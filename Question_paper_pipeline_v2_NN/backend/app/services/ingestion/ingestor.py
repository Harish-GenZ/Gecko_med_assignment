"""
Document ingestion: accepts an uploaded file (PDF or image) and returns
a list of (page_index, PIL.Image) tuples ready for preprocessing.

Strategy:
  - For each PDF page, attempt native text extraction first.
  - If chars < threshold → render page as image for OCR.
  - For image inputs (JPEG/PNG) → pass directly.
"""

import io
import re
from pathlib import Path
from typing import List, Tuple

import fitz  # PyMuPDF
from PIL import Image

from app.core.config import PAGE_SCANNED_THRESHOLD, DPI_RENDER
from app.models.schemas import PageType

# Question indicators to distinguish genuine digital question paper pages
# from short watermarks, copyright notices, telegram links, or scanner headers
_QUESTION_PATTERN_RE = re.compile(
    r'(?:(?:^|\n)\s*(?:Q\.?\s*\d+|\b\d{1,2}\s*[\.\)]|\([a-z0-9ivx]+\))\s+)|'
    r'\b(?:describe|explain|define|enlist|enumerate|differentiate|discuss|'
    r'write\s+short\s+notes?|classify|what\s+is|what\s+are)\b|'
    r'(?:\(\s*\d+\s*marks?\s*\)|\[\s*\d+\s*marks?\s*\]|\bmarks?[:\s-]+\d+)',
    re.I
)


def ingest_pdf(pdf_path: Path) -> List[dict]:
    """
    Process a PDF file.
    Applies adaptive render scaling (300 DPI / minimum 2800px height) and
    realistic digital vs. scanned detection (>= PAGE_SCANNED_THRESHOLD chars + verified question patterns).
    Pages falling short are routed as SCANNED to the neural OCR pipeline.

    Returns a list of page dicts:
    {
        "page_index": int,        # 0-based
        "page_type": PageType,
        "native_text": str,       # Non-empty if digital
        "image": PIL.Image,       # Always provided (for OCR fallback)
    }
    """
    pages = []
    doc = fitz.open(pdf_path)

    for page_index, page in enumerate(doc):
        rect = page.rect
        # 1. Adaptive Render Scaling:
        # Ensure DPI is at least DPI_RENDER (300 DPI), or dynamically scaled
        # so the rendered height is at least ~2800px (capped at 360 DPI to protect memory/speed)
        target_dpi = max(float(DPI_RENDER), (2800.0 / max(rect.height, 1.0)) * 72.0)
        target_dpi = min(target_dpi, 360.0)
        scale = target_dpi / 72.0
        mat = fitz.Matrix(scale, scale)

        native_text = page.get_text("text").strip()
        char_count  = len(native_text)

        # 2. Realistic "Digital vs Scanned" Detection:
        # Requires sufficient character volume AND presence of genuine question patterns
        has_question_patterns = bool(_QUESTION_PATTERN_RE.search(native_text))
        if char_count >= PAGE_SCANNED_THRESHOLD and has_question_patterns:
            page_type = PageType.DIGITAL
        else:
            page_type = PageType.SCANNED

        # Always render to image in RGB (needed for both OCR and Vision models)
        pix  = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        pages.append({
            "page_index":  page_index,
            "page_type":   page_type,
            "native_text": native_text,
            "image":       img,
        })

    doc.close()
    return pages


def ingest_images(image_paths: List[Path]) -> List[dict]:
    """
    Process a list of standalone image files (UHSR WhatsApp images).

    Returns same structure as ingest_pdf but page_type is always SCANNED.
    """
    pages = []
    for idx, img_path in enumerate(sorted(image_paths)):
        img = Image.open(img_path).convert("RGB")
        pages.append({
            "page_index":  idx,
            "page_type":   PageType.SCANNED,
            "native_text": "",
            "image":       img,
        })
    return pages
