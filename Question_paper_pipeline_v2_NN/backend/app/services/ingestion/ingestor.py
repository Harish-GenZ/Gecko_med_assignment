"""
Document ingestion: accepts an uploaded file (PDF or image) and returns
a list of (page_index, PIL.Image) tuples ready for preprocessing.

Strategy:
  - For each PDF page, attempt native text extraction first.
  - If chars < threshold → render page as image for OCR.
  - For image inputs (JPEG/PNG) → pass directly.
"""

import io
from pathlib import Path
from typing import List, Tuple

import fitz  # PyMuPDF
from PIL import Image

from app.core.config import PAGE_SCANNED_THRESHOLD, DPI_RENDER
from app.models.schemas import PageType


def ingest_pdf(pdf_path: Path) -> List[dict]:
    """
    Process a PDF file.

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
    mat = fitz.Matrix(DPI_RENDER / 72, DPI_RENDER / 72)  # 200 DPI

    for page_index, page in enumerate(doc):
        native_text = page.get_text("text").strip()
        char_count  = len(native_text)

        if char_count >= PAGE_SCANNED_THRESHOLD:
            page_type = PageType.DIGITAL
        else:
            page_type = PageType.SCANNED

        # Always render to image (OCR needs it for scanned; digital uses native_text)
        pix  = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
        img  = Image.frombytes("L", [pix.width, pix.height], pix.samples)

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
        img = Image.open(img_path).convert("L")  # grayscale
        pages.append({
            "page_index":  idx,
            "page_type":   PageType.SCANNED,
            "native_text": "",
            "image":       img,
        })
    return pages
