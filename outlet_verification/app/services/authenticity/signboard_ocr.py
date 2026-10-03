import difflib
import logging
import re
import threading
from typing import Any, NamedTuple
import numpy as np

from app.services.authenticity.utils import load_pil_image

logger = logging.getLogger("outlet_verification.authenticity.ocr")

RETAIL_SIGNBOARD_KEYWORDS = {
    "supermarket", "super market", "general store", "general stores", "store", "stores",
    "mart", "provisions", "provision", "grocery", "groceries", "departmental",
    "department store", "hypermarket", "retail", "bazaar", "traders", "kirana",
    "daily needs", "foods", "organics", "wholesale", "agency", "enterprises",
    "pharmacy", "medicals", "bakery", "sweets"
}


class OCRAnalysisResult(NamedTuple):
    extracted_text_lines: list[str]
    detected_retail_keywords: list[str]
    name_match_score: float | None
    has_readable_signboard: bool
    rejection_reason: str | None


class SignboardOCRService:
    """
    Signboard OCR and cross-verification service for retail outlets and supermarkets.
    Extracts text from the storefront photograph using EasyOCR, checks for store branding,
    and cross-matches with the submitted outlet name.
    """
    _instance: "SignboardOCRService | None" = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self.reader: Any = None
        self._is_loaded: bool = False

    @classmethod
    def get_instance(cls) -> "SignboardOCRService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def load_reader(self) -> None:
        if self._is_loaded:
            return

        with self._lock:
            if self._is_loaded:
                return

            try:
                import easyocr
                logger.info("Initializing EasyOCR reader for signboard text extraction...")
                self.reader = easyocr.Reader(["en"], gpu=False, verbose=False)
                self._is_loaded = True
                logger.info("EasyOCR reader successfully initialized.")
            except Exception as exc:
                logger.warning("Failed to initialize EasyOCR reader: %s. OCR checks will be bypassed.", exc)
                self.reader = None
                self._is_loaded = True

    def analyze_image(self, image_input: Any, submitted_name: str | None = None) -> OCRAnalysisResult:
        """
        Extracts text from the image and performs keyword and name cross-matching.
        """
        self.load_reader()
        if self.reader is None:
            return OCRAnalysisResult(
                extracted_text_lines=[],
                detected_retail_keywords=[],
                name_match_score=None,
                has_readable_signboard=False,
                rejection_reason=None,
            )

        pil_img = load_pil_image(image_input)
        img_np = np.array(pil_img)

        try:
            results = self.reader.readtext(img_np)
        except Exception as exc:
            logger.warning("EasyOCR execution failed: %s", exc)
            return OCRAnalysisResult(
                extracted_text_lines=[],
                detected_retail_keywords=[],
                name_match_score=None,
                has_readable_signboard=False,
                rejection_reason=None,
            )

        extracted_lines: list[str] = []
        for _, text, conf in results:
            if conf >= 0.25 and text and len(text.strip()) > 1:
                extracted_lines.append(text.strip())

        full_ocr_text = " ".join(extracted_lines).lower()
        words = set(re.findall(r"\w+", full_ocr_text))

        # 1. Detect retail keywords
        detected_retail = [kw for kw in RETAIL_SIGNBOARD_KEYWORDS if kw in words or kw in full_ocr_text]

        # 2. Cross-match with submitted name
        name_match_score: float | None = None
        if submitted_name and extracted_lines:
            name_norm = submitted_name.strip().lower()
            name_words = set(re.findall(r"\w+", name_norm))

            # Token overlap score
            overlap_words = words.intersection(name_words)
            token_overlap_score = len(overlap_words) / max(1, len(name_words))

            # String sequence matching score against full OCR text
            seq_score = difflib.SequenceMatcher(None, name_norm, full_ocr_text).ratio()
            name_match_score = round(max(token_overlap_score, seq_score), 4)

        has_readable_signboard = len(extracted_lines) > 0

        return OCRAnalysisResult(
            extracted_text_lines=extracted_lines,
            detected_retail_keywords=detected_retail,
            name_match_score=name_match_score,
            has_readable_signboard=has_readable_signboard,
            rejection_reason=None,
        )
