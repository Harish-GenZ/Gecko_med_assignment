import logging
from typing import Any
from pydantic import BaseModel, Field

from app.services.authenticity.name_classifier import NameClassifier
from app.services.authenticity.signboard_ocr import SignboardOCRService
from app.services.authenticity.storefront_classifier import StorefrontClassifier

logger = logging.getLogger("outlet_verification.authenticity.service")


class AuthenticityAssessment(BaseModel):
    is_authentic_store: bool = Field(..., description="Whether the submission qualifies as a genuine retail outlet / store")
    status: str = Field(..., description="'AUTHENTIC', 'SUSPICIOUS', or 'REJECTED'")
    authenticity_confidence: float = Field(..., description="Confidence score [0.0, 1.0] of domain authenticity")
    visual_is_store: bool
    visual_probability: float
    visual_category: str
    name_is_valid_outlet: bool
    name_category: str
    ocr_lines: list[str] = Field(default_factory=list)
    ocr_retail_detected: bool = False
    ocr_name_match_score: float | None = None
    rejection_reasons: list[str] = Field(default_factory=list)
    summary: str


class StoreAuthenticityService:
    """
    Unified Domain Authenticity Service for Retail Outlets (Supermarkets, General Stores, Grocery, Marts).
    Determines whether a submission is genuinely a commercial retail outlet / storefront by combining:
      1. Zero-shot CLIP Storefront Visual Classification
      2. Signboard OCR & Cross-Verification (EasyOCR)
      3. Semantic Name Classification (English + Tamil)
    """

    def __init__(self) -> None:
        self.name_classifier = NameClassifier()
        self.storefront_classifier = StorefrontClassifier.get_instance()
        self.ocr_service = SignboardOCRService.get_instance()

    def evaluate_authenticity(self, name: str, image: Any) -> AuthenticityAssessment:
        rejection_reasons: list[str] = []

        # 1. Semantic Name Classification
        name_res = self.name_classifier.classify(name)
        if not name_res.is_valid_outlet_name and name_res.category == "NON_OUTLET":
            rejection_reasons.append("NON_OUTLET_NAME")
            logger.info("Name '%s' flagged as non-outlet: %s", name, name_res.rejection_reason)

        # 2. Storefront Visual Domain Classification (CLIP)
        visual_res = self.storefront_classifier.classify_storefront(image)
        if not visual_res.is_retail_store:
            rejection_reasons.append("NOT_A_RETAIL_STOREFRONT_IMAGE")
            logger.info("Image rejected by visual classifier: %s", visual_res.rejection_reason)

        # 3. Signboard OCR & Cross-Verification (EasyOCR)
        ocr_res = self.ocr_service.analyze_image(image, submitted_name=name)
        if (
            ocr_res.has_readable_signboard
            and ocr_res.detected_retail_keywords
            and ocr_res.name_match_score is not None
            and ocr_res.name_match_score < 0.20
        ):
            rejection_reasons.append("SIGNBOARD_MISMATCH")
            logger.info(
                "Signboard mismatch detected: OCR read '%s', but submitted name is '%s' (score: %.2f)",
                " ".join(ocr_res.extracted_text_lines),
                name,
                ocr_res.name_match_score,
            )

        # Synthesis & Decision
        if rejection_reasons:
            is_authentic = False
            status = "REJECTED"
            confidence = round(1.0 - visual_res.store_probability if not visual_res.is_retail_store else 0.95, 4)
            summary_parts = []
            if "NOT_A_RETAIL_STOREFRONT_IMAGE" in rejection_reasons:
                summary_parts.append(f"Image appears to be '{visual_res.top_predicted_label}' rather than an authentic store storefront")
            if "NON_OUTLET_NAME" in rejection_reasons:
                summary_parts.append(f"Name '{name}' does not represent a valid commercial outlet")
            if "SIGNBOARD_MISMATCH" in rejection_reasons:
                detected_text = " ".join(ocr_res.extracted_text_lines[:3])
                summary_parts.append(f"Signboard in photo reads '{detected_text}' which contradicts the submitted name '{name}'")
            summary = "; ".join(summary_parts) + "."
        else:
            is_authentic = True
            status = "AUTHENTIC"
            confidence = round(visual_res.store_probability, 4)
            summary = (
                f"Verified authentic retail outlet storefront "
                f"(storefront probability: {visual_res.store_probability:.1%}, "
                f"name category: {name_res.category})."
            )

        return AuthenticityAssessment(
            is_authentic_store=is_authentic,
            status=status,
            authenticity_confidence=confidence,
            visual_is_store=visual_res.is_retail_store,
            visual_probability=visual_res.store_probability,
            visual_category=visual_res.top_predicted_label,
            name_is_valid_outlet=name_res.is_valid_outlet_name,
            name_category=name_res.category,
            ocr_lines=ocr_res.extracted_text_lines,
            ocr_retail_detected=len(ocr_res.detected_retail_keywords) > 0,
            ocr_name_match_score=ocr_res.name_match_score,
            rejection_reasons=rejection_reasons,
            summary=summary,
        )


_authenticity_service: StoreAuthenticityService | None = None


def get_authenticity_service() -> StoreAuthenticityService:
    global _authenticity_service
    if _authenticity_service is None:
        _authenticity_service = StoreAuthenticityService()
    return _authenticity_service
