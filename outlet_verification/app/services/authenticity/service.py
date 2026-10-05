import logging
from typing import Any
from pydantic import BaseModel, Field

from app.services.authenticity.name_classifier import NameClassifier
from app.services.authenticity.signboard_ocr import OCRAnalysisResult, SignboardOCRService
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
    is_cross_lingual_match: bool = False
    is_incidental_signage: bool = False
    is_unclear_or_multilingual: bool = False
    ocr_clarity_status: str = "LEGIBLE"
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

    def warm_up(self) -> None:
        """Pre-load heavyweight visual and OCR neural networks during app startup."""
        self.storefront_classifier.load_model()
        self.ocr_service.load_reader()

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

        # Smart Bypass: If image is decisively NOT a retail storefront (e.g. bedroom, living room, car, road, selfie)
        # or name is explicitly a non-outlet term (e.g. 'bedroom'), skip expensive OCR on non-store elements.
        if not visual_res.is_retail_store and (not name_res.is_valid_outlet_name or visual_res.store_probability < 0.20):
            logger.info(
                "Skipping OCR analysis: Image is not a retail storefront (store_prob: %.2f%%) and name is '%s'",
                visual_res.store_probability * 100,
                name,
            )
            ocr_res = OCRAnalysisResult(
                extracted_text_lines=[],
                detected_retail_keywords=[],
                name_match_score=None,
                has_readable_signboard=False,
                is_cross_lingual_or_category_match=False,
                is_incidental_signage=False,
                has_conflicting_brand=False,
                conflicting_brand_name=None,
                is_unclear_or_multilingual=False,
                clarity_status="NO_SIGNBOARD",
                rejection_reason=None,
            )
        else:
            # 3. Signboard OCR & Cross-Verification (EasyOCR + Multilingual & Semantic Matching)
            ocr_res = self.ocr_service.analyze_image(image, submitted_name=name)

        # Signboard Cross-Verification:
        # A submission is ONLY rejected for NAME_MISMATCH / SIGNBOARD_MISMATCH if:
        # 1. OCR extracted clear, high-confidence signboard text (not low confidence / blurry / multilingual), AND
        # 2. There is an explicit, verified conflicting commercial brand name (e.g. 'MedPlus' on an 'Apollo' submission).
        # If OCR details cannot be extracted with high confidence, OCR is marked as inconclusive
        # and bypassed from comparison so genuine outlets are NOT falsely rejected.
        if (
            ocr_res.has_conflicting_brand
            and ocr_res.conflicting_brand_name
            and not ocr_res.is_cross_lingual_or_category_match
            and not ocr_res.is_incidental_signage
            and not ocr_res.is_unclear_or_multilingual
            and visual_res.is_retail_store
        ):
            rejection_reasons.append("NAME_MISMATCH")
            rejection_reasons.append("SIGNBOARD_MISMATCH")
            logger.info(
                "Name mismatch detected: Signboard has confirmed conflicting brand '%s', which contradicts '%s'",
                ocr_res.conflicting_brand_name,
                name,
            )
        elif ocr_res.is_cross_lingual_or_category_match or ocr_res.is_incidental_signage:
            logger.info(
                "Semantic / cross-lingual or incidental match confirmed for '%s' (OCR text: '%s', score: %.2f, incidental: %s)",
                name,
                " ".join(ocr_res.extracted_text_lines),
                ocr_res.name_match_score or 0.0,
                ocr_res.is_incidental_signage,
            )
        elif ocr_res.is_unclear_or_multilingual or not ocr_res.has_readable_signboard:
            logger.info(
                "Signboard details obtained with low confidence / unclear for '%s' (status: %s). Bypassing OCR from comparison.",
                name,
                ocr_res.clarity_status,
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
            if "NAME_MISMATCH" in rejection_reasons or "SIGNBOARD_MISMATCH" in rejection_reasons:
                detected_text = ocr_res.conflicting_brand_name or " ".join(ocr_res.extracted_text_lines[:3])
                score_pct = f"{ocr_res.name_match_score * 100:.0f}%" if ocr_res.name_match_score is not None else "0%"
                summary_parts.append(
                    f"Name mismatch detected: Signboard in photo contains conflicting brand '{detected_text}' which contradicts the submitted name '{name}' (similarity: {score_pct})"
                )
            summary = "; ".join(summary_parts) + "."
        else:
            is_authentic = True
            status = "AUTHENTIC"
            confidence = round(visual_res.store_probability, 4)
            if ocr_res.is_cross_lingual_or_category_match and not ocr_res.is_incidental_signage:
                score_pct = f"{ocr_res.name_match_score * 100:.0f}%" if ocr_res.name_match_score is not None else "100%"
                summary = (
                    f"Verified authentic retail storefront. "
                    f"Semantic and cross-lingual match confirmed between submitted name '{name}' "
                    f"and physical signboard (match score: {score_pct})."
                )
            elif ocr_res.is_incidental_signage:
                detected_incidental = ", ".join(ocr_res.extracted_text_lines[:3])
                summary = (
                    f"Verified authentic retail storefront (storefront probability: {visual_res.store_probability:.1%}, "
                    f"category: {name_res.category}). "
                    f"Storefront verified with incidental product signage ('{detected_incidental}') and no conflicting commercial brand."
                )
            elif ocr_res.is_unclear_or_multilingual or not ocr_res.has_readable_signboard:
                if ocr_res.clarity_status == "MULTILINGUAL_OR_STYLIZED":
                    clarity_desc = "signboard is in regional script or stylized font (bypassed from comparison)"
                elif ocr_res.clarity_status == "UNCLEAR_OR_BLURRY":
                    clarity_desc = "signboard photo is blurry or unreadable (bypassed from comparison)"
                else:
                    clarity_desc = "signboard text obtained with low confidence / unclear (bypassed from comparison)"
                summary = (
                    f"Verified authentic retail storefront (storefront probability: {visual_res.store_probability:.1%}, "
                    f"category: {name_res.category}). "
                    f"Physical {clarity_desc}; visual and semantic signals confirmed authenticity."
                )
            else:
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
            is_cross_lingual_match=ocr_res.is_cross_lingual_or_category_match,
            is_incidental_signage=ocr_res.is_incidental_signage,
            is_unclear_or_multilingual=ocr_res.is_unclear_or_multilingual,
            ocr_clarity_status=ocr_res.clarity_status,
            rejection_reasons=rejection_reasons,
            summary=summary,
        )


_authenticity_service: StoreAuthenticityService | None = None


def get_authenticity_service() -> StoreAuthenticityService:
    global _authenticity_service
    if _authenticity_service is None:
        _authenticity_service = StoreAuthenticityService()
    return _authenticity_service
