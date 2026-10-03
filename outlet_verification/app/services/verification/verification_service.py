import logging
import time
from typing import Any
from sqlalchemy.orm import Session

from app.models.outlet import Outlet
from app.services.embeddings import generate_image_embedding, generate_name_embedding
from app.services.retrieval import (
    CandidateRetrievalService,
    get_candidate_retrieval_service,
)
from app.services.authenticity import AuthenticityAssessment, get_authenticity_service
from app.services.storage_service import StorageService, get_storage_service
from app.services.verification.decision import DecisionEngine, get_decision_engine
from app.services.verification.models import (
    ReasonCode,
    VerificationDecision,
    VerificationResponse,
)
from app.services.verification.scoring import CandidateScorer, get_candidate_scorer

logger = logging.getLogger("outlet_verification.verification.service")


class VerificationService:
    """
    High-level orchestrator for multimodal outlet verification.
    Coordinates candidate retrieval, evidence normalization, multimodal scoring,
    deterministic decision classification, and genuine outlet persistence.
    """

    def __init__(
        self,
        retrieval_service: CandidateRetrievalService | None = None,
        scorer: CandidateScorer | None = None,
        decision_engine: DecisionEngine | None = None,
        storage_service: StorageService | None = None,
    ) -> None:
        self.retrieval_service = (
            retrieval_service
            if retrieval_service is not None
            else get_candidate_retrieval_service()
        )
        self.scorer = scorer if scorer is not None else get_candidate_scorer()
        self.decision_engine = (
            decision_engine if decision_engine is not None else get_decision_engine()
        )
        self.storage_service = (
            storage_service if storage_service is not None else get_storage_service()
        )
        self.authenticity_service = get_authenticity_service()

    def register_outlet(
        self,
        db: Session,
        name: str,
        latitude: float,
        longitude: float,
        image: Any,
        filename: str = "storefront.jpg",
    ) -> Outlet:
        """
        Uploads outlet photograph to Railway Object Storage (S3 bucket),
        computes text and image embeddings, and registers the verified genuine
        outlet into PostgreSQL.
        """
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Outlet name cannot be empty.")
        if not (-90.0 <= latitude <= 90.0):
            raise ValueError(f"Latitude must be between -90 and 90 degrees. Got {latitude}.")
        if not (-180.0 <= longitude <= 180.0):
            raise ValueError(f"Longitude must be between -180 and 180 degrees. Got {longitude}.")

        # 1. Upload photograph to Railway S3 Object Storage
        stored_url, object_key = self.storage_service.upload_outlet_image(
            image_data=image,
            filename=filename,
        )

        # 2. Compute embeddings
        name_emb = generate_name_embedding(clean_name)
        img_emb = generate_image_embedding(image)

        # 3. Create Outlet record in PostgreSQL
        new_outlet = Outlet(
            name=clean_name,
            latitude=latitude,
            longitude=longitude,
            image_url=stored_url,
            name_embedding=name_emb,
            image_embedding=img_emb,
        )
        db.add(new_outlet)
        db.commit()
        db.refresh(new_outlet)

        logger.info(
            "Registered new genuine outlet in PostgreSQL | ID: %s | Name: '%s' | S3 Object: %s",
            new_outlet.id,
            clean_name,
            object_key,
        )
        return new_outlet

    def verify_outlet(
        self,
        db: Session,
        name: str,
        latitude: float | None = None,
        longitude: float | None = None,
        image: Any = None,
        image_filename: str = "storefront.jpg",
        auto_register_if_genuine: bool = False,
        name_top_k: int | None = None,
        image_top_k: int | None = None,
        geo_radius_meters: float | None = None,
        geo_top_k: int | None = None,
    ) -> VerificationResponse:
        """
        Executes the end-to-end deterministic verification pipeline for a submitted outlet.
        Supports missing coordinates (latitude/longitude=None) and missing photos (image=None).
        If auto_register_if_genuine is True and decision is GENUINE, automatically uploads the
        storefront image to Railway Object Storage and registers the outlet in PostgreSQL.
        """
        start_time = time.perf_counter()

        # 1. Coordinate Validation (if coordinates provided)
        if latitude is not None and not (-90.0 <= latitude <= 90.0):
            raise ValueError(f"Latitude must be between -90 and 90 degrees. Got {latitude}.")
        if longitude is not None and not (-180.0 <= longitude <= 180.0):
            raise ValueError(f"Longitude must be between -180 and 180 degrees. Got {longitude}.")

        clean_name = name.strip() if isinstance(name, str) else ""
        if not clean_name:
            raise ValueError("Outlet name cannot be empty.")

        coord_str = f"({latitude:.6f}, {longitude:.6f})" if (latitude is not None and longitude is not None) else "(None, None)"
        logger.info(
            "Starting verification for outlet '%s' at %s (image provided: %s)...",
            clean_name,
            coord_str,
            image is not None,
        )

        # 1B. Authenticity Pre-check (Storefront Visual, Signboard OCR, Name Validation)
        auth_assessment: AuthenticityAssessment | None = None
        if image is not None:
            auth_assessment = self.authenticity_service.evaluate_authenticity(clean_name, image)
            if not auth_assessment.is_authentic_store:
                logger.info(
                    "Outlet '%s' failed domain authenticity pre-check: %s",
                    clean_name,
                    auth_assessment.rejection_reasons,
                )
                duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
                return VerificationResponse(
                    decision=VerificationDecision.REJECTED,
                    duplicate_confidence=0.0,
                    evidence_coverage=1.0,
                    evidence_agreement=1.0,
                    matched_outlet=None,
                    evidence=None,
                    matched_methods=[],
                    reason_codes=auth_assessment.rejection_reasons or [ReasonCode.FAKE_OUTLET_REJECTED.value],
                    reason_summary=f"Outlet rejected as fake / non-commercial store: {auth_assessment.summary}",
                    candidate_margin=None,
                    evidence_status="REJECTED_NON_STORE",
                    candidates_evaluated=0,
                    registered=False,
                    registered_outlet_id=None,
                    stored_image_url=None,
                    authenticity=auth_assessment,
                )

        # 2. Retrieve Candidate Pool (Multi-channel: Name, Image, Geo)
        retrieval_result = self.retrieval_service.retrieve_candidates(
            db=db,
            name=clean_name,
            latitude=latitude,
            longitude=longitude,
            image=image,
            name_top_k=name_top_k,
            image_top_k=image_top_k,
            geo_radius_meters=geo_radius_meters,
            geo_top_k=geo_top_k,
        )

        candidates = retrieval_result.candidates
        logger.info(
            "Candidate retrieval returned %d potential candidate(s).",
            len(candidates),
        )

        # 3. Multimodal Evidence Fusion & Scoring for each candidate
        scored_candidates = [self.scorer.score_candidate(c) for c in candidates]

        # 4. Deterministic Decision Engine Evaluation
        response = self.decision_engine.evaluate(scored_candidates)

        # 5. Auto-Registration for Genuine Outlets
        if response.decision == VerificationDecision.GENUINE and auto_register_if_genuine:
            if latitude is not None and longitude is not None and image is not None:
                try:
                    registered_outlet = self.register_outlet(
                        db=db,
                        name=clean_name,
                        latitude=latitude,
                        longitude=longitude,
                        image=image,
                        filename=image_filename,
                    )
                    response.registered = True
                    response.registered_outlet_id = str(registered_outlet.id)
                    response.stored_image_url = registered_outlet.image_url
                    logger.info(
                        "Auto-registered genuine outlet %s ('%s') into PostgreSQL and S3",
                        registered_outlet.id,
                        clean_name,
                    )
                except Exception as reg_exc:
                    logger.error("Failed to auto-register genuine outlet '%s': %s", clean_name, reg_exc)
                    response.registered = False
            else:
                logger.warning(
                    "Skipping auto-registration for genuine outlet '%s': missing coordinates or image.",
                    clean_name,
                )

        response.authenticity = auth_assessment
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # 6. Structured Audit Logging
        matched_id = response.matched_outlet.outlet_id if response.matched_outlet else "None"
        logger.info(
            "Verification completed in %.2fms | Decision: %s | Confidence: %.4f | "
            "Top Candidate: %s | Registered: %s | Reasons: %s",
            duration_ms,
            response.decision.value,
            response.duplicate_confidence,
            matched_id,
            response.registered,
            ", ".join(response.reason_codes),
        )

        return response


def get_verification_service() -> VerificationService:
    """Factory getter for VerificationService."""
    return VerificationService()
