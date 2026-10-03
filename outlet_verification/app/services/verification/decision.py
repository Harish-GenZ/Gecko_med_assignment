import logging
from typing import Any

from app.config import get_settings
from app.services.verification.models import (
    MatchedOutlet,
    ReasonCode,
    ScoredCandidate,
    VerificationDecision,
    VerificationEvidence,
    VerificationResponse,
)

logger = logging.getLogger("outlet_verification.verification.decision")


class DecisionEngine:
    """
    Deterministic verification decision engine.
    Ranks candidates, inspects signal coverage and agreement, evaluates ambiguity margins,
    applies safeguard rules, and yields a categorical decision with structured reason codes.
    """

    def __init__(
        self,
        duplicate_threshold: float | None = None,
        genuine_threshold: float | None = None,
        min_coverage: float | None = None,
        min_margin: float | None = None,
    ) -> None:
        settings = get_settings()
        self.duplicate_threshold = (
            duplicate_threshold
            if duplicate_threshold is not None
            else settings.VERIFICATION_DUPLICATE_THRESHOLD
        )
        self.genuine_threshold = (
            genuine_threshold
            if genuine_threshold is not None
            else settings.VERIFICATION_GENUINE_THRESHOLD
        )
        self.min_coverage = (
            min_coverage
            if min_coverage is not None
            else settings.VERIFICATION_REVIEW_MIN_COVERAGE
        )
        self.min_margin = (
            min_margin if min_margin is not None else settings.VERIFICATION_MIN_MARGIN
        )

    def evaluate(self, scored_candidates: list[ScoredCandidate]) -> VerificationResponse:
        """
        Evaluates a pool of scored candidates and returns the final verification response.
        """
        # ----------------------------------------------------------------------
        # Case 1: Zero candidates retrieved
        # ----------------------------------------------------------------------
        if not scored_candidates:
            logger.info("Candidate retrieval returned 0 candidates. Classifying as GENUINE (no match).")
            return VerificationResponse(
                decision=VerificationDecision.GENUINE,
                duplicate_confidence=0.0,
                evidence_coverage=0.0,
                evidence_agreement=1.0,
                matched_outlet=None,
                evidence=None,
                matched_methods=[],
                reason_codes=[ReasonCode.NO_MATCHING_CANDIDATE.value],
                reason_summary="No plausible existing outlet was retrieved in database search.",
                candidate_margin=None,
                evidence_status="NO_MATCHING_CANDIDATE",
                candidates_evaluated=0,
            )

        # ----------------------------------------------------------------------
        # Step 2: Sort and rank candidates by duplicate_confidence descending
        # ----------------------------------------------------------------------
        ranked = sorted(
            scored_candidates,
            key=lambda sc: (sc.duplicate_confidence, sc.base_score, sc.evidence_coverage),
            reverse=True,
        )

        top_sc = ranked[0]
        runner_up_sc = ranked[1] if len(ranked) > 1 else None

        margin: float | None = None
        if runner_up_sc is not None:
            margin = round(top_sc.duplicate_confidence - runner_up_sc.duplicate_confidence, 4)

        # Prepare matched outlet and evidence details
        matched_outlet = MatchedOutlet(
            outlet_id=top_sc.candidate.outlet_id,
            name=top_sc.candidate.name,
            image_url=top_sc.candidate.image_url,
            latitude=top_sc.candidate.latitude,
            longitude=top_sc.candidate.longitude,
        )

        evidence = VerificationEvidence(
            name_similarity=top_sc.candidate.name_similarity,
            image_similarity=top_sc.candidate.image_similarity,
            distance_meters=top_sc.candidate.distance_meters,
            geo_proximity_score=top_sc.geo_proximity_score,
        )

        reason_codes: list[str] = []
        flags = top_sc.flags

        # Record secondary observation flags
        if flags.possible_same_brand_different_location:
            reason_codes.append(ReasonCode.POSSIBLE_SAME_BRAND_DIFFERENT_LOCATION.value)
        if flags.strong_duplicate_evidence:
            reason_codes.append(ReasonCode.STRONG_IMAGE_GEO_MATCH.value)

        # ----------------------------------------------------------------------
        # Step 3: Safeguard Check — Strong Contradictory Evidence
        # (e.g. name is almost identical but image is completely different)
        # ----------------------------------------------------------------------
        if flags.conflicting_evidence:
            reason_codes.append(ReasonCode.CONFLICTING_NAME_IMAGE.value)
            logger.info(
                "Outlet %s flagged for conflicting evidence (high name similarity with low image similarity).",
                top_sc.candidate.outlet_id,
            )
            return VerificationResponse(
                decision=VerificationDecision.NEEDS_REVIEW,
                duplicate_confidence=top_sc.duplicate_confidence,
                evidence_coverage=top_sc.evidence_coverage,
                evidence_agreement=top_sc.evidence_agreement,
                matched_outlet=matched_outlet,
                evidence=evidence,
                matched_methods=top_sc.candidate.matched_methods,
                reason_codes=reason_codes,
                reason_summary="High name similarity conflicts with low visual image similarity. Manual review required.",
                candidate_margin=margin,
                evidence_status="CONFLICTING_EVIDENCE",
                candidates_evaluated=len(scored_candidates),
            )

        # ----------------------------------------------------------------------
        # Step 4: Safeguard Check — Ambiguous Top Candidates (Margin Check)
        # If top candidate is in duplicate range but runner-up is nearly tied
        # ----------------------------------------------------------------------
        if (
            top_sc.duplicate_confidence >= self.duplicate_threshold
            and runner_up_sc is not None
            and margin is not None
            and margin < self.min_margin
        ):
            reason_codes.append(ReasonCode.AMBIGUOUS_TOP_CANDIDATES.value)
            logger.info(
                "Top candidates are ambiguous (margin %.4f < %.2f). Marking as NEEDS_REVIEW.",
                margin,
                self.min_margin,
            )
            return VerificationResponse(
                decision=VerificationDecision.NEEDS_REVIEW,
                duplicate_confidence=top_sc.duplicate_confidence,
                evidence_coverage=top_sc.evidence_coverage,
                evidence_agreement=top_sc.evidence_agreement,
                matched_outlet=matched_outlet,
                evidence=evidence,
                matched_methods=top_sc.candidate.matched_methods,
                reason_codes=reason_codes,
                reason_summary=(
                    f"Top candidate ({top_sc.duplicate_confidence:.2f}) and runner-up "
                    f"({runner_up_sc.duplicate_confidence:.2f}) have ambiguous margin "
                    f"({margin:.4f} < {self.min_margin:.2f}). Manual review required."
                ),
                candidate_margin=margin,
                evidence_status="AMBIGUOUS_CANDIDATES",
                candidates_evaluated=len(scored_candidates),
            )

        # ----------------------------------------------------------------------
        # Step 5: Safeguard Check — Insufficient Evidence Coverage
        # (Cannot declare DUPLICATE if coverage is below threshold)
        # ----------------------------------------------------------------------
        if top_sc.evidence_coverage < self.min_coverage:
            if top_sc.duplicate_confidence > self.genuine_threshold:
                reason_codes.append(ReasonCode.INSUFFICIENT_EVIDENCE.value)
                return VerificationResponse(
                    decision=VerificationDecision.NEEDS_REVIEW,
                    duplicate_confidence=top_sc.duplicate_confidence,
                    evidence_coverage=top_sc.evidence_coverage,
                    evidence_agreement=top_sc.evidence_agreement,
                    matched_outlet=matched_outlet,
                    evidence=evidence,
                    matched_methods=top_sc.candidate.matched_methods,
                    reason_codes=reason_codes,
                    reason_summary=(
                        f"Evidence coverage ({top_sc.evidence_coverage:.2f}) is below minimum "
                        f"required coverage ({self.min_coverage:.2f}) for duplicate confirmation."
                    ),
                    candidate_margin=margin,
                    evidence_status="LOW_COVERAGE",
                    candidates_evaluated=len(scored_candidates),
                )

        # ----------------------------------------------------------------------
        # Step 6: Safeguard Check — Weak / Inconclusive Signal Combinations
        # e.g. Missing image + moderate ambiguous name similarity (0.40 - 0.75)
        # or Missing name + moderate ambiguous image similarity (0.40 - 0.75)
        # Only applies to candidates in close geographic vicinity (<= 500m or GPS unknown).
        # Distant candidates with moderate similarity are genuinely distinct establishments.
        # ----------------------------------------------------------------------
        c = top_sc.candidate
        dist = c.distance_meters
        if dist is None or dist <= 500.0:
            if c.image_similarity is None and c.name_similarity is not None:
                if 0.40 <= c.name_similarity <= 0.75:
                    reason_codes.append(ReasonCode.WEAK_EVIDENCE_REQUIRES_REVIEW.value)
                    return VerificationResponse(
                        decision=VerificationDecision.NEEDS_REVIEW,
                        duplicate_confidence=top_sc.duplicate_confidence,
                        evidence_coverage=top_sc.evidence_coverage,
                        evidence_agreement=top_sc.evidence_agreement,
                        matched_outlet=matched_outlet,
                        evidence=evidence,
                        matched_methods=top_sc.candidate.matched_methods,
                        reason_codes=reason_codes,
                        reason_summary="Image evidence is unavailable and name similarity is moderate/ambiguous in close vicinity. Manual review required.",
                        candidate_margin=margin,
                        evidence_status="WEAK_AMBIGUOUS_EVIDENCE",
                        candidates_evaluated=len(scored_candidates),
                    )

            if c.name_similarity is None and c.image_similarity is not None:
                if 0.40 <= c.image_similarity <= 0.75:
                    reason_codes.append(ReasonCode.WEAK_EVIDENCE_REQUIRES_REVIEW.value)
                    return VerificationResponse(
                        decision=VerificationDecision.NEEDS_REVIEW,
                        duplicate_confidence=top_sc.duplicate_confidence,
                        evidence_coverage=top_sc.evidence_coverage,
                        evidence_agreement=top_sc.evidence_agreement,
                        matched_outlet=matched_outlet,
                        evidence=evidence,
                        matched_methods=top_sc.candidate.matched_methods,
                        reason_codes=reason_codes,
                        reason_summary="Name evidence is unavailable and image similarity is moderate/ambiguous in close vicinity. Manual review required.",
                        candidate_margin=margin,
                        evidence_status="WEAK_AMBIGUOUS_EVIDENCE",
                        candidates_evaluated=len(scored_candidates),
                    )

        # ----------------------------------------------------------------------
        # Step 7: Confidence Threshold Evaluation
        # ----------------------------------------------------------------------
        if top_sc.duplicate_confidence >= self.duplicate_threshold:
            decision = VerificationDecision.DUPLICATE
            if (
                c.name_similarity is not None
                and c.image_similarity is not None
                and c.distance_meters is not None
                and top_sc.duplicate_confidence >= 0.85
            ):
                if ReasonCode.STRONG_MULTIMODAL_MATCH.value not in reason_codes:
                    reason_codes.append(ReasonCode.STRONG_MULTIMODAL_MATCH.value)
            elif not flags.strong_duplicate_evidence:
                if ReasonCode.HIGH_NAME_IMAGE_GEO_MATCH.value not in reason_codes:
                    reason_codes.append(ReasonCode.HIGH_NAME_IMAGE_GEO_MATCH.value)

            reason_summary = (
                f"High duplicate confidence ({top_sc.duplicate_confidence:.2f} >= "
                f"{self.duplicate_threshold:.2f}). Matched with existing outlet."
            )
        elif top_sc.duplicate_confidence <= self.genuine_threshold:
            decision = VerificationDecision.GENUINE
            reason_codes.append(ReasonCode.LOW_CONFIDENCE_GENUINE.value)
            reason_summary = (
                f"Duplicate confidence ({top_sc.duplicate_confidence:.2f}) is below genuine "
                f"threshold ({self.genuine_threshold:.2f}). Outlet appears distinct and genuine."
            )
        else:
            # Uncertain range [self.genuine_threshold, self.duplicate_threshold]
            # Human review is only requested when outlets are in close geographic vicinity
            # (<= 500m or GPS unknown) and the system is not certain.
            # Distant candidates (> 500m) without an exact image clone (i_sim < 0.90) are physically
            # distinct separate outlets and must be classified as GENUINE.
            if dist is not None and dist > 500.0 and (c.image_similarity is None or c.image_similarity < 0.90):
                decision = VerificationDecision.GENUINE
                reason_codes.append(ReasonCode.LOW_CONFIDENCE_GENUINE.value)
                reason_summary = (
                    f"Candidate outlet is physically distant ({dist:.0f}m > 500m) with distinct storefront and identity. "
                    f"Outlet is verified as GENUINE."
                )
            else:
                decision = VerificationDecision.NEEDS_REVIEW
                reason_codes.append(ReasonCode.INSUFFICIENT_EVIDENCE.value)
                reason_summary = (
                    f"Outlets are in close geographic vicinity ({dist if dist is not None else 'unknown'}m) "
                    f"with ambiguous evidence (confidence {top_sc.duplicate_confidence:.2f}). Manual review required."
                )

        return VerificationResponse(
            decision=decision,
            duplicate_confidence=top_sc.duplicate_confidence,
            evidence_coverage=top_sc.evidence_coverage,
            evidence_agreement=top_sc.evidence_agreement,
            matched_outlet=matched_outlet,
            evidence=evidence,
            matched_methods=top_sc.candidate.matched_methods,
            reason_codes=reason_codes,
            reason_summary=reason_summary,
            candidate_margin=margin,
            evidence_status="MATCHED_CANDIDATE",
            candidates_evaluated=len(scored_candidates),
        )


def get_decision_engine() -> DecisionEngine:
    """Factory getter for DecisionEngine."""
    return DecisionEngine()
