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
        review_threshold: float | None = None,
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
        self.review_threshold = (
            review_threshold
            if review_threshold is not None
            else getattr(settings, "VERIFICATION_REVIEW_THRESHOLD", 0.65)
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
        Enforces the strong 3-metric average rule:
          - 0% to 65%: GENUINE
          - 65% to 75%: MANUAL REVIEW REQUIRED (NEEDS_REVIEW)
          - Above 75%: DUPLICATE
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
        # Step 2: Sort and rank candidates by three_metric_average / duplicate_confidence descending
        # ----------------------------------------------------------------------
        ranked = sorted(
            scored_candidates,
            key=lambda sc: (
                sc.three_metric_average if sc.three_metric_average is not None else -1.0,
                sc.duplicate_confidence,
                sc.base_score,
                sc.evidence_coverage,
            ),
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
            three_metric_average=top_sc.three_metric_average,
        )

        reason_codes: list[str] = []
        flags = top_sc.flags

        # Record secondary observation flags
        if flags.possible_same_brand_different_location:
            reason_codes.append(ReasonCode.POSSIBLE_SAME_BRAND_DIFFERENT_LOCATION.value)
        if flags.strong_duplicate_evidence:
            reason_codes.append(ReasonCode.STRONG_IMAGE_GEO_MATCH.value)

        c = top_sc.candidate

        # ----------------------------------------------------------------------
        # Primary Evaluation: Strong 3-Metric Combined Average Rule
        # Evaluates Distance Proximity + Visual Storefront Image + Name Match
        # Rule:
        #   - 0.0 to 0.65 (< 65%): GENUINE
        #   - 0.65 to 0.75 (65% - 75%): MANUAL REVIEW REQUIRED (NEEDS_REVIEW)
        #   - Above 0.75 (>= 75%): DUPLICATE
        # ----------------------------------------------------------------------
        has_all_three_signals = (
            top_sc.three_metric_average is not None
            and top_sc.geo_proximity_score is not None
            and c.image_similarity is not None
            and c.name_similarity is not None
        )

        if has_all_three_signals:
            avg_score = top_sc.three_metric_average

            # Safeguard Check: Conflicting evidence (e.g. high name similarity but low image similarity)
            if flags.conflicting_evidence:
                reason_codes.append(ReasonCode.CONFLICTING_NAME_IMAGE.value)
                logger.info(
                    "Outlet %s flagged for conflicting evidence (high name similarity with low image similarity).",
                    c.outlet_id,
                )
                return VerificationResponse(
                    decision=VerificationDecision.NEEDS_REVIEW,
                    duplicate_confidence=top_sc.duplicate_confidence,
                    evidence_coverage=top_sc.evidence_coverage,
                    evidence_agreement=top_sc.evidence_agreement,
                    matched_outlet=matched_outlet,
                    evidence=evidence,
                    matched_methods=c.matched_methods,
                    reason_codes=reason_codes,
                    reason_summary="High name similarity conflicts with low visual image similarity. Manual review required.",
                    candidate_margin=margin,
                    evidence_status="CONFLICTING_EVIDENCE",
                    candidates_evaluated=len(scored_candidates),
                    candidates=ranked,
                )

            # Strong 3-Tier Threshold Evaluation
            if avg_score >= self.duplicate_threshold:
                decision = VerificationDecision.DUPLICATE
                if ReasonCode.STRONG_MULTIMODAL_MATCH.value not in reason_codes:
                    reason_codes.append(ReasonCode.STRONG_MULTIMODAL_MATCH.value)
                if margin is not None and margin < self.min_margin:
                    reason_codes.append(ReasonCode.AMBIGUOUS_TOP_CANDIDATES.value)

                margin_note = (
                    f" (Top candidate margin: {margin:.4f} compared to runner-up '{runner_up_sc.candidate.name}')."
                    if runner_up_sc is not None and margin is not None and margin < self.min_margin
                    else ""
                )
                reason_summary = (
                    f"3-Metric Average ({avg_score * 100:.1f}%) crosses the 75% duplicate threshold "
                    f"across Distance Proximity ({(top_sc.geo_proximity_score or 0) * 100:.1f}%), "
                    f"Visual Image Similarity ({(c.image_similarity or 0) * 100:.1f}%), "
                    f"and Name Similarity ({(c.name_similarity or 0) * 100:.1f}%). "
                    f"Matched duplicate with existing outlet '{c.name}'.{margin_note}"
                )
            elif avg_score >= self.review_threshold:
                decision = VerificationDecision.NEEDS_REVIEW
                if ReasonCode.INSUFFICIENT_EVIDENCE.value not in reason_codes:
                    reason_codes.append(ReasonCode.INSUFFICIENT_EVIDENCE.value)
                if margin is not None and margin < self.min_margin:
                    reason_codes.append(ReasonCode.AMBIGUOUS_TOP_CANDIDATES.value)

                reason_summary = (
                    f"3-Metric Average ({avg_score * 100:.1f}%) is in the manual review range "
                    f"(65.0% - 75.0%) compared to existing outlet '{c.name}'. "
                    f"Distance Proximity: {(top_sc.geo_proximity_score or 0) * 100:.1f}%, "
                    f"Image Similarity: {(c.image_similarity or 0) * 100:.1f}%, "
                    f"Name Similarity: {(c.name_similarity or 0) * 100:.1f}%. "
                    f"Evidence is borderline. Manual review required."
                )
            else:
                decision = VerificationDecision.GENUINE
                if ReasonCode.LOW_CONFIDENCE_GENUINE.value not in reason_codes:
                    reason_codes.append(ReasonCode.LOW_CONFIDENCE_GENUINE.value)

                reason_summary = (
                    f"3-Metric Average ({avg_score * 100:.1f}%) is below 65.0% "
                    f"(Distance Proximity: {(top_sc.geo_proximity_score or 0) * 100:.1f}%, "
                    f"Image Similarity: {(c.image_similarity or 0) * 100:.1f}%, "
                    f"Name Similarity: {(c.name_similarity or 0) * 100:.1f}%). "
                    f"Outlet is verified as GENUINE."
                )

            return VerificationResponse(
                decision=decision,
                duplicate_confidence=top_sc.duplicate_confidence,
                evidence_coverage=top_sc.evidence_coverage,
                evidence_agreement=top_sc.evidence_agreement,
                matched_outlet=matched_outlet,
                evidence=evidence,
                matched_methods=c.matched_methods,
                reason_codes=reason_codes,
                reason_summary=reason_summary,
                candidate_margin=margin,
                evidence_status="MATCHED_CANDIDATE",
                candidates_evaluated=len(scored_candidates),
                candidates=ranked,
            )

        # ----------------------------------------------------------------------
        # Fallback Processing: Incomplete Signals (Missing Distance, Image, or Name)
        # ----------------------------------------------------------------------

        # Safeguard Check: Conflicting Evidence
        if flags.conflicting_evidence:
            reason_codes.append(ReasonCode.CONFLICTING_NAME_IMAGE.value)
            logger.info(
                "Outlet %s flagged for conflicting evidence (high name similarity with low image similarity).",
                c.outlet_id,
            )
            return VerificationResponse(
                decision=VerificationDecision.NEEDS_REVIEW,
                duplicate_confidence=top_sc.duplicate_confidence,
                evidence_coverage=top_sc.evidence_coverage,
                evidence_agreement=top_sc.evidence_agreement,
                matched_outlet=matched_outlet,
                evidence=evidence,
                matched_methods=c.matched_methods,
                reason_codes=reason_codes,
                reason_summary="High name similarity conflicts with low visual image similarity. Manual review required.",
                candidate_margin=margin,
                evidence_status="CONFLICTING_EVIDENCE",
                candidates_evaluated=len(scored_candidates),
                candidates=ranked,
            )

        # Safeguard Check: Ambiguous Top Candidates when signals are incomplete
        if (
            top_sc.duplicate_confidence >= self.duplicate_threshold
            and runner_up_sc is not None
            and margin is not None
            and margin < self.min_margin
        ):
            reason_codes.append(ReasonCode.AMBIGUOUS_TOP_CANDIDATES.value)
            logger.info(
                "Top candidates are ambiguous (margin %.4f < %.2f) under incomplete signals. Marking as NEEDS_REVIEW.",
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
                matched_methods=c.matched_methods,
                reason_codes=reason_codes,
                reason_summary=(
                    f"Top candidate ({top_sc.duplicate_confidence:.2f}) and runner-up "
                    f"({runner_up_sc.duplicate_confidence:.2f}) have ambiguous margin "
                    f"({margin:.4f} < {self.min_margin:.2f}) with incomplete signals. Manual review required."
                ),
                candidate_margin=margin,
                evidence_status="AMBIGUOUS_CANDIDATES",
                candidates_evaluated=len(scored_candidates),
                candidates=ranked,
            )

        # Safeguard Check: Insufficient Coverage
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
                    matched_methods=c.matched_methods,
                    reason_codes=reason_codes,
                    reason_summary=(
                        f"Evidence coverage ({top_sc.evidence_coverage:.2f}) is below minimum "
                        f"required coverage ({self.min_coverage:.2f}) for duplicate confirmation."
                    ),
                    candidate_margin=margin,
                    evidence_status="LOW_COVERAGE",
                    candidates_evaluated=len(scored_candidates),
                    candidates=ranked,
                )

        # Safeguard Check: Weak / Inconclusive Signal Combinations in close vicinity
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
                        matched_methods=c.matched_methods,
                        reason_codes=reason_codes,
                        reason_summary="Image evidence is unavailable and name similarity is moderate/ambiguous in close vicinity. Manual review required.",
                        candidate_margin=margin,
                        evidence_status="WEAK_AMBIGUOUS_EVIDENCE",
                        candidates_evaluated=len(scored_candidates),
                        candidates=ranked,
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
                        matched_methods=c.matched_methods,
                        reason_codes=reason_codes,
                        reason_summary="Name evidence is unavailable and image similarity is moderate/ambiguous in close vicinity. Manual review required.",
                        candidate_margin=margin,
                        evidence_status="WEAK_AMBIGUOUS_EVIDENCE",
                        candidates_evaluated=len(scored_candidates),
                        candidates=ranked,
                    )

        # General threshold decision on available duplicate_confidence
        if top_sc.duplicate_confidence >= self.duplicate_threshold:
            decision = VerificationDecision.DUPLICATE
            if ReasonCode.STRONG_MULTIMODAL_MATCH.value not in reason_codes:
                reason_codes.append(ReasonCode.STRONG_MULTIMODAL_MATCH.value)
            reason_summary = (
                f"Confidence ({top_sc.duplicate_confidence * 100:.1f}%) crosses the 75% duplicate threshold. "
                f"Matched duplicate with existing outlet '{c.name}'."
            )
        elif top_sc.duplicate_confidence >= self.review_threshold:
            decision = VerificationDecision.NEEDS_REVIEW
            if ReasonCode.INSUFFICIENT_EVIDENCE.value not in reason_codes:
                reason_codes.append(ReasonCode.INSUFFICIENT_EVIDENCE.value)
            reason_summary = (
                f"Confidence ({top_sc.duplicate_confidence * 100:.1f}%) is in the manual review range (65.0% - 75.0%). "
                f"Manual review required."
            )
        else:
            decision = VerificationDecision.GENUINE
            if ReasonCode.LOW_CONFIDENCE_GENUINE.value not in reason_codes:
                reason_codes.append(ReasonCode.LOW_CONFIDENCE_GENUINE.value)
            reason_summary = (
                f"Confidence ({top_sc.duplicate_confidence * 100:.1f}%) is below 65.0%. "
                f"Outlet is verified as GENUINE."
            )

        return VerificationResponse(
            decision=decision,
            duplicate_confidence=top_sc.duplicate_confidence,
            evidence_coverage=top_sc.evidence_coverage,
            evidence_agreement=top_sc.evidence_agreement,
            matched_outlet=matched_outlet,
            evidence=evidence,
            matched_methods=c.matched_methods,
            reason_codes=reason_codes,
            reason_summary=reason_summary,
            candidate_margin=margin,
            evidence_status="MATCHED_CANDIDATE",
            candidates_evaluated=len(scored_candidates),
            candidates=ranked,
        )


def get_decision_engine() -> DecisionEngine:
    """Factory getter for DecisionEngine."""
    return DecisionEngine()
