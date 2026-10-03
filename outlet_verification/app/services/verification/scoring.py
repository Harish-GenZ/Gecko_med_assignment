import math
from typing import Any

from app.config import get_settings
from app.services.retrieval.models import CandidateEvidence
from app.services.verification.models import (
    CandidateFlags,
    ScoredCandidate,
)


def compute_geo_proximity(
    distance_meters: float | None,
    scale_meters: float = 100.0,
) -> float | None:
    """
    Transforms physical distance in meters into a continuous proximity score in [0.0, 1.0]
    using an exponential decay function:
        G = exp(-d / D)
    where:
        d = distance_meters
        D = scale_meters (default: 100.0m)
    
    If distance is None, returns None (missing signal).
    """
    if distance_meters is None:
        return None
    d = max(0.0, float(distance_meters))
    scale = max(1.0, float(scale_meters))
    return round(math.exp(-d / scale), 4)


class CandidateScorer:
    """
    Computes evidence fusion scores, signal agreement, coverage, and duplicate confidence
    for candidate outlets. Handles missing signals via dynamic weight renormalization.
    """

    def __init__(
        self,
        name_weight: float | None = None,
        image_weight: float | None = None,
        geo_weight: float | None = None,
        geo_scale_meters: float | None = None,
        strong_image_threshold: float | None = None,
        strong_image_geo_meters: float | None = None,
    ) -> None:
        settings = get_settings()
        self.name_weight = (
            name_weight if name_weight is not None else settings.VERIFICATION_NAME_WEIGHT
        )
        self.image_weight = (
            image_weight if image_weight is not None else settings.VERIFICATION_IMAGE_WEIGHT
        )
        self.geo_weight = (
            geo_weight if geo_weight is not None else settings.VERIFICATION_GEO_WEIGHT
        )
        self.geo_scale_meters = (
            geo_scale_meters
            if geo_scale_meters is not None
            else settings.VERIFICATION_GEO_SCALE_METERS
        )
        self.strong_image_threshold = (
            strong_image_threshold
            if strong_image_threshold is not None
            else settings.VERIFICATION_STRONG_IMAGE_THRESHOLD
        )
        self.strong_image_geo_meters = (
            strong_image_geo_meters
            if strong_image_geo_meters is not None
            else settings.VERIFICATION_STRONG_IMAGE_GEO_METERS
        )

    def score_candidate(self, candidate: CandidateEvidence) -> ScoredCandidate:
        """
        Calculates multimodal fusion, agreement, coverage, and duplicate confidence for a single candidate.
        Missing signals are excluded from both numerator and denominator (weight renormalization).
        """
        # 1. Geographic proximity score
        geo_proximity = compute_geo_proximity(
            candidate.distance_meters,
            scale_meters=self.geo_scale_meters,
        )

        # 2. Collect available signals and their configured weights
        signals: list[tuple[float, float]] = []

        if candidate.name_similarity is not None:
            n_val = max(0.0, min(1.0, float(candidate.name_similarity)))
            signals.append((n_val, self.name_weight))

        if candidate.image_similarity is not None:
            i_val = max(0.0, min(1.0, float(candidate.image_similarity)))
            signals.append((i_val, self.image_weight))

        if geo_proximity is not None:
            g_val = max(0.0, min(1.0, float(geo_proximity)))
            signals.append((g_val, self.geo_weight))

        # 3. Evidence Coverage
        total_configured_weight = self.name_weight + self.image_weight + self.geo_weight
        available_weight_sum = sum(weight for _, weight in signals)

        coverage = (
            available_weight_sum / total_configured_weight
            if total_configured_weight > 0
            else 0.0
        )
        coverage = max(0.0, min(1.0, coverage))

        # 4. Base Multimodal Score (Missing-Signal Weight Renormalization)
        if available_weight_sum > 0:
            base_score = sum(score * weight for score, weight in signals) / available_weight_sum
            base_score = max(0.0, min(1.0, base_score))
        else:
            base_score = 0.0

        # 5. Signal Agreement (Weighted Mean Absolute Deviation)
        if len(signals) <= 1:
            # Single available signal or no signals:
            # With 1 signal there is no conflict; agreement is 1.0 (or 0.0 if empty).
            agreement = 1.0 if len(signals) == 1 else 0.0
        else:
            weighted_mean = base_score
            disagreement = (
                sum(weight * abs(score - weighted_mean) for score, weight in signals)
                / available_weight_sum
            )
            agreement = max(0.0, min(1.0, 1.0 - disagreement))

        # 6. Duplicate Confidence (Engineered Evidence Fusion)
        confidence = base_score * (0.70 + 0.30 * coverage) * agreement
        duplicate_confidence = max(0.0, min(1.0, confidence))

        # Storefront Category & Physical Distance Gating:
        # In retail medical shop verification, any two storefronts have baseline CLIP visual
        # similarity (~0.50-0.70) simply from common retail features (shelves, counter, signboards).
        # When an outlet is physically distant (dist > 500m), it cannot be a physical duplicate
        # unless there is an exact photographic clone (i_sim >= 0.90) indicating photo reuse fraud.
        # For non-cloned images across distant locations, duplicate confidence is attenuated by distance.
        dist = candidate.distance_meters
        i_sim = candidate.image_similarity
        if dist is not None and dist > 500.0:
            if i_sim is None or i_sim < 0.90:
                distance_factor = max(0.0, min(1.0, 500.0 / dist))
                duplicate_confidence = duplicate_confidence * distance_factor

        # 7. Strong Evidence & Safeguard Flags
        n_sim = candidate.name_similarity
        i_sim = candidate.image_similarity
        dist = candidate.distance_meters

        strong_duplicate = False
        if i_sim is not None and dist is not None:
            # Rule 11.1: High image, close distance, moderate name
            if (
                i_sim >= self.strong_image_threshold
                and dist <= self.strong_image_geo_meters
                and (n_sim is not None and n_sim >= 0.60)
            ):
                strong_duplicate = True
            # Rule 11.2: Very high image + close distance even if names differ
            elif i_sim >= 0.95 and dist <= self.strong_image_geo_meters:
                strong_duplicate = True

        # Rule 11.3: Strong name but conflicting image
        conflicting_evidence = False
        if n_sim is not None and i_sim is not None:
            if n_sim >= 0.90 and i_sim <= 0.50:
                conflicting_evidence = True

        # Rule 11.4: Same name but far away (possible different branch)
        same_brand_diff_location = False
        if n_sim is not None and dist is not None:
            if n_sim >= 0.90 and dist >= 1000.0:
                same_brand_diff_location = True

        flags = CandidateFlags(
            strong_duplicate_evidence=strong_duplicate,
            conflicting_evidence=conflicting_evidence,
            possible_same_brand_different_location=same_brand_diff_location,
        )

        return ScoredCandidate(
            candidate=candidate,
            geo_proximity_score=round(geo_proximity, 4) if geo_proximity is not None else None,
            base_score=round(base_score, 4),
            evidence_coverage=round(coverage, 4),
            evidence_agreement=round(agreement, 4),
            duplicate_confidence=round(duplicate_confidence, 4),
            flags=flags,
        )


def get_candidate_scorer() -> CandidateScorer:
    """Factory getter for CandidateScorer."""
    return CandidateScorer()
