"""
Verification Service Package (Phase 5).
Multimodal Fusion & Confidence-Based Verification Decision Engine.
"""

from app.services.verification.decision import DecisionEngine, get_decision_engine
from app.services.verification.models import (
    CandidateFlags,
    MatchedOutlet,
    ReasonCode,
    ScoredCandidate,
    VerificationDecision,
    VerificationEvidence,
    VerificationResponse,
)
from app.services.verification.scoring import (
    CandidateScorer,
    compute_geo_proximity,
    get_candidate_scorer,
)
from app.services.verification.verification_service import (
    VerificationService,
    get_verification_service,
)

__all__ = [
    "VerificationDecision",
    "ReasonCode",
    "MatchedOutlet",
    "VerificationEvidence",
    "CandidateFlags",
    "ScoredCandidate",
    "VerificationResponse",
    "compute_geo_proximity",
    "CandidateScorer",
    "get_candidate_scorer",
    "DecisionEngine",
    "get_decision_engine",
    "VerificationService",
    "get_verification_service",
]
