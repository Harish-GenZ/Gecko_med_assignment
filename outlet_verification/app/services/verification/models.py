from enum import Enum
from pydantic import BaseModel, Field

from app.services.authenticity.service import AuthenticityAssessment
from app.services.retrieval.models import CandidateEvidence


class VerificationDecision(str, Enum):
    """
    Deterministic classification of the submitted outlet.
    """
    DUPLICATE = "DUPLICATE"
    GENUINE = "GENUINE"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    REJECTED = "REJECTED"


class ReasonCode(str, Enum):
    """
    Structured reason codes deterministically explaining the verification outcome.
    """
    STRONG_MULTIMODAL_MATCH = "STRONG_MULTIMODAL_MATCH"
    STRONG_IMAGE_GEO_MATCH = "STRONG_IMAGE_GEO_MATCH"
    HIGH_NAME_IMAGE_GEO_MATCH = "HIGH_NAME_IMAGE_GEO_MATCH"
    NO_MATCHING_CANDIDATE = "NO_MATCHING_CANDIDATE"
    CONFLICTING_NAME_IMAGE = "CONFLICTING_NAME_IMAGE"
    AMBIGUOUS_TOP_CANDIDATES = "AMBIGUOUS_TOP_CANDIDATES"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    POSSIBLE_SAME_BRAND_DIFFERENT_LOCATION = "POSSIBLE_SAME_BRAND_DIFFERENT_LOCATION"
    WEAK_EVIDENCE_REQUIRES_REVIEW = "WEAK_EVIDENCE_REQUIRES_REVIEW"
    LOW_CONFIDENCE_GENUINE = "LOW_CONFIDENCE_GENUINE"
    NOT_A_RETAIL_STOREFRONT_IMAGE = "NOT_A_RETAIL_STOREFRONT_IMAGE"
    NON_OUTLET_NAME = "NON_OUTLET_NAME"
    SIGNBOARD_MISMATCH = "SIGNBOARD_MISMATCH"
    NAME_MISMATCH = "NAME_MISMATCH"
    FAKE_OUTLET_REJECTED = "FAKE_OUTLET_REJECTED"


class MatchedOutlet(BaseModel):
    """
    Summary profile of an existing matched outlet.
    """
    outlet_id: str
    name: str
    image_url: str
    latitude: float
    longitude: float


class VerificationEvidence(BaseModel):
    """
    Normalized multi-modal evidence components for the matched outlet.
    """
    name_similarity: float | None = None
    image_similarity: float | None = None
    distance_meters: float | None = None
    geo_proximity_score: float | None = None
    three_metric_average: float | None = None


class CandidateFlags(BaseModel):
    """
    Deterministic evidence flags evaluated per candidate.
    """
    strong_duplicate_evidence: bool = False
    conflicting_evidence: bool = False
    possible_same_brand_different_location: bool = False


class ScoredCandidate(BaseModel):
    """
    Evaluation metrics and confidence score for a candidate outlet.
    """
    candidate: CandidateEvidence
    geo_proximity_score: float | None = None
    three_metric_average: float | None = None
    is_duplicate_candidate: bool = False
    base_score: float
    evidence_coverage: float
    evidence_agreement: float
    duplicate_confidence: float
    flags: CandidateFlags = Field(default_factory=CandidateFlags)


class VerificationResponse(BaseModel):
    """
    Complete verification result for an outlet submission.
    Note: duplicate_confidence is an engineered evidence fusion score, NOT a calibrated probability.
    """
    decision: VerificationDecision
    duplicate_confidence: float = Field(
        ...,
        description="Engineered duplicate confidence score [0.0, 1.0]. NOT a calibrated probability.",
    )
    evidence_coverage: float = Field(
        ...,
        description="Proportion of available signal weights [0.0, 1.0].",
    )
    evidence_agreement: float = Field(
        ...,
        description="Inter-signal agreement score [0.0, 1.0] derived from weighted mean absolute deviation.",
    )
    matched_outlet: MatchedOutlet | None = Field(
        default=None,
        description="Top matching candidate outlet from the database, if any.",
    )
    evidence: VerificationEvidence | None = Field(
        default=None,
        description="Granular evidence profile of the top candidate.",
    )
    matched_methods: list[str] = Field(
        default_factory=list,
        description="Retrieval channels that surfaced the top candidate.",
    )
    reason_codes: list[str] = Field(
        default_factory=list,
        description="Deterministic structured reason codes explaining the decision.",
    )
    reason_summary: str = Field(
        ...,
        description="Deterministic human-readable explanation of the decision.",
    )
    candidate_margin: float | None = Field(
        default=None,
        description="Difference in confidence between best and second-best candidate.",
    )
    evidence_status: str | None = Field(
        default=None,
        description="Evidence status indicator (e.g., 'NO_MATCHING_CANDIDATE').",
    )
    candidates_evaluated: int = Field(
        default=0,
        description="Total number of candidates retrieved and evaluated.",
    )
    registered: bool = Field(
        default=False,
        description="True if a genuine outlet was registered and stored in PostgreSQL.",
    )
    registered_outlet_id: str | None = Field(
        default=None,
        description="UUID of the newly registered outlet in PostgreSQL.",
    )
    stored_image_url: str | None = Field(
        default=None,
        description="Public bucket storage URL of the uploaded storefront image.",
    )
    authenticity: AuthenticityAssessment | None = Field(
        default=None,
        description="Domain authenticity analysis (storefront visual classifier, signboard OCR, and name check).",
    )
    candidates: list[ScoredCandidate] = Field(
        default_factory=list,
        description="All evaluated candidate matches retrieved from database search.",
    )
