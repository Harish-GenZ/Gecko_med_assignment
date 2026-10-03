"""
Pydantic schemas for the Question Paper Extraction Pipeline.
"""

from __future__ import annotations
from enum import Enum
from typing import Optional, List, Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PageType(str, Enum):
    DIGITAL = "digital"
    SCANNED = "scanned"
    MIXED   = "mixed"


class QuestionType(str, Enum):
    ESSAY              = "Essay"
    SHORT_NOTES        = "Short Notes"
    VERY_SHORT_ANSWERS = "Very Short Answers"
    MCQ                = "MCQ"
    UNKNOWN            = "Unknown"


class JobStatus(str, Enum):
    QUEUED     = "queued"
    PROCESSING = "processing"
    COMPLETED  = "completed"
    FAILED     = "failed"
    CANCELLED  = "cancelled"


class ReviewStatus(str, Enum):
    OK           = "ok"
    NEEDS_REVIEW = "needs_review"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class MCQOption(BaseModel):
    label: str   # "a", "b", "c", "d"
    text:  str


class AnomalySeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class StructuralAnomaly(BaseModel):
    type: str # e.g. "question_like_text_in_instructions", "question_numbering_gap"
    severity: AnomalySeverity
    text_preview: Optional[str] = None
    reason: str


class PageProcessingState(str, Enum):
    OCR_PROCESSING   = "OCR_PROCESSING"
    OCR_COMPLETE     = "OCR_COMPLETE"
    AUDIT_COMPLETE   = "AUDIT_COMPLETE"
    VISION_PENDING    = "VISION_PENDING"
    VISION_PROCESSING = "VISION_PROCESSING"
    VISION_VERIFIED   = "VISION_VERIFIED"
    VISION_CORRECTED  = "VISION_CORRECTED"
    VISION_FAILED     = "VISION_FAILED"
    FINALIZED        = "FINALIZED"


class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


# ---------------------------------------------------------------------------
# Vision Verification Structured Models
# ---------------------------------------------------------------------------

class VisionCorrectionType(str, Enum):
    MOVE_TEXT        = "move_text"
    SPLIT_QUESTION   = "split_question"
    ADD_QUESTION     = "add_question"
    REMOVE_QUESTION  = "remove_question"
    MODIFY_QUESTION  = "modify_question"
    UPDATE_BOUNDARY  = "update_boundary"
    FIX_NUMBERING    = "fix_numbering"
    REASSIGN_SECTION = "reassign_section"
    OTHER            = "other"


class VisionCorrectionItem(BaseModel):
    type: str  # e.g. "move_text", "split_question"
    source: Optional[str] = None  # e.g. "instructions", "questions", "header"
    target: Optional[str] = None  # e.g. "questions", "instructions"
    question_number: Optional[str] = None
    section: Optional[str] = None
    reason: Optional[str] = None


class VisionQuestionCorrection(BaseModel):
    question_number: str
    action: str  # "ADD", "MODIFY", "REMOVE", "MOVE", "REPLACE"
    section: Optional[str] = None
    text: Optional[str] = None
    heading: Optional[str] = None
    marks: Optional[str] = None
    options: Optional[List[MCQOption]] = None
    type: Optional[str] = None
    confidence: float = 0.90


class VisionInstructionCorrection(BaseModel):
    action: str  # "REMOVE", "MODIFY", "ADD"
    text: Optional[str] = None
    index: Optional[int] = None
    reason: Optional[str] = None


class VisionMetadataCorrection(BaseModel):
    field: str
    value: Optional[str] = None
    confidence: float = 0.90


class VisionVerificationResult(BaseModel):
    page_number: int
    is_structure_correct: bool = True
    confidence: float = 0.90  # vision_confidence
    corrections: List[VisionCorrectionItem] = []
    question_corrections: List[VisionQuestionCorrection] = []
    instruction_corrections: List[VisionInstructionCorrection] = []
    metadata_corrections: List[VisionMetadataCorrection] = []
    instructions: Optional[List[str]] = None
    needs_manual_review: bool = False
    review_reason: Optional[str] = None
    raw_explanation: Optional[str] = None


class QuestionConfidenceBreakdown(BaseModel):
    ocr: Optional[float] = None
    boundary: float
    classification: float
    overall: float


class PaperConfidenceBreakdown(BaseModel):
    ocr_confidence: Optional[float] = None
    ocr_confidence_available: bool = True
    structure_confidence: float
    segmentation_confidence: float
    metadata_confidence: float
    classification_confidence: float
    consistency_confidence: float
    overall_confidence: float


class PageConfidenceAudit(BaseModel):
    page: int
    confidence: PaperConfidenceBreakdown
    status: ConfidenceLevel
    anomalies: List[StructuralAnomaly] = []
    needs_visual_verification: bool = False
    visual_verification_reasons: List[str] = []
    processing_state: PageProcessingState = PageProcessingState.FINALIZED
    vision_confidence: Optional[float] = None
    vision_status: Optional[str] = None
    vision_error: Optional[str] = None
    needs_manual_review: bool = False
    corrections_applied: int = 0


class Question(BaseModel):
    number:               str
    text:                 str
    type:                 QuestionType
    part:                 Optional[str] = None               # e.g. "PART-A", "PART-B", "Section A"
    heading:              Optional[str] = None               # Section/prompt heading e.g. "Write short notes on:", "Explain why?"
    options:              Optional[List[MCQOption]] = None   # Only for MCQ
    marks:                Optional[str] = None               # e.g. "11", "25", "9", "5"
    confidence:           float = Field(default=0.95, ge=0.0, le=1.0)      # Overall question confidence (0.0 - 1.0)
    confidence_breakdown: Optional[QuestionConfidenceBreakdown] = None
    status:               ReviewStatus = ReviewStatus.OK
    confidence_level:     Optional[ConfidenceLevel] = None
    anomalies:            List[StructuralAnomaly] = []


class ExamMetadata(BaseModel):
    university:     Optional[str]  = None
    exam_month:     Optional[str]  = None
    exam_year:      Optional[str]  = None
    session_code:   Optional[str]  = None
    subject:        Optional[str]  = None
    paper_code:     Optional[str]  = None
    max_marks:      Optional[str]  = None
    duration:       Optional[str]  = None
    confidence:     float = Field(default=0.0, ge=0.0, le=1.0)
    status:         ReviewStatus = ReviewStatus.OK
    missing_fields: List[str] = []


class QuestionPaper(BaseModel):
    paper_index:                  int
    page_range:                   List[int]          # [start_page, end_page] 1-indexed
    metadata:                     ExamMetadata
    questions:                    List[Question]
    instructions:                 Optional[List[str]] = None  # Instructions separated from header & questions
    parts:                        Optional[List[str]] = None  # Distinct parts detected e.g. ["PART-A", "PART-B"]
    raw_text:                     Optional[str] = None  # Full extracted text for debugging
    confidence:                   Optional[PaperConfidenceBreakdown] = None
    confidence_level:             ConfidenceLevel = ConfidenceLevel.HIGH
    needs_visual_verification:   bool = False
    visual_verification_reasons: List[str] = []
    anomalies:                    List[StructuralAnomaly] = []
    vision_verified:              bool = False
    vision_confidence:            Optional[float] = None
    vision_status:                Optional[str] = None
    needs_manual_review:          bool = False


class PageInfo(BaseModel):
    page_number:      int
    type:             PageType
    ocr_used:         bool
    char_count:       int
    processing_state: PageProcessingState = PageProcessingState.FINALIZED


# ---------------------------------------------------------------------------
# Top-level job / response models
# ---------------------------------------------------------------------------

class JobCreateResponse(BaseModel):
    job_id:   str
    status:   JobStatus
    filename: str
    message:  str


class WorkerInfo(BaseModel):
    worker_pid:      int
    page_number:     int
    status:          str
    elapsed_seconds: Optional[float] = None


class JobStatusResponse(BaseModel):
    job_id:         str
    status:         JobStatus
    filename:       str
    progress:       int = Field(description="Percentage 0–100")
    current_step:   str = ""
    error:          Optional[str] = None
    pages_info:     List[PageInfo] = []
    result:         Optional[ExtractionResult] = None
    workers_info:   Optional[List[WorkerInfo]] = None


class ExtractionResult(BaseModel):
    document:                     str
    total_pages:                  int
    question_papers:              List[QuestionPaper]
    page_audits:                  List[PageConfidenceAudit] = []
    processing_notes:             List[str] = []
    needs_visual_verification:   bool = False
    visual_verification_reasons: List[str] = []
    vision_verified_pages:        List[int] = []
    needs_manual_review:          bool = False


# Forward reference resolution
JobStatusResponse.model_rebuild()

