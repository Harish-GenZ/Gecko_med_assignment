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


class ReviewStatus(str, Enum):
    OK           = "ok"
    NEEDS_REVIEW = "needs_review"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class MCQOption(BaseModel):
    label: str   # "a", "b", "c", "d"
    text:  str


class Question(BaseModel):
    number:     str
    text:       str
    type:       QuestionType
    part:       Optional[str] = None               # e.g. "PART-A", "PART-B", "Section A"
    heading:    Optional[str] = None               # Section/prompt heading e.g. "Write short notes on:", "Explain why?"
    options:    Optional[List[MCQOption]] = None   # Only for MCQ
    marks:      Optional[str] = None               # e.g. "11", "25", "9", "5"
    confidence: float = Field(ge=0.0, le=1.0)
    status:     ReviewStatus = ReviewStatus.OK


class ExamMetadata(BaseModel):
    university:   Optional[str]  = None
    exam_month:   Optional[str]  = None
    exam_year:    Optional[str]  = None
    session_code: Optional[str]  = None
    subject:      Optional[str]  = None
    paper_code:   Optional[str]  = None
    max_marks:    Optional[str]  = None
    duration:     Optional[str]  = None
    confidence:   float = Field(default=0.0, ge=0.0, le=1.0)
    status:       ReviewStatus = ReviewStatus.OK


class QuestionPaper(BaseModel):
    paper_index:  int
    page_range:   List[int]          # [start_page, end_page] 1-indexed
    metadata:     ExamMetadata
    questions:    List[Question]
    instructions: Optional[List[str]] = None  # Instructions separated from header & questions
    parts:        Optional[List[str]] = None  # Distinct parts detected e.g. ["PART-A", "PART-B"]
    raw_text:     Optional[str] = None  # Full extracted text for debugging


class PageInfo(BaseModel):
    page_number: int
    type:        PageType
    ocr_used:    bool
    char_count:  int


# ---------------------------------------------------------------------------
# Top-level job / response models
# ---------------------------------------------------------------------------

class JobCreateResponse(BaseModel):
    job_id:   str
    status:   JobStatus
    filename: str
    message:  str


class JobStatusResponse(BaseModel):
    job_id:         str
    status:         JobStatus
    filename:       str
    progress:       int = Field(description="Percentage 0–100")
    current_step:   str = ""
    error:          Optional[str] = None
    pages_info:     List[PageInfo] = []
    result:         Optional[ExtractionResult] = None


class ExtractionResult(BaseModel):
    document:         str
    total_pages:      int
    question_papers:  List[QuestionPaper]
    processing_notes: List[str] = []


# Forward reference resolution
JobStatusResponse.model_rebuild()
