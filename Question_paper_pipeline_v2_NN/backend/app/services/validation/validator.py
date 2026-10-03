"""
Validation layer.

Enriches extraction results with:
- Multi-dimensional confidence scoring & quality audit
- Structural anomaly detection
- Review flags (needs_review, ConfidenceLevel)
- Visual verification trigger flags (needs_visual_verification, reasons)
- Page-level audits
"""

import logging
from typing import List, Dict, Any, Optional

from app.models.schemas import (
    Question, QuestionPaper, ExtractionResult, ReviewStatus,
    QuestionType, ConfidenceLevel, AnomalySeverity
)
from app.services.validation.quality_audit import audit_paper, generate_page_audits

logger = logging.getLogger(__name__)

MIN_QUESTION_TEXT_LENGTH = 5   # Characters


def validate_and_flag(
    result: ExtractionResult,
    pages_list: Optional[List[Dict[str, Any]]] = None,
    page_states: Optional[Dict[int, Any]] = None,
    vision_results: Optional[Dict[int, Any]] = None,
) -> ExtractionResult:
    """
    Run validation checks on the extraction result.
    Applies quality audits and flags low-confidence items for review.
    """
    notes = []
    overall_needs_vision = False
    overall_vision_reasons: List[str] = []
    manual_review_needed = False

    for paper in result.question_papers:
        # Filter completely empty questions
        valid_questions = []
        for q in paper.questions:
            if len(q.text.strip()) < MIN_QUESTION_TEXT_LENGTH:
                logger.debug(f"Dropping empty question {q.number}")
                continue
            valid_questions.append(q)
        paper.questions = valid_questions

        # Propagate paper vision trigger
        if paper.needs_visual_verification:
            overall_needs_vision = True
            for r in paper.visual_verification_reasons:
                if r not in overall_vision_reasons:
                    overall_vision_reasons.append(r)

        if paper.needs_manual_review:
            manual_review_needed = True

        # Collect critical/high anomalies into processing notes
        for anom in paper.anomalies:
            if anom.severity in (AnomalySeverity.HIGH, AnomalySeverity.CRITICAL):
                notes.append(f"Paper {paper.paper_index + 1}: [{anom.severity.value.upper()}] {anom.reason}")

        if paper.confidence_level in (ConfidenceLevel.LOW, ConfidenceLevel.REVIEW_REQUIRED):
            notes.append(
                f"Paper {paper.paper_index + 1}: Overall confidence {paper.confidence.overall_confidence:.1%} "
                f"({paper.confidence_level.value}) — visual review recommended."
            )

    result.needs_visual_verification = overall_needs_vision
    result.visual_verification_reasons = overall_vision_reasons

    verified_pages = []
    if vision_results:
        for pg_idx, v_res in vision_results.items():
            verified_pages.append(pg_idx + 1)
            if getattr(v_res, "needs_manual_review", False):
                manual_review_needed = True

    result.vision_verified_pages = sorted(verified_pages)
    result.needs_manual_review = manual_review_needed

    if pages_list:
        result.page_audits = generate_page_audits(
            paper_pages=pages_list,
            papers=result.question_papers,
            page_states=page_states,
            vision_results=vision_results,
        )

    result.processing_notes.extend(notes)
    return result
