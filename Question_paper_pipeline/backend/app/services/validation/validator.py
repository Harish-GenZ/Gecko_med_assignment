"""
Validation layer.

Validates extraction results and enriches them with:
- Review flags (needs_review)
- Confidence scores
- Post-processing corrections
"""

import re
import logging
from typing import List

from app.models.schemas import (
    Question, QuestionPaper, ExtractionResult, ReviewStatus, QuestionType
)
from app.core.config import OCR_MIN_CONFIDENCE, CLASSIFICATION_CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)

MIN_QUESTION_TEXT_LENGTH = 5   # Characters


def validate_and_flag(result: ExtractionResult) -> ExtractionResult:
    """
    Run validation checks on the extraction result.
    Flags low-confidence items for review.
    Returns mutated result (in-place modification is fine here).
    """
    notes = []

    for paper in result.question_papers:
        # Validate metadata
        if paper.metadata.confidence < 0.5:
            paper.metadata.status = ReviewStatus.NEEDS_REVIEW
            notes.append(
                f"Paper {paper.paper_index}: low metadata confidence "
                f"({paper.metadata.confidence:.2f}) — manual review needed."
            )

        # Validate questions
        valid_questions = []
        for q in paper.questions:
            # Skip empty questions
            if len(q.text.strip()) < MIN_QUESTION_TEXT_LENGTH:
                logger.debug(f"Dropping empty question {q.number}")
                continue

            # Flag low classification confidence
            if q.confidence < CLASSIFICATION_CONFIDENCE_THRESHOLD:
                q.status = ReviewStatus.NEEDS_REVIEW

            # Flag MCQ questions that are missing options
            if q.type == QuestionType.MCQ and not q.options:
                q.status = ReviewStatus.NEEDS_REVIEW
                notes.append(
                    f"Paper {paper.paper_index}, Q{q.number}: classified as MCQ "
                    f"but no options found — review needed."
                )

            valid_questions.append(q)

        paper.questions = valid_questions

        # Flag paper with no questions
        if not paper.questions:
            paper.metadata.status = ReviewStatus.NEEDS_REVIEW
            notes.append(f"Paper {paper.paper_index}: no questions extracted.")

    result.processing_notes.extend(notes)
    return result
