"""
Question type classifier.

Classifies each question into one of:
  - Essay
  - Short Notes
  - Very Short Answers
  - MCQ

Approach: Rule-based classification, calibrated on actual dataset patterns.
No LLM or heavy ML needed — the classification signals are explicit
(section headers, marks, MCQ options, keyword cues).

Priority order (in case of conflict):
  1. MCQ — if options (a)(b)(c)(d) are present → always MCQ
  2. Marks-based classification (most reliable signal in this dataset)
  3. Section header context (inherited from paper section)
  4. Keyword matching (write short notes / explain / define)
  5. Question length heuristic (last resort)
"""

import re
import logging
from typing import Optional, Tuple, Dict, Any

from app.models.schemas import QuestionType, ReviewStatus

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Confidence levels
# ---------------------------------------------------------------------------
HIGH_CONF   = 0.95
MEDIUM_CONF = 0.80
LOW_CONF    = 0.55
FALLBACK_CONF = 0.45


# ---------------------------------------------------------------------------
# Keyword patterns (from dataset observation)
# ---------------------------------------------------------------------------

# Essay-type keywords: long explanatory questions
_ESSAY_RE = re.compile(
    r'\b(describe\s+in\s+detail|enumerate\s+and\s+describe|explain\s+in\s+detail|'
    r'discuss\s+in\s+detail|write\s+an?\s+essay|write\s+in\s+detail|'
    r'classify\s+and\s+describe|what\s+is\s+.*\s+describe|write\s+about\s+.*\s+and\s+describe)\b',
    re.I
)

# Short notes / Short answer keywords
_SHORT_NOTES_RE = re.compile(
    r'\b(write\s+short\s+note[s]?|brief\s+note[s]?|short\s+note[s]?|'
    r'write\s+about|write\s+briefly|enumerate|list|mention)\b',
    re.I
)

# Very short answer keywords
_VERY_SHORT_RE = re.compile(
    r'\b(define|what\s+is|what\s+are|name|give\s+the|state|'
    r'expand|abbreviate|full\s+form|normal\s+value)\b',
    re.I
)

# Marks thresholds (calibrated to the 3 universities' actual mark distributions)
# Essay: 10–15 marks, Short Notes: 5–7 marks, VSA: 1–4 marks, MCQ: 1 mark
def _classify_by_marks(marks_str: Optional[str]) -> Optional[Tuple[QuestionType, float]]:
    if not marks_str:
        return None

    # Parse numeric value from marks string like "15", "5x5=25", "2½"
    num_m = re.search(r'(\d+)', marks_str)
    if not num_m:
        return None

    marks = int(num_m.group(1))

    # Handle "NxM" format (e.g., "5x5=25" → each question is 5 marks)
    multi = re.search(r'(\d+)\s*[x×]\s*(\d+)', marks_str, re.I)
    if multi:
        marks = int(multi.group(1))  # Per-question marks

    if marks == 1:
        return QuestionType.MCQ, HIGH_CONF
    elif marks <= 4:
        return QuestionType.VERY_SHORT_ANSWERS, HIGH_CONF
    elif marks <= 7:
        return QuestionType.SHORT_NOTES, HIGH_CONF
    elif marks >= 10:
        return QuestionType.ESSAY, HIGH_CONF

    return None


# Section context → default question type mapping
_SECTION_DEFAULTS: Dict[str, QuestionType] = {
    "long question":        QuestionType.ESSAY,
    "long essay":           QuestionType.ESSAY,
    "structured long essay":QuestionType.ESSAY,
    "problem based":        QuestionType.ESSAY,
    "modified essay":       QuestionType.ESSAY,
    "essay":                QuestionType.ESSAY,
    "short question":       QuestionType.SHORT_NOTES,
    "short notes":          QuestionType.SHORT_NOTES,
    "short answer":         QuestionType.SHORT_NOTES,
    "short essay":          QuestionType.SHORT_NOTES,
    "seq":                  QuestionType.SHORT_NOTES,
    "saq":                  QuestionType.SHORT_NOTES,
    "very short":           QuestionType.VERY_SHORT_ANSWERS,
    "define":               QuestionType.VERY_SHORT_ANSWERS,
    "multiple choice":      QuestionType.MCQ,
    "mcq":                  QuestionType.MCQ,
    "objective":            QuestionType.MCQ,
}

def _classify_by_section(section_context: Optional[str]) -> Optional[Tuple[QuestionType, float]]:
    if not section_context:
        return None
    ctx = section_context.lower()
    for key, qtype in _SECTION_DEFAULTS.items():
        if key in ctx:
            return qtype, MEDIUM_CONF
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_question(
    question_data: Dict[str, Any],
    section_context: Optional[str] = None,
) -> Tuple[QuestionType, float]:
    """
    Classify a question into one of the four types.

    Args:
        question_data: Dict from segmenter with keys: text, options, marks, number, category
        section_context: The section heading under which this question appears

    Returns:
        (QuestionType, confidence_float)
    """
    # Rule 0: Pre-detected category from category-first segmenter (Top Priority)
    cat_str = question_data.get("category") or question_data.get("type")
    if cat_str:
        for q_type in QuestionType:
            if q_type.value.lower() == str(cat_str).lower():
                return q_type, HIGH_CONF

    text    = question_data.get("text", "")
    options = question_data.get("options", [])
    marks   = question_data.get("marks")

    # Rule 1: If MCQ options are present → MCQ
    if options and len(options) >= 2:
        return QuestionType.MCQ, HIGH_CONF

    # Rule 2: Marks-based
    marks_result = _classify_by_marks(marks)
    if marks_result:
        return marks_result

    # Rule 3: Inherit from section context
    section_result = _classify_by_section(section_context or question_data.get("section"))

    # Rule 4: Keyword matching on question text
    keyword_result = _classify_by_keywords(text)

    # Choose between section and keyword result
    if section_result and keyword_result:
        sk_type, sk_conf = keyword_result
        if sk_conf >= section_result[1]:
            return keyword_result
        return section_result
    elif section_result:
        return section_result
    elif keyword_result:
        return keyword_result

    # Rule 5: Heuristic by question text length
    return _classify_by_length(text)


def _classify_by_keywords(text: str) -> Optional[Tuple[QuestionType, float]]:
    if _ESSAY_RE.search(text):
        return QuestionType.ESSAY, MEDIUM_CONF
    if _SHORT_NOTES_RE.search(text):
        return QuestionType.SHORT_NOTES, MEDIUM_CONF
    if _VERY_SHORT_RE.search(text):
        return QuestionType.VERY_SHORT_ANSWERS, MEDIUM_CONF
    return None


def _classify_by_length(text: str) -> Tuple[QuestionType, float]:
    word_count = len(text.split())
    if word_count <= 10:
        return QuestionType.VERY_SHORT_ANSWERS, FALLBACK_CONF
    elif word_count <= 30:
        return QuestionType.SHORT_NOTES, FALLBACK_CONF
    else:
        return QuestionType.ESSAY, FALLBACK_CONF
