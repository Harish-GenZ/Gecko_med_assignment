"""
Metadata extraction from question-paper header text.

Extracts:
- university
- exam_month
- exam_year
- session_code (paper code / ANQP code / Q.P. code)
- subject
- max_marks
- duration

Confidence is based on how many fields were found.

Evidence from dataset:
  BIU  : "Paper Code: 11016 / Send-up Examination – January 2021 / MBBS 3rd Professional"
  KNRUHS: "PAPER CODE: MB2019101 / MBBS FIRST YEAR EXAMINATIONS: NOVEMBER, 2023 / Max Marks: 100"
  UHSR : "ANQP Code: MBY2P5 / Q.P. Code: 1725N / 10th September, 2026 / M. Marks: 100"
"""

import re
import logging
from typing import Optional, Tuple

from app.models.schemas import ExamMetadata, ReviewStatus

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# University name — full line heuristic
_UNIV_LINE_RE = re.compile(
    r'((?:kaloji|narayana|bareilly|international|university|health\s+sciences|'
    r'KNRUHS|UHSR|BIU).{0,80})', re.I
)

# Exam month + year
_MONTH_YEAR_RE = re.compile(
    r'\b(january|february|march|april|may|june|july|august|september|'
    r'october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)'
    r'[\s\-,]*(20\d{2}|19\d{2})\b', re.I
)
# Standalone year (fallback)
_YEAR_RE = re.compile(r'\b(20\d{2}|19\d{2})\b')

# Paper / session codes — multiple institutions, multiple formats
_CODE_PATTERNS = [
    re.compile(r'paper\s*code\s*[:\-]?\s*([A-Z0-9]+)', re.I),
    re.compile(r'ANQP\s*code\s*[:\-]?\s*([A-Z0-9]+)', re.I),
    re.compile(r'Q\.?P\.?\s*code\s*[:\-]?\s*([A-Z0-9]+)', re.I),
    re.compile(r'session\s*code\s*[:\-]?\s*([A-Z0-9]+)', re.I),
    re.compile(r'code\s*[:\-]?\s*([A-Z]{2}\d+[A-Z0-9]*)', re.I),
]

# Subject
_SUBJECT_RE = re.compile(
    r'\b(biochemistry|physiology|anatomy|pharmacology|pathology|'
    r'microbiology|community\s+medicine|forensic|PSM|FMT|'
    r'surgery|medicine|obstetrics|gynaecology|paediatrics|'
    r'ophthalmology|ENT|orthopaedics|radiology|psychiatry)(?=[A-Za-z0-9]|\b)', re.I
)

# Max marks
_MAX_MARKS_RE = re.compile(
    r'(?:max(?:imum)?\s*marks?|full\s*marks?|m\.\s*marks?|marks?)\s*[:\-]?\s*(?:[\d\+\s]+=\s*)?(\d+)', re.I
)

# Duration
_DURATION_RE = re.compile(
    r'time\s*(?:allowed)?\s*[:\-]?\s*(\d+(?::\d+)?\s*(?:hours?|hrs?|minutes?|mins?)(?:\s*\d+\s*(?:minutes?|mins?))?|three\s+hours?|two\s+hours?|1[.\s]5\s*hours?)', re.I
)

# Date (Nth Month YYYY)
_DATE_RE = re.compile(
    r'(\d{1,2}(?:st|nd|rd|th)?\s+'
    r'(?:january|february|march|april|may|june|july|august|september|'
    r'october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)'
    r'(?:\s*,?\s*(?:20\d{2}|19\d{2}))?)', re.I
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_metadata(header_text: str) -> ExamMetadata:
    """
    Parse examination metadata from the header text of a question paper.
    Returns an ExamMetadata object with confidence and review status.
    """
    university   = _extract_university(header_text)
    month, year  = _extract_month_year(header_text)
    session_code = _extract_code(header_text)
    subject      = _extract_subject(header_text)
    max_marks    = _extract_max_marks(header_text)
    duration     = _extract_duration(header_text)

    # Confidence: based on how many of the 4 required fields were found
    required = [university, month, year, session_code]
    found    = sum(1 for v in required if v)
    confidence = found / 4.0

    status = ReviewStatus.OK if confidence >= 0.5 else ReviewStatus.NEEDS_REVIEW

    logger.debug(
        f"Metadata extracted: university={university}, month={month}, "
        f"year={year}, code={session_code}, conf={confidence:.2f}"
    )

    return ExamMetadata(
        university   = university,
        exam_month   = month,
        exam_year    = year,
        session_code = session_code,
        subject      = subject,
        max_marks    = max_marks,
        duration     = duration,
        confidence   = confidence,
        status       = status,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _extract_university(text: str) -> Optional[str]:
    # Try full university-name match first
    m = _UNIV_LINE_RE.search(text)
    if m:
        val = m.group(1).strip()
        # Normalise whitespace
        val = re.sub(r'\s+', ' ', val)
        if len(val) > 5:
            return val[:120]  # Truncate excessively long matches

    # Keyword-based fallback
    keywords = ["KNRUHS", "UHSR", "BIU", "Bareilly International"]
    for kw in keywords:
        if re.search(re.escape(kw), text, re.I):
            return kw

    return None


def _extract_month_year(text: str) -> Tuple[Optional[str], Optional[str]]:
    m = _MONTH_YEAR_RE.search(text)
    if m:
        return m.group(1).capitalize(), m.group(2)

    # Try date format: "10th September, 2026"
    m2 = _DATE_RE.search(text)
    if m2:
        raw = m2.group(1)
        month_m = re.search(
            r'(january|february|march|april|may|june|july|august|september|'
            r'october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)',
            raw, re.I
        )
        year_m = _YEAR_RE.search(raw)
        month = month_m.group(1).capitalize() if month_m else None
        year  = year_m.group(1)              if year_m  else None
        return month, year

    # Standalone year fallback
    year_m = _YEAR_RE.search(text)
    if year_m:
        return None, year_m.group(1)

    return None, None


def _extract_code(text: str) -> Optional[str]:
    for pattern in _CODE_PATTERNS:
        m = pattern.search(text)
        if m:
            return m.group(1).strip()
    return None


def _extract_subject(text: str) -> Optional[str]:
    m = _SUBJECT_RE.search(text)
    if m:
        return m.group(0).strip().title()
    return None


def _extract_max_marks(text: str) -> Optional[str]:
    m = _MAX_MARKS_RE.search(text)
    return m.group(1) if m else None


def _extract_duration(text: str) -> Optional[str]:
    m = _DURATION_RE.search(text)
    return m.group(1).strip() if m else None
