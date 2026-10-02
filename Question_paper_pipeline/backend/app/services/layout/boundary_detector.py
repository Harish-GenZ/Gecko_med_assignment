"""
Question-paper boundary detection.

A single PDF may contain multiple question papers.
This module identifies where each paper starts and ends.

Strategy (evidence-based from dataset inspection):
- Papers begin with a header block containing:
    * University name
    * Exam type (e.g., "Send-up Examination", "MBBS First Year")
    * Paper code
    * Date / Month+Year
    * "Time" and "Maximum marks" / "Max Marks"
- These signals appear on the FIRST page of each paper.
- We score each page by how many boundary signals it contains.
- Pages scoring ≥ BOUNDARY_THRESHOLD are treated as paper starts.

Observed patterns from dataset:
  BIU  : "SEND UP EXAMINATIONS MAY-JUNE, 2018", "Paper Code: 11016", "Section A"
  KNRUHS: "KALOJI NARAYANA RAO UNIVERSITY OF HEALTH SCIENCES",
           "MBBS FIRST YEAR EXAMINATIONS: NOVEMBER, 2023", "PAPER CODE: MB2019101"
  UHSR : "UHSR EXAMINATIONS", "ANQP Code:", "Q.P. Code:", "10th September, 2026"
"""

import re
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Minimum signal score to treat a page as a new paper boundary
BOUNDARY_THRESHOLD = 2

# ---------------------------------------------------------------------------
# Compiled patterns (from dataset analysis)
# ---------------------------------------------------------------------------

_UNIV_RE = re.compile(
    r'(university|college|institute|KNRUHS|UHSR|BIU|Bareilly)', re.I
)
_EXAM_TYPE_RE = re.compile(
    r'(examination[s]?|send.?up|annual|supplementary|mains?|internal)', re.I
)
_PAPER_CODE_RE = re.compile(
    r'(paper\s*code|ANQP\s*code|Q\.?P\.?\s*code)', re.I
)
_DATE_RE = re.compile(
    r'\b(january|february|march|april|may|june|july|august|september|'
    r'october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)'
    r'[\s\-,]*(20\d{2}|19\d{2})\b', re.I
)
_YEAR_STANDALONE_RE = re.compile(r'\b(20\d{2}|19\d{2})\b')
_MAX_MARKS_RE = re.compile(r'(max(imum)?\s*marks?|full\s*marks?|m\.\s*marks)', re.I)
_TIME_RE = re.compile(r'\b(time\s*[:]\s*\d|time\s*[:]\s*three|time\s*[:]\s*two)', re.I)
_MBBS_RE = re.compile(r'\bMBBS\b', re.I)
_SECTION_RESET_RE = re.compile(r'\bsection\s+[Aa]\b')


def score_page_as_boundary(text: str) -> Tuple[int, List[str]]:
    """
    Score a page's text for likelihood of being a new paper header.

    Returns (score, list_of_matched_signals).
    """
    signals = []
    if _UNIV_RE.search(text):
        signals.append("university")
    if _EXAM_TYPE_RE.search(text):
        signals.append("exam_type")
    if _PAPER_CODE_RE.search(text):
        signals.append("paper_code")
    if _DATE_RE.search(text):
        signals.append("date_month_year")
    elif _YEAR_STANDALONE_RE.search(text):
        signals.append("year")
    if _MAX_MARKS_RE.search(text):
        signals.append("max_marks")
    if _TIME_RE.search(text):
        signals.append("time_duration")
    if _MBBS_RE.search(text):
        signals.append("mbbs_level")
    if _SECTION_RESET_RE.search(text):
        signals.append("section_a_reset")

    return len(signals), signals


def detect_paper_boundaries(pages: List[dict]) -> List[int]:
    """
    Given a list of page dicts (each with a "text" key), return a list
    of page indices (0-based) where a new question paper likely starts.

    Always includes page 0 as the first paper boundary.
    """
    boundaries = []

    for page in pages:
        idx  = page["page_index"]
        text = page.get("text", "")
        score, signals = score_page_as_boundary(text)

        if score >= BOUNDARY_THRESHOLD:
            boundaries.append(idx)
            logger.debug(f"Paper boundary at page {idx+1}: signals={signals}")

    # Always start at page 0
    if not boundaries or boundaries[0] != 0:
        boundaries.insert(0, 0)

    # Deduplicate and sort
    boundaries = sorted(set(boundaries))
    return boundaries


def assign_pages_to_papers(pages: List[dict]) -> List[List[dict]]:
    """
    Split the flat list of pages into groups, one group per detected paper.
    """
    boundaries = detect_paper_boundaries(pages)

    # Add sentinel
    sentinel = [pages[-1]["page_index"] + 1] if pages else [1]
    groups   = []

    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else sentinel[0]
        group = [p for p in pages if start <= p["page_index"] < end]
        if group:
            groups.append(group)

    return groups
