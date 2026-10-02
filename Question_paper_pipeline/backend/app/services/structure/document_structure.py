"""
Document Structure Analysis module.

Separates a question paper into 4 distinct semantic zones before question extraction:
  1. HEADER / METADATA : University, exam type, codes, date, subject, max marks, duration
  2. INSTRUCTIONS      : General notes, numbered exam instructions, booklets/pencil rules
  3. SECTION HEADINGS  : PART-A, PART-B, Section A, Group I, etc.
  4. ACTUAL QUESTIONS  : Genuine question blocks under their respective sections

Decoupling ensures:
  - Metadata is extracted strictly from the header region.
  - Instructions are never mistakenly treated as questions or classified as Essay.
  - Only genuine question blocks enter question segmentation and classification.
"""

from __future__ import annotations
import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

# Section boundary patterns (e.g. "Section A", "PART-A", "Part B", "(PART-A)", "[Group 1]")
SECTION_RE = re.compile(
    r'^[\s\(\[]*(?:section|part|group)\s*[-:]?\s*([A-Za-z0-9]+|first|second|part\s*-[iI]+|[iI]+)[\s\(\):]*.*$',
    re.I
)

# Instruction headers (e.g. "*AR IMPORTANT NOTE:", "IMPORTANT NOTE:", "Note:", "Instructions:")
INSTRUCTION_HEADER_RE = re.compile(
    r'^[\s*\-•·~_]*(?:[A-Za-z0-9*]{1,6}\s+)?(?:important\s+note|instructions?|notes?|general\s+instructions?)\s*[:\-]*\s*$',
    re.I
)

# Inline instruction prefix (e.g. "Note: Attempt all questions...", "Instructions: 1. ...")
INLINE_INSTRUCTION_PREFIX_RE = re.compile(
    r'^[\s*\-•·~_]*(?:[A-Za-z0-9*]{1,6}\s+)?(?:note|instructions?|general\s+instructions?)\s*[:\-]\s*(.+)$',
    re.I
)

# Instruction keyword patterns indicating exam rules / candidate guidelines
INSTRUCTION_KEYWORDS = [
    r'\battempt\s+all\b',
    r'\battempt\s+any\b',
    r'\battempt\s+the\s+(?:following\s+)?questions\b',
    r'\bpreferably\s+attempt\b',
    r'\banswer\s+(?:all|any)\b',
    r'\banswer\s+booklets?\b',
    r'\bseparate\s+answer\b',
    r'\banswer\s+sheets?\b',
    r'\bblank\s+spaces?\b',
    r'\bblack\s+lead\b',
    r'\bgraphite\s+pencil\b',
    r'\bunfairmeans\b',
    r'\bunfair\s+means\b',
    r'\binstructions?\s+at\s+\d\b',
    r'\bnot\s+complied\b',
    r'\bEND\s+stamp\b',
    r'\bdraw\s+(?:appropriate\s+)?diagrams\b',
    r'\bexplain\s+with\s+examples\b',
    r'\bgiven\s+sequence\b',
    r'\bserially\b',
    r'\bown\s+words\b',
    r'\bnon[- ]programmable\s+calculator\b',
    r'\belectronic\s+gadgets?\b',
    r'\bmobile\s+phones?\b',
    r'\bdo\s+not\s+write\s+(?:your\s+)?(?:name|roll)\b',
    r'\bdo\s+not\s+(?:leave|use)\b',
    r'\bfigures?\s+to\s+the\s+right\b',
    r'\billustrate\s+your\s+answers?\b',
    r'\ball\s+questions\s+carry\s+equal\s+marks\b',
    r'\buse\s+of\s+calculators?\b',
    r'\bcase\s+of\s+use\s+of\s+unfair\b',
    r'\btwo\s+answer\s+booklets\b',
    r'\bwriting\s+area\b',
    r'\bwriting\s+in\s+pencil\b',
]
INSTRUCTION_KEYWORD_RE = re.compile('|'.join(INSTRUCTION_KEYWORDS), re.I)

# Explicit question prefix: Q.1, Q1., Question 1, etc.
Q_PREF_RE = re.compile(r'^[\s]*(?:[·•\*\-]\s*)?Q\.?\s*(\d+)[\.\):]?', re.I)

# Standalone formula: e.g. "3+3+3+2=11", "5x5=25", "1x5=5"
STANDALONE_FORMULA_RE = re.compile(
    r'^[\s\(\[]*(?:[\d\+\s]+=\s*\d+|\d+\s*[x×]\s*\d+(?:\s*=\s*\d+)?)\s*[\)\]]*$',
    re.I
)

# Category headings (e.g. "Write short notes on:", "Multiple choice questions", "Structured Long Essay")
CATEGORY_HEADING_PATTERNS = [
    re.compile(r'\b(?:multiple\s+choice\s+questions?|mcqs?|objective(?:\s+type)?\s+questions?)', re.I),
    re.compile(r'\b(?:short\s+essay\s+questions?|short\s+answer\s+questions?|write\s+short\s+notes?\s+on|short\s+notes?|brief\s+notes?)', re.I),
    re.compile(r'\b(?:very\s+short\s+answers?|brief\s+answers?|define\s*:)', re.I),
    re.compile(r'\b(?:structured\s+long\s+essay|long\s+essay|long\s+answer|long\s+questions?|essay\s+questions?|problem\s+based)', re.I),
    re.compile(r'\b(?:explain\s+why[\?:]?|give\s+reasons?\s+(?:for|why)[\?:]?)', re.I),
]

# Numbered item prefix: e.g. "1.", "2.", "(i)", "(ii)", "(a)"
NUMBERED_ITEM_RE = re.compile(r'^[\s\.\-•*]*(?:\(?\d{1,2}[\.\)]|\([a-z0-9ivx]+\))\s*(.*)', re.I)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class DocumentStructure:
    """
    Structured representation of a question paper partitioned into zones.
    """
    header_text: str = ""                         # Isolated header & metadata region
    instructions: List[str] = field(default_factory=list) # Clean list of instruction items
    questions_raw_text: str = ""                  # Section headings + genuine question blocks
    sections: List[str] = field(default_factory=list)      # Detected section names e.g. ["PART-A", "PART-B"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_section_heading(line: str) -> bool:
    line_s = line.strip()
    if not line_s:
        return False
    m = SECTION_RE.match(line_s)
    if m:
        # Exclude questions that accidentally contain the word "section" or "part" in a sentence
        if re.search(r'\b(question|answer|study|explain|describe|patient|following)\b', line_s, re.I):
            return False
        return True
    return False


def _is_category_heading(line: str) -> bool:
    line_s = line.strip()
    return any(bool(pat.search(line_s)) for pat in CATEGORY_HEADING_PATTERNS)


def _is_instruction_keyword_line(line: str) -> bool:
    return bool(INSTRUCTION_KEYWORD_RE.search(line))


def _clean_instruction_item(text: str) -> str:
    # Normalize internal spaces
    return re.sub(r'\s+', ' ', text).strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_document_structure(paper_text: str) -> DocumentStructure:
    """
    Analyzes document layout and partitions text into:
      - Header/Metadata zone
      - Instructions zone
      - Section headings
      - Genuine question blocks

    Guarantees:
      - Header lines never enter question extraction.
      - Instructions are separated into structured guidelines, never classified as questions.
      - Section transitions and genuine question blocks are preserved for question segmentation.
    """
    lines = paper_text.split('\n')
    header_lines: List[str] = []
    instruction_items: List[str] = []
    question_lines: List[str] = []
    detected_sections: List[str] = []

    # States: HEADER -> INSTRUCTIONS -> QUESTIONS
    state = "HEADER"

    for line in lines:
        line_s = line.strip()

        # Preserve empty lines in question blocks to maintain spacing between subparts
        if not line_s:
            if state == "QUESTIONS":
                question_lines.append(line)
            continue

        # Signal checks
        is_sec = _is_section_heading(line_s)
        is_q_pref = bool(Q_PREF_RE.match(line_s))
        is_cat = _is_category_heading(line_s)
        is_formula = bool(STANDALONE_FORMULA_RE.match(line_s))
        is_inst_hdr = bool(INSTRUCTION_HEADER_RE.match(line_s))
        m_inline_inst = INLINE_INSTRUCTION_PREFIX_RE.match(line_s)
        is_inst_kw = _is_instruction_keyword_line(line_s)

        # -------------------------------------------------------------------
        # State: HEADER
        # -------------------------------------------------------------------
        if state == "HEADER":
            # Check transitions out of HEADER:
            if is_sec:
                state = "QUESTIONS"
                sec_match = SECTION_RE.match(line_s)
                if sec_match:
                    clean_sec = re.sub(r'^[\s\(\[]+|[\s\)\]]+$', '', sec_match.group(0)).strip()
                    detected_sections.append(clean_sec)
                question_lines.append(line)

            elif is_q_pref or (is_cat and not is_inst_kw) or is_formula:
                state = "QUESTIONS"
                question_lines.append(line)

            elif is_inst_hdr:
                state = "INSTRUCTIONS"
                # Pure header like "*AR IMPORTANT NOTE:", nothing more on this line

            elif m_inline_inst:
                state = "INSTRUCTIONS"
                body = m_inline_inst.group(1).strip()
                if body:
                    instruction_items.append(_clean_instruction_item(body))

            elif is_inst_kw:
                state = "INSTRUCTIONS"
                instruction_items.append(_clean_instruction_item(line_s))

            else:
                # Still in header (University, exam title, roll no, subject, time, marks, etc.)
                header_lines.append(line)

        # -------------------------------------------------------------------
        # State: INSTRUCTIONS
        # -------------------------------------------------------------------
        elif state == "INSTRUCTIONS":
            # Transitions OUT of INSTRUCTIONS → only on an unambiguous question start
            # i.e. explicit section heading (PART-A) or explicit Q. prefix.
            # Plain numbered items (1. 2. 3. 4.) are ALWAYS instructions in this state.
            if is_sec:
                state = "QUESTIONS"
                sec_match = SECTION_RE.match(line_s)
                if sec_match:
                    clean_sec = re.sub(r'^[\s\(\[]+|[\s\)\]]+$', '', sec_match.group(0)).strip()
                    detected_sections.append(clean_sec)
                question_lines.append(line)

            elif is_q_pref or is_formula:
                # Explicit "Q.1" or standalone marks formula → questions have started
                state = "QUESTIONS"
                question_lines.append(line)

            elif is_cat and not is_inst_kw:
                # Category heading like "Long Essay Questions" without instruction keywords
                state = "QUESTIONS"
                question_lines.append(line)

            else:
                # ----------------------------------------------------------
                # Still in INSTRUCTIONS: consume everything as instruction
                # content.  Numbered items (1. 2. 3. 4.) start a new entry;
                # every other non-section/non-question line is a continuation.
                # ----------------------------------------------------------
                num_m = NUMBERED_ITEM_RE.match(line_s)
                if num_m:
                    # A numbered item — always starts a new instruction entry
                    instruction_items.append(_clean_instruction_item(line_s))
                elif instruction_items:
                    # Any continuation line (keyword or not) appends to the
                    # last instruction item (handles wrapped lines like
                    # "any blank spaces... before the END stamp...")
                    instruction_items[-1] += " " + _clean_instruction_item(line_s)
                else:
                    # No instruction items yet — this is a standalone line
                    # (e.g. inline note without a number prefix)
                    instruction_items.append(_clean_instruction_item(line_s))

        # -------------------------------------------------------------------
        # State: QUESTIONS
        # -------------------------------------------------------------------
        elif state == "QUESTIONS":
            if is_sec:
                sec_match = SECTION_RE.match(line_s)
                if sec_match:
                    sec_name = sec_match.group(0).strip()
                    if sec_name not in detected_sections:
                        detected_sections.append(sec_name)
            question_lines.append(line)

    # Clean and consolidate instructions
    cleaned_instructions = [
        item for item in instruction_items
        if len(item) > 3 and not INSTRUCTION_HEADER_RE.match(item)
    ]

    header_text = '\n'.join(header_lines).strip()
    questions_raw_text = '\n'.join(question_lines).strip()

    logger.debug(
        f"Document structure parsed: {len(header_lines)} header lines, "
        f"{len(cleaned_instructions)} instructions, "
        f"{len(detected_sections)} sections, "
        f"{len(question_lines)} question lines"
    )

    return DocumentStructure(
        header_text        = header_text,
        instructions       = cleaned_instructions,
        questions_raw_text = questions_raw_text,
        sections           = detected_sections,
    )
