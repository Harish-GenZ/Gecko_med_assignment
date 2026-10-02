"""
Question segmentation module.

Splits the extracted text of a question paper into structured question blocks
by classifying categories and sections first.

Category-First Architecture:
  1. Detect Section boundaries (e.g., "Section A", "Part B").
  2. Detect Category headings first in priority order:
     - MCQ
     - Short Notes (SEQs, SAQs, Short Questions, Write short notes on...)
     - Very Short Answers (Definitions, 1-mark brief questions)
     - Essay (Structured Long Essay, Long Answer, Problem-based)
  3. Strip category heading text out of question bodies.
  4. Preserve subparts (a, b, c, d, 1, 2, i, ii) within Essay and Short Note questions.
  5. Parse options (a, b, c, d) ONLY when question category is MCQ.
  6. Extract marks hints per question or inherit from category level.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Categories in strict priority order
# Short Notes MUST be checked before Essay because "Short Essay Questions" contains "Essay"
# ---------------------------------------------------------------------------
CATEGORY_RULES = [
    (
        "MCQ",
        re.compile(
            r'\b(?:multiple\s+choice\s+questions?|mcqs?|objective(?:\s+type)?\s+questions?)',
            re.I
        )
    ),
    (
        "Short Notes",
        re.compile(
            r'\b(?:short\s+essay\s+questions?(?:\s*\([a-z0-9\s]+\))?|'
            r'short\s+answer\s+questions?(?:\s*\([a-z0-9\s]+\))?|'
            r'[cs]hort\s+questions?|'
            r'write\s+short\s+notes?\s+on(?:\s+the)?(?:\s+following)?|'
            r'write\s+a\s+short\s+notes?\s+on|'
            r'write\s+in\s+short\s+the\s+following|'
            r'short\s+notes?|brief\s+notes?|'
            r'explain\s+why[\?:]?|give\s+reasons?\s+(?:for|why)[\?:]?|'
            r'short\s+answer\s+questions?)',
            re.I
        )
    ),
    (
        "Very Short Answers",
        re.compile(
            r'\b(?:very\s+short\s+answers?|very\s+short\s+answer\s+questions?|'
            r'brief\s+answers?|define\s*:|one\s+word\s+answers?|'
            r'reasoning\s+questions?)',
            re.I
        )
    ),
    (
        "Essay",
        re.compile(
            r'\b(?:structured\s+long\s+essay(?:\s+questions?)?|'
            r'long\s+essay(?:\s+questions?)?|'
            r'long\s+answer(?:\s+questions?)?|'
            r'long\s+questions?|'
            r'essay\s*\/\s*long\s+answer|'
            r'essay\s+questions?|'
            r'problem\s+based\s+questions?|'
            r'modified\s+essay(?:\s+questions?)?)',
            re.I
        )
    ),
]

# Section boundary patterns
SECTION_RE = re.compile(
    r'^[\s]*(?:section|part|group)\s*[-:]?\s*([A-Za-z0-9]+|first|second|part\s*-[iI]+|[iI]+)[\s\(\):]*.*$',
    re.I
)

# Question number line patterns
Q_PREF_RE = re.compile(r'^[\s]*(?:[·•\*\-]\s*)?Q\.?\s*(\d+)[\.\):]?', re.I)
PLAIN_NUM_RE = re.compile(r'^[\s]*(?:[·•\*\-]\s*)?(\d{1,2})[\.\)]\s*', re.I)

# MCQ option pattern: (a) (b) (c) (d) or a) b) c) d)
MCQ_OPT_RE = re.compile(r'^[\s\.\-•*]*\(?([a-dA-D])[\)\.\s]\s*(.+)')

# Subpart start pattern for implicit question detection: (a), (b), (i), (1) etc.
SUBPART_START_RE = re.compile(
    r'^[\s\.\-•*]*\(?([a-hA-H]|[1-9]|(?:i|ii|iii|iv|v|vi))[\)\.\s]\s*(.+)',
    re.I
)

# Instruction header pattern
INSTRUCTIONS_HEADER_RE = re.compile(
    r'^[\s*\-•·~_]*(?:[A-Za-z0-9*]{1,6}\s+)?(?:important\s+note|instructions?|notes?|general\s+instructions?)[\s:]*$',
    re.I
)

# Metadata / instruction detection to prevent false question generation
_META_INST_FILTER_RE = re.compile(
    r'\b(examinations?|university|anqp\s*code|q\.?p\.?\s*code|paper\s*code|roll\s*no|'
    r'attempt\s+all|attempt\s+any|preferably\s+attempt|answer\s+booklet|separate\s+answer|'
    r'unfairmeans|unfair\s+means|graphite\s+pencil|black\s+lead|'
    r'blank\s+spaces?|writing\s+area|END\s+stamp|not\s+complied|'
    r'two\s+answer\s+booklets|do\s+not\s+(?:leave|use)\b.*(?:booklet|pencil|blank)|'
    r'm\.\s*marks|max(?:imum)?\s*marks|time\s*:\s*\d)\b',
    re.I
)

# Standalone marks formula pattern (e.g., "3+3+3+2=11", "5x5=25", "1x5=5")
STANDALONE_FORMULA_RE = re.compile(
    r'^[\s\(\[]*(?:[\d\+\s]+=\s*\d+|\d+\s*[x×]\s*\d+(?:\s*=\s*\d+)?)\s*[\)\]]*$',
    re.I
)

# Noise filters (stamps, watermarks, timestamps, URLs)
NOISE_PATTERNS = [
    re.compile(r'@medi_circle', re.I),
    re.compile(r'https?://\S+'),
    re.compile(r'instagram|telegram|youtube|whatsapp', re.I),
    re.compile(r'unfairmeans|unfair\s+means', re.I),
    re.compile(r'^\s*page\s*\d+\s*$', re.I),
    re.compile(r'^\s*\*+\s*$'),
    re.compile(r'for more useful content', re.I),
    re.compile(r'join us', re.I),
    re.compile(r'^\s*\d{8,}\.?\d*\s*$'),  # e.g., 20240704.2041 timestamps
]

# Marks regexes
MARKS_PATTERNS = [
    re.compile(r'\(?\s*mark.?[\s:=]*[\d\+\s]+=\s*(\d+)\s*(?:mark.?)?\s*\)?', re.I),
    re.compile(r'\b(?:\d+\s*\+\s*)+[\d\+]+\s*=\s*(\d+)\b', re.I),
    re.compile(r'[\d\+]+\s*=\s*(\d+)', re.I),
    re.compile(r'\b\d+\s*[x×*]\s*\d+\s*=\s*(\d+)\b', re.I),
    re.compile(r'\(?\s*(\d+(?:\.\d+|½)?)\s*mark.?\s*(?:each|per\s*question)\s*\)?', re.I),
    re.compile(r'\(?\s*(\d+)\s*mark.?\s*[x×]\s*\d+\s*=\s*(\d+)\s*mark.?\s*\)?', re.I),
    re.compile(r'\(?\s*(\d+)\s*mark.?\s*each\s*=\s*(\d+)\s*mark.?\s*\)?', re.I),
    re.compile(r'[\(\[]\s*mark.?[\s:=]*(\d+)\s*[\)\]]', re.I),
    re.compile(r'[\(\[]\s*(\d+)\s*mark.?\s*[\)\]]', re.I),
    re.compile(r'\(?\s*(\d+)\s*mark.?\s*\)?$', re.I),
]

# Marks cleanup patterns to strip out of question headers
MARKS_CLEANUP_RES = [
    re.compile(r'\(?\s*mark.?[\s:=]*[\d\+\s]+=\s*\d+\s*(?:mark.?)?\s*\)?', re.I),
    re.compile(r'\(?\s*\d+\s*mark.?\s*each(?:\s*=\s*\d+\s*(?:mark.?)?)?\s*\)?', re.I),
    re.compile(r'\(?\s*\d+\s*mark.?\s*[x×]\s*\d+(?:\s*=\s*\d+\s*(?:mark.?)?)?\s*\)?', re.I),
    re.compile(r'\(?\s*mark.?[\s:=]*\d+\s*\)?', re.I),
    re.compile(r'\(?\s*\d+\s*mark.?\s*\)?', re.I),
    re.compile(r'\[\s*\d+\s*mark.?\s*\]', re.I),
    re.compile(r'=\s*\d+\s*\)?', re.I),
    re.compile(r'\(?\s*each\s*=\s*\d+\s*mark.?\s*\)?', re.I),
]

# Continuation of category header wrapped to next line
CATEGORY_CONTINUATION_RE = re.compile(
    r'^[\s/]*(?:questions?|modified\s+essay(?:\s+questions?)?|problem\s+based(?:\s+questions?)?)[\s/:]*$',
    re.I
)

# Standalone marks hint line, e.g. "(5 Marks each)"
STANDALONE_MARKS_HINT_RE = re.compile(
    r'^[\s\(\[]*\d+(?:\.\d+|½)?\s*marks?\s*each[\s\)\]]*$',
    re.I
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def segment_questions(paper_text: str) -> List[Dict[str, Any]]:
    """
    Split paper text into individual structured question dicts,
    classifying categories and sections first.
    """
    lines = paper_text.split("\n")
    questions: List[Dict[str, Any]] = []

    # Check if this document contains section / part headers
    has_any_section = any(
        bool(SECTION_RE.match(l.strip()) and not re.search(r'question|answer|study|explain|describe', l, re.I))
        for l in lines
    )
    first_detected_section = None
    if has_any_section:
        for l in lines:
            m = SECTION_RE.match(l.strip())
            if m and not re.search(r'question|answer|study|explain|describe', l, re.I):
                first_detected_section = m.group(0).strip()
                break

    current_section: Optional[str] = None
    current_category: Optional[str] = None
    current_category_marks: Optional[str] = None
    current_heading: Optional[str] = None
    pending_marks: Optional[str] = None

    current_q_num: Optional[str] = None
    current_q_lines: List[str] = []
    current_q_marks: Optional[str] = None
    current_q_category: Optional[str] = None
    current_q_heading: Optional[str] = None
    current_q_options: List[Dict[str, str]] = []
    current_q_raw: List[str] = []
    current_q_is_q_prefixed: bool = False

    hanging_lines: List[str] = []
    in_instructions: bool = False

    container_q_num: Optional[str] = None
    container_is_mcq: bool = False

    def _flush_hanging_as_question():
        nonlocal hanging_lines, current_section, current_category, current_heading, pending_marks
        clean_lines = [l for l in hanging_lines if not _is_noise(l) and not _is_heading_line(l)]
        joined_text = " ".join(clean_lines).strip()
        # Never flush metadata or instructions as a question
        if _META_INST_FILTER_RE.search(joined_text):
            hanging_lines = []
            pending_marks = None
            return

        if len(joined_text) >= 15 and re.search(r'[a-zA-Z]{3,}', joined_text):
            if not questions:
                inferred_num = "1"
            else:
                last_num = questions[-1]["number"].split('.')[0]
                inferred_num = str(int(last_num) + 1) if last_num.isdigit() else str(len(questions) + 1)

            full_body = "\n".join(clean_lines).strip()
            full_body = re.sub(
                r'([^\n])\s*(\([a-hA-H]\)|\([1-9]\)|\((?:i|ii|iii|iv|v|vi)\))\s*',
                r'\1\n\2 ',
                full_body
            ).strip()

            q_type = current_category or "Essay"
            q_marks = pending_marks or _extract_marks(full_body)
            if q_marks and full_body:
                full_body = _strip_marks_from_text(full_body)

            questions.append({
                "number":   inferred_num,
                "text":     full_body,
                "type":     q_type,
                "category": q_type,
                "part":     current_section,
                "heading":  current_heading,
                "options":  None,
                "marks":    q_marks,
                "section":  current_section,
                "raw":      full_body,
            })
            hanging_lines = []
            pending_marks = None

    def finish_current_question():
        nonlocal current_q_num, current_q_lines, current_q_marks, current_q_category, current_q_heading
        nonlocal current_q_options, current_q_raw, current_q_is_q_prefixed
        if current_q_num is not None:
            full_body = "\n".join(current_q_lines).strip()

            # Separate prompt heading if present at the start of full_body
            if full_body:
                first_line = full_body.split("\n")[0].strip()
                if _is_heading_line(first_line):
                    if not current_q_heading:
                        current_q_heading = _strip_marks_from_heading(first_line)
                    rest_lines = full_body.split("\n")[1:]
                    full_body = "\n".join(rest_lines).strip()

            # Ensure subparts (a), (b), (c), (d), (e), (i), (ii) have a newline before them if inline
            full_body = re.sub(
                r'([^\n])\s*(\([a-hA-H]\)|\([1-9]\)|\((?:i|ii|iii|iv|v|vi)\))\s*',
                r'\1\n\2 ',
                full_body
            ).strip()

            if full_body or current_q_options:
                q_marks = current_q_marks or current_category_marks
                q_type = current_q_category or current_category or "Essay"
                if not q_marks:
                    q_marks = _extract_marks(full_body)
                if q_marks and full_body:
                    full_body = _strip_marks_from_text(full_body)

                questions.append({
                    "number":   current_q_num,
                    "text":     full_body,
                    "type":     q_type,
                    "category": q_type,
                    "part":     current_section,
                    "heading":  current_q_heading or current_heading,
                    "options":  current_q_options if q_type == "MCQ" and current_q_options else None,
                    "marks":    q_marks,
                    "section":  current_section,
                    "raw":      "\n".join(current_q_raw).strip(),
                })

            current_q_num = None
            current_q_lines = []
            current_q_marks = None
            current_q_category = None
            current_q_heading = None
            current_q_options = []
            current_q_raw = []
            current_q_is_q_prefixed = False

    for raw_line in lines:
        line = raw_line.strip()
        if _is_noise(line):
            continue

        # 0. Instructions Detection
        if INSTRUCTIONS_HEADER_RE.match(line):
            in_instructions = True
            continue

        if in_instructions:
            if (SECTION_RE.match(line) or Q_PREF_RE.match(line) or
                _detect_category(line) or STANDALONE_FORMULA_RE.match(line)):
                in_instructions = False
            else:
                continue

        # 1. Section Boundary Detection
        sec_m = SECTION_RE.match(line)
        if sec_m and not re.search(r'question|answer|study|explain|describe', line, re.I):
            finish_current_question()
            current_section = sec_m.group(0).strip()
            current_category = None
            current_category_marks = None
            hanging_lines = []
            container_q_num = None
            container_is_mcq = False
            continue

        # 2. Standalone Marks Formula line (e.g., "3+3+3+2=11", "5x5=25", "1x5=5")
        if STANDALONE_FORMULA_RE.match(line):
            finish_current_question()
            m = _extract_marks(line)
            if m:
                pending_marks = m
            continue

        # 3. Category Detection
        cat_info = _detect_category(line)
        q_pref_m = Q_PREF_RE.match(line)
        plain_num_m = PLAIN_NUM_RE.match(line)

        # Standalone Category Header
        if cat_info and not q_pref_m and not (plain_num_m and not current_q_is_q_prefixed and not container_is_mcq):
            cat_name, _ = cat_info
            cat_marks = _extract_marks(line)

            if current_q_num is not None and not current_q_lines:
                current_q_category = cat_name
                current_category = cat_name
                if cat_marks:
                    current_q_marks = cat_marks
                    current_category_marks = cat_marks
                clean_heading = _strip_marks_from_heading(line)
                if clean_heading:
                    current_heading = clean_heading
                    current_q_heading = clean_heading
                continue
            else:
                finish_current_question()
                current_category = cat_name
                if cat_marks:
                    current_category_marks = cat_marks
                clean_heading = _strip_marks_from_heading(line)
                if clean_heading and not re.search(r'^(?:mcqs?|objective|multiple choice)', clean_heading, re.I):
                    current_heading = clean_heading
                else:
                    current_heading = None
                hanging_lines = []
                continue

        # Standalone marks hint line, e.g. "(5 Marks each)"
        if STANDALONE_MARKS_HINT_RE.match(line):
            m = _extract_marks(line)
            if m:
                current_category_marks = m
            continue

        # 4. Question Start Detection
        is_new_q = False
        num = None
        rem_line = ""
        is_q_pref = False

        if q_pref_m:
            is_new_q = True
            num = q_pref_m.group(1)
            rem_line = line[q_pref_m.end():].strip()
            is_q_pref = True
            container_q_num = num
            container_is_mcq = (current_category == "MCQ")
        elif plain_num_m:
            active_cat = current_q_category or current_category
            if active_cat == "MCQ" or container_is_mcq:
                is_new_q = True
                sub_idx = plain_num_m.group(1)
                num = f"{container_q_num}.{sub_idx}" if container_q_num else sub_idx
                rem_line = line[plain_num_m.end():].strip()
                is_q_pref = False
            elif not current_q_is_q_prefixed:
                is_new_q = True
                num = plain_num_m.group(1)
                rem_line = line[plain_num_m.end():].strip()
                is_q_pref = False
        elif current_q_num is None and not container_is_mcq:
            # Implicit question start: subpart (a), (b), (1), (i) when no question is active
            sub_m = SUBPART_START_RE.match(line)
            if sub_m:
                label = sub_m.group(1).lower()
                if label in ('a', '1', 'i') or hanging_lines:
                    if not questions:
                        num = "1"
                    else:
                        last_num = questions[-1]["number"].split('.')[0]
                        num = str(int(last_num) + 1) if last_num.isdigit() else str(len(questions) + 1)

                    current_q_num = num
                    current_q_is_q_prefixed = False
                    current_q_category = current_category or "Essay"
                    current_q_heading = current_heading
                    current_q_marks = pending_marks
                    pending_marks = None
                    current_q_lines = list(hanging_lines) + [line]
                    hanging_lines = []
                    current_q_raw = list(current_q_lines)
                    continue

        if is_new_q:
            if current_q_num is not None:
                finish_current_question()
            else:
                _flush_hanging_as_question()
            current_q_num = num
            current_q_is_q_prefixed = is_q_pref
            current_q_heading = current_heading
            current_q_raw.append(line)

            if pending_marks:
                current_q_marks = pending_marks
                pending_marks = None

            inline_cat = _detect_category(rem_line)
            if inline_cat:
                current_q_category, _ = inline_cat
                current_category = current_q_category
                if current_q_category == "MCQ":
                    container_is_mcq = True
                inline_marks = _extract_marks(rem_line)
                if inline_marks:
                    current_q_marks = inline_marks
                    current_category_marks = inline_marks
                rem_line = _strip_category_text(rem_line)
            else:
                current_q_category = current_category
                m = _extract_marks(rem_line)
                if m:
                    current_q_marks = m

            if container_is_mcq:
                current_q_category = "MCQ"
                if not current_q_marks:
                    current_q_marks = "1"

            if hanging_lines:
                current_q_lines.extend(hanging_lines)
                hanging_lines = []

            if rem_line:
                current_q_lines.append(rem_line)
            continue

        # 5. Continuation Line for Current Question
        if current_q_num is not None:
            current_q_raw.append(line)

            if not current_q_lines and CATEGORY_CONTINUATION_RE.match(line):
                continue

            if not current_q_lines and cat_info:
                current_q_category = cat_info[0]
                current_category = cat_info[0]
                if current_q_category == "MCQ":
                    container_is_mcq = True
                m = _extract_marks(line)
                if m:
                    current_q_marks = m
                    current_category_marks = m
                continue

            active_cat = current_q_category or current_category

            if active_cat == "MCQ":
                opt_m = MCQ_OPT_RE.match(line)
                if opt_m:
                    opt_label = opt_m.group(1).lower()
                    if opt_label == 'a' and len(current_q_options) >= 2:
                        finish_current_question()
                        existing_subs = [
                            q["number"] for q in questions
                            if container_q_num and q["number"].startswith(f"{container_q_num}.")
                        ]
                        sub_idx = len(existing_subs) + 1
                        num = f"{container_q_num}.{sub_idx}" if container_q_num else str(sub_idx)
                        current_q_num = num
                        current_q_is_q_prefixed = False
                        current_q_category = "MCQ"
                        current_q_marks = "1"
                        current_q_heading = current_heading
                        current_q_lines = list(hanging_lines)
                        hanging_lines = []
                        current_q_options = [{
                            "label": opt_label,
                            "text":  opt_m.group(2).strip()
                        }]
                        current_q_raw = [line]
                        continue
                    else:
                        current_q_options.append({
                            "label": opt_label,
                            "text":  opt_m.group(2).strip(),
                        })
                        continue
                elif len(current_q_options) >= 3:
                    finish_current_question()
                    hanging_lines.append(line)
                    continue

            if not current_q_marks:
                m = _extract_marks(line)
                if m:
                    current_q_marks = m

            current_q_lines.append(line)
        else:
            # When current_q_num is None: check if this is an MCQ option starting a question whose number was missing/cut off
            if (container_is_mcq or current_category == "MCQ"):
                opt_m = MCQ_OPT_RE.match(line)
                if opt_m:
                    opt_label = opt_m.group(1).lower()
                    existing_subs = [
                        q["number"] for q in questions
                        if container_q_num and q["number"].startswith(f"{container_q_num}.")
                    ]
                    sub_idx = len(existing_subs) + 1
                    num = f"{container_q_num}.{sub_idx}" if container_q_num else str(sub_idx)
                    current_q_num = num
                    current_q_is_q_prefixed = False
                    current_q_category = "MCQ"
                    current_q_marks = "1"
                    current_q_heading = current_heading
                    current_q_lines = list(hanging_lines)
                    hanging_lines = []
                    current_q_options = [{
                        "label": opt_label,
                        "text":  opt_m.group(2).strip()
                    }]
                    current_q_raw = [line]
                    continue

            m = _extract_marks(line)
            if m:
                pending_marks = m
            else:
                hanging_lines.append(line)

    finish_current_question()
    logger.debug(f"Segmented {len(questions)} questions")
    return questions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_noise(line: str) -> bool:
    line_s = line.strip()
    if not line_s:
        return True
    for pat in NOISE_PATTERNS:
        if pat.search(line_s):
            return True
    return False


def _detect_category(line: str) -> Optional[Tuple[str, str]]:
    for cat_name, pat in CATEGORY_RULES:
        m = pat.search(line)
        if m:
            return cat_name, m.group(0)
    return None


def _extract_marks(text: str) -> Optional[str]:
    for pat in MARKS_PATTERNS:
        m = pat.search(text)
        if m:
            for g in m.groups():
                if g:
                    return g.strip()
    return None


def _strip_category_text(line: str) -> str:
    cleaned = line
    for _, pat in CATEGORY_RULES:
        cleaned = pat.sub("", cleaned)
    for pat in MARKS_CLEANUP_RES:
        cleaned = pat.sub("", cleaned)
    cleaned = cleaned.strip()
    # If nothing substantial remains except symbols/numbers/brackets, return empty
    if not re.search(r'[a-zA-Z]{2,}', cleaned):
        return ""
    # Strip leading punctuation
    cleaned = re.sub(r'^[\s\.\-:/]+', '', cleaned)
    return cleaned.strip()


def _strip_marks_from_text(text: str) -> str:
    cleaned_lines = []
    for line in text.split('\n'):
        l = re.sub(r'\(?[ \t]*\b(?:\d+[ \t]*\+[ \t]*)+[\d\+]+[ \t]*=[ \t]*\d+\b[ \t]*(?:marks?)?[ \t]*\)?', '', line, flags=re.I)
        l = re.sub(r'\(?[ \t]*\b\d+[ \t]*[x×*][ \t]*\d+(?:[ \t]*=[ \t]*\d+)?[ \t]*(?:marks?)?[ \t]*\)?', '', l, flags=re.I)
        l = re.sub(r'\(?[ \t]*\b\d+(?:\.\d+|½)?[ \t]*marks?(?:[ \t]*(?:each|per[ \t]*question))?[ \t]*\)?', '', l, flags=re.I)
        l = re.sub(r'\[[ \t]*\d+[ \t]*marks?[ \t]*\]', '', l, flags=re.I)
        l = re.sub(r'[ \t]+', ' ', l).strip()
        if l:
            cleaned_lines.append(l)
    return '\n'.join(cleaned_lines)


def _strip_marks_from_heading(line: str) -> str:
    cleaned = line
    cleaned = re.sub(r'\(?\s*\d+\s*[x×]\s*\d+(?:\s*=\s*\d+)?\s*\)?', '', cleaned)
    cleaned = re.sub(r'\(?\s*[\d\+\s]+=\s*\d+\s*\)?', '', cleaned)
    cleaned = re.sub(r'\(?\s*\d+(?:\.\d+|½)?\s*marks?\s*(?:each|per\s*question)?\s*\)?', '', cleaned, flags=re.I)
    cleaned = re.sub(r'\[\s*\d+\s*marks?\s*\]', '', cleaned, flags=re.I)
    return cleaned.strip()


def _is_heading_line(line: str) -> bool:
    line_s = line.strip()
    if not line_s:
        return False
    heading_pats = [
        r'^(?:write\s+(?:short|brief|a\s+short)\s+notes?\s+on|write\s+in\s+short\s+the\s+following)',
        r'^(?:explain\s+why[\?:]?|explain\s+the\s+following[\?:]?|give\s+reasons?\s+(?:for|why)[\?:]?)',
        r'^(?:structured\s+long\s+essay|long\s+essay|short\s+notes?|brief\s+notes?)',
        r'^(?:multiple\s+choice\s+questions?\.?|objective\s+questions?\.?)'
    ]
    for pat in heading_pats:
        if re.search(pat, line_s, re.I):
            return True
    return False

