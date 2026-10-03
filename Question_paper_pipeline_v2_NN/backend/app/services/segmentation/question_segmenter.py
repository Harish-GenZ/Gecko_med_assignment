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
            r'\b(?:multiple\s+choice(?:\s+[a-z]+)?|mcqs?|objective(?:\s+type)?(?:\s+[a-z]+)?)',
            re.I
        )
    ),
    (
        "Short Notes",
        re.compile(
            r'\b(?:short\s+essay\s+questions?(?:\s*\([a-z0-9\s]+\))?|'
            r'short\s+answer\s+questions?(?:\s*\([a-z0-9\s]+\))?|'
            r'write\s*(?:the\s*)?differences?\s*between\s*[:\-]|'
            r'writethedifferencesbetween\s*[:\-]?|'
            r'differentiate\s*between\s*[:\-]|'
            r'enumerate\s*[:\-]|'
            r'writeshortnoteson\s*[:\-]?|'
            r'write\s+short\s+notes?\s+on(?:\s+the)?(?:\s+following)?\s*[:\-]?|'
            r'write\s+a\s+short\s+notes?\s+on\s*[:\-]?|'
            r'write\s+in\s+short\s+the\s+following\s*[:\-]?|'
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
    r'^[\s\(\[]*(?:section|part|group)\s*[-:]?\s*([A-Za-z0-9]+|first|second|part\s*-[iI]+|[iI]+)[\s\(\):]*.*$',
    re.I
)

_ROMAN_PAT = r'(?:X{0,2}(?:IX|IV|V?I{1,3}))'

# Question number line patterns
Q_PREF_RE = re.compile(rf'^[\s]*(?:[·•\*\-]\s*)?Q\.?\s*([0-9]{{1,2}}|[lIL|]|I\d|It|{_ROMAN_PAT})[\.\):,]?', re.I)
PLAIN_NUM_RE = re.compile(rf'^[\s]*(?:[·•\*\-]\s*)?([0-9]{{1,2}}|[lIL|]|I\d|It|{_ROMAN_PAT})[\.\),]\s*', re.I)

_ROMAN_MAP = {
    'i': '1', 'ii': '2', 'iii': '3', 'iv': '4', 'v': '5',
    'vi': '6', 'vii': '7', 'viii': '8', 'ix': '9', 'x': '10',
    'xi': '11', 'xii': '12', 'xiii': '13', 'xiv': '14', 'xv': '15',
    'xvi': '16', 'xvii': '17', 'xviii': '18', 'xix': '19', 'xx': '20'
}


def _normalize_question_number(raw_num: str) -> str:
    """Normalizes OCR confusion and Roman numerals to standard decimal numbers."""
    if not raw_num:
        return "1"
    raw_clean = raw_num.strip()
    low = raw_clean.lower()
    if low in ('l', 'i', '|'):
        return "1"
    if low == 'it':
        return "11"
    if low.startswith('i') and len(low) > 1 and low[1:].isdigit():
        return f"1{low[1:]}"
    if low in _ROMAN_MAP:
        return _ROMAN_MAP[low]
    if raw_clean.isdigit():
        return raw_clean
    return raw_clean


# MCQ option pattern: (a) (b) (c) (d) or a) b) c) d)
MCQ_OPT_RE = re.compile(r'^[\s\.\-•*]*\(?([a-dA-D])[\)\.\s]\s*(.+)')


def _extract_mcq_options_from_line(line: str) -> List[Tuple[str, str]]:
    """
    Extracts all MCQ options from a single line.
    Handles:
      a) Gustafsonmethod b) DNAprofiling
      a)Cuboid b)Talus
      a) NuclearDNA
      (a) option1 (b) option2
      a. option1 b. option2
    """
    line_s = line.strip()
    if not line_s:
        return []

    opt_pattern = re.compile(
        r'(?:^|[\s\t]+)\(?([a-dA-D])[\)\.\-:]\s*(.*?)(?=(?:[\s\t]+\(?[a-dA-D][\)\.\-:]|$))',
        re.DOTALL
    )
    matches = opt_pattern.findall(line_s)
    if not matches:
        single_m = MCQ_OPT_RE.match(line_s)
        if single_m:
            return [(single_m.group(1).lower(), single_m.group(2).strip())]
        return []

    results = []
    for label, text in matches:
        t = text.strip()
        t = re.sub(r'^[s\-_*•·~]+\s*', '', t)
        if t:
            results.append((label.lower(), t))
    return results

# Subpart start pattern for implicit question detection: (a), (b), (i), (1) etc.
SUBPART_START_RE = re.compile(
    r'^[\s\.\-•*]*\(?([a-hA-H]|[1-9]|(?:i|ii|iii|iv|v|vi))[\)\.\s]\s*(.+)',
    re.I
)

# Instruction header pattern
INSTRUCTIONS_HEADER_RE = re.compile(
    r'^[\s*\-•·~_]*(?:[A-Za-z0-9*]{1,6}\s+)?(?:important\s*notes?|instructions?|notes?|general\s*instructions?)[\s:]*$',
    re.I
)

# Metadata / instruction detection to prevent false question generation
_META_INST_FILTER_RE = re.compile(
    r'\b(examinations?|university|anqp\s*code|q\.?p\.?\s*code|paper\s*code|roll\s*no|'
    r'attempt\s+all|attempt\s+any|preferably\s+attempt|answer\s+booklet|separate\s+answer|'
    r'unfairmeans|unfair\s+means|graphite\s*pencil|black\s+lead|'
    r'blank\s+spaces?|writing\s+area|END\s+stamp|not\s+complied|'
    r'two\s+answer\s+booklets|do\s+not\s+(?:leave|use|write)\b.*(?:booklet|pencil|blank)|'
    r'm\.\s*marks|max(?:imum)?\s*marks|total\s*marks|time[\.:\s]|important\s*notes?)\b',
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
    re.compile(r'\(?\s*mark.?[\s:=]*[\d\+\s]+[=-]\s*(\d+)\s*(?:mark.?)?\s*\)?', re.I),
    re.compile(r'\b(?:\d+\s*\+\s*)+[\d\+]+\s*[=-]\s*(\d+)\b', re.I),
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

# Common action verbs initiating question subparts
VERB_PAT = r'(?:Describe|Explain|Define|Enlist|Enumerate|Draw|Differentiate|Discuss|Write|List|Mention|State|Classify|Give|Compare|Contrast|Evaluate|Outline|Calculate|Illustrate|Name|Distinguish)'


def clean_scribbles_and_text(text: str) -> str:
    """
    Cleans student annotations, scribbles, ticks, attached verbs, and handwriting noise
    from question bodies, ensuring only genuine printed/typeset content is retained.
    """
    if not text:
        return ""

    # 1. Strip leading prompt leakage (e.g. "Questions\n", "Modified Essay Questions\n")
    cleaned = re.sub(
        r'^(?:Questions?|Modified\s+Essay(?:\s+Questions?)?|Problem\s+based(?:\s+questions?)?)[\s:]*\n+',
        '',
        text.strip(),
        flags=re.I
    )

    lines = cleaned.split('\n')
    out_lines = []
    last_subpart = None

    for raw_l in lines:
        l = raw_l.strip()
        if not l:
            continue

        # Strip trailing student scribbles, underscores, pen dashes: ec--, --+, ==+, __+
        l = re.sub(r'\s+(?:ec|ce|cc)?[\-_=~]{2,}\s*$', '', l, flags=re.I)
        l = re.sub(r'[\-_=~]{2,}\s*$', '', l)

        # Strip student checkmarks / ticks / dashes before subparts:
        # Je. -> e. , va. -> a. , -d. -> d. , C- -> c.
        l = re.sub(r'^[JvVILl/|\-•·*]\s*([a-hA-H])[\.\)]\s*', r'\1. ', l)
        l = re.sub(r'^([a-hA-H])\s*-\s*', r'\1. ', l)

        # Merge subpart prefix + action verb: aDescribe -> a. Describe, CExplain -> c. Explain
        l = re.sub(r'^([a-hA-H])(' + VERB_PAT + r')\b', r'\1. \2', l)

        # Scribble before subpart + verb: JeDescribe -> e. Describe
        l = re.sub(r'^[JvVILl/|\-•·*]\s*([a-hA-H])(' + VERB_PAT + r')\b', r'\1. \2', l)

        # Fix missing space after subpart period: a.Define -> a. Define
        l = re.sub(r'^([a-hA-H]|[1-9])\.\s*([A-Za-z])', r'\1. \2', l)

        # Fix orphaned period before action verb: .Explain -> c. Explain (inferred from sequence)
        if re.match(r'^\.\s*' + VERB_PAT + r'\b', l, re.I):
            next_char = 'c'
            if last_subpart and ord(last_subpart) >= ord('a') and ord(last_subpart) < ord('z'):
                next_char = chr(ord(last_subpart) + 1)
            l = re.sub(r'^\.\s*', f'{next_char}. ', l)

        # Track subpart letter
        sub_m = re.match(r'^([a-hA-H])\.\s+', l)
        if sub_m:
            last_subpart = sub_m.group(1).lower()

        # Fix missing space before opening parenthesis: graphs(frequently -> graphs (frequently
        l = re.sub(r'([a-zA-Z0-9])\(', r'\1 (', l)

        # Clean student handwriting artifacts & OCR typos:
        # "an&" -> "and"
        l = re.sub(r'\ban&(?:\s+|$)', 'and ', l)
        # "jused" -> "used"
        l = re.sub(r'\bjused\b', 'used', l)
        # Space before period: "diseases ." -> "diseases."
        l = re.sub(r'\s+\.', '.', l)

        # Close unclosed trailing parenthesis if appropriate
        if l.count('(') > l.count(')'):
            l += ')'

        # If line is just student noise or empty
        if not re.search(r'[a-zA-Z]{2,}', l):
            continue

        out_lines.append(l)

    # If all lines start with an action verb and NONE have subpart bullets, auto-assign a., b., c., etc.
    if len(out_lines) >= 2:
        has_any_prefix = any(re.match(r'^[a-hA-H1-9]\.', line) for line in out_lines)
        all_verbs = all(re.match(r'^' + VERB_PAT + r'\b', line, re.I) for line in out_lines)
        if not has_any_prefix and all_verbs:
            letters = 'abcdefghijklmnopqrstuvwxyz'
            out_lines = [f'{letters[i]}. {line}' for i, line in enumerate(out_lines)]

    return '\n'.join(out_lines).strip()


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

        if len(joined_text) >= 8 and re.search(r'[a-zA-Z]{3,}', joined_text):
            # Check questions in the current section:
            current_sec_qs = [
                q for q in questions
                if q.get("section") == current_section or q.get("part") == current_section
            ]
            if not current_sec_qs:
                inferred_num = "1"
            else:
                last_num = current_sec_qs[-1]["number"].split('.')[0]
                inferred_num = str(int(last_num) + 1) if last_num.isdigit() else str(len(current_sec_qs) + 1)

            full_body = "\n".join(clean_lines).strip()
            full_body = re.sub(
                r'([^\n])\s*(\([a-hA-H]\)|\([1-9]\)|\((?:i|ii|iii|iv|v|vi)\))\s*',
                r'\1\n\2 ',
                full_body
            ).strip()

            full_body = clean_scribbles_and_text(full_body)
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
            # ONLY if there are subsequent subparts/lines in the question!
            if full_body:
                split_lines = full_body.split("\n")
                if len(split_lines) > 1 and _is_heading_line(split_lines[0].strip()):
                    if not current_q_heading:
                        current_q_heading = _strip_marks_from_heading(split_lines[0].strip())
                    full_body = "\n".join(split_lines[1:]).strip()

            # Ensure subparts (a), (b), (c), (d), (e), (i), (ii) have a newline before them if inline
            full_body = re.sub(
                r'([^\n])\s*(\([a-hA-H]\)|\([1-9]\)|\((?:i|ii|iii|iv|v|vi)\))\s*',
                r'\1\n\2 ',
                full_body
            ).strip()

            # Clean student scribbles, handwriting noise, checkmarks, and OCR typos
            full_body = clean_scribbles_and_text(full_body)

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
            sec_raw = re.sub(r'^[\s\(\[]+|[\s\)\]]+$', '', sec_m.group(0)).strip()
            # Clean marks hints from section name: "Section A (Marks: 40" -> "Section A"
            current_section = re.sub(r'\(?\s*marks?.*$', '', sec_raw, flags=re.I).strip()
            current_category = None
            current_category_marks = None
            hanging_lines = []
            pending_marks = None
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

        # Check for category continuation line (e.g. "Questions" wrapped to next line)
        if CATEGORY_CONTINUATION_RE.match(line):
            if current_heading and not current_heading.endswith(line):
                current_heading = f"{current_heading} {line}".strip()
            if current_q_heading and not current_q_heading.endswith(line):
                current_q_heading = f"{current_q_heading} {line}".strip()
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
                if hanging_lines:
                    _flush_hanging_as_question()
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
            raw_num = q_pref_m.group(1)
            num = _normalize_question_number(raw_num)
            rem_line = line[q_pref_m.end():].strip()
            is_q_pref = True
            container_q_num = num
            container_is_mcq = (current_category == "MCQ")
        elif plain_num_m:
            raw_num = plain_num_m.group(1)
            norm_num = _normalize_question_number(raw_num)
            active_cat = current_q_category or current_category
            if active_cat == "MCQ" or container_is_mcq:
                is_new_q = True
                sub_idx = norm_num
                num = f"{container_q_num}.{sub_idx}" if container_q_num else sub_idx
                rem_line = line[plain_num_m.end():].strip()
                is_q_pref = False
            elif not current_q_is_q_prefixed:
                rem_line = line[plain_num_m.end():].strip()
                # Skip candidate instruction lines like "1. Attempt all...", "2. Part-A and Part-B..."
                if _META_INST_FILTER_RE.search(line) or _META_INST_FILTER_RE.search(rem_line):
                    continue
                is_new_q = True
                num = norm_num
                is_q_pref = False
        elif current_q_num is None and not container_is_mcq:
            # Implicit question start: subpart (a), (b), (1), (i) when no question is active
            sub_m = SUBPART_START_RE.match(line)
            if sub_m:
                label = sub_m.group(1).lower()
                if label in ('a', '1', 'i') or hanging_lines:
                    current_sec_qs = [
                        q for q in questions
                        if q.get("section") == current_section or q.get("part") == current_section
                    ]
                    if not current_sec_qs:
                        num = "1"
                    else:
                        last_num = current_sec_qs[-1]["number"].split('.')[0]
                        num = str(int(last_num) + 1) if last_num.isdigit() else str(len(current_sec_qs) + 1)

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
                if hanging_lines:
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
                clean_heading = _strip_marks_from_heading(rem_line)
                if clean_heading:
                    current_heading = clean_heading
                    current_q_heading = clean_heading
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
                parsed_opts = _extract_mcq_options_from_line(line)
                if parsed_opts:
                    opt_label = parsed_opts[0][0]
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
                        current_q_options = []
                        for o_lbl, o_txt in parsed_opts:
                            current_q_options.append({"label": o_lbl, "text": o_txt})
                        current_q_raw = [line]
                        continue
                    else:
                        for o_lbl, o_txt in parsed_opts:
                            current_q_options.append({"label": o_lbl, "text": o_txt})
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
                parsed_opts = _extract_mcq_options_from_line(line)
                if parsed_opts:
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
                    current_q_options = []
                    for o_lbl, o_txt in parsed_opts:
                        current_q_options.append({"label": o_lbl, "text": o_txt})
                    current_q_raw = [line]
                    continue

            if _META_INST_FILTER_RE.search(line):
                continue

            clean_l = _strip_marks_from_text(line).strip()
            m = _extract_marks(line)
            # If line is purely a standalone marks allocation e.g. "(Marks: 10)", "(Marks: 3+5)", "(1+2+5=8)"
            if m and (not clean_l or not re.search(r'[a-zA-Z]{3,}', clean_l)):
                pending_marks = m
            else:
                if m:
                    pending_marks = m
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
    if not text:
        return None

    # Check for explicit sum with equals, e.g. (1+2+5=8) or [2+3=5]
    eq_m = re.search(r'[\(\[]\s*[\d\+\s]+=\s*(\d+)\s*[\)\]]', text)
    if eq_m:
        return eq_m.group(1).strip()

    # Check for pure addition formula without equals, e.g. (4+8+4) or [4+6] or (Marks:3+5)
    sum_m = re.search(r'[\(\[]\s*(?:marks?[:\s]*)?(\d+(?:\s*\+\s*\d+)+)\s*[\)\]]', text, re.I)
    if sum_m:
        parts = [int(p.strip()) for p in sum_m.group(1).split('+') if p.strip().isdigit()]
        if parts:
            return str(sum(parts))

    for pat in MARKS_PATTERNS:
        m = pat.search(text)
        if m:
            for g in m.groups():
                if g:
                    return g.strip()
    return None


def _strip_category_text(line: str) -> str:
    # If line is a complete question sentence (starts with verb + noun/clause, e.g. "Enumerate positive signs of pregnancy")
    # do NOT strip the leading verb!
    if re.search(r'^(?:Describe|Explain|Define|Enlist|Enumerate|Differentiate|Discuss|Classify)\s+[a-zA-Z]{3,}', line.strip(), re.I):
        # Only strip trailing marks allocations
        cleaned = line
        for pat in MARKS_CLEANUP_RES:
            cleaned = pat.sub("", cleaned)
        return cleaned.strip()

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
        l = re.sub(r'\(?[ \t]*\b(?:marks?[ \t:=]*)?(?:\d+[ \t]*\+[ \t]*)+[\d\+]+[ \t]*[=-][ \t]*\d+\b[ \t]*(?:marks?)?[ \t]*\)?', '', line, flags=re.I)
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
    cleaned = re.sub(r'[\s=\-_:]+$', '', cleaned)
    return cleaned.strip()


def _is_heading_line(line: str) -> bool:
    line_s = line.strip()
    if not line_s:
        return False
    heading_pats = [
        r'^(?:write\s+(?:short|brief|a\s+short)\s+notes?\s+on|write\s+in\s+short\s+the\s+following|writeshortnoteson)\b.*[:\-]?$',
        r'^(?:write\s*(?:the\s*)?differences?\s*between|writethedifferencesbetween|differentiate\s*between)\b.*[:\-]?$',
        r'^(?:enumerate|outline\s+the\s+differences?\s*between)\s*[:\-]',
        r'^(?:explain\s+why[\?:]?|explain\s+the\s+following[\?:]?|give\s+reasons?\s+(?:for|why)[\?:]?)',
        r'^(?:structured\s+long\s+essay|long\s+essay|short\s+notes?|brief\s+notes?)\b.*[:\-]?$',
        r'^(?:multiple\s+choice(?:\s+[a-z]+)?\.?|objective\s+[a-z]+\.?)'
    ]
    for pat in heading_pats:
        if re.search(pat, line_s, re.I):
            return True
    return False

