"""
Correction Validator for Vision proposed adjustments.

Enforces strict rules:
1. Anti-hallucination: Question text must actually exist in the OCR/instruction text.
2. Safe transformations: Verifies boundaries and sections.
3. Does not blindly overwrite: Only applies verified corrections.
4. Marks questionable proposals as needs_manual_review.
"""

import re
import logging
from typing import List, Tuple, Set, Optional

from app.models.schemas import (
    QuestionPaper,
    Question,
    QuestionType,
    ReviewStatus,
    MCQOption,
    VisionVerificationResult,
    VisionQuestionCorrection,
    VisionInstructionCorrection,
    ConfidenceLevel,
)
from app.services.classification.question_classifier import classify_question

logger = logging.getLogger(__name__)

_WORD_TOKEN_RE = re.compile(r'\b[a-zA-Z0-9]{3,}\b')


_METADATA_FIELD_MAP = {
    "university": "university",
    "university_name": "university",
    "college": "university",
    "institution": "university",
    "board": "university",
    "subject": "subject",
    "subject_name": "subject",
    "course": "subject",
    "paper_name": "subject",
    "exam_year": "exam_year",
    "year": "exam_year",
    "examination_year": "exam_year",
    "exam_month": "exam_month",
    "month": "exam_month",
    "session_code": "session_code",
    "paper_code": "paper_code",
    "qp_code": "paper_code",
    "code": "session_code",
    "anqp_code": "session_code",
    "sub_code": "session_code",
    "subject_code": "session_code",
    "max_marks": "max_marks",
    "total_marks": "max_marks",
    "marks": "max_marks",
    "full_marks": "max_marks",
    "m_marks": "max_marks",
    "duration": "duration",
    "time": "duration",
    "time_allowed": "duration",
    "exam_duration": "duration",
}


def _tokenize(text: str) -> Set[str]:
    """Tokenize text into lowercase alphanumeric words (min 3 chars)."""
    if not text:
        return set()
    return set(w.lower() for w in _WORD_TOKEN_RE.findall(text))


def _text_grounded_in_source(text: str, source_text: str, threshold: float = 0.55) -> bool:
    """
    Verifies that proposed text is genuinely grounded in source OCR text.
    Prevents LLM hallucinations of questions that don't exist on the page.
    """
    if not text or not source_text:
        return False

    # Direct substring match
    if text.strip().lower() in source_text.lower():
        return True

    text_tokens = _tokenize(text)
    if not text_tokens:
        return False

    source_tokens = _tokenize(source_text)
    overlap = len(text_tokens.intersection(source_tokens))
    overlap_ratio = overlap / len(text_tokens)

    return overlap_ratio >= threshold


def validate_and_filter_corrections(
    vision_result: VisionVerificationResult,
    paper: QuestionPaper,
    page_ocr_text: str,
) -> VisionVerificationResult:
    """
    Inspects proposed corrections from the Vision model and filters out
    any invalid or hallucinated entries.
    """
    combined_source_text = page_ocr_text + " " + " ".join(paper.instructions or [])
    if paper.raw_text:
        combined_source_text += " " + paper.raw_text

    validated_q_corrections: List[VisionQuestionCorrection] = []
    for q_corr in vision_result.question_corrections:
        # Check text grounding
        if q_corr.action in ("ADD", "MODIFY", "MOVE", "REPLACE"):
            if not q_corr.text or not q_corr.text.strip():
                logger.warning(f"[VisionValidator] Rejecting empty text for Q{q_corr.question_number}")
                vision_result.needs_manual_review = True
                continue

            # If vision confidence is high (>= 0.85), trust the multimodal vision model's direct image reading
            # even when OCR dropped lines on faint/noisy/camera-tilted scans
            if vision_result.confidence < 0.85 and not _text_grounded_in_source(q_corr.text, combined_source_text):
                logger.warning(
                    f"[VisionValidator] Anti-hallucination trigger: Proposed Q{q_corr.question_number} "
                    f"text not grounded in OCR text ({q_corr.text[:60]}...). Rejecting proposal."
                )
                vision_result.needs_manual_review = True
                vision_result.review_reason = (
                    f"Vision suggested Q{q_corr.question_number} with text not found in OCR."
                )
                continue

        validated_q_corrections.append(q_corr)

    vision_result.question_corrections = validated_q_corrections

    # Validate instruction removals
    validated_inst_corrections: List[VisionInstructionCorrection] = []
    for inst_corr in vision_result.instruction_corrections:
        if inst_corr.action == "REMOVE" and inst_corr.text:
            # Verify text exists in current instructions
            found = False
            for cur_inst in (paper.instructions or []):
                if _text_grounded_in_source(inst_corr.text, cur_inst, threshold=0.50):
                    found = True
                    break
            if found or inst_corr.index is not None:
                validated_inst_corrections.append(inst_corr)
        else:
            validated_inst_corrections.append(inst_corr)

    vision_result.instruction_corrections = validated_inst_corrections

    # Validate metadata corrections
    validated_meta_corrections = []
    for meta_corr in vision_result.metadata_corrections:
        if not meta_corr.field or not meta_corr.value:
            continue
        field_norm = meta_corr.field.strip().lower()
        canonical = _METADATA_FIELD_MAP.get(field_norm)
        if not canonical:
            continue
        val = str(meta_corr.value).strip()
        if not val or val.lower() in ("null", "none", "n/a", "undefined"):
            continue
        meta_corr.field = canonical
        meta_corr.value = val
        validated_meta_corrections.append(meta_corr)

    vision_result.metadata_corrections = validated_meta_corrections

    return vision_result


def _norm_sec(s: Optional[str]) -> str:
    """Normalize section label for robust matching, e.g. 'Sectiom A' -> 'SECTIONA'."""
    if not s:
        return ""
    clean = re.sub(r'[\s\(\)\[\]\-_:]', '', str(s)).upper()
    clean = clean.replace("SECTIOM", "SECTION").replace("SECTON", "SECTION")
    clean = clean.replace("PART", "SECTION").replace("GROUP", "SECTION")
    return clean


def apply_validated_corrections(
    paper: QuestionPaper,
    vision_result: VisionVerificationResult,
) -> Tuple[QuestionPaper, int]:
    """
    Applies validated corrections to a QuestionPaper.
    Returns (updated_paper, count_of_corrections_applied).
    """
    corrections_applied = 0

    # 1. Apply instructions if provided by high-fidelity Vision
    if vision_result.instructions and len(vision_result.instructions) > 0:
        if (paper.page_range and len(paper.page_range) == 2 and
                paper.page_range[1] > paper.page_range[0] and paper.instructions):
            # Multi-page paper: merge instructions without clobbering previous pages
            merged_inst = list(paper.instructions)
            for new_inst in vision_result.instructions:
                s = str(new_inst).strip()
                if s and not any(_text_grounded_in_source(s, cur, 0.70) for cur in merged_inst):
                    merged_inst.append(s)
            paper.instructions = merged_inst
        else:
            paper.instructions = [str(x).strip() for x in vision_result.instructions if str(x).strip()]
        corrections_applied += 1
        logger.info(f"[VisionValidator] Replaced instructions with {len(paper.instructions)} verified items from Vision.")
    elif paper.instructions:
        new_instructions = list(paper.instructions)
        for inst_corr in vision_result.instruction_corrections:
            if inst_corr.action == "REMOVE":
                if inst_corr.text:
                    cleaned = []
                    for inst in new_instructions:
                        if _text_grounded_in_source(inst_corr.text, inst, threshold=0.40):
                            corrections_applied += 1
                        else:
                            cleaned.append(inst)
                    new_instructions = cleaned
                elif inst_corr.index is not None and 0 <= inst_corr.index < len(new_instructions):
                    new_instructions.pop(inst_corr.index)
                    corrections_applied += 1

        # Also remove instruction items whose text matches questions added/moved by Vision
        for q_corr in vision_result.question_corrections:
            if q_corr.text and len(q_corr.text.strip()) > 10:
                new_instructions = [
                    inst for inst in new_instructions
                    if not _text_grounded_in_source(q_corr.text, inst, threshold=0.45)
                ]

        for item in vision_result.corrections:
            if item.type == "move_text" and item.source == "instructions":
                q_num = item.question_number
                if q_num:
                    pattern = re.compile(rf'^\s*(?:Q\.?\s*{re.escape(q_num)}|\b{re.escape(q_num)}\s*[\.\)]|\({re.escape(q_num)}\))\s+', re.I)
                    new_instructions = [inst for inst in new_instructions if not pattern.search(inst)]

        # Safety clean: Strip any instruction item that still contains per-question marks allocations
        final_instructions = []
        for inst in new_instructions:
            if re.search(r'(?:\(\s*marks?[:\s-]*\d+|\(\s*\d+\s*marks?\s*\)|marks?[:\s-]+\d+\s*each)', inst, re.I):
                corrections_applied += 1
            else:
                final_instructions.append(inst)

        paper.instructions = final_instructions if final_instructions else None

    # 2. Apply question corrections
    # The vision model's duty is strictly to correct alignment, typography, mismatched words,
    # marks, headings, and options. It must NEVER remove or drop existing OCR questions!
    new_questions: List[Question] = list(paper.questions)

    for q_corr in vision_result.question_corrections:
        # User requirement: Vision must never remove questions or drop text
        if q_corr.action == "REMOVE":
            continue

        q_corr_sec_norm = _norm_sec(q_corr.section)
        clean_target_sec = re.sub(r'^[\s\(\[]+|[\s\)\]]+$', '', q_corr.section).strip() if q_corr.section else None
        q_corr_num = re.sub(r'^[Qq][\.\s]*', '', q_corr.question_number).strip()

        # Step A: Attempt to match an existing question in the paper
        matched_q: Optional[Question] = None

        # Priority 1: Match by both question number AND section
        for q in new_questions:
            if q.number.strip() == q_corr_num and (not q_corr_sec_norm or _norm_sec(q.part) == q_corr_sec_norm):
                matched_q = q
                break

        # Priority 2: Match by text grounding / token overlap (for OCR typos or mislabeled sections)
        if matched_q is None and q_corr.text:
            for q in new_questions:
                if q.text and (_text_grounded_in_source(q_corr.text, q.text, threshold=0.45) or
                               _text_grounded_in_source(q.text, q_corr.text, threshold=0.45)):
                    matched_q = q
                    break

        # Priority 3: Match by question number alone if unambiguous
        if matched_q is None:
            same_num_qs = [q for q in new_questions if q.number.strip() == q_corr_num]
            if len(same_num_qs) == 1:
                matched_q = same_num_qs[0]

        # Step B: Apply corrections to matched question
        if matched_q is not None:
            # Correct mismatched words and broken OCR alignment
            if q_corr.text and len(q_corr.text.strip()) > 5:
                matched_q.text = q_corr.text
            if q_corr.marks:
                matched_q.marks = q_corr.marks
            if clean_target_sec and (not matched_q.part or "sectiom" in matched_q.part.lower() or matched_q.part.lower() in ("none", "part-a", "part a")):
                matched_q.part = clean_target_sec
            if q_corr.heading:
                matched_q.heading = q_corr.heading
            if q_corr.options:
                mcq_opts = []
                for opt in q_corr.options:
                    if isinstance(opt, dict):
                        mcq_opts.append(MCQOption(label=opt.get("label", ""), text=opt.get("text", "")))
                    elif isinstance(opt, MCQOption):
                        mcq_opts.append(opt)
                matched_q.options = mcq_opts
            if q_corr.type:
                for qt in QuestionType:
                    if qt.value.lower() == str(q_corr.type).lower():
                        matched_q.type = qt
                        break
            matched_q.confidence = max(matched_q.confidence or 0.0, 0.95)
            corrections_applied += 1
        else:
            # Genuinely new question detected by Vision that OCR missed completely
            if q_corr.text and len(q_corr.text.strip()) > 20:
                q_text = q_corr.text
                raw_dict = {"text": q_text, "heading": q_corr.heading, "marks": q_corr.marks, "options": q_corr.options}
                q_type, q_conf = classify_question(raw_dict, clean_target_sec)
                mcq_opts = [
                    MCQOption(label=opt.get("label", ""), text=opt.get("text", "")) if isinstance(opt, dict) else opt
                    for opt in q_corr.options
                ] if q_corr.options else None

                new_q = Question(
                    number=q_corr_num,
                    text=q_text,
                    type=q_type,
                    part=clean_target_sec,
                    heading=q_corr.heading,
                    marks=q_corr.marks,
                    options=mcq_opts,
                    confidence=round(min(q_corr.confidence, q_conf), 3),
                    status=ReviewStatus.OK,
                )
                new_questions.append(new_q)
                corrections_applied += 1

    # Ensure no duplicate questions exist within the same section
    deduped_questions: List[Question] = []
    seen_sec_nums = set()
    for q in new_questions:
        key = (_norm_sec(q.part), q.number.strip())
        if key not in seen_sec_nums:
            seen_sec_nums.add(key)
            deduped_questions.append(q)
    new_questions = deduped_questions

    # If any questions lack part attribution while others have PART-A / PART-B, attribute remaining
    has_part_a = any(_norm_sec(q.part) == "PARTA" for q in new_questions)
    for q in new_questions:
        if not q.part and has_part_a:
            q.part = "PART-B"

    # Sort questions logically: PART-A first, then PART-B, then by question number
    def _sort_key(q: Question) -> Tuple[str, int, str]:
        part_str = (q.part or "").upper()
        part_prio = 0 if "PART-A" in part_str or "PART A" in part_str or "PART - I" in part_str or "SECTION A" in part_str else 1
        num_digits = re.findall(r'\d+', q.number)
        num_val = int(num_digits[0]) if num_digits else 999
        return (str(part_prio), num_val, q.number)

    new_questions.sort(key=_sort_key)
    paper.questions = new_questions
    if paper.parts:
        paper.parts = [re.sub(r'\bSectio[nm]\b', 'Section', p, flags=re.I) for p in paper.parts]

    # 3. Update metadata if corrections present
    if not paper.metadata:
        from app.models.schemas import ExamMetadata
        paper.metadata = ExamMetadata()

    for meta_corr in vision_result.metadata_corrections:
        field = meta_corr.field.lower()
        canonical_field = _METADATA_FIELD_MAP.get(field, field)
        val = str(meta_corr.value).strip() if meta_corr.value else None
        if not val:
            continue

        # Format specific fields cleanly
        if canonical_field == "max_marks":
            m_digits = re.findall(r'\d+', val)
            if m_digits:
                val = m_digits[0]
        elif canonical_field == "exam_year":
            y_digits = re.findall(r'\b(20\d{2}|19\d{2})\b', val)
            if y_digits:
                val = y_digits[0]

        if hasattr(paper.metadata, canonical_field):
            curr_val = getattr(paper.metadata, canonical_field)
            # Only update if current value is missing or if new value is more complete/different
            if not curr_val or curr_val.strip().lower() != val.lower():
                setattr(paper.metadata, canonical_field, val)
                corrections_applied += 1
                logger.info(
                    f"[VisionValidator] Corrected metadata: {canonical_field} = '{val}' "
                    f"(was '{curr_val or 'empty'}')"
                )
                # Remove from missing_fields
                if paper.metadata.missing_fields:
                    paper.metadata.missing_fields = [
                        f for f in paper.metadata.missing_fields
                        if f.lower() not in (canonical_field, field)
                    ]
                    if canonical_field in ("exam_year", "exam_month"):
                        paper.metadata.missing_fields = [
                            f for f in paper.metadata.missing_fields if f != "exam_date"
                        ]

    # 4. Update parts list
    distinct_parts = list(dict.fromkeys(q.part for q in new_questions if q.part))
    if distinct_parts:
        paper.parts = distinct_parts

    return paper, corrections_applied
