"""
Multi-dimensional Confidence Scoring & Quality Audit Layer.

Evaluates:
1. OCR Quality (recognition confidence, token noise, unreadable text)
2. Document Structure Quality (header integrity, instruction leakage, section transitions)
3. Question Segmentation Quality (boundary clarity, embedded boundaries, subparts)
4. Question Numbering Consistency (sequence continuity, missing Q1, unexplained gaps)
5. Metadata Quality (required vs optional fields)
6. Classification Quality (category certainty)
7. Overall Confidence with structural hard-caps
8. Quality status (HIGH, MEDIUM, LOW, REVIEW_REQUIRED)
9. Vision Verification Trigger Flag (needs_visual_verification)
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple

from app.models.schemas import (
    Question, QuestionPaper, ExamMetadata,
    AnomalySeverity, StructuralAnomaly, ConfidenceLevel,
    QuestionConfidenceBreakdown, PaperConfidenceBreakdown,
    PageConfidenceAudit, ReviewStatus, QuestionType
)

logger = logging.getLogger(__name__)

# Configurable thresholds
CONFIDENCE_HIGH_THRESHOLD   = 0.85
CONFIDENCE_REVIEW_THRESHOLD = 0.65
CONFIDENCE_LOW_THRESHOLD    = 0.45

# Question-like signals inside non-question regions
_QUESTION_VERBS_RE = re.compile(
    r'\b(?:classify|explain|describe|discuss|write(?:\s+(?:short|brief|a\s+short))?\s+notes?|'
    r'define|list|compare|mention|enumerate|differentiate|distinguish|illustrate|'
    r'evaluate|critically\s+evaluate|give\s+an\s+account|how\s+will\s+you|what\s+is|'
    r'what\s+are|why\s+is|draw|outline|state|name|suggest|give\s+reasons?)\b',
    re.I
)

_MARKS_RE = re.compile(
    r'(?:\(\s*marks?[:\s-]*\d+|\(\s*\d+\s*marks?\s*\)|\[\s*\d+\s*marks?\s*\]|'
    r'\[\s*\d+\s*x\s*\d+\s*=\s*\d+\s*\]|\(\s*marks?:\s*\d+\s*each\s*\)|'
    r'marks?[:\s-]+\d+\s*each|\(\s*\d+\s*each\s*\)|'
    r'\(\s*marks?-\s*\d+\s*\))',
    re.I
)

_NUMBERED_ITEM_RE = re.compile(
    r'(?:^|\n)\s*(?:Q\.?\s*\d+|\b\d+\s*[\.\)]|\(\d+\))\s+[A-Za-z]',
    re.I
)

_SUBPARTS_RE = re.compile(
    r'(?:^|\s)(?:[a-d]\.|\([a-d]\)|\(i{1,3}\))\s+[A-Za-z]',
    re.I
)

_SECTION_MARKER_RE = re.compile(
    r'\((?:PART|SECTION)\s*[-:]?\s*[A-Z]\)|(?:PART|SECTION)\s*[-:]?\s*[A-Z]',
    re.I
)

_EMBEDDED_BOUNDARY_RE = re.compile(
    r'\n\s*(?:Q\.?\s*\d+|\b\d{1,2}\s*[\.\)]|\(\d{1,2}\))\s+[A-Z]',
    re.I
)

_NOISE_CHAR_RE = re.compile(r'[~^|\\_`§±¥]')

_GENUINE_INSTRUCTION_RE = re.compile(
    r'\b(?:illustrate\s+(?:your\s+)?answers?|answer\s+all(?:\s+of\s+the\s+following)?\s+questions?|'
    r'attempt\s+all(?:\s+of\s+the\s+following)?\s+questions?|attempt\s+all\s+the\s+questions\s+serially|'
    r'draw\s+(?:neat\s+)?diagrams?\s+wherever|write\s+(?:your\s+)?(?:roll|hall\s*ticket|reg|index)\s*(?:no|number)|'
    r'use\s+of\s+calculators?|answers?\s+must\s+be\s+written|read\s+(?:the\s+)?instructions|'
    r'figures\s+to\s+the\s+right\s+indicate|all\s+questions\s+carry\s+equal|'
    r'separate\s+answer\s+(?:booklets?|books?|sheets?)|answer\s+booklets?|'
    r'do\s+not\s+leave\s+blank\s+spaces?|cross\s+(?:any\s+)?blank\s+spaces?|end\s+stamp|'
    r'do\s+not\s+use\s+(?:black\s+lead|graphite\s+pencil|pencil|calculators?)|unfair\s*means|'
    r'instructions?\s+at\s+[\d\s,&]+\s+are\s+not\s+complied|complied\s+with|'
    r'candidate\s+must\s+verify|two\s+answer\s+booklets?\s+to\s+be\s+used)\b',
    re.I
)


# ===========================================================================
# 1. OCR Confidence
# ===========================================================================

def audit_ocr(paper_pages: List[Dict[str, Any]]) -> Tuple[Optional[float], bool, List[StructuralAnomaly]]:
    """
    Evaluates real OCR quality across the paper's pages.
    Differentiates between digital pages (100% native fidelity) and scanned pages.
    """
    anomalies: List[StructuralAnomaly] = []
    if not paper_pages:
        return None, False, anomalies

    scanned_pages = [p for p in paper_pages if p.get("ocr_used")]
    if not scanned_pages:
        # All pages were digital vector text
        return 1.0, True, anomalies

    confs = [p.get("ocr_conf", 0.0) for p in scanned_pages if p.get("ocr_conf") is not None]
    if not confs:
        return None, False, anomalies

    base_ocr_conf = sum(confs) / len(confs)
    penalty = 0.0

    # Token noise audit
    combined_text = "\n".join(p.get("text", "") for p in paper_pages)
    if combined_text:
        noise_chars = len(_NOISE_CHAR_RE.findall(combined_text))
        noise_ratio = noise_chars / max(len(combined_text), 1)
        if noise_ratio > 0.05:
            penalty += min(0.25, noise_ratio * 3)
            anomalies.append(StructuralAnomaly(
                type="excessive_ocr_noise",
                severity=AnomalySeverity.MEDIUM if noise_ratio < 0.10 else AnomalySeverity.HIGH,
                text_preview=combined_text[:100],
                reason=f"Suspicious OCR symbol noise detected ({noise_ratio:.1%} of text characters are noise)."
            ))

        # Check for unreadable/sparse scanned pages (< 80 chars on a scanned page)
        for p in scanned_pages:
            pg_len = len(p.get("text", "").strip())
            if pg_len < 80:
                penalty += 0.15
                anomalies.append(StructuralAnomaly(
                    type="low_page_text_density",
                    severity=AnomalySeverity.MEDIUM,
                    reason=f"Page {p.get('page_index', 0) + 1} has very low extracted text density ({pg_len} chars)."
                ))
                break

    final_ocr_conf = max(0.10, min(1.0, round(base_ocr_conf - penalty, 3)))
    return final_ocr_conf, True, anomalies


# ===========================================================================
# 2. Document Structure Confidence (Most Critical Dimension)
# ===========================================================================

def audit_document_structure(
    instructions: Optional[List[str]],
    sections: Optional[List[str]],
    header_text: Optional[str],
    questions: List[Question],
    raw_text: str
) -> Tuple[float, List[StructuralAnomaly]]:
    """
    Audits the structural integrity of the paper.
    Detects if real questions or marks were misplaced into instructions or headers.
    """
    anomalies: List[StructuralAnomaly] = []
    score = 1.0

    # 1. Check for Question-Like content trapped in Instructions
    if instructions:
        for idx, inst in enumerate(instructions):
            inst_clean = inst.strip()
            if not inst_clean:
                continue

            has_marks = bool(_MARKS_RE.search(inst_clean))
            has_verbs = bool(_QUESTION_VERBS_RE.search(inst_clean))
            has_numbering = bool(_NUMBERED_ITEM_RE.search(inst_clean))
            has_subparts = bool(_SUBPARTS_RE.search(inst_clean))
            has_section = bool(_SECTION_MARKER_RE.search(inst_clean))
            is_long = len(inst_clean) > 80

            is_genuine = bool(_GENUINE_INSTRUCTION_RE.search(inst_clean))
            is_section_guideline = bool(re.search(
                r'\b(?:separate\s+answer|attempted\s+in|written\s+in|booklets?|sheets?|to\s+be\s+used|both\s+parts)\b',
                inst_clean, re.I
            ))

            # CRITICAL FAILURE: Marks detected inside instructions
            if has_marks:
                marks_match = _MARKS_RE.search(inst_clean).group(0)
                score -= 0.35
                anomalies.append(StructuralAnomaly(
                    type="question_like_text_in_instructions",
                    severity=AnomalySeverity.CRITICAL,
                    text_preview=inst_clean[:120],
                    reason=f"Instruction item {idx + 1} contains per-question marks allocation '{marks_match}', "
                           f"proving real questions were trapped inside instructions."
                ))
            elif has_verbs and (has_numbering or has_subparts) and not is_genuine:
                score -= 0.25
                anomalies.append(StructuralAnomaly(
                    type="question_like_text_in_instructions",
                    severity=AnomalySeverity.HIGH,
                    text_preview=inst_clean[:120],
                    reason=f"Instruction item {idx + 1} contains question action verbs and structured question numbering/subparts."
                ))
            elif has_section and not is_section_guideline and len(inst_clean) < 40:
                score -= 0.15
                anomalies.append(StructuralAnomaly(
                    type="section_marker_in_instructions",
                    severity=AnomalySeverity.MEDIUM,
                    text_preview=inst_clean[:100],
                    reason=f"Section marker trapped inside instructions block instead of defining a section partition."
                ))
            elif has_numbering and is_long and has_verbs and not is_genuine:
                score -= 0.10
                anomalies.append(StructuralAnomaly(
                    type="suspicious_long_numbered_instruction",
                    severity=AnomalySeverity.LOW,
                    text_preview=inst_clean[:100],
                    reason=f"Instruction item {idx + 1} has long numbered text that may contain misclassified content."
                ))

    # 2. Check instruction text length ratio vs question text length
    inst_total_len = sum(len(i) for i in (instructions or []))
    q_total_len = sum(len(q.text) for q in questions)
    if inst_total_len > 300 and inst_total_len > q_total_len:
        score -= 0.25
        anomalies.append(StructuralAnomaly(
            type="excessive_instruction_text_ratio",
            severity=AnomalySeverity.HIGH,
            reason=f"Instructions contain {inst_total_len} characters while extracted questions contain only {q_total_len} characters. Content is trapped in header/instructions."
        ))

    # 3. Check for total absence of extracted questions
    if not questions:
        score -= 0.70
        anomalies.append(StructuralAnomaly(
            type="no_questions_extracted",
            severity=AnomalySeverity.CRITICAL,
            reason="Zero questions were extracted from this question paper."
        ))

    # 4. Check for presence of examination header
    if not header_text or len(header_text.strip()) < 10:
        score -= 0.15
        anomalies.append(StructuralAnomaly(
            type="weak_header_detected",
            severity=AnomalySeverity.LOW,
            reason="Examination header block was absent or unusually brief."
        ))

    final_score = max(0.05, min(1.0, round(score, 3)))
    return final_score, anomalies


# ===========================================================================
# 3. Question Numbering & Consistency Audit
# ===========================================================================

def audit_question_numbering(questions: List[Question]) -> Tuple[float, List[StructuralAnomaly]]:
    """
    Audits the question numbering sequence for missing questions, gaps, and duplicate numbers.
    """
    anomalies: List[StructuralAnomaly] = []
    if not questions:
        return 0.10, [StructuralAnomaly(
            type="no_questions_to_audit",
            severity=AnomalySeverity.HIGH,
            reason="Cannot audit numbering consistency because no questions exist."
        )]

    score = 1.0

    def parse_num(num_str: str) -> Optional[int]:
        m = re.search(r'\b(\d+)\b', num_str)
        return int(m.group(1)) if m else None

    # Check first question
    first_num = parse_num(questions[0].number)
    if first_num is not None and first_num > 1:
        # First question is Q2 or higher! This strongly indicates Q1 was swallowed or missed
        score -= 0.35
        anomalies.append(StructuralAnomaly(
            type="question_numbering_gap",
            severity=AnomalySeverity.HIGH,
            reason=f"First extracted question is Q{questions[0].number}; expected Q1. Initial questions appear lost or unsegmented."
        ))

    # Check for duplicate numbers within the same section/part
    seen_in_part: Dict[Optional[str], set] = {}
    for q in questions:
        part_key = q.part or "DEFAULT"
        seen_in_part.setdefault(part_key, set())
        clean_num = q.number.strip().lower()
        if clean_num in seen_in_part[part_key]:
            score -= 0.20
            anomalies.append(StructuralAnomaly(
                type="duplicate_question_number",
                severity=AnomalySeverity.MEDIUM,
                reason=f"Duplicate question number Q{q.number} detected in section '{q.part or 'Main'}'."
            ))
        else:
            seen_in_part[part_key].add(clean_num)

    # Check for sequence continuity within sections
    by_section: Dict[Optional[str], List[int]] = {}
    for q in questions:
        n = parse_num(q.number)
        if n is not None:
            by_section.setdefault(q.part, []).append(n)

    for section_name, nums in by_section.items():
        if len(nums) >= 2:
            for i in range(len(nums) - 1):
                curr, nxt = nums[i], nums[i + 1]
                if nxt > curr + 1:
                    gap_size = nxt - curr - 1
                    score -= min(0.25, 0.10 * gap_size)
                    anomalies.append(StructuralAnomaly(
                        type="question_numbering_gap",
                        severity=AnomalySeverity.MEDIUM,
                        reason=f"Numbering gap detected in section '{section_name or 'Main'}': Q{curr} is followed by Q{nxt} (missing {gap_size} question(s))."
                    ))

    final_score = max(0.10, min(1.0, round(score, 3)))
    return final_score, anomalies


# ===========================================================================
# 4. Question Boundary Quality Audit
# ===========================================================================

def audit_question_boundaries(questions: List[Question], ocr_conf: Optional[float]) -> Tuple[float, List[StructuralAnomaly]]:
    """
    Evaluates each question's boundary quality, attaches individual breakdowns,
    and returns overall segmentation confidence.
    """
    paper_anomalies: List[StructuralAnomaly] = []
    if not questions:
        return 0.10, paper_anomalies

    q_boundary_scores: List[float] = []

    for q in questions:
        q_anomalies: List[StructuralAnomaly] = []
        b_score = 1.0

        # Check question length
        text_len = len(q.text.strip())
        if text_len < 12:
            b_score -= 0.40
            q_anomalies.append(StructuralAnomaly(
                type="suspiciously_short_question",
                severity=AnomalySeverity.HIGH,
                text_preview=q.text,
                reason=f"Question body is extremely short ({text_len} chars) and may be an OCR fragment or truncated."
            ))
        elif text_len < 25:
            b_score -= 0.15

        # Check for unsegmented embedded questions in body
        embedded_match = _EMBEDDED_BOUNDARY_RE.search(q.text)
        if embedded_match:
            b_score -= 0.35
            snippet = embedded_match.group(0).strip()
            anom = StructuralAnomaly(
                type="embedded_question_boundary",
                severity=AnomalySeverity.HIGH,
                text_preview=q.text[:100],
                reason=f"Question body appears to contain an unsegmented subsequent question boundary '{snippet}'."
            )
            q_anomalies.append(anom)
            paper_anomalies.append(anom)

        # Check for header/instruction leakage into question body
        if re.search(r'\b(?:time:\s*three\s*hours|maximum\s*marks|attempt\s*all\s*questions)\b', q.text, re.I):
            b_score -= 0.30
            anom = StructuralAnomaly(
                type="header_leakage_in_question",
                severity=AnomalySeverity.MEDIUM,
                text_preview=q.text[:100],
                reason="Question body contains examination header or instruction phrases."
            )
            q_anomalies.append(anom)
            paper_anomalies.append(anom)

        # Reward presence of explicit marks
        if q.marks:
            b_score = min(1.0, b_score + 0.05)

        # Reward clean MCQ options
        if q.type == QuestionType.MCQ:
            if not q.options or len(q.options) < 2:
                b_score -= 0.30
                q_anomalies.append(StructuralAnomaly(
                    type="mcq_missing_options",
                    severity=AnomalySeverity.MEDIUM,
                    reason="Question classified as MCQ but fewer than 2 option choices were extracted."
                ))

        final_b_score = max(0.10, min(1.0, round(b_score, 3)))
        q_boundary_scores.append(final_b_score)

        # Question classification confidence (already populated by classify_question)
        class_conf = q.confidence

        # Question overall confidence = weighted boundary (40%) + classification (35%) + OCR (25%)
        effective_ocr = ocr_conf if ocr_conf is not None else 0.90
        q_overall = (0.40 * final_b_score) + (0.35 * class_conf) + (0.25 * effective_ocr)
        if q_anomalies and any(a.severity in (AnomalySeverity.HIGH, AnomalySeverity.CRITICAL) for a in q_anomalies):
            q_overall = min(q_overall, 0.55)

        q_overall = max(0.10, min(1.0, round(q_overall, 3)))

        # Assign to question model
        q.confidence_breakdown = QuestionConfidenceBreakdown(
            ocr=round(effective_ocr, 3),
            boundary=final_b_score,
            classification=round(class_conf, 3),
            overall=q_overall
        )
        q.confidence = q_overall
        q.anomalies = q_anomalies

        if q_overall >= CONFIDENCE_HIGH_THRESHOLD:
            q.confidence_level = ConfidenceLevel.HIGH
        elif q_overall >= CONFIDENCE_REVIEW_THRESHOLD:
            q.confidence_level = ConfidenceLevel.MEDIUM
        elif q_overall >= CONFIDENCE_LOW_THRESHOLD:
            q.confidence_level = ConfidenceLevel.LOW
        else:
            q.confidence_level = ConfidenceLevel.REVIEW_REQUIRED

        if q_overall < CONFIDENCE_REVIEW_THRESHOLD:
            q.status = ReviewStatus.NEEDS_REVIEW

    segmentation_confidence = sum(q_boundary_scores) / len(q_boundary_scores)
    return round(segmentation_confidence, 3), paper_anomalies


# ===========================================================================
# 5. Metadata Quality Audit
# ===========================================================================

def audit_metadata(metadata: ExamMetadata) -> Tuple[float, List[StructuralAnomaly]]:
    """
    Evaluates required vs optional metadata fields.
    Does not heavily penalize optional fields.
    """
    anomalies: List[StructuralAnomaly] = []
    missing_fields: List[str] = []
    score = 0.0

    # Required fields (total 0.80)
    if metadata.university and len(metadata.university.strip()) > 3:
        score += 0.35
    else:
        missing_fields.append("university")
        anomalies.append(StructuralAnomaly(
            type="missing_required_metadata",
            severity=AnomalySeverity.MEDIUM,
            reason="University name could not be extracted from header."
        ))

    if metadata.subject and len(metadata.subject.strip()) > 3:
        score += 0.25
    else:
        missing_fields.append("subject")
        anomalies.append(StructuralAnomaly(
            type="missing_required_metadata",
            severity=AnomalySeverity.LOW,
            reason="Examination subject could not be extracted."
        ))

    if metadata.exam_year or metadata.exam_month:
        score += 0.20
    else:
        missing_fields.append("exam_date")

    # Optional fields (total 0.20)
    if metadata.session_code:
        score += 0.10
    else:
        missing_fields.append("session_code")

    if metadata.max_marks:
        score += 0.05

    if metadata.duration:
        score += 0.05

    metadata.missing_fields = missing_fields
    final_meta_conf = max(0.10, min(1.0, round(score, 3)))
    metadata.confidence = final_meta_conf

    if "university" in missing_fields or final_meta_conf < CONFIDENCE_REVIEW_THRESHOLD:
        metadata.status = ReviewStatus.NEEDS_REVIEW

    return final_meta_conf, anomalies


# ===========================================================================
# 6. Overall Paper Confidence & Hard Caps
# ===========================================================================

def calculate_overall_confidence(
    ocr_conf: Optional[float],
    structure_conf: float,
    segmentation_conf: float,
    metadata_conf: float,
    classification_conf: float,
    consistency_conf: float,
    anomalies: List[StructuralAnomaly]
) -> Tuple[float, ConfidenceLevel, bool, List[str]]:
    """
    Calculates overall paper confidence with negative override / hard caps.
    Severe structural anomalies or lost questions CANNOT be masked by high classification scores.
    """
    effective_ocr = ocr_conf if ocr_conf is not None else 0.85

    # Suggested baseline weights:
    # OCR: 20%, Structure: 30%, Segmentation: 20%, Metadata: 10%, Classification: 10%, Consistency: 10%
    raw_score = (
        (0.20 * effective_ocr) +
        (0.30 * structure_conf) +
        (0.20 * segmentation_conf) +
        (0.10 * metadata_conf) +
        (0.10 * classification_conf) +
        (0.10 * consistency_conf)
    )

    # -----------------------------------------------------------------------
    # HARD ANOMALY PENALTIES & CAPS
    # -----------------------------------------------------------------------
    final_score = raw_score
    critical_anomalies = [a for a in anomalies if a.severity == AnomalySeverity.CRITICAL]
    high_anomalies     = [a for a in anomalies if a.severity == AnomalySeverity.HIGH]

    # Rule 1: Critical anomaly caps overall confidence at 0.40
    if critical_anomalies:
        final_score = min(final_score, 0.40)

    # Rule 2: High anomaly caps overall confidence at 0.58
    elif high_anomalies or structure_conf < 0.50:
        final_score = min(final_score, 0.58)

    # Rule 3: Structural Domination Cap: Overall can NEVER exceed structure_conf + 0.15
    final_score = min(final_score, structure_conf + 0.15)

    final_score = max(0.05, min(1.0, round(final_score, 3)))

    # -----------------------------------------------------------------------
    # Status Level Assignment
    # -----------------------------------------------------------------------
    if critical_anomalies or final_score < CONFIDENCE_LOW_THRESHOLD:
        status = ConfidenceLevel.REVIEW_REQUIRED
    elif high_anomalies or final_score < CONFIDENCE_REVIEW_THRESHOLD:
        status = ConfidenceLevel.LOW
    elif final_score < CONFIDENCE_HIGH_THRESHOLD:
        status = ConfidenceLevel.MEDIUM
    else:
        status = ConfidenceLevel.HIGH

    # -----------------------------------------------------------------------
    # Vision Trigger Evaluation
    # -----------------------------------------------------------------------
    needs_vision = False
    vision_reasons: List[str] = []

    if status in (ConfidenceLevel.LOW, ConfidenceLevel.REVIEW_REQUIRED):
        needs_vision = True

    for a in critical_anomalies + high_anomalies:
        needs_vision = True
        if a.type not in vision_reasons:
            vision_reasons.append(a.type)

    if structure_conf < CONFIDENCE_REVIEW_THRESHOLD and "low_structure_confidence" not in vision_reasons:
        needs_vision = True
        vision_reasons.append("low_structure_confidence")

    return final_score, status, needs_vision, vision_reasons


# ===========================================================================
# Master Entry Point: Audit Whole Paper
# ===========================================================================

def audit_paper(
    paper: QuestionPaper,
    paper_pages: List[Dict[str, Any]],
    doc_structure: Any
) -> QuestionPaper:
    """
    Full multi-dimensional audit of an extracted QuestionPaper.
    Enriches paper and its questions with breakdown scores, anomalies,
    and visual verification trigger flags.
    """
    all_anomalies: List[StructuralAnomaly] = []

    # 1. OCR Audit
    ocr_conf, ocr_avail, ocr_anomalies = audit_ocr(paper_pages)
    all_anomalies.extend(ocr_anomalies)

    # 2. Document Structure Audit
    instructions = paper.instructions if paper.instructions is not None else (doc_structure.instructions if doc_structure else None)
    sections = paper.parts if paper.parts is not None else (doc_structure.sections if doc_structure else None)
    header_text = doc_structure.header_text if doc_structure else ""
    raw_text = paper.raw_text or ""

    struct_conf, struct_anomalies = audit_document_structure(
        instructions=instructions,
        sections=sections,
        header_text=header_text,
        questions=paper.questions,
        raw_text=raw_text
    )
    all_anomalies.extend(struct_anomalies)

    # 3. Question Numbering Audit
    consist_conf, consist_anomalies = audit_question_numbering(paper.questions)
    all_anomalies.extend(consist_anomalies)

    # 4. Question Boundary Audit
    seg_conf, bound_anomalies = audit_question_boundaries(paper.questions, ocr_conf)
    all_anomalies.extend(bound_anomalies)

    # 5. Metadata Audit
    meta_conf, meta_anomalies = audit_metadata(paper.metadata)
    all_anomalies.extend(meta_anomalies)

    # 6. Classification Audit (average category confidence)
    if paper.questions:
        class_conf = sum(q.confidence_breakdown.classification for q in paper.questions if q.confidence_breakdown) / len(paper.questions)
    else:
        class_conf = 0.50
    class_conf = round(class_conf, 3)

    # 7. Overall Confidence Calculation
    overall_conf, status, needs_vision, vision_reasons = calculate_overall_confidence(
        ocr_conf=ocr_conf,
        structure_conf=struct_conf,
        segmentation_conf=seg_conf,
        metadata_conf=meta_conf,
        classification_conf=class_conf,
        consistency_conf=consist_conf,
        anomalies=all_anomalies
    )

    paper.confidence = PaperConfidenceBreakdown(
        ocr_confidence=ocr_conf,
        ocr_confidence_available=ocr_avail,
        structure_confidence=struct_conf,
        segmentation_confidence=seg_conf,
        metadata_confidence=meta_conf,
        classification_confidence=class_conf,
        consistency_confidence=consist_conf,
        overall_confidence=overall_conf
    )
    paper.confidence_level = status
    paper.needs_visual_verification = needs_vision
    paper.visual_verification_reasons = vision_reasons
    paper.anomalies = all_anomalies

    logger.info(
        f"Paper {paper.paper_index + 1} Quality Audit: Overall={overall_conf:.2f} ({status.value}) | "
        f"Struct={struct_conf:.2f}, OCR={ocr_conf or 'N/A'}, Seg={seg_conf:.2f}, "
        f"Meta={meta_conf:.2f}, Class={class_conf:.2f}, VisionTrigger={needs_vision}"
    )

    return paper


def generate_page_audits(
    paper_pages: List[Dict[str, Any]],
    papers: List[QuestionPaper],
    page_states: Optional[Dict[int, Any]] = None,
    vision_results: Optional[Dict[int, Any]] = None,
) -> List[PageConfidenceAudit]:
    """
    Generates page-level audits for inspection, quality audit display, and Vision tracking.
    """
    from app.models.schemas import PageProcessingState

    page_audits: List[PageConfidenceAudit] = []

    for idx, p in enumerate(paper_pages):
        pg_num = idx + 1
        # Find corresponding paper
        matched_paper = None
        for paper in papers:
            if paper.page_range[0] <= pg_num <= paper.page_range[1]:
                matched_paper = paper
                break

        pg_state = PageProcessingState.FINALIZED
        if page_states and idx in page_states:
            st = page_states[idx]
            if isinstance(st, PageProcessingState):
                pg_state = st
            elif isinstance(st, str) and hasattr(PageProcessingState, st):
                pg_state = PageProcessingState(st)

        v_res = vision_results.get(idx) if vision_results else None
        v_conf = v_res.confidence if v_res else (matched_paper.vision_confidence if matched_paper else None)
        v_status = (
            "CORRECTED" if (v_res and (len(v_res.corrections) + len(v_res.question_corrections) > 0))
            else ("VERIFIED" if (v_res and v_res.is_structure_correct) else (matched_paper.vision_status if matched_paper else None))
        )
        manual_rev = (v_res.needs_manual_review if v_res else False) or (matched_paper.needs_manual_review if matched_paper else False)
        applied_cnt = (len(v_res.corrections) + len(v_res.question_corrections)) if v_res else 0

        if matched_paper and matched_paper.confidence:
            conf = matched_paper.confidence
            page_audits.append(PageConfidenceAudit(
                page=pg_num,
                confidence=conf,
                status=matched_paper.confidence_level,
                anomalies=matched_paper.anomalies,
                needs_visual_verification=matched_paper.needs_visual_verification,
                visual_verification_reasons=matched_paper.visual_verification_reasons,
                processing_state=pg_state,
                vision_confidence=v_conf,
                vision_status=v_status,
                needs_manual_review=manual_rev,
                corrections_applied=applied_cnt,
            ))
        else:
            default_conf = PaperConfidenceBreakdown(
                ocr_confidence=p.get("ocr_conf"),
                ocr_confidence_available=bool(p.get("ocr_used")),
                structure_confidence=0.50,
                segmentation_confidence=0.50,
                metadata_confidence=0.50,
                classification_confidence=0.50,
                consistency_confidence=0.50,
                overall_confidence=0.50
            )
            page_audits.append(PageConfidenceAudit(
                page=pg_num,
                confidence=default_conf,
                status=ConfidenceLevel.LOW,
                anomalies=[],
                needs_visual_verification=True,
                visual_verification_reasons=["unassigned_page"],
                processing_state=pg_state,
                vision_confidence=v_conf,
                vision_status=v_status,
                needs_manual_review=manual_rev,
                corrections_applied=applied_cnt,
            ))

    return page_audits
