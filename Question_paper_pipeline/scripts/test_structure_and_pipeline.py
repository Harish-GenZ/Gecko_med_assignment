import sys
sys.path.insert(0, 'backend')
import json
from pathlib import Path
from app.services.structure.document_structure import analyze_document_structure
from app.services.extraction.metadata_extractor import extract_metadata
from app.services.segmentation.question_segmenter import segment_questions
from app.services.classification.question_classifier import classify_question
from app.models.schemas import Question, QuestionPaper, QuestionType, MCQOption, ReviewStatus

def run_test():
    print("=" * 70)
    print("TEST 1: UHSR Paper (The exact failure case from user)")
    print("=" * 70)
    uhsr_job = json.load(open('backend/storage/jobs/b704ccfa-69ff-4f99-b301-2854cc6fccc7.json'))
    uhsr_raw = uhsr_job['result']['question_papers'][0]['raw_text']

    doc_struct = analyze_document_structure(uhsr_raw)
    print("\n[DOCUMENT STRUCTURE PARSER OUTPUT]")
    print(f"Header lines:       {len(doc_struct.header_text.splitlines())}")
    print(f"Instructions count: {len(doc_struct.instructions)}")
    print(f"Sections detected:  {doc_struct.sections}")
    print(f"Questions lines:    {len(doc_struct.questions_raw_text.splitlines())}")

    print("\n--- EXTRACTED INSTRUCTIONS ---")
    for idx, inst in enumerate(doc_struct.instructions, 1):
        print(f"  {idx}. {inst}")

    print("\n--- INDEPENDENT METADATA EXTRACTION ---")
    meta = extract_metadata(doc_struct.header_text)
    print(f"  University:   {meta.university}")
    print(f"  Subject:      {meta.subject}")
    print(f"  Session Code: {meta.session_code}")
    print(f"  Exam Date:    {meta.exam_month} {meta.exam_year}")
    print(f"  Max Marks:    {meta.max_marks}")
    print(f"  Duration:     {meta.duration}")
    print(f"  Confidence:   {meta.confidence:.2f}")

    print("\n--- QUESTION SEGMENTATION & CLASSIFICATION ---")
    raw_qs = segment_questions(doc_struct.questions_raw_text)
    validated_qs = []
    for rq in raw_qs:
        q_type, conf = classify_question(rq, rq.get('section'))
        opts = None
        if q_type == QuestionType.MCQ and rq.get('options'):
            opts = [MCQOption(label=o['label'], text=o['text']) for o in rq['options']]
        q = Question(
            number=rq['number'],
            text=rq['text'],
            type=q_type,
            part=rq.get('part'),
            heading=rq.get('heading'),
            options=opts,
            marks=rq.get('marks'),
            confidence=round(conf, 3),
            status=ReviewStatus.OK
        )
        validated_qs.append(q)

    print(f"Total genuine questions: {len(validated_qs)}")
    for q in validated_qs:
        opt_str = f" [options: {len(q.options)}]" if q.options else ""
        part_str = f" [{q.part}]" if q.part else ""
        print(f"  Q{q.number:<4} | {q.type.value:<18} | Marks: {str(q.marks):<4}{part_str}{opt_str}")
        print(f"         Preview: {repr(q.text.splitlines()[0][:70])}")

    # Assertions for the failure case
    q_texts = [q.text.lower() for q in validated_qs]
    assert not any("uhsr examinations" in t for t in q_texts), "FAILURE: Header became a question!"
    assert not any("attempt all of the following questions" in t for t in q_texts), "FAILURE: Instruction 1 became a question!"
    assert not any("separate answer booklets" in t for t in q_texts), "FAILURE: Instruction 2 became a question!"
    assert not any("blank spaces" in t for t in q_texts), "FAILURE: Instruction 3 became a question!"
    assert not any("graphite pencil" in t for t in q_texts), "FAILURE: Instruction 4 became a question!"
    assert len(validated_qs) == 7, f"Expected 7 questions, got {len(validated_qs)}"
    print("\n>>> ALL UHSR ASSERTIONS PASSED! No metadata/instructions in questions! <<<")

    print("\n" + "=" * 70)
    print("TEST 2: BIU 20-Paper Batch Test")
    print("=" * 70)
    biu_job = json.load(open('backend/storage/jobs/32ed76c1-6c56-4d08-9eba-27975db4c5a1.json'))
    papers = biu_job['result']['question_papers']
    total_q = 0
    total_inst = 0

    for i, p in enumerate(papers):
        doc = analyze_document_structure(p['raw_text'])
        total_inst += len(doc.instructions)
        meta = extract_metadata(doc.header_text or p['raw_text'][:500])
        p_qs = segment_questions(doc.questions_raw_text or p['raw_text'])
        total_q += len(p_qs)
        pr = p['page_range']
        print(f"Paper {i:2d} (pages {pr}): {len(p_qs):2d} qs | {len(doc.instructions):1d} insts | Subj: {meta.subject or 'N/A':<20} | Marks: {meta.max_marks or 'N/A'}")

    print(f"\nSUCCESS! Extracted total of {total_q} questions across 20 papers.")
    print(f"Separated total of {total_inst} instruction items across 20 papers.")

if __name__ == '__main__':
    run_test()
