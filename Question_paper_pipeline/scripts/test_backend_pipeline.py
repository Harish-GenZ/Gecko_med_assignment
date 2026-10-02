import sys
sys.path.insert(0, 'backend')
import json
from app.services.segmentation.question_segmenter import segment_questions
from app.services.classification.question_classifier import classify_question
from app.models.schemas import Question, QuestionPaper, ExamMetadata, QuestionType, MCQOption, ReviewStatus

data = json.load(open('backend/storage/jobs/32ed76c1-6c56-4d08-9eba-27975db4c5a1.json'))
papers = data['result']['question_papers']

print("Testing segmentation and classification on all 20 papers...\n")
total_questions = 0
for i, p in enumerate(papers):
    raw_qs = segment_questions(p['raw_text'])
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
            options=opts,
            marks=rq.get('marks'),
            confidence=round(conf, 3),
            status=ReviewStatus.OK
        )
        validated_qs.append(q)
    total_questions += len(validated_qs)
    pr = p['page_range']
    print(f"\n=== PAPER {i:2d} (pages {pr}): {len(validated_qs):2d} questions ===")
    for q in validated_qs:
        opt_str = f" [options: {len(q.options)}]" if q.options else " [options: None]"
        print(f"   Q{q.number:<3} | {q.type.value:<18} | Marks: {str(q.marks):<5} | Conf: {q.confidence:.2f}{opt_str}")
        first_line = q.text.split('\n')[0]
        print(f"         Preview: {repr(first_line[:80])}")

print(f"\nSUCCESS! Extracted total of {total_questions} questions across 20 papers.")
