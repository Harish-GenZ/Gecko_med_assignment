import sys
import time
import json
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.pipeline import run_pipeline
from app.core import job_store

def main():
    image_path = Path(__file__).parent / "storage" / "uploads" / "38112463-13f7-45d6-9a24-3e1963d225ec.jpeg"
    if not image_path.exists():
        print(f"Error: {image_path} does not exist!")
        return

    job_id = job_store.create_job(image_path.name)

    print(f"Starting pipeline on {image_path.name}...")
    t0 = time.time()
    result = run_pipeline(image_path, job_id)
    t1 = time.time()

    print(f"\nPipeline finished in {t1 - t0:.2f} seconds.")
    if result:
        res_dict = result.model_dump()
        print("\n--- RESULT JSON SUMMARY ---")
        qp0 = res_dict["question_papers"][0]
        print(f"Metadata: {json.dumps(qp0['metadata'], indent=2)}")
        print(f"Instructions ({len(qp0.get('instructions') or [])} items):")
        for inst in (qp0.get("instructions") or []):
            print(f"  - {inst}")
        print(f"\nQuestions ({len(qp0['questions'])} questions):")
        for q in qp0["questions"]:
            print(f"  [{q['number']}] ({q['type']}, marks={q['marks']}) {q['text'][:80]}...")
            if q.get("options"):
                print(f"      Options: {len(q['options'])} options")
        print(f"\nVision Verified: {qp0.get('vision_verified')}")
        print(f"Vision Status: {qp0.get('vision_status')}")
        print(f"Overall Confidence: {qp0['confidence']['overall_confidence']}")

        # Save full output for inspection
        out_file = Path(__file__).parent / "test_result_output.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(res_dict, f, indent=2)
        print(f"\nFull result written to {out_file}")

if __name__ == "__main__":
    main()
