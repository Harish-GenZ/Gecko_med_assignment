import json
import logging
import sys
import uuid
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.db.session import SessionLocal
from app.models.outlet import Outlet

logger = logging.getLogger("outlet_verification.evaluation.cleanup")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DATASET_ROOT = Path(__file__).resolve().parent.parent
BASE_OUTLETS_DIR = DATASET_ROOT / "base_outlets"


def cleanup_synthetic_dataset():
    print("========================================================")
    print("  PHASE 6A: CLEANING UP SYNTHETIC EVALUATION DATASET")
    print("========================================================\n")

    meta_file = BASE_OUTLETS_DIR / "metadata.json"
    synth_uuids = []
    if meta_file.exists():
        with open(meta_file, "r", encoding="utf-8") as f:
            base_outlets = json.load(f)
            synth_uuids = [uuid.UUID(o["db_uuid"]) for o in base_outlets if "db_uuid" in o]

    db = SessionLocal()
    try:
        # Match either by the exact deterministic UUIDs from metadata OR by synthetic URL prefix
        query = db.query(Outlet).filter(
            (Outlet.id.in_(synth_uuids) if synth_uuids else False)
            | Outlet.image_url.like("synthetic://evaluation_dataset%")
        )

        matched_count = query.count()
        print(f"Found {matched_count} synthetic dataset outlet(s) in PostgreSQL database.")

        if matched_count == 0:
            print("No synthetic dataset records to delete. Database is already clean.")
            return

        deleted_count = query.delete(synchronize_session=False)
        db.commit()

        print(f"\n[PASS] Successfully deleted {deleted_count} synthetic records.")
        print("Database returned to clean state. Unrelated records were untouched.\n")

    except Exception as exc:
        db.rollback()
        print(f"\n[ERROR] Cleanup failed: {exc}")
        raise exc
    finally:
        db.close()


if __name__ == "__main__":
    cleanup_synthetic_dataset()
