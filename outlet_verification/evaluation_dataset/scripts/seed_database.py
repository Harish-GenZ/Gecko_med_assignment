import json
import logging
import sys
import uuid
from pathlib import Path
from PIL import Image

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.db.session import SessionLocal
from app.models.outlet import Outlet
from app.services.embeddings import (
    generate_image_embedding,
    generate_name_embedding,
)

logger = logging.getLogger("outlet_verification.evaluation.seed")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DATASET_ROOT = Path(__file__).resolve().parent.parent
BASE_OUTLETS_DIR = DATASET_ROOT / "base_outlets"
BASE_IMAGES_DIR = BASE_OUTLETS_DIR / "images"


def seed_database():
    print("========================================================")
    print("  PHASE 6A: SEEDING SYNTHETIC BASE OUTLETS INTO DATABASE")
    print("========================================================\n")

    meta_file = BASE_OUTLETS_DIR / "metadata.json"
    if not meta_file.exists():
        print(f"Error: Metadata file not found at {meta_file}")
        sys.exit(1)

    with open(meta_file, "r", encoding="utf-8") as f:
        base_outlets = json.load(f)

    db = SessionLocal()
    inserted_count = 0
    updated_count = 0

    try:
        total = len(base_outlets)
        print(f"Seeding {total} base outlets into Railway PostgreSQL table 'outlets'...")

        for idx, item in enumerate(base_outlets, 1):
            outlet_uuid = uuid.UUID(item["db_uuid"])
            img_path = BASE_IMAGES_DIR / item["image_file"]

            # Load image from disk for CLIP embedding
            with Image.open(img_path) as pil_img:
                img_rgb = pil_img.convert("RGB")
                img_embedding = generate_image_embedding(img_rgb)

            # Compute text embedding for outlet name
            name_embedding = generate_name_embedding(item["name"])

            # Check if record already exists (idempotent seeding)
            existing = db.query(Outlet).filter(Outlet.id == outlet_uuid).first()
            synth_image_url = f"synthetic://evaluation_dataset/base_outlets/images/{item['image_file']}"

            if existing:
                existing.name = item["name"]
                existing.latitude = item["latitude"]
                existing.longitude = item["longitude"]
                existing.image_url = synth_image_url
                existing.name_embedding = name_embedding
                existing.image_embedding = img_embedding
                updated_count += 1
            else:
                new_outlet = Outlet(
                    id=outlet_uuid,
                    name=item["name"],
                    latitude=item["latitude"],
                    longitude=item["longitude"],
                    image_url=synth_image_url,
                    name_embedding=name_embedding,
                    image_embedding=img_embedding,
                )
                db.add(new_outlet)
                inserted_count += 1

            if idx % 10 == 0 or idx == total:
                db.commit()
                print(f"   Processed {idx}/{total} outlets (Inserted: {inserted_count}, Updated: {updated_count})...")

        db.commit()
        print("\n--------------------------------------------------------")
        print(f"  SEEDING COMPLETE: {inserted_count} inserted, {updated_count} updated.")
        print(f"  All {total} synthetic base outlets have 384d & 512d embeddings.")
        print("--------------------------------------------------------\n")

    except Exception as exc:
        db.rollback()
        print(f"\n[ERROR] Seeding failed: {exc}")
        raise exc
    finally:
        db.close()


if __name__ == "__main__":
    seed_database()
