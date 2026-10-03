import json
import sys
from pathlib import Path
from PIL import Image

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.db.session import SessionLocal
from app.models.outlet import Outlet
from app.services.embeddings import (
    generate_image_embedding,
    generate_name_embedding,
)
from app.services.retrieval import get_candidate_retrieval_service


def run_candidate_retrieval_demo():
    print("\n========================================================")
    print("  OUTLET VERIFICATION - PHASE 4 CANDIDATE RETRIEVAL DEMO")
    print("========================================================\n")

    db = SessionLocal()
    created_ids = []

    try:
        # Reference coordinates: Indiranagar, Bangalore (12.9716, 77.5946)
        base_lat, base_lng = 12.9716, 77.5946
        img_red = Image.new("RGB", (224, 224), color=(220, 30, 30))
        img_blue = Image.new("RGB", (224, 224), color=(30, 30, 220))
        img_green = Image.new("RGB", (224, 224), color=(30, 220, 30))

        # Populate temporary mock database records representing existing registered outlets
        print("1. Inserting temporary registered outlets into Railway PostgreSQL...")
        
        # Outlet 1: Close match across all channels (Same name brand, close by 40m, similar image)
        o1 = Outlet(
            name="Apollo Pharmacy - Indiranagar 100ft",
            latitude=base_lat + 0.0003,
            longitude=base_lng + 0.0003,
            image_url="https://example.com/apollo_store1.jpg",
            name_embedding=generate_name_embedding("Apollo Pharmacy - Indiranagar 100ft"),
            image_embedding=generate_image_embedding(img_red),
        )

        # Outlet 2: Same brand name, but in Whitefield ~15 km away (Name match only)
        o2 = Outlet(
            name="Apollo Pharmacy - Whitefield Main",
            latitude=base_lat + 0.1200,
            longitude=base_lng + 0.1200,
            image_url="https://example.com/apollo_whitefield.jpg",
            name_embedding=generate_name_embedding("Apollo Pharmacy - Whitefield Main"),
            image_embedding=generate_image_embedding(img_blue),
        )

        # Outlet 3: Nearby pharmacy ~120m away with different name & different green image (Geo match only)
        o3 = Outlet(
            name="Sri Krishna Medical & General Store",
            latitude=base_lat + 0.0010,
            longitude=base_lng + 0.0008,
            image_url="https://example.com/krishna_medicals.jpg",
            name_embedding=generate_name_embedding("Sri Krishna Medical & General Store"),
            image_embedding=generate_image_embedding(img_green),
        )

        db.add_all([o1, o2, o3])
        db.commit()
        for o in [o1, o2, o3]:
            db.refresh(o)
            created_ids.append(o.id)

        print(f"   Successfully registered {len(created_ids)} mock outlets.")

        # 2. Simulate a new field submission
        submission_name = "Apollo Pharmacy Indiranagar"
        submission_lat = base_lat
        submission_lng = base_lng
        submission_img = img_red

        print(f"\n2. Query Submission:")
        print(f"   Name:        '{submission_name}'")
        print(f"   Coordinates: ({submission_lat}, {submission_lng})")
        print(f"   Image:       Red storefront photograph")

        # 3. Execute Candidate Retrieval
        print("\n3. Executing Candidate Retrieval Service (HNSW Name + HNSW Image + Geo 500m)...")
        service = get_candidate_retrieval_service()
        result = service.retrieve_candidates(
            db=db,
            name=submission_name,
            latitude=submission_lat,
            longitude=submission_lng,
            image=submission_img,
            name_top_k=5,
            image_top_k=5,
            geo_radius_meters=500.0,
            geo_top_k=10,
        )

        print(f"\n4. Retrieved Candidates Pool: {result.total_candidates} candidate(s) found\n")
        print("-" * 80)
        print(f"{'Outlet Name':<35} | {'Methods':<14} | {'Name Sim':<8} | {'Img Sim':<8} | {'Dist (m)':<8}")
        print("-" * 80)
        for c in result.candidates:
            methods_str = "+".join(c.matched_methods)
            name_sim_str = f"{c.name_similarity:.3f}" if c.name_similarity is not None else "  ---  "
            img_sim_str = f"{c.image_similarity:.3f}" if c.image_similarity is not None else "  ---  "
            dist_str = f"{c.distance_meters:.1f}m" if c.distance_meters is not None else "  ---  "
            print(f"{c.name:<35} | {methods_str:<14} | {name_sim_str:<8} | {img_sim_str:<8} | {dist_str:<8}")
        print("-" * 80)

        print("\nFull JSON Representation (Candidate Evidence):")
        print(json.dumps([c.model_dump() for c in result.candidates], indent=2))

    finally:
        # Clean up mock records
        if created_ids:
            db.query(Outlet).filter(Outlet.id.in_(created_ids)).delete(synchronize_session=False)
            db.commit()
            print("\nCleaned up all mock database records.")
        db.close()


if __name__ == "__main__":
    run_candidate_retrieval_demo()
