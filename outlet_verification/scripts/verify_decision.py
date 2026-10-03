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
from app.services.verification import get_verification_service


def run_phase5_verification_demo():
    print("\n========================================================")
    print("  OUTLET VERIFICATION - PHASE 5 DECISION ENGINE DEMO")
    print("========================================================\n")

    db = SessionLocal()
    created_ids = []
    service = get_verification_service()

    try:
        # Reference coordinates: Indiranagar, Bangalore (12.9716, 77.5946)
        base_lat, base_lng = 12.9716, 77.5946
        img_red = Image.new("RGB", (224, 224), color=(220, 30, 30))
        img_blue = Image.new("RGB", (224, 224), color=(30, 30, 220))
        img_green = Image.new("RGB", (224, 224), color=(30, 220, 30))

        print("1. Inserting registered outlets into Railway PostgreSQL...")
        
        # Existing Outlet 1: Apollo Pharmacy in Indiranagar
        o1 = Outlet(
            name="Apollo Pharmacy - Indiranagar 100ft Road",
            latitude=base_lat + 0.0003,
            longitude=base_lng + 0.0003,
            image_url="https://example.com/apollo_indiranagar.jpg",
            name_embedding=generate_name_embedding("Apollo Pharmacy - Indiranagar 100ft Road"),
            image_embedding=generate_image_embedding(img_red),
        )

        # Existing Outlet 2: Apollo Pharmacy in Whitefield (~15 km away)
        o2 = Outlet(
            name="Apollo Pharmacy - Whitefield Main Branch",
            latitude=base_lat + 0.1200,
            longitude=base_lng + 0.1200,
            image_url="https://example.com/apollo_whitefield.jpg",
            name_embedding=generate_name_embedding("Apollo Pharmacy - Whitefield Main Branch"),
            image_embedding=generate_image_embedding(img_blue),
        )

        # Existing Outlet 3: Nearby different clinic ~60m away
        o3 = Outlet(
            name="CarePlus Diagnostic Center",
            latitude=base_lat + 0.0005,
            longitude=base_lng + 0.0004,
            image_url="https://example.com/careplus.jpg",
            name_embedding=generate_name_embedding("CarePlus Diagnostic Center"),
            image_embedding=generate_image_embedding(img_green),
        )

        db.add_all([o1, o2, o3])
        db.commit()
        db.refresh(o1)
        db.refresh(o2)
        db.refresh(o3)

        created_ids = [o1.id, o2.id, o3.id]
        print(f"   Registered {len(created_ids)} existing outlets successfully.\n")

        # ----------------------------------------------------------------------
        # SUBMISSION 1: Strong Duplicate
        # ----------------------------------------------------------------------
        print("2. Verifying Submission 1 (Duplicate Expected)...")
        print("   Submitted: 'Apollo Pharmacy - Indiranagar' with similar red storefront image at same location")
        resp1 = service.verify_outlet(
            db=db,
            name="Apollo Pharmacy - Indiranagar",
            latitude=base_lat + 0.00032,
            longitude=base_lng + 0.00031,
            image=img_red,
        )
        print(f"   Decision:              {resp1.decision.value}")
        print(f"   Duplicate Confidence:  {resp1.duplicate_confidence:.4f}")
        print(f"   Evidence Coverage:     {resp1.evidence_coverage:.4f}")
        print(f"   Evidence Agreement:    {resp1.evidence_agreement:.4f}")
        print(f"   Reason Codes:          {resp1.reason_codes}")
        print(f"   Matched Outlet:        {resp1.matched_outlet.name if resp1.matched_outlet else None}")
        print(f"   Candidate Margin:      {resp1.candidate_margin}")
        print(f"   Summary:               {resp1.reason_summary}\n")

        # ----------------------------------------------------------------------
        # SUBMISSION 2: Same Brand Different Branch (Far away)
        # ----------------------------------------------------------------------
        print("3. Verifying Submission 2 (Same brand far away / Different branch)...")
        print("   Submitted: 'Apollo Pharmacy - Electronic City' ~20 km away with distinct green storefront image")
        resp2 = service.verify_outlet(
            db=db,
            name="Apollo Pharmacy - Electronic City",
            latitude=base_lat - 0.1800,
            longitude=base_lng - 0.1800,
            image=img_green,
        )
        print(f"   Decision:              {resp2.decision.value}")
        print(f"   Duplicate Confidence:  {resp2.duplicate_confidence:.4f}")
        print(f"   Evidence Coverage:     {resp2.evidence_coverage:.4f}")
        print(f"   Evidence Agreement:    {resp2.evidence_agreement:.4f}")
        print(f"   Reason Codes:          {resp2.reason_codes}")
        print(f"   Summary:               {resp2.reason_summary}\n")

        # ----------------------------------------------------------------------
        # SUBMISSION 3: Brand New Genuine Outlet (No candidates in area)
        # ----------------------------------------------------------------------
        print("4. Verifying Submission 3 (New Genuine Outlet)...")
        print("   Submitted: 'Himalayan Organic Wellness' at remote coordinates with novel storefront image")
        img_purple = Image.new("RGB", (224, 224), color=(180, 50, 220))
        resp3 = service.verify_outlet(
            db=db,
            name="Himalayan Organic Wellness Store",
            latitude=0.0,
            longitude=0.0,
            image=img_purple,
        )
        print(f"   Decision:              {resp3.decision.value}")
        print(f"   Duplicate Confidence:  {resp3.duplicate_confidence:.4f}")
        print(f"   Evidence Status:       {resp3.evidence_status}")
        print(f"   Reason Codes:          {resp3.reason_codes}")
        print(f"   Summary:               {resp3.reason_summary}\n")

    finally:
        if created_ids:
            print("5. Cleaning up temporary test outlets from database...")
            db.query(Outlet).filter(Outlet.id.in_(created_ids)).delete(synchronize_session=False)
            db.commit()
            print(f"   Deleted {len(created_ids)} test outlets. Database clean.")
        db.close()

    print("\n--------------------------------------------------------")
    print("      DEMO EXECUTION COMPLETED SUCCESSFULLY!")
    print("--------------------------------------------------------\n")


if __name__ == "__main__":
    run_phase5_verification_demo()
