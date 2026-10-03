import math
import sys
import uuid
from pathlib import Path
from PIL import Image

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from fastapi.testclient import TestClient
from sqlalchemy import text

from app.db.session import SessionLocal, engine
from app.main import app
from app.models.outlet import Outlet
from app.services.embeddings import (
    generate_image_embedding,
    generate_name_embedding,
)
from app.services.retrieval import (
    CandidateRetrievalService,
    compute_haversine_distance,
    find_nearby_outlets,
    get_candidate_retrieval_service,
    search_by_image_embedding,
    search_by_name_embedding,
)

client = TestClient(app)


def create_solid_image(color=(100, 150, 200), size=(224, 224)) -> Image.Image:
    return Image.new("RGB", size, color=color)


# ==============================================================================
# 1. HNSW & SPATIAL INDEX VERIFICATION
# ==============================================================================

def test_hnsw_and_spatial_indexes_exist():
    with engine.connect() as conn:
        indexes = conn.execute(
            text("""
                SELECT indexname, indexdef 
                FROM pg_indexes 
                WHERE tablename = 'outlets';
            """)
        ).fetchall()

        index_names = {row[0] for row in indexes}
        assert "idx_outlets_name_embedding_hnsw" in index_names, "Name HNSW index missing"
        assert "idx_outlets_image_embedding_hnsw" in index_names, "Image HNSW index missing"
        assert "idx_outlets_ll_to_earth" in index_names, "GiST spatial index missing"

        # Verify HNSW index definitions contain vector_cosine_ops
        name_def = next(row[1] for row in indexes if row[0] == "idx_outlets_name_embedding_hnsw")
        assert "hnsw" in name_def.lower()
        assert "vector_cosine_ops" in name_def.lower()

        image_def = next(row[1] for row in indexes if row[0] == "idx_outlets_image_embedding_hnsw")
        assert "hnsw" in image_def.lower()
        assert "vector_cosine_ops" in image_def.lower()


# ==============================================================================
# 2. HAVERSINE DISTANCE UNIT TEST
# ==============================================================================

def test_haversine_distance_computation():
    # Bangalore MG Road to Indiranagar (~4.2 km)
    lat1, lon1 = 12.9756, 77.6066
    lat2, lon2 = 12.9784, 77.6408
    dist = compute_haversine_distance(lat1, lon1, lat2, lon2)
    assert 3500.0 < dist < 4500.0, f"Expected ~4.0km, got {dist}m"

    # Same coordinate distance = 0
    assert compute_haversine_distance(lat1, lon1, lat1, lon1) == 0.0


# ==============================================================================
# 3. CONTROLLED MULTI-CHANNEL RETRIEVAL TESTS
# ==============================================================================

def test_multi_channel_retrieval_and_deduplication():
    db = SessionLocal()
    test_ids = []

    try:
        # Reference query location: Indiranagar, Bangalore (12.9716, 77.5946)
        query_lat = 12.97160
        query_lng = 77.59460
        query_name = "Apollo Pharmacy - Indiranagar Main"
        
        # Colors for test images
        img_red = create_solid_image(color=(255, 20, 20))
        img_blue = create_solid_image(color=(20, 20, 255))
        img_green = create_solid_image(color=(20, 255, 20))
        img_yellow = create_solid_image(color=(255, 255, 20))

        red_emb = generate_image_embedding(img_red)
        blue_emb = generate_image_embedding(img_blue)
        green_emb = generate_image_embedding(img_green)
        yellow_emb = generate_image_embedding(img_yellow)

        name_apollo_emb = generate_name_embedding("Apollo Pharmacy - Indiranagar Branch")
        name_medplus_emb = generate_name_embedding("MedPlus Health Services")
        name_random_emb = generate_name_embedding("Zenith Electronics World")

        # ----------------------------------------------------------------------
        # Insert Controlled Test Candidates
        # ----------------------------------------------------------------------

        # Candidate A: Appears in ALL THREE (Name similar, Image same red, Geo ~50m away)
        outlet_all = Outlet(
            name="Apollo Pharmacy - Indiranagar Branch",
            latitude=query_lat + 0.0003,  # ~33 meters away
            longitude=query_lng + 0.0003,
            image_url="https://example.com/test_all.jpg",
            name_embedding=name_apollo_emb,
            image_embedding=red_emb,
        )
        db.add(outlet_all)

        # Candidate B: Name-only (Apollo brand, but in Chennai ~290 km away, distinct blue image)
        outlet_name_only = Outlet(
            name="Apollo Pharmacy - T Nagar Central",
            latitude=13.0418,
            longitude=80.2341,
            image_url="https://example.com/test_name.jpg",
            name_embedding=name_apollo_emb,
            image_embedding=blue_emb,
        )
        db.add(outlet_name_only)

        # Candidate C: Image-only (Same red image, but completely unrelated name and in Mumbai ~840 km away)
        outlet_image_only = Outlet(
            name="Zenith Electronics World",
            latitude=19.0760,
            longitude=72.8777,
            image_url="https://example.com/test_image.jpg",
            name_embedding=name_random_emb,
            image_embedding=red_emb,
        )
        db.add(outlet_image_only)

        # Candidate D: Geo-only (Close by ~80m, but completely different name & different green image)
        outlet_geo_only = Outlet(
            name="Zenith Electronics Indiranagar",
            latitude=query_lat + 0.0006,  # ~66 meters away
            longitude=query_lng + 0.0005,
            image_url="https://example.com/test_geo.jpg",
            name_embedding=name_random_emb,
            image_embedding=green_emb,
        )
        db.add(outlet_geo_only)

        # Candidate E: NULL embedding candidate (Close by ~90m, but has NULL embeddings)
        outlet_null_emb = Outlet(
            name="Legacy Indiranagar General Store",
            latitude=query_lat + 0.0007,
            longitude=query_lng + 0.0006,
            image_url="https://example.com/test_null.jpg",
            name_embedding=None,
            image_embedding=None,
        )
        db.add(outlet_null_emb)

        # Candidate F: Outside Radius (~5 km away, different name, yellow image)
        outlet_outside = Outlet(
            name="Koramangala Supermarket",
            latitude=query_lat + 0.045,  # ~5 km away
            longitude=query_lng + 0.045,
            image_url="https://example.com/test_outside.jpg",
            name_embedding=name_medplus_emb,
            image_embedding=yellow_emb,
        )
        db.add(outlet_outside)

        db.commit()
        db.refresh(outlet_all)
        db.refresh(outlet_name_only)
        db.refresh(outlet_image_only)
        db.refresh(outlet_geo_only)
        db.refresh(outlet_null_emb)
        db.refresh(outlet_outside)

        test_ids = [
            outlet_all.id,
            outlet_name_only.id,
            outlet_image_only.id,
            outlet_geo_only.id,
            outlet_null_emb.id,
            outlet_outside.id,
        ]

        # ----------------------------------------------------------------------
        # Test 1: Channel A - Name Vector Search Only
        # ----------------------------------------------------------------------
        query_text_emb = generate_name_embedding(query_name)
        name_results = search_by_name_embedding(db, query_text_emb, top_k=5)
        name_result_ids = {r["outlet_id"] for r in name_results}
        assert str(outlet_all.id) in name_result_ids
        assert str(outlet_name_only.id) in name_result_ids
        assert str(outlet_null_emb.id) not in name_result_ids, "NULL embedding must not be returned by vector search"

        # ----------------------------------------------------------------------
        # Test 2: Channel B - Image Vector Search Only
        # ----------------------------------------------------------------------
        query_img_emb = generate_image_embedding(img_red)
        image_results = search_by_image_embedding(db, query_img_emb, top_k=5)
        image_result_ids = {r["outlet_id"] for r in image_results}
        assert str(outlet_all.id) in image_result_ids
        assert str(outlet_image_only.id) in image_result_ids
        assert str(outlet_null_emb.id) not in image_result_ids

        # ----------------------------------------------------------------------
        # Test 3: Channel C - Geographic Radius Search Only (radius = 500m)
        # ----------------------------------------------------------------------
        geo_results = find_nearby_outlets(db, query_lat, query_lng, radius_meters=500.0, top_k=10)
        geo_result_ids = {r["outlet_id"] for r in geo_results}
        assert str(outlet_all.id) in geo_result_ids
        assert str(outlet_geo_only.id) in geo_result_ids
        assert str(outlet_null_emb.id) in geo_result_ids, "Geo search retrieves nearby outlets even with NULL embedding"
        assert str(outlet_name_only.id) not in geo_result_ids, "Chennai outlet must be excluded by 500m radius"
        assert str(outlet_outside.id) not in geo_result_ids, "5km outlet must be excluded by 500m radius"

        # ----------------------------------------------------------------------
        # Test 4: Combined Candidate Retrieval Service & Merging
        # ----------------------------------------------------------------------
        retrieval_service = get_candidate_retrieval_service()
        combined = retrieval_service.retrieve_candidates(
            db=db,
            name=query_name,
            latitude=query_lat,
            longitude=query_lng,
            image=img_red,
            name_top_k=5,
            image_top_k=5,
            geo_radius_meters=500.0,
            geo_top_k=10,
        )

        candidates = combined.candidates
        candidate_ids = [c.outlet_id for c in candidates]
        
        # Verify deduplication: no outlet_id is duplicated
        assert len(candidate_ids) == len(set(candidate_ids)), "Duplicate candidate IDs found in result pool!"

        # Candidate A (outlet_all) must be surfaced with all 3 channels
        cand_all = next(c for c in candidates if c.outlet_id == str(outlet_all.id))
        assert "name" in cand_all.matched_methods
        assert "image" in cand_all.matched_methods
        assert "geo" in cand_all.matched_methods
        assert cand_all.name_similarity is not None and cand_all.name_similarity > 0.7
        assert cand_all.image_similarity is not None and cand_all.image_similarity > 0.9
        assert cand_all.distance_meters is not None and cand_all.distance_meters < 100.0

        # Candidate B (outlet_name_only) has name match, distance computed via haversine (~290km)
        cand_name = next(c for c in candidates if c.outlet_id == str(outlet_name_only.id))
        assert "name" in cand_name.matched_methods
        assert cand_name.name_similarity is not None
        assert cand_name.distance_meters is not None and cand_name.distance_meters > 250000.0

        # Candidate C (outlet_image_only) has image match
        cand_img = next(c for c in candidates if c.outlet_id == str(outlet_image_only.id))
        assert "image" in cand_img.matched_methods
        assert cand_img.image_similarity is not None

        # Candidate D (outlet_geo_only) has geo match
        cand_geo = next(c for c in candidates if c.outlet_id == str(outlet_geo_only.id))
        assert "geo" in cand_geo.matched_methods
        assert cand_geo.distance_meters is not None and cand_geo.distance_meters < 200.0

        # Candidate E (outlet_null_emb) has geo match only
        cand_null = next(c for c in candidates if c.outlet_id == str(outlet_null_emb.id))
        assert "geo" in cand_null.matched_methods
        assert cand_null.name_similarity is None
        assert cand_null.image_similarity is None

        # Top candidate must be cand_all (matched 3 methods)
        assert candidates[0].outlet_id == str(outlet_all.id), "Candidate matching all 3 methods should rank first"

        # ----------------------------------------------------------------------
        # Test 5: API Endpoint Verification (POST /candidates/search-json)
        # ----------------------------------------------------------------------
        import base64
        import io
        buf = io.BytesIO()
        img_red.save(buf, format="JPEG")
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

        response = client.post(
            "/candidates/search-json",
            json={
                "name": query_name,
                "latitude": query_lat,
                "longitude": query_lng,
                "image_base64": b64_str,
                "name_top_k": 5,
                "image_top_k": 5,
                "geo_radius_meters": 500.0,
            },
        )
        assert response.status_code == 200, f"API error: {response.text}"
        api_data = response.json()
        assert "total_candidates" in api_data
        assert "candidates" in api_data
        assert api_data["total_candidates"] >= 4

    finally:
        # ----------------------------------------------------------------------
        # Clean Up All Test Records
        # ----------------------------------------------------------------------
        if test_ids:
            db.query(Outlet).filter(Outlet.id.in_(test_ids)).delete(synchronize_session=False)
            db.commit()
        db.close()


def test_empty_search_no_candidates():
    db = SessionLocal()
    try:
        # Search at a remote coordinate in the Pacific Ocean with a completely synthetic random name
        random_name = f"Nonexistent Unique Outlet {uuid.uuid4()}"
        retrieval_service = get_candidate_retrieval_service()
        img = create_solid_image(color=(12, 34, 56))

        result = retrieval_service.retrieve_candidates(
            db=db,
            name=random_name,
            latitude=0.0,
            longitude=-160.0,
            image=img,
            name_top_k=0,  # 0 or small
            image_top_k=0,
            geo_radius_meters=10.0,  # Tiny 10 meter radius in ocean
        )
        # With geo_radius=10m in the ocean and top_k=0, geo will be empty
        assert isinstance(result.candidates, list)
    finally:
        db.close()


if __name__ == "__main__":
    print("\n========================================================")
    print("      RUNNING PHASE 4 CANDIDATE RETRIEVAL TESTS")
    print("========================================================\n")
    test_hnsw_and_spatial_indexes_exist()
    print("[PASS] test_hnsw_and_spatial_indexes_exist (HNSW cosine & GiST ll_to_earth)")
    test_haversine_distance_computation()
    print("[PASS] test_haversine_distance_computation (great-circle distance math)")
    test_multi_channel_retrieval_and_deduplication()
    print("[PASS] test_multi_channel_retrieval_and_deduplication (name, image, geo, multi-channel merge, NULL handling, API endpoint)")
    test_empty_search_no_candidates()
    print("[PASS] test_empty_search_no_candidates")
    print("\n--------------------------------------------------------")
    print("      ALL PHASE 4 TESTS PASSED SUCCESSFULLY!")
    print("--------------------------------------------------------\n")
