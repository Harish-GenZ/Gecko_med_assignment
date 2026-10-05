import io
import math
import sys
from pathlib import Path
from PIL import Image

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.services.retrieval.models import CandidateEvidence
from app.services.verification import (
    CandidateScorer,
    DecisionEngine,
    ReasonCode,
    VerificationDecision,
    compute_geo_proximity,
    get_candidate_scorer,
    get_decision_engine,
)

client = TestClient(app)


def make_candidate(
    outlet_id: str = "cand-001",
    name: str = "Test Outlet",
    lat: float = 12.9716,
    lng: float = 77.5946,
    name_sim: float | None = None,
    image_sim: float | None = None,
    distance_m: float | None = None,
    matched_methods: list[str] | None = None,
) -> CandidateEvidence:
    return CandidateEvidence(
        outlet_id=outlet_id,
        name=name,
        latitude=lat,
        longitude=lng,
        image_url="https://example.com/test.jpg",
        name_similarity=name_sim,
        image_similarity=image_sim,
        distance_meters=distance_m,
        matched_methods=matched_methods or ["name"],
    )


# ==============================================================================
# 1. FORMULA UNIT TESTS (Section 21)
# ==============================================================================

def test_formula_geo_proximity():
    """Verify exponential decay function G = exp(-d / D)."""
    # d = 0 -> 1.0
    assert compute_geo_proximity(0.0, 100.0) == 1.0
    # d = 100 -> exp(-1) ≈ 0.3679
    assert math.isclose(compute_geo_proximity(100.0, 100.0), math.exp(-1), abs_tol=1e-3)
    # d = 500 -> exp(-5) ≈ 0.0067
    assert math.isclose(compute_geo_proximity(500.0, 100.0), math.exp(-5), abs_tol=1e-3)
    # distance is None -> None
    assert compute_geo_proximity(None, 100.0) is None
    # distance is negative -> clamped to 0.0
    assert compute_geo_proximity(-10.0, 100.0) == 1.0
    print("[PASS] test_formula_geo_proximity passed.")


def test_formula_weight_renormalization():
    """Verify missing signals are NOT treated as zero; weights renormalize."""
    scorer = CandidateScorer(name_weight=0.30, image_weight=0.45, geo_weight=0.25, geo_scale_meters=100.0)
    
    # N = 0.90, G = 0.80 (distance chosen so G ≈ 0.8), I = NULL
    # To test pure renormalization:
    # Let candidate have name_sim=0.90, distance=22.314 (exp(-0.22314) ≈ 0.80), image_sim=None
    d_for_g80 = -100.0 * math.log(0.80)
    cand = make_candidate(name_sim=0.90, image_sim=None, distance_m=d_for_g80)
    scored = scorer.score_candidate(cand)

    expected_available_weight = 0.30 + 0.25  # 0.55
    expected_base_score = (0.30 * 0.90 + 0.25 * 0.80) / expected_available_weight  # 0.47 / 0.55 ≈ 0.8545
    
    assert math.isclose(scored.evidence_coverage, 0.55, abs_tol=1e-3)
    assert math.isclose(scored.base_score, expected_base_score, abs_tol=1e-3)
    # Base score must NOT equal penalized version with image=0: (0.3*0.9 + 0.45*0 + 0.25*0.8) = 0.47
    assert scored.base_score > 0.80
    print("[PASS] test_formula_weight_renormalization passed.")


def test_formula_agreement_and_confidence():
    """Verify agreement is 1.0 for single signal and properly reflects disagreement."""
    scorer = get_candidate_scorer()

    # Single available signal -> agreement is 1.0
    single_cand = make_candidate(name_sim=0.85, image_sim=None, distance_m=None)
    scored_single = scorer.score_candidate(single_cand)
    assert scored_single.evidence_agreement == 1.0

    # Conflicting signals -> agreement decreases
    # N = 0.95, I = 0.10, d = 0m (G = 1.0)
    conflict_cand = make_candidate(name_sim=0.95, image_sim=0.10, distance_m=0.0)
    scored_conflict = scorer.score_candidate(conflict_cand)
    assert scored_conflict.evidence_agreement < 0.70
    print("[PASS] test_formula_agreement_and_confidence passed.")


def test_formula_confidence_bounds():
    """Sweep a broad range of similarity and distance values to verify 0 <= duplicate_confidence <= 1."""
    scorer = get_candidate_scorer()
    test_sims = [None, 0.0, 0.25, 0.50, 0.75, 0.95, 1.0]
    test_dists = [None, 0.0, 50.0, 100.0, 500.0, 5000.0]

    for n in test_sims:
        for i in test_sims:
            for d in test_dists:
                cand = make_candidate(name_sim=n, image_sim=i, distance_m=d)
                scored = scorer.score_candidate(cand)
                assert 0.0 <= scored.duplicate_confidence <= 1.0, (
                    f"Out of bounds confidence {scored.duplicate_confidence} for n={n}, i={i}, d={d}"
                )
                assert 0.0 <= scored.evidence_coverage <= 1.0
                assert 0.0 <= scored.evidence_agreement <= 1.0
    print("[PASS] test_formula_confidence_bounds passed across all combinations.")


# ==============================================================================
# 2. REQUIRED CORE SCENARIO TESTS (Section 20)
# ==============================================================================

def test_scenario_1_strong_duplicate():
    """Test 1: High name, high image, very close distance -> DUPLICATE."""
    cand = make_candidate(name_sim=0.92, image_sim=0.97, distance_m=30.0)
    scorer = get_candidate_scorer()
    decision_engine = get_decision_engine()

    scored = scorer.score_candidate(cand)
    result = decision_engine.evaluate([scored])

    assert result.decision == VerificationDecision.DUPLICATE
    assert result.duplicate_confidence >= 0.80
    assert (
        ReasonCode.STRONG_MULTIMODAL_MATCH.value in result.reason_codes
        or ReasonCode.STRONG_IMAGE_GEO_MATCH.value in result.reason_codes
    )
    print("[PASS] test_scenario_1_strong_duplicate passed:", result.decision, result.duplicate_confidence)


def test_scenario_2_strong_visual_geo_different_name():
    """Test 2: Moderate/different name, very high image, close distance -> strong duplicate evidence."""
    cand = make_candidate(name_sim=0.40, image_sim=0.97, distance_m=25.0)
    scorer = get_candidate_scorer()
    decision_engine = get_decision_engine()

    scored = scorer.score_candidate(cand)
    assert scored.flags.strong_duplicate_evidence is True

    result = decision_engine.evaluate([scored])
    assert ReasonCode.STRONG_IMAGE_GEO_MATCH.value in result.reason_codes
    print("[PASS] test_scenario_2_strong_visual_geo_different_name passed. Flags:", scored.flags)


def test_scenario_3_same_name_different_outlet():
    """Test 3: Same name but different outlet (far away & different image) -> NOT DUPLICATE."""
    cand = make_candidate(name_sim=0.95, image_sim=0.20, distance_m=2500.0)
    scorer = get_candidate_scorer()
    decision_engine = get_decision_engine()

    scored = scorer.score_candidate(cand)
    assert scored.flags.possible_same_brand_different_location is True
    assert scored.flags.conflicting_evidence is True

    result = decision_engine.evaluate([scored])
    assert result.decision != VerificationDecision.DUPLICATE
    assert ReasonCode.POSSIBLE_SAME_BRAND_DIFFERENT_LOCATION.value in result.reason_codes
    print("[PASS] test_scenario_3_same_name_different_outlet passed:", result.decision, result.reason_codes)


def test_scenario_4_conflicting_signals():
    """Test 4: High name, low image, close distance -> NEEDS_REVIEW due to conflicting signals."""
    cand = make_candidate(name_sim=0.95, image_sim=0.20, distance_m=30.0)
    scorer = get_candidate_scorer()
    decision_engine = get_decision_engine()

    scored = scorer.score_candidate(cand)
    assert scored.flags.conflicting_evidence is True

    result = decision_engine.evaluate([scored])
    assert result.decision == VerificationDecision.NEEDS_REVIEW
    assert ReasonCode.CONFLICTING_NAME_IMAGE.value in result.reason_codes
    print("[PASS] test_scenario_4_conflicting_signals passed:", result.decision, result.reason_codes)


def test_scenario_5_missing_image():
    """Test 5: Missing image -> weight renormalized, not treated as 0."""
    cand = make_candidate(name_sim=0.88, image_sim=None, distance_m=50.0)
    scorer = get_candidate_scorer()
    scored = scorer.score_candidate(cand)

    assert math.isclose(scored.evidence_coverage, scorer.name_weight + scorer.geo_weight, abs_tol=1e-3)
    # Base score should be ~0.7557, NOT ~0.415
    assert scored.base_score > 0.70
    print("[PASS] test_scenario_5_missing_image passed:", scored.base_score, scored.evidence_coverage)


def test_scenario_6_missing_gps():
    """Test 6: Missing GPS -> weight renormalized, not treated as 0."""
    cand = make_candidate(name_sim=0.88, image_sim=0.93, distance_m=None)
    scorer = get_candidate_scorer()
    decision_engine = get_decision_engine()

    scored = scorer.score_candidate(cand)
    assert math.isclose(scored.evidence_coverage, scorer.name_weight + scorer.image_weight, abs_tol=1e-3)
    assert scored.base_score > 0.85

    result = decision_engine.evaluate([scored])
    assert result.decision == VerificationDecision.DUPLICATE
    print("[PASS] test_scenario_6_missing_gps passed:", result.decision, result.duplicate_confidence)


def test_scenario_7_weak_evidence():
    """Test 7: Only weak evidence (moderate name, missing image, far) -> NEEDS_REVIEW."""
    cand = make_candidate(name_sim=0.55, image_sim=None, distance_m=300.0)
    scorer = get_candidate_scorer()
    decision_engine = get_decision_engine()

    scored = scorer.score_candidate(cand)
    result = decision_engine.evaluate([scored])

    assert result.decision == VerificationDecision.NEEDS_REVIEW
    assert (
        ReasonCode.WEAK_EVIDENCE_REQUIRES_REVIEW.value in result.reason_codes
        or ReasonCode.INSUFFICIENT_EVIDENCE.value in result.reason_codes
    )
    print("[PASS] test_scenario_7_weak_evidence passed:", result.decision, result.reason_codes)


def test_scenario_8_no_candidates():
    """Test 8: No candidates found -> GENUINE with confidence = 0.0 and NO_MATCHING_CANDIDATE."""
    decision_engine = get_decision_engine()
    result = decision_engine.evaluate([])

    assert result.decision == VerificationDecision.GENUINE
    assert result.duplicate_confidence == 0.0
    assert result.evidence_status == "NO_MATCHING_CANDIDATE"
    assert ReasonCode.NO_MATCHING_CANDIDATE.value in result.reason_codes
    assert result.matched_outlet is None
    print("[PASS] test_scenario_8_no_candidates passed:", result.decision, result.reason_codes)


def test_scenario_9_ambiguous_candidates():
    """Test 9: Two candidates with 3-metric average >= 0.75 are DUPLICATE (strong 75% rule) with margin recorded."""
    cand_a = make_candidate(
        outlet_id="cand-A",
        name="Apollo Pharmacy Branch 1",
        name_sim=0.94,
        image_sim=0.97,
        distance_m=15.0,
    )
    cand_b = make_candidate(
        outlet_id="cand-B",
        name="Apollo Pharmacy Branch 2",
        name_sim=0.92,
        image_sim=0.95,
        distance_m=22.0,
    )

    scorer = get_candidate_scorer()
    decision_engine = get_decision_engine()

    scored_a = scorer.score_candidate(cand_a)
    scored_b = scorer.score_candidate(cand_b)

    # Both are high confidence (>= 0.80), and margin is < 0.10
    assert scored_a.duplicate_confidence >= 0.80
    assert scored_b.duplicate_confidence >= 0.80
    margin_diff = scored_a.duplicate_confidence - scored_b.duplicate_confidence
    assert 0.0 < margin_diff < 0.10

    result = decision_engine.evaluate([scored_a, scored_b])

    # Above 75% is strictly DUPLICATE
    assert result.decision == VerificationDecision.DUPLICATE
    assert ReasonCode.AMBIGUOUS_TOP_CANDIDATES.value in result.reason_codes
    assert result.candidate_margin is not None
    assert result.candidate_margin < 0.10

    print("[PASS] test_scenario_9_ambiguous_candidates passed:", result.decision, result.candidate_margin)


def test_scenario_manual_review_range_65_to_75():
    """Test: 3-Metric average in the 65% to 75% range correctly yields NEEDS_REVIEW."""
    # Let average be ~0.70: name=0.70, image=0.70, distance=35.6m (geo ≈ 0.70)
    cand = make_candidate(name_sim=0.70, image_sim=0.70, distance_m=35.667)
    scorer = get_candidate_scorer()
    decision_engine = get_decision_engine()

    scored = scorer.score_candidate(cand)
    assert 0.65 <= scored.duplicate_confidence < 0.75

    result = decision_engine.evaluate([scored])
    assert result.decision == VerificationDecision.NEEDS_REVIEW
    print("[PASS] test_scenario_manual_review_range_65_to_75 passed:", result.decision, result.duplicate_confidence)


def test_scenario_10_perfect_match():
    """Test 10: Perfect match (name=1.0, image=1.0, distance=0) -> DUPLICATE with very high confidence."""
    cand = make_candidate(name_sim=1.0, image_sim=1.0, distance_m=0.0)
    scorer = get_candidate_scorer()
    decision_engine = get_decision_engine()

    scored = scorer.score_candidate(cand)
    result = decision_engine.evaluate([scored])

    assert result.decision == VerificationDecision.DUPLICATE
    assert result.duplicate_confidence >= 0.95
    assert ReasonCode.STRONG_MULTIMODAL_MATCH.value in result.reason_codes
    print("[PASS] test_scenario_10_perfect_match passed:", result.duplicate_confidence)


def test_scenario_11_completely_different_outlet():
    """Test 11: Completely different outlet (name=0.05, image=0.10, distance=2000m) -> GENUINE."""
    cand = make_candidate(name_sim=0.05, image_sim=0.10, distance_m=2000.0)
    scorer = get_candidate_scorer()
    decision_engine = get_decision_engine()

    scored = scorer.score_candidate(cand)
    result = decision_engine.evaluate([scored])

    assert result.decision == VerificationDecision.GENUINE
    assert result.duplicate_confidence <= 0.35
    assert ReasonCode.LOW_CONFIDENCE_GENUINE.value in result.reason_codes
    print("[PASS] test_scenario_11_completely_different_outlet passed:", result.decision, result.duplicate_confidence)


# ==============================================================================
# 3. ENDPOINT INTEGRATION TESTS (POST /verify & POST /verify-json)
# ==============================================================================

def test_api_verify_json_and_validation():
    """Test API endpoint response structure, JSON submission, and validation."""
    # Test coordinate validation failure
    bad_req = {
        "name": "Invalid Outlet",
        "latitude": 95.0,  # Invalid latitude > 90
        "longitude": 77.0,
        "image_url": "https://example.com/test.jpg",
    }
    resp = client.post("/verify-json", json=bad_req)
    assert resp.status_code == 422, "Expected 422 for lat > 90"

    # Test empty name failure
    empty_name_req = {
        "name": "   ",
        "latitude": 12.9716,
        "longitude": 77.5946,
        "image_url": "https://example.com/test.jpg",
    }
    resp = client.post("/verify-json", json=empty_name_req)
    assert resp.status_code == 400

    print("[PASS] test_api_verify_json_and_validation passed.")


def test_api_verify_form_upload():
    """Test POST /verify with multipart form upload containing a generated image."""
    img = Image.new("RGB", (100, 100), color=(120, 200, 80))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)

    response = client.post(
        "/verify",
        data={
            "name": "Test Form Submission Pharmacy",
            "latitude": "12.9716",
            "longitude": "77.5946",
        },
        files={"image": ("store.jpg", img_byte_arr, "image/jpeg")},
    )

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()

    assert "decision" in data
    assert data["decision"] in ["DUPLICATE", "GENUINE", "NEEDS_REVIEW", "REJECTED"]
    assert "duplicate_confidence" in data
    assert 0.0 <= data["duplicate_confidence"] <= 1.0
    assert "evidence_coverage" in data
    assert "evidence_agreement" in data
    assert "reason_codes" in data
    assert "reason_summary" in data

    print("[PASS] test_api_verify_form_upload passed:", data["decision"], data["duplicate_confidence"], data["reason_codes"])


def test_authenticity_rejection():
    """Test that a non-store input (e.g. blank square image) fails domain authenticity and returns REJECTED."""
    img = Image.new("RGB", (64, 64), color=(0, 0, 0))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)

    response = client.post(
        "/verify",
        data={
            "name": "Random Non Store Entity",
            "latitude": "12.9716",
            "longitude": "77.5946",
        },
        files={"image": ("blank.jpg", img_byte_arr, "image/jpeg")},
    )

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert data["decision"] == "REJECTED"
    assert "NOT_A_RETAIL_STOREFRONT_IMAGE" in data["reason_codes"] or "NON_OUTLET_NAME" in data["reason_codes"]
    assert data["registered"] is False
    assert data["registered_outlet_id"] is None
    assert data["authenticity"] is not None
    assert data["authenticity"]["is_authentic_store"] is False
    print("[PASS] test_authenticity_rejection passed: correctly rejected non-store input with REJECTED decision.")


if __name__ == "__main__":
    print("\n========================================================")
    print("      RUNNING PHASE 5 VERIFICATION DECISION TESTS")
    print("========================================================\n")

    test_formula_geo_proximity()
    test_formula_weight_renormalization()
    test_formula_agreement_and_confidence()
    test_formula_confidence_bounds()

    test_scenario_1_strong_duplicate()
    test_scenario_2_strong_visual_geo_different_name()
    test_scenario_3_same_name_different_outlet()
    test_scenario_4_conflicting_signals()
    test_scenario_5_missing_image()
    test_scenario_6_missing_gps()
    test_scenario_7_weak_evidence()
    test_scenario_8_no_candidates()
    test_scenario_9_ambiguous_candidates()
    test_scenario_manual_review_range_65_to_75()
    test_scenario_10_perfect_match()
    test_scenario_11_completely_different_outlet()

    test_api_verify_json_and_validation()
    test_api_verify_form_upload()
    test_authenticity_rejection()

    print("\n--------------------------------------------------------")
    print("      ALL PHASE 5 TESTS PASSED SUCCESSFULLY!")
    print("--------------------------------------------------------\n")
