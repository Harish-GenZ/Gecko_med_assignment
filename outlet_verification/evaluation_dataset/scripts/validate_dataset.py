import json
import sys
from pathlib import Path
from PIL import Image

DATASET_ROOT = Path(__file__).resolve().parent.parent
BASE_OUTLETS_DIR = DATASET_ROOT / "base_outlets"
BASE_IMAGES_DIR = BASE_OUTLETS_DIR / "images"
SUBMISSIONS_DIR = DATASET_ROOT / "submissions"
SUB_IMAGES_DIR = SUBMISSIONS_DIR / "images"
GROUND_TRUTH_DIR = DATASET_ROOT / "ground_truth"
GENERATED_DIR = DATASET_ROOT / "generated"

VALID_DECISIONS = {"DUPLICATE", "GENUINE", "NEEDS_REVIEW"}
REQUIRED_SCENARIOS = {
    "STRONG_DUPLICATE",
    "SAME_OUTLET_DIFFERENT_NAME",
    "SAME_BRAND_DIFFERENT_BRANCH",
    "SIMILAR_NAME_DIFFERENT_OUTLET",
    "GPS_DRIFT",
    "CONFLICTING_EVIDENCE",
    "MISSING_IMAGE",
    "MISSING_GPS",
    "WEAK_EVIDENCE",
    "NEW_OUTLET",
    "PERFECT_MATCH",
}


def validate_dataset() -> bool:
    print("========================================================")
    print("  PHASE 6A: VALIDATING SYNTHETIC EVALUATION DATASET")
    print("========================================================\n")

    errors: list[str] = []

    # 1. Validate Base Outlets
    base_meta_file = BASE_OUTLETS_DIR / "metadata.json"
    if not base_meta_file.exists():
        errors.append(f"Missing base outlets metadata file: {base_meta_file}")
        print("FATAL:", errors[-1])
        return False

    with open(base_meta_file, "r", encoding="utf-8") as f:
        base_outlets = json.load(f)

    if not (50 <= len(base_outlets) <= 60):
        errors.append(f"Expected 50-60 base outlets, found {len(base_outlets)}.")

    base_outlet_ids = set()
    for o in base_outlets:
        oid = o.get("outlet_id")
        if not oid:
            errors.append("Base outlet missing 'outlet_id'.")
        elif oid in base_outlet_ids:
            errors.append(f"Duplicate base outlet_id: {oid}")
        base_outlet_ids.add(oid)

        # Coordinates
        lat, lng = o.get("latitude"), o.get("longitude")
        if not (isinstance(lat, (int, float)) and -90 <= lat <= 90):
            errors.append(f"Invalid latitude for base outlet {oid}: {lat}")
        if not (isinstance(lng, (int, float)) and -180 <= lng <= 180):
            errors.append(f"Invalid longitude for base outlet {oid}: {lng}")

        # Image existence & validity
        img_name = o.get("image_file")
        if not img_name:
            errors.append(f"Base outlet {oid} missing image_file.")
        else:
            img_path = BASE_IMAGES_DIR / img_name
            if not img_path.exists():
                errors.append(f"Base image not found on disk: {img_path}")
            else:
                try:
                    with Image.open(img_path) as im:
                        im.verify()
                except Exception as exc:
                    errors.append(f"Corrupt base image {img_path}: {exc}")

    # 2. Validate Submissions
    sub_meta_file = SUBMISSIONS_DIR / "metadata.json"
    if not sub_meta_file.exists():
        errors.append(f"Missing submissions metadata file: {sub_meta_file}")
        print("FATAL:", errors[-1])
        return False

    with open(sub_meta_file, "r", encoding="utf-8") as f:
        submissions = json.load(f)

    if len(submissions) < 140:
        errors.append(f"Expected ~150 submissions, found {len(submissions)}.")

    sub_ids = set()
    scenarios_found = set()
    decisions_count = {"DUPLICATE": 0, "GENUINE": 0, "NEEDS_REVIEW": 0}
    missing_images_count = 0
    missing_gps_count = 0

    for s in submissions:
        sid = s.get("submission_id")
        if not sid:
            errors.append("Submission missing 'submission_id'.")
        elif sid in sub_ids:
            errors.append(f"Duplicate submission_id: {sid}")
        sub_ids.add(sid)

        # Expected decision
        dec = s.get("expected_decision")
        if dec not in VALID_DECISIONS:
            errors.append(f"Invalid expected_decision '{dec}' in submission {sid}")
        else:
            decisions_count[dec] += 1

        # Scenario
        scen = s.get("scenario")
        if scen not in REQUIRED_SCENARIOS:
            errors.append(f"Unknown scenario '{scen}' in submission {sid}")
        else:
            scenarios_found.add(scen)

        # Expected match outlet ID
        match_id = s.get("expected_match_outlet_id")
        if match_id is not None:
            if match_id not in base_outlet_ids:
                errors.append(f"Submission {sid} references non-existent base outlet {match_id}")

        # Coordinates
        lat, lng = s.get("latitude"), s.get("longitude")
        if lat is None or lng is None:
            missing_gps_count += 1
            if scen != "MISSING_GPS":
                errors.append(f"Submission {sid} has missing GPS but scenario is '{scen}'")
        else:
            if not (-90 <= lat <= 90):
                errors.append(f"Latitude out of bounds in {sid}: {lat}")
            if not (-180 <= lng <= 180):
                errors.append(f"Longitude out of bounds in {sid}: {lng}")

        # Image
        img_name = s.get("image_file")
        if img_name is None:
            missing_images_count += 1
            if scen not in {"MISSING_IMAGE", "WEAK_EVIDENCE"}:
                errors.append(f"Submission {sid} has missing image but scenario is '{scen}'")
        else:
            img_path = SUB_IMAGES_DIR / img_name
            if not img_path.exists():
                errors.append(f"Submission image not found: {img_path}")
            else:
                try:
                    with Image.open(img_path) as im:
                        im.verify()
                except Exception as exc:
                    errors.append(f"Corrupt submission image {img_path}: {exc}")

    # Check all required scenarios are covered
    missing_scenarios = REQUIRED_SCENARIOS - scenarios_found
    if missing_scenarios:
        errors.append(f"Missing required scenarios: {missing_scenarios}")

    # 3. Validate Ground Truth Files
    gt_json = GROUND_TRUTH_DIR / "ground_truth.json"
    gt_csv = GROUND_TRUTH_DIR / "ground_truth.csv"
    if not gt_json.exists():
        errors.append(f"Missing ground truth JSON file: {gt_json}")
    if not gt_csv.exists():
        errors.append(f"Missing ground truth CSV file: {gt_csv}")

    # 4. Summary & Report
    print("--------------------------------------------------------")
    print(f"  Base outlets:         {len(base_outlets)}")
    print(f"  Total submissions:    {len(submissions)}")
    print()
    print("  Ground Truth Decisions:")
    for k, v in decisions_count.items():
        print(f"    {k:<16}: {v}")
    print()
    print("  Scenario Coverage:")
    for s_name in sorted(REQUIRED_SCENARIOS):
        cnt = sum(1 for s in submissions if s.get("scenario") == s_name)
        print(f"    {s_name:<30}: {cnt}")
    print()
    print("  Image & Signal Integrity:")
    print(f"    Base Images on disk:     {len(list(BASE_IMAGES_DIR.glob('*.jpg')))}")
    print(f"    Sub Images on disk:      {len(list(SUB_IMAGES_DIR.glob('*.jpg')))}")
    print(f"    Submissions w/o Image:   {missing_images_count}")
    print(f"    Submissions w/o GPS:     {missing_gps_count}")
    print("--------------------------------------------------------")

    if errors:
        print(f"\n[FAIL] Validation found {len(errors)} error(s):")
        for err in errors[:10]:
            print(f"  - {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more.")
        return False

    print("\n[PASS] DATASET VALIDATION: PASS")
    print("All IDs unique, coordinates valid, images intact, scenarios complete.")
    print("--------------------------------------------------------\n")
    return True


if __name__ == "__main__":
    success = validate_dataset()
    sys.exit(0 if success else 1)
