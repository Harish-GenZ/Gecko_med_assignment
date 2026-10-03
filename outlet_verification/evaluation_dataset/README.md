# Synthetic Outlet Verification Evaluation Dataset (Phase 6A)

> **IMPORTANT DISCLAIMER**
> 
> **THIS IS ENTIRELY SYNTHETIC / DUMMY EVALUATION DATA.**
> - The outlet names, addresses, and coordinates are programmatically generated test fixtures.
> - None of the data represents real-world businesses, individuals, or customers.
> - Storefront images are programmatically synthesized 2D vectors rendered via Python Pillow.
> - This dataset is a development and pipeline verification benchmark, **NOT** a substitute for real-world field photographs. Real field photographs will be used in future evaluation phases.

---

## 1. Objective & Overview
The Phase 6A synthetic dataset provides a controlled, ground-truth labeled benchmark to quantitatively evaluate the complete outlet verification system (embedding inference, pgvector HNSW candidate retrieval, multimodal evidence fusion, and decision classification).

The dataset allows us to verify:
1. *When the system says `DUPLICATE`, is it actually a duplicate?*
2. *When the system says `GENUINE`, is it actually a distinct/new outlet?*
3. *When the system says `NEEDS_REVIEW`, was the case truly ambiguous or conflicting?*

---

## 2. Dataset Dimensions & Statistics

- **Base Outlets**: `55` registered outlets (with 384d MiniLM text embeddings & 512d CLIP visual embeddings).
- **Submissions**: `155` evaluation submissions.
- **Ground Truth Distribution**:
  - `DUPLICATE`: `70` (45.2%)
  - `GENUINE`: `58` (37.4%)
  - `NEEDS_REVIEW`: `27` (17.4%)
- **Image Assets**:
  - Base outlet images: `55` JPEG images (`256x256`)
  - Submission images: `135` JPEG images
  - Submissions with missing images: `20`
- **Signal Integrity**:
  - Submissions with GPS: `145`
  - Submissions with missing GPS: `10`

---

## 3. Scenario Categories

| Category | Code | Count | Expected Ground Truth | Description |
| :--- | :--- | :--- | :--- | :--- |
| **A** | `STRONG_DUPLICATE` | 25 | `DUPLICATE` | Same outlet, identical/close name, close GPS ($\le 25$m), near-duplicate photo. |
| **B** | `SAME_OUTLET_DIFFERENT_NAME` | 15 | `DUPLICATE` | Same outlet, altered naming ("Medicals" vs "Pharmacy"), close GPS, near-duplicate photo. |
| **C** | `SAME_BRAND_DIFFERENT_BRANCH` | 15 | `GENUINE` | Same brand name, distant GPS ($> 2.5$km), distinct photo. |
| **D** | `SIMILAR_NAME_DIFFERENT_OUTLET`| 15 | `GENUINE` | Similar name prefix, distant location ($> 1.5$km), different storefront photo. |
| **E** | `GPS_DRIFT` | 15 | `DUPLICATE` | Controlled GPS drift (5m, 15m, 30m, 50m, 75m, 100m, 150m) with identical storefront photo. |
| **F** | `CONFLICTING_EVIDENCE` | 15 | `NEEDS_REVIEW` | Conflicting signals (e.g. identical name + close GPS but different photo; or identical photo but far away). |
| **G** | `MISSING_IMAGE` | 10 | 5 `DUPLICATE`, 5 `GENUINE` | `image_file = NULL`. Verifies weight renormalization without penalizing missing signals as 0. |
| **H** | `MISSING_GPS` | 10 | 5 `DUPLICATE`, 3 `GENUINE`, 2 `NEEDS_REVIEW` | `latitude = NULL, longitude = NULL`. Verifies distance = NULL handling. |
| **I** | `WEAK_EVIDENCE` | 10 | `NEEDS_REVIEW` | Moderate name similarity (~0.55), missing photo, moderate distance (~300m). |
| **J** | `NEW_OUTLET` | 20 | `GENUINE` | Completely novel outlet name, novel storefront photo, and distant coordinates. |
| **K** | `PERFECT_MATCH` | 5 | `DUPLICATE` | 100% identical name, exact identical photo file, identical GPS coordinates ($0$m). |

---

## 4. Directory Structure

```
evaluation_dataset/
├── README.md
├── base_outlets/
│   ├── metadata.json           # 55 base outlet profiles (ID, name, coordinates, image file)
│   └── images/                 # 55 synthetic storefront images (OUTLET_001.jpg - OUTLET_055.jpg)
├── submissions/
│   ├── metadata.json           # 155 submission test profiles
│   └── images/                 # 135 synthetic submission images (SUB_001.jpg - SUB_155.jpg)
├── ground_truth/
│   ├── ground_truth.json       # Independent ground truth labels & metadata
│   └── ground_truth.csv        # Tabular ground truth labels for metrics analysis
├── generated/
│   ├── dataset_statistics.json # Dataset composition & scenario counts
│   └── evaluation_results.csv  # Detailed execution benchmark results (from run_evaluation.py)
└── scripts/
    ├── generate_dataset.py     # Deterministic dataset and image generator
    ├── validate_dataset.py     # Rigorous structural & integrity validator
    ├── seed_database.py        # Safe database seeder (inserts 55 base outlets with embeddings)
    ├── cleanup_dataset.py      # Safe database cleanup script (deletes only synthetic outlets)
    └── run_evaluation.py       # End-to-end evaluation runner comparing predictions vs ground truth
```

---

## 5. Usage Instructions

From the `outlet_verification/` root folder:

### 1. Validate Dataset Integrity
```powershell
python evaluation_dataset/scripts/validate_dataset.py
```

### 2. Seed Base Outlets into Database
Inserts the 55 synthetic base outlets into the PostgreSQL `outlets` table and generates their 384-dimensional text embeddings and 512-dimensional CLIP image embeddings:
```powershell
python evaluation_dataset/scripts/seed_database.py
```

### 3. Run Benchmark Evaluation
Runs all 155 submissions through the Phase 5 verification engine and compares predictions against ground truth:
```powershell
# Run full 155 submissions benchmark
python evaluation_dataset/scripts/run_evaluation.py

# Or test a small subset (e.g. first 15 submissions)
python evaluation_dataset/scripts/run_evaluation.py --limit 15
```

### 4. Clean Up Database
Safely deletes all synthetic base outlets using their deterministic UUIDs and synthetic URL prefixes, leaving all unrelated database records completely untouched:
```powershell
python evaluation_dataset/scripts/cleanup_dataset.py
```
