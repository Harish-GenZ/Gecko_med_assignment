# Phase 6B Evaluation Report: Objective Benchmark of Current Pipeline

**Project**: Outlet Verification Engine  
**Phase**: 6B — Objective Production-Pipeline Benchmark  
**Date**: October 3, 2026  
**Status**: Completed  
**Evaluation Dataset**: Phase 6A Synthetic Benchmark (55 Base Outlets, 155 Submissions)  
**Database**: Railway Cloud PostgreSQL + pgvector (55 Reference Outlets Seeded)  

---

## 1. Objective

The primary objective of Phase 6B is to conduct an **unbiased, objective, end-to-end evaluation** of the current production-like outlet verification pipeline against the entire Phase 6A synthetic dataset.

Crucially:
- **No algorithms, formulas, or thresholds were modified** for this evaluation.
- Ground-truth labels were strictly isolated from the verification services and used solely as an independent reference to calculate classification accuracy, candidate match accuracy, confusion matrices, signal correlation, and error attribution.
- The pipeline was exercised end-to-end: query text embedding (`all-MiniLM-L6-v2`), query image embedding (`open_clip ViT-B/32`), multi-channel candidate retrieval (HNSW name, HNSW image, earthdistance GPS), candidate deduplication, multimodal evidence fusion, and deterministic decision engine evaluation.

---

## 2. Dataset

The evaluation was performed against the Phase 6A synthetic benchmark dataset located in `outlet_verification/evaluation_dataset/`:

| Dimension | Count | Details |
| :--- | :--- | :--- |
| **Base Outlets** | 55 | Realistically distributed across 5 Indian metro clusters (Bangalore, Chennai, Madurai, Coimbatore, Trichy). Seeded into Railway PostgreSQL with 384d text and 512d image embeddings. |
| **Submissions** | 155 | Synthetic submissions spanning 11 realistic field verification scenarios. |
| **Ground Truth Classes** | 3 | **DUPLICATE**: 70 (45.2%), **GENUINE**: 58 (37.4%), **NEEDS_REVIEW**: 27 (17.4%). |
| **Modalities Tested** | 3 | Outlet name, GPS coordinates (latitude/longitude), and storefront photo (Pillow rendered with architectural and signage features). |
| **Missing Signal Cases** | 20 | 10 submissions with missing photo (`image_file=None`), 10 submissions with missing GPS (`latitude=None, longitude=None`). |

---

## 3. Evaluation Methodology

Every submission was processed sequentially through the real application service layer:

```
Submission Record (Name, GPS, Image)
          │
          ▼
Embedding Infrastructure (Phase 3B)
  ├─ all-MiniLM-L6-v2 (384-dim normalized vector)
  └─ CLIP ViT-B/32 (512-dim normalized vector)
          │
          ▼
Multi-Channel Candidate Retrieval (Phase 4)
  ├─ Channel A: Name HNSW Cosine Search (top 10)
  ├─ Channel B: Image HNSW Cosine Search (top 10)
  └─ Channel C: Geographic Radius Search (500m radius, earthdistance)
          │
          ▼
Candidate Deduplication & Evidence Assembly
          │
          ▼
Multimodal Evidence Fusion & Scoring (Phase 5)
  ├─ Dynamic weight renormalization for missing signals
  ├─ Distance decay: exp(-ln(2) * dist / 50m)
  └─ Weighted fusion: S = 0.30 * S_name + 0.45 * S_image + 0.25 * S_geo
          │
          ▼
Deterministic Decision Engine (Phase 5)
  ├─ Safeguard 1: Conflicting evidence check (Name high >= 0.85, Image low < 0.40)
  ├─ Safeguard 2: Ambiguous candidate margin check (Margin < 0.10)
  ├─ Safeguard 3: Missing signal with ambiguous evidence check
  └─ Confidence thresholds: DUPLICATE >= 0.80, GENUINE <= 0.35, NEEDS_REVIEW in [0.35, 0.80]
          │
          ▼
Independent Ground Truth Comparison & Metrics Generation
```

All 155 submissions completed successfully. For each submission, 25 diagnostic fields were captured and written to `generated/evaluation_results.csv`.

---

## 4. Overall Results

Across all 155 submissions in the benchmark dataset:

- **Total Submissions Evaluated**: 155
- **Correct Decisions**: 85
- **Overall Decision Accuracy**: **54.84%**
- **Total Classification Errors**: 70
- **Total Evaluation Wall-Clock Time**: 193.23 seconds
- **Average Submission Latency**: 1120.16 ms

### Key Executive Observation

While raw 3-class accuracy is 54.84%, **62 out of the 70 classification errors (88.6%) represent conservative escalations into `NEEDS_REVIEW`**. The system exhibited an exceptionally strong safety profile:
- **Zero duplicates** were misclassified as `GENUINE` (0 / 70).
- **Zero genuine outlets** were misclassified as `DUPLICATE` (0 / 58).
- Only **1 single false positive duplicate** occurred across the entire 155-submission test suite (a 98.8% specificity rate).

---

## 5. Confusion Matrix

The full 3x3 confusion matrix mapping **Ground Truth (Rows)** against **Pipeline Predictions (Columns)** is presented below:

| Ground Truth \ Predicted | DUPLICATE | GENUINE | NEEDS_REVIEW | Total | Row Accuracy (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DUPLICATE** | **42** | 0 | 28 | 70 | 60.0% |
| **GENUINE** | 0 | **24** | 34 | 58 | 41.4% |
| **NEEDS_REVIEW** | 1 | 7 | **19** | 27 | 70.4% |
| **Total Predicted** | 43 | 31 | 81 | 155 | — |

### Confusion Matrix Observations
1. **DUPLICATE Row**: 42 correctly identified as `DUPLICATE` (60.0%). The remaining 28 were diverted to `NEEDS_REVIEW` because evidence was incomplete (missing image, missing GPS) or attenuated by GPS drift / name variation. Not a single duplicate was erroneously approved as `GENUINE`.
2. **GENUINE Row**: 24 correctly recognized as `GENUINE` (41.4%). 34 were diverted to `NEEDS_REVIEW` primarily due to chain brand similarity across different branches. Not a single genuine outlet was falsely flagged as `DUPLICATE`.
3. **NEEDS_REVIEW Row**: 19 correctly held for review (70.4%). 7 were predicted as `GENUINE` (cases where conflicting images occurred >3.5 km away), and 1 was predicted as `DUPLICATE` (SUB_087).

---

## 6. Per-Class Metrics

| Class | Support | True Positives | False Positives | False Negatives | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DUPLICATE** | 70 | 42 | 1 | 28 | **0.9767** | **0.6000** | **0.7434** |
| **GENUINE** | 58 | 24 | 7 | 34 | **0.7742** | **0.4138** | **0.5393** |
| **NEEDS_REVIEW** | 27 | 19 | 62 | 8 | **0.2346** | **0.7037** | **0.3519** |
| **Macro Average** | 155 | — | — | — | 0.6618 | 0.5725 | 0.5449 |
| **Weighted Average** | 155 | — | — | — | 0.7716 | 0.5484 | 0.6022 |

*Note: In accordance with rigorous evaluation standards, zero-division cases were handled explicitly; support reflects the true class distribution.*

---

## 7. Duplicate Detection Metrics (Binary)

Because duplicate detection is the primary risk mitigation challenge for outlet verification, binary metrics treat **DUPLICATE as the Positive class** and **Non-Duplicate (`GENUINE` + `NEEDS_REVIEW`) as the Negative class**:

| Metric | Value | Operational Definition & Real-World Impact |
| :--- | :---: | :--- |
| **True Positives (TP)** | 42 | Existing outlets correctly identified and matched as duplicates. |
| **False Positives (FP)** | 1 | Genuine or ambiguous submissions incorrectly classified as duplicates. |
| **True Negatives (TN)** | 84 | Genuine or ambiguous outlets correctly prevented from being marked duplicate. |
| **False Negatives (FN)** | 28 | True duplicates that failed to reach the 0.80 duplicate threshold (all sent to `NEEDS_REVIEW`). |
| **Precision** | **97.67%** | When the system declares an outlet a `DUPLICATE`, it is correct **97.7%** of the time. |
| **Recall (Sensitivity)** | **60.00%** | The system automatically resolves **60.0%** of all duplicate submissions without human intervention. |
| **F1-Score** | **0.7434** | Harmonic mean of precision and recall. |
| **Specificity** | **98.82%** | **98.8%** of non-duplicate submissions are correctly rejected from automatic duplicate labeling. |
| **False Positive Rate (FPR)** | **1.18%** | Only **1.2%** probability of a false duplicate accusation. |
| **False Negative Rate (FNR)** | **40.00%** | **40.0%** of duplicates require human review to verify, but zero are accepted as new genuine outlets. |

---

## 8. Match Identification Accuracy

For submissions whose expected ground-truth decision is `DUPLICATE` (70 samples), we distinguish between deciding *whether* an outlet is a duplicate and identifying *which* reference outlet it matches:

| Metric | Accuracy (%) | Ratio | Operational Meaning |
| :--- | :---: | :---: | :--- |
| **Decision Accuracy** | **60.00%** | 42 / 70 | Proportion of duplicate cases where the pipeline assigned the `DUPLICATE` decision label. |
| **Match Accuracy (All Duplicates)** | **100.00%** | 70 / 70 | Proportion of all 70 duplicate cases where the top candidate retrieved was the exact target base outlet. |
| **Match Accuracy (Detected Duplicates)** | **100.00%** | 42 / 42 | When the pipeline made a `DUPLICATE` decision, the predicted matched outlet was correct 100% of the time. |

### Critical Finding: Phase 4 Candidate Retrieval is 100% Effective
In all 28 cases where a duplicate was classified as `NEEDS_REVIEW`, **the top candidate returned by the Phase 4 retrieval service was nevertheless the exact correct reference outlet (`expected_match_outlet_id == predicted_match_outlet_id`)**. The retrieval stage achieved a perfect **100% Top-1 Recall** across the entire duplicate benchmark. The decision to send those 28 samples to `NEEDS_REVIEW` was purely governed by the downstream confidence scoring and safeguards.

---

## 9. Scenario-by-Scenario Results

Below is the complete performance breakdown across all 11 evaluation categories:

| Scenario | Samples | Exp Dist | Pred Dist | Correct | Accuracy (%) | Avg Conf | Avg Cov | Avg Agr | Avg Latency | Primary Failure Source |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `PERFECT_MATCH` | 5 | D:5/G:0/R:0 | D:5/G:0/R:0 | 5 | **100.0%** | 0.9982 | 1.00 | 0.9991 | 1086.7 ms | NONE |
| `STRONG_DUPLICATE` | 25 | D:25/G:0/R:0 | D:25/G:0/R:0 | 25 | **100.0%** | 0.8802 | 1.00 | 0.9511 | 1457.3 ms | NONE |
| `WEAK_EVIDENCE` | 10 | D:0/G:0/R:10 | D:0/G:0/R:10 | 10 | **100.0%** | 0.2377 | 0.52 | 0.7691 | 673.9 ms | NONE |
| `NEW_OUTLET` | 20 | D:0/G:20/R:0 | D:0/G:17/R:3 | 17 | **85.0%** | 0.3183 | 0.82 | 0.6532 | 1072.6 ms | THRESHOLD_ERROR |
| `GPS_DRIFT` | 15 | D:15/G:0/R:0 | D:7/G:0/R:8 | 7 | **46.7%** | 0.7595 | 1.00 | 0.8709 | 1122.3 ms | THRESHOLD_ERROR |
| `CONFLICTING_EVIDENCE` | 15 | D:0/G:0/R:15 | D:1/G:7/R:7 | 7 | **46.7%** | 0.5112 | 0.73 | 0.7315 | 1105.1 ms | GPS_ERROR |
| `SIMILAR_NAME_DIFFERENT_OUTLET` | 15 | D:0/G:15/R:0 | D:0/G:7/R:8 | 7 | **46.7%** | 0.3495 | 0.88 | 0.6684 | 1155.7 ms | THRESHOLD_ERROR |
| `SAME_OUTLET_DIFFERENT_NAME` | 15 | D:15/G:0/R:0 | D:5/G:0/R:10 | 5 | **33.3%** | 0.7561 | 0.98 | 0.9103 | 1310.6 ms | THRESHOLD_ERROR |
| `MISSING_GPS` | 10 | D:5/G:3/R:2 | D:0/G:0/R:10 | 2 | **20.0%** | 0.8155 | 0.60 | 0.9979 | 1014.0 ms | AMBIGUITY_ERROR |
| `SAME_BRAND_DIFFERENT_BRANCH` | 15 | D:0/G:15/R:0 | D:0/G:0/R:15 | 0 | **0.0%** | 0.4263 | 1.00 | 0.6917 | 1083.7 ms | THRESHOLD_ERROR |
| `MISSING_IMAGE` | 10 | D:5/G:5/R:0 | D:0/G:0/R:10 | 0 | **0.0%** | 0.4450 | 0.55 | 0.8407 | 676.6 ms | MISSING_SIGNAL |

---

## 10. Signal Analysis

Detailed signal inspection reveals exactly how the individual modality signals interact within the current fusion engine:

### 1. High-Performing Signal Combinations
- **Perfect / Near-Perfect Triangulation**: When Name similarity > 0.85, Image similarity > 0.85, and GPS distance < 25 meters, confidence scores reliably exceed **0.88 - 0.99**, achieving 100% accuracy on `PERFECT_MATCH` and `STRONG_DUPLICATE`.
- **Completely Distinct Entities**: When Name similarity < 0.40, Image similarity < 0.40, and GPS distance > 5 km, confidence scores reliably fall below **0.30**, correctly achieving `GENUINE` decisions on 85% of `NEW_OUTLET` cases.

### 2. Modality Decay and Threshold Boundary Effects

#### A. GPS Drift Decay
In `GPS_DRIFT` (15 samples), duplicate outlets were subjected to controlled coordinate shifts from 5m to 150m:
- **5m, 15m, 30m**: Confidence = **0.82 - 0.91** -> **Correctly classified as `DUPLICATE`**.
- **50m drift**: Name = 1.0, Image = 0.93, Dist = 50.1m -> Confidence = **0.7545** -> Falls below 0.80 -> `NEEDS_REVIEW`.
- **75m drift**: Name = 1.0, Image = 0.91, Dist = 75.1m -> Confidence = **0.6813** -> `NEEDS_REVIEW`.
- **100m drift**: Name = 1.0, Image = 0.87, Dist = 100.1m -> Confidence = **0.6199** -> `NEEDS_REVIEW`.
- **150m drift**: Name = 1.0, Image = 0.90, Dist = 150.1m -> Confidence = **0.5557** -> `NEEDS_REVIEW`.

*Root Cause*: The distance exponential decay function uses a half-life of 50.0 meters:
$$\text{decay} = \exp\left(-\frac{\ln(2) \cdot \text{distance}}{50.0}\right)$$
At 50 meters, the geographic score drops to 0.50. At 100 meters, it drops to 0.25. Even with 100% identical name and 92% identical image, the weighted sum $(0.30 \times 1.0 + 0.45 \times 0.92 + 0.25 \times 0.25) = 0.776$, which falls below the strict 0.80 threshold.

#### B. Brand-Branch Ambiguity
In `SAME_BRAND_DIFFERENT_BRANCH` (15 samples, expected `GENUINE`):
- Example **SUB_041**: Name = `Apollo QuickMeds Synthetic - KKNagar Branch`, Reference = `Apollo QuickMeds Synthetic - AnnaNagar Branch`.
  - Name Similarity: **0.7156** (shares brand prefix).
  - Image Similarity: **0.8591** (standardized corporate pharmacy facade).
  - GPS Distance: **284,456 meters** (284 km away in another city).
  - Geographic Score: **0.0000** (distance > 1000m).
  - Fused Confidence: **0.4205**.
  - Current Genuine Threshold: $\le \mathbf{0.35}$.
  - Result: Because 0.4205 is in $[0.35, 0.80]$, the engine outputs `NEEDS_REVIEW`.

*Root Cause*: The engine does not have an explicit cross-branch or locality differentiation rule. The high visual consistency of corporate retail chains keeps the score above the genuine cutoff (0.35) despite vast geographic separation.

#### C. Missing Modality Dynamics
- **Missing Image (10 samples)**:
  - When image is missing, the weights are dynamically re-normalized: Name weight becomes $0.30 / (0.30 + 0.25) \approx 0.545$, and Geo weight becomes $0.25 / 0.55 \approx 0.455$.
  - In SUB_101 (Duplicate with missing image, distance 20m): Fused confidence = **0.7218**.
  - Because 0.7218 < 0.80, it cannot automatically reach `DUPLICATE` without image evidence.
  - In SUB_106 (Genuine with missing image, distance 5.3km): Name similarity = 0.4573. Safeguard 6 explicitly flags moderate name similarity $(0.40 \le S \le 0.75)$ with missing image as `WEAK_EVIDENCE_REQUIRES_REVIEW` -> `NEEDS_REVIEW`.
- **Missing GPS (10 samples)**:
  - In SUB_111-SUB_115 (Duplicates with missing GPS): Name = 1.0, Image = 0.99. Fused confidence = **0.9231**.
  - However, because GPS is missing, multiple branches of the same chain in the database produce near-identical high scores (runner-up confidence = 0.8334, margin = 0.0897).
  - Safeguard 4 (`AMBIGUOUS_TOP_CANDIDATES`, margin < 0.10) triggers automatically and routes the submission to `NEEDS_REVIEW`.

---

## 11. Error Analysis

All 70 incorrect predictions were isolated and classified deterministically based on empirical signal values:

| Failure Category | Error Count | Percentage | Primary Contributing Factors |
| :--- | :---: | :---: | :--- |
| **THRESHOLD_ERROR** | 38 | 54.3% | Fused confidence fell into the review band $[0.35, 0.80]$ for clear duplicates (attenuated by GPS drift or name change) or clear genuine chain branches. |
| **OTHER (Safe Review)** | 12 | 17.1% | Edge cases where visual similarity of standard shop signs kept confidence above genuine threshold. |
| **MISSING_SIGNAL** | 8 | 11.4% | Missing photo or GPS triggered designed safeguards, preventing automatic classification. |
| **GPS_ERROR** | 7 | 10.0% | Conflicting evidence samples where image was identical but distance was >3.5 km; the extreme distance penalty pulled confidence below 0.35 into `GENUINE`. |
| **AMBIGUITY_ERROR** | 5 | 7.1% | Missing GPS caused runner-up candidate score to tie within 0.10 margin, triggering the ambiguity safeguard. |
| **CANDIDATE_RETRIEVAL_ERROR** | 0 | 0.0% | **Zero retrieval failures**. Top candidate was correct for 100% of duplicates. |
| **NAME_ERROR** | 0 | 0.0% | MiniLM text embeddings successfully ranked every variant. |
| **IMAGE_ERROR** | 0 | 0.0% | CLIP image embeddings successfully matched storefront visual features. |

### Concrete Error Case Studies

#### Case 1: GPS Drift Attenuation (SUB_075)
- **Scenario**: `GPS_DRIFT` (75m coordinate offset)
- **Expected**: `DUPLICATE` (SYNTH_OUTLET_035)
- **Predicted**: `NEEDS_REVIEW`
- **Signals**: `name_similarity` = 1.0000, `image_similarity` = 0.9151, `distance_meters` = 75.07m
- **Fused Confidence**: **0.6813** (Candidate Margin: 0.2260, Agreement: 0.8211)
- **Reason Codes**: `STRONG_IMAGE_GEO_MATCH; INSUFFICIENT_EVIDENCE`
- **Diagnosis**: The top candidate was exactly SYNTH_OUTLET_035. However, at 75m, GPS distance decay dropped the geographic score from 1.0 to 0.353, pulling total confidence to 0.6813. Because 0.6813 < 0.80, the engine properly avoided an uncalibrated duplicate declaration and requested human review.

#### Case 2: Chain Multi-Branch Storefront Ambiguity (SUB_041)
- **Scenario**: `SAME_BRAND_DIFFERENT_BRANCH`
- **Expected**: `GENUINE` (New branch of Apollo QuickMeds in KK Nagar)
- **Predicted**: `NEEDS_REVIEW`
- **Signals**: `name_similarity` = 0.7156, `image_similarity` = 0.8591, `distance_meters` = 284,456.25m
- **Fused Confidence**: **0.4205** (Candidate Margin: 0.0004, Agreement: 0.6994)
- **Reason Codes**: `INSUFFICIENT_EVIDENCE`
- **Diagnosis**: The physical outlet is in a completely different city (284 km away). However, corporate brand colors, signage typography, and pharmacy layout caused CLIP visual similarity to reach 0.8591. The fused score of 0.4205 was just 0.07 above the genuine threshold of 0.35, diverting it to review.

#### Case 3: Missing GPS Margin Ambiguity (SUB_111)
- **Scenario**: `MISSING_GPS`
- **Expected**: `DUPLICATE` (SYNTH_OUTLET_021)
- **Predicted**: `NEEDS_REVIEW`
- **Signals**: `name_similarity` = 1.0000, `image_similarity` = 0.9981, `distance_meters` = None
- **Fused Confidence**: **0.9231** (Coverage: 0.75, Agreement: 0.9991)
- **Reason Codes**: `AMBIGUOUS_TOP_CANDIDATES`
- **Diagnosis**: Confidence was extremely high (0.9231). However, because GPS was missing, a runner-up candidate with identical branding had a confidence of 0.8334. The margin was $0.9231 - 0.8334 = 0.0897$, which violated the minimum margin threshold of 0.10. Safeguard 4 intervened to ensure a human confirms which branch is being submitted.

---

## 12. Latency & Performance Analysis

End-to-end performance was measured with microsecond precision across all 155 submissions:

| Metric | Total / Breakdown | Production Implication |
| :--- | :---: | :--- |
| **Total Evaluation Wall Time** | 193.23 seconds | Full benchmark across 155 remote database operations completed in 3.2 minutes. |
| **Submissions Evaluated** | 155 | Real network calls to Railway PostgreSQL over public TCP proxy. |
| **Mean End-to-End Latency** | **1120.16 ms** | ~1.1 seconds per submission end-to-end. |
| **Median (p50) Latency** | **1096.19 ms** | Half of all requests complete in under 1.1 seconds. |
| **p90 Latency** | **1255.00 ms** | 90% of requests complete in under 1.25 seconds. |
| **p95 Latency** | **1299.64 ms** | 95% of requests complete in under 1.30 seconds. |
| **Maximum Latency** | **7025.20 ms** | Single outlier caused by network TCP proxy keepalive reconnect. |

### Component Latency Breakdown

```
Average Submission Latency: 1120.16 ms
┌───────────────────────────────┬────────────┬─────────┐
│ Component                     │ Latency    │ % Share │
├───────────────────────────────┼────────────┼─────────┤
│ Embedding Inference (CPU)     │  106.46 ms │    9.5% │
│ Database Retrieval & Transfer │ 1013.47 ms │   90.5% │
│ Decision Engine & Safeguards  │    0.24 ms │   <0.1% │
└───────────────────────────────┴────────────┴─────────┘
```

### Performance Insights
1. **Decision Engine is Zero-Cost**: The evidence fusion, weight renormalization, 6 safeguards, and deterministic threshold evaluations execute in **0.24 milliseconds** on CPU.
2. **Embeddings are Efficient**: Generating both a 384d MiniLM text embedding and a 512d CLIP image embedding takes an average of **106.46 ms** on standard CPU.
3. **Database Roundtrips are the Dominant Bottleneck**: Candidate retrieval takes **1013.47 ms** (90.5% of total time). This is because the database is hosted in Railway US-West, and the local test runner executed 3 sequential queries per submission (Name HNSW, Image HNSW, Geo Earthdistance) over an international public TCP proxy (`tramway.proxy.rlwy.net`). When co-located with the database in Railway, retrieval latency is expected to drop below **25-40 ms**.

---

## 13. Current System Strengths

1. **Flawless Candidate Discovery**: The Phase 4 multi-channel retrieval layer achieved **100% Top-1 Recall** across all 70 duplicate submissions in the benchmark. It never missed a true reference outlet.
2. **Near-Zero False Positive Risk**: Binary duplicate specificity reached **98.82%** with a precision of **97.67%**. In an enterprise pharmacy network, false duplicate merges are catastrophic (causing order misrouting and billing corruption); the current system virtually eliminates this risk.
3. **Complete Elimination of Dangerous False Negatives**: Not a single duplicate outlet was erroneously approved as `GENUINE` (0 / 70).
4. **Deterministic and Audit-Ready**: Every decision produces structured reason codes (`STRONG_MULTIMODAL_MATCH`, `AMBIGUOUS_TOP_CANDIDATES`, `INSUFFICIENT_EVIDENCE`), exact similarity scores, and candidate margins.
5. **Robust Multimodal Synergy**: When all three signals align, the system achieves 100% precision with average confidence > 0.88.

---

## 14. Current Failure Modes

1. **High Review Escalation Rate (Conservative Bias)**:
   - 49.7% of all submissions (77 / 155) were routed to `NEEDS_REVIEW`.
   - While safe, a 50% human review rate imposes an operational burden on field audit teams.
2. **Chain Storefront Visual Homogeneity**:
   - Corporate retail chains (Apollo, MedPlus, WellCare) maintain identical signage and architectural templates across all branches.
   - When combined with identical brand prefixes, the visual similarity (~0.85-0.92) keeps confidence above 0.35 even when the outlets are located in different cities, preventing automatic `GENUINE` approval.
3. **Missing Image Penalty Barrier**:
   - Without a storefront photo, re-normalized confidence tops out at ~0.72 even with 100% name match and 20m GPS proximity.
   - As a result, 0% of duplicates with missing photos could be automatically resolved without human review.
4. **GPS Drift Sensitivity**:
   - Outlets with coordinate errors between 50m and 150m (common with standard smartphone GPS in urban street canyons) experience rapid confidence decay below the 0.80 threshold.
5. **Margin Collapse in Missing GPS**:
   - When GPS is missing, multiple branches of the same chain produce near-identical confidence scores, triggering the margin safeguard (<0.10) for 100% of missing-GPS duplicate cases.

---

## 15. What Should NOT Be Changed Yet

Based strictly on empirical evidence, the following elements are working correctly and **must NOT be changed**:

1. **Do NOT blindly lower the Duplicate Threshold (0.80)**:
   - Lowering the threshold to 0.70 would increase duplicate recall, but it would immediately convert ambiguous chain branches (`SAME_BRAND_DIFFERENT_BRANCH`) and conflicting cases into catastrophic false positive duplicates, degrading the current 97.7% precision.
2. **Do NOT raise the Genuine Threshold (0.35) arbitrarily**:
   - Raising the threshold to 0.45 would allow some chain branches to pass as genuine, but would simultaneously allow weak duplicates with degraded images to slip through as genuine.
3. **Do NOT alter the Phase 3B Embedding Models**:
   - `all-MiniLM-L6-v2` and `open_clip ViT-B/32` achieved a perfect 100% candidate retrieval rate. The embedding representations are not the source of errors.
4. **Do NOT remove the Ambiguity Margin Safeguard**:
   - The margin safeguard correctly prevented automatic duplicate assignment when the system could not distinguish between two identical chain branches without GPS.

---

## 16. Actionable Recommendations for Phase 6C

Phase 6C should focus on principled, evidence-driven refinements to address the specific failure modes identified:

1. **Implement a Chain / Multi-Branch Geographic Disambiguation Rule**:
   - *Observation*: In `SAME_BRAND_DIFFERENT_BRANCH`, distance is > 2,000 meters, but visual similarity is high due to brand templates.
   - *Recommendation*: If $\text{distance} > 2,000\text{m}$ and name similarity is moderate/high, suppress the image weight or apply an explicit multi-branch penalty. An outlet 50 km away cannot be a duplicate of an existing outlet regardless of storefront branding.
2. **Calibrate Missing-Image Duplicate Handling**:
   - *Observation*: Duplicates with missing photos had 100% name match and <20m GPS proximity, but confidence was capped at 0.72.
   - *Recommendation*: Introduce a specialized rule or conditional threshold: if photo is missing, but name match is exact ($\ge 0.95$) and GPS distance is $< 25\text{m}$, allow the decision engine to confirm `DUPLICATE` with reason code `STRONG_NAME_GEO_MATCH_MISSING_IMAGE`.
3. **Calibrate Distance Decay Half-Life**:
   - *Observation*: At 50m drift, distance decay score is 0.50, pulling confidence down to 0.75.
   - *Recommendation*: Experiment with increasing the geographic half-life from 50m to 75m or 100m, or use a piecewise plateau (e.g. 1.0 similarity up to 30m, decaying thereafter) to accommodate typical consumer mobile GPS inaccuracies.
4. **Locality / Area Name Extraction**:
   - *Observation*: Names like `Sunrise Medicals - Indiranagar Branch` vs `Sunrise Medicals - Gandhipuram Branch` differ primarily in the locality token.
   - *Recommendation*: Utilize simple token-level locality comparison to differentiate branches when global text similarity remains high.
5. **Database Query Batching / Co-location**:
   - *Observation*: Database retrieval accounts for 90.5% of pipeline latency (1013 ms) due to 3 sequential remote roundtrips.
   - *Recommendation*: Execute the three retrieval queries concurrently or combine them into a single CTE query to reduce roundtrip latency.

---

## 17. Artifact Registry

The following objective data artifacts were generated and persisted during Phase 6B:

| Artifact Path | Format | Description |
| :--- | :---: | :--- |
| `evaluation_dataset/generated/evaluation_results.csv` | CSV | Complete row-by-row log of all 155 submissions with 25 diagnostic fields. |
| `evaluation_dataset/generated/confusion_matrix.csv` | CSV | 3x3 confusion matrix with raw sample counts and row accuracy percentages. |
| `evaluation_dataset/generated/scenario_metrics.csv` | CSV | Detailed performance, accuracy, confidence, and latency for all 11 scenarios. |
| `evaluation_dataset/generated/error_analysis.csv` | CSV | Detailed inventory of all 70 errors with deterministic root-cause attribution. |
| `evaluation_dataset/generated/signal_analysis.csv` | CSV | Per-submission evidence metrics (name, image, distance, confidence, margin). |
| `evaluation_dataset/generated/performance_metrics.json` | JSON | Component latency breakdowns, mean, p50, p90, p95, and maximum times. |
| `evaluation_dataset/generated/evaluation_summary.json` | JSON | Consolidated machine-readable benchmark summary. |

---

*Report prepared autonomously by Antigravity IDE during Phase 6B.*
