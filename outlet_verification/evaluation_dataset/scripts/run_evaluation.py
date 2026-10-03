import argparse
import csv
import json
import logging
import math
import sys
import time
from collections import defaultdict
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
from app.services.retrieval import get_candidate_retrieval_service
from app.services.verification import (
    get_candidate_scorer,
    get_decision_engine,
    get_verification_service,
)

logger = logging.getLogger("outlet_verification.evaluation.runner")
logging.basicConfig(level=logging.WARNING)

DATASET_ROOT = Path(__file__).resolve().parent.parent
SUBMISSIONS_DIR = DATASET_ROOT / "submissions"
SUB_IMAGES_DIR = SUBMISSIONS_DIR / "images"
BASE_OUTLETS_DIR = DATASET_ROOT / "base_outlets"
GENERATED_DIR = DATASET_ROOT / "generated"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

DECISION_CLASSES = ["DUPLICATE", "GENUINE", "NEEDS_REVIEW"]


def compute_percentiles(values: list[float], percentiles: list[float]) -> dict[str, float]:
    """Calculates requested percentiles for a list of values."""
    if not values:
        return {f"p{int(p)}": 0.0 for p in percentiles}
    sorted_v = sorted(values)
    n = len(sorted_v)
    res = {}
    for p in percentiles:
        idx = int(math.ceil((p / 100.0) * n)) - 1
        idx = max(0, min(n - 1, idx))
        res[f"p{int(p)}"] = round(sorted_v[idx], 2)
    return res


def classify_failure_source(
    scenario: str,
    exp_decision: str,
    pred_decision: str,
    candidate_count: int,
    name_sim: float | None,
    image_sim: float | None,
    distance_m: float | None,
    confidence: float,
    coverage: float,
    margin: float | None,
    reason_codes: list[str],
) -> str:
    """
    Deterministically attributes the root cause of an evaluation mismatch.
    """
    if candidate_count == 0:
        return "CANDIDATE_RETRIEVAL_ERROR"
    if "CONFLICTING_NAME_IMAGE" in reason_codes or "AMBIGUOUS_TOP_CANDIDATES" in reason_codes:
        if margin is not None and margin < 0.10:
            return "AMBIGUITY_ERROR"
        return "FUSION_ERROR"
    if image_sim is None or distance_m is None:
        return "MISSING_SIGNAL"
    if exp_decision == "DUPLICATE" and pred_decision == "NEEDS_REVIEW":
        if 0.35 < confidence < 0.80:
            return "THRESHOLD_ERROR"
    if exp_decision == "GENUINE" and pred_decision == "NEEDS_REVIEW":
        if confidence > 0.35:
            return "THRESHOLD_ERROR"
    if distance_m is not None and distance_m > 1000.0:
        return "GPS_ERROR"
    if image_sim is not None and image_sim < 0.70:
        return "IMAGE_ERROR"
    if name_sim is not None and name_sim < 0.70:
        return "NAME_ERROR"
    return "OTHER"


def run_complete_evaluation(limit: int | None = None):
    print("========================================================")
    print("  PHASE 6B: OBJECTIVE BENCHMARK EVALUATION OF CURRENT PIPELINE")
    print("========================================================\n")

    sub_meta_file = SUBMISSIONS_DIR / "metadata.json"
    if not sub_meta_file.exists():
        print(f"Error: Submissions metadata not found at {sub_meta_file}")
        sys.exit(1)

    with open(sub_meta_file, "r", encoding="utf-8") as f:
        submissions = json.load(f)

    # Load base outlets to map db_uuid back to SYNTH_OUTLET_xxx
    base_meta_file = BASE_OUTLETS_DIR / "metadata.json"
    uuid_to_synth_id = {}
    if base_meta_file.exists():
        with open(base_meta_file, "r", encoding="utf-8") as f:
            base_list = json.load(f)
            for b in base_list:
                uuid_to_synth_id[str(b["db_uuid"]).lower()] = b["outlet_id"]

    if limit is not None:
        submissions = submissions[:limit]
        print(f"Limiting benchmark run to first {limit} submissions.")

    # Initialize production services
    retrieval_service = get_candidate_retrieval_service()
    scorer = get_candidate_scorer()
    decision_engine = get_decision_engine()
    service = get_verification_service()
    db = SessionLocal()

    print("Warming up embedding models...", flush=True)
    _ = generate_name_embedding("warmup model initialization")
    _ = generate_image_embedding(Image.new("RGB", (64, 64)))
    print("Embedding models warmed up successfully.\n", flush=True)

    total = len(submissions)
    print(f"Executing real production pipeline for {total} submissions...", flush=True)
    print("Pipeline: Query Embeddings -> HNSW/Spatial Retrieval -> Evidence Fusion -> Deterministic Decision\n", flush=True)

    eval_rows = []
    latencies_total = []
    latencies_emb = []
    latencies_ret = []
    latencies_dec = []

    overall_start = time.perf_counter()

    try:
        for idx, sub in enumerate(submissions, 1):
            sub_id = sub["submission_id"]
            scenario = sub["scenario"]
            name = sub["name"]
            lat = sub["latitude"]
            lng = sub["longitude"]
            img_file = sub["image_file"]

            exp_decision = sub["expected_decision"]
            exp_match_id = sub.get("expected_match_outlet_id")

            # 1. Load image if provided
            image_obj = None
            if img_file:
                img_path = SUB_IMAGES_DIR / img_file
                if img_path.exists():
                    image_obj = Image.open(img_path).convert("RGB")

            # 2. Benchmark with granular component timing
            # Embedding generation timing
            t_emb_start = time.perf_counter()
            _ = generate_name_embedding(name) if name else None
            _ = generate_image_embedding(image_obj) if image_obj is not None else None
            t_emb_ms = (time.perf_counter() - t_emb_start) * 1000

            # Retrieval timing (candidate retrieval from DB)
            t_ret_start = time.perf_counter()
            ret_result = retrieval_service.retrieve_candidates(
                db=db,
                name=name,
                latitude=lat,
                longitude=lng,
                image=image_obj,
            )
            raw_ret_ms = (time.perf_counter() - t_ret_start) * 1000
            # Retrieve_candidates internally generated embeddings, so DB time is:
            t_db_ms = max(0.0, raw_ret_ms - t_emb_ms)

            # Decision timing (Scoring + Safeguards + Thresholds)
            t_dec_start = time.perf_counter()
            scored_candidates = [scorer.score_candidate(c) for c in ret_result.candidates]
            resp = decision_engine.evaluate(scored_candidates)
            t_dec_ms = (time.perf_counter() - t_dec_start) * 1000

            # Production pipeline latency
            t_total_ms = t_emb_ms + t_db_ms + t_dec_ms

            latencies_total.append(t_total_ms)
            latencies_emb.append(t_emb_ms)
            latencies_ret.append(t_db_ms)
            latencies_dec.append(t_dec_ms)

            # 3. Extract evaluation data
            pred_decision = resp.decision.value

            pred_match_id = None
            if resp.matched_outlet and resp.matched_outlet.outlet_id:
                raw_id = str(resp.matched_outlet.outlet_id).lower()
                pred_match_id = uuid_to_synth_id.get(raw_id, raw_id)

            # Correct decision & match flags
            is_decision_correct = (pred_decision == exp_decision)
            
            is_match_correct = False
            if exp_decision == "DUPLICATE":
                is_match_correct = (pred_match_id == exp_match_id) if exp_match_id else False
            elif exp_decision in {"GENUINE", "NEEDS_REVIEW"}:
                # For genuine / review cases, correct match is satisfied if no duplicate false positive was assigned
                is_match_correct = (pred_decision != "DUPLICATE")

            overall_correct = is_decision_correct and (is_match_correct if exp_decision == "DUPLICATE" else True)

            # Runner up candidate confidence
            runner_up_conf = None
            if len(scored_candidates) > 1:
                ranked = sorted(scored_candidates, key=lambda c: c.duplicate_confidence, reverse=True)
                runner_up_conf = ranked[1].duplicate_confidence

            ev = resp.evidence
            name_sim = round(ev.name_similarity, 4) if ev and ev.name_similarity is not None else None
            image_sim = round(ev.image_similarity, 4) if ev and ev.image_similarity is not None else None
            dist_m = round(ev.distance_meters, 2) if ev and ev.distance_meters is not None else None

            failure_src = ""
            if not overall_correct:
                failure_src = classify_failure_source(
                    scenario=scenario,
                    exp_decision=exp_decision,
                    pred_decision=pred_decision,
                    candidate_count=resp.candidates_evaluated,
                    name_sim=name_sim,
                    image_sim=image_sim,
                    distance_m=dist_m,
                    confidence=resp.duplicate_confidence,
                    coverage=resp.evidence_coverage,
                    margin=resp.candidate_margin,
                    reason_codes=resp.reason_codes,
                )

            row = {
                "submission_id": sub_id,
                "scenario": scenario,
                "expected_decision": exp_decision,
                "predicted_decision": pred_decision,
                "expected_match_outlet_id": exp_match_id or "",
                "predicted_match_outlet_id": pred_match_id or "",
                "duplicate_confidence": round(resp.duplicate_confidence, 4),
                "evidence_coverage": round(resp.evidence_coverage, 4),
                "evidence_agreement": round(resp.evidence_agreement, 4),
                "name_similarity": name_sim if name_sim is not None else "",
                "image_similarity": image_sim if image_sim is not None else "",
                "distance_meters": dist_m if dist_m is not None else "",
                "candidate_count": resp.candidates_evaluated,
                "top_candidate_confidence": round(resp.duplicate_confidence, 4) if resp.candidates_evaluated > 0 else "",
                "runner_up_confidence": round(runner_up_conf, 4) if runner_up_conf is not None else "",
                "candidate_margin": round(resp.candidate_margin, 4) if resp.candidate_margin is not None else "",
                "reason_codes": ";".join(resp.reason_codes),
                "latency_ms": round(t_total_ms, 2),
                "embedding_latency_ms": round(t_emb_ms, 2),
                "retrieval_latency_ms": round(t_db_ms, 2),
                "decision_latency_ms": round(t_dec_ms, 2),
                "correct_decision": "YES" if is_decision_correct else "NO",
                "correct_match": "YES" if is_match_correct else "NO",
                "correct": "YES" if overall_correct else "NO",
                "likely_failure_source": failure_src,
            }
            eval_rows.append(row)

            if idx % 10 == 0 or idx == total:
                print(f"   Processed {idx:>3}/{total} submissions | Latest latency: {t_total_ms:.1f}ms", flush=True)

    finally:
        db.close()

    total_time_sec = round(time.perf_counter() - overall_start, 2)
    print(f"\nExecution finished in {total_time_sec}s.\n")

    # ==========================================================================
    # 4. SAVE EVALUATION RESULTS CSV
    # ==========================================================================
    results_csv_path = GENERATED_DIR / "evaluation_results.csv"
    fieldnames = list(eval_rows[0].keys())
    with open(results_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(eval_rows)
    print(f"Saved detailed results to {results_csv_path}")

    # ==========================================================================
    # 5. COMPUTE 3x3 CONFUSION MATRIX
    # ==========================================================================
    # matrix[expected][predicted]
    conf_matrix = {e: {p: 0 for p in DECISION_CLASSES} for e in DECISION_CLASSES}
    for r in eval_rows:
        conf_matrix[r["expected_decision"]][r["predicted_decision"]] += 1

    cm_csv_path = GENERATED_DIR / "confusion_matrix.csv"
    with open(cm_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Expected / Predicted", "DUPLICATE", "GENUINE", "NEEDS_REVIEW", "Total", "Row_Accuracy_%"])
        for exp in DECISION_CLASSES:
            row_vals = [conf_matrix[exp][p] for p in DECISION_CLASSES]
            row_sum = sum(row_vals)
            acc_pct = (conf_matrix[exp][exp] / row_sum * 100) if row_sum > 0 else 0.0
            writer.writerow([exp, *row_vals, row_sum, round(acc_pct, 2)])
    print(f"Saved confusion matrix to {cm_csv_path}")

    # ==========================================================================
    # 6. PER-CLASS METRICS
    # ==========================================================================
    per_class_metrics = {}
    for c in DECISION_CLASSES:
        tp = conf_matrix[c][c]
        fp = sum(conf_matrix[other][c] for other in DECISION_CLASSES if other != c)
        fn = sum(conf_matrix[c][other] for other in DECISION_CLASSES if other != c)
        support = sum(conf_matrix[c].values())

        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_class_metrics[c] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support": support,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
        }

    overall_accuracy = (
        sum(conf_matrix[c][c] for c in DECISION_CLASSES) / total
    ) if total > 0 else 0.0

    # ==========================================================================
    # 7. BINARY DUPLICATE METRICS
    # ==========================================================================
    bin_tp = conf_matrix["DUPLICATE"]["DUPLICATE"]
    bin_fp = conf_matrix["GENUINE"]["DUPLICATE"] + conf_matrix["NEEDS_REVIEW"]["DUPLICATE"]
    bin_fn = conf_matrix["DUPLICATE"]["GENUINE"] + conf_matrix["DUPLICATE"]["NEEDS_REVIEW"]
    bin_tn = (
        conf_matrix["GENUINE"]["GENUINE"]
        + conf_matrix["GENUINE"]["NEEDS_REVIEW"]
        + conf_matrix["NEEDS_REVIEW"]["GENUINE"]
        + conf_matrix["NEEDS_REVIEW"]["NEEDS_REVIEW"]
    )

    bin_prec = (bin_tp / (bin_tp + bin_fp)) if (bin_tp + bin_fp) > 0 else 0.0
    bin_rec = (bin_tp / (bin_tp + bin_fn)) if (bin_tp + bin_fn) > 0 else 0.0
    bin_f1 = (2 * bin_prec * bin_rec / (bin_prec + bin_rec)) if (bin_prec + bin_rec) > 0 else 0.0
    bin_spec = (bin_tn / (bin_tn + bin_fp)) if (bin_tn + bin_fp) > 0 else 0.0
    bin_fpr = (bin_fp / (bin_fp + bin_tn)) if (bin_fp + bin_tn) > 0 else 0.0
    bin_fnr = (bin_fn / (bin_fn + bin_tp)) if (bin_fn + bin_tp) > 0 else 0.0

    binary_metrics = {
        "TP": bin_tp,
        "FP": bin_fp,
        "TN": bin_tn,
        "FN": bin_fn,
        "precision": round(bin_prec, 4),
        "recall": round(bin_rec, 4),
        "f1_score": round(bin_f1, 4),
        "specificity": round(bin_spec, 4),
        "false_positive_rate": round(bin_fpr, 4),
        "false_negative_rate": round(bin_fnr, 4),
    }

    # ==========================================================================
    # 8. MATCH IDENTIFICATION ACCURACY (DUPLICATE CASES)
    # ==========================================================================
    dup_rows = [r for r in eval_rows if r["expected_decision"] == "DUPLICATE"]
    dup_total = len(dup_rows)
    dup_decision_correct = sum(1 for r in dup_rows if r["predicted_decision"] == "DUPLICATE")
    dup_match_correct_all = sum(1 for r in dup_rows if r["predicted_match_outlet_id"] == r["expected_match_outlet_id"])
    
    detected_dup_rows = [r for r in dup_rows if r["predicted_decision"] == "DUPLICATE"]
    dup_match_correct_detected = sum(1 for r in detected_dup_rows if r["predicted_match_outlet_id"] == r["expected_match_outlet_id"])

    match_metrics = {
        "expected_duplicate_samples": dup_total,
        "decision_accuracy": round(dup_decision_correct / dup_total, 4) if dup_total > 0 else 0.0,
        "match_accuracy_among_all_duplicates": round(dup_match_correct_all / dup_total, 4) if dup_total > 0 else 0.0,
        "match_accuracy_among_detected_duplicates": (
            round(dup_match_correct_detected / len(detected_dup_rows), 4)
            if detected_dup_rows
            else 0.0
        ),
    }

    # ==========================================================================
    # 9. SCENARIO-BY-SCENARIO ANALYSIS
    # ==========================================================================
    scenario_groups = defaultdict(list)
    for r in eval_rows:
        scenario_groups[r["scenario"]].append(r)

    scenario_metrics_rows = []
    for sc_name, rows in sorted(scenario_groups.items()):
        sc_total = len(rows)
        sc_correct = sum(1 for r in rows if r["correct"] == "YES")
        sc_acc = (sc_correct / sc_total) if sc_total > 0 else 0.0

        avg_conf = sum(float(r["duplicate_confidence"]) for r in rows) / sc_total
        avg_cov = sum(float(r["evidence_coverage"]) for r in rows) / sc_total
        avg_agr = sum(float(r["evidence_agreement"]) for r in rows) / sc_total
        avg_lat = sum(float(r["latency_ms"]) for r in rows) / sc_total

        exp_dist = {c: sum(1 for r in rows if r["expected_decision"] == c) for c in DECISION_CLASSES}
        pred_dist = {c: sum(1 for r in rows if r["predicted_decision"] == c) for c in DECISION_CLASSES}

        # Identify most common failure mode
        failures = [r["likely_failure_source"] for r in rows if r["correct"] == "NO"]
        common_fail = max(set(failures), key=failures.count) if failures else "NONE"

        sc_row = {
            "scenario": sc_name,
            "sample_count": sc_total,
            "correct_count": sc_correct,
            "accuracy": round(sc_acc, 4),
            "expected_distribution": f"D:{exp_dist['DUPLICATE']}/G:{exp_dist['GENUINE']}/R:{exp_dist['NEEDS_REVIEW']}",
            "predicted_distribution": f"D:{pred_dist['DUPLICATE']}/G:{pred_dist['GENUINE']}/R:{pred_dist['NEEDS_REVIEW']}",
            "avg_confidence": round(avg_conf, 4),
            "avg_coverage": round(avg_cov, 4),
            "avg_agreement": round(avg_agr, 4),
            "avg_latency_ms": round(avg_lat, 2),
            "primary_failure_source": common_fail,
        }
        scenario_metrics_rows.append(sc_row)

    scenario_csv_path = GENERATED_DIR / "scenario_metrics.csv"
    with open(scenario_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(scenario_metrics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(scenario_metrics_rows)
    print(f"Saved scenario metrics to {scenario_csv_path}")

    # ==========================================================================
    # 10. ERROR ANALYSIS & SIGNAL ANALYSIS CSVs
    # ==========================================================================
    error_rows = [r for r in eval_rows if r["correct"] == "NO"]
    error_csv_path = GENERATED_DIR / "error_analysis.csv"
    error_fields = [
        "submission_id", "scenario", "expected_decision", "predicted_decision",
        "expected_match_outlet_id", "predicted_match_outlet_id", "name_similarity",
        "image_similarity", "distance_meters", "duplicate_confidence", "evidence_coverage",
        "evidence_agreement", "candidate_margin", "reason_codes", "likely_failure_source"
    ]
    with open(error_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=error_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(error_rows)
    print(f"Saved error analysis ({len(error_rows)} errors) to {error_csv_path}")

    # Signal Analysis CSV (all rows with evidence metrics)
    signal_csv_path = GENERATED_DIR / "signal_analysis.csv"
    signal_fields = [
        "submission_id", "scenario", "expected_decision", "predicted_decision",
        "duplicate_confidence", "evidence_coverage", "evidence_agreement",
        "name_similarity", "image_similarity", "distance_meters",
        "candidate_count", "candidate_margin", "reason_codes", "correct"
    ]
    with open(signal_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=signal_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(eval_rows)
    print(f"Saved signal analysis to {signal_csv_path}")

    # ==========================================================================
    # 11. PERFORMANCE & LATENCY METRICS
    # ==========================================================================
    pct_total = compute_percentiles(latencies_total, [50, 90, 95, 99])
    perf_metrics = {
        "total_evaluation_time_seconds": total_time_sec,
        "submissions_evaluated": total,
        "mean_latency_ms": round(sum(latencies_total) / total, 2) if total else 0.0,
        "p50_latency_ms": pct_total["p50"],
        "p90_latency_ms": pct_total["p90"],
        "p95_latency_ms": pct_total["p95"],
        "max_latency_ms": round(max(latencies_total), 2) if latencies_total else 0.0,
        "latency_breakdown_averages_ms": {
            "embedding_generation": round(sum(latencies_emb) / total, 2) if total else 0.0,
            "candidate_retrieval_and_db": round(sum(latencies_ret) / total, 2) if total else 0.0,
            "evidence_scoring_and_decision": round(sum(latencies_dec) / total, 2) if total else 0.0,
        },
    }

    perf_json_path = GENERATED_DIR / "performance_metrics.json"
    with open(perf_json_path, "w", encoding="utf-8") as f:
        json.dump(perf_metrics, f, indent=2)
    print(f"Saved performance metrics to {perf_json_path}")

    # ==========================================================================
    # 12. EVALUATION SUMMARY JSON
    # ==========================================================================
    summary = {
        "overall_accuracy": round(overall_accuracy, 4),
        "total_submissions": total,
        "correct_decisions": sum(1 for r in eval_rows if r["correct_decision"] == "YES"),
        "total_errors": len(error_rows),
        "confusion_matrix": conf_matrix,
        "per_class_metrics": per_class_metrics,
        "binary_duplicate_metrics": binary_metrics,
        "match_identification_metrics": match_metrics,
        "performance_metrics": perf_metrics,
    }

    summary_json_path = GENERATED_DIR / "evaluation_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved evaluation summary to {summary_json_path}")

    # ==========================================================================
    # 13. PRINT HIGH-LEVEL CONSOLE REPORT
    # ==========================================================================
    print("\n========================================================")
    print("      PHASE 6B BENCHMARK EVALUATION SUMMARY")
    print("========================================================")
    print(f"  Overall Accuracy:            {overall_accuracy * 100:.2f}% ({summary['correct_decisions']}/{total})")
    print()
    print("  3x3 Confusion Matrix (Rows: Expected, Cols: Predicted):")
    print(f"    {'':<14} | {'DUPLICATE':<10} | {'GENUINE':<10} | {'NEEDS_REVIEW':<12} | {'Row Acc %':<10}")
    print("    " + "-" * 65)
    for c in DECISION_CLASSES:
        vals = [conf_matrix[c][p] for p in DECISION_CLASSES]
        row_sum = sum(vals)
        r_acc = (conf_matrix[c][c] / row_sum * 100) if row_sum > 0 else 0.0
        print(f"    {c:<14} | {vals[0]:<10} | {vals[1]:<10} | {vals[2]:<12} | {r_acc:.1f}%")
    print()
    print("  Per-Class Metrics:")
    for c, m in per_class_metrics.items():
        print(f"    {c:<14}: Precision={m['precision']:.3f} | Recall={m['recall']:.3f} | F1={m['f1_score']:.3f} | Support={m['support']}")
    print()
    print("  Binary Duplicate Metrics (Positive: DUPLICATE):")
    print(f"    Precision:  {binary_metrics['precision']:.3f} | Recall: {binary_metrics['recall']:.3f} | F1: {binary_metrics['f1_score']:.3f}")
    print(f"    Specificity: {binary_metrics['specificity']:.3f} | FPR: {binary_metrics['false_positive_rate']:.3f} | FNR: {binary_metrics['false_negative_rate']:.3f}")
    print()
    print("  Duplicate Match ID Accuracy:")
    print(f"    Decision Accuracy:               {match_metrics['decision_accuracy'] * 100:.1f}%")
    print(f"    Match Accuracy (All Duplicates): {match_metrics['match_accuracy_among_all_duplicates'] * 100:.1f}%")
    print(f"    Match Accuracy (Detected Only):  {match_metrics['match_accuracy_among_detected_duplicates'] * 100:.1f}%")
    print()
    print("  Latency:")
    print(f"    Mean: {perf_metrics['mean_latency_ms']:.1f}ms | p50: {perf_metrics['p50_latency_ms']:.1f}ms | p95: {perf_metrics['p95_latency_ms']:.1f}ms")
    print("========================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete Phase 6B synthetic benchmark evaluation")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for dry-run")
    args = parser.parse_args()
    run_complete_evaluation(limit=args.limit)
