from __future__ import annotations

import csv
import json
import math
import statistics
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
RUN_ID = "run_16_batch20_teacher_metrics_ingestion_and_combined80_ranking"
RUN_NAME = "batch20 teacher metrics ingestion and combined80 ranking"

BATCH20_METRICS = ROOT / "outputs" / "stage3_run_14_batch20_odb_teacher_validation" / "run14_batch20_teacher_metrics.csv"
BATCH20_EXTRACTION = ROOT / "outputs" / "stage3_run_14_batch20_odb_teacher_validation" / "run14_batch20_odb_extraction_summary.csv"
BATCH20_SOLVER = ROOT / "outputs" / "stage3_run_14_batch20_odb_teacher_validation" / "run14_batch20_solver_completion_audit.csv"
BATCH20_SUMMARY = ROOT / "outputs" / "stage3_run_14_batch20_odb_teacher_validation" / "run14_batch20_odb_teacher_validation_summary.json"
BATCH20_REPORT = ROOT / "outputs" / "stage3_run_14_batch20_odb_teacher_validation" / "run14_batch20_odb_teacher_validation_report.md"
RUN13_HANDOFF = ROOT / "outputs" / "stage3_run_13_batch20_surrogate_screened_teacher_handoff" / "stage3_run13_batch20_candidate_orders.csv"
RUN12_BATCH20 = ROOT / "outputs" / "stage3_run_12_offline_surrogate_screened_candidate_generation" / "run12_recommended_future_teacher_batch20.csv"
RUN12_SCORED = ROOT / "outputs" / "stage3_run_12_offline_surrogate_screened_candidate_generation" / "run12_candidate_pool_scored.csv"
PROBE60_RANKED = ROOT / "outputs" / "stage3_run_09_variable_n_probe60_teacher_ranking_analysis" / "probe60_teacher_ranked_canonical.csv"
PROBE60_LABELS = ROOT / "outputs" / "stage3_run_08_probe60_odb_teacher_validation" / "probe60_odb_teacher_labels.csv"
RUN10_REWARD = ROOT / "outputs" / "stage3_run_10_variable_n_normalized_reward_surrogate_dataset" / "probe60_variable_n_reward_dataset.csv"
RUN11_BEST = ROOT / "outputs" / "stage3_run_11_variable_n_surrogate_reward_model_validation" / "best_surrogate_configurations.csv"
RUN12_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_12_offline_surrogate_screened_candidate_generation" / "RUN_12_OFFLINE_SURROGATE_SCREENED_CANDIDATE_GENERATION_REPORT.md"
RUN13_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_13_batch20_surrogate_screened_teacher_handoff" / "RUN_13_BATCH20_SURROGATE_SCREENED_TEACHER_HANDOFF_REPORT.md"

OUTPUT_DIR = ROOT / "outputs" / "stage3_run_16_batch20_teacher_metrics_ingestion_and_combined80_ranking"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = ROOT / "docs" / "stage3" / "runs" / RUN_ID
REPORT_PATH = REPORT_DIR / "RUN_16_BATCH20_TEACHER_METRICS_INGESTION_AND_COMBINED80_RANKING_REPORT.md"
MANIFEST_PATH = ROOT / "artifacts" / "manifests" / "stage3_run_16_manifest.json"
RUN_INDEX_PATH = ROOT / "docs" / "stage3" / "STAGE3_RUN_INDEX.md"

EXPECTED_N = [12, 16, 24, 40]
REPAIRED_CASE = "S3B20_N40_B02_diversity_top"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_int(value: Any) -> int:
    text = str(value).strip()
    if text.upper().startswith("N"):
        text = text[1:]
    return int(text)


def parse_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def safe_divide(num: float, den: float, default: float = 0.0) -> float:
    return num / den if den else default


def mean(values: list[float], default: float = math.nan) -> float:
    clean = [x for x in values if math.isfinite(x)]
    return statistics.fmean(clean) if clean else default


def rank_ascending(rows: list[dict[str, Any]], metric: str, rank_col: str) -> None:
    sorted_rows = sorted(rows, key=lambda row: row[metric])
    i = 0
    while i < len(sorted_rows):
        j = i + 1
        while j < len(sorted_rows) and sorted_rows[j][metric] == sorted_rows[i][metric]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            sorted_rows[k][rank_col] = avg
        i = j


def add_minmax_cost(rows: list[dict[str, Any]], metric: str, col: str) -> None:
    values = [row[metric] for row in rows]
    mn, mx = min(values), max(values)
    span = mx - mn
    for row in rows:
        row[col] = safe_divide(row[metric] - mn, span, default=0.0)


def add_rank_score(rows: list[dict[str, Any]], rank_col: str, score_col: str) -> None:
    denom = max(1, len(rows) - 1)
    for row in rows:
        row[score_col] = 1.0 - safe_divide(row[rank_col] - 1.0, denom)


def pareto_flags(rows: list[dict[str, Any]], metrics: list[str], flag_col: str) -> None:
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            no_worse = all(other[m] <= row[m] for m in metrics)
            strictly_better = any(other[m] < row[m] for m in metrics)
            if no_worse and strictly_better:
                dominated = True
                break
        row[flag_col] = not dominated


def spearman(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return math.nan
    def ranks(vals: list[float]) -> list[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        out = [0.0] * len(vals)
        for rank, idx in enumerate(order, start=1):
            out[idx] = rank
        return out
    rx = ranks([p[0] for p in pairs])
    ry = ranks([p[1] for p in pairs])
    mx, my = mean(rx), mean(ry)
    den = math.sqrt(sum((x - mx) ** 2 for x in rx) * sum((y - my) ** 2 for y in ry))
    return safe_divide(sum((x - mx) * (y - my) for x, y in zip(rx, ry)), den, default=math.nan)


def validate_batch20(rows: list[dict[str, str]]) -> dict[str, Any]:
    errors: list[str] = []
    counts = Counter()
    names_by_n: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        try:
            n = parse_int(row.get("n"))
        except Exception:
            errors.append(f"Invalid N: {row.get('n')}")
            continue
        counts[n] += 1
        name = row.get("handoff_strategy_name") or row.get("strategy_name")
        if not name:
            errors.append("Missing strategy_name/handoff_strategy_name")
        if name in names_by_n[n]:
            errors.append(f"Duplicate strategy within N{n}: {name}")
        names_by_n[n].add(name)
        status = row.get("teacher_validation_status", "")
        if "PASS" not in status.upper():
            errors.append(f"{name}: non-pass teacher_validation_status={status}")
        for col in ["u2_range", "peeq_max", "mises_max"]:
            if not math.isfinite(parse_float(row.get(col))):
                errors.append(f"{name}: missing {col}")
        surface = row.get("surface_t_proxy_max_tensile_pa") or row.get("surface_t_proxy") or row.get("surface_t_proxy_max_tensile_mpa")
        if not math.isfinite(parse_float(surface)):
            errors.append(f"{name}: missing surface_t_proxy")
    if len(rows) != 20:
        errors.append(f"Expected 20 rows, found {len(rows)}")
    if sorted(counts) != EXPECTED_N:
        errors.append(f"Expected N values {EXPECTED_N}, found {sorted(counts)}")
    for n in EXPECTED_N:
        if counts[n] != 5:
            errors.append(f"Expected 5 rows for N{n}, found {counts[n]}")
    repaired = next((r for r in rows if (r.get("handoff_strategy_name") or r.get("strategy_name")) == REPAIRED_CASE), None)
    if repaired is None:
        errors.append(f"{REPAIRED_CASE} missing")
    elif "PASS" not in repaired.get("teacher_validation_status", "").upper():
        errors.append(f"{REPAIRED_CASE} is not pass")
    summary = json.loads(BATCH20_SUMMARY.read_text(encoding="utf-8")) if BATCH20_SUMMARY.exists() else {}
    solver = summary.get("audit_summary", {})
    extraction = summary.get("extraction_summary", {})
    if solver.get("total_complete") != 20 or extraction.get("total_pass") != 20:
        errors.append("Summary does not confirm 20/20 completion and extraction")
    if solver.get("total_lck_present", 0) != 0:
        errors.append("Summary reports nonzero lck count")
    verdict = "PASS_RUN16_BATCH20_TEACHER_METRICS_20_OF_20_READY" if not errors else "FAIL_RUN16_BATCH20_INPUT_INVALID"
    return {
        "verdict": verdict,
        "errors": errors,
        "row_count": len(rows),
        "per_n_counts": dict(sorted(counts.items())),
        "solver_completion_verdict": solver.get("verdict"),
        "extraction_verdict": extraction.get("verdict"),
        "total_lck_present": solver.get("total_lck_present"),
        "repaired_case_present_and_pass": repaired is not None and "PASS" in repaired.get("teacher_validation_status", "").upper(),
    }


def canonicalize_batch20(metrics_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    handoff = {row["handoff_strategy_name"]: row for row in read_csv(RUN13_HANDOFF)}
    scored_by_id = {row["candidate_id"]: row for row in read_csv(RUN12_SCORED)}
    canonical: list[dict[str, Any]] = []
    for row in metrics_rows:
        name = row.get("handoff_strategy_name") or row.get("strategy_name")
        meta = handoff.get(name, {})
        scored = scored_by_id.get(meta.get("original_run12_candidate_id", ""), {})
        surface_pa = parse_float(row.get("surface_t_proxy_max_tensile_pa"))
        if not math.isfinite(surface_pa):
            surface_pa = parse_float(row.get("surface_t_proxy_max_tensile_mpa")) * 1_000_000.0
        canonical.append(
            {
                "dataset_source": "batch20_run14",
                "n": parse_int(row["n"]),
                "strategy_name": name,
                "handoff_strategy_name": name,
                "job_name": row.get("job_name", ""),
                "candidate_id": meta.get("original_run12_candidate_id", ""),
                "original_run12_candidate_id": meta.get("original_run12_candidate_id", ""),
                "candidate_family": meta.get("candidate_family", scored.get("candidate_family", "")),
                "selection_bucket": meta.get("selection_bucket", scored.get("selection_bucket", "")),
                "predicted_reward_mean_all": parse_float(meta.get("predicted_reward_mean_all", scored.get("pred_reward_mean_all"))),
                "predicted_rank_within_n": parse_float(meta.get("predicted_rank_within_n", scored.get("pred_rank_within_n"))),
                "predicted_uncertainty_std": parse_float(meta.get("predicted_uncertainty_std", scored.get("pred_uncertainty_std"))),
                "novelty_distance_to_teacher": parse_float(meta.get("novelty_distance_to_teacher", scored.get("novelty_distance_to_nearest_existing"))),
                "nearest_existing_teacher_strategy": meta.get("nearest_existing_teacher_strategy", scored.get("nearest_existing_strategy", "")),
                "order_json": meta.get("order_json", scored.get("order_json", "")),
                "u2_range": parse_float(row["u2_range"]),
                "peeq_max": parse_float(row["peeq_max"]),
                "surface_t_proxy": surface_pa,
                "surface_t_proxy_mpa": safe_divide(surface_pa, 1_000_000.0),
                "mises_max": parse_float(row["mises_max"]),
                "teacher_validation_status": row.get("teacher_validation_status", ""),
                "final_step": row.get("final_step_name", ""),
                "final_frame_time": parse_float(row.get("final_frame_time")),
                "extracted_fields": row.get("extracted_field_names", ""),
            }
        )
    return canonical


def add_batch20_ranking(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for n in EXPECTED_N:
        group = [row for row in rows if row["n"] == n]
        for metric, prefix in [("u2_range", "u2"), ("peeq_max", "peeq"), ("surface_t_proxy", "surfaceT"), ("mises_max", "mises")]:
            rank_ascending(group, metric, f"{prefix}_rank_batch20_within_n")
            add_minmax_cost(group, metric, f"{prefix}_cost_minmax_batch20_within_n")
            add_rank_score(group, f"{prefix}_rank_batch20_within_n", f"{prefix}_score_batch20_rank")
        for row in group:
            row["reward_batch20_u2_primary"] = (
                0.65 * row["u2_score_batch20_rank"]
                + 0.20 * row["peeq_score_batch20_rank"]
                + 0.10 * row["surfaceT_score_batch20_rank"]
                + 0.05 * row["mises_score_batch20_rank"]
            )
        rank_ascending([{**row, "neg_reward": -row["reward_batch20_u2_primary"]} for row in []], "neg_reward", "unused")
        sorted_reward = sorted(group, key=lambda row: row["reward_batch20_u2_primary"], reverse=True)
        for idx, row in enumerate(sorted_reward, start=1):
            row["batch20_constrained_rank_within_n"] = idx
        pareto_flags(group, ["u2_range", "peeq_max"], "batch20_pareto_flag_u2_peeq")
        pareto_flags(group, ["u2_range", "peeq_max", "surface_t_proxy", "mises_max"], "batch20_pareto_flag_u2_peeq_surfaceT_mises")
    return rows


def surrogate_audit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    audit: list[dict[str, Any]] = []
    for n in EXPECTED_N:
        group = [row for row in rows if row["n"] == n]
        for row in group:
            pred = row["predicted_reward_mean_all"]
            realized = row["reward_batch20_u2_primary"]
            audit.append(
                {
                    "n": n,
                    "strategy_name": row["strategy_name"],
                    "predicted_reward_mean_all": pred,
                    "realized_reward_batch20_u2_primary": realized,
                    "prediction_error": realized - pred if math.isfinite(pred) else math.nan,
                    "absolute_error": abs(realized - pred) if math.isfinite(pred) else math.nan,
                    "predicted_rank_within_n": row["predicted_rank_within_n"],
                    "realized_rank_within_n": row["batch20_constrained_rank_within_n"],
                    "rank_error": abs(row["batch20_constrained_rank_within_n"] - row["predicted_rank_within_n"]) if math.isfinite(row["predicted_rank_within_n"]) else math.nan,
                    "top_predicted_within_n": row is max(group, key=lambda r: r["predicted_reward_mean_all"]),
                    "top_realized_within_n": row is max(group, key=lambda r: r["reward_batch20_u2_primary"]),
                    "spearman_pred_vs_realized_within_n": spearman([r["predicted_reward_mean_all"] for r in group], [r["reward_batch20_u2_primary"] for r in group]),
                    "overall_spearman_pred_vs_realized": spearman([r["predicted_reward_mean_all"] for r in rows], [r["reward_batch20_u2_primary"] for r in rows]),
                }
            )
    return audit


def probe60_rows() -> list[dict[str, Any]]:
    run10 = {row["strategy_name"]: row for row in read_csv(RUN10_REWARD)} if RUN10_REWARD.exists() else {}
    rows: list[dict[str, Any]] = []
    for row in read_csv(PROBE60_RANKED):
        name = row["strategy_name"]
        r10 = run10.get(name, {})
        rows.append(
            {
                "dataset_source": "probe60_run08",
                "n": parse_int(row["n"]),
                "strategy_name": name,
                "handoff_strategy_name": "",
                "job_name": row.get("job_name_canonical", row.get("raw_job_name", "")),
                "candidate_id": row.get("raw_strategy_id", ""),
                "candidate_family": row.get("strategy_family", ""),
                "selection_bucket": "",
                "predicted_reward_mean_all": math.nan,
                "predicted_rank_within_n": math.nan,
                "predicted_uncertainty_std": math.nan,
                "novelty_distance_to_teacher": math.nan,
                "nearest_existing_teacher_strategy": "",
                "order_json": row.get("order_json", row.get("raw_scan_order", "")),
                "u2_range": parse_float(row["u2_range_canonical"]),
                "peeq_max": parse_float(row["peeq_max_canonical"]),
                "surface_t_proxy": parse_float(row["surfaceT_proxy_canonical"]),
                "surface_t_proxy_mpa": safe_divide(parse_float(row["surfaceT_proxy_canonical"]), 1_000_000.0),
                "mises_max": parse_float(row.get("raw_mises_max")),
                "teacher_validation_status": row.get("teacher_status_canonical", ""),
                "final_step": row.get("raw_final_step_name", ""),
                "final_frame_time": parse_float(row.get("raw_final_frame_value")),
                "extracted_fields": "",
                "previous_reward_mean_all": parse_float(r10.get("reward_mean_all")),
            }
        )
    return rows


def compare_batch20_probe60(batch20: list[dict[str, Any]], probe60: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for n in EXPECTED_N:
        p = [r for r in probe60 if r["n"] == n]
        b = [r for r in batch20 if r["n"] == n]
        row: dict[str, Any] = {"n": n}
        for metric, label in [("u2_range", "u2"), ("peeq_max", "peeq"), ("surface_t_proxy", "surfaceT"), ("mises_max", "mises")]:
            pb = min(p, key=lambda r: r[metric])
            bb = min(b, key=lambda r: r[metric])
            row[f"probe60_best_{label}_strategy"] = pb["strategy_name"]
            row[f"probe60_best_{label}_value"] = pb[metric]
            row[f"batch20_best_{label}_strategy"] = bb["strategy_name"]
            row[f"batch20_best_{label}_value"] = bb[metric]
            row[f"{label}_improvement_ratio"] = safe_divide(pb[metric] - bb[metric], pb[metric], default=math.nan)
            row[f"batch20_beats_probe60_best_{label}"] = bb[metric] < pb[metric]
        bc = max(b, key=lambda r: r["reward_batch20_u2_primary"])
        row["batch20_best_constrained_strategy"] = bc["strategy_name"]
        out.append(row)
    return out


def build_combined80(probe60: list[dict[str, Any]], batch20: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [dict(r) for r in probe60] + [dict(r) for r in batch20]
    for n in EXPECTED_N:
        group = [row for row in rows if row["n"] == n]
        for metric, prefix in [("u2_range", "u2"), ("peeq_max", "peeq"), ("surface_t_proxy", "surfaceT"), ("mises_max", "mises")]:
            rank_ascending(group, metric, f"{prefix}_rank_combined80_within_n")
            add_rank_score(group, f"{prefix}_rank_combined80_within_n", f"{prefix}_score_combined80_rank")
            add_minmax_cost(group, metric, f"{prefix}_cost_minmax_combined80_within_n")
        for row in group:
            row["reward_combined80_u2_primary"] = (
                0.65 * row["u2_score_combined80_rank"]
                + 0.20 * row["peeq_score_combined80_rank"]
                + 0.10 * row["surfaceT_score_combined80_rank"]
                + 0.05 * row["mises_score_combined80_rank"]
            )
        for idx, row in enumerate(sorted(group, key=lambda r: r["reward_combined80_u2_primary"], reverse=True), start=1):
            row["combined80_constrained_rank_within_n"] = idx
        pareto_flags(group, ["u2_range", "peeq_max"], "combined80_pareto_flag_u2_peeq")
        pareto_flags(group, ["u2_range", "peeq_max", "surface_t_proxy", "mises_max"], "combined80_pareto_flag_u2_peeq_surfaceT_mises")
        for metric, flag in [
            ("u2_range", "is_new_best_u2_within_n"),
            ("peeq_max", "is_new_best_peeq_within_n"),
            ("surface_t_proxy", "is_new_best_surfaceT_within_n"),
        ]:
            best = min(group, key=lambda r: r[metric])
            for row in group:
                row[flag] = row is best and row["dataset_source"] == "batch20_run14"
        best_reward = max(group, key=lambda r: r["reward_combined80_u2_primary"])
        for row in group:
            row["is_new_best_combined_reward_within_n"] = row is best_reward and row["dataset_source"] == "batch20_run14"
    return rows


def validate_combined(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["n"] for row in rows)
    errors = []
    if len(rows) != 80:
        errors.append(f"Expected 80 rows, found {len(rows)}")
    if sorted(counts) != EXPECTED_N:
        errors.append(f"Expected N values {EXPECTED_N}, found {sorted(counts)}")
    for n in EXPECTED_N:
        if counts[n] != 20:
            errors.append(f"Expected 20 rows for N{n}, found {counts[n]}")
    seen = set()
    for row in rows:
        key = (row["dataset_source"], row["n"], row["strategy_name"])
        if key in seen:
            errors.append(f"Duplicate combined key: {key}")
        seen.add(key)
        for col in ["u2_range", "peeq_max", "surface_t_proxy", "mises_max"]:
            if not math.isfinite(row[col]):
                errors.append(f"{row['strategy_name']} missing {col}")
    return {"errors": errors, "total_rows": len(rows), "per_n_counts": dict(sorted(counts.items()))}


def rl_ready(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keep = [
        "n", "strategy_name", "dataset_source", "order_json", "candidate_family", "selection_bucket",
        "u2_range", "peeq_max", "surface_t_proxy", "mises_max",
        "u2_rank_combined80_within_n", "peeq_rank_combined80_within_n", "surfaceT_rank_combined80_within_n", "mises_rank_combined80_within_n",
        "u2_cost_minmax_combined80_within_n", "peeq_cost_minmax_combined80_within_n", "surfaceT_cost_minmax_combined80_within_n", "mises_cost_minmax_combined80_within_n",
        "reward_combined80_u2_primary", "combined80_constrained_rank_within_n",
        "combined80_pareto_flag_u2_peeq", "combined80_pareto_flag_u2_peeq_surfaceT_mises", "teacher_validation_status",
    ]
    out = []
    for row in rows:
        item = {k: row.get(k, "") for k in keep}
        item.update(
            {
                "target_reward_combined80_u2_primary": row["reward_combined80_u2_primary"],
                "target_u2_score_combined80_rank": row["u2_score_combined80_rank"],
                "target_peeq_score_combined80_rank": row["peeq_score_combined80_rank"],
                "target_surfaceT_score_combined80_rank": row["surfaceT_score_combined80_rank"],
                "target_mises_score_combined80_rank": row["mises_score_combined80_rank"],
                "target_combined80_constrained_rank_within_n": row["combined80_constrained_rank_within_n"],
            }
        )
        out.append(item)
    return out


def leaderboard(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    specs = [
        ("top5_u2", "u2_range", False, 5),
        ("top5_peeq", "peeq_max", False, 5),
        ("top5_surfaceT", "surface_t_proxy", False, 5),
        ("top5_mises", "mises_max", False, 5),
        ("top5_combined_reward", "reward_combined80_u2_primary", True, 5),
        ("worst3_u2", "u2_range", True, 3),
        ("worst3_peeq", "peeq_max", True, 3),
        ("worst3_surfaceT", "surface_t_proxy", True, 3),
    ]
    for n in EXPECTED_N:
        group = [row for row in rows if row["n"] == n]
        for name, metric, reverse, limit in specs:
            for pos, row in enumerate(sorted(group, key=lambda r: r[metric], reverse=reverse)[:limit], start=1):
                output.append({"n": n, "leaderboard": name, "position": pos, "strategy_name": row["strategy_name"], "dataset_source": row["dataset_source"], "value": row[metric], "combined_reward_rank": row["combined80_constrained_rank_within_n"]})
    return output


def highlights(batch20: list[dict[str, Any]], combined: list[dict[str, Any]]) -> str:
    lines = ["# Batch20 Highlight Cases", ""]
    repaired = next((row for row in combined if row["strategy_name"] == REPAIRED_CASE), None)
    if repaired:
        lines += [
            f"## {REPAIRED_CASE}",
            f"- Status: `{repaired['teacher_validation_status']}`",
            f"- U2 range: `{repaired['u2_range']}`",
            f"- PEEQ max: `{repaired['peeq_max']}`",
            f"- SurfaceT proxy MPa: `{safe_divide(repaired['surface_t_proxy'], 1_000_000.0)}`",
            f"- Combined80 U2 rank within N: `{repaired['u2_rank_combined80_within_n']}`",
            f"- Combined80 reward rank within N: `{repaired['combined80_constrained_rank_within_n']}`",
            "",
        ]
    new_bests = [row for row in combined if row["dataset_source"] == "batch20_run14" and (row["is_new_best_u2_within_n"] or row["is_new_best_peeq_within_n"] or row["is_new_best_surfaceT_within_n"] or row["is_new_best_combined_reward_within_n"])]
    lines.append("## New Combined80 Best Flags")
    lines.extend([f"- {row['strategy_name']}: U2={row['is_new_best_u2_within_n']}, PEEQ={row['is_new_best_peeq_within_n']}, SurfaceT={row['is_new_best_surfaceT_within_n']}, reward={row['is_new_best_combined_reward_within_n']}" for row in new_bests] or ["- None."])
    lines += ["", "## Batch20 Pareto Non-Dominated Cases"]
    pareto = [row for row in combined if row["dataset_source"] == "batch20_run14" and row["combined80_pareto_flag_u2_peeq_surfaceT_mises"]]
    lines.extend([f"- {row['strategy_name']} (N{row['n']})" for row in pareto] or ["- None."])
    tradeoffs = [row for row in combined if row["dataset_source"] == "batch20_run14" and ((row["u2_rank_combined80_within_n"] <= 5 and row["surfaceT_rank_combined80_within_n"] > 10) or (row["surfaceT_rank_combined80_within_n"] <= 5 and row["u2_rank_combined80_within_n"] > 10))]
    lines += ["", "## U2 / SurfaceT Tradeoff Flags"]
    lines.extend([f"- {row['strategy_name']} (N{row['n']}): U2 rank {row['u2_rank_combined80_within_n']}, SurfaceT rank {row['surfaceT_rank_combined80_within_n']}" for row in tradeoffs] or ["- None."])
    return "\n".join(lines) + "\n"


def maybe_plots(combined: list[dict[str, Any]], audit: list[dict[str, Any]], comparison: list[dict[str, Any]]) -> list[str]:
    paths = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return [f"PLOTTING_SKIPPED: {exc}"]
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    colors = {"probe60_run08": "tab:blue", "batch20_run14": "tab:orange"}
    for ymetric, filename in [("peeq_max", "u2_vs_peeq_by_source.png"), ("surface_t_proxy", "u2_vs_surfaceT_by_source.png")]:
        plt.figure(figsize=(6, 4))
        for source in colors:
            group = [r for r in combined if r["dataset_source"] == source]
            plt.scatter([r["u2_range"] for r in group], [r[ymetric] for r in group], label=source, s=25)
        plt.xlabel("u2_range")
        plt.ylabel(ymetric)
        plt.legend()
        path = FIGURE_DIR / filename
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        paths.append(str(path))
    plt.figure(figsize=(6, 4))
    plt.scatter([r["predicted_reward_mean_all"] for r in audit], [r["realized_reward_batch20_u2_primary"] for r in audit], s=30)
    plt.xlabel("predicted reward")
    plt.ylabel("realized batch20 reward")
    path = FIGURE_DIR / "batch20_predicted_vs_realized_reward.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    paths.append(str(path))
    return paths


def write_claim_boundary(md: Path, js: Path) -> None:
    safe = [
        "Batch20 ODB teacher metrics were ingested successfully for 20/20 cases.",
        "Combined teacher-labelled dataset now contains 80 cases, with 20 per N.",
        "Batch20 can be compared against the original probe60 dataset using within-N ranks.",
        "Combined80 rankings and normalized costs are ready for updated surrogate/RL analysis.",
        "S3B20_N40_B02_diversity_top is teacher-valid and can be included in RL/analysis.",
    ]
    unsafe = [
        "Do not claim trained variable-N RL policy superiority.",
        "Do not claim surrogate predictions are ground truth.",
        "Do not claim arbitrary-N generalization.",
        "Do not claim batch20 candidates are physically superior unless teacher comparison proves specific metric-level improvement.",
        "Do not claim fixed-32 U2 guard transfer.",
        "Do not claim final optimum.",
    ]
    md.write_text("# Run 16 Claim Boundary\n\n## Safe Claims\n" + "\n".join(f"- {x}" for x in safe) + "\n\n## Unsafe Claims\n" + "\n".join(f"- {x}" for x in unsafe) + "\n", encoding="utf-8")
    write_json(js, {"verdict": "RUN16_COMBINED80_TEACHER_DATASET_READY_WITH_CLAIM_BOUNDARY", "safe_claims": safe, "unsafe_claims": unsafe})


def update_run_index(verdict: str) -> None:
    if not RUN_INDEX_PATH.exists():
        return
    entry = (
        "| run_16 | Batch20 teacher metrics ingestion and combined80 ranking | Ingest 20 batch20 teacher metrics, audit surrogate predictions, compare with probe60, and build combined80 RL-ready within-N ranked dataset. | "
        "`scripts/stage3/run_16_ingest_batch20_and_build_combined80_teacher_dataset.py` | "
        "`docs/stage3/runs/run_16_batch20_teacher_metrics_ingestion_and_combined80_ranking/RUN_16_BATCH20_TEACHER_METRICS_INGESTION_AND_COMBINED80_RANKING_REPORT.md` | "
        "`outputs/stage3_run_16_batch20_teacher_metrics_ingestion_and_combined80_ranking/` | "
        f"`{verdict}` | No Abaqus, no ODB opening, no abqjobpilot, no CAE/INP/JNL generation, no final RL policy training, no commit/push. Next: run17 combined80 surrogate validation update. |"
    )
    lines = RUN_INDEX_PATH.read_text(encoding="utf-8").splitlines()
    for idx, line in enumerate(lines):
        if line.startswith("| run_16 | Batch20 teacher metrics ingestion and combined80 ranking |"):
            lines[idx] = entry
            break
    else:
        lines.append(entry)
    RUN_INDEX_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(validation: dict[str, Any], comparison: list[dict[str, Any]], combined: list[dict[str, Any]], audit: list[dict[str, Any]], output_files: list[str]) -> None:
    lines = [
        "# Stage 3 Run 16 - Batch20 Teacher Metrics Ingestion and Combined80 Ranking",
        "",
        "## Purpose",
        "Ingest 20 official batch20 teacher metrics, audit surrogate predictions, compare against probe60, and build a combined 80-case teacher-labelled dataset.",
        "",
        "## Inputs",
        f"- `{BATCH20_METRICS}`",
        f"- `{RUN13_HANDOFF}`",
        f"- `{PROBE60_RANKED}`",
        f"- `{RUN10_REWARD}`",
        "",
        "## Batch20 Validation Status",
        f"- `{validation['verdict']}`",
        f"- Per-N counts: {validation['per_n_counts']}",
        f"- Solver completion verdict: `{validation.get('solver_completion_verdict')}`",
        f"- Extraction verdict: `{validation.get('extraction_verdict')}`",
        "",
        "## Batch20 Within-N Ranking",
        "Batch20 ranks, min-max costs, U2-primary reward, and Pareto flags were recomputed over the 5 candidates per N.",
        "",
        "## Surrogate Prediction Audit",
        f"- Overall Spearman predicted vs realized batch20 reward: `{audit[0]['overall_spearman_pred_vs_realized']}`",
        "- Top1/top2 within-N comparisons are recorded in `batch20_surrogate_prediction_audit.csv`.",
        "",
        "## Batch20 vs Probe60 Best Comparison",
    ]
    for row in comparison:
        lines.append(f"- N{row['n']}: beats probe60 best U2={row['batch20_beats_probe60_best_u2']}, PEEQ={row['batch20_beats_probe60_best_peeq']}, SurfaceT={row['batch20_beats_probe60_best_surfaceT']}, Mises={row['batch20_beats_probe60_best_mises']}.")
    lines += [
        "",
        "## Combined80 Teacher Dataset Construction",
        "- Total rows: 80.",
        "- Per-N rows: 20 for N12/N16/N24/N40.",
        "",
        "## Combined80 Within-N Rankings",
        "Ranks, rank scores, min-max costs, U2-primary reward, and Pareto flags were recomputed over 20 cases per N.",
        "",
        "## RL-Ready Dataset",
        "- `combined80_RL_ready_dataset.csv` contains target columns for updated surrogate/RL analysis.",
        "",
        "## Highlight Cases",
        f"- `{REPAIRED_CASE}` is included and teacher-valid.",
        "",
        "## Claim Boundary",
        "- No final RL policy superiority, arbitrary-N generalization, fixed-32 guard transfer, or final optimum is claimed.",
        "",
        "## Output Files",
        *[f"- `{path}`" for path in output_files],
        "",
        "## Recommended Run17",
        "Use `combined80_RL_ready_dataset.csv` to update lightweight surrogate validation with the expanded 80-case dataset. Re-run leave-N-out validation and compare against run11. Do not train final RL policy yet unless explicitly instructed.",
    ]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def git_branch() -> str:
    try:
        result = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception:
        return "UNKNOWN"


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics_rows = read_csv(BATCH20_METRICS)
    validation = validate_batch20(metrics_rows)
    validation_path = OUTPUT_DIR / "run16_batch20_input_validation_summary.json"
    write_json(validation_path, validation)
    if validation["verdict"].startswith("FAIL"):
        print(validation["verdict"])
        print(json.dumps(validation, indent=2))
        return 2
    batch20 = add_batch20_ranking(canonicalize_batch20(metrics_rows))
    batch20_csv = OUTPUT_DIR / "batch20_teacher_metrics_canonical.csv"
    batch20_json = OUTPUT_DIR / "batch20_teacher_metrics_canonical.json"
    write_csv(batch20_csv, batch20)
    write_json(batch20_json, batch20)
    batch20_rank_csv = OUTPUT_DIR / "batch20_within_N_rank_table.csv"
    batch20_rank_json = OUTPUT_DIR / "batch20_within_N_rank_table.json"
    write_csv(batch20_rank_csv, batch20)
    write_json(batch20_rank_json, batch20)
    audit = surrogate_audit(batch20)
    audit_csv = OUTPUT_DIR / "batch20_surrogate_prediction_audit.csv"
    audit_json = OUTPUT_DIR / "batch20_surrogate_prediction_audit.json"
    write_csv(audit_csv, audit)
    write_json(audit_json, audit)
    probe60 = probe60_rows()
    comparison = compare_batch20_probe60(batch20, probe60)
    comparison_csv = OUTPUT_DIR / "batch20_vs_probe60_best_comparison.csv"
    comparison_json = OUTPUT_DIR / "batch20_vs_probe60_best_comparison.json"
    write_csv(comparison_csv, comparison)
    write_json(comparison_json, comparison)
    combined = build_combined80(probe60, batch20)
    combined_validation = validate_combined(combined)
    combined_csv = OUTPUT_DIR / "combined80_teacher_dataset.csv"
    combined_json = OUTPUT_DIR / "combined80_teacher_dataset.json"
    write_csv(combined_csv, combined)
    write_json(combined_json, combined)
    ready = rl_ready(combined)
    ready_csv = OUTPUT_DIR / "combined80_RL_ready_dataset.csv"
    ready_json = OUTPUT_DIR / "combined80_RL_ready_dataset.json"
    write_csv(ready_csv, ready)
    write_json(ready_json, ready)
    leaders = leaderboard(combined)
    leader_csv = OUTPUT_DIR / "combined80_per_N_leaderboard.csv"
    leader_json = OUTPUT_DIR / "combined80_per_N_leaderboard.json"
    write_csv(leader_csv, leaders)
    write_json(leader_json, leaders)
    highlight_md = OUTPUT_DIR / "batch20_highlight_cases.md"
    highlight_md.write_text(highlights(batch20, combined), encoding="utf-8")
    plots = maybe_plots(combined, audit, comparison)
    claim_md = OUTPUT_DIR / "run16_claim_boundary.md"
    claim_json = OUTPUT_DIR / "run16_claim_boundary.json"
    write_claim_boundary(claim_md, claim_json)
    outputs = [
        str(validation_path), str(batch20_csv), str(batch20_json), str(batch20_rank_csv), str(batch20_rank_json),
        str(audit_csv), str(audit_json), str(comparison_csv), str(comparison_json), str(combined_csv), str(combined_json),
        str(ready_csv), str(ready_json), str(leader_csv), str(leader_json), str(highlight_md), str(claim_md), str(claim_json),
        *[p for p in plots if not p.startswith("PLOTTING_SKIPPED")], str(REPORT_PATH),
    ]
    write_report(validation, comparison, combined, audit, outputs)
    update_run_index(validation["verdict"])
    manifest = {
        "run_id": RUN_ID,
        "run_name": RUN_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "branch": git_branch(),
        "script_path": str(Path(__file__).resolve()),
        "input_files": [str(BATCH20_METRICS), str(BATCH20_EXTRACTION), str(BATCH20_SOLVER), str(BATCH20_SUMMARY), str(BATCH20_REPORT), str(RUN13_HANDOFF), str(RUN12_BATCH20), str(RUN12_SCORED), str(PROBE60_RANKED), str(PROBE60_LABELS), str(RUN10_REWARD), str(RUN11_BEST), str(RUN12_REPORT), str(RUN13_REPORT)],
        "output_files": outputs,
        "total_batch20_rows": len(batch20),
        "total_combined_rows": len(combined),
        "per_N_combined_rows": combined_validation["per_n_counts"],
        "report_path": str(REPORT_PATH),
        "claim_boundary_path": str(claim_md),
        "validation_verdict": validation["verdict"],
        "combined_validation": combined_validation,
        "no_solver_run": True,
        "no_odb_opened": True,
        "no_abqjobpilot_run": True,
        "no_cae_inp_generated": True,
        "no_rl_policy_training": True,
        "no_commit_or_push": True,
    }
    write_json(MANIFEST_PATH, manifest)
    print(validation["verdict"])
    print(f"batch20_rows={len(batch20)} per_n={validation['per_n_counts']}")
    print(f"combined_rows={len(combined)} per_n={combined_validation['per_n_counts']}")
    print(f"overall_spearman_pred_vs_realized={audit[0]['overall_spearman_pred_vs_realized']}")
    print(f"report={REPORT_PATH}")
    print(f"manifest={MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
