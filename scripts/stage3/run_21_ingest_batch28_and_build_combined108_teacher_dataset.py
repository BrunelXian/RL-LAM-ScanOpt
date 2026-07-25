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
RUN_ID = "run_21_batch28_teacher_metrics_ingestion_and_combined108_ranking"
RUN_NAME = "batch28 teacher metrics ingestion and combined108 ranking"

BATCH28_METRICS = ROOT / "outputs" / "stage3_run_20_batch28_odb_teacher_validation" / "run20_batch28_teacher_metrics.csv"
BATCH28_EXTRACTION = ROOT / "outputs" / "stage3_run_20_batch28_odb_teacher_validation" / "run20_batch28_odb_extraction_summary.csv"
BATCH28_SOLVER = ROOT / "outputs" / "stage3_run_20_batch28_odb_teacher_validation" / "run20_batch28_solver_completion_audit.csv"
BATCH28_SUMMARY = ROOT / "outputs" / "stage3_run_20_batch28_odb_teacher_validation" / "run20_batch28_odb_teacher_validation_summary.json"
BATCH28_REPORT = ROOT / "outputs" / "stage3_run_20_batch28_odb_teacher_validation" / "run20_batch28_odb_teacher_validation_report.md"
RUN19_BATCH28 = ROOT / "outputs" / "stage3_run_19_run18_candidate_handoff_review_package" / "batch28" / "stage3_run19_batch28_candidate_orders.csv"
RUN18_BATCH28 = ROOT / "outputs" / "stage3_run_18_combined80_surrogate_screened_candidate_generation" / "run18_recommended_future_teacher_batch28.csv"
RUN18_SCORED = ROOT / "outputs" / "stage3_run_18_combined80_surrogate_screened_candidate_generation" / "run18_candidate_pool_scored.csv"
COMBINED80_TEACHER = ROOT / "outputs" / "stage3_run_16_batch20_teacher_metrics_ingestion_and_combined80_ranking" / "combined80_teacher_dataset.csv"
COMBINED80_READY = ROOT / "outputs" / "stage3_run_16_batch20_teacher_metrics_ingestion_and_combined80_ranking" / "combined80_RL_ready_dataset.csv"
COMBINED80_LEADERBOARD = ROOT / "outputs" / "stage3_run_16_batch20_teacher_metrics_ingestion_and_combined80_ranking" / "combined80_per_N_leaderboard.csv"
RUN17_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_17_combined80_surrogate_reward_model_validation_update" / "RUN_17_COMBINED80_SURROGATE_REWARD_MODEL_VALIDATION_UPDATE_REPORT.md"
RUN18_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_18_combined80_surrogate_screened_candidate_generation" / "RUN_18_COMBINED80_SURROGATE_SCREENED_CANDIDATE_GENERATION_REPORT.md"
RUN19_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_19_run18_candidate_handoff_review_package" / "RUN_19_RUN18_CANDIDATE_HANDOFF_REVIEW_PACKAGE_REPORT.md"

OUTPUT_DIR = ROOT / "outputs" / "stage3_run_21_batch28_teacher_metrics_ingestion_and_combined108_ranking"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = ROOT / "docs" / "stage3" / "runs" / RUN_ID
REPORT_PATH = REPORT_DIR / "RUN_21_BATCH28_TEACHER_METRICS_INGESTION_AND_COMBINED108_RANKING_REPORT.md"
MANIFEST_PATH = ROOT / "artifacts" / "manifests" / "stage3_run_21_manifest.json"
RUN_INDEX_PATH = ROOT / "docs" / "stage3" / "STAGE3_RUN_INDEX.md"

EXPECTED_N = [12, 16, 24, 40]
EXPECTED_BATCH28_COUNTS = {12: 4, 16: 4, 24: 10, 40: 10}
EXPECTED_COMBINED108_COUNTS = {12: 24, 16: 24, 24: 30, 40: 30}
SPECIAL_CASE = "S3R19B28_N40_B01_surrogate_top"


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


def write_table_json(path: Path, rows: list[dict[str, Any]]) -> None:
    cols: list[str] = []
    for row in rows:
        for key in row:
            if key not in cols:
                cols.append(key)
    write_json(path, {"schema": "columns_and_rows", "columns": cols, "rows": [[row.get(col) for col in cols] for row in rows]})


def parse_int(value: Any) -> int:
    text = str(value).strip()
    if text.upper().startswith("N"):
        text = text[1:]
    return int(float(text))


def parse_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def safe_divide(num: float, den: float, default: float = 0.0) -> float:
    return num / den if den else default


def mean(values: list[float], default: float = math.nan) -> float:
    vals = [x for x in values if math.isfinite(x)]
    return statistics.fmean(vals) if vals else default


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
    vals = [row[metric] for row in rows]
    mn, mx = min(vals), max(vals)
    for row in rows:
        row[col] = safe_divide(row[metric] - mn, mx - mn, default=0.0)


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
    if len(pairs) < 3:
        return math.nan

    def ranks(vals: list[float]) -> list[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        out = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i + 1
            while j < len(order) and vals[order[j]] == vals[order[i]]:
                j += 1
            avg = (i + 1 + j) / 2.0
            for k in range(i, j):
                out[order[k]] = avg
            i = j
        return out

    rx = ranks([p[0] for p in pairs])
    ry = ranks([p[1] for p in pairs])
    mx, my = mean(rx), mean(ry)
    den = math.sqrt(sum((x - mx) ** 2 for x in rx) * sum((y - my) ** 2 for y in ry))
    return safe_divide(sum((x - mx) * (y - my) for x, y in zip(rx, ry)), den, default=math.nan)


def validate_batch28(rows: list[dict[str, str]]) -> dict[str, Any]:
    errors: list[str] = []
    counts: Counter[int] = Counter()
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
        if "PASS" not in row.get("teacher_validation_status", "").upper():
            errors.append(f"{name}: non-pass teacher_validation_status={row.get('teacher_validation_status')}")
        for col in ["u2_range", "peeq_max", "mises_max"]:
            if not math.isfinite(parse_float(row.get(col))):
                errors.append(f"{name}: missing {col}")
        surface = row.get("surface_t_proxy_max_tensile_pa") or row.get("surface_t_proxy") or row.get("surface_t_proxy_max_tensile_mpa")
        if not math.isfinite(parse_float(surface)):
            errors.append(f"{name}: missing surface_t_proxy")
    if len(rows) != 28:
        errors.append(f"Expected 28 rows, found {len(rows)}")
    if sorted(counts) != EXPECTED_N:
        errors.append(f"Expected N values {EXPECTED_N}, found {sorted(counts)}")
    for n, expected in EXPECTED_BATCH28_COUNTS.items():
        if counts[n] != expected:
            errors.append(f"Expected {expected} rows for N{n}, found {counts[n]}")
    special = next((r for r in rows if (r.get("handoff_strategy_name") or r.get("strategy_name")) == SPECIAL_CASE), None)
    if special is None:
        errors.append(f"{SPECIAL_CASE} missing")
    elif "PASS" not in special.get("teacher_validation_status", "").upper():
        errors.append(f"{SPECIAL_CASE} is not pass")

    summary = json.loads(BATCH28_SUMMARY.read_text(encoding="utf-8")) if BATCH28_SUMMARY.exists() else {}
    audit = summary.get("audit_summary", {})
    extraction = summary.get("extraction_summary", {})
    if audit.get("total_complete") != 28 or extraction.get("total_pass") != 28:
        errors.append("Summary does not confirm 28/28 completion and extraction")
    if audit.get("total_lck_present", 0) != 0:
        errors.append("Summary reports nonzero lck count")
    verdict = "PASS_RUN21_BATCH28_TEACHER_METRICS_28_OF_28_READY" if not errors else "FAIL_RUN21_BATCH28_INPUT_INVALID"
    payload = {
        "verdict": verdict,
        "errors": errors,
        "row_count": len(rows),
        "per_n_counts": dict(sorted(counts.items())),
        "solver_completion_verdict": audit.get("verdict"),
        "extraction_verdict": extraction.get("verdict"),
        "total_lck_present": audit.get("total_lck_present"),
        "special_case_present_and_pass": special is not None and "PASS" in special.get("teacher_validation_status", "").upper(),
    }
    write_json(OUTPUT_DIR / "run21_batch28_input_validation_summary.json", payload)
    return payload


def canonicalize_batch28(metrics_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    handoff = {row["handoff_strategy_name"]: row for row in read_csv(RUN19_BATCH28)}
    scored_by_id = {row["candidate_id"]: row for row in read_csv(RUN18_SCORED)}
    missing_optional: Counter[str] = Counter()
    canonical: list[dict[str, Any]] = []
    for row in metrics_rows:
        name = row.get("handoff_strategy_name") or row.get("strategy_name")
        meta = handoff.get(name, {})
        scored = scored_by_id.get(meta.get("original_run18_candidate_id", ""), {})
        if not meta:
            missing_optional["run19_handoff_metadata"] += 1
        surface_pa = parse_float(row.get("surface_t_proxy_max_tensile_pa"))
        if not math.isfinite(surface_pa):
            surface_pa = parse_float(row.get("surface_t_proxy")) 
        if not math.isfinite(surface_pa):
            surface_pa = parse_float(row.get("surface_t_proxy_max_tensile_mpa")) * 1_000_000.0
        canonical.append(
            {
                "dataset_source": "batch28_run20",
                "n": parse_int(row["n"]),
                "strategy_name": name,
                "handoff_strategy_name": name,
                "job_name": row.get("job_name", ""),
                "candidate_id": meta.get("original_run18_candidate_id", ""),
                "original_run18_candidate_id": meta.get("original_run18_candidate_id", ""),
                "candidate_family": meta.get("candidate_family", scored.get("candidate_family", "")),
                "candidate_source": meta.get("candidate_source", scored.get("candidate_source", "")),
                "selection_bucket": meta.get("selection_bucket", scored.get("selection_bucket", "")),
                "priority_role": meta.get("priority_role", scored.get("priority_role", "")),
                "predicted_reward_combined80_u2_primary": parse_float(meta.get("predicted_reward_combined80_u2_primary", scored.get("pred_reward_combined80_u2_primary"))),
                "predicted_rank_within_n": parse_float(meta.get("predicted_rank_within_n", scored.get("pred_rank_within_n"))),
                "predicted_uncertainty_std": parse_float(meta.get("predicted_uncertainty_std", scored.get("pred_uncertainty_std"))),
                "novelty_distance_to_combined80": parse_float(meta.get("novelty_distance_to_combined80", scored.get("novelty_distance_to_combined80"))),
                "nearest_existing_teacher_strategy": meta.get("nearest_existing_teacher_strategy", scored.get("nearest_existing_teacher_strategy", "")),
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
    return canonical, {"missing_optional_metadata": dict(missing_optional)}


def add_within_batch_ranking(rows: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    for n in EXPECTED_N:
        group = [row for row in rows if row["n"] == n]
        for metric, prefix in [("u2_range", "u2"), ("peeq_max", "peeq"), ("surface_t_proxy", "surfaceT"), ("mises_max", "mises")]:
            rank_ascending(group, metric, f"{prefix}_rank_{label}_within_n")
            add_minmax_cost(group, metric, f"{prefix}_cost_minmax_{label}_within_n")
            add_rank_score(group, f"{prefix}_rank_{label}_within_n", f"{prefix}_score_{label}_rank")
        for row in group:
            row[f"reward_{label}_u2_primary"] = (
                0.65 * row[f"u2_score_{label}_rank"]
                + 0.20 * row[f"peeq_score_{label}_rank"]
                + 0.10 * row[f"surfaceT_score_{label}_rank"]
                + 0.05 * row[f"mises_score_{label}_rank"]
            )
        rank_ascending(group, f"reward_{label}_u2_primary", f"{label}_reward_rank_ascending_tmp")
        total = len(group)
        for row in group:
            row[f"{label}_constrained_rank_within_n"] = total + 1 - row[f"{label}_reward_rank_ascending_tmp"]
            row.pop(f"{label}_reward_rank_ascending_tmp", None)
        pareto_flags(group, ["u2_range", "peeq_max"], f"{label}_pareto_flag_u2_peeq")
        pareto_flags(group, ["u2_range", "peeq_max", "surface_t_proxy", "mises_max"], f"{label}_pareto_flag_u2_peeq_surfaceT_mises")
    return rows


def realized_prediction_audit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for n in EXPECTED_N:
        group = [row for row in rows if row["n"] == n]
        pred_values = [row["predicted_reward_combined80_u2_primary"] for row in group]
        realized = [row["reward_batch28_u2_primary"] for row in group]
        sp = spearman(pred_values, realized)
        pred_ranked = sorted(group, key=lambda r: r["predicted_reward_combined80_u2_primary"], reverse=True)
        real_ranked = sorted(group, key=lambda r: r["reward_batch28_u2_primary"], reverse=True)
        pred_rank = {row["strategy_name"]: i + 1 for i, row in enumerate(pred_ranked)}
        real_rank = {row["strategy_name"]: i + 1 for i, row in enumerate(real_ranked)}
        top3_pred = {row["strategy_name"] for row in pred_ranked[:3]}
        top3_real = {row["strategy_name"] for row in real_ranked[:3]}
        for row in group:
            pr = pred_rank[row["strategy_name"]]
            rr = real_rank[row["strategy_name"]]
            out.append(
                {
                    "n": n,
                    "strategy_name": row["strategy_name"],
                    "predicted_reward_combined80_u2_primary": row["predicted_reward_combined80_u2_primary"],
                    "realized_reward_batch28_u2_primary": row["reward_batch28_u2_primary"],
                    "prediction_error": row["reward_batch28_u2_primary"] - row["predicted_reward_combined80_u2_primary"],
                    "absolute_error": abs(row["reward_batch28_u2_primary"] - row["predicted_reward_combined80_u2_primary"]),
                    "predicted_rank_within_n": pr,
                    "realized_rank_within_n": rr,
                    "rank_error": rr - pr,
                    "abs_rank_error": abs(rr - pr),
                    "within_n_spearman_predicted_vs_realized": sp,
                    "top_predicted_candidate_for_n": pred_ranked[0]["strategy_name"],
                    "best_realized_candidate_for_n": real_ranked[0]["strategy_name"],
                    "top_predicted_was_top_realized": pred_ranked[0]["strategy_name"] == real_ranked[0]["strategy_name"],
                    "top3_overlap_within_n": len(top3_pred & top3_real),
                    "note": "Diagnostic only; predictions are not teacher labels.",
                }
            )
    all_sp = spearman([row["predicted_reward_combined80_u2_primary"] for row in rows], [row["reward_batch28_u2_primary"] for row in rows])
    for row in out:
        row["overall_spearman_across_batch28"] = all_sp
    return out


def compare_batch28_combined80(batch28: list[dict[str, Any]], combined80: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for n in EXPECTED_N:
        old = [row for row in combined80 if row["n"] == n]
        new = [row for row in batch28 if row["n"] == n]
        row: dict[str, Any] = {"n": n}
        for metric, label in [("u2_range", "u2"), ("peeq_max", "peeq"), ("surface_t_proxy", "surfaceT"), ("mises_max", "mises")]:
            old_best = min(old, key=lambda r: r[metric])
            new_best = min(new, key=lambda r: r[metric])
            row[f"combined80_best_{label}_strategy"] = old_best["strategy_name"]
            row[f"combined80_best_{label}_value"] = old_best[metric]
            row[f"batch28_best_{label}_strategy"] = new_best["strategy_name"]
            row[f"batch28_best_{label}_value"] = new_best[metric]
            row[f"batch28_beats_combined80_best_{label}"] = new_best[metric] < old_best[metric]
            row[f"{label}_improvement_ratio_vs_combined80_best"] = safe_divide(old_best[metric] - new_best[metric], old_best[metric], default=math.nan)
        new_reward_best = max(new, key=lambda r: r["reward_batch28_u2_primary"])
        old_reward_best = max(old, key=lambda r: parse_float(r.get("reward_combined80_u2_primary"), -1.0))
        row["combined80_best_combined_reward_strategy"] = old_reward_best["strategy_name"]
        row["combined80_best_combined_reward"] = parse_float(old_reward_best.get("reward_combined80_u2_primary"), math.nan)
        row["batch28_best_constrained_strategy"] = new_reward_best["strategy_name"]
        row["batch28_best_constrained_reward"] = new_reward_best["reward_batch28_u2_primary"]
        row["batch28_introduces_new_combined_best_any_metric"] = any(row[f"batch28_beats_combined80_best_{label}"] for label in ["u2", "peeq", "surfaceT", "mises"])
        special = next((r for r in new if r["strategy_name"] == SPECIAL_CASE), None)
        if n == 40 and special:
            old_u2 = min(old, key=lambda r: r["u2_range"])
            row["special_case"] = SPECIAL_CASE
            row["special_case_u2_range"] = special["u2_range"]
            row["special_case_beats_previous_n40_u2_best"] = special["u2_range"] < old_u2["u2_range"]
            row["previous_n40_u2_best_strategy"] = old_u2["strategy_name"]
            row["previous_n40_u2_best_u2_range"] = old_u2["u2_range"]
        out.append(row)
    return out


def normalize_combined80_row(row: dict[str, str]) -> dict[str, Any]:
    return {
        "dataset_source": row.get("dataset_source", ""),
        "n": parse_int(row["n"]),
        "strategy_name": row["strategy_name"],
        "handoff_strategy_name": row.get("handoff_strategy_name", ""),
        "job_name": row.get("job_name", ""),
        "candidate_id": row.get("candidate_id", ""),
        "original_run18_candidate_id": row.get("original_run18_candidate_id", ""),
        "candidate_family": row.get("candidate_family", ""),
        "candidate_source": row.get("candidate_source", ""),
        "selection_bucket": row.get("selection_bucket", ""),
        "priority_role": row.get("priority_role", ""),
        "predicted_reward_combined80_u2_primary": parse_float(row.get("predicted_reward_combined80_u2_primary", row.get("predicted_reward_mean_all"))),
        "predicted_rank_within_n": parse_float(row.get("predicted_rank_within_n")),
        "predicted_uncertainty_std": parse_float(row.get("predicted_uncertainty_std")),
        "novelty_distance_to_combined80": parse_float(row.get("novelty_distance_to_combined80", row.get("novelty_distance_to_teacher"))),
        "nearest_existing_teacher_strategy": row.get("nearest_existing_teacher_strategy", ""),
        "order_json": row.get("order_json", ""),
        "u2_range": parse_float(row["u2_range"]),
        "peeq_max": parse_float(row["peeq_max"]),
        "surface_t_proxy": parse_float(row["surface_t_proxy"]),
        "surface_t_proxy_mpa": parse_float(row.get("surface_t_proxy_mpa"), default=safe_divide(parse_float(row["surface_t_proxy"]), 1_000_000.0)),
        "mises_max": parse_float(row["mises_max"]),
        "teacher_validation_status": row.get("teacher_validation_status", ""),
        "final_step": row.get("final_step", ""),
        "final_frame_time": parse_float(row.get("final_frame_time")),
        "extracted_fields": row.get("extracted_fields", ""),
    }


def build_combined108(combined80_rows: list[dict[str, Any]], batch28: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in combined80_rows] + [dict(row) for row in batch28]
    counts = Counter(row["n"] for row in rows)
    if len(rows) != 108 or dict(sorted(counts.items())) != EXPECTED_COMBINED108_COUNTS:
        raise RuntimeError(f"combined108 validation failed: rows={len(rows)} counts={dict(counts)}")
    seen: set[tuple[int, str, str]] = set()
    for row in rows:
        key = (row["n"], row["dataset_source"], row["strategy_name"])
        if key in seen:
            raise RuntimeError(f"Duplicate strategy within dataset_source/N: {key}")
        seen.add(key)
        for metric in ["u2_range", "peeq_max", "surface_t_proxy", "mises_max"]:
            if not math.isfinite(row[metric]):
                raise RuntimeError(f"Missing {metric}: {row['strategy_name']}")
    for n in EXPECTED_N:
        group = [row for row in rows if row["n"] == n]
        for metric, prefix in [("u2_range", "u2"), ("peeq_max", "peeq"), ("surface_t_proxy", "surfaceT"), ("mises_max", "mises")]:
            rank_ascending(group, metric, f"{prefix}_rank_combined108_within_n")
            add_minmax_cost(group, metric, f"{prefix}_cost_minmax_combined108_within_n")
            add_rank_score(group, f"{prefix}_rank_combined108_within_n", f"{prefix}_score_combined108_rank")
        for row in group:
            row["reward_combined108_u2_primary"] = (
                0.65 * row["u2_score_combined108_rank"]
                + 0.20 * row["peeq_score_combined108_rank"]
                + 0.10 * row["surfaceT_score_combined108_rank"]
                + 0.05 * row["mises_score_combined108_rank"]
            )
        rank_ascending(group, "reward_combined108_u2_primary", "combined108_reward_rank_ascending_tmp")
        total = len(group)
        for row in group:
            row["combined108_constrained_rank_within_n"] = total + 1 - row["combined108_reward_rank_ascending_tmp"]
            row.pop("combined108_reward_rank_ascending_tmp", None)
        pareto_flags(group, ["u2_range", "peeq_max"], "combined108_pareto_flag_u2_peeq")
        pareto_flags(group, ["u2_range", "peeq_max", "surface_t_proxy", "mises_max"], "combined108_pareto_flag_u2_peeq_surfaceT_mises")
        best_u2 = min(row["u2_rank_combined108_within_n"] for row in group)
        best_peeq = min(row["peeq_rank_combined108_within_n"] for row in group)
        best_surface = min(row["surfaceT_rank_combined108_within_n"] for row in group)
        best_mises = min(row["mises_rank_combined108_within_n"] for row in group)
        best_reward = min(row["combined108_constrained_rank_within_n"] for row in group)
        for row in group:
            row["is_new_best_u2_within_n"] = row["dataset_source"] == "batch28_run20" and row["u2_rank_combined108_within_n"] == best_u2
            row["is_new_best_peeq_within_n"] = row["dataset_source"] == "batch28_run20" and row["peeq_rank_combined108_within_n"] == best_peeq
            row["is_new_best_surfaceT_within_n"] = row["dataset_source"] == "batch28_run20" and row["surfaceT_rank_combined108_within_n"] == best_surface
            row["is_new_best_mises_within_n"] = row["dataset_source"] == "batch28_run20" and row["mises_rank_combined108_within_n"] == best_mises
            row["is_new_best_combined_reward_within_n"] = row["dataset_source"] == "batch28_run20" and row["combined108_constrained_rank_within_n"] == best_reward
    return rows


def rl_ready(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keep = [
        "n", "strategy_name", "dataset_source", "order_json", "candidate_family", "selection_bucket", "priority_role",
        "u2_range", "peeq_max", "surface_t_proxy", "mises_max",
        "u2_rank_combined108_within_n", "peeq_rank_combined108_within_n", "surfaceT_rank_combined108_within_n", "mises_rank_combined108_within_n",
        "u2_cost_minmax_combined108_within_n", "peeq_cost_minmax_combined108_within_n", "surfaceT_cost_minmax_combined108_within_n", "mises_cost_minmax_combined108_within_n",
        "reward_combined108_u2_primary", "combined108_constrained_rank_within_n",
        "combined108_pareto_flag_u2_peeq", "combined108_pareto_flag_u2_peeq_surfaceT_mises",
        "teacher_validation_status",
    ]
    out = []
    for row in rows:
        item = {col: row.get(col, "") for col in keep}
        item["target_reward_combined108_u2_primary"] = row["reward_combined108_u2_primary"]
        item["target_u2_score_combined108_rank"] = row["u2_score_combined108_rank"]
        item["target_peeq_score_combined108_rank"] = row["peeq_score_combined108_rank"]
        item["target_surfaceT_score_combined108_rank"] = row["surfaceT_score_combined108_rank"]
        item["target_mises_score_combined108_rank"] = row["mises_score_combined108_rank"]
        item["target_combined108_constrained_rank_within_n"] = row["combined108_constrained_rank_within_n"]
        out.append(item)
    return out


def leaderboard(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs = [
        ("top5_u2", "u2_range", False, 5),
        ("top5_peeq", "peeq_max", False, 5),
        ("top5_surfaceT", "surface_t_proxy", False, 5),
        ("top5_mises", "mises_max", False, 5),
        ("top5_combined_reward", "reward_combined108_u2_primary", True, 5),
        ("worst3_u2", "u2_range", True, 3),
        ("worst3_peeq", "peeq_max", True, 3),
        ("worst3_surfaceT", "surface_t_proxy", True, 3),
    ]
    out = []
    for n in EXPECTED_N:
        group = [row for row in rows if row["n"] == n]
        for section, metric, reverse, limit in specs:
            ordered = sorted(group, key=lambda row: row[metric], reverse=reverse)
            for rank, row in enumerate(ordered[:limit], start=1):
                out.append(
                    {
                        "n": n,
                        "leaderboard_section": section,
                        "rank": rank,
                        "strategy_name": row["strategy_name"],
                        "dataset_source": row["dataset_source"],
                        "metric": metric,
                        "metric_value": row[metric],
                        "u2_range": row["u2_range"],
                        "peeq_max": row["peeq_max"],
                        "surface_t_proxy": row["surface_t_proxy"],
                        "mises_max": row["mises_max"],
                        "reward_combined108_u2_primary": row["reward_combined108_u2_primary"],
                    }
                )
    return out


def highlight_cases(batch28: list[dict[str, Any]], combined108: list[dict[str, Any]], comparison: list[dict[str, Any]]) -> None:
    lines = ["# Batch28 Highlight Cases", ""]
    special = next((row for row in combined108 if row["strategy_name"] == SPECIAL_CASE), None)
    if special:
        lines.extend(
            [
                f"## {SPECIAL_CASE}",
                f"- Teacher status: `{special['teacher_validation_status']}`",
                f"- N40 combined108 U2 rank: `{special['u2_rank_combined108_within_n']}`",
                f"- N40 combined108 PEEQ rank: `{special['peeq_rank_combined108_within_n']}`",
                f"- N40 combined108 SurfaceT rank: `{special['surfaceT_rank_combined108_within_n']}`",
                f"- U2 range: `{special['u2_range']}`",
                f"- PEEQ max: `{special['peeq_max']}`",
                f"- SurfaceT proxy MPa: `{safe_divide(special['surface_t_proxy'], 1_000_000.0)}`",
                "",
            ]
        )
    new_bests = [row for row in combined108 if row["dataset_source"] == "batch28_run20" and any(row.get(flag) for flag in ["is_new_best_u2_within_n", "is_new_best_peeq_within_n", "is_new_best_surfaceT_within_n", "is_new_best_mises_within_n", "is_new_best_combined_reward_within_n"])]
    lines.append("## Batch28 New Combined108 Best Flags")
    if new_bests:
        for row in new_bests:
            flags = [flag for flag in ["is_new_best_u2_within_n", "is_new_best_peeq_within_n", "is_new_best_surfaceT_within_n", "is_new_best_mises_within_n", "is_new_best_combined_reward_within_n"] if row.get(flag)]
            lines.append(f"- `{row['strategy_name']}` N{row['n']}: {', '.join(flags)}")
    else:
        lines.append("- No batch28 candidate became a new combined108 best.")
    lines.extend(["", "## Batch28 Pareto Non-Dominated Cases"])
    pareto = [row for row in combined108 if row["dataset_source"] == "batch28_run20" and row.get("combined108_pareto_flag_u2_peeq_surfaceT_mises")]
    if pareto:
        for row in pareto:
            lines.append(f"- `{row['strategy_name']}` N{row['n']}: 4-objective Pareto non-dominated.")
    else:
        lines.append("- No batch28 cases are 4-objective Pareto non-dominated.")
    lines.extend(["", "## Tradeoff Notes"])
    for row in batch28:
        if row["u2_rank_batch28_within_n"] <= 2 and (row["peeq_rank_batch28_within_n"] > len([r for r in batch28 if r["n"] == row["n"]]) / 2 or row["surfaceT_rank_batch28_within_n"] > len([r for r in batch28 if r["n"] == row["n"]]) / 2):
            lines.append(f"- `{row['strategy_name']}` N{row['n']}: strong batch28 U2 rank with weaker PEEQ/SurfaceT rank.")
    (OUTPUT_DIR / "batch28_highlight_cases.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_claim_boundary() -> tuple[Path, Path]:
    md = OUTPUT_DIR / "run21_claim_boundary.md"
    js = OUTPUT_DIR / "run21_claim_boundary.json"
    md.write_text(
        "\n".join(
            [
                "# Run21 Claim Boundary",
                "",
                "## Safe claims",
                "- Batch28 ODB teacher metrics were ingested successfully for 28/28 cases.",
                "- Combined teacher-labelled dataset now contains 108 cases.",
                "- Combined108 contains N12=24, N16=24, N24=30, and N40=30.",
                "- Batch28 can be compared against combined80 using within-N ranks.",
                "- Combined108 rankings and normalized costs are ready for updated surrogate/RL analysis.",
                f"- `{SPECIAL_CASE}` is teacher-valid and can be included in RL/analysis.",
                "- Specific metric-level improvements can be claimed only if confirmed by combined108 comparison.",
                "",
                "## Unsafe claims",
                "- Do not claim trained variable-N RL policy superiority.",
                "- Do not claim surrogate predictions are ground truth.",
                "- Do not claim arbitrary-N generalization.",
                "- Do not claim fixed-32 U2 guard transfer.",
                "- Do not claim final optimum.",
                "- Do not claim physical superiority except for explicitly supported metric-level teacher comparisons.",
                "",
                "Verdict: RUN21_BATCH28_INGESTION_AND_COMBINED108_DATASET_ONLY_NO_RL_POLICY_TRAINING",
                "",
            ]
        ),
        encoding="utf-8",
    )
    write_json(
        js,
        {
            "verdict": "RUN21_BATCH28_INGESTION_AND_COMBINED108_DATASET_ONLY_NO_RL_POLICY_TRAINING",
            "safe_claims": [
                "batch28 metrics ingested 28/28",
                "combined108 teacher-labelled dataset built",
                "within-N ranks and normalized costs ready",
                f"{SPECIAL_CASE} is teacher-valid",
            ],
            "unsafe_claims": [
                "trained variable-N RL policy superiority",
                "surrogate predictions as ground truth",
                "arbitrary-N generalization",
                "fixed-32 U2 guard transfer",
                "final optimum",
            ],
        },
    )
    return md, js


def write_report(validation: dict[str, Any], batch28: list[dict[str, Any]], audit: list[dict[str, Any]], comparison: list[dict[str, Any]], combined108: list[dict[str, Any]], outputs: list[str]) -> None:
    lines = [
        "# Stage 3 Run 21 - Batch28 Teacher Metrics Ingestion and Combined108 Ranking",
        "",
        "## Purpose",
        "Ingest official batch28 teacher metrics, audit surrogate predictions, compare against combined80, and build a combined108 RL-ready teacher-labelled dataset.",
        "",
        "## Inputs",
        f"- Batch28 metrics: `{BATCH28_METRICS}`",
        f"- Run19 batch28 handoff: `{RUN19_BATCH28}`",
        f"- Previous combined80 dataset: `{COMBINED80_TEACHER}`",
        "",
        "## Batch28 Validation Status",
        f"- Verdict: `{validation['verdict']}`",
        f"- Per-N counts: `{validation['per_n_counts']}`",
        f"- Solver completion verdict: `{validation.get('solver_completion_verdict')}`",
        f"- Extraction verdict: `{validation.get('extraction_verdict')}`",
        "",
        "## Batch28 Within-N Ranking",
    ]
    for n in EXPECTED_N:
        g = [row for row in batch28 if row["n"] == n]
        lines.append(f"- N{n}: best U2 `{min(g, key=lambda r: r['u2_range'])['strategy_name']}`, best reward `{max(g, key=lambda r: r['reward_batch28_u2_primary'])['strategy_name']}`.")
    overall_sp = audit[0]["overall_spearman_across_batch28"] if audit else math.nan
    lines.extend(
        [
            "",
            "## Surrogate Prediction Audit",
            f"- Overall Spearman predicted vs realized batch28 reward: `{overall_sp}`.",
            "- This is diagnostic calibration evidence only.",
            "",
            "## Batch28 vs Combined80 Best Comparison",
        ]
    )
    for row in comparison:
        wins = [label for label in ["u2", "peeq", "surfaceT", "mises"] if row[f"batch28_beats_combined80_best_{label}"]]
        lines.append(f"- N{row['n']}: batch28 beats combined80 best metrics: `{wins or []}`.")
    lines.extend(
        [
            "",
            "## Combined108 Teacher Dataset Construction",
            f"- Total rows: `{len(combined108)}`",
            f"- Per-N rows: `{dict(Counter(row['n'] for row in combined108))}`",
            "",
            "## Claim Boundary",
            "RUN21_BATCH28_INGESTION_AND_COMBINED108_DATASET_ONLY_NO_RL_POLICY_TRAINING. Do not claim trained RL superiority, arbitrary-N generalization, or physical superiority beyond teacher-supported metric-level comparisons.",
            "",
            "## Output Files",
        ]
    )
    lines.extend(f"- `{path}`" for path in outputs)
    lines.extend(
        [
            "",
            "## Recommended Run22",
            "Use `combined108_RL_ready_dataset.csv` to update lightweight surrogate validation again. Re-run leave-N-out validation and compare against run17, with special attention to N40 stability. Do not train final RL policy unless explicitly instructed.",
            "",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def update_run_index() -> None:
    row = "| run_21 | Batch28 teacher metrics ingestion and combined108 ranking | Ingest 28 batch28 teacher metrics, audit surrogate predictions, compare against combined80, and build combined108 RL-ready dataset. | `scripts/stage3/run_21_ingest_batch28_and_build_combined108_teacher_dataset.py` | `docs/stage3/runs/run_21_batch28_teacher_metrics_ingestion_and_combined108_ranking/RUN_21_BATCH28_TEACHER_METRICS_INGESTION_AND_COMBINED108_RANKING_REPORT.md` | `outputs/stage3_run_21_batch28_teacher_metrics_ingestion_and_combined108_ranking/` | `PASS_RUN21_BATCH28_TEACHER_METRICS_28_OF_28_READY` | No Abaqus, no ODB opening, no abqjobpilot, no CAE/INP/JNL generation, no final RL policy training, no commit/push. Next: run22 combined108 surrogate validation update. |"
    if RUN_INDEX_PATH.exists():
        text = RUN_INDEX_PATH.read_text(encoding="utf-8")
        if "| run_21 |" not in text:
            RUN_INDEX_PATH.write_text(text.rstrip() + "\n" + row + "\n", encoding="utf-8")


def write_plots(combined108: list[dict[str, Any]], batch28: list[dict[str, Any]]) -> list[str]:
    paths: list[str] = []
    try:
        import matplotlib.pyplot as plt

        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        colors = {"probe60_run08": "tab:blue", "batch20_run14": "tab:orange", "batch28_run20": "tab:green"}
        fig, ax = plt.subplots(figsize=(7, 5))
        for source, group in defaultdict(list, {s: [r for r in combined108 if r["dataset_source"] == s] for s in set(r["dataset_source"] for r in combined108)}).items():
            ax.scatter([r["u2_range"] for r in group], [r["peeq_max"] for r in group], s=22, alpha=0.7, label=source, color=colors.get(source))
        ax.set_xlabel("U2 range")
        ax.set_ylabel("PEEQ max")
        ax.set_title("Combined108 U2 vs PEEQ")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = FIGURE_DIR / "combined108_u2_vs_peeq_by_source.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        paths.append(str(p))

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter([r["predicted_reward_combined80_u2_primary"] for r in batch28], [r["reward_batch28_u2_primary"] for r in batch28], s=24)
        ax.set_xlabel("Predicted combined80 reward")
        ax.set_ylabel("Realized batch28 reward")
        ax.set_title("Batch28 predicted vs realized reward")
        fig.tight_layout()
        p = FIGURE_DIR / "batch28_predicted_vs_realized_reward.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        paths.append(str(p))
    except Exception as exc:  # noqa: BLE001
        write_json(OUTPUT_DIR / "run21_plotting_warning.json", {"plotting_warning": str(exc)})
    return paths


def git_branch() -> str:
    try:
        result = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception:
        return ""


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    metrics_rows = read_csv(BATCH28_METRICS)
    validation = validate_batch28(metrics_rows)
    if validation["verdict"].startswith("FAIL"):
        print(validation["verdict"])
        return 2

    batch28, optional_meta = canonicalize_batch28(metrics_rows)
    add_within_batch_ranking(batch28, "batch28")
    audit = realized_prediction_audit(batch28)
    combined80 = [normalize_combined80_row(row) for row in read_csv(COMBINED80_TEACHER)]
    comparison = compare_batch28_combined80(batch28, combined80)
    combined108 = build_combined108(combined80, batch28)
    ready = rl_ready(combined108)
    board = leaderboard(combined108)
    highlight_cases(batch28, combined108, comparison)
    claim_md, claim_json = write_claim_boundary()

    outputs: list[str] = []
    for filename, rows in [
        ("batch28_teacher_metrics_canonical.csv", batch28),
        ("batch28_within_N_rank_table.csv", batch28),
        ("batch28_surrogate_prediction_audit.csv", audit),
        ("batch28_vs_combined80_best_comparison.csv", comparison),
        ("combined108_teacher_dataset.csv", combined108),
        ("combined108_RL_ready_dataset.csv", ready),
        ("combined108_per_N_leaderboard.csv", board),
    ]:
        path = OUTPUT_DIR / filename
        write_csv(path, rows)
        outputs.append(str(path))
    for filename, rows in [
        ("batch28_teacher_metrics_canonical.json", batch28),
        ("batch28_within_N_rank_table.json", batch28),
        ("batch28_surrogate_prediction_audit.json", audit),
        ("batch28_vs_combined80_best_comparison.json", comparison),
        ("combined108_teacher_dataset.json", combined108),
        ("combined108_RL_ready_dataset.json", ready),
        ("combined108_per_N_leaderboard.json", board),
    ]:
        path = OUTPUT_DIR / filename
        write_table_json(path, rows)
        outputs.append(str(path))
    outputs.extend([str(OUTPUT_DIR / "run21_batch28_input_validation_summary.json"), str(OUTPUT_DIR / "batch28_highlight_cases.md"), str(claim_md), str(claim_json)])
    outputs.extend(write_plots(combined108, batch28))
    write_report(validation, batch28, audit, comparison, combined108, outputs)
    outputs.append(str(REPORT_PATH))
    update_run_index()

    manifest = {
        "run_id": RUN_ID,
        "run_name": RUN_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "branch": git_branch(),
        "script_path": str(ROOT / "scripts" / "stage3" / "run_21_ingest_batch28_and_build_combined108_teacher_dataset.py"),
        "input_files": [str(p) for p in [BATCH28_METRICS, BATCH28_EXTRACTION, BATCH28_SOLVER, BATCH28_SUMMARY, BATCH28_REPORT, RUN19_BATCH28, RUN18_BATCH28, RUN18_SCORED, COMBINED80_TEACHER, COMBINED80_READY, COMBINED80_LEADERBOARD, RUN17_REPORT, RUN18_REPORT, RUN19_REPORT] if p.exists()],
        "output_files": outputs,
        "validation_verdict": validation["verdict"],
        "total_batch28_rows": len(batch28),
        "total_combined_rows": len(combined108),
        "per_N_combined_rows": dict(Counter(row["n"] for row in combined108)),
        "optional_metadata_summary": optional_meta,
        "report_path": str(REPORT_PATH),
        "claim_boundary_path": str(claim_md),
        "no_solver_run": True,
        "no_odb_opened": True,
        "no_abqjobpilot_run": True,
        "no_cae_inp_generated": True,
        "no_rl_policy_training": True,
        "no_commit_or_push": True,
    }
    write_json(MANIFEST_PATH, manifest)

    print(validation["verdict"])
    print(f"batch28={len(batch28)} per_n={dict(Counter(row['n'] for row in batch28))}")
    print(f"combined108={len(combined108)} per_n={dict(Counter(row['n'] for row in combined108))}")
    print(f"overall_spearman={audit[0]['overall_spearman_across_batch28'] if audit else math.nan}")
    print(f"report={REPORT_PATH}")
    print(f"manifest={MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
