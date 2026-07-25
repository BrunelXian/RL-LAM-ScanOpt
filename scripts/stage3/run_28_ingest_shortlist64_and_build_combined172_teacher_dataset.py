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
RUN_ID = "run_28_shortlist64_teacher_metrics_ingestion_and_combined172_ranking"
RUN_NAME = "shortlist64 teacher metrics ingestion and combined172 ranking"

RUN27_METRICS = ROOT / "outputs" / "stage3_run_27_shortlist64_odb_teacher_validation" / "run27_shortlist64_teacher_metrics.csv"
RUN27_EXTRACTION = ROOT / "outputs" / "stage3_run_27_shortlist64_odb_teacher_validation" / "run27_shortlist64_odb_extraction_summary.csv"
RUN27_SOLVER = ROOT / "outputs" / "stage3_run_27_shortlist64_odb_teacher_validation" / "run27_shortlist64_solver_completion_audit.csv"
RUN27_SUMMARY = ROOT / "outputs" / "stage3_run_27_shortlist64_odb_teacher_validation" / "run27_shortlist64_odb_teacher_validation_summary.json"
RUN27_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_27_shortlist64_odb_teacher_validation" / "RUN_27_SHORTLIST64_ODB_TEACHER_VALIDATION_REPORT.md"
RUN27_MANIFEST = ROOT / "artifacts" / "manifests" / "stage3_run_27_manifest.json"

RUN24_HANDOFF = ROOT / "outputs" / "stage3_run_24_run23_shortlist64_active_learning_handoff_package" / "stage3_run24_shortlist64_candidate_orders.csv"
RUN24_REVIEW = ROOT / "outputs" / "stage3_run_24_run23_shortlist64_active_learning_handoff_package" / "shortlist64_review_summary.csv"
RUN23_SCORED = ROOT / "outputs" / "stage3_run_23_combined108_active_learning_coverage_calibration_design" / "run23_candidate_pool_scored.csv"
RUN23_SHORTLIST = ROOT / "outputs" / "stage3_run_23_combined108_active_learning_coverage_calibration_design" / "run23_candidate_shortlist64.csv"

COMBINED108_READY = ROOT / "outputs" / "stage3_run_21_batch28_teacher_metrics_ingestion_and_combined108_ranking" / "combined108_RL_ready_dataset.csv"
COMBINED108_TEACHER = ROOT / "outputs" / "stage3_run_21_batch28_teacher_metrics_ingestion_and_combined108_ranking" / "combined108_teacher_dataset.csv"
COMBINED108_LEADERBOARD = ROOT / "outputs" / "stage3_run_21_batch28_teacher_metrics_ingestion_and_combined108_ranking" / "combined108_per_N_leaderboard.csv"
RUN22_RESULTS = ROOT / "outputs" / "stage3_run_22_combined108_surrogate_reward_model_validation_update" / "combined108_surrogate_validation_results_detailed.csv"
RUN26_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_26_combined108_gnn_graph_pointer_policy_candidate_generation" / "RUN_26_COMBINED108_GNN_GRAPH_POINTER_POLICY_CANDIDATE_GENERATION_REPORT.md"

OUTPUT_DIR = ROOT / "outputs" / "stage3_run_28_shortlist64_teacher_metrics_ingestion_and_combined172_ranking"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = ROOT / "docs" / "stage3" / "runs" / RUN_ID
REPORT_PATH = REPORT_DIR / "RUN_28_SHORTLIST64_TEACHER_METRICS_INGESTION_AND_COMBINED172_RANKING_REPORT.md"
MANIFEST_PATH = ROOT / "artifacts" / "manifests" / "stage3_run_28_manifest.json"
RUN_INDEX_PATH = ROOT / "docs" / "stage3" / "STAGE3_RUN_INDEX.md"

EXPECTED_N = [12, 16, 24, 40]
EXPECTED_RUN27_COUNTS = {12: 8, 16: 8, 24: 24, 40: 24}
EXPECTED_COMBINED108_COUNTS = {12: 24, 16: 24, 24: 30, 40: 30}
EXPECTED_COMBINED172_COUNTS = {12: 32, 16: 32, 24: 54, 40: 54}
PATCHED_N40_CASES = [
    "S3R24L64_N40_B02",
    "S3R24L64_N40_B03",
    "S3R24L64_N40_B04",
    "S3R24L64_N40_B05",
]
METRICS = [
    ("u2", "u2_range", True),
    ("peeq", "peeq_max", True),
    ("surfaceT", "surface_t_proxy", True),
    ("mises", "mises_max", True),
]


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
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def write_table_json(path: Path, rows: list[dict[str, Any]]) -> None:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    write_json(path, {"schema": "columns_and_rows", "columns": columns, "rows": [[row.get(col) for col in columns] for row in rows]})


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


def boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def safe_divide(num: float, den: float, default: float = 0.0) -> float:
    if den == 0 or not math.isfinite(den):
        return default
    result = num / den
    return result if math.isfinite(result) else default


def mean(values: list[float], default: float = math.nan) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return statistics.fmean(vals) if vals else default


def median(values: list[float], default: float = math.nan) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return statistics.median(vals) if vals else default


def rank_ascending(rows: list[dict[str, Any]], metric: str, rank_col: str) -> None:
    sorted_rows = sorted(rows, key=lambda row: (row[metric], row.get("strategy_name", "")))
    i = 0
    while i < len(sorted_rows):
        j = i + 1
        while j < len(sorted_rows) and sorted_rows[j][metric] == sorted_rows[i][metric]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            sorted_rows[k][rank_col] = avg_rank
        i = j


def rank_descending(rows: list[dict[str, Any]], metric: str, rank_col: str) -> None:
    sorted_rows = sorted(rows, key=lambda row: (-row[metric], row.get("strategy_name", "")))
    i = 0
    while i < len(sorted_rows):
        j = i + 1
        while j < len(sorted_rows) and sorted_rows[j][metric] == sorted_rows[i][metric]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            sorted_rows[k][rank_col] = avg_rank
        i = j


def add_minmax_cost(rows: list[dict[str, Any]], metric: str, col: str) -> None:
    vals = [row[metric] for row in rows if math.isfinite(row[metric])]
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

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda idx: values[idx])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i + 1
            while j < len(order) and values[order[j]] == values[order[i]]:
                j += 1
            avg_rank = (i + 1 + j) / 2.0
            for k in range(i, j):
                out[order[k]] = avg_rank
            i = j
        return out

    rx = ranks([p[0] for p in pairs])
    ry = ranks([p[1] for p in pairs])
    mx, my = mean(rx), mean(ry)
    den = math.sqrt(sum((x - mx) ** 2 for x in rx) * sum((y - my) ** 2 for y in ry))
    return safe_divide(sum((x - mx) * (y - my) for x, y in zip(rx, ry)), den, default=math.nan)


def surface_from_run27(row: dict[str, str]) -> tuple[float, float]:
    pa = parse_float(row.get("surface_t_proxy_max_tensile_pa"))
    mpa = parse_float(row.get("surface_t_proxy_max_tensile_mpa"))
    if not math.isfinite(pa):
        pa = parse_float(row.get("surface_t_proxy"))
    if not math.isfinite(pa) and math.isfinite(mpa):
        pa = mpa * 1_000_000.0
    if not math.isfinite(mpa) and math.isfinite(pa):
        mpa = pa / 1_000_000.0
    return pa, mpa


def normalize_name(row: dict[str, str]) -> str:
    return row.get("handoff_strategy_name") or row.get("strategy_name") or row.get("job_name", "").replace("J2D_", "")


def git_branch() -> str:
    try:
        result = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception:
        return ""


def validate_inputs(run27_rows: list[dict[str, str]], combined108_rows: list[dict[str, str]]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    run27_counts: Counter[int] = Counter()
    names_by_n: dict[int, set[str]] = defaultdict(set)
    patched_present: dict[str, bool] = {prefix: False for prefix in PATCHED_N40_CASES}

    for row in run27_rows:
        name = normalize_name(row)
        try:
            n = parse_int(row.get("n"))
        except Exception:
            errors.append(f"{name}: invalid n={row.get('n')}")
            continue
        run27_counts[n] += 1
        if not name:
            errors.append("Run27 row missing strategy/handoff name")
        if name in names_by_n[n]:
            errors.append(f"Duplicate Run27 strategy within N{n}: {name}")
        names_by_n[n].add(name)
        if "PASS" not in row.get("teacher_validation_status", "").upper():
            errors.append(f"{name}: non-pass teacher_validation_status={row.get('teacher_validation_status')}")
        if row.get("final_step_name") and row.get("final_step_name") != "step_final_cooling":
            errors.append(f"{name}: final step is {row.get('final_step_name')}")
        fields = row.get("extracted_field_names", "")
        if fields:
            for field in ["U", "PEEQ", "S", "NT11"]:
                if field not in {part.strip() for part in fields.replace(",", ";").split(";")}:
                    errors.append(f"{name}: missing extracted field {field}")
        surf, _ = surface_from_run27(row)
        for col, val in [
            ("u2_range", parse_float(row.get("u2_range"))),
            ("peeq_max", parse_float(row.get("peeq_max"))),
            ("surface_t_proxy", surf),
            ("mises_max", parse_float(row.get("mises_max"))),
        ]:
            if not math.isfinite(val):
                errors.append(f"{name}: missing metric {col}")
        for prefix in PATCHED_N40_CASES:
            if name.startswith(prefix):
                patched_present[prefix] = True

    if len(run27_rows) != 64:
        errors.append(f"Expected 64 Run27 rows, found {len(run27_rows)}")
    for n, expected in EXPECTED_RUN27_COUNTS.items():
        if run27_counts[n] != expected:
            errors.append(f"Expected Run27 N{n}={expected}, found {run27_counts[n]}")
    for prefix, present in patched_present.items():
        if not present:
            warnings.append(f"Patched N40 case prefix not found: {prefix}")

    combined_counts: Counter[int] = Counter()
    for row in combined108_rows:
        try:
            combined_counts[parse_int(row.get("n"))] += 1
        except Exception:
            errors.append(f"Combined108 row has invalid N: {row.get('n')}")
    if len(combined108_rows) != 108:
        errors.append(f"Expected 108 combined108 rows, found {len(combined108_rows)}")
    for n, expected in EXPECTED_COMBINED108_COUNTS.items():
        if combined_counts[n] != expected:
            errors.append(f"Expected combined108 N{n}={expected}, found {combined_counts[n]}")

    handoff_rows = read_csv(RUN24_HANDOFF) if RUN24_HANDOFF.exists() else []
    handoff_names = {row.get("handoff_strategy_name") for row in handoff_rows}
    direct_matches = 0
    fallback_matches = 0
    unmatched: list[str] = []
    for row in run27_rows:
        name = normalize_name(row)
        if name in handoff_names:
            direct_matches += 1
        elif name.replace("J2D_", "") in handoff_names:
            fallback_matches += 1
        else:
            unmatched.append(name)
    if unmatched:
        errors.append(f"{len(unmatched)} Run27 rows did not match Run24 metadata")

    summary_payload: dict[str, Any] = {}
    if RUN27_SUMMARY.exists():
        summary_payload = json.loads(RUN27_SUMMARY.read_text(encoding="utf-8"))
        audit = summary_payload.get("audit_summary", {})
        extraction = summary_payload.get("extraction_summary", {})
        if audit.get("total_complete") not in (None, 64):
            errors.append(f"Run27 summary total_complete={audit.get('total_complete')}")
        if audit.get("total_lck_present", 0) not in (None, 0):
            errors.append(f"Run27 summary total_lck_present={audit.get('total_lck_present')}")
        if extraction.get("total_pass") not in (None, 64):
            errors.append(f"Run27 summary total_pass={extraction.get('total_pass')}")

    verdict = "PASS_RUN28_SHORTLIST64_TEACHER_METRICS_64_OF_64_READY" if not errors else "FAIL_RUN28_INPUT_VALIDATION"
    payload = {
        "verdict": verdict,
        "errors": errors,
        "warnings": warnings,
        "run27_row_count": len(run27_rows),
        "run27_per_n_counts": dict(sorted(run27_counts.items())),
        "combined108_row_count": len(combined108_rows),
        "combined108_per_n_counts": dict(sorted(combined_counts.items())),
        "run24_metadata_direct_matches": direct_matches,
        "run24_metadata_fallback_matches": fallback_matches,
        "run24_metadata_unmatched": unmatched,
        "patched_n40_case_prefixes_present": patched_present,
        "run27_summary_verdict": summary_payload.get("extraction_summary", {}).get("verdict"),
        "solver_audit_status": summary_payload.get("audit_summary", {}).get("audit_status"),
        "lck_count": summary_payload.get("audit_summary", {}).get("total_lck_present"),
        "important_boundary": "Run27 validates Run23/Run24 active-learning shortlist64 only; it is not Run26 GNN-policy validation.",
    }
    write_json(OUTPUT_DIR / "run28_input_validation_summary.json", payload)
    return payload


def load_metadata() -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]], dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    handoff = {row.get("handoff_strategy_name", ""): row for row in read_csv(RUN24_HANDOFF)} if RUN24_HANDOFF.exists() else {}
    run23_scored = {row.get("candidate_id", ""): row for row in read_csv(RUN23_SCORED)} if RUN23_SCORED.exists() else {}
    run23_shortlist = {row.get("candidate_id", ""): row for row in read_csv(RUN23_SHORTLIST)} if RUN23_SHORTLIST.exists() else {}
    solver = {normalize_name(row): row for row in read_csv(RUN27_SOLVER)} if RUN27_SOLVER.exists() else {}
    return handoff, run23_scored, run23_shortlist, solver


def canonicalize_run27(run27_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    handoff, scored_by_id, shortlist_by_id, solver_by_name = load_metadata()
    missing = Counter()
    enriched: list[dict[str, Any]] = []
    for row in run27_rows:
        name = normalize_name(row)
        meta = handoff.get(name, {})
        if not meta:
            missing["run24_handoff_metadata"] += 1
        original_id = meta.get("original_run23_candidate_id") or meta.get("candidate_id") or ""
        scored = scored_by_id.get(original_id, {})
        shortlist = shortlist_by_id.get(original_id, {})
        solver = solver_by_name.get(name, {})
        surf_pa, surf_mpa = surface_from_run27(row)
        order_json = meta.get("order_json") or scored.get("order_json") or shortlist.get("order_json") or ""
        order_hash = meta.get("order_hash") or scored.get("order_hash") or shortlist.get("order_hash") or ""
        enriched.append(
            {
                "n": parse_int(row.get("n")),
                "strategy_name": name,
                "handoff_strategy_name": name,
                "job_name": row.get("job_name", ""),
                "dataset_source": "shortlist64_run27",
                "original_run23_candidate_id": original_id,
                "candidate_family": meta.get("candidate_family") or scored.get("candidate_family") or shortlist.get("candidate_family", ""),
                "candidate_source": meta.get("candidate_source") or scored.get("candidate_source") or shortlist.get("candidate_source", ""),
                "generation_method": meta.get("generation_method") or scored.get("generation_method") or shortlist.get("generation_method", ""),
                "selection_bucket": meta.get("selection_bucket") or scored.get("selection_bucket") or shortlist.get("selection_bucket", ""),
                "priority_role": meta.get("priority_role") or scored.get("priority_role") or shortlist.get("priority_role", ""),
                "order_json": order_json,
                "order_compact": meta.get("order_compact") or scored.get("order_compact") or shortlist.get("order_compact", ""),
                "order_hash": order_hash,
                "u2_range": parse_float(row.get("u2_range")),
                "peeq_max": parse_float(row.get("peeq_max")),
                "surface_t_proxy": surf_pa,
                "surface_t_proxy_mpa": surf_mpa,
                "mises_max": parse_float(row.get("mises_max")),
                "final_step": row.get("final_step_name", ""),
                "final_frame_time": parse_float(row.get("final_frame_time")),
                "extracted_fields": row.get("extracted_field_names", ""),
                "teacher_validation_status": row.get("teacher_validation_status", ""),
                "solver_audit_status": solver.get("solver_audit_status") or solver.get("audit_status", ""),
                "completion_status": row.get("completion_status", solver.get("completion_status", "")),
                "odb_extraction_status": row.get("odb_extraction_status", ""),
                "nonfatal_warning_flag": boolish(solver.get("nonfatal_warning_marker", "")) or "WARNING" in row.get("completion_status", "").upper(),
                "patched_after_cool_initialInc_failure": boolish(row.get("patched_after_cool_initialInc_failure", "")),
                "pred_reward_ET_F01": parse_float(meta.get("pred_reward_ET_F01", scored.get("pred_reward_ET_F01"))),
                "pred_reward_Ridge_F03": parse_float(meta.get("pred_reward_Ridge_F03", scored.get("pred_reward_Ridge_F03"))),
                "pred_reward_Ridge_F06": parse_float(meta.get("pred_reward_Ridge_F06", scored.get("pred_reward_Ridge_F06"))),
                "pred_reward_ET_F04": parse_float(meta.get("pred_reward_ET_F04", scored.get("pred_reward_ET_F04"))),
                "pred_reward_ET_F05": parse_float(meta.get("pred_reward_ET_F05", scored.get("pred_reward_ET_F05"))),
                "model_prediction_mean": parse_float(meta.get("model_prediction_mean", scored.get("model_prediction_mean"))),
                "model_prediction_std": parse_float(meta.get("model_prediction_std", scored.get("model_prediction_std"))),
                "pred_uncertainty_ET_F01_std": parse_float(meta.get("pred_uncertainty_ET_F01_std", scored.get("pred_uncertainty_ET_F01_std"))),
                "disagreement_ET_F01_vs_Ridge_F03": parse_float(meta.get("disagreement_ET_F01_vs_Ridge_F03", scored.get("disagreement_ET_F01_vs_Ridge_F03"))),
                "disagreement_ET_F01_vs_Ridge_F06": parse_float(meta.get("disagreement_ET_F01_vs_Ridge_F06", scored.get("disagreement_ET_F01_vs_Ridge_F06"))),
                "novelty_distance_to_combined108": parse_float(meta.get("novelty_distance_to_combined108", scored.get("novelty_distance_to_combined108"))),
                "nearest_existing_teacher_strategy": meta.get("nearest_existing_teacher_strategy") or scored.get("nearest_existing_teacher_strategy", ""),
            }
        )
    return enriched, {"missing_optional_metadata_counts": dict(missing)}


def add_within_batch_ranking(rows: list[dict[str, Any]], prefix: str) -> None:
    for n in EXPECTED_N:
        group = [row for row in rows if row["n"] == n]
        for label, metric, _lower_better in METRICS:
            rank_col = f"rank_{label.upper() if label != 'surfaceT' else 'SurfaceT'}_{prefix}_within_n"
            score_col = f"score_{label.upper() if label != 'surfaceT' else 'SurfaceT'}_{prefix}_within_n"
            rank_ascending(group, metric, rank_col)
            add_rank_score(group, rank_col, score_col)
            add_minmax_cost(group, metric, f"cost_{label}_{prefix}_within_n")
        for row in group:
            row[f"reward_{prefix}_u2_primary"] = (
                0.65 * row[f"score_U2_{prefix}_within_n"]
                + 0.20 * row[f"score_PEEQ_{prefix}_within_n"]
                + 0.10 * row[f"score_SurfaceT_{prefix}_within_n"]
                + 0.05 * row[f"score_MISES_{prefix}_within_n"]
            )
        rank_descending(group, f"reward_{prefix}_u2_primary", f"rank_reward_{prefix}_within_n")


def run27_leaderboard(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    board: list[dict[str, Any]] = []
    for n in EXPECTED_N:
        group = [row for row in rows if row["n"] == n]
        for label, metric, lower_better in METRICS:
            ordered = sorted(group, key=lambda row: row[metric], reverse=not lower_better)
            for idx, row in enumerate(ordered[:5], start=1):
                board.append({"n": n, "category": f"top5_{label}", "position": idx, "strategy_name": row["strategy_name"], "dataset_source": row["dataset_source"], "value": row[metric], "selection_bucket": row.get("selection_bucket", ""), "priority_role": row.get("priority_role", "")})
        ordered_reward = sorted(group, key=lambda row: row["reward_run27_u2_primary"], reverse=True)
        for idx, row in enumerate(ordered_reward[:5], start=1):
            board.append({"n": n, "category": "top5_reward_run27_u2_primary", "position": idx, "strategy_name": row["strategy_name"], "dataset_source": row["dataset_source"], "value": row["reward_run27_u2_primary"], "selection_bucket": row.get("selection_bucket", ""), "priority_role": row.get("priority_role", "")})
    return board


def normalize_combined108_row(row: dict[str, str]) -> dict[str, Any]:
    surf = parse_float(row.get("surface_t_proxy"))
    surf_mpa = parse_float(row.get("surface_t_proxy_mpa"))
    if not math.isfinite(surf) and math.isfinite(surf_mpa):
        surf = surf_mpa * 1_000_000.0
    if not math.isfinite(surf_mpa) and math.isfinite(surf):
        surf_mpa = surf / 1_000_000.0
    source = row.get("dataset_source", "")
    return {
        "n": parse_int(row.get("n")),
        "strategy_name": row.get("strategy_name") or row.get("handoff_strategy_name") or "",
        "handoff_strategy_name": row.get("handoff_strategy_name", ""),
        "job_name": row.get("job_name", ""),
        "dataset_source": source,
        "original_run23_candidate_id": row.get("original_run23_candidate_id", ""),
        "candidate_family": row.get("candidate_family", ""),
        "candidate_source": row.get("candidate_source", ""),
        "generation_method": row.get("generation_method", ""),
        "selection_bucket": row.get("selection_bucket", ""),
        "priority_role": row.get("priority_role", ""),
        "order_json": row.get("order_json", ""),
        "order_compact": row.get("order_compact", ""),
        "order_hash": row.get("order_hash", ""),
        "u2_range": parse_float(row.get("u2_range")),
        "peeq_max": parse_float(row.get("peeq_max")),
        "surface_t_proxy": surf,
        "surface_t_proxy_mpa": surf_mpa,
        "mises_max": parse_float(row.get("mises_max")),
        "final_step": row.get("final_step", ""),
        "final_frame_time": parse_float(row.get("final_frame_time")),
        "extracted_fields": row.get("extracted_fields", ""),
        "teacher_validation_status": row.get("teacher_validation_status", ""),
        "solver_audit_status": row.get("solver_audit_status", ""),
        "completion_status": row.get("completion_status", ""),
        "odb_extraction_status": row.get("odb_extraction_status", ""),
        "nonfatal_warning_flag": boolish(row.get("nonfatal_warning_flag", "")),
        "patched_after_cool_initialInc_failure": boolish(row.get("patched_after_cool_initialInc_failure", "")),
        "pred_reward_ET_F01": parse_float(row.get("pred_reward_ET_F01")),
        "pred_reward_Ridge_F03": parse_float(row.get("pred_reward_Ridge_F03")),
        "pred_reward_Ridge_F06": parse_float(row.get("pred_reward_Ridge_F06")),
        "pred_reward_ET_F04": parse_float(row.get("pred_reward_ET_F04")),
        "pred_reward_ET_F05": parse_float(row.get("pred_reward_ET_F05")),
        "model_prediction_mean": parse_float(row.get("model_prediction_mean")),
        "model_prediction_std": parse_float(row.get("model_prediction_std")),
        "pred_uncertainty_ET_F01_std": parse_float(row.get("pred_uncertainty_ET_F01_std")),
        "disagreement_ET_F01_vs_Ridge_F03": parse_float(row.get("disagreement_ET_F01_vs_Ridge_F03")),
        "disagreement_ET_F01_vs_Ridge_F06": parse_float(row.get("disagreement_ET_F01_vs_Ridge_F06")),
        "novelty_distance_to_combined108": parse_float(row.get("novelty_distance_to_combined108")),
        "nearest_existing_teacher_strategy": row.get("nearest_existing_teacher_strategy", ""),
        "is_probe60": source == "probe60_run08",
        "is_batch20": source == "batch20_run14",
        "is_batch28": source == "batch28_run20",
        "is_shortlist64_run27": False,
    }


def build_combined172(combined108: list[dict[str, Any]], run27: list[dict[str, Any]]) -> list[dict[str, Any]]:
    combined = [dict(row) for row in combined108]
    for row in run27:
        new = dict(row)
        new["is_probe60"] = False
        new["is_batch20"] = False
        new["is_batch28"] = False
        new["is_shortlist64_run27"] = True
        combined.append(new)

    counts = Counter(row["n"] for row in combined)
    if len(combined) != 172 or dict(sorted(counts.items())) != EXPECTED_COMBINED172_COUNTS:
        raise RuntimeError(f"Combined172 validation failed: rows={len(combined)} counts={dict(sorted(counts.items()))}")

    for n in EXPECTED_N:
        group = [row for row in combined if row["n"] == n]
        for label, metric, _lower_better in METRICS:
            rank_col = f"{label}_rank_combined172_within_n"
            if label == "surfaceT":
                rank_col = "surfaceT_rank_combined172_within_n"
            score_col = f"target_{label}_score_combined172_rank"
            if label == "surfaceT":
                score_col = "target_surfaceT_score_combined172_rank"
            cost_col = f"{label}_cost_minmax_combined172_within_n"
            if label == "surfaceT":
                cost_col = "surfaceT_cost_minmax_combined172_within_n"
            rank_ascending(group, metric, rank_col)
            add_rank_score(group, rank_col, score_col)
            add_minmax_cost(group, metric, cost_col)
        for row in group:
            row["target_reward_combined172_u2_primary"] = (
                0.65 * row["target_u2_score_combined172_rank"]
                + 0.20 * row["target_peeq_score_combined172_rank"]
                + 0.10 * row["target_surfaceT_score_combined172_rank"]
                + 0.05 * row["target_mises_score_combined172_rank"]
            )
        reward_min = min(r["target_reward_combined172_u2_primary"] for r in group)
        reward_max = max(r["target_reward_combined172_u2_primary"] for r in group)
        for row in group:
            denom = max(1, len(group) - 1)
            row["u2_percentile_combined172_within_n"] = safe_divide(row["u2_rank_combined172_within_n"] - 1.0, denom)
            row["reward_percentile_combined172_within_n"] = safe_divide(row["target_reward_combined172_u2_primary"] - reward_min, reward_max - reward_min, default=0.0)
        rank_descending(group, "target_reward_combined172_u2_primary", "reward_rank_combined172_within_n")
        pareto_flags(group, ["u2_range", "peeq_max"], "combined172_pareto_flag_u2_peeq")
        pareto_flags(group, ["u2_range", "peeq_max", "surface_t_proxy", "mises_max"], "combined172_pareto_flag_u2_peeq_surfaceT_mises")
        for metric_label, metric, _lower_better in METRICS:
            best_value = min(r[metric] for r in group)
            flag_col = f"is_new_best_{metric_label}_within_n"
            if metric_label == "surfaceT":
                flag_col = "is_new_best_surfaceT_within_n"
            for row in group:
                row[flag_col] = row[metric] == best_value and row["dataset_source"] == "shortlist64_run27"
        best_reward = max(r["target_reward_combined172_u2_primary"] for r in group)
        for row in group:
            row["is_new_best_combined_reward_within_n"] = row["target_reward_combined172_u2_primary"] == best_reward and row["dataset_source"] == "shortlist64_run27"
    return combined


def rl_ready_dataset(combined: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keep = [
        "n",
        "strategy_name",
        "dataset_source",
        "order_json",
        "candidate_family",
        "candidate_source",
        "generation_method",
        "selection_bucket",
        "priority_role",
        "u2_range",
        "peeq_max",
        "surface_t_proxy",
        "surface_t_proxy_mpa",
        "mises_max",
        "u2_rank_combined172_within_n",
        "peeq_rank_combined172_within_n",
        "surfaceT_rank_combined172_within_n",
        "mises_rank_combined172_within_n",
        "u2_cost_minmax_combined172_within_n",
        "peeq_cost_minmax_combined172_within_n",
        "surfaceT_cost_minmax_combined172_within_n",
        "mises_cost_minmax_combined172_within_n",
        "target_u2_score_combined172_rank",
        "target_peeq_score_combined172_rank",
        "target_surfaceT_score_combined172_rank",
        "target_mises_score_combined172_rank",
        "target_reward_combined172_u2_primary",
        "reward_rank_combined172_within_n",
        "combined172_pareto_flag_u2_peeq",
        "combined172_pareto_flag_u2_peeq_surfaceT_mises",
        "is_probe60",
        "is_batch20",
        "is_batch28",
        "is_shortlist64_run27",
        "teacher_validation_status",
    ]
    return [{key: row.get(key, "") for key in keep} for row in combined]


def combined_leaderboard(combined: list[dict[str, Any]]) -> list[dict[str, Any]]:
    board: list[dict[str, Any]] = []
    for n in EXPECTED_N:
        group = [row for row in combined if row["n"] == n]
        for label, metric, lower_better in METRICS:
            ordered = sorted(group, key=lambda row: row[metric], reverse=not lower_better)
            for idx, row in enumerate(ordered[:5], start=1):
                board.append({"n": n, "category": f"top5_{label}", "position": idx, "strategy_name": row["strategy_name"], "dataset_source": row["dataset_source"], "value": row[metric], "rank_combined172_within_n": row.get(f"{label}_rank_combined172_within_n", row.get("surfaceT_rank_combined172_within_n"))})
            worst = sorted(group, key=lambda row: row[metric], reverse=lower_better)
            for idx, row in enumerate(worst[:3], start=1):
                board.append({"n": n, "category": f"worst3_{label}", "position": idx, "strategy_name": row["strategy_name"], "dataset_source": row["dataset_source"], "value": row[metric], "rank_combined172_within_n": row.get(f"{label}_rank_combined172_within_n", row.get("surfaceT_rank_combined172_within_n"))})
        ordered_reward = sorted(group, key=lambda row: row["target_reward_combined172_u2_primary"], reverse=True)
        for idx, row in enumerate(ordered_reward[:5], start=1):
            board.append({"n": n, "category": "top5_reward_combined172_u2_primary", "position": idx, "strategy_name": row["strategy_name"], "dataset_source": row["dataset_source"], "value": row["target_reward_combined172_u2_primary"], "rank_combined172_within_n": row["reward_rank_combined172_within_n"]})
    return board


def compare_run27_vs_combined108(combined108: list[dict[str, Any]], run27: list[dict[str, Any]], combined172: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n in EXPECTED_N:
        old = [row for row in combined108 if row["n"] == n]
        new = [row for row in run27 if row["n"] == n]
        c172_old = [row for row in combined172 if row["n"] == n and row["dataset_source"] != "shortlist64_run27"]
        c172_new = [row for row in combined172 if row["n"] == n and row["dataset_source"] == "shortlist64_run27"]
        for label, metric, _lower_better in METRICS:
            old_best = min(old, key=lambda row: row[metric])
            new_best = min(new, key=lambda row: row[metric])
            combined_best = min([row for row in combined172 if row["n"] == n], key=lambda row: row[metric])
            diff = old_best[metric] - new_best[metric]
            rows.append(
                {
                    "n": n,
                    "metric": label,
                    "combined108_best_strategy": old_best["strategy_name"],
                    "combined108_best_source": old_best["dataset_source"],
                    "combined108_best_value": old_best[metric],
                    "run27_best_strategy": new_best["strategy_name"],
                    "run27_best_value": new_best[metric],
                    "run27_beats_combined108_best": new_best[metric] < old_best[metric],
                    "absolute_improvement_lower_is_better": diff,
                    "relative_improvement_pct": safe_divide(diff, old_best[metric], default=math.nan) * 100.0,
                    "combined172_best_strategy": combined_best["strategy_name"],
                    "combined172_best_source": combined_best["dataset_source"],
                    "combined172_best_value": combined_best[metric],
                }
            )
        old_reward_best = max(c172_old, key=lambda row: row["target_reward_combined172_u2_primary"])
        new_reward_best = max(c172_new, key=lambda row: row["target_reward_combined172_u2_primary"])
        combined_reward_best = max([row for row in combined172 if row["n"] == n], key=lambda row: row["target_reward_combined172_u2_primary"])
        rows.append(
            {
                "n": n,
                "metric": "combined_reward",
                "combined108_best_strategy": old_reward_best["strategy_name"],
                "combined108_best_source": old_reward_best["dataset_source"],
                "combined108_best_value": old_reward_best["target_reward_combined172_u2_primary"],
                "run27_best_strategy": new_reward_best["strategy_name"],
                "run27_best_value": new_reward_best["target_reward_combined172_u2_primary"],
                "run27_beats_combined108_best": new_reward_best["target_reward_combined172_u2_primary"] > old_reward_best["target_reward_combined172_u2_primary"],
                "absolute_improvement_higher_is_better": new_reward_best["target_reward_combined172_u2_primary"] - old_reward_best["target_reward_combined172_u2_primary"],
                "relative_improvement_pct": safe_divide(new_reward_best["target_reward_combined172_u2_primary"] - old_reward_best["target_reward_combined172_u2_primary"], old_reward_best["target_reward_combined172_u2_primary"], default=math.nan) * 100.0,
                "combined172_best_strategy": combined_reward_best["strategy_name"],
                "combined172_best_source": combined_reward_best["dataset_source"],
                "combined172_best_value": combined_reward_best["target_reward_combined172_u2_primary"],
            }
        )
    return rows


def prediction_audit(run27_ranked: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    detailed: list[dict[str, Any]] = []
    for row in run27_ranked:
        realized = row["reward_run27_u2_primary"]
        pred = row.get("model_prediction_mean")
        if not math.isfinite(pred):
            pred = row.get("pred_reward_ET_F01")
        detailed.append(
            {
                "n": row["n"],
                "strategy_name": row["strategy_name"],
                "selection_bucket": row.get("selection_bucket", ""),
                "priority_role": row.get("priority_role", ""),
                "predicted_reward": pred,
                "pred_reward_ET_F01": row.get("pred_reward_ET_F01"),
                "model_prediction_mean": row.get("model_prediction_mean"),
                "realized_reward_run27": realized,
                "prediction_error": realized - pred if math.isfinite(pred) else math.nan,
                "absolute_error": abs(realized - pred) if math.isfinite(pred) else math.nan,
                "realized_u2_score_run27": row["score_U2_run27_within_n"],
                "realized_peeq_score_run27": row["score_PEEQ_run27_within_n"],
                "realized_surfaceT_score_run27": row["score_SurfaceT_run27_within_n"],
                "rank_reward_run27_within_n": row["rank_reward_run27_within_n"],
                "rank_u2_run27_within_n": row["rank_U2_run27_within_n"],
                "pred_uncertainty_ET_F01_std": row.get("pred_uncertainty_ET_F01_std"),
                "model_prediction_std": row.get("model_prediction_std"),
                "disagreement_ET_F01_vs_Ridge_F03": row.get("disagreement_ET_F01_vs_Ridge_F03"),
                "disagreement_ET_F01_vs_Ridge_F06": row.get("disagreement_ET_F01_vs_Ridge_F06"),
                "novelty_distance_to_combined108": row.get("novelty_distance_to_combined108"),
            }
        )

    top_region_rows: list[dict[str, Any]] = []
    top5_overlaps: list[int] = []
    top10_overlaps: list[int] = []
    top1_hits: list[bool] = []
    per_n_spearman: dict[str, float] = {}
    for n in EXPECTED_N:
        group = [row for row in detailed if row["n"] == n]
        predicted = sorted(group, key=lambda row: (row["predicted_reward"] if math.isfinite(row["predicted_reward"]) else -999.0), reverse=True)
        realized = sorted(group, key=lambda row: row["realized_reward_run27"], reverse=True)
        k5 = min(5, len(group))
        k10 = min(10, len(group))
        pred5 = {row["strategy_name"] for row in predicted[:k5]}
        real5 = {row["strategy_name"] for row in realized[:k5]}
        pred10 = {row["strategy_name"] for row in predicted[:k10]}
        real10 = {row["strategy_name"] for row in realized[:k10]}
        top5_overlap = len(pred5 & real5)
        top10_overlap = len(pred10 & real10)
        top5_overlaps.append(top5_overlap)
        top10_overlaps.append(top10_overlap)
        top1_hits.append(predicted[0]["strategy_name"] == realized[0]["strategy_name"])
        sp = spearman([row["predicted_reward"] for row in group], [row["realized_reward_run27"] for row in group])
        per_n_spearman[f"N{n}"] = sp
        top_region_rows.append(
            {
                "n": n,
                "candidate_count": len(group),
                "top_predicted_strategy": predicted[0]["strategy_name"],
                "top_predicted_realized_rank": predicted[0]["rank_reward_run27_within_n"],
                "best_realized_strategy": realized[0]["strategy_name"],
                "top1_hit": predicted[0]["strategy_name"] == realized[0]["strategy_name"],
                "top5_overlap": top5_overlap,
                "top5_possible": k5,
                "top10_overlap": top10_overlap,
                "top10_possible": k10,
                "spearman_predicted_vs_realized": sp,
            }
        )

    summary = {
        "overall_spearman_predicted_vs_realized_reward": spearman([row["predicted_reward"] for row in detailed], [row["realized_reward_run27"] for row in detailed]),
        "per_n_spearman_predicted_vs_realized_reward": per_n_spearman,
        "mean_top5_overlap": mean([float(x) for x in top5_overlaps]),
        "mean_top10_overlap": mean([float(x) for x in top10_overlaps]),
        "top1_hits": sum(1 for hit in top1_hits if hit),
        "top1_possible": len(top1_hits),
        "uncertainty_abs_error_spearman": spearman([row["pred_uncertainty_ET_F01_std"] for row in detailed], [row["absolute_error"] for row in detailed]),
        "model_std_abs_error_spearman": spearman([row["model_prediction_std"] for row in detailed], [row["absolute_error"] for row in detailed]),
        "disagreement_f03_abs_error_spearman": spearman([row["disagreement_ET_F01_vs_Ridge_F03"] for row in detailed], [row["absolute_error"] for row in detailed]),
        "disagreement_f06_abs_error_spearman": spearman([row["disagreement_ET_F01_vs_Ridge_F06"] for row in detailed], [row["absolute_error"] for row in detailed]),
        "novelty_realized_reward_spearman": spearman([row["novelty_distance_to_combined108"] for row in detailed], [row["realized_reward_run27"] for row in detailed]),
        "interpretation": "Run23 active-learning prediction audit only; Run27 is not GNN-policy validation.",
    }
    return detailed, top_region_rows, summary


def grouped_performance(run27_ranked: list[dict[str, Any]], combined172: list[dict[str, Any]]) -> list[dict[str, Any]]:
    combined_by_strategy = {row["strategy_name"]: row for row in combined172}
    rows: list[dict[str, Any]] = []
    specs = [
        ("selection_bucket", "selection_bucket"),
        ("priority_role", "priority_role"),
        ("candidate_family", "candidate_family"),
        ("generation_method", "generation_method"),
        ("n", "n"),
    ]
    for group_type, key in specs:
        grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for row in run27_ranked:
            grouped[row.get(key, "")].append(row)
        for value, group in sorted(grouped.items(), key=lambda item: str(item[0])):
            combined_group = [combined_by_strategy[row["strategy_name"]] for row in group if row["strategy_name"] in combined_by_strategy]
            rows.append(
                {
                    "group_type": group_type,
                    "group_value": value,
                    "count": len(group),
                    "median_u2_rank_run27": median([row["rank_U2_run27_within_n"] for row in group]),
                    "best_u2_rank_run27": min(row["rank_U2_run27_within_n"] for row in group),
                    "median_reward_rank_run27": median([row["rank_reward_run27_within_n"] for row in group]),
                    "best_reward_rank_run27": min(row["rank_reward_run27_within_n"] for row in group),
                    "median_u2_rank_combined172": median([row["u2_rank_combined172_within_n"] for row in combined_group]),
                    "best_u2_rank_combined172": min([row["u2_rank_combined172_within_n"] for row in combined_group], default=math.nan),
                    "median_reward_rank_combined172": median([row["reward_rank_combined172_within_n"] for row in combined_group]),
                    "best_reward_rank_combined172": min([row["reward_rank_combined172_within_n"] for row in combined_group], default=math.nan),
                    "top5_u2_run27_count": sum(1 for row in group if row["rank_U2_run27_within_n"] <= 5),
                    "top10_u2_run27_count": sum(1 for row in group if row["rank_U2_run27_within_n"] <= 10),
                    "top5_reward_run27_count": sum(1 for row in group if row["rank_reward_run27_within_n"] <= 5),
                    "top10_reward_run27_count": sum(1 for row in group if row["rank_reward_run27_within_n"] <= 10),
                    "new_combined172_top5_u2_entries": sum(1 for row in combined_group if row["u2_rank_combined172_within_n"] <= 5),
                    "new_combined172_top10_u2_entries": sum(1 for row in combined_group if row["u2_rank_combined172_within_n"] <= 10),
                    "new_combined172_top5_reward_entries": sum(1 for row in combined_group if row["reward_rank_combined172_within_n"] <= 5),
                    "new_combined172_top10_reward_entries": sum(1 for row in combined_group if row["reward_rank_combined172_within_n"] <= 10),
                }
            )
    return rows


def run27_bucket_performance(run27_ranked: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bucket, group in sorted(defaultdict(list, {bucket: [row for row in run27_ranked if row.get("selection_bucket", "") == bucket] for bucket in {row.get("selection_bucket", "") for row in run27_ranked}}).items()):
        if not group:
            continue
        rows.append(
            {
                "selection_bucket": bucket,
                "count": len(group),
                "mean_reward_run27": mean([row["reward_run27_u2_primary"] for row in group]),
                "median_reward_rank_run27": median([row["rank_reward_run27_within_n"] for row in group]),
                "best_reward_rank_run27": min(row["rank_reward_run27_within_n"] for row in group),
                "median_u2_rank_run27": median([row["rank_U2_run27_within_n"] for row in group]),
                "best_u2_rank_run27": min(row["rank_U2_run27_within_n"] for row in group),
                "top5_reward_count": sum(1 for row in group if row["rank_reward_run27_within_n"] <= 5),
                "top5_u2_count": sum(1 for row in group if row["rank_U2_run27_within_n"] <= 5),
            }
        )
    return rows


def combined172_summary(combined172: list[dict[str, Any]], comparison: list[dict[str, Any]], prediction_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": RUN_ID,
        "combined172_rows": len(combined172),
        "per_n_counts": dict(sorted(Counter(row["n"] for row in combined172).items())),
        "dataset_source_counts": dict(sorted(Counter(row["dataset_source"] for row in combined172).items())),
        "new_best_records": [row for row in comparison if row.get("run27_beats_combined108_best")],
        "new_best_count": sum(1 for row in comparison if row.get("run27_beats_combined108_best")),
        "prediction_audit_summary": prediction_summary,
        "claim_boundary": "Run28 ingests completed Run27 active-learning shortlist64 metrics only; it does not validate Run26 GNN-policy candidates.",
    }


def write_gnn_context_note() -> Path:
    path = OUTPUT_DIR / "run28_gnn_context_note.md"
    path.write_text(
        "\n".join(
            [
                "# Run28 GNN Context Note",
                "",
                "Run27 teacher metrics validate the Run23/Run24 active-learning shortlist64 batch, not the Run26 GNN-policy batch.",
                "",
                "Run26 remains an offline GNN / graph-pointer policy prototype with no teacher validation in this run.",
                "",
                "The combined172 dataset produced by Run28 can be used to update both lightweight surrogate models and offline GNN / graph-pointer policy models in a later run.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def write_claim_boundary() -> tuple[Path, Path]:
    md = OUTPUT_DIR / "run28_claim_boundary.md"
    js = OUTPUT_DIR / "run28_claim_boundary.json"
    safe = [
        "Run28 ingests 64/64 teacher-validated Run27 shortlist64 cases.",
        "Run28 builds combined172 with N12=32, N16=32, N24=54, N40=54.",
        "Run28 recomputes within-N ranks and U2-primary rewards.",
        "Run28 evaluates whether Run27 shortlist64 active-learning candidates beat combined108 best cases.",
        "Run28 audits Run23 prediction/calibration against realized Run27 teacher metrics.",
        "Run28 does not validate Run26 GNN-policy candidates.",
    ]
    unsafe = [
        "Run27 is GNN-policy validation.",
        "online RL success.",
        "arbitrary-N generalization.",
        "physical optimum.",
        "deployment readiness.",
        "feature importance or bucket performance is causal.",
        "solver/ODB extraction happened in Run28.",
    ]
    md.write_text(
        "# Run28 Claim Boundary\n\n"
        "## Safe Claims\n"
        + "\n".join(f"- {item}" for item in safe)
        + "\n\n## Unsafe Claims\n"
        + "\n".join(f"- Do not claim {item}" for item in unsafe)
        + "\n",
        encoding="utf-8",
    )
    write_json(js, {"verdict": "RUN28_SHORTLIST64_INGESTION_AND_COMBINED172_RANKING_ONLY_NOT_GNN_VALIDATION", "safe_claims": safe, "unsafe_claims": unsafe})
    return md, js


def write_report(
    validation: dict[str, Any],
    run27_ranked: list[dict[str, Any]],
    combined172: list[dict[str, Any]],
    comparison: list[dict[str, Any]],
    prediction_summary: dict[str, Any],
    bucket_summary: list[dict[str, Any]],
    outputs: list[str],
) -> None:
    lines = [
        "# Stage 3 Run 28 - Shortlist64 Teacher Metrics Ingestion and Combined172 Ranking",
        "",
        "## Purpose",
        "Ingest completed Run27 shortlist64 teacher metrics, merge active-learning metadata, recompute within-N rankings, build combined172, and audit Run23 calibration against realized teacher metrics.",
        "",
        "## Inputs",
        f"- Run27 teacher metrics: `{RUN27_METRICS}`",
        f"- Run24 shortlist64 handoff metadata: `{RUN24_HANDOFF}`",
        f"- Previous combined108 teacher dataset: `{COMBINED108_TEACHER}`",
        f"- Run26 GNN report used only for boundary context: `{RUN26_REPORT}`",
        "",
        "## Run27 Teacher-Validation Status",
        "- User-provided upstream verdict: `PASS_RUN27_SHORTLIST64_ODB_TEACHER_VALIDATION_64_OF_64`.",
        "- Run28 did not open ODB files or perform solver/extraction work; it read the completed CSV/JSON outputs only.",
        "",
        "## Input Validation",
        f"- Verdict: `{validation['verdict']}`",
        f"- Run27 rows: `{validation['run27_row_count']}`; per-N: `{validation['run27_per_n_counts']}`",
        f"- Combined108 rows: `{validation['combined108_row_count']}`; per-N: `{validation['combined108_per_n_counts']}`",
        f"- Run24 metadata direct matches: `{validation['run24_metadata_direct_matches']}`",
        "",
        "## Run27 Enriched Teacher Dataset",
        "The enriched table preserves handoff strategy names, Run23 candidate IDs, bucket metadata, prediction metadata, novelty/disagreement fields, and official teacher metrics.",
        "",
        "## Run27 Within-Batch Ranking",
    ]
    for n in EXPECTED_N:
        group = [row for row in run27_ranked if row["n"] == n]
        lines.append(
            f"- N{n}: best U2 `{min(group, key=lambda r: r['u2_range'])['strategy_name']}`, "
            f"best reward `{max(group, key=lambda r: r['reward_run27_u2_primary'])['strategy_name']}`."
        )
    lines.extend(
        [
            "",
            "## Combined172 Construction",
            f"- Total rows: `{len(combined172)}`",
            f"- Per-N rows: `{dict(sorted(Counter(row['n'] for row in combined172).items()))}`",
            "",
            "## Run27 vs Combined108 Best Comparison",
        ]
    )
    for n in EXPECTED_N:
        wins = [row["metric"] for row in comparison if row["n"] == n and row.get("run27_beats_combined108_best")]
        lines.append(f"- N{n}: Run27 beats combined108 best metrics: `{wins or []}`.")
    lines.extend(
        [
            "",
            "## Prediction Audit for Run23 Active-Learning Design",
            f"- Overall Spearman predicted vs realized Run27 reward: `{prediction_summary['overall_spearman_predicted_vs_realized_reward']}`",
            f"- Mean top5 overlap: `{prediction_summary['mean_top5_overlap']} / 5`",
            f"- Top1 hits: `{prediction_summary['top1_hits']} / {prediction_summary['top1_possible']}`",
            "",
            "## Bucket/Source Performance",
        ]
    )
    best_bucket = min((row for row in bucket_summary if row.get("group_type") == "selection_bucket"), key=lambda row: row["best_reward_rank_run27"], default=None)
    if best_bucket:
        lines.append(f"- Best bucket by Run27 reward rank: `{best_bucket['group_value']}` with best reward rank `{best_bucket['best_reward_rank_run27']}`.")
    lines.extend(
        [
            "",
            "## N24/N40 Focus Analysis",
        ]
    )
    for n in [24, 40]:
        wins = [row["metric"] for row in comparison if row["n"] == n and row.get("run27_beats_combined108_best")]
        group = [row for row in run27_ranked if row["n"] == n]
        lines.append(f"- N{n}: `{len(group)}` Run27 active-learning cases; new-best metrics `{wins or []}`.")
    lines.extend(
        [
            "",
            "## Context Note on Run26 GNN Prototype",
            "Run27 teacher metrics are not GNN-policy validation. They validate the Run23/Run24 active-learning shortlist64 batch only. Combined172 can support a later update of both surrogate and offline GNN / graph-pointer policy models.",
            "",
            "## Claim Boundary",
            "`RUN28_SHORTLIST64_INGESTION_AND_COMBINED172_RANKING_ONLY_NOT_GNN_VALIDATION`.",
            "",
            "## Output Files",
        ]
    )
    lines.extend(f"- `{path}`" for path in outputs)
    lines.extend(
        [
            "",
            "## Recommended Run29",
            "Use the combined172 dataset to update surrogate and offline GNN / graph-pointer policy models. If Run27 produced strong new bests, prepare a refined hybrid-policy candidate batch; if it mainly improved calibration, focus Run29 on top-region model calibration before proposing more solver cases.",
            "",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def update_run_index(validation_verdict: str) -> None:
    if not RUN_INDEX_PATH.exists():
        return
    row = (
        "| run_28 | Shortlist64 teacher metrics ingestion and combined172 ranking | "
        "Ingest Run27 active-learning shortlist64 teacher metrics, audit Run23 prediction calibration, compare against combined108, and build combined172 RL-ready dataset. | "
        "`scripts/stage3/run_28_ingest_shortlist64_and_build_combined172_teacher_dataset.py` | "
        "`docs/stage3/runs/run_28_shortlist64_teacher_metrics_ingestion_and_combined172_ranking/RUN_28_SHORTLIST64_TEACHER_METRICS_INGESTION_AND_COMBINED172_RANKING_REPORT.md` | "
        "`outputs/stage3_run_28_shortlist64_teacher_metrics_ingestion_and_combined172_ranking/` | "
        f"`{validation_verdict}` | "
        "No Abaqus, no ODB opening, no abqjobpilot, no CAE/INP/JNL generation, no online RL, no commit/push. Run27 is active-learning validation, not Run26 GNN-policy validation. |"
    )
    text = RUN_INDEX_PATH.read_text(encoding="utf-8")
    if "| run_28 |" not in text:
        RUN_INDEX_PATH.write_text(text.rstrip() + "\n" + row + "\n", encoding="utf-8")


def write_optional_plots(combined172: list[dict[str, Any]], run27_ranked: list[dict[str, Any]], prediction_detail: list[dict[str, Any]]) -> list[str]:
    paths: list[str] = []
    try:
        import matplotlib.pyplot as plt

        FIGURE_DIR.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        for source in sorted({row["dataset_source"] for row in combined172}):
            group = [row for row in combined172 if row["dataset_source"] == source]
            ax.scatter([row["u2_range"] for row in group], [row["peeq_max"] for row in group], s=18, alpha=0.65, label=source)
        ax.set_xlabel("U2 range")
        ax.set_ylabel("PEEQ max")
        ax.set_title("Combined172 U2 vs PEEQ by source")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = FIGURE_DIR / "combined172_u2_vs_peeq_by_source.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(str(path))

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter([row["predicted_reward"] for row in prediction_detail], [row["realized_reward_run27"] for row in prediction_detail], s=22, alpha=0.75)
        ax.set_xlabel("Run23 predicted reward")
        ax.set_ylabel("Realized Run27 reward")
        ax.set_title("Run27 predicted vs realized reward")
        fig.tight_layout()
        path = FIGURE_DIR / "run27_predicted_vs_realized_reward.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(str(path))

        buckets = sorted({row.get("selection_bucket", "") for row in run27_ranked})
        medians = [median([row["rank_reward_run27_within_n"] for row in run27_ranked if row.get("selection_bucket", "") == bucket]) for bucket in buckets]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(range(len(buckets)), medians)
        ax.set_xticks(range(len(buckets)))
        ax.set_xticklabels(buckets, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Median Run27 reward rank")
        ax.set_title("Run27 bucket reward-rank performance")
        fig.tight_layout()
        path = FIGURE_DIR / "run27_bucket_reward_rank_performance.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(str(path))
    except Exception as exc:  # noqa: BLE001
        write_json(OUTPUT_DIR / "run28_plotting_warning.json", {"plotting_warning": str(exc)})
    return paths


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    run27_rows = read_csv(RUN27_METRICS)
    combined108_rows_raw = read_csv(COMBINED108_TEACHER)
    validation = validate_inputs(run27_rows, combined108_rows_raw)
    if validation["verdict"].startswith("FAIL"):
        print(validation["verdict"])
        return 2

    run27_enriched, optional_meta = canonicalize_run27(run27_rows)
    add_within_batch_ranking(run27_enriched, "run27")
    run27_board = run27_leaderboard(run27_enriched)
    run27_bucket_rows = run27_bucket_performance(run27_enriched)
    combined108 = [normalize_combined108_row(row) for row in combined108_rows_raw]
    combined172 = build_combined172(combined108, run27_enriched)
    ready = rl_ready_dataset(combined172)
    combined_board = combined_leaderboard(combined172)
    comparison = compare_run27_vs_combined108(combined108, run27_enriched, combined172)
    prediction_detail, top_region, prediction_summary = prediction_audit(run27_enriched)
    bucket_source = grouped_performance(run27_enriched, combined172)
    summary = combined172_summary(combined172, comparison, prediction_summary)
    claim_md, claim_json = write_claim_boundary()
    gnn_note = write_gnn_context_note()

    output_files: list[str] = []
    csv_outputs = [
        ("run27_shortlist64_teacher_dataset_enriched.csv", run27_enriched),
        ("run27_shortlist64_ranked_within_batch.csv", run27_enriched),
        ("run27_shortlist64_per_N_leaderboard.csv", run27_board),
        ("run27_shortlist64_bucket_performance.csv", run27_bucket_rows),
        ("combined172_teacher_dataset.csv", combined172),
        ("combined172_RL_ready_dataset.csv", ready),
        ("combined172_per_N_leaderboard.csv", combined_board),
        ("run27_vs_combined108_best_comparison.csv", comparison),
        ("run27_shortlist64_prediction_audit.csv", prediction_detail),
        ("run27_shortlist64_top_region_retrieval_audit.csv", top_region),
        ("run27_bucket_source_performance_summary.csv", bucket_source),
    ]
    for filename, rows in csv_outputs:
        path = OUTPUT_DIR / filename
        write_csv(path, rows)
        output_files.append(str(path))
    json_table_outputs = [
        ("run27_shortlist64_teacher_dataset_enriched.json", run27_enriched),
        ("combined172_teacher_dataset.json", combined172),
        ("run27_vs_combined108_best_comparison.json", comparison),
        ("run27_bucket_source_performance_summary.json", bucket_source),
    ]
    for filename, rows in json_table_outputs:
        path = OUTPUT_DIR / filename
        write_table_json(path, rows)
        output_files.append(str(path))
    write_json(OUTPUT_DIR / "combined172_summary.json", summary)
    write_json(OUTPUT_DIR / "run27_shortlist64_prediction_audit_summary.json", prediction_summary)
    output_files.extend(
        [
            str(OUTPUT_DIR / "run28_input_validation_summary.json"),
            str(OUTPUT_DIR / "combined172_summary.json"),
            str(OUTPUT_DIR / "run27_shortlist64_prediction_audit_summary.json"),
            str(claim_md),
            str(claim_json),
            str(gnn_note),
        ]
    )
    output_files.extend(write_optional_plots(combined172, run27_enriched, prediction_detail))
    write_report(validation, run27_enriched, combined172, comparison, prediction_summary, bucket_source, output_files)
    output_files.append(str(REPORT_PATH))
    update_run_index(validation["verdict"])

    manifest = {
        "run_id": RUN_ID,
        "run_name": RUN_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "branch": git_branch(),
        "script_path": str(ROOT / "scripts" / "stage3" / "run_28_ingest_shortlist64_and_build_combined172_teacher_dataset.py"),
        "input_files": [
            str(path)
            for path in [
                RUN27_METRICS,
                RUN27_EXTRACTION,
                RUN27_SOLVER,
                RUN27_SUMMARY,
                RUN27_REPORT,
                RUN27_MANIFEST,
                RUN24_HANDOFF,
                RUN24_REVIEW,
                RUN23_SCORED,
                RUN23_SHORTLIST,
                COMBINED108_READY,
                COMBINED108_TEACHER,
                COMBINED108_LEADERBOARD,
                RUN22_RESULTS,
                RUN26_REPORT,
            ]
            if path.exists()
        ],
        "output_files": output_files,
        "report_path": str(REPORT_PATH),
        "claim_boundary_path": str(claim_md),
        "validation_verdict": validation["verdict"],
        "run27_teacher_rows": len(run27_enriched),
        "combined172_rows": len(combined172),
        "per_N_combined172_counts": dict(sorted(Counter(row["n"] for row in combined172).items())),
        "new_best_count": summary["new_best_count"],
        "prediction_audit_summary": prediction_summary,
        "optional_metadata_summary": optional_meta,
        "no_solver_run": True,
        "no_odb_opened": True,
        "no_abqjobpilot_run": True,
        "no_cae_inp_generated": True,
        "no_teacher_validation_performed_by_run28": True,
        "no_online_rl": True,
        "no_commit_or_push": True,
    }
    write_json(MANIFEST_PATH, manifest)

    print(validation["verdict"])
    print(f"run27={len(run27_enriched)} per_n={dict(sorted(Counter(row['n'] for row in run27_enriched).items()))}")
    print(f"combined172={len(combined172)} per_n={dict(sorted(Counter(row['n'] for row in combined172).items()))}")
    print(f"new_best_count={summary['new_best_count']}")
    print(f"prediction_spearman={prediction_summary['overall_spearman_predicted_vs_realized_reward']}")
    print(f"report={REPORT_PATH}")
    print(f"manifest={MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
