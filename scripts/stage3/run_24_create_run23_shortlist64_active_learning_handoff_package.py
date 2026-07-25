from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
RUN_ID = "run_24_run23_shortlist64_active_learning_handoff_package"
RUN_NAME = "run23 shortlist64 active-learning handoff package"

RUN23_SCORED = ROOT / "outputs" / "stage3_run_23_combined108_active_learning_coverage_calibration_design" / "run23_candidate_pool_scored.csv"
RUN23_SHORTLIST64 = ROOT / "outputs" / "stage3_run_23_combined108_active_learning_coverage_calibration_design" / "run23_candidate_shortlist64.csv"
RUN23_BATCH32 = ROOT / "outputs" / "stage3_run_23_combined108_active_learning_coverage_calibration_design" / "run23_recommended_active_learning_batch32.csv"
RUN23_BATCH24 = ROOT / "outputs" / "stage3_run_23_combined108_active_learning_coverage_calibration_design" / "run23_conservative_active_learning_batch24.csv"
RUN23_COMPARISON = ROOT / "outputs" / "stage3_run_23_combined108_active_learning_coverage_calibration_design" / "run23_predicted_comparison_vs_combined108.csv"
RUN23_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_23_combined108_active_learning_coverage_calibration_design" / "RUN_23_COMBINED108_ACTIVE_LEARNING_COVERAGE_CALIBRATION_DESIGN_REPORT.md"
COMBINED108_TEACHER = ROOT / "outputs" / "stage3_run_21_batch28_teacher_metrics_ingestion_and_combined108_ranking" / "combined108_teacher_dataset.csv"
COMBINED108_READY = ROOT / "outputs" / "stage3_run_21_batch28_teacher_metrics_ingestion_and_combined108_ranking" / "combined108_RL_ready_dataset.csv"

OUTPUT_DIR = ROOT / "outputs" / "stage3_run_24_run23_shortlist64_active_learning_handoff_package"
SCAN_ORDER_DIR = OUTPUT_DIR / "scan_orders"
REPORT_DIR = ROOT / "docs" / "stage3" / "runs" / RUN_ID
REPORT_PATH = REPORT_DIR / "RUN_24_RUN23_SHORTLIST64_ACTIVE_LEARNING_HANDOFF_PACKAGE_REPORT.md"
MANIFEST_PATH = ROOT / "artifacts" / "manifests" / "stage3_run_24_manifest.json"
RUN_INDEX_PATH = ROOT / "docs" / "stage3" / "STAGE3_RUN_INDEX.md"

EXPECTED_N = [12, 16, 24, 40]
SHORTLIST64_COUNTS = {12: 8, 16: 8, 24: 24, 40: 24}
SELECTED_BATCH = "shortlist64"
BATCH_NAME = "stage3_run24_shortlist64_active_learning_calibration_v01"
FUTURE_CASE_ROOT = ROOT / "cae_model" / BATCH_NAME


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


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def parse_order(text: Any) -> list[int] | None:
    try:
        value = json.loads(str(text))
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(value, list):
        return None
    try:
        return [int(x) for x in value]
    except (TypeError, ValueError):
        return None


def order_hash(order: list[int]) -> str:
    return hashlib.sha1(",".join(str(x) for x in order).encode("ascii")).hexdigest()[:16]


def validate_order(order: list[int] | None, n: int) -> tuple[bool, str]:
    if order is None:
        return False, "missing or unparsable scan order"
    if len(order) != n:
        return False, f"length {len(order)} != N {n}"
    if len(set(order)) != n:
        return False, "duplicate track ids"
    expected = set(range(n))
    actual = set(order)
    if actual != expected:
        return False, f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
    return True, "legal permutation"


def safe_token(text: Any, fallback: str = "candidate", max_len: int = 30) -> str:
    token = re.sub(r"[^A-Za-z0-9_]+", "_", str(text).strip())
    token = re.sub(r"_+", "_", token).strip("_").lower()
    return (token or fallback)[:max_len]


def bucket_token(row: dict[str, str]) -> str:
    bucket = str(row.get("selection_bucket", "")).strip()
    family = str(row.get("candidate_family", "")).strip()
    mapping = {
        "top_region_local_search": "top_region",
        "model_disagreement": "model_disagreement",
        "uncertainty_calibration": "uncertainty_calibration",
        "diversity_coverage": "diversity_coverage",
        "tradeoff_probe": "tradeoff_probe",
        "sentinel_control": "sentinel_control",
        "exploitation_reference": "exploitation_reference",
        "batch_fill": "batch_fill",
        "u2_reference_fill": "u2_reference",
        "combined_selection_fill": "selection_fill",
    }
    return safe_token(mapping.get(bucket, bucket or family), "candidate", 28)


def combined108_best_and_hashes() -> tuple[dict[int, dict[str, Any]], dict[int, set[str]]]:
    rows = read_csv(COMBINED108_READY)
    best: dict[int, dict[str, Any]] = {}
    hashes: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        n = parse_int(row.get("n"))
        order = parse_order(row.get("order_json"))
        if order:
            hashes[n].add(order_hash(order))
    for n in EXPECTED_N:
        group = [row for row in rows if parse_int(row.get("n")) == n]
        reward_best = max(group, key=lambda row: parse_float(row.get("target_reward_combined108_u2_primary"), -1.0))
        best[n] = {
            "combined108_best_reward_strategy": reward_best.get("strategy_name", ""),
            "combined108_best_reward": parse_float(reward_best.get("target_reward_combined108_u2_primary")),
        }
    return best, hashes


def validate_shortlist(rows: list[dict[str, str]], teacher_hashes: dict[int, set[str]]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    counts: Counter[int] = Counter()
    n24_n40_count = 0
    candidate_ids: Counter[str] = Counter()
    seen_order: dict[int, set[str]] = defaultdict(set)
    buckets: Counter[str] = Counter()
    families: Counter[str] = Counter()

    for row in rows:
        try:
            n = parse_int(row.get("n"))
        except (TypeError, ValueError):
            errors.append(f"invalid N {row.get('n')}")
            continue
        counts[n] += 1
        if n in (24, 40):
            n24_n40_count += 1
        cid = str(row.get("candidate_id", "")).strip()
        strategy = str(row.get("strategy_name", "")).strip()
        candidate_ids[cid] += 1
        if not cid and not strategy:
            errors.append("row missing candidate_id and strategy_name")
        order = parse_order(row.get("order_json", ""))
        legal, reason = validate_order(order, n)
        if not legal:
            errors.append(f"{cid or strategy}: {reason}")
            continue
        digest = order_hash(order or [])
        if digest in seen_order[n]:
            errors.append(f"duplicate order within N{n}: {cid or strategy}")
        seen_order[n].add(digest)
        if digest in teacher_hashes[n] and not parse_bool(row.get("is_existing_teacher_order")):
            errors.append(f"{cid or strategy}: duplicates combined108 teacher order")
        if not math.isfinite(parse_float(row.get("pred_reward_ET_F01"))):
            errors.append(f"{cid or strategy}: missing pred_reward_ET_F01")
        if not math.isfinite(parse_float(row.get("model_prediction_mean"))):
            warnings.append(f"{cid or strategy}: missing model_prediction_mean")
        if not math.isfinite(parse_float(row.get("pred_uncertainty_ET_F01_std"))):
            warnings.append(f"{cid or strategy}: missing uncertainty metadata")
        if not str(row.get("selection_bucket", "")).strip():
            warnings.append(f"{cid or strategy}: missing selection_bucket")
        if not str(row.get("priority_role", "")).strip():
            warnings.append(f"{cid or strategy}: missing priority_role")
        buckets[str(row.get("selection_bucket", "")).strip()] += 1
        families[str(row.get("candidate_family", "")).strip()] += 1

    if len(rows) != 64:
        errors.append(f"expected 64 rows, found {len(rows)}")
    if sorted(counts) != EXPECTED_N:
        errors.append(f"expected N values {EXPECTED_N}, found {sorted(counts)}")
    for n, count in SHORTLIST64_COUNTS.items():
        if counts[n] != count:
            errors.append(f"expected {count} rows for N{n}, found {counts[n]}")
    if n24_n40_count != 48:
        errors.append(f"expected 48 N24/N40 rows, found {n24_n40_count}")
    duplicate_ids = [cid for cid, count in candidate_ids.items() if cid and count > 1]
    if duplicate_ids:
        errors.append(f"duplicate candidate_id values {duplicate_ids}")
    exploitation_count = buckets.get("exploitation_reference", 0)
    if exploitation_count >= len(rows) / 2:
        errors.append("selection appears exploitation-dominated, not active-learning balanced")
    if set(buckets) <= {"exploitation_reference", "surrogate_top"}:
        errors.append("bucket composition is pure exploitation")

    summary = {
        "verdict": "PASS_RUN24_SHORTLIST64_INPUT_READY" if not errors else "FAIL_RUN24_SHORTLIST64_INPUT_INVALID",
        "selected_batch": SELECTED_BATCH,
        "row_count": len(rows),
        "per_n_counts": dict(sorted(counts.items())),
        "n24_n40_count": n24_n40_count,
        "bucket_composition": dict(buckets),
        "family_composition": dict(families),
        "errors": errors,
        "warnings": warnings,
        "duplicate_candidate_ids": duplicate_ids,
        "selected_batch_is_shortlist64_not_batch32_or_batch24": True,
    }
    write_json(OUTPUT_DIR / "run24_input_validation_summary.json", summary)
    return summary


def make_handoff_rows(rows: list[dict[str, str]], best_by_n: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    handoff_rows: list[dict[str, Any]] = []
    per_n_index: Counter[int] = Counter()
    for row in sorted(rows, key=lambda r: (parse_int(r["n"]), int(float(r.get("shortlist_rank_within_n", "999"))), r.get("candidate_id", ""))):
        n = parse_int(row["n"])
        per_n_index[n] += 1
        bucket = bucket_token(row)
        handoff_name = f"S3R24L64_N{n}_B{per_n_index[n]:02d}_{bucket}"
        pred_reward = parse_float(row.get("pred_reward_ET_F01"))
        gap = pred_reward - parse_float(best_by_n[n]["combined108_best_reward"])
        order = parse_order(row.get("order_json"))
        assert order is not None
        handoff_rows.append(
            {
                "run_id": RUN_ID,
                "batch_option": SELECTED_BATCH,
                "batch_name": BATCH_NAME,
                "n": n,
                "handoff_strategy_name": handoff_name,
                "original_run23_candidate_id": row.get("candidate_id", ""),
                "original_run23_strategy_name": row.get("strategy_name", ""),
                "candidate_family": row.get("candidate_family", ""),
                "candidate_source": row.get("candidate_source", ""),
                "generation_method": row.get("generation_method", ""),
                "selection_bucket": row.get("selection_bucket", ""),
                "priority_role": row.get("priority_role", ""),
                "pred_reward_ET_F01": row.get("pred_reward_ET_F01", ""),
                "pred_reward_Ridge_F03": row.get("pred_reward_Ridge_F03", ""),
                "pred_reward_Ridge_F06": row.get("pred_reward_Ridge_F06", ""),
                "pred_reward_ET_F04": row.get("pred_reward_ET_F04", ""),
                "pred_reward_ET_F05": row.get("pred_reward_ET_F05", ""),
                "model_prediction_mean": row.get("model_prediction_mean", ""),
                "model_prediction_std": row.get("model_prediction_std", ""),
                "pred_uncertainty_ET_F01_std": row.get("pred_uncertainty_ET_F01_std", ""),
                "disagreement_ET_F01_vs_Ridge_F03": row.get("disagreement_ET_F01_vs_Ridge_F03", ""),
                "disagreement_ET_F01_vs_Ridge_F06": row.get("disagreement_ET_F01_vs_Ridge_F06", ""),
                "novelty_distance_to_combined108": row.get("novelty_distance_to_combined108", ""),
                "nearest_existing_teacher_strategy": row.get("nearest_existing_teacher_strategy", ""),
                "predicted_gap_vs_combined108_best": gap,
                "order_json": json.dumps(order, separators=(",", ":")),
                "order_compact": "-".join(str(x) for x in order),
                "order_hash": row.get("order_hash") or order_hash(order),
                "teacher_validated": False,
                "teacher_validation_status": "NOT_RUN",
                "notes": "Run24 handoff only. Not teacher-validated. Do not claim physical superiority.",
            }
        )
    return handoff_rows


def write_scan_order_jsons(handoff_rows: list[dict[str, Any]]) -> list[str]:
    paths: list[str] = []
    SCAN_ORDER_DIR.mkdir(parents=True, exist_ok=True)
    for row in handoff_rows:
        order = parse_order(row["order_json"])
        payload = {
            "run_id": RUN_ID,
            "batch_option": SELECTED_BATCH,
            "batch_name": BATCH_NAME,
            "n": row["n"],
            "handoff_strategy_name": row["handoff_strategy_name"],
            "original_run23_candidate_id": row["original_run23_candidate_id"],
            "candidate_family": row["candidate_family"],
            "selection_bucket": row["selection_bucket"],
            "priority_role": row["priority_role"],
            "prediction_metadata": {
                "pred_reward_ET_F01": parse_float(row["pred_reward_ET_F01"], None),
                "pred_reward_Ridge_F03": parse_float(row["pred_reward_Ridge_F03"], None),
                "pred_reward_Ridge_F06": parse_float(row["pred_reward_Ridge_F06"], None),
                "pred_reward_ET_F04": parse_float(row["pred_reward_ET_F04"], None),
                "pred_reward_ET_F05": parse_float(row["pred_reward_ET_F05"], None),
                "model_prediction_mean": parse_float(row["model_prediction_mean"], None),
                "model_prediction_std": parse_float(row["model_prediction_std"], None),
                "pred_uncertainty_ET_F01_std": parse_float(row["pred_uncertainty_ET_F01_std"], None),
                "disagreement_ET_F01_vs_Ridge_F03": parse_float(row["disagreement_ET_F01_vs_Ridge_F03"], None),
                "disagreement_ET_F01_vs_Ridge_F06": parse_float(row["disagreement_ET_F01_vs_Ridge_F06"], None),
                "novelty_distance_to_combined108": parse_float(row["novelty_distance_to_combined108"], None),
                "predicted_gap_vs_combined108_best": parse_float(row["predicted_gap_vs_combined108_best"], None),
            },
            "nearest_existing_teacher_strategy": row["nearest_existing_teacher_strategy"],
            "scan_order": order,
            "order_hash": row["order_hash"],
            "teacher_validated": False,
            "teacher_validation_status": "NOT_RUN",
            "notes": "Run24 handoff only. Not teacher-validated. Do not claim physical superiority.",
        }
        path = SCAN_ORDER_DIR / f"scan_order_{row['handoff_strategy_name']}.json"
        write_json(path, payload)
        paths.append(str(path))
    return paths


def future_paths(row: dict[str, Any]) -> dict[str, str]:
    n = int(row["n"])
    strategy = str(row["handoff_strategy_name"])
    case_dir = FUTURE_CASE_ROOT / f"N{n}{strategy}"
    job_name = f"J2D_{strategy}"
    return {
        "expected_case_dir": str(case_dir),
        "expected_job_name": job_name,
        "expected_cae": str(case_dir / f"{job_name}.cae"),
        "expected_inp": str(case_dir / f"{job_name}.inp"),
        "expected_jnl": str(case_dir / f"{job_name}.jnl"),
        "expected_odb": str(case_dir / f"{job_name}.odb"),
    }


def build_cae_manifest(handoff_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in handoff_rows:
        paths = future_paths(row)
        rows.append(
            {
                "n": row["n"],
                "handoff_strategy_name": row["handoff_strategy_name"],
                "expected_case_dir": paths["expected_case_dir"],
                "expected_job_name": paths["expected_job_name"],
                "scan_order_json": str(SCAN_ORDER_DIR / f"scan_order_{row['handoff_strategy_name']}.json"),
                "expected_cae": paths["expected_cae"],
                "expected_inp": paths["expected_inp"],
                "expected_jnl": paths["expected_jnl"],
                "expected_odb": paths["expected_odb"],
                "teacher_validated": False,
                "generation_status": "NOT_GENERATED",
                "solver_status": "NOT_SUBMITTED",
                "notes": "Template only. Do not run until CAE/INP generation is completed and checked.",
            }
        )
    return rows


def write_abqjobpilot_template(handoff_rows: list[dict[str, Any]]) -> Path:
    path = OUTPUT_DIR / "stage3_run24_shortlist64_abqjobpilot_commands_TEMPLATE.txt"
    lines = []
    for row in handoff_rows:
        paths = future_paths(row)
        lines.append(f'enqueue --inp "{paths["expected_inp"]}" --cpus 14 --batch {BATCH_NAME} --strategy {row["handoff_strategy_name"]}')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def summarize_review(handoff_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    rows: list[dict[str, Any]] = []
    total = len(handoff_rows)
    n24n40 = sum(1 for row in handoff_rows if int(row["n"]) in (24, 40))
    bucket_counts = Counter(str(row["selection_bucket"]) for row in handoff_rows)
    family_counts = Counter(str(row["candidate_family"]) for row in handoff_rows)
    exceeds = [row for row in handoff_rows if parse_float(row["predicted_gap_vs_combined108_best"]) > 0]
    for n in EXPECTED_N:
        group = [row for row in handoff_rows if int(row["n"]) == n]
        rewards = [parse_float(row["pred_reward_ET_F01"]) for row in group]
        novelty = [parse_float(row["novelty_distance_to_combined108"]) for row in group]
        uncertainty = [parse_float(row["pred_uncertainty_ET_F01_std"]) for row in group]
        disagreement = [parse_float(row["model_prediction_std"]) for row in group]
        rows.append(
            {
                "n": n,
                "count": len(group),
                "bucket_composition_json": json.dumps(dict(Counter(row["selection_bucket"] for row in group)), sort_keys=True),
                "candidate_family_composition_json": json.dumps(dict(Counter(row["candidate_family"] for row in group)), sort_keys=True),
                "mean_pred_reward_ET_F01": sum(rewards) / len(rewards) if rewards else "",
                "max_pred_reward_ET_F01": max(rewards) if rewards else "",
                "mean_novelty_distance_to_combined108": sum(novelty) / len(novelty) if novelty else "",
                "mean_uncertainty_ET_F01_std": sum(uncertainty) / len(uncertainty) if uncertainty else "",
                "mean_model_disagreement_std": sum(disagreement) / len(disagreement) if disagreement else "",
            }
        )
    summary = {
        "total_count": total,
        "per_n_counts": dict(Counter(int(row["n"]) for row in handoff_rows)),
        "n24_n40_count": n24n40,
        "n24_n40_share": n24n40 / total if total else 0.0,
        "bucket_composition": dict(bucket_counts),
        "candidate_family_composition": dict(family_counts),
        "top_region_candidate_count": bucket_counts.get("top_region_local_search", 0),
        "model_disagreement_candidate_count": bucket_counts.get("model_disagreement", 0),
        "uncertainty_candidate_count": bucket_counts.get("uncertainty_calibration", 0),
        "diversity_candidate_count": bucket_counts.get("diversity_coverage", 0),
        "tradeoff_candidate_count": bucket_counts.get("tradeoff_probe", 0),
        "sentinel_candidate_count": bucket_counts.get("sentinel_control", 0),
        "surrogate_only_predicted_exceeds_combined108_best_count": len(exceeds),
        "expected_abaqus_cost": "64 jobs total, with 48 jobs from N24/N40",
        "selected_batch_reason": "shortlist64 selected because the user wants an overnight run with 60+ candidates.",
        "claim_boundary": "shortlist64 remains unvalidated until future teacher validation; run24 did not create CAE/INP files.",
    }
    md_lines = [
        "# Shortlist64 Review Summary",
        "",
        f"- Total count: {total}",
        f"- Per-N counts: {summary['per_n_counts']}",
        f"- N24/N40 share: {n24n40}/{total}",
        f"- Bucket composition: {dict(bucket_counts)}",
        f"- Candidate family composition: {dict(family_counts)}",
        f"- Surrogate-only candidates predicted above combined108 best: {len(exceeds)}",
        "- Expected Abaqus cost: 64 jobs total, with 48 jobs from N24/N40.",
        "",
        "Shortlist64 is selected because the user wants an overnight run with 60+ candidates. It gives stronger N24/N40 calibration coverage than batch32 or batch24.",
        "",
        "This package is handoff only. No CAE/INP files were created, and no candidate is teacher-validated.",
        "",
    ]
    return rows, summary, "\n".join(md_lines)


def write_readme() -> Path:
    path = OUTPUT_DIR / "README_FOR_FUTURE_CAE_GENERATION.md"
    text = "\n".join(
        [
            "# README For Future CAE Generation",
            "",
            "Run24 created a handoff package for the selected shortlist64 active-learning candidates.",
            "",
            "No CAE/INP files exist yet. No abqjobpilot command is executable yet.",
            "",
            "The CAE module should generate true variable-N models using corrected sanity_base CAE files.",
            "",
            "Established heat-load mapping:",
            "- `set_body_heat_{track:02d}`",
            "- `step_scan_{seq:02d}`",
            "- `step_cool_{seq:02d}`",
            "- `load_body_hflux_{seq:02d}`",
            "- BodyHeatFlux magnitude `80000000000.0`",
            "",
            "Final cooling controls must remain:",
            "- `step_final_cooling` duration = `1200.0`",
            "- `initialInc = 0.01`",
            "- `maxInc = 60.0`",
            "",
            "The CAE module should not run solver until the user approves. The future abqjobpilot command template must not be executed until INPs exist and pass checks.",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")
    return path


def write_claim_boundary() -> tuple[Path, Path]:
    md_path = OUTPUT_DIR / "run24_claim_boundary.md"
    json_path = OUTPUT_DIR / "run24_claim_boundary.json"
    md = "\n".join(
        [
            "# Run24 Claim Boundary",
            "",
            "## Safe claims",
            "- Run24 packages selected run23 shortlist64 active-learning candidates for human review and future CAE generation.",
            "- Shortlist64 includes N24/N40-heavy active-learning coverage.",
            "- Handoff files include scan orders, metadata, future CAE manifest template, and abqjobpilot command template.",
            "- No CAE/INP files were generated.",
            "",
            "## Unsafe claims",
            "- Do not claim candidates are teacher-validated.",
            "- Do not claim physical superiority.",
            "- Do not claim surrogate predictions are ground truth.",
            "- Do not claim trained variable-N RL success.",
            "- Do not claim arbitrary-N generalization.",
            "- Do not claim abqjobpilot commands are ready to execute.",
            "- Do not claim CAE/INP files exist.",
            "",
            "Verdict: RUN24_SHORTLIST64_HANDOFF_ONLY_NO_TEACHER_VALIDATION",
            "",
        ]
    )
    md_path.write_text(md, encoding="utf-8")
    write_json(
        json_path,
        {
            "verdict": "RUN24_SHORTLIST64_HANDOFF_ONLY_NO_TEACHER_VALIDATION",
            "safe_claims": [
                "shortlist64 handoff package created for human review and future CAE generation",
                "N24/N40-heavy active-learning coverage included",
                "scan orders, metadata, future CAE manifest template, and abqjobpilot command template written",
                "no CAE/INP files generated",
            ],
            "unsafe_claims": [
                "teacher validation",
                "physical superiority",
                "surrogate predictions as ground truth",
                "trained variable-N RL success",
                "arbitrary-N generalization",
                "abqjobpilot commands ready to execute",
                "CAE/INP files exist",
            ],
        },
    )
    return md_path, json_path


def write_report(validation: dict[str, Any], review_summary: dict[str, Any], outputs: list[str]) -> None:
    lines = [
        "# Stage 3 Run 24 - Run23 Shortlist64 Active-Learning Handoff Package",
        "",
        "## Purpose",
        "Package the user-selected run23 shortlist64 active-learning candidates for future CAE generation. This run is handoff-only.",
        "",
        "## Inputs",
        f"- Run23 scored candidate pool: `{RUN23_SCORED}`",
        f"- Run23 selected shortlist64: `{RUN23_SHORTLIST64}`",
        f"- Run23 batch32 and batch24 were read only as reference inputs and were not packaged as the selected batch.",
        f"- Combined108 teacher dataset: `{COMBINED108_TEACHER}`",
        "",
        "## User-Selected Batch",
        f"- Selected batch: `{SELECTED_BATCH}`",
        f"- Batch name: `{BATCH_NAME}`",
        "- Batch24 and batch32 are explicitly not selected.",
        "",
        "## Validation Status",
        f"- Verdict: `{validation['verdict']}`",
        f"- Per-N counts: `{validation['per_n_counts']}`",
        f"- N24/N40 count: `{validation['n24_n40_count']}`",
        "",
        "## Stable Naming Convention",
        "- Format: `S3R24L64_N{N}_B{index:02d}_{short_bucket_or_family}`",
        "- Names are filesystem-safe and preserve original run23 candidate IDs in metadata.",
        "",
        "## Shortlist64 Handoff Package",
        "- Candidate order CSV contains stable handoff names, run23 provenance, active-learning bucket metadata, predictions, uncertainty, disagreement, novelty, scan orders, and NOT_RUN teacher status.",
        "",
        "## Per-Candidate Scan-Order JSON Outputs",
        f"- JSON directory: `{SCAN_ORDER_DIR}`",
        "- Each JSON is handoff metadata only and is not teacher-validated.",
        "",
        "## Future CAE Handoff Template",
        "- The future CAE manifest template lists expected paths only. It does not create CAE case directories and does not generate CAE/INP/JNL files.",
        "",
        "## Future abqjobpilot Command Template",
        "- The command template is not executable yet because INP files do not exist.",
        "- Commands must not be run until CAE/INP generation has completed and passed checks.",
        "",
        "## Shortlist64 Review Summary",
        f"- Total count: `{review_summary['total_count']}`",
        f"- Per-N counts: `{review_summary['per_n_counts']}`",
        f"- N24/N40 share: `{review_summary['n24_n40_count']}/{review_summary['total_count']}`",
        f"- Bucket composition: `{review_summary['bucket_composition']}`",
        f"- Expected Abaqus cost: `{review_summary['expected_abaqus_cost']}`",
        "",
        "## Claim Boundary",
        "RUN24_SHORTLIST64_HANDOFF_ONLY_NO_TEACHER_VALIDATION. No CAE/INP generation, solver execution, abqjobpilot execution, teacher validation, or physical superiority is claimed.",
        "",
        "## Output Files",
    ]
    lines.extend(f"- `{path}`" for path in outputs)
    lines.extend(
        [
            "",
            "## Recommended Run25",
            "CAE module should generate CAE/INP/JNL for selected shortlist64 only. Do not run solver, do not execute abqjobpilot, and do not generate batch24 or batch32.",
            "",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def update_run_index() -> None:
    row = "| run_24 | Run23 shortlist64 active-learning handoff package | Package the user-selected run23 shortlist64 active-learning candidates for future CAE generation without generating CAE/INP/JNL or executing solver/job commands. | `scripts/stage3/run_24_create_run23_shortlist64_active_learning_handoff_package.py` | `docs/stage3/runs/run_24_run23_shortlist64_active_learning_handoff_package/RUN_24_RUN23_SHORTLIST64_ACTIVE_LEARNING_HANDOFF_PACKAGE_REPORT.md` | `outputs/stage3_run_24_run23_shortlist64_active_learning_handoff_package/` | `PASS_RUN24_SHORTLIST64_INPUT_READY` | No Abaqus, no ODB, no abqjobpilot, no CAE/INP/JNL generation, no teacher validation, no RL policy training, no commit/push. Next: run25 CAE/INP generation for shortlist64 only after user approval. |"
    if RUN_INDEX_PATH.exists():
        text = RUN_INDEX_PATH.read_text(encoding="utf-8")
        if "| run_24 |" not in text:
            RUN_INDEX_PATH.write_text(text.rstrip() + "\n" + row + "\n", encoding="utf-8")


def git_branch() -> str:
    try:
        result = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception:
        return ""


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_csv(RUN23_SHORTLIST64)
    best_by_n, teacher_hashes = combined108_best_and_hashes()
    validation = validate_shortlist(rows, teacher_hashes)
    if validation["verdict"].startswith("FAIL"):
        print(validation["verdict"])
        return 2

    handoff_rows = make_handoff_rows(rows, best_by_n)
    scan_json_paths = write_scan_order_jsons(handoff_rows)
    cae_manifest = build_cae_manifest(handoff_rows)
    abq_template = write_abqjobpilot_template(handoff_rows)
    review_rows, review_summary, review_md = summarize_review(handoff_rows)
    claim_md, claim_json = write_claim_boundary()
    readme_path = write_readme()

    outputs: list[str] = []
    candidate_csv = OUTPUT_DIR / "stage3_run24_shortlist64_candidate_orders.csv"
    write_csv(candidate_csv, handoff_rows)
    outputs.append(str(candidate_csv))
    outputs.extend(scan_json_paths)
    cae_template = OUTPUT_DIR / "stage3_run24_shortlist64_future_cae_handoff_manifest_TEMPLATE.csv"
    write_csv(cae_template, cae_manifest)
    outputs.append(str(cae_template))
    outputs.append(str(abq_template))
    review_csv = OUTPUT_DIR / "shortlist64_review_summary.csv"
    review_json = OUTPUT_DIR / "shortlist64_review_summary.json"
    review_md_path = OUTPUT_DIR / "shortlist64_review_summary.md"
    write_csv(review_csv, review_rows)
    write_json(review_json, review_summary)
    review_md_path.write_text(review_md, encoding="utf-8")
    outputs.extend([str(review_csv), str(review_json), str(review_md_path), str(readme_path), str(claim_md), str(claim_json), str(OUTPUT_DIR / "run24_input_validation_summary.json")])
    write_report(validation, review_summary, outputs)
    outputs.append(str(REPORT_PATH))
    update_run_index()

    manifest = {
        "run_id": RUN_ID,
        "run_name": RUN_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "branch": git_branch(),
        "script_path": str(ROOT / "scripts" / "stage3" / "run_24_create_run23_shortlist64_active_learning_handoff_package.py"),
        "input_files": [str(p) for p in [RUN23_SCORED, RUN23_SHORTLIST64, RUN23_BATCH32, RUN23_BATCH24, RUN23_COMPARISON, RUN23_REPORT, COMBINED108_TEACHER, COMBINED108_READY] if p.exists()],
        "output_files": outputs,
        "selected_batch": SELECTED_BATCH,
        "shortlist64_count": len(handoff_rows),
        "per_n_counts": dict(Counter(row["n"] for row in handoff_rows)),
        "report_path": str(REPORT_PATH),
        "claim_boundary_path": str(claim_md),
        "validation_verdict": validation["verdict"],
        "no_solver_run": True,
        "no_odb_opened": True,
        "no_abqjobpilot_run": True,
        "no_cae_inp_generated": True,
        "no_teacher_validation": True,
        "no_rl_policy_training": True,
        "no_commit_or_push": True,
    }
    write_json(MANIFEST_PATH, manifest)

    print(validation["verdict"])
    print(f"selected_batch={SELECTED_BATCH}")
    print(f"shortlist64={len(handoff_rows)} per_n={dict(Counter(row['n'] for row in handoff_rows))}")
    print(f"candidate_csv={candidate_csv}")
    print(f"scan_order_dir={SCAN_ORDER_DIR}")
    print(f"cae_template={cae_template}")
    print(f"abqjobpilot_template={abq_template}")
    print(f"report={REPORT_PATH}")
    print(f"manifest={MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
