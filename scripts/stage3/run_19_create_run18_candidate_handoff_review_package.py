from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
RUN_ID = "run_19_run18_candidate_handoff_review_package"
RUN_NAME = "human-review handoff package for run18 batch24 and batch28 candidates"

RUN18_SCORED = ROOT / "outputs" / "stage3_run_18_combined80_surrogate_screened_candidate_generation" / "run18_candidate_pool_scored.csv"
RUN18_SHORTLIST48 = ROOT / "outputs" / "stage3_run_18_combined80_surrogate_screened_candidate_generation" / "run18_candidate_shortlist48.csv"
RUN18_BATCH28 = ROOT / "outputs" / "stage3_run_18_combined80_surrogate_screened_candidate_generation" / "run18_recommended_future_teacher_batch28.csv"
RUN18_BATCH24 = ROOT / "outputs" / "stage3_run_18_combined80_surrogate_screened_candidate_generation" / "run18_recommended_future_teacher_batch24.csv"
RUN18_IMPROVEMENT = ROOT / "outputs" / "stage3_run_18_combined80_surrogate_screened_candidate_generation" / "run18_predicted_improvement_vs_combined80.csv"
RUN18_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_18_combined80_surrogate_screened_candidate_generation" / "RUN_18_COMBINED80_SURROGATE_SCREENED_CANDIDATE_GENERATION_REPORT.md"
COMBINED80_TEACHER = ROOT / "outputs" / "stage3_run_16_batch20_teacher_metrics_ingestion_and_combined80_ranking" / "combined80_teacher_dataset.csv"
COMBINED80_READY = ROOT / "outputs" / "stage3_run_16_batch20_teacher_metrics_ingestion_and_combined80_ranking" / "combined80_RL_ready_dataset.csv"

OUTPUT_DIR = ROOT / "outputs" / "stage3_run_19_run18_candidate_handoff_review_package"
BATCH24_DIR = OUTPUT_DIR / "batch24"
BATCH28_DIR = OUTPUT_DIR / "batch28"
REPORT_DIR = ROOT / "docs" / "stage3" / "runs" / RUN_ID
REPORT_PATH = REPORT_DIR / "RUN_19_RUN18_CANDIDATE_HANDOFF_REVIEW_PACKAGE_REPORT.md"
MANIFEST_PATH = ROOT / "artifacts" / "manifests" / "stage3_run_19_manifest.json"
RUN_INDEX_PATH = ROOT / "docs" / "stage3" / "STAGE3_RUN_INDEX.md"

EXPECTED_N = [12, 16, 24, 40]
BATCH24_COUNTS = {12: 3, 16: 3, 24: 9, 40: 9}
BATCH28_COUNTS = {12: 4, 16: 4, 24: 10, 40: 10}
BATCH24_NAME = "stage3_run19_batch24_combined80_surrogate_screened_v01"
BATCH28_NAME = "stage3_run19_batch28_combined80_surrogate_screened_v01"
TARGET_COL = "target_reward_combined80_u2_primary"


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


def safe_token(text: str, fallback: str = "candidate", max_len: int = 30) -> str:
    token = re.sub(r"[^A-Za-z0-9_]+", "_", str(text).strip())
    token = re.sub(r"_+", "_", token).strip("_").lower()
    return (token or fallback)[:max_len]


def bucket_token(row: dict[str, str]) -> str:
    bucket = row.get("selection_bucket", "")
    family = row.get("candidate_family", "")
    if bucket == "method_c_or_known_best_inspired":
        return "method_c_inspired"
    if bucket == "negative_control_sentinel":
        return "control_sentinel"
    if bucket == "batch_fill":
        return safe_token(family, "batch_fill")
    return safe_token(bucket or family, "candidate")


def load_combined80_best() -> tuple[dict[int, dict[str, Any]], dict[int, set[str]]]:
    rows = read_csv(COMBINED80_READY)
    best: dict[int, dict[str, Any]] = {}
    hashes: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        n = parse_int(row["n"])
        order = parse_order(row.get("order_json"))
        if order:
            hashes[n].add(order_hash(order))
    for n in EXPECTED_N:
        group = [row for row in rows if parse_int(row["n"]) == n]
        reward_best = max(group, key=lambda row: parse_float(row.get(TARGET_COL), -1.0))
        u2_best = max(group, key=lambda row: parse_float(row.get("target_u2_score_combined80_rank"), -1.0))
        best[n] = {
            "combined80_best_reward_strategy": reward_best.get("strategy_name", ""),
            "combined80_best_reward": parse_float(reward_best.get(TARGET_COL)),
            "combined80_best_u2_strategy": u2_best.get("strategy_name", ""),
            "combined80_best_u2_score": parse_float(u2_best.get("target_u2_score_combined80_rank")),
        }
    return best, hashes


def validate_batch(rows: list[dict[str, str]], expected_counts: dict[int, int], label: str, teacher_hashes: dict[int, set[str]]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    counts: Counter[int] = Counter()
    candidate_ids: Counter[str] = Counter()
    seen_order: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        try:
            n = parse_int(row.get("n"))
        except (TypeError, ValueError):
            errors.append(f"{label}: invalid N {row.get('n')}")
            continue
        counts[n] += 1
        cid = str(row.get("candidate_id", "")).strip()
        strategy = str(row.get("strategy_name", "")).strip()
        candidate_ids[cid] += 1
        if not cid and not strategy:
            errors.append(f"{label}: row missing candidate_id and strategy_name")
        order = parse_order(row.get("order_json", ""))
        legal, reason = validate_order(order, n)
        if not legal:
            errors.append(f"{label}:{cid or strategy}: {reason}")
            continue
        digest = order_hash(order or [])
        if digest in seen_order[n]:
            errors.append(f"{label}: duplicate order within N{n}: {cid or strategy}")
        seen_order[n].add(digest)
        if digest in teacher_hashes[n] and not parse_bool(row.get("is_existing_teacher_order")):
            errors.append(f"{label}:{cid or strategy}: duplicates combined80 teacher order")
        if not math.isfinite(parse_float(row.get("pred_reward_combined80_u2_primary"))):
            errors.append(f"{label}:{cid or strategy}: missing predicted reward")
        for col in ["candidate_family", "candidate_source", "selection_bucket"]:
            if not str(row.get(col, "")).strip():
                warnings.append(f"{label}:{cid or strategy}: missing optional {col}")
    expected_total = sum(expected_counts.values())
    if len(rows) != expected_total:
        errors.append(f"{label}: expected {expected_total} rows, found {len(rows)}")
    if sorted(counts) != EXPECTED_N:
        errors.append(f"{label}: expected N values {EXPECTED_N}, found {sorted(counts)}")
    for n, count in expected_counts.items():
        if counts[n] != count:
            errors.append(f"{label}: expected {count} rows for N{n}, found {counts[n]}")
    duplicate_ids = [cid for cid, count in candidate_ids.items() if cid and count > 1]
    if duplicate_ids:
        errors.append(f"{label}: duplicate candidate_id values {duplicate_ids}")
    return {
        "batch_option": label,
        "row_count": len(rows),
        "per_n_counts": dict(sorted(counts.items())),
        "errors": errors,
        "warnings": warnings,
        "duplicate_candidate_ids": duplicate_ids,
    }


def validation_summary(batch24: list[dict[str, str]], batch28: list[dict[str, str]], teacher_hashes: dict[int, set[str]]) -> dict[str, Any]:
    v24 = validate_batch(batch24, BATCH24_COUNTS, "batch24", teacher_hashes)
    v28 = validate_batch(batch28, BATCH28_COUNTS, "batch28", teacher_hashes)
    errors = v24["errors"] + v28["errors"]
    verdict = "PASS_RUN19_BATCH24_BATCH28_INPUTS_READY" if not errors else "FAIL_RUN19_BATCH_INPUTS_INVALID"
    payload = {
        "verdict": verdict,
        "batch24": v24,
        "batch28": v28,
        "errors": errors,
        "warnings": v24["warnings"] + v28["warnings"],
    }
    write_json(OUTPUT_DIR / "run19_input_validation_summary.json", payload)
    return payload


def stable_handoff_name(prefix: str, n: int, idx: int, row: dict[str, str]) -> str:
    token = bucket_token(row)
    name = f"{prefix}_N{n}_B{idx:02d}_{token}"
    return re.sub(r"[^A-Za-z0-9_]", "_", name)[:72].rstrip("_")


def predicted_gap(row: dict[str, str], combined80_best: dict[int, dict[str, Any]]) -> float:
    n = parse_int(row["n"])
    return parse_float(row.get("pred_reward_combined80_u2_primary")) - parse_float(combined80_best[n]["combined80_best_reward"])


def build_handoff_rows(rows: list[dict[str, str]], option: str, batch_name: str, combined80_best: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    prefix = "S3R19B24" if option == "batch24" else "S3R19B28"
    rank_col = "batch24_rank_within_n" if option == "batch24" else "batch28_rank_within_n"
    output: list[dict[str, Any]] = []
    for n in EXPECTED_N:
        group = [row for row in rows if parse_int(row["n"]) == n]
        group.sort(key=lambda row: parse_int(row.get(rank_col) or row.get("pred_rank_within_n") or "999"))
        for idx, row in enumerate(group, start=1):
            order = parse_order(row["order_json"]) or []
            handoff_name = stable_handoff_name(prefix, n, idx, row)
            output.append(
                {
                    "run_id": RUN_ID,
                    "batch_option": option,
                    "batch_name": batch_name,
                    "n": n,
                    "handoff_strategy_name": handoff_name,
                    "original_run18_candidate_id": row.get("candidate_id", ""),
                    "original_run18_strategy_name": row.get("strategy_name", ""),
                    "candidate_family": row.get("candidate_family", ""),
                    "candidate_source": row.get("candidate_source", ""),
                    "generation_method": row.get("generation_method", ""),
                    "selection_bucket": row.get("selection_bucket", ""),
                    "priority_role": row.get("priority_role", ""),
                    "predicted_reward_combined80_u2_primary": row.get("pred_reward_combined80_u2_primary", ""),
                    "predicted_rank_within_n": row.get("pred_rank_within_n", ""),
                    "predicted_percentile_within_n": row.get("pred_percentile_within_n", ""),
                    "predicted_uncertainty_std": row.get("pred_uncertainty_std", ""),
                    "novelty_distance_to_combined80": row.get("novelty_distance_to_combined80", ""),
                    "nearest_existing_teacher_strategy": row.get("nearest_existing_teacher_strategy", ""),
                    "predicted_gap_vs_combined80_best": predicted_gap(row, combined80_best),
                    "order_json": json.dumps(order, separators=(",", ":")),
                    "order_compact": row.get("order_compact", "-".join(str(x) for x in order)),
                    "order_hash": row.get("order_hash", order_hash(order)),
                    "teacher_validated": False,
                    "teacher_validation_status": "NOT_RUN",
                    "notes": "Run19 handoff only. Not teacher-validated. Do not claim physical superiority.",
                }
            )
    return output


def create_scan_order_jsons(rows: list[dict[str, Any]], scan_dir: Path) -> list[str]:
    scan_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for row in rows:
        order = parse_order(row["order_json"]) or []
        payload = {
            "run_id": RUN_ID,
            "batch_option": row["batch_option"],
            "batch_name": row["batch_name"],
            "n": row["n"],
            "handoff_strategy_name": row["handoff_strategy_name"],
            "original_run18_candidate_id": row["original_run18_candidate_id"],
            "candidate_family": row["candidate_family"],
            "selection_bucket": row["selection_bucket"],
            "predicted_reward_combined80_u2_primary": parse_float(row["predicted_reward_combined80_u2_primary"]),
            "predicted_rank_within_n": parse_int(row["predicted_rank_within_n"]),
            "predicted_uncertainty_std": parse_float(row["predicted_uncertainty_std"]),
            "novelty_distance_to_combined80": parse_float(row["novelty_distance_to_combined80"]),
            "nearest_existing_teacher_strategy": row["nearest_existing_teacher_strategy"],
            "scan_order": order,
            "order_hash": row["order_hash"],
            "teacher_validated": False,
            "teacher_validation_status": "NOT_RUN",
            "notes": "Run19 handoff only. Not teacher-validated. Do not claim physical superiority.",
        }
        path = scan_dir / f"scan_order_{row['handoff_strategy_name']}.json"
        write_json(path, payload)
        written.append(str(path))
    return written


def future_case_root(batch_name: str) -> Path:
    return ROOT / "cae_model" / batch_name


def create_future_cae_manifest(rows: list[dict[str, Any]], option_dir: Path, option: str, batch_name: str) -> Path:
    root = future_case_root(batch_name)
    manifest_rows: list[dict[str, Any]] = []
    for row in rows:
        n = parse_int(row["n"])
        name = row["handoff_strategy_name"]
        expected_case_dir = root / f"N{n}{name}"
        expected_job_name = f"J2D_{name}"
        scan_order_json = option_dir / "scan_orders" / f"scan_order_{name}.json"
        manifest_rows.append(
            {
                "n": n,
                "handoff_strategy_name": name,
                "expected_case_dir": str(expected_case_dir),
                "expected_job_name": expected_job_name,
                "scan_order_json": str(scan_order_json),
                "expected_cae": str(expected_case_dir / f"{expected_job_name}.cae"),
                "expected_inp": str(expected_case_dir / f"{expected_job_name}.inp"),
                "expected_jnl": str(expected_case_dir / f"{expected_job_name}.jnl"),
                "expected_odb": str(expected_case_dir / f"{expected_job_name}.odb"),
                "teacher_validated": False,
                "generation_status": "NOT_GENERATED",
                "solver_status": "NOT_SUBMITTED",
                "notes": "Template only. Do not run until CAE/INP generation exists and passes checks.",
            }
        )
    filename = f"stage3_run19_{option}_future_cae_handoff_manifest_TEMPLATE.csv"
    path = option_dir / filename
    write_csv(path, manifest_rows)
    return path


def create_abqjobpilot_template(rows: list[dict[str, Any]], option_dir: Path, option: str, batch_name: str) -> Path:
    root = future_case_root(batch_name)
    lines: list[str] = []
    for row in rows:
        n = parse_int(row["n"])
        name = row["handoff_strategy_name"]
        inp = root / f"N{n}{name}" / f"J2D_{name}.inp"
        lines.append(f'enqueue --inp "{inp}" --cpus 14 --batch {batch_name} --strategy {name}')
    path = option_dir / f"stage3_run19_{option}_abqjobpilot_commands_TEMPLATE.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def summarize_option(rows: list[dict[str, Any]], option: str, combined80_best: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for n in EXPECTED_N:
        group = [row for row in rows if parse_int(row["n"]) == n]
        pred = [parse_float(row["predicted_reward_combined80_u2_primary"]) for row in group]
        novelty = [parse_float(row["novelty_distance_to_combined80"]) for row in group]
        uncertainty = [parse_float(row["predicted_uncertainty_std"]) for row in group]
        top = max(group, key=lambda row: parse_float(row["predicted_reward_combined80_u2_primary"]))
        out.append(
            {
                "batch_option": option,
                "n": n,
                "count": len(group),
                "mean_predicted_reward": statistics.fmean(pred) if pred else math.nan,
                "max_predicted_reward": max(pred) if pred else math.nan,
                "mean_novelty_distance": statistics.fmean(novelty) if novelty else math.nan,
                "mean_uncertainty": statistics.fmean(uncertainty) if uncertainty else math.nan,
                "candidate_family_composition": json.dumps(dict(Counter(row["candidate_family"] for row in group)), sort_keys=True),
                "bucket_composition": json.dumps(dict(Counter(row["selection_bucket"] for row in group)), sort_keys=True),
                "exploitation_count": sum(1 for row in group if row["selection_bucket"] in {"surrogate_top", "U2_primary_top", "geometry_signal_top", "method_c_or_known_best_inspired", "batch_fill"}),
                "diversity_count": sum(1 for row in group if row["selection_bucket"] == "diversity_top"),
                "uncertainty_calibration_count": sum(1 for row in group if row["selection_bucket"] == "uncertainty_calibration"),
                "sentinel_control_count": sum(1 for row in group if row["selection_bucket"] == "negative_control_sentinel"),
                "top_candidate": top["handoff_strategy_name"],
                "top_candidate_predicted_gap_vs_combined80_best": parse_float(top["predicted_gap_vs_combined80_best"]),
                "any_candidate_predicted_above_combined80_best_surrogate_only": any(parse_float(row["predicted_gap_vs_combined80_best"]) > 0 for row in group),
                "combined80_best_reward_strategy": combined80_best[n]["combined80_best_reward_strategy"],
                "combined80_best_reward": combined80_best[n]["combined80_best_reward"],
            }
        )
    return out


def write_review_summary(batch24_rows: list[dict[str, Any]], batch28_rows: list[dict[str, Any]], combined80_best: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = summarize_option(batch24_rows, "batch24", combined80_best) + summarize_option(batch28_rows, "batch28", combined80_best)
    write_csv(OUTPUT_DIR / "batch24_vs_batch28_review_summary.csv", rows)
    write_table_json(OUTPUT_DIR / "batch24_vs_batch28_review_summary.json", rows)
    total24 = len(batch24_rows)
    total28 = len(batch28_rows)
    n2440_24 = sum(1 for row in batch24_rows if parse_int(row["n"]) in {24, 40})
    n2440_28 = sum(1 for row in batch28_rows if parse_int(row["n"]) in {24, 40})
    md = [
        "# Batch24 vs Batch28 Review Summary",
        "",
        f"- Batch24 total: `{total24}`; N24/N40 share: `{n2440_24}/{total24}`.",
        f"- Batch28 total: `{total28}`; N24/N40 share: `{n2440_28}/{total28}`.",
        "- Batch24 is cheaper and more conservative.",
        "- Batch28 gives slightly broader N24/N40 coverage at the cost of 4 additional future jobs.",
        "- Neither option is teacher-validated.",
        "- Neither option should be treated as a guaranteed physical improvement over combined80 best cases.",
        "",
        "| Batch | N | Count | Mean Pred Reward | Max Pred Reward | Mean Novelty | Mean Uncertainty | Above Combined80 Best? |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        md.append(
            f"| {row['batch_option']} | {row['n']} | {row['count']} | {float(row['mean_predicted_reward']):.4f} | "
            f"{float(row['max_predicted_reward']):.4f} | {float(row['mean_novelty_distance']):.4f} | "
            f"{float(row['mean_uncertainty']):.4f} | {row['any_candidate_predicted_above_combined80_best_surrogate_only']} |"
        )
    md.extend(
        [
            "",
            "Review position: do not recommend either option as universally superior. Choose batch24 for a lower-cost calibration pass, or batch28 for broader N24/N40 coverage.",
            "",
        ]
    )
    (OUTPUT_DIR / "batch24_vs_batch28_review_summary.md").write_text("\n".join(md), encoding="utf-8")
    return rows


def write_readme() -> Path:
    path = OUTPUT_DIR / "README_FOR_FUTURE_CAE_GENERATION.md"
    text = "\n".join(
        [
            "# README For Future CAE Generation",
            "",
            "Run19 created human-review handoff packages for batch24 and batch28.",
            "",
            "- No CAE/INP files exist yet.",
            "- No abqjobpilot command is executable yet.",
            "- The user must choose either batch24 or batch28 before CAE generation.",
            "- CAE generation should use true variable-N models and the corrected sanity_base CAE files.",
            "",
            "Established heat-load mapping to preserve:",
            "",
            "- `set_body_heat_{track:02d}`",
            "- `step_scan_{seq:02d}`",
            "- `step_cool_{seq:02d}`",
            "- `load_body_hflux_{seq:02d}`",
            "- `BodyHeatFlux` magnitude `80000000000.0`",
            "",
            "Final cooling controls to preserve:",
            "",
            "- `step_final_cooling` duration = `1200.0`",
            "- `initialInc = 0.01`",
            "- `maxInc = 60.0`",
            "",
            "The CAE module should not run solver until the user approves. Future abqjobpilot command templates must not be executed until INPs exist and pass checks.",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")
    return path


def write_claim_boundary() -> tuple[Path, Path]:
    md_path = OUTPUT_DIR / "run19_claim_boundary.md"
    json_path = OUTPUT_DIR / "run19_claim_boundary.json"
    md = "\n".join(
        [
            "# Run19 Claim Boundary",
            "",
            "## Safe claims",
            "- Run19 packages run18 surrogate-screened candidates for human review.",
            "- Both batch24 and batch28 handoff options are prepared.",
            "- Batch24 and batch28 include N24/N40-biased candidate distributions.",
            "- Handoff files include scan orders, metadata, future CAE manifest templates, and abqjobpilot command templates.",
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
            "Verdict: RUN19_HANDOFF_REVIEW_PACKAGE_ONLY_NO_TEACHER_VALIDATION",
            "",
        ]
    )
    md_path.write_text(md, encoding="utf-8")
    write_json(
        json_path,
        {
            "verdict": "RUN19_HANDOFF_REVIEW_PACKAGE_ONLY_NO_TEACHER_VALIDATION",
            "safe_claims": [
                "batch24 and batch28 handoff options prepared",
                "scan orders and metadata packaged for human review",
                "future CAE manifest and abqjobpilot command templates produced",
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


def write_report(outputs: list[str], validation: dict[str, Any], review_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Stage 3 Run 19 - Run18 Candidate Handoff Review Package",
        "",
        "## Purpose",
        "Create human-review handoff packages for both run18 batch24 and batch28 options without generating CAE/INP/JNL files or running any solver/job tooling.",
        "",
        "## Inputs",
        f"- Run18 scored candidate pool: `{RUN18_SCORED}`",
        f"- Run18 shortlist48: `{RUN18_SHORTLIST48}`",
        f"- Run18 batch24: `{RUN18_BATCH24}`",
        f"- Run18 batch28: `{RUN18_BATCH28}`",
        f"- Combined80 RL-ready dataset: `{COMBINED80_READY}`",
        "",
        "## Validation Status",
        f"- Verdict: `{validation['verdict']}`",
        f"- Batch24 counts: `{validation['batch24']['per_n_counts']}`",
        f"- Batch28 counts: `{validation['batch28']['per_n_counts']}`",
        "",
        "## Batch24 Handoff Package",
        f"- Candidate order table: `{BATCH24_DIR / 'stage3_run19_batch24_candidate_orders.csv'}`",
        f"- Scan order JSON directory: `{BATCH24_DIR / 'scan_orders'}`",
        f"- Future CAE handoff template: `{BATCH24_DIR / 'stage3_run19_batch24_future_cae_handoff_manifest_TEMPLATE.csv'}`",
        f"- Future abqjobpilot template: `{BATCH24_DIR / 'stage3_run19_batch24_abqjobpilot_commands_TEMPLATE.txt'}`",
        "",
        "## Batch28 Handoff Package",
        f"- Candidate order table: `{BATCH28_DIR / 'stage3_run19_batch28_candidate_orders.csv'}`",
        f"- Scan order JSON directory: `{BATCH28_DIR / 'scan_orders'}`",
        f"- Future CAE handoff template: `{BATCH28_DIR / 'stage3_run19_batch28_future_cae_handoff_manifest_TEMPLATE.csv'}`",
        f"- Future abqjobpilot template: `{BATCH28_DIR / 'stage3_run19_batch28_abqjobpilot_commands_TEMPLATE.txt'}`",
        "",
        "## Batch24 vs Batch28 Review Summary",
        "- Batch24 is cheaper and more conservative.",
        "- Batch28 gives slightly broader N24/N40 coverage at the cost of 4 additional future jobs.",
        "- Neither option is universally superior; both remain unvalidated until future teacher validation.",
        "",
        "## Future CAE Handoff Templates",
        "The CAE manifest templates list expected future paths only. Run19 did not create case directories and did not generate CAE/INP/JNL files.",
        "",
        "## Future Abqjobpilot Command Templates",
        "Command templates are not executable yet because INP files do not exist. They must not be run until CAE/INP generation has completed and passed checks.",
        "",
        "## Claim Boundary",
        "RUN19_HANDOFF_REVIEW_PACKAGE_ONLY_NO_TEACHER_VALIDATION. No physical superiority, teacher validation, trained RL success, arbitrary-N generalization, or executable job readiness is claimed.",
        "",
        "## Output Files",
    ]
    lines.extend(f"- `{path}`" for path in outputs)
    lines.extend(
        [
            "",
            "## Recommended Next Step",
            "User reviews batch24 vs batch28. If batch24 is selected, the CAE module should create run20 batch24 CAE/INP generation only. If batch28 is selected, the CAE module should create run20 batch28 CAE/INP generation only. Do not generate both unless explicitly requested.",
            "",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def update_run_index() -> None:
    row = "| run_19 | Run18 candidate handoff review package | Package run18 batch24 and batch28 options for human review with scan orders, future CAE templates, and abqjobpilot command templates. | `scripts/stage3/run_19_create_run18_candidate_handoff_review_package.py` | `docs/stage3/runs/run_19_run18_candidate_handoff_review_package/RUN_19_RUN18_CANDIDATE_HANDOFF_REVIEW_PACKAGE_REPORT.md` | `outputs/stage3_run_19_run18_candidate_handoff_review_package/` | `PASS_RUN19_BATCH24_BATCH28_INPUTS_READY` | No Abaqus, no ODB, no abqjobpilot, no CAE/INP/JNL generation, no teacher validation, no RL training, no commit/push. Next: user selects batch24 or batch28 for run20 CAE/INP generation only. |"
    if RUN_INDEX_PATH.exists():
        text = RUN_INDEX_PATH.read_text(encoding="utf-8")
        if "| run_19 |" not in text:
            RUN_INDEX_PATH.write_text(text.rstrip() + "\n" + row + "\n", encoding="utf-8")


def git_branch() -> str:
    try:
        result = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception:
        return ""


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BATCH24_DIR.mkdir(parents=True, exist_ok=True)
    BATCH28_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    batch24 = read_csv(RUN18_BATCH24)
    batch28 = read_csv(RUN18_BATCH28)
    combined80_best, teacher_hashes = load_combined80_best()
    validation = validation_summary(batch24, batch28, teacher_hashes)
    if validation["verdict"].startswith("FAIL"):
        print(validation["verdict"])
        return 2

    handoff24 = build_handoff_rows(batch24, "batch24", BATCH24_NAME, combined80_best)
    handoff28 = build_handoff_rows(batch28, "batch28", BATCH28_NAME, combined80_best)
    outputs: list[str] = []

    path24 = BATCH24_DIR / "stage3_run19_batch24_candidate_orders.csv"
    path28 = BATCH28_DIR / "stage3_run19_batch28_candidate_orders.csv"
    write_csv(path24, handoff24)
    write_csv(path28, handoff28)
    outputs.extend([str(path24), str(path28)])

    outputs.extend(create_scan_order_jsons(handoff24, BATCH24_DIR / "scan_orders"))
    outputs.extend(create_scan_order_jsons(handoff28, BATCH28_DIR / "scan_orders"))
    cae24 = create_future_cae_manifest(handoff24, BATCH24_DIR, "batch24", BATCH24_NAME)
    cae28 = create_future_cae_manifest(handoff28, BATCH28_DIR, "batch28", BATCH28_NAME)
    cmd24 = create_abqjobpilot_template(handoff24, BATCH24_DIR, "batch24", BATCH24_NAME)
    cmd28 = create_abqjobpilot_template(handoff28, BATCH28_DIR, "batch28", BATCH28_NAME)
    outputs.extend([str(cae24), str(cae28), str(cmd24), str(cmd28)])

    review_rows = write_review_summary(handoff24, handoff28, combined80_best)
    outputs.extend(
        [
            str(OUTPUT_DIR / "batch24_vs_batch28_review_summary.csv"),
            str(OUTPUT_DIR / "batch24_vs_batch28_review_summary.json"),
            str(OUTPUT_DIR / "batch24_vs_batch28_review_summary.md"),
        ]
    )
    readme = write_readme()
    claim_md, claim_json = write_claim_boundary()
    outputs.extend([str(readme), str(claim_md), str(claim_json), str(OUTPUT_DIR / "run19_input_validation_summary.json")])
    write_report(outputs, validation, review_rows)
    outputs.append(str(REPORT_PATH))
    update_run_index()

    manifest = {
        "run_id": RUN_ID,
        "run_name": RUN_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "branch": git_branch(),
        "script_path": str(ROOT / "scripts" / "stage3" / "run_19_create_run18_candidate_handoff_review_package.py"),
        "input_files": [str(path) for path in [RUN18_SCORED, RUN18_SHORTLIST48, RUN18_BATCH28, RUN18_BATCH24, RUN18_IMPROVEMENT, RUN18_REPORT, COMBINED80_TEACHER, COMBINED80_READY] if path.exists()],
        "output_files": outputs,
        "validation_verdict": validation["verdict"],
        "batch24_count": len(handoff24),
        "batch24_per_n_counts": dict(Counter(row["n"] for row in handoff24)),
        "batch28_count": len(handoff28),
        "batch28_per_n_counts": dict(Counter(row["n"] for row in handoff28)),
        "report_path": str(REPORT_PATH),
        "claim_boundary_path": str(claim_md),
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
    print(f"batch24={len(handoff24)} per_n={dict(Counter(row['n'] for row in handoff24))}")
    print(f"batch28={len(handoff28)} per_n={dict(Counter(row['n'] for row in handoff28))}")
    print(f"batch24_csv={path24}")
    print(f"batch28_csv={path28}")
    print(f"report={REPORT_PATH}")
    print(f"manifest={MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
