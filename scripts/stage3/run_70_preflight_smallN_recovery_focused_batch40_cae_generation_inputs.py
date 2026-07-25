"""Preflight Stage 3 run40 native_N24_N40_focused_batch60 CAE/INP generation inputs."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
RUN_ID = "run_70_smallN_recovery_focused_batch40_cae_inp_generation"
BATCH_NAME = "stage3_run69_smallN_recovery_focused_batch40_v01"
RUN44_DIR = PROJECT_ROOT / "outputs" / "stage3_run_69_run68_smallN_recovery_focused_batch40_handoff_package"
INPUT_CSV = RUN44_DIR / "stage3_run69_smallN_recovery_focused_batch40_candidate_orders.csv"
SCAN_ORDER_DIR = RUN44_DIR / "scan_orders"
CASE_ROOT = PROJECT_ROOT / "cae_model" / BATCH_NAME
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage3_run_70_smallN_recovery_focused_batch40_cae_inp_generation"
SUMMARY_PATH = OUTPUT_DIR / "run70_preflight_summary.json"
CASES_CSV = OUTPUT_DIR / "run70_preflight_cases.csv"
PLAN_CSV = OUTPUT_DIR / "run70_generation_plan.csv"
SAFETY_AUDIT_PATH = OUTPUT_DIR / "run70_future_cae_root_safety_audit.json"

EXPECTED_NS = (12, 16, 24, 40)
EXPECTED_COUNTS = {12: 16, 16: 16, 24: 4, 40: 4}
BASES = {
    12: PROJECT_ROOT / "cae_model" / "12track_full" / "sanity_base" / "12track_sanity_base.cae",
    16: PROJECT_ROOT / "cae_model" / "16track_full" / "sanity_base" / "16track_sanity_base.cae",
    24: PROJECT_ROOT / "cae_model" / "24track_full" / "sanity_base" / "24track_sanity_base.cae",
    40: PROJECT_ROOT / "cae_model" / "40track_full" / "sanity_base" / "40track_sanity_base.cae",
}
BAD_SCHEMA_TOKENS = ("N12N12_", "N16N16_", "N24N24_", "N40N40_")
SOLVER_EXTS = (".odb", ".sim", ".sta", ".dat", ".msg", ".lck")
MODEL_EXTS = (".cae", ".inp", ".jnl")
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def load_json(path: Path) -> object:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def validate_order(n: int, order: object) -> None:
    if not isinstance(order, list):
        raise ValueError("scan_order is not a list")
    if len(order) != n:
        raise ValueError("scan_order length {} != N {}".format(len(order), n))
    if not all(isinstance(v, int) for v in order):
        raise ValueError("scan_order contains non-integer entries")
    if sorted(order) != list(range(n)):
        raise ValueError("scan_order is not a permutation of 0..{}".format(n - 1))


def resolve_order_json(row: dict[str, str], strategy: str) -> Path:
    raw = row.get("order_json", "")
    if raw:
        raw_path = Path(raw)
        if raw_path.is_absolute() and raw_path.exists():
            return raw_path
        rel = SCAN_ORDER_DIR / raw
        if rel.exists():
            return rel
    fallback = SCAN_ORDER_DIR / "scan_order_{}.json".format(strategy)
    return fallback


def extract_scan_order(data: object) -> list[int]:
    if isinstance(data, dict):
        order = data.get("scan_order")
    else:
        order = data
    if not isinstance(order, list):
        raise ValueError("no scan_order list found")
    return order


def build_case_paths(n: int, strategy: str) -> dict[str, str]:
    case_dir = CASE_ROOT / ("N{}{}".format(n, strategy))
    job_name = "J2D_{}".format(strategy)
    return {
        "case_dir": str(case_dir),
        "job_name": job_name,
        "scan_order_json": str(case_dir / "scan_order_{}.json".format(strategy)),
        "cae_path": str(case_dir / "{}.cae".format(job_name)),
        "inp_path": str(case_dir / "{}.inp".format(job_name)),
        "jnl_path": str(case_dir / "{}.jnl".format(job_name)),
        "generation_log_path": str(case_dir / "{}.generation_log.json".format(job_name)),
    }


def safety_audit() -> tuple[dict[str, object], list[str]]:
    errors: list[str] = []
    counts = {ext: 0 for ext in MODEL_EXTS + SOLVER_EXTS}
    existing_solver_files: list[str] = []
    existing_model_files: list[str] = []
    if CASE_ROOT.exists():
        for ext in counts:
            files = list(CASE_ROOT.rglob("*{}".format(ext)))
            counts[ext] = len(files)
            if ext in SOLVER_EXTS:
                existing_solver_files.extend(str(p) for p in files)
            elif ext in MODEL_EXTS:
                existing_model_files.extend(str(p) for p in files)
    if existing_solver_files:
        errors.append("future CAE root contains solver outputs; stop before generation")
    if existing_model_files:
        errors.append("future CAE root contains existing CAE/INP/JNL; archive explicitly before regeneration")
    audit = {
        "case_root": str(CASE_ROOT),
        "case_root_exists": CASE_ROOT.exists(),
        "counts": counts,
        "existing_solver_files": existing_solver_files,
        "existing_model_files": existing_model_files,
        "verdict": "PASS_RUN70_FUTURE_CAE_ROOT_SAFE" if not errors else "FAIL_RUN70_FUTURE_CAE_ROOT_NOT_SAFE",
    }
    SAFETY_AUDIT_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit, errors


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    warnings: list[str] = []
    rows: list[dict[str, str]] = []
    normalized: list[dict[str, object]] = []
    orders_by_n: dict[int, set[tuple[int, ...]]] = defaultdict(set)

    if not INPUT_CSV.exists():
        errors.append("missing input CSV: {}".format(INPUT_CSV))
    else:
        with INPUT_CSV.open(newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))

    if len(rows) != 40:
        errors.append("row count {} != 40".format(len(rows)))

    n_counts = Counter()
    strategy_counts = Counter()
    for idx, row in enumerate(rows, 1):
        try:
            n = int(row.get("n", ""))
        except ValueError:
            errors.append("row {} invalid n={!r}".format(idx, row.get("n")))
            continue
        strategy = row.get("handoff_strategy_name", "")
        n_counts[n] += 1
        strategy_counts[strategy] += 1
        if row.get("batch_option") != "smallN_recovery_focused_batch40":
            errors.append("row {} is not smallN_recovery_focused_batch40: {}".format(idx, row.get("batch_option")))
        if row.get("batch_name") != BATCH_NAME:
            errors.append("row {} batch_name mismatch: {}".format(idx, row.get("batch_name")))
        lowered = "|".join(str(value) for value in row.values()).lower()
        if "small-n recovery batch32" in lowered or "final diagnostic batch24" in lowered:
            errors.append("row {} references forbidden Run68 Option B/C text".format(idx))
        if n not in EXPECTED_NS:
            errors.append("row {} unexpected N {}".format(idx, n))
        if n == 32:
            errors.append("row {} contains forbidden N32 case".format(idx))
        if row.get("native_validation_N") not in ("True", "true", True):
            errors.append("row {} is not marked native_validation_N=True".format(idx))
        if not strategy:
            errors.append("row {} missing handoff_strategy_name".format(idx))
        if strategy and not SAFE_NAME.match(strategy):
            errors.append("row {} strategy is not filesystem-safe: {}".format(idx, strategy))
        order_json = resolve_order_json(row, strategy)
        if not order_json.exists():
            errors.append("row {} missing order_json: {}".format(idx, order_json))
            continue
        try:
            order = extract_scan_order(load_json(order_json))
            validate_order(n, order)
        except Exception as exc:
            errors.append("row {} invalid scan_order {}: {}".format(idx, order_json, exc))
            continue
        order_tuple = tuple(order)
        if order_tuple in orders_by_n[n]:
            errors.append("row {} duplicate order within N{}".format(idx, n))
        orders_by_n[n].add(order_tuple)
        paths = build_case_paths(n, strategy)
        all_paths = [paths[key] for key in ("case_dir", "cae_path", "inp_path", "jnl_path")]
        if any(token in p for token in BAD_SCHEMA_TOKENS for p in all_paths):
            errors.append("row {} contains bad doubled-N path schema".format(idx))
        normalized.append(
            {
                "run_id": RUN_ID,
                "batch_name": BATCH_NAME,
                "selected_batch": "run68_smallN_recovery_focused_batch40",
                "n": n,
                "handoff_strategy_name": strategy,
                "original_run48_candidate_id": row.get("original_run48_candidate_id", ""),
                "original_run48_strategy_name": row.get("original_run48_strategy_name", ""),
                "candidate_source": row.get("candidate_source", ""),
                "generation_method": row.get("generation_method", ""),
                "selection_bucket": row.get("selection_bucket", ""),
                "priority_role": row.get("priority_role", ""),
                "surrogate_prediction": row.get("surrogate_prediction", ""),
                "strict_penalty_guard_prediction": row.get("strict_penalty_guard_prediction", ""),
                "u2_guarded_prediction": row.get("u2_guarded_prediction", ""),
                "predicted_peeq_guarded_score": row.get("predicted_peeq_guarded_score", ""),
                "predicted_surfaceT_guarded_score": row.get("predicted_surfaceT_guarded_score", ""),
                "gnn_reward_prediction": row.get("gnn_reward_prediction", ""),
                "graph_pointer_policy_score": row.get("graph_pointer_policy_score", ""),
                "hybrid_score": row.get("hybrid_score", ""),
                "uncertainty_score": row.get("uncertainty_score", ""),
                "gnn_vs_surrogate_disagreement": row.get("gnn_vs_surrogate_disagreement", ""),
                "novelty_distance": row.get("novelty_distance", ""),
                "nearest_existing_teacher_strategy": row.get("nearest_existing_teacher_strategy", ""),
                "N24_N40_focused": row.get("N24_N40_focused", ""),
                "stricter_penalty_guard": row.get("stricter_penalty_guard", ""),
                "native_validation_N": row.get("native_validation_N", ""),
                "source_order_json": str(order_json),
                "order_hash": row.get("order_hash", ""),
                "job_name": paths["job_name"],
                "case_dir": paths["case_dir"],
                "scan_order_json": paths["scan_order_json"],
                "cae_path": paths["cae_path"],
                "inp_path": paths["inp_path"],
                "jnl_path": paths["jnl_path"],
                "generation_log_path": paths["generation_log_path"],
                "teacher_validated": False,
                "teacher_validation_status": "NOT_RUN",
                "solver_status": "NOT_SUBMITTED",
                "scan_order": json.dumps(order, separators=(",", ":")),
            }
        )

    if set(n_counts) != set(EXPECTED_NS):
        errors.append("N values {} != {}".format(sorted(n_counts), list(EXPECTED_NS)))
    for n, expected in EXPECTED_COUNTS.items():
        if n_counts[n] != expected:
            errors.append("N{} count {} != {}".format(n, n_counts[n], expected))
    duplicates = [name for name, count in strategy_counts.items() if count > 1]
    if duplicates:
        errors.append("duplicate handoff_strategy_name: {}".format(duplicates))
    missing_bases = [str(path) for path in BASES.values() if not path.exists()]
    if missing_bases:
        errors.append("missing base CAE files: {}".format(missing_bases))

    audit, audit_errors = safety_audit()
    errors.extend(audit_errors)

    fieldnames = [
        "run_id", "batch_name", "selected_batch", "n", "handoff_strategy_name",
        "original_run48_candidate_id", "original_run48_strategy_name", "candidate_source",
        "generation_method", "selection_bucket", "priority_role",
        "surrogate_prediction", "strict_penalty_guard_prediction", "u2_guarded_prediction",
        "predicted_peeq_guarded_score", "predicted_surfaceT_guarded_score",
        "gnn_reward_prediction", "graph_pointer_policy_score",
        "hybrid_score", "uncertainty_score", "gnn_vs_surrogate_disagreement",
        "novelty_distance", "nearest_existing_teacher_strategy",
        "N24_N40_focused", "stricter_penalty_guard", "native_validation_N",
        "source_order_json", "order_hash", "job_name", "case_dir", "scan_order_json",
        "cae_path", "inp_path", "jnl_path", "generation_log_path", "teacher_validated",
        "teacher_validation_status", "solver_status", "scan_order",
    ]
    for path in (CASES_CSV, PLAN_CSV):
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(normalized)

    verdict = "PASS_RUN70_SMALLN_RECOVERY_FOCUSED_BATCH40_INPUT_READY_FOR_CAE_GENERATION" if not errors else "FAIL_RUN70_SMALLN_RECOVERY_FOCUSED_BATCH40_INPUT_INVALID"
    summary = {
        "verdict": verdict,
        "input_csv": str(INPUT_CSV),
        "scan_order_dir": str(SCAN_ORDER_DIR),
        "case_root": str(CASE_ROOT),
        "row_count": len(rows),
        "per_n_counts": {str(n): n_counts[n] for n in EXPECTED_NS},
        "selected_batch": "run68_smallN_recovery_focused_batch40",
        "batch_name": BATCH_NAME,
        "generation_plan": str(PLAN_CSV),
        "preflight_cases_csv": str(CASES_CSV),
        "future_cae_root_safety_audit": str(SAFETY_AUDIT_PATH),
        "future_cae_root_safety_verdict": audit["verdict"],
        "errors": errors,
        "warnings": warnings,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(verdict)
    print("summary={}".format(SUMMARY_PATH))
    print("generation_plan={}".format(PLAN_CSV))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())




