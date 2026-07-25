"""Validate Stage 3 run14 batch20 CAE/INP generation inputs."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
RUN_ID = "run_14_batch20_cae_inp_generation"
BATCH_NAME = "stage3_run13_batch20_surrogate_screened_v01"
RUN13_DIR = PROJECT_ROOT / "outputs" / "stage3_run_13_batch20_surrogate_screened_teacher_handoff"
INPUT_CSV = RUN13_DIR / "stage3_run13_batch20_candidate_orders.csv"
SCAN_ORDER_DIR = RUN13_DIR / "scan_orders"
CASE_ROOT = PROJECT_ROOT / "cae_model" / BATCH_NAME
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage3_run_14_batch20_cae_inp_generation"
SUMMARY_PATH = OUTPUT_DIR / "run14_input_validation_summary.json"
NORMALIZED_MANIFEST = OUTPUT_DIR / "run14_generation_plan.csv"

EXPECTED_NS = (12, 16, 24, 40)
BAD_SCHEMA_TOKENS = ("N12N12_", "N16N16_", "N24N24_", "N40N40_")
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def load_json(path: Path) -> dict:
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


def build_case_paths(n: int, strategy: str) -> dict[str, str]:
    case_dir = CASE_ROOT / ("N{}{}".format(n, strategy))
    job_name = "J2D_{}".format(strategy)
    return {
        "case_dir": str(case_dir),
        "job_name": job_name,
        "case_scan_order_json": str(case_dir / "scan_order_{}.json".format(strategy)),
        "cae_path": str(case_dir / "{}.cae".format(job_name)),
        "inp_path": str(case_dir / "{}.inp".format(job_name)),
        "jnl_path": str(case_dir / "{}.jnl".format(job_name)),
        "generation_log_path": str(case_dir / "{}.generation_log.json".format(job_name)),
    }


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

    if len(rows) != 20:
        errors.append("row count {} != 20".format(len(rows)))

    n_counts = Counter()
    strategy_counts = Counter()
    for idx, row in enumerate(rows, 1):
        try:
            n = int(row.get("n", ""))
        except ValueError:
            errors.append("row {} invalid n={!r}".format(idx, row.get("n")))
            continue
        n_counts[n] += 1
        strategy = row.get("handoff_strategy_name", "")
        strategy_counts[strategy] += 1
        if n not in EXPECTED_NS:
            errors.append("row {} unexpected N {}".format(idx, n))
        if not strategy:
            errors.append("row {} missing handoff_strategy_name".format(idx))
        if strategy and not SAFE_NAME.match(strategy):
            errors.append("row {} strategy is not filesystem-safe: {}".format(idx, strategy))

        order_json = Path(row.get("order_json", ""))
        if not order_json.is_absolute():
            order_json = SCAN_ORDER_DIR / order_json
        if not order_json.exists():
            candidate = SCAN_ORDER_DIR / "scan_order_{}.json".format(strategy)
            if candidate.exists():
                order_json = candidate
            else:
                errors.append("row {} missing order_json: {}".format(idx, order_json))
                continue
        try:
            data = load_json(order_json)
            order = data.get("scan_order")
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
            errors.append("row {} contains bad path schema token".format(idx))
        generated = [
            p
            for p in (Path(paths["cae_path"]), Path(paths["inp_path"]), Path(paths["jnl_path"]))
            if p.exists()
        ]
        if generated:
            errors.append("row {} generated CAE/INP/JNL already exists: {}".format(idx, generated))

        normalized.append(
            {
                "run_id": RUN_ID,
                "batch_name": BATCH_NAME,
                "n": n,
                "handoff_strategy_name": strategy,
                "original_run12_candidate_id": row.get("original_run12_candidate_id", ""),
                "candidate_family": row.get("candidate_family", ""),
                "selection_bucket": row.get("selection_bucket", ""),
                "predicted_reward_mean_all": row.get("predicted_reward_mean_all", ""),
                "predicted_rank_within_n": row.get("predicted_rank_within_n", ""),
                "source_order_json": str(order_json),
                "order_hash": row.get("order_hash", data.get("order_hash", "")),
                "job_name": paths["job_name"],
                "case_dir": paths["case_dir"],
                "scan_order_json": paths["case_scan_order_json"],
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
    for n in EXPECTED_NS:
        if n_counts[n] != 5:
            errors.append("N{} count {} != 5".format(n, n_counts[n]))
    duplicate_strategies = [name for name, count in strategy_counts.items() if count > 1]
    if duplicate_strategies:
        errors.append("duplicate handoff_strategy_name: {}".format(duplicate_strategies))

    verdict = (
        "PASS_RUN14_BATCH20_INPUT_READY_FOR_CAE_GENERATION"
        if not errors
        else "FAIL_RUN14_BATCH20_INPUT_INVALID"
    )

    fieldnames = [
        "run_id",
        "batch_name",
        "n",
        "handoff_strategy_name",
        "original_run12_candidate_id",
        "candidate_family",
        "selection_bucket",
        "predicted_reward_mean_all",
        "predicted_rank_within_n",
        "source_order_json",
        "order_hash",
        "job_name",
        "case_dir",
        "scan_order_json",
        "cae_path",
        "inp_path",
        "jnl_path",
        "generation_log_path",
        "teacher_validated",
        "teacher_validation_status",
        "solver_status",
        "scan_order",
    ]
    with NORMALIZED_MANIFEST.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized)

    summary = {
        "verdict": verdict,
        "input_csv": str(INPUT_CSV),
        "scan_order_dir": str(SCAN_ORDER_DIR),
        "case_root": str(CASE_ROOT),
        "row_count": len(rows),
        "per_n_counts": {str(n): n_counts[n] for n in EXPECTED_NS},
        "normalized_manifest": str(NORMALIZED_MANIFEST),
        "errors": errors,
        "warnings": warnings,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(verdict)
    print("summary={}".format(SUMMARY_PATH))
    print("normalized_manifest={}".format(NORMALIZED_MANIFEST))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
