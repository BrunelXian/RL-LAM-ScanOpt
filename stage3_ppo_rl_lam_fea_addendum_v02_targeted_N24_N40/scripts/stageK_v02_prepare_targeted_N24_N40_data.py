"""Prepare PPO v02 targeted N24/N40 teacher-labelled dataset.

Reads frozen v01/Run78 evidence only. Does not run Abaqus, open ODB files,
generate CAE/INP/JNL, train models, or generate candidates.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
V01_NAMESPACE = "stage3_ppo_rl_lam_fea_addendum_v01"
V02_NAMESPACE = "stage3_ppo_rl_lam_fea_addendum_v02_targeted_N24_N40"
BRANCH_FALLBACK = "stage3-variable-n-graph-pointer-init-v01"
SUPPORTED_N = [24, 40]

COMBINED552 = PROJECT_ROOT / "outputs" / "stage3_run_78_final_evidence_freeze_package" / "FROZEN_stage3_native_combined552_teacher_dataset.csv"
PPO_V01_METRICS = PROJECT_ROOT / "outputs" / V01_NAMESPACE / "stageI_final_ppo_evidence_freeze" / "frozen_tables" / "FROZEN_PPO_batch32_teacher_metrics.csv"
PPO_V01_RANKING = PROJECT_ROOT / "outputs" / V01_NAMESPACE / "stageI_final_ppo_evidence_freeze" / "frozen_tables" / "FROZEN_PPO_batch32_teacher_metric_ranking_full.csv"

OUT_ROOT = PROJECT_ROOT / "outputs" / V02_NAMESPACE
DATA_DIR = OUT_ROOT / "data"
CHECKS_DIR = OUT_ROOT / "checks"
DOCS_DIR = PROJECT_ROOT / "docs" / V02_NAMESPACE

DATASET_CSV = DATA_DIR / "v02_targeted_N24_N40_teacher_dataset.csv"
SUMMARY_JSON = DATA_DIR / "v02_targeted_N24_N40_teacher_dataset_summary.json"
AUDIT_CSV = CHECKS_DIR / "v02_data_integrity_audit.csv"
AUDIT_JSON = CHECKS_DIR / "v02_data_integrity_audit_summary.json"


def ensure_dirs() -> None:
    for path in [
        DATA_DIR,
        OUT_ROOT / "surrogate_v02" / "models",
        OUT_ROOT / "surrogate_v02" / "tables",
        OUT_ROOT / "surrogate_v02" / "plots",
        OUT_ROOT / "ppo_training_v02" / "checkpoints",
        OUT_ROOT / "ppo_training_v02" / "logs",
        OUT_ROOT / "ppo_training_v02" / "tables",
        OUT_ROOT / "candidate_generation_v02" / "rollout_pool",
        OUT_ROOT / "candidate_generation_v02" / "selected_batch32" / "scan_orders",
        OUT_ROOT / "candidate_generation_v02" / "tables",
        OUT_ROOT / "tables",
        OUT_ROOT / "reports",
        OUT_ROOT / "plots",
        CHECKS_DIR,
        DOCS_DIR,
        PROJECT_ROOT / V02_NAMESPACE / "src",
        PROJECT_ROOT / V02_NAMESPACE / "scripts",
        PROJECT_ROOT / V02_NAMESPACE / "tests",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def git_branch() -> str:
    try:
        result = subprocess.run(["git", "-C", str(PROJECT_ROOT), "branch", "--show-current"], capture_output=True, text=True, timeout=10)
        return result.stdout.strip() or BRANCH_FALLBACK
    except Exception:
        return BRANCH_FALLBACK


def order_json_from_text(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    if text.startswith("["):
        nums = [int(x) for x in re.findall(r"-?\d+", text)]
    else:
        nums = [int(x) for x in re.findall(r"-?\d+", text)]
    return json.dumps(nums, separators=(",", ":"))


def order_compact_from_json(order_json: str) -> str:
    nums = [int(x) for x in re.findall(r"-?\d+", order_json)]
    return ",".join(str(x) for x in nums)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def canonical_combined(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["n"].astype(int).isin(SUPPORTED_N)].copy()
    rows = []
    for _, row in sub.iterrows():
        order_json = order_json_from_text(row.get("order_json", row.get("order_compact", "")))
        rows.append(
            {
                "dataset_source": "stage3_native_combined552",
                "teacher_metric_extracted": True,
                "n": int(row["n"]),
                "strategy_name": str(row.get("strategy_name", row.get("handoff_strategy_name", ""))),
                "source_strategy_name": str(row.get("strategy_name", row.get("handoff_strategy_name", ""))),
                "order_json": order_json,
                "order_compact": order_compact_from_json(order_json),
                "order_hash": str(row.get("order_hash", "")),
                "u2_range": float(row["u2_range"]),
                "peeq_max": float(row["peeq_max"]),
                "surface_t_proxy": float(row["surface_t_proxy"]),
                "mises_max": float(row["mises_max"]),
                "teacher_validation_status": str(row.get("teacher_validation_status", "")),
            }
        )
    return pd.DataFrame(rows)


def canonical_ppo_v01(metrics: pd.DataFrame, ranking: pd.DataFrame) -> pd.DataFrame:
    merged = metrics.merge(
        ranking[["strategy_name", "predicted_surrogate_reward_lex", "ppo_order_hash", "duplicate_vs_combined552", "duplicate_role"]].drop_duplicates("strategy_name"),
        left_on="handoff_strategy_name",
        right_on="strategy_name",
        how="left",
        suffixes=("", "_rank"),
    )
    sub = merged[merged["n"].astype(int).isin(SUPPORTED_N)].copy()
    rows = []
    for _, row in sub.iterrows():
        order_json = order_json_from_text(row.get("order_compact", ""))
        strategy = str(row["handoff_strategy_name"])
        rows.append(
            {
                "dataset_source": "ppo_v01_teacher_validated",
                "teacher_metric_extracted": True,
                "n": int(row["n"]),
                "strategy_name": f"ppo_v01::{strategy}",
                "source_strategy_name": strategy,
                "order_json": order_json,
                "order_compact": order_compact_from_json(order_json),
                "order_hash": str(row.get("order_hash", "")),
                "u2_range": float(row["u2_range"]),
                "peeq_max": float(row["peeq_max"]),
                "surface_t_proxy": float(row["surface_t_proxy_max_tensile_pa"]),
                "mises_max": float(row["mises_max"]),
                "teacher_validation_status": str(row.get("teacher_validation_status", "")),
                "ppo_v01_predicted_surrogate_reward_lex": row.get("predicted_surrogate_reward_lex", ""),
                "duplicate_vs_combined552": row.get("duplicate_vs_combined552", ""),
                "duplicate_role": row.get("duplicate_role", ""),
            }
        )
    return pd.DataFrame(rows)


def validate_order(n: int, order_json: str) -> bool:
    order = [int(x) for x in re.findall(r"-?\d+", str(order_json))]
    return len(order) == int(n) and sorted(order) == list(range(int(n)))


def audit_dataset(dataset: pd.DataFrame) -> str:
    rows: list[dict[str, Any]] = []

    def add(check: str, passed: bool, severity: str, detail: Any) -> None:
        rows.append({"check": check, "passed": bool(passed), "severity": severity, "detail": str(detail)})

    add("combined552_exists", COMBINED552.exists(), "FAIL", COMBINED552)
    add("ppo_v01_metrics_exists", PPO_V01_METRICS.exists(), "FAIL", PPO_V01_METRICS)
    add("row_count_412", len(dataset) == 412, "WARNING", len(dataset))
    add("only_N24_N40", set(dataset["n"].astype(int).unique()) <= {24, 40}, "FAIL", sorted(dataset["n"].unique()))
    add("no_N12_N16_N32", not dataset["n"].astype(int).isin([12, 16, 32]).any(), "FAIL", sorted(dataset["n"].unique()))
    counts = dataset.groupby(["n", "dataset_source"]).size().to_dict()
    add("expected_source_counts", counts == {(24, "ppo_v01_teacher_validated"): 8, (24, "stage3_native_combined552"): 190, (40, "ppo_v01_teacher_validated"): 8, (40, "stage3_native_combined552"): 206}, "WARNING", counts)
    for metric in ["u2_range", "peeq_max", "surface_t_proxy", "mises_max"]:
        add(f"metric_{metric}_present", metric in dataset.columns and dataset[metric].notna().all(), "FAIL", metric)
    add("valid_orders_all", all(validate_order(int(r.n), r.order_json) for r in dataset.itertuples()), "FAIL", "legal permutation check")
    add("unique_strategy_names", dataset["strategy_name"].is_unique, "FAIL", dataset["strategy_name"].duplicated().sum())
    ppo = dataset[dataset["dataset_source"] == "ppo_v01_teacher_validated"]
    add("ppo_v01_teacher_metric_extracted", len(ppo) == 16 and ppo["teacher_metric_extracted"].astype(bool).all(), "FAIL", len(ppo))
    write_csv(AUDIT_CSV, rows)
    fail_count = sum(1 for r in rows if r["severity"] == "FAIL" and not r["passed"])
    warn_count = sum(1 for r in rows if r["severity"] == "WARNING" and not r["passed"])
    verdict = "FAIL_V02_TARGETED_DATA_NOT_READY" if fail_count else ("WARNING_V02_TARGETED_DATA_REVIEW" if warn_count else "PASS_V02_TARGETED_DATA_READY")
    AUDIT_JSON.write_text(json.dumps({"verdict": verdict, "fail_count": fail_count, "warning_count": warn_count, "counts_by_N_source": {f"N{k[0]}::{k[1]}": int(v) for k, v in counts.items()}, "no_Abaqus": True, "no_ODB_opening": True, "no_solver": True}, indent=2), encoding="utf-8")
    return verdict


def main() -> int:
    ensure_dirs()
    combined = pd.read_csv(COMBINED552)
    ppo_metrics = pd.read_csv(PPO_V01_METRICS)
    ppo_ranking = pd.read_csv(PPO_V01_RANKING)
    dataset = pd.concat([canonical_combined(combined), canonical_ppo_v01(ppo_metrics, ppo_ranking)], ignore_index=True, sort=False)
    dataset = dataset.sort_values(["n", "dataset_source", "strategy_name"]).reset_index(drop=True)
    dataset.to_csv(DATASET_CSV, index=False)
    verdict = audit_dataset(dataset)
    summary = {
        "branch": git_branch(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "dataset_path": str(DATASET_CSV),
        "row_count": int(len(dataset)),
        "counts_by_N_source": {f"N{k[0]}::{k[1]}": int(v) for k, v in dataset.groupby(["n", "dataset_source"]).size().to_dict().items()},
        "no_Abaqus": True,
        "no_ODB_opening": True,
        "no_solver": True,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if not verdict.startswith("FAIL") else 1


if __name__ == "__main__":
    raise SystemExit(main())
