"""Stage K2 preflight for N40 PPO v02 completion.

This script reads existing v02 artifacts only. It does not run Abaqus, open ODB
files, run solver/datacheck, enqueue jobs, or generate CAE/INP/JNL.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
NS = "stage3_ppo_rl_lam_fea_addendum_v02_targeted_N24_N40"
OUT_ROOT = PROJECT_ROOT / "outputs" / NS
K2_ROOT = OUT_ROOT / "stageK2_n40_completion"
CHECKS_DIR = K2_ROOT / "checks"

SELECTED_K = OUT_ROOT / "candidate_generation_v02" / "selected_batch32" / "v02_ppo_targeted_N24_N40_candidate_batch32.csv"
N40_SURROGATE = OUT_ROOT / "surrogate_v02" / "models" / "N40_surrogate_reward_v02.joblib"
N40_PARTIAL_CKPT = OUT_ROOT / "ppo_training_v02" / "checkpoints" / "N40_seed20260624_maskable_ppo_v02.zip"
SURROGATE_REPORT = PROJECT_ROOT / "docs" / NS / "PPO_V02_SURROGATE_REPORT.md"

CSV_OUT = CHECKS_DIR / "stageK2_preflight_n40_completion.csv"
JSON_OUT = CHECKS_DIR / "stageK2_preflight_n40_completion_summary.json"


def ensure_dirs() -> None:
    for path in [
        K2_ROOT,
        K2_ROOT / "checkpoints",
        K2_ROOT / "rollout_pool",
        K2_ROOT / "selected_batch32_K2" / "scan_orders",
        K2_ROOT / "tables",
        K2_ROOT / "reports",
        K2_ROOT / "plots",
        CHECKS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def write_rows(rows: list[dict]) -> None:
    with CSV_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["check", "passed", "severity", "detail"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ensure_dirs()
    rows: list[dict] = []

    def add(check: str, passed: bool, severity: str, detail: str) -> None:
        rows.append({"check": check, "passed": bool(passed), "severity": severity, "detail": detail})

    add("existing_selected_batch_exists", SELECTED_K.exists(), "FAIL", str(SELECTED_K))
    if SELECTED_K.exists():
        selected = pd.read_csv(SELECTED_K)
        counts = selected["n"].astype(int).value_counts().sort_index().to_dict()
        add("existing_N24_count_16", counts.get(24, 0) == 16, "FAIL", str(counts))
        add("existing_N40_count_16", counts.get(40, 0) == 16, "FAIL", str(counts))
        add("no_N12_N16_N32_in_existing_batch", not selected["n"].astype(int).isin([12, 16, 32]).any(), "FAIL", str(counts))
    else:
        counts = {}
    add("existing_N40_surrogate_exists", N40_SURROGATE.exists(), "FAIL", str(N40_SURROGATE))
    add("existing_partial_N40_checkpoint_exists", N40_PARTIAL_CKPT.exists(), "FAIL", str(N40_PARTIAL_CKPT))
    add("N40_surrogate_validation_report_exists", SURROGATE_REPORT.exists(), "WARNING", str(SURROGATE_REPORT))
    add("no_Abaqus_or_solver_required", True, "FAIL", "normal Python preflight only")
    add("no_CAE_INP_generation_attempted", True, "FAIL", "Stage K2 preflight does not generate CAE/INP/JNL")

    write_rows(rows)
    fail_count = sum(1 for r in rows if r["severity"] == "FAIL" and not r["passed"])
    warning_count = sum(1 for r in rows if r["severity"] == "WARNING" and not r["passed"])
    verdict = "FAIL_STAGEK2_N40_COMPLETION_PREFLIGHT_BLOCKED" if fail_count else ("WARNING_STAGEK2_N40_COMPLETION_PREFLIGHT_REVIEW" if warning_count else "PASS_STAGEK2_N40_COMPLETION_PREFLIGHT_READY")
    summary = {
        "verdict": verdict,
        "fail_count": fail_count,
        "warning_count": warning_count,
        "existing_selected_counts_by_N": {str(k): int(v) for k, v in counts.items()},
        "no_Abaqus": True,
        "no_ODB_opening": True,
        "no_solver": True,
        "no_CAE_INP_JNL": True,
    }
    JSON_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if not verdict.startswith("FAIL") else 1


if __name__ == "__main__":
    raise SystemExit(main())
