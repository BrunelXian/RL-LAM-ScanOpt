"""Stage T preflight for final PPO expansion to a 320-case pool.

This script performs file/source inventory checks only. It does not run Abaqus,
open ODB files, generate CAE/INP/JNL files, train models, or submit jobs.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
NS = "stage3_ppo_final_expansion_224_to_320"
CODE_ROOT = ROOT / NS
OUT_ROOT = ROOT / "outputs" / NS
DOCS_ROOT = ROOT / "docs" / NS

PATHS = {
    "native_combined552": ROOT / "outputs" / "stage3_run_78_final_evidence_freeze_package" / "FROZEN_stage3_native_combined552_teacher_dataset.csv",
    "ppo_v01_metrics": ROOT / "outputs" / "stage3_ppo_rl_lam_fea_addendum_v01" / "stageI_final_ppo_evidence_freeze" / "frozen_tables" / "FROZEN_PPO_batch32_teacher_metrics.csv",
    "ppo_v02K2_metrics": ROOT / "outputs" / "stage3_ppo_rl_lam_fea_addendum_v02_targeted_N24_N40" / "stageM_ODB_teacher_metric_extraction" / "stageM_v02K2_teacher_metrics.csv",
    "ppo_v03_metrics": ROOT / "outputs" / "stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40" / "stageR_ODB_teacher_metric_extraction" / "stageR_v03_teacher_metrics.csv",
    "ppo_v01_checkpoint": ROOT / "outputs" / "stage3_ppo_rl_lam_fea_addendum_v01" / "ppo_training" / "checkpoints" / "maskable_ppo_lam_scan_order_final.zip",
    "ppo_v02K2_checkpoint": ROOT / "outputs" / "stage3_ppo_rl_lam_fea_addendum_v02_targeted_N24_N40" / "stageK2_n40_completion" / "checkpoints" / "N40_seed20260624_maskable_ppo_v02_K2.zip",
    "ppo_v03_N24_checkpoint": ROOT / "outputs" / "stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40" / "ppo_training_v03" / "checkpoints" / "N24_seed20260627_maskable_ppo_v03.zip",
    "ppo_v03_N40_checkpoint": ROOT / "outputs" / "stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40" / "ppo_training_v03" / "checkpoints" / "N40_seed20260627_maskable_ppo_v03.zip",
    "v01_rollout_pool": ROOT / "outputs" / "stage3_ppo_rl_lam_fea_addendum_v01" / "ppo_candidate_generation" / "rollout_pool" / "ppo_generated_rollout_pool.csv",
    "v02_rollout_pool": ROOT / "outputs" / "stage3_ppo_rl_lam_fea_addendum_v02_targeted_N24_N40" / "candidate_generation_v02" / "rollout_pool" / "v02_ppo_rollout_pool.csv",
    "v02K2_rollout_pool": ROOT / "outputs" / "stage3_ppo_rl_lam_fea_addendum_v02_targeted_N24_N40" / "stageK2_n40_completion" / "rollout_pool" / "stageK2_n40_ppo_rollout_pool.csv",
    "v03_rollout_pool": ROOT / "outputs" / "stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40" / "candidate_generation_v03" / "rollout_pool" / "v03_ppo_rollout_pool.csv",
}

SUBDIRS = [
    "checks",
    "rollout_pools",
    "selected_candidates",
    "selected_candidates/batches",
    "selected_candidates/scan_orders",
    "audits",
    "handoff_preview",
    "reports",
    "plots",
]


def git_branch() -> str:
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=str(ROOT),
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def count_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(len(pd.read_csv(path)))
    except Exception:
        return None


def n_counts(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
        col = "n" if "n" in df.columns else ("N" if "N" in df.columns else None)
        if col is None:
            return {}
        return {str(int(k)): int(v) for k, v in df[col].dropna().astype(int).value_counts().sort_index().items()}
    except Exception:
        return {}


def write_selection_protocol() -> Path:
    DOCS_ROOT.mkdir(parents=True, exist_ok=True)
    path = DOCS_ROOT / "PPO_FINAL_EXPANSION_SELECTION_PROTOCOL.md"
    path.write_text(
        """# PPO Final Expansion Selection Protocol

## Purpose

Stage T creates a fixed-budget 224-case PPO-generated expansion set intended to bring the cumulative PPO teacher-validation pool from 96 cases to 320 cases after future validation.

This is not another open-ended reward-redesign loop. Stage S showed weak surrogate-to-teacher alignment for v03, so Stage T uses existing PPO checkpoints and controlled selection buckets to build a broad, auditable candidate pool.

## Allocation

| N | Selected candidates |
|---:|---:|
| 12 | 32 |
| 16 | 32 |
| 24 | 80 |
| 40 | 80 |
| **Total** | **224** |

## Batch Structure

| Batch | Allocation |
|---|---|
| final_expansion_batch01 | N12=16, N16=16 |
| final_expansion_batch02 | N12=16, N16=16 |
| final_expansion_batch03 | N24=32 |
| final_expansion_batch04 | N24=32 |
| final_expansion_batch05 | N24=16, N40=16 |
| final_expansion_batch06 | N40=32 |
| final_expansion_batch07 | N40=32 |

## Selection Buckets

| Bucket | Target share | Role |
|---|---:|---|
| quality-seeking | 35% | Prefer high available PPO/surrogate score from existing models. |
| diversity-seeking | 25% | Maximize scan-order distance from already selected candidates. |
| industrial-efficiency-seeking | 20% | Prefer smoother/shorter proxy paths using sequence descriptors. |
| novelty-seeking | 10% | Prefer candidates distant from combined552 and previous PPO pools. |
| baseline-proximity / conventional-comparison | 10% | Preserve candidates near recognizable conventional patterns for later comparison. |

## Industrial-Efficiency Proxy Descriptors

The expansion records:

- `mean_abs_jump`
- `max_abs_jump`
- `long_jump_count`
- `adjacent_fraction`
- `total_travel_proxy`
- `jump_variance`
- `local_continuity_score`
- `path_complexity_score`

These are sequence descriptors only. They are not physical teacher metrics and must not be claimed as physically validated efficiency improvements until separately justified or validated.

## Claim Boundary

Safe after Stage T: a legal PPO-generated 224-case candidate expansion set is ready for later CAE/INP handoff.

Unsafe after Stage T: physical improvement, teacher validation, industrial efficiency validation, or superiority over combined552.
""",
        encoding="utf-8",
    )
    return path


def main() -> None:
    for rel in SUBDIRS:
        (OUT_ROOT / rel).mkdir(parents=True, exist_ok=True)
    (CODE_ROOT / "scripts").mkdir(parents=True, exist_ok=True)
    (CODE_ROOT / "src").mkdir(parents=True, exist_ok=True)
    (CODE_ROOT / "tests").mkdir(parents=True, exist_ok=True)
    DOCS_ROOT.mkdir(parents=True, exist_ok=True)

    branch = git_branch()
    rows = []
    for label, path in PATHS.items():
        required = label not in {"v02_rollout_pool", "v02K2_rollout_pool", "v03_rollout_pool"}
        exists = path.exists()
        rows.append(
            {
                "label": label,
                "path": str(path),
                "required": bool(required),
                "exists": bool(exists),
                "row_count": count_rows(path) if path.suffix.lower() == ".csv" else None,
                "n_counts": json.dumps(n_counts(path), sort_keys=True),
                "status": "PASS" if exists else ("FAIL" if required else "WARNING"),
            }
        )

    inventory = pd.DataFrame(rows)
    inventory_path = OUT_ROOT / "checks" / "stageT_source_inventory.csv"
    inventory.to_csv(inventory_path, index=False)

    required_missing = inventory[(inventory["required"]) & (~inventory["exists"])]
    validated_counts = {
        "ppo_v01": count_rows(PATHS["ppo_v01_metrics"]) or 0,
        "ppo_v02K2": count_rows(PATHS["ppo_v02K2_metrics"]) or 0,
        "ppo_v03": count_rows(PATHS["ppo_v03_metrics"]) or 0,
    }
    current_validated_total = int(sum(validated_counts.values()))
    target_total = 320
    remaining = target_total - current_validated_total

    protocol_path = write_selection_protocol()

    warnings: list[str] = []
    if current_validated_total != 96:
        warnings.append(f"expected_current_validated_total_96_observed_{current_validated_total}")
    combined_counts = n_counts(PATHS["native_combined552"])
    for n in ("12", "16", "24", "40"):
        if n not in combined_counts:
            warnings.append(f"native_combined552_missing_N{n}")

    if len(required_missing) > 0:
        verdict = "FAIL_STAGET_FINAL_EXPANSION_BLOCKED"
    elif warnings:
        verdict = "WARNING_STAGET_FINAL_EXPANSION_REVIEW"
    else:
        verdict = "PASS_STAGET_FINAL_EXPANSION_PREFLIGHT_READY"

    summary = {
        "branch": branch,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "namespace": NS,
        "current_validated_count_by_source": validated_counts,
        "current_PPO_teacher_validated_count": current_validated_total,
        "target_PPO_teacher_validated_count": target_total,
        "remaining_required_count": remaining,
        "source_inventory_path": str(inventory_path),
        "selection_protocol_path": str(protocol_path),
        "required_missing": required_missing["label"].tolist(),
        "warnings": warnings,
        "no_Abaqus": True,
        "no_ODB_opening": True,
        "no_ODB_extraction": True,
        "no_solver": True,
        "no_datacheck": True,
        "no_enqueue": True,
        "no_CAE_INP_JNL": True,
        "no_training": True,
        "no_commit_or_push": True,
        "verdict": verdict,
    }
    summary_path = OUT_ROOT / "checks" / "stageT_preflight_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
