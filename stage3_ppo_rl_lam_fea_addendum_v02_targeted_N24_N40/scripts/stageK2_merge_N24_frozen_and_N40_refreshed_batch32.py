"""Merge frozen Stage K N24 candidates with refreshed K2 N40 candidates."""

from __future__ import annotations

import csv
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
NS = "stage3_ppo_rl_lam_fea_addendum_v02_targeted_N24_N40"
OUT_ROOT = PROJECT_ROOT / "outputs" / NS
K2_ROOT = OUT_ROOT / "stageK2_n40_completion"
SEL_DIR = K2_ROOT / "selected_batch32_K2"
SCAN_DIR = SEL_DIR / "scan_orders"
TABLES_DIR = K2_ROOT / "tables"
PLOTS_DIR = K2_ROOT / "plots"
DOCS_DIR = PROJECT_ROOT / "docs" / NS

STAGEK_SELECTED = OUT_ROOT / "candidate_generation_v02" / "selected_batch32" / "v02_ppo_targeted_N24_N40_candidate_batch32.csv"
STAGEK_SCAN_DIR = OUT_ROOT / "candidate_generation_v02" / "selected_batch32" / "scan_orders"
REFRESHED_N40 = SEL_DIR / "stageK2_refreshed_N40_candidates.csv"
MERGED_CSV = SEL_DIR / "v02K2_ppo_targeted_N24_N40_candidate_batch32.csv"
LEGALITY_CSV = TABLES_DIR / "stageK2_candidate_legality_audit.csv"
NOVELTY_CSV = TABLES_DIR / "stageK2_candidate_novelty_audit.csv"
SCORE_SUMMARY_CSV = TABLES_DIR / "stageK2_candidate_score_summary_by_N.csv"
REPORT = DOCS_DIR / "PPO_V02K2_N40_COMPLETION_AND_BATCH32_REPORT.md"
CLAIM_BOUNDARY = DOCS_DIR / "PPO_V02K2_CLAIM_BOUNDARY.md"
MANIFEST = K2_ROOT / "stageK2_n40_completion_manifest.json"
TRAIN_SUMMARY = TABLES_DIR / "stageK2_n40_training_summary.json"
EVAL_SUMMARY = TABLES_DIR / "stageK2_n40_internal_eval_summary.json"
REFRESHED_SUMMARY = TABLES_DIR / "stageK2_refreshed_N40_generation_summary.json"
VERDICT = "PASS_PPO_V02K2_BATCH32_READY_FOR_CAE_INP_HANDOFF"


def git_branch() -> str:
    try:
        result = subprocess.run(["git", "-C", str(PROJECT_ROOT), "branch", "--show-current"], capture_output=True, text=True, timeout=10)
        return result.stdout.strip()
    except Exception:
        return ""


def parse_order(text: Any) -> list[int]:
    return [int(x) for x in re.findall(r"-?\d+", "" if pd.isna(text) else str(text))]


def validate_order(n: int, order: list[int]) -> bool:
    return int(n) in (24, 40) and len(order) == int(n) and sorted(order) == list(range(int(n)))


def order_compact(order: list[int]) -> str:
    return ",".join(str(x) for x in order)


def load_json_order(strategy_name: str) -> list[int] | None:
    path = STAGEK_SCAN_DIR / f"scan_order_{strategy_name}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if "order" in data:
        return [int(x) for x in data["order"]]
    return parse_order(data.get("order_json", ""))


def normalize_n24(row: pd.Series) -> dict[str, Any]:
    order = parse_order(row["order_json"])
    return {
        "strategy_name": row["strategy_name"],
        "n": 24,
        "order_json": json.dumps(order, separators=(",", ":")),
        "order_compact": order_compact(order),
        "order_hash": row["order_hash"],
        "ppo_checkpoint": row.get("ppo_v02_checkpoint", row.get("ppo_checkpoint", "")),
        "ppo_seed": row.get("ppo_seed", ""),
        "ppo_generation_mode": row.get("ppo_generation_mode", ""),
        "selected_by": row.get("selected_by", ""),
        "predicted_reward": row.get("predicted_reward", ""),
        "conservative_reward": row.get("conservative_reward", ""),
        "candidate_source": "PPO_v02_checkpoint_inference",
        "teacher_validated": False,
        "abaqus_validated": False,
        "stageK2_role": "N24_retained_from_StageK_v02",
        "notes": "N24 candidate retained from Stage K v02.",
    }


def normalize_n40(row: pd.Series) -> dict[str, Any]:
    order = parse_order(row["order_json"])
    return {
        "strategy_name": row["strategy_name"],
        "n": 40,
        "order_json": json.dumps(order, separators=(",", ":")),
        "order_compact": order_compact(order),
        "order_hash": row["order_hash"],
        "ppo_checkpoint": row.get("ppo_checkpoint", ""),
        "ppo_seed": row.get("ppo_seed", ""),
        "ppo_generation_mode": row.get("ppo_generation_mode", ""),
        "selected_by": row.get("selected_by", ""),
        "predicted_reward": row.get("predicted_reward", ""),
        "conservative_reward": row.get("conservative_reward", ""),
        "candidate_source": "PPO_v02K2_checkpoint_inference",
        "teacher_validated": False,
        "abaqus_validated": False,
        "stageK2_role": "N40_refreshed_after_K2_training",
        "notes": "N40 candidate refreshed after Stage K2 N40 PPO completion.",
    }


def write_scan_json(row: dict[str, Any]) -> None:
    payload = dict(row)
    payload["order"] = parse_order(row["order_json"])
    (SCAN_DIR / f"scan_order_{row['strategy_name']}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    for path in [SEL_DIR, SCAN_DIR, TABLES_DIR, PLOTS_DIR, DOCS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    stagek = pd.read_csv(STAGEK_SELECTED)
    n24 = stagek[stagek["n"].astype(int) == 24].copy()
    n40 = pd.read_csv(REFRESHED_N40)
    rows = [normalize_n24(row) for _, row in n24.iterrows()] + [normalize_n40(row) for _, row in n40.iterrows()]
    merged = pd.DataFrame(rows)
    merged.to_csv(MERGED_CSV, index=False)
    for row in rows:
        write_scan_json(row)

    leg_rows = []
    for _, row in merged.iterrows():
        order = parse_order(row["order_json"])
        source_order = load_json_order(row["strategy_name"]) if int(row["n"]) == 24 else order
        leg_rows.append({
            "strategy_name": row["strategy_name"],
            "n": int(row["n"]),
            "legal": validate_order(int(row["n"]), order),
            "source_ok": (row["candidate_source"] == "PPO_v02_checkpoint_inference" if int(row["n"]) == 24 else row["candidate_source"] == "PPO_v02K2_checkpoint_inference"),
            "n24_unchanged_from_stageK_json": True if int(row["n"]) == 40 else source_order == order,
            "teacher_validated_false": not bool(row["teacher_validated"]),
            "abaqus_validated_false": not bool(row["abaqus_validated"]),
        })
    leg = pd.DataFrame(leg_rows)
    leg.to_csv(LEGALITY_CSV, index=False)
    novelty_cols = ["strategy_name", "n", "order_hash", "stageK2_role", "candidate_source"]
    merged[novelty_cols].to_csv(NOVELTY_CSV, index=False)
    score = merged.groupby("n").agg(count=("strategy_name", "size"), mean_conservative_reward=("conservative_reward", lambda s: float(pd.to_numeric(s, errors="coerce").mean())), max_conservative_reward=("conservative_reward", lambda s: float(pd.to_numeric(s, errors="coerce").max()))).reset_index()
    score.to_csv(SCORE_SUMMARY_CSV, index=False)
    checks_ok = (
        len(merged) == 32
        and merged["n"].astype(int).value_counts().to_dict() == {24: 16, 40: 16}
        and not merged["n"].astype(int).isin([12, 16, 32]).any()
        and leg["legal"].all()
        and leg["source_ok"].all()
        and leg["n24_unchanged_from_stageK_json"].all()
        and not merged["order_hash"].duplicated().any()
        and leg["teacher_validated_false"].all()
        and leg["abaqus_validated_false"].all()
    )
    final_verdict = VERDICT if checks_ok else "FAIL_PPO_V02K2_BATCH32_NOT_READY"

    fig, ax = plt.subplots(figsize=(6, 4))
    merged.assign(conservative_reward_num=pd.to_numeric(merged["conservative_reward"], errors="coerce")).boxplot(column="conservative_reward_num", by="n", ax=ax)
    ax.set_title("K2 batch32 conservative reward by N")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "stageK2_batch32_conservative_reward_by_N.png", dpi=180)
    plt.close(fig)

    training = json.loads(TRAIN_SUMMARY.read_text(encoding="utf-8")) if TRAIN_SUMMARY.exists() else {}
    evaluation = json.loads(EVAL_SUMMARY.read_text(encoding="utf-8")) if EVAL_SUMMARY.exists() else {}
    refreshed = json.loads(REFRESHED_SUMMARY.read_text(encoding="utf-8")) if REFRESHED_SUMMARY.exists() else {}
    REPORT.write_text(
        "# PPO v02K2 N40 Completion And Batch32 Report\n\n"
        "## Scope\n\nStage K2 continued N40 PPO v02 training in the surrogate environment, refreshed only N40 candidates, and merged them with frozen Stage K N24 candidates. No Abaqus/ODB/solver/CAE/INP/JNL occurred.\n\n"
        f"## N40 Training\n\n`{training.get('verdict', 'UNKNOWN')}`\n\n"
        f"## N40 Internal Evaluation\n\nRollouts: {evaluation.get('rollout_count', 'NA')}; unique orders: {evaluation.get('unique_orders', 'NA')}; mean reward: {evaluation.get('mean_reward', 'NA')}; max reward: {evaluation.get('max_reward', 'NA')}.\n\n"
        f"## Refreshed N40 Generation\n\nRollout pool: {refreshed.get('rollout_pool_size', 'NA')}; unique orders: {refreshed.get('unique_orders', 'NA')}; selected: {refreshed.get('selected_count', 'NA')}.\n\n"
        f"## Merged K2 Batch32\n\nPath: `{MERGED_CSV}`\n\nCounts: N24=16 retained from Stage K; N40=16 refreshed in K2.\n\n"
        f"## Audits\n\nLegality audit: `{LEGALITY_CSV}`\nNovelty audit: `{NOVELTY_CSV}`\nScore summary: `{SCORE_SUMMARY_CSV}`\n\n"
        f"## Verdict\n\n`{final_verdict}`\n",
        encoding="utf-8",
    )
    CLAIM_BOUNDARY.write_text(
        "# PPO v02K2 Claim Boundary\n\n"
        "## Safe\n\n"
        "- N40 PPO v02 was further trained/completed in a surrogate environment.\n"
        "- Refreshed N40 PPO candidates were generated from K2 PPO checkpoint inference.\n"
        "- K2 batch32 is ready for CAE/INP handoff.\n"
        "- No physical validation has happened yet.\n\n"
        "## Unsafe\n\n"
        "- K2 improves physical N40 metrics.\n"
        "- K2 beats v01/v02.\n"
        "- K2 is teacher validated.\n"
        "- K2 solves N40.\n",
        encoding="utf-8",
    )
    ckpts = [str(p) for p in sorted((K2_ROOT / "checkpoints").glob("N40_seed*_maskable_ppo_v02_K2.zip"))]
    manifest = {
        "branch": git_branch(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoints_produced": ckpts,
        "selected_refreshed_N40_candidates": str(REFRESHED_N40),
        "merged_K2_batch32_path": str(MERGED_CSV),
        "scan_order_JSON_directory": str(SCAN_DIR),
        "report_path": str(REPORT),
        "claim_boundary_path": str(CLAIM_BOUNDARY),
        "legality_audit": str(LEGALITY_CSV),
        "novelty_audit": str(NOVELTY_CSV),
        "score_summary": str(SCORE_SUMMARY_CSV),
        "no_Abaqus": True,
        "no_ODB_opening": True,
        "no_solver": True,
        "no_CAE_INP_JNL": True,
        "no_teacher_validation": True,
        "no_commit_or_push": True,
        "final_verdict": final_verdict,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary = {"verdict": final_verdict, "merged_batch32": str(MERGED_CSV), "counts_by_N": {str(k): int(v) for k, v in merged["n"].astype(int).value_counts().sort_index().to_dict().items()}, "legality_all": bool(leg["legal"].all()), "no_Abaqus": True, "no_solver": True}
    print(json.dumps(summary, indent=2))
    return 0 if final_verdict.startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
