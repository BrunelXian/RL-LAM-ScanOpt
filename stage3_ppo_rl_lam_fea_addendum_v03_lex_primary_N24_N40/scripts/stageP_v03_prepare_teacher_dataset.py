"""Prepare PPO v03 lex-primary N24/N40 teacher dataset.

Reads existing teacher-labelled CSVs only. No Abaqus, ODB, solver, CAE/INP/JNL,
training, or candidate generation is performed here.
"""

from __future__ import annotations

import ast
import csv
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
BRANCH_FALLBACK = "stage3-variable-n-graph-pointer-init-v01"
NS = "stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40"
SUPPORTED_N = [24, 40]
EXPECTED_COUNTS = {
    ("stage3_native_combined552", 24): 190,
    ("stage3_native_combined552", 40): 206,
    ("ppo_v01_teacher_validated", 24): 8,
    ("ppo_v01_teacher_validated", 40): 8,
    ("ppo_v02K2_teacher_validated", 24): 16,
    ("ppo_v02K2_teacher_validated", 40): 16,
}
METRICS = ["u2_range", "peeq_max", "surface_t_proxy", "mises_max"]

COMBINED552 = PROJECT_ROOT / "outputs" / "stage3_run_78_final_evidence_freeze_package" / "FROZEN_stage3_native_combined552_teacher_dataset.csv"
V01_METRICS = PROJECT_ROOT / "outputs" / "stage3_ppo_rl_lam_fea_addendum_v01" / "stageI_final_ppo_evidence_freeze" / "frozen_tables" / "FROZEN_PPO_batch32_teacher_metrics.csv"
V01_RANKING = PROJECT_ROOT / "outputs" / "stage3_ppo_rl_lam_fea_addendum_v01" / "stageI_final_ppo_evidence_freeze" / "frozen_tables" / "FROZEN_PPO_batch32_teacher_metric_ranking_full.csv"
V02K2_METRICS = PROJECT_ROOT / "outputs" / "stage3_ppo_rl_lam_fea_addendum_v02_targeted_N24_N40" / "stageM_ODB_teacher_metric_extraction" / "stageM_v02K2_teacher_metrics.csv"
V02K2_RANKING = PROJECT_ROOT / "outputs" / "stage3_ppo_rl_lam_fea_addendum_v02_targeted_N24_N40" / "stageN_teacher_metric_ranking" / "tables" / "v02K2_teacher_metric_ranking_full.csv"

OUT_ROOT = PROJECT_ROOT / "outputs" / NS
DATA_DIR = OUT_ROOT / "data"
CHECKS_DIR = OUT_ROOT / "checks"
DOCS_DIR = PROJECT_ROOT / "docs" / NS
CODE_ROOT = PROJECT_ROOT / NS
DATASET_CSV = DATA_DIR / "v03_N24_N40_teacher_dataset.csv"
SUMMARY_JSON = DATA_DIR / "v03_N24_N40_teacher_dataset_summary.json"
AUDIT_CSV = CHECKS_DIR / "v03_data_integrity_audit.csv"
AUDIT_JSON = CHECKS_DIR / "v03_data_integrity_audit_summary.json"


def ensure_dirs() -> None:
    for path in [
        OUT_ROOT,
        DATA_DIR,
        CHECKS_DIR,
        OUT_ROOT / "surrogate_v03",
        OUT_ROOT / "surrogate_v03" / "models",
        OUT_ROOT / "surrogate_v03" / "tables",
        OUT_ROOT / "surrogate_v03" / "plots",
        OUT_ROOT / "ppo_training_v03",
        OUT_ROOT / "ppo_training_v03" / "checkpoints",
        OUT_ROOT / "ppo_training_v03" / "logs",
        OUT_ROOT / "ppo_training_v03" / "tables",
        OUT_ROOT / "candidate_generation_v03",
        OUT_ROOT / "candidate_generation_v03" / "rollout_pool",
        OUT_ROOT / "candidate_generation_v03" / "selected_batch32",
        OUT_ROOT / "candidate_generation_v03" / "selected_batch32" / "scan_orders",
        OUT_ROOT / "candidate_generation_v03" / "tables",
        OUT_ROOT / "tables",
        OUT_ROOT / "reports",
        OUT_ROOT / "plots",
        DOCS_DIR,
        CODE_ROOT / "src",
        CODE_ROOT / "scripts",
        CODE_ROOT / "tests",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def git_branch() -> str:
    try:
        result = subprocess.run(
            ["git", "-c", "safe.directory=E:/Projects/RL-LAM-ScanOpt", "-C", str(PROJECT_ROOT), "branch", "--show-current"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return result.stdout.strip() or BRANCH_FALLBACK
    except Exception:
        return BRANCH_FALLBACK


def metric_col(df: pd.DataFrame, metric: str) -> str | None:
    if metric in df.columns:
        return metric
    if metric == "surface_t_proxy":
        for col in ["surface_t_proxy_max_tensile_pa", "surface_t_proxy_pa", "surface_t_proxy_mpa"]:
            if col in df.columns:
                return col
    return None


def parse_order(value: Any) -> list[int]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            return [int(x) for x in ast.literal_eval(text)]
        except Exception:
            pass
    return [int(x) for x in re.findall(r"-?\d+", text)]


def order_json_compact(order: list[int]) -> tuple[str, str]:
    return json.dumps(order, separators=(",", ":")), ",".join(str(x) for x in order)


def validate_order(n: int, order: list[int]) -> bool:
    return int(n) in SUPPORTED_N and len(order) == int(n) and sorted(order) == list(range(int(n)))


def canonical_combined(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["n"].astype(int).isin(SUPPORTED_N)].copy()
    rows: list[dict[str, Any]] = []
    for _, row in sub.iterrows():
        order = parse_order(row.get("order_json", row.get("order_compact", "")))
        oj, oc = order_json_compact(order)
        rec = {
            "n": int(row["n"]),
            "strategy_name": str(row.get("strategy_name", row.get("handoff_strategy_name", ""))),
            "dataset_source": "stage3_native_combined552",
            "candidate_source": str(row.get("candidate_source", "")),
            "order_json": oj,
            "order_compact": oc,
            "order_hash": str(row.get("order_hash", "")),
            "teacher_metrics_extracted": True,
            "teacher_validation_status": str(row.get("teacher_validation_status", "PASS_TEACHER_LABELLED_REFERENCE")),
        }
        for metric in METRICS:
            col = metric_col(sub, metric)
            rec[metric] = float(row[col]) if col and pd.notna(row[col]) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def canonical_v01(metrics: pd.DataFrame) -> pd.DataFrame:
    sub = metrics[metrics["n"].astype(int).isin(SUPPORTED_N)].copy()
    rows: list[dict[str, Any]] = []
    for _, row in sub.iterrows():
        order = parse_order(row.get("order_compact", row.get("order_json", "")))
        oj, oc = order_json_compact(order)
        rec = {
            "n": int(row["n"]),
            "strategy_name": str(row.get("handoff_strategy_name", row.get("strategy_name", ""))),
            "dataset_source": "ppo_v01_teacher_validated",
            "candidate_source": "PPO_checkpoint_inference",
            "order_json": oj,
            "order_compact": oc,
            "order_hash": str(row.get("order_hash", "")),
            "teacher_metrics_extracted": True,
            "teacher_validation_status": str(row.get("teacher_validation_status", "PASS_TEACHER_FIELDS_EXTRACTED")),
        }
        for metric in METRICS:
            col = metric_col(sub, metric)
            rec[metric] = float(row[col]) if col and pd.notna(row[col]) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def canonical_v02k2(metrics: pd.DataFrame) -> pd.DataFrame:
    sub = metrics[metrics["n"].astype(int).isin(SUPPORTED_N)].copy()
    rows: list[dict[str, Any]] = []
    for _, row in sub.iterrows():
        order = parse_order(row.get("order_compact", row.get("order_json", "")))
        oj, oc = order_json_compact(order)
        rec = {
            "n": int(row["n"]),
            "strategy_name": str(row.get("handoff_strategy_name", row.get("strategy_name", ""))),
            "dataset_source": "ppo_v02K2_teacher_validated",
            "candidate_source": "PPO_v02K2_or_v02_checkpoint_inference",
            "order_json": oj,
            "order_compact": oc,
            "order_hash": str(row.get("order_hash", "")),
            "teacher_metrics_extracted": True,
            "teacher_validation_status": str(row.get("teacher_validation_status", "PASS_TEACHER_FIELDS_EXTRACTED")),
            "predicted_reward_previous": row.get("predicted_reward", np.nan),
            "conservative_reward_previous": row.get("conservative_reward", np.nan),
        }
        for metric in METRICS:
            col = metric_col(sub, metric)
            rec[metric] = float(row[col]) if col and pd.notna(row[col]) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def add_targets(df: pd.DataFrame, combined_ref: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["surfaceT_only_false_positive_teacher"] = False
    for n in SUPPORTED_N:
        idx = out[out["n"] == n].index
        sub = out.loc[idx].copy()
        count = len(sub)
        for metric, label in [
            ("u2_range", "u2"),
            ("peeq_max", "peeq"),
            ("surface_t_proxy", "surface_t"),
            ("mises_max", "mises"),
        ]:
            ranks = sub[metric].rank(method="min", ascending=True)
            out.loc[idx, f"rank_{label}"] = ranks
            out.loc[idx, f"percentile_{label}"] = ranks / float(count)
            out.loc[idx, f"is_top10_{label}"] = ranks <= int(np.ceil(0.10 * count))
            out.loc[idx, f"is_top25_{label}"] = ranks <= int(np.ceil(0.25 * count))
        lex_order = sub.sort_values(["u2_range", "peeq_max", "surface_t_proxy"]).index
        lex_rank = pd.Series(np.arange(1, len(lex_order) + 1), index=lex_order)
        out.loc[idx, "lex_rank_u2_peeq_surfacet"] = lex_rank
        out.loc[idx, "lex_percentile_u2_peeq_surfacet"] = lex_rank / float(count)
        out.loc[idx, "is_top10_lex"] = lex_rank <= int(np.ceil(0.10 * count))
        out.loc[idx, "is_top25_lex"] = lex_rank <= int(np.ceil(0.25 * count))

        lex_score = 1.0 - (out.loc[idx, "lex_rank_u2_peeq_surfacet"] - 1.0) / max(1.0, count - 1.0)
        u2_score = 1.0 - (out.loc[idx, "rank_u2"] - 1.0) / max(1.0, count - 1.0)
        peeq_score = 1.0 - (out.loc[idx, "rank_peeq"] - 1.0) / max(1.0, count - 1.0)
        surface_score = 1.0 - (out.loc[idx, "rank_surface_t"] - 1.0) / max(1.0, count - 1.0)
        top10_bonus = out.loc[idx, "is_top10_lex"].astype(float) * 0.35 + out.loc[idx, "is_top10_u2"].astype(float) * 0.25
        top25_bonus = out.loc[idx, "is_top25_lex"].astype(float) * 0.15 + out.loc[idx, "is_top25_u2"].astype(float) * 0.10
        out.loc[idx, "reward_lex_primary_v03"] = np.clip(lex_score + top10_bonus + top25_bonus, 0.0, 1.7)

        peeq_gate = np.where(out.loc[idx, "percentile_peeq"] <= 0.55, 1.0, 0.55)
        u2_gate = np.where(out.loc[idx, "percentile_u2"] <= 0.50, 1.0, 0.45)
        surface_allowed = np.where((out.loc[idx, "percentile_u2"] <= 0.50) & (out.loc[idx, "percentile_peeq"] <= 0.60), surface_score, 0.0)
        surface_false_positive = (out.loc[idx, "percentile_surface_t"] <= 0.25) & (out.loc[idx, "percentile_u2"] > 0.50)
        out.loc[idx, "surfaceT_only_false_positive_teacher"] = surface_false_positive
        out.loc[idx, "reward_u2_guarded_v03"] = np.clip(0.72 * u2_score * peeq_gate + 0.20 * peeq_score * u2_gate + 0.08 * surface_allowed - 0.35 * surface_false_positive.astype(float), 0.0, 1.0)

        out.loc[idx, "reward_topk_classifier_v03"] = ((out.loc[idx, "is_top25_lex"]) | (out.loc[idx, "is_top25_u2"])).astype(int)
        ref_n = combined_ref[combined_ref["n"] == n].copy()
        ref_best_u2 = float(ref_n["u2_range"].min())
        ref_best_key = tuple(ref_n.sort_values(["u2_range", "peeq_max", "surface_t_proxy"]).iloc[0][["u2_range", "peeq_max", "surface_t_proxy"]])
        u2_ratio = ref_best_u2 / out.loc[idx, "u2_range"].astype(float)
        near_best_u2 = np.clip((u2_ratio - 0.35) / 0.65, 0.0, 1.0)
        lex_ref_rank = []
        ref_keys = [tuple(x) for x in ref_n[["u2_range", "peeq_max", "surface_t_proxy"]].to_numpy()]
        for _, row in out.loc[idx].iterrows():
            key = (row["u2_range"], row["peeq_max"], row["surface_t_proxy"])
            lex_ref_rank.append(1 + sum(k < key for k in ref_keys))
        lex_ref_rank = pd.Series(lex_ref_rank, index=idx)
        near_best_lex = 1.0 - (lex_ref_rank - 1.0) / max(1.0, len(ref_n) - 1.0)
        out.loc[idx, "ref_lex_rank_vs_combined552"] = lex_ref_rank
        out.loc[idx, "reward_record_seeking_v03"] = np.clip(0.65 * near_best_u2 + 0.35 * near_best_lex - 0.25 * surface_false_positive.astype(float), 0.0, 1.0)
        out.loc[idx, "combined552_best_u2"] = ref_best_u2
        out.loc[idx, "combined552_best_lex_key"] = str(ref_best_key)
    return out


def write_audit(df: pd.DataFrame, verdict: str) -> None:
    rows: list[dict[str, Any]] = []

    def add(check: str, passed: bool, severity: str, detail: Any) -> None:
        rows.append({"check": check, "passed": bool(passed), "severity": severity, "detail": str(detail)})

    add("combined552_exists", COMBINED552.exists(), "FAIL", COMBINED552)
    add("v01_metrics_exists", V01_METRICS.exists(), "FAIL", V01_METRICS)
    add("v01_ranking_exists", V01_RANKING.exists(), "WARNING", V01_RANKING)
    add("v02K2_metrics_exists", V02K2_METRICS.exists(), "FAIL", V02K2_METRICS)
    add("v02K2_ranking_exists", V02K2_RANKING.exists(), "WARNING", V02K2_RANKING)
    add("row_count_444", len(df) == 444, "FAIL", len(df))
    add("only_N24_N40", set(df["n"].astype(int)) == {24, 40}, "FAIL", sorted(df["n"].astype(int).unique()))
    add("no_N12_N16_N32", not df["n"].astype(int).isin([12, 16, 32]).any(), "FAIL", sorted(df["n"].astype(int).unique()))
    counts = df.groupby(["dataset_source", "n"]).size().to_dict()
    add("counts_by_source_N_expected", all(counts.get(k, 0) == v for k, v in EXPECTED_COUNTS.items()), "FAIL", counts)
    for metric in METRICS:
        add(f"metric_{metric}_complete", metric in df.columns and df[metric].notna().all(), "FAIL", df[metric].isna().sum() if metric in df.columns else "missing")
    add("all_orders_legal", all(validate_order(int(r["n"]), parse_order(r["order_json"])) for _, r in df.iterrows()), "FAIL", "order_json legality")
    add("target_columns_created", all(c in df.columns for c in ["reward_lex_primary_v03", "reward_u2_guarded_v03", "reward_record_seeking_v03", "reward_topk_classifier_v03"]), "FAIL", "v03 reward targets")

    with AUDIT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["check", "passed", "severity", "detail"])
        writer.writeheader()
        writer.writerows(rows)
    fail_count = sum(1 for r in rows if r["severity"] == "FAIL" and not r["passed"])
    warn_count = sum(1 for r in rows if r["severity"] == "WARNING" and not r["passed"])
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "branch": git_branch(),
        "verdict": verdict,
        "fail_count": fail_count,
        "warning_count": warn_count,
        "row_count": int(len(df)),
        "counts_by_N_source": {f"{src}_N{n}": int(v) for (src, n), v in counts.items()},
        "dataset_path": str(DATASET_CSV),
        "no_Abaqus": True,
        "no_ODB_opening": True,
        "no_solver": True,
    }
    AUDIT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> int:
    ensure_dirs()
    combined = pd.read_csv(COMBINED552)
    v01 = pd.read_csv(V01_METRICS)
    v02 = pd.read_csv(V02K2_METRICS)
    c = canonical_combined(combined)
    p1 = canonical_v01(v01)
    p2 = canonical_v02k2(v02)
    df = pd.concat([c, p1, p2], ignore_index=True)
    df = add_targets(df, c)
    df.to_csv(DATASET_CSV, index=False)
    counts = df.groupby(["dataset_source", "n"]).size().to_dict()
    fail = len(df) != 444 or not all(counts.get(k, 0) == v for k, v in EXPECTED_COUNTS.items()) or not set(df["n"].astype(int)) == {24, 40}
    verdict = "FAIL_V03_LEX_PRIMARY_DATA_NOT_READY" if fail else "PASS_V03_LEX_PRIMARY_DATA_READY"
    write_audit(df, verdict)
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "branch": git_branch(),
        "verdict": verdict,
        "dataset_path": str(DATASET_CSV),
        "row_count": int(len(df)),
        "counts_by_N_source": {f"{src}_N{n}": int(v) for (src, n), v in counts.items()},
        "reward_targets": ["reward_lex_primary_v03", "reward_u2_guarded_v03", "reward_topk_classifier_v03", "reward_record_seeking_v03"],
        "no_N12_N16_N32": True,
        "no_Abaqus": True,
        "no_ODB_opening": True,
        "no_solver": True,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if verdict.startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
