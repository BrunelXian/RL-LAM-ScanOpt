"""Stage H PPO batch32 teacher-metric ranking and claim audit.

This script reads existing CSV/JSON evidence only. It does not run Abaqus,
open ODB files, run solver/datacheck, enqueue jobs, generate CAE/INP/JNL,
train models, or generate candidates.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import ast
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
BRANCH_FALLBACK = "stage3-variable-n-graph-pointer-init-v01"
NAMESPACE = "stage3_ppo_rl_lam_fea_addendum_v01"
SUPPORTED_N = [12, 16, 24, 40]
EXPECTED_COUNTS = {12: 8, 16: 8, 24: 8, 40: 8}
REF_COUNTS = {12: 78, 16: 78, 24: 190, 40: 206}

STAGE_G_METRICS = PROJECT_ROOT / "outputs" / NAMESPACE / "stageG_odb_teacher_metrics" / "tables" / "stageG_ppo_batch32_teacher_metrics.csv"
STAGE_G_SUMMARY = PROJECT_ROOT / "outputs" / NAMESPACE / "stageG_odb_teacher_metrics" / "tables" / "stageG_ppo_batch32_teacher_metrics_summary.json"
STAGE_D_SELECTED = PROJECT_ROOT / "outputs" / NAMESPACE / "ppo_candidate_generation" / "selected_batch32" / "ppo_policy_only_candidate_batch32.csv"
COMBINED552 = PROJECT_ROOT / "outputs" / "stage3_run_78_final_evidence_freeze_package" / "FROZEN_stage3_native_combined552_teacher_dataset.csv"

OUT_ROOT = PROJECT_ROOT / "outputs" / NAMESPACE / "stageH_teacher_metric_ranking"
TABLES_DIR = OUT_ROOT / "tables"
REPORTS_DIR = OUT_ROOT / "reports"
PLOTS_DIR = OUT_ROOT / "plots"
CHECKS_DIR = OUT_ROOT / "checks"
DOCS_DIR = PROJECT_ROOT / "docs" / NAMESPACE

AUDIT_CSV = CHECKS_DIR / "stageH_input_integrity_audit.csv"
AUDIT_JSON = CHECKS_DIR / "stageH_input_integrity_audit_summary.json"
ANALYSIS_DATASET_CSV = TABLES_DIR / "combined552_plus_ppo32_analysis_dataset.csv"
PPO_RANKING_CSV = TABLES_DIR / "ppo_batch32_teacher_metric_ranking_full.csv"
PPO_SUMMARY_BY_N_CSV = TABLES_DIR / "ppo_batch32_summary_by_N.csv"
PPO_GLOBAL_SUMMARY_CSV = TABLES_DIR / "ppo_batch32_global_summary.csv"
PPO_NEW_RECORDS_CSV = TABLES_DIR / "ppo_batch32_new_record_candidates.csv"
PPO_TOPK_CSV = TABLES_DIR / "ppo_batch32_topk_candidates.csv"
ALIGNMENT_CSV = TABLES_DIR / "ppo_surrogate_vs_teacher_alignment.csv"
ALIGNMENT_JSON = TABLES_DIR / "ppo_surrogate_vs_teacher_alignment_summary.json"
RECOVERY_ANCHOR_CSV = TABLES_DIR / "ppo_recovery_anchor_duplicate_audit.csv"
REPORT_PATH = DOCS_DIR / "PPO_STAGEH_TEACHER_METRIC_RANKING_REPORT.md"
CLAIM_BOUNDARY_PATH = DOCS_DIR / "PPO_STAGEH_CLAIM_BOUNDARY.md"
MANIFEST_PATH = OUT_ROOT / "stageH_ppo_batch32_teacher_metric_ranking_manifest.json"

METRICS = ["u2_range", "peeq_max", "surface_t_proxy", "mises_max"]
PRIMARY_METRICS = ["u2_range", "peeq_max", "surface_t_proxy"]
LEX_KEYS = PRIMARY_METRICS
LEX_DIAGNOSTIC_KEYS = ["u2_range", "peeq_max", "surface_t_proxy", "mises_max"]


def ensure_dirs() -> None:
    for directory in [OUT_ROOT, TABLES_DIR, REPORTS_DIR, PLOTS_DIR, CHECKS_DIR, DOCS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def git_branch() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "branch", "--show-current"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        branch = result.stdout.strip()
        return branch or BRANCH_FALLBACK
    except Exception:
        return BRANCH_FALLBACK


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fieldnames:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def as_float(value: Any) -> float:
    try:
        if pd.isna(value):
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def parse_order_value(value: Any) -> list[int] | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.startswith("["):
            return [int(x) for x in ast.literal_eval(text)]
        normalized = text.replace(";", ",").replace("-", ",")
        return [int(float(x.strip())) for x in normalized.split(",") if x.strip()]
    except Exception:
        return None


def metric_col(df: pd.DataFrame, metric: str, source: str) -> str | None:
    if metric in df.columns:
        return metric
    if metric == "surface_t_proxy":
        options = ["surface_t_proxy_max_tensile_pa", "surface_t_proxy_pa", "surface_t_proxy_mpa"]
        for option in options:
            if option in df.columns:
                return option
    return None


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stage_g = pd.read_csv(STAGE_G_METRICS)
    stage_d = pd.read_csv(STAGE_D_SELECTED)
    combined = pd.read_csv(COMBINED552)
    return stage_g, stage_d, combined


def input_audit(stage_g: pd.DataFrame, stage_d: pd.DataFrame, combined: pd.DataFrame) -> str:
    rows: list[dict[str, Any]] = []

    def add(check: str, passed: bool, severity: str, detail: str) -> None:
        rows.append({"check": check, "passed": bool(passed), "severity": severity, "detail": detail})

    add("stageG_metrics_exists", STAGE_G_METRICS.exists(), "FAIL", str(STAGE_G_METRICS))
    add("stageD_selected_exists", STAGE_D_SELECTED.exists(), "FAIL", str(STAGE_D_SELECTED))
    add("combined552_exists", COMBINED552.exists(), "FAIL", str(COMBINED552))
    add("stageG_row_count_32", len(stage_g) == 32, "FAIL", f"rows={len(stage_g)}")
    add("stageD_row_count_32", len(stage_d) == 32, "FAIL", f"rows={len(stage_d)}")
    add("combined552_row_count_552", len(combined) == 552, "FAIL", f"rows={len(combined)}")
    add("stageG_no_N32", set(stage_g["n"].dropna().astype(int)) <= set(SUPPORTED_N), "FAIL", str(sorted(stage_g["n"].unique())))
    add("stageD_no_N32", set(stage_d["n"].dropna().astype(int)) <= set(SUPPORTED_N), "FAIL", str(sorted(stage_d["n"].unique())))
    add("combined552_no_N32_primary", set(combined["n"].dropna().astype(int)) <= set(SUPPORTED_N), "FAIL", str(sorted(combined["n"].unique())))
    add("stageG_counts_by_N", stage_g["n"].value_counts().sort_index().to_dict() == EXPECTED_COUNTS, "FAIL", str(stage_g["n"].value_counts().sort_index().to_dict()))
    add("stageD_counts_by_N", stage_d["n"].value_counts().sort_index().to_dict() == EXPECTED_COUNTS, "FAIL", str(stage_d["n"].value_counts().sort_index().to_dict()))
    add("combined552_counts_by_N", combined["n"].value_counts().sort_index().to_dict() == REF_COUNTS, "FAIL", str(combined["n"].value_counts().sort_index().to_dict()))

    for metric in METRICS:
        add(f"stageG_metric_{metric}_mapped", metric_col(stage_g, metric, "stageG") is not None, "FAIL", str(metric_col(stage_g, metric, "stageG")))
        add(f"combined552_metric_{metric}_mapped", metric_col(combined, metric, "combined") is not None, "FAIL", str(metric_col(combined, metric, "combined")))

    status_ok = "odb_extraction_status" in stage_g.columns and stage_g["odb_extraction_status"].astype(str).str.contains("PASS", case=False, na=False).all()
    teacher_ok = "teacher_validation_status" in stage_g.columns and stage_g["teacher_validation_status"].astype(str).str.contains("PASS", case=False, na=False).all()
    add("stageG_extraction_success_32", status_ok, "FAIL", stage_g.get("odb_extraction_status", pd.Series(dtype=str)).value_counts().to_dict())
    add("stageG_teacher_status_mappable", teacher_ok, "FAIL", stage_g.get("teacher_validation_status", pd.Series(dtype=str)).value_counts().to_dict())
    add("stageG_final_step_metadata_available", "final_step_name" in stage_g.columns and "final_frame_time" in stage_g.columns, "WARNING", "final_step_name/final_frame_time")
    fields_ok = "extracted_field_names" in stage_g.columns and stage_g["extracted_field_names"].astype(str).str.contains("U").all() and stage_g["extracted_field_names"].astype(str).str.contains("PEEQ").all() and stage_g["extracted_field_names"].astype(str).str.contains("S").all() and stage_g["extracted_field_names"].astype(str).str.contains("NT11|NT", regex=True).all()
    add("stageG_required_fields_U_PEEQ_S_NT_visible", fields_ok, "WARNING", "checked extracted_field_names")

    stage_g_names = set(stage_g["handoff_strategy_name"].astype(str))
    stage_d_names = set(stage_d["strategy_name"].astype(str))
    add("stageD_stageG_strategy_names_match", stage_g_names == stage_d_names, "FAIL", f"missing_in_G={sorted(stage_d_names-stage_g_names)} missing_in_D={sorted(stage_g_names-stage_d_names)}")
    add("stageD_candidate_source_all_PPO", "candidate_source" in stage_d.columns and (stage_d["candidate_source"].astype(str) == "PPO_checkpoint_inference").all(), "FAIL", stage_d.get("candidate_source", pd.Series(dtype=str)).value_counts().to_dict())
    add("stageD_predicted_surrogate_reward_lex_exists", "predicted_surrogate_reward_lex" in stage_d.columns, "FAIL", "predicted_surrogate_reward_lex")
    add("stageD_order_hash_exists", "order_hash" in stage_d.columns, "FAIL", "order_hash")
    anchor = stage_d[stage_d["strategy_name"].astype(str) == "PPOV01_N12_B02_surrogate_top"]
    anchor_ok = len(anchor) == 1 and boolish(anchor.iloc[0].get("duplicate_order_hash_in_combined552", False))
    add("recovery_anchor_duplicate_flag_exists", anchor_ok, "WARNING", "PPOV01_N12_B02_surrogate_top")

    write_csv(AUDIT_CSV, rows)
    fail_count = sum(1 for row in rows if row["severity"] == "FAIL" and not row["passed"])
    warning_count = sum(1 for row in rows if row["severity"] == "WARNING" and not row["passed"])
    verdict = "FAIL_STAGEH_INPUTS_NOT_READY" if fail_count else ("WARNING_STAGEH_INPUTS_REVIEW" if warning_count else "PASS_STAGEH_INPUTS_READY")
    summary = {
        "verdict": verdict,
        "fail_count": fail_count,
        "warning_count": warning_count,
        "stageG_rows": int(len(stage_g)),
        "stageD_rows": int(len(stage_d)),
        "combined552_rows": int(len(combined)),
        "stageG_counts_by_N": {str(k): int(v) for k, v in stage_g["n"].value_counts().sort_index().to_dict().items()},
        "stageD_counts_by_N": {str(k): int(v) for k, v in stage_d["n"].value_counts().sort_index().to_dict().items()},
        "combined552_counts_by_N": {str(k): int(v) for k, v in combined["n"].value_counts().sort_index().to_dict().items()},
        "audit_csv": str(AUDIT_CSV),
        "no_Abaqus": True,
        "no_ODB_opening": True,
        "no_solver": True,
    }
    AUDIT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return verdict


def canonical_combined(combined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in combined.iterrows():
        rec: dict[str, Any] = {
            "n": int(row["n"]),
            "strategy_name": row.get("strategy_name", row.get("handoff_strategy_name", "")),
            "order_json": row.get("order_json", ""),
            "order_compact": row.get("order_compact", ""),
            "order_hash": row.get("order_hash", ""),
            "dataset_source": "stage3_native_combined552",
            "is_ppo_candidate": False,
            "candidate_source": row.get("candidate_source", ""),
            "ppo_checkpoint": "",
            "ppo_generation_mode": "",
            "ppo_selection_tag": "",
            "predicted_surrogate_reward_lex": np.nan,
            "ppo_order_hash": "",
            "duplicate_vs_combined552": False,
            "duplicate_role": "",
            "teacher_metrics_extracted": True,
            "teacher_validation_status": row.get("teacher_validation_status", row.get("teacher_validation_status_run74", "")),
            "stageG_metric_source_file": "",
        }
        for metric in METRICS:
            col = metric_col(combined, metric, "combined")
            rec[metric] = as_float(row[col]) if col else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def canonical_ppo(stage_g: pd.DataFrame, stage_d: pd.DataFrame) -> pd.DataFrame:
    merged = stage_g.merge(stage_d, left_on="handoff_strategy_name", right_on="strategy_name", how="left", suffixes=("_stageG", "_stageD"))
    rows = []
    for _, row in merged.iterrows():
        rec: dict[str, Any] = {
            "n": int(row["n_stageG"] if "n_stageG" in row else row["n"]),
            "strategy_name": row["handoff_strategy_name"],
            "order_json": row.get("order_json", row.get("scan_order_json", "")),
            "order_compact": row.get("order_compact_stageG", row.get("order_compact", "")),
            "order_hash": row.get("order_hash_stageG", row.get("order_hash", "")),
            "dataset_source": "stage3_ppo_policy_only_batch32",
            "is_ppo_candidate": True,
            "candidate_source": row.get("candidate_source", "PPO_checkpoint_inference"),
            "ppo_checkpoint": row.get("ppo_checkpoint", ""),
            "ppo_generation_mode": row.get("generation_mode", ""),
            "ppo_selection_tag": row.get("selection_tag_stageD", row.get("selection_tag", "")),
            "predicted_surrogate_reward_lex": as_float(row.get("predicted_surrogate_reward_lex", np.nan)),
            "ppo_order_hash": row.get("order_hash_stageG", row.get("order_hash", "")),
            "duplicate_vs_combined552": boolish(row.get("duplicate_vs_combined552_stageG", row.get("duplicate_vs_combined552", False))),
            "duplicate_role": row.get("duplicate_role_stageG", row.get("duplicate_role", "")),
            "teacher_metrics_extracted": bool(str(row.get("odb_extraction_status", "")).startswith("PASS")),
            "teacher_validation_status": row.get("teacher_validation_status", "PASS_TEACHER_METRICS_EXTRACTED"),
            "stageG_metric_source_file": str(STAGE_G_METRICS),
            "final_step_name": row.get("final_step_name", ""),
            "final_frame_time": row.get("final_frame_time", np.nan),
            "extracted_field_names": row.get("extracted_field_names", ""),
            "completion_status": row.get("completion_status", ""),
            "odb_extraction_status": row.get("odb_extraction_status", ""),
        }
        rec["u2_range"] = as_float(row.get("u2_range", np.nan))
        rec["peeq_max"] = as_float(row.get("peeq_max", np.nan))
        rec["surface_t_proxy"] = as_float(row.get("surface_t_proxy_max_tensile_pa", row.get("surface_t_proxy", np.nan)))
        rec["mises_max"] = as_float(row.get("mises_max", np.nan))
        rows.append(rec)
    return pd.DataFrame(rows)


def minmax_norm(values: pd.Series) -> pd.Series:
    vmin = values.min()
    vmax = values.max()
    if pd.isna(vmin) or pd.isna(vmax) or math.isclose(float(vmin), float(vmax)):
        return pd.Series([0.0] * len(values), index=values.index)
    return (values - vmin) / (vmax - vmin)


def ref_position(ref_values: pd.Series, value: float) -> tuple[int, float, bool, bool, bool, bool]:
    ref = ref_values.dropna().sort_values().to_numpy()
    n = len(ref)
    if n == 0 or pd.isna(value):
        return 0, float("nan"), False, False, False, False
    pos = int(np.searchsorted(ref, value, side="left")) + 1
    percentile = pos / n
    beats_best = bool(value < ref[0])
    better_median = bool(value < np.median(ref))
    top10 = pos <= math.ceil(0.10 * n)
    top25 = pos <= math.ceil(0.25 * n)
    return pos, percentile, beats_best, better_median, top10, top25


def lex_key(row: pd.Series, keys: list[str]) -> tuple[float, ...]:
    return tuple(as_float(row.get(k, np.nan)) for k in keys)


def lex_position(ref: pd.DataFrame, row: pd.Series, keys: list[str]) -> tuple[int, float, bool, bool, bool]:
    tuples = sorted(tuple(lex_key(r, keys)) for _, r in ref.iterrows())
    value = lex_key(row, keys)
    n = len(tuples)
    if n == 0 or any(pd.isna(x) for x in value):
        return 0, float("nan"), False, False, False
    pos = sum(1 for item in tuples if item < value) + 1
    beats_best = value < tuples[0]
    top10 = pos <= math.ceil(0.10 * n)
    top25 = pos <= math.ceil(0.25 * n)
    return pos, pos / n, beats_best, top10, top25


def add_rankings(analysis: pd.DataFrame) -> pd.DataFrame:
    df = analysis.copy()
    for metric in METRICS:
        df[f"combined_rank_{metric}"] = np.nan
        df[f"combined_percentile_{metric}"] = np.nan
        df[f"combined_cost_norm_{metric}"] = np.nan
        df[f"ref_rank_{metric}"] = np.nan
        df[f"ref_percentile_{metric}"] = np.nan
        df[f"ref_best_{metric}"] = np.nan
        df[f"ref_median_{metric}"] = np.nan
        df[f"ppo_beats_ref_best_{metric}"] = False
        df[f"ppo_better_than_ref_median_{metric}"] = False
        df[f"ppo_top10pct_ref_{metric}"] = False
        df[f"ppo_top25pct_ref_{metric}"] = False
        df[f"ppo_rank_{metric}_within_N"] = np.nan
        df[f"ppo_best_{metric}_within_N"] = False

    df["lex_rank_combined"] = np.nan
    df["lex_percentile_combined"] = np.nan
    df["ppo_beats_ref_lex_best"] = False
    df["ppo_top10pct_ref_lex"] = False
    df["ppo_top25pct_ref_lex"] = False
    df["ppo_best_lex_within_N"] = False
    df["lex_diag_rank_combined"] = np.nan
    df["lex_diag_percentile_combined"] = np.nan
    df["teacher_lex_reward_rank_normalized"] = np.nan

    for n in SUPPORTED_N:
        mask_n = df["n"] == n
        ref = df[mask_n & (~df["is_ppo_candidate"])].copy()
        ppo = df[mask_n & (df["is_ppo_candidate"])].copy()
        all_n = df[mask_n].copy()
        for metric in METRICS:
            df.loc[mask_n, f"combined_rank_{metric}"] = all_n[metric].rank(method="min", ascending=True)
            df.loc[mask_n, f"combined_percentile_{metric}"] = df.loc[mask_n, f"combined_rank_{metric}"] / len(all_n)
            df.loc[mask_n, f"combined_cost_norm_{metric}"] = minmax_norm(all_n[metric])
            ref_best = ref[metric].min()
            ref_median = ref[metric].median()
            df.loc[mask_n, f"ref_best_{metric}"] = ref_best
            df.loc[mask_n, f"ref_median_{metric}"] = ref_median
            ppo_metric_ranks = ppo[metric].rank(method="min", ascending=True)
            for idx, row in ppo.iterrows():
                pos, pct, beats, median, top10, top25 = ref_position(ref[metric], as_float(row[metric]))
                df.loc[idx, f"ref_rank_{metric}"] = pos
                df.loc[idx, f"ref_percentile_{metric}"] = pct
                df.loc[idx, f"ppo_beats_ref_best_{metric}"] = beats
                df.loc[idx, f"ppo_better_than_ref_median_{metric}"] = median
                df.loc[idx, f"ppo_top10pct_ref_{metric}"] = top10
                df.loc[idx, f"ppo_top25pct_ref_{metric}"] = top25
                df.loc[idx, f"ppo_rank_{metric}_within_N"] = ppo_metric_ranks.loc[idx]
            if len(ppo_metric_ranks):
                df.loc[ppo_metric_ranks[ppo_metric_ranks == 1].index, f"ppo_best_{metric}_within_N"] = True

        sorted_all = all_n.sort_values(LEX_KEYS, ascending=True)
        lex_ranks = pd.Series(range(1, len(sorted_all) + 1), index=sorted_all.index)
        df.loc[lex_ranks.index, "lex_rank_combined"] = lex_ranks
        df.loc[lex_ranks.index, "lex_percentile_combined"] = lex_ranks / len(sorted_all)
        df.loc[lex_ranks.index, "teacher_lex_reward_rank_normalized"] = 1.0 - ((lex_ranks - 1) / max(len(sorted_all) - 1, 1))

        sorted_diag = all_n.sort_values(LEX_DIAGNOSTIC_KEYS, ascending=True)
        diag_ranks = pd.Series(range(1, len(sorted_diag) + 1), index=sorted_diag.index)
        df.loc[diag_ranks.index, "lex_diag_rank_combined"] = diag_ranks
        df.loc[diag_ranks.index, "lex_diag_percentile_combined"] = diag_ranks / len(sorted_diag)

        ppo_lex_ranks = df.loc[ppo.index, "lex_rank_combined"]
        if len(ppo_lex_ranks):
            df.loc[ppo_lex_ranks[ppo_lex_ranks == ppo_lex_ranks.min()].index, "ppo_best_lex_within_N"] = True
        for idx, row in ppo.iterrows():
            pos, pct, beats, top10, top25 = lex_position(ref, row, LEX_KEYS)
            df.loc[idx, "ref_rank_lex"] = pos
            df.loc[idx, "ref_percentile_lex"] = pct
            df.loc[idx, "ppo_beats_ref_lex_best"] = beats
            df.loc[idx, "ppo_top10pct_ref_lex"] = top10
            df.loc[idx, "ppo_top25pct_ref_lex"] = top25

    return df


def summarize_by_n(ranked: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for n in SUPPORTED_N:
        ref = ranked[(ranked["n"] == n) & (~ranked["is_ppo_candidate"])]
        ppo = ranked[(ranked["n"] == n) & (ranked["is_ppo_candidate"])]
        rec: dict[str, Any] = {"n": n, "ppo_count": len(ppo), "reference_count": len(ref)}
        for metric in METRICS:
            rec[f"best_ppo_{metric}"] = ppo[metric].min()
            rec[f"best_reference_{metric}"] = ref[metric].min()
            rec[f"ppo_best_over_reference_best_ratio_{metric}"] = rec[f"best_ppo_{metric}"] / rec[f"best_reference_{metric}"] if rec[f"best_reference_{metric}"] else np.nan
            rec[f"median_ppo_{metric}"] = ppo[metric].median()
            rec[f"median_reference_{metric}"] = ref[metric].median()
            rec[f"ppo_median_over_reference_median_ratio_{metric}"] = rec[f"median_ppo_{metric}"] / rec[f"median_reference_{metric}"] if rec[f"median_reference_{metric}"] else np.nan
            rec[f"ppo_beats_reference_best_count_{metric}"] = int(ppo[f"ppo_beats_ref_best_{metric}"].sum())
            rec[f"ppo_top10pct_ref_count_{metric}"] = int(ppo[f"ppo_top10pct_ref_{metric}"].sum())
            rec[f"ppo_top25pct_ref_count_{metric}"] = int(ppo[f"ppo_top25pct_ref_{metric}"].sum())
        rec["best_ppo_lex_rank_combined"] = int(ppo["lex_rank_combined"].min())
        rec["best_ppo_ref_lex_rank"] = int(ppo["ref_rank_lex"].min())
        rec["ppo_beats_reference_lex_best_count"] = int(ppo["ppo_beats_ref_lex_best"].sum())
        rec["ppo_top10pct_ref_lex_count"] = int(ppo["ppo_top10pct_ref_lex"].sum())
        rec["ppo_top25pct_ref_lex_count"] = int(ppo["ppo_top25pct_ref_lex"].sum())
        best_row = ppo.sort_values("lex_rank_combined").iloc[0]
        rec["best_ppo_lex_strategy_name"] = best_row["strategy_name"]
        rec["best_ppo_lex_percentile_combined"] = best_row["lex_percentile_combined"]
        rows.append(rec)
    return pd.DataFrame(rows)


def global_summary(ranked: pd.DataFrame, by_n: pd.DataFrame) -> pd.DataFrame:
    ppo = ranked[ranked["is_ppo_candidate"]].copy()
    rows: list[dict[str, Any]] = []
    rows.append({"metric": "ppo_candidate_count", "value": int(len(ppo))})
    rows.append({"metric": "new_record_any_primary_metric_or_lex_count", "value": int(len(new_record_rows(ppo)))})
    rows.append({"metric": "top10_any_primary_metric_or_lex_count", "value": int(ppo[[*(f"ppo_top10pct_ref_{m}" for m in PRIMARY_METRICS), "ppo_top10pct_ref_lex"]].any(axis=1).sum())})
    rows.append({"metric": "top25_any_primary_metric_or_lex_count", "value": int(ppo[[*(f"ppo_top25pct_ref_{m}" for m in PRIMARY_METRICS), "ppo_top25pct_ref_lex"]].any(axis=1).sum())})
    for metric in METRICS:
        rows.append({"metric": f"total_beats_reference_best_{metric}", "value": int(ppo[f"ppo_beats_ref_best_{metric}"].sum())})
    rows.append({"metric": "total_beats_reference_lex_best", "value": int(ppo["ppo_beats_ref_lex_best"].sum())})
    rows.append({"metric": "N_with_top25_lex", "value": int((by_n["ppo_top25pct_ref_lex_count"] > 0).sum())})
    rows.append({"metric": "N_with_top10_lex", "value": int((by_n["ppo_top10pct_ref_lex_count"] > 0).sum())})
    return pd.DataFrame(rows)


def new_record_rows(ppo: pd.DataFrame) -> pd.DataFrame:
    flags = [f"ppo_beats_ref_best_{m}" for m in METRICS] + ["ppo_beats_ref_lex_best"]
    return ppo[ppo[flags].any(axis=1)].copy()


def topk_rows(ppo: pd.DataFrame) -> pd.DataFrame:
    flags = [f"ppo_top10pct_ref_{m}" for m in PRIMARY_METRICS] + [f"ppo_top25pct_ref_{m}" for m in PRIMARY_METRICS] + ["ppo_top10pct_ref_lex", "ppo_top25pct_ref_lex"]
    return ppo[ppo[flags].any(axis=1)].copy()


def surrogate_alignment(ppo: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = ppo.copy()
    df["predicted_surrogate_rank_within_all_ppo"] = df["predicted_surrogate_reward_lex"].rank(method="min", ascending=False)
    df["teacher_lex_rank_within_all_ppo"] = df.groupby("n")["lex_rank_combined"].rank(method="min", ascending=True)
    df["surrogate_optimism_gap_rank"] = df["teacher_lex_rank_within_all_ppo"] - df["predicted_surrogate_rank_within_all_ppo"]
    q_pred = df["predicted_surrogate_reward_lex"].quantile(0.75)
    q_teacher_bad = df["lex_percentile_combined"].quantile(0.75)
    q_teacher_good = df["lex_percentile_combined"].quantile(0.25)
    df["surrogate_false_positive_flag"] = (df["predicted_surrogate_reward_lex"] >= q_pred) & (df["lex_percentile_combined"] >= q_teacher_bad)
    df["surrogate_true_positive_flag"] = (df["predicted_surrogate_reward_lex"] >= q_pred) & (df["lex_percentile_combined"] <= q_teacher_good)

    summary: dict[str, Any] = {
        "overall_spearman_predicted_vs_teacher_reward": float(df["predicted_surrogate_reward_lex"].corr(df["teacher_lex_reward_rank_normalized"], method="spearman")),
        "overall_pearson_predicted_vs_teacher_reward": float(df["predicted_surrogate_reward_lex"].corr(df["teacher_lex_reward_rank_normalized"], method="pearson")),
        "false_positive_count": int(df["surrogate_false_positive_flag"].sum()),
        "true_positive_count": int(df["surrogate_true_positive_flag"].sum()),
        "by_N": {},
    }
    for n in SUPPORTED_N:
        sub = df[df["n"] == n]
        summary["by_N"][str(n)] = {
            "count": int(len(sub)),
            "spearman": None if sub["predicted_surrogate_reward_lex"].nunique() < 2 or sub["teacher_lex_reward_rank_normalized"].nunique() < 2 else float(sub["predicted_surrogate_reward_lex"].corr(sub["teacher_lex_reward_rank_normalized"], method="spearman")),
            "pearson": None if sub["predicted_surrogate_reward_lex"].nunique() < 2 or sub["teacher_lex_reward_rank_normalized"].nunique() < 2 else float(sub["predicted_surrogate_reward_lex"].corr(sub["teacher_lex_reward_rank_normalized"], method="pearson")),
            "false_positive_count": int(sub["surrogate_false_positive_flag"].sum()),
            "true_positive_count": int(sub["surrogate_true_positive_flag"].sum()),
        }
    ALIGNMENT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return df, summary


def recovery_anchor_audit(ranked: pd.DataFrame) -> pd.DataFrame:
    ppo_anchor = ranked[(ranked["is_ppo_candidate"]) & (ranked["strategy_name"] == "PPOV01_N12_B02_surrogate_top")]
    rows = []
    if ppo_anchor.empty:
        rows.append({"strategy_name": "PPOV01_N12_B02_surrogate_top", "status": "NOT_FOUND"})
        return pd.DataFrame(rows)
    anchor = ppo_anchor.iloc[0]
    ref_same_n = ranked[(~ranked["is_ppo_candidate"]) & (ranked["n"] == int(anchor["n"]))].copy()
    matches = ref_same_n[ref_same_n["order_hash"].astype(str) == str(anchor["order_hash"])]
    match_mode = "order_hash"
    if matches.empty:
        anchor_order = parse_order_value(anchor.get("order_json")) or parse_order_value(anchor.get("order_compact"))
        parsed_matches = []
        if anchor_order is not None:
            for idx, ref in ref_same_n.iterrows():
                ref_order = parse_order_value(ref.get("order_json")) or parse_order_value(ref.get("order_compact"))
                if ref_order == anchor_order:
                    parsed_matches.append(idx)
        matches = ref_same_n.loc[parsed_matches]
        match_mode = "parsed_order_equality"
    for _, ref in matches.iterrows():
        rec = {
            "strategy_name": anchor["strategy_name"],
            "n": int(anchor["n"]),
            "order_hash": anchor["order_hash"],
            "duplicate_order_hash_source_strategy": ref["strategy_name"],
            "duplicate_order_hash_source_dataset": ref["dataset_source"],
            "duplicate_source_match_mode": match_mode,
            "ppo_order_hash": anchor["order_hash"],
            "source_order_hash": ref["order_hash"],
            "duplicate_role": anchor["duplicate_role"],
            "interpretation": "PPO recovered a known teacher-validated strategy; not a new novel PPO discovery",
        }
        all_match = True
        for metric in METRICS:
            diff = abs(as_float(anchor[metric]) - as_float(ref[metric]))
            tol = max(1e-12, abs(as_float(ref[metric])) * 1e-8)
            rec[f"{metric}_ppo"] = anchor[metric]
            rec[f"{metric}_reference"] = ref[metric]
            rec[f"{metric}_abs_diff"] = diff
            rec[f"{metric}_matches_within_tolerance"] = diff <= tol
            all_match = all_match and diff <= tol
        rec["all_metrics_match_within_tolerance"] = all_match
        rows.append(rec)
    if not rows:
        rows.append({
            "strategy_name": anchor["strategy_name"],
            "n": int(anchor["n"]),
            "order_hash": anchor["order_hash"],
            "duplicate_role": anchor["duplicate_role"],
            "status": "ORDER_HASH_SOURCE_NOT_FOUND_IN_COMBINED552",
            "interpretation": "duplicate flag exists but source row was not identified",
        })
    return pd.DataFrame(rows)


def plot_metric_distribution(ranked: pd.DataFrame, metric: str, filename: str, ylabel: str) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=False)
    for ax, n in zip(axes, SUPPORTED_N):
        ref = ranked[(ranked["n"] == n) & (~ranked["is_ppo_candidate"])][metric].dropna()
        ppo = ranked[(ranked["n"] == n) & (ranked["is_ppo_candidate"])][metric].dropna()
        ax.boxplot([ref, ppo], tick_labels=["combined552", "PPO32"], showfliers=False)
        ax.scatter(np.full(len(ppo), 2), ppo, s=18, alpha=0.75)
        ax.set_title(f"N{n}")
        ax.tick_params(axis="x", rotation=20)
    axes[0].set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=180)
    plt.close(fig)


def make_plots(ranked: pd.DataFrame, alignment: pd.DataFrame, by_n: pd.DataFrame) -> list[str]:
    plot_paths = []
    specs = [
        ("u2_range", "ppo_vs_combined552_u2_distribution_by_N.png", "U2 range"),
        ("peeq_max", "ppo_vs_combined552_peeq_distribution_by_N.png", "PEEQ max"),
        ("surface_t_proxy", "ppo_vs_combined552_surfaceT_distribution_by_N.png", "SurfaceT proxy"),
        ("mises_max", "ppo_vs_combined552_mises_distribution_by_N.png", "Mises max"),
    ]
    for metric, filename, ylabel in specs:
        plot_metric_distribution(ranked, metric, filename, ylabel)
        plot_paths.append(str(PLOTS_DIR / filename))

    ppo = ranked[ranked["is_ppo_candidate"]].copy()
    fig, ax = plt.subplots(figsize=(8, 4))
    for n in SUPPORTED_N:
        sub = ppo[ppo["n"] == n]
        ax.scatter([n] * len(sub), sub["lex_percentile_combined"], label=f"N{n}", alpha=0.8)
    ax.axhline(0.10, color="tab:green", linestyle="--", linewidth=1, label="top 10%")
    ax.axhline(0.25, color="tab:orange", linestyle="--", linewidth=1, label="top 25%")
    ax.set_xlabel("N")
    ax.set_ylabel("Lexicographic percentile in combined552+PPO32")
    ax.set_title("PPO lexicographic rank percentile by N")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "ppo_lexicographic_rank_percentile_by_N.png", dpi=180)
    plt.close(fig)
    plot_paths.append(str(PLOTS_DIR / "ppo_lexicographic_rank_percentile_by_N.png"))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(alignment["predicted_surrogate_reward_lex"], alignment["teacher_lex_reward_rank_normalized"], alpha=0.85)
    ax.set_xlabel("Stage D predicted surrogate reward")
    ax.set_ylabel("Teacher-derived lex reward")
    ax.set_title("Surrogate predicted reward vs teacher-derived reward")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "surrogate_predicted_reward_vs_teacher_reward.png", dpi=180)
    plt.close(fig)
    plot_paths.append(str(PLOTS_DIR / "surrogate_predicted_reward_vs_teacher_reward.png"))

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [f"N{int(row.n)}" for _, row in by_n.iterrows()]
    x = np.arange(len(labels))
    ax.bar(x - 0.18, by_n["ppo_top10pct_ref_lex_count"], width=0.36, label="top10 lex")
    ax.bar(x + 0.18, by_n["ppo_top25pct_ref_lex_count"], width=0.36, label="top25 lex")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("PPO candidate count")
    ax.set_title("PPO top-k lexicographic status summary")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "ppo_candidate_topk_status_summary.png", dpi=180)
    plt.close(fig)
    plot_paths.append(str(PLOTS_DIR / "ppo_candidate_topk_status_summary.png"))
    return plot_paths


def verdict_from_tables(new_records: pd.DataFrame, topk: pd.DataFrame, by_n: pd.DataFrame) -> str:
    if len(new_records) > 0:
        return "PASS_STAGEH_PPO_BATCH32_TEACHER_VALIDATED_WITH_NEW_RECORDS"
    top25_count = int(topk[[c for c in topk.columns if "top25pct_ref" in c]].any(axis=1).sum()) if len(topk) else 0
    top10_count = int(topk[[c for c in topk.columns if "top10pct_ref" in c]].any(axis=1).sum()) if len(topk) else 0
    if top10_count > 0 or top25_count >= 4 or int((by_n["ppo_top25pct_ref_lex_count"] > 0).sum()) >= 2:
        return "PASS_STAGEH_PPO_BATCH32_TEACHER_VALIDATED_AND_COMPETITIVE"
    return "WARNING_STAGEH_PPO_BATCH32_TEACHER_VALIDATED_BUT_NOT_COMPETITIVE"


def fmt_float(value: Any, digits: int = 6) -> str:
    try:
        f = float(value)
    except Exception:
        return str(value)
    if math.isnan(f):
        return "nan"
    return f"{f:.{digits}g}"


def make_report(
    input_verdict: str,
    ranked: pd.DataFrame,
    by_n: pd.DataFrame,
    global_df: pd.DataFrame,
    new_records: pd.DataFrame,
    topk: pd.DataFrame,
    alignment_summary: dict[str, Any],
    recovery: pd.DataFrame,
    verdict: str,
    plot_paths: list[str],
) -> None:
    ppo = ranked[ranked["is_ppo_candidate"]].copy()
    lines = [
        "# PPO Stage H Teacher-Metric Ranking Report",
        "",
        "## 1. Purpose",
        "",
        "Stage H ranks the 32 Abaqus teacher-metric-extracted PPO candidates against the native combined552 Stage 3 reference dataset. This is analysis only.",
        "",
        "## 2. Inputs",
        "",
        f"- Stage G teacher metrics: `{STAGE_G_METRICS}`",
        f"- Stage D PPO selected batch: `{STAGE_D_SELECTED}`",
        f"- Native combined552 reference: `{COMBINED552}`",
        "- N32 is not used in the primary ranking.",
        "",
        "## 3. Stage G Extraction Status",
        "",
        "Stage G teacher metrics were available for 32/32 PPO cases. No failed or incomplete cases were present in the Stage H input.",
        "",
        "## 4. Input Integrity Audit",
        "",
        f"Input verdict: `{input_verdict}`",
        "",
        f"- Audit CSV: `{AUDIT_CSV}`",
        f"- Audit summary JSON: `{AUDIT_JSON}`",
        "",
        "## 5. Analysis Dataset",
        "",
        f"Analysis dataset: `{ANALYSIS_DATASET_CSV}`",
        "",
        "The dataset contains 584 rows: 552 native combined reference rows plus 32 PPO batch rows. It is an analysis artifact, not a frozen replacement for Run78 evidence.",
        "",
        "## 6. PPO Batch32 Ranking Against Native Combined552",
        "",
        f"- Full PPO ranking table: `{PPO_RANKING_CSV}`",
        f"- Summary by N: `{PPO_SUMMARY_BY_N_CSV}`",
        f"- Global summary: `{PPO_GLOBAL_SUMMARY_CSV}`",
        "",
        "Best PPO candidates by lexicographic U2 -> PEEQ -> SurfaceT:",
        "",
        "| N | Strategy | Combined lex rank | Ref lex position | U2 range | PEEQ max | SurfaceT proxy |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in by_n.iterrows():
        best = ppo[(ppo["n"] == int(row["n"]))].sort_values("lex_rank_combined").iloc[0]
        lines.append(
            f"| {int(row['n'])} | `{best['strategy_name']}` | {int(best['lex_rank_combined'])} | {int(best['ref_rank_lex'])} | {fmt_float(best['u2_range'])} | {fmt_float(best['peeq_max'])} | {fmt_float(best['surface_t_proxy'])} |"
        )
    lines.extend([
        "",
        "## 7. Metric-Wise Comparison By N",
        "",
        "| N | Best PPO U2 / Ref Best | Best PPO PEEQ / Ref Best | Best PPO SurfaceT / Ref Best | Best PPO Mises / Ref Best | Top25 Lex Count |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for _, row in by_n.iterrows():
        lines.append(
            f"| {int(row['n'])} | {fmt_float(row['ppo_best_over_reference_best_ratio_u2_range'], 4)} | {fmt_float(row['ppo_best_over_reference_best_ratio_peeq_max'], 4)} | {fmt_float(row['ppo_best_over_reference_best_ratio_surface_t_proxy'], 4)} | {fmt_float(row['ppo_best_over_reference_best_ratio_mises_max'], 4)} | {int(row['ppo_top25pct_ref_lex_count'])} |"
        )
    lines.extend([
        "",
        "Ratios below 1.0 indicate that the best PPO candidate beats the prior combined552 best for that metric. Ratios at or above 1.0 do not support a new-record claim for that metric.",
        "",
        "## 8. Lexicographic U2->PEEQ->SurfaceT Comparison",
        "",
        "Lexicographic ranking is computed within each N using U2 range first, then PEEQ max, then SurfaceT proxy, all smaller-is-better.",
        "",
        f"- PPO candidates beating the prior combined552 lexicographic best: {int(ppo['ppo_beats_ref_lex_best'].sum())}",
        f"- PPO candidates in reference top10pct lex: {int(ppo['ppo_top10pct_ref_lex'].sum())}",
        f"- PPO candidates in reference top25pct lex: {int(ppo['ppo_top25pct_ref_lex'].sum())}",
        "",
        "## 9. New-Record Audit",
        "",
        f"New-record table: `{PPO_NEW_RECORDS_CSV}`",
        "",
        f"New-record candidate count: {len(new_records)}",
        "",
        "A new-record row is included only when a PPO candidate beats the prior combined552 best in at least one primary metric, Mises diagnostic metric, or lexicographic ranking.",
        "",
        "## 10. Top-K Audit",
        "",
        f"Top-k table: `{PPO_TOPK_CSV}`",
        "",
        f"Top-k candidate count: {len(topk)}",
        "",
        "A top-k row is included when a PPO candidate falls in the reference top10pct or top25pct by a primary metric or lexicographic ranking.",
        "",
        "## 11. Surrogate-Vs-Teacher Alignment",
        "",
        f"- Alignment table: `{ALIGNMENT_CSV}`",
        f"- Alignment summary JSON: `{ALIGNMENT_JSON}`",
        f"- Overall Spearman: {fmt_float(alignment_summary.get('overall_spearman_predicted_vs_teacher_reward'), 4)}",
        f"- Overall Pearson: {fmt_float(alignment_summary.get('overall_pearson_predicted_vs_teacher_reward'), 4)}",
        f"- False-positive count: {alignment_summary.get('false_positive_count')}",
        f"- True-positive count: {alignment_summary.get('true_positive_count')}",
        "",
        "## 12. Recovery-Anchor Duplicate Audit",
        "",
        f"Recovery-anchor audit: `{RECOVERY_ANCHOR_CSV}`",
        "",
    ])
    if not recovery.empty:
        first = recovery.iloc[0]
        lines.append(f"- `PPOV01_N12_B02_surrogate_top`: {first.get('interpretation', first.get('status', ''))}")
        if "all_metrics_match_within_tolerance" in first:
            lines.append(f"- Metrics match source row within tolerance: {bool(first.get('all_metrics_match_within_tolerance'))}")
    lines.extend([
        "",
        "## 13. Claim Boundary",
        "",
        "Safe claims after Stage H are limited to teacher validation and rankings demonstrated by these tables. New-record or top-k claims are allowed only where the tables prove them.",
        "",
        "Unsafe claims remain: PPO globally optimised scan order, PPO solved arbitrary-N optimisation, online Abaqus PPO was performed, PPO is experimentally validated, or PPO is first in the world.",
        "",
        "## 14. Manuscript Implication",
        "",
    ])
    if len(new_records) > 0:
        lines.append("The PPO addendum can report that the PPO-generated batch was independently Abaqus teacher-evaluated and produced new record(s) in the specific metric/N combinations listed in the new-record audit. It must not generalize beyond those proven cases.")
    elif len(topk) > 0:
        lines.append("The PPO addendum can report that the PPO-generated batch was independently Abaqus teacher-evaluated and achieved bounded top-k competitiveness in the specific rankings listed in the top-k audit. It must not claim broad superiority.")
    else:
        lines.append("The PPO addendum can report that PPO-generated candidates were teacher-validated, but the evidence does not support competitiveness or superiority claims against combined552.")
    lines.extend([
        "",
        "## 15. Plots",
        "",
    ])
    for path in plot_paths:
        lines.append(f"- `{path}`")
    lines.extend([
        "",
        "## 16. Verdict",
        "",
        f"`{verdict}`",
    ])
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_claim_boundary() -> None:
    CLAIM_BOUNDARY_PATH.write_text(
        "\n".join(
            [
                "# PPO Stage H Claim Boundary",
                "",
                "## Safe After Stage H",
                "",
                "- 32 PPO-generated scan-order candidates were independently evaluated by Abaqus teacher simulations.",
                "- Teacher metrics were extracted for 32/32 PPO candidates.",
                "- PPO performance can be reported according to the Stage H ranking evidence:",
                "  - new records only where `ppo_batch32_new_record_candidates.csv` proves them;",
                "  - top-k competitiveness only where `ppo_batch32_topk_candidates.csv` proves it;",
                "  - teacher validation only where no competitiveness is proven.",
                "",
                "## Unsafe Unless Separately Proven",
                "",
                "- PPO outperforms all known strategies.",
                "- PPO solves arbitrary-N scan-order optimisation.",
                "- PPO is globally optimal.",
                "- PPO is experimentally validated.",
                "- PPO is online Abaqus RL.",
                "- PPO is first in the world.",
                "",
                "Any `first` claim requires a separate literature-priority audit before manuscript submission.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def make_manifest(verdict: str, input_verdict: str, plot_paths: list[str]) -> None:
    manifest = {
        "branch": git_branch(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stageG_metrics_input": str(STAGE_G_METRICS),
        "combined552_reference_input": str(COMBINED552),
        "stageD_selected_batch_input": str(STAGE_D_SELECTED),
        "input_integrity_verdict": input_verdict,
        "analysis_dataset_path": str(ANALYSIS_DATASET_CSV),
        "ranking_table_paths": {
            "ppo_full_ranking": str(PPO_RANKING_CSV),
            "ppo_summary_by_N": str(PPO_SUMMARY_BY_N_CSV),
            "ppo_global_summary": str(PPO_GLOBAL_SUMMARY_CSV),
            "ppo_new_records": str(PPO_NEW_RECORDS_CSV),
            "ppo_topk": str(PPO_TOPK_CSV),
            "surrogate_alignment": str(ALIGNMENT_CSV),
            "recovery_anchor_duplicate_audit": str(RECOVERY_ANCHOR_CSV),
        },
        "summary_paths": {
            "input_audit_summary": str(AUDIT_JSON),
            "surrogate_alignment_summary": str(ALIGNMENT_JSON),
        },
        "plot_paths": plot_paths,
        "report_path": str(REPORT_PATH),
        "claim_boundary_path": str(CLAIM_BOUNDARY_PATH),
        "no_Abaqus": True,
        "no_ODB_opening": True,
        "no_solver": True,
        "no_CAE_INP_JNL": True,
        "no_candidate_generation": True,
        "no_training": True,
        "no_commit_or_push": True,
        "verdict": verdict,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    ensure_dirs()
    stage_g, stage_d, combined = load_inputs()
    input_verdict = input_audit(stage_g, stage_d, combined)
    if input_verdict.startswith("FAIL"):
        print(json.dumps({"verdict": input_verdict, "audit_json": str(AUDIT_JSON)}, indent=2))
        return 1

    combined_core = canonical_combined(combined)
    ppo_core = canonical_ppo(stage_g, stage_d)
    analysis = pd.concat([combined_core, ppo_core], ignore_index=True, sort=False)
    ranked = add_rankings(analysis)
    ranked.to_csv(ANALYSIS_DATASET_CSV, index=False)

    ppo_ranked = ranked[ranked["is_ppo_candidate"]].copy().sort_values(["n", "lex_rank_combined", "strategy_name"])
    ppo_ranked.to_csv(PPO_RANKING_CSV, index=False)
    by_n = summarize_by_n(ranked)
    by_n.to_csv(PPO_SUMMARY_BY_N_CSV, index=False)
    global_df = global_summary(ranked, by_n)
    global_df.to_csv(PPO_GLOBAL_SUMMARY_CSV, index=False)
    new_records = new_record_rows(ppo_ranked)
    new_records.to_csv(PPO_NEW_RECORDS_CSV, index=False)
    topk = topk_rows(ppo_ranked)
    topk.to_csv(PPO_TOPK_CSV, index=False)
    alignment_df, alignment_summary = surrogate_alignment(ppo_ranked)
    alignment_df.to_csv(ALIGNMENT_CSV, index=False)
    recovery = recovery_anchor_audit(ranked)
    recovery.to_csv(RECOVERY_ANCHOR_CSV, index=False)

    plot_paths = make_plots(ranked, alignment_df, by_n)
    verdict = verdict_from_tables(new_records, topk, by_n)
    make_claim_boundary()
    make_report(input_verdict, ranked, by_n, global_df, new_records, topk, alignment_summary, recovery, verdict, plot_paths)
    make_manifest(verdict, input_verdict, plot_paths)

    summary = {
        "verdict": verdict,
        "input_integrity_verdict": input_verdict,
        "analysis_dataset": str(ANALYSIS_DATASET_CSV),
        "ppo_ranking_table": str(PPO_RANKING_CSV),
        "summary_by_N": str(PPO_SUMMARY_BY_N_CSV),
        "new_record_count": int(len(new_records)),
        "topk_count": int(len(topk)),
        "alignment_spearman": alignment_summary.get("overall_spearman_predicted_vs_teacher_reward"),
        "alignment_pearson": alignment_summary.get("overall_pearson_predicted_vs_teacher_reward"),
        "report": str(REPORT_PATH),
        "manifest": str(MANIFEST_PATH),
        "no_Abaqus": True,
        "no_ODB_opening": True,
        "no_solver": True,
        "no_training": True,
        "no_candidate_generation": True,
    }
    print(json.dumps(summary, indent=2))
    return 0 if not verdict.startswith("FAIL") else 1


if __name__ == "__main__":
    raise SystemExit(main())
