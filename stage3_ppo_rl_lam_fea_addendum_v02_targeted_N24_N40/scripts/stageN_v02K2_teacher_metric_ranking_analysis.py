"""Stage N: PPO v02K2 N24/N40 teacher-metric ranking and claim audit.

This script is analysis-only. It reads existing teacher-metric CSV evidence,
computes ranking/comparison/claim-support tables, and writes Stage N reports.
It does not run Abaqus, open ODB files, run solver/datacheck, generate
CAE/INP/JNL, train models, or generate candidates.
"""

from __future__ import annotations

import csv
import json
import math
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
BRANCH_FALLBACK = "stage3-variable-n-graph-pointer-init-v01"
NAMESPACE = "stage3_ppo_rl_lam_fea_addendum_v02_targeted_N24_N40"
V01_NAMESPACE = "stage3_ppo_rl_lam_fea_addendum_v01"
SUPPORTED_N = [24, 40]
EXPECTED_V02_COUNTS = {24: 16, 40: 16}
EXPECTED_V01_COUNTS = {24: 8, 40: 8}
REF_COUNTS = {24: 190, 40: 206}
METRICS = ["u2_range", "peeq_max", "surface_t_proxy", "mises_max"]
PRIMARY_METRICS = ["u2_range", "peeq_max", "surface_t_proxy"]
BOOTSTRAP_TRIALS = 10000
RNG_SEED = 20260628

STAGE_M_METRICS = PROJECT_ROOT / "outputs" / NAMESPACE / "stageM_ODB_teacher_metric_extraction" / "stageM_v02K2_teacher_metrics.csv"
STAGE_M_SUMMARY = PROJECT_ROOT / "outputs" / NAMESPACE / "stageM_ODB_teacher_metric_extraction" / "stageM_v02K2_extraction_summary.json"
STAGE_M_SOLVER_AUDIT = PROJECT_ROOT / "outputs" / NAMESPACE / "stageM_ODB_teacher_metric_extraction" / "stageM_v02K2_solver_completion_audit.csv"
K2_SELECTED = PROJECT_ROOT / "outputs" / NAMESPACE / "stageK2_n40_completion" / "selected_batch32_K2" / "v02K2_ppo_targeted_N24_N40_candidate_batch32.csv"
COMBINED552 = PROJECT_ROOT / "outputs" / "stage3_run_78_final_evidence_freeze_package" / "FROZEN_stage3_native_combined552_teacher_dataset.csv"
V01_METRICS = PROJECT_ROOT / "outputs" / V01_NAMESPACE / "stageI_final_ppo_evidence_freeze" / "frozen_tables" / "FROZEN_PPO_batch32_teacher_metrics.csv"
V01_RANKING = PROJECT_ROOT / "outputs" / V01_NAMESPACE / "stageI_final_ppo_evidence_freeze" / "frozen_tables" / "FROZEN_PPO_batch32_teacher_metric_ranking_full.csv"

OUT_ROOT = PROJECT_ROOT / "outputs" / NAMESPACE / "stageN_teacher_metric_ranking"
CHECKS_DIR = OUT_ROOT / "checks"
TABLES_DIR = OUT_ROOT / "tables"
PLOTS_DIR = OUT_ROOT / "plots"
REPORTS_DIR = OUT_ROOT / "reports"
DOCS_DIR = PROJECT_ROOT / "docs" / NAMESPACE

AUDIT_CSV = CHECKS_DIR / "stageN_input_integrity_audit.csv"
AUDIT_JSON = CHECKS_DIR / "stageN_input_integrity_audit_summary.json"
ANALYSIS_V02_CSV = TABLES_DIR / "combined552_N24N40_plus_v02K2_analysis_dataset.csv"
ANALYSIS_ALL_CSV = TABLES_DIR / "combined552_N24N40_plus_v01_plus_v02K2_analysis_dataset.csv"
FULL_RANKING_CSV = TABLES_DIR / "v02K2_teacher_metric_ranking_full.csv"
SUMMARY_BY_N_CSV = TABLES_DIR / "v02K2_summary_by_N.csv"
GLOBAL_SUMMARY_CSV = TABLES_DIR / "v02K2_global_summary.csv"
BEST_BY_N_CSV = TABLES_DIR / "v02K2_best_candidates_by_N.csv"
NEW_RECORDS_CSV = TABLES_DIR / "v02K2_new_record_candidates.csv"
TOPK_CSV = TABLES_DIR / "v02K2_topk_competitive_candidates.csv"
TOPK_SUMMARY_CSV = TABLES_DIR / "v02K2_topk_summary_by_N.csv"
V02_VS_V01_CSV = TABLES_DIR / "v02K2_vs_v01_targeted_comparison_by_N.csv"
V02_VS_V01_BOOTSTRAP_CSV = TABLES_DIR / "v02K2_vs_v01_equal_budget_bootstrap.csv"
BOOTSTRAP_BY_N_CSV = TABLES_DIR / "v02K2_vs_bootstrap_random_reference_by_N.csv"
BOOTSTRAP_GLOBAL_CSV = TABLES_DIR / "v02K2_vs_bootstrap_random_reference_global.csv"
BASELINE_COMPARE_CSV = TABLES_DIR / "v02K2_vs_identified_baseline_families.csv"
BASELINE_INVENTORY_CSV = TABLES_DIR / "v02K2_identified_baseline_family_inventory.csv"
ALIGNMENT_CSV = TABLES_DIR / "v02K2_surrogate_vs_teacher_alignment.csv"
ALIGNMENT_JSON = TABLES_DIR / "v02K2_surrogate_vs_teacher_alignment_summary.json"
REPORT_PATH = DOCS_DIR / "PPO_V02K2_STAGEN_TEACHER_METRIC_RANKING_REPORT.md"
CLAIM_BOUNDARY_PATH = DOCS_DIR / "PPO_V02K2_STAGEN_CLAIM_BOUNDARY.md"
MANIFEST_PATH = OUT_ROOT / "stageN_v02K2_teacher_metric_ranking_manifest.json"


def ensure_dirs() -> None:
    for directory in [OUT_ROOT, CHECKS_DIR, TABLES_DIR, PLOTS_DIR, REPORTS_DIR, DOCS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def git_branch() -> str:
    try:
        result = subprocess.run(
            ["git", "-c", "safe.directory=E:/Projects/RL-LAM-ScanOpt", "-C", str(PROJECT_ROOT), "branch", "--show-current"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        return result.stdout.strip() or BRANCH_FALLBACK
    except Exception:
        return BRANCH_FALLBACK


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def md_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    def cell(value: Any) -> str:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return ""
        return str(value).replace("|", "\\|").replace("\n", "<br>")

    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(cell(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def metric_col(df: pd.DataFrame, metric: str) -> str | None:
    if metric in df.columns:
        return metric
    if metric == "surface_t_proxy":
        for col in ["surface_t_proxy_max_tensile_pa", "surface_t_proxy_pa", "surface_t_proxy_mpa"]:
            if col in df.columns:
                return col
    return None


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def canonical_reference(ref: pd.DataFrame) -> pd.DataFrame:
    ref = ref[ref["n"].astype(int).isin(SUPPORTED_N)].copy()
    out = pd.DataFrame()
    out["n"] = ref["n"].astype(int)
    out["strategy_name"] = ref.get("strategy_name", ref.get("handoff_strategy_name", pd.Series([""] * len(ref)))).astype(str)
    out["order_hash"] = ref.get("order_hash", pd.Series([""] * len(ref))).astype(str)
    out["order_compact"] = ref.get("order_compact", pd.Series([""] * len(ref))).astype(str)
    out["dataset_source"] = "stage3_native_combined552"
    out["candidate_source"] = ref.get("candidate_source", pd.Series([""] * len(ref))).astype(str)
    out["ppo_checkpoint"] = ""
    out["ppo_seed"] = ""
    out["predicted_reward"] = np.nan
    out["conservative_reward"] = np.nan
    out["teacher_metrics_extracted"] = True
    out["teacher_validation_status"] = ref.get("teacher_validation_status", pd.Series(["PASS_TEACHER_LABELLED_REFERENCE"] * len(ref))).astype(str)
    out["stageM_metric_source_file"] = ""
    for metric in METRICS:
        col = metric_col(ref, metric)
        out[metric] = pd.to_numeric(ref[col], errors="coerce") if col else np.nan
    return out


def canonical_v02(stage_m: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    m = stage_m.copy()
    m["strategy_name"] = m["handoff_strategy_name"].astype(str)
    s = selected.copy()
    merged = m.merge(
        s[["strategy_name", "ppo_checkpoint", "ppo_seed", "candidate_source", "teacher_validated", "abaqus_validated"]],
        on="strategy_name",
        how="left",
        suffixes=("", "_selected"),
    )
    out = pd.DataFrame()
    out["n"] = merged["n"].astype(int)
    out["strategy_name"] = merged["strategy_name"].astype(str)
    out["order_hash"] = merged.get("order_hash", pd.Series([""] * len(merged))).astype(str)
    out["order_compact"] = merged.get("order_compact", pd.Series([""] * len(merged))).astype(str)
    out["dataset_source"] = "ppo_v02K2_targeted_batch32"
    out["candidate_source"] = merged.get("candidate_source_selected", merged.get("candidate_source", pd.Series([""] * len(merged)))).astype(str)
    out["ppo_checkpoint"] = merged.get("ppo_checkpoint", pd.Series([""] * len(merged))).astype(str)
    out["ppo_seed"] = merged.get("ppo_seed", pd.Series([""] * len(merged))).astype(str)
    out["ppo_generation_mode"] = merged.get("ppo_generation_mode", pd.Series([""] * len(merged))).astype(str)
    out["selection_tag"] = merged.get("selection_tag", pd.Series([""] * len(merged))).astype(str)
    out["selected_by"] = merged.get("selected_by", pd.Series([""] * len(merged))).astype(str)
    out["predicted_reward"] = pd.to_numeric(merged.get("predicted_reward", pd.Series([np.nan] * len(merged))), errors="coerce")
    out["conservative_reward"] = pd.to_numeric(merged.get("conservative_reward", pd.Series([np.nan] * len(merged))), errors="coerce")
    out["teacher_metrics_extracted"] = True
    out["teacher_validation_status"] = merged.get("teacher_validation_status", pd.Series(["PASS_TEACHER_FIELDS_EXTRACTED"] * len(merged))).astype(str)
    out["stageM_metric_source_file"] = str(STAGE_M_METRICS)
    out["final_step_name"] = merged.get("final_step_name", pd.Series([""] * len(merged))).astype(str)
    out["final_frame_time"] = pd.to_numeric(merged.get("final_frame_time", pd.Series([np.nan] * len(merged))), errors="coerce")
    out["extracted_field_names"] = merged.get("extracted_field_names", pd.Series([""] * len(merged))).astype(str)
    out["completion_status"] = merged.get("completion_status", pd.Series([""] * len(merged))).astype(str)
    out["odb_extraction_status"] = merged.get("odb_extraction_status", pd.Series([""] * len(merged))).astype(str)
    for metric in METRICS:
        col = metric_col(merged, metric)
        out[metric] = pd.to_numeric(merged[col], errors="coerce") if col else np.nan
    return out


def canonical_v01(v01_metrics: pd.DataFrame, v01_ranking: pd.DataFrame) -> pd.DataFrame:
    m = v01_metrics[v01_metrics["n"].astype(int).isin(SUPPORTED_N)].copy()
    m["strategy_name"] = m["handoff_strategy_name"].astype(str)
    rank_cols = ["strategy_name", "predicted_surrogate_reward_lex", "ppo_checkpoint", "ppo_generation_mode", "ppo_selection_tag", "candidate_source"]
    available = [c for c in rank_cols if c in v01_ranking.columns]
    merged = m.merge(v01_ranking[available], on="strategy_name", how="left") if available else m
    out = pd.DataFrame()
    out["n"] = merged["n"].astype(int)
    out["strategy_name"] = merged["strategy_name"].astype(str)
    out["order_hash"] = merged.get("order_hash", pd.Series([""] * len(merged))).astype(str)
    out["order_compact"] = merged.get("order_compact", pd.Series([""] * len(merged))).astype(str)
    out["dataset_source"] = "ppo_v01_batch32"
    out["candidate_source"] = merged.get("candidate_source", pd.Series(["PPO_checkpoint_inference"] * len(merged))).astype(str)
    out["ppo_checkpoint"] = merged.get("ppo_checkpoint", pd.Series([""] * len(merged))).astype(str)
    out["ppo_seed"] = ""
    out["ppo_generation_mode"] = merged.get("ppo_generation_mode", pd.Series([""] * len(merged))).astype(str)
    out["predicted_reward"] = pd.to_numeric(merged.get("predicted_surrogate_reward_lex", pd.Series([np.nan] * len(merged))), errors="coerce")
    out["conservative_reward"] = np.nan
    out["teacher_metrics_extracted"] = True
    out["teacher_validation_status"] = merged.get("teacher_validation_status", pd.Series(["PASS_TEACHER_FIELDS_EXTRACTED"] * len(merged))).astype(str)
    out["stageM_metric_source_file"] = ""
    for metric in METRICS:
        col = metric_col(merged, metric)
        out[metric] = pd.to_numeric(merged[col], errors="coerce") if col else np.nan
    return out


def input_audit(stage_m: pd.DataFrame, selected: pd.DataFrame, ref: pd.DataFrame, v01_metrics: pd.DataFrame, v01_ranking: pd.DataFrame) -> str:
    rows: list[dict[str, Any]] = []

    def add(check: str, passed: bool, severity: str, detail: Any) -> None:
        rows.append({"check": check, "passed": bool(passed), "severity": severity, "detail": str(detail)})

    for label, path in [
        ("stageM_metrics", STAGE_M_METRICS),
        ("stageM_summary", STAGE_M_SUMMARY),
        ("stageM_solver_audit", STAGE_M_SOLVER_AUDIT),
        ("K2_selected_batch", K2_SELECTED),
        ("combined552", COMBINED552),
        ("v01_metrics", V01_METRICS),
        ("v01_ranking", V01_RANKING),
    ]:
        add(f"{label}_exists", path.exists(), "FAIL", path)

    add("stageM_row_count_32", len(stage_m) == 32, "FAIL", len(stage_m))
    add("stageM_counts_N24_N40", stage_m["n"].astype(int).value_counts().sort_index().to_dict() == EXPECTED_V02_COUNTS, "FAIL", stage_m["n"].astype(int).value_counts().sort_index().to_dict())
    add("stageM_no_N12_N16_N32", set(stage_m["n"].astype(int)) == set(SUPPORTED_N), "FAIL", sorted(stage_m["n"].astype(int).unique()))
    add("stageM_extraction_success_32", stage_m["odb_extraction_status"].astype(str).str.contains("PASS", case=False, na=False).all(), "FAIL", stage_m.get("odb_extraction_status", pd.Series(dtype=str)).value_counts().to_dict())
    add("stageM_teacher_status_success_32", stage_m["teacher_validation_status"].astype(str).str.contains("PASS", case=False, na=False).all(), "FAIL", stage_m.get("teacher_validation_status", pd.Series(dtype=str)).value_counts().to_dict())
    add("stageM_final_metadata_available", "final_step_name" in stage_m.columns and "final_frame_time" in stage_m.columns, "WARNING", "final_step_name/final_frame_time")
    fields_text = stage_m.get("extracted_field_names", pd.Series([""] * len(stage_m))).astype(str)
    fields_ok = fields_text.str.contains("U").all() and fields_text.str.contains("PEEQ").all() and fields_text.str.contains("S").all() and fields_text.str.contains("NT11|NT", regex=True).all()
    add("stageM_required_fields_visible", fields_ok, "WARNING", "U/PEEQ/S/NT or NT11")
    for metric in METRICS:
        add(f"stageM_metric_{metric}_mapped", metric_col(stage_m, metric) is not None, "FAIL", metric_col(stage_m, metric))

    add("K2_row_count_32", len(selected) == 32, "FAIL", len(selected))
    add("K2_counts_N24_N40", selected["n"].astype(int).value_counts().sort_index().to_dict() == EXPECTED_V02_COUNTS, "FAIL", selected["n"].astype(int).value_counts().sort_index().to_dict())
    add("K2_strategy_names_match_stageM", set(selected["strategy_name"].astype(str)) == set(stage_m["handoff_strategy_name"].astype(str)), "FAIL", "strategy set equality")
    add("K2_candidate_source_valid", selected.apply(lambda r: (int(r["n"]) == 24 and str(r.get("candidate_source", "")) == "PPO_v02_checkpoint_inference") or (int(r["n"]) == 40 and str(r.get("candidate_source", "")) == "PPO_v02K2_checkpoint_inference"), axis=1).all(), "FAIL", selected.get("candidate_source", pd.Series(dtype=str)).value_counts().to_dict())
    add("K2_predicted_or_conservative_exists", "predicted_reward" in selected.columns or "conservative_reward" in selected.columns, "FAIL", "predicted_reward/conservative_reward")

    ref_n = ref[ref["n"].astype(int).isin(SUPPORTED_N)]
    add("combined552_row_count_552", len(ref) == 552, "FAIL", len(ref))
    add("combined552_N24_N40_counts", ref_n["n"].astype(int).value_counts().sort_index().to_dict() == REF_COUNTS, "FAIL", ref_n["n"].astype(int).value_counts().sort_index().to_dict())
    for metric in METRICS:
        add(f"combined552_metric_{metric}_mapped", metric_col(ref, metric) is not None, "FAIL", metric_col(ref, metric))

    v01_n = v01_metrics[v01_metrics["n"].astype(int).isin(SUPPORTED_N)]
    add("v01_N24_N40_count_16", len(v01_n) == 16, "FAIL", len(v01_n))
    add("v01_N24_N40_counts", v01_n["n"].astype(int).value_counts().sort_index().to_dict() == EXPECTED_V01_COUNTS, "FAIL", v01_n["n"].astype(int).value_counts().sort_index().to_dict())
    add("v01_ranking_available", len(v01_ranking) >= 16, "FAIL", len(v01_ranking))

    write_csv(AUDIT_CSV, rows)
    fail_count = sum(1 for r in rows if r["severity"] == "FAIL" and not r["passed"])
    warn_count = sum(1 for r in rows if r["severity"] == "WARNING" and not r["passed"])
    verdict = "FAIL_STAGEN_V02K2_INPUTS_NOT_READY" if fail_count else ("WARNING_STAGEN_V02K2_INPUTS_REVIEW" if warn_count else "PASS_STAGEN_V02K2_INPUTS_READY")
    summary = {
        "verdict": verdict,
        "fail_count": fail_count,
        "warning_count": warn_count,
        "stageM_rows": int(len(stage_m)),
        "stageM_counts_by_N": {str(k): int(v) for k, v in stage_m["n"].astype(int).value_counts().sort_index().to_dict().items()},
        "combined552_N24_N40_counts": {str(k): int(v) for k, v in ref_n["n"].astype(int).value_counts().sort_index().to_dict().items()},
        "v01_N24_N40_counts": {str(k): int(v) for k, v in v01_n["n"].astype(int).value_counts().sort_index().to_dict().items()},
        "no_Abaqus": True,
        "no_ODB_opening": True,
        "no_ODB_extraction": True,
        "no_solver": True,
    }
    AUDIT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return verdict


def ref_rank(value: float, ref_values: np.ndarray) -> int:
    return int(np.sum(ref_values < value) + 1)


def ref_percentile(value: float, ref_values: np.ndarray) -> float:
    return ref_rank(value, ref_values) / float(len(ref_values) + 1)


def top_threshold(count: int, fraction: float) -> int:
    return int(math.ceil(count * fraction))


def annotate_v02_ranks(ref_df: pd.DataFrame, v02_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for n in SUPPORTED_N:
        ref_n = ref_df[ref_df["n"] == n].copy()
        v02_n = v02_df[v02_df["n"] == n].copy()
        combined = pd.concat([ref_n, v02_n], ignore_index=True)
        for metric in METRICS:
            combined[f"combined_rank_{metric}"] = combined[metric].rank(method="min", ascending=True)
            combined[f"combined_percentile_{metric}"] = combined[f"combined_rank_{metric}"] / len(combined)
            mn, mx = combined[metric].min(), combined[metric].max()
            combined[f"combined_cost_norm_{metric}"] = 0.0 if mx == mn else (combined[metric] - mn) / (mx - mn)
        combined = assign_lex_ranks(combined, prefix="combined")
        combined = assign_lex_ranks(combined, keys=METRICS, prefix="combined_diag")
        combined_v02 = combined[combined["dataset_source"] == "ppo_v02K2_targeted_batch32"].copy()

        for metric in METRICS:
            ref_values = ref_n[metric].dropna().to_numpy()
            threshold10 = top_threshold(len(ref_values), 0.10)
            threshold25 = top_threshold(len(ref_values), 0.25)
            combined_v02[f"ref_rank_{metric}"] = combined_v02[metric].apply(lambda x: ref_rank(float(x), ref_values))
            combined_v02[f"ref_percentile_{metric}"] = combined_v02[metric].apply(lambda x: ref_percentile(float(x), ref_values))
            combined_v02[f"ref_best_{metric}"] = float(np.min(ref_values))
            combined_v02[f"ref_median_{metric}"] = float(np.median(ref_values))
            combined_v02[f"ref_q25_{metric}"] = float(np.quantile(ref_values, 0.25))
            combined_v02[f"ref_q75_{metric}"] = float(np.quantile(ref_values, 0.75))
            combined_v02[f"candidate_beats_ref_best_{metric}"] = combined_v02[metric] < np.min(ref_values)
            combined_v02[f"candidate_better_than_ref_median_{metric}"] = combined_v02[metric] < np.median(ref_values)
            combined_v02[f"candidate_top10pct_ref_{metric}"] = combined_v02[f"ref_rank_{metric}"] <= threshold10
            combined_v02[f"candidate_top25pct_ref_{metric}"] = combined_v02[f"ref_rank_{metric}"] <= threshold25
            combined_v02[f"v02K2_rank_{metric}_within_N"] = combined_v02[metric].rank(method="min", ascending=True)
            combined_v02[f"v02K2_best_{metric}_within_N"] = combined_v02[f"v02K2_rank_{metric}_within_N"] == 1

        ref_lex = sorted([tuple(x) for x in ref_n[PRIMARY_METRICS].to_numpy()])
        ref_diag = sorted([tuple(x) for x in ref_n[METRICS].to_numpy()])
        lex10 = top_threshold(len(ref_lex), 0.10)
        lex25 = top_threshold(len(ref_lex), 0.25)
        for idx, row in combined_v02.iterrows():
            key = tuple(row[m] for m in PRIMARY_METRICS)
            diag_key = tuple(row[m] for m in METRICS)
            combined_v02.loc[idx, "ref_lex_rank"] = 1 + sum(k < key for k in ref_lex)
            combined_v02.loc[idx, "ref_lex_percentile"] = combined_v02.loc[idx, "ref_lex_rank"] / (len(ref_lex) + 1)
            combined_v02.loc[idx, "candidate_beats_ref_lex_best"] = key < ref_lex[0]
            combined_v02.loc[idx, "candidate_top10pct_ref_lex"] = combined_v02.loc[idx, "ref_lex_rank"] <= lex10
            combined_v02.loc[idx, "candidate_top25pct_ref_lex"] = combined_v02.loc[idx, "ref_lex_rank"] <= lex25
            combined_v02.loc[idx, "ref_lex_diag_rank"] = 1 + sum(k < diag_key for k in ref_diag)
            combined_v02.loc[idx, "ref_lex_diag_percentile"] = combined_v02.loc[idx, "ref_lex_diag_rank"] / (len(ref_diag) + 1)
        combined_v02["v02K2_lex_rank_within_N"] = range(1, len(combined_v02.sort_values(PRIMARY_METRICS)) + 1)
        rows.extend(combined_v02.to_dict("records"))
    out = pd.DataFrame(rows)
    for n in SUPPORTED_N:
        mask = out["n"] == n
        order = out[mask].sort_values(PRIMARY_METRICS).index
        out.loc[order, "v02K2_lex_rank_within_N"] = np.arange(1, len(order) + 1)
        out.loc[order, "v02K2_best_lex_within_N"] = np.arange(1, len(order) + 1) == 1
    out["teacher_lex_reward_rank_normalized"] = 1.0 - (out["ref_lex_rank"] - 1) / out["n"].map(REF_COUNTS)
    return out.sort_values(["n", "ref_lex_rank", "strategy_name"]).reset_index(drop=True)


def assign_lex_ranks(df: pd.DataFrame, keys: list[str] | None = None, prefix: str = "combined") -> pd.DataFrame:
    keys = keys or PRIMARY_METRICS
    out = df.copy()
    for n in sorted(out["n"].unique()):
        idx = out[out["n"] == n].sort_values(keys).index
        ranks = np.arange(1, len(idx) + 1)
        if prefix == "combined":
            out.loc[idx, "combined_lex_rank"] = ranks
            out.loc[idx, "combined_lex_percentile"] = ranks / len(idx)
        else:
            out.loc[idx, f"{prefix}_lex_rank"] = ranks
            out.loc[idx, f"{prefix}_lex_percentile"] = ranks / len(idx)
    return out


def row_has_primary_topk(row: pd.Series) -> bool:
    flags = []
    for metric in PRIMARY_METRICS:
        flags.extend([row.get(f"candidate_top10pct_ref_{metric}", False), row.get(f"candidate_top25pct_ref_{metric}", False)])
    flags.extend([row.get("candidate_top10pct_ref_lex", False), row.get("candidate_top25pct_ref_lex", False)])
    return any(to_bool(x) for x in flags)


def write_analysis_datasets(ref_df: pd.DataFrame, v01_df: pd.DataFrame, v02_df: pd.DataFrame) -> None:
    v02_analysis = pd.concat([ref_df, v02_df], ignore_index=True)
    v02_analysis["is_ppo_v01"] = False
    v02_analysis["is_ppo_v02K2"] = v02_analysis["dataset_source"] == "ppo_v02K2_targeted_batch32"
    v02_analysis.to_csv(ANALYSIS_V02_CSV, index=False)

    all_analysis = pd.concat([ref_df, v01_df, v02_df], ignore_index=True)
    all_analysis["is_ppo_v01"] = all_analysis["dataset_source"] == "ppo_v01_batch32"
    all_analysis["is_ppo_v02K2"] = all_analysis["dataset_source"] == "ppo_v02K2_targeted_batch32"
    all_analysis.to_csv(ANALYSIS_ALL_CSV, index=False)


def new_record_table(ranking: pd.DataFrame, ref_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    rows: list[dict[str, Any]] = []
    for n in SUPPORTED_N:
        ref_n = ref_df[ref_df["n"] == n]
        cand_n = ranking[ranking["n"] == n]
        for metric in METRICS:
            ref_best = float(ref_n[metric].min())
            better = cand_n[cand_n[metric] < ref_best]
            for _, row in better.iterrows():
                rows.append({
                    "n": n,
                    "strategy_name": row["strategy_name"],
                    "metric_or_lex": metric,
                    "v02K2_value": row[metric],
                    "prior_combined552_best": ref_best,
                    "improvement_ratio": row[metric] / ref_best if ref_best else np.nan,
                    "improvement_percent": (ref_best - row[metric]) / ref_best * 100 if ref_best else np.nan,
                    "primary_or_diagnostic": "primary" if metric in PRIMARY_METRICS else "diagnostic",
                    "caution_note": "Teacher-metric new record relative to combined552 only; experimental validation not implied.",
                })
        ref_lex = sorted([tuple(x) for x in ref_n[PRIMARY_METRICS].to_numpy()])
        for _, row in cand_n.iterrows():
            key = tuple(row[m] for m in PRIMARY_METRICS)
            if key < ref_lex[0]:
                rows.append({
                    "n": n,
                    "strategy_name": row["strategy_name"],
                    "metric_or_lex": "primary_lex_u2_peeq_surfaceT",
                    "v02K2_value": str(key),
                    "prior_combined552_best": str(ref_lex[0]),
                    "improvement_ratio": "",
                    "improvement_percent": "",
                    "primary_or_diagnostic": "primary",
                    "caution_note": "Lexicographic new record relative to combined552 only.",
                })
    fieldnames = ["n", "strategy_name", "metric_or_lex", "v02K2_value", "prior_combined552_best", "improvement_ratio", "improvement_percent", "primary_or_diagnostic", "caution_note"]
    df = pd.DataFrame(rows, columns=fieldnames)
    df.to_csv(NEW_RECORDS_CSV, index=False)
    counts = {
        "total": int(len(df)),
        "primary": int((df["primary_or_diagnostic"] == "primary").sum()) if len(df) else 0,
        "diagnostic": int((df["primary_or_diagnostic"] == "diagnostic").sum()) if len(df) else 0,
    }
    return df, counts


def topk_tables(ranking: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for _, row in ranking.iterrows():
        reasons = []
        for metric in PRIMARY_METRICS:
            if to_bool(row.get(f"candidate_top10pct_ref_{metric}", False)):
                reasons.append(f"top10_{metric}")
            if to_bool(row.get(f"candidate_top25pct_ref_{metric}", False)):
                reasons.append(f"top25_{metric}")
        if to_bool(row.get("candidate_top10pct_ref_lex", False)):
            reasons.append("top10_primary_lex")
        if to_bool(row.get("candidate_top25pct_ref_lex", False)):
            reasons.append("top25_primary_lex")
        if to_bool(row.get("candidate_top10pct_ref_mises_max", False)):
            reasons.append("diagnostic_top10_mises")
        if to_bool(row.get("candidate_top25pct_ref_mises_max", False)):
            reasons.append("diagnostic_top25_mises")
        if reasons:
            rec = row.to_dict()
            rec["topk_reasons"] = ";".join(reasons)
            rec["primary_topk"] = any("mises" not in r for r in reasons)
            rec["diagnostic_mises_only"] = (not rec["primary_topk"]) and any("mises" in r for r in reasons)
            rows.append(rec)
    topk = pd.DataFrame(rows)
    topk.to_csv(TOPK_CSV, index=False)

    summary_rows = []
    for n in SUPPORTED_N:
        sub = ranking[ranking["n"] == n]
        unique_topk = sub[sub.apply(row_has_primary_topk, axis=1)]
        summary_rows.append({
            "n": n,
            "v02K2_count": int(len(sub)),
            "top10pct_U2_count": int(sub["candidate_top10pct_ref_u2_range"].sum()),
            "top25pct_U2_count": int(sub["candidate_top25pct_ref_u2_range"].sum()),
            "top10pct_PEEQ_count": int(sub["candidate_top10pct_ref_peeq_max"].sum()),
            "top25pct_PEEQ_count": int(sub["candidate_top25pct_ref_peeq_max"].sum()),
            "top10pct_SurfaceT_count": int(sub["candidate_top10pct_ref_surface_t_proxy"].sum()),
            "top25pct_SurfaceT_count": int(sub["candidate_top25pct_ref_surface_t_proxy"].sum()),
            "top10pct_lex_count": int(sub["candidate_top10pct_ref_lex"].sum()),
            "top25pct_lex_count": int(sub["candidate_top25pct_ref_lex"].sum()),
            "diagnostic_Mises_topk_count": int((sub["candidate_top10pct_ref_mises_max"] | sub["candidate_top25pct_ref_mises_max"]).sum()),
            "total_unique_primary_topk_candidates": int(len(unique_topk)),
            "total_unique_any_topk_candidates": int(len(topk[topk["n"] == n])) if len(topk) else 0,
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(TOPK_SUMMARY_CSV, index=False)
    return topk, summary


def annotate_any(df: pd.DataFrame, ref_df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    fake = df.copy()
    fake["dataset_source"] = source_label
    return annotate_v02_ranks(ref_df, fake)


def v02_vs_v01_tables(ref_df: pd.DataFrame, v02_rank: pd.DataFrame, v01_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    v01_rank = annotate_any(v01_df, ref_df, "ppo_v02K2_targeted_batch32")
    comparison_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(RNG_SEED)
    bootstrap_rows: list[dict[str, Any]] = []
    known_v01_best = {24: 134, 40: 147}
    known_v01_topk = {24: 3, 40: 0}

    for n in SUPPORTED_N:
        v01n = v01_rank[v01_rank["n"] == n]
        v02n = v02_rank[v02_rank["n"] == n]
        v01_topk = int(v01n.apply(row_has_primary_topk, axis=1).sum())
        v02_topk = int(v02n.apply(row_has_primary_topk, axis=1).sum())
        row: dict[str, Any] = {
            "n": n,
            "v01_candidate_count": int(len(v01n)),
            "v02K2_candidate_count": int(len(v02n)),
            "v01_best_lex_rank": int(v01n["ref_lex_rank"].min()),
            "v02K2_best_lex_rank": int(v02n["ref_lex_rank"].min()),
            "known_v01_best_lex_rank_from_stageH": known_v01_best[n],
            "v01_topk_count": v01_topk,
            "v02K2_topk_count": v02_topk,
            "known_v01_topk_count_from_stageH": known_v01_topk[n],
            "v02K2_improves_v01_best_lex_rank": bool(v02n["ref_lex_rank"].min() < v01n["ref_lex_rank"].min()),
            "v02K2_improves_known_v01_best_lex_rank": bool(v02n["ref_lex_rank"].min() < known_v01_best[n]),
            "v02K2_improves_v01_topk_count": bool(v02_topk > v01_topk),
            "v02K2_improves_known_v01_topk_count": bool(v02_topk > known_v01_topk[n]),
        }
        for metric in METRICS:
            row[f"v01_best_{metric}"] = float(v01n[metric].min())
            row[f"v02K2_best_{metric}"] = float(v02n[metric].min())
            row[f"v01_median_{metric}"] = float(v01n[metric].median())
            row[f"v02K2_median_{metric}"] = float(v02n[metric].median())
            row[f"v02K2_best_improves_v01_{metric}"] = bool(v02n[metric].min() < v01n[metric].min())
            row[f"v02K2_median_improves_v01_{metric}"] = bool(v02n[metric].median() < v01n[metric].median())
        if row["v02K2_improves_v01_best_lex_rank"] and row["v02K2_improves_v01_topk_count"]:
            row["interpretation"] = "v02K2 improves v01 in best lex rank and top-k count"
        elif row["v02K2_improves_v01_best_lex_rank"] or row["v02K2_improves_v01_topk_count"]:
            row["interpretation"] = "v02K2 partially improves v01"
        else:
            row["interpretation"] = "v02K2 does not improve v01 on primary targeted summary"
        comparison_rows.append(row)

        lex_ranks = v02n["ref_lex_rank"].to_numpy()
        topk_flags = v02n.apply(row_has_primary_topk, axis=1).to_numpy()
        best_samples, topk_samples = [], []
        for _ in range(BOOTSTRAP_TRIALS):
            idx = rng.choice(len(v02n), size=8, replace=False)
            best_samples.append(float(np.min(lex_ranks[idx])))
            topk_samples.append(int(np.sum(topk_flags[idx])))
        best_arr = np.array(best_samples)
        topk_arr = np.array(topk_samples)
        bootstrap_rows.append({
            "n": n,
            "trials": BOOTSTRAP_TRIALS,
            "v01_best_lex_rank": int(v01n["ref_lex_rank"].min()),
            "known_v01_best_lex_rank_from_stageH": known_v01_best[n],
            "expected_v02K2_best_lex_rank_equal8_mean": float(best_arr.mean()),
            "expected_v02K2_best_lex_rank_equal8_median": float(np.median(best_arr)),
            "expected_v02K2_best_lex_rank_equal8_q05": float(np.quantile(best_arr, 0.05)),
            "expected_v02K2_best_lex_rank_equal8_q95": float(np.quantile(best_arr, 0.95)),
            "prob_v02K2_subsample_beats_v01_best_lex_rank": float(np.mean(best_arr < v01n["ref_lex_rank"].min())),
            "prob_v02K2_subsample_beats_known_v01_best_lex_rank": float(np.mean(best_arr < known_v01_best[n])),
            "v01_topk_count": v01_topk,
            "known_v01_topk_count_from_stageH": known_v01_topk[n],
            "expected_v02K2_topk_count_equal8_mean": float(topk_arr.mean()),
            "expected_v02K2_topk_count_equal8_median": float(np.median(topk_arr)),
            "expected_v02K2_topk_count_equal8_q05": float(np.quantile(topk_arr, 0.05)),
            "expected_v02K2_topk_count_equal8_q95": float(np.quantile(topk_arr, 0.95)),
            "prob_v02K2_subsample_beats_v01_topk_count": float(np.mean(topk_arr > v01_topk)),
            "prob_v02K2_subsample_beats_known_v01_topk_count": float(np.mean(topk_arr > known_v01_topk[n])),
        })
    comparison = pd.DataFrame(comparison_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    comparison.to_csv(V02_VS_V01_CSV, index=False)
    bootstrap.to_csv(V02_VS_V01_BOOTSTRAP_CSV, index=False)
    return comparison, bootstrap


def topk_count_for_ref_sample(ranks: pd.DataFrame) -> int:
    flags = []
    for _, row in ranks.iterrows():
        is_top = False
        for metric in PRIMARY_METRICS:
            is_top = is_top or row[f"ref_rank_{metric}"] <= top_threshold(REF_COUNTS[int(row["n"])], 0.25)
        is_top = is_top or row["ref_lex_rank"] <= top_threshold(REF_COUNTS[int(row["n"])], 0.25)
        flags.append(is_top)
    return int(np.sum(flags))


def bootstrap_random_reference(ref_df: pd.DataFrame, v02_rank: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RNG_SEED + 1)
    rows: list[dict[str, Any]] = []
    global_values = []

    ref_ranks = annotate_reference_self(ref_df)
    observed_global = int(v02_rank.apply(row_has_primary_topk, axis=1).sum())

    for n in SUPPORTED_N:
        ref_n = ref_ranks[ref_ranks["n"] == n].reset_index(drop=True)
        v02_n = v02_rank[v02_rank["n"] == n]
        observed = {
            "top10pct_lex_count": int(v02_n["candidate_top10pct_ref_lex"].sum()),
            "top25pct_lex_count": int(v02_n["candidate_top25pct_ref_lex"].sum()),
            "top10pct_U2_count": int(v02_n["candidate_top10pct_ref_u2_range"].sum()),
            "top25pct_U2_count": int(v02_n["candidate_top25pct_ref_u2_range"].sum()),
            "top10pct_PEEQ_count": int(v02_n["candidate_top10pct_ref_peeq_max"].sum()),
            "top25pct_PEEQ_count": int(v02_n["candidate_top25pct_ref_peeq_max"].sum()),
            "top10pct_SurfaceT_count": int(v02_n["candidate_top10pct_ref_surface_t_proxy"].sum()),
            "top25pct_SurfaceT_count": int(v02_n["candidate_top25pct_ref_surface_t_proxy"].sum()),
            "best_lex_rank": float(v02_n["ref_lex_rank"].min()),
            "median_lex_rank": float(v02_n["ref_lex_rank"].median()),
        }
        sample_metrics = {key: [] for key in observed}
        for _ in range(BOOTSTRAP_TRIALS):
            sample = ref_n.iloc[rng.choice(len(ref_n), size=16, replace=False)]
            sample_metrics["top10pct_lex_count"].append(int((sample["ref_lex_rank"] <= top_threshold(len(ref_n), 0.10)).sum()))
            sample_metrics["top25pct_lex_count"].append(int((sample["ref_lex_rank"] <= top_threshold(len(ref_n), 0.25)).sum()))
            sample_metrics["top10pct_U2_count"].append(int((sample["ref_rank_u2_range"] <= top_threshold(len(ref_n), 0.10)).sum()))
            sample_metrics["top25pct_U2_count"].append(int((sample["ref_rank_u2_range"] <= top_threshold(len(ref_n), 0.25)).sum()))
            sample_metrics["top10pct_PEEQ_count"].append(int((sample["ref_rank_peeq_max"] <= top_threshold(len(ref_n), 0.10)).sum()))
            sample_metrics["top25pct_PEEQ_count"].append(int((sample["ref_rank_peeq_max"] <= top_threshold(len(ref_n), 0.25)).sum()))
            sample_metrics["top10pct_SurfaceT_count"].append(int((sample["ref_rank_surface_t_proxy"] <= top_threshold(len(ref_n), 0.10)).sum()))
            sample_metrics["top25pct_SurfaceT_count"].append(int((sample["ref_rank_surface_t_proxy"] <= top_threshold(len(ref_n), 0.25)).sum()))
            sample_metrics["best_lex_rank"].append(float(sample["ref_lex_rank"].min()))
            sample_metrics["median_lex_rank"].append(float(sample["ref_lex_rank"].median()))
        for metric_name, values in sample_metrics.items():
            arr = np.array(values)
            if metric_name in {"best_lex_rank", "median_lex_rank"}:
                p = float(np.mean(arr <= observed[metric_name]))
                interp = "enriched" if observed[metric_name] < np.quantile(arr, 0.05) else ("weak" if observed[metric_name] > np.quantile(arr, 0.95) else "comparable")
            else:
                p = float(np.mean(arr >= observed[metric_name]))
                interp = "enriched" if observed[metric_name] > np.quantile(arr, 0.95) else ("weak" if observed[metric_name] < np.quantile(arr, 0.05) else "comparable")
            rows.append({
                "n": n,
                "metric": metric_name,
                "observed": observed[metric_name],
                "bootstrap_mean": float(arr.mean()),
                "bootstrap_median": float(np.median(arr)),
                "q05": float(np.quantile(arr, 0.05)),
                "q95": float(np.quantile(arr, 0.95)),
                "empirical_p_value_greater_equal": p,
                "interpretation": interp,
            })

    for _ in range(BOOTSTRAP_TRIALS):
        total = 0
        for n in SUPPORTED_N:
            ref_n = ref_ranks[ref_ranks["n"] == n].reset_index(drop=True)
            sample = ref_n.iloc[rng.choice(len(ref_n), size=16, replace=False)]
            total += topk_count_for_ref_sample(sample)
        global_values.append(total)
    garr = np.array(global_values)
    global_row = {
        "metric": "total_unique_primary_top25_or_lex_top25_count",
        "observed": observed_global,
        "bootstrap_mean": float(garr.mean()),
        "bootstrap_median": float(np.median(garr)),
        "q05": float(np.quantile(garr, 0.05)),
        "q95": float(np.quantile(garr, 0.95)),
        "empirical_p_value_greater_equal": float(np.mean(garr >= observed_global)),
        "interpretation": "enriched" if observed_global > np.quantile(garr, 0.95) else ("weak" if observed_global < np.quantile(garr, 0.05) else "comparable"),
        "note": "Equal-budget bootstrap against existing teacher-labelled reference distribution, not the full scan-order universe.",
    }
    by_n = pd.DataFrame(rows)
    global_df = pd.DataFrame([global_row])
    by_n.to_csv(BOOTSTRAP_BY_N_CSV, index=False)
    global_df.to_csv(BOOTSTRAP_GLOBAL_CSV, index=False)
    return by_n, global_df


def annotate_reference_self(ref_df: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    for n in SUPPORTED_N:
        sub = ref_df[ref_df["n"] == n].copy().reset_index(drop=True)
        for metric in METRICS:
            sub[f"ref_rank_{metric}"] = sub[metric].rank(method="min", ascending=True)
        idx = sub.sort_values(PRIMARY_METRICS).index
        sub.loc[idx, "ref_lex_rank"] = np.arange(1, len(sub) + 1)
        out_rows.append(sub)
    return pd.concat(out_rows, ignore_index=True)


def baseline_family_tables(ref_df: pd.DataFrame, v02_rank: pd.DataFrame, raw_ref: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_cols = [c for c in ["strategy_name", "candidate_family", "family", "source", "generation_tag", "batch", "strategy_type", "candidate_source", "selection_tag", "generation_method"] if c in raw_ref.columns]
    patterns = {
        "raster": r"raster",
        "odd_even": r"odd[_ -]?even|even[_ -]?odd",
        "edge_in": r"edge[_ -]?in",
        "center_out": r"center[_ -]?out",
        "center_edge": r"center[_ -]?edge",
        "method_c": r"method[_ -]?c",
        "regular_jump": r"regular[_ -]?jump|jump",
        "engineering": r"engineering",
        "random": r"random",
        "heuristic": r"heuristic|baseline",
    }
    inventory_rows: list[dict[str, Any]] = []
    compare_rows: list[dict[str, Any]] = []
    raw = raw_ref[raw_ref["n"].astype(int).isin(SUPPORTED_N)].copy()
    raw["_label_text"] = ""
    for col in source_cols:
        raw["_label_text"] += " " + raw[col].astype(str).str.lower()

    for family, pattern in patterns.items():
        fam_raw = raw[raw["_label_text"].str.contains(pattern, regex=True, na=False)].copy()
        for n in SUPPORTED_N:
            count = int((fam_raw["n"].astype(int) == n).sum())
            inventory_rows.append({"family": family, "n": n, "count": count, "source_columns_used": ";".join(source_cols), "reliability": "FOUND" if count else "NOT_FOUND"})
            if count:
                fam = canonical_reference(fam_raw[fam_raw["n"].astype(int) == n])
                v02n = v02_rank[v02_rank["n"] == n]
                refn = ref_df[ref_df["n"] == n]
                fam_rank = annotate_reference_self(pd.concat([refn, fam], ignore_index=True))
                fam_only = fam_rank[fam_rank["dataset_source"] == "stage3_native_combined552"].copy()
                row = {"family": family, "n": n, "family_count": count, "v02K2_count": int(len(v02n))}
                for metric in METRICS:
                    row[f"family_best_{metric}"] = float(fam[metric].min())
                    row[f"family_median_{metric}"] = float(fam[metric].median())
                    row[f"v02K2_best_{metric}"] = float(v02n[metric].min())
                    row[f"v02K2_median_{metric}"] = float(v02n[metric].median())
                    row[f"v02K2_best_beats_family_best_{metric}"] = bool(v02n[metric].min() < fam[metric].min())
                    row[f"v02K2_median_beats_family_median_{metric}"] = bool(v02n[metric].median() < fam[metric].median())
                row["comparison_boundary"] = "Label-derived family comparison; use only where family labels are reliable."
                compare_rows.append(row)

    inventory = pd.DataFrame(inventory_rows)
    compare = pd.DataFrame(compare_rows)
    if compare.empty:
        compare = pd.DataFrame([{"status": "NOT_RELIABLE", "detail": "No reliable explicit baseline-family labels found for N24/N40."}])
    inventory.to_csv(BASELINE_INVENTORY_CSV, index=False)
    compare.to_csv(BASELINE_COMPARE_CSV, index=False)
    return inventory, compare


def corr_pair(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    tmp = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(tmp) < 3 or tmp["x"].nunique() < 2 or tmp["y"].nunique() < 2:
        return np.nan, np.nan
    pearson = float(tmp["x"].corr(tmp["y"], method="pearson"))
    spearman = float(tmp["x"].corr(tmp["y"], method="spearman"))
    return spearman, pearson


def surrogate_alignment(ranking: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = ranking.copy()
    out["teacher_reward"] = out["teacher_lex_reward_rank_normalized"]
    out["predicted_rank_desc"] = out["predicted_reward"].rank(method="min", ascending=False)
    out["conservative_rank_desc"] = out["conservative_reward"].rank(method="min", ascending=False)
    out["teacher_rank_asc"] = out["ref_lex_rank"]
    high_pred = out["predicted_rank_desc"] <= max(1, math.ceil(len(out) * 0.25))
    poor_teacher = out["teacher_rank_asc"] > out["n"].map(lambda n: top_threshold(REF_COUNTS[int(n)], 0.50))
    strong_teacher = out.apply(row_has_primary_topk, axis=1)
    out["false_positive_high_pred_poor_teacher"] = high_pred & poor_teacher
    out["true_positive_high_pred_strong_teacher"] = high_pred & strong_teacher
    out.to_csv(ALIGNMENT_CSV, index=False)

    pred_s, pred_p = corr_pair(out["predicted_reward"], out["teacher_reward"])
    cons_s, cons_p = corr_pair(out["conservative_reward"], out["teacher_reward"])
    by_n = {}
    for n in SUPPORTED_N:
        sub = out[out["n"] == n]
        ps, pp = corr_pair(sub["predicted_reward"], sub["teacher_reward"])
        cs, cp = corr_pair(sub["conservative_reward"], sub["teacher_reward"])
        by_n[str(n)] = {
            "predicted_spearman": ps,
            "predicted_pearson": pp,
            "conservative_spearman": cs,
            "conservative_pearson": cp,
            "false_positive_count": int(sub["false_positive_high_pred_poor_teacher"].sum()),
            "true_positive_count": int(sub["true_positive_high_pred_strong_teacher"].sum()),
        }
    summary = {
        "overall_predicted_spearman": pred_s,
        "overall_predicted_pearson": pred_p,
        "overall_conservative_spearman": cons_s,
        "overall_conservative_pearson": cons_p,
        "false_positive_count": int(out["false_positive_high_pred_poor_teacher"].sum()),
        "true_positive_count": int(out["true_positive_high_pred_strong_teacher"].sum()),
        "by_N": by_n,
    }
    ALIGNMENT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out, summary


def summary_tables(ranking: pd.DataFrame, ref_df: pd.DataFrame, new_records: pd.DataFrame, topk_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    best_rows = []
    for n in SUPPORTED_N:
        sub = ranking[ranking["n"] == n]
        refn = ref_df[ref_df["n"] == n]
        topk_row = topk_summary[topk_summary["n"] == n].iloc[0].to_dict()
        row: dict[str, Any] = {
            "n": n,
            "v02K2_count": int(len(sub)),
            "reference_count": int(len(refn)),
            "best_v02K2_lex_candidate": sub.sort_values("ref_lex_rank").iloc[0]["strategy_name"],
            "best_v02K2_ref_lex_rank": int(sub["ref_lex_rank"].min()),
            "new_record_count": int((new_records["n"] == n).sum()) if len(new_records) else 0,
            "primary_new_record_count": int(((new_records["n"] == n) & (new_records["primary_or_diagnostic"] == "primary")).sum()) if len(new_records) else 0,
            "total_unique_primary_topk_candidates": int(topk_row["total_unique_primary_topk_candidates"]),
            "total_unique_any_topk_candidates": int(topk_row["total_unique_any_topk_candidates"]),
        }
        for metric in METRICS:
            row[f"best_v02K2_{metric}"] = float(sub[metric].min())
            row[f"best_reference_{metric}"] = float(refn[metric].min())
            row[f"v02K2_best_over_reference_best_ratio_{metric}"] = float(sub[metric].min() / refn[metric].min())
            row[f"median_v02K2_{metric}"] = float(sub[metric].median())
            row[f"median_reference_{metric}"] = float(refn[metric].median())
            row[f"v02K2_median_over_reference_median_ratio_{metric}"] = float(sub[metric].median() / refn[metric].median())
            row[f"v02K2_beats_ref_best_count_{metric}"] = int(sub[f"candidate_beats_ref_best_{metric}"].sum())
            row[f"v02K2_top10pct_count_{metric}"] = int(sub[f"candidate_top10pct_ref_{metric}"].sum())
            row[f"v02K2_top25pct_count_{metric}"] = int(sub[f"candidate_top25pct_ref_{metric}"].sum())
        summary_rows.append(row)

        for label, sort_key in [
            ("best_U2", "u2_range"),
            ("best_PEEQ", "peeq_max"),
            ("best_SurfaceT", "surface_t_proxy"),
            ("best_primary_lex", "ref_lex_rank"),
            ("best_diagnostic_Mises", "mises_max"),
        ]:
            b = sub.sort_values(sort_key).iloc[0]
            metric = "primary_lex" if label == "best_primary_lex" else ("mises_max" if label == "best_diagnostic_Mises" else sort_key)
            best_rows.append({
                "n": n,
                "best_type": label,
                "strategy_name": b["strategy_name"],
                "metric_or_rank": metric,
                "value": b[sort_key],
                "ref_rank_u2": b["ref_rank_u2_range"],
                "ref_rank_peeq": b["ref_rank_peeq_max"],
                "ref_rank_surfaceT": b["ref_rank_surface_t_proxy"],
                "ref_rank_mises": b["ref_rank_mises_max"],
                "ref_lex_rank": b["ref_lex_rank"],
                "is_new_record": bool(b.get(f"candidate_beats_ref_best_{sort_key}", False)) if sort_key in METRICS else bool(b.get("candidate_beats_ref_lex_best", False)),
                "is_top10pct": bool(b.get(f"candidate_top10pct_ref_{sort_key}", False)) if sort_key in METRICS else bool(b.get("candidate_top10pct_ref_lex", False)),
                "is_top25pct": bool(b.get(f"candidate_top25pct_ref_{sort_key}", False)) if sort_key in METRICS else bool(b.get("candidate_top25pct_ref_lex", False)),
            })
    summary = pd.DataFrame(summary_rows)
    best = pd.DataFrame(best_rows)
    global_summary = pd.DataFrame([{
        "v02K2_total_count": int(len(ranking)),
        "new_record_count_total": int(len(new_records)),
        "primary_new_record_count_total": int((new_records["primary_or_diagnostic"] == "primary").sum()) if len(new_records) else 0,
        "total_unique_primary_topk_candidates": int(topk_summary["total_unique_primary_topk_candidates"].sum()),
        "total_unique_any_topk_candidates": int(topk_summary["total_unique_any_topk_candidates"].sum()),
        "best_overall_ref_lex_rank": int(ranking["ref_lex_rank"].min()),
        "best_overall_ref_lex_candidate": ranking.sort_values("ref_lex_rank").iloc[0]["strategy_name"],
    }])
    summary.to_csv(SUMMARY_BY_N_CSV, index=False)
    best.to_csv(BEST_BY_N_CSV, index=False)
    global_summary.to_csv(GLOBAL_SUMMARY_CSV, index=False)
    return summary, global_summary, best


def make_plots(ref_df: pd.DataFrame, v02_rank: pd.DataFrame, v01_compare: pd.DataFrame, bootstrap_global: pd.DataFrame, alignment: pd.DataFrame) -> list[str]:
    paths: list[str] = []
    for metric, title in [
        ("u2_range", "U2 Range"),
        ("peeq_max", "PEEQ Max"),
        ("surface_t_proxy", "SurfaceT Proxy"),
        ("mises_max", "Mises Max"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, n in zip(axes, SUPPORTED_N):
            ref_vals = ref_df[ref_df["n"] == n][metric].dropna()
            v02_vals = v02_rank[v02_rank["n"] == n][metric].dropna()
            ax.boxplot([ref_vals, v02_vals], labels=["combined552", "v02K2"])
            ax.set_title(f"N{n}")
            ax.set_ylabel(metric)
        fig.suptitle(f"v02K2 vs combined552 {title}")
        fig.tight_layout()
        path = PLOTS_DIR / f"v02K2_vs_combined552_{metric}_by_N.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(v02_rank["n"], v02_rank["ref_lex_percentile"], alpha=0.8)
    ax.set_xlabel("N")
    ax.set_ylabel("Reference lex percentile")
    ax.set_title("v02K2 lexicographic percentile by N")
    path = PLOTS_DIR / "v02K2_lex_rank_percentile_by_N.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(v01_compare))
    ax.bar(x - 0.2, v01_compare["v01_best_lex_rank"], width=0.4, label="v01")
    ax.bar(x + 0.2, v01_compare["v02K2_best_lex_rank"], width=0.4, label="v02K2")
    ax.set_xticks(x, [f"N{n}" for n in v01_compare["n"]])
    ax.invert_yaxis()
    ax.set_ylabel("Best ref lex rank (lower is better)")
    ax.legend()
    path = PLOTS_DIR / "v02K2_vs_v01_best_lex_rank.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(7, 4))
    observed = bootstrap_global.iloc[0]["observed"]
    mean = bootstrap_global.iloc[0]["bootstrap_mean"]
    ax.bar(["v02K2 observed", "bootstrap mean"], [observed, mean])
    ax.set_ylabel("Top-k count")
    ax.set_title("v02K2 vs random-reference bootstrap top-k count")
    path = PLOTS_DIR / "v02K2_vs_bootstrap_random_reference_topk_count.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(alignment["conservative_reward"], alignment["teacher_reward"], label="conservative", alpha=0.8)
    ax.scatter(alignment["predicted_reward"], alignment["teacher_reward"], label="predicted", alpha=0.6, marker="x")
    ax.set_xlabel("Surrogate reward")
    ax.set_ylabel("Teacher-derived lex reward")
    ax.legend()
    ax.set_title("Surrogate reward vs teacher-derived reward")
    path = PLOTS_DIR / "v02K2_surrogate_reward_vs_teacher_reward.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))
    return paths


def decide_verdict(new_counts: dict[str, int], topk_summary: pd.DataFrame, v01_compare: pd.DataFrame) -> str:
    if new_counts["primary"] > 0:
        return "PASS_STAGEN_V02K2_TEACHER_VALIDATED_WITH_NEW_RECORDS"
    improves_v01 = bool(v01_compare["v02K2_improves_v01_best_lex_rank"].any() or v01_compare["v02K2_improves_v01_topk_count"].any())
    if improves_v01:
        return "PASS_STAGEN_V02K2_TEACHER_VALIDATED_AND_IMPROVES_V01_TARGETED"
    if int(topk_summary["total_unique_primary_topk_candidates"].sum()) > 0:
        return "PASS_STAGEN_V02K2_TEACHER_VALIDATED_AND_COMPETITIVE"
    return "WARNING_STAGEN_V02K2_TEACHER_VALIDATED_BUT_NOT_COMPETITIVE"


def write_report(
    audit_verdict: str,
    verdict: str,
    summary_by_n: pd.DataFrame,
    new_counts: dict[str, int],
    topk_summary: pd.DataFrame,
    v01_compare: pd.DataFrame,
    v01_boot: pd.DataFrame,
    boot_global: pd.DataFrame,
    baseline_compare: pd.DataFrame,
    align_summary: dict[str, Any],
    best_by_n: pd.DataFrame,
) -> None:
    best_lex = best_by_n[best_by_n["best_type"] == "best_primary_lex"][["n", "strategy_name", "ref_lex_rank", "is_top25pct"]].to_dict("records")
    safe_claim = (
        "v02K2 candidates were teacher-validated for N24/N40; performance claims are limited to the ranking, "
        "top-k, v01-comparison, and bootstrap evidence generated in Stage N."
    )
    if new_counts["primary"] == 0:
        new_record_text = "No primary-metric or primary-lex new records over native combined552 were found."
    else:
        new_record_text = f"{new_counts['primary']} primary new-record rows were found relative to combined552."
    report = f"""# PPO v02K2 Stage N Teacher-Metric Ranking Report

## Purpose
Compare the teacher-metric-extracted PPO v02K2 N24/N40 batch32 against native combined552, PPO v01 N24/N40, identified baseline labels where available, and equal-budget bootstrap draws from combined552.

## Inputs
- Stage M metrics: `{STAGE_M_METRICS}`
- K2 selected batch: `{K2_SELECTED}`
- Native combined552: `{COMBINED552}`
- PPO v01 metrics: `{V01_METRICS}`
- PPO v01 ranking: `{V01_RANKING}`

## Stage M Extraction Status
Stage M reported 32/32 extracted teacher-metric rows: N24=16 and N40=16.

## Input Integrity Verdict
`{audit_verdict}`

## Analysis Datasets
- combined552 + v02K2: `{ANALYSIS_V02_CSV}`
- combined552 + v01 + v02K2: `{ANALYSIS_ALL_CSV}`

## v02K2 Ranking Against Native combined552
{md_table(summary_by_n.to_dict("records"), ["n", "v02K2_count", "best_v02K2_lex_candidate", "best_v02K2_ref_lex_rank", "total_unique_primary_topk_candidates", "new_record_count"])}

## New-Record Audit
{new_record_text}

Table: `{NEW_RECORDS_CSV}`

## Top-k Competitiveness Audit
{md_table(topk_summary.to_dict("records"), ["n", "v02K2_count", "top10pct_U2_count", "top25pct_U2_count", "top10pct_lex_count", "top25pct_lex_count", "total_unique_primary_topk_candidates", "diagnostic_Mises_topk_count"])}

## v02K2 vs PPO v01 N24/N40
{md_table(v01_compare.to_dict("records"), ["n", "v01_candidate_count", "v02K2_candidate_count", "v01_best_lex_rank", "v02K2_best_lex_rank", "v01_topk_count", "v02K2_topk_count", "interpretation"])}

## Equal-Budget v02K2-vs-v01 Bootstrap
{md_table(v01_boot.to_dict("records"), ["n", "expected_v02K2_best_lex_rank_equal8_median", "prob_v02K2_subsample_beats_v01_best_lex_rank", "expected_v02K2_topk_count_equal8_median", "prob_v02K2_subsample_beats_v01_topk_count"])}

## v02K2 vs Random-Reference Bootstrap
Global bootstrap summary:

{md_table(boot_global.to_dict("records"), ["metric", "observed", "bootstrap_mean", "q05", "q95", "empirical_p_value_greater_equal", "interpretation"])}

This is an equal-budget bootstrap against the existing teacher-labelled reference distribution, not against the full scan-order universe.

## Identified Baseline-Family Comparison
Output: `{BASELINE_COMPARE_CSV}`

Status preview:
{md_table(baseline_compare.head(8).to_dict("records"), list(baseline_compare.head(8).columns))}

## Surrogate-to-Teacher Alignment
- Predicted reward Spearman: `{align_summary["overall_predicted_spearman"]:.4f}`
- Predicted reward Pearson: `{align_summary["overall_predicted_pearson"]:.4f}`
- Conservative reward Spearman: `{align_summary["overall_conservative_spearman"]:.4f}`
- Conservative reward Pearson: `{align_summary["overall_conservative_pearson"]:.4f}`
- False positives: `{align_summary["false_positive_count"]}`
- True positives: `{align_summary["true_positive_count"]}`

## Best Candidates By N
{md_table(best_lex, ["n", "strategy_name", "ref_lex_rank", "is_top25pct"])}

## Claim Implications
{safe_claim}

## Limitations
- This is teacher-metric analysis only and does not create new candidates.
- The random-reference bootstrap samples from the active-learning-enriched combined552 pool, not from all possible permutations.
- Surrogate score alignment is diagnostic and cannot replace Abaqus teacher validation.

## Verdict
`{verdict}`
"""
    REPORT_PATH.write_text(report, encoding="utf-8")

    claim_boundary = f"""# PPO v02K2 Stage N Claim Boundary

## Safe Claims If Supported By Stage N Tables
- v02K2 candidates were teacher-validated 32/32 for targeted N24/N40.
- v02K2 can be compared against native combined552, PPO v01 N24/N40, baseline-family labels where reliable, and equal-budget reference bootstrap draws.
- v02K2 improved over PPO v01 N24/N40 only where the Stage N comparison tables show better teacher-metric ranks or top-k counts.
- v02K2 achieved top-k competitiveness only where the Stage N top-k tables show it.
- v02K2 set new records only if `{NEW_RECORDS_CSV}` contains primary-metric or primary-lex rows.

## Unsafe Unless Proven
- v02K2 outperformed the combined552 best.
- v02K2 solved N40.
- v02K2 dominated all N.
- v02K2 was online Abaqus PPO.
- v02K2 is experimentally validated.

## Stage N Verdict
`{verdict}`
"""
    CLAIM_BOUNDARY_PATH.write_text(claim_boundary, encoding="utf-8")


def write_manifest(verdict: str, plot_paths: list[str]) -> None:
    manifest = {
        "branch": git_branch(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "Stage M metrics input": str(STAGE_M_METRICS),
        "combined552 reference input": str(COMBINED552),
        "v01 metrics input": str(V01_METRICS),
        "v01 ranking input": str(V01_RANKING),
        "K2 selected batch input": str(K2_SELECTED),
        "analysis_dataset_paths": [str(ANALYSIS_V02_CSV), str(ANALYSIS_ALL_CSV)],
        "ranking_table_paths": [str(FULL_RANKING_CSV), str(NEW_RECORDS_CSV), str(TOPK_CSV), str(BEST_BY_N_CSV)],
        "summary_table_paths": [str(SUMMARY_BY_N_CSV), str(GLOBAL_SUMMARY_CSV), str(TOPK_SUMMARY_CSV), str(V02_VS_V01_CSV), str(BOOTSTRAP_BY_N_CSV), str(BOOTSTRAP_GLOBAL_CSV), str(ALIGNMENT_JSON)],
        "plot_paths": plot_paths,
        "report_path": str(REPORT_PATH),
        "claim_boundary_path": str(CLAIM_BOUNDARY_PATH),
        "no_Abaqus": True,
        "no_ODB_opening": True,
        "no_ODB_extraction": True,
        "no_solver": True,
        "no_datacheck": True,
        "no_enqueue": True,
        "no_training": True,
        "no_candidate_generation": True,
        "no_commit_or_push": True,
        "final_verdict": verdict,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    stage_m = pd.read_csv(STAGE_M_METRICS)
    selected = pd.read_csv(K2_SELECTED)
    raw_ref = pd.read_csv(COMBINED552)
    v01_metrics = pd.read_csv(V01_METRICS)
    v01_ranking = pd.read_csv(V01_RANKING)

    audit_verdict = input_audit(stage_m, selected, raw_ref, v01_metrics, v01_ranking)
    if audit_verdict.startswith("FAIL"):
        write_manifest("FAIL_STAGEN_V02K2_ANALYSIS_NOT_READY", [])
        raise SystemExit(audit_verdict)

    ref_df = canonical_reference(raw_ref)
    v02_df = canonical_v02(stage_m, selected)
    v01_df = canonical_v01(v01_metrics, v01_ranking)
    write_analysis_datasets(ref_df, v01_df, v02_df)

    ranking = annotate_v02_ranks(ref_df, v02_df)
    ranking.to_csv(FULL_RANKING_CSV, index=False)

    new_records, new_counts = new_record_table(ranking, ref_df)
    topk, topk_summary = topk_tables(ranking)
    v01_compare, v01_boot = v02_vs_v01_tables(ref_df, ranking, v01_df)
    boot_by_n, boot_global = bootstrap_random_reference(ref_df, ranking)
    inventory, baseline_compare = baseline_family_tables(ref_df, ranking, raw_ref)
    alignment, align_summary = surrogate_alignment(ranking)
    summary_by_n, global_summary, best_by_n = summary_tables(ranking, ref_df, new_records, topk_summary)
    plot_paths = make_plots(ref_df, ranking, v01_compare, boot_global, alignment)

    verdict = decide_verdict(new_counts, topk_summary, v01_compare)
    write_report(audit_verdict, verdict, summary_by_n, new_counts, topk_summary, v01_compare, v01_boot, boot_global, baseline_compare, align_summary, best_by_n)
    write_manifest(verdict, plot_paths)

    print(json.dumps({
        "audit_verdict": audit_verdict,
        "final_verdict": verdict,
        "new_records": new_counts,
        "topk_summary_by_N": topk_summary.to_dict("records"),
        "best_lex_by_N": best_by_n[best_by_n["best_type"] == "best_primary_lex"][["n", "strategy_name", "ref_lex_rank"]].to_dict("records"),
        "report": str(REPORT_PATH),
        "manifest": str(MANIFEST_PATH),
    }, indent=2))


if __name__ == "__main__":
    main()
