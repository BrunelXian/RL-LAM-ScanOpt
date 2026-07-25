"""Stage S: PPO v03 teacher-metric ranking and cumulative PPO-pool update.

Analysis-only script. It reads existing CSV/JSON evidence and writes ranking,
comparison, bootstrap, baseline-family, cumulative-pool, report, and manifest
artifacts. It does not run Abaqus, open/extract ODBs, run solver/datacheck,
enqueue jobs, generate CAE/INP/JNL, train models, or generate candidates.
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
NS = "stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40"
V01_NS = "stage3_ppo_rl_lam_fea_addendum_v01"
V02_NS = "stage3_ppo_rl_lam_fea_addendum_v02_targeted_N24_N40"
SUPPORTED_N = [24, 40]
METRICS = ["u2_range", "peeq_max", "surface_t_proxy", "mises_max"]
PRIMARY = ["u2_range", "peeq_max", "surface_t_proxy"]
REF_COUNTS = {24: 190, 40: 206}
V03_COUNTS = {24: 16, 40: 16}
BOOTSTRAP_TRIALS = 2000
RNG_SEED = 20260629

STAGE_R_METRICS = PROJECT_ROOT / "outputs" / NS / "stageR_ODB_teacher_metric_extraction" / "stageR_v03_teacher_metrics.csv"
STAGE_R_SUMMARY = PROJECT_ROOT / "outputs" / NS / "stageR_ODB_teacher_metric_extraction" / "stageR_v03_extraction_summary.json"
STAGE_R_SOLVER_AUDIT = PROJECT_ROOT / "outputs" / NS / "stageR_ODB_teacher_metric_extraction" / "stageR_v03_solver_completion_audit.csv"
STAGE_P_SELECTED = PROJECT_ROOT / "outputs" / NS / "candidate_generation_v03" / "selected_batch32" / "v03_ppo_lex_primary_N24_N40_candidate_batch32.csv"
COMBINED552 = PROJECT_ROOT / "outputs" / "stage3_run_78_final_evidence_freeze_package" / "FROZEN_stage3_native_combined552_teacher_dataset.csv"
V01_METRICS = PROJECT_ROOT / "outputs" / V01_NS / "stageI_final_ppo_evidence_freeze" / "frozen_tables" / "FROZEN_PPO_batch32_teacher_metrics.csv"
V01_RANKING = PROJECT_ROOT / "outputs" / V01_NS / "stageI_final_ppo_evidence_freeze" / "frozen_tables" / "FROZEN_PPO_batch32_teacher_metric_ranking_full.csv"
V02_METRICS = PROJECT_ROOT / "outputs" / V02_NS / "stageM_ODB_teacher_metric_extraction" / "stageM_v02K2_teacher_metrics.csv"
V02_RANKING = PROJECT_ROOT / "outputs" / V02_NS / "stageN_teacher_metric_ranking" / "tables" / "v02K2_teacher_metric_ranking_full.csv"

OUT_ROOT = PROJECT_ROOT / "outputs" / NS / "stageS_teacher_metric_ranking"
CHECKS = OUT_ROOT / "checks"
TABLES = OUT_ROOT / "tables"
PLOTS = OUT_ROOT / "plots"
REPORTS = OUT_ROOT / "reports"
DOCS = PROJECT_ROOT / "docs" / NS

AUDIT_CSV = CHECKS / "stageS_input_integrity_audit.csv"
AUDIT_JSON = CHECKS / "stageS_input_integrity_audit_summary.json"
ANALYSIS_V03 = TABLES / "combined552_N24N40_plus_v03_analysis_dataset.csv"
ANALYSIS_ALL = TABLES / "combined552_N24N40_plus_ppo_v01_v02K2_v03_analysis_dataset.csv"
PPO_POOL = TABLES / "ppo_teacher_validated_pool_v01_v02K2_v03_96cases.csv"
FULL_RANK = TABLES / "v03_teacher_metric_ranking_full.csv"
SUMMARY_BY_N = TABLES / "v03_summary_by_N.csv"
GLOBAL_SUMMARY = TABLES / "v03_global_summary.csv"
BEST_BY_N = TABLES / "v03_best_candidates_by_N.csv"
NEW_RECORDS = TABLES / "v03_new_record_candidates.csv"
TOPK = TABLES / "v03_topk_competitive_candidates.csv"
TOPK_SUMMARY = TABLES / "v03_topk_summary_by_N.csv"
V03_VS_PRIOR = TABLES / "v03_vs_v01_v02K2_targeted_comparison_by_N.csv"
PRIOR_BOOT = TABLES / "v03_vs_prior_ppo_equal_budget_bootstrap.csv"
BOOT_BY_N = TABLES / "v03_vs_bootstrap_random_reference_by_N.csv"
BOOT_GLOBAL = TABLES / "v03_vs_bootstrap_random_reference_global.csv"
BASELINE_INV = TABLES / "v03_identified_baseline_family_inventory.csv"
BASELINE_V03 = TABLES / "v03_vs_identified_baseline_families.csv"
BASELINE_POOL = TABLES / "cumulative_ppo_pool_vs_identified_baseline_families.csv"
ALIGN_CSV = TABLES / "v03_surrogate_vs_teacher_alignment.csv"
ALIGN_JSON = TABLES / "v03_surrogate_vs_teacher_alignment_summary.json"
POOL_SUMMARY = TABLES / "cumulative_ppo_pool_96_summary_by_N.csv"
POOL_PROGRESS = TABLES / "ppo_pool_progress_to_320_target.csv"
REPORT = DOCS / "PPO_V03_STAGES_TEACHER_METRIC_RANKING_REPORT.md"
CLAIM_BOUNDARY = DOCS / "PPO_V03_STAGES_CLAIM_BOUNDARY.md"
MANIFEST = OUT_ROOT / "stageS_v03_teacher_metric_ranking_manifest.json"


def ensure_dirs() -> None:
    for p in [OUT_ROOT, CHECKS, TABLES, PLOTS, REPORTS, DOCS]:
        p.mkdir(parents=True, exist_ok=True)


def git_branch() -> str:
    try:
        r = subprocess.run(
            ["git", "-c", "safe.directory=E:/Projects/RL-LAM-ScanOpt", "-C", str(PROJECT_ROOT), "branch", "--show-current"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        return r.stdout.strip() or BRANCH_FALLBACK
    except Exception:
        return BRANCH_FALLBACK


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for k in row:
                if k not in fieldnames:
                    fieldnames.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def md_table(rows: list[dict[str, Any]], cols: list[str]) -> str:
    def c(v: Any) -> str:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return ""
        return str(v).replace("|", "\\|").replace("\n", "<br>")
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(c(row.get(col, "")) for col in cols) + " |")
    return "\n".join(lines)


def metric_col(df: pd.DataFrame, metric: str) -> str | None:
    if metric in df.columns:
        return metric
    if metric == "surface_t_proxy":
        for c in ["surface_t_proxy_max_tensile_pa", "surface_t_proxy_pa", "surface_t_proxy_mpa"]:
            if c in df.columns:
                return c
    return None


def parse_order(text: Any) -> list[int]:
    return [int(x) for x in re.findall(r"-?\d+", "" if pd.isna(text) else str(text))]


def rank_threshold(n_ref: int, frac: float) -> int:
    return int(math.ceil(n_ref * frac))


def canonical_ref(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw[raw["n"].astype(int).isin(SUPPORTED_N)].copy()
    out = pd.DataFrame()
    out["n"] = raw["n"].astype(int)
    out["strategy_name"] = raw.get("strategy_name", raw.get("handoff_strategy_name", pd.Series([""] * len(raw)))).astype(str)
    out["order_hash"] = raw.get("order_hash", pd.Series([""] * len(raw))).astype(str)
    out["order_compact"] = raw.get("order_compact", pd.Series([""] * len(raw))).astype(str)
    out["dataset_source"] = "stage3_native_combined552"
    out["ppo_version"] = ""
    out["is_ppo"] = False
    out["candidate_source"] = raw.get("candidate_source", pd.Series([""] * len(raw))).astype(str)
    out["ppo_checkpoint"] = ""
    out["ppo_seed"] = ""
    out["predicted_lex_primary_score"] = np.nan
    out["predicted_u2_guarded_score"] = np.nan
    out["final_v03_score"] = np.nan
    out["partial_training_caveat"] = ""
    out["teacher_metrics_extracted"] = True
    out["teacher_validation_status"] = raw.get("teacher_validation_status", pd.Series(["PASS_TEACHER_LABELLED_REFERENCE"] * len(raw))).astype(str)
    out["source_metric_file"] = str(COMBINED552)
    for m in METRICS:
        col = metric_col(raw, m)
        out[m] = pd.to_numeric(raw[col], errors="coerce") if col else np.nan
    return out


def canonical_v03(stage_r: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    m = stage_r.copy()
    m["strategy_name"] = m["handoff_strategy_name"].astype(str)
    sel_cols = [c for c in ["strategy_name", "candidate_source", "ppo_v03_checkpoint", "ppo_seed", "predicted_lex_primary_score", "predicted_u2_guarded_score", "final_v03_score"] if c in selected.columns]
    s = selected.copy()
    merged = m.merge(s[sel_cols], on="strategy_name", how="left", suffixes=("", "_selected")) if sel_cols else m
    out = pd.DataFrame()
    out["n"] = merged["n"].astype(int)
    out["strategy_name"] = merged["strategy_name"].astype(str)
    out["order_hash"] = merged.get("order_hash", pd.Series([""] * len(merged))).astype(str)
    out["order_compact"] = merged.get("order_compact", pd.Series([""] * len(merged))).astype(str)
    out["dataset_source"] = "ppo_v03_batch32"
    out["ppo_version"] = "v03"
    out["is_ppo"] = True
    out["candidate_source"] = merged.get("candidate_source_selected", merged.get("candidate_source", pd.Series(["PPO_v03_checkpoint_inference"] * len(merged)))).astype(str)
    out["ppo_checkpoint"] = merged.get("ppo_v03_checkpoint_selected", merged.get("ppo_v03_checkpoint", pd.Series([""] * len(merged)))).astype(str)
    out["ppo_seed"] = merged.get("ppo_seed_selected", merged.get("ppo_seed", pd.Series([""] * len(merged)))).astype(str)
    for c in ["predicted_lex_primary_score", "predicted_u2_guarded_score", "final_v03_score"]:
        out[c] = pd.to_numeric(merged.get(f"{c}_selected", merged.get(c, pd.Series([np.nan] * len(merged)))), errors="coerce")
    out["partial_training_caveat"] = merged.get("partial_training_caveat", pd.Series([""] * len(merged))).astype(str)
    out["teacher_metrics_extracted"] = True
    out["teacher_validation_status"] = merged.get("teacher_validation_status", pd.Series(["PASS_TEACHER_FIELDS_EXTRACTED"] * len(merged))).astype(str)
    out["source_metric_file"] = str(STAGE_R_METRICS)
    out["final_step_name"] = merged.get("final_step_name", pd.Series([""] * len(merged))).astype(str)
    out["final_frame_time"] = pd.to_numeric(merged.get("final_frame_time", pd.Series([np.nan] * len(merged))), errors="coerce")
    out["extracted_field_names"] = merged.get("extracted_field_names", pd.Series([""] * len(merged))).astype(str)
    out["completion_status"] = merged.get("completion_status", pd.Series([""] * len(merged))).astype(str)
    out["odb_extraction_status"] = merged.get("odb_extraction_status", pd.Series([""] * len(merged))).astype(str)
    for m in METRICS:
        col = metric_col(merged, m)
        out[m] = pd.to_numeric(merged[col], errors="coerce") if col else np.nan
    return out


def canonical_v01(v01: pd.DataFrame, include_all: bool = True) -> pd.DataFrame:
    sub = v01.copy()
    if not include_all:
        sub = sub[sub["n"].astype(int).isin(SUPPORTED_N)].copy()
    out = pd.DataFrame()
    out["n"] = sub["n"].astype(int)
    out["strategy_name"] = sub.get("handoff_strategy_name", sub.get("strategy_name", pd.Series([""] * len(sub)))).astype(str)
    out["order_hash"] = sub.get("order_hash", pd.Series([""] * len(sub))).astype(str)
    out["order_compact"] = sub.get("order_compact", pd.Series([""] * len(sub))).astype(str)
    out["dataset_source"] = "ppo_v01_batch32"
    out["ppo_version"] = "v01"
    out["is_ppo"] = True
    out["candidate_source"] = "PPO_checkpoint_inference"
    out["ppo_checkpoint"] = ""
    out["ppo_seed"] = ""
    out["predicted_lex_primary_score"] = np.nan
    out["predicted_u2_guarded_score"] = np.nan
    out["final_v03_score"] = np.nan
    out["partial_training_caveat"] = ""
    out["teacher_metrics_extracted"] = True
    out["teacher_validation_status"] = sub.get("teacher_validation_status", pd.Series(["PASS_TEACHER_FIELDS_EXTRACTED"] * len(sub))).astype(str)
    out["source_metric_file"] = str(V01_METRICS)
    for m in METRICS:
        col = metric_col(sub, m)
        out[m] = pd.to_numeric(sub[col], errors="coerce") if col else np.nan
    return out


def canonical_v02(v02: pd.DataFrame) -> pd.DataFrame:
    sub = v02.copy()
    out = pd.DataFrame()
    out["n"] = sub["n"].astype(int)
    out["strategy_name"] = sub.get("handoff_strategy_name", sub.get("strategy_name", pd.Series([""] * len(sub)))).astype(str)
    out["order_hash"] = sub.get("order_hash", pd.Series([""] * len(sub))).astype(str)
    out["order_compact"] = sub.get("order_compact", pd.Series([""] * len(sub))).astype(str)
    out["dataset_source"] = "ppo_v02K2_batch32"
    out["ppo_version"] = "v02K2"
    out["is_ppo"] = True
    out["candidate_source"] = sub.get("candidate_source", pd.Series(["PPO_v02K2_checkpoint_inference"] * len(sub))).astype(str)
    out["ppo_checkpoint"] = ""
    out["ppo_seed"] = sub.get("ppo_seed", pd.Series([""] * len(sub))).astype(str)
    out["predicted_lex_primary_score"] = np.nan
    out["predicted_u2_guarded_score"] = np.nan
    out["final_v03_score"] = pd.to_numeric(sub.get("conservative_reward", sub.get("predicted_reward", pd.Series([np.nan] * len(sub)))), errors="coerce")
    out["partial_training_caveat"] = ""
    out["teacher_metrics_extracted"] = True
    out["teacher_validation_status"] = sub.get("teacher_validation_status", pd.Series(["PASS_TEACHER_FIELDS_EXTRACTED"] * len(sub))).astype(str)
    out["source_metric_file"] = str(V02_METRICS)
    for m in METRICS:
        col = metric_col(sub, m)
        out[m] = pd.to_numeric(sub[col], errors="coerce") if col else np.nan
    return out


def input_audit(stage_r: pd.DataFrame, selected: pd.DataFrame, ref: pd.DataFrame, v01: pd.DataFrame, v02: pd.DataFrame) -> str:
    rows: list[dict[str, Any]] = []
    def add(check: str, passed: bool, severity: str, detail: Any) -> None:
        rows.append({"check": check, "passed": bool(passed), "severity": severity, "detail": str(detail)})
    for label, path in [
        ("stageR_metrics", STAGE_R_METRICS), ("stageR_summary", STAGE_R_SUMMARY), ("stageR_solver_audit", STAGE_R_SOLVER_AUDIT),
        ("stageP_selected", STAGE_P_SELECTED), ("combined552", COMBINED552), ("v01_metrics", V01_METRICS), ("v01_ranking", V01_RANKING),
        ("v02K2_metrics", V02_METRICS), ("v02K2_ranking", V02_RANKING),
    ]:
        add(f"{label}_exists", path.exists(), "FAIL" if "ranking" not in label and "summary" not in label else "WARNING", path)
    add("stageR_row_count_32", len(stage_r) == 32, "FAIL", len(stage_r))
    add("stageR_counts_N24_N40", stage_r["n"].astype(int).value_counts().sort_index().to_dict() == V03_COUNTS, "FAIL", stage_r["n"].astype(int).value_counts().sort_index().to_dict())
    add("stageR_no_N12_N16_N32", set(stage_r["n"].astype(int)) == set(SUPPORTED_N), "FAIL", sorted(stage_r["n"].astype(int).unique()))
    for m in METRICS:
        add(f"stageR_metric_{m}_mapped", metric_col(stage_r, m) is not None, "FAIL", metric_col(stage_r, m))
        add(f"combined_metric_{m}_mapped", metric_col(ref, m) is not None, "FAIL", metric_col(ref, m))
        add(f"v01_metric_{m}_mapped", metric_col(v01, m) is not None, "FAIL", metric_col(v01, m))
        add(f"v02_metric_{m}_mapped", metric_col(v02, m) is not None, "FAIL", metric_col(v02, m))
    add("stageR_extraction_success", stage_r["odb_extraction_status"].astype(str).str.contains("PASS", case=False, na=False).all(), "FAIL", stage_r.get("odb_extraction_status", pd.Series(dtype=str)).value_counts().to_dict())
    add("stageR_teacher_status_success", stage_r["teacher_validation_status"].astype(str).str.contains("PASS", case=False, na=False).all(), "FAIL", stage_r.get("teacher_validation_status", pd.Series(dtype=str)).value_counts().to_dict())
    add("stageR_partial_training_caveat_present_32", "partial_training_caveat" in stage_r.columns and stage_r["partial_training_caveat"].astype(str).str.len().gt(0).all(), "WARNING", "partial_training_caveat")
    fields = stage_r.get("extracted_field_names", pd.Series([""] * len(stage_r))).astype(str)
    add("stageR_required_fields_visible", fields.str.contains("U").all() and fields.str.contains("PEEQ").all() and fields.str.contains("S").all() and fields.str.contains("NT11|NT", regex=True).all(), "WARNING", "U/PEEQ/S/NT")
    add("selected_row_count_32", len(selected) == 32, "FAIL", len(selected))
    add("selected_counts_N24_N40", selected["n"].astype(int).value_counts().sort_index().to_dict() == V03_COUNTS, "FAIL", selected["n"].astype(int).value_counts().sort_index().to_dict())
    add("selected_names_match_stageR", set(selected["strategy_name"].astype(str)) == set(stage_r["handoff_strategy_name"].astype(str)), "FAIL", "strategy sets")
    add("selected_candidate_source_v03", selected["candidate_source"].astype(str).str.contains("PPO_v03_checkpoint_inference").all(), "FAIL", selected.get("candidate_source", pd.Series(dtype=str)).value_counts().to_dict())
    for c in ["predicted_lex_primary_score", "predicted_u2_guarded_score", "final_v03_score"]:
        add(f"selected_{c}_exists", c in selected.columns, "FAIL", c)
    ref_n = ref[ref["n"].astype(int).isin(SUPPORTED_N)]
    add("combined552_row_count_552", len(ref) == 552, "FAIL", len(ref))
    add("combined552_N24_N40_counts", ref_n["n"].astype(int).value_counts().sort_index().to_dict() == REF_COUNTS, "FAIL", ref_n["n"].astype(int).value_counts().sort_index().to_dict())
    add("v01_N24_N40_rows_available", v01[v01["n"].astype(int).isin(SUPPORTED_N)]["n"].astype(int).value_counts().sort_index().to_dict() == {24: 8, 40: 8}, "FAIL", v01[v01["n"].astype(int).isin(SUPPORTED_N)]["n"].astype(int).value_counts().sort_index().to_dict())
    add("v02_N24_N40_rows_available", v02["n"].astype(int).value_counts().sort_index().to_dict() == {24: 16, 40: 16}, "FAIL", v02["n"].astype(int).value_counts().sort_index().to_dict())
    write_csv(AUDIT_CSV, rows)
    fail = sum(1 for r in rows if r["severity"] == "FAIL" and not r["passed"])
    warn = sum(1 for r in rows if r["severity"] == "WARNING" and not r["passed"])
    verdict = "FAIL_STAGES_V03_INPUTS_NOT_READY" if fail else ("WARNING_STAGES_V03_INPUTS_REVIEW" if warn else "PASS_STAGES_V03_INPUTS_READY")
    AUDIT_JSON.write_text(json.dumps({
        "verdict": verdict, "fail_count": fail, "warning_count": warn,
        "stageR_rows": int(len(stage_r)), "stageR_counts_by_N": {str(k): int(v) for k, v in stage_r["n"].astype(int).value_counts().sort_index().to_dict().items()},
        "combined552_N24_N40_counts": {str(k): int(v) for k, v in ref_n["n"].astype(int).value_counts().sort_index().to_dict().items()},
        "v01_N24_N40_counts": {str(k): int(v) for k, v in v01[v01["n"].astype(int).isin(SUPPORTED_N)]["n"].astype(int).value_counts().sort_index().to_dict().items()},
        "v02_N24_N40_counts": {str(k): int(v) for k, v in v02["n"].astype(int).value_counts().sort_index().to_dict().items()},
        "no_Abaqus": True, "no_ODB_opening": True, "no_ODB_extraction": True, "no_solver": True,
    }, indent=2), encoding="utf-8")
    return verdict


def assign_ref_self(ref: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for n in SUPPORTED_N:
        sub = ref[ref["n"] == n].copy()
        for m in METRICS:
            sub[f"ref_rank_{m}"] = sub[m].rank(method="min", ascending=True)
        idx = sub.sort_values(PRIMARY).index
        sub.loc[idx, "ref_lex_rank"] = np.arange(1, len(idx) + 1)
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def annotate_candidates(ref: pd.DataFrame, cand: pd.DataFrame, prefix: str = "v03") -> pd.DataFrame:
    out_rows = []
    for n in SUPPORTED_N:
        refn = ref[ref["n"] == n].copy()
        candn = cand[cand["n"] == n].copy()
        combo = pd.concat([refn, candn], ignore_index=True)
        for m in METRICS:
            combo[f"combined_rank_{m}"] = combo[m].rank(method="min", ascending=True)
            combo[f"combined_percentile_{m}"] = combo[f"combined_rank_{m}"] / len(combo)
            mn, mx = combo[m].min(), combo[m].max()
            combo[f"combined_cost_norm_{m}"] = 0.0 if mx == mn else (combo[m] - mn) / (mx - mn)
        idx = combo[combo["n"] == n].sort_values(PRIMARY).index
        combo.loc[idx, "combined_lex_rank"] = np.arange(1, len(idx) + 1)
        combo.loc[idx, "combined_lex_percentile"] = combo.loc[idx, "combined_lex_rank"] / len(idx)
        idxd = combo[combo["n"] == n].sort_values(METRICS).index
        combo.loc[idxd, "combined_diag_lex_rank"] = np.arange(1, len(idxd) + 1)
        combo.loc[idxd, "combined_diag_lex_percentile"] = combo.loc[idxd, "combined_diag_lex_rank"] / len(idxd)
        cn = combo[combo["dataset_source"] == candn.iloc[0]["dataset_source"]].copy() if len(candn) else candn.copy()
        for m in METRICS:
            vals = refn[m].dropna().to_numpy()
            cn[f"ref_rank_{m}"] = cn[m].apply(lambda x: int(np.sum(vals < float(x)) + 1))
            cn[f"ref_percentile_{m}"] = cn[f"ref_rank_{m}"] / (len(vals) + 1)
            cn[f"ref_best_{m}"] = float(vals.min())
            cn[f"ref_median_{m}"] = float(np.median(vals))
            cn[f"ref_q25_{m}"] = float(np.quantile(vals, 0.25))
            cn[f"ref_q75_{m}"] = float(np.quantile(vals, 0.75))
            cn[f"candidate_beats_ref_best_{m}"] = cn[m] < vals.min()
            cn[f"candidate_better_than_ref_median_{m}"] = cn[m] < np.median(vals)
            cn[f"candidate_top10pct_ref_{m}"] = cn[f"ref_rank_{m}"] <= rank_threshold(len(vals), 0.10)
            cn[f"candidate_top25pct_ref_{m}"] = cn[f"ref_rank_{m}"] <= rank_threshold(len(vals), 0.25)
            cn[f"{prefix}_rank_{m}_within_N"] = cn[m].rank(method="min", ascending=True)
            cn[f"{prefix}_best_{m}_within_N"] = cn[f"{prefix}_rank_{m}_within_N"] == 1
        ref_keys = [tuple(x) for x in refn[PRIMARY].to_numpy()]
        ref_diag = [tuple(x) for x in refn[METRICS].to_numpy()]
        for i, r in cn.iterrows():
            key = tuple(r[m] for m in PRIMARY)
            dkey = tuple(r[m] for m in METRICS)
            cn.loc[i, "ref_lex_rank"] = 1 + sum(k < key for k in ref_keys)
            cn.loc[i, "ref_lex_percentile"] = cn.loc[i, "ref_lex_rank"] / (len(ref_keys) + 1)
            cn.loc[i, "candidate_beats_ref_lex_best"] = key < min(ref_keys)
            cn.loc[i, "candidate_top10pct_ref_lex"] = cn.loc[i, "ref_lex_rank"] <= rank_threshold(len(ref_keys), 0.10)
            cn.loc[i, "candidate_top25pct_ref_lex"] = cn.loc[i, "ref_lex_rank"] <= rank_threshold(len(ref_keys), 0.25)
            cn.loc[i, "ref_diag_lex_rank"] = 1 + sum(k < dkey for k in ref_diag)
            cn.loc[i, "ref_diag_lex_percentile"] = cn.loc[i, "ref_diag_lex_rank"] / (len(ref_diag) + 1)
        order = cn.sort_values(PRIMARY).index
        cn.loc[order, f"{prefix}_lex_rank_within_N"] = np.arange(1, len(order) + 1)
        cn.loc[order, f"{prefix}_best_lex_within_N"] = np.arange(1, len(order) + 1) == 1
        cn["teacher_lex_reward_rank_normalized"] = 1.0 - (cn["ref_lex_rank"] - 1) / refn.shape[0]
        cn["teacher_u2_reward_rank_normalized"] = 1.0 - (cn["ref_rank_u2_range"] - 1) / refn.shape[0]
        out_rows.append(cn)
    return pd.concat(out_rows, ignore_index=True).sort_values(["n", "ref_lex_rank", "strategy_name"]).reset_index(drop=True)


def row_primary_topk(row: pd.Series) -> bool:
    flags = [row.get(f"candidate_top10pct_ref_{m}", False) or row.get(f"candidate_top25pct_ref_{m}", False) for m in PRIMARY]
    flags += [row.get("candidate_top10pct_ref_lex", False), row.get("candidate_top25pct_ref_lex", False)]
    return any(bool(x) for x in flags)


def make_analysis_datasets(ref: pd.DataFrame, v01_all: pd.DataFrame, v02: pd.DataFrame, v03: pd.DataFrame) -> None:
    pd.concat([ref, v03], ignore_index=True).to_csv(ANALYSIS_V03, index=False)
    v01_targeted = v01_all[v01_all["n"].astype(int).isin(SUPPORTED_N)].copy()
    pd.concat([ref, v01_targeted, v02, v03], ignore_index=True).to_csv(ANALYSIS_ALL, index=False)
    pool = pd.concat([v01_all, v02, v03], ignore_index=True)
    pool.to_csv(PPO_POOL, index=False)


def new_records(ranking: pd.DataFrame, ref: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    rows = []
    for n in SUPPORTED_N:
        refn = ref[ref["n"] == n]
        cand = ranking[ranking["n"] == n]
        for m in METRICS:
            best = float(refn[m].min())
            for _, r in cand[cand[m] < best].iterrows():
                rows.append({"n": n, "strategy_name": r["strategy_name"], "metric_or_lex": m, "v03_value": r[m], "prior_combined552_best": best, "improvement_ratio": r[m] / best if best else np.nan, "improvement_percent": (best - r[m]) / best * 100 if best else np.nan, "primary_or_diagnostic": "primary" if m in PRIMARY else "diagnostic", "caveat_note": "Relative to combined552 teacher metrics; v03 training was partial."})
        ref_best_key = min([tuple(x) for x in refn[PRIMARY].to_numpy()])
        for _, r in cand.iterrows():
            key = tuple(r[m] for m in PRIMARY)
            if key < ref_best_key:
                rows.append({"n": n, "strategy_name": r["strategy_name"], "metric_or_lex": "primary_lex_u2_peeq_surfaceT", "v03_value": str(key), "prior_combined552_best": str(ref_best_key), "improvement_ratio": "", "improvement_percent": "", "primary_or_diagnostic": "primary", "caveat_note": "Relative to combined552 teacher metrics; v03 training was partial."})
    cols = ["n", "strategy_name", "metric_or_lex", "v03_value", "prior_combined552_best", "improvement_ratio", "improvement_percent", "primary_or_diagnostic", "caveat_note"]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(NEW_RECORDS, index=False)
    return df, {"total": len(df), "primary": int((df["primary_or_diagnostic"] == "primary").sum()) if len(df) else 0, "diagnostic": int((df["primary_or_diagnostic"] == "diagnostic").sum()) if len(df) else 0}


def topk_tables(ranking: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for _, r in ranking.iterrows():
        reasons = []
        for m in PRIMARY:
            if r[f"candidate_top10pct_ref_{m}"]:
                reasons.append(f"top10_{m}")
            if r[f"candidate_top25pct_ref_{m}"]:
                reasons.append(f"top25_{m}")
        if r["candidate_top10pct_ref_lex"]:
            reasons.append("top10_primary_lex")
        if r["candidate_top25pct_ref_lex"]:
            reasons.append("top25_primary_lex")
        if r["candidate_top10pct_ref_mises_max"]:
            reasons.append("diagnostic_top10_mises")
        if r["candidate_top25pct_ref_mises_max"]:
            reasons.append("diagnostic_top25_mises")
        if reasons:
            rec = r.to_dict()
            rec["topk_reasons"] = ";".join(reasons)
            rec["primary_topk"] = any("mises" not in x for x in reasons)
            rows.append(rec)
    topk = pd.DataFrame(rows)
    topk.to_csv(TOPK, index=False)
    summary = []
    for n in SUPPORTED_N:
        sub = ranking[ranking["n"] == n]
        summary.append({
            "n": n, "v03_count": len(sub),
            "top10pct_U2_count": int(sub["candidate_top10pct_ref_u2_range"].sum()),
            "top25pct_U2_count": int(sub["candidate_top25pct_ref_u2_range"].sum()),
            "top10pct_PEEQ_count": int(sub["candidate_top10pct_ref_peeq_max"].sum()),
            "top25pct_PEEQ_count": int(sub["candidate_top25pct_ref_peeq_max"].sum()),
            "top10pct_SurfaceT_count": int(sub["candidate_top10pct_ref_surface_t_proxy"].sum()),
            "top25pct_SurfaceT_count": int(sub["candidate_top25pct_ref_surface_t_proxy"].sum()),
            "top10pct_lex_count": int(sub["candidate_top10pct_ref_lex"].sum()),
            "top25pct_lex_count": int(sub["candidate_top25pct_ref_lex"].sum()),
            "diagnostic_Mises_topk_count": int((sub["candidate_top10pct_ref_mises_max"] | sub["candidate_top25pct_ref_mises_max"]).sum()),
            "total_unique_primary_topk_candidates": int(sub.apply(row_primary_topk, axis=1).sum()),
            "total_unique_any_topk_candidates": int(len(topk[topk["n"] == n])) if len(topk) else 0,
        })
    summ = pd.DataFrame(summary)
    summ.to_csv(TOPK_SUMMARY, index=False)
    return topk, summ


def annotate_prior(ref: pd.DataFrame, prior: pd.DataFrame, prefix: str) -> pd.DataFrame:
    temp = prior[prior["n"].astype(int).isin(SUPPORTED_N)].copy()
    temp["dataset_source"] = "ppo_v03_batch32"
    return annotate_candidates(ref, temp, prefix=prefix)


def compare_prior(ref: pd.DataFrame, v01: pd.DataFrame, v02: pd.DataFrame, v03_rank: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    v01r = annotate_prior(ref, v01, "v01")
    v02r = annotate_prior(ref, v02, "v02K2")
    rows, boot_rows = [], []
    rng = np.random.default_rng(RNG_SEED)
    known = {("v01", 24, "lex"): 134, ("v01", 40, "lex"): 147, ("v01", 24, "topk"): 3, ("v01", 40, "topk"): 0, ("v02", 24, "lex"): 114, ("v02", 40, "lex"): 147, ("v02", 24, "topk"): 0, ("v02", 40, "topk"): 16}
    for n in SUPPORTED_N:
        a = v01r[v01r["n"] == n]
        b = v02r[v02r["n"] == n]
        c = v03_rank[v03_rank["n"] == n]
        top_a, top_b, top_c = int(a.apply(row_primary_topk, axis=1).sum()), int(b.apply(row_primary_topk, axis=1).sum()), int(c.apply(row_primary_topk, axis=1).sum())
        row = {
            "n": n, "v01_candidate_count": len(a), "v02K2_candidate_count": len(b), "v03_candidate_count": len(c),
            "v01_best_lex_rank": int(a["ref_lex_rank"].min()), "v02K2_best_lex_rank": int(b["ref_lex_rank"].min()), "v03_best_lex_rank": int(c["ref_lex_rank"].min()),
            "known_v01_best_lex_rank": known[("v01", n, "lex")], "known_v02K2_best_lex_rank": known[("v02", n, "lex")],
            "v01_topk_count": top_a, "v02K2_topk_count": top_b, "v03_topk_count": top_c,
            "known_v01_topk_count": known[("v01", n, "topk")], "known_v02K2_topk_count": known[("v02", n, "topk")],
            "v03_improves_v01_best_lex_rank": bool(c["ref_lex_rank"].min() < a["ref_lex_rank"].min()),
            "v03_improves_v02K2_best_lex_rank": bool(c["ref_lex_rank"].min() < b["ref_lex_rank"].min()),
            "v03_improves_v01_topk_count": bool(top_c > top_a),
            "v03_improves_v02K2_topk_count": bool(top_c > top_b),
        }
        for m in METRICS:
            for name, df in [("v01", a), ("v02K2", b), ("v03", c)]:
                row[f"{name}_best_{m}"] = float(df[m].min())
                row[f"{name}_median_{m}"] = float(df[m].median())
        if row["v03_improves_v02K2_best_lex_rank"] or row["v03_improves_v02K2_topk_count"]:
            row["interpretation"] = "v03 improves prior PPO on at least one primary targeted criterion"
        elif top_c > 0:
            row["interpretation"] = "v03 is teacher-validated but not stronger than prior PPO"
        else:
            row["interpretation"] = "v03 is weak under primary targeted criteria"
        rows.append(row)
        c_lex = c["ref_lex_rank"].to_numpy()
        c_top = c.apply(row_primary_topk, axis=1).to_numpy()
        best8, top8 = [], []
        for _ in range(BOOTSTRAP_TRIALS):
            idx = rng.choice(len(c), size=8, replace=False)
            best8.append(np.min(c_lex[idx]))
            top8.append(np.sum(c_top[idx]))
        boot_rows.append({
            "n": n, "trials": BOOTSTRAP_TRIALS,
            "v01_best_lex_rank": int(a["ref_lex_rank"].min()), "v02K2_best_lex_rank": int(b["ref_lex_rank"].min()), "v03_best_lex_rank": int(c["ref_lex_rank"].min()),
            "v03_equal8_best_lex_mean": float(np.mean(best8)), "v03_equal8_best_lex_median": float(np.median(best8)),
            "prob_v03_equal8_beats_v01_best_lex": float(np.mean(np.array(best8) < a["ref_lex_rank"].min())),
            "prob_v03_equal8_beats_v02K2_best_lex": float(np.mean(np.array(best8) < b["ref_lex_rank"].min())),
            "v01_topk_count": top_a, "v02K2_topk_count": top_b, "v03_topk_count": top_c,
            "v03_equal8_topk_mean": float(np.mean(top8)), "v03_equal8_topk_median": float(np.median(top8)),
            "prob_v03_equal8_beats_v01_topk": float(np.mean(np.array(top8) > top_a)),
            "prob_v03_equal8_beats_v02K2_topk": float(np.mean(np.array(top8) > top_b)),
            "v03_direct16_beats_v02K2_best_lex": bool(c["ref_lex_rank"].min() < b["ref_lex_rank"].min()),
            "v03_direct16_beats_v02K2_topk": bool(top_c > top_b),
        })
    comp = pd.DataFrame(rows)
    boot = pd.DataFrame(boot_rows)
    comp.to_csv(V03_VS_PRIOR, index=False)
    boot.to_csv(PRIOR_BOOT, index=False)
    return comp, boot


def bootstrap_reference(ref: pd.DataFrame, v03_rank: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RNG_SEED + 1)
    ref_ranked = assign_ref_self(ref)
    rows = []
    obs_global = int(v03_rank.apply(row_primary_topk, axis=1).sum())
    global_vals = []
    for n in SUPPORTED_N:
        refn = ref_ranked[ref_ranked["n"] == n].reset_index(drop=True)
        v = v03_rank[v03_rank["n"] == n]
        observed = {
            "top10pct_lex_count": int(v["candidate_top10pct_ref_lex"].sum()),
            "top25pct_lex_count": int(v["candidate_top25pct_ref_lex"].sum()),
            "top10pct_U2_count": int(v["candidate_top10pct_ref_u2_range"].sum()),
            "top25pct_U2_count": int(v["candidate_top25pct_ref_u2_range"].sum()),
            "top10pct_PEEQ_count": int(v["candidate_top10pct_ref_peeq_max"].sum()),
            "top25pct_PEEQ_count": int(v["candidate_top25pct_ref_peeq_max"].sum()),
            "top10pct_SurfaceT_count": int(v["candidate_top10pct_ref_surface_t_proxy"].sum()),
            "top25pct_SurfaceT_count": int(v["candidate_top25pct_ref_surface_t_proxy"].sum()),
            "best_lex_rank": float(v["ref_lex_rank"].min()),
            "median_lex_rank": float(v["ref_lex_rank"].median()),
        }
        vals = {k: [] for k in observed}
        for _ in range(BOOTSTRAP_TRIALS):
            s = refn.iloc[rng.choice(len(refn), size=16, replace=False)]
            vals["top10pct_lex_count"].append(int((s["ref_lex_rank"] <= rank_threshold(len(refn), 0.10)).sum()))
            vals["top25pct_lex_count"].append(int((s["ref_lex_rank"] <= rank_threshold(len(refn), 0.25)).sum()))
            vals["top10pct_U2_count"].append(int((s["ref_rank_u2_range"] <= rank_threshold(len(refn), 0.10)).sum()))
            vals["top25pct_U2_count"].append(int((s["ref_rank_u2_range"] <= rank_threshold(len(refn), 0.25)).sum()))
            vals["top10pct_PEEQ_count"].append(int((s["ref_rank_peeq_max"] <= rank_threshold(len(refn), 0.10)).sum()))
            vals["top25pct_PEEQ_count"].append(int((s["ref_rank_peeq_max"] <= rank_threshold(len(refn), 0.25)).sum()))
            vals["top10pct_SurfaceT_count"].append(int((s["ref_rank_surface_t_proxy"] <= rank_threshold(len(refn), 0.10)).sum()))
            vals["top25pct_SurfaceT_count"].append(int((s["ref_rank_surface_t_proxy"] <= rank_threshold(len(refn), 0.25)).sum()))
            vals["best_lex_rank"].append(float(s["ref_lex_rank"].min()))
            vals["median_lex_rank"].append(float(s["ref_lex_rank"].median()))
        for k, arr0 in vals.items():
            arr = np.asarray(arr0)
            if "rank" in k:
                p = float(np.mean(arr <= observed[k]))
                interp = "enriched" if observed[k] < np.quantile(arr, 0.05) else ("weak" if observed[k] > np.quantile(arr, 0.95) else "comparable")
            else:
                p = float(np.mean(arr >= observed[k]))
                interp = "enriched" if observed[k] > np.quantile(arr, 0.95) else ("weak" if observed[k] < np.quantile(arr, 0.05) else "comparable")
            rows.append({"n": n, "metric": k, "observed": observed[k], "bootstrap_mean": float(arr.mean()), "bootstrap_median": float(np.median(arr)), "q05": float(np.quantile(arr, 0.05)), "q95": float(np.quantile(arr, 0.95)), "empirical_p_value_greater_equal": p, "interpretation": interp})
    for _ in range(BOOTSTRAP_TRIALS):
        total = 0
        for n in SUPPORTED_N:
            refn = ref_ranked[ref_ranked["n"] == n].reset_index(drop=True)
            s = refn.iloc[rng.choice(len(refn), size=16, replace=False)]
            is_top = (
                (s["ref_rank_u2_range"] <= rank_threshold(len(refn), 0.25))
                | (s["ref_rank_peeq_max"] <= rank_threshold(len(refn), 0.25))
                | (s["ref_rank_surface_t_proxy"] <= rank_threshold(len(refn), 0.25))
                | (s["ref_lex_rank"] <= rank_threshold(len(refn), 0.25))
            )
            total += int(is_top.sum())
        global_vals.append(total)
    g = np.asarray(global_vals)
    global_df = pd.DataFrame([{"metric": "total_unique_primary_top25_or_lex_top25_count", "observed": obs_global, "bootstrap_mean": float(g.mean()), "bootstrap_median": float(np.median(g)), "q05": float(np.quantile(g, 0.05)), "q95": float(np.quantile(g, 0.95)), "empirical_p_value_greater_equal": float(np.mean(g >= obs_global)), "interpretation": "enriched" if obs_global > np.quantile(g, 0.95) else ("weak" if obs_global < np.quantile(g, 0.05) else "comparable"), "note": "Equal-budget bootstrap against existing teacher-labelled reference distribution, not the full scan-order universe."}])
    byn = pd.DataFrame(rows)
    byn.to_csv(BOOT_BY_N, index=False)
    global_df.to_csv(BOOT_GLOBAL, index=False)
    return byn, global_df


def baseline_tables(raw_ref: pd.DataFrame, ref: pd.DataFrame, v03_rank: pd.DataFrame, pool: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cols = [c for c in ["strategy_name", "candidate_family", "family", "source", "generation_tag", "batch", "strategy_type", "candidate_source", "selection_tag", "generation_method"] if c in raw_ref.columns]
    raw = raw_ref[raw_ref["n"].astype(int).isin(SUPPORTED_N)].copy()
    raw["_label_text"] = ""
    for c in cols:
        raw["_label_text"] += " " + raw[c].astype(str).str.lower()
    pats = {"raster": r"raster", "odd_even": r"odd[_ -]?even|even[_ -]?odd", "edge_in": r"edge[_ -]?in", "center_out": r"center[_ -]?out", "center_edge": r"center[_ -]?edge", "method_c": r"method[_ -]?c", "regular_jump": r"regular[_ -]?jump|jump", "engineering": r"engineering", "random": r"random", "heuristic": r"heuristic|baseline"}
    inv, comp, pool_comp = [], [], []
    for fam, pat in pats.items():
        fam_raw = raw[raw["_label_text"].str.contains(pat, regex=True, na=False)]
        for n in SUPPORTED_N:
            count = int((fam_raw["n"].astype(int) == n).sum())
            inv.append({"family": fam, "n": n, "count": count, "source_columns_used": ";".join(cols), "reliability": "FOUND" if count else "NOT_FOUND"})
            if not count:
                continue
            fam_df = canonical_ref(fam_raw[fam_raw["n"].astype(int) == n])
            v = v03_rank[v03_rank["n"] == n]
            p = pool[(pool["n"].astype(int) == n)]
            row = {"family": fam, "n": n, "family_count": count, "v03_count": len(v)}
            prow = {"family": fam, "n": n, "family_count": count, "cumulative_ppo_count": len(p)}
            for m in METRICS:
                row[f"family_best_{m}"] = float(fam_df[m].min()); row[f"family_median_{m}"] = float(fam_df[m].median())
                row[f"v03_best_{m}"] = float(v[m].min()); row[f"v03_median_{m}"] = float(v[m].median())
                row[f"v03_best_beats_family_best_{m}"] = bool(v[m].min() < fam_df[m].min())
                row[f"v03_median_beats_family_median_{m}"] = bool(v[m].median() < fam_df[m].median())
                prow[f"family_best_{m}"] = float(fam_df[m].min()); prow[f"cumulative_ppo_best_{m}"] = float(p[m].min())
                prow[f"cumulative_ppo_best_beats_family_best_{m}"] = bool(p[m].min() < fam_df[m].min())
            comp.append(row); pool_comp.append(prow)
    inv_df, comp_df, pool_df = pd.DataFrame(inv), pd.DataFrame(comp), pd.DataFrame(pool_comp)
    if comp_df.empty:
        comp_df = pd.DataFrame([{"status": "NOT_RELIABLE", "detail": "No explicit baseline labels found."}])
    inv_df.to_csv(BASELINE_INV, index=False); comp_df.to_csv(BASELINE_V03, index=False); pool_df.to_csv(BASELINE_POOL, index=False)
    return inv_df, comp_df, pool_df


def corr_pair(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    tmp = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(tmp) < 3 or tmp["x"].nunique() < 2 or tmp["y"].nunique() < 2:
        return np.nan, np.nan
    return float(tmp["x"].corr(tmp["y"], method="spearman")), float(tmp["x"].corr(tmp["y"], method="pearson"))


def alignment(ranking: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = ranking.copy()
    out["teacher_lex_reward"] = out["teacher_lex_reward_rank_normalized"]
    out["teacher_u2_lex_reward"] = 0.7 * out["teacher_u2_reward_rank_normalized"] + 0.3 * out["teacher_lex_reward_rank_normalized"]
    high = out["final_v03_score"].rank(method="min", ascending=False) <= max(1, math.ceil(len(out) * 0.25))
    poor = out["ref_lex_rank"] > out["n"].map(lambda n: rank_threshold(REF_COUNTS[int(n)], 0.50))
    strong = out.apply(row_primary_topk, axis=1)
    surface_only_teacher = (out["ref_rank_surface_t_proxy"] <= out["n"].map(lambda n: rank_threshold(REF_COUNTS[int(n)], 0.25))) & (out["ref_rank_u2_range"] > out["n"].map(lambda n: rank_threshold(REF_COUNTS[int(n)], 0.50))) & (out["ref_lex_rank"] > out["n"].map(lambda n: rank_threshold(REF_COUNTS[int(n)], 0.50)))
    out["false_positive_high_score_poor_teacher"] = high & poor
    out["true_positive_high_score_strong_teacher"] = high & strong
    out["teacher_surfaceT_only_false_positive"] = surface_only_teacher
    out.to_csv(ALIGN_CSV, index=False)
    lx_s, lx_p = corr_pair(out["predicted_lex_primary_score"], out["teacher_lex_reward"])
    u2_s, u2_p = corr_pair(out["predicted_u2_guarded_score"], out["teacher_u2_lex_reward"])
    f_s, f_p = corr_pair(out["final_v03_score"], out["teacher_lex_reward"])
    byn = {}
    for n in SUPPORTED_N:
        sub = out[out["n"] == n]
        byn[str(n)] = {
            "lex_score_spearman": corr_pair(sub["predicted_lex_primary_score"], sub["teacher_lex_reward"])[0],
            "lex_score_pearson": corr_pair(sub["predicted_lex_primary_score"], sub["teacher_lex_reward"])[1],
            "u2_guarded_spearman": corr_pair(sub["predicted_u2_guarded_score"], sub["teacher_u2_lex_reward"])[0],
            "u2_guarded_pearson": corr_pair(sub["predicted_u2_guarded_score"], sub["teacher_u2_lex_reward"])[1],
            "final_score_spearman": corr_pair(sub["final_v03_score"], sub["teacher_lex_reward"])[0],
            "final_score_pearson": corr_pair(sub["final_v03_score"], sub["teacher_lex_reward"])[1],
            "false_positive_count": int(sub["false_positive_high_score_poor_teacher"].sum()),
            "true_positive_count": int(sub["true_positive_high_score_strong_teacher"].sum()),
            "teacher_surfaceT_only_false_positive_count": int(sub["teacher_surfaceT_only_false_positive"].sum()),
        }
    summary = {"overall_predicted_lex_spearman": lx_s, "overall_predicted_lex_pearson": lx_p, "overall_u2_guarded_spearman": u2_s, "overall_u2_guarded_pearson": u2_p, "overall_final_v03_score_spearman": f_s, "overall_final_v03_score_pearson": f_p, "false_positive_count": int(out["false_positive_high_score_poor_teacher"].sum()), "true_positive_count": int(out["true_positive_high_score_strong_teacher"].sum()), "teacher_surfaceT_only_false_positive_count": int(out["teacher_surfaceT_only_false_positive"].sum()), "by_N": byn}
    ALIGN_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out, summary


def summaries(ranking: pd.DataFrame, ref: pd.DataFrame, new: pd.DataFrame, topk_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, best_rows = [], []
    for n in SUPPORTED_N:
        sub, refn = ranking[ranking["n"] == n], ref[ref["n"] == n]
        tk = topk_summary[topk_summary["n"] == n].iloc[0].to_dict()
        row = {"n": n, "v03_count": len(sub), "reference_count": len(refn), "best_v03_lex_candidate": sub.sort_values("ref_lex_rank").iloc[0]["strategy_name"], "best_v03_ref_lex_rank": int(sub["ref_lex_rank"].min()), "new_record_count": int((new["n"] == n).sum()) if len(new) else 0, "primary_new_record_count": int(((new["n"] == n) & (new["primary_or_diagnostic"] == "primary")).sum()) if len(new) else 0, "total_unique_primary_topk_candidates": int(tk["total_unique_primary_topk_candidates"]), "total_unique_any_topk_candidates": int(tk["total_unique_any_topk_candidates"])}
        for m in METRICS:
            row[f"best_v03_{m}"] = float(sub[m].min()); row[f"best_reference_{m}"] = float(refn[m].min()); row[f"v03_best_over_reference_best_ratio_{m}"] = float(sub[m].min() / refn[m].min()); row[f"median_v03_{m}"] = float(sub[m].median()); row[f"median_reference_{m}"] = float(refn[m].median()); row[f"v03_top25pct_count_{m}"] = int(sub[f"candidate_top25pct_ref_{m}"].sum())
        rows.append(row)
        for label, sort_key in [("best_U2", "u2_range"), ("best_PEEQ", "peeq_max"), ("best_SurfaceT", "surface_t_proxy"), ("best_primary_lex", "ref_lex_rank"), ("best_diagnostic_Mises", "mises_max")]:
            b = sub.sort_values(sort_key).iloc[0]
            best_rows.append({"n": n, "best_type": label, "strategy_name": b["strategy_name"], "value": b[sort_key], "ref_rank_u2": b["ref_rank_u2_range"], "ref_rank_peeq": b["ref_rank_peeq_max"], "ref_rank_surfaceT": b["ref_rank_surface_t_proxy"], "ref_rank_mises": b["ref_rank_mises_max"], "ref_lex_rank": b["ref_lex_rank"], "is_new_record": bool(b.get(f"candidate_beats_ref_best_{sort_key}", False)) if sort_key in METRICS else bool(b.get("candidate_beats_ref_lex_best", False)), "is_top25pct": bool(b.get(f"candidate_top25pct_ref_{sort_key}", False)) if sort_key in METRICS else bool(b.get("candidate_top25pct_ref_lex", False))})
    byn = pd.DataFrame(rows); best = pd.DataFrame(best_rows)
    glob = pd.DataFrame([{"v03_total_count": len(ranking), "new_record_count_total": len(new), "primary_new_record_count_total": int((new["primary_or_diagnostic"] == "primary").sum()) if len(new) else 0, "total_unique_primary_topk_candidates": int(topk_summary["total_unique_primary_topk_candidates"].sum()), "total_unique_any_topk_candidates": int(topk_summary["total_unique_any_topk_candidates"].sum()), "best_overall_ref_lex_rank": int(ranking["ref_lex_rank"].min()), "best_overall_ref_lex_candidate": ranking.sort_values("ref_lex_rank").iloc[0]["strategy_name"]}])
    byn.to_csv(SUMMARY_BY_N, index=False); glob.to_csv(GLOBAL_SUMMARY, index=False); best.to_csv(BEST_BY_N, index=False)
    return byn, glob, best


def cumulative_pool_summary(ref: pd.DataFrame, pool_all: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pool_target = pool_all[pool_all["n"].astype(int).isin(SUPPORTED_N)].copy()
    pool_rank = annotate_candidates(ref, pool_target.assign(dataset_source="ppo_v03_batch32"), prefix="ppo_pool")
    # Restore ppo version columns lost only if present via values kept in rows.
    rows = []
    for n in sorted(pool_all["n"].astype(int).unique()):
        sub = pool_all[pool_all["n"].astype(int) == n]
        row = {"n": int(n), "total_ppo_count": int(len(sub)), "count_by_ppo_version": json.dumps({k: int(v) for k, v in sub["ppo_version"].value_counts().to_dict().items()})}
        if n in SUPPORTED_N:
            rankn = pool_rank[pool_rank["n"] == n]
            for label, key in [("U2", "u2_range"), ("PEEQ", "peeq_max"), ("SurfaceT", "surface_t_proxy"), ("primary_lex", "ref_lex_rank"), ("diagnostic_Mises", "mises_max")]:
                b = rankn.sort_values(key).iloc[0]
                row[f"best_PPO_{label}_candidate"] = b["strategy_name"]
                row[f"best_PPO_{label}_value_or_rank"] = b[key]
            row["primary_topk_count"] = int(rankn.apply(row_primary_topk, axis=1).sum())
            row["new_record_count"] = int(sum(rankn[f"candidate_beats_ref_best_{m}"].sum() for m in PRIMARY) + rankn["candidate_beats_ref_lex_best"].sum())
            row["comparison_against_combined552_best"] = "no combined552 best beat" if row["new_record_count"] == 0 else "PPO pool contains combined552-best improvements"
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(POOL_SUMMARY, index=False)
    progress = pd.DataFrame([
        {"stage": "v01", "teacher_validated_count": 32},
        {"stage": "v02K2", "teacher_validated_count": 32},
        {"stage": "v03", "teacher_validated_count": 32},
        {"stage": "current_total", "teacher_validated_count": 96},
        {"stage": "target_total", "teacher_validated_count": 320},
        {"stage": "remaining", "teacher_validated_count": 224},
    ])
    progress.to_csv(POOL_PROGRESS, index=False)
    return summary, progress


def make_plots(ref: pd.DataFrame, rank: pd.DataFrame, comp: pd.DataFrame, boot_global: pd.DataFrame, align_df: pd.DataFrame, pool_progress: pd.DataFrame) -> list[str]:
    paths = []
    for m in METRICS:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, n in zip(axes, SUPPORTED_N):
            ax.boxplot([ref[ref["n"] == n][m].dropna(), rank[rank["n"] == n][m].dropna()], labels=["combined552", "v03"])
            ax.set_title(f"N{n}")
            ax.set_ylabel(m)
        fig.tight_layout(); p = PLOTS / f"v03_vs_combined552_{m}_by_N.png"; fig.savefig(p, dpi=160); plt.close(fig); paths.append(str(p))
    fig, ax = plt.subplots(figsize=(6, 4)); ax.scatter(rank["n"], rank["ref_lex_percentile"]); ax.set_xlabel("N"); ax.set_ylabel("ref lex percentile"); p = PLOTS / "v03_lex_rank_percentile_by_N.png"; fig.savefig(p, dpi=160); plt.close(fig); paths.append(str(p))
    fig, ax = plt.subplots(figsize=(7, 4)); x = np.arange(len(comp)); ax.bar(x - .25, comp["v01_best_lex_rank"], .25, label="v01"); ax.bar(x, comp["v02K2_best_lex_rank"], .25, label="v02K2"); ax.bar(x + .25, comp["v03_best_lex_rank"], .25, label="v03"); ax.set_xticks(x, [f"N{n}" for n in comp["n"]]); ax.invert_yaxis(); ax.legend(); p = PLOTS / "v03_vs_v01_v02K2_best_lex_rank.png"; fig.savefig(p, dpi=160); plt.close(fig); paths.append(str(p))
    fig, ax = plt.subplots(figsize=(6, 4)); ax.bar(["v03", "bootstrap mean"], [boot_global.iloc[0]["observed"], boot_global.iloc[0]["bootstrap_mean"]]); p = PLOTS / "v03_vs_bootstrap_random_reference_topk_count.png"; fig.savefig(p, dpi=160); plt.close(fig); paths.append(str(p))
    fig, ax = plt.subplots(figsize=(6, 4)); ax.scatter(align_df["final_v03_score"], align_df["teacher_lex_reward"], label="final"); ax.scatter(align_df["predicted_lex_primary_score"], align_df["teacher_lex_reward"], marker="x", label="lex pred"); ax.legend(); ax.set_xlabel("predicted score"); ax.set_ylabel("teacher lex reward"); p = PLOTS / "v03_predicted_score_vs_teacher_reward.png"; fig.savefig(p, dpi=160); plt.close(fig); paths.append(str(p))
    fig, ax = plt.subplots(figsize=(7, 4)); ax.bar(pool_progress["stage"], pool_progress["teacher_validated_count"]); ax.tick_params(axis="x", rotation=30); p = PLOTS / "cumulative_ppo_pool_progress_to_320.png"; fig.tight_layout(); fig.savefig(p, dpi=160); plt.close(fig); paths.append(str(p))
    fig, ax = plt.subplots(figsize=(7, 4)); comp[["n", "v01_topk_count", "v02K2_topk_count", "v03_topk_count"]].set_index("n").plot(kind="bar", ax=ax); p = PLOTS / "cumulative_ppo_pool_vs_baseline_family_summary.png"; fig.tight_layout(); fig.savefig(p, dpi=160); plt.close(fig); paths.append(str(p))
    return paths


def decide_verdict(new_counts: dict[str, int], comp: pd.DataFrame, topk_summary: pd.DataFrame) -> str:
    if new_counts["primary"] > 0:
        return "PASS_STAGES_V03_TEACHER_VALIDATED_WITH_NEW_RECORDS"
    if comp["v03_improves_v02K2_best_lex_rank"].any() or comp["v03_improves_v02K2_topk_count"].any() or comp["v03_improves_v01_best_lex_rank"].any() or comp["v03_improves_v01_topk_count"].any():
        return "PASS_STAGES_V03_TEACHER_VALIDATED_AND_IMPROVES_PRIOR_PPO"
    if int(topk_summary["total_unique_primary_topk_candidates"].sum()) > 0:
        return "PASS_STAGES_V03_TEACHER_VALIDATED_AND_COMPETITIVE"
    return "WARNING_STAGES_V03_TEACHER_VALIDATED_BUT_WEAK"


def write_report(audit: str, verdict: str, byn: pd.DataFrame, new_counts: dict[str, int], topk_summary: pd.DataFrame, comp: pd.DataFrame, prior_boot: pd.DataFrame, boot_global: pd.DataFrame, baseline_v03: pd.DataFrame, align_summary: dict[str, Any], pool_summary: pd.DataFrame, progress: pd.DataFrame, best: pd.DataFrame) -> None:
    best_lex = best[best["best_type"] == "best_primary_lex"][["n", "strategy_name", "ref_lex_rank", "is_top25pct"]].to_dict("records")
    REPORT.write_text(f"""# PPO v03 Stage S Teacher-Metric Ranking Report

## Purpose
Rank PPO v03 teacher metrics against native combined552, PPO v01, PPO v02K2, conventional baseline families, and the cumulative PPO teacher-validated pool.

## Inputs
- Stage R metrics: `{STAGE_R_METRICS}`
- v03 selected batch: `{STAGE_P_SELECTED}`
- combined552: `{COMBINED552}`
- v01 metrics: `{V01_METRICS}`
- v02K2 metrics: `{V02_METRICS}`

## Stage R Extraction Status
Stage R extracted 32/32 PPO v03 teacher-metric rows: N24=16 and N40=16. No failed cases are included.

## v03 Partial-Training Caveat
The Stage R rows preserve the caveat that N24 used a 100000-step interrupted checkpoint and N40 used 61440 timesteps. This limits claim strength.

## Input Integrity Verdict
`{audit}`

## Analysis Datasets
- `{ANALYSIS_V03}`
- `{ANALYSIS_ALL}`
- `{PPO_POOL}`

## v03 Ranking Against Native combined552
{md_table(byn.to_dict("records"), ["n", "v03_count", "best_v03_lex_candidate", "best_v03_ref_lex_rank", "total_unique_primary_topk_candidates", "new_record_count"])}

## New-Record Audit
New-record rows: `{new_counts["total"]}`; primary new-record rows: `{new_counts["primary"]}`; diagnostic rows: `{new_counts["diagnostic"]}`.

Table: `{NEW_RECORDS}`

## Top-k Competitiveness Audit
{md_table(topk_summary.to_dict("records"), ["n", "v03_count", "top10pct_U2_count", "top25pct_U2_count", "top10pct_lex_count", "top25pct_lex_count", "total_unique_primary_topk_candidates", "diagnostic_Mises_topk_count"])}

## v03 vs v01/v02K2
{md_table(comp.to_dict("records"), ["n", "v01_best_lex_rank", "v02K2_best_lex_rank", "v03_best_lex_rank", "v01_topk_count", "v02K2_topk_count", "v03_topk_count", "interpretation"])}

## Equal-Budget v03-vs-Prior PPO Bootstrap
{md_table(prior_boot.to_dict("records"), ["n", "prob_v03_equal8_beats_v01_best_lex", "prob_v03_equal8_beats_v02K2_best_lex", "prob_v03_equal8_beats_v01_topk", "prob_v03_equal8_beats_v02K2_topk", "v03_direct16_beats_v02K2_best_lex", "v03_direct16_beats_v02K2_topk"])}

## v03 vs Random-Reference Bootstrap
{md_table(boot_global.to_dict("records"), ["metric", "observed", "bootstrap_mean", "q05", "q95", "empirical_p_value_greater_equal", "interpretation"])}

This bootstrap samples the existing teacher-labelled reference distribution, not the full scan-order universe.

## Conventional Baseline-Family Comparison
Inventory: `{BASELINE_INV}`. v03 comparison: `{BASELINE_V03}`. Cumulative PPO comparison: `{BASELINE_POOL}`.

Preview:
{md_table(baseline_v03.head(8).to_dict("records"), list(baseline_v03.head(8).columns))}

## Surrogate-to-Teacher Alignment
- Final v03 score Spearman: `{align_summary["overall_final_v03_score_spearman"]:.4f}`
- Final v03 score Pearson: `{align_summary["overall_final_v03_score_pearson"]:.4f}`
- Lex-primary score Spearman: `{align_summary["overall_predicted_lex_spearman"]:.4f}`
- U2-guarded score Spearman: `{align_summary["overall_u2_guarded_spearman"]:.4f}`
- False positives: `{align_summary["false_positive_count"]}`
- True positives: `{align_summary["true_positive_count"]}`
- Teacher SurfaceT-only false positives: `{align_summary["teacher_surfaceT_only_false_positive_count"]}`

## Best v03 Candidates By N
{md_table(best_lex, ["n", "strategy_name", "ref_lex_rank", "is_top25pct"])}

## Cumulative PPO Pool 96-Case Summary
{md_table(pool_summary.to_dict("records"), ["n", "total_ppo_count", "count_by_ppo_version", "primary_topk_count", "new_record_count", "comparison_against_combined552_best"])}

## Progress Toward 320-Case PPO Target
{md_table(progress.to_dict("records"), ["stage", "teacher_validated_count"])}

## Claim Implications
Use v03 claims only where supported by these tables. The cumulative PPO teacher-validated pool now contains 96 cases, but v03 retains the partial-training caveat.

## Limitations
- Stage S is ranking/analysis only.
- Bootstrap is against combined552, an active-learning-enriched reference distribution, not the universe of scan orders.
- Partial v03 training limits claims about policy convergence.

## Verdict
`{verdict}`
""", encoding="utf-8")
    CLAIM_BOUNDARY.write_text(f"""# PPO v03 Stage S Claim Boundary

## Safe Only If Supported
- v03 candidates were teacher-metric extracted 32/32.
- v03 improves over v01/v02K2 only where `{V03_VS_PRIOR}` shows improvement.
- v03 achieves top-k competitiveness only where `{TOPK_SUMMARY}` shows it.
- v03 sets new records only if `{NEW_RECORDS}` contains primary rows.
- The cumulative PPO teacher-validated pool now contains 96 cases.

## Unsafe Unless Proven
- v03 outperformed combined552 best.
- v03 solved N24/N40.
- v03 dominated prior PPO.
- v03 was fully trained without caveat.
- v03 was online Abaqus PPO.
- v03 is experimentally validated.

## Verdict
`{verdict}`
""", encoding="utf-8")


def write_manifest(verdict: str, plots: list[str]) -> None:
    MANIFEST.write_text(json.dumps({
        "branch": git_branch(), "timestamp": datetime.now(timezone.utc).isoformat(),
        "Stage R metrics input": str(STAGE_R_METRICS), "combined552 reference input": str(COMBINED552),
        "v01 metrics input": str(V01_METRICS), "v02K2 metrics input": str(V02_METRICS), "v03 selected batch input": str(STAGE_P_SELECTED),
        "analysis_dataset_paths": [str(ANALYSIS_V03), str(ANALYSIS_ALL), str(PPO_POOL)],
        "ranking_table_paths": [str(FULL_RANK), str(NEW_RECORDS), str(TOPK), str(BEST_BY_N)],
        "summary_table_paths": [str(SUMMARY_BY_N), str(GLOBAL_SUMMARY), str(TOPK_SUMMARY), str(V03_VS_PRIOR), str(PRIOR_BOOT), str(BOOT_BY_N), str(BOOT_GLOBAL), str(ALIGN_JSON), str(POOL_SUMMARY), str(POOL_PROGRESS)],
        "plot_paths": plots, "report_path": str(REPORT), "claim_boundary_path": str(CLAIM_BOUNDARY),
        "current_PPO_teacher_validated_total": 96, "target_PPO_teacher_validated_total": 320, "remaining_to_target": 224,
        "no_Abaqus": True, "no_ODB_opening": True, "no_ODB_extraction": True, "no_solver": True, "no_datacheck": True, "no_enqueue": True, "no_training": True, "no_candidate_generation": True, "no_commit_or_push": True,
        "final_verdict": verdict,
    }, indent=2), encoding="utf-8")


def main() -> int:
    ensure_dirs()
    stage_r = pd.read_csv(STAGE_R_METRICS)
    selected = pd.read_csv(STAGE_P_SELECTED)
    raw_ref = pd.read_csv(COMBINED552)
    v01_raw = pd.read_csv(V01_METRICS)
    v02_raw = pd.read_csv(V02_METRICS)
    audit = input_audit(stage_r, selected, raw_ref, v01_raw, v02_raw)
    if audit.startswith("FAIL"):
        write_manifest("FAIL_STAGES_V03_ANALYSIS_NOT_READY", [])
        raise SystemExit(audit)
    ref = canonical_ref(raw_ref)
    v03 = canonical_v03(stage_r, selected)
    v01_all = canonical_v01(v01_raw, include_all=True)
    v01_target = canonical_v01(v01_raw, include_all=False)
    v02 = canonical_v02(v02_raw)
    make_analysis_datasets(ref, v01_all, v02, v03)
    ranking = annotate_candidates(ref, v03, prefix="v03")
    ranking.to_csv(FULL_RANK, index=False)
    new_df, new_counts = new_records(ranking, ref)
    topk_df, topk_summary = topk_tables(ranking)
    comp, prior_boot = compare_prior(ref, v01_target, v02, ranking)
    boot_by_n, boot_global = bootstrap_reference(ref, ranking)
    pool_all = pd.concat([v01_all, v02, v03], ignore_index=True)
    baseline_inv, baseline_v03, baseline_pool = baseline_tables(raw_ref, ref, ranking, pool_all)
    align_df, align_summary = alignment(ranking)
    byn, glob, best = summaries(ranking, ref, new_df, topk_summary)
    pool_summary, progress = cumulative_pool_summary(ref, pool_all)
    plots = make_plots(ref, ranking, comp, boot_global, align_df, progress)
    verdict = decide_verdict(new_counts, comp, topk_summary)
    write_report(audit, verdict, byn, new_counts, topk_summary, comp, prior_boot, boot_global, baseline_v03, align_summary, pool_summary, progress, best)
    write_manifest(verdict, plots)
    print(json.dumps({
        "audit_verdict": audit,
        "final_verdict": verdict,
        "new_records": new_counts,
        "topk_summary_by_N": topk_summary.to_dict("records"),
        "best_lex_by_N": best[best["best_type"] == "best_primary_lex"][["n", "strategy_name", "ref_lex_rank"]].to_dict("records"),
        "v03_vs_prior": comp[["n", "v01_best_lex_rank", "v02K2_best_lex_rank", "v03_best_lex_rank", "v01_topk_count", "v02K2_topk_count", "v03_topk_count", "interpretation"]].to_dict("records"),
        "bootstrap_global": boot_global.to_dict("records"),
        "alignment_summary": align_summary,
        "report": str(REPORT),
        "manifest": str(MANIFEST),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
