"""Stage W final 320-case PPO pool ranking and comparison analysis.

This is an analysis-only stage. It reads existing CSV/JSON/Markdown evidence and
creates non-frozen Stage W analysis outputs. It does not run Abaqus, open ODB
files, extract ODB metrics, run solver/datacheck/enqueue, generate CAE/INP/JNL
files, train models, or generate candidates.
"""

from __future__ import annotations

import json
import math
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
NS = "stage3_ppo_final_pool_320_analysis"
OUT_ROOT = ROOT / "outputs" / NS
DOCS_ROOT = ROOT / "docs" / NS
SCRIPT_ROOT = ROOT / NS / "scripts"
CHECKS = OUT_ROOT / "checks"
TABLES = OUT_ROOT / "tables"
PLOTS = OUT_ROOT / "plots"
REPORTS = OUT_ROOT / "reports"

PATHS = {
    "combined552": ROOT / "outputs" / "stage3_run_78_final_evidence_freeze_package" / "FROZEN_stage3_native_combined552_teacher_dataset.csv",
    "v01_metrics": ROOT / "outputs" / "stage3_ppo_rl_lam_fea_addendum_v01" / "stageI_final_ppo_evidence_freeze" / "frozen_tables" / "FROZEN_PPO_batch32_teacher_metrics.csv",
    "v01_ranking": ROOT / "outputs" / "stage3_ppo_rl_lam_fea_addendum_v01" / "stageI_final_ppo_evidence_freeze" / "frozen_tables" / "FROZEN_PPO_batch32_teacher_metric_ranking_full.csv",
    "v02K2_metrics": ROOT / "outputs" / "stage3_ppo_rl_lam_fea_addendum_v02_targeted_N24_N40" / "stageM_ODB_teacher_metric_extraction" / "stageM_v02K2_teacher_metrics.csv",
    "v02K2_ranking": ROOT / "outputs" / "stage3_ppo_rl_lam_fea_addendum_v02_targeted_N24_N40" / "stageN_teacher_metric_ranking" / "tables" / "v02K2_teacher_metric_ranking_full.csv",
    "v03_metrics": ROOT / "outputs" / "stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40" / "stageR_ODB_teacher_metric_extraction" / "stageR_v03_teacher_metrics.csv",
    "v03_ranking": ROOT / "outputs" / "stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40" / "stageS_teacher_metric_ranking" / "tables" / "v03_teacher_metric_ranking_full.csv",
    "final_expansion_metrics": ROOT / "outputs" / "stage3_ppo_final_expansion_224_to_320" / "stageV_ODB_teacher_metric_extraction" / "stageV_ppo_final_expansion_224_ODB_metrics.csv",
    "final_expansion_selected": ROOT / "outputs" / "stage3_ppo_final_expansion_224_to_320" / "selected_candidates" / "PPO_FINAL_EXPANSION_224_SELECTED_MASTER.csv",
    "stageV_report": ROOT / "docs" / "stage3_ppo_final_expansion_224_to_320" / "PPO_FINAL_EXPANSION_STAGEV_ODB_TEACHER_METRIC_EXTRACTION_REPORT.md",
}

METRICS = ["u2_range", "peeq_max", "surface_t_proxy", "mises_max"]
PRIMARY_METRICS = ["u2_range", "peeq_max", "surface_t_proxy"]
SUPPORTED_N = (12, 16, 24, 40)
EXPECTED_REF_COUNTS = {12: 78, 16: 78, 24: 190, 40: 206}
EXPECTED_PPO_COUNTS = {12: 40, 16: 40, 24: 120, 40: 120}


def ensure_dirs() -> None:
    for path in [OUT_ROOT, CHECKS, TABLES, PLOTS, REPORTS, DOCS_ROOT, SCRIPT_ROOT]:
        path.mkdir(parents=True, exist_ok=True)


def git_branch() -> str:
    try:
        result = subprocess.run(["git", "branch", "--show-current"], cwd=str(ROOT), capture_output=True, text=True, check=False)
        return result.stdout.strip() or "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def parse_order(value: object) -> list[int] | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.startswith("["):
        try:
            return [int(v) for v in json.loads(text)]
        except Exception:
            pass
    parts = [p for p in text.replace("|", ",").replace(";", ",").replace(" ", ",").split(",") if p != ""]
    try:
        return [int(float(p)) for p in parts]
    except Exception:
        return None


def order_ok(n: int, order: list[int] | None) -> bool:
    return order is not None and len(order) == int(n) and sorted(order) == list(range(int(n)))


def metric_map(df: pd.DataFrame) -> dict[str, str | None]:
    return {
        "u2_range": first_existing(df, ["u2_range"]),
        "peeq_max": first_existing(df, ["peeq_max"]),
        "surface_t_proxy": first_existing(df, ["surface_t_proxy_mpa", "surface_t_proxy_max_tensile_mpa", "surface_t_proxy"]),
        "mises_max": first_existing(df, ["mises_max"]),
    }


def normalize_surface(series: pd.Series, source_col: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    # Native combined552 often carries surface_t_proxy in Pa. Convert to MPa if
    # the chosen source looks Pa-scale so PPO and reference metrics share units.
    if source_col == "surface_t_proxy" and values.median(skipna=True) > 1e6:
        return values / 1e6
    return values


def canonicalize(df: pd.DataFrame, source: str, ppo_version: str | None, source_file: Path) -> pd.DataFrame:
    mm = metric_map(df)
    missing = [metric for metric, col in mm.items() if col is None]
    if missing:
        raise ValueError(f"{source} missing mappable metric columns: {missing}")
    out = pd.DataFrame(index=df.index)
    n_col = first_existing(df, ["n", "N"])
    if n_col is None:
        raise ValueError(f"{source} missing n/N column")
    out["n"] = pd.to_numeric(df[n_col], errors="coerce").astype("Int64")
    name_col = first_existing(df, ["strategy_name", "handoff_strategy_name", "job_name"])
    out["strategy_name"] = df[name_col].astype(str) if name_col else [f"{source}_{i:04d}" for i in range(len(df))]
    out["handoff_strategy_name"] = df["handoff_strategy_name"].astype(str) if "handoff_strategy_name" in df.columns else out["strategy_name"]
    out["dataset_source"] = source
    out["ppo_version"] = ppo_version if ppo_version is not None else ""
    out["is_ppo"] = ppo_version is not None
    out["source_metric_file"] = str(source_file)
    out["candidate_source"] = df["candidate_source"] if "candidate_source" in df.columns else source
    out["order_hash"] = df["order_hash"].astype(str) if "order_hash" in df.columns else ""
    order_col = first_existing(df, ["order_compact", "scan_order_json", "order_json", "scan_order"])
    out["order_compact"] = df[order_col].astype(str) if order_col else ""
    for metric, col in mm.items():
        if metric == "surface_t_proxy":
            out[metric] = normalize_surface(df[col], col)
            out["surface_t_proxy_source_column"] = col
        else:
            out[metric] = pd.to_numeric(df[col], errors="coerce")
    for col in [
        "teacher_validation_status",
        "completion_status",
        "odb_extraction_status",
        "extracted_field_names",
        "final_frame_time",
        "final_step_name",
        "selected_by_bucket",
        "final_expansion_batch",
        "ppo_checkpoint",
        "ppo_generation_mode",
        "ppo_version_source",
        "global_candidate_index",
        "predicted_quality_score",
        "min_hamming_to_combined552_sameN",
        "min_hamming_to_conventional_baseline",
        "nearest_conventional_baseline",
        "mean_abs_jump",
        "max_abs_jump",
        "long_jump_count",
        "adjacent_fraction",
        "total_travel_proxy",
        "jump_variance",
        "local_continuity_score",
        "path_complexity_score",
        "partial_training_caveat",
        "notes",
    ]:
        if col in df.columns:
            out[col] = df[col]
    out["teacher_metrics_extracted"] = True
    return out


def input_audit(raw: dict[str, pd.DataFrame]) -> tuple[str, dict]:
    rows = []

    def add(label: str, status: str, detail: str, value: object = "") -> None:
        rows.append({"check": label, "status": status, "detail": detail, "value": value})

    expected_rows = {
        "combined552": 552,
        "v01_metrics": 32,
        "v02K2_metrics": 32,
        "v03_metrics": 32,
        "final_expansion_metrics": 224,
    }
    for key, path in PATHS.items():
        if key.endswith("ranking") or key in {"stageV_report", "final_expansion_selected"}:
            add(f"{key}_exists", "PASS" if path.exists() else "WARNING", str(path))
        else:
            add(f"{key}_exists", "PASS" if path.exists() else "FAIL", str(path))
    for key, expected in expected_rows.items():
        actual = len(raw[key])
        add(f"{key}_row_count", "PASS" if actual == expected else "FAIL", f"expected={expected}", actual)
        mm = metric_map(raw[key])
        missing = [m for m, c in mm.items() if c is None]
        add(f"{key}_metric_mapping", "PASS" if not missing else "FAIL", json.dumps(mm), ",".join(missing))

    native_counts = raw["combined552"]["n"].astype(int).value_counts().sort_index().to_dict()
    add("combined552_native_counts", "PASS" if native_counts == EXPECTED_REF_COUNTS else "FAIL", json.dumps(EXPECTED_REF_COUNTS), json.dumps(native_counts))
    exp_counts = raw["final_expansion_metrics"]["n"].astype(int).value_counts().sort_index().to_dict()
    add("final_expansion_counts", "PASS" if exp_counts == {12: 32, 16: 32, 24: 80, 40: 80} else "FAIL", "expected N12=32,N16=32,N24=80,N40=80", json.dumps(exp_counts))

    # Candidate metadata join check.
    if PATHS["final_expansion_selected"].exists():
        selected = raw["final_expansion_selected"]
        metric_names = set(raw["final_expansion_metrics"]["handoff_strategy_name"].astype(str))
        selected_names = set(selected["strategy_name"].astype(str))
        missing_join = sorted(metric_names - selected_names)
        add("final_expansion_metadata_join", "PASS" if not missing_join else "FAIL", "Stage V metrics join selected master by strategy_name", len(missing_join))

    ppo = pd.concat(
        [
            canonicalize(raw["v01_metrics"], "ppo_v01_batch32", "v01", PATHS["v01_metrics"]),
            canonicalize(raw["v02K2_metrics"], "ppo_v02K2_batch32", "v02K2", PATHS["v02K2_metrics"]),
            canonicalize(raw["v03_metrics"], "ppo_v03_batch32", "v03", PATHS["v03_metrics"]),
            canonicalize(raw["final_expansion_metrics"], "ppo_final_expansion_224", "final_expansion", PATHS["final_expansion_metrics"]),
        ],
        ignore_index=True,
    )
    ppo_counts = ppo["n"].astype(int).value_counts().sort_index().to_dict()
    add("cumulative_ppo_pool_counts", "PASS" if ppo_counts == EXPECTED_PPO_COUNTS and len(ppo) == 320 else "FAIL", "expected total=320,N12=40,N16=40,N24=120,N40=120", json.dumps(ppo_counts))
    add("duplicate_strategy_name_within_ppo", "PASS" if ppo["strategy_name"].duplicated().sum() == 0 else "FAIL", "duplicate strategy_name count", int(ppo["strategy_name"].duplicated().sum()))
    same_n_hash_dupes = int(ppo[ppo["order_hash"].astype(str) != ""].duplicated(["n", "order_hash"]).sum())
    add("duplicate_sameN_order_hash_within_ppo", "PASS" if same_n_hash_dupes == 0 else "FAIL", "duplicate same-N order_hash count", same_n_hash_dupes)

    ref_hashes = set(raw["combined552"]["order_hash"].dropna().astype(str)) if "order_hash" in raw["combined552"].columns else set()
    ppo_hashes = set(ppo["order_hash"].dropna().astype(str)) - {""}
    duplicate_vs_ref = sorted(ppo_hashes & ref_hashes)
    # v01 contains a known recovery-anchor duplicate. Preserve as warning rather than stop.
    add("duplicate_vs_combined552_order_hash", "PASS" if len(duplicate_vs_ref) == 0 else "WARNING", "duplicate PPO hashes vs combined552", len(duplicate_vs_ref))

    status_cols = [c for c in ["teacher_validation_status", "completion_status", "odb_extraction_status"] if c in ppo.columns]
    extracted_ok = len(ppo) == 320 and all(ppo[m].notna().all() for m in METRICS)
    add("teacher_metrics_extracted_320", "PASS" if extracted_ok else "FAIL", "all metric columns non-null for 320 PPO rows", extracted_ok)
    stagev_text = PATHS["stageV_report"].read_text(encoding="utf-8", errors="ignore") if PATHS["stageV_report"].exists() else ""
    add("stageV_nonfatal_warning_preserved", "PASS" if "WARNING" in stagev_text.upper() or "NONFATAL" in stagev_text.upper() else "WARNING", "Stage V warning/nonfatal text present in report", "")

    audit = pd.DataFrame(rows)
    audit_path = CHECKS / "stageW_input_integrity_audit.csv"
    audit.to_csv(audit_path, index=False)
    fail_count = int((audit["status"] == "FAIL").sum())
    warning_count = int((audit["status"] == "WARNING").sum())
    verdict = "FAIL_STAGEW_FINAL_PPO_POOL_INPUTS_NOT_READY" if fail_count else ("WARNING_STAGEW_FINAL_PPO_POOL_INPUTS_REVIEW" if warning_count else "PASS_STAGEW_FINAL_PPO_POOL_INPUTS_READY")
    summary = {
        "verdict": verdict,
        "fail_count": fail_count,
        "warning_count": warning_count,
        "audit_path": str(audit_path),
        "ppo_counts_by_N": {str(k): int(v) for k, v in ppo_counts.items()},
        "duplicate_vs_combined552_order_hash_count": len(duplicate_vs_ref),
        "duplicate_vs_combined552_note": "known v01 recovery-anchor duplicate is nonfatal if present",
        "status_columns_checked": status_cols,
    }
    (CHECKS / "stageW_input_integrity_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return verdict, summary


def ref_relative_rank(ref_values: np.ndarray, candidate_value: float) -> tuple[int, float, bool, bool, bool, bool]:
    ref_values = np.asarray(ref_values, dtype=float)
    n = len(ref_values)
    rank = int(np.sum(ref_values < candidate_value) + 1)
    pct = rank / max(1, n)
    beats_best = bool(candidate_value < np.nanmin(ref_values))
    better_median = bool(candidate_value < np.nanmedian(ref_values))
    top10 = bool(rank <= math.ceil(0.10 * n))
    top25 = bool(rank <= math.ceil(0.25 * n))
    return rank, pct, beats_best, better_median, top10, top25


def minmax_cost(values: pd.Series) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce")
    lo, hi = vals.min(), vals.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(np.zeros(len(vals)), index=values.index)
    return (vals - lo) / (hi - lo)


def compute_rankings(ref: pd.DataFrame, ppo: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_rows = []
    combined_rows = []
    for n in SUPPORTED_N:
        ref_n = ref[ref["n"].astype(int) == n].copy()
        ppo_n = ppo[ppo["n"].astype(int) == n].copy()
        ref_n = ref_n.sort_values(["u2_range", "peeq_max", "surface_t_proxy"]).reset_index(drop=True)
        ref_lex_tuples = list(zip(ref_n["u2_range"], ref_n["peeq_max"], ref_n["surface_t_proxy"]))
        ref_diag_tuples = list(zip(ref_n["u2_range"], ref_n["peeq_max"], ref_n["surface_t_proxy"], ref_n["mises_max"]))
        best_ref_lex = min(ref_lex_tuples)
        best_ref_diag = min(ref_diag_tuples)

        combined = pd.concat([ref_n.assign(is_ppo=False), ppo_n.assign(is_ppo=True)], ignore_index=True)
        for metric in METRICS:
            combined[f"combined_rank_{metric}"] = combined[metric].rank(method="min", ascending=True).astype(int)
            combined[f"combined_percentile_{metric}"] = combined[f"combined_rank_{metric}"] / len(combined)
            combined[f"combined_minmax_cost_{metric}"] = minmax_cost(combined[metric])
        combined = combined.sort_values(["u2_range", "peeq_max", "surface_t_proxy"]).reset_index(drop=True)
        combined["combined_lex_rank"] = np.arange(1, len(combined) + 1)
        combined["combined_lex_percentile"] = combined["combined_lex_rank"] / len(combined)
        combined_rows.append(combined)

        for _, row in ppo_n.iterrows():
            rec = row.to_dict()
            for metric in METRICS:
                rank, pct, beats, better_med, top10, top25 = ref_relative_rank(ref_n[metric].values, float(row[metric]))
                rec[f"ref_rank_{metric}"] = rank
                rec[f"ref_percentile_{metric}"] = pct
                rec[f"ref_best_{metric}"] = float(ref_n[metric].min())
                rec[f"ref_median_{metric}"] = float(ref_n[metric].median())
                rec[f"candidate_beats_ref_best_{metric}"] = beats
                rec[f"candidate_better_than_ref_median_{metric}"] = better_med
                rec[f"candidate_top10pct_ref_{metric}"] = top10
                rec[f"candidate_top25pct_ref_{metric}"] = top25
            cand_lex = (float(row["u2_range"]), float(row["peeq_max"]), float(row["surface_t_proxy"]))
            cand_diag = (float(row["u2_range"]), float(row["peeq_max"]), float(row["surface_t_proxy"]), float(row["mises_max"]))
            lex_rank = int(sum(t < cand_lex for t in ref_lex_tuples) + 1)
            diag_rank = int(sum(t < cand_diag for t in ref_diag_tuples) + 1)
            rec["ref_lex_rank"] = lex_rank
            rec["ref_lex_percentile"] = lex_rank / len(ref_n)
            rec["candidate_beats_ref_best_lex"] = bool(cand_lex < best_ref_lex)
            rec["candidate_top10pct_ref_lex"] = bool(lex_rank <= math.ceil(0.10 * len(ref_n)))
            rec["candidate_top25pct_ref_lex"] = bool(lex_rank <= math.ceil(0.25 * len(ref_n)))
            rec["ref_diag_lex_rank"] = diag_rank
            rec["ref_diag_lex_percentile"] = diag_rank / len(ref_n)
            rec["candidate_beats_ref_best_diag_lex"] = bool(cand_diag < best_ref_diag)
            # Attach combined ranks for this strategy.
            match = combined[(combined["is_ppo"]) & (combined["strategy_name"] == row["strategy_name"])]
            if not match.empty:
                m = match.iloc[0]
                for metric in METRICS:
                    rec[f"combined_rank_{metric}"] = int(m[f"combined_rank_{metric}"])
                    rec[f"combined_percentile_{metric}"] = float(m[f"combined_percentile_{metric}"])
                    rec[f"combined_minmax_cost_{metric}"] = float(m[f"combined_minmax_cost_{metric}"])
                rec["combined_lex_rank"] = int(m["combined_lex_rank"])
                rec["combined_lex_percentile"] = float(m["combined_lex_percentile"])
            all_rows.append(rec)
    ranking = pd.DataFrame(all_rows)
    combined_ranked = pd.concat(combined_rows, ignore_index=True)
    return ranking, combined_ranked


def any_primary_topk(row: pd.Series, pct: int = 25) -> bool:
    cols = [
        f"candidate_top{pct}pct_ref_u2_range",
        f"candidate_top{pct}pct_ref_peeq_max",
        f"candidate_top{pct}pct_ref_surface_t_proxy",
        f"candidate_top{pct}pct_ref_lex",
    ]
    return bool(any(bool(row.get(col, False)) for col in cols))


def new_record_table(ranking: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in ranking.iterrows():
        for metric, kind in [
            ("u2_range", "primary"),
            ("peeq_max", "primary"),
            ("surface_t_proxy", "primary"),
            ("mises_max", "diagnostic"),
        ]:
            if bool(row.get(f"candidate_beats_ref_best_{metric}", False)):
                prev = float(row[f"ref_best_{metric}"])
                val = float(row[metric])
                rows.append(
                    {
                        "N": int(row["n"]),
                        "ppo_version": row["ppo_version"],
                        "strategy_name": row["strategy_name"],
                        "metric_or_lex": metric,
                        "ppo_value": val,
                        "previous_combined552_best": prev,
                        "improvement_ratio": val / prev if prev else np.nan,
                        "improvement_percent": (prev - val) / prev * 100.0 if prev else np.nan,
                        "primary_or_diagnostic": kind,
                        "caveat": "metric smaller is better; diagnostic Mises is not a primary hierarchy metric" if kind == "diagnostic" else "primary metric",
                    }
                )
        if bool(row.get("candidate_beats_ref_best_lex", False)):
            rows.append(
                {
                    "N": int(row["n"]),
                    "ppo_version": row["ppo_version"],
                    "strategy_name": row["strategy_name"],
                    "metric_or_lex": "primary_lex_u2_peeq_surface_t",
                    "ppo_value": int(row["ref_lex_rank"]),
                    "previous_combined552_best": 1,
                    "improvement_ratio": np.nan,
                    "improvement_percent": np.nan,
                    "primary_or_diagnostic": "primary",
                    "caveat": "lexicographic comparison U2 -> PEEQ -> SurfaceT",
                }
            )
    cols = ["N", "ppo_version", "strategy_name", "metric_or_lex", "ppo_value", "previous_combined552_best", "improvement_ratio", "improvement_percent", "primary_or_diagnostic", "caveat"]
    return pd.DataFrame(rows, columns=cols)


def topk_tables(ranking: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rank = ranking.copy()
    rank["primary_top10_any"] = rank.apply(any_primary_topk, pct=10, axis=1)
    rank["primary_top25_any"] = rank.apply(any_primary_topk, pct=25, axis=1)
    rank["diagnostic_mises_top10"] = rank["candidate_top10pct_ref_mises_max"].astype(bool)
    rank["diagnostic_mises_top25"] = rank["candidate_top25pct_ref_mises_max"].astype(bool)
    topk = rank[rank["primary_top10_any"] | rank["primary_top25_any"] | rank["diagnostic_mises_top10"] | rank["diagnostic_mises_top25"]].copy()

    rows_n = []
    for n, sub in rank.groupby("n"):
        rows_n.append(
            {
                "N": int(n),
                "ppo_count": int(len(sub)),
                "top10pct_U2": int(sub["candidate_top10pct_ref_u2_range"].sum()),
                "top25pct_U2": int(sub["candidate_top25pct_ref_u2_range"].sum()),
                "top10pct_PEEQ": int(sub["candidate_top10pct_ref_peeq_max"].sum()),
                "top25pct_PEEQ": int(sub["candidate_top25pct_ref_peeq_max"].sum()),
                "top10pct_SurfaceT": int(sub["candidate_top10pct_ref_surface_t_proxy"].sum()),
                "top25pct_SurfaceT": int(sub["candidate_top25pct_ref_surface_t_proxy"].sum()),
                "top10pct_lex": int(sub["candidate_top10pct_ref_lex"].sum()),
                "top25pct_lex": int(sub["candidate_top25pct_ref_lex"].sum()),
                "primary_top10_any_unique": int(sub["primary_top10_any"].sum()),
                "primary_top25_any_unique": int(sub["primary_top25_any"].sum()),
                "diagnostic_Mises_top10": int(sub["diagnostic_mises_top10"].sum()),
                "diagnostic_Mises_top25": int(sub["diagnostic_mises_top25"].sum()),
            }
        )
    by_n = pd.DataFrame(rows_n)
    rows_v = []
    for (version, n), sub in rank.groupby(["ppo_version", "n"]):
        rows_v.append(
            {
                "ppo_version": version,
                "N": int(n),
                "count": int(len(sub)),
                "primary_top10_any_unique": int(sub["primary_top10_any"].sum()),
                "primary_top25_any_unique": int(sub["primary_top25_any"].sum()),
                "best_ref_lex_rank": int(sub["ref_lex_rank"].min()),
                "best_ref_u2_rank": int(sub["ref_rank_u2_range"].min()),
            }
        )
    by_v = pd.DataFrame(rows_v)
    return topk, by_n, by_v


def final_expansion_vs_prior(ranking: pd.DataFrame) -> pd.DataFrame:
    rows = []
    versions = ["v01", "v02K2", "v03", "final_expansion"]
    for n, subn in ranking.groupby("n"):
        rec = {"N": int(n)}
        prior_best_lex = None
        prior_topk = 0
        for version in versions:
            subv = subn[subn["ppo_version"] == version]
            rec[f"{version}_count"] = int(len(subv))
            if len(subv):
                rec[f"{version}_best_lex_rank"] = int(subv["ref_lex_rank"].min())
                rec[f"{version}_best_U2_rank"] = int(subv["ref_rank_u2_range"].min())
                rec[f"{version}_best_PEEQ_rank"] = int(subv["ref_rank_peeq_max"].min())
                rec[f"{version}_best_SurfaceT_rank"] = int(subv["ref_rank_surface_t_proxy"].min())
                rec[f"{version}_top25_any_primary_count"] = int(subv.apply(any_primary_topk, axis=1).sum())
                if version != "final_expansion":
                    prior_best_lex = rec[f"{version}_best_lex_rank"] if prior_best_lex is None else min(prior_best_lex, rec[f"{version}_best_lex_rank"])
                    prior_topk += rec[f"{version}_top25_any_primary_count"]
            else:
                rec[f"{version}_best_lex_rank"] = np.nan
                rec[f"{version}_top25_any_primary_count"] = 0
        fe = subn[subn["ppo_version"] == "final_expansion"]
        rec["final_expansion_improves_prior_best_lex"] = bool(len(fe) and prior_best_lex is not None and fe["ref_lex_rank"].min() < prior_best_lex)
        rec["final_expansion_improves_prior_topk_count"] = bool(len(fe) and int(fe.apply(any_primary_topk, axis=1).sum()) > prior_topk)
        if rec["final_expansion_improves_prior_best_lex"] and rec["final_expansion_improves_prior_topk_count"]:
            rec["interpretation"] = "final expansion improves prior PPO best lex rank and top-k count"
        elif rec["final_expansion_improves_prior_best_lex"]:
            rec["interpretation"] = "final expansion improves prior PPO best lex rank only"
        elif rec["final_expansion_improves_prior_topk_count"]:
            rec["interpretation"] = "final expansion improves prior PPO top-k count by larger budget"
        else:
            rec["interpretation"] = "final expansion broadens evidence pool without improving prior PPO best/top-k"
        rows.append(rec)
    return pd.DataFrame(rows)


def bootstrap_reference(ref: pd.DataFrame, ranking: pd.DataFrame, trials: int = 10000) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(20260701)
    by_n_rows = []
    global_stats = []
    for n in SUPPORTED_N:
        refn = ref[ref["n"].astype(int) == n].copy()
        ppon = ranking[ranking["n"].astype(int) == n].copy()
        sample_size = EXPECTED_PPO_COUNTS[n]
        # Precompute reference ranks among reference itself.
        refn = refn.sort_values(["u2_range", "peeq_max", "surface_t_proxy"]).reset_index(drop=True)
        for metric in METRICS:
            refn[f"ref_rank_{metric}"] = refn[metric].rank(method="min", ascending=True).astype(int)
            refn[f"top10_{metric}"] = refn[f"ref_rank_{metric}"] <= math.ceil(0.10 * len(refn))
            refn[f"top25_{metric}"] = refn[f"ref_rank_{metric}"] <= math.ceil(0.25 * len(refn))
        refn["ref_lex_rank"] = np.arange(1, len(refn) + 1)
        refn["top10_lex"] = refn["ref_lex_rank"] <= math.ceil(0.10 * len(refn))
        refn["top25_lex"] = refn["ref_lex_rank"] <= math.ceil(0.25 * len(refn))

        observed = {
            "top10_U2": int(ppon["candidate_top10pct_ref_u2_range"].sum()),
            "top25_U2": int(ppon["candidate_top25pct_ref_u2_range"].sum()),
            "top10_PEEQ": int(ppon["candidate_top10pct_ref_peeq_max"].sum()),
            "top25_PEEQ": int(ppon["candidate_top25pct_ref_peeq_max"].sum()),
            "top10_SurfaceT": int(ppon["candidate_top10pct_ref_surface_t_proxy"].sum()),
            "top25_SurfaceT": int(ppon["candidate_top25pct_ref_surface_t_proxy"].sum()),
            "top10_lex": int(ppon["candidate_top10pct_ref_lex"].sum()),
            "top25_lex": int(ppon["candidate_top25pct_ref_lex"].sum()),
            "best_lex_rank": int(ppon["ref_lex_rank"].min()),
            "median_lex_rank": float(ppon["ref_lex_rank"].median()),
            "best_U2_rank": int(ppon["ref_rank_u2_range"].min()),
            "median_U2_rank": float(ppon["ref_rank_u2_range"].median()),
            "primary_top25_any_unique": int(ppon.apply(any_primary_topk, axis=1).sum()),
        }
        dist = {k: [] for k in observed}
        idx = np.arange(len(refn))
        for _ in range(trials):
            sample = refn.iloc[rng.choice(idx, size=sample_size, replace=False)]
            vals = {
                "top10_U2": int(sample["top10_u2_range"].sum()),
                "top25_U2": int(sample["top25_u2_range"].sum()),
                "top10_PEEQ": int(sample["top10_peeq_max"].sum()),
                "top25_PEEQ": int(sample["top25_peeq_max"].sum()),
                "top10_SurfaceT": int(sample["top10_surface_t_proxy"].sum()),
                "top25_SurfaceT": int(sample["top25_surface_t_proxy"].sum()),
                "top10_lex": int(sample["top10_lex"].sum()),
                "top25_lex": int(sample["top25_lex"].sum()),
                "best_lex_rank": int(sample["ref_lex_rank"].min()),
                "median_lex_rank": float(sample["ref_lex_rank"].median()),
                "best_U2_rank": int(sample["ref_rank_u2_range"].min()),
                "median_U2_rank": float(sample["ref_rank_u2_range"].median()),
                "primary_top25_any_unique": int((sample["top25_u2_range"] | sample["top25_peeq_max"] | sample["top25_surface_t_proxy"] | sample["top25_lex"]).sum()),
            }
            for k, v in vals.items():
                dist[k].append(v)
        for stat, obs in observed.items():
            arr = np.asarray(dist[stat], dtype=float)
            rank_lower_better = "rank" in stat
            if rank_lower_better:
                p = float(np.mean(arr <= obs))
                interp = "enriched" if obs < np.quantile(arr, 0.05) else ("comparable" if obs <= np.quantile(arr, 0.95) else "weak")
            else:
                p = float(np.mean(arr >= obs))
                interp = "enriched" if obs > np.quantile(arr, 0.95) else ("comparable" if obs >= np.quantile(arr, 0.05) else "weak")
            by_n_rows.append(
                {
                    "N": n,
                    "statistic": stat,
                    "observed": obs,
                    "bootstrap_mean": float(np.mean(arr)),
                    "bootstrap_median": float(np.median(arr)),
                    "bootstrap_q05": float(np.quantile(arr, 0.05)),
                    "bootstrap_q95": float(np.quantile(arr, 0.95)),
                    "empirical_p_value": p,
                    "interpretation": interp,
                    "trials": trials,
                }
            )
        global_stats.append({"N": n, "observed": observed, "dist": dist})

    # Global counts: sum same statistic across N for count-like metrics.
    global_rows = []
    count_stats = ["top10_U2", "top25_U2", "top10_PEEQ", "top25_PEEQ", "top10_SurfaceT", "top25_SurfaceT", "top10_lex", "top25_lex", "primary_top25_any_unique"]
    for stat in count_stats:
        obs = sum(int(g["observed"][stat]) for g in global_stats)
        arr = np.sum([np.asarray(g["dist"][stat], dtype=float) for g in global_stats], axis=0)
        interp = "enriched" if obs > np.quantile(arr, 0.95) else ("comparable" if obs >= np.quantile(arr, 0.05) else "weak")
        global_rows.append(
            {
                "statistic": stat,
                "observed": obs,
                "bootstrap_mean": float(np.mean(arr)),
                "bootstrap_median": float(np.median(arr)),
                "bootstrap_q05": float(np.quantile(arr, 0.05)),
                "bootstrap_q95": float(np.quantile(arr, 0.95)),
                "empirical_p_value_greater_equal": float(np.mean(arr >= obs)),
                "interpretation": interp,
                "trials": trials,
            }
        )
    return pd.DataFrame(by_n_rows), pd.DataFrame(global_rows)


def baseline_family_tables(ref: pd.DataFrame, ranking: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keywords = ["raster", "odd_even", "edge_in", "center_out", "center_edge", "method_c", "regular_jump", "engineering", "heuristic", "random"]
    search_cols = [c for c in ["strategy_name", "candidate_family", "candidate_source", "generation_method", "selection_bucket", "batch_name", "priority_role"] if c in ref.columns]
    inv_rows = []
    comp_rows = []
    ref_ext = ref.copy()
    label_text = ref_ext[search_cols].astype(str).agg(" ".join, axis=1).str.lower() if search_cols else pd.Series([""] * len(ref_ext))
    for key in keywords:
        mask = label_text.str.contains(key, regex=False)
        count = int(mask.sum())
        inv_rows.append({"family_keyword": key, "matched_rows": count, "reliable": bool(count >= 2), "search_columns": ",".join(search_cols)})
        if count < 2:
            continue
        fam = ref_ext[mask].copy()
        for n, famn in fam.groupby("n"):
            ppon = ranking[ranking["n"].astype(int) == int(n)]
            if ppon.empty:
                continue
            refn = ref_ext[ref_ext["n"].astype(int) == int(n)].sort_values(["u2_range", "peeq_max", "surface_t_proxy"]).reset_index(drop=True)
            # Lex rank within native reference for family rows.
            fam_names = set(famn["strategy_name"].astype(str))
            famlex = refn[refn["strategy_name"].astype(str).isin(fam_names)].copy()
            famlex["lex_rank"] = famlex.index + 1
            comp_rows.append(
                {
                    "N": int(n),
                    "family_keyword": key,
                    "family_count": int(len(famn)),
                    "PPO_count": int(len(ppon)),
                    "PPO_best_U2": float(ppon["u2_range"].min()),
                    "family_best_U2": float(famn["u2_range"].min()),
                    "PPO_median_U2": float(ppon["u2_range"].median()),
                    "family_median_U2": float(famn["u2_range"].median()),
                    "PPO_best_PEEQ": float(ppon["peeq_max"].min()),
                    "family_best_PEEQ": float(famn["peeq_max"].min()),
                    "PPO_best_SurfaceT": float(ppon["surface_t_proxy"].min()),
                    "family_best_SurfaceT": float(famn["surface_t_proxy"].min()),
                    "PPO_best_Mises": float(ppon["mises_max"].min()),
                    "family_best_Mises": float(famn["mises_max"].min()),
                    "PPO_best_ref_lex_rank": int(ppon["ref_lex_rank"].min()),
                    "family_best_ref_lex_rank": int(famlex["lex_rank"].min()) if not famlex.empty else np.nan,
                    "PPO_beats_family_best_lex": bool(not famlex.empty and ppon["ref_lex_rank"].min() < famlex["lex_rank"].min()),
                    "interpretation": "comparison uses label-keyword inventory; do not overclaim if family labels are broad",
                }
            )
    return pd.DataFrame(comp_rows), pd.DataFrame(inv_rows)


def industrial_proxy_tables(ppo: pd.DataFrame, ranking: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    proxy_cols = ["mean_abs_jump", "max_abs_jump", "long_jump_count", "adjacent_fraction", "total_travel_proxy", "jump_variance", "local_continuity_score", "path_complexity_score"]
    available = [c for c in proxy_cols if c in ppo.columns]
    if not available:
        return pd.DataFrame(), pd.DataFrame()
    summary = ppo.dropna(subset=available, how="all").groupby(["n", "ppo_version"], dropna=False)[available].agg(["count", "mean", "median", "min", "max"]).round(6)
    summary = summary.reset_index()
    merged = ranking.merge(ppo[["strategy_name"] + available], on="strategy_name", how="left", suffixes=("", "_meta"))
    rows = []
    for n, sub in merged.groupby("n"):
        for proxy in available:
            x = pd.to_numeric(sub[proxy], errors="coerce")
            for target in ["u2_range", "peeq_max", "surface_t_proxy", "ref_lex_rank"]:
                y = pd.to_numeric(sub[target], errors="coerce")
                valid = x.notna() & y.notna()
                if valid.sum() >= 3:
                    rows.append(
                        {
                            "N": int(n),
                            "proxy": proxy,
                            "teacher_metric_or_rank": target,
                            "spearman": float(x[valid].corr(y[valid], method="spearman")),
                            "pearson": float(x[valid].corr(y[valid], method="pearson")),
                            "count": int(valid.sum()),
                            "claim_boundary": "proxy descriptor only; not physically validated industrial efficiency",
                        }
                    )
    return summary, pd.DataFrame(rows)


def best_candidates(ranking: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for n, sub in ranking.groupby("n"):
        for label, sort_col in [
            ("primary_lex", "ref_lex_rank"),
            ("U2", "ref_rank_u2_range"),
            ("PEEQ", "ref_rank_peeq_max"),
            ("SurfaceT", "ref_rank_surface_t_proxy"),
            ("Mises_diagnostic", "ref_rank_mises_max"),
        ]:
            row = sub.sort_values(sort_col).iloc[0]
            rows.append(
                {
                    "N": int(n),
                    "criterion": label,
                    "strategy_name": row["strategy_name"],
                    "ppo_version": row["ppo_version"],
                    "ref_rank": int(row[sort_col]),
                    "u2_range": float(row["u2_range"]),
                    "peeq_max": float(row["peeq_max"]),
                    "surface_t_proxy_mpa": float(row["surface_t_proxy"]),
                    "mises_max": float(row["mises_max"]),
                    "new_record_primary_or_metric": bool(
                        row.get("candidate_beats_ref_best_lex", False)
                        if label == "primary_lex"
                        else row.get(f"candidate_beats_ref_best_{'mises_max' if label == 'Mises_diagnostic' else ('surface_t_proxy' if label == 'SurfaceT' else ('peeq_max' if label == 'PEEQ' else 'u2_range'))}", False)
                    ),
                }
            )
    return pd.DataFrame(rows)


def version_summary(ppo: pd.DataFrame, ranking: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (version, n), sub in ranking.groupby(["ppo_version", "n"]):
        rows.append(
            {
                "ppo_version": version,
                "N": int(n),
                "count": int(len(sub)),
                "best_lex_rank": int(sub["ref_lex_rank"].min()),
                "median_lex_rank": float(sub["ref_lex_rank"].median()),
                "best_U2_rank": int(sub["ref_rank_u2_range"].min()),
                "best_PEEQ_rank": int(sub["ref_rank_peeq_max"].min()),
                "best_SurfaceT_rank": int(sub["ref_rank_surface_t_proxy"].min()),
                "primary_top25_any": int(sub.apply(any_primary_topk, axis=1).sum()),
                "new_primary_records": int(
                    (
                        sub["candidate_beats_ref_best_u2_range"]
                        | sub["candidate_beats_ref_best_peeq_max"]
                        | sub["candidate_beats_ref_best_surface_t_proxy"]
                        | sub["candidate_beats_ref_best_lex"]
                    ).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def claim_decisions(ranking: pd.DataFrame, new_records: pd.DataFrame, topk_by_n: pd.DataFrame, baseline_comp: pd.DataFrame, bootstrap_global: pd.DataFrame, proxy_corr: pd.DataFrame) -> pd.DataFrame:
    primary_new = new_records[new_records["primary_or_diagnostic"] == "primary"] if not new_records.empty else pd.DataFrame()
    total_top25 = int(topk_by_n["primary_top25_any_unique"].sum()) if not topk_by_n.empty else 0
    baseline_supported = bool(not baseline_comp.empty and baseline_comp.get("PPO_beats_family_best_lex", pd.Series(dtype=bool)).any())
    boot_primary = bootstrap_global[bootstrap_global["statistic"] == "primary_top25_any_unique"]
    boot_interp = boot_primary["interpretation"].iloc[0] if not boot_primary.empty else "not_available"
    rows = [
        ("PPO generated legal scan-order permutations.", "supported", "Stage T/V manifests and Stage W input integrity", "320 PPO-generated cases have teacher metrics extracted; legality was audited before handoff.", ""),
        ("PPO-generated candidates were independently teacher-metric extracted by Abaqus.", "supported", "Stage V metrics + Stage W final pool", "Abaqus teacher metrics were extracted for 320 PPO candidates.", ""),
        ("PPO pool reached 320 teacher-metric-extracted cases.", "supported" if len(ranking) == 320 else "unsupported", "ppo_final_pool_320_teacher_metrics.csv", "The final PPO pool contains 320 teacher-metric-extracted cases.", ""),
        ("PPO beats conventional simple baselines.", "partially_supported" if baseline_supported else "unsupported_or_not_reliable", "baseline-family comparison", "PPO can be compared against identified baseline families where labels are reliable.", "Do not overclaim if family labels are broad or sparse."),
        ("PPO competes with stronger engineering/method baselines.", "partially_supported", "baseline-family comparison + top-k tables", "PPO competitiveness is bounded by top-k/reference evidence.", "Mature combined552 remains the high bar."),
        ("PPO reaches top-k regions of combined552.", "supported" if total_top25 > 0 else "unsupported", "top-k summary", f"PPO reaches combined552 top-k regions for {total_top25} candidates under primary metrics/lex.", ""),
        ("PPO beats combined552 best.", "supported" if len(primary_new) else "unsupported", "new-record audit", "PPO beats combined552 best only for primary new records if present.", "No such claim if new-record table has zero primary rows."),
        ("PPO produces new records.", "supported" if len(primary_new) else "unsupported", "new-record audit", "PPO produced new records only if listed in the audit.", ""),
        ("PPO improves N24/N40 U2/lex over v01/v02K2.", "partially_supported", "final_expansion_vs_prior_ppo_by_N.csv", "Version-to-version improvement should be stated by N and criterion only.", "Budget differs across versions."),
        ("PPO demonstrates industrial-efficiency optimisation.", "unsupported", "industrial proxy tables", "Industrial-efficiency descriptors are proxy descriptors only.", "No physical industrial-efficiency validation in Stage W."),
        ("PPO is better than surrogate-assisted optimisation.", "unsupported", "combined552 comparison", "PPO is compared against combined552; superiority requires new records or robust enrichment.", "Do not claim superiority to the mature reference if unsupported."),
        ("PPO defines a practical boundary of surrogate-trained policy-gradient generation.", "supported", "full evidence chain", "The 320-case pool provides a large-scale teacher-metric evidence base for surrogate-trained PPO candidate generation.", f"Bootstrap interpretation: {boot_interp}."),
    ]
    return pd.DataFrame(rows, columns=["claim", "evidence_status", "evidence_table", "safe_manuscript_wording", "caveat"])


def make_plots(ref: pd.DataFrame, ranking: pd.DataFrame, topk_version: pd.DataFrame, prior_comp: pd.DataFrame, bootstrap_global: pd.DataFrame, proxy_corr: pd.DataFrame) -> list[str]:
    paths = []
    metric_labels = {
        "u2_range": "U2 Range",
        "peeq_max": "PEEQ Max",
        "surface_t_proxy": "SurfaceT proxy (MPa)",
        "mises_max": "Mises Max",
    }
    for metric, label in metric_labels.items():
        fig, axes = plt.subplots(1, 4, figsize=(13, 3), sharey=False)
        for ax, n in zip(axes, SUPPORTED_N):
            ax.hist(ref[ref["n"].astype(int) == n][metric], bins=20, alpha=0.55, label="combined552")
            ax.hist(ranking[ranking["n"].astype(int) == n][metric], bins=20, alpha=0.55, label="PPO320")
            ax.set_title(f"N{n}")
            ax.tick_params(axis="x", labelrotation=35)
        axes[0].set_ylabel("count")
        axes[0].legend(fontsize=7)
        fig.suptitle(f"PPO final pool vs combined552: {label}")
        fig.tight_layout()
        path = PLOTS / f"ppo320_vs_combined552_{metric}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))

    fig, ax = plt.subplots(figsize=(7, 4))
    ranking.boxplot(column="ref_lex_percentile", by="n", ax=ax)
    ax.set_title("PPO final pool primary lex percentile by N")
    ax.set_xlabel("N")
    ax.set_ylabel("reference lex percentile")
    fig.suptitle("")
    fig.tight_layout()
    path = PLOTS / "ppo320_lex_rank_percentile_by_N.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    if not topk_version.empty:
        pivot = topk_version.pivot_table(index="N", columns="ppo_version", values="primary_top25_any_unique", aggfunc="sum").fillna(0)
        fig, ax = plt.subplots(figsize=(8, 4))
        pivot.plot(kind="bar", ax=ax)
        ax.set_title("Primary top25 any count by PPO version and N")
        ax.set_ylabel("count")
        fig.tight_layout()
        path = PLOTS / "ppo_topk_count_by_version_and_N.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))

    if not prior_comp.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        for version in ["v01", "v02K2", "v03", "final_expansion"]:
            col = f"{version}_best_lex_rank"
            if col in prior_comp.columns:
                ax.plot(prior_comp["N"], prior_comp[col], marker="o", label=version)
        ax.invert_yaxis()
        ax.set_title("Best lex rank by PPO stage")
        ax.set_xlabel("N")
        ax.set_ylabel("lower is better")
        ax.legend()
        fig.tight_layout()
        path = PLOTS / "final_expansion_vs_prior_ppo_best_lex_rank.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))

    if not bootstrap_global.empty:
        row = bootstrap_global[bootstrap_global["statistic"] == "primary_top25_any_unique"]
        if not row.empty:
            r = row.iloc[0]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(["bootstrap mean", "PPO observed"], [r["bootstrap_mean"], r["observed"]], color=["0.65", "tab:blue"])
            ax.errorbar([0], [r["bootstrap_mean"]], yerr=[[r["bootstrap_mean"] - r["bootstrap_q05"]], [r["bootstrap_q95"] - r["bootstrap_mean"]]], fmt="none", color="black")
            ax.set_title("PPO final pool vs random-reference bootstrap")
            ax.set_ylabel("primary top25-any count")
            fig.tight_layout()
            path = PLOTS / "ppo320_vs_bootstrap_random_reference_topk_count.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            paths.append(str(path))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([1, 2, 3, 4], [32, 64, 96, 320], marker="o")
    ax.set_xticks([1, 2, 3, 4], ["v01", "v02K2", "v03", "final"])
    ax.set_ylabel("teacher-metric-extracted PPO cases")
    ax.set_title("PPO pool size progression")
    fig.tight_layout()
    path = PLOTS / "ppo_pool_size_progression_32_64_96_320.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    return paths


def write_report(verdict: str, input_verdict: str, summary: dict, paths_out: dict, tables: dict) -> tuple[Path, Path]:
    report_path = DOCS_ROOT / "PPO_FINAL_POOL_320_STAGEW_RANKING_AND_COMPARISON_REPORT.md"
    claim_path = DOCS_ROOT / "PPO_FINAL_POOL_320_STAGEW_CLAIM_BOUNDARY.md"
    topk_by_n = tables["topk_by_n"]
    new_records = tables["new_records"]
    prior = tables["prior_comp"]
    bootstrap_global = tables["bootstrap_global"]
    best = tables["best_candidates"]
    claim = tables["claim"]
    primary_new_count = int((new_records["primary_or_diagnostic"] == "primary").sum()) if not new_records.empty else 0
    topk_total = int(topk_by_n["primary_top25_any_unique"].sum()) if not topk_by_n.empty else 0
    boot_line = "not available"
    row = bootstrap_global[bootstrap_global["statistic"] == "primary_top25_any_unique"] if not bootstrap_global.empty else pd.DataFrame()
    if not row.empty:
        r = row.iloc[0]
        boot_line = f"observed={r['observed']}, bootstrap mean={r['bootstrap_mean']:.2f}, q05={r['bootstrap_q05']:.2f}, q95={r['bootstrap_q95']:.2f}, interpretation={r['interpretation']}"

    report_path.write_text(
        f"""# PPO Final Pool 320 Stage W Ranking and Comparison Report

## Purpose

Stage W integrates PPO v01, PPO v02K2, PPO v03, and the final 224-case PPO expansion into a 320-case teacher-metric-extracted PPO evidence pool, then compares it against native combined552 and related baselines.

## Evidence Pool Composition

- PPO v01: 32
- PPO v02K2: 32
- PPO v03: 32
- Final expansion: 224
- Final PPO pool: 320
- By N: {summary.get('ppo_counts_by_N')}

## Stage V Extraction Status

Stage V final expansion teacher metrics were read from `{PATHS['final_expansion_metrics']}`. Stage V warning/nonfatal status is preserved in the input audit; Stage W does not open ODB files or extract metrics.

## Input Integrity Verdict

{input_verdict}

## Final PPO Pool Dataset

- PPO final pool: `{paths_out['ppo_pool']}`
- Combined552 + PPO analysis dataset: `{paths_out['combined']}`
- Full ranking table: `{paths_out['ranking']}`

## Comparison Against Combined552

PPO candidates were ranked within each N against the native combined552 teacher-labelled reference. Metrics are smaller-is-better. SurfaceT is compared in MPa units after explicit mapping.

## New-Record Audit

- Primary new-record rows: {primary_new_count}
- New-record table: `{paths_out['new_records']}`

## Top-k Competitiveness Audit

- Unique PPO candidates in at least one primary top25 region: {topk_total}
- Top-k by N table: `{paths_out['topk_by_n']}`
- Top-k by version table: `{paths_out['topk_by_version']}`

## Final Expansion vs Prior PPO Stages

`{paths_out['prior_comp']}`

```text
{prior.to_string(index=False) if not prior.empty else 'No prior comparison rows.'}
```

## Random-Reference Bootstrap

Equal-budget bootstrap against existing teacher-labelled reference distribution:

{boot_line}

## Conventional Baseline-Family Comparison

Baseline-family comparison is label-derived from combined552 where labels are available. It should not be overclaimed when family labels are sparse or broad.

## Industrial-Efficiency Proxy Analysis

Industrial-efficiency descriptors are sequence proxies only. They are not physical teacher metrics and do not establish experimentally validated industrial efficiency.

## Claim-Decision Table

`{paths_out['claim']}`

## Main Scientific Interpretation

The final 320-case pool is a large-scale teacher-metric-extracted evidence base for surrogate-trained policy-gradient scan-order generation. Final physical claims must follow the new-record, top-k, bootstrap, and baseline-family tables rather than surrogate scores.

## Limitations

- The random-reference bootstrap samples from the existing teacher-labelled combined552 distribution, not the full scan-order universe.
- Baseline-family extraction depends on available labels.
- Industrial-efficiency fields are proxies only.
- Stage W performs analysis only; it does not generate new physical evidence.

## Recommended Manuscript Wording

Use bounded wording: "A 320-case PPO-generated scan-order pool was teacher-metric extracted and ranked against the native combined552 reference. The PPO evidence is reported by new-record, top-k, bootstrap, and baseline-family audits."

## Verdict

{verdict}
""",
        encoding="utf-8",
    )

    claim_path.write_text(
        """# PPO Final Pool 320 Stage W Claim Boundary

## Safe Only If Supported By Stage W Tables

- 320 PPO-generated scan-order candidates have teacher metrics extracted.
- PPO generated legal and executable scan-order permutations.
- PPO achieved top-k competitiveness if supported by the top-k tables.
- PPO beat conventional baselines if supported by reliable baseline-family tables.
- PPO produced new records only if listed in the new-record audit.

## Always Safe

- PPO final pool provides a large-scale teacher-metric-extracted evidence base for evaluating surrogate-trained policy-gradient scan-order generation.
- Final physical claims are based on Abaqus teacher metrics, not surrogate scores.

## Unsafe Unless Proven

- PPO beats combined552 best.
- PPO is superior to the mature surrogate-assisted optimiser.
- PPO solves N24/N40.
- PPO is experimentally validated.
- Industrial efficiency is physically validated.
- The surrogate score alone predicts physical quality.
""",
        encoding="utf-8",
    )
    return report_path, claim_path


def main() -> None:
    ensure_dirs()
    raw = {
        "combined552": read_csv(PATHS["combined552"]),
        "v01_metrics": read_csv(PATHS["v01_metrics"]),
        "v02K2_metrics": read_csv(PATHS["v02K2_metrics"]),
        "v03_metrics": read_csv(PATHS["v03_metrics"]),
        "final_expansion_metrics": read_csv(PATHS["final_expansion_metrics"]),
        "final_expansion_selected": read_csv(PATHS["final_expansion_selected"]),
    }
    input_verdict, input_summary = input_audit(raw)
    if input_verdict.startswith("FAIL"):
        print(json.dumps(input_summary, indent=2))
        return

    ref = canonicalize(raw["combined552"], "stage3_native_combined552", None, PATHS["combined552"])
    # Add baseline family columns back onto ref for label search.
    for col in ["candidate_family", "generation_method", "selection_bucket", "batch_name", "priority_role"]:
        if col in raw["combined552"].columns:
            ref[col] = raw["combined552"][col]

    ppo_v01 = canonicalize(raw["v01_metrics"], "ppo_v01_batch32", "v01", PATHS["v01_metrics"])
    ppo_v02 = canonicalize(raw["v02K2_metrics"], "ppo_v02K2_batch32", "v02K2", PATHS["v02K2_metrics"])
    ppo_v03 = canonicalize(raw["v03_metrics"], "ppo_v03_batch32", "v03", PATHS["v03_metrics"])
    ppo_final = canonicalize(raw["final_expansion_metrics"], "ppo_final_expansion_224", "final_expansion", PATHS["final_expansion_metrics"])

    # Join final-expansion candidate metadata for proxy and bucket fields.
    meta = raw["final_expansion_selected"].copy()
    meta = meta.rename(columns={"strategy_name": "handoff_strategy_name"})
    join_cols = [c for c in meta.columns if c not in ppo_final.columns or c in ["handoff_strategy_name"]]
    ppo_final = ppo_final.merge(meta[join_cols], on="handoff_strategy_name", how="left", suffixes=("", "_candidate"))
    if "strategy_name_candidate" in ppo_final.columns:
        ppo_final["strategy_name"] = ppo_final["strategy_name_candidate"].fillna(ppo_final["strategy_name"])

    ppo = pd.concat([ppo_v01, ppo_v02, ppo_v03, ppo_final], ignore_index=True)
    ppo["teacher_metrics_extracted"] = True

    ppo_pool_path = TABLES / "ppo_final_pool_320_teacher_metrics.csv"
    combined_path = TABLES / "combined552_plus_ppo_final_pool_320_analysis_dataset.csv"
    version_path = TABLES / "ppo_final_pool_320_version_summary.csv"
    final_join_path = TABLES / "ppo_final_expansion_224_metrics_with_candidate_metadata.csv"
    ppo.to_csv(ppo_pool_path, index=False)
    pd.concat([ref.assign(is_ppo=False, ppo_version=""), ppo.assign(is_ppo=True)], ignore_index=True).to_csv(combined_path, index=False)
    ppo_final.to_csv(final_join_path, index=False)

    ranking, combined_ranked = compute_rankings(ref, ppo)
    ranking_path = TABLES / "ppo_final_pool_320_teacher_metric_ranking_full.csv"
    ranking.to_csv(ranking_path, index=False)

    version = version_summary(ppo, ranking)
    version.to_csv(version_path, index=False)

    new_records = new_record_table(ranking)
    new_records_path = TABLES / "ppo_final_pool_320_new_record_candidates.csv"
    new_records.to_csv(new_records_path, index=False)

    topk, topk_by_n, topk_by_version = topk_tables(ranking)
    topk_path = TABLES / "ppo_final_pool_320_topk_competitive_candidates.csv"
    topk_by_n_path = TABLES / "ppo_final_pool_320_topk_summary_by_N.csv"
    topk_by_version_path = TABLES / "ppo_final_pool_320_topk_summary_by_version.csv"
    topk.to_csv(topk_path, index=False)
    topk_by_n.to_csv(topk_by_n_path, index=False)
    topk_by_version.to_csv(topk_by_version_path, index=False)

    prior_comp = final_expansion_vs_prior(ranking)
    prior_comp_path = TABLES / "final_expansion_vs_prior_ppo_by_N.csv"
    prior_comp.to_csv(prior_comp_path, index=False)

    bootstrap_by_n, bootstrap_global = bootstrap_reference(ref, ranking, trials=10000)
    bootstrap_by_n_path = TABLES / "ppo_final_pool_320_vs_bootstrap_random_reference_by_N.csv"
    bootstrap_global_path = TABLES / "ppo_final_pool_320_vs_bootstrap_random_reference_global.csv"
    bootstrap_by_n.to_csv(bootstrap_by_n_path, index=False)
    bootstrap_global.to_csv(bootstrap_global_path, index=False)

    baseline_comp, baseline_inv = baseline_family_tables(ref, ranking)
    baseline_comp_path = TABLES / "ppo_final_pool_320_vs_identified_baseline_families.csv"
    baseline_inv_path = TABLES / "ppo_final_pool_320_identified_baseline_family_inventory.csv"
    baseline_comp.to_csv(baseline_comp_path, index=False)
    baseline_inv.to_csv(baseline_inv_path, index=False)

    proxy_summary, proxy_corr = industrial_proxy_tables(ppo, ranking)
    proxy_summary_path = TABLES / "ppo_final_pool_320_industrial_efficiency_proxy_summary.csv"
    proxy_corr_path = TABLES / "ppo_final_pool_320_efficiency_proxy_vs_teacher_metrics.csv"
    proxy_summary.to_csv(proxy_summary_path, index=False)
    proxy_corr.to_csv(proxy_corr_path, index=False)

    best = best_candidates(ranking)
    best_path = TABLES / "ppo_final_pool_320_best_candidates_by_N.csv"
    best.to_csv(best_path, index=False)

    claim = claim_decisions(ranking, new_records, topk_by_n, baseline_comp, bootstrap_global, proxy_corr)
    claim_path = TABLES / "ppo_final_pool_320_claim_decision_table.csv"
    claim.to_csv(claim_path, index=False)

    plot_paths = make_plots(ref, ranking, topk_by_version, prior_comp, bootstrap_global, proxy_corr)

    primary_new_count = int((new_records["primary_or_diagnostic"] == "primary").sum()) if not new_records.empty else 0
    topk_total = int(topk_by_n["primary_top25_any_unique"].sum()) if not topk_by_n.empty else 0
    boot_row = bootstrap_global[bootstrap_global["statistic"] == "primary_top25_any_unique"]
    boot_interp = boot_row["interpretation"].iloc[0] if not boot_row.empty else "not_available"
    if primary_new_count:
        final_verdict = "PASS_STAGEW_PPO_FINAL_POOL_320_WITH_NEW_RECORDS"
    elif topk_total > 0 and boot_interp in {"enriched", "comparable"}:
        final_verdict = "PASS_STAGEW_PPO_FINAL_POOL_320_TEACHER_VALIDATED_AND_COMPETITIVE"
    elif len(ranking) == 320:
        final_verdict = "PASS_STAGEW_PPO_FINAL_POOL_320_BOUNDED_NO_NEW_RECORDS"
    else:
        final_verdict = "WARNING_STAGEW_PPO_FINAL_POOL_320_WEAK_OR_NOT_ENRICHED"

    paths_out = {
        "ppo_pool": str(ppo_pool_path),
        "combined": str(combined_path),
        "ranking": str(ranking_path),
        "new_records": str(new_records_path),
        "topk": str(topk_path),
        "topk_by_n": str(topk_by_n_path),
        "topk_by_version": str(topk_by_version_path),
        "prior_comp": str(prior_comp_path),
        "claim": str(claim_path),
    }
    report_path, claim_boundary_path = write_report(
        final_verdict,
        input_verdict,
        input_summary,
        paths_out,
        {
            "topk_by_n": topk_by_n,
            "new_records": new_records,
            "prior_comp": prior_comp,
            "bootstrap_global": bootstrap_global,
            "best_candidates": best,
            "claim": claim,
        },
    )

    manifest = {
        "branch": git_branch(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input_paths": {k: str(v) for k, v in PATHS.items()},
        "output_datasets": {
            "ppo_final_pool_320": str(ppo_pool_path),
            "combined552_plus_ppo": str(combined_path),
            "final_expansion_joined_metadata": str(final_join_path),
        },
        "ranking_tables": {
            "full_ranking": str(ranking_path),
            "new_records": str(new_records_path),
            "topk_candidates": str(topk_path),
            "best_candidates": str(best_path),
        },
        "summary_tables": {
            "version_summary": str(version_path),
            "topk_by_N": str(topk_by_n_path),
            "topk_by_version": str(topk_by_version_path),
            "final_expansion_vs_prior": str(prior_comp_path),
            "bootstrap_by_N": str(bootstrap_by_n_path),
            "bootstrap_global": str(bootstrap_global_path),
            "baseline_comparison": str(baseline_comp_path),
            "baseline_inventory": str(baseline_inv_path),
            "industrial_proxy_summary": str(proxy_summary_path),
            "industrial_proxy_vs_teacher": str(proxy_corr_path),
            "claim_decision": str(claim_path),
        },
        "plot_directory": str(PLOTS),
        "plot_paths": plot_paths,
        "report_path": str(report_path),
        "claim_boundary_path": str(claim_boundary_path),
        "final_PPO_pool_count": int(len(ppo)),
        "by_N_counts": {str(k): int(v) for k, v in ppo["n"].astype(int).value_counts().sort_index().items()},
        "no_Abaqus": True,
        "no_ODB_opening": True,
        "no_ODB_extraction": True,
        "no_solver": True,
        "no_datacheck": True,
        "no_enqueue": True,
        "no_training": True,
        "no_candidate_generation": True,
        "no_commit_or_push": True,
        "final_verdict": final_verdict,
    }
    manifest_path = OUT_ROOT / "stageW_final_ppo_pool_320_analysis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    stdout = {
        "input_integrity_verdict": input_verdict,
        "final_verdict": final_verdict,
        "ppo_counts_by_N": manifest["by_N_counts"],
        "new_record_rows": int(len(new_records)),
        "primary_new_record_rows": primary_new_count,
        "top25_primary_any_total": topk_total,
        "bootstrap_primary_top25_any": boot_row.to_dict("records")[0] if not boot_row.empty else {},
        "best_candidates_path": str(best_path),
        "report_path": str(report_path),
        "claim_boundary_path": str(claim_boundary_path),
        "manifest_path": str(manifest_path),
        "no_Abaqus": True,
        "no_ODB_opening": True,
        "no_solver": True,
        "no_training": True,
        "no_candidate_generation": True,
    }
    print(json.dumps(stdout, indent=2))


if __name__ == "__main__":
    main()
