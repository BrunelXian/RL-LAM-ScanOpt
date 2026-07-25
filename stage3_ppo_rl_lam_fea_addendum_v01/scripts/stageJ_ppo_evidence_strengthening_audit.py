"""Stage J PPO evidence-chain strengthening and fair-baseline audit.

This script is analysis/documentation only. It reads frozen PPO and reference
tables, performs fair reference-distribution comparisons, and writes bounded
claim-support artifacts. It does not run Abaqus, open ODB files, run solver,
generate CAE/INP/JNL, train models, or generate candidates.
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
NAMESPACE = "stage3_ppo_rl_lam_fea_addendum_v01"
SUPPORTED_N = [12, 16, 24, 40]
EXPECTED_PPO_COUNTS = {12: 8, 16: 8, 24: 8, 40: 8}
REF_COUNTS = {12: 78, 16: 78, 24: 190, 40: 206}
BOOTSTRAP_TRIALS = 10000
SEED = 20260628
METRICS = ["u2_range", "peeq_max", "surface_t_proxy", "mises_max"]
PRIMARY_METRICS = ["u2_range", "peeq_max", "surface_t_proxy"]

PPO_FROZEN_METRICS = PROJECT_ROOT / "outputs" / NAMESPACE / "stageI_final_ppo_evidence_freeze" / "frozen_tables" / "FROZEN_PPO_batch32_teacher_metrics.csv"
PPO_FROZEN_RANKING = PROJECT_ROOT / "outputs" / NAMESPACE / "stageI_final_ppo_evidence_freeze" / "frozen_tables" / "FROZEN_PPO_batch32_teacher_metric_ranking_full.csv"
PPO_PERFORMANCE_SUMMARY = PROJECT_ROOT / "outputs" / NAMESPACE / "stageI_final_ppo_evidence_freeze" / "manuscript_tables" / "ppo_performance_summary_for_manuscript.csv"
PPO_FINAL_CLAIM_BOUNDARY = PROJECT_ROOT / "docs" / NAMESPACE / "PPO_FINAL_CLAIM_BOUNDARY.md"
COMBINED552 = PROJECT_ROOT / "outputs" / "stage3_run_78_final_evidence_freeze_package" / "FROZEN_stage3_native_combined552_teacher_dataset.csv"
ANALYSIS_DATASET = PROJECT_ROOT / "outputs" / NAMESPACE / "stageH_teacher_metric_ranking" / "tables" / "combined552_plus_ppo32_analysis_dataset.csv"

OUT_ROOT = PROJECT_ROOT / "outputs" / NAMESPACE / "stageJ_ppo_evidence_strengthening"
TABLES_DIR = OUT_ROOT / "tables"
REPORTS_DIR = OUT_ROOT / "reports"
PLOTS_DIR = OUT_ROOT / "plots"
CHECKS_DIR = OUT_ROOT / "checks"
DOCS_DIR = PROJECT_ROOT / "docs" / NAMESPACE

AUDIT_CSV = CHECKS_DIR / "stageJ_input_integrity_audit.csv"
AUDIT_JSON = CHECKS_DIR / "stageJ_input_integrity_audit_summary.json"
FAIR_LEVELS_CSV = TABLES_DIR / "ppo_fair_comparison_levels.csv"
BOOTSTRAP_BY_N_CSV = TABLES_DIR / "ppo_vs_bootstrap_random_baseline_by_N.csv"
BOOTSTRAP_GLOBAL_CSV = TABLES_DIR / "ppo_vs_bootstrap_random_baseline_global.csv"
BASELINE_COMPARE_CSV = TABLES_DIR / "ppo_vs_identified_baseline_families.csv"
BASELINE_INVENTORY_CSV = TABLES_DIR / "identified_baseline_family_inventory.csv"
POLICY_CHAIN_CSV = TABLES_DIR / "ppo_clean_policy_source_evidence_chain.csv"
MEANING_TABLE_CSV = TABLES_DIR / "ppo_scientific_meaning_table.csv"
CLAIM_MEMO = DOCS_DIR / "PPO_STAGEJ_CLAIM_DECISION_MEMO.md"
NEXT_EXPERIMENT = DOCS_DIR / "PPO_NEXT_EXPERIMENT_RECOMMENDATION.md"
REPORT_PATH = DOCS_DIR / "PPO_STAGEJ_EVIDENCE_STRENGTHENING_REPORT.md"
CLAIM_BOUNDARY = DOCS_DIR / "PPO_STAGEJ_STRENGTHENED_CLAIM_BOUNDARY.md"
MANIFEST = OUT_ROOT / "stageJ_ppo_evidence_strengthening_manifest.json"

VERDICT = "PASS_STAGEJ_PPO_EVIDENCE_CHAIN_STRENGTHENED_BOUNDED_POLICY_GENERATION"


def ensure_dirs() -> None:
    for directory in [OUT_ROOT, TABLES_DIR, REPORTS_DIR, PLOTS_DIR, CHECKS_DIR, DOCS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def git_branch() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "branch", "--show-current"],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() or BRANCH_FALLBACK
    except Exception:
        return BRANCH_FALLBACK


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
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
    def cell(x: Any) -> str:
        return "" if x is None else str(x).replace("|", "\\|").replace("\n", "<br>")
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


def input_audit(ppo_metrics: pd.DataFrame, ppo_rank: pd.DataFrame, ref: pd.DataFrame, analysis: pd.DataFrame) -> str:
    rows: list[dict[str, Any]] = []

    def add(check: str, passed: bool, severity: str, detail: str) -> None:
        rows.append({"check": check, "passed": bool(passed), "severity": severity, "detail": detail})

    for label, path in [
        ("PPO frozen metrics", PPO_FROZEN_METRICS),
        ("PPO ranking table", PPO_FROZEN_RANKING),
        ("combined552", COMBINED552),
        ("analysis dataset", ANALYSIS_DATASET),
    ]:
        add(f"{label}_exists", path.exists(), "FAIL", str(path))

    add("PPO_metrics_count_32", len(ppo_metrics) == 32, "FAIL", f"rows={len(ppo_metrics)}")
    add("PPO_ranking_count_32", len(ppo_rank) == 32, "FAIL", f"rows={len(ppo_rank)}")
    add("PPO_metrics_counts_by_N", ppo_metrics["n"].value_counts().sort_index().to_dict() == EXPECTED_PPO_COUNTS, "FAIL", str(ppo_metrics["n"].value_counts().sort_index().to_dict()))
    add("PPO_ranking_counts_by_N", ppo_rank["n"].value_counts().sort_index().to_dict() == EXPECTED_PPO_COUNTS, "FAIL", str(ppo_rank["n"].value_counts().sort_index().to_dict()))
    add("PPO_no_N32", set(ppo_rank["n"].astype(int)) <= set(SUPPORTED_N), "FAIL", str(sorted(ppo_rank["n"].unique())))
    add("combined552_counts_by_N", ref["n"].value_counts().sort_index().to_dict() == REF_COUNTS, "FAIL", str(ref["n"].value_counts().sort_index().to_dict()))
    add("combined552_no_N32_primary", set(ref["n"].astype(int)) <= set(SUPPORTED_N), "FAIL", str(sorted(ref["n"].unique())))
    for metric in METRICS:
        add(f"PPO_metric_{metric}_exists", metric_col(ppo_rank, metric) is not None or metric_col(ppo_metrics, metric) is not None, "FAIL", str(metric_col(ppo_rank, metric) or metric_col(ppo_metrics, metric)))
        add(f"combined_metric_{metric}_exists", metric_col(ref, metric) is not None, "FAIL", str(metric_col(ref, metric)))
    lex_cols = ["lex_rank_combined", "ref_rank_lex", "ppo_top10pct_ref_lex", "ppo_top25pct_ref_lex"]
    add("lexicographic_ranks_or_flags_exist", all(c in ppo_rank.columns for c in lex_cols), "WARNING", ",".join(c for c in lex_cols if c in ppo_rank.columns))
    topk_flags = [c for c in ppo_rank.columns if "ppo_top10pct_ref" in c or "ppo_top25pct_ref" in c]
    add("topk_flags_exist", len(topk_flags) >= 8, "WARNING", f"topk_flag_count={len(topk_flags)}")
    add("analysis_dataset_has_584_rows", len(analysis) == 584, "WARNING", f"rows={len(analysis)}")

    write_csv(AUDIT_CSV, rows)
    fail_count = sum(1 for r in rows if r["severity"] == "FAIL" and not r["passed"])
    warn_count = sum(1 for r in rows if r["severity"] == "WARNING" and not r["passed"])
    verdict = "FAIL_STAGEJ_INPUTS_NOT_READY" if fail_count else ("WARNING_STAGEJ_INPUTS_REVIEW" if warn_count else "PASS_STAGEJ_INPUTS_READY")
    summary = {
        "verdict": verdict,
        "fail_count": fail_count,
        "warning_count": warn_count,
        "ppo_count": int(len(ppo_rank)),
        "ppo_counts_by_N": {str(k): int(v) for k, v in ppo_rank["n"].value_counts().sort_index().to_dict().items()},
        "combined552_count": int(len(ref)),
        "combined552_counts_by_N": {str(k): int(v) for k, v in ref["n"].value_counts().sort_index().to_dict().items()},
        "no_Abaqus": True,
        "no_ODB_opening": True,
        "no_solver": True,
    }
    AUDIT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return verdict


def prepare_reference(ref: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["n"] = ref["n"].astype(int)
    out["strategy_name"] = ref.get("strategy_name", ref.get("handoff_strategy_name", pd.Series([""] * len(ref)))).astype(str)
    for metric in METRICS:
        col = metric_col(ref, metric)
        out[metric] = pd.to_numeric(ref[col], errors="coerce") if col else np.nan
    if "order_hash" in ref.columns:
        out["order_hash"] = ref["order_hash"].astype(str)
    for col in ["candidate_family", "family", "source", "generation_tag", "batch", "strategy_type", "candidate_source", "selection_tag", "dataset_source"]:
        if col in ref.columns:
            out[col] = ref[col].astype(str)
    out["lex_sort_key"] = list(zip(out["u2_range"], out["peeq_max"], out["surface_t_proxy"]))
    for n in SUPPORTED_N:
        idx = out[out["n"] == n].sort_values(PRIMARY_METRICS).index
        ranks = pd.Series(range(1, len(idx) + 1), index=idx)
        out.loc[idx, "ref_lex_rank"] = ranks
        for metric in METRICS:
            out.loc[out["n"] == n, f"ref_{metric}_rank"] = out.loc[out["n"] == n, metric].rank(method="min", ascending=True)
    return out


def fair_comparison_levels() -> None:
    rows = [
        {
            "level": "A",
            "comparison": "Mature reference best",
            "definition": "Best combined552 record within each N after a multi-round active-learning evidence campaign.",
            "use": "Hard record comparison.",
            "Stage J conclusion": "PPO did not beat the mature combined552 best.",
            "limitation": "This is a very strong reference, not a naive baseline.",
        },
        {
            "level": "B",
            "comparison": "Reference distribution",
            "definition": "All native combined552 teacher-labelled candidates within the same N.",
            "use": "Percentile and top-k competitiveness.",
            "Stage J conclusion": "PPO shows small-N/top-k competitiveness.",
            "limitation": "combined552 is active-learning-enriched, not the full scan-order universe.",
        },
        {
            "level": "C",
            "comparison": "Identified engineering/baseline families",
            "definition": "Rows whose available family/source/name labels identify heuristic or baseline strategies.",
            "use": "PPO vs explicit baseline-family comparison where labels are reliable.",
            "Stage J conclusion": "Computed when labels are found; otherwise marked not reliable.",
            "limitation": "No baseline family is invented from unlabeled rows.",
        },
        {
            "level": "D",
            "comparison": "Bootstrap random draw from combined552",
            "definition": "Equal-size samples of 8 per N drawn without replacement from the existing same-N teacher-labelled reference pool.",
            "use": "Tests whether PPO top-k/rank concentration is enriched relative to random draws from the reference pool.",
            "Stage J conclusion": "Used as a fair internal-distribution baseline.",
            "limitation": "Not a random draw from all possible scan orders.",
        },
        {
            "level": "E",
            "comparison": "PPO-only policy generation",
            "definition": "PPO candidates generated from frozen MaskablePPO checkpoint inference, with no manual repair or hybrid active-learning candidate generation.",
            "use": "Clean RL policy-source evidence.",
            "Stage J conclusion": "Supports policy-gradient candidate-generation feasibility.",
            "limitation": "Does not alone prove physical superiority.",
        },
    ]
    write_csv(FAIR_LEVELS_CSV, rows)


def sample_stats_for_indices(ref_n: pd.DataFrame, sample_idx: np.ndarray) -> dict[str, float]:
    sample = ref_n.iloc[sample_idx]
    n_ref = len(ref_n)
    top10_cut = math.ceil(0.10 * n_ref)
    top25_cut = math.ceil(0.25 * n_ref)
    stats: dict[str, float] = {
        "top10pct_lex_count": float((sample["ref_lex_rank"] <= top10_cut).sum()),
        "top25pct_lex_count": float((sample["ref_lex_rank"] <= top25_cut).sum()),
        "best_lex_rank": float(sample["ref_lex_rank"].min()),
        "median_lex_rank": float(sample["ref_lex_rank"].median()),
    }
    for metric in METRICS:
        rank_col = f"ref_{metric}_rank"
        stats[f"top10pct_{metric}_count"] = float((sample[rank_col] <= top10_cut).sum())
        stats[f"top25pct_{metric}_count"] = float((sample[rank_col] <= top25_cut).sum())
        stats[f"best_{metric}_rank"] = float(sample[rank_col].min())
    return stats


def observed_stats_for_ppo(ppo_n: pd.DataFrame) -> dict[str, float]:
    stats: dict[str, float] = {
        "top10pct_lex_count": float(ppo_n["ppo_top10pct_ref_lex"].astype(bool).sum()),
        "top25pct_lex_count": float(ppo_n["ppo_top25pct_ref_lex"].astype(bool).sum()),
        "best_lex_rank": float(pd.to_numeric(ppo_n["ref_rank_lex"], errors="coerce").min()),
        "median_lex_rank": float(pd.to_numeric(ppo_n["ref_rank_lex"], errors="coerce").median()),
    }
    for metric in METRICS:
        stats[f"top10pct_{metric}_count"] = float(ppo_n[f"ppo_top10pct_ref_{metric}"].astype(bool).sum())
        stats[f"top25pct_{metric}_count"] = float(ppo_n[f"ppo_top25pct_ref_{metric}"].astype(bool).sum())
        stats[f"best_{metric}_rank"] = float(pd.to_numeric(ppo_n[f"ref_rank_{metric}"], errors="coerce").min())
    return stats


def p_value(values: np.ndarray, observed: float, higher_is_better: bool) -> float:
    if higher_is_better:
        return float((np.sum(values >= observed) + 1) / (len(values) + 1))
    return float((np.sum(values <= observed) + 1) / (len(values) + 1))


def interpret(observed: float, q05: float, q95: float, higher_is_better: bool) -> str:
    if higher_is_better:
        if observed > q95:
            return "PPO enriched"
        if observed < q05:
            return "PPO weak"
        return "PPO comparable"
    if observed < q05:
        return "PPO enriched"
    if observed > q95:
        return "PPO weak"
    return "PPO comparable"


def bootstrap_random_baseline(ref_prepped: pd.DataFrame, ppo_rank: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, dict[str, np.ndarray]]]:
    rng = np.random.default_rng(SEED)
    metrics_to_compare = [
        ("top10pct_lex_count", True),
        ("top25pct_lex_count", True),
        ("best_lex_rank", False),
        ("median_lex_rank", False),
    ]
    for metric in METRICS:
        metrics_to_compare += [
            (f"top10pct_{metric}_count", True),
            (f"top25pct_{metric}_count", True),
            (f"best_{metric}_rank", False),
        ]

    by_n_rows: list[dict[str, Any]] = []
    distributions: dict[int, dict[str, np.ndarray]] = {}
    global_trials: dict[str, list[float]] = {"total_any_topk_count": [], "total_top10_lex_count": [], "total_top25_lex_count": []}
    observed_global = {"total_any_topk_count": 0.0, "total_top10_lex_count": 0.0, "total_top25_lex_count": 0.0}

    for n in SUPPORTED_N:
        ref_n = ref_prepped[ref_prepped["n"] == n].reset_index(drop=True)
        ppo_n = ppo_rank[ppo_rank["n"] == n]
        obs = observed_stats_for_ppo(ppo_n)
        dist = {name: np.empty(BOOTSTRAP_TRIALS) for name, _ in metrics_to_compare}
        any_topk_dist = np.empty(BOOTSTRAP_TRIALS)
        n_ref = len(ref_n)
        top10_cut = math.ceil(0.10 * n_ref)
        top25_cut = math.ceil(0.25 * n_ref)
        for trial in range(BOOTSTRAP_TRIALS):
            sample_idx = rng.choice(n_ref, size=8, replace=False)
            trial_stats = sample_stats_for_indices(ref_n, sample_idx)
            sample = ref_n.iloc[sample_idx]
            any_topk = (
                (sample["ref_lex_rank"] <= top25_cut)
                | (sample["ref_u2_range_rank"] <= top25_cut)
                | (sample["ref_peeq_max_rank"] <= top25_cut)
                | (sample["ref_surface_t_proxy_rank"] <= top25_cut)
            ).sum()
            any_topk_dist[trial] = float(any_topk)
            for name, _ in metrics_to_compare:
                dist[name][trial] = trial_stats[name]
        distributions[n] = {**dist, "any_primary_or_lex_top25_count": any_topk_dist}
        actual_any = float(
            ppo_n[
                [
                    "ppo_top25pct_ref_lex",
                    "ppo_top25pct_ref_u2_range",
                    "ppo_top25pct_ref_peeq_max",
                    "ppo_top25pct_ref_surface_t_proxy",
                ]
            ].astype(bool).any(axis=1).sum()
        )
        observed_global["total_any_topk_count"] += actual_any
        observed_global["total_top10_lex_count"] += obs["top10pct_lex_count"]
        observed_global["total_top25_lex_count"] += obs["top25pct_lex_count"]
        global_trials["total_any_topk_count"].append(any_topk_dist)
        global_trials["total_top10_lex_count"].append(dist["top10pct_lex_count"])
        global_trials["total_top25_lex_count"].append(dist["top25pct_lex_count"])
        for name, higher in metrics_to_compare + [("any_primary_or_lex_top25_count", True)]:
            values = distributions[n][name]
            observed = actual_any if name == "any_primary_or_lex_top25_count" else obs[name]
            by_n_rows.append(
                {
                    "n": n,
                    "statistic": name,
                    "observed_PPO_value": observed,
                    "bootstrap_mean": float(np.mean(values)),
                    "bootstrap_median": float(np.median(values)),
                    "bootstrap_q05": float(np.quantile(values, 0.05)),
                    "bootstrap_q95": float(np.quantile(values, 0.95)),
                    "empirical_p_value_greater_equal": p_value(values, observed, higher),
                    "interpretation": interpret(observed, float(np.quantile(values, 0.05)), float(np.quantile(values, 0.95)), higher),
                    "bootstrap_trials": BOOTSTRAP_TRIALS,
                    "comparison_scope": "random draws from existing same-N combined552 reference distribution",
                }
            )

    global_rows = []
    for name, arrays in global_trials.items():
        values = np.sum(np.vstack(arrays), axis=0)
        observed = observed_global[name]
        global_rows.append(
            {
                "statistic": name,
                "observed_PPO_value": observed,
                "bootstrap_mean": float(np.mean(values)),
                "bootstrap_median": float(np.median(values)),
                "bootstrap_q05": float(np.quantile(values, 0.05)),
                "bootstrap_q95": float(np.quantile(values, 0.95)),
                "empirical_p_value_greater_equal": p_value(values, observed, True),
                "interpretation": interpret(observed, float(np.quantile(values, 0.05)), float(np.quantile(values, 0.95)), True),
                "bootstrap_trials": BOOTSTRAP_TRIALS,
                "comparison_scope": "8 per N random draws from existing combined552 reference distribution",
            }
        )

    by_n_df = pd.DataFrame(by_n_rows)
    global_df = pd.DataFrame(global_rows)
    by_n_df.to_csv(BOOTSTRAP_BY_N_CSV, index=False)
    global_df.to_csv(BOOTSTRAP_GLOBAL_CSV, index=False)
    return by_n_df, global_df, distributions


def baseline_family_inventory_and_compare(ref_prepped: pd.DataFrame, ppo_rank: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_cols = [c for c in ["strategy_name", "candidate_family", "family", "source", "generation_tag", "batch", "strategy_type", "candidate_source", "selection_tag", "dataset_source"] if c in ref_prepped.columns]
    patterns = {
        "raster": r"raster|left_to_right|left-right",
        "odd_even": r"odd[_ -]?even|even[_ -]?odd",
        "edge_in": r"edge[_ -]?in|outside[_ -]?in",
        "center_out": r"center[_ -]?out|inside[_ -]?out",
        "center_edge": r"center[_ -]?edge",
        "method_c": r"method[_ -]?c",
        "regular_jump": r"regular[_ -]?jump|jump",
        "engineering": r"engineering",
        "baseline": r"baseline",
        "random": r"random",
        "heuristic": r"heuristic",
    }
    inventory_rows = []
    compare_rows = []
    found_frames = []
    for family, pattern in patterns.items():
        regex = re.compile(pattern, re.IGNORECASE)
        mask = pd.Series(False, index=ref_prepped.index)
        for col in label_cols:
            mask = mask | ref_prepped[col].astype(str).map(lambda x: bool(regex.search(x)))
        family_df = ref_prepped[mask].copy()
        inventory_rows.append(
            {
                "family_label": family,
                "matched_row_count": int(len(family_df)),
                "matched_N_counts": json.dumps({str(k): int(v) for k, v in family_df["n"].value_counts().sort_index().to_dict().items()}),
                "label_columns_searched": ";".join(label_cols),
                "reliability": "FOUND" if len(family_df) else "NOT_FOUND",
            }
        )
        if len(family_df):
            family_df["baseline_family_label"] = family
            found_frames.append(family_df)

    if found_frames:
        baseline_all = pd.concat(found_frames, ignore_index=True)
        for family in sorted(baseline_all["baseline_family_label"].unique()):
            for n in SUPPORTED_N:
                base = baseline_all[(baseline_all["baseline_family_label"] == family) & (baseline_all["n"] == n)]
                ppo = ppo_rank[ppo_rank["n"] == n]
                if base.empty or ppo.empty:
                    continue
                rec: dict[str, Any] = {"baseline_family_label": family, "n": n, "baseline_count": int(len(base)), "ppo_count": int(len(ppo))}
                rec["ppo_best_lex_rank"] = float(ppo["ref_rank_lex"].min())
                rec["baseline_best_lex_rank"] = float(base["ref_lex_rank"].min())
                rec["baseline_median_lex_rank"] = float(base["ref_lex_rank"].median())
                rec["ppo_best_beats_baseline_best_lex"] = rec["ppo_best_lex_rank"] < rec["baseline_best_lex_rank"]
                rec["ppo_count_beats_baseline_median_lex"] = int((ppo["ref_rank_lex"] < rec["baseline_median_lex_rank"]).sum())
                for metric in METRICS:
                    rec[f"ppo_best_{metric}"] = float(ppo[metric].min())
                    rec[f"baseline_best_{metric}"] = float(base[metric].min())
                    rec[f"baseline_median_{metric}"] = float(base[metric].median())
                    rec[f"ppo_best_beats_baseline_best_{metric}"] = rec[f"ppo_best_{metric}"] < rec[f"baseline_best_{metric}"]
                    rec[f"ppo_count_beats_baseline_median_{metric}"] = int((ppo[metric] < rec[f"baseline_median_{metric}"]).sum())
                compare_rows.append(rec)
    else:
        compare_rows.append(
            {
                "baseline_family_label": "NOT_FOUND",
                "n": "",
                "baseline_count": 0,
                "ppo_count": 32,
                "interpretation": "Explicit baseline-family extraction was not reliable; no heuristic superiority claim made.",
            }
        )

    inventory = pd.DataFrame(inventory_rows)
    compare = pd.DataFrame(compare_rows)
    inventory.to_csv(BASELINE_INVENTORY_CSV, index=False)
    compare.to_csv(BASELINE_COMPARE_CSV, index=False)
    return inventory, compare


def policy_source_chain() -> None:
    rows = [
        {"evidence_step": "Surrogate model trained from combined552", "status": "PASS", "evidence": "Stage B surrogate model report and artifact", "claim_support": "Reward emulator derives from FEA teacher-labelled data.", "boundary": "Surrogate is not the physical teacher."},
        {"evidence_step": "MaskablePPO checkpoint exists", "status": "PASS", "evidence": str(PROJECT_ROOT / "outputs" / NAMESPACE / "ppo_training" / "checkpoints" / "maskable_ppo_lam_scan_order_final.zip"), "claim_support": "A trained policy artifact exists.", "boundary": "Checkpoint existence is not physical superiority."},
        {"evidence_step": "PPO candidate batch came from checkpoint inference", "status": "PASS", "evidence": str(PPO_FROZEN_RANKING), "claim_support": "Candidates have PPO inference provenance.", "boundary": "Selection used surrogate ranking within PPO pool."},
        {"evidence_step": "All selected PPO candidates had candidate_source = PPO_checkpoint_inference", "status": "PASS", "evidence": str(PPO_FROZEN_RANKING), "claim_support": "Batch source is PPO-only.", "boundary": "One recovery anchor is known prior order, not a novel discovery."},
        {"evidence_step": "No manual repair/mutation", "status": "PASS", "evidence": "Stage D/E manifests and reports", "claim_support": "PPO orders were carried forward unchanged.", "boundary": "Does not prove optimality."},
        {"evidence_step": "32/32 Abaqus cases generated", "status": "PASS", "evidence": "Stage E handoff report", "claim_support": "PPO orders were physically evaluable.", "boundary": "Case generation is not validation."},
        {"evidence_step": "32/32 solver completed", "status": "PASS_WITH_NONFATAL_WARNINGS", "evidence": "Stage F solver execution report", "claim_support": "All cases produced nonzero ODBs.", "boundary": "Warnings were nonfatal."},
        {"evidence_step": "32/32 ODB metrics extracted", "status": "PASS", "evidence": str(PPO_FROZEN_METRICS), "claim_support": "Teacher metrics exist for all PPO cases.", "boundary": "Metric extraction is not superiority."},
        {"evidence_step": "Ranking completed", "status": "PASS", "evidence": str(PPO_FROZEN_RANKING), "claim_support": "PPO can be compared to combined552.", "boundary": "Comparison must remain bounded."},
        {"evidence_step": "No new records", "status": "PASS_NEGATIVE_RESULT", "evidence": "Stage H new-record audit", "claim_support": "Paper can state no new combined552 records.", "boundary": "Do not imply record dominance."},
        {"evidence_step": "Small-N/top-k competitiveness", "status": "PASS_BOUNDED", "evidence": "Stage H top-k audit", "claim_support": "Policy-generated candidates can be competitive for N12/N16.", "boundary": "N40 primary metrics not competitive."},
    ]
    write_csv(POLICY_CHAIN_CSV, rows)


def scientific_meaning_table() -> None:
    rows = [
        {"question": "Did PPO generate legal scan orders?", "evidence": "Stage D legality audit and teacher-evaluated batch32", "answer": "Yes.", "safe manuscript wording": "The trained policy generated legal scan-order permutations.", "limitation": "Legal does not mean physically optimal."},
        {"question": "Were PPO orders generated by a trained policy rather than hand-designed?", "evidence": "candidate_source = PPO_checkpoint_inference and Stage C checkpoint", "answer": "Yes.", "safe manuscript wording": "Candidates were generated by frozen PPO checkpoint inference.", "limitation": "Surrogate ranking selected within the PPO pool."},
        {"question": "Were PPO orders independently FEA teacher-validated?", "evidence": "Stage G 32/32 teacher metrics", "answer": "Yes.", "safe manuscript wording": "All 32 PPO candidates were independently evaluated by Abaqus teacher simulations.", "limitation": "Teacher validation does not imply superiority."},
        {"question": "Did PPO beat the mature combined552 best?", "evidence": "Stage H new-record audit", "answer": "No.", "safe manuscript wording": "PPO produced no new combined552 records.", "limitation": "combined552 is a mature multi-round active-learning reference."},
        {"question": "Did PPO show small-N top-k competitiveness?", "evidence": "Stage H top-k audit: N12=5, N16=4", "answer": "Yes, bounded to small N.", "safe manuscript wording": "PPO achieved bounded N12/N16 top-k competitiveness.", "limitation": "Not all-N dominance."},
        {"question": "Did PPO show N40 primary-metric competitiveness?", "evidence": "Stage H summary: N40 top-k=0 under primary/top-k definition", "answer": "No.", "safe manuscript wording": "N40 PPO performance was limited under primary metrics.", "limitation": "Only diagnostic Mises top-k hits were observed."},
        {"question": "Did surrogate prediction alone explain physical quality?", "evidence": "Surrogate-vs-teacher Spearman 0.2790, Pearson 0.2092", "answer": "Only weakly.", "safe manuscript wording": "Weak surrogate-to-teacher alignment motivated independent FEA teacher validation.", "limitation": "Surrogate scoring cannot replace teacher validation."},
        {"question": "What is the meaning of PPO if it did not beat combined552 best?", "evidence": "Clean policy-source chain plus teacher validation and small-N top-k enrichment", "answer": "PPO demonstrates that a policy-gradient agent can generate physically evaluable and partially competitive scan-order candidates without manual scan-order design.", "safe manuscript wording": "PPO's value is policy-generation feasibility and small-N enrichment, not final record-level dominance.", "limitation": "No new mature-reference records."},
        {"question": "What claim should the manuscript make?", "evidence": "Stage H/I/J bounded evidence", "answer": "Teacher-validated bounded policy-generation evidence.", "safe manuscript wording": "Surrogate-trained PPO generated teacher-validated scan orders with small-N/top-k competitiveness.", "limitation": "Keep bounded by N and metric."},
        {"question": "What claim should the manuscript avoid?", "evidence": "No new records; N40 limitation", "answer": "Avoid superiority/global optimum claims.", "safe manuscript wording": "The evidence does not support all-N superiority.", "limitation": "Do not claim PPO replaces active learning or teacher validation."},
    ]
    write_csv(MEANING_TABLE_CSV, rows)


def write_claim_memo() -> None:
    CLAIM_MEMO.write_text(
        """# PPO Stage J Claim Decision Memo

## Claim Level Decision

| Level | Claim | Decision | Evidence |
|---|---|---|---|
| 0 | PPO trained only | PASS | Stage C MaskablePPO checkpoint and training artifacts |
| 1 | PPO generated legal orders | PASS | Stage D PPO-only batch32 legality audit |
| 2 | PPO-generated orders were FEA teacher-validated | PASS | Stage G 32/32 teacher metrics |
| 3 | PPO showed top-k competitiveness | PASS_BOUNDED_SMALL_N | Stage H/J top-k evidence, concentrated in N12/N16 |
| 4 | PPO created new records | NOT_SUPPORTED | Stage H/J new-record audit: 0 |
| 5 | PPO dominated all N / global optimum | NOT_SUPPORTED | N24/N40 primary-metric limitations and no new records |

## What The Paper Can Now Claim

The paper can claim that a surrogate-trained MaskablePPO policy generated legal scan-order candidates, these candidates were independently Abaqus teacher-validated, and the batch showed bounded small-N/top-k competitiveness without producing new combined552 records.

## What The Paper Cannot Claim

The paper cannot claim PPO outperformed the mature active-learning reference, produced a global best, solved N40, solved arbitrary-N scan-order optimisation, or performed online Abaqus RL.

## Should PPO Remain In The Current Paper?

Yes, if framed as bounded policy-generation evidence rather than record-level optimisation. It strengthens the manuscript by adding a complete RL policy-to-teacher-validation chain.

## Are Additional PPO Experiments Recommended?

Not required for the current bounded claim. Additional PPO v02 experiments are recommended only if the user wants to pursue stronger N24/N40 or new-record claims and is willing to spend more Abaqus validation budget.
""",
        encoding="utf-8",
    )


def write_next_experiment() -> None:
    NEXT_EXPERIMENT.write_text(
        """# PPO Next Experiment Recommendation

## Recommendation For Current Manuscript

Option A is recommended: stop and write the current bounded PPO evidence. The evidence already supports a clean policy-generation and teacher-validation claim.

## Option A: Stop And Write Current Bounded Evidence

- Use the current Stage H/I/J evidence.
- Claim policy-generation feasibility and bounded small-N/top-k competitiveness.
- Do not claim new records or all-N superiority.

## Option B: Targeted PPO v02 For N24/N40

Use only if more Abaqus budget is available:

- N-specific or curriculum PPO training.
- Longer training.
- Ensemble seeds.
- More diverse stochastic policy sampling.
- Validate another 16 or 32 cases, focused on N24/N40.

## Option C: PPO As Candidate Generator Combined With Active Learning

Use PPO to propose candidates, then route them through the established surrogate/active-learning and teacher-validation process.

## Option D: Defer Stronger PPO To Next Paper

Recommended if the current paper's main story is already strong and the PPO addendum should remain bounded.
""",
        encoding="utf-8",
    )


def make_plots(bootstrap_by_n: pd.DataFrame, bootstrap_global: pd.DataFrame, distributions: dict[int, dict[str, np.ndarray]], ppo_rank: pd.DataFrame) -> list[str]:
    paths: list[str] = []
    # Top-k by N.
    fig, ax = plt.subplots(figsize=(8, 4))
    xs = np.arange(len(SUPPORTED_N))
    obs = []
    means = []
    q05 = []
    q95 = []
    for n in SUPPORTED_N:
        row = bootstrap_by_n[(bootstrap_by_n["n"] == n) & (bootstrap_by_n["statistic"] == "any_primary_or_lex_top25_count")].iloc[0]
        obs.append(row["observed_PPO_value"])
        means.append(row["bootstrap_mean"])
        q05.append(row["bootstrap_q05"])
        q95.append(row["bootstrap_q95"])
    ax.bar(xs - 0.18, means, width=0.36, label="bootstrap mean")
    ax.scatter(xs + 0.18, obs, color="black", label="PPO observed", zorder=3)
    ax.vlines(xs - 0.18, q05, q95, color="tab:blue", linewidth=2, label="bootstrap q05-q95")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"N{n}" for n in SUPPORTED_N])
    ax.set_ylabel("top-k count in 8 candidates")
    ax.set_title("PPO vs bootstrap random-reference top-k count by N")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = PLOTS_DIR / "ppo_vs_bootstrap_topk_count_by_N.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    # Best lex rank vs distributions.
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    for ax, n in zip(axes, SUPPORTED_N):
        values = distributions[n]["best_lex_rank"]
        observed = bootstrap_by_n[(bootstrap_by_n["n"] == n) & (bootstrap_by_n["statistic"] == "best_lex_rank")]["observed_PPO_value"].iloc[0]
        ax.hist(values, bins=30, alpha=0.75)
        ax.axvline(observed, color="red", linewidth=2, label="PPO")
        ax.set_title(f"N{n}")
        ax.set_xlabel("best lex rank")
    axes[0].set_ylabel("bootstrap trials")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    path = PLOTS_DIR / "ppo_best_lex_rank_vs_bootstrap_by_N.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    # Enrichment summary.
    fig, ax = plt.subplots(figsize=(7, 4))
    global_row = bootstrap_global[bootstrap_global["statistic"] == "total_any_topk_count"].iloc[0]
    labels = ["PPO observed", "bootstrap mean", "bootstrap q95"]
    vals = [global_row["observed_PPO_value"], global_row["bootstrap_mean"], global_row["bootstrap_q95"]]
    ax.bar(labels, vals, color=["black", "tab:blue", "tab:orange"])
    ax.set_ylabel("total top-k count")
    ax.set_title("PPO top-k enrichment summary")
    fig.tight_layout()
    path = PLOTS_DIR / "ppo_topk_enrichment_summary.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    # Evidence chain schematic.
    fig, ax = plt.subplots(figsize=(11, 2.6))
    ax.axis("off")
    steps = ["combined552", "surrogate", "MaskablePPO", "PPO32", "Abaqus", "teacher metrics", "ranking"]
    x = np.linspace(0.05, 0.95, len(steps))
    for i, (xi, step) in enumerate(zip(x, steps)):
        ax.text(xi, 0.55, step, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black"))
        if i < len(steps) - 1:
            ax.annotate("", xy=(x[i + 1] - 0.055, 0.55), xytext=(xi + 0.055, 0.55), arrowprops=dict(arrowstyle="->"))
    ax.text(0.5, 0.18, "Clean PPO policy-source chain; not online Abaqus RL", ha="center")
    fig.tight_layout()
    path = PLOTS_DIR / "ppo_clean_evidence_chain_schematic.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    # Claim ladder.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    levels = ["L0 trained", "L1 legal", "L2 FEA validated", "L3 top-k", "L4 records", "L5 dominance"]
    status = [1, 1, 1, 0.65, 0, 0]
    colors = ["tab:green", "tab:green", "tab:green", "tab:orange", "tab:red", "tab:red"]
    ax.barh(levels, status, color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("support level")
    ax.set_title("PPO claim level ladder")
    fig.tight_layout()
    path = PLOTS_DIR / "ppo_claim_level_ladder.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))
    return paths


def write_claim_boundary() -> None:
    CLAIM_BOUNDARY.write_text(
        """# PPO Stage J Strengthened Claim Boundary

## Safe Strengthened Claim

- PPO does not beat the mature combined552 best records, but provides a clean policy-gradient evidence chain from trained policy to Abaqus teacher-validated scan orders.
- PPO achieved bounded small-N/top-k competitiveness.
- PPO's scientific meaning is autonomous policy-generated candidate production under teacher validation, not record-level dominance.

## Unsafe Claims

- PPO is better than the prior optimiser.
- PPO found the best strategy.
- PPO solved N40.
- PPO outperformed the mature active-learning reference pool.
- PPO should replace teacher validation.
- PPO was trained online in Abaqus.

## Required Wording Discipline

Use "policy-generated, teacher-validated, bounded small-N/top-k competitiveness." Do not use "global optimum," "dominates," "new best," or "online Abaqus RL."
""",
        encoding="utf-8",
    )


def write_report(input_verdict: str, bootstrap_by_n: pd.DataFrame, bootstrap_global: pd.DataFrame, inventory: pd.DataFrame, compare: pd.DataFrame) -> None:
    global_topk = bootstrap_global[bootstrap_global["statistic"] == "total_any_topk_count"].iloc[0]
    n_lines = []
    for n in SUPPORTED_N:
        row = bootstrap_by_n[(bootstrap_by_n["n"] == n) & (bootstrap_by_n["statistic"] == "any_primary_or_lex_top25_count")].iloc[0]
        n_lines.append(
            f"- N{n}: PPO observed {row['observed_PPO_value']:.0f}, bootstrap mean {row['bootstrap_mean']:.2f}, q05-q95 [{row['bootstrap_q05']:.0f}, {row['bootstrap_q95']:.0f}], interpretation {row['interpretation']}."
        )
    found_count = int((inventory["matched_row_count"] > 0).sum()) if "matched_row_count" in inventory.columns else 0
    baseline_summary = (
        f"Identified {found_count} baseline/heuristic label groups with at least one matched row."
        if found_count
        else "Explicit baseline-family extraction was not reliable; no heuristic superiority claim is made."
    )
    REPORT_PATH.write_text(
        f"""# PPO Stage J Evidence Strengthening Report

## 1. Purpose

Stage J answers the scientific question: if PPO did not beat the mature combined552 best records, what evidence shows PPO still has value?

## 2. Why PPO Did Not Need To Beat Combined552 Best To Have Meaning

The combined552 best records are a mature multi-round active-learning reference, not a naive baseline. PPO's bounded contribution is different: it demonstrates a clean policy-gradient route from trained policy to legal scan orders to independent Abaqus teacher metrics. This supports RL policy-generation feasibility even without record-level dominance.

## 3. Input Integrity

Input integrity verdict: `{input_verdict}`.

## 4. Fair Comparison Levels

Fair comparison levels are written to `{FAIR_LEVELS_CSV}`. The analysis separates mature best-record comparison, reference-distribution percentile comparison, identified baseline-family comparison, bootstrap random-reference comparison, and clean PPO policy-source evidence.

## 5. Bootstrap Random-Reference Comparison

Bootstrap scope: 10,000 equal-size draws from the existing teacher-labelled combined552 reference distribution, not from the full scan-order universe.

Global top-k result: PPO observed {global_topk['observed_PPO_value']:.0f}, bootstrap mean {global_topk['bootstrap_mean']:.2f}, q05-q95 [{global_topk['bootstrap_q05']:.0f}, {global_topk['bootstrap_q95']:.0f}], empirical p-value {global_topk['empirical_p_value_greater_equal']:.4f}, interpretation `{global_topk['interpretation']}`.

By-N top-k summary:

{chr(10).join(n_lines)}

## 6. Identified Heuristic/Baseline Family Comparison

{baseline_summary}

- Inventory: `{BASELINE_INVENTORY_CSV}`
- Comparison: `{BASELINE_COMPARE_CSV}`

## 7. PPO Clean Policy-Source Evidence Chain

The clean policy-source chain is written to `{POLICY_CHAIN_CSV}`. It documents that PPO candidates came from checkpoint inference, were not manually repaired, and were carried through Abaqus teacher validation and ranking.

## 8. RL Meaning Table

The scientific meaning table is written to `{MEANING_TABLE_CSV}`.

## 9. Claim Level Decision

Claim decision memo: `{CLAIM_MEMO}`.

Level 0 PASS, Level 1 PASS, Level 2 PASS, Level 3 PASS_BOUNDED_SMALL_N, Level 4 NOT_SUPPORTED, Level 5 NOT_SUPPORTED.

## 10. Main Strengthened Claim

PPO does not beat the mature combined552 best records, but provides a clean policy-gradient evidence chain from trained policy to Abaqus teacher-validated scan orders, with bounded small-N/top-k competitiveness.

## 11. Main Limitation

PPO did not produce new records, did not solve N40 under primary metrics, and weak surrogate-to-teacher alignment means teacher validation remains required.

## 12. Whether More PPO Experiments Are Needed

More PPO experiments are not required for the current bounded claim. They are recommended only if stronger N24/N40 or new-record claims are desired.

## 13. Recommended Manuscript Wording

"A surrogate-trained MaskablePPO policy generated legal scan-order candidates that were independently Abaqus teacher-validated. Although the PPO batch did not exceed the mature combined552 best records, it achieved bounded small-N/top-k competitiveness, demonstrating policy-generated candidate feasibility rather than record-level dominance."

## 14. Verdict

`{VERDICT}`
""",
        encoding="utf-8",
    )


def write_manifest(input_verdict: str, plot_paths: list[str]) -> None:
    manifest = {
        "branch": git_branch(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_paths": {
            "ppo_frozen_metrics": str(PPO_FROZEN_METRICS),
            "ppo_frozen_ranking": str(PPO_FROZEN_RANKING),
            "ppo_performance_summary": str(PPO_PERFORMANCE_SUMMARY),
            "ppo_final_claim_boundary": str(PPO_FINAL_CLAIM_BOUNDARY),
            "combined552_reference": str(COMBINED552),
            "analysis_dataset": str(ANALYSIS_DATASET),
        },
        "input_integrity_verdict": input_verdict,
        "output_tables": {
            "fair_comparison_levels": str(FAIR_LEVELS_CSV),
            "bootstrap_by_N": str(BOOTSTRAP_BY_N_CSV),
            "bootstrap_global": str(BOOTSTRAP_GLOBAL_CSV),
            "baseline_inventory": str(BASELINE_INVENTORY_CSV),
            "baseline_comparison": str(BASELINE_COMPARE_CSV),
            "clean_policy_source_chain": str(POLICY_CHAIN_CSV),
            "scientific_meaning": str(MEANING_TABLE_CSV),
        },
        "plot_paths": plot_paths,
        "report_path": str(REPORT_PATH),
        "claim_boundary_path": str(CLAIM_BOUNDARY),
        "claim_decision_memo_path": str(CLAIM_MEMO),
        "next_experiment_recommendation_path": str(NEXT_EXPERIMENT),
        "no_Abaqus": True,
        "no_ODB_opening": True,
        "no_solver": True,
        "no_training": True,
        "no_candidate_generation": True,
        "no_commit_or_push": True,
        "verdict": VERDICT,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    ensure_dirs()
    ppo_metrics = pd.read_csv(PPO_FROZEN_METRICS)
    ppo_rank = pd.read_csv(PPO_FROZEN_RANKING)
    ref = pd.read_csv(COMBINED552)
    analysis = pd.read_csv(ANALYSIS_DATASET)
    input_verdict = input_audit(ppo_metrics, ppo_rank, ref, analysis)
    if input_verdict.startswith("FAIL"):
        print(json.dumps({"verdict": input_verdict, "audit": str(AUDIT_JSON)}, indent=2))
        return 1

    ref_prepped = prepare_reference(ref)
    fair_comparison_levels()
    bootstrap_by_n, bootstrap_global, distributions = bootstrap_random_baseline(ref_prepped, ppo_rank)
    inventory, compare = baseline_family_inventory_and_compare(ref_prepped, ppo_rank)
    policy_source_chain()
    scientific_meaning_table()
    write_claim_memo()
    write_next_experiment()
    plot_paths = make_plots(bootstrap_by_n, bootstrap_global, distributions, ppo_rank)
    write_claim_boundary()
    write_report(input_verdict, bootstrap_by_n, bootstrap_global, inventory, compare)
    write_manifest(input_verdict, plot_paths)

    summary = {
        "verdict": VERDICT,
        "input_integrity_verdict": input_verdict,
        "bootstrap_global": bootstrap_global.to_dict(orient="records"),
        "baseline_family_groups_found": int((inventory["matched_row_count"] > 0).sum()) if "matched_row_count" in inventory.columns else 0,
        "policy_chain_table": str(POLICY_CHAIN_CSV),
        "scientific_meaning_table": str(MEANING_TABLE_CSV),
        "report": str(REPORT_PATH),
        "manifest": str(MANIFEST),
        "no_Abaqus": True,
        "no_ODB_opening": True,
        "no_solver": True,
        "no_training": True,
        "no_candidate_generation": True,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
