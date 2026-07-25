from __future__ import annotations

import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
RUN_ID = "run_57_calibrated_N24_N40_batch64_teacher_metrics_ingestion_and_combined392_ranking"
RUN_NAME = "calibrated N24/N40 batch64 teacher metrics ingestion and combined392 ranking"
SCRIPT_PATH = ROOT / "scripts" / "stage3" / "run_57_ingest_calibrated_N24_N40_batch64_and_build_combined392.py"

RUN56_DIR = ROOT / "outputs" / "stage3_run_56_calibrated_N24_N40_batch64_odb_teacher_validation"
RUN56_METRICS = RUN56_DIR / "run56_calibrated_N24_N40_batch64_teacher_metrics.csv"
RUN56_EXTRACTION = RUN56_DIR / "run56_calibrated_N24_N40_batch64_odb_extraction_summary.csv"
RUN56_SOLVER = RUN56_DIR / "run56_calibrated_N24_N40_batch64_solver_completion_audit.csv"
RUN56_SUMMARY = RUN56_DIR / "run56_calibrated_N24_N40_batch64_odb_teacher_validation_summary.json"
RUN56_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_56_calibrated_N24_N40_batch64_odb_teacher_validation" / "RUN_56_CALIBRATED_N24_N40_BATCH64_ODB_TEACHER_VALIDATION_REPORT.md"
RUN56_MANIFEST = ROOT / "artifacts" / "manifests" / "stage3_run_56_manifest.json"

RUN54_HANDOFF = ROOT / "outputs" / "stage3_run_54_run53_calibrated_N24_N40_batch64_handoff_package" / "stage3_run54_calibrated_N24_N40_batch64_candidate_orders.csv"
RUN54_SCAN_DIR = ROOT / "outputs" / "stage3_run_54_run53_calibrated_N24_N40_batch64_handoff_package" / "scan_orders"
RUN53_POOL = ROOT / "outputs" / "stage3_run_53_combined328_calibrated_N24_N40_batch64_candidate_generation" / "run53_candidate_pool_scored.csv"
RUN53_BATCH64_COMPARISON = ROOT / "outputs" / "stage3_run_53_combined328_calibrated_N24_N40_batch64_candidate_generation" / "run53_batch64_comparison_to_previous.csv"
RUN53_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_53_combined328_calibrated_N24_N40_batch64_candidate_generation" / "RUN_53_COMBINED328_CALIBRATED_N24_N40_BATCH64_CANDIDATE_GENERATION_REPORT.md"

COMBINED328_TEACHER = ROOT / "outputs" / "stage3_run_52_stricter_constrained_N24_N40_batch32_teacher_metrics_ingestion_and_combined328_ranking" / "combined328_teacher_dataset.csv"
COMBINED328_READY = ROOT / "outputs" / "stage3_run_52_stricter_constrained_N24_N40_batch32_teacher_metrics_ingestion_and_combined328_ranking" / "combined328_RL_ready_dataset.csv"
COMBINED328_PLUS_N32_READY = ROOT / "outputs" / "stage3_run_52_stricter_constrained_N24_N40_batch32_teacher_metrics_ingestion_and_combined328_ranking" / "combined328_plus_N32_RL_ready_dataset.csv"
N32_DEDUP = ROOT / "outputs" / "stage3_run_32a_stage2_n32_legacy_teacher_label_ingestion_for_stage3" / "n32_legacy_teacher_dataset_dedup_training_332.csv"
RUN52_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_52_stricter_constrained_N24_N40_batch32_teacher_metrics_ingestion_and_combined328_ranking" / "RUN_52_STRICTER_CONSTRAINED_N24_N40_BATCH32_TEACHER_METRICS_INGESTION_AND_COMBINED328_RANKING_REPORT.md"

OUTPUT_DIR = ROOT / "outputs" / "stage3_run_57_calibrated_N24_N40_batch64_teacher_metrics_ingestion_and_combined392_ranking"
REPORT_DIR = ROOT / "docs" / "stage3" / "runs" / "run_57_calibrated_N24_N40_batch64_teacher_metrics_ingestion_and_combined392_ranking"
REPORT_PATH = REPORT_DIR / "RUN_57_CALIBRATED_N24_N40_BATCH64_TEACHER_METRICS_INGESTION_AND_COMBINED392_RANKING_REPORT.md"
MANIFEST_PATH = ROOT / "artifacts" / "manifests" / "stage3_run_57_manifest.json"
CLAIM_BOUNDARY_MD = OUTPUT_DIR / "run57_claim_boundary.md"
CLAIM_BOUNDARY_JSON = OUTPUT_DIR / "run57_claim_boundary.json"

EXPECTED_RUN56_COUNTS = {24: 32, 40: 32}
EXPECTED_COMBINED328_COUNTS = {12: 36, 16: 36, 24: 128, 40: 128}
EXPECTED_COMBINED328_PLUS_N32_COUNTS = {12: 36, 16: 36, 24: 128, 32: 332, 40: 128}
EXPECTED_COMBINED392_COUNTS = {12: 36, 16: 36, 24: 160, 40: 160}
EXPECTED_COMBINED392_PLUS_N32_COUNTS = {12: 36, 16: 36, 24: 160, 32: 332, 40: 160}

RAW_METRICS = {
    "u2": "u2_range",
    "peeq": "peeq_max",
    "surfaceT": "surface_t_proxy",
    "mises": "mises_max",
}


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def clean_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_for_json(v) for v in value]
    if isinstance(value, tuple):
        return [clean_for_json(v) for v in value]
    if isinstance(value, (pd.Series, pd.Index)):
        return [clean_for_json(v) for v in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return clean_for_json(value.to_dict(orient="records"))
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return clean_for_json(value.item())
        except Exception:
            pass
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_for_json(payload), indent=2, sort_keys=False) + "\n", encoding="utf-8")


def write_table_json(path: Path, df: pd.DataFrame) -> None:
    write_json(path, {"schema": "records", "rows": df.to_dict(orient="records")})


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def current_branch() -> str:
    try:
        return subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def parse_n_counts(df: pd.DataFrame) -> dict[int, int]:
    return {int(k): int(v) for k, v in df["n"].astype(int).value_counts().sort_index().to_dict().items()}


def parse_order(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(x) for x in value]
    text = str(value).strip()
    if not text:
        raise ValueError("empty order")
    if text.startswith("["):
        return [int(x) for x in json.loads(text)]
    cleaned = text.replace(";", "-").replace(",", "-").replace(" ", "")
    return [int(x) for x in cleaned.split("-") if x != ""]


def valid_order(value: Any, n: int) -> bool:
    try:
        order = parse_order(value)
    except Exception:
        return False
    return len(order) == n and sorted(order) == list(range(n))


def order_hash(order: list[int]) -> str:
    payload = ",".join(str(x) for x in order).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def as_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def normalize_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "surface_t_proxy" in df.columns:
        surface_pa = as_float(df["surface_t_proxy"])
    elif "surface_t_proxy_max_tensile_pa" in df.columns:
        surface_pa = as_float(df["surface_t_proxy_max_tensile_pa"])
    elif "surface_t_proxy_max_tensile_mpa" in df.columns:
        surface_pa = as_float(df["surface_t_proxy_max_tensile_mpa"]) * 1_000_000.0
    else:
        surface_pa = pd.Series([math.nan] * len(df), index=df.index)

    if "surface_t_proxy_max_tensile_mpa" in df.columns:
        surface_mpa = as_float(df["surface_t_proxy_max_tensile_mpa"])
    else:
        surface_mpa = surface_pa / 1_000_000.0

    df["surface_t_proxy"] = surface_pa
    df["surface_t_proxy_mpa"] = surface_mpa
    for col in ["u2_range", "peeq_max", "mises_max"]:
        if col in df.columns:
            df[col] = as_float(df[col])
    return df


def minmax_cost(series: pd.Series) -> pd.Series:
    vals = as_float(series)
    mn = vals.min()
    mx = vals.max()
    if not math.isfinite(float(mn)) or not math.isfinite(float(mx)) or mx == mn:
        return pd.Series([0.0] * len(vals), index=vals.index)
    return (vals - mn) / (mx - mn)


def rank_score(ranks: pd.Series, count: int) -> pd.Series:
    denom = max(1, count - 1)
    return 1.0 - ((ranks.astype(float) - 1.0) / denom)


def add_scores(
    df: pd.DataFrame,
    rank_suffix: str,
    target_prefix: str,
    cost_suffix: str,
    reward_col: str,
    reward_rank_col: str,
) -> pd.DataFrame:
    out = df.copy()
    for metric_key, metric_col in RAW_METRICS.items():
        out[metric_col] = as_float(out[metric_col])
        target_col = f"{target_prefix}_{metric_key}_score"
        if metric_key == "surfaceT":
            target_col = f"{target_prefix}_surfaceT_score"
        rank_col = f"{metric_key}_rank_{rank_suffix}"
        cost_col = f"{metric_key}_cost_minmax_{cost_suffix}"
        if metric_key == "surfaceT":
            cost_col = f"surfaceT_cost_minmax_{cost_suffix}"
        out[rank_col] = math.nan
        out[target_col] = math.nan
        out[cost_col] = math.nan
        for _, idx in out.groupby("n").groups.items():
            ranks = out.loc[idx, metric_col].rank(method="average", ascending=True)
            out.loc[idx, rank_col] = ranks
            out.loc[idx, target_col] = rank_score(ranks, len(idx))
            out.loc[idx, cost_col] = minmax_cost(out.loc[idx, metric_col])

    out[reward_col] = (
        0.65 * out[f"{target_prefix}_u2_score"]
        + 0.20 * out[f"{target_prefix}_peeq_score"]
        + 0.10 * out[f"{target_prefix}_surfaceT_score"]
        + 0.05 * out[f"{target_prefix}_mises_score"]
    )
    out[reward_rank_col] = math.nan
    for _, idx in out.groupby("n").groups.items():
        out.loc[idx, reward_rank_col] = out.loc[idx, reward_col].rank(method="average", ascending=False)
    return out


def add_run56_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    labels = {"u2": "U2", "peeq": "PEEQ", "surfaceT": "SurfaceT", "mises": "Mises"}
    for metric_key, metric_col in RAW_METRICS.items():
        label = labels[metric_key]
        out[f"rank_{label}_run56_within_n"] = math.nan
        out[f"score_{label}_run56_within_n"] = math.nan
        out[f"cost_{metric_key}_run56_within_n"] = math.nan
        for _, idx in out.groupby("n").groups.items():
            ranks = out.loc[idx, metric_col].rank(method="average", ascending=True)
            out.loc[idx, f"rank_{label}_run56_within_n"] = ranks
            out.loc[idx, f"score_{label}_run56_within_n"] = rank_score(ranks, len(idx))
            out.loc[idx, f"cost_{metric_key}_run56_within_n"] = minmax_cost(out.loc[idx, metric_col])
    out["reward_run56_u2_primary"] = (
        0.65 * out["score_U2_run56_within_n"]
        + 0.20 * out["score_PEEQ_run56_within_n"]
        + 0.10 * out["score_SurfaceT_run56_within_n"]
        + 0.05 * out["score_Mises_run56_within_n"]
    )
    out["reward_run56_constrained_u2_reward_balanced"] = (
        0.50 * out["score_U2_run56_within_n"]
        + 0.25 * out["score_PEEQ_run56_within_n"]
        + 0.15 * out["score_SurfaceT_run56_within_n"]
        + 0.10 * out["score_Mises_run56_within_n"]
    )
    out["reward_run56_strict_penalty_guard"] = (
        0.40 * out["score_U2_run56_within_n"]
        + 0.30 * out["score_PEEQ_run56_within_n"]
        + 0.20 * out["score_SurfaceT_run56_within_n"]
        + 0.10 * out["score_Mises_run56_within_n"]
    )
    out["reward_run56_penalty_repair"] = (
        0.30 * out["score_U2_run56_within_n"]
        + 0.30 * out["score_PEEQ_run56_within_n"]
        + 0.25 * out["score_SurfaceT_run56_within_n"]
        + 0.15 * out["score_Mises_run56_within_n"]
    )
    out["rank_reward_run56_within_n"] = math.nan
    out["rank_constrained_reward_run56_within_n"] = math.nan
    out["rank_strict_penalty_guard_run56_within_n"] = math.nan
    out["rank_penalty_repair_run56_within_n"] = math.nan
    for _, idx in out.groupby("n").groups.items():
        out.loc[idx, "rank_reward_run56_within_n"] = out.loc[idx, "reward_run56_u2_primary"].rank(method="average", ascending=False)
        out.loc[idx, "rank_constrained_reward_run56_within_n"] = out.loc[idx, "reward_run56_constrained_u2_reward_balanced"].rank(method="average", ascending=False)
        out.loc[idx, "rank_strict_penalty_guard_run56_within_n"] = out.loc[idx, "reward_run56_strict_penalty_guard"].rank(method="average", ascending=False)
        out.loc[idx, "rank_penalty_repair_run56_within_n"] = out.loc[idx, "reward_run56_penalty_repair"].rank(method="average", ascending=False)
    return out


def first_col(df: pd.DataFrame, names: list[str]) -> str | None:
    return next((c for c in names if c in df.columns), None)


def make_run56_enriched(run56: pd.DataFrame, handoff: pd.DataFrame, solver: pd.DataFrame) -> pd.DataFrame:
    run56 = normalize_metric_columns(run56)
    handoff = handoff.copy()
    for frame in [run56, handoff]:
        if "n" in frame.columns:
            frame["n"] = frame["n"].astype(int)
        if "order_json" in frame.columns:
            frame["order_json"] = frame["order_json"].astype(str)

    merged = run56.merge(
        handoff,
        on=["handoff_strategy_name", "n"],
        how="left",
        suffixes=("", "_run54"),
        validate="one_to_one",
    )
    if "order_json" not in merged.columns and "order_json_run54" in merged.columns:
        merged["order_json"] = merged["order_json_run54"]
    if "order_json_run54" in merged.columns:
        merged["order_json"] = merged["order_json"].fillna(merged["order_json_run54"])
    if "order_hash" in merged.columns and "order_hash_run54" in merged.columns:
        merged["order_hash"] = merged["order_hash"].fillna(merged["order_hash_run54"])
    elif "order_hash_run54" in merged.columns:
        merged["order_hash"] = merged["order_hash_run54"]

    for i, row in merged.iterrows():
        order = parse_order(row["order_json"])
        if not str(row.get("order_hash", "")).strip():
            merged.at[i, "order_hash"] = order_hash(order)
        if not str(row.get("order_compact", "")).strip():
            merged.at[i, "order_compact"] = "-".join(str(x) for x in order)

    if not solver.empty:
        keep = [c for c in ["handoff_strategy_name", "job_name", "completion_status", "sta_success_marker_present", "lck_present"] if c in solver.columns]
        if "handoff_strategy_name" in keep:
            solver_small = solver[keep].drop_duplicates("handoff_strategy_name")
            merged = merged.merge(solver_small, on="handoff_strategy_name", how="left", suffixes=("", "_solver"))

    merged["strategy_name"] = merged["handoff_strategy_name"]
    merged["dataset_source"] = "run56_calibrated_N24_N40_batch64"
    merged["batch_name"] = "stage3_run54_calibrated_N24_N40_batch64_v01"
    merged["native_validation_N"] = True
    merged["N24_N40_focused"] = True
    merged["calibrated_batch64"] = True
    merged["overnight_batch64"] = True
    merged["includes_N12_case"] = False
    merged["includes_N16_case"] = False
    merged["includes_N32_case"] = False
    merged["final_step"] = merged.get("final_step_name", "")
    merged["extracted_fields"] = merged.get("extracted_field_names", "")
    merged["solver_audit_status"] = merged.get("completion_status", "")
    merged["nonfatal_warning_flag"] = merged.get("completion_status", "").astype(str).str.contains("WARNING", case=False, na=False)
    merged["notes"] = "Run57 ingestion of Run56 calibrated N24/N40 batch64 teacher metrics. No N12/N16/N32 cases."

    preferred = [
        "n", "strategy_name", "handoff_strategy_name", "job_name", "dataset_source", "batch_name",
        "native_validation_N", "N24_N40_focused", "calibrated_batch64", "overnight_batch64",
        "includes_N12_case", "includes_N16_case", "includes_N32_case",
        "original_run53_candidate_id", "original_run53_strategy_name", "candidate_source",
        "generation_method", "selection_bucket", "priority_role", "surrogate_prediction",
        "calibrated_reward_prediction", "penalty_repair_prediction",
        "N24_u2_retention_prediction", "N40_strict_reward_retention_prediction",
        "two_stage_repair_prediction", "median_guard_prediction", "u2_primary_prediction",
        "strict_penalty_guard_prediction", "predicted_peeq_guarded_score",
        "predicted_surfaceT_guarded_score", "predicted_mises_guarded_score",
        "gnn_reward_prediction", "graph_pointer_policy_score",
        "hybrid_score", "uncertainty_score", "gnn_vs_surrogate_disagreement", "novelty_distance",
        "nearest_existing_teacher_strategy",
        "order_json", "order_compact", "order_hash", "u2_range", "peeq_max", "surface_t_proxy",
        "surface_t_proxy_mpa", "mises_max", "final_step", "final_step_name", "final_frame_time",
        "extracted_fields", "extracted_field_names", "teacher_validation_status",
        "solver_audit_status", "completion_status", "odb_extraction_status", "nonfatal_warning_flag",
        "notes",
    ]
    cols = [c for c in preferred if c in merged.columns]
    cols += [c for c in merged.columns if c not in cols]
    return merged[cols]


def make_leaderboard(df: pd.DataFrame, reward_col: str, extra_reward_cols: dict[str, str] | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    extra_reward_cols = extra_reward_cols or {}
    for n, group in df.groupby("n"):
        row: dict[str, Any] = {"n": int(n), "count": int(len(group))}
        for metric_key, metric_col in RAW_METRICS.items():
            best = group.sort_values(metric_col, ascending=True).iloc[0]
            label = "surfaceT" if metric_key == "surfaceT" else metric_key
            row[f"best_{label}_strategy"] = best.get("strategy_name", best.get("handoff_strategy_name", ""))
            row[f"best_{label}_source"] = best.get("dataset_source", "")
            row[f"best_{label}_value"] = float(best[metric_col])
        best_reward = group.sort_values(reward_col, ascending=False).iloc[0]
        row["best_reward_strategy"] = best_reward.get("strategy_name", best_reward.get("handoff_strategy_name", ""))
        row["best_reward_source"] = best_reward.get("dataset_source", "")
        row["best_reward_value"] = float(best_reward[reward_col])
        for label, col in extra_reward_cols.items():
            if col not in group.columns:
                continue
            best_extra = group.sort_values(col, ascending=False).iloc[0]
            row[f"best_{label}_strategy"] = best_extra.get("strategy_name", best_extra.get("handoff_strategy_name", ""))
            row[f"best_{label}_source"] = best_extra.get("dataset_source", "")
            row[f"best_{label}_value"] = float(best_extra[col])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n")


def compare_best(
    baseline: pd.DataFrame,
    new: pd.DataFrame,
    combined: pd.DataFrame,
    baseline_reward_col: str,
    new_reward_col: str,
    combined_reward_col: str,
    ns: list[int],
    label: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for n in ns:
        base_n = baseline[baseline["n"].astype(int) == n]
        new_n = new[new["n"].astype(int) == n]
        comb_n = combined[combined["n"].astype(int) == n]
        for metric_name, metric_col, lower in [
            ("U2", "u2_range", True),
            ("PEEQ", "peeq_max", True),
            ("SurfaceT", "surface_t_proxy", True),
            ("Mises", "mises_max", True),
            ("combined_reward", combined_reward_col, False),
        ]:
            base_col = baseline_reward_col if metric_name == "combined_reward" else metric_col
            new_col = new_reward_col if metric_name == "combined_reward" else metric_col
            comb_col = combined_reward_col if metric_name == "combined_reward" else metric_col
            base_best = base_n.sort_values(base_col, ascending=lower).iloc[0]
            new_best = new_n.sort_values(new_col, ascending=lower).iloc[0]
            comb_best = comb_n.sort_values(comb_col, ascending=lower).iloc[0]
            base_val = float(base_best[base_col])
            new_val = float(new_best[new_col])
            if lower:
                beat = new_val < base_val
                abs_improvement = base_val - new_val
            else:
                beat = new_val > base_val
                abs_improvement = new_val - base_val
            rel = abs_improvement / abs(base_val) if base_val != 0 else math.nan
            rows.append({
                "comparison": label,
                "n": n,
                "metric": metric_name,
                "baseline_best_strategy": base_best.get("strategy_name", base_best.get("handoff_strategy_name", "")),
                "baseline_best_source": base_best.get("dataset_source", ""),
                "baseline_best_value": base_val,
                "run56_best_strategy": new_best.get("strategy_name", new_best.get("handoff_strategy_name", "")),
                "run56_best_source": new_best.get("dataset_source", ""),
                "run56_best_value": new_val,
                "run56_beats_baseline": bool(beat),
                "absolute_improvement": abs_improvement,
                "relative_improvement_fraction": rel,
                "combined392_best_strategy": comb_best.get("strategy_name", comb_best.get("handoff_strategy_name", "")),
                "combined392_best_source": comb_best.get("dataset_source", ""),
                "combined392_best_value": float(comb_best[comb_col]),
            })
    return pd.DataFrame(rows)


def top_counts(df: pd.DataFrame, source_col: str, reward_rank_col: str) -> dict[str, Any]:
    run56 = df[df["dataset_source"].astype(str) == "run56_calibrated_N24_N40_batch64"]
    payload: dict[str, Any] = {"total_run56_rows": int(len(run56)), "by_n": {}}
    for n, group in run56.groupby("n"):
        payload["by_n"][f"N{int(n)}"] = {
            "top5_u2_entries": int((group["u2_rank_combined392_within_n"] <= 5).sum()),
            "top10_u2_entries": int((group["u2_rank_combined392_within_n"] <= 10).sum()),
            "top5_reward_entries": int((group[reward_rank_col] <= 5).sum()),
            "top10_reward_entries": int((group[reward_rank_col] <= 10).sum()),
        }
    if source_col in run56.columns:
        payload["by_source"] = run56[source_col].fillna("unknown").value_counts().to_dict()
    return payload


def group_effectiveness(combined: pd.DataFrame) -> pd.DataFrame:
    run56 = combined[combined["dataset_source"].astype(str) == "run56_calibrated_N24_N40_batch64"].copy()
    rows: list[dict[str, Any]] = []
    group_cols = ["n", "candidate_source", "generation_method", "selection_bucket", "priority_role"]
    for col in group_cols:
        if col not in run56.columns:
            continue
        for key, group in run56.groupby(col, dropna=False):
            rows.append({
                "group_type": col,
                "group_value": str(key),
                "count": int(len(group)),
                "median_u2_rank_combined392": float(group["u2_rank_combined392_within_n"].median()),
                "best_u2_rank_combined392": float(group["u2_rank_combined392_within_n"].min()),
                "median_reward_rank_combined392": float(group["reward_rank_combined392_within_n"].median()),
                "best_reward_rank_combined392": float(group["reward_rank_combined392_within_n"].min()),
                "median_strict_guard_rank_combined392": float(group.get("strict_penalty_guard_rank_combined392_within_n", pd.Series([math.nan])).median()),
                "best_strict_guard_rank_combined392": float(group.get("strict_penalty_guard_rank_combined392_within_n", pd.Series([math.nan])).min()),
                "top5_u2_count": int((group["u2_rank_combined392_within_n"] <= 5).sum()),
                "top10_u2_count": int((group["u2_rank_combined392_within_n"] <= 10).sum()),
                "top5_reward_count": int((group["reward_rank_combined392_within_n"] <= 5).sum()),
                "top10_reward_count": int((group["reward_rank_combined392_within_n"] <= 10).sum()),
                "top5_strict_guard_count": int((group.get("strict_penalty_guard_rank_combined392_within_n", pd.Series([math.inf] * len(group), index=group.index)) <= 5).sum()),
                "top10_strict_guard_count": int((group.get("strict_penalty_guard_rank_combined392_within_n", pd.Series([math.inf] * len(group), index=group.index)) <= 10).sum()),
            })
    return pd.DataFrame(rows)


def spearman_safe(a: pd.Series, b: pd.Series) -> float | None:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 3 or x[mask].nunique() < 2 or y[mask].nunique() < 2:
        return None
    return float(x[mask].corr(y[mask], method="spearman"))


def prediction_audit(run56_ranked: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = run56_ranked.copy()
    pred_col = first_col(df, [
        "calibrated_reward_prediction",
        "penalty_repair_prediction",
        "N24_u2_retention_prediction",
        "N40_strict_reward_retention_prediction",
        "two_stage_repair_prediction",
        "median_guard_prediction",
        "strict_penalty_guard_prediction",
        "u2_primary_prediction",
        "hybrid_score",
        "surrogate_prediction",
        "model_prediction_mean",
        "pred_reward_native_combined392",
    ])
    u2_pred_col = first_col(df, ["u2_primary_prediction", "surrogate_prediction", "pred_u2_score", "hybrid_score"])
    peeq_pred_col = first_col(df, ["predicted_peeq_guarded_score"])
    surface_pred_col = first_col(df, ["predicted_surfaceT_guarded_score"])
    mises_pred_col = first_col(df, ["predicted_mises_guarded_score"])
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "prediction_column_used_for_reward": pred_col,
        "prediction_column_used_for_u2": u2_pred_col,
        "overall_reward_spearman": None,
        "overall_constrained_reward_spearman": None,
        "overall_strict_penalty_guard_spearman": None,
        "overall_u2_spearman": None,
        "overall_peeq_spearman": None,
        "overall_surfaceT_spearman": None,
        "overall_mises_spearman": None,
        "top1_hit": None,
        "mean_top5_overlap": None,
        "per_n": {},
        "by_selection_bucket": {},
    }
    if pred_col:
        df[pred_col] = pd.to_numeric(df[pred_col], errors="coerce")
        summary["overall_reward_spearman"] = spearman_safe(df[pred_col], df["reward_run56_u2_primary"])
        if "reward_run56_constrained_u2_reward_balanced" in df.columns:
            summary["overall_constrained_reward_spearman"] = spearman_safe(df[pred_col], df["reward_run56_constrained_u2_reward_balanced"])
        if "reward_run56_strict_penalty_guard" in df.columns:
            summary["overall_strict_penalty_guard_spearman"] = spearman_safe(df[pred_col], df["reward_run56_strict_penalty_guard"])
        if "reward_run56_penalty_repair" in df.columns:
            summary["overall_penalty_repair_spearman"] = spearman_safe(df[pred_col], df["reward_run56_penalty_repair"])
    if u2_pred_col:
        df[u2_pred_col] = pd.to_numeric(df[u2_pred_col], errors="coerce")
        summary["overall_u2_spearman"] = spearman_safe(df[u2_pred_col], df["score_U2_run56_within_n"])
    if peeq_pred_col:
        summary["overall_peeq_spearman"] = spearman_safe(df[peeq_pred_col], df["score_PEEQ_run56_within_n"])
    if surface_pred_col:
        summary["overall_surfaceT_spearman"] = spearman_safe(df[surface_pred_col], df["score_SurfaceT_run56_within_n"])
    if mises_pred_col:
        summary["overall_mises_spearman"] = spearman_safe(df[mises_pred_col], df["score_Mises_run56_within_n"])

    top1_hits = 0
    top5_overlaps: list[int] = []
    for n, group in df.groupby("n"):
        item: dict[str, Any] = {"count": int(len(group))}
        if pred_col and group[pred_col].notna().any():
            item["reward_spearman"] = spearman_safe(group[pred_col], group["reward_run56_u2_primary"])
            if "reward_run56_constrained_u2_reward_balanced" in group.columns:
                item["constrained_reward_spearman"] = spearman_safe(group[pred_col], group["reward_run56_constrained_u2_reward_balanced"])
            if "reward_run56_strict_penalty_guard" in group.columns:
                item["strict_penalty_guard_spearman"] = spearman_safe(group[pred_col], group["reward_run56_strict_penalty_guard"])
            if "reward_run56_penalty_repair" in group.columns:
                item["penalty_repair_spearman"] = spearman_safe(group[pred_col], group["reward_run56_penalty_repair"])
            realized_col = "reward_run56_penalty_repair" if "penalty_repair" in str(pred_col) else "reward_run56_strict_penalty_guard" if "strict" in str(pred_col) else "reward_run56_u2_primary"
            predicted = list(group.sort_values(pred_col, ascending=False)["handoff_strategy_name"].head(5))
            realized = list(group.sort_values(realized_col, ascending=False)["handoff_strategy_name"].head(5))
            overlap = len(set(predicted) & set(realized))
            top5_overlaps.append(overlap)
            top1_hits += int(predicted[:1] == realized[:1])
            item["top5_overlap"] = overlap
            item["predicted_top1"] = predicted[0] if predicted else ""
            item["realized_top1"] = realized[0] if realized else ""
        if u2_pred_col and group[u2_pred_col].notna().any():
            item["u2_spearman"] = spearman_safe(group[u2_pred_col], group["score_U2_run56_within_n"])
        if peeq_pred_col and group[peeq_pred_col].notna().any():
            item["peeq_spearman"] = spearman_safe(group[peeq_pred_col], group["score_PEEQ_run56_within_n"])
        if surface_pred_col and group[surface_pred_col].notna().any():
            item["surfaceT_spearman"] = spearman_safe(group[surface_pred_col], group["score_SurfaceT_run56_within_n"])
        if mises_pred_col and group[mises_pred_col].notna().any():
            item["mises_spearman"] = spearman_safe(group[mises_pred_col], group["score_Mises_run56_within_n"])
        summary["per_n"][f"N{int(n)}"] = item
    if top5_overlaps:
        summary["top1_hit"] = f"{top1_hits}/{len(top5_overlaps)}"
        summary["mean_top5_overlap"] = sum(top5_overlaps) / len(top5_overlaps)

    for bucket_col in ["selection_bucket", "candidate_source"]:
        if bucket_col in df.columns and pred_col:
            bucket_rows = []
            for bucket, group in df.groupby(bucket_col, dropna=False):
                bucket_rows.append({
                    "group_type": bucket_col,
                    "group_value": str(bucket),
                    "count": int(len(group)),
                    "reward_spearman": spearman_safe(group[pred_col], group["reward_run56_u2_primary"]),
                    "mean_abs_reward_error": float((pd.to_numeric(group[pred_col], errors="coerce") - group["reward_run56_u2_primary"]).abs().mean()),
                })
            rows.extend(bucket_rows)

    for diag_col in ["gnn_vs_surrogate_disagreement", "uncertainty_score", "novelty_distance"]:
        if diag_col in df.columns and pred_col:
            err = (pd.to_numeric(df[pred_col], errors="coerce") - df["reward_run56_u2_primary"]).abs()
            summary[f"{diag_col}_vs_abs_error_spearman"] = spearman_safe(df[diag_col], err)

    return pd.DataFrame(rows), summary


def validate_inputs(run56: pd.DataFrame, handoff: pd.DataFrame, combined328: pd.DataFrame, plus_n32: pd.DataFrame) -> dict[str, Any]:
    errors: list[str] = []
    run56_counts = parse_n_counts(run56)
    combined328_counts = parse_n_counts(combined328)
    plus_counts = parse_n_counts(plus_n32)
    if run56_counts != EXPECTED_RUN56_COUNTS:
        errors.append(f"Run56 counts mismatch: {run56_counts}")
    if set(run56["n"].astype(int)) - {24, 40}:
        errors.append("Run56 contains non-N24/N40 rows")
    if combined328_counts != EXPECTED_COMBINED328_COUNTS:
        errors.append(f"combined328 counts mismatch: {combined328_counts}")
    if plus_counts != EXPECTED_COMBINED328_PLUS_N32_COUNTS:
        errors.append(f"combined328_plus_N32 counts mismatch: {plus_counts}")
    for col in ["u2_range", "peeq_max", "mises_max"]:
        if col not in run56.columns:
            errors.append(f"Run56 missing metric {col}")
    if "surface_t_proxy" not in run56.columns and "surface_t_proxy_max_tensile_pa" not in run56.columns:
        errors.append("Run56 missing surface_t_proxy metric")
    if "teacher_validation_status" not in run56.columns or not (run56["teacher_validation_status"].astype(str) == "PASS_TEACHER_FIELDS_EXTRACTED").all():
        errors.append("Run56 teacher_validation_status is not PASS for all rows")
    if "final_step_name" in run56.columns and not (run56["final_step_name"].astype(str) == "step_final_cooling").all():
        errors.append("Run56 final_step_name is not step_final_cooling for all rows")
    if "extracted_field_names" in run56.columns:
        for field in ["U", "PEEQ", "S", "NT11"]:
            if not run56["extracted_field_names"].astype(str).str.contains(field).all():
                errors.append(f"Run56 extracted fields missing {field}")
    for col in RAW_METRICS.values():
        source = "surface_t_proxy_max_tensile_pa" if col == "surface_t_proxy" and col not in run56.columns else col
        if source in run56.columns and pd.to_numeric(run56[source], errors="coerce").isna().any():
            errors.append(f"Run56 has missing metric values in {source}")
    missing_handoff = sorted(set(run56["handoff_strategy_name"]) - set(handoff["handoff_strategy_name"]))
    if missing_handoff:
        errors.append(f"Run56 rows unmatched to Run54 handoff: {missing_handoff[:5]}")
    bad_orders = []
    for _, row in handoff.iterrows():
        if not valid_order(row["order_json"], int(row["n"])):
            bad_orders.append(row.get("handoff_strategy_name", "UNKNOWN"))
    if bad_orders:
        errors.append(f"Run54 handoff has invalid scan orders: {bad_orders[:5]}")
    n32 = plus_n32[plus_n32["n"].astype(int) == 32]
    warning_col = first_col(n32, ["metric_semantic_warning", "legacy_compatibility_status", "compatibility_status"])
    if warning_col is None:
        errors.append("combined392_plus_N32 N32 rows do not carry semantic warning/status columns")

    verdict = "PASS_RUN57_CALIBRATED_N24_N40_BATCH64_TEACHER_METRICS_64_OF_64_READY" if not errors else "FAIL_RUN57_INPUT_VALIDATION"
    return {
        "timestamp": now_iso(),
        "verdict": verdict,
        "errors": errors,
        "run56_rows": int(len(run56)),
        "run56_per_N_counts": run56_counts,
        "combined328_rows": int(len(combined328)),
        "combined328_per_N_counts": combined328_counts,
        "combined328_plus_N32_rows": int(len(plus_n32)),
        "combined328_plus_N32_per_N_counts": plus_counts,
        "run56_has_N12": bool((run56["n"].astype(int) == 12).any()),
        "run56_has_N16": bool((run56["n"].astype(int) == 16).any()),
        "run56_has_N32": bool((run56["n"].astype(int) == 32).any()),
        "run56_rows_matched_to_run54_handoff": int(len(run56) - len(missing_handoff)),
        "n32_semantic_warning_column": warning_col,
    }


def write_claim_boundary() -> None:
    safe = [
        "Run57 ingests 64/64 teacher-validated Run56 calibrated N24/N40 batch64 cases.",
        "Run57 builds native combined392 with N12=36, N16=36, N24=160, N40=160.",
        "Run57 builds combined392_plus_N32 with N12=36, N16=36, N24=160, N32=332, N40=160.",
        "Run57 evaluates whether calibrated batch64 candidate selection improved native Stage 3 teacher metrics.",
        "Run56 is teacher validation of native N24/N40 calibrated candidates, not N32 cases.",
        "Run57 audits whether N24/N40 are approaching a mature teacher-label density for offline active-learning / offline-RL evidence.",
    ]
    unsafe = [
        "Do not claim N32 itself was newly teacher-validated in Run56.",
        "Do not claim N32 caused Run56 improvements.",
        "Do not claim GNN-RL superiority.",
        "Do not claim online RL.",
        "Do not claim arbitrary-N generalization.",
        "Do not claim physical optimum.",
        "Do not claim solver/ODB extraction happened in Run57.",
        "Do not claim full variable-N RL maturity if N12/N16 remain under-sampled.",
    ]
    CLAIM_BOUNDARY_MD.write_text(
        "# Run57 Claim Boundary\n\n## Safe claims\n\n"
        + "\n".join(f"- {x}" for x in safe)
        + "\n\n## Unsafe claims\n\n"
        + "\n".join(f"- {x}" for x in unsafe)
        + "\n",
        encoding="utf-8",
    )
    write_json(CLAIM_BOUNDARY_JSON, {
        "verdict": "RUN57_INGESTION_AND_COMBINED392_RANKING_ONLY_NO_SOLVER_OR_TRAINING",
        "safe_claims": safe,
        "unsafe_claims": unsafe,
    })


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_No rows._"
    rows = df[columns].copy()
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for _, row in rows.iterrows():
        body.append("| " + " | ".join(str(row[c]) for c in columns) + " |")
    return "\n".join([header, sep] + body)


def add_existing_constrained_reward(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = df.copy()
    cols = [
        f"target_u2_score_{prefix}_rank",
        f"target_peeq_score_{prefix}_rank",
        f"target_surfaceT_score_{prefix}_rank",
        f"target_mises_score_{prefix}_rank",
    ]
    if all(c in out.columns for c in cols):
        out[f"target_reward_{prefix}_constrained_u2_reward_balanced"] = (
            0.50 * pd.to_numeric(out[cols[0]], errors="coerce")
            + 0.25 * pd.to_numeric(out[cols[1]], errors="coerce")
            + 0.15 * pd.to_numeric(out[cols[2]], errors="coerce")
            + 0.10 * pd.to_numeric(out[cols[3]], errors="coerce")
        )
    return out


def maturity_audit(combined392: pd.DataFrame, prediction_summary: dict[str, Any], new_best_count: int) -> tuple[pd.DataFrame, dict[str, Any], str]:
    counts_by_n = parse_n_counts(combined392)
    rows = []
    for n in [12, 16, 24, 32, 40]:
        count = 332 if n == 32 else counts_by_n.get(n, 0)
        rows.append({
            "n": n,
            "teacher_rows": int(count),
            "native_stage3_rows": int(count if n != 32 else 0),
            "legacy_compatible_rows": int(count if n == 32 else 0),
            "maturity_note": (
                "legacy N32 auxiliary data; not native Stage 3"
                if n == 32 else
                "under-sampled native anchor"
                if n in {12, 16} else
                "mature N24/N40 focused native teacher-label density"
            ),
        })
    audit = pd.DataFrame(rows)
    n24_n40_ready = counts_by_n.get(24, 0) >= 160 and counts_by_n.get(40, 0) >= 160
    anchors_limited = counts_by_n.get(12, 0) <= 36 and counts_by_n.get(16, 0) <= 36
    headline = (
        "N24/N40 now have 160 native teacher rows each, enough to support a stronger offline active-learning/RL evidence freeze for those N values, while N12/N16 remain under-sampled anchors."
        if n24_n40_ready else
        "N24/N40 teacher-label density improved, but the requested 160-row maturity threshold was not reached for both N values."
    )
    summary = {
        "headline": headline,
        "n24_teacher_rows": int(counts_by_n.get(24, 0)),
        "n40_teacher_rows": int(counts_by_n.get(40, 0)),
        "n32_legacy_teacher_rows": 332,
        "n12_teacher_rows": int(counts_by_n.get(12, 0)),
        "n16_teacher_rows": int(counts_by_n.get(16, 0)),
        "n24_n40_mature_for_offline_active_learning_evidence": bool(n24_n40_ready),
        "full_variable_n_rl_maturity_limited_by_n12_n16": bool(anchors_limited),
        "recent_new_best_count_vs_combined328": int(new_best_count),
        "prediction_calibration_summary": prediction_summary,
        "safe_paper_claim": "N24/N40 have substantially denser native teacher labels and repeated active-learning loops; full variable-N maturity remains limited by N12/N16 sample counts.",
    }
    md = f"""# N24/N40 Maturity and RL-Readiness Audit

{headline}

- N24 teacher rows: {summary['n24_teacher_rows']}
- N40 teacher rows: {summary['n40_teacher_rows']}
- N32 legacy-compatible rows: 332
- N12 teacher rows: {summary['n12_teacher_rows']}
- N16 teacher rows: {summary['n16_teacher_rows']}
- Full variable-N maturity limitation: N12/N16 remain under-sampled relative to N24/N40 and N32.

Safe paper/report boundary: {summary['safe_paper_claim']}
"""
    return audit, summary, md


def write_report(payload: dict[str, Any]) -> None:
    validation = payload["validation"]
    combined_summary = payload["combined392_summary"]
    plus_summary = payload["combined392_plus_N32_summary"]
    comparison = payload["run56_vs_combined392"]
    prior_summary = payload["prior_summary"]
    effectiveness = payload["effectiveness_summary"]
    pred = payload["prediction_summary"]
    maturity = payload["maturity_summary"]

    def fmt_counts(counts: dict[int, int] | dict[str, int]) -> str:
        return ", ".join(f"N{k}={v}" for k, v in counts.items())

    run56_best = payload["run56_leaderboard"]
    combined_best = payload["combined392_leaderboard"]
    report = f"""# Stage 3 Run 57 - Calibrated N24/N40 Batch64 Teacher Metrics Ingestion and Combined392 Ranking

## 1. Purpose
Run57 ingests the completed Run56 user-selected overnight calibrated N24/N40 batch64 teacher metrics, merges them with Run54/Run53 candidate metadata, recomputes within-N rankings, and builds native combined392 plus combined392_plus_N32 datasets.

## 2. Inputs
- Run56 teacher metrics: `{RUN56_METRICS}`
- Run54 handoff metadata: `{RUN54_HANDOFF}`
- Native combined328 RL-ready dataset: `{COMBINED328_READY}`
- combined328_plus_N32 RL-ready dataset: `{COMBINED328_PLUS_N32_READY}`
- N32 deduplicated legacy-compatible table: `{N32_DEDUP}`

## 3. Run56 Teacher-Extraction Status
Run56 was complete for 64/64 teacher-validated cases: N24=32 and N40=32. It contains no N12, N16, or N32 cases. Run56 is native calibrated N24/N40 teacher validation, not N32 teacher validation.

## 4. Input Validation
Verdict: `{validation['verdict']}`.

Run56 counts: {fmt_counts(validation['run56_per_N_counts'])}. Native combined328 input counts: {fmt_counts(validation['combined328_per_N_counts'])}. combined328_plus_N32 input counts: {fmt_counts(validation['combined328_plus_N32_per_N_counts'])}.

## 5. Run56 Enriched Teacher Dataset
Run57 produced an enriched Run56 teacher dataset with handoff names, Run53/Run54 prediction metadata, candidate-source metadata, scan orders, hashes, raw teacher metrics, extraction status, and nonfatal-warning flags.

## 6. Run56 Within-Batch Ranking
Within Run56, lower raw metric values are better for U2, PEEQ, SurfaceT, and Mises. The U2-primary reward uses 0.65 U2, 0.20 PEEQ, 0.10 SurfaceT, and 0.05 Mises rank scores.

Run56 best U2 by N:
{markdown_table(run56_best, ['n', 'best_u2_strategy', 'best_u2_value'])}

Run56 best combined reward by N:
{markdown_table(run56_best, ['n', 'best_reward_strategy', 'best_reward_value'])}

## 7. Native Combined392 Construction
Native combined392 rows: {combined_summary['rows']}. Counts: {fmt_counts(combined_summary['per_N_counts'])}. There are no N32 rows in native combined392.

## 8. combined392_plus_N32 Construction
combined392_plus_N32 rows: {plus_summary['rows']}. Counts: {fmt_counts(plus_summary['per_N_counts'])}. N32 rows preserve the legacy metric semantic warnings from Run32A.

## 9. Run56 vs Combined328 Best Comparison
Run56 was compared against the native combined328 best records for N24 and N40 across U2, PEEQ, SurfaceT, Mises, U2-primary reward, constrained reward, strict penalty guard, and penalty-repair reward.

{markdown_table(comparison, ['n', 'metric', 'run56_beats_baseline', 'baseline_best_strategy', 'run56_best_strategy', 'absolute_improvement'])}

## 10. Run56 vs Prior Key Records
Run56 was compared against combined328, Run46, Run41, Run36, Run27, and earlier baselines where available. Summary: {prior_summary['headline']}

## 11. Calibrated N24/N40 Batch64 Effectiveness Audit
{effectiveness['headline']}

Top-entry counts by N:
```json
{json.dumps(effectiveness['top_entry_counts_by_n'], indent=2)}
```

## 12. Prediction Audit for Run53 Batch64
Prediction column used for reward audit: `{pred.get('prediction_column_used_for_reward')}`. Overall reward Spearman: `{pred.get('overall_reward_spearman')}`. Top1 hit: `{pred.get('top1_hit')}`. Mean top5 overlap: `{pred.get('mean_top5_overlap')}`.

Per-N prediction audit:
```json
{json.dumps(pred.get('per_n', {}), indent=2)}
```

## 13. U2 Gain Versus Penalty Analysis
{payload['penalty_headline']}

## 14. N24/N40 Maturity and RL-Readiness Audit
{maturity['headline']}

N24 rows: {maturity['n24_teacher_rows']}. N40 rows: {maturity['n40_teacher_rows']}. N32 legacy rows: {maturity['n32_legacy_teacher_rows']}. N12/N16 remain at {maturity['n12_teacher_rows']} and {maturity['n16_teacher_rows']}.

## 15. Metric Semantic Boundary for N32
combined392_plus_N32 includes N32 legacy-compatible rows. These rows are not native Stage 3 teacher validation. PEEQ is mapped from Stage 2 `peeq_guard`, and Mises is mapped from `mises_P95_top_band`; they are proxy-compatible fields with warnings, not literal native Stage 3 metric identities.

## 16. Claim Boundary
Claim boundary verdict: `RUN57_INGESTION_AND_COMBINED392_RANKING_ONLY_NO_SOLVER_OR_TRAINING`.

## 17. Output Files
- Run56 enriched dataset: `{OUTPUT_DIR / 'run56_calibrated_N24_N40_batch64_teacher_dataset_enriched.csv'}`
- Run56 ranked within batch: `{OUTPUT_DIR / 'run56_calibrated_N24_N40_batch64_ranked_within_batch.csv'}`
- Native combined392 RL-ready dataset: `{OUTPUT_DIR / 'combined392_RL_ready_dataset.csv'}`
- combined392_plus_N32 RL-ready dataset: `{OUTPUT_DIR / 'combined392_plus_N32_RL_ready_dataset.csv'}`
- Run56 vs combined328 comparison: `{OUTPUT_DIR / 'run56_vs_combined328_best_comparison.csv'}`
- Effectiveness audit: `{OUTPUT_DIR / 'run56_calibrated_batch64_effectiveness_audit.csv'}`
- Prediction audit: `{OUTPUT_DIR / 'run56_prediction_audit_for_run53_batch64.csv'}`
- Maturity audit: `{OUTPUT_DIR / 'n24_n40_maturity_and_rl_readiness_audit.md'}`
- Manifest: `{MANIFEST_PATH}`

## 18. Recommended Run58
{payload['recommended_run58']}
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    ensure_dirs()

    run56_raw = read_csv(RUN56_METRICS)
    handoff = read_csv(RUN54_HANDOFF)
    combined328 = normalize_metric_columns(read_csv(COMBINED328_READY))
    combined328 = add_existing_constrained_reward(combined328, "combined328")
    if "target_reward_combined328_strict_penalty_guard" not in combined328.columns:
        score_cols_328 = [
            "target_u2_score_combined328_rank",
            "target_peeq_score_combined328_rank",
            "target_surfaceT_score_combined328_rank",
            "target_mises_score_combined328_rank",
        ]
        if all(c in combined328.columns for c in score_cols_328):
            combined328["target_reward_combined328_strict_penalty_guard"] = (
                0.40 * pd.to_numeric(combined328[score_cols_328[0]], errors="coerce")
                + 0.30 * pd.to_numeric(combined328[score_cols_328[1]], errors="coerce")
                + 0.20 * pd.to_numeric(combined328[score_cols_328[2]], errors="coerce")
                + 0.10 * pd.to_numeric(combined328[score_cols_328[3]], errors="coerce")
            )
    score_cols_328 = [
        "target_u2_score_combined328_rank",
        "target_peeq_score_combined328_rank",
        "target_surfaceT_score_combined328_rank",
        "target_mises_score_combined328_rank",
    ]
    if "target_reward_combined328_penalty_repair" not in combined328.columns and all(c in combined328.columns for c in score_cols_328):
        combined328["target_reward_combined328_penalty_repair"] = (
            0.30 * pd.to_numeric(combined328[score_cols_328[0]], errors="coerce")
            + 0.30 * pd.to_numeric(combined328[score_cols_328[1]], errors="coerce")
            + 0.25 * pd.to_numeric(combined328[score_cols_328[2]], errors="coerce")
            + 0.15 * pd.to_numeric(combined328[score_cols_328[3]], errors="coerce")
        )
    combined328_teacher = normalize_metric_columns(read_csv(COMBINED328_TEACHER))
    combined328_plus = normalize_metric_columns(read_csv(COMBINED328_PLUS_N32_READY))
    n32 = normalize_metric_columns(read_csv(N32_DEDUP))
    solver = read_csv(RUN56_SOLVER) if RUN56_SOLVER.exists() else pd.DataFrame()

    validation = validate_inputs(run56_raw, handoff, combined328, combined328_plus)
    write_json(OUTPUT_DIR / "run57_input_validation_summary.json", validation)
    if not validation["verdict"].startswith("PASS"):
        raise SystemExit(f"Input validation failed: {validation['errors']}")

    run56_enriched = make_run56_enriched(run56_raw, handoff, solver)
    write_csv(OUTPUT_DIR / "run56_calibrated_N24_N40_batch64_teacher_dataset_enriched.csv", run56_enriched)
    write_table_json(OUTPUT_DIR / "run56_calibrated_N24_N40_batch64_teacher_dataset_enriched.json", run56_enriched)

    run56_ranked = add_run56_scores(run56_enriched)
    write_csv(OUTPUT_DIR / "run56_calibrated_N24_N40_batch64_ranked_within_batch.csv", run56_ranked)
    run56_leaderboard = make_leaderboard(
        run56_ranked,
        "reward_run56_u2_primary",
        {
            "constrained_reward": "reward_run56_constrained_u2_reward_balanced",
            "strict_penalty_guard": "reward_run56_strict_penalty_guard",
            "penalty_repair": "reward_run56_penalty_repair",
        },
    )
    write_csv(OUTPUT_DIR / "run56_calibrated_N24_N40_batch64_per_N_leaderboard.csv", run56_leaderboard)

    combined392 = pd.concat([combined328, run56_enriched], ignore_index=True, sort=False)
    combined392["n"] = combined392["n"].astype(int)
    combined392 = normalize_metric_columns(combined392)
    combined392 = add_scores(
        combined392,
        rank_suffix="combined392_within_n",
        target_prefix="target",
        cost_suffix="combined392_within_n",
        reward_col="target_reward_combined392_u2_primary",
        reward_rank_col="reward_rank_combined392_within_n",
    )
    combined392 = combined392.rename(columns={
        "target_u2_score": "target_u2_score_combined392_rank",
        "target_peeq_score": "target_peeq_score_combined392_rank",
        "target_surfaceT_score": "target_surfaceT_score_combined392_rank",
        "target_mises_score": "target_mises_score_combined392_rank",
    })
    combined392["target_reward_combined392_constrained_u2_reward_balanced"] = (
        0.50 * combined392["target_u2_score_combined392_rank"]
        + 0.25 * combined392["target_peeq_score_combined392_rank"]
        + 0.15 * combined392["target_surfaceT_score_combined392_rank"]
        + 0.10 * combined392["target_mises_score_combined392_rank"]
    )
    combined392["target_reward_combined392_strict_penalty_guard"] = (
        0.40 * combined392["target_u2_score_combined392_rank"]
        + 0.30 * combined392["target_peeq_score_combined392_rank"]
        + 0.20 * combined392["target_surfaceT_score_combined392_rank"]
        + 0.10 * combined392["target_mises_score_combined392_rank"]
    )
    combined392["target_reward_combined392_penalty_repair"] = (
        0.30 * combined392["target_u2_score_combined392_rank"]
        + 0.30 * combined392["target_peeq_score_combined392_rank"]
        + 0.25 * combined392["target_surfaceT_score_combined392_rank"]
        + 0.15 * combined392["target_mises_score_combined392_rank"]
    )
    combined392["constrained_reward_rank_combined392_within_n"] = math.nan
    combined392["strict_penalty_guard_rank_combined392_within_n"] = math.nan
    combined392["penalty_repair_rank_combined392_within_n"] = math.nan
    for _, idx in combined392.groupby("n").groups.items():
        combined392.loc[idx, "constrained_reward_rank_combined392_within_n"] = combined392.loc[idx, "target_reward_combined392_constrained_u2_reward_balanced"].rank(method="average", ascending=False)
        combined392.loc[idx, "strict_penalty_guard_rank_combined392_within_n"] = combined392.loc[idx, "target_reward_combined392_strict_penalty_guard"].rank(method="average", ascending=False)
        combined392.loc[idx, "penalty_repair_rank_combined392_within_n"] = combined392.loc[idx, "target_reward_combined392_penalty_repair"].rank(method="average", ascending=False)
    write_csv(OUTPUT_DIR / "combined392_teacher_dataset.csv", combined392)
    write_csv(OUTPUT_DIR / "combined392_RL_ready_dataset.csv", combined392)
    combined392_leaderboard = make_leaderboard(
        combined392,
        "target_reward_combined392_u2_primary",
        {
            "constrained_reward": "target_reward_combined392_constrained_u2_reward_balanced",
            "strict_penalty_guard": "target_reward_combined392_strict_penalty_guard",
            "penalty_repair": "target_reward_combined392_penalty_repair",
        },
    )
    write_csv(OUTPUT_DIR / "combined392_per_N_leaderboard.csv", combined392_leaderboard)
    combined392_summary = {
        "rows": int(len(combined392)),
        "per_N_counts": parse_n_counts(combined392),
        "no_N32_rows": bool((combined392["n"].astype(int) == 32).sum() == 0),
        "leaderboard": combined392_leaderboard.to_dict(orient="records"),
    }
    write_json(OUTPUT_DIR / "combined392_summary.json", combined392_summary)

    n32_for_plus = n32.copy()
    n32_for_plus["n"] = n32_for_plus["n"].astype(int)
    n32_for_plus["legacy_compatibility_status"] = "LEGACY_COMPATIBLE_WITH_WARNINGS"
    n32_for_plus["metric_semantic_warning"] = True
    combined392_plus = pd.concat([combined392, n32_for_plus], ignore_index=True, sort=False)
    combined392_plus["n"] = combined392_plus["n"].astype(int)
    combined392_plus["metric_semantic_warning"] = combined392_plus["metric_semantic_warning"].fillna(False)
    combined392_plus["legacy_compatibility_status"] = combined392_plus.get("legacy_compatibility_status", "").replace("", pd.NA)
    combined392_plus["legacy_compatibility_status"] = combined392_plus["legacy_compatibility_status"].fillna("NATIVE_STAGE3")
    combined392_plus = normalize_metric_columns(combined392_plus)
    combined392_plus = add_scores(
        combined392_plus,
        rank_suffix="combined392_plus_N32_within_n",
        target_prefix="target",
        cost_suffix="combined392_plus_N32_within_n",
        reward_col="target_reward_combined392_plus_N32_mapped_u2_primary",
        reward_rank_col="reward_rank_combined392_plus_N32_within_n",
    )
    combined392_plus = combined392_plus.rename(columns={
        "target_u2_score": "target_u2_score_combined392_plus_N32_rank",
        "target_peeq_score": "target_peeq_score_combined392_plus_N32_rank",
        "target_surfaceT_score": "target_surfaceT_score_combined392_plus_N32_rank",
        "target_mises_score": "target_mises_score_combined392_plus_N32_rank",
    })
    combined392_plus["target_reward_combined392_plus_N32_strict_u2_surfaceT"] = (
        0.80 * combined392_plus["target_u2_score_combined392_plus_N32_rank"]
        + 0.20 * combined392_plus["target_surfaceT_score_combined392_plus_N32_rank"]
    )
    combined392_plus["target_reward_combined392_plus_N32_constrained_u2_reward_balanced"] = (
        0.50 * combined392_plus["target_u2_score_combined392_plus_N32_rank"]
        + 0.25 * combined392_plus["target_peeq_score_combined392_plus_N32_rank"]
        + 0.15 * combined392_plus["target_surfaceT_score_combined392_plus_N32_rank"]
        + 0.10 * combined392_plus["target_mises_score_combined392_plus_N32_rank"]
    )
    combined392_plus["target_reward_combined392_plus_N32_strict_penalty_guard"] = (
        0.40 * combined392_plus["target_u2_score_combined392_plus_N32_rank"]
        + 0.30 * combined392_plus["target_peeq_score_combined392_plus_N32_rank"]
        + 0.20 * combined392_plus["target_surfaceT_score_combined392_plus_N32_rank"]
        + 0.10 * combined392_plus["target_mises_score_combined392_plus_N32_rank"]
    )
    combined392_plus["target_reward_combined392_plus_N32_penalty_repair"] = (
        0.30 * combined392_plus["target_u2_score_combined392_plus_N32_rank"]
        + 0.30 * combined392_plus["target_peeq_score_combined392_plus_N32_rank"]
        + 0.25 * combined392_plus["target_surfaceT_score_combined392_plus_N32_rank"]
        + 0.15 * combined392_plus["target_mises_score_combined392_plus_N32_rank"]
    )
    write_csv(OUTPUT_DIR / "combined392_plus_N32_teacher_dataset.csv", combined392_plus)
    write_csv(OUTPUT_DIR / "combined392_plus_N32_RL_ready_dataset.csv", combined392_plus)
    combined392_plus_leaderboard = make_leaderboard(
        combined392_plus,
        "target_reward_combined392_plus_N32_mapped_u2_primary",
        {
            "constrained_reward": "target_reward_combined392_plus_N32_constrained_u2_reward_balanced",
            "strict_penalty_guard": "target_reward_combined392_plus_N32_strict_penalty_guard",
            "penalty_repair": "target_reward_combined392_plus_N32_penalty_repair",
        },
    )
    write_csv(OUTPUT_DIR / "combined392_plus_N32_per_N_leaderboard.csv", combined392_plus_leaderboard)
    combined392_plus_summary = {
        "rows": int(len(combined392_plus)),
        "per_N_counts": parse_n_counts(combined392_plus),
        "n32_metric_semantic_warning_active": True,
        "n32_rows": int((combined392_plus["n"].astype(int) == 32).sum()),
        "leaderboard": combined392_plus_leaderboard.to_dict(orient="records"),
    }
    write_json(OUTPUT_DIR / "combined392_plus_N32_summary.json", combined392_plus_summary)

    comparison = compare_best(
        baseline=combined328,
        new=run56_ranked,
        combined=combined392,
        baseline_reward_col="target_reward_combined328_u2_primary",
        new_reward_col="reward_run56_u2_primary",
        combined_reward_col="target_reward_combined392_u2_primary",
        ns=[24, 40],
        label="run56_vs_combined328",
    )
    constrained_comparison = compare_best(
        baseline=combined328,
        new=run56_ranked,
        combined=combined392,
        baseline_reward_col="target_reward_combined328_constrained_u2_reward_balanced",
        new_reward_col="reward_run56_constrained_u2_reward_balanced",
        combined_reward_col="target_reward_combined392_constrained_u2_reward_balanced",
        ns=[24, 40],
        label="run56_vs_combined328_constrained",
    )
    constrained_comparison = constrained_comparison[constrained_comparison["metric"] == "combined_reward"].copy()
    constrained_comparison["metric"] = "constrained_u2_reward_balanced"
    strict_comparison = compare_best(
        baseline=combined328,
        new=run56_ranked,
        combined=combined392,
        baseline_reward_col="target_reward_combined328_strict_penalty_guard",
        new_reward_col="reward_run56_strict_penalty_guard",
        combined_reward_col="target_reward_combined392_strict_penalty_guard",
        ns=[24, 40],
        label="run56_vs_combined328_strict_penalty_guard",
    )
    strict_comparison = strict_comparison[strict_comparison["metric"] == "combined_reward"].copy()
    strict_comparison["metric"] = "strict_penalty_guard"
    penalty_repair_comparison = compare_best(
        baseline=combined328,
        new=run56_ranked,
        combined=combined392,
        baseline_reward_col="target_reward_combined328_penalty_repair",
        new_reward_col="reward_run56_penalty_repair",
        combined_reward_col="target_reward_combined392_penalty_repair",
        ns=[24, 40],
        label="run56_vs_combined328_penalty_repair",
    )
    penalty_repair_comparison = penalty_repair_comparison[penalty_repair_comparison["metric"] == "combined_reward"].copy()
    penalty_repair_comparison["metric"] = "penalty_repair"
    comparison = pd.concat([comparison, constrained_comparison, strict_comparison, penalty_repair_comparison], ignore_index=True, sort=False)
    write_csv(OUTPUT_DIR / "run56_vs_combined328_best_comparison.csv", comparison)
    write_json(OUTPUT_DIR / "run56_vs_combined328_best_comparison.json", comparison.to_dict(orient="records"))

    # Key-record context: Run36/Run27 information is already represented in combined392 sources.
    prior_rows: list[dict[str, Any]] = []
    for source in [
        "run51_stricter_constrained_N24_N40_batch32",
        "run46_constrained_N24_N40_batch32",
        "run41_native_N24_N40_focused_batch60",
        "run36_N32_informed_native_batch32",
        "shortlist64_run27",
        "run56_calibrated_N24_N40_batch64",
    ]:
        source_df = combined392[combined392["dataset_source"].astype(str) == source]
        if source_df.empty:
            continue
        for n in [24, 40]:
            group = source_df[source_df["n"].astype(int) == n]
            if group.empty:
                continue
            row = {"source": source, "n": n, "count": int(len(group))}
            for key, col in RAW_METRICS.items():
                best = group.sort_values(col).iloc[0]
                row[f"best_{key}_strategy"] = best.get("strategy_name", "")
                row[f"best_{key}_value"] = float(best[col])
            best_reward = group.sort_values("target_reward_combined392_u2_primary", ascending=False).iloc[0]
            best_constrained = group.sort_values("target_reward_combined392_constrained_u2_reward_balanced", ascending=False).iloc[0]
            best_strict = group.sort_values("target_reward_combined392_strict_penalty_guard", ascending=False).iloc[0]
            best_penalty = group.sort_values("target_reward_combined392_penalty_repair", ascending=False).iloc[0]
            row["best_reward_strategy"] = best_reward.get("strategy_name", "")
            row["best_reward_value_combined392_basis"] = float(best_reward["target_reward_combined392_u2_primary"])
            row["best_constrained_reward_strategy"] = best_constrained.get("strategy_name", "")
            row["best_constrained_reward_value_combined392_basis"] = float(best_constrained["target_reward_combined392_constrained_u2_reward_balanced"])
            row["best_strict_penalty_guard_strategy"] = best_strict.get("strategy_name", "")
            row["best_strict_penalty_guard_value_combined392_basis"] = float(best_strict["target_reward_combined392_strict_penalty_guard"])
            row["best_penalty_repair_strategy"] = best_penalty.get("strategy_name", "")
            row["best_penalty_repair_value_combined392_basis"] = float(best_penalty["target_reward_combined392_penalty_repair"])
            row["top5_u2_entries_combined392"] = int((group["u2_rank_combined392_within_n"] <= 5).sum())
            row["top10_u2_entries_combined392"] = int((group["u2_rank_combined392_within_n"] <= 10).sum())
            row["top5_reward_entries_combined392"] = int((group["reward_rank_combined392_within_n"] <= 5).sum())
            row["top10_reward_entries_combined392"] = int((group["reward_rank_combined392_within_n"] <= 10).sum())
            prior_rows.append(row)
    prior_df = pd.DataFrame(prior_rows)
    write_csv(OUTPUT_DIR / "run56_vs_prior_key_records.csv", prior_df)
    prior_summary = {
        "headline": "Run56 is compared as a native N24/N40 teacher-validation batch against earlier Run36 and Run27 sources through combined392 ranks.",
        "rows": prior_df.to_dict(orient="records"),
    }
    write_json(OUTPUT_DIR / "run56_vs_prior_key_records_summary.json", prior_summary)

    effectiveness_df = group_effectiveness(combined392)
    write_csv(OUTPUT_DIR / "run56_calibrated_batch64_effectiveness_audit.csv", effectiveness_df)
    top_entry_counts = top_counts(combined392, "candidate_source", "reward_rank_combined392_within_n")
    beat_count = int(comparison["run56_beats_baseline"].sum())
    effectiveness_summary = {
        "headline": f"Run56 created {beat_count} new best metric-level records versus combined328 and contributed top5/top10 density in combined392.",
        "new_best_record_count_vs_combined328": beat_count,
        "top_entry_counts_by_n": top_entry_counts.get("by_n", {}),
        "candidate_source_counts": top_entry_counts.get("by_source", {}),
        "group_rows": effectiveness_df.to_dict(orient="records"),
    }
    write_json(OUTPUT_DIR / "run56_calibrated_batch64_effectiveness_summary.json", effectiveness_summary)

    prediction_df, prediction_summary = prediction_audit(run56_ranked)
    write_csv(OUTPUT_DIR / "run56_prediction_audit_for_run53_batch64.csv", prediction_df)
    write_json(OUTPUT_DIR / "run56_prediction_audit_for_run53_batch64_summary.json", prediction_summary)

    u2_beats = comparison[(comparison["metric"] == "U2") & (comparison["run56_beats_baseline"])]
    penalty_rows = []
    for n in [24, 40]:
        if n not in set(u2_beats["n"].astype(int)):
            continue
        run56_best_u2 = run56_ranked[run56_ranked["n"].astype(int) == n].sort_values("u2_range").iloc[0]
        combined_row = combined392[combined392["order_hash"].astype(str) == str(run56_best_u2["order_hash"])].iloc[0]
        penalty_rows.append({
            "n": n,
            "strategy_name": run56_best_u2["strategy_name"],
            "combined392_u2_rank": float(combined_row["u2_rank_combined392_within_n"]),
            "combined392_peeq_rank": float(combined_row["peeq_rank_combined392_within_n"]),
            "combined392_surfaceT_rank": float(combined_row["surfaceT_rank_combined392_within_n"]),
            "combined392_mises_rank": float(combined_row["mises_rank_combined392_within_n"]),
            "combined392_reward_rank": float(combined_row["reward_rank_combined392_within_n"]),
            "combined392_penalty_repair_rank": float(combined_row["penalty_repair_rank_combined392_within_n"]),
        })
    penalty_df = pd.DataFrame(penalty_rows)
    write_csv(OUTPUT_DIR / "run56_u2_gain_vs_penalty_analysis.csv", penalty_df)
    if len(penalty_df):
        penalty_headline = "Run56 U2-best candidates were audited against PEEQ, SurfaceT, Mises, and reward ranks to identify any safety or balance penalties."
    else:
        penalty_headline = "Run56 did not create new U2 bests versus combined328, so no U2-gain penalty rows were generated."

    write_claim_boundary()

    if beat_count > 0 and not u2_beats.empty:
        recommended = "Run58 should update models with native combined392 and combined392_plus_N32, then decide whether to run targeted penalty-repair generation or freeze N24/N40 policy-learning evidence depending on prediction calibration."
    elif effectiveness_summary["top_entry_counts_by_n"]:
        recommended = "Run58 should update models with combined392 and continue a calibrated diagnostic loop around high-density top5/top10 regions."
    else:
        recommended = "Run58 should diagnose over-constraining or weak calibration and return toward a broader native combined392 search."

    maturity_df, maturity_summary, maturity_md = maturity_audit(combined392, prediction_summary, beat_count)
    write_csv(OUTPUT_DIR / "n24_n40_maturity_and_rl_readiness_audit.csv", maturity_df)
    write_json(OUTPUT_DIR / "n24_n40_maturity_and_rl_readiness_summary.json", maturity_summary)
    (OUTPUT_DIR / "n24_n40_maturity_and_rl_readiness_audit.md").write_text(maturity_md, encoding="utf-8")

    payload = {
        "validation": validation,
        "run56_leaderboard": run56_leaderboard,
        "combined392_leaderboard": combined392_leaderboard,
        "combined392_summary": combined392_summary,
        "combined392_plus_N32_summary": combined392_plus_summary,
        "run56_vs_combined392": comparison,
        "prior_summary": prior_summary,
        "effectiveness_summary": effectiveness_summary,
        "prediction_summary": prediction_summary,
        "maturity_summary": maturity_summary,
        "penalty_headline": penalty_headline,
        "recommended_run58": recommended,
    }
    write_report(payload)

    output_files = [
        OUTPUT_DIR / "run57_input_validation_summary.json",
        OUTPUT_DIR / "run56_calibrated_N24_N40_batch64_teacher_dataset_enriched.csv",
        OUTPUT_DIR / "run56_calibrated_N24_N40_batch64_teacher_dataset_enriched.json",
        OUTPUT_DIR / "run56_calibrated_N24_N40_batch64_ranked_within_batch.csv",
        OUTPUT_DIR / "run56_calibrated_N24_N40_batch64_per_N_leaderboard.csv",
        OUTPUT_DIR / "combined392_teacher_dataset.csv",
        OUTPUT_DIR / "combined392_RL_ready_dataset.csv",
        OUTPUT_DIR / "combined392_per_N_leaderboard.csv",
        OUTPUT_DIR / "combined392_summary.json",
        OUTPUT_DIR / "combined392_plus_N32_teacher_dataset.csv",
        OUTPUT_DIR / "combined392_plus_N32_RL_ready_dataset.csv",
        OUTPUT_DIR / "combined392_plus_N32_per_N_leaderboard.csv",
        OUTPUT_DIR / "combined392_plus_N32_summary.json",
        OUTPUT_DIR / "run56_vs_combined328_best_comparison.csv",
        OUTPUT_DIR / "run56_vs_combined328_best_comparison.json",
        OUTPUT_DIR / "run56_vs_prior_key_records.csv",
        OUTPUT_DIR / "run56_vs_prior_key_records_summary.json",
        OUTPUT_DIR / "run56_calibrated_batch64_effectiveness_audit.csv",
        OUTPUT_DIR / "run56_calibrated_batch64_effectiveness_summary.json",
        OUTPUT_DIR / "run56_prediction_audit_for_run53_batch64.csv",
        OUTPUT_DIR / "run56_prediction_audit_for_run53_batch64_summary.json",
        OUTPUT_DIR / "run56_u2_gain_vs_penalty_analysis.csv",
        OUTPUT_DIR / "n24_n40_maturity_and_rl_readiness_audit.csv",
        OUTPUT_DIR / "n24_n40_maturity_and_rl_readiness_summary.json",
        OUTPUT_DIR / "n24_n40_maturity_and_rl_readiness_audit.md",
        CLAIM_BOUNDARY_MD,
        CLAIM_BOUNDARY_JSON,
        REPORT_PATH,
        MANIFEST_PATH,
    ]
    manifest = {
        "run_id": RUN_ID,
        "run_name": RUN_NAME,
        "timestamp": now_iso(),
        "branch": current_branch(),
        "script_path": str(SCRIPT_PATH),
        "input_files": [
            str(RUN56_METRICS), str(RUN56_EXTRACTION), str(RUN56_SOLVER), str(RUN56_SUMMARY),
            str(RUN54_HANDOFF), str(RUN53_POOL), str(RUN53_BATCH64_COMPARISON), str(COMBINED328_TEACHER), str(COMBINED328_READY),
            str(COMBINED328_PLUS_N32_READY), str(N32_DEDUP),
        ],
        "output_files": [str(p) for p in output_files],
        "run56_teacher_rows": 64,
        "combined392_rows": int(len(combined392)),
        "combined392_plus_N32_rows": int(len(combined392_plus)),
        "per_N_combined392_counts": parse_n_counts(combined392),
        "per_N_combined392_plus_N32_counts": parse_n_counts(combined392_plus),
        "new_best_counts": {"run56_vs_combined328_metric_level_records": beat_count},
        "prediction_audit_summary": prediction_summary,
        "maturity_audit_summary": maturity_summary,
        "report_path": str(REPORT_PATH),
        "claim_boundary_path": str(CLAIM_BOUNDARY_MD),
        "no_solver_run": True,
        "no_odb_opened": True,
        "no_abqjobpilot_run": True,
        "no_cae_inp_generated": True,
        "no_teacher_validation_performed_by_run57": True,
        "no_training": True,
        "no_candidate_generation": True,
        "no_commit_or_push": True,
    }
    write_json(MANIFEST_PATH, manifest)
    print(json.dumps({
        "verdict": validation["verdict"],
        "combined392_rows": len(combined392),
        "combined392_plus_N32_rows": len(combined392_plus),
        "new_best_count_vs_combined328": beat_count,
        "report": str(REPORT_PATH),
        "manifest": str(MANIFEST_PATH),
    }, indent=2))


if __name__ == "__main__":
    main()


