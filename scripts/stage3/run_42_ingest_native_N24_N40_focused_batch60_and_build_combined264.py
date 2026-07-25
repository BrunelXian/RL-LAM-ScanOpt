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
RUN_ID = "run_42_native_N24_N40_focused_batch60_teacher_metrics_ingestion_and_combined264_ranking"
RUN_NAME = "native N24/N40 focused batch60 teacher metrics ingestion and combined264 ranking"
SCRIPT_PATH = ROOT / "scripts" / "stage3" / "run_42_ingest_native_N24_N40_focused_batch60_and_build_combined264.py"

RUN41_DIR = ROOT / "outputs" / "stage3_run_41_native_N24_N40_focused_batch60_odb_teacher_extraction"
RUN41_METRICS = RUN41_DIR / "run41_native_N24_N40_focused_batch60_teacher_metrics.csv"
RUN41_EXTRACTION = RUN41_DIR / "run41_native_N24_N40_focused_batch60_odb_extraction_summary.csv"
RUN41_SOLVER = RUN41_DIR / "run41_native_N24_N40_focused_batch60_solver_completion_audit.csv"
RUN41_SUMMARY = RUN41_DIR / "run41_native_N24_N40_focused_batch60_odb_teacher_extraction_summary.json"
RUN41_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_41_native_N24_N40_focused_batch60_odb_teacher_extraction" / "RUN_41_NATIVE_N24_N40_FOCUSED_BATCH60_ODB_TEACHER_EXTRACTION_REPORT.md"
RUN41_MANIFEST = ROOT / "artifacts" / "manifests" / "stage3_run_41_manifest.json"

RUN39_HANDOFF = ROOT / "outputs" / "stage3_run_39_run38_native_N24_N40_focused_batch60_handoff_package" / "stage3_run39_native_N24_N40_focused_batch60_candidate_orders.csv"
RUN39_SCAN_DIR = ROOT / "outputs" / "stage3_run_39_run38_native_N24_N40_focused_batch60_handoff_package" / "scan_orders"
RUN38_POOL = ROOT / "outputs" / "stage3_run_38_combined204_and_combined204_plus_N32_model_update_candidate_generation" / "run38_candidate_pool_scored.csv"
RUN38_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_38_combined204_and_combined204_plus_N32_model_update_candidate_generation" / "RUN_38_COMBINED204_AND_COMBINED204_PLUS_N32_MODEL_UPDATE_CANDIDATE_GENERATION_REPORT.md"

COMBINED204_TEACHER = ROOT / "outputs" / "stage3_run_37_N32_informed_native_batch32_teacher_metrics_ingestion_and_combined204_ranking" / "combined204_teacher_dataset.csv"
COMBINED204_READY = ROOT / "outputs" / "stage3_run_37_N32_informed_native_batch32_teacher_metrics_ingestion_and_combined204_ranking" / "combined204_RL_ready_dataset.csv"
COMBINED204_PLUS_N32_READY = ROOT / "outputs" / "stage3_run_37_N32_informed_native_batch32_teacher_metrics_ingestion_and_combined204_ranking" / "combined204_plus_N32_RL_ready_dataset.csv"
N32_DEDUP = ROOT / "outputs" / "stage3_run_32a_stage2_n32_legacy_teacher_label_ingestion_for_stage3" / "n32_legacy_teacher_dataset_dedup_training_332.csv"
RUN37_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_37_N32_informed_native_batch32_teacher_metrics_ingestion_and_combined204_ranking" / "RUN_37_N32_INFORMED_NATIVE_BATCH32_TEACHER_METRICS_INGESTION_AND_COMBINED204_RANKING_REPORT.md"

OUTPUT_DIR = ROOT / "outputs" / "stage3_run_42_native_N24_N40_focused_batch60_teacher_metrics_ingestion_and_combined264_ranking"
REPORT_DIR = ROOT / "docs" / "stage3" / "runs" / "run_42_native_N24_N40_focused_batch60_teacher_metrics_ingestion_and_combined264_ranking"
REPORT_PATH = REPORT_DIR / "RUN_42_NATIVE_N24_N40_FOCUSED_BATCH60_TEACHER_METRICS_INGESTION_AND_COMBINED264_RANKING_REPORT.md"
MANIFEST_PATH = ROOT / "artifacts" / "manifests" / "stage3_run_42_manifest.json"
CLAIM_BOUNDARY_MD = OUTPUT_DIR / "run42_claim_boundary.md"
CLAIM_BOUNDARY_JSON = OUTPUT_DIR / "run42_claim_boundary.json"

EXPECTED_RUN41_COUNTS = {24: 30, 40: 30}
EXPECTED_COMBINED204_COUNTS = {12: 36, 16: 36, 24: 66, 40: 66}
EXPECTED_COMBINED204_PLUS_N32_COUNTS = {12: 36, 16: 36, 24: 66, 32: 332, 40: 66}
EXPECTED_COMBINED264_COUNTS = {12: 36, 16: 36, 24: 96, 40: 96}
EXPECTED_COMBINED264_PLUS_N32_COUNTS = {12: 36, 16: 36, 24: 96, 32: 332, 40: 96}

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


def add_run41_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    labels = {"u2": "U2", "peeq": "PEEQ", "surfaceT": "SurfaceT", "mises": "Mises"}
    for metric_key, metric_col in RAW_METRICS.items():
        label = labels[metric_key]
        out[f"rank_{label}_run41_within_n"] = math.nan
        out[f"score_{label}_run41_within_n"] = math.nan
        out[f"cost_{metric_key}_run41_within_n"] = math.nan
        for _, idx in out.groupby("n").groups.items():
            ranks = out.loc[idx, metric_col].rank(method="average", ascending=True)
            out.loc[idx, f"rank_{label}_run41_within_n"] = ranks
            out.loc[idx, f"score_{label}_run41_within_n"] = rank_score(ranks, len(idx))
            out.loc[idx, f"cost_{metric_key}_run41_within_n"] = minmax_cost(out.loc[idx, metric_col])
    out["reward_run41_u2_primary"] = (
        0.65 * out["score_U2_run41_within_n"]
        + 0.20 * out["score_PEEQ_run41_within_n"]
        + 0.10 * out["score_SurfaceT_run41_within_n"]
        + 0.05 * out["score_Mises_run41_within_n"]
    )
    out["rank_reward_run41_within_n"] = math.nan
    for _, idx in out.groupby("n").groups.items():
        out.loc[idx, "rank_reward_run41_within_n"] = out.loc[idx, "reward_run41_u2_primary"].rank(method="average", ascending=False)
    return out


def first_col(df: pd.DataFrame, names: list[str]) -> str | None:
    return next((c for c in names if c in df.columns), None)


def make_run41_enriched(run41: pd.DataFrame, handoff: pd.DataFrame, solver: pd.DataFrame) -> pd.DataFrame:
    run41 = normalize_metric_columns(run41)
    handoff = handoff.copy()
    for frame in [run41, handoff]:
        if "n" in frame.columns:
            frame["n"] = frame["n"].astype(int)
        if "order_json" in frame.columns:
            frame["order_json"] = frame["order_json"].astype(str)

    merged = run41.merge(
        handoff,
        on=["handoff_strategy_name", "n"],
        how="left",
        suffixes=("", "_run39"),
        validate="one_to_one",
    )
    if "order_json" not in merged.columns and "order_json_run39" in merged.columns:
        merged["order_json"] = merged["order_json_run39"]
    if "order_json_run39" in merged.columns:
        merged["order_json"] = merged["order_json"].fillna(merged["order_json_run39"])
    if "order_hash" in merged.columns and "order_hash_run39" in merged.columns:
        merged["order_hash"] = merged["order_hash"].fillna(merged["order_hash_run39"])
    elif "order_hash_run39" in merged.columns:
        merged["order_hash"] = merged["order_hash_run39"]

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
    merged["dataset_source"] = "run41_native_N24_N40_focused_batch60"
    merged["batch_name"] = "stage3_run39_native_N24_N40_focused_batch60_v01"
    merged["native_validation_N"] = True
    merged["N24_N40_focused"] = True
    merged["overnight_batch60"] = True
    merged["includes_N12_case"] = False
    merged["includes_N16_case"] = False
    merged["includes_N32_case"] = False
    merged["final_step"] = merged.get("final_step_name", "")
    merged["extracted_fields"] = merged.get("extracted_field_names", "")
    merged["solver_audit_status"] = merged.get("completion_status", "")
    merged["nonfatal_warning_flag"] = merged.get("completion_status", "").astype(str).str.contains("WARNING", case=False, na=False)
    merged["notes"] = "Run42 ingestion of Run41 native N24/N40 focused batch60 teacher metrics. No N12/N16/N32 cases."

    preferred = [
        "n", "strategy_name", "handoff_strategy_name", "job_name", "dataset_source", "batch_name",
        "native_validation_N", "N24_N40_focused", "overnight_batch60",
        "includes_N12_case", "includes_N16_case", "includes_N32_case",
        "original_run38_candidate_id", "original_run38_strategy_name", "candidate_source",
        "generation_method", "selection_bucket", "priority_role", "surrogate_prediction",
        "gnn_reward_prediction", "graph_pointer_policy_score", "hybrid_score", "uncertainty_score",
        "gnn_vs_surrogate_disagreement", "novelty_distance", "nearest_existing_teacher_strategy",
        "order_json", "order_compact", "order_hash", "u2_range", "peeq_max", "surface_t_proxy",
        "surface_t_proxy_mpa", "mises_max", "final_step", "final_step_name", "final_frame_time",
        "extracted_fields", "extracted_field_names", "teacher_validation_status",
        "solver_audit_status", "completion_status", "odb_extraction_status", "nonfatal_warning_flag",
        "notes",
    ]
    cols = [c for c in preferred if c in merged.columns]
    cols += [c for c in merged.columns if c not in cols]
    return merged[cols]


def make_leaderboard(df: pd.DataFrame, reward_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
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
                "run41_best_strategy": new_best.get("strategy_name", new_best.get("handoff_strategy_name", "")),
                "run41_best_source": new_best.get("dataset_source", ""),
                "run41_best_value": new_val,
                "run41_beats_baseline": bool(beat),
                "absolute_improvement": abs_improvement,
                "relative_improvement_fraction": rel,
                "combined264_best_strategy": comb_best.get("strategy_name", comb_best.get("handoff_strategy_name", "")),
                "combined264_best_source": comb_best.get("dataset_source", ""),
                "combined264_best_value": float(comb_best[comb_col]),
            })
    return pd.DataFrame(rows)


def top_counts(df: pd.DataFrame, source_col: str, reward_rank_col: str) -> dict[str, Any]:
    run41 = df[df["dataset_source"].astype(str) == "run41_native_N24_N40_focused_batch60"]
    payload: dict[str, Any] = {"total_run41_rows": int(len(run41)), "by_n": {}}
    for n, group in run41.groupby("n"):
        payload["by_n"][f"N{int(n)}"] = {
            "top5_u2_entries": int((group["u2_rank_combined264_within_n"] <= 5).sum()),
            "top10_u2_entries": int((group["u2_rank_combined264_within_n"] <= 10).sum()),
            "top5_reward_entries": int((group[reward_rank_col] <= 5).sum()),
            "top10_reward_entries": int((group[reward_rank_col] <= 10).sum()),
        }
    if source_col in run41.columns:
        payload["by_source"] = run41[source_col].fillna("unknown").value_counts().to_dict()
    return payload


def group_effectiveness(combined: pd.DataFrame) -> pd.DataFrame:
    run41 = combined[combined["dataset_source"].astype(str) == "run41_native_N24_N40_focused_batch60"].copy()
    rows: list[dict[str, Any]] = []
    group_cols = ["n", "candidate_source", "generation_method", "selection_bucket", "priority_role"]
    for col in group_cols:
        if col not in run41.columns:
            continue
        for key, group in run41.groupby(col, dropna=False):
            rows.append({
                "group_type": col,
                "group_value": str(key),
                "count": int(len(group)),
                "median_u2_rank_combined264": float(group["u2_rank_combined264_within_n"].median()),
                "best_u2_rank_combined264": float(group["u2_rank_combined264_within_n"].min()),
                "median_reward_rank_combined264": float(group["reward_rank_combined264_within_n"].median()),
                "best_reward_rank_combined264": float(group["reward_rank_combined264_within_n"].min()),
                "top5_u2_count": int((group["u2_rank_combined264_within_n"] <= 5).sum()),
                "top10_u2_count": int((group["u2_rank_combined264_within_n"] <= 10).sum()),
                "top5_reward_count": int((group["reward_rank_combined264_within_n"] <= 5).sum()),
                "top10_reward_count": int((group["reward_rank_combined264_within_n"] <= 10).sum()),
            })
    return pd.DataFrame(rows)


def spearman_safe(a: pd.Series, b: pd.Series) -> float | None:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 3 or x[mask].nunique() < 2 or y[mask].nunique() < 2:
        return None
    return float(x[mask].corr(y[mask], method="spearman"))


def prediction_audit(run41_ranked: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = run41_ranked.copy()
    pred_col = first_col(df, ["hybrid_score", "surrogate_prediction", "model_prediction_mean", "pred_reward_native_combined204"])
    u2_pred_col = first_col(df, ["surrogate_prediction", "pred_u2_score", "hybrid_score"])
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "prediction_column_used_for_reward": pred_col,
        "prediction_column_used_for_u2": u2_pred_col,
        "overall_reward_spearman": None,
        "overall_u2_spearman": None,
        "top1_hit": None,
        "mean_top5_overlap": None,
        "per_n": {},
        "by_selection_bucket": {},
    }
    if pred_col:
        df[pred_col] = pd.to_numeric(df[pred_col], errors="coerce")
        summary["overall_reward_spearman"] = spearman_safe(df[pred_col], df["reward_run41_u2_primary"])
    if u2_pred_col:
        df[u2_pred_col] = pd.to_numeric(df[u2_pred_col], errors="coerce")
        summary["overall_u2_spearman"] = spearman_safe(df[u2_pred_col], df["score_U2_run41_within_n"])

    top1_hits = 0
    top5_overlaps: list[int] = []
    for n, group in df.groupby("n"):
        item: dict[str, Any] = {"count": int(len(group))}
        if pred_col and group[pred_col].notna().any():
            item["reward_spearman"] = spearman_safe(group[pred_col], group["reward_run41_u2_primary"])
            predicted = list(group.sort_values(pred_col, ascending=False)["handoff_strategy_name"].head(5))
            realized = list(group.sort_values("reward_run41_u2_primary", ascending=False)["handoff_strategy_name"].head(5))
            overlap = len(set(predicted) & set(realized))
            top5_overlaps.append(overlap)
            top1_hits += int(predicted[:1] == realized[:1])
            item["top5_overlap"] = overlap
            item["predicted_top1"] = predicted[0] if predicted else ""
            item["realized_top1"] = realized[0] if realized else ""
        if u2_pred_col and group[u2_pred_col].notna().any():
            item["u2_spearman"] = spearman_safe(group[u2_pred_col], group["score_U2_run41_within_n"])
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
                    "reward_spearman": spearman_safe(group[pred_col], group["reward_run41_u2_primary"]),
                    "mean_abs_reward_error": float((pd.to_numeric(group[pred_col], errors="coerce") - group["reward_run41_u2_primary"]).abs().mean()),
                })
            rows.extend(bucket_rows)

    for diag_col in ["gnn_vs_surrogate_disagreement", "uncertainty_score", "novelty_distance"]:
        if diag_col in df.columns and pred_col:
            err = (pd.to_numeric(df[pred_col], errors="coerce") - df["reward_run41_u2_primary"]).abs()
            summary[f"{diag_col}_vs_abs_error_spearman"] = spearman_safe(df[diag_col], err)

    return pd.DataFrame(rows), summary


def validate_inputs(run41: pd.DataFrame, handoff: pd.DataFrame, combined204: pd.DataFrame, plus_n32: pd.DataFrame) -> dict[str, Any]:
    errors: list[str] = []
    run41_counts = parse_n_counts(run41)
    combined204_counts = parse_n_counts(combined204)
    plus_counts = parse_n_counts(plus_n32)
    if run41_counts != EXPECTED_RUN41_COUNTS:
        errors.append(f"Run41 counts mismatch: {run41_counts}")
    if set(run41["n"].astype(int)) - {24, 40}:
        errors.append("Run41 contains non-N24/N40 rows")
    if combined204_counts != EXPECTED_COMBINED204_COUNTS:
        errors.append(f"combined204 counts mismatch: {combined204_counts}")
    if plus_counts != EXPECTED_COMBINED204_PLUS_N32_COUNTS:
        errors.append(f"combined204_plus_N32 counts mismatch: {plus_counts}")
    for col in ["u2_range", "peeq_max", "mises_max"]:
        if col not in run41.columns:
            errors.append(f"Run41 missing metric {col}")
    if "surface_t_proxy" not in run41.columns and "surface_t_proxy_max_tensile_pa" not in run41.columns:
        errors.append("Run41 missing surface_t_proxy metric")
    if "teacher_validation_status" not in run41.columns or not (run41["teacher_validation_status"].astype(str) == "PASS_TEACHER_FIELDS_EXTRACTED").all():
        errors.append("Run41 teacher_validation_status is not PASS for all rows")
    if "final_step_name" in run41.columns and not (run41["final_step_name"].astype(str) == "step_final_cooling").all():
        errors.append("Run41 final_step_name is not step_final_cooling for all rows")
    if "extracted_field_names" in run41.columns:
        for field in ["U", "PEEQ", "S", "NT11"]:
            if not run41["extracted_field_names"].astype(str).str.contains(field).all():
                errors.append(f"Run41 extracted fields missing {field}")
    for col in RAW_METRICS.values():
        source = "surface_t_proxy_max_tensile_pa" if col == "surface_t_proxy" and col not in run41.columns else col
        if source in run41.columns and pd.to_numeric(run41[source], errors="coerce").isna().any():
            errors.append(f"Run41 has missing metric values in {source}")
    missing_handoff = sorted(set(run41["handoff_strategy_name"]) - set(handoff["handoff_strategy_name"]))
    if missing_handoff:
        errors.append(f"Run41 rows unmatched to Run39 handoff: {missing_handoff[:5]}")
    bad_orders = []
    for _, row in handoff.iterrows():
        if not valid_order(row["order_json"], int(row["n"])):
            bad_orders.append(row.get("handoff_strategy_name", "UNKNOWN"))
    if bad_orders:
        errors.append(f"Run39 handoff has invalid scan orders: {bad_orders[:5]}")
    n32 = plus_n32[plus_n32["n"].astype(int) == 32]
    warning_col = first_col(n32, ["metric_semantic_warning", "legacy_compatibility_status", "compatibility_status"])
    if warning_col is None:
        errors.append("combined204_plus_N32 N32 rows do not carry semantic warning/status columns")

    verdict = "PASS_RUN42_NATIVE_N24_N40_FOCUSED_BATCH60_TEACHER_METRICS_60_OF_60_READY" if not errors else "FAIL_RUN42_INPUT_VALIDATION"
    return {
        "timestamp": now_iso(),
        "verdict": verdict,
        "errors": errors,
        "run41_rows": int(len(run41)),
        "run41_per_N_counts": run41_counts,
        "combined204_rows": int(len(combined204)),
        "combined204_per_N_counts": combined204_counts,
        "combined204_plus_N32_rows": int(len(plus_n32)),
        "combined204_plus_N32_per_N_counts": plus_counts,
        "run41_has_N12": bool((run41["n"].astype(int) == 12).any()),
        "run41_has_N16": bool((run41["n"].astype(int) == 16).any()),
        "run41_has_N32": bool((run41["n"].astype(int) == 32).any()),
        "run41_rows_matched_to_run39_handoff": int(len(run41) - len(missing_handoff)),
        "n32_semantic_warning_column": warning_col,
    }


def write_claim_boundary() -> None:
    safe = [
        "Run42 ingests 60/60 teacher-validated Run41 native N24/N40 focused batch60 cases.",
        "Run42 builds native combined264 with N12=36, N16=36, N24=96, N40=96.",
        "Run42 builds combined264_plus_N32 with N12=36, N16=36, N24=96, N32=332, N40=96.",
        "Run42 evaluates whether the N24/N40 focused batch60 improved native Stage 3 teacher metrics.",
        "Run41 is teacher validation of native N24/N40 candidates, not N32 cases.",
    ]
    unsafe = [
        "Do not claim N32 itself was newly teacher-validated in Run41.",
        "Do not claim N32 caused Run41 improvements.",
        "Do not claim GNN-RL superiority.",
        "Do not claim online RL.",
        "Do not claim arbitrary-N generalization.",
        "Do not claim physical optimum.",
        "Do not claim solver/ODB extraction happened in Run42.",
    ]
    CLAIM_BOUNDARY_MD.write_text(
        "# Run42 Claim Boundary\n\n## Safe claims\n\n"
        + "\n".join(f"- {x}" for x in safe)
        + "\n\n## Unsafe claims\n\n"
        + "\n".join(f"- {x}" for x in unsafe)
        + "\n",
        encoding="utf-8",
    )
    write_json(CLAIM_BOUNDARY_JSON, {
        "verdict": "RUN42_INGESTION_AND_COMBINED264_RANKING_ONLY_NO_SOLVER_OR_TRAINING",
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


def write_report(payload: dict[str, Any]) -> None:
    validation = payload["validation"]
    combined_summary = payload["combined264_summary"]
    plus_summary = payload["combined264_plus_N32_summary"]
    comparison = payload["run41_vs_combined204"]
    prior_summary = payload["prior_summary"]
    effectiveness = payload["effectiveness_summary"]
    pred = payload["prediction_summary"]

    def fmt_counts(counts: dict[int, int] | dict[str, int]) -> str:
        return ", ".join(f"N{k}={v}" for k, v in counts.items())

    run41_best = payload["run41_leaderboard"]
    combined_best = payload["combined264_leaderboard"]
    report = f"""# Stage 3 Run 42 - Native N24/N40 Focused Batch60 Teacher Metrics Ingestion and Combined264 Ranking

## 1. Purpose
Run42 ingests the completed Run41 native N24/N40 focused batch60 teacher metrics, merges them with Run39/Run38 candidate metadata, recomputes within-N rankings, and builds native combined264 plus combined264_plus_N32 datasets.

## 2. Inputs
- Run41 teacher metrics: `{RUN41_METRICS}`
- Run39 handoff metadata: `{RUN39_HANDOFF}`
- Native combined204 RL-ready dataset: `{COMBINED204_READY}`
- combined204_plus_N32 RL-ready dataset: `{COMBINED204_PLUS_N32_READY}`
- N32 deduplicated legacy-compatible table: `{N32_DEDUP}`

## 3. Run41 Teacher-Extraction Status
Run41 was complete for 60/60 teacher-extracted cases: N24=30 and N40=30. It contains no N12, N16, or N32 cases. Run41 is native N24/N40 teacher validation, not N32 teacher validation.

## 4. Input Validation
Verdict: `{validation['verdict']}`.

Run41 counts: {fmt_counts(validation['run41_per_N_counts'])}. Native combined204 counts: {fmt_counts(validation['combined204_per_N_counts'])}. combined204_plus_N32 counts: {fmt_counts(validation['combined204_plus_N32_per_N_counts'])}.

## 5. Run41 Enriched Teacher Dataset
Run42 produced an enriched Run41 teacher dataset with handoff names, Run38/Run39 prediction metadata, candidate-source metadata, scan orders, hashes, raw teacher metrics, extraction status, and nonfatal-warning flags.

## 6. Run41 Within-Batch Ranking
Within Run41, lower raw metric values are better for U2, PEEQ, SurfaceT, and Mises. The U2-primary reward uses 0.65 U2, 0.20 PEEQ, 0.10 SurfaceT, and 0.05 Mises rank scores.

Run41 best U2 by N:
{markdown_table(run41_best, ['n', 'best_u2_strategy', 'best_u2_value'])}

Run41 best combined reward by N:
{markdown_table(run41_best, ['n', 'best_reward_strategy', 'best_reward_value'])}

## 7. Native Combined264 Construction
Native combined264 rows: {combined_summary['rows']}. Counts: {fmt_counts(combined_summary['per_N_counts'])}. There are no N32 rows in native combined264.

## 8. combined264_plus_N32 Construction
combined264_plus_N32 rows: {plus_summary['rows']}. Counts: {fmt_counts(plus_summary['per_N_counts'])}. N32 rows preserve the legacy metric semantic warnings from Run32A.

## 9. Run41 vs Combined204 Best Comparison
Run41 was compared against the native combined204 best records for N24 and N40 across U2, PEEQ, SurfaceT, Mises, and recomputed combined reward.

{markdown_table(comparison, ['n', 'metric', 'run41_beats_baseline', 'baseline_best_strategy', 'run41_best_strategy', 'absolute_improvement'])}

## 10. Run41 vs Prior Key Records
Run41 was compared against combined204, Run36, Run27, and the earlier combined172 baseline where available. Summary: {prior_summary['headline']}

## 11. N24/N40 Focused Batch60 Effectiveness Audit
{effectiveness['headline']}

Top-entry counts by N:
```json
{json.dumps(effectiveness['top_entry_counts_by_n'], indent=2)}
```

## 12. Prediction Audit for Run38 Batch60
Prediction column used for reward audit: `{pred.get('prediction_column_used_for_reward')}`. Overall reward Spearman: `{pred.get('overall_reward_spearman')}`. Top1 hit: `{pred.get('top1_hit')}`. Mean top5 overlap: `{pred.get('mean_top5_overlap')}`.

Per-N prediction audit:
```json
{json.dumps(pred.get('per_n', {}), indent=2)}
```

## 13. U2 Gain Versus Penalty Analysis
{payload['penalty_headline']}

## 14. Metric Semantic Boundary for N32
combined264_plus_N32 includes N32 legacy-compatible rows. These rows are not native Stage 3 teacher validation. PEEQ is mapped from Stage 2 `peeq_guard`, and Mises is mapped from `mises_P95_top_band`; they are proxy-compatible fields with warnings, not literal native Stage 3 metric identities.

## 15. Claim Boundary
Claim boundary verdict: `RUN42_INGESTION_AND_COMBINED264_RANKING_ONLY_NO_SOLVER_OR_TRAINING`.

## 16. Output Files
- Run41 enriched dataset: `{OUTPUT_DIR / 'run41_native_N24_N40_focused_batch60_teacher_dataset_enriched.csv'}`
- Run41 ranked within batch: `{OUTPUT_DIR / 'run41_native_N24_N40_focused_batch60_ranked_within_batch.csv'}`
- Native combined264 RL-ready dataset: `{OUTPUT_DIR / 'combined264_RL_ready_dataset.csv'}`
- combined264_plus_N32 RL-ready dataset: `{OUTPUT_DIR / 'combined264_plus_N32_RL_ready_dataset.csv'}`
- Run41 vs combined204 comparison: `{OUTPUT_DIR / 'run41_vs_combined204_best_comparison.csv'}`
- Effectiveness audit: `{OUTPUT_DIR / 'run41_N24_N40_focused_batch60_effectiveness_audit.csv'}`
- Prediction audit: `{OUTPUT_DIR / 'run41_prediction_audit_for_run38_batch60.csv'}`
- Manifest: `{MANIFEST_PATH}`

## 17. Recommended Run43
{payload['recommended_run43']}
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    ensure_dirs()

    run41_raw = read_csv(RUN41_METRICS)
    handoff = read_csv(RUN39_HANDOFF)
    combined204 = normalize_metric_columns(read_csv(COMBINED204_READY))
    combined204_teacher = normalize_metric_columns(read_csv(COMBINED204_TEACHER))
    combined204_plus = normalize_metric_columns(read_csv(COMBINED204_PLUS_N32_READY))
    n32 = normalize_metric_columns(read_csv(N32_DEDUP))
    solver = read_csv(RUN41_SOLVER) if RUN41_SOLVER.exists() else pd.DataFrame()

    validation = validate_inputs(run41_raw, handoff, combined204, combined204_plus)
    write_json(OUTPUT_DIR / "run42_input_validation_summary.json", validation)
    if not validation["verdict"].startswith("PASS"):
        raise SystemExit(f"Input validation failed: {validation['errors']}")

    run41_enriched = make_run41_enriched(run41_raw, handoff, solver)
    write_csv(OUTPUT_DIR / "run41_native_N24_N40_focused_batch60_teacher_dataset_enriched.csv", run41_enriched)
    write_table_json(OUTPUT_DIR / "run41_native_N24_N40_focused_batch60_teacher_dataset_enriched.json", run41_enriched)

    run41_ranked = add_run41_scores(run41_enriched)
    write_csv(OUTPUT_DIR / "run41_native_N24_N40_focused_batch60_ranked_within_batch.csv", run41_ranked)
    run41_leaderboard = make_leaderboard(run41_ranked, "reward_run41_u2_primary")
    write_csv(OUTPUT_DIR / "run41_native_N24_N40_focused_batch60_per_N_leaderboard.csv", run41_leaderboard)

    combined264 = pd.concat([combined204, run41_enriched], ignore_index=True, sort=False)
    combined264["n"] = combined264["n"].astype(int)
    combined264 = normalize_metric_columns(combined264)
    combined264 = add_scores(
        combined264,
        rank_suffix="combined264_within_n",
        target_prefix="target",
        cost_suffix="combined264_within_n",
        reward_col="target_reward_combined264_u2_primary",
        reward_rank_col="reward_rank_combined264_within_n",
    )
    combined264 = combined264.rename(columns={
        "target_u2_score": "target_u2_score_combined264_rank",
        "target_peeq_score": "target_peeq_score_combined264_rank",
        "target_surfaceT_score": "target_surfaceT_score_combined264_rank",
        "target_mises_score": "target_mises_score_combined264_rank",
    })
    write_csv(OUTPUT_DIR / "combined264_teacher_dataset.csv", combined264)
    write_csv(OUTPUT_DIR / "combined264_RL_ready_dataset.csv", combined264)
    combined264_leaderboard = make_leaderboard(combined264, "target_reward_combined264_u2_primary")
    write_csv(OUTPUT_DIR / "combined264_per_N_leaderboard.csv", combined264_leaderboard)
    combined264_summary = {
        "rows": int(len(combined264)),
        "per_N_counts": parse_n_counts(combined264),
        "no_N32_rows": bool((combined264["n"].astype(int) == 32).sum() == 0),
        "leaderboard": combined264_leaderboard.to_dict(orient="records"),
    }
    write_json(OUTPUT_DIR / "combined264_summary.json", combined264_summary)

    n32_for_plus = n32.copy()
    n32_for_plus["n"] = n32_for_plus["n"].astype(int)
    n32_for_plus["legacy_compatibility_status"] = "LEGACY_COMPATIBLE_WITH_WARNINGS"
    n32_for_plus["metric_semantic_warning"] = True
    combined264_plus = pd.concat([combined264, n32_for_plus], ignore_index=True, sort=False)
    combined264_plus["n"] = combined264_plus["n"].astype(int)
    combined264_plus["metric_semantic_warning"] = combined264_plus["metric_semantic_warning"].fillna(False)
    combined264_plus["legacy_compatibility_status"] = combined264_plus.get("legacy_compatibility_status", "").replace("", pd.NA)
    combined264_plus["legacy_compatibility_status"] = combined264_plus["legacy_compatibility_status"].fillna("NATIVE_STAGE3")
    combined264_plus = normalize_metric_columns(combined264_plus)
    combined264_plus = add_scores(
        combined264_plus,
        rank_suffix="combined264_plus_N32_within_n",
        target_prefix="target",
        cost_suffix="combined264_plus_N32_within_n",
        reward_col="target_reward_combined264_plus_N32_mapped_u2_primary",
        reward_rank_col="reward_rank_combined264_plus_N32_within_n",
    )
    combined264_plus = combined264_plus.rename(columns={
        "target_u2_score": "target_u2_score_combined264_plus_N32_rank",
        "target_peeq_score": "target_peeq_score_combined264_plus_N32_rank",
        "target_surfaceT_score": "target_surfaceT_score_combined264_plus_N32_rank",
        "target_mises_score": "target_mises_score_combined264_plus_N32_rank",
    })
    combined264_plus["target_reward_combined264_plus_N32_strict_u2_surfaceT"] = (
        0.80 * combined264_plus["target_u2_score_combined264_plus_N32_rank"]
        + 0.20 * combined264_plus["target_surfaceT_score_combined264_plus_N32_rank"]
    )
    write_csv(OUTPUT_DIR / "combined264_plus_N32_teacher_dataset.csv", combined264_plus)
    write_csv(OUTPUT_DIR / "combined264_plus_N32_RL_ready_dataset.csv", combined264_plus)
    combined264_plus_leaderboard = make_leaderboard(combined264_plus, "target_reward_combined264_plus_N32_mapped_u2_primary")
    write_csv(OUTPUT_DIR / "combined264_plus_N32_per_N_leaderboard.csv", combined264_plus_leaderboard)
    combined264_plus_summary = {
        "rows": int(len(combined264_plus)),
        "per_N_counts": parse_n_counts(combined264_plus),
        "n32_metric_semantic_warning_active": True,
        "n32_rows": int((combined264_plus["n"].astype(int) == 32).sum()),
        "leaderboard": combined264_plus_leaderboard.to_dict(orient="records"),
    }
    write_json(OUTPUT_DIR / "combined264_plus_N32_summary.json", combined264_plus_summary)

    comparison = compare_best(
        baseline=combined204,
        new=run41_ranked,
        combined=combined264,
        baseline_reward_col="target_reward_combined204_u2_primary",
        new_reward_col="reward_run41_u2_primary",
        combined_reward_col="target_reward_combined264_u2_primary",
        ns=[24, 40],
        label="run41_vs_combined204",
    )
    write_csv(OUTPUT_DIR / "run41_vs_combined204_best_comparison.csv", comparison)
    write_json(OUTPUT_DIR / "run41_vs_combined204_best_comparison.json", comparison.to_dict(orient="records"))

    # Key-record context: Run36/Run27 information is already represented in combined204 sources.
    prior_rows: list[dict[str, Any]] = []
    for source in ["run36_N32_informed_native_batch32", "shortlist64_run27", "run41_native_N24_N40_focused_batch60"]:
        source_df = combined264[combined264["dataset_source"].astype(str) == source]
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
            best_reward = group.sort_values("target_reward_combined264_u2_primary", ascending=False).iloc[0]
            row["best_reward_strategy"] = best_reward.get("strategy_name", "")
            row["best_reward_value_combined264_basis"] = float(best_reward["target_reward_combined264_u2_primary"])
            row["top5_u2_entries_combined264"] = int((group["u2_rank_combined264_within_n"] <= 5).sum())
            row["top10_u2_entries_combined264"] = int((group["u2_rank_combined264_within_n"] <= 10).sum())
            row["top5_reward_entries_combined264"] = int((group["reward_rank_combined264_within_n"] <= 5).sum())
            row["top10_reward_entries_combined264"] = int((group["reward_rank_combined264_within_n"] <= 10).sum())
            prior_rows.append(row)
    prior_df = pd.DataFrame(prior_rows)
    write_csv(OUTPUT_DIR / "run41_vs_prior_key_records.csv", prior_df)
    prior_summary = {
        "headline": "Run41 is compared as a native N24/N40 teacher-validation batch against earlier Run36 and Run27 sources through combined264 ranks.",
        "rows": prior_df.to_dict(orient="records"),
    }
    write_json(OUTPUT_DIR / "run41_vs_prior_key_records_summary.json", prior_summary)

    effectiveness_df = group_effectiveness(combined264)
    write_csv(OUTPUT_DIR / "run41_N24_N40_focused_batch60_effectiveness_audit.csv", effectiveness_df)
    top_entry_counts = top_counts(combined264, "candidate_source", "reward_rank_combined264_within_n")
    beat_count = int(comparison["run41_beats_baseline"].sum())
    effectiveness_summary = {
        "headline": f"Run41 created {beat_count} new best metric-level records versus combined204 and contributed top5/top10 density in combined264.",
        "new_best_record_count_vs_combined204": beat_count,
        "top_entry_counts_by_n": top_entry_counts.get("by_n", {}),
        "candidate_source_counts": top_entry_counts.get("by_source", {}),
        "group_rows": effectiveness_df.to_dict(orient="records"),
    }
    write_json(OUTPUT_DIR / "run41_N24_N40_focused_batch60_effectiveness_summary.json", effectiveness_summary)

    prediction_df, prediction_summary = prediction_audit(run41_ranked)
    write_csv(OUTPUT_DIR / "run41_prediction_audit_for_run38_batch60.csv", prediction_df)
    write_json(OUTPUT_DIR / "run41_prediction_audit_for_run38_batch60_summary.json", prediction_summary)

    u2_beats = comparison[(comparison["metric"] == "U2") & (comparison["run41_beats_baseline"])]
    penalty_rows = []
    for n in [24, 40]:
        if n not in set(u2_beats["n"].astype(int)):
            continue
        run41_best_u2 = run41_ranked[run41_ranked["n"].astype(int) == n].sort_values("u2_range").iloc[0]
        combined_row = combined264[combined264["order_hash"].astype(str) == str(run41_best_u2["order_hash"])].iloc[0]
        penalty_rows.append({
            "n": n,
            "strategy_name": run41_best_u2["strategy_name"],
            "combined264_u2_rank": float(combined_row["u2_rank_combined264_within_n"]),
            "combined264_peeq_rank": float(combined_row["peeq_rank_combined264_within_n"]),
            "combined264_surfaceT_rank": float(combined_row["surfaceT_rank_combined264_within_n"]),
            "combined264_mises_rank": float(combined_row["mises_rank_combined264_within_n"]),
            "combined264_reward_rank": float(combined_row["reward_rank_combined264_within_n"]),
        })
    penalty_df = pd.DataFrame(penalty_rows)
    write_csv(OUTPUT_DIR / "run41_u2_gain_vs_penalty_analysis.csv", penalty_df)
    if len(penalty_df):
        penalty_headline = "Run41 U2-best candidates were audited against PEEQ, SurfaceT, Mises, and reward ranks to identify any safety or balance penalties."
    else:
        penalty_headline = "Run41 did not create new U2 bests versus combined204, so no U2-gain penalty rows were generated."

    write_claim_boundary()

    if beat_count > 0 and not u2_beats.empty:
        recommended = "Run43 should update models with native combined264 and combined264_plus_N32, then generate a constrained N24/N40 candidate design that exploits U2 gains while explicitly guarding PEEQ and SurfaceT."
    elif effectiveness_summary["top_entry_counts_by_n"]:
        recommended = "Run43 should update models with combined264 and continue local calibration around high-density top5/top10 regions rather than declaring the focused search exhausted."
    else:
        recommended = "Run43 should diagnose over-exploitation and return to a broader native combined264 search."

    payload = {
        "validation": validation,
        "run41_leaderboard": run41_leaderboard,
        "combined264_leaderboard": combined264_leaderboard,
        "combined264_summary": combined264_summary,
        "combined264_plus_N32_summary": combined264_plus_summary,
        "run41_vs_combined204": comparison,
        "prior_summary": prior_summary,
        "effectiveness_summary": effectiveness_summary,
        "prediction_summary": prediction_summary,
        "penalty_headline": penalty_headline,
        "recommended_run43": recommended,
    }
    write_report(payload)

    output_files = [
        OUTPUT_DIR / "run42_input_validation_summary.json",
        OUTPUT_DIR / "run41_native_N24_N40_focused_batch60_teacher_dataset_enriched.csv",
        OUTPUT_DIR / "run41_native_N24_N40_focused_batch60_teacher_dataset_enriched.json",
        OUTPUT_DIR / "run41_native_N24_N40_focused_batch60_ranked_within_batch.csv",
        OUTPUT_DIR / "run41_native_N24_N40_focused_batch60_per_N_leaderboard.csv",
        OUTPUT_DIR / "combined264_teacher_dataset.csv",
        OUTPUT_DIR / "combined264_RL_ready_dataset.csv",
        OUTPUT_DIR / "combined264_per_N_leaderboard.csv",
        OUTPUT_DIR / "combined264_summary.json",
        OUTPUT_DIR / "combined264_plus_N32_teacher_dataset.csv",
        OUTPUT_DIR / "combined264_plus_N32_RL_ready_dataset.csv",
        OUTPUT_DIR / "combined264_plus_N32_per_N_leaderboard.csv",
        OUTPUT_DIR / "combined264_plus_N32_summary.json",
        OUTPUT_DIR / "run41_vs_combined204_best_comparison.csv",
        OUTPUT_DIR / "run41_vs_combined204_best_comparison.json",
        OUTPUT_DIR / "run41_vs_prior_key_records.csv",
        OUTPUT_DIR / "run41_vs_prior_key_records_summary.json",
        OUTPUT_DIR / "run41_N24_N40_focused_batch60_effectiveness_audit.csv",
        OUTPUT_DIR / "run41_N24_N40_focused_batch60_effectiveness_summary.json",
        OUTPUT_DIR / "run41_prediction_audit_for_run38_batch60.csv",
        OUTPUT_DIR / "run41_prediction_audit_for_run38_batch60_summary.json",
        OUTPUT_DIR / "run41_u2_gain_vs_penalty_analysis.csv",
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
            str(RUN41_METRICS), str(RUN41_EXTRACTION), str(RUN41_SOLVER), str(RUN41_SUMMARY),
            str(RUN39_HANDOFF), str(RUN38_POOL), str(COMBINED204_TEACHER), str(COMBINED204_READY),
            str(COMBINED204_PLUS_N32_READY), str(N32_DEDUP),
        ],
        "output_files": [str(p) for p in output_files],
        "run41_teacher_rows": 60,
        "combined264_rows": int(len(combined264)),
        "combined264_plus_N32_rows": int(len(combined264_plus)),
        "per_N_combined264_counts": parse_n_counts(combined264),
        "per_N_combined264_plus_N32_counts": parse_n_counts(combined264_plus),
        "new_best_counts": {"run41_vs_combined204_metric_level_records": beat_count},
        "prediction_audit_summary": prediction_summary,
        "report_path": str(REPORT_PATH),
        "claim_boundary_path": str(CLAIM_BOUNDARY_MD),
        "no_solver_run": True,
        "no_odb_opened": True,
        "no_abqjobpilot_run": True,
        "no_cae_inp_generated": True,
        "no_teacher_validation_performed_by_run42": True,
        "no_training": True,
        "no_candidate_generation": True,
        "no_commit_or_push": True,
    }
    write_json(MANIFEST_PATH, manifest)
    print(json.dumps({
        "verdict": validation["verdict"],
        "combined264_rows": len(combined264),
        "combined264_plus_N32_rows": len(combined264_plus),
        "new_best_count_vs_combined204": beat_count,
        "report": str(REPORT_PATH),
        "manifest": str(MANIFEST_PATH),
    }, indent=2))


if __name__ == "__main__":
    main()
