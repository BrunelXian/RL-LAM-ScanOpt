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
RUN_ID = "run_37_N32_informed_native_batch32_teacher_metrics_ingestion_and_combined204_ranking"
RUN_NAME = "N32-informed native batch32 teacher metrics ingestion and combined204 ranking"
SCRIPT_PATH = ROOT / "scripts" / "stage3" / "run_37_ingest_N32_informed_native_batch32_and_build_combined204.py"

RUN36_DIR = ROOT / "outputs" / "stage3_run_36_N32_informed_native_batch32_odb_teacher_validation"
RUN36_METRICS = RUN36_DIR / "run36_N32_informed_native_batch32_teacher_metrics.csv"
RUN36_EXTRACTION = RUN36_DIR / "run36_N32_informed_native_batch32_odb_extraction_summary.csv"
RUN36_SOLVER = RUN36_DIR / "run36_N32_informed_native_batch32_solver_completion_audit.csv"
RUN36_SUMMARY = RUN36_DIR / "run36_N32_informed_native_batch32_odb_teacher_validation_summary.json"
RUN36_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_36_N32_informed_native_batch32_odb_teacher_validation" / "RUN_36_N32_INFORMED_NATIVE_BATCH32_ODB_TEACHER_VALIDATION_REPORT.md"
RUN36_MANIFEST = ROOT / "artifacts" / "manifests" / "stage3_run_36_manifest.json"

RUN34_HANDOFF = ROOT / "outputs" / "stage3_run_34_run33_N32_informed_native_batch32_handoff_package" / "stage3_run34_N32_informed_native_batch32_candidate_orders.csv"
RUN33_POOL = ROOT / "outputs" / "stage3_run_33_combined172_plus_N32_balanced_surrogate_gnn_candidate_generation" / "run33_candidate_pool_scored.csv"
RUN33_OPTION_A = ROOT / "outputs" / "stage3_run_33_combined172_plus_N32_balanced_surrogate_gnn_candidate_generation" / "run33_N32_informed_native_batch32_candidate_orders.csv"
RUN33_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_33_combined172_plus_N32_balanced_surrogate_gnn_candidate_generation" / "RUN_33_COMBINED172_PLUS_N32_BALANCED_SURROGATE_GNN_CANDIDATE_GENERATION_REPORT.md"

COMBINED172_TEACHER = ROOT / "outputs" / "stage3_run_28_shortlist64_teacher_metrics_ingestion_and_combined172_ranking" / "combined172_teacher_dataset.csv"
COMBINED172_READY = ROOT / "outputs" / "stage3_run_28_shortlist64_teacher_metrics_ingestion_and_combined172_ranking" / "combined172_RL_ready_dataset.csv"
COMBINED172_PLUS_N32_READY = ROOT / "outputs" / "stage3_run_32a_stage2_n32_legacy_teacher_label_ingestion_for_stage3" / "combined172_plus_N32_RL_ready_dataset.csv"
N32_DEDUP = ROOT / "outputs" / "stage3_run_32a_stage2_n32_legacy_teacher_label_ingestion_for_stage3" / "n32_legacy_teacher_dataset_dedup_training_332.csv"
RUN32A_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_32a_stage2_n32_legacy_teacher_label_ingestion_for_stage3" / "RUN_32A_STAGE2_N32_LEGACY_TEACHER_LABEL_INGESTION_FOR_STAGE3_REPORT.md"

OUTPUT_DIR = ROOT / "outputs" / "stage3_run_37_N32_informed_native_batch32_teacher_metrics_ingestion_and_combined204_ranking"
REPORT_DIR = ROOT / "docs" / "stage3" / "runs" / "run_37_N32_informed_native_batch32_teacher_metrics_ingestion_and_combined204_ranking"
REPORT_PATH = REPORT_DIR / "RUN_37_N32_INFORMED_NATIVE_BATCH32_TEACHER_METRICS_INGESTION_AND_COMBINED204_RANKING_REPORT.md"
MANIFEST_PATH = ROOT / "artifacts" / "manifests" / "stage3_run_37_manifest.json"
CLAIM_BOUNDARY_MD = OUTPUT_DIR / "run37_claim_boundary.md"
CLAIM_BOUNDARY_JSON = OUTPUT_DIR / "run37_claim_boundary.json"

EXPECTED_RUN36_COUNTS = {12: 4, 16: 4, 24: 12, 40: 12}
EXPECTED_COMBINED172_COUNTS = {12: 32, 16: 32, 24: 54, 40: 54}
EXPECTED_COMBINED204_COUNTS = {12: 36, 16: 36, 24: 66, 40: 66}
EXPECTED_PLUS_N32_COUNTS = {12: 36, 16: 36, 24: 66, 32: 332, 40: 66}

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
    if pd.isna(value):
        return None
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


def is_valid_order(value: Any, n: int) -> bool:
    try:
        order = parse_order(value)
    except Exception:
        return False
    return len(order) == n and sorted(order) == list(range(n))


def stable_order_hash(order: list[int]) -> str:
    payload = ",".join(str(x) for x in order).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def first_existing(row: pd.Series, names: list[str], default: Any = "") -> Any:
    for name in names:
        if name in row.index and not pd.isna(row[name]) and str(row[name]) != "":
            return row[name]
    return default


def as_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def canonicalize_surface_proxy(df: pd.DataFrame) -> pd.DataFrame:
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
    return df


def normalize_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = canonicalize_surface_proxy(df)
    for col in ["u2_range", "peeq_max", "mises_max"]:
        if col in df.columns:
            df[col] = as_float(df[col])
    return df


def rank_score_from_rank(rank: pd.Series, count: int) -> pd.Series:
    denom = max(1, count - 1)
    return 1.0 - ((rank.astype(float) - 1.0) / denom)


def minmax_cost(series: pd.Series) -> pd.Series:
    vals = as_float(series)
    mn = vals.min()
    mx = vals.max()
    if not math.isfinite(float(mn)) or not math.isfinite(float(mx)) or mx == mn:
        return pd.Series([0.0] * len(vals), index=vals.index)
    return (vals - mn) / (mx - mn)


def add_within_n_scores(
    df: pd.DataFrame,
    rank_suffix: str,
    score_prefix: str,
    cost_suffix: str,
    reward_col: str,
    reward_rank_col: str,
) -> pd.DataFrame:
    out = df.copy()
    for metric_key, metric_col in RAW_METRICS.items():
        out[metric_col] = as_float(out[metric_col])
        rank_col = f"{metric_key}_rank_{rank_suffix}"
        score_col = f"{score_prefix}_{metric_key}_score"
        if metric_key == "surfaceT":
            score_col = f"{score_prefix}_surfaceT_score"
        cost_col = f"{metric_key}_cost_minmax_{cost_suffix}"
        if metric_key == "surfaceT":
            cost_col = f"surfaceT_cost_minmax_{cost_suffix}"
        out[rank_col] = math.nan
        out[score_col] = math.nan
        out[cost_col] = math.nan
        for n, idx in out.groupby("n").groups.items():
            ranks = out.loc[idx, metric_col].rank(method="average", ascending=True)
            out.loc[idx, rank_col] = ranks
            out.loc[idx, score_col] = rank_score_from_rank(ranks, len(idx))
            out.loc[idx, cost_col] = minmax_cost(out.loc[idx, metric_col])

    u2_score = f"{score_prefix}_u2_score"
    peeq_score = f"{score_prefix}_peeq_score"
    surface_score = f"{score_prefix}_surfaceT_score"
    mises_score = f"{score_prefix}_mises_score"
    out[reward_col] = (
        0.65 * out[u2_score].astype(float)
        + 0.20 * out[peeq_score].astype(float)
        + 0.10 * out[surface_score].astype(float)
        + 0.05 * out[mises_score].astype(float)
    )
    out[reward_rank_col] = math.nan
    for _, idx in out.groupby("n").groups.items():
        out.loc[idx, reward_rank_col] = out.loc[idx, reward_col].rank(method="average", ascending=False)
    return out


def add_run36_batch_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    labels = {
        "u2": ("rank_U2_run36_within_n", "score_U2_run36_within_n"),
        "peeq": ("rank_PEEQ_run36_within_n", "score_PEEQ_run36_within_n"),
        "surfaceT": ("rank_SurfaceT_run36_within_n", "score_SurfaceT_run36_within_n"),
        "mises": ("rank_Mises_run36_within_n", "score_Mises_run36_within_n"),
    }
    for metric_key, metric_col in RAW_METRICS.items():
        rank_col, score_col = labels[metric_key]
        cost_col = f"cost_{metric_key}_run36_within_n"
        if metric_key == "surfaceT":
            cost_col = "cost_surfaceT_run36_within_n"
        out[rank_col] = math.nan
        out[score_col] = math.nan
        out[cost_col] = math.nan
        for _, idx in out.groupby("n").groups.items():
            ranks = out.loc[idx, metric_col].rank(method="average", ascending=True)
            out.loc[idx, rank_col] = ranks
            out.loc[idx, score_col] = rank_score_from_rank(ranks, len(idx))
            out.loc[idx, cost_col] = minmax_cost(out.loc[idx, metric_col])
    out["reward_run36_u2_primary"] = (
        0.65 * out["score_U2_run36_within_n"]
        + 0.20 * out["score_PEEQ_run36_within_n"]
        + 0.10 * out["score_SurfaceT_run36_within_n"]
        + 0.05 * out["score_Mises_run36_within_n"]
    )
    out["rank_reward_run36_within_n"] = math.nan
    for _, idx in out.groupby("n").groups.items():
        out.loc[idx, "rank_reward_run36_within_n"] = out.loc[idx, "reward_run36_u2_primary"].rank(method="average", ascending=False)
    return out


def pareto_flags(df: pd.DataFrame, metrics: list[str], flag_col: str) -> pd.DataFrame:
    out = df.copy()
    out[flag_col] = False
    for _, idx in out.groupby("n").groups.items():
        block = out.loc[idx, metrics]
        flags: list[bool] = []
        for i, row in block.iterrows():
            dominated = False
            for j, other in block.iterrows():
                if i == j:
                    continue
                if all(other[m] <= row[m] for m in metrics) and any(other[m] < row[m] for m in metrics):
                    dominated = True
                    break
            flags.append(not dominated)
        out.loc[idx, flag_col] = flags
    return out


def top_records(df: pd.DataFrame, metric_col: str, ascending: bool, n_top: int, label: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for n, block in df.groupby("n"):
        for rank, (_, row) in enumerate(block.sort_values(metric_col, ascending=ascending).head(n_top).iterrows(), start=1):
            rows.append(
                {
                    "n": int(n),
                    "leaderboard": label,
                    "rank": rank,
                    "strategy_name": row.get("strategy_name", row.get("handoff_strategy_name", "")),
                    "dataset_source": row.get("dataset_source", ""),
                    "metric_value": row.get(metric_col),
                    "u2_range": row.get("u2_range"),
                    "peeq_max": row.get("peeq_max"),
                    "surface_t_proxy": row.get("surface_t_proxy"),
                    "mises_max": row.get("mises_max"),
                }
            )
    return pd.DataFrame(rows)


def make_leaderboard(df: pd.DataFrame, reward_col: str, top_n: int = 5) -> pd.DataFrame:
    parts = [
        top_records(df, "u2_range", True, top_n, "top_u2"),
        top_records(df, "peeq_max", True, top_n, "top_peeq"),
        top_records(df, "surface_t_proxy", True, top_n, "top_surfaceT"),
        top_records(df, "mises_max", True, top_n, "top_mises"),
        top_records(df, reward_col, False, top_n, "top_reward"),
    ]
    return pd.concat(parts, ignore_index=True)


def best_by_metric(df: pd.DataFrame, metric_col: str, ascending: bool) -> pd.DataFrame:
    rows = []
    for n, block in df.groupby("n"):
        row = block.sort_values(metric_col, ascending=ascending).iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def metric_best_map(df: pd.DataFrame, metric_col: str, ascending: bool) -> dict[int, pd.Series]:
    return {int(row["n"]): row for _, row in best_by_metric(df, metric_col, ascending).iterrows()}


def spearman(x: pd.Series, y: pd.Series) -> float:
    data = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(data) < 3:
        return math.nan
    return float(data["x"].rank().corr(data["y"].rank()))


def correlation(x: pd.Series, y: pd.Series) -> float:
    data = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(data) < 3:
        return math.nan
    return float(data["x"].corr(data["y"]))


def validate_inputs() -> dict[str, Any]:
    paths = [
        RUN36_METRICS,
        RUN36_EXTRACTION,
        RUN36_SOLVER,
        RUN36_SUMMARY,
        RUN34_HANDOFF,
        RUN33_OPTION_A,
        COMBINED172_TEACHER,
        COMBINED172_READY,
        COMBINED172_PLUS_N32_READY,
        N32_DEDUP,
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")

    metrics = normalize_metric_columns(read_csv(RUN36_METRICS))
    handoff = read_csv(RUN34_HANDOFF)
    combined172 = normalize_metric_columns(read_csv(COMBINED172_TEACHER))
    plus_n32 = read_csv(COMBINED172_PLUS_N32_READY)

    metrics["n"] = metrics["n"].astype(int)
    handoff["n"] = handoff["n"].astype(int)
    combined172["n"] = combined172["n"].astype(int)
    plus_n32["n"] = plus_n32["n"].astype(int)

    run36_counts = parse_n_counts(metrics)
    combined172_counts = parse_n_counts(combined172)
    plus_counts = parse_n_counts(plus_n32)
    required_fields_ok = all(col in metrics.columns and metrics[col].notna().all() for col in ["u2_range", "peeq_max", "surface_t_proxy", "mises_max"])
    pass_status_ok = "teacher_validation_status" in metrics.columns and metrics["teacher_validation_status"].astype(str).eq("PASS_TEACHER_FIELDS_EXTRACTED").all()
    final_step_ok = "final_step_name" not in metrics.columns or metrics["final_step_name"].astype(str).eq("step_final_cooling").all()
    fields_ok = True
    if "extracted_field_names" in metrics.columns:
        field_text = metrics["extracted_field_names"].astype(str)
        fields_ok = all(field_text.str.contains(field).all() for field in ["U", "PEEQ", "S", "NT11"])

    valid_orders = []
    source_order_col = "order_json" if "order_json" in handoff.columns else "scan_order"
    for _, row in handoff.iterrows():
        valid_orders.append(is_valid_order(row[source_order_col], int(row["n"])))
    duplicate_orders = handoff.duplicated(["n", "order_hash"]).sum() if "order_hash" in handoff.columns else 0

    matched = metrics.merge(
        handoff[["handoff_strategy_name", "order_hash"]],
        on="handoff_strategy_name",
        how="left",
        suffixes=("", "_handoff"),
    )
    all_matched = matched["order_hash"].notna().all()
    n32_warning_ok = True
    if "metric_semantic_warning" in plus_n32.columns:
        n32_rows = plus_n32[plus_n32["n"] == 32]
        n32_warning_ok = n32_rows["metric_semantic_warning"].astype(str).str.lower().isin(["true", "1", "yes"]).all()

    validations = {
        "run36_file_exists": RUN36_METRICS.exists(),
        "run36_row_count": int(len(metrics)),
        "run36_per_N_counts": run36_counts,
        "run36_has_no_N32_rows": 32 not in run36_counts,
        "run36_required_metrics_present": required_fields_ok,
        "run36_teacher_status_pass": pass_status_ok,
        "run36_final_step_ok": final_step_ok,
        "run36_extracted_fields_ok": fields_ok,
        "run34_order_validation_count": int(sum(valid_orders)),
        "run34_all_orders_valid": all(valid_orders),
        "run34_duplicate_order_count_within_n": int(duplicate_orders),
        "run36_all_rows_matched_to_run34": bool(all_matched),
        "combined172_row_count": int(len(combined172)),
        "combined172_per_N_counts": combined172_counts,
        "combined172_plus_N32_row_count": int(len(plus_n32)),
        "combined172_plus_N32_per_N_counts": plus_counts,
        "combined172_plus_N32_N32_warning_ok": bool(n32_warning_ok),
    }
    verdict_ok = (
        len(metrics) == 32
        and run36_counts == EXPECTED_RUN36_COUNTS
        and 32 not in run36_counts
        and required_fields_ok
        and pass_status_ok
        and final_step_ok
        and fields_ok
        and all(valid_orders)
        and duplicate_orders == 0
        and all_matched
        and len(combined172) == 172
        and combined172_counts == EXPECTED_COMBINED172_COUNTS
        and len(plus_n32) == 504
        and plus_counts == {12: 32, 16: 32, 24: 54, 32: 332, 40: 54}
        and n32_warning_ok
    )
    return {
        "run_id": RUN_ID,
        "timestamp": now_iso(),
        "verdict": "PASS_RUN37_N32_INFORMED_NATIVE_BATCH32_TEACHER_METRICS_32_OF_32_READY" if verdict_ok else "FAIL_RUN37_INPUT_VALIDATION",
        "validations": validations,
        "input_files": [str(path) for path in paths],
    }


def create_enriched_run36() -> pd.DataFrame:
    metrics = normalize_metric_columns(read_csv(RUN36_METRICS))
    handoff = read_csv(RUN34_HANDOFF)
    solver = read_csv(RUN36_SOLVER)
    extraction = read_csv(RUN36_EXTRACTION)
    option_a = read_csv(RUN33_OPTION_A)

    metrics["n"] = metrics["n"].astype(int)
    handoff["n"] = handoff["n"].astype(int)
    option_a["n"] = option_a["n"].astype(int)

    solver_keep = [
        "handoff_strategy_name",
        "completion_status",
        "nonfatal_warning_marker",
        "lck_present",
        "sta_success_marker",
        "sta_fatal_marker",
        "dat_fatal_marker",
        "msg_fatal_marker",
    ]
    solver_keep = [c for c in solver_keep if c in solver.columns]
    solver_small = solver[solver_keep].copy()
    solver_small = solver_small.rename(columns={"completion_status": "solver_audit_status", "nonfatal_warning_marker": "nonfatal_warning_flag"})

    extraction_keep = ["handoff_strategy_name", "odb_extraction_status", "final_step_name", "final_frame_time", "extracted_field_names", "missing_required_fields"]
    extraction_keep = [c for c in extraction_keep if c in extraction.columns]
    extraction_small = extraction[extraction_keep].copy()

    meta = handoff.copy()
    option_keep = [
        "candidate_id",
        "strategy_name",
        "order_hash",
        "surrogate_reward_pred",
        "f01_extra_trees_reward_pred",
        "gnn_reward_pred",
        "graph_pointer_mean_logprob",
        "gnn_surrogate_disagreement",
        "hybrid_score",
        "uncertainty_score",
        "acquisition_score",
        "seed_strategy",
    ]
    option_keep = [c for c in option_keep if c in option_a.columns]
    meta = meta.merge(option_a[option_keep], on="order_hash", how="left", suffixes=("", "_run33"))

    enriched = metrics.merge(meta, on=["handoff_strategy_name", "n"], how="left", suffixes=("", "_handoff"))
    enriched = enriched.merge(solver_small, on="handoff_strategy_name", how="left")
    enriched = enriched.merge(extraction_small, on="handoff_strategy_name", how="left", suffixes=("", "_extraction"))

    enriched["strategy_name"] = enriched["handoff_strategy_name"]
    enriched["dataset_source"] = "run36_N32_informed_native_batch32"
    enriched["batch_name"] = enriched.get("batch_name", "stage3_run34_N32_informed_native_batch32_v01")
    enriched["N32_informed"] = True
    enriched["native_validation_N"] = True
    enriched["includes_N32_case"] = False
    enriched["is_run36_N32_informed_native_batch32"] = True
    enriched["is_shortlist64_run27"] = False
    enriched["is_batch28"] = False
    enriched["is_batch20"] = False
    enriched["is_probe60"] = False
    enriched["metric_semantic_warning"] = False
    enriched["legacy_compatibility_status"] = "NATIVE_STAGE3"

    rename_map = {
        "original_run33_candidate_id": "original_run33_candidate_id",
        "surrogate_prediction": "surrogate_prediction",
        "gnn_reward_prediction": "gnn_reward_prediction",
        "graph_pointer_policy_score": "graph_pointer_policy_score",
        "gnn_vs_surrogate_disagreement": "gnn_vs_surrogate_disagreement",
        "novelty_distance_to_combined172_plus_N32": "novelty_distance_to_combined172_plus_N32",
    }
    for col in rename_map:
        if col not in enriched.columns:
            enriched[col] = ""

    enriched["surrogate_prediction"] = pd.to_numeric(enriched["surrogate_prediction"], errors="coerce").fillna(pd.to_numeric(enriched.get("surrogate_reward_pred", pd.Series(index=enriched.index)), errors="coerce"))
    enriched["gnn_reward_prediction"] = pd.to_numeric(enriched["gnn_reward_prediction"], errors="coerce").fillna(pd.to_numeric(enriched.get("gnn_reward_pred", pd.Series(index=enriched.index)), errors="coerce"))
    enriched["graph_pointer_policy_score"] = pd.to_numeric(enriched["graph_pointer_policy_score"], errors="coerce").fillna(pd.to_numeric(enriched.get("graph_pointer_mean_logprob", pd.Series(index=enriched.index)), errors="coerce"))
    enriched["gnn_vs_surrogate_disagreement"] = pd.to_numeric(enriched["gnn_vs_surrogate_disagreement"], errors="coerce").fillna(pd.to_numeric(enriched.get("gnn_surrogate_disagreement", pd.Series(index=enriched.index)), errors="coerce"))
    enriched["hybrid_score"] = pd.to_numeric(enriched.get("hybrid_score", pd.Series(index=enriched.index)), errors="coerce")
    enriched["uncertainty_score"] = pd.to_numeric(enriched.get("uncertainty_score", pd.Series(index=enriched.index)), errors="coerce")

    if "order_json" not in enriched.columns:
        enriched["order_json"] = ""
    if "order_hash" not in enriched.columns:
        enriched["order_hash"] = enriched["order_json"].apply(lambda value: stable_order_hash(parse_order(value)) if str(value).strip() else "")
    if "order_compact" not in enriched.columns:
        enriched["order_compact"] = enriched["order_json"].apply(lambda value: "-".join(map(str, parse_order(value))) if str(value).strip() else "")

    if "final_step_name" in enriched.columns:
        enriched["final_step"] = enriched["final_step_name"]
    if "extracted_field_names" in enriched.columns:
        enriched["extracted_fields"] = enriched["extracted_field_names"]

    required_cols = [
        "n",
        "strategy_name",
        "handoff_strategy_name",
        "job_name",
        "dataset_source",
        "batch_name",
        "N32_informed",
        "native_validation_N",
        "includes_N32_case",
        "original_run33_candidate_id",
        "candidate_source",
        "generation_method",
        "selection_bucket",
        "priority_role",
        "surrogate_prediction",
        "gnn_reward_prediction",
        "graph_pointer_policy_score",
        "hybrid_score",
        "uncertainty_score",
        "gnn_vs_surrogate_disagreement",
        "novelty_distance_to_combined172_plus_N32",
        "nearest_existing_teacher_strategy",
        "order_json",
        "order_compact",
        "order_hash",
        "u2_range",
        "peeq_max",
        "surface_t_proxy",
        "surface_t_proxy_mpa",
        "mises_max",
        "final_step",
        "final_frame_time",
        "extracted_fields",
        "teacher_validation_status",
        "solver_audit_status",
        "completion_status",
        "odb_extraction_status",
        "nonfatal_warning_flag",
        "metric_semantic_warning",
        "legacy_compatibility_status",
        "notes",
    ]
    for col in required_cols:
        if col not in enriched.columns:
            enriched[col] = ""
    return enriched[required_cols].copy()


def align_for_concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    columns: list[str] = []
    for df in frames:
        for col in df.columns:
            if col not in columns:
                columns.append(col)
    return pd.concat([df.reindex(columns=columns) for df in frames], ignore_index=True)


def build_combined204(combined172: pd.DataFrame, run36: pd.DataFrame) -> pd.DataFrame:
    combined172 = normalize_metric_columns(combined172.copy())
    run36 = normalize_metric_columns(run36.copy())
    combined172["n"] = combined172["n"].astype(int)
    run36["n"] = run36["n"].astype(int)
    combined = align_for_concat([combined172, run36])
    combined["n"] = combined["n"].astype(int)
    combined = normalize_metric_columns(combined)
    combined["metric_semantic_warning"] = False
    combined["legacy_compatibility_status"] = "NATIVE_STAGE3"
    combined["includes_N32_case"] = combined.get("includes_N32_case", False)
    combined = add_within_n_scores(
        combined,
        rank_suffix="combined204_within_n",
        score_prefix="target",
        cost_suffix="combined204_within_n",
        reward_col="target_reward_combined204_u2_primary",
        reward_rank_col="reward_rank_combined204_within_n",
    )
    combined = combined.rename(
        columns={
            "target_u2_score": "target_u2_score_combined204_rank",
            "target_peeq_score": "target_peeq_score_combined204_rank",
            "target_surfaceT_score": "target_surfaceT_score_combined204_rank",
            "target_mises_score": "target_mises_score_combined204_rank",
        }
    )
    combined = pareto_flags(combined, ["u2_range", "peeq_max"], "combined204_pareto_flag_u2_peeq")
    combined = pareto_flags(combined, ["u2_range", "peeq_max", "surface_t_proxy", "mises_max"], "combined204_pareto_flag_u2_peeq_surfaceT_mises")
    for key, metric in RAW_METRICS.items():
        col = f"is_new_best_{key}_within_n"
        if key == "surfaceT":
            col = "is_new_best_surfaceT_within_n"
        combined[col] = False
        for _, idx in combined.groupby("n").groups.items():
            min_val = combined.loc[idx, metric].min()
            combined.loc[idx, col] = combined.loc[idx, metric].eq(min_val)
    combined["is_new_best_combined_reward_within_n"] = False
    for _, idx in combined.groupby("n").groups.items():
        max_val = combined.loc[idx, "target_reward_combined204_u2_primary"].max()
        combined.loc[idx, "is_new_best_combined_reward_within_n"] = combined.loc[idx, "target_reward_combined204_u2_primary"].eq(max_val)
    return combined


def build_combined204_plus_n32(combined204: pd.DataFrame, n32: pd.DataFrame) -> pd.DataFrame:
    native = combined204.copy()
    native["metric_semantic_warning"] = False
    native["legacy_compatibility_status"] = "NATIVE_STAGE3"
    n32 = normalize_metric_columns(n32.copy())
    n32["n"] = n32["n"].astype(int)
    n32["dataset_source"] = n32.get("dataset_source", "stage2_n32_gnn_rl_legacy")
    n32["includes_N32_case"] = True
    n32["metric_semantic_warning"] = True
    n32["legacy_compatibility_status"] = "LEGACY_COMPATIBLE_WITH_WARNINGS"
    plus = align_for_concat([native, n32])
    plus["n"] = plus["n"].astype(int)
    plus = normalize_metric_columns(plus)
    plus["metric_semantic_warning"] = plus["metric_semantic_warning"].astype(str).str.lower().isin(["true", "1", "yes"])
    plus["legacy_compatibility_status"] = plus["legacy_compatibility_status"].replace("", pd.NA).fillna("NATIVE_STAGE3")
    plus = add_within_n_scores(
        plus,
        rank_suffix="combined204_plus_N32_within_n",
        score_prefix="target",
        cost_suffix="combined204_plus_N32_within_n",
        reward_col="target_reward_combined204_plus_N32_mapped_u2_primary",
        reward_rank_col="reward_rank_combined204_plus_N32_within_n",
    )
    plus = plus.rename(
        columns={
            "target_u2_score": "target_u2_score_combined204_plus_N32_rank",
            "target_peeq_score": "target_peeq_score_combined204_plus_N32_rank",
            "target_surfaceT_score": "target_surfaceT_score_combined204_plus_N32_rank",
            "target_mises_score": "target_mises_score_combined204_plus_N32_rank",
        }
    )
    plus["target_reward_combined204_plus_N32_strict_u2_surfaceT"] = (
        0.80 * plus["target_u2_score_combined204_plus_N32_rank"]
        + 0.20 * plus["target_surfaceT_score_combined204_plus_N32_rank"]
    )
    return plus


def compare_run36_to_combined172(combined172: pd.DataFrame, run36_ranked: pd.DataFrame, combined204: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_specs = [
        ("U2", "u2_range", True),
        ("PEEQ", "peeq_max", True),
        ("SurfaceT", "surface_t_proxy", True),
        ("Mises", "mises_max", True),
        ("combined_reward", "target_reward_combined204_u2_primary", False),
    ]
    c172 = normalize_metric_columns(combined172.copy())
    if "target_reward_combined172_u2_primary" not in c172.columns:
        c172 = build_combined204(c172, pd.DataFrame(columns=c172.columns)).rename(columns={"target_reward_combined204_u2_primary": "target_reward_combined172_u2_primary"})

    for n in EXPECTED_RUN36_COUNTS:
        c172_block = c172[c172["n"].astype(int) == n]
        run36_block = run36_ranked[run36_ranked["n"].astype(int) == n]
        c204_block = combined204[combined204["n"].astype(int) == n]
        for metric_label, metric_col, ascending in metric_specs:
            c172_col = "target_reward_combined172_u2_primary" if metric_label == "combined_reward" else metric_col
            run36_col = "reward_run36_u2_primary" if metric_label == "combined_reward" else metric_col
            c204_col = "target_reward_combined204_u2_primary" if metric_label == "combined_reward" else metric_col
            c172_best = c172_block.sort_values(c172_col, ascending=ascending).iloc[0]
            run36_best = run36_block.sort_values(run36_col, ascending=ascending).iloc[0]
            c204_best = c204_block.sort_values(c204_col, ascending=ascending).iloc[0]
            c172_val = float(c172_best[c172_col])
            run36_val = float(run36_best[run36_col])
            beats = run36_val < c172_val if ascending else run36_val > c172_val
            abs_improvement = (c172_val - run36_val) if ascending else (run36_val - c172_val)
            rel_improvement = abs_improvement / abs(c172_val) if c172_val not in [0.0, -0.0] else math.nan
            rows.append(
                {
                    "n": n,
                    "metric": metric_label,
                    "combined172_best_strategy": c172_best.get("strategy_name", c172_best.get("handoff_strategy_name", "")),
                    "combined172_best_source": c172_best.get("dataset_source", ""),
                    "combined172_best_value": c172_val,
                    "run36_best_strategy": run36_best.get("strategy_name", run36_best.get("handoff_strategy_name", "")),
                    "run36_best_source": run36_best.get("candidate_source", ""),
                    "run36_best_bucket": run36_best.get("selection_bucket", ""),
                    "run36_best_value": run36_val,
                    "run36_beats_combined172_best": bool(beats),
                    "absolute_improvement": abs_improvement,
                    "relative_improvement_fraction": rel_improvement,
                    "combined204_best_strategy": c204_best.get("strategy_name", c204_best.get("handoff_strategy_name", "")),
                    "combined204_best_source": c204_best.get("dataset_source", ""),
                    "combined204_best_value": float(c204_best[c204_col]),
                    "combined204_new_best_is_run36": c204_best.get("dataset_source", "") == "run36_N32_informed_native_batch32",
                }
            )
    return pd.DataFrame(rows)


def effectiveness_audit(run36: pd.DataFrame, combined204: pd.DataFrame, comparison: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    run36_keys = set(run36["order_hash"].astype(str))
    c204_run36 = combined204[combined204["order_hash"].astype(str).isin(run36_keys)].copy()
    rows: list[dict[str, Any]] = []
    group_cols = ["n", "candidate_source", "generation_method", "selection_bucket", "priority_role"]
    for col in group_cols:
        if col not in c204_run36.columns:
            c204_run36[col] = ""
    for group_col in group_cols:
        for value, block in c204_run36.groupby(group_col, dropna=False):
            rows.append(
                {
                    "group_type": group_col,
                    "group_value": value,
                    "count": int(len(block)),
                    "median_u2_rank_combined204": float(block["u2_rank_combined204_within_n"].median()),
                    "best_u2_rank_combined204": float(block["u2_rank_combined204_within_n"].min()),
                    "median_reward_rank_combined204": float(block["reward_rank_combined204_within_n"].median()),
                    "best_reward_rank_combined204": float(block["reward_rank_combined204_within_n"].min()),
                    "top5_u2_count": int((block["u2_rank_combined204_within_n"] <= 5).sum()),
                    "top10_u2_count": int((block["u2_rank_combined204_within_n"] <= 10).sum()),
                    "top5_reward_count": int((block["reward_rank_combined204_within_n"] <= 5).sum()),
                    "top10_reward_count": int((block["reward_rank_combined204_within_n"] <= 10).sum()),
                    "new_best_u2_count": int((block["u2_rank_combined204_within_n"] == 1).sum()),
                    "new_best_reward_count": int((block["reward_rank_combined204_within_n"] == 1).sum()),
                }
            )

    run36_new_best_records = comparison[comparison["run36_beats_combined172_best"] == True]  # noqa: E712
    summary = {
        "run36_rows": int(len(run36)),
        "run36_top5_u2_total": int((c204_run36["u2_rank_combined204_within_n"] <= 5).sum()),
        "run36_top10_u2_total": int((c204_run36["u2_rank_combined204_within_n"] <= 10).sum()),
        "run36_top5_reward_total": int((c204_run36["reward_rank_combined204_within_n"] <= 5).sum()),
        "run36_top10_reward_total": int((c204_run36["reward_rank_combined204_within_n"] <= 10).sum()),
        "run36_new_metric_best_records_vs_combined172": int(len(run36_new_best_records)),
        "new_best_records": run36_new_best_records[["n", "metric", "run36_best_strategy", "absolute_improvement", "relative_improvement_fraction"]].to_dict(orient="records"),
        "n24_top10_reward_count": int(((c204_run36["n"] == 24) & (c204_run36["reward_rank_combined204_within_n"] <= 10)).sum()),
        "n40_top10_reward_count": int(((c204_run36["n"] == 40) & (c204_run36["reward_rank_combined204_within_n"] <= 10)).sum()),
        "interpretation": "Run36 is evaluated as N32-informed native-N candidate validation; it contains no N32 cases.",
    }
    return pd.DataFrame(rows), summary


def prediction_audit(run36_ranked: pd.DataFrame, combined204: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    run36_keys = set(run36_ranked["order_hash"].astype(str))
    realized = combined204[combined204["order_hash"].astype(str).isin(run36_keys)].copy()
    pred_cols = ["hybrid_score", "surrogate_prediction", "gnn_reward_prediction", "graph_pointer_policy_score"]
    rows: list[dict[str, Any]] = []
    for col in pred_cols:
        if col not in realized.columns:
            realized[col] = math.nan
        realized[col] = pd.to_numeric(realized[col], errors="coerce")

    primary_pred = "hybrid_score" if realized["hybrid_score"].notna().sum() >= 3 else "surrogate_prediction"
    realized["prediction_error_combined204_reward"] = realized["target_reward_combined204_u2_primary"] - realized[primary_pred]
    realized["absolute_error_combined204_reward"] = realized["prediction_error_combined204_reward"].abs()

    for n, block in realized.groupby("n"):
        true_top5 = set(block.sort_values("target_reward_combined204_u2_primary", ascending=False).head(5)["order_hash"].astype(str))
        pred_top5 = set(block.sort_values(primary_pred, ascending=False).head(5)["order_hash"].astype(str))
        true_top1 = set(block.sort_values("target_reward_combined204_u2_primary", ascending=False).head(1)["order_hash"].astype(str))
        pred_top1 = set(block.sort_values(primary_pred, ascending=False).head(1)["order_hash"].astype(str))
        rows.append(
            {
                "scope": "per_N",
                "n": int(n),
                "prediction_column": primary_pred,
                "count": int(len(block)),
                "spearman_pred_vs_combined204_reward": spearman(block[primary_pred], block["target_reward_combined204_u2_primary"]),
                "top1_hit": int(bool(true_top1 & pred_top1)),
                "top5_overlap": int(len(true_top5 & pred_top5)),
                "mean_absolute_error": float(block["absolute_error_combined204_reward"].mean()),
            }
        )
    rows.append(
        {
            "scope": "overall",
            "n": "all",
            "prediction_column": primary_pred,
            "count": int(len(realized)),
            "spearman_pred_vs_combined204_reward": spearman(realized[primary_pred], realized["target_reward_combined204_u2_primary"]),
            "top1_hit": "",
            "top5_overlap": "",
            "mean_absolute_error": float(realized["absolute_error_combined204_reward"].mean()),
        }
    )

    for group_col in ["candidate_source", "selection_bucket"]:
        if group_col not in realized.columns:
            continue
        for value, block in realized.groupby(group_col, dropna=False):
            rows.append(
                {
                    "scope": group_col,
                    "n": value,
                    "prediction_column": primary_pred,
                    "count": int(len(block)),
                    "spearman_pred_vs_combined204_reward": spearman(block[primary_pred], block["target_reward_combined204_u2_primary"]),
                    "top1_hit": "",
                    "top5_overlap": "",
                    "mean_absolute_error": float(block["absolute_error_combined204_reward"].mean()),
                }
            )

    summary = {
        "primary_prediction_column": primary_pred,
        "overall_spearman_pred_vs_combined204_reward": spearman(realized[primary_pred], realized["target_reward_combined204_u2_primary"]),
        "per_N_spearman": {
            str(int(n)): spearman(block[primary_pred], block["target_reward_combined204_u2_primary"])
            for n, block in realized.groupby("n")
        },
        "uncertainty_vs_abs_error_correlation": correlation(realized.get("uncertainty_score", pd.Series(dtype=float)), realized["absolute_error_combined204_reward"]),
        "disagreement_vs_abs_error_correlation": correlation(realized.get("gnn_vs_surrogate_disagreement", pd.Series(dtype=float)), realized["absolute_error_combined204_reward"]),
        "novelty_vs_combined204_reward_correlation": correlation(realized.get("novelty_distance_to_combined172_plus_N32", pd.Series(dtype=float)), realized["target_reward_combined204_u2_primary"]),
    }
    return pd.DataFrame(rows), summary


def write_claim_boundary() -> None:
    safe_claims = [
        "Run37 ingests 32/32 teacher-validated Run36 N32-informed native batch32 cases.",
        "Run37 builds native combined204 with N12=36, N16=36, N24=66, N40=66.",
        "Run37 builds combined204_plus_N32 with N12=36, N16=36, N24=66, N32=332, N40=66.",
        "Run37 evaluates whether the N32-informed native batch32 improved native Stage 3 teacher metrics.",
        "Run36 is teacher validation of N32-informed native candidates, not N32 cases.",
    ]
    unsafe_claims = [
        "Do not claim N32 itself was newly teacher-validated in Run36.",
        "Do not claim N32 caused improvement unless supported by Run37 comparison.",
        "Do not claim GNN-RL superiority.",
        "Do not claim online RL.",
        "Do not claim arbitrary-N generalization.",
        "Do not claim physical optimum.",
        "Do not claim solver/ODB extraction happened in Run37.",
    ]
    CLAIM_BOUNDARY_MD.write_text(
        "# Run37 Claim Boundary\n\n"
        "## Safe claims\n\n"
        + "\n".join(f"- {claim}" for claim in safe_claims)
        + "\n\n## Unsafe claims\n\n"
        + "\n".join(f"- {claim}" for claim in unsafe_claims)
        + "\n",
        encoding="utf-8",
    )
    write_json(
        CLAIM_BOUNDARY_JSON,
        {
            "verdict": "RUN37_INGESTION_AND_RANKING_ONLY_NO_SOLVER_NO_TRAINING_NO_CANDIDATE_GENERATION",
            "safe_claims": safe_claims,
            "unsafe_claims": unsafe_claims,
        },
    )


def best_summary(df: pd.DataFrame, reward_col: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for n, block in df.groupby("n"):
        out[str(int(n))] = {
            "best_u2": str(block.sort_values("u2_range", ascending=True).iloc[0].get("strategy_name", "")),
            "best_peeq": str(block.sort_values("peeq_max", ascending=True).iloc[0].get("strategy_name", "")),
            "best_surfaceT": str(block.sort_values("surface_t_proxy", ascending=True).iloc[0].get("strategy_name", "")),
            "best_mises": str(block.sort_values("mises_max", ascending=True).iloc[0].get("strategy_name", "")),
            "best_reward": str(block.sort_values(reward_col, ascending=False).iloc[0].get("strategy_name", "")),
        }
    return out


def write_report(
    validation: dict[str, Any],
    run36_ranked: pd.DataFrame,
    combined204: pd.DataFrame,
    combined204_plus_n32: pd.DataFrame,
    comparison: pd.DataFrame,
    effectiveness_summary: dict[str, Any],
    prediction_summary: dict[str, Any],
    output_files: list[Path],
) -> None:
    run36_best = best_summary(run36_ranked, "reward_run36_u2_primary")
    c204_best = best_summary(combined204, "target_reward_combined204_u2_primary")
    beat_rows = comparison[comparison["run36_beats_combined172_best"] == True]  # noqa: E712
    report = [
        "# Stage 3 Run 37 - N32-Informed Native Batch32 Teacher Metrics Ingestion and Combined204 Ranking",
        "",
        "## 1. Purpose",
        "Run37 ingests the 32 teacher metrics from Run36, ranks the N32-informed native batch32, builds native combined204 and combined204_plus_N32 datasets, and audits whether the N32-informed candidate route improved native Stage 3 performance.",
        "",
        "## 2. Inputs",
        f"- Run36 teacher metrics: `{RUN36_METRICS}`",
        f"- Run34 handoff metadata: `{RUN34_HANDOFF}`",
        f"- Native combined172 teacher dataset: `{COMBINED172_TEACHER}`",
        f"- N32 dedup training dataset: `{N32_DEDUP}`",
        "",
        "## 3. Run36 Teacher-Validation Status",
        "Run36 completed 32/32 teacher extraction with N12=4, N16=4, N24=12, N40=12. No N32 cases are present in Run36.",
        "",
        "## 4. Input Validation",
        f"Verdict: `{validation['verdict']}`.",
        "",
        "## 5. Run36 Enriched Teacher Dataset",
        f"Run36 rows were merged with Run34 handoff metadata and written to `{OUTPUT_DIR / 'run36_N32_informed_native_batch32_teacher_dataset_enriched.csv'}`.",
        "",
        "## 6. Run36 Within-Batch Ranking",
        "Within each native N, Run36 rankings were recomputed for U2, PEEQ, SurfaceT, Mises, and the U2-primary reward.",
        "",
        "## 7. Native Combined204 Construction",
        f"Native combined204 contains {len(combined204)} rows with counts {parse_n_counts(combined204)} and no N32 rows.",
        "",
        "## 8. combined204_plus_N32 Construction",
        f"combined204_plus_N32 contains {len(combined204_plus_n32)} rows with counts {parse_n_counts(combined204_plus_n32)}. N32 remains legacy-compatible and carries metric semantic warnings.",
        "",
        "## 9. Run36 vs Combined172 Best Comparison",
        f"Run36 produced {len(beat_rows)} metric-level records beating the previous combined172 best.",
    ]
    if len(beat_rows):
        for _, row in beat_rows.iterrows():
            report.append(f"- N{int(row['n'])} {row['metric']}: `{row['run36_best_strategy']}` improved by {row['absolute_improvement']:.6g}.")
    else:
        report.append("- No Run36 candidate beat a combined172 best metric/reward record.")

    report.extend(
        [
            "",
            "## 10. N32-Informed Candidate Effectiveness Audit",
            f"Run36 candidates entering combined204 top5 U2: {effectiveness_summary['run36_top5_u2_total']}; top10 U2: {effectiveness_summary['run36_top10_u2_total']}; top5 reward: {effectiveness_summary['run36_top5_reward_total']}; top10 reward: {effectiveness_summary['run36_top10_reward_total']}.",
            "",
            "## 11. Prediction Audit for Run33 Option A",
            f"Primary prediction column: `{prediction_summary['primary_prediction_column']}`. Overall Spearman vs combined204 reward: {prediction_summary['overall_spearman_pred_vs_combined204_reward']:.4f}.",
            "",
            "## 12. N24/N40 Focus Analysis",
            f"N24 top10 reward entries from Run36: {effectiveness_summary['n24_top10_reward_count']}. N40 top10 reward entries from Run36: {effectiveness_summary['n40_top10_reward_count']}.",
            "",
            "## 13. Metric Semantic Boundary for N32",
            "N32 rows in combined204_plus_N32 come from the legacy-compatible Stage 2 ingestion. Run36 contains no N32 rows, and Run37 performs no new N32 teacher validation.",
            "",
            "## 14. Claim Boundary",
            f"Claim boundary files: `{CLAIM_BOUNDARY_MD}` and `{CLAIM_BOUNDARY_JSON}`.",
            "",
            "## 15. Output Files",
        ]
    )
    report.extend([f"- `{path}`" for path in output_files])
    report.extend(
        [
            "",
            "## 16. Recommended Run38",
        ]
    )
    if len(beat_rows) or effectiveness_summary["run36_top10_reward_total"] > 0:
        report.append("Update models using combined204 and combined204_plus_N32, then generate the next candidate batch with explicit native-only versus N32-augmented diagnostics.")
    else:
        report.append("Diagnose the N32-informed selection route against the native-only model before generating another teacher-validation batch.")
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    validation = validate_inputs()
    write_json(OUTPUT_DIR / "run37_input_validation_summary.json", validation)
    if validation["verdict"] != "PASS_RUN37_N32_INFORMED_NATIVE_BATCH32_TEACHER_METRICS_32_OF_32_READY":
        raise RuntimeError(f"Input validation failed: {validation['verdict']}")

    run36_enriched = create_enriched_run36()
    write_csv(OUTPUT_DIR / "run36_N32_informed_native_batch32_teacher_dataset_enriched.csv", run36_enriched)
    write_table_json(OUTPUT_DIR / "run36_N32_informed_native_batch32_teacher_dataset_enriched.json", run36_enriched)

    run36_ranked = add_run36_batch_scores(run36_enriched)
    write_csv(OUTPUT_DIR / "run36_N32_informed_native_batch32_ranked_within_batch.csv", run36_ranked)
    run36_leaderboard = make_leaderboard(run36_ranked, "reward_run36_u2_primary", top_n=5)
    write_csv(OUTPUT_DIR / "run36_N32_informed_native_batch32_per_N_leaderboard.csv", run36_leaderboard)

    combined172 = read_csv(COMBINED172_TEACHER)
    combined204 = build_combined204(combined172, run36_ranked)
    write_csv(OUTPUT_DIR / "combined204_teacher_dataset.csv", combined204)
    write_csv(OUTPUT_DIR / "combined204_RL_ready_dataset.csv", combined204)
    c204_leaderboard = make_leaderboard(combined204, "target_reward_combined204_u2_primary", top_n=5)
    write_csv(OUTPUT_DIR / "combined204_per_N_leaderboard.csv", c204_leaderboard)
    write_json(
        OUTPUT_DIR / "combined204_summary.json",
        {
            "rows": int(len(combined204)),
            "per_N_counts": parse_n_counts(combined204),
            "contains_N32": bool((combined204["n"].astype(int) == 32).any()),
            "best_by_N": best_summary(combined204, "target_reward_combined204_u2_primary"),
        },
    )

    n32 = read_csv(N32_DEDUP)
    combined204_plus_n32 = build_combined204_plus_n32(combined204, n32)
    write_csv(OUTPUT_DIR / "combined204_plus_N32_teacher_dataset.csv", combined204_plus_n32)
    write_csv(OUTPUT_DIR / "combined204_plus_N32_RL_ready_dataset.csv", combined204_plus_n32)
    plus_leaderboard = make_leaderboard(combined204_plus_n32, "target_reward_combined204_plus_N32_mapped_u2_primary", top_n=5)
    write_csv(OUTPUT_DIR / "combined204_plus_N32_per_N_leaderboard.csv", plus_leaderboard)
    write_json(
        OUTPUT_DIR / "combined204_plus_N32_summary.json",
        {
            "rows": int(len(combined204_plus_n32)),
            "per_N_counts": parse_n_counts(combined204_plus_n32),
            "N32_metric_semantic_warning_active": True,
            "N32_rows": int((combined204_plus_n32["n"].astype(int) == 32).sum()),
            "best_by_N": best_summary(combined204_plus_n32, "target_reward_combined204_plus_N32_mapped_u2_primary"),
        },
    )

    comparison = compare_run36_to_combined172(combined172, run36_ranked, combined204)
    write_csv(OUTPUT_DIR / "run36_vs_combined172_best_comparison.csv", comparison)
    write_table_json(OUTPUT_DIR / "run36_vs_combined172_best_comparison.json", comparison)

    effectiveness, effectiveness_summary = effectiveness_audit(run36_ranked, combined204, comparison)
    write_csv(OUTPUT_DIR / "run36_N32_informed_candidate_effectiveness_audit.csv", effectiveness)
    write_json(OUTPUT_DIR / "run36_N32_informed_candidate_effectiveness_summary.json", effectiveness_summary)

    prediction_audit_df, prediction_summary = prediction_audit(run36_ranked, combined204)
    write_csv(OUTPUT_DIR / "run36_prediction_audit_for_run33_optionA.csv", prediction_audit_df)
    write_json(OUTPUT_DIR / "run36_prediction_audit_for_run33_optionA_summary.json", prediction_summary)

    write_claim_boundary()

    output_files = [
        OUTPUT_DIR / "run37_input_validation_summary.json",
        OUTPUT_DIR / "run36_N32_informed_native_batch32_teacher_dataset_enriched.csv",
        OUTPUT_DIR / "run36_N32_informed_native_batch32_ranked_within_batch.csv",
        OUTPUT_DIR / "run36_N32_informed_native_batch32_per_N_leaderboard.csv",
        OUTPUT_DIR / "combined204_teacher_dataset.csv",
        OUTPUT_DIR / "combined204_RL_ready_dataset.csv",
        OUTPUT_DIR / "combined204_per_N_leaderboard.csv",
        OUTPUT_DIR / "combined204_summary.json",
        OUTPUT_DIR / "combined204_plus_N32_teacher_dataset.csv",
        OUTPUT_DIR / "combined204_plus_N32_RL_ready_dataset.csv",
        OUTPUT_DIR / "combined204_plus_N32_per_N_leaderboard.csv",
        OUTPUT_DIR / "combined204_plus_N32_summary.json",
        OUTPUT_DIR / "run36_vs_combined172_best_comparison.csv",
        OUTPUT_DIR / "run36_N32_informed_candidate_effectiveness_audit.csv",
        OUTPUT_DIR / "run36_N32_informed_candidate_effectiveness_summary.json",
        OUTPUT_DIR / "run36_prediction_audit_for_run33_optionA.csv",
        OUTPUT_DIR / "run36_prediction_audit_for_run33_optionA_summary.json",
        CLAIM_BOUNDARY_MD,
        CLAIM_BOUNDARY_JSON,
        REPORT_PATH,
        MANIFEST_PATH,
    ]
    write_report(validation, run36_ranked, combined204, combined204_plus_n32, comparison, effectiveness_summary, prediction_summary, output_files)

    manifest = {
        "run_id": RUN_ID,
        "run_name": RUN_NAME,
        "timestamp": now_iso(),
        "branch": current_branch(),
        "script_path": str(SCRIPT_PATH),
        "input_files": [
            str(RUN36_METRICS),
            str(RUN36_EXTRACTION),
            str(RUN36_SOLVER),
            str(RUN36_SUMMARY),
            str(RUN34_HANDOFF),
            str(RUN33_POOL),
            str(RUN33_OPTION_A),
            str(COMBINED172_TEACHER),
            str(COMBINED172_READY),
            str(COMBINED172_PLUS_N32_READY),
            str(N32_DEDUP),
            str(RUN32A_REPORT),
            str(RUN33_REPORT),
            str(RUN36_REPORT),
            str(RUN36_MANIFEST),
        ],
        "output_files": [str(path) for path in output_files],
        "run36_teacher_rows": int(len(run36_enriched)),
        "run36_per_N_counts": parse_n_counts(run36_enriched),
        "combined204_rows": int(len(combined204)),
        "combined204_plus_N32_rows": int(len(combined204_plus_n32)),
        "per_N_combined204_counts": parse_n_counts(combined204),
        "per_N_combined204_plus_N32_counts": parse_n_counts(combined204_plus_n32),
        "new_best_counts": {
            "run36_beats_combined172_metric_records": int((comparison["run36_beats_combined172_best"] == True).sum()),  # noqa: E712
        },
        "prediction_audit_summary": prediction_summary,
        "report_path": str(REPORT_PATH),
        "claim_boundary_path": str(CLAIM_BOUNDARY_MD),
        "no_solver_run": True,
        "no_odb_opened": True,
        "no_abqjobpilot_run": True,
        "no_cae_inp_generated": True,
        "no_teacher_validation_performed_by_run37": True,
        "no_training": True,
        "no_candidate_generation": True,
        "no_commit_or_push": True,
    }
    write_json(MANIFEST_PATH, manifest)

    print(json.dumps({"verdict": validation["verdict"], "combined204_counts": parse_n_counts(combined204), "combined204_plus_N32_counts": parse_n_counts(combined204_plus_n32)}, indent=2))


if __name__ == "__main__":
    main()
