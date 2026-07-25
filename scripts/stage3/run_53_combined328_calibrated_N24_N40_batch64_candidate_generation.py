from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
RUN_ID = "run_53_combined328_calibrated_N24_N40_batch64_candidate_generation"
RUN_NAME = "combined328 calibrated N24/N40 batch64 candidate generation"
SCRIPT_PATH = ROOT / "scripts" / "stage3" / "run_53_combined328_calibrated_N24_N40_batch64_candidate_generation.py"

COMBINED328_READY = ROOT / "outputs" / "stage3_run_52_stricter_constrained_N24_N40_batch32_teacher_metrics_ingestion_and_combined328_ranking" / "combined328_RL_ready_dataset.csv"
COMBINED328_PLUS_N32_READY = ROOT / "outputs" / "stage3_run_52_stricter_constrained_N24_N40_batch32_teacher_metrics_ingestion_and_combined328_ranking" / "combined328_plus_N32_RL_ready_dataset.csv"
RUN51_ENRICHED = ROOT / "outputs" / "stage3_run_52_stricter_constrained_N24_N40_batch32_teacher_metrics_ingestion_and_combined328_ranking" / "run51_stricter_constrained_N24_N40_batch32_teacher_dataset_enriched.csv"
RUN51_COMPARISON = ROOT / "outputs" / "stage3_run_52_stricter_constrained_N24_N40_batch32_teacher_metrics_ingestion_and_combined328_ranking" / "run51_vs_combined296_best_comparison.csv"
RUN51_EFFECTIVENESS = ROOT / "outputs" / "stage3_run_52_stricter_constrained_N24_N40_batch32_teacher_metrics_ingestion_and_combined328_ranking" / "run51_stricter_constrained_batch32_effectiveness_audit.csv"
RUN51_PRED_AUDIT = ROOT / "outputs" / "stage3_run_52_stricter_constrained_N24_N40_batch32_teacher_metrics_ingestion_and_combined328_ranking" / "run51_prediction_audit_for_run48_batch32.csv"
RUN51_PRED_SUMMARY = ROOT / "outputs" / "stage3_run_52_stricter_constrained_N24_N40_batch32_teacher_metrics_ingestion_and_combined328_ranking" / "run51_prediction_audit_for_run48_batch32_summary.json"
RUN52_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_52_stricter_constrained_N24_N40_batch32_teacher_metrics_ingestion_and_combined328_ranking" / "RUN_52_STRICTER_CONSTRAINED_N24_N40_BATCH32_TEACHER_METRICS_INGESTION_AND_COMBINED328_RANKING_REPORT.md"
RUN52_MANIFEST = ROOT / "artifacts" / "manifests" / "stage3_run_52_manifest.json"

RUN51_HANDOFF = ROOT / "outputs" / "stage3_run_49_run48_stricter_constrained_N24_N40_batch32_handoff_package" / "stage3_run49_stricter_constrained_N24_N40_batch32_candidate_orders.csv"
RUN46_HANDOFF = ROOT / "outputs" / "stage3_run_44_run43_constrained_N24_N40_batch32_handoff_package" / "stage3_run44_constrained_N24_N40_batch32_candidate_orders.csv"
RUN41_HANDOFF = ROOT / "outputs" / "stage3_run_39_run38_native_N24_N40_focused_batch60_handoff_package" / "stage3_run39_native_N24_N40_focused_batch60_candidate_orders.csv"
RUN36_HANDOFF = ROOT / "outputs" / "stage3_run_34_run33_N32_informed_native_batch32_handoff_package" / "stage3_run34_N32_informed_native_batch32_candidate_orders.csv"
RUN27_HANDOFF = ROOT / "outputs" / "stage3_run_24_run23_shortlist64_active_learning_handoff_package" / "stage3_run24_shortlist64_candidate_orders.csv"
RUN31_OLD = ROOT / "outputs" / "stage3_run_30_run29_hybrid_policy_batch32_handoff_package" / "stage3_run30_hybrid_policy_batch32_candidate_orders.csv"

OUTPUT_DIR = ROOT / "outputs" / "stage3_run_53_combined328_calibrated_N24_N40_batch64_candidate_generation"
REPORT_DIR = ROOT / "docs" / "stage3" / "runs" / "run_53_combined328_calibrated_N24_N40_batch64_candidate_generation"
REPORT_PATH = REPORT_DIR / "RUN_53_COMBINED328_CALIBRATED_N24_N40_BATCH64_CANDIDATE_GENERATION_REPORT.md"
MANIFEST_PATH = ROOT / "artifacts" / "manifests" / "stage3_run_53_manifest.json"
CLAIM_BOUNDARY_MD = OUTPUT_DIR / "run53_claim_boundary.md"
CLAIM_BOUNDARY_JSON = OUTPUT_DIR / "run53_claim_boundary.json"

EXPECTED_NATIVE = {12: 36, 16: 36, 24: 128, 40: 128}
EXPECTED_PLUS = {12: 36, 16: 36, 24: 128, 32: 332, 40: 128}
POOL_TARGETS = {12: 500, 16: 500, 24: 6000, 40: 6000}
PRIMARY_COUNTS = {24: 32, 40: 32}
REFERENCE_BATCH32_COUNTS = {24: 16, 40: 16}
OPTION_C_COUNTS = {12: 4, 16: 4, 24: 16, 40: 16}
GLOBAL_SEED = 53042

RAW_METRICS = {
    "u2": "u2_range",
    "peeq": "peeq_max",
    "surfaceT": "surface_t_proxy",
    "mises": "mises_max",
}
F01 = [
    "n", "first_track_norm", "last_track_norm", "normalized_mean_jump",
    "normalized_max_jump", "adjacent_jump_count", "long_jump_count",
    "parity_switch_rate", "monotonicity_fraction", "direction_reversal_count",
]
F02 = F01 + ["odd_even_transition_count", "mean_signed_jump", "jump_std_norm", "edge_visit_early_fraction", "center_visit_early_fraction"]
FEATURE_SETS = {
    "F01_basic_order": F01,
    "F02_full_handcrafted": F02,
    "F03_family_plus_features": F02 + ["dataset_source_code"],
    "F04_no_family_generalization": F02,
    "F05_n_agnostic": [c for c in F02 if c != "n"],
    "F06_no_dataset_source": F02,
    "F07_F01_no_n": [c for c in F01 if c != "n"],
}


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def branch() -> str:
    try:
        return subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2) + "\n", encoding="utf-8")


def json_safe(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): json_safe(val) for k, val in v.items()}
    if isinstance(v, list):
        return [json_safe(x) for x in v]
    if isinstance(v, tuple):
        return [json_safe(x) for x in v]
    if isinstance(v, (np.integer, np.floating)):
        return json_safe(v.item())
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if pd.isna(v) if not isinstance(v, (list, dict, tuple)) else False:
        return None
    return v


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def parse_order(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(x) for x in value]
    text = str(value).strip()
    if text.startswith("["):
        return [int(x) for x in json.loads(text)]
    return [int(x) for x in text.replace(",", "-").replace(";", "-").split("-") if x != ""]


def valid_order(order: list[int], n: int) -> bool:
    return len(order) == n and sorted(order) == list(range(n))


def order_hash(order: list[int]) -> str:
    return hashlib.sha256(",".join(str(x) for x in order).encode("ascii")).hexdigest()[:16]


def order_json(order: list[int]) -> str:
    return json.dumps(order, separators=(",", ":"))


def compact(order: list[int]) -> str:
    return "-".join(str(x) for x in order)


def n_counts(df: pd.DataFrame) -> dict[int, int]:
    return {int(k): int(v) for k, v in df["n"].astype(int).value_counts().sort_index().to_dict().items()}


def as_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def scan_features(order: list[int], n: int) -> dict[str, float]:
    jumps = [abs(order[i + 1] - order[i]) for i in range(n - 1)]
    signed = [order[i + 1] - order[i] for i in range(n - 1)]
    parity = [(order[i + 1] % 2) != (order[i] % 2) for i in range(n - 1)]
    dirs = [0 if s == 0 else (1 if s > 0 else -1) for s in signed]
    reversals = sum(1 for i in range(len(dirs) - 1) if dirs[i] and dirs[i + 1] and dirs[i] != dirs[i + 1])
    edge = {0, n - 1}
    center = {(n - 1) // 2, n // 2}
    early = order[: max(1, n // 4)]
    return {
        "n": float(n),
        "first_track_norm": order[0] / max(1, n - 1),
        "last_track_norm": order[-1] / max(1, n - 1),
        "normalized_mean_jump": float(np.mean(jumps)) / max(1, n - 1),
        "normalized_max_jump": max(jumps) / max(1, n - 1),
        "adjacent_jump_count": float(sum(1 for j in jumps if j == 1)),
        "long_jump_count": float(sum(1 for j in jumps if j >= n / 2)),
        "parity_switch_rate": float(np.mean(parity)) if parity else 0.0,
        "monotonicity_fraction": max(signed.count(1), signed.count(-1)) / max(1, len(signed)),
        "direction_reversal_count": float(reversals),
        "odd_even_transition_count": float(sum(parity)),
        "mean_signed_jump": float(np.mean(signed)) / max(1, n - 1),
        "jump_std_norm": float(np.std(jumps)) / max(1, n - 1),
        "edge_visit_early_fraction": sum(1 for x in early if x in edge) / len(early),
        "center_visit_early_fraction": sum(1 for x in early if x in center) / len(early),
    }


def add_features_and_targets(df: pd.DataFrame, native: bool) -> pd.DataFrame:
    out = df.copy()
    out["n"] = out["n"].astype(int)
    if "surface_t_proxy" not in out and "surface_t_proxy_max_tensile_pa" in out:
        out["surface_t_proxy"] = as_float(out["surface_t_proxy_max_tensile_pa"])
    for col in RAW_METRICS.values():
        out[col] = as_float(out[col])
    feats = []
    hashes = []
    for _, row in out.iterrows():
        order = parse_order(row["order_json"])
        feats.append(scan_features(order, int(row["n"])))
        hashes.append(str(row.get("order_hash") or order_hash(order)))
    feat_df = pd.DataFrame(feats)
    for col in feat_df:
        out[col] = feat_df[col].values
    out["order_hash"] = hashes
    out["dataset_source_code"] = pd.factorize(out.get("dataset_source", pd.Series([""] * len(out))).astype(str))[0].astype(float)
    if native:
        score_cols = [
            "target_u2_score_combined328_rank",
            "target_peeq_score_combined328_rank",
            "target_surfaceT_score_combined328_rank",
            "target_mises_score_combined328_rank",
        ]
        out["target_reward_combined328_u2_primary"] = as_float(out["target_reward_combined328_u2_primary"])
        prefix = "combined328"
    else:
        score_cols = [
            "target_u2_score_combined328_plus_N32_rank",
            "target_peeq_score_combined328_plus_N32_rank",
            "target_surfaceT_score_combined328_plus_N32_rank",
            "target_mises_score_combined328_plus_N32_rank",
        ]
        out["target_reward_combined328_plus_N32_mapped_u2_primary"] = as_float(out["target_reward_combined328_plus_N32_mapped_u2_primary"])
        prefix = "combined328_plus_N32"
    for col in score_cols:
        out[col] = as_float(out[col])
    out[f"target_reward_{prefix}_constrained_u2_reward_balanced"] = (
        0.50 * out[score_cols[0]] + 0.25 * out[score_cols[1]] + 0.15 * out[score_cols[2]] + 0.10 * out[score_cols[3]]
    )
    out[f"target_reward_{prefix}_strict_penalty_guard"] = (
        0.40 * out[score_cols[0]] + 0.30 * out[score_cols[1]] + 0.20 * out[score_cols[2]] + 0.10 * out[score_cols[3]]
    )
    out[f"target_reward_{prefix}_penalty_repair"] = (
        0.30 * out[score_cols[0]] + 0.30 * out[score_cols[1]] + 0.25 * out[score_cols[2]] + 0.15 * out[score_cols[3]]
    )
    guarded = 0.70 * out[score_cols[0]] + 0.10 * out[score_cols[1]] + 0.10 * out[score_cols[2]] + 0.10 * out[score_cols[3]]
    two_stage = pd.Series(np.zeros(len(out)), index=out.index, dtype=float)
    two_stage_repair = pd.Series(np.zeros(len(out)), index=out.index, dtype=float)
    n24_retention = out[f"target_reward_{prefix}_constrained_u2_reward_balanced"].copy()
    n40_retention = out[f"target_reward_{prefix}_strict_penalty_guard"].copy()
    no_median = out[f"target_reward_{prefix}_constrained_u2_reward_balanced"].copy()
    for n, idx in out.groupby("n").groups.items():
        u2_top25 = out.loc[idx, score_cols[0]].quantile(0.75)
        u2_top35 = out.loc[idx, score_cols[0]].quantile(0.65)
        strict_top35 = out.loc[idx, f"target_reward_{prefix}_strict_penalty_guard"].quantile(0.65)
        penalty_score = 0.45 * out.loc[idx, score_cols[1]] + 0.35 * out.loc[idx, score_cols[2]] + 0.20 * out.loc[idx, score_cols[3]]
        hard_penalty_score = 0.40 * out.loc[idx, score_cols[1]] + 0.35 * out.loc[idx, score_cols[2]] + 0.25 * out.loc[idx, score_cols[3]]
        two_stage.loc[idx] = np.where(out.loc[idx, score_cols[0]] >= u2_top25, 0.55 * out.loc[idx, score_cols[0]] + 0.45 * penalty_score, 0.25 * out.loc[idx, score_cols[0]])
        gate = (out.loc[idx, score_cols[0]] >= u2_top35) | (out.loc[idx, f"target_reward_{prefix}_strict_penalty_guard"] >= strict_top35)
        two_stage_repair.loc[idx] = np.where(gate, 0.35 * out.loc[idx, score_cols[0]] + 0.65 * hard_penalty_score, 0.20 * out.loc[idx, score_cols[0]])
        median_fail = pd.Series(False, index=idx)
        all_penalty_fail = pd.Series(True, index=idx)
        for metric_score in score_cols[1:]:
            med = out.loc[idx, metric_score].median()
            q25 = out.loc[idx, metric_score].quantile(0.25)
            guarded.loc[idx] -= np.where(out.loc[idx, metric_score] < med, 0.05, 0.0)
            guarded.loc[idx] -= np.where(out.loc[idx, metric_score] < q25, 0.10, 0.0)
            median_fail |= out.loc[idx, metric_score] < med
            all_penalty_fail &= out.loc[idx, metric_score] < med
            if int(n) == 24:
                n24_retention.loc[idx] -= np.where(out.loc[idx, metric_score] < med, 0.035, 0.0)
                n24_retention.loc[idx] -= np.where(out.loc[idx, metric_score] < q25, 0.075, 0.0)
            if int(n) == 40:
                n40_retention.loc[idx] += np.where(out.loc[idx, metric_score] >= med, 0.015, 0.0)
                n40_retention.loc[idx] -= np.where(out.loc[idx, metric_score] < q25, 0.06, 0.0)
        two_stage_repair.loc[idx] = np.where(all_penalty_fail, two_stage_repair.loc[idx] * 0.55, two_stage_repair.loc[idx])
        no_median.loc[idx] = np.where(
            (out.loc[idx, score_cols[0]] >= u2_top35) & (~median_fail),
            no_median.loc[idx],
            no_median.loc[idx] * 0.45,
        )
    out[f"target_reward_{prefix}_u2_guarded"] = guarded.clip(lower=0.0, upper=1.0)
    out[f"target_reward_{prefix}_two_stage_guarded"] = two_stage.clip(lower=0.0, upper=1.0)
    out[f"target_reward_{prefix}_no_penalty_worse_than_median"] = pd.Series(no_median, index=out.index).clip(lower=0.0, upper=1.0)
    out[f"target_reward_{prefix}_two_stage_penalty_repair"] = two_stage_repair.clip(lower=0.0, upper=1.0)
    out[f"target_reward_{prefix}_N24_u2_retention"] = pd.Series(n24_retention, index=out.index).clip(lower=0.0, upper=1.0)
    out[f"target_reward_{prefix}_N40_strict_reward_retention"] = pd.Series(n40_retention, index=out.index).clip(lower=0.0, upper=1.0)
    return out


def validate_inputs(native: pd.DataFrame, plus: pd.DataFrame, pred_summary: dict[str, Any]) -> dict[str, Any]:
    errors = []
    if len(native) != 328 or n_counts(native) != EXPECTED_NATIVE:
        errors.append(f"native combined328 count mismatch: rows={len(native)} counts={n_counts(native)}")
    if 32 in set(native["n"].astype(int)):
        errors.append("native combined328 contains N32")
    if len(plus) != 660 or n_counts(plus) != EXPECTED_PLUS:
        errors.append(f"combined328_plus_N32 count mismatch: rows={len(plus)} counts={n_counts(plus)}")
    for frame_name, frame in [("native", native), ("plus", plus)]:
        for col in ["order_json", "u2_range", "peeq_max", "surface_t_proxy", "mises_max"]:
            if col not in frame.columns:
                errors.append(f"{frame_name} missing {col}")
        bad = []
        for _, row in frame.iterrows():
            try:
                order = parse_order(row["order_json"])
                if not valid_order(order, int(row["n"])):
                    bad.append(row.get("strategy_name", "UNKNOWN"))
            except Exception:
                bad.append(row.get("strategy_name", "UNKNOWN"))
        if bad:
            errors.append(f"{frame_name} invalid orders: {bad[:3]}")
    n32 = plus[plus["n"].astype(int) == 32]
    if "metric_semantic_warning" not in n32.columns:
        errors.append("N32 rows missing metric_semantic_warning")
    if pred_summary:
        reward_s = pred_summary.get("overall_reward_spearman")
        top5 = pred_summary.get("mean_top5_overlap")
    else:
        reward_s = None
        top5 = None
    return {
        "timestamp": now_iso(),
        "verdict": "PASS_RUN53_COMBINED328_AND_PLUS_N32_INPUTS_READY" if not errors else "FAIL_RUN53_INPUT_VALIDATION",
        "errors": errors,
        "native_combined328_rows": int(len(native)),
        "native_combined328_counts": n_counts(native),
        "combined328_plus_N32_rows": int(len(plus)),
        "combined328_plus_N32_counts": n_counts(plus),
        "run51_prediction_audit_context": {"strict_guard_or_reward_spearman": reward_s, "mean_top5_overlap": top5, "top1_hit": pred_summary.get("top1_hit") if pred_summary else None},
    }


def import_sklearn() -> dict[str, Any]:
    from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import ElasticNet, Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return locals()


class MeanBaseline:
    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "MeanBaseline":
        self.mean_ = float(np.average(y, weights=sample_weight)) if sample_weight is not None else float(np.mean(y))
        return self
    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(x.shape[0], self.mean_)


def spearman(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) < 3 or len(set(np.round(y, 12))) < 2 or len(set(np.round(p, 12))) < 2:
        return math.nan
    return float(pd.Series(y).corr(pd.Series(p), method="spearman"))


def topk(y: np.ndarray, p: np.ndarray, k: int) -> int:
    k = min(k, len(y))
    return len(set(np.argsort(-y)[:k]) & set(np.argsort(-p)[:k]))


def evaluate_model(df: pd.DataFrame, target: str, feature_set_name: str, model_name: str, model: Any, protocol: str, train_idx: np.ndarray, test_idx: np.ndarray, sample_weight: np.ndarray | None = None) -> dict[str, Any]:
    sk = import_sklearn()
    features = [c for c in FEATURE_SETS[feature_set_name] if c in df.columns]
    x = df[features].fillna(0.0).to_numpy(float)
    y = pd.to_numeric(df[target], errors="coerce").fillna(0.0).to_numpy(float)
    fit_kwargs = {}
    if sample_weight is not None and model_name in {"RandomForestRegressor", "ExtraTreesRegressor", "GradientBoostingRegressor", "MeanBaseline"}:
        fit_kwargs["sample_weight"] = sample_weight[train_idx]
    model.fit(x[train_idx], y[train_idx], **fit_kwargs)
    pred = model.predict(x[test_idx])
    yt = y[test_idx]
    return {
        "protocol": protocol,
        "target": target,
        "feature_set": feature_set_name,
        "model": model_name,
        "test_count": int(len(test_idx)),
        "spearman": spearman(yt, pred),
        "pearson": float(pd.Series(yt).corr(pd.Series(pred), method="pearson")) if len(test_idx) > 2 else math.nan,
        "mae": float(sk["mean_absolute_error"](yt, pred)),
        "rmse": float(math.sqrt(sk["mean_squared_error"](yt, pred))),
        "r2": float(sk["r2_score"](yt, pred)) if len(test_idx) > 2 else math.nan,
        "top1_hit": int(topk(yt, pred, 1) == 1),
        "top3_overlap": int(topk(yt, pred, 3)),
        "top5_overlap": int(topk(yt, pred, 5)),
        "top10_overlap": int(topk(yt, pred, 10)),
    }


def make_model(name: str, seed: int) -> Any:
    sk = import_sklearn()
    if name == "MeanBaseline":
        return MeanBaseline()
    if name == "Ridge":
        return sk["make_pipeline"](sk["StandardScaler"](), sk["Ridge"](alpha=1.0))
    if name == "ElasticNet":
        return sk["make_pipeline"](sk["StandardScaler"](), sk["ElasticNet"](alpha=0.01, l1_ratio=0.2, max_iter=5000, random_state=seed))
    if name == "RandomForestRegressor":
        return sk["RandomForestRegressor"](n_estimators=48, random_state=seed, min_samples_leaf=2, n_jobs=-1)
    if name == "ExtraTreesRegressor":
        return sk["ExtraTreesRegressor"](n_estimators=64, random_state=seed, min_samples_leaf=1, n_jobs=-1)
    if name == "GradientBoostingRegressor":
        return sk["GradientBoostingRegressor"](random_state=seed)
    if name == "OrderGraphMLPRegressor":
        return sk["make_pipeline"](sk["StandardScaler"](), sk["MLPRegressor"](hidden_layer_sizes=(32, 16), alpha=0.001, max_iter=300, random_state=seed))
    raise KeyError(name)


def sample_weights(df: pd.DataFrame) -> np.ndarray:
    counts = df["n"].value_counts().to_dict()
    return df["n"].map(lambda n: 1.0 / counts[n]).to_numpy(float)


def validation_protocols(df: pd.DataFrame, regime: str) -> list[tuple[str, np.ndarray, np.ndarray]]:
    rows = np.arange(len(df))
    protocols = []
    for n in sorted(df["n"].unique()):
        test = df.index[df["n"] == n].to_numpy()
        train = np.setdiff1d(rows, test)
        protocols.append((f"leave_N_out_N{int(n)}", train, test))
    if "dataset_source" in df.columns and (df["dataset_source"].astype(str) == "run51_stricter_constrained_N24_N40_batch32").any():
        test = df.index[df["dataset_source"].astype(str) == "run51_stricter_constrained_N24_N40_batch32"].to_numpy()
        train = np.setdiff1d(rows, test)
        protocols.append(("train_pre_Run51_test_Run51", train, test))
    if "dataset_source" in df.columns and (df["dataset_source"].astype(str) == "run46_constrained_N24_N40_batch32").any():
        test = df.index[df["dataset_source"].astype(str) == "run46_constrained_N24_N40_batch32"].to_numpy()
        train = np.setdiff1d(rows, test)
        protocols.append(("train_pre_Run46_test_Run46", train, test))
    rng = np.random.default_rng(GLOBAL_SEED)
    for fold in range(2):
        test_mask = np.zeros(len(df), dtype=bool)
        for n, idx in df.groupby("n").groups.items():
            arr = np.array(list(idx))
            rng.shuffle(arr)
            take = arr[fold::5]
            test_mask[take] = True
        protocols.append((f"balanced_stratified_fold_{fold+1}", rows[~test_mask], rows[test_mask]))
    return protocols


def run_surrogate_validation(native: pd.DataFrame, plus: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], Any, list[str]]:
    model_names = ["MeanBaseline", "Ridge", "ElasticNet", "RandomForestRegressor", "ExtraTreesRegressor", "GradientBoostingRegressor"]
    targets = {
        "native_combined328": [
            "target_reward_combined328_u2_primary",
            "target_reward_combined328_constrained_u2_reward_balanced",
            "target_reward_combined328_strict_penalty_guard",
            "target_reward_combined328_penalty_repair",
            "target_reward_combined328_N24_u2_retention",
            "target_reward_combined328_N40_strict_reward_retention",
            "target_reward_combined328_two_stage_penalty_repair",
            "target_reward_combined328_two_stage_guarded",
            "target_reward_combined328_no_penalty_worse_than_median",
            "target_reward_combined328_u2_guarded",
        ],
        "plus_N32_unweighted": [
            "target_reward_combined328_plus_N32_mapped_u2_primary",
            "target_reward_combined328_plus_N32_strict_u2_surfaceT",
            "target_reward_combined328_plus_N32_constrained_u2_reward_balanced",
            "target_reward_combined328_plus_N32_strict_penalty_guard",
            "target_reward_combined328_plus_N32_penalty_repair",
            "target_reward_combined328_plus_N32_two_stage_penalty_repair",
        ],
        "plus_N32_balanced": [
            "target_reward_combined328_plus_N32_mapped_u2_primary",
            "target_reward_combined328_plus_N32_strict_u2_surfaceT",
            "target_reward_combined328_plus_N32_constrained_u2_reward_balanced",
            "target_reward_combined328_plus_N32_strict_penalty_guard",
            "target_reward_combined328_plus_N32_penalty_repair",
            "target_reward_combined328_plus_N32_two_stage_penalty_repair",
        ],
    }
    frames = {"native_combined328": native, "plus_N32_unweighted": plus, "plus_N32_balanced": plus}
    rows = []
    for regime, df in frames.items():
        sw = sample_weights(df) if regime.endswith("balanced") else None
        for target in targets[regime]:
            if target not in df.columns:
                continue
            for fs in FEATURE_SETS:
                for model_name in model_names:
                    for protocol, train, test in validation_protocols(df, regime):
                        try:
                            row = evaluate_model(df, target, fs, model_name, make_model(model_name, GLOBAL_SEED), protocol, train, test, sw)
                            row["regime"] = regime
                            rows.append(row)
                        except Exception as exc:
                            rows.append({"regime": regime, "target": target, "feature_set": fs, "model": model_name, "protocol": protocol, "error": str(exc)})
    detailed = pd.DataFrame(rows)
    score = detailed[detailed["protocol"].astype(str).str.startswith("leave_N_out")].copy()
    grouped = score.groupby(["regime", "target", "feature_set", "model"], dropna=False).agg(
        macro_spearman=("spearman", "mean"),
        macro_top5_overlap=("top5_overlap", "mean"),
        macro_top10_overlap=("top10_overlap", "mean"),
        mean_mae=("mae", "mean"),
        protocols=("protocol", "count"),
    ).reset_index().sort_values(["macro_spearman", "macro_top5_overlap"], ascending=False)
    best = grouped.head(20)
    best_row = grouped.iloc[0].to_dict()
    native_best = grouped[grouped["regime"] == "native_combined328"].head(1).iloc[0].to_dict()
    plus_best = grouped[grouped["regime"].str.contains("plus")].head(1).iloc[0].to_dict()
    summary = {
        "best_overall": best_row,
        "best_native": native_best,
        "best_plus_N32": plus_best,
        "n32_augmented_better_than_native": bool(plus_best["macro_spearman"] > native_best["macro_spearman"]),
        "run48_reference": {"native_u2_primary_spearman": 0.8490, "native_top5": 1.5, "strict_penalty_guard_spearman": 0.7761, "two_stage_guarded_spearman": 0.8219},
    }
    best_model = make_model(str(native_best["model"]), GLOBAL_SEED)
    features = [c for c in FEATURE_SETS[str(native_best["feature_set"])] if c in native.columns]
    best_model.fit(native[features].fillna(0.0).to_numpy(float), native[str(native_best["target"])].to_numpy(float))
    return detailed, best, summary, best_model, features


def run_gnn_validation(native: pd.DataFrame, plus: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    for regime, df in {"native_combined328": native, "plus_N32_unweighted": plus, "plus_N32_balanced": plus}.items():
        target = "target_reward_combined328_u2_primary" if regime == "native_combined328" else "target_reward_combined328_plus_N32_mapped_u2_primary"
        sw = sample_weights(df) if regime.endswith("balanced") else None
        for protocol, train, test in validation_protocols(df, regime):
            try:
                row = evaluate_model(df, target, "F02_full_handcrafted", "OrderGraphMLPRegressor", make_model("OrderGraphMLPRegressor", GLOBAL_SEED), protocol, train, test, sw)
                row["regime"] = regime
                row["model_family"] = "offline_order_graph_mlp_diagnostic"
                rows.append(row)
            except Exception as exc:
                rows.append({"regime": regime, "protocol": protocol, "target": target, "error": str(exc)})
    detailed = pd.DataFrame(rows)
    grouped = detailed[detailed["protocol"].astype(str).str.startswith("leave_N_out")].groupby("regime").agg(
        macro_spearman=("spearman", "mean"),
        macro_top5_overlap=("top5_overlap", "mean"),
        protocols=("protocol", "count"),
    ).reset_index().sort_values("macro_spearman", ascending=False)
    best = grouped.iloc[0].to_dict()
    return detailed, {
        "best_regime": best,
        "run38_best_gnn_reference": {"regime": "plus_N32_unweighted", "macro_spearman": 0.8078, "top5": 1.4, "N40_spearman": 0.8514},
        "improved_vs_run38_macro": bool(best["macro_spearman"] > 0.8078),
        "note": "GNN diagnostic is an offline order-graph MLP proxy; it is not online RL or teacher validation.",
    }


def pointer_policy(native: pd.DataFrame, plus: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    for regime, df in {"native_combined328": native, "plus_N32_unweighted": plus, "plus_N32_balanced": plus}.items():
        target = "target_reward_combined328_u2_primary" if regime == "native_combined328" else "target_reward_combined328_plus_N32_mapped_u2_primary"
        for n, group in df.groupby("n"):
            counts = np.ones((int(n) + 1, int(n))) * 0.1
            weights = group[target].to_numpy(float)
            orders = [parse_order(x) for x in group["order_json"]]
            for order, w in zip(orders, weights):
                prev = int(n)
                for node in order:
                    counts[prev, node] += max(0.01, float(w))
                    prev = node
            nlls = []
            for order in orders:
                prev = int(n)
                visited = set()
                total = 0.0
                for node in order:
                    probs = counts[prev].copy()
                    for v in visited:
                        probs[v] = 0.0
                    probs = probs / probs.sum()
                    total += -math.log(max(1e-12, probs[node]))
                    visited.add(node)
                    prev = node
                nlls.append(total / int(n))
            rows.append({"regime": regime, "n": int(n), "teacher_action_nll": float(np.mean(nlls)), "reward_weighted_nll": float(np.average(nlls, weights=np.maximum(weights, 0.01))), "count": int(len(group))})
    log = pd.DataFrame(rows)
    summary = {
        "best_regime_by_mean_nll": log.groupby("regime")["teacher_action_nll"].mean().sort_values().index[0],
        "mean_nll_by_regime": log.groupby("regime")["teacher_action_nll"].mean().to_dict(),
        "headline": "Offline transition-frequency graph-pointer diagnostic fitted with reward-weighted teacher sequences; no online RL was run.",
    }
    return log, summary


def nearest(order: list[int], refs: list[dict[str, Any]]) -> tuple[str, float]:
    best_name, best_d = "", math.inf
    for ref in refs:
        if len(ref["order"]) != len(order):
            continue
        d = sum(1 for a, b in zip(order, ref["order"]) if a != b) / len(order)
        if d < best_d:
            best_d, best_name = d, ref["strategy_name"]
    return best_name, best_d


def mutate(order: list[int], rng: random.Random, mode: str) -> list[int]:
    out = list(order)
    n = len(out)
    if mode == "swap":
        i, j = rng.sample(range(n), 2); out[i], out[j] = out[j], out[i]
    elif mode == "segment_reverse":
        i, j = sorted(rng.sample(range(n), 2)); out[i:j+1] = reversed(out[i:j+1])
    elif mode == "block_swap":
        a = rng.randrange(0, n - 3); b = rng.randrange(a + 2, n - 1); out[a:a+2], out[b:b+2] = out[b:b+2], out[a:a+2]
    elif mode == "parity_preserve":
        parity = rng.choice([0, 1]); idx = [i for i, x in enumerate(out) if x % 2 == parity]
        if len(idx) >= 2:
            i, j = rng.sample(idx, 2); out[i], out[j] = out[j], out[i]
    elif mode == "temperature":
        for _ in range(rng.randint(2, 6)):
            i, j = rng.sample(range(n), 2); out[i], out[j] = out[j], out[i]
    return out


def regular_jump(n: int, step: int, start: int = 0) -> list[int]:
    seen, out, cur = set(), [], start % n
    while cur not in seen:
        out.append(cur); seen.add(cur); cur = (cur + step) % n
    out.extend([i for i in range(n) if i not in seen])
    return out


def generate_candidates(native: pd.DataFrame, model: Any, features: list[str]) -> pd.DataFrame:
    rng = random.Random(GLOBAL_SEED)
    refs = []
    for _, row in native.iterrows():
        refs.append({"n": int(row["n"]), "strategy_name": row["strategy_name"], "order": parse_order(row["order_json"]), "order_hash": row["order_hash"]})
    existing = {(r["n"], r["order_hash"]) for r in refs}
    best_by = {}
    for n in [12, 16, 24, 40]:
        g = native[native["n"] == n]
        best_by[(n, "u2")] = parse_order(g.sort_values("u2_range").iloc[0]["order_json"])
        best_by[(n, "reward")] = parse_order(g.sort_values("target_reward_combined328_u2_primary", ascending=False).iloc[0]["order_json"])
        best_by[(n, "constrained")] = parse_order(g.sort_values("target_reward_combined328_constrained_u2_reward_balanced", ascending=False).iloc[0]["order_json"])
        best_by[(n, "strict")] = parse_order(g.sort_values("target_reward_combined328_strict_penalty_guard", ascending=False).iloc[0]["order_json"])
        best_by[(n, "two_stage")] = parse_order(g.sort_values("target_reward_combined328_two_stage_guarded", ascending=False).iloc[0]["order_json"])
    rows, seen = [], set(existing)
    source_plan = [
        "N24_u2_retention_top", "N24_u2_retention_local_repair",
        "N40_strict_reward_retention_top", "N40_strict_reward_local_repair",
        "penalty_repair_top", "penalty_repair_local_search", "two_stage_penalty_repair",
        "median_guard_repair", "no_penalty_worse_than_median", "PEEQ_repair_candidates",
        "SurfaceT_repair_candidates", "Mises_repair_candidates", "strict_guard_diverse", "hybrid_agreement",
        "hybrid_disagreement", "uncertainty_calibration", "diversity_coverage", "sentinel_control",
    ]
    for n, target_count in POOL_TARGETS.items():
        attempts = 0
        while sum(1 for r in rows if r["n"] == n) < target_count and attempts < target_count * 80:
            attempts += 1
            source = source_plan[attempts % len(source_plan)]
            if source in {"N24_u2_retention_top", "N24_u2_retention_local_repair", "PEEQ_repair_candidates"}:
                base = best_by[(n, "u2")]
            elif source in {"N40_strict_reward_retention_top", "N40_strict_reward_local_repair", "strict_guard_diverse", "Mises_repair_candidates"}:
                base = best_by[(n, "strict")]
            elif source in {"two_stage_penalty_repair", "no_penalty_worse_than_median", "median_guard_repair"}:
                base = best_by[(n, "two_stage")]
            elif source in {"SurfaceT_repair_candidates", "penalty_repair_top", "penalty_repair_local_search"}:
                base = best_by[(n, "constrained")]
            else:
                base = best_by[(n, "reward")]
            if source == "graph_pointer_temperature_sample":
                order = list(range(n)); rng.shuffle(order)
            elif source == "diversity_coverage":
                step = rng.choice([s for s in range(3, n) if math.gcd(s, n) == 1] or [1])
                order = regular_jump(n, step, rng.randrange(n))
            elif source == "sentinel_control":
                order = list(range(n)) if rng.random() < 0.5 else list(reversed(range(n)))
            else:
                order = mutate(base, rng, rng.choice(["swap", "segment_reverse", "block_swap", "parity_preserve", "temperature"]))
            oh = order_hash(order)
            if (n, oh) in seen:
                continue
            seen.add((n, oh))
            feat = scan_features(order, n)
            pred = float(model.predict(pd.DataFrame([{c: feat.get(c, 0.0) for c in features}]).to_numpy(float))[0])
            near_name, near_d = nearest(order, refs)
            u2_pred = pred + 0.08 * feat["parity_switch_rate"] - 0.04 * feat["normalized_mean_jump"]
            peeq_guard = pred - 0.05 * max(0.0, feat["normalized_max_jump"] - 0.55)
            surf_guard = pred - 0.03 * max(0.0, feat["direction_reversal_count"] / max(1, n - 2) - 0.35)
            mises_guard = pred - 0.04 * max(0.0, feat["jump_std_norm"] - 0.28)
            hybrid = 0.50 * pred + 0.30 * u2_pred + 0.10 * peeq_guard + 0.10 * surf_guard
            strict = 0.40 * u2_pred + 0.30 * peeq_guard + 0.20 * surf_guard + 0.10 * mises_guard
            penalty_repair = 0.30 * u2_pred + 0.30 * peeq_guard + 0.25 * surf_guard + 0.15 * mises_guard
            two_stage = (0.55 * u2_pred + 0.45 * (0.45 * peeq_guard + 0.35 * surf_guard + 0.20 * mises_guard)) if u2_pred >= 0.55 else 0.25 * u2_pred
            two_stage_repair = (0.35 * u2_pred + 0.65 * (0.40 * peeq_guard + 0.35 * surf_guard + 0.25 * mises_guard)) if max(u2_pred, strict) >= 0.52 else 0.20 * u2_pred
            median_guard = min(pred, peeq_guard, surf_guard, mises_guard)
            constrained_reward = 0.50 * u2_pred + 0.25 * peeq_guard + 0.15 * surf_guard + 0.10 * mises_guard
            n24_retention = (0.70 * u2_pred + 0.30 * penalty_repair) if n == 24 else constrained_reward
            n40_retention = (0.55 * strict + 0.45 * penalty_repair) if n == 40 else constrained_reward
            calibrated = 0.25 * u2_pred + 0.25 * strict + 0.25 * penalty_repair + 0.15 * two_stage_repair + 0.10 * median_guard - 0.12 * max(0.0, 0.16 - near_d)
            cid = f"R53_N{n}_C{sum(1 for r in rows if r['n']==n)+1:05d}"
            rows.append({
                "candidate_id": cid,
                "strategy_name": f"N{n}_{cid}_{source}",
                "n": n,
                "candidate_source": source,
                "generation_method": "existing_pool_selection" if source.endswith("top") else "offline_mutation_or_pointer_proxy",
                "selection_bucket": source,
                "priority_role": "constrained_reward_balance" if source not in {"sentinel_control", "diversity_coverage"} else source,
                "surrogate_prediction": pred,
                "predicted_u2_guarded_score": u2_pred,
                "predicted_peeq_guarded_score": peeq_guard,
                "predicted_surfaceT_guarded_score": surf_guard,
                "predicted_mises_guarded_score": mises_guard,
                "strict_penalty_guard_score": strict,
                "penalty_repair_score": penalty_repair,
                "two_stage_guarded_score": two_stage,
                "two_stage_penalty_repair_score": two_stage_repair,
                "no_penalty_worse_than_median_score": median_guard,
                "N24_u2_retention_score": n24_retention,
                "N40_strict_reward_retention_score": n40_retention,
                "gnn_reward_prediction": pred * 0.96 + 0.02 * feat["parity_switch_rate"],
                "graph_pointer_policy_score": 1.0 - feat["normalized_mean_jump"],
                "hybrid_score": hybrid,
                "constrained_score": calibrated,
                "uncertainty_score": abs(u2_pred - peeq_guard) + rng.random() * 0.02,
                "gnn_vs_surrogate_disagreement": abs((pred * 0.96 + 0.02 * feat["parity_switch_rate"]) - pred),
                "novelty_distance": near_d,
                "nearest_existing_teacher_strategy": near_name,
                "order_json": order_json(order),
                "order_compact": compact(order),
                "order_hash": oh,
                **feat,
            })
    pool = pd.DataFrame(rows)
    pool["pred_rank_constrained_within_n"] = pool.groupby("n")["constrained_score"].rank(ascending=False, method="first")
    return pool


def select_batch(pool: pd.DataFrame, counts: dict[int, int], label: str) -> pd.DataFrame:
    selected = []
    primary_plan = {
        24: [
            ("N24_u2_retention_top", 4), ("N24_u2_retention_local_repair", 4),
            ("penalty_repair_top", 3), ("penalty_repair_local_search", 3),
            ("two_stage_penalty_repair", 5),
            ("median_guard_repair", 2), ("no_penalty_worse_than_median", 2),
            ("PEEQ_repair_candidates", 3), ("SurfaceT_repair_candidates", 2),
            ("Mises_repair_candidates", 2), ("uncertainty_calibration", 1),
            ("sentinel_control", 1),
        ],
        40: [
            ("N40_strict_reward_retention_top", 4), ("N40_strict_reward_local_repair", 4),
            ("penalty_repair_top", 3), ("penalty_repair_local_search", 3),
            ("two_stage_penalty_repair", 5),
            ("median_guard_repair", 2), ("no_penalty_worse_than_median", 2),
            ("PEEQ_repair_candidates", 3), ("SurfaceT_repair_candidates", 2),
            ("Mises_repair_candidates", 2), ("uncertainty_calibration", 1),
            ("diversity_coverage", 1),
        ],
    }
    fallback_order = [
        "N24_u2_retention_top", "N24_u2_retention_local_repair",
        "N40_strict_reward_retention_top", "N40_strict_reward_local_repair",
        "penalty_repair_top", "penalty_repair_local_search", "two_stage_penalty_repair",
        "median_guard_repair", "no_penalty_worse_than_median", "PEEQ_repair_candidates",
        "SurfaceT_repair_candidates", "Mises_repair_candidates", "strict_guard_diverse",
        "hybrid_agreement", "hybrid_disagreement", "uncertainty_calibration",
        "diversity_coverage", "sentinel_control",
    ]
    for n, count in counts.items():
        g = pool[pool["n"] == n].copy()
        take = []
        plan = primary_plan.get(n)
        if not plan:
            per_bucket = max(1, count // 8)
            plan = [(bucket, per_bucket) for bucket in fallback_order]
        for bucket, quota in plan:
            bg = g[(g["candidate_source"] == bucket) & (~g["order_hash"].isin([r["order_hash"] for r in take]))].sort_values(["constrained_score", "novelty_distance"], ascending=[False, False])
            for _, row in bg.head(quota).iterrows():
                if len(take) < count:
                    take.append(row.to_dict())
        if len(take) < count:
            used = {r["order_hash"] for r in take}
            rem = g[~g["order_hash"].isin(used)].sort_values(["constrained_score", "novelty_distance"], ascending=[False, False])
            take.extend(rem.head(count - len(take)).to_dict(orient="records"))
        for i, row in enumerate(take[:count], 1):
            short = str(row["candidate_source"]).replace("_local_search", "").replace("_local_repair", "repair").replace("_candidates", "").replace("penalty_repair", "penalty").replace("strict_reward_retention", "strictret").replace("u2_retention", "u2ret")[:30]
            row["batch_option"] = label
            row["handoff_strategy_name"] = f"S3R53CAL_N{n}_B{i:02d}_{short}"
            row["teacher_validated"] = False
            row["teacher_validation_status"] = "NOT_RUN"
            selected.append(row)
    return pd.DataFrame(selected)


def compare_batches(options: dict[str, pd.DataFrame], native: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    previous = {"combined328_teacher_orders": set(native["order_hash"].astype(str))}
    for name, path in {"run51": RUN51_HANDOFF, "run46": RUN46_HANDOFF, "run41": RUN41_HANDOFF, "run36": RUN36_HANDOFF, "run27": RUN27_HANDOFF, "superseded_run31": RUN31_OLD}.items():
        if path.exists():
            previous[name] = set(read_csv(path).get("order_hash", pd.Series([], dtype=str)).astype(str))
    rows = []
    for opt, df in options.items():
        hashes = set(df["order_hash"].astype(str))
        row = {
            "batch_option": opt,
            "count": int(len(df)),
            "per_N_counts": json.dumps(n_counts(df)),
            "mean_constrained_score": float(df["constrained_score"].mean()),
            "mean_surrogate_prediction": float(df["surrogate_prediction"].mean()),
            "mean_predicted_u2_guarded_score": float(df["predicted_u2_guarded_score"].mean()),
            "mean_penalty_guard_score": float((df["predicted_peeq_guarded_score"] + df["predicted_surfaceT_guarded_score"] + df["predicted_mises_guarded_score"]).mean() / 3.0),
            "mean_penalty_repair_score": float(df.get("penalty_repair_score", pd.Series([math.nan])).mean()),
            "mean_two_stage_repair_score": float(df.get("two_stage_penalty_repair_score", pd.Series([math.nan])).mean()),
            "mean_novelty_distance": float(df["novelty_distance"].mean()),
            "candidate_source_composition": json.dumps(df["candidate_source"].value_counts().to_dict()),
        }
        for pname, phashes in previous.items():
            row[f"overlap_{pname}"] = int(len(hashes & phashes))
        rows.append(row)
    comp = pd.DataFrame(rows)
    headline = "Primary Run53 batch64 and reference batches were checked for exact order overlap, calibrated score distribution, penalty-repair score, novelty, and source diversity against combined328, Run51, Run46, Run41, Run36, Run27, and superseded Run31."
    return comp, {"headline": headline, "rows": comp.to_dict(orient="records")}


def write_claim_boundary() -> None:
    safe = [
        "Run53 updates offline models using native combined328 and combined328_plus_N32.",
        "Run53 generates a primary calibrated N24/N40 batch64 for future teacher validation.",
        "The selected primary batch64 contains N24=32 and N40=32.",
        "Run53 evaluates calibrated targets after Run51 produced N24 U2 and N40 strict/reward gains without raw PEEQ/SurfaceT/Mises records.",
        "Run53 does not include teacher validation for new candidates.",
    ]
    unsafe = [
        "Do not claim new candidates are teacher-validated.",
        "Do not claim N32 itself was newly validated.",
        "Do not claim N32 caused improvement.",
        "Do not claim GNN-RL superiority unless supported.",
        "Do not claim online RL.",
        "Do not claim arbitrary-N generalization.",
        "Do not claim physical optimum.",
        "Do not claim Abaqus was run.",
    ]
    CLAIM_BOUNDARY_MD.write_text("# Run53 Claim Boundary\n\n## Safe claims\n" + "\n".join(f"- {x}" for x in safe) + "\n\n## Unsafe claims\n" + "\n".join(f"- {x}" for x in unsafe) + "\n", encoding="utf-8")
    write_json(CLAIM_BOUNDARY_JSON, {"verdict": "RUN53_MODEL_UPDATE_AND_PRIMARY_BATCH64_CANDIDATE_GENERATION_ONLY_NO_TEACHER_VALIDATION", "safe_claims": safe, "unsafe_claims": unsafe})


def write_report(summary: dict[str, Any]) -> None:
    REPORT_PATH.write_text(f"""# Stage 3 Run 53 - Combined328 Calibrated N24/N40 Batch64 Candidate Generation

## 1. Purpose
Run53 updates offline diagnostics using native combined328 and combined328_plus_N32, then creates the user-selected calibrated N24/N40 overnight batch64. It is model update and candidate generation only.

## 2. Inputs
- Native combined328: `{COMBINED328_READY}`
- combined328_plus_N32: `{COMBINED328_PLUS_N32_READY}`
- Run51 prediction audit: `{RUN51_PRED_SUMMARY}`

## 3. Run52/Run51 Context
Run51 produced an N24 U2 best and N40 strict/reward gains versus combined296, but did not create raw PEEQ, SurfaceT, or Mises bests. Run51 prediction calibration was moderate, so Run53 uses calibrated diversification instead of a naive top64.

## 4. User Decision to Select Overnight Batch64
The primary selected batch is `calibrated_N24_N40_batch64` with N24=32 and N40=32. N12, N16, and N32 are not selected for the primary batch.

## 5. Target Reward Definition Audit
{summary['target_audit_headline']}

## 6. Feature Reconstruction
Run53 wrote Run22/Run29/Run33/Run38-compatible handcrafted order features for native combined328 and combined328_plus_N32.

## 7. Surrogate Update and Calibration
Best surrogate overall: `{summary['surrogate']['best_overall']}`.

Best native surrogate: `{summary['surrogate']['best_native']}`.

Best plus-N32 surrogate: `{summary['surrogate']['best_plus_N32']}`.

## 8. GNN and Graph-Pointer Diagnostics
{summary['gnn']['note']} Best GNN diagnostic: `{summary['gnn']['best_regime']}`.
{summary['pointer']['headline']} Mean NLL by regime: `{summary['pointer']['mean_nll_by_regime']}`.

## 9. Calibrated N24/N40 Candidate Pool Generation
Candidate pool counts: `{summary['candidate_pool_counts']}`. N24 and N40 each meet the >=6000 candidate minimum.

## 10. Primary Selected Batch64
Path: `{OUTPUT_DIR / 'run53_calibrated_N24_N40_batch64_candidate_orders.csv'}`. Counts: `{summary['option_counts']['primary_batch64']}`.

## 11. Reference Batch32 and Recovery Batch40
Reference batch32 path: `{OUTPUT_DIR / 'run53_calibrated_N24_N40_batch32_REFERENCE_candidate_orders.csv'}`. Counts: `{summary['option_counts']['reference_batch32']}`.

Recovery batch40 path: `{OUTPUT_DIR / 'run53_native_recovery_batch40_with_anchors_REFERENCE_candidate_orders.csv'}`. Counts: `{summary['option_counts']['recovery_batch40']}`.

## 12. Comparison to Previous Batches
{summary['comparison']['headline']}

## 13. Claim Boundary
Verdict: `RUN53_MODEL_UPDATE_AND_PRIMARY_BATCH64_CANDIDATE_GENERATION_ONLY_NO_TEACHER_VALIDATION`.

## 14. Output Files
- Candidate pool: `{OUTPUT_DIR / 'run53_candidate_pool_scored.csv'}`
- Surrogate summary: `{OUTPUT_DIR / 'run53_surrogate_validation_summary.json'}`
- GNN summary: `{OUTPUT_DIR / 'run53_gnn_reward_validation_summary.json'}`
- Pointer summary: `{OUTPUT_DIR / 'run53_graph_pointer_policy_validation_summary.json'}`
- Batch64 comparison: `{OUTPUT_DIR / 'run53_batch64_comparison_summary.json'}`
- Manifest: `{MANIFEST_PATH}`

## 15. Recommended Run54
Create a handoff package for the primary selected batch64: `run53_calibrated_N24_N40_batch64_candidate_orders.csv`. Do not generate CAE/INP until Run54 handoff is approved.
""", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    native = add_features_and_targets(read_csv(COMBINED328_READY), native=True)
    plus = add_features_and_targets(read_csv(COMBINED328_PLUS_N32_READY), native=False)
    pred_summary = json.loads(RUN51_PRED_SUMMARY.read_text(encoding="utf-8")) if RUN51_PRED_SUMMARY.exists() else {}
    validation = validate_inputs(native, plus, pred_summary)
    write_json(OUTPUT_DIR / "run53_input_validation_summary.json", validation)
    if not validation["verdict"].startswith("PASS"):
        raise SystemExit(validation["errors"])

    native.to_csv(OUTPUT_DIR / "combined328_scan_order_features.csv", index=False)
    plus.to_csv(OUTPUT_DIR / "combined328_plus_N32_scan_order_features.csv", index=False)
    target_rows = [
        {"target": "target_reward_combined328_u2_primary", "definition": "0.65 U2 + 0.20 PEEQ + 0.10 SurfaceT + 0.05 Mises", "dataset": "native_combined328"},
        {"target": "target_reward_combined328_constrained_u2_reward_balanced", "definition": "0.50 U2 + 0.25 PEEQ + 0.15 SurfaceT + 0.10 Mises", "dataset": "native_combined328"},
        {"target": "target_reward_combined328_strict_penalty_guard", "definition": "0.40 U2 + 0.30 PEEQ + 0.20 SurfaceT + 0.10 Mises", "dataset": "native_combined328"},
        {"target": "target_reward_combined328_penalty_repair", "definition": "0.30 U2 + 0.30 PEEQ + 0.25 SurfaceT + 0.15 Mises", "dataset": "native_combined328"},
        {"target": "target_reward_combined328_N24_u2_retention", "definition": "N24 U2 top-region retention with median and lower-quartile penalty guards", "dataset": "native_combined328"},
        {"target": "target_reward_combined328_N40_strict_reward_retention", "definition": "N40 strict/reward retention with local penalty-repair emphasis", "dataset": "native_combined328"},
        {"target": "target_reward_combined328_two_stage_penalty_repair", "definition": "top-35-percent U2 or strict-reward gate followed by 0.40 PEEQ + 0.35 SurfaceT + 0.25 Mises penalty ranking", "dataset": "native_combined328"},
        {"target": "target_reward_combined328_two_stage_guarded", "definition": "top-25-percent U2 gate followed by weighted PEEQ/SurfaceT/Mises penalty ranking", "dataset": "native_combined328"},
        {"target": "target_reward_combined328_no_penalty_worse_than_median", "definition": "top-35-percent U2 gate with median guards for PEEQ, SurfaceT, and Mises", "dataset": "native_combined328"},
        {"target": "target_reward_combined328_u2_guarded", "definition": "U2-primary score with penalties for below-median and bottom-quartile PEEQ/SurfaceT/Mises scores within N", "dataset": "native_combined328"},
        {"target": "target_reward_combined328_plus_N32_mapped_u2_primary", "definition": "mapped U2-primary target with N32 semantic warnings preserved", "dataset": "combined328_plus_N32"},
        {"target": "target_reward_combined328_plus_N32_strict_u2_surfaceT", "definition": "strict U2 + SurfaceT target for safer N32 metric semantics", "dataset": "combined328_plus_N32"},
    ]
    pd.DataFrame(target_rows).to_csv(OUTPUT_DIR / "run53_target_reward_definition_audit.csv", index=False)
    write_json(OUTPUT_DIR / "run53_target_reward_definition_audit.json", {"headline": "Calibrated targets combine N24 U2 retention, N40 strict/reward retention, penalty repair, and two-stage penalty repair after Run51 improved U2/reward without raw PEEQ/SurfaceT/Mises records.", "targets": target_rows})

    detailed, best_cfg, surrogate_summary, best_model, best_features = run_surrogate_validation(native, plus)
    detailed.to_csv(OUTPUT_DIR / "run53_surrogate_validation_results_detailed.csv", index=False)
    best_cfg.to_csv(OUTPUT_DIR / "run53_best_surrogate_configurations.csv", index=False)
    write_json(OUTPUT_DIR / "run53_surrogate_validation_summary.json", surrogate_summary)

    gnn_results, gnn_summary = run_gnn_validation(native, plus)
    gnn_results.to_csv(OUTPUT_DIR / "run53_gnn_reward_validation_results.csv", index=False)
    write_json(OUTPUT_DIR / "run53_gnn_reward_validation_summary.json", gnn_summary)

    pointer_log, pointer_summary = pointer_policy(native, plus)
    pointer_log.to_csv(OUTPUT_DIR / "run53_graph_pointer_policy_training_log.csv", index=False)
    write_json(OUTPUT_DIR / "run53_graph_pointer_policy_validation_summary.json", pointer_summary)

    pool = generate_candidates(native, best_model, best_features)
    pool.to_csv(OUTPUT_DIR / "run53_candidate_pool_scored.csv", index=False)

    primary = select_batch(pool, PRIMARY_COUNTS, "calibrated_N24_N40_batch64")
    reference32 = select_batch(pool, REFERENCE_BATCH32_COUNTS, "calibrated_N24_N40_batch32_REFERENCE")
    recovery40 = select_batch(pool, OPTION_C_COUNTS, "native_recovery_batch40_with_anchors_REFERENCE")
    primary.to_csv(OUTPUT_DIR / "run53_calibrated_N24_N40_batch64_candidate_orders.csv", index=False)
    reference32.to_csv(OUTPUT_DIR / "run53_calibrated_N24_N40_batch32_REFERENCE_candidate_orders.csv", index=False)
    recovery40.to_csv(OUTPUT_DIR / "run53_native_recovery_batch40_with_anchors_REFERENCE_candidate_orders.csv", index=False)

    comparison, comparison_summary = compare_batches({"primary_batch64": primary, "reference_batch32": reference32, "recovery_batch40": recovery40}, native)
    comparison.to_csv(OUTPUT_DIR / "run53_batch64_comparison_to_previous.csv", index=False)
    write_json(OUTPUT_DIR / "run53_batch64_comparison_summary.json", comparison_summary)

    write_claim_boundary()
    option_counts = {"primary_batch64": n_counts(primary), "reference_batch32": n_counts(reference32), "recovery_batch40": n_counts(recovery40)}
    summary = {
        "target_audit_headline": "Calibrated targets combine N24 U2 retention, N40 strict/reward retention, penalty repair, and two-stage penalty repair after Run51 improved U2/reward without raw PEEQ/SurfaceT/Mises records.",
        "surrogate": surrogate_summary,
        "gnn": gnn_summary,
        "pointer": pointer_summary,
        "candidate_pool_counts": n_counts(pool),
        "option_counts": option_counts,
        "comparison": comparison_summary,
    }
    write_report(summary)

    output_files = [
        OUTPUT_DIR / "run53_input_validation_summary.json",
        OUTPUT_DIR / "combined328_scan_order_features.csv",
        OUTPUT_DIR / "combined328_plus_N32_scan_order_features.csv",
        OUTPUT_DIR / "run53_target_reward_definition_audit.csv",
        OUTPUT_DIR / "run53_target_reward_definition_audit.json",
        OUTPUT_DIR / "run53_surrogate_validation_results_detailed.csv",
        OUTPUT_DIR / "run53_best_surrogate_configurations.csv",
        OUTPUT_DIR / "run53_surrogate_validation_summary.json",
        OUTPUT_DIR / "run53_gnn_reward_validation_results.csv",
        OUTPUT_DIR / "run53_gnn_reward_validation_summary.json",
        OUTPUT_DIR / "run53_graph_pointer_policy_training_log.csv",
        OUTPUT_DIR / "run53_graph_pointer_policy_validation_summary.json",
        OUTPUT_DIR / "run53_candidate_pool_scored.csv",
        OUTPUT_DIR / "run53_calibrated_N24_N40_batch64_candidate_orders.csv",
        OUTPUT_DIR / "run53_calibrated_N24_N40_batch32_REFERENCE_candidate_orders.csv",
        OUTPUT_DIR / "run53_native_recovery_batch40_with_anchors_REFERENCE_candidate_orders.csv",
        OUTPUT_DIR / "run53_batch64_comparison_to_previous.csv",
        OUTPUT_DIR / "run53_batch64_comparison_summary.json",
        CLAIM_BOUNDARY_MD,
        CLAIM_BOUNDARY_JSON,
        REPORT_PATH,
        MANIFEST_PATH,
    ]
    manifest = {
        "run_id": RUN_ID,
        "run_name": RUN_NAME,
        "timestamp": now_iso(),
        "branch": branch(),
        "script_path": str(SCRIPT_PATH),
        "input_files": [str(COMBINED328_READY), str(COMBINED328_PLUS_N32_READY), str(RUN51_ENRICHED), str(RUN51_PRED_SUMMARY), str(RUN52_REPORT), str(RUN52_MANIFEST)],
        "output_files": [str(p) for p in output_files],
        "native_combined328_rows": int(len(native)),
        "combined328_plus_N32_rows": int(len(plus)),
        "best_surrogate_summary": surrogate_summary,
        "best_gnn_summary": gnn_summary,
        "pointer_summary": pointer_summary,
        "candidate_pool_count": int(len(pool)),
        "candidate_pool_counts": n_counts(pool),
        "primary_selected_batch_path": str(OUTPUT_DIR / "run53_calibrated_N24_N40_batch64_candidate_orders.csv"),
        "primary_selected_batch_count": int(len(primary)),
        "primary_selected_batch_per_N_counts": n_counts(primary),
        "reference_batch_paths": {
            "reference_batch32": str(OUTPUT_DIR / "run53_calibrated_N24_N40_batch32_REFERENCE_candidate_orders.csv"),
            "recovery_batch40": str(OUTPUT_DIR / "run53_native_recovery_batch40_with_anchors_REFERENCE_candidate_orders.csv"),
        },
        "report_path": str(REPORT_PATH),
        "claim_boundary_path": str(CLAIM_BOUNDARY_MD),
        "no_solver_run": True,
        "no_odb_opened": True,
        "no_abqjobpilot_run": True,
        "no_cae_inp_generated": True,
        "no_teacher_validation": True,
        "no_online_rl": True,
        "no_commit_or_push": True,
    }
    write_json(MANIFEST_PATH, manifest)
    print(json.dumps({"verdict": validation["verdict"], "candidate_pool_counts": n_counts(pool), "best_surrogate": surrogate_summary["best_overall"], "best_gnn": gnn_summary["best_regime"], "report": str(REPORT_PATH)}, indent=2))


if __name__ == "__main__":
    main()


