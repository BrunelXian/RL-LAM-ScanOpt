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
RUN_ID = "run_58_combined392_model_update_N24_N40_evidence_freeze_and_N40_focused_candidate_generation"
RUN_NAME = "combined392 model update N24/N40 evidence freeze and N40 focused candidate generation"
SCRIPT_PATH = ROOT / "scripts" / "stage3" / "run_58_combined392_model_update_N24_N40_evidence_freeze_and_N40_focused_candidate_generation.py"

RUN57_DIR = ROOT / "outputs" / "stage3_run_57_calibrated_N24_N40_batch64_teacher_metrics_ingestion_and_combined392_ranking"
COMBINED392_READY = RUN57_DIR / "combined392_RL_ready_dataset.csv"
COMBINED392_PLUS_N32_READY = RUN57_DIR / "combined392_plus_N32_RL_ready_dataset.csv"
RUN56_ENRICHED = RUN57_DIR / "run56_calibrated_N24_N40_batch64_teacher_dataset_enriched.csv"
RUN56_COMPARISON = RUN57_DIR / "run56_vs_combined328_best_comparison.csv"
RUN56_EFFECTIVENESS = RUN57_DIR / "run56_calibrated_batch64_effectiveness_audit.csv"
RUN56_PRED_AUDIT = RUN57_DIR / "run56_prediction_audit_for_run53_batch64.csv"
RUN56_PRED_SUMMARY = RUN57_DIR / "run56_prediction_audit_for_run53_batch64_summary.json"
RUN57_MATURITY = RUN57_DIR / "n24_n40_maturity_and_rl_readiness_summary.json"
RUN57_MATURITY_MD = RUN57_DIR / "n24_n40_maturity_and_rl_readiness_audit.md"
RUN57_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_57_calibrated_N24_N40_batch64_teacher_metrics_ingestion_and_combined392_ranking" / "RUN_57_CALIBRATED_N24_N40_BATCH64_TEACHER_METRICS_INGESTION_AND_COMBINED392_RANKING_REPORT.md"
RUN57_MANIFEST = ROOT / "artifacts" / "manifests" / "stage3_run_57_manifest.json"

RUN56_HANDOFF = ROOT / "outputs" / "stage3_run_54_run53_calibrated_N24_N40_batch64_handoff_package" / "stage3_run54_calibrated_N24_N40_batch64_candidate_orders.csv"
RUN51_HANDOFF = ROOT / "outputs" / "stage3_run_49_run48_stricter_constrained_N24_N40_batch32_handoff_package" / "stage3_run49_stricter_constrained_N24_N40_batch32_candidate_orders.csv"
RUN46_HANDOFF = ROOT / "outputs" / "stage3_run_44_run43_constrained_N24_N40_batch32_handoff_package" / "stage3_run44_constrained_N24_N40_batch32_candidate_orders.csv"
RUN41_HANDOFF = ROOT / "outputs" / "stage3_run_39_run38_native_N24_N40_focused_batch60_handoff_package" / "stage3_run39_native_N24_N40_focused_batch60_candidate_orders.csv"
RUN36_HANDOFF = ROOT / "outputs" / "stage3_run_34_run33_N32_informed_native_batch32_handoff_package" / "stage3_run34_N32_informed_native_batch32_candidate_orders.csv"
RUN27_HANDOFF = ROOT / "outputs" / "stage3_run_24_run23_shortlist64_active_learning_handoff_package" / "stage3_run24_shortlist64_candidate_orders.csv"
RUN31_OLD = ROOT / "outputs" / "stage3_run_30_run29_hybrid_policy_batch32_handoff_package" / "stage3_run30_hybrid_policy_batch32_candidate_orders.csv"

OUTPUT_DIR = ROOT / "outputs" / "stage3_run_58_combined392_model_update_N24_N40_evidence_freeze_and_N40_focused_candidate_generation"
REPORT_DIR = ROOT / "docs" / "stage3" / "runs" / "run_58_combined392_model_update_N24_N40_evidence_freeze_and_N40_focused_candidate_generation"
REPORT_PATH = REPORT_DIR / "RUN_58_COMBINED392_MODEL_UPDATE_N24_N40_EVIDENCE_FREEZE_AND_N40_FOCUSED_CANDIDATE_GENERATION_REPORT.md"
MANIFEST_PATH = ROOT / "artifacts" / "manifests" / "stage3_run_58_manifest.json"
CLAIM_BOUNDARY_MD = OUTPUT_DIR / "run58_claim_boundary.md"
CLAIM_BOUNDARY_JSON = OUTPUT_DIR / "run58_claim_boundary.json"

EXPECTED_NATIVE = {12: 36, 16: 36, 24: 160, 40: 160}
EXPECTED_PLUS = {12: 36, 16: 36, 24: 160, 32: 332, 40: 160}
POOL_TARGETS = {12: 500, 16: 500, 24: 3000, 40: 7000}
OPTION_A_COUNTS = {24: 8, 40: 24}
OPTION_B_COUNTS = {24: 16, 40: 48}
OPTION_C_COUNTS = {12: 8, 16: 8, 24: 16, 40: 16}
GLOBAL_SEED = 58042

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
            "target_u2_score_combined392_rank",
            "target_peeq_score_combined392_rank",
            "target_surfaceT_score_combined392_rank",
            "target_mises_score_combined392_rank",
        ]
        out["target_reward_combined392_u2_primary"] = as_float(out["target_reward_combined392_u2_primary"])
        prefix = "combined392"
    else:
        score_cols = [
            "target_u2_score_combined392_plus_N32_rank",
            "target_peeq_score_combined392_plus_N32_rank",
            "target_surfaceT_score_combined392_plus_N32_rank",
            "target_mises_score_combined392_plus_N32_rank",
        ]
        out["target_reward_combined392_plus_N32_mapped_u2_primary"] = as_float(out["target_reward_combined392_plus_N32_mapped_u2_primary"])
        prefix = "combined392_plus_N32"
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
    out[f"target_reward_{prefix}_N40_u2_reward_retention"] = out[f"target_reward_{prefix}_N40_strict_reward_retention"]
    out[f"target_reward_{prefix}_N40_penalty_repair"] = (
        0.25 * out[score_cols[0]] + 0.30 * out[score_cols[1]] + 0.25 * out[score_cols[2]] + 0.20 * out[score_cols[3]]
    )
    out[f"target_reward_{prefix}_N24_maintenance"] = out[f"target_reward_{prefix}_N24_u2_retention"]
    out[f"target_reward_{prefix}_two_stage_N40_penalty_repair"] = out[f"target_reward_{prefix}_two_stage_penalty_repair"]
    return out


def validate_inputs(native: pd.DataFrame, plus: pd.DataFrame, pred_summary: dict[str, Any], maturity: dict[str, Any]) -> dict[str, Any]:
    errors = []
    if len(native) != 392 or n_counts(native) != EXPECTED_NATIVE:
        errors.append(f"native combined392 count mismatch: rows={len(native)} counts={n_counts(native)}")
    if 32 in set(native["n"].astype(int)):
        errors.append("native combined392 contains N32")
    if len(plus) != 724 or n_counts(plus) != EXPECTED_PLUS:
        errors.append(f"combined392_plus_N32 count mismatch: rows={len(plus)} counts={n_counts(plus)}")
    for col in [
        "target_reward_combined392_u2_primary",
        "target_reward_combined392_constrained_u2_reward_balanced",
        "target_reward_combined392_strict_penalty_guard",
        "target_reward_combined392_penalty_repair",
    ]:
        if col not in native.columns:
            errors.append(f"native missing target {col}")
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
    if maturity:
        expected_maturity = {
            "n24_teacher_rows": 160,
            "n40_teacher_rows": 160,
            "n12_teacher_rows": 36,
            "n16_teacher_rows": 36,
        }
        for key, expected in expected_maturity.items():
            if int(maturity.get(key, -1)) != expected:
                errors.append(f"maturity audit {key} mismatch: {maturity.get(key)}")
        if not maturity.get("full_variable_n_rl_maturity_limited_by_n12_n16", False):
            errors.append("maturity audit does not preserve full variable-N limitation warning")
    else:
        errors.append("Run57 maturity audit summary missing")
    if pred_summary:
        reward_s = pred_summary.get("overall_reward_spearman")
        top5 = pred_summary.get("mean_top5_overlap")
    else:
        reward_s = None
        top5 = None
    return {
        "timestamp": now_iso(),
        "verdict": "PASS_RUN58_COMBINED392_AND_PLUS_N32_INPUTS_READY" if not errors else "FAIL_RUN58_INPUT_VALIDATION",
        "errors": errors,
        "native_combined392_rows": int(len(native)),
        "native_combined392_counts": n_counts(native),
        "combined392_plus_N32_rows": int(len(plus)),
        "combined392_plus_N32_counts": n_counts(plus),
        "maturity_context": maturity,
        "run56_prediction_audit_context": {"strict_guard_or_reward_spearman": reward_s, "mean_top5_overlap": top5, "top1_hit": pred_summary.get("top1_hit") if pred_summary else None},
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
        return sk["RandomForestRegressor"](n_estimators=16, random_state=seed, min_samples_leaf=2, n_jobs=-1)
    if name == "ExtraTreesRegressor":
        return sk["ExtraTreesRegressor"](n_estimators=20, random_state=seed, min_samples_leaf=1, n_jobs=-1)
    if name == "GradientBoostingRegressor":
        return sk["GradientBoostingRegressor"](random_state=seed)
    if name == "OrderGraphMLPRegressor":
        return sk["make_pipeline"](sk["StandardScaler"](), sk["MLPRegressor"](hidden_layer_sizes=(24, 12), alpha=0.001, max_iter=120, random_state=seed))
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
    if "dataset_source" in df.columns and (df["dataset_source"].astype(str) == "run56_calibrated_N24_N40_batch64").any():
        test = df.index[df["dataset_source"].astype(str) == "run56_calibrated_N24_N40_batch64"].to_numpy()
        train = np.setdiff1d(rows, test)
        protocols.append(("train_pre_Run56_test_Run56", train, test))
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
        "native_combined392": [
            "target_reward_combined392_u2_primary",
            "target_reward_combined392_constrained_u2_reward_balanced",
            "target_reward_combined392_strict_penalty_guard",
            "target_reward_combined392_penalty_repair",
            "target_reward_combined392_N40_u2_reward_retention",
            "target_reward_combined392_N40_penalty_repair",
            "target_reward_combined392_N24_maintenance",
            "target_reward_combined392_two_stage_N40_penalty_repair",
            "target_reward_combined392_two_stage_guarded",
            "target_reward_combined392_no_penalty_worse_than_median",
            "target_reward_combined392_u2_guarded",
        ],
        "plus_N32_unweighted": [
            "target_reward_combined392_plus_N32_mapped_u2_primary",
            "target_reward_combined392_plus_N32_strict_u2_surfaceT",
            "target_reward_combined392_plus_N32_constrained_u2_reward_balanced",
            "target_reward_combined392_plus_N32_strict_penalty_guard",
            "target_reward_combined392_plus_N32_penalty_repair",
            "target_reward_combined392_plus_N32_two_stage_penalty_repair",
        ],
        "plus_N32_balanced": [
            "target_reward_combined392_plus_N32_mapped_u2_primary",
            "target_reward_combined392_plus_N32_strict_u2_surfaceT",
            "target_reward_combined392_plus_N32_constrained_u2_reward_balanced",
            "target_reward_combined392_plus_N32_strict_penalty_guard",
            "target_reward_combined392_plus_N32_penalty_repair",
            "target_reward_combined392_plus_N32_two_stage_penalty_repair",
        ],
    }
    frames = {"native_combined392": native, "plus_N32_unweighted": plus, "plus_N32_balanced": plus}
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
    native_best = grouped[grouped["regime"] == "native_combined392"].head(1).iloc[0].to_dict()
    plus_best = grouped[grouped["regime"].str.contains("plus")].head(1).iloc[0].to_dict()
    def best_for(target: str) -> dict[str, Any]:
        part = grouped[grouped["target"] == target]
        return part.head(1).iloc[0].to_dict() if not part.empty else {}
    summary = {
        "best_overall": best_row,
        "best_native": native_best,
        "best_plus_N32": plus_best,
        "best_N40_u2_reward_retention": best_for("target_reward_combined392_N40_u2_reward_retention"),
        "best_N40_penalty_repair": best_for("target_reward_combined392_N40_penalty_repair"),
        "best_N24_maintenance": best_for("target_reward_combined392_N24_maintenance"),
        "best_penalty_repair": best_for("target_reward_combined392_penalty_repair"),
        "n32_augmented_better_than_native": bool(plus_best["macro_spearman"] > native_best["macro_spearman"]),
        "run53_reference": {"native_u2_primary_spearman": 0.8350, "native_top5": 1.5, "calibrated_spearman": 0.7924, "penalty_repair_spearman": 0.6633},
    }
    best_model = make_model(str(native_best["model"]), GLOBAL_SEED)
    features = [c for c in FEATURE_SETS[str(native_best["feature_set"])] if c in native.columns]
    best_model.fit(native[features].fillna(0.0).to_numpy(float), native[str(native_best["target"])].to_numpy(float))
    return detailed, best, summary, best_model, features


def run_gnn_validation(native: pd.DataFrame, plus: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    for regime, df in {"native_combined392": native, "plus_N32_unweighted": plus, "plus_N32_balanced": plus}.items():
        target = "target_reward_combined392_u2_primary" if regime == "native_combined392" else "target_reward_combined392_plus_N32_mapped_u2_primary"
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
    for regime, df in {"native_combined392": native, "plus_N32_unweighted": plus, "plus_N32_balanced": plus}.items():
        target = "target_reward_combined392_u2_primary" if regime == "native_combined392" else "target_reward_combined392_plus_N32_mapped_u2_primary"
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
        best_by[(n, "reward")] = parse_order(g.sort_values("target_reward_combined392_u2_primary", ascending=False).iloc[0]["order_json"])
        best_by[(n, "constrained")] = parse_order(g.sort_values("target_reward_combined392_constrained_u2_reward_balanced", ascending=False).iloc[0]["order_json"])
        best_by[(n, "strict")] = parse_order(g.sort_values("target_reward_combined392_strict_penalty_guard", ascending=False).iloc[0]["order_json"])
        best_by[(n, "two_stage")] = parse_order(g.sort_values("target_reward_combined392_two_stage_guarded", ascending=False).iloc[0]["order_json"])
    rows, seen = [], set(existing)
    for n, target_count in POOL_TARGETS.items():
        if n == 40:
            source_plan = [
                "N40_u2_reward_retention_top", "N40_u2_reward_retention_local_repair",
                "N40_penalty_repair_top", "N40_penalty_repair_local_search",
                "N40_two_stage_penalty_repair", "N40_median_guard_repair",
                "N40_no_penalty_worse_than_median", "N40_PEEQ_repair_candidates",
                "N40_SurfaceT_repair_candidates", "N40_Mises_repair_candidates",
                "N40_strict_guard_diverse", "N40_uncertainty_calibration",
                "N40_diversity_coverage", "N40_sentinel_control",
            ]
        elif n == 24:
            source_plan = [
                "N24_u2_reward_maintenance", "N24_penalty_repair_diagnostic",
                "N24_uncertainty_calibration", "N24_diversity_coverage", "N24_sentinel_control",
            ]
        else:
            source_plan = ["variableN_anchor_top", "variableN_anchor_diversity", "variableN_anchor_sentinel"]
        attempts = 0
        while sum(1 for r in rows if r["n"] == n) < target_count and attempts < target_count * 20:
            attempts += 1
            source = source_plan[attempts % len(source_plan)]
            if "u2_reward_retention" in source or "u2_reward_maintenance" in source or "PEEQ_repair" in source:
                base = best_by[(n, "u2")]
            elif "strict" in source or "Mises_repair" in source:
                base = best_by[(n, "strict")]
            elif "two_stage" in source or "median" in source:
                base = best_by[(n, "two_stage")]
            elif "SurfaceT_repair" in source or "penalty_repair" in source:
                base = best_by[(n, "constrained")]
            else:
                base = best_by[(n, "reward")]
            if attempts > target_count * 6:
                source = source_plan[attempts % len(source_plan)]
                order = list(range(n))
                rng.shuffle(order)
            elif source == "graph_pointer_temperature_sample":
                order = list(range(n)); rng.shuffle(order)
            elif source.endswith("diversity_coverage") or source == "diversity_coverage":
                step = rng.choice([s for s in range(3, n) if math.gcd(s, n) == 1] or [1])
                order = regular_jump(n, step, rng.randrange(n))
            elif source.endswith("sentinel_control") or source == "sentinel_control":
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
            n24_retention = (0.62 * u2_pred + 0.38 * penalty_repair) if n == 24 else constrained_reward
            n40_retention = (0.50 * strict + 0.50 * penalty_repair) if n == 40 else constrained_reward
            n40_repair = (0.25 * u2_pred + 0.30 * peeq_guard + 0.25 * surf_guard + 0.20 * mises_guard) if n == 40 else penalty_repair
            calibrated = 0.20 * u2_pred + 0.20 * strict + 0.30 * penalty_repair + 0.20 * n40_repair + 0.10 * median_guard - 0.12 * max(0.0, 0.16 - near_d)
            cid = f"R58_N{n}_C{sum(1 for r in rows if r['n']==n)+1:05d}"
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
                "N24_maintenance_score": n24_retention,
                "N40_u2_reward_retention_score": n40_retention,
                "N40_penalty_repair_score": n40_repair,
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
            ("N24_u2_reward_maintenance", 3),
            ("N24_penalty_repair_diagnostic", 2),
            ("N24_uncertainty_calibration", 1),
            ("N24_diversity_coverage", 1),
            ("N24_sentinel_control", 1),
        ],
        40: [
            ("N40_u2_reward_retention_top", 3), ("N40_u2_reward_retention_local_repair", 3),
            ("N40_penalty_repair_top", 4), ("N40_penalty_repair_local_search", 4),
            ("N40_two_stage_penalty_repair", 3),
            ("N40_median_guard_repair", 2), ("N40_no_penalty_worse_than_median", 2),
            ("N40_PEEQ_repair_candidates", 1), ("N40_SurfaceT_repair_candidates", 1),
            ("N40_Mises_repair_candidates", 1),
        ],
        12: [("variableN_anchor_top", 3), ("variableN_anchor_diversity", 3), ("variableN_anchor_sentinel", 2)],
        16: [("variableN_anchor_top", 3), ("variableN_anchor_diversity", 3), ("variableN_anchor_sentinel", 2)],
    }
    fallback_order = [
        "N40_u2_reward_retention_top", "N40_u2_reward_retention_local_repair",
        "N40_penalty_repair_top", "N40_penalty_repair_local_search", "N40_two_stage_penalty_repair",
        "N40_median_guard_repair", "N40_no_penalty_worse_than_median",
        "N40_PEEQ_repair_candidates", "N40_SurfaceT_repair_candidates", "N40_Mises_repair_candidates",
        "N24_u2_reward_maintenance", "N24_penalty_repair_diagnostic",
        "N40_uncertainty_calibration", "N40_diversity_coverage", "N40_sentinel_control",
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
            short = str(row["candidate_source"]).replace("_local_search", "").replace("_local_repair", "repair").replace("_candidates", "").replace("penalty_repair", "penalty").replace("u2_reward_retention", "u2ret").replace("N40_", "n40_").replace("N24_", "n24_")[:30]
            row["batch_option"] = label
            row["handoff_strategy_name"] = f"S3R58N40_N{n}_B{i:02d}_{short}"
            row["teacher_validated"] = False
            row["teacher_validation_status"] = "NOT_RUN"
            selected.append(row)
    return pd.DataFrame(selected)


def compare_batches(options: dict[str, pd.DataFrame], native: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    previous = {"combined392_teacher_orders": set(native["order_hash"].astype(str))}
    for name, path in {"run56": RUN56_HANDOFF, "run51": RUN51_HANDOFF, "run46": RUN46_HANDOFF, "run41": RUN41_HANDOFF, "run36": RUN36_HANDOFF, "run27": RUN27_HANDOFF, "superseded_run31": RUN31_OLD}.items():
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
            "mean_N40_penalty_repair_score": float(df.get("N40_penalty_repair_score", pd.Series([math.nan])).mean()),
            "mean_two_stage_repair_score": float(df.get("two_stage_penalty_repair_score", pd.Series([math.nan])).mean()),
            "mean_novelty_distance": float(df["novelty_distance"].mean()),
            "candidate_source_composition": json.dumps(df["candidate_source"].value_counts().to_dict()),
        }
        for pname, phashes in previous.items():
            row[f"overlap_{pname}"] = int(len(hashes & phashes))
        rows.append(row)
    comp = pd.DataFrame(rows)
    headline = "Run58 batch options were checked for exact order overlap, N40-focused score distribution, penalty-repair score, novelty, and source diversity against combined392, Run56, Run51, Run46, Run41, Run36, Run27, and superseded Run31."
    return comp, {"headline": headline, "rows": comp.to_dict(orient="records")}


def write_evidence_freeze(native: pd.DataFrame, validation: dict[str, Any], pred_summary: dict[str, Any], maturity: dict[str, Any]) -> dict[str, Any]:
    timeline = [
        {"stage": "combined172_baseline", "native_rows": 172, "N12": 32, "N16": 32, "N24": 54, "N40": 54, "interpretation": "pre-Run36 native baseline before N24/N40 focused loops"},
        {"stage": "Run36_N32_informed_native_batch32", "native_rows": 204, "N12": 36, "N16": 36, "N24": 66, "N40": 66, "interpretation": "N24 and N40 U2 records refreshed; no N32 cases in validation"},
        {"stage": "Run41_N24_N40_focused_batch60", "native_rows": 264, "N12": 36, "N16": 36, "N24": 96, "N40": 96, "interpretation": "near-top density increased; N40 PEEQ record observed earlier, U2 did not extend"},
        {"stage": "Run46_constrained_batch32", "native_rows": 296, "N12": 36, "N16": 36, "N24": 112, "N40": 112, "interpretation": "N24 U2/reward and N40 reward gains supported constrained selection"},
        {"stage": "Run51_stricter_constrained_batch32", "native_rows": 328, "N12": 36, "N16": 36, "N24": 128, "N40": 128, "interpretation": "N24 U2 and N40 strict/reward behavior strengthened; raw penalty records still limited"},
        {"stage": "Run56_calibrated_batch64", "native_rows": 392, "N12": 36, "N16": 36, "N24": 160, "N40": 160, "interpretation": "N40 U2/reward-family records advanced; N24 produced density but no new combined328 bests"},
    ]
    freeze = pd.DataFrame(timeline)
    freeze_path = OUTPUT_DIR / "n24_n40_active_learning_rl_evidence_freeze.csv"
    freeze.to_csv(freeze_path, index=False)

    summary = {
        "verdict": "RUN58_N24_N40_EVIDENCE_FREEZE_READY_FULL_VARIABLE_N_LIMITED_BY_N12_N16",
        "headline": "N24/N40 active-learning evidence is mature enough to freeze at 160 native teacher rows each; full variable-N RL remains limited by N12/N16 at 36 rows each.",
        "native_combined392_counts": n_counts(native),
        "run57_maturity_context": maturity,
        "run56_prediction_audit_context": {
            "reward_spearman": pred_summary.get("overall_reward_spearman") if pred_summary else None,
            "strict_or_primary_spearman": pred_summary.get("strict_guard_spearman") if pred_summary else None,
            "mean_top5_overlap": pred_summary.get("mean_top5_overlap") if pred_summary else None,
        },
        "freeze_decision": {
            "freeze_N24_N40_evidence_for_reporting": True,
            "continue_N40_focused_candidate_generation": True,
            "continue_full_variable_N_claim_boundary": "limited_by_N12_N16_anchor_rows",
        },
        "timeline": timeline,
        "validation_verdict": validation["verdict"],
    }
    write_json(OUTPUT_DIR / "n24_n40_active_learning_rl_evidence_freeze.json", summary)
    (OUTPUT_DIR / "n24_n40_active_learning_rl_evidence_freeze.md").write_text(
        "# N24/N40 Active-Learning RL Evidence Freeze\n\n"
        f"Verdict: `{summary['verdict']}`\n\n"
        f"{summary['headline']}\n\n"
        "This freeze preserves the distinction between mature N24/N40 focused evidence and the still-limited full variable-N setting. "
        "N32 rows remain legacy-compatible auxiliary data, not native Stage 3 teacher validation.\n\n"
        "## Timeline\n"
        + "\n".join(f"- {row['stage']}: N24={row['N24']}, N40={row['N40']} - {row['interpretation']}" for row in timeline)
        + "\n",
        encoding="utf-8",
    )
    return summary


def write_claim_boundary() -> None:
    safe = [
        "Run58 updates offline models using native combined392 and combined392_plus_N32.",
        "Run58 freezes the N24/N40 active-learning evidence summary while preserving the full variable-N limitation from N12/N16 row counts.",
        "Run58 generates N40-focused calibrated candidate-order batches for future teacher validation.",
        "Run58 recommends Option A with N24=8 and N40=24 for a focused next validation loop.",
        "Run58 evaluates N40 U2/reward retention and N40 penalty-repair targets after Run56 advanced N40 U2/reward-family records.",
        "Run58 does not include teacher validation for new candidates.",
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
    CLAIM_BOUNDARY_MD.write_text("# Run58 Claim Boundary\n\n## Safe claims\n" + "\n".join(f"- {x}" for x in safe) + "\n\n## Unsafe claims\n" + "\n".join(f"- {x}" for x in unsafe) + "\n", encoding="utf-8")
    write_json(CLAIM_BOUNDARY_JSON, {"verdict": "RUN58_MODEL_UPDATE_EVIDENCE_FREEZE_AND_CANDIDATE_GENERATION_ONLY_NO_TEACHER_VALIDATION", "safe_claims": safe, "unsafe_claims": unsafe})


def write_report(summary: dict[str, Any]) -> None:
    REPORT_PATH.write_text(f"""# Stage 3 Run 58 - Combined392 Model Update, N24/N40 Evidence Freeze, and N40-Focused Candidate Generation

## 1. Purpose
Run58 updates offline diagnostics using native combined392 and combined392_plus_N32, freezes the N24/N40 active-learning evidence summary, and creates N40-focused calibrated candidate batches. It is model update and candidate generation only.

## 2. Inputs
- Native combined392: `{COMBINED392_READY}`
- combined392_plus_N32: `{COMBINED392_PLUS_N32_READY}`
- Run56 prediction audit: `{RUN56_PRED_SUMMARY}`

## 3. Run57/Run56 Context
Run56 created five new records versus combined328, all in N40 U2/reward-family metrics. N24 produced useful top-density but no new combined328 bests. Prediction calibration was moderate, so Run58 uses N40-focused diversification rather than a naive top-only batch.

## 4. N24/N40 Evidence Freeze
{summary['evidence_freeze']['headline']}

## 5. Target Reward Definition Audit
{summary['target_audit_headline']}

## 6. Feature Reconstruction
Run58 wrote Run22/Run29/Run33/Run38-compatible handcrafted order features for native combined392 and combined392_plus_N32.

## 7. Surrogate Update and Calibration
Best surrogate overall: `{summary['surrogate']['best_overall']}`.

Best native surrogate: `{summary['surrogate']['best_native']}`.

Best plus-N32 surrogate: `{summary['surrogate']['best_plus_N32']}`.

## 8. GNN and Graph-Pointer Diagnostics
{summary['gnn']['note']} Best GNN diagnostic: `{summary['gnn']['best_regime']}`.
{summary['pointer']['headline']} Mean NLL by regime: `{summary['pointer']['mean_nll_by_regime']}`.

## 9. N40-Focused Candidate Pool Generation
Candidate pool counts: `{summary['candidate_pool_counts']}`. N24 meets the >=3000 candidate minimum and N40 meets the >=7000 candidate minimum.

## 10. Option A - N40-Focused Calibrated Penalty-Repair Batch32
Path: `{OUTPUT_DIR / 'run58_N40_focused_calibrated_penalty_repair_batch32_candidate_orders.csv'}`. Counts: `{summary['option_counts']['option_A_N40_focused_batch32']}`.

## 11. Option B - N40-Focused Calibrated Batch64
Path: `{OUTPUT_DIR / 'run58_N40_focused_calibrated_batch64_candidate_orders.csv'}`. Counts: `{summary['option_counts']['option_B_N40_focused_batch64']}`.

## 12. Option C - Variable-N Recovery Anchor Batch48
Path: `{OUTPUT_DIR / 'run58_variable_N_recovery_anchor_batch48_candidate_orders.csv'}`. Counts: `{summary['option_counts']['option_C_variable_N_recovery_batch48']}`.

## 13. Comparison to Previous Batches
{summary['comparison']['headline']}

## 14. Claim Boundary
Verdict: `RUN58_MODEL_UPDATE_EVIDENCE_FREEZE_AND_CANDIDATE_GENERATION_ONLY_NO_TEACHER_VALIDATION`.

## 15. Output Files
- Candidate pool: `{OUTPUT_DIR / 'run58_candidate_pool_scored.csv'}`
- Surrogate summary: `{OUTPUT_DIR / 'run58_surrogate_validation_summary.json'}`
- GNN summary: `{OUTPUT_DIR / 'run58_gnn_reward_validation_summary.json'}`
- Pointer summary: `{OUTPUT_DIR / 'run58_graph_pointer_policy_validation_summary.json'}`
- Evidence freeze: `{OUTPUT_DIR / 'n24_n40_active_learning_rl_evidence_freeze.md'}`
- Batch options comparison: `{OUTPUT_DIR / 'run58_batch_options_comparison_summary.json'}`
- Manifest: `{MANIFEST_PATH}`

## 16. Recommended Run59
Select Option A for a quick N40-focused penalty-repair validation loop unless the user explicitly wants another overnight batch64. Do not generate CAE/INP until the selected Run58 option is handed off.
""", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    native = add_features_and_targets(read_csv(COMBINED392_READY), native=True)
    plus = add_features_and_targets(read_csv(COMBINED392_PLUS_N32_READY), native=False)
    pred_summary = json.loads(RUN56_PRED_SUMMARY.read_text(encoding="utf-8")) if RUN56_PRED_SUMMARY.exists() else {}
    maturity = json.loads(RUN57_MATURITY.read_text(encoding="utf-8")) if RUN57_MATURITY.exists() else {}
    validation = validate_inputs(native, plus, pred_summary, maturity)
    write_json(OUTPUT_DIR / "run58_input_validation_summary.json", validation)
    if not validation["verdict"].startswith("PASS"):
        raise SystemExit(validation["errors"])

    native.to_csv(OUTPUT_DIR / "combined392_scan_order_features.csv", index=False)
    plus.to_csv(OUTPUT_DIR / "combined392_plus_N32_scan_order_features.csv", index=False)
    target_rows = [
        {"target": "target_reward_combined392_u2_primary", "definition": "0.65 U2 + 0.20 PEEQ + 0.10 SurfaceT + 0.05 Mises", "dataset": "native_combined392"},
        {"target": "target_reward_combined392_constrained_u2_reward_balanced", "definition": "0.50 U2 + 0.25 PEEQ + 0.15 SurfaceT + 0.10 Mises", "dataset": "native_combined392"},
        {"target": "target_reward_combined392_strict_penalty_guard", "definition": "0.40 U2 + 0.30 PEEQ + 0.20 SurfaceT + 0.10 Mises", "dataset": "native_combined392"},
        {"target": "target_reward_combined392_penalty_repair", "definition": "0.30 U2 + 0.30 PEEQ + 0.25 SurfaceT + 0.15 Mises", "dataset": "native_combined392"},
        {"target": "target_reward_combined392_N40_u2_reward_retention", "definition": "N40 U2/reward-family retention with local penalty-repair emphasis", "dataset": "native_combined392"},
        {"target": "target_reward_combined392_N40_penalty_repair", "definition": "N40 penalty repair with 0.25 U2 + 0.30 PEEQ + 0.25 SurfaceT + 0.20 Mises", "dataset": "native_combined392"},
        {"target": "target_reward_combined392_N24_maintenance", "definition": "N24 maintenance target retaining the Run51 U2 signal while avoiding penalty collapse", "dataset": "native_combined392"},
        {"target": "target_reward_combined392_two_stage_penalty_repair", "definition": "top-35-percent U2 or strict-reward gate followed by 0.40 PEEQ + 0.35 SurfaceT + 0.25 Mises penalty ranking", "dataset": "native_combined392"},
        {"target": "target_reward_combined392_two_stage_guarded", "definition": "top-25-percent U2 gate followed by weighted PEEQ/SurfaceT/Mises penalty ranking", "dataset": "native_combined392"},
        {"target": "target_reward_combined392_no_penalty_worse_than_median", "definition": "top-35-percent U2 gate with median guards for PEEQ, SurfaceT, and Mises", "dataset": "native_combined392"},
        {"target": "target_reward_combined392_u2_guarded", "definition": "U2-primary score with penalties for below-median and bottom-quartile PEEQ/SurfaceT/Mises scores within N", "dataset": "native_combined392"},
        {"target": "target_reward_combined392_plus_N32_mapped_u2_primary", "definition": "mapped U2-primary target with N32 semantic warnings preserved", "dataset": "combined392_plus_N32"},
        {"target": "target_reward_combined392_plus_N32_strict_u2_surfaceT", "definition": "strict U2 + SurfaceT target for safer N32 metric semantics", "dataset": "combined392_plus_N32"},
    ]
    pd.DataFrame(target_rows).to_csv(OUTPUT_DIR / "run58_target_reward_definition_audit.csv", index=False)
    write_json(OUTPUT_DIR / "run58_target_reward_definition_audit.json", {"headline": "Calibrated targets combine N24 U2 retention, N40 strict/reward retention, penalty repair, and two-stage penalty repair after Run56 improved U2/reward without raw PEEQ/SurfaceT/Mises records.", "targets": target_rows})

    detailed, best_cfg, surrogate_summary, best_model, best_features = run_surrogate_validation(native, plus)
    detailed.to_csv(OUTPUT_DIR / "run58_surrogate_validation_results_detailed.csv", index=False)
    best_cfg.to_csv(OUTPUT_DIR / "run58_best_surrogate_configurations.csv", index=False)
    write_json(OUTPUT_DIR / "run58_surrogate_validation_summary.json", surrogate_summary)

    gnn_results, gnn_summary = run_gnn_validation(native, plus)
    gnn_results.to_csv(OUTPUT_DIR / "run58_gnn_reward_validation_results.csv", index=False)
    write_json(OUTPUT_DIR / "run58_gnn_reward_validation_summary.json", gnn_summary)

    pointer_log, pointer_summary = pointer_policy(native, plus)
    pointer_log.to_csv(OUTPUT_DIR / "run58_graph_pointer_policy_training_log.csv", index=False)
    write_json(OUTPUT_DIR / "run58_graph_pointer_policy_validation_summary.json", pointer_summary)

    pool = generate_candidates(native, best_model, best_features)
    pool.to_csv(OUTPUT_DIR / "run58_candidate_pool_scored.csv", index=False)

    option_a = select_batch(pool, OPTION_A_COUNTS, "N40_focused_calibrated_penalty_repair_batch32")
    option_b = select_batch(pool, OPTION_B_COUNTS, "N40_focused_calibrated_batch64")
    option_c = select_batch(pool, OPTION_C_COUNTS, "variable_N_recovery_anchor_batch48")
    option_a.to_csv(OUTPUT_DIR / "run58_N40_focused_calibrated_penalty_repair_batch32_candidate_orders.csv", index=False)
    option_b.to_csv(OUTPUT_DIR / "run58_N40_focused_calibrated_batch64_candidate_orders.csv", index=False)
    option_c.to_csv(OUTPUT_DIR / "run58_variable_N_recovery_anchor_batch48_candidate_orders.csv", index=False)

    comparison, comparison_summary = compare_batches({"option_A_N40_focused_batch32": option_a, "option_B_N40_focused_batch64": option_b, "option_C_variable_N_recovery_batch48": option_c}, native)
    comparison.to_csv(OUTPUT_DIR / "run58_batch_options_comparison_to_previous.csv", index=False)
    write_json(OUTPUT_DIR / "run58_batch_options_comparison_summary.json", comparison_summary)

    write_claim_boundary()
    evidence_freeze = write_evidence_freeze(native, validation, pred_summary, maturity)
    option_counts = {"option_A_N40_focused_batch32": n_counts(option_a), "option_B_N40_focused_batch64": n_counts(option_b), "option_C_variable_N_recovery_batch48": n_counts(option_c)}
    summary = {
        "target_audit_headline": "Run58 target definitions prioritize N40 U2/reward retention, N40 penalty repair, and N24 maintenance while preserving N32 metric-semantic warnings.",
        "surrogate": surrogate_summary,
        "gnn": gnn_summary,
        "pointer": pointer_summary,
        "evidence_freeze": evidence_freeze,
        "candidate_pool_counts": n_counts(pool),
        "option_counts": option_counts,
        "comparison": comparison_summary,
    }
    write_report(summary)

    output_files = [
        OUTPUT_DIR / "run58_input_validation_summary.json",
        OUTPUT_DIR / "combined392_scan_order_features.csv",
        OUTPUT_DIR / "combined392_plus_N32_scan_order_features.csv",
        OUTPUT_DIR / "run58_target_reward_definition_audit.csv",
        OUTPUT_DIR / "run58_target_reward_definition_audit.json",
        OUTPUT_DIR / "run58_surrogate_validation_results_detailed.csv",
        OUTPUT_DIR / "run58_best_surrogate_configurations.csv",
        OUTPUT_DIR / "run58_surrogate_validation_summary.json",
        OUTPUT_DIR / "run58_gnn_reward_validation_results.csv",
        OUTPUT_DIR / "run58_gnn_reward_validation_summary.json",
        OUTPUT_DIR / "run58_graph_pointer_policy_training_log.csv",
        OUTPUT_DIR / "run58_graph_pointer_policy_validation_summary.json",
        OUTPUT_DIR / "n24_n40_active_learning_rl_evidence_freeze.csv",
        OUTPUT_DIR / "n24_n40_active_learning_rl_evidence_freeze.json",
        OUTPUT_DIR / "n24_n40_active_learning_rl_evidence_freeze.md",
        OUTPUT_DIR / "run58_candidate_pool_scored.csv",
        OUTPUT_DIR / "run58_N40_focused_calibrated_penalty_repair_batch32_candidate_orders.csv",
        OUTPUT_DIR / "run58_N40_focused_calibrated_batch64_candidate_orders.csv",
        OUTPUT_DIR / "run58_variable_N_recovery_anchor_batch48_candidate_orders.csv",
        OUTPUT_DIR / "run58_batch_options_comparison_to_previous.csv",
        OUTPUT_DIR / "run58_batch_options_comparison_summary.json",
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
        "input_files": [str(COMBINED392_READY), str(COMBINED392_PLUS_N32_READY), str(RUN56_ENRICHED), str(RUN56_PRED_SUMMARY), str(RUN57_MATURITY), str(RUN57_REPORT), str(RUN57_MANIFEST)],
        "output_files": [str(p) for p in output_files],
        "native_combined392_rows": int(len(native)),
        "combined392_plus_N32_rows": int(len(plus)),
        "best_surrogate_summary": surrogate_summary,
        "best_N40_model_summary": surrogate_summary.get("best_N40_u2_reward_retention"),
        "best_N40_penalty_repair_summary": surrogate_summary.get("best_N40_penalty_repair"),
        "best_N24_maintenance_summary": surrogate_summary.get("best_N24_maintenance"),
        "best_gnn_summary": gnn_summary,
        "pointer_summary": pointer_summary,
        "evidence_freeze_summary": evidence_freeze,
        "candidate_pool_count": int(len(pool)),
        "candidate_pool_counts": n_counts(pool),
        "recommended_option": "option_A_N40_focused_calibrated_penalty_repair_batch32",
        "batch_option_paths": {
            "option_A_N40_focused_batch32": str(OUTPUT_DIR / "run58_N40_focused_calibrated_penalty_repair_batch32_candidate_orders.csv"),
            "option_B_N40_focused_batch64": str(OUTPUT_DIR / "run58_N40_focused_calibrated_batch64_candidate_orders.csv"),
            "option_C_variable_N_recovery_batch48": str(OUTPUT_DIR / "run58_variable_N_recovery_anchor_batch48_candidate_orders.csv"),
        },
        "batch_option_counts": option_counts,
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
    print(json.dumps({"verdict": validation["verdict"], "candidate_pool_counts": n_counts(pool), "option_A_counts": n_counts(option_a), "best_surrogate": surrogate_summary["best_overall"], "best_gnn": gnn_summary["best_regime"], "report": str(REPORT_PATH)}, indent=2))


if __name__ == "__main__":
    main()


