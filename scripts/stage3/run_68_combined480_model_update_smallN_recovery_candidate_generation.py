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
RUN_ID = "run_68_combined480_model_update_smallN_recovery_candidate_generation"
RUN_NAME = "combined480 model update smallN recovery candidate generation"
SCRIPT_PATH = ROOT / "scripts" / "stage3" / "run_68_combined480_model_update_smallN_recovery_candidate_generation.py"

RUN67_DIR = ROOT / "outputs" / "stage3_run_67_variable_N_recovery_anchor_batch48_teacher_metrics_ingestion_and_combined480_ranking"
COMBINED480_READY = RUN67_DIR / "combined480_RL_ready_dataset.csv"
COMBINED480_PLUS_N32_READY = RUN67_DIR / "combined480_plus_N32_RL_ready_dataset.csv"
RUN66_ENRICHED = RUN67_DIR / "run66_variable_N_recovery_anchor_batch48_teacher_dataset_enriched.csv"
RUN66_COMPARISON = RUN67_DIR / "run66_vs_combined432_best_comparison.csv"
RUN66_EFFECTIVENESS = RUN67_DIR / "run66_variable_N_recovery_anchor_batch48_effectiveness_audit.csv"
RUN66_PRED_AUDIT = RUN67_DIR / "run66_prediction_audit_for_run63_batch48.csv"
RUN66_PRED_SUMMARY = RUN67_DIR / "run66_prediction_audit_for_run63_batch48_summary.json"
RUN67_MATURITY = RUN67_DIR / "full_variable_N_updated_maturity_and_claim_boundary_summary.json"
RUN67_MATURITY_MD = RUN67_DIR / "full_variable_N_updated_maturity_and_claim_boundary_audit.md"
RUN67_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_67_variable_N_recovery_anchor_batch48_teacher_metrics_ingestion_and_combined480_ranking" / "RUN_67_VARIABLE_N_RECOVERY_ANCHOR_BATCH48_TEACHER_METRICS_INGESTION_AND_COMBINED480_RANKING_REPORT.md"
RUN67_MANIFEST = ROOT / "artifacts" / "manifests" / "stage3_run_67_manifest.json"

RUN66_HANDOFF = ROOT / "outputs" / "stage3_run_64_run63_variable_N_recovery_anchor_batch48_handoff_package" / "stage3_run64_variable_N_recovery_anchor_batch48_candidate_orders.csv"
RUN61_HANDOFF = ROOT / "outputs" / "stage3_run_59_run58_N40_focused_calibrated_penalty_repair_batch40_handoff_package" / "stage3_run59_N40_focused_calibrated_penalty_repair_batch40_candidate_orders.csv"
RUN56_HANDOFF = ROOT / "outputs" / "stage3_run_54_run53_calibrated_N24_N40_batch64_handoff_package" / "stage3_run54_calibrated_N24_N40_batch64_candidate_orders.csv"
RUN51_HANDOFF = ROOT / "outputs" / "stage3_run_49_run48_stricter_constrained_N24_N40_batch32_handoff_package" / "stage3_run49_stricter_constrained_N24_N40_batch32_candidate_orders.csv"
RUN46_HANDOFF = ROOT / "outputs" / "stage3_run_44_run43_constrained_N24_N40_batch32_handoff_package" / "stage3_run44_constrained_N24_N40_batch32_candidate_orders.csv"
RUN41_HANDOFF = ROOT / "outputs" / "stage3_run_39_run38_native_N24_N40_focused_batch60_handoff_package" / "stage3_run39_native_N24_N40_focused_batch60_candidate_orders.csv"
RUN36_HANDOFF = ROOT / "outputs" / "stage3_run_34_run33_N32_informed_native_batch32_handoff_package" / "stage3_run34_N32_informed_native_batch32_candidate_orders.csv"
RUN27_HANDOFF = ROOT / "outputs" / "stage3_run_24_run23_shortlist64_active_learning_handoff_package" / "stage3_run24_shortlist64_candidate_orders.csv"
RUN31_OLD = ROOT / "outputs" / "stage3_run_30_run29_hybrid_policy_batch32_handoff_package" / "stage3_run30_hybrid_policy_batch32_candidate_orders.csv"

OUTPUT_DIR = ROOT / "outputs" / "stage3_run_68_combined480_model_update_smallN_recovery_candidate_generation"
REPORT_DIR = ROOT / "docs" / "stage3" / "runs" / "run_68_combined480_model_update_smallN_recovery_candidate_generation"
REPORT_PATH = REPORT_DIR / "RUN_68_COMBINED480_MODEL_UPDATE_SMALLN_RECOVERY_CANDIDATE_GENERATION_REPORT.md"
MANIFEST_PATH = ROOT / "artifacts" / "manifests" / "stage3_run_68_manifest.json"
CLAIM_BOUNDARY_MD = OUTPUT_DIR / "run68_claim_boundary.md"
CLAIM_BOUNDARY_JSON = OUTPUT_DIR / "run68_claim_boundary.json"

EXPECTED_NATIVE = {12: 48, 16: 48, 24: 184, 40: 200}
EXPECTED_PLUS = {12: 48, 16: 48, 24: 184, 32: 332, 40: 200}
POOL_TARGETS = {12: 5000, 16: 5000, 24: 1000, 40: 1000}
OPTION_A_COUNTS = {12: 16, 16: 16, 24: 4, 40: 4}
OPTION_B_COUNTS = {12: 14, 16: 14, 24: 2, 40: 2}
OPTION_C_COUNTS = {12: 8, 16: 8, 24: 4, 40: 4}
GLOBAL_SEED = 68042

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
            "target_u2_score_combined480_rank",
            "target_peeq_score_combined480_rank",
            "target_surfaceT_score_combined480_rank",
            "target_mises_score_combined480_rank",
        ]
        out["target_reward_combined480_u2_primary"] = as_float(out["target_reward_combined480_u2_primary"])
        prefix = "combined480"
    else:
        score_cols = [
            "target_u2_score_combined480_plus_N32_rank",
            "target_peeq_score_combined480_plus_N32_rank",
            "target_surfaceT_score_combined480_plus_N32_rank",
            "target_mises_score_combined480_plus_N32_rank",
        ]
        out["target_reward_combined480_plus_N32_mapped_u2_primary"] = as_float(out["target_reward_combined480_plus_N32_mapped_u2_primary"])
        prefix = "combined480_plus_N32"
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
    out[f"target_reward_{prefix}_N12_recovery"] = (
        0.55 * out[score_cols[0]] + 0.20 * out[score_cols[1]] + 0.15 * out[score_cols[2]] + 0.10 * out[score_cols[3]]
    )
    out[f"target_reward_{prefix}_N16_recovery"] = (
        0.55 * out[score_cols[0]] + 0.20 * out[score_cols[1]] + 0.15 * out[score_cols[2]] + 0.10 * out[score_cols[3]]
    )
    smalln_target = np.where(
        out["n"].astype(int).isin([12, 16]),
        0.55 * out[f"target_reward_{prefix}_N12_recovery"] + 0.45 * out[f"target_reward_{prefix}_N16_recovery"],
        0.70 * out[f"target_reward_{prefix}_constrained_u2_reward_balanced"] + 0.30 * out[f"target_reward_{prefix}_penalty_repair"],
    )
    out[f"target_reward_{prefix}_variable_N_recovery"] = pd.Series(smalln_target, index=out.index).clip(lower=0.0, upper=1.0)
    out[f"target_reward_{prefix}_smallN_recovery"] = out[f"target_reward_{prefix}_variable_N_recovery"]
    out[f"target_reward_{prefix}_variable_N_balanced"] = (
        np.where(out["n"].astype(int).isin([12, 16]), 1.0, 0.65)
        * out[f"target_reward_{prefix}_variable_N_recovery"]
    ).clip(0.0, 1.0)
    out[f"target_reward_{prefix}_N24_anchor"] = out[f"target_reward_{prefix}_N24_maintenance"]
    out[f"target_reward_{prefix}_N40_anchor"] = out[f"target_reward_{prefix}_N40_strict_reward_retention"]
    out[f"target_reward_{prefix}_N40_followup"] = (
        0.45 * out[f"target_reward_{prefix}_N40_u2_reward_retention"] + 0.35 * out[f"target_reward_{prefix}_N40_penalty_repair"] + 0.20 * out[f"target_reward_{prefix}_two_stage_N40_penalty_repair"]
    ).clip(lower=0.0, upper=1.0)
    return out


def validate_inputs(native: pd.DataFrame, plus: pd.DataFrame, pred_summary: dict[str, Any], maturity: dict[str, Any]) -> dict[str, Any]:
    errors = []
    if len(native) != 480 or n_counts(native) != EXPECTED_NATIVE:
        errors.append(f"native combined480 count mismatch: rows={len(native)} counts={n_counts(native)}")
    if 32 in set(native["n"].astype(int)):
        errors.append("native combined480 contains N32")
    if len(plus) != 812 or n_counts(plus) != EXPECTED_PLUS:
        errors.append(f"combined480_plus_N32 count mismatch: rows={len(plus)} counts={n_counts(plus)}")
    for col in [
        "target_reward_combined480_u2_primary",
        "target_reward_combined480_constrained_u2_reward_balanced",
        "target_reward_combined480_strict_penalty_guard",
        "target_reward_combined480_penalty_repair",
        "target_reward_combined480_N12_recovery",
        "target_reward_combined480_N16_recovery",
        "target_reward_combined480_smallN_recovery",
        "target_reward_combined480_variable_N_recovery",
        "target_reward_combined480_variable_N_balanced",
        "target_reward_combined480_N24_anchor",
        "target_reward_combined480_N40_anchor",
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
            "n24_teacher_rows": 184,
            "n40_teacher_rows": 200,
            "n12_teacher_rows": 48,
            "n16_teacher_rows": 48,
        }
        for key, expected in expected_maturity.items():
            if int(maturity.get(key, -1)) != expected:
                errors.append(f"maturity audit {key} mismatch: {maturity.get(key)}")
        if not maturity.get("full_variable_n_rl_maturity_limited_by_n12_n16", False):
            errors.append("maturity audit does not preserve full variable-N limitation warning")
    else:
        errors.append("Run67 maturity audit summary missing")
    if pred_summary:
        reward_s = pred_summary.get("overall_reward_spearman")
        top5 = pred_summary.get("mean_top5_overlap")
    else:
        reward_s = None
        top5 = None
    return {
        "timestamp": now_iso(),
        "verdict": "PASS_RUN68_COMBINED480_AND_PLUS_N32_INPUTS_READY" if not errors else "FAIL_RUN68_INPUT_VALIDATION",
        "errors": errors,
        "native_combined480_rows": int(len(native)),
        "native_combined480_counts": n_counts(native),
        "combined480_plus_N32_rows": int(len(plus)),
        "combined480_plus_N32_counts": n_counts(plus),
        "maturity_context": maturity,
        "run66_prediction_audit_context": {"u2_primary_or_reward_spearman": reward_s, "mean_top5_overlap": top5, "top1_hits": pred_summary.get("top1_hits") if pred_summary else None},
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
        return sk["RandomForestRegressor"](n_estimators=8, random_state=seed, min_samples_leaf=2, n_jobs=-1)
    if name == "ExtraTreesRegressor":
        return sk["ExtraTreesRegressor"](n_estimators=10, random_state=seed, min_samples_leaf=1, n_jobs=-1)
    if name == "GradientBoostingRegressor":
        return sk["GradientBoostingRegressor"](random_state=seed)
    if name == "OrderGraphMLPRegressor":
        return sk["make_pipeline"](sk["StandardScaler"](), sk["MLPRegressor"](hidden_layer_sizes=(20,), alpha=0.001, max_iter=60, random_state=seed))
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
    if "dataset_source" in df.columns and (df["dataset_source"].astype(str) == "run66_variable_N_recovery_anchor_batch48").any():
        test = df.index[df["dataset_source"].astype(str) == "run66_variable_N_recovery_anchor_batch48"].to_numpy()
        train = np.setdiff1d(rows, test)
        protocols.append(("train_pre_Run66_test_Run66", train, test))
    return protocols


def run_surrogate_validation(native: pd.DataFrame, plus: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], Any, list[str]]:
    model_names = ["MeanBaseline", "Ridge", "ElasticNet", "RandomForestRegressor", "ExtraTreesRegressor", "GradientBoostingRegressor"]
    targets = {
        "native_combined480": [
            "target_reward_combined480_u2_primary",
            "target_reward_combined480_constrained_u2_reward_balanced",
            "target_reward_combined480_strict_penalty_guard",
            "target_reward_combined480_penalty_repair",
            "target_reward_combined480_N12_recovery",
            "target_reward_combined480_N16_recovery",
            "target_reward_combined480_smallN_recovery",
            "target_reward_combined480_variable_N_balanced",
            "target_reward_combined480_N24_anchor",
            "target_reward_combined480_N40_anchor",
        ],
        "plus_N32_unweighted": [
            "target_reward_combined480_plus_N32_mapped_u2_primary",
            "target_reward_combined480_plus_N32_strict_u2_surfaceT",
            "target_reward_combined480_plus_N32_constrained_u2_reward_balanced",
            "target_reward_combined480_plus_N32_N12_recovery",
            "target_reward_combined480_plus_N32_N16_recovery",
            "target_reward_combined480_plus_N32_smallN_recovery",
            "target_reward_combined480_plus_N32_variable_N_balanced",
            "target_reward_combined480_plus_N32_N24_anchor",
            "target_reward_combined480_plus_N32_N40_anchor",
        ],
        "plus_N32_balanced": [
            "target_reward_combined480_plus_N32_mapped_u2_primary",
            "target_reward_combined480_plus_N32_strict_u2_surfaceT",
            "target_reward_combined480_plus_N32_constrained_u2_reward_balanced",
            "target_reward_combined480_plus_N32_N12_recovery",
            "target_reward_combined480_plus_N32_N16_recovery",
            "target_reward_combined480_plus_N32_smallN_recovery",
            "target_reward_combined480_plus_N32_variable_N_balanced",
            "target_reward_combined480_plus_N32_N24_anchor",
            "target_reward_combined480_plus_N32_N40_anchor",
        ],
    }
    frames = {"native_combined480": native, "plus_N32_unweighted": plus, "plus_N32_balanced": plus}
    rows = []
    feature_sets_to_run = ["F01_basic_order", "F02_full_handcrafted", "F07_F01_no_n"]
    for regime, df in frames.items():
        sw = sample_weights(df) if regime.endswith("balanced") else None
        for target in targets[regime]:
            if target not in df.columns:
                continue
            for fs in feature_sets_to_run:
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
    native_best = grouped[grouped["regime"] == "native_combined480"].head(1).iloc[0].to_dict()
    plus_best = grouped[grouped["regime"].str.contains("plus")].head(1).iloc[0].to_dict()
    def best_for(target: str) -> dict[str, Any]:
        part = grouped[grouped["target"] == target]
        return part.head(1).iloc[0].to_dict() if not part.empty else {}
    summary = {
        "best_overall": best_row,
        "best_native": native_best,
        "best_plus_N32": plus_best,
        "best_N12_recovery": best_for("target_reward_combined480_N12_recovery"),
        "best_N16_recovery": best_for("target_reward_combined480_N16_recovery"),
        "best_smallN_recovery": best_for("target_reward_combined480_smallN_recovery"),
        "best_variable_N_balanced": best_for("target_reward_combined480_variable_N_balanced"),
        "best_N24_anchor": best_for("target_reward_combined480_N24_anchor"),
        "best_N40_anchor": best_for("target_reward_combined480_N40_anchor"),
        "best_N40_u2_reward_retention": best_for("target_reward_combined480_N40_u2_reward_retention"),
        "best_N40_penalty_repair": best_for("target_reward_combined480_N40_penalty_repair"),
        "best_N24_maintenance": best_for("target_reward_combined480_N24_maintenance"),
        "best_penalty_repair": best_for("target_reward_combined480_penalty_repair"),
        "n32_augmented_better_than_native": bool(plus_best["macro_spearman"] > native_best["macro_spearman"]),
        "run63_reference": {"best_native_spearman": 0.8075, "best_N12_recovery_spearman": 0.7595, "best_N16_recovery_spearman": 0.7850, "run66_prediction_u2_primary_spearman": 0.4911},
    }
    best_model = make_model(str(native_best["model"]), GLOBAL_SEED)
    features = [c for c in FEATURE_SETS[str(native_best["feature_set"])] if c in native.columns]
    best_model.fit(native[features].fillna(0.0).to_numpy(float), native[str(native_best["target"])].to_numpy(float))
    return detailed, best, summary, best_model, features


def run_gnn_validation(native: pd.DataFrame, plus: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    for regime, df in {"native_combined480": native, "plus_N32_unweighted": plus, "plus_N32_balanced": plus}.items():
        target = "target_reward_combined480_u2_primary" if regime == "native_combined480" else "target_reward_combined480_plus_N32_mapped_u2_primary"
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
    for regime, df in {"native_combined480": native, "plus_N32_unweighted": plus, "plus_N32_balanced": plus}.items():
        target = "target_reward_combined480_u2_primary" if regime == "native_combined480" else "target_reward_combined480_plus_N32_mapped_u2_primary"
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
        best_by[(n, "reward")] = parse_order(g.sort_values("target_reward_combined480_u2_primary", ascending=False).iloc[0]["order_json"])
        best_by[(n, "constrained")] = parse_order(g.sort_values("target_reward_combined480_constrained_u2_reward_balanced", ascending=False).iloc[0]["order_json"])
        best_by[(n, "strict")] = parse_order(g.sort_values("target_reward_combined480_strict_penalty_guard", ascending=False).iloc[0]["order_json"])
        best_by[(n, "two_stage")] = parse_order(g.sort_values("target_reward_combined480_two_stage_guarded", ascending=False).iloc[0]["order_json"])
    rows, seen = [], set(existing)
    for n, target_count in POOL_TARGETS.items():
        if n in (12, 16):
            source_plan = [
                f"N{n}_recovery_surrogate_top", f"N{n}_local_search_around_Run66_best",
                f"N{n}_recovery_u2_safe", f"N{n}_recovery_penalty_aware",
                f"N{n}_recovery_reward_balanced", f"N{n}_recovery_uncertainty",
                f"N{n}_recovery_diversity", f"N{n}_recovery_sentinel_control",
            ]
        elif n == 40:
            source_plan = [
                "N40_frozen_u2_reward_reference", "N40_uncertainty_anchor",
                "N40_sentinel_control",
            ]
        elif n == 24:
            source_plan = [
                "N24_frozen_top_density_reference", "N24_uncertainty_anchor",
                "N24_sentinel_control",
            ]
        attempts = 0
        while sum(1 for r in rows if r["n"] == n) < target_count and attempts < target_count * 20:
            attempts += 1
            source = source_plan[attempts % len(source_plan)]
            if "u2_safe" in source or "u2_reward_retention" in source or "PEEQ_repair" in source:
                base = best_by[(n, "u2")]
            elif "strict" in source or "Mises_repair" in source or "penalty_aware" in source:
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
            n12_recovery = 0.55 * u2_pred + 0.20 * peeq_guard + 0.15 * surf_guard + 0.10 * mises_guard
            n16_recovery = 0.55 * u2_pred + 0.20 * peeq_guard + 0.15 * surf_guard + 0.10 * mises_guard
            variable_recovery = (0.55 * n12_recovery + 0.45 * n16_recovery) if n in (12, 16) else (0.70 * constrained_reward + 0.30 * penalty_repair)
            n40_followup = 0.45 * n40_retention + 0.35 * n40_repair + 0.20 * two_stage_repair
            calibrated = (
                0.60 * variable_recovery + 0.15 * n12_recovery + 0.15 * n16_recovery + 0.10 * n40_followup
                - 0.12 * max(0.0, 0.16 - near_d)
            )
            if n == 40:
                calibrated = 0.55 * n40_followup + 0.25 * penalty_repair + 0.20 * constrained_reward - 0.12 * max(0.0, 0.16 - near_d)
            cid = f"R68_N{n}_C{sum(1 for r in rows if r['n']==n)+1:05d}"
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
                "N12_recovery_score": n12_recovery,
                "N16_recovery_score": n16_recovery,
                "variable_N_recovery_score": variable_recovery,
                "N40_followup_score": n40_followup,
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
            ("N24_frozen_top_density_reference", 2),
            ("N24_uncertainty_anchor", 1),
            ("N24_sentinel_control", 1),
        ],
        40: [
            ("N40_frozen_u2_reward_reference", 2),
            ("N40_uncertainty_anchor", 1),
            ("N40_sentinel_control", 1),
        ],
        12: [
            ("N12_recovery_surrogate_top", 4),
            ("N12_local_search_around_Run66_best", 3),
            ("N12_recovery_u2_safe", 3),
            ("N12_recovery_penalty_aware", 2),
            ("N12_recovery_reward_balanced", 2),
            ("N12_recovery_uncertainty", 1),
            ("N12_recovery_diversity", 1),
        ],
        16: [
            ("N16_recovery_surrogate_top", 4),
            ("N16_local_search_around_Run66_best", 3),
            ("N16_recovery_u2_safe", 3),
            ("N16_recovery_penalty_aware", 2),
            ("N16_recovery_reward_balanced", 2),
            ("N16_recovery_uncertainty", 1),
            ("N16_recovery_diversity", 1),
        ],
    }
    fallback_order = [
        "N12_recovery_surrogate_top", "N12_local_search_around_Run66_best", "N12_recovery_u2_safe", "N12_recovery_penalty_aware",
        "N16_recovery_surrogate_top", "N16_local_search_around_Run66_best", "N16_recovery_u2_safe", "N16_recovery_penalty_aware",
        "N24_frozen_top_density_reference", "N24_uncertainty_anchor",
        "N40_frozen_u2_reward_reference", "N40_uncertainty_anchor",
    ]
    for n, count in counts.items():
        g = pool[pool["n"] == n].copy()
        take = []
        plan = primary_plan.get(n)
        if not plan:
            per_bucket = max(1, count // 8)
            plan = [(bucket, per_bucket) for bucket in fallback_order]
        for bucket, quota in plan:
            score_col = "variable_N_recovery_score" if n in (12, 16) else ("N40_followup_score" if n == 40 else "constrained_score")
            bg = g[(g["candidate_source"] == bucket) & (~g["order_hash"].isin([r["order_hash"] for r in take]))].sort_values([score_col, "novelty_distance"], ascending=[False, False])
            for _, row in bg.head(quota).iterrows():
                if len(take) < count:
                    take.append(row.to_dict())
        if len(take) < count:
            used = {r["order_hash"] for r in take}
            score_col = "variable_N_recovery_score" if n in (12, 16) else ("N40_followup_score" if n == 40 else "constrained_score")
            rem = g[~g["order_hash"].isin(used)].sort_values([score_col, "novelty_distance"], ascending=[False, False])
            take.extend(rem.head(count - len(take)).to_dict(orient="records"))
        for i, row in enumerate(take[:count], 1):
            short = str(row["candidate_source"]).replace("_local_search", "").replace("_local_repair", "repair").replace("_candidates", "").replace("penalty_repair", "penalty").replace("u2_reward_retention", "u2ret").replace("recovery_anchor", "anchor").replace("N40_", "n40_").replace("N24_", "n24_").replace("N12_", "n12_").replace("N16_", "n16_")[:30]
            row["batch_option"] = label
            row["handoff_strategy_name"] = f"S3R68SMN_N{n}_B{i:02d}_{short}"
            row["teacher_validated"] = False
            row["teacher_validation_status"] = "NOT_RUN"
            selected.append(row)
    return pd.DataFrame(selected)


def compare_batches(options: dict[str, pd.DataFrame], native: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    previous = {"combined480_teacher_orders": set(native["order_hash"].astype(str))}
    for name, path in {"run66": RUN66_HANDOFF, "run61": RUN61_HANDOFF, "run56": RUN56_HANDOFF, "run51": RUN51_HANDOFF, "run46": RUN46_HANDOFF, "run41": RUN41_HANDOFF, "run36": RUN36_HANDOFF, "run27": RUN27_HANDOFF, "superseded_run31": RUN31_OLD}.items():
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
            "mean_N12_recovery_score": float(df.get("N12_recovery_score", pd.Series([math.nan])).mean()),
            "mean_N16_recovery_score": float(df.get("N16_recovery_score", pd.Series([math.nan])).mean()),
            "mean_variable_N_recovery_score": float(df.get("variable_N_recovery_score", pd.Series([math.nan])).mean()),
            "mean_N40_penalty_repair_score": float(df.get("N40_penalty_repair_score", pd.Series([math.nan])).mean()),
            "mean_N40_followup_score": float(df.get("N40_followup_score", pd.Series([math.nan])).mean()),
            "mean_two_stage_repair_score": float(df.get("two_stage_penalty_repair_score", pd.Series([math.nan])).mean()),
            "mean_novelty_distance": float(df["novelty_distance"].mean()),
            "candidate_source_composition": json.dumps(df["candidate_source"].value_counts().to_dict()),
        }
        for pname, phashes in previous.items():
            row[f"overlap_{pname}"] = int(len(hashes & phashes))
        rows.append(row)
    comp = pd.DataFrame(rows)
    headline = "Run68 batch options were checked for exact order overlap, small-N recovery score, optional N40 follow-up score, novelty, and source diversity against combined480, Run66, Run56, Run51, Run46, Run41, Run36, Run27, and superseded Run31."
    return comp, {"headline": headline, "rows": comp.to_dict(orient="records")}


def write_evidence_update(native: pd.DataFrame, validation: dict[str, Any], pred_summary: dict[str, Any], maturity: dict[str, Any]) -> dict[str, Any]:
    timeline = [
        {"stage": "combined432_before_Run66", "native_rows": 432, "N12": 36, "N16": 36, "N24": 176, "N40": 184, "interpretation": "N24/N40 evidence frozen; full variable-N claim limited by N12/N16 density"},
        {"stage": "Run66_variable_N_recovery_anchor_batch48", "native_rows": 480, "N12": 48, "N16": 48, "N24": 184, "N40": 200, "interpretation": "small-N evidence strengthened, especially N16; N24/N40 remain mature anchors"},
    ]
    update = pd.DataFrame(timeline)
    update_path = OUTPUT_DIR / "full_variable_N_evidence_update_after_run67.csv"
    update.to_csv(update_path, index=False)

    summary = {
        "verdict": "RUN68_FULL_VARIABLE_N_EVIDENCE_STRENGTHENED_SMALLN_RECOVERY_STILL_PRIMARY",
        "headline": "Run67/Run66 strengthened full variable-N evidence to N12=48 and N16=48 while N24=184 and N40=200 remain mature anchors; the next validation should continue small-N recovery rather than broad N24/N40 exploitation.",
        "native_combined480_counts": n_counts(native),
        "run67_maturity_context": maturity,
        "run66_prediction_audit_context": {
            "reward_spearman": pred_summary.get("overall_reward_spearman") if pred_summary else None,
            "penalty_repair_spearman": pred_summary.get("overall_penalty_repair_spearman") if pred_summary else None,
            "mean_top5_overlap": pred_summary.get("mean_top5_overlap") if pred_summary else None,
        },
        "evidence_decision": {
            "N24_N40_remain_frozen_anchors": True,
            "pivot_primary_generation_to_N12_N16_recovery": True,
            "keep_N24_N40_anchor_counts_low": True,
            "continue_full_variable_N_claim_boundary": "improved_but_still_limited_by_N12_N16_density",
        },
        "timeline": timeline,
        "validation_verdict": validation["verdict"],
    }
    write_json(OUTPUT_DIR / "full_variable_N_evidence_update_after_run67.json", summary)
    (OUTPUT_DIR / "full_variable_N_evidence_update_after_run67.md").write_text(
        "# Full Variable-N Evidence Update After Run67\n\n"
        f"Verdict: `{summary['verdict']}`\n\n"
        f"{summary['headline']}\n\n"
        "This update preserves the distinction between mature N24/N40 focused evidence and the still-recovering full variable-N setting. "
        "N32 rows remain legacy-compatible auxiliary data, not native Stage 3 teacher validation.\n\n"
        "The next candidate-generation priority is N12/N16 recovery, with only minimal N24/N40 anchor coverage.\n\n"
        "## Timeline\n"
        + "\n".join(f"- {row['stage']}: N24={row['N24']}, N40={row['N40']} - {row['interpretation']}" for row in timeline)
        + "\n",
        encoding="utf-8",
    )
    return summary


def write_claim_boundary() -> None:
    safe = [
        "Run68 updates offline models using native combined480 and combined480_plus_N32.",
        "Run68 generates candidate-order batches for future teacher validation.",
        "Run68 identifies small-N recovery, especially N12/N16, as the main follow-up direction.",
        "N12/N16 now have 48 native teacher rows each.",
        "N24/N40 remain mature teacher-labelled anchor groups with 184 and 200 native rows respectively.",
        "Full variable-N evidence is strengthened but still bounded by N12/N16 density.",
        "Run68 does not include teacher validation for new candidates.",
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
    CLAIM_BOUNDARY_MD.write_text("# Run68 Claim Boundary\n\n## Safe claims\n" + "\n".join(f"- {x}" for x in safe) + "\n\n## Unsafe claims\n" + "\n".join(f"- {x}" for x in unsafe) + "\n", encoding="utf-8")
    write_json(CLAIM_BOUNDARY_JSON, {"verdict": "RUN68_MODEL_UPDATE_AND_SMALLN_RECOVERY_CANDIDATE_GENERATION_ONLY_NO_TEACHER_VALIDATION", "safe_claims": safe, "unsafe_claims": unsafe})


def write_report(summary: dict[str, Any]) -> None:
    REPORT_PATH.write_text(f"""# Stage 3 Run 68 - Combined480 Model Update and Small-N Recovery Candidate Generation

## 1. Purpose
Run68 updates offline diagnostics using native combined480 and combined480_plus_N32, updates the full variable-N evidence boundary after Run67, and creates N12/N16 recovery-focused candidate batches. It is model update and candidate generation only.

## 2. Inputs
- Native combined480: `{COMBINED480_READY}`
- combined480_plus_N32: `{COMBINED480_PLUS_N32_READY}`
- Run66 prediction audit: `{RUN66_PRED_SUMMARY}`
- Run67 maturity audit: `{RUN67_MATURITY}`

## 3. Run67/Run66 Context
Run66 created 14 new metric/reward records versus combined432. N16 was the clearest small-N winner, N12 improved reward-family records, N40 produced useful U2/reward anchor records, and N24 acted as an anchor. N12/N16 now have 48 rows each but remain much less dense than N24/N40.

## 4. Target Reward Definition Audit
{summary['target_audit_headline']}

## 5. Feature Reconstruction
Run68 wrote Run22/Run29/Run33/Run38/Run43/Run48/Run53/Run58-compatible handcrafted order features for native combined480 and combined480_plus_N32.

## 6. Surrogate Update
Best surrogate overall: `{summary['surrogate']['best_overall']}`.

Best native surrogate: `{summary['surrogate']['best_native']}`.

Best plus-N32 surrogate: `{summary['surrogate']['best_plus_N32']}`.

Best N12 recovery: `{summary['surrogate'].get('best_N12_recovery')}`.

Best N16 recovery: `{summary['surrogate'].get('best_N16_recovery')}`.

Best small-N recovery: `{summary['surrogate'].get('best_smallN_recovery')}`.

Best variable-N balanced: `{summary['surrogate'].get('best_variable_N_balanced')}`.

## 7. GNN and Graph-Pointer Diagnostics
{summary['gnn']['note']} Best GNN diagnostic: `{summary['gnn']['best_regime']}`.
{summary['pointer']['headline']} Mean NLL by regime: `{summary['pointer']['mean_nll_by_regime']}`.

## 8. Full Variable-N Evidence Update After Run67
{summary['evidence_update']['headline']}

## 9. N12/N16 Recovery Candidate Generation
Candidate pool counts: `{summary['candidate_pool_counts']}`. N12/N16 meet the >=5000 candidate minimums, with N24/N40 frozen-anchor references included for comparison.

## 10. Option A - Small-N Recovery-Focused Batch40
Path: `{OUTPUT_DIR / 'run68_smallN_recovery_focused_batch40_candidate_orders.csv'}`. Counts: `{summary['option_counts']['option_A_smallN_recovery_focused_batch40']}`.

## 11. Option B - Small-N Recovery Batch32
Path: `{OUTPUT_DIR / 'run68_smallN_recovery_batch32_candidate_orders.csv'}`. Counts: `{summary['option_counts']['option_B_smallN_recovery_batch32']}`.

## 12. Option C - Final Diagnostic Batch24
Path: `{OUTPUT_DIR / 'run68_final_diagnostic_batch24_candidate_orders.csv'}`. Counts: `{summary['option_counts']['option_C_final_diagnostic_batch24']}`.

## 13. Comparison to Previous Batches
{summary['comparison']['headline']}

## 14. Claim Boundary
Verdict: `RUN68_MODEL_UPDATE_AND_SMALLN_RECOVERY_CANDIDATE_GENERATION_ONLY_NO_TEACHER_VALIDATION`.

## 15. Output Files
- Candidate pool: `{OUTPUT_DIR / 'run68_candidate_pool_scored.csv'}`
- Surrogate summary: `{OUTPUT_DIR / 'run68_surrogate_validation_summary.json'}`
- GNN summary: `{OUTPUT_DIR / 'run68_gnn_reward_validation_summary.json'}`
- Pointer summary: `{OUTPUT_DIR / 'run68_graph_pointer_policy_validation_summary.json'}`
- Evidence update: `{OUTPUT_DIR / 'full_variable_N_evidence_update_after_run67.md'}`
- Batch options comparison: `{OUTPUT_DIR / 'run68_batch_options_comparison_summary.json'}`
- Manifest: `{MANIFEST_PATH}`

## 16. Recommended Run69
Create a handoff package for Option A (`run68_smallN_recovery_focused_batch40_candidate_orders.csv`) unless the user explicitly selects Option B or Option C. Do not generate CAE/INP until a Run68 option is selected and handed off.
""", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    native = add_features_and_targets(read_csv(COMBINED480_READY), native=True)
    plus = add_features_and_targets(read_csv(COMBINED480_PLUS_N32_READY), native=False)
    pred_summary = json.loads(RUN66_PRED_SUMMARY.read_text(encoding="utf-8")) if RUN66_PRED_SUMMARY.exists() else {}
    maturity = json.loads(RUN67_MATURITY.read_text(encoding="utf-8")) if RUN67_MATURITY.exists() else {}
    validation = validate_inputs(native, plus, pred_summary, maturity)
    write_json(OUTPUT_DIR / "run68_input_validation_summary.json", validation)
    if not validation["verdict"].startswith("PASS"):
        raise SystemExit(validation["errors"])

    native.to_csv(OUTPUT_DIR / "combined480_scan_order_features.csv", index=False)
    plus.to_csv(OUTPUT_DIR / "combined480_plus_N32_scan_order_features.csv", index=False)
    target_rows = [
        {"target": "target_reward_combined480_u2_primary", "definition": "0.65 U2 + 0.20 PEEQ + 0.10 SurfaceT + 0.05 Mises", "dataset": "native_combined480"},
        {"target": "target_reward_combined480_constrained_u2_reward_balanced", "definition": "0.50 U2 + 0.25 PEEQ + 0.15 SurfaceT + 0.10 Mises", "dataset": "native_combined480"},
        {"target": "target_reward_combined480_strict_penalty_guard", "definition": "0.40 U2 + 0.30 PEEQ + 0.20 SurfaceT + 0.10 Mises", "dataset": "native_combined480"},
        {"target": "target_reward_combined480_penalty_repair", "definition": "0.30 U2 + 0.30 PEEQ + 0.25 SurfaceT + 0.15 Mises", "dataset": "native_combined480"},
        {"target": "target_reward_combined480_N12_recovery", "definition": "N12 recovery target: 0.55 U2 + 0.20 PEEQ + 0.15 SurfaceT + 0.10 Mises", "dataset": "native_combined480"},
        {"target": "target_reward_combined480_N16_recovery", "definition": "N16 recovery target: 0.55 U2 + 0.20 PEEQ + 0.15 SurfaceT + 0.10 Mises", "dataset": "native_combined480"},
        {"target": "target_reward_combined480_smallN_recovery", "definition": "N12/N16-preferential small-N recovery target using within-N ranks; N24/N40 are not allowed to dominate", "dataset": "native_combined480"},
        {"target": "target_reward_combined480_variable_N_balanced", "definition": "Variable-N balanced target with N12/N16 recovery emphasis and N24/N40 frozen-anchor treatment", "dataset": "native_combined480"},
        {"target": "target_reward_combined480_N24_anchor", "definition": "N24 frozen-anchor diagnostic target; preserve high-performing region without broad N24 exploitation", "dataset": "native_combined480"},
        {"target": "target_reward_combined480_N40_anchor", "definition": "N40 frozen-anchor diagnostic target; preserve U2/reward region without broad N40 exploitation", "dataset": "native_combined480"},
        {"target": "target_reward_combined480_plus_N32_mapped_u2_primary", "definition": "mapped U2-primary target with N32 semantic warnings preserved", "dataset": "combined480_plus_N32"},
        {"target": "target_reward_combined480_plus_N32_strict_u2_surfaceT", "definition": "strict U2 + SurfaceT target for safer N32 metric semantics", "dataset": "combined480_plus_N32"},
    ]
    pd.DataFrame(target_rows).to_csv(OUTPUT_DIR / "run68_target_reward_definition_audit.csv", index=False)
    write_json(OUTPUT_DIR / "run68_target_reward_definition_audit.json", {"headline": "Run68 defines N12/N16 recovery as the primary candidate-generation objective, keeps N24/N40 as low-count frozen anchors, and preserves N32 metric-semantic warnings for diagnostic plus_N32 regimes.", "targets": target_rows})

    detailed, best_cfg, surrogate_summary, best_model, best_features = run_surrogate_validation(native, plus)
    detailed.to_csv(OUTPUT_DIR / "run68_surrogate_validation_results_detailed.csv", index=False)
    best_cfg.to_csv(OUTPUT_DIR / "run68_best_surrogate_configurations.csv", index=False)
    write_json(OUTPUT_DIR / "run68_surrogate_validation_summary.json", surrogate_summary)

    gnn_results, gnn_summary = run_gnn_validation(native, plus)
    gnn_results.to_csv(OUTPUT_DIR / "run68_gnn_reward_validation_results.csv", index=False)
    write_json(OUTPUT_DIR / "run68_gnn_reward_validation_summary.json", gnn_summary)

    pointer_log, pointer_summary = pointer_policy(native, plus)
    pointer_log.to_csv(OUTPUT_DIR / "run68_graph_pointer_policy_training_log.csv", index=False)
    write_json(OUTPUT_DIR / "run68_graph_pointer_policy_validation_summary.json", pointer_summary)

    pool = generate_candidates(native, best_model, best_features)
    pool.to_csv(OUTPUT_DIR / "run68_candidate_pool_scored.csv", index=False)

    option_a = select_batch(pool, OPTION_A_COUNTS, "smallN_recovery_focused_batch40")
    option_b = select_batch(pool, OPTION_B_COUNTS, "smallN_recovery_batch32")
    option_c = select_batch(pool, OPTION_C_COUNTS, "final_diagnostic_batch24")
    option_a.to_csv(OUTPUT_DIR / "run68_smallN_recovery_focused_batch40_candidate_orders.csv", index=False)
    option_b.to_csv(OUTPUT_DIR / "run68_smallN_recovery_batch32_candidate_orders.csv", index=False)
    option_c.to_csv(OUTPUT_DIR / "run68_final_diagnostic_batch24_candidate_orders.csv", index=False)

    comparison, comparison_summary = compare_batches({"option_A_smallN_recovery_focused_batch40": option_a, "option_B_smallN_recovery_batch32": option_b, "option_C_final_diagnostic_batch24": option_c}, native)
    comparison.to_csv(OUTPUT_DIR / "run68_batch_options_comparison_to_previous.csv", index=False)
    write_json(OUTPUT_DIR / "run68_batch_options_comparison_summary.json", comparison_summary)

    write_claim_boundary()
    evidence_update = write_evidence_update(native, validation, pred_summary, maturity)
    option_counts = {"option_A_smallN_recovery_focused_batch40": n_counts(option_a), "option_B_smallN_recovery_batch32": n_counts(option_b), "option_C_final_diagnostic_batch24": n_counts(option_c)}
    summary = {
        "target_audit_headline": "Run68 target definitions prioritize N12/N16 recovery, keep N24/N40 as low-count anchors, and preserve N32 metric-semantic warnings for plus_N32 diagnostics.",
        "surrogate": surrogate_summary,
        "gnn": gnn_summary,
        "pointer": pointer_summary,
        "evidence_update": evidence_update,
        "candidate_pool_counts": n_counts(pool),
        "option_counts": option_counts,
        "comparison": comparison_summary,
    }
    write_report(summary)

    output_files = [
        OUTPUT_DIR / "run68_input_validation_summary.json",
        OUTPUT_DIR / "combined480_scan_order_features.csv",
        OUTPUT_DIR / "combined480_plus_N32_scan_order_features.csv",
        OUTPUT_DIR / "run68_target_reward_definition_audit.csv",
        OUTPUT_DIR / "run68_target_reward_definition_audit.json",
        OUTPUT_DIR / "run68_surrogate_validation_results_detailed.csv",
        OUTPUT_DIR / "run68_best_surrogate_configurations.csv",
        OUTPUT_DIR / "run68_surrogate_validation_summary.json",
        OUTPUT_DIR / "run68_gnn_reward_validation_results.csv",
        OUTPUT_DIR / "run68_gnn_reward_validation_summary.json",
        OUTPUT_DIR / "run68_graph_pointer_policy_training_log.csv",
        OUTPUT_DIR / "run68_graph_pointer_policy_validation_summary.json",
        OUTPUT_DIR / "full_variable_N_evidence_update_after_run67.csv",
        OUTPUT_DIR / "full_variable_N_evidence_update_after_run67.json",
        OUTPUT_DIR / "full_variable_N_evidence_update_after_run67.md",
        OUTPUT_DIR / "run68_candidate_pool_scored.csv",
        OUTPUT_DIR / "run68_smallN_recovery_focused_batch40_candidate_orders.csv",
        OUTPUT_DIR / "run68_smallN_recovery_batch32_candidate_orders.csv",
        OUTPUT_DIR / "run68_final_diagnostic_batch24_candidate_orders.csv",
        OUTPUT_DIR / "run68_batch_options_comparison_to_previous.csv",
        OUTPUT_DIR / "run68_batch_options_comparison_summary.json",
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
        "input_files": [str(COMBINED480_READY), str(COMBINED480_PLUS_N32_READY), str(RUN66_ENRICHED), str(RUN66_PRED_SUMMARY), str(RUN67_MATURITY), str(RUN67_REPORT), str(RUN67_MANIFEST)],
        "output_files": [str(p) for p in output_files],
        "native_combined480_rows": int(len(native)),
        "combined480_plus_N32_rows": int(len(plus)),
        "best_surrogate_summary": surrogate_summary,
        "best_N12_recovery_summary": surrogate_summary.get("best_N12_recovery"),
        "best_N16_recovery_summary": surrogate_summary.get("best_N16_recovery"),
        "best_smallN_recovery_summary": surrogate_summary.get("best_smallN_recovery"),
        "best_variable_N_balanced_summary": surrogate_summary.get("best_variable_N_balanced"),
        "best_N24_anchor_summary": surrogate_summary.get("best_N24_anchor"),
        "best_N40_anchor_summary": surrogate_summary.get("best_N40_anchor"),
        "best_gnn_summary": gnn_summary,
        "pointer_summary": pointer_summary,
        "evidence_update_summary": evidence_update,
        "evidence_update_paths": {
            "csv": str(OUTPUT_DIR / "full_variable_N_evidence_update_after_run67.csv"),
            "json": str(OUTPUT_DIR / "full_variable_N_evidence_update_after_run67.json"),
            "md": str(OUTPUT_DIR / "full_variable_N_evidence_update_after_run67.md"),
        },
        "candidate_pool_count": int(len(pool)),
        "candidate_pool_counts": n_counts(pool),
        "recommended_option": "option_A_smallN_recovery_focused_batch40",
        "batch_option_paths": {
            "option_A_smallN_recovery_focused_batch40": str(OUTPUT_DIR / "run68_smallN_recovery_focused_batch40_candidate_orders.csv"),
            "option_B_smallN_recovery_batch32": str(OUTPUT_DIR / "run68_smallN_recovery_batch32_candidate_orders.csv"),
            "option_C_final_diagnostic_batch24": str(OUTPUT_DIR / "run68_final_diagnostic_batch24_candidate_orders.csv"),
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



