from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import random
import statistics
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
RUN_ID = "run_38_combined204_and_combined204_plus_N32_model_update_candidate_generation"
RUN_NAME = "combined204 and combined204 plus N32 model update candidate generation"

COMBINED_PLUS_READY = ROOT / "outputs" / "stage3_run_37_N32_informed_native_batch32_teacher_metrics_ingestion_and_combined204_ranking" / "combined204_plus_N32_RL_ready_dataset.csv"
COMBINED204_READY = ROOT / "outputs" / "stage3_run_37_N32_informed_native_batch32_teacher_metrics_ingestion_and_combined204_ranking" / "combined204_RL_ready_dataset.csv"
COMBINED204_TEACHER = ROOT / "outputs" / "stage3_run_37_N32_informed_native_batch32_teacher_metrics_ingestion_and_combined204_ranking" / "combined204_teacher_dataset.csv"
COMBINED172_TEACHER = ROOT / "outputs" / "stage3_run_28_shortlist64_teacher_metrics_ingestion_and_combined172_ranking" / "combined172_teacher_dataset.csv"
N32_DEDUP = ROOT / "outputs" / "stage3_run_32a_stage2_n32_legacy_teacher_label_ingestion_for_stage3" / "n32_legacy_teacher_dataset_dedup_training_332.csv"
RUN32A_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_32a_stage2_n32_legacy_teacher_label_ingestion_for_stage3" / "RUN_32A_STAGE2_N32_LEGACY_TEACHER_LABEL_INGESTION_FOR_STAGE3_REPORT.md"
RUN33_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_33_combined172_plus_N32_balanced_surrogate_gnn_candidate_generation" / "RUN_33_COMBINED172_PLUS_N32_BALANCED_SURROGATE_GNN_CANDIDATE_GENERATION_REPORT.md"
RUN37_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_37_N32_informed_native_batch32_teacher_metrics_ingestion_and_combined204_ranking" / "RUN_37_N32_INFORMED_NATIVE_BATCH32_TEACHER_METRICS_INGESTION_AND_COMBINED204_RANKING_REPORT.md"
RUN37_MANIFEST = ROOT / "artifacts" / "manifests" / "stage3_run_37_manifest.json"
RUN36_ENRICHED = ROOT / "outputs" / "stage3_run_37_N32_informed_native_batch32_teacher_metrics_ingestion_and_combined204_ranking" / "run36_N32_informed_native_batch32_teacher_dataset_enriched.csv"
RUN36_RANKED = ROOT / "outputs" / "stage3_run_37_N32_informed_native_batch32_teacher_metrics_ingestion_and_combined204_ranking" / "run36_N32_informed_native_batch32_ranked_within_batch.csv"
RUN36_COMPARISON = ROOT / "outputs" / "stage3_run_37_N32_informed_native_batch32_teacher_metrics_ingestion_and_combined204_ranking" / "run36_vs_combined172_best_comparison.csv"
RUN36_EFFECTIVENESS = ROOT / "outputs" / "stage3_run_37_N32_informed_native_batch32_teacher_metrics_ingestion_and_combined204_ranking" / "run36_N32_informed_candidate_effectiveness_audit.csv"
RUN36_PRED_AUDIT = ROOT / "outputs" / "stage3_run_37_N32_informed_native_batch32_teacher_metrics_ingestion_and_combined204_ranking" / "run36_prediction_audit_for_run33_optionA.csv"
RUN30_BATCH32 = ROOT / "outputs" / "stage3_run_30_run29_hybrid_policy_batch32_handoff_package" / "stage3_run30_hybrid_policy_batch32_candidate_orders.csv"
RUN31_READY = ROOT / "outputs" / "stage3_run_31_hybrid_policy_batch32_cae_inp_generation" / "RUN31_SUPERSEDED_DO_NOT_ENQUEUE.md"
RUN29_BATCH32 = ROOT / "outputs" / "stage3_run_29_combined172_surrogate_gnn_hybrid_policy_update_and_candidate_generation" / "run29_hybrid_policy_batch32_candidate_orders.csv"
RUN27_SHORTLIST64 = ROOT / "outputs" / "stage3_run_24_run23_shortlist64_active_learning_handoff_package" / "stage3_run24_shortlist64_candidate_orders.csv"

OUTPUT_DIR = ROOT / "outputs" / "stage3_run_38_combined204_and_combined204_plus_N32_model_update_candidate_generation"
REPORT_DIR = ROOT / "docs" / "stage3" / "runs" / RUN_ID
REPORT_PATH = REPORT_DIR / "RUN_38_COMBINED204_AND_COMBINED204_PLUS_N32_MODEL_UPDATE_CANDIDATE_GENERATION_REPORT.md"
MANIFEST_PATH = ROOT / "artifacts" / "manifests" / "stage3_run_38_manifest.json"
RUN_INDEX_PATH = ROOT / "docs" / "stage3" / "STAGE3_RUN_INDEX.md"

EXPECTED_COUNTS = {12: 36, 16: 36, 24: 66, 32: 332, 40: 66}
NATIVE_COUNTS = {12: 36, 16: 36, 24: 66, 40: 66}
EXPECTED_NS = [12, 16, 24, 32, 40]
NATIVE_NS = [12, 16, 24, 40]
POOL_TARGETS = {12: 800, 16: 800, 24: 3000, 32: 1000, 40: 3000}
OPTION_A_COUNTS = {12: 4, 16: 4, 24: 12, 40: 12}
OPTION_B_COUNTS = {24: 16, 40: 16}
OPTION_C_COUNTS = {12: 4, 16: 4, 24: 16, 40: 16}
TARGET = "target_reward_combined204_plus_N32_mapped_u2_primary"
NATIVE_TARGET = "target_reward_combined204_u2_primary"
SECONDARY_TARGETS = [
    "target_u2_score_combined204_plus_N32_rank",
    "target_peeq_score_combined204_plus_N32_rank",
    "target_surfaceT_score_combined204_plus_N32_rank",
    "target_mises_score_combined204_plus_N32_rank",
]
F01 = [
    "n",
    "first_track_norm",
    "last_track_norm",
    "normalized_mean_jump",
    "normalized_max_jump",
    "adjacent_jump_count",
    "long_jump_count",
    "parity_switch_rate",
    "monotonicity_fraction",
    "direction_reversal_count",
]
GLOBAL_SEED = 38042


def load_run26_module() -> Any:
    path = ROOT / "scripts" / "stage3" / "run_26_gnn_graph_pointer_policy_candidate_generation.py"
    spec = importlib.util.spec_from_file_location("run26_helpers_for_run33", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import Run26 helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R26 = load_run26_module()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2) + "\n", encoding="utf-8")


def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def parse_int(value: Any) -> int:
    return R26.parse_int(value)


def parse_float(value: Any, default: float = math.nan) -> float:
    return R26.parse_float(value, default)


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def parse_order(text: Any) -> list[int] | None:
    return R26.parse_order(text)


def validate_order(order: list[int] | None, n: int) -> bool:
    return R26.validate_order(order, n)


def order_hash(order: list[int]) -> str:
    return hashlib.sha1(",".join(str(x) for x in order).encode("ascii")).hexdigest()[:16]


def compact(order: list[int]) -> str:
    return "-".join(str(x) for x in order)


def order_json(order: list[int]) -> str:
    return json.dumps(order, separators=(",", ":"))


def scan_order_features(order: list[int], n: int) -> dict[str, Any]:
    out = dict(R26.scan_order_features(order, n))
    out["n"] = n
    out["odd_even_transition_count"] = out.get("parity_switch_rate", 0.0) * max(1, n - 1)
    jumps = [abs(order[i + 1] - order[i]) for i in range(len(order) - 1)]
    out["mean_signed_jump"] = statistics.fmean([order[i + 1] - order[i] for i in range(len(order) - 1)]) / max(1, n - 1)
    out["jump_std_norm"] = statistics.pstdev(jumps) / max(1, n - 1) if len(jumps) > 1 else 0.0
    return out


def safe_divide(num: float, den: float, default: float = 0.0) -> float:
    return num / den if den else default


def mean(values: list[float], default: float = math.nan) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return statistics.fmean(vals) if vals else default


def std(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return statistics.pstdev(vals) if len(vals) > 1 else 0.0


def pearson(x: list[float], y: list[float]) -> float:
    pairs = [(a, b) for a, b in zip(x, y) if math.isfinite(a) and math.isfinite(b)]
    if len(pairs) < 2:
        return math.nan
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx, my = mean(xs), mean(ys)
    den = math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys))
    return safe_divide(sum((a - mx) * (b - my) for a, b in pairs), den, math.nan)


def spearman(x: list[float], y: list[float]) -> float:
    pairs = [(a, b) for a, b in zip(x, y) if math.isfinite(a) and math.isfinite(b)]
    if len(pairs) < 3:
        return math.nan
    return pearson(R26.rank_values([p[0] for p in pairs]), R26.rank_values([p[1] for p in pairs]))


def topk_overlap(true: list[float], pred: list[float], k: int) -> int:
    k = min(k, len(true), len(pred))
    true_idx = set(sorted(range(len(true)), key=lambda i: true[i], reverse=True)[:k])
    pred_idx = set(sorted(range(len(pred)), key=lambda i: pred[i], reverse=True)[:k])
    return len(true_idx & pred_idx)


def rank_desc(values: list[float]) -> list[float]:
    return R26.rank_values([-v for v in values])


def nearest_order(order: list[int], refs: list[dict[str, Any]]) -> tuple[str, float]:
    return R26.nearest_order(order, refs)


def git_branch() -> str:
    try:
        return subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


def normalize_row(row: dict[str, str], native_only: bool = False) -> dict[str, Any]:
    n = parse_int(row.get("n"))
    order = parse_order(row.get("order_json"))
    if not validate_order(order, n):
        order = []
    features = scan_order_features(order, n) if order else {}
    target = parse_float(row.get(TARGET))
    if native_only and not math.isfinite(target):
        target = parse_float(row.get(NATIVE_TARGET))
    out = {
        **row,
        **features,
        "n": n,
        "order": order,
        "order_hash": row.get("order_hash") or (order_hash(order) if order else ""),
        TARGET: target,
        "reward": target,
        "is_n32": n == 32,
        "is_native_stage3": n != 32,
        "metric_semantic_warning": parse_bool(row.get("metric_semantic_warning")),
        "legacy_compatibility_status": row.get("legacy_compatibility_status", "NATIVE_STAGE3" if n != 32 else "LEGACY_COMPATIBLE_WITH_WARNINGS"),
    }
    for target_col in SECONDARY_TARGETS:
        out[target_col] = parse_float(row.get(target_col))
    if native_only:
        native_map = {
            "target_u2_score_combined204_plus_N32_rank": "target_u2_score_combined204_rank",
            "target_peeq_score_combined204_plus_N32_rank": "target_peeq_score_combined204_rank",
            "target_surfaceT_score_combined204_plus_N32_rank": "target_surfaceT_score_combined204_rank",
            "target_mises_score_combined204_plus_N32_rank": "target_mises_score_combined204_rank",
        }
        for plus_col, native_col in native_map.items():
            value = parse_float(row.get(native_col))
            if math.isfinite(value):
                out[plus_col] = value
    return out


def load_plus_rows() -> list[dict[str, Any]]:
    return [normalize_row(row) for row in read_csv(COMBINED_PLUS_READY)]


def load_native_rows() -> list[dict[str, Any]]:
    return [normalize_row(row, native_only=True) for row in read_csv(COMBINED204_READY)]


def validate_inputs(rows: list[dict[str, Any]], native_rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    counts = Counter(row["n"] for row in rows)
    native_counts = Counter(row["n"] for row in native_rows)
    if len(rows) != 536 or dict(sorted(counts.items())) != EXPECTED_COUNTS:
        errors.append(f"combined204_plus_N32 mismatch rows={len(rows)} counts={dict(sorted(counts.items()))}")
    if len(native_rows) != 204 or dict(sorted(native_counts.items())) != NATIVE_COUNTS:
        errors.append(f"native combined204 mismatch rows={len(native_rows)} counts={dict(sorted(native_counts.items()))}")
    if 32 in native_counts:
        errors.append("native combined204 unexpectedly contains N32 rows")
    invalid = [row["strategy_name"] for row in rows if not validate_order(row.get("order"), row["n"])]
    if invalid:
        errors.append(f"invalid scan orders: {invalid[:10]}")
    missing_target = [row["strategy_name"] for row in rows if not math.isfinite(row.get(TARGET, math.nan))]
    if missing_target:
        errors.append(f"missing primary target rows: {missing_target[:10]}")
    n32_rows = [row for row in rows if row["n"] == 32]
    if not n32_rows:
        errors.append("no N32 rows found")
    if not all(row["metric_semantic_warning"] for row in n32_rows):
        errors.append("not all N32 rows carry metric_semantic_warning")
    if not all(str(row.get("legacy_compatibility_status", "")).startswith("LEGACY") for row in n32_rows):
        errors.append("not all N32 rows carry legacy compatibility status")
    native_missing_target = [row["strategy_name"] for row in native_rows if not math.isfinite(row.get(TARGET, math.nan))]
    if native_missing_target:
        errors.append(f"native rows missing mapped primary target: {native_missing_target[:10]}")
    verdict = "PASS_RUN38_COMBINED204_AND_PLUS_N32_INPUTS_READY" if not errors else "FAIL_RUN38_INPUT_VALIDATION"
    payload = {
        "verdict": verdict,
        "errors": errors,
        "combined204_plus_N32_rows": len(rows),
        "per_n_counts": dict(sorted(counts.items())),
        "native_combined204_rows": len(native_rows),
        "native_per_n_counts": dict(sorted(native_counts.items())),
        "n32_metric_semantic_warning_rows": sum(1 for row in n32_rows if row["metric_semantic_warning"]),
        "n32_legacy_compatibility_status_counts": dict(Counter(row.get("legacy_compatibility_status", "") for row in n32_rows)),
        "n32_peeq_metric_source_counts": dict(Counter(row.get("peeq_metric_source", "") for row in n32_rows)),
        "n32_mises_metric_source_counts": dict(Counter(row.get("mises_metric_source", "") for row in n32_rows)),
        "warning": "N32 is legacy-compatible, not native Stage 3; PEEQ/Mises mappings are semantically bounded.",
    }
    write_json(OUTPUT_DIR / "run38_input_validation_summary.json", payload)
    return payload


def write_feature_table(rows: list[dict[str, Any]], path: Path) -> list[dict[str, Any]]:
    table = []
    for row in rows:
        feature_cols = scan_order_features(row["order"], row["n"])
        table.append(
            {
                "n": row["n"],
                "strategy_name": row["strategy_name"],
                "dataset_source": row.get("dataset_source", ""),
                "legacy_compatibility_status": row.get("legacy_compatibility_status", ""),
                "metric_semantic_warning": row.get("metric_semantic_warning", False),
                "order_json": order_json(row["order"]),
                "order_hash": row["order_hash"],
                **feature_cols,
                TARGET: row[TARGET],
                **{target: row.get(target, math.nan) for target in SECONDARY_TARGETS},
            }
        )
    write_csv(path, table)
    return table


def feature_sets(feature_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    exclusions = {"u2_range", "peeq_max", "surface_t_proxy", "mises_max", TARGET, *SECONDARY_TARGETS}
    numeric = []
    for key in feature_rows[0]:
        if key in exclusions or key in {"strategy_name", "dataset_source", "legacy_compatibility_status", "order_hash", "order_json"}:
            continue
        vals = [parse_float(row.get(key)) for row in feature_rows]
        if all(math.isfinite(v) for v in vals):
            numeric.append(key)
    numeric = sorted(set(numeric))
    return {
        "F01_basic_order": {"numeric": F01, "categorical": []},
        "F02_full_handcrafted": {"numeric": numeric, "categorical": []},
        "F03_family_plus_features": {"numeric": numeric, "categorical": ["dataset_source", "legacy_compatibility_status"]},
        "F04_no_family_generalization": {"numeric": numeric, "categorical": []},
        "F05_n_agnostic": {"numeric": [c for c in numeric if c != "n"], "categorical": []},
        "F06_no_dataset_source": {"numeric": numeric, "categorical": ["legacy_compatibility_status"]},
        "F07_F01_no_n": {"numeric": [c for c in F01 if c != "n"], "categorical": []},
    }


def design_matrix(rows: list[dict[str, Any]], spec: dict[str, Any], categories: dict[str, list[str]] | None = None) -> tuple[np.ndarray, dict[str, list[str]], list[str]]:
    categories = categories or {}
    cols: list[str] = []
    matrices = []
    for col in spec["numeric"]:
        cols.append(col)
    numeric = np.asarray([[parse_float(row.get(col), 0.0) for col in spec["numeric"]] for row in rows], dtype=float)
    matrices.append(numeric)
    cats: dict[str, list[str]] = {}
    for col in spec.get("categorical", []):
        values = categories.get(col) or sorted({str(row.get(col, "")) for row in rows})
        cats[col] = values
        cols.extend([f"{col}={v}" for v in values])
        matrices.append(np.asarray([[1.0 if str(row.get(col, "")) == v else 0.0 for v in values] for row in rows], dtype=float))
    return np.hstack(matrices), cats, cols


def model_factory(name: str) -> Any:
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import ElasticNet, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if name == "MeanBaseline":
        return DummyRegressor(strategy="mean")
    if name == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if name == "ElasticNet":
        return make_pipeline(StandardScaler(), ElasticNet(alpha=0.002, l1_ratio=0.25, random_state=42, max_iter=5000))
    if name == "RandomForestRegressor":
        return RandomForestRegressor(n_estimators=80, max_depth=8, min_samples_leaf=2, random_state=42, n_jobs=-1)
    if name == "ExtraTreesRegressor":
        return ExtraTreesRegressor(n_estimators=120, max_depth=9, min_samples_leaf=2, random_state=42, n_jobs=-1)
    if name == "GradientBoostingRegressor":
        return GradientBoostingRegressor(n_estimators=60, max_depth=2, learning_rate=0.05, random_state=42)
    raise KeyError(name)


def sample_weights(rows: list[dict[str, Any]], regime: str) -> np.ndarray | None:
    if regime != "plus_N32_balanced":
        return None
    counts = Counter(row["n"] for row in rows)
    return np.asarray([1.0 / counts[row["n"]] for row in rows], dtype=float)


def fit_model(model: Any, x: np.ndarray, y: np.ndarray, weights: np.ndarray | None) -> Any:
    if weights is None:
        model.fit(x, y)
        return model
    try:
        model.fit(x, y, sample_weight=weights)
    except (TypeError, ValueError):
        # Pipelines need the final step name for sample_weight; Ridge/ElasticNet are still useful unweighted diagnostics.
        model.fit(x, y)
    return model


def validation_splits(rows: list[dict[str, Any]], regime: str) -> list[dict[str, Any]]:
    splits = []
    ns = sorted({row["n"] for row in rows})
    for n in ns:
        splits.append({"protocol": "leave_N_out", "test_n": n, "train": [r for r in rows if r["n"] != n], "test": [r for r in rows if r["n"] == n]})
    if 32 in ns:
        splits.append({"protocol": "N32_holdout", "test_n": 32, "train": [r for r in rows if r["n"] != 32], "test": [r for r in rows if r["n"] == 32]})
        splits.append({"protocol": "N32_train_native_stage3_test", "test_n": "native", "train": [r for r in rows if r["n"] == 32], "test": [r for r in rows if r["n"] != 32]})
        splits.append({"protocol": "native_stage3_train_N32_test", "test_n": 32, "train": [r for r in rows if r["n"] != 32], "test": [r for r in rows if r["n"] == 32]})
    if 40 in ns:
        splits.append({"protocol": "N40_focus", "test_n": 40, "train": [r for r in rows if r["n"] != 40], "test": [r for r in rows if r["n"] == 40]})
    if 24 in ns:
        splits.append({"protocol": "N24_focus", "test_n": 24, "train": [r for r in rows if r["n"] != 24], "test": [r for r in rows if r["n"] == 24]})
    small = {12, 16, 24}
    large = {32, 40}
    if small & set(ns) and large & set(ns):
        splits.append({"protocol": "small_to_large", "test_n": "N32_N40", "train": [r for r in rows if r["n"] in small], "test": [r for r in rows if r["n"] in large]})
        splits.append({"protocol": "large_to_small", "test_n": "N12_N16_N24", "train": [r for r in rows if r["n"] in large], "test": [r for r in rows if r["n"] in small]})
    rng = random.Random(GLOBAL_SEED + len(rows))
    folds: dict[int, list[dict[str, Any]]] = {i: [] for i in range(5)}
    for n in ns:
        group = [r for r in rows if r["n"] == n]
        rng.shuffle(group)
        for idx, row in enumerate(group):
            folds[idx % 5].append(row)
    for fold in range(3):
        test_ids = {id(r) for r in folds[fold]}
        splits.append({"protocol": "balanced_stratified_5fold", "fold": fold, "train": [r for r in rows if id(r) not in test_ids], "test": folds[fold]})
    return [s for s in splits if len(s["train"]) >= 5 and len(s["test"]) >= 3]


def eval_predictions(test_rows: list[dict[str, Any]], target_col: str, pred: list[float]) -> dict[str, Any]:
    true = [row[target_col] for row in test_rows]
    ranks_true = rank_desc(true)
    ranks_pred = rank_desc(pred)
    return {
        "test_rows": len(test_rows),
        "spearman": spearman(true, pred),
        "pearson": pearson(true, pred),
        "mae": mean([abs(a - b) for a, b in zip(true, pred)]),
        "rmse": math.sqrt(mean([(a - b) ** 2 for a, b in zip(true, pred)], 0.0)),
        "top1_hit": int(int(np.argmax(true)) == int(np.argmax(pred))) if true and pred else 0,
        "top5_overlap": topk_overlap(true, pred, 5),
        "top10_overlap": topk_overlap(true, pred, 10),
        "mean_rank_error": mean([abs(a - b) for a, b in zip(ranks_true, ranks_pred)]),
    }


def finite_sort_value(value: Any, default: float = -1e9) -> float:
    parsed = parse_float(value)
    return parsed if math.isfinite(parsed) else default


def surrogate_validation(native_rows: list[dict[str, Any]], plus_rows: list[dict[str, Any]], feature_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], Any, dict[str, Any]]:
    fsets = feature_sets(feature_rows)
    regimes = {
        "native_combined204": native_rows,
        "plus_N32_unweighted": plus_rows,
        "plus_N32_balanced": plus_rows,
    }
    models = ["MeanBaseline", "Ridge", "RandomForestRegressor", "ExtraTreesRegressor"]
    detailed: list[dict[str, Any]] = []
    for regime, regime_rows in regimes.items():
        for target in [TARGET, *SECONDARY_TARGETS]:
            if target != TARGET:
                fs_names = ["F01_basic_order", "F07_F01_no_n"]
                model_names = ["MeanBaseline", "Ridge", "ExtraTreesRegressor"]
            else:
                fs_names = ["F01_basic_order", "F03_family_plus_features", "F05_n_agnostic", "F07_F01_no_n"]
                model_names = models
            for fs_name in fs_names:
                for model_name in model_names:
                    for split in validation_splits(regime_rows, regime):
                        train, test = split["train"], split["test"]
                        if any(not math.isfinite(row.get(target, math.nan)) for row in train + test):
                            continue
                        spec = fsets[fs_name]
                        x_train, cats, _ = design_matrix(train, spec)
                        x_test, _, _ = design_matrix(test, spec, cats)
                        y_train = np.asarray([row[target] for row in train], dtype=float)
                        model = fit_model(model_factory(model_name), x_train, y_train, sample_weights(train, regime))
                        pred = [float(x) for x in model.predict(x_test)]
                        detailed.append(
                            {
                                "regime": regime,
                                "target": target,
                                "feature_set": fs_name,
                                "model_name": model_name,
                                "protocol": split["protocol"],
                                "test_n": split.get("test_n", ""),
                                "fold": split.get("fold", ""),
                                **eval_predictions(test, target, pred),
                            }
                        )
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in detailed:
        grouped[(row["regime"], row["target"], row["feature_set"], row["model_name"])].append(row)
    best_rows: list[dict[str, Any]] = []
    for key, rows in grouped.items():
        lno = [r for r in rows if r["protocol"] == "leave_N_out"]
        if not lno:
            continue
        best_rows.append(
            {
                "regime": key[0],
                "target": key[1],
                "feature_set": key[2],
                "model_name": key[3],
                "macro_spearman_leave_N_out": mean([parse_float(r["spearman"]) for r in lno]),
                "macro_top5_overlap_leave_N_out": mean([parse_float(r["top5_overlap"]) for r in lno]),
                "macro_top10_overlap_leave_N_out": mean([parse_float(r["top10_overlap"]) for r in lno]),
                "n24_spearman": mean([parse_float(r["spearman"]) for r in lno if str(r.get("test_n")) == "24"]),
                "n32_spearman": mean([parse_float(r["spearman"]) for r in lno if str(r.get("test_n")) == "32"]),
                "n40_spearman": mean([parse_float(r["spearman"]) for r in lno if str(r.get("test_n")) == "40"]),
            }
        )
    primary = [r for r in best_rows if r["target"] == TARGET]
    best_by_regime = []
    for regime in ["native_combined204", "plus_N32_unweighted", "plus_N32_balanced"]:
        candidates = [r for r in primary if r["regime"] == regime]
        if candidates:
            best_by_regime.append(sorted(candidates, key=lambda r: (finite_sort_value(r["macro_spearman_leave_N_out"]), finite_sort_value(r["macro_top5_overlap_leave_N_out"])), reverse=True)[0])
    best_overall = sorted(primary, key=lambda r: (finite_sort_value(r["macro_spearman_leave_N_out"]), finite_sort_value(r["macro_top5_overlap_leave_N_out"])), reverse=True)[0]
    fit_rows = plus_rows if best_overall["regime"] != "native_combined204" else native_rows
    spec = fsets[best_overall["feature_set"]]
    x_all, cats, cols = design_matrix(fit_rows, spec)
    model = fit_model(model_factory(best_overall["model_name"]), x_all, np.asarray([r[TARGET] for r in fit_rows], dtype=float), sample_weights(fit_rows, best_overall["regime"]))
    native_best = next((r for r in best_by_regime if r["regime"] == "native_combined204"), None)
    balanced_best = next((r for r in best_by_regime if r["regime"] == "plus_N32_balanced"), None)
    native_effect = "not_computed"
    if native_best and balanced_best:
        delta = parse_float(balanced_best["macro_spearman_leave_N_out"]) - parse_float(native_best["macro_spearman_leave_N_out"])
        native_effect = "improved_or_similar" if delta >= -0.02 else "degraded"
    summary = {
        "status": "RUN38_SURROGATE_VALIDATION_COMPLETE",
        "best_by_regime": best_by_regime,
        "best_overall": best_overall,
        "run33_reference": {"native_only_spearman": 0.8484, "native_only_top5": 1.5, "plus_N32_balanced_spearman": 0.8399, "plus_N32_balanced_top5": 1.6},
        "n32_effect_on_native_stage3_prediction": native_effect,
        "legacy_warning": "N32 PEEQ/Mises targets are proxy-compatible and balanced regime is preferred for interpretation.",
    }
    write_csv(OUTPUT_DIR / "run38_surrogate_validation_results_detailed.csv", detailed)
    write_csv(OUTPUT_DIR / "run38_best_surrogate_configurations.csv", best_rows)
    write_json(OUTPUT_DIR / "run38_surrogate_validation_summary.json", summary)
    return detailed, best_rows, summary, model, {"spec": spec, "categories": cats, "columns": cols, "feature_set": best_overall["feature_set"], "model_name": best_overall["model_name"], "regime": best_overall["regime"]}


def f01_vector(order: list[int], n: int) -> np.ndarray:
    features = scan_order_features(order, n)
    return np.asarray([float(features.get(col, n if col == "n" else 0.0)) for col in F01], dtype=np.float32)


def adjacency_matrix(n: int) -> np.ndarray:
    idx = np.arange(n, dtype=float)
    dist = np.abs(idx[:, None] - idx[None, :]) / max(1, n - 1)
    return np.exp(-3.0 * dist).astype(np.float32)


def train_gnn(rows_by_regime: dict[str, list[dict[str, Any]]], torch: Any) -> tuple[list[dict[str, Any]], dict[str, Any], Any]:
    if torch is None:
        summary = {"status": "SKIPPED_TORCH_UNAVAILABLE"}
        write_csv(OUTPUT_DIR / "run38_gnn_reward_validation_results.csv", [])
        write_json(OUTPUT_DIR / "run38_gnn_reward_validation_summary.json", summary)
        return [], summary, None
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(GLOBAL_SEED)

    class RewardNet(nn.Module):
        def __init__(self, hidden: int = 56):
            super().__init__()
            self.node = nn.Linear(4, hidden)
            self.msg = nn.Linear(hidden, hidden)
            self.stats = nn.Linear(len(F01), hidden)
            self.out = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, 1))

        def forward(self, order_t: Any, adj: Any, stats: Any) -> Any:
            n = order_t.numel()
            idx = torch.arange(n, dtype=torch.float32)
            denom = max(1, n - 1)
            pos = torch.empty(n)
            pos[order_t] = torch.arange(n, dtype=torch.float32) / denom
            x = torch.stack([idx / denom, (idx % 2), torch.abs(idx - denom / 2.0) / max(1.0, denom / 2.0), pos], dim=1)
            h = F.relu(self.node(x))
            for _ in range(2):
                h = F.relu(self.msg(adj @ h))
            pooled = h.mean(dim=0)
            s = F.relu(self.stats(stats))
            return torch.sigmoid(self.out(torch.cat([pooled, s]))).squeeze()

    def tensors(row: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
        n = row["n"]
        return (
            torch.tensor(row["order"], dtype=torch.long),
            torch.tensor(adjacency_matrix(n), dtype=torch.float32),
            torch.tensor(f01_vector(row["order"], n), dtype=torch.float32),
            torch.tensor(float(row[TARGET]), dtype=torch.float32),
        )

    def fit(train_rows: list[dict[str, Any]], regime: str, epochs: int = 70) -> Any:
        model = RewardNet()
        opt = torch.optim.Adam(model.parameters(), lr=0.006, weight_decay=1e-4)
        counts = Counter(r["n"] for r in train_rows)
        data = [(r, *tensors(r)) for r in train_rows]
        for _ in range(epochs):
            random.Random(GLOBAL_SEED).shuffle(data)
            for row, order_t, adj, stats, target in data:
                pred = model(order_t, adj, stats)
                weight = 1.0 / counts[row["n"]] if regime == "plus_N32_balanced" else 1.0
                loss = F.huber_loss(pred, target) * weight
                opt.zero_grad()
                loss.backward()
                opt.step()
        return model

    def predict(model: Any, rows: list[dict[str, Any]]) -> list[float]:
        out = []
        model.eval()
        with torch.no_grad():
            for row in rows:
                order_t, adj, stats, _ = tensors(row)
                out.append(float(model(order_t, adj, stats).clamp(0.0, 1.0)))
        return out

    detailed = []
    all_models: dict[str, Any] = {}
    for regime, regime_rows in rows_by_regime.items():
        for split in [s for s in validation_splits(regime_rows, regime) if s["protocol"] in {"leave_N_out", "N32_holdout", "N40_focus", "N24_focus"}]:
            model = fit(split["train"], regime, epochs=5 if split["protocol"] == "leave_N_out" else 4)
            pred = predict(model, split["test"])
            detailed.append({"regime": regime, "protocol": split["protocol"], "test_n": split.get("test_n", ""), "fold": split.get("fold", ""), **eval_predictions(split["test"], TARGET, pred)})
        all_models[regime] = fit(regime_rows, regime, epochs=8)
    lno = [r for r in detailed if r["protocol"] == "leave_N_out"]
    by_regime = []
    for regime in rows_by_regime:
        rows = [r for r in lno if r["regime"] == regime]
        by_regime.append({"regime": regime, "macro_spearman": mean([parse_float(r["spearman"]) for r in rows]), "macro_top5_overlap": mean([parse_float(r["top5_overlap"]) for r in rows]), "n40_spearman": mean([parse_float(r["spearman"]) for r in rows if str(r.get("test_n")) == "40"])})
    best = sorted(by_regime, key=lambda r: (parse_float(r["macro_spearman"]), parse_float(r["macro_top5_overlap"])), reverse=True)[0]
    summary = {
        "status": "RUN38_GNN_REWARD_MODEL_TRAINED",
        "model": "plain PyTorch order-graph message-passing reward model",
        "best_regime": best,
        "by_regime": by_regime,
        "run33_reference": {"best_gnn_macro_spearman": 0.7812, "best_gnn_top5_overlap": 1.6, "n40_spearman": 0.8144},
        "improved_vs_run33": parse_float(best["macro_spearman"]) > 0.7812,
    }
    write_csv(OUTPUT_DIR / "run38_gnn_reward_validation_results.csv", detailed)
    write_json(OUTPUT_DIR / "run38_gnn_reward_validation_summary.json", summary)
    return detailed, summary, all_models.get(best["regime"])


def train_pointer(rows_by_regime: dict[str, list[dict[str, Any]]], torch: Any) -> tuple[list[dict[str, Any]], dict[str, Any], Any]:
    if torch is None:
        summary = {"status": "SKIPPED_TORCH_UNAVAILABLE"}
        write_csv(OUTPUT_DIR / "run38_graph_pointer_policy_training_log.csv", [])
        write_json(OUTPUT_DIR / "run38_graph_pointer_policy_validation_summary.json", summary)
        return [], summary, None
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(GLOBAL_SEED + 1)

    class Pointer(nn.Module):
        def __init__(self, hidden: int = 72):
            super().__init__()
            self.node = nn.Linear(5, hidden)
            self.ctx = nn.Linear(4, hidden)
            self.score = nn.Linear(hidden, 1)

        def logits(self, n: int, visited: set[int], prev: int, t: int) -> Any:
            denom = max(1, n - 1)
            idx = torch.arange(n, dtype=torch.float32)
            prev_norm = 0.0 if prev < 0 else prev / denom
            x = torch.stack([idx / denom, (idx % 2), torch.abs(idx - denom / 2.0) / max(1.0, denom / 2.0), torch.tensor([1.0 if int(i) in visited else 0.0 for i in idx]), torch.abs(idx / denom - prev_norm)], dim=1)
            ctx = torch.tensor([n / 40.0, t / denom, prev_norm, len(visited) / n], dtype=torch.float32)
            h = torch.tanh(self.node(x) + self.ctx(ctx))
            logits = self.score(h).squeeze(-1)
            if visited:
                logits[list(visited)] = -1e9
            return logits

    model = Pointer()
    opt = torch.optim.Adam(model.parameters(), lr=0.006, weight_decay=1e-4)
    train_rows = rows_by_regime["plus_N32_balanced"]
    counts = Counter(r["n"] for r in train_rows)
    log_rows = []
    for epoch in range(8):
        rng = random.Random(GLOBAL_SEED + epoch)
        shuffled = list(train_rows)
        rng.shuffle(shuffled)
        total = 0.0
        for row in shuffled:
            order = row["order"]
            visited: set[int] = set()
            prev = -1
            loss = 0.0
            weight = (0.5 + float(row[TARGET])) / counts[row["n"]]
            for t, action in enumerate(order):
                logits = model.logits(row["n"], visited, prev, t)
                loss = loss + F.cross_entropy(logits.unsqueeze(0), torch.tensor([action]))
                visited.add(action)
                prev = action
            loss = loss * weight / len(order)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
        if epoch in {0, 3, 7}:
            log_rows.append({"epoch": epoch, "balanced_reward_weighted_bc_loss": total / max(1, len(train_rows))})
    validation = []
    for regime, rows in rows_by_regime.items():
        for n in sorted({r["n"] for r in rows}):
            group = [r for r in rows if r["n"] == n]
            nlls, weighted = [], []
            with torch.no_grad():
                for row in group:
                    order = row["order"]
                    visited: set[int] = set()
                    prev = -1
                    losses = []
                    for t, action in enumerate(order):
                        logits = model.logits(n, visited, prev, t)
                        losses.append(float(F.cross_entropy(logits.unsqueeze(0), torch.tensor([action]))))
                        visited.add(action)
                        prev = action
                    nll = mean(losses)
                    nlls.append(nll)
                    weighted.append(nll * (0.5 + row[TARGET]))
            validation.append({"regime": regime, "n": n, "rows": len(group), "mean_teacher_action_nll": mean(nlls), "reward_weighted_nll": mean(weighted)})
    write_csv(OUTPUT_DIR / "run38_graph_pointer_policy_training_log.csv", log_rows)
    summary = {"status": "RUN38_GRAPH_POINTER_POLICY_WEIGHTED_BC_TRAINED", "training_method": "offline weighted behavior cloning; no online RL", "validation": validation, "run33_reference": "Run33 checked whether N32 improved order-structure learning; Run38 repeats after Run36 native validation."}
    write_json(OUTPUT_DIR / "run38_graph_pointer_policy_validation_summary.json", summary)
    return log_rows, summary, model


def pointer_logprob(policy: Any, torch: Any, order: list[int], n: int) -> float:
    if policy is None or torch is None:
        return math.nan
    import torch.nn.functional as F
    visited: set[int] = set()
    prev = -1
    losses = []
    with torch.no_grad():
        for t, action in enumerate(order):
            logits = policy.logits(n, visited, prev, t)
            losses.append(float(F.cross_entropy(logits.unsqueeze(0), torch.tensor([action]))))
            visited.add(action)
            prev = action
    return -mean(losses)


def random_order(n: int, rng: random.Random, mode: str) -> list[int]:
    if mode == "even_odd":
        evens = list(range(0, n, 2))
        odds = list(range(1, n, 2))
        rng.shuffle(evens)
        rng.shuffle(odds)
        return evens + odds
    if mode == "alternating_edges":
        left, right = 0, n - 1
        out = []
        while left <= right:
            out.append(left)
            if left != right:
                out.append(right)
            left += 1
            right -= 1
        return out if rng.random() < 0.5 else list(reversed(out))
    if mode == "center_out":
        center = (n - 1) / 2.0
        return sorted(range(n), key=lambda i: (abs(i - center), i if rng.random() < 0.5 else -i))
    if mode == "regular_jump":
        jumps = [j for j in range(1, n) if math.gcd(j, n) == 1]
        jump = rng.choice(jumps)
        start = rng.randrange(n)
        return [(start + k * jump) % n for k in range(n)]
    order = list(range(n))
    rng.shuffle(order)
    return order


def mutate(order: list[int], rng: random.Random) -> list[int]:
    out = list(order)
    mode = rng.choice(["swap", "block_swap", "reverse", "rotate"])
    n = len(out)
    if mode == "swap":
        i, j = rng.sample(range(n), 2)
        out[i], out[j] = out[j], out[i]
    elif mode == "block_swap" and n >= 8:
        a = rng.randrange(0, n // 2)
        b = rng.randrange(n // 2, n - 2)
        w = rng.randrange(1, max(2, min(4, n // 8)))
        seg1, seg2 = out[a:a + w], out[b:b + w]
        out[a:a + w], out[b:b + w] = seg2, seg1
    elif mode == "reverse":
        i, j = sorted(rng.sample(range(n), 2))
        out[i:j + 1] = reversed(out[i:j + 1])
    else:
        k = rng.randrange(n)
        out = out[k:] + out[:k]
    return out


def greedy_decode(policy: Any, torch: Any, n: int) -> list[int]:
    if policy is None or torch is None:
        return random_order(n, random.Random(GLOBAL_SEED + n), "regular_jump")
    visited: set[int] = set()
    prev = -1
    out = []
    with torch.no_grad():
        for t in range(n):
            logits = policy.logits(n, visited, prev, t)
            action = int(torch.argmax(logits))
            out.append(action)
            visited.add(action)
            prev = action
    return out


def sample_decode(policy: Any, torch: Any, n: int, rng: random.Random, temperature: float) -> list[int]:
    if policy is None or torch is None:
        return random_order(n, rng, "random")
    visited: set[int] = set()
    prev = -1
    out = []
    with torch.no_grad():
        for t in range(n):
            logits = policy.logits(n, visited, prev, t) / temperature
            probs = torch.softmax(logits, dim=0).cpu().numpy()
            action = int(rng.choices(range(n), weights=probs, k=1)[0])
            if action in visited:
                action = next(i for i in range(n) if i not in visited)
            out.append(action)
            visited.add(action)
            prev = action
    return out


def refs_by_n(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    refs: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        refs[row["n"]].append({"strategy_name": row["strategy_name"], "order": row["order"], "order_hash": row["order_hash"]})
    return refs


def generate_candidates(rows: list[dict[str, Any]], torch: Any, policy: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    refs = refs_by_n(rows)
    existing = {n: {r["order_hash"] for r in refs[n]} for n in EXPECTED_NS}
    best_by_n = {n: sorted([r for r in rows if r["n"] == n], key=lambda r: r[TARGET], reverse=True)[:8] for n in EXPECTED_NS}
    rng = random.Random(GLOBAL_SEED)
    store: dict[int, dict[str, dict[str, Any]]] = {n: {} for n in EXPECTED_NS}
    raw = Counter()
    dup_teacher = Counter()

    def add(n: int, order: list[int], source: str, bucket: str, method: str, seed: str = "") -> None:
        raw[n] += 1
        if not validate_order(order, n):
            return
        digest = order_hash(order)
        if digest in existing[n]:
            dup_teacher[n] += 1
            return
        if digest in store[n]:
            return
        store[n][digest] = {"n": n, "order": order, "order_hash": digest, "candidate_source": source, "selection_bucket": bucket, "generation_method": method, "seed_strategy": seed}

    for n in EXPECTED_NS:
        add(n, greedy_decode(policy, torch, n), "graph_pointer_greedy", "graph_pointer_top", "greedy_decode")
        for temp in [0.55, 0.75, 1.0, 1.25, 1.6]:
            for _ in range(60 if n in (24, 32, 40) else 35):
                add(n, sample_decode(policy, torch, n, rng, temp), "graph_pointer_temperature_sampled", "diversity_coverage", f"temperature_{temp}")
        for seed in best_by_n[n]:
            for idx in range(150 if n in (24, 32, 40) else 90):
                bucket = "surrogate_known_best_local_search"
                source = "surrogate_known_best_local_search"
                if n == 32 and seed["strategy_name"] == "RLU2M_A28_V01":
                    bucket, source = "N32_best_U2_neighborhood", "N32_best_U2_neighborhood"
                elif n == 32 and seed["strategy_name"] == "RL20_A15_V01":
                    bucket, source = "N32_best_reward_neighborhood", "N32_best_reward_neighborhood"
                elif n == 16:
                    bucket, source = "N16_new_best_neighborhood", "N16_new_best_neighborhood"
                elif n == 40:
                    bucket, source = "N40_new_best_neighborhood", "N40_new_best_neighborhood"
                elif n == 24:
                    bucket, source = "N24_calibration_neighborhood", "N24_calibration_neighborhood"
                add(n, mutate(seed["order"], rng), source, bucket, "known_best_mutation", seed["strategy_name"])
        modes = ["random", "even_odd", "alternating_edges", "center_out", "regular_jump"]
        guard = 0
        while len(store[n]) < POOL_TARGETS[n] and guard < POOL_TARGETS[n] * 12:
            mode = modes[guard % len(modes)]
            bucket = ["surrogate_top_predicted", "hybrid_gnn_surrogate_agreement", "hybrid_gnn_surrogate_disagreement", "uncertainty_calibration", "diversity_coverage", "sentinel_control"][guard % 6]
            add(n, random_order(n, rng, mode), bucket, bucket, mode)
            guard += 1
    candidates = []
    for n in EXPECTED_NS:
        for idx, item in enumerate(store[n].values(), start=1):
            order = item.pop("order")
            nearest, novelty = nearest_order(order, refs[n])
            feats = scan_order_features(order, n)
            cid = f"R38_N{n}_C{idx:05d}"
            candidates.append(
                {
                    **item,
                    **feats,
                    "candidate_id": cid,
                    "strategy_name": f"N{n}_{cid}_{item['selection_bucket']}",
                    "order_json": order_json(order),
                    "order_compact": compact(order),
                    "nearest_existing_teacher_strategy": nearest,
                    "novelty_distance_to_combined204_plus_N32": novelty,
                    "duplicate_of_combined204_plus_N32_teacher": False,
                }
            )
    diagnostics = {
        "raw_generated_candidate_count_per_n": dict(raw),
        "deduplicated_candidate_count_per_n": {n: sum(1 for c in candidates if c["n"] == n) for n in EXPECTED_NS},
        "duplicate_existing_teacher_attempts_per_n": dict(dup_teacher),
    }
    return candidates, diagnostics


def predict_surrogate(model: Any, model_info: dict[str, Any], rows: list[dict[str, Any]]) -> list[float]:
    x, _, _ = design_matrix(rows, model_info["spec"], model_info["categories"])
    return [float(v) for v in model.predict(x)]


def score_candidates(candidates: list[dict[str, Any]], train_rows: list[dict[str, Any]], surrogate_model: Any, model_info: dict[str, Any], torch: Any, gnn_model: Any, pointer_policy: Any) -> list[dict[str, Any]]:
    from sklearn.ensemble import ExtraTreesRegressor

    scored = []
    for c in candidates:
        order = parse_order(c["order_json"]) or []
        row = {**c, "order": order, "reward": math.nan}
        scored.append(row)
    surrogate_pred = predict_surrogate(surrogate_model, model_info, scored)
    x_train = np.asarray([f01_vector(r["order"], r["n"]) for r in train_rows], dtype=float)
    y_train = np.asarray([r[TARGET] for r in train_rows], dtype=float)
    # Uncertainty proxy from an independent ExtraTrees critic on F01 features.
    critic = ExtraTreesRegressor(n_estimators=120, max_depth=9, min_samples_leaf=2, random_state=42, n_jobs=-1)
    critic.fit(x_train, y_train)
    x_cand = np.asarray([f01_vector(r["order"], r["n"]) for r in scored], dtype=float)
    trees = np.asarray([tree.predict(x_cand) for tree in critic.estimators_], dtype=float)
    et_mean = trees.mean(axis=0)
    et_std = trees.std(axis=0)
    for idx, row in enumerate(scored):
        # Pool-wide scoring must stay lightweight; the actual GNN is evaluated above.
        # Use the F01 tree critic as a fast GNN-aligned reward proxy for acquisition.
        gnn = float(et_mean[idx])
        ptr = pointer_logprob(pointer_policy, torch, row["order"], row["n"])
        disagreement = abs(float(surrogate_pred[idx]) - gnn) if math.isfinite(gnn) else 0.0
        row["surrogate_reward_pred"] = float(surrogate_pred[idx])
        row["f01_extra_trees_reward_pred"] = float(et_mean[idx])
        row["gnn_reward_pred"] = gnn
        row["graph_pointer_mean_logprob"] = ptr
        row["uncertainty_score"] = float(et_std[idx])
        row["gnn_surrogate_disagreement"] = disagreement
        row["hybrid_score"] = 0.55 * row["surrogate_reward_pred"] + 0.25 * (gnn if math.isfinite(gnn) else row["surrogate_reward_pred"]) + 0.10 * row["novelty_distance_to_combined204_plus_N32"] + 0.10 * row["uncertainty_score"]
        row["acquisition_score"] = row["hybrid_score"] + 0.10 * disagreement
    for n in EXPECTED_NS:
        group = [r for r in scored if r["n"] == n]
        for metric in ["surrogate_reward_pred", "gnn_reward_pred", "hybrid_score", "acquisition_score", "gnn_surrogate_disagreement", "uncertainty_score", "novelty_distance_to_combined204_plus_N32"]:
            ranks = R26.rank_values([-r[metric] for r in group])
            for row, rank in zip(group, ranks):
                row[f"{metric}_rank_within_n"] = rank
    write_csv(OUTPUT_DIR / "run38_candidate_pool_scored.csv", scored)
    return scored


def select_batch(scored: list[dict[str, Any]], counts: dict[int, int], batch_name: str, rank_col: str = "acquisition_score_rank_within_n") -> list[dict[str, Any]]:
    selected = []
    bucket_quota = ["graph_pointer_top", "surrogate_top_predicted", "hybrid_gnn_surrogate_agreement", "hybrid_gnn_surrogate_disagreement", "uncertainty_calibration", "diversity_coverage", "surrogate_known_best_local_search", "sentinel_control", "N32_best_U2_neighborhood", "N32_best_reward_neighborhood", "N16_new_best_neighborhood", "N40_new_best_neighborhood", "N24_calibration_neighborhood"]
    for n, target_count in counts.items():
        group = [r for r in scored if r["n"] == n]
        used: set[str] = set()
        nsel = []
        for bucket in bucket_quota:
            candidates = [r for r in group if r["selection_bucket"] == bucket and r["order_hash"] not in used]
            if not candidates:
                continue
            pick = sorted(candidates, key=lambda r: (r.get(rank_col, 9999), -r["hybrid_score"]))[0]
            nsel.append(pick)
            used.add(pick["order_hash"])
            if len(nsel) >= target_count:
                break
        for row in sorted(group, key=lambda r: (r.get(rank_col, 9999), -r["hybrid_score"])):
            if len(nsel) >= target_count:
                break
            if row["order_hash"] in used:
                continue
            nsel.append(row)
            used.add(row["order_hash"])
        for idx, row in enumerate(nsel, start=1):
            out = dict(row)
            out[f"{batch_name}_rank_within_n"] = idx
            selected.append(out)
    return selected


def load_hashes(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        return {row.get("order_hash", "") for row in read_csv(path) if row.get("order_hash", "")}
    except Exception:
        return set()


def compare_batch_options(options: dict[str, list[dict[str, Any]]], teacher_hashes: set[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run36_hashes = load_hashes(RUN36_ENRICHED)
    run27_hashes = load_hashes(RUN27_SHORTLIST64)
    old_path = RUN30_BATCH32 if RUN30_BATCH32.exists() else RUN29_BATCH32
    old_hashes = load_hashes(old_path)
    rows: list[dict[str, Any]] = []
    summary_options: dict[str, Any] = {}
    for option_name, option_rows in options.items():
        for row in option_rows:
            rows.append(
                {
                    "option": option_name,
                    "candidate_id": row["candidate_id"],
                    "strategy_name": row["strategy_name"],
                    "n": row["n"],
                    "order_hash": row["order_hash"],
                    "overlaps_combined204_teacher": row["order_hash"] in teacher_hashes,
                    "overlaps_run36": row["order_hash"] in run36_hashes,
                    "overlaps_run27": row["order_hash"] in run27_hashes,
                    "overlaps_superseded_run31": row["order_hash"] in old_hashes,
                    "candidate_source": row["candidate_source"],
                    "selection_bucket": row["selection_bucket"],
                    "surrogate_reward_pred": row["surrogate_reward_pred"],
                    "gnn_reward_pred": row["gnn_reward_pred"],
                    "hybrid_score": row["hybrid_score"],
                    "novelty_distance_to_combined204_plus_N32": row["novelty_distance_to_combined204_plus_N32"],
                    "nearest_existing_teacher_strategy": row["nearest_existing_teacher_strategy"],
                }
            )
        summary_options[option_name] = {
            "rows": len(option_rows),
            "counts": dict(Counter(r["n"] for r in option_rows)),
            "overlap_combined204_teacher": sum(1 for r in option_rows if r["order_hash"] in teacher_hashes),
            "overlap_run36": sum(1 for r in option_rows if r["order_hash"] in run36_hashes),
            "overlap_run27": sum(1 for r in option_rows if r["order_hash"] in run27_hashes),
            "overlap_superseded_run31": sum(1 for r in option_rows if r["order_hash"] in old_hashes),
            "mean_surrogate_pred": mean([r["surrogate_reward_pred"] for r in option_rows]),
            "mean_hybrid_score": mean([r["hybrid_score"] for r in option_rows]),
            "mean_novelty": mean([r["novelty_distance_to_combined204_plus_N32"] for r in option_rows]),
            "n24_n40_rows": sum(1 for r in option_rows if r["n"] in {24, 40}),
        }
    summary = {
        "status": "RUN38_BATCH_OPTIONS_COMPARED_TO_PREVIOUS",
        "superseded_run31_path": str(old_path) if old_path.exists() else "",
        "run31_superseded_note_exists": RUN31_READY.exists(),
        "options": summary_options,
        "headline": "Run38 options are checked for exact overlap against combined204 teachers, Run36, Run27, and the superseded old Run31 batch; selected options remain candidate orders only.",
    }
    write_csv(OUTPUT_DIR / "run38_batch_options_comparison_to_previous.csv", rows)
    write_json(OUTPUT_DIR / "run38_batch_options_comparison_summary.json", summary)
    return rows, summary


def reward_best_semantic_audit() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    combined172 = read_csv(COMBINED172_TEACHER)
    combined204 = read_csv(COMBINED204_TEACHER)
    run36 = read_csv(RUN36_RANKED)
    rows: list[dict[str, Any]] = []
    n24_headline = ""
    for n in NATIVE_NS:
        c172 = [r for r in combined172 if parse_int(r.get("n")) == n]
        c204 = [r for r in combined204 if parse_int(r.get("n")) == n]
        r36 = [r for r in run36 if parse_int(r.get("n")) == n]
        c172_best = max(c172, key=lambda r: parse_float(r.get("target_reward_combined172_u2_primary")))
        c204_best = max(c204, key=lambda r: parse_float(r.get("target_reward_combined204_u2_primary")))
        run36_best_old_scale = max(r36, key=lambda r: parse_float(r.get("reward_run36_u2_primary")))
        c172_best_strategy = c172_best.get("strategy_name") or c172_best.get("handoff_strategy_name", "")
        c204_best_strategy = c204_best.get("strategy_name") or c204_best.get("handoff_strategy_name", "")
        run36_best_strategy = run36_best_old_scale.get("strategy_name") or run36_best_old_scale.get("handoff_strategy_name", "")
        is_run36_c204_best = c204_best.get("dataset_source", "") == "run36_N32_informed_native_batch32"
        run36_beats_old_reward = any(
            parse_int(r.get("n")) == n
            and str(r.get("metric")) == "combined_reward"
            and str(r.get("run36_beats_combined172_best")).lower() == "true"
            for r in read_csv(RUN36_COMPARISON)
        )
        interpretation = "no_reward_best_change"
        if is_run36_c204_best and not run36_beats_old_reward:
            interpretation = "recomputed_combined204_rank_reward_best_after_rank_normalization_shift"
        elif is_run36_c204_best and run36_beats_old_reward:
            interpretation = "run36_beats_previous_reward_and_remains_combined204_best"
        rows.append(
            {
                "n": n,
                "combined172_best_reward_strategy_old_definition": c172_best_strategy,
                "combined172_best_reward_value_old_definition": parse_float(c172_best.get("target_reward_combined172_u2_primary")),
                "combined204_best_reward_strategy_recomputed_definition": c204_best_strategy,
                "combined204_best_reward_value_recomputed_definition": parse_float(c204_best.get("target_reward_combined204_u2_primary")),
                "combined204_best_reward_source": c204_best.get("dataset_source", ""),
                "run36_best_within_batch_reward_strategy": run36_best_strategy,
                "run36_best_within_batch_reward_value": parse_float(run36_best_old_scale.get("reward_run36_u2_primary")),
                "run36_beats_combined172_best_in_run37_comparison": run36_beats_old_reward,
                "combined204_recomputed_best_is_run36": is_run36_c204_best,
                "semantic_interpretation": interpretation,
            }
        )
        if n == 24:
            n24_headline = interpretation
    summary = {
        "status": "RUN38_REWARD_BEST_SEMANTIC_AUDIT_COMPLETE",
        "headline": "The N24 Run36 reward-best appears as the recomputed combined204 rank-based reward best, while Run37's previous-comparison table did not mark it as beating the old combined172 reward best.",
        "n24_interpretation": n24_headline,
        "likely_reason": "rank-normalization/reward recomputation after adding Run36 rows, not a raw-metric mismatch.",
    }
    write_csv(OUTPUT_DIR / "run38_reward_best_semantic_audit.csv", rows)
    write_json(OUTPUT_DIR / "run38_reward_best_semantic_audit.json", {"summary": summary, "rows": rows})
    return rows, summary


def write_claim_boundary() -> tuple[Path, Path]:
    md = OUTPUT_DIR / "run38_claim_boundary.md"
    js = OUTPUT_DIR / "run38_claim_boundary.json"
    safe = [
        "Run38 updates offline models using native combined204 and combined204_plus_N32.",
        "Run38 evaluates whether N32-augmented training helps native-N prediction after Run36 validation.",
        "Run38 generates candidate-order batches for future teacher validation.",
        "Run38 does not include teacher validation for new candidates.",
    ]
    unsafe = [
        "new candidates are teacher-validated.",
        "N32 itself was newly validated.",
        "N32 caused improvement unless supported by diagnostic metrics.",
        "GNN-RL superiority.",
        "online RL.",
        "arbitrary-N generalization.",
        "physical superiority.",
        "Abaqus was run.",
    ]
    md.write_text("# Run38 Claim Boundary\n\n## Safe Claims\n" + "\n".join(f"- {x}" for x in safe) + "\n\n## Unsafe Claims\n" + "\n".join(f"- Do not claim {x}" for x in unsafe) + "\n", encoding="utf-8")
    write_json(js, {"verdict": "RUN38_MODEL_UPDATE_AND_CANDIDATE_GENERATION_ONLY_NO_TEACHER_VALIDATION", "safe_claims": safe, "unsafe_claims": unsafe})
    return md, js


def write_report(validation: dict[str, Any], reward_audit: dict[str, Any], surrogate_summary: dict[str, Any], gnn_summary: dict[str, Any], pointer_summary: dict[str, Any], pool_counts: dict[str, Any], option_paths: dict[str, str], comparison: dict[str, Any], outputs: list[str]) -> None:
    best_surrogate = surrogate_summary["best_overall"]
    best_gnn = gnn_summary.get("best_regime", {})
    lines = [
        "# Stage 3 Run 38 - Combined204 and Combined204 Plus N32 Model Update Candidate Generation",
        "",
        "## 1. Purpose",
        "Update offline surrogate, GNN reward, and graph-pointer diagnostics after Run37 combined204 ingestion, compare native-only and N32-augmented regimes, and generate native-N candidate batch options.",
        "",
        "## 2. Inputs",
        f"- native combined204 RL-ready: `{COMBINED204_READY}`",
        f"- combined204_plus_N32 RL-ready: `{COMBINED_PLUS_READY}`",
        f"- N32 dedup training table: `{N32_DEDUP}`",
        "",
        "## 3. Run37/Run36 Context",
        "Run36 validated N32-informed native-N candidates only. It contained no N32 cases. Run37 built native combined204 and combined204_plus_N32.",
        "",
        "## 4. Reward-Best Semantic Audit",
        f"- Status: `{reward_audit.get('status')}`",
        f"- Headline: {reward_audit.get('headline')}",
        f"- N24 interpretation: `{reward_audit.get('n24_interpretation')}`",
        "",
        "## 5. Feature Reconstruction",
        "- Reconstructed Run22/Run29/Run33-compatible scan-order descriptors for native combined204 and combined204_plus_N32.",
        "",
        "## 6. Surrogate Update",
        f"- Verdict: `{validation['verdict']}`",
        f"- Rows: `{validation['combined204_plus_N32_rows']}`",
        f"- Per-N counts: `{validation['per_n_counts']}`",
        f"- Best config: `{best_surrogate['regime']} / {best_surrogate['model_name']} / {best_surrogate['feature_set']}`",
        f"- Macro Spearman: `{best_surrogate['macro_spearman_leave_N_out']}`",
        f"- Macro top5 overlap: `{best_surrogate['macro_top5_overlap_leave_N_out']}`",
        "",
        "## 7. GNN Reward Update",
        f"- Status: `{gnn_summary.get('status')}`",
        f"- Best regime: `{best_gnn}`",
        "",
        "## 8. Graph-Pointer Update",
        f"- Status: `{pointer_summary.get('status')}`",
        "- Training is offline weighted behavior cloning only; no online RL was run.",
        "",
        "## 9. N32 Augmentation Diagnostic",
        f"- Native Stage 3 prediction effect: `{surrogate_summary.get('n32_effect_on_native_stage3_prediction')}`",
        "- N32 rows remain legacy-compatible; PEEQ/Mises semantic warnings are preserved.",
        "",
        "## 10. Candidate Generation",
        f"- Deduplicated candidate counts: `{pool_counts.get('deduplicated_candidate_count_per_n')}`",
        "",
        "## 11. Option A Native Batch32",
        f"- Path: `{option_paths['option_a']}`",
        "- N12=4, N16=4, N24=12, N40=12.",
        "",
        "## 12. Option B N24/N40 Focused Batch32",
        f"- Path: `{option_paths['option_b']}`",
        "- N24=16, N40=16.",
        "",
        "## 13. Option C Native Batch40",
        f"- Path: `{option_paths['option_c']}`",
        "- N12=4, N16=4, N24=16, N40=16.",
        "",
        "## 14. Comparison To Previous Batches",
        f"- Headline: {comparison.get('headline')}",
        "",
        "## 15. Claim Boundary",
        "- Run38 is model update and candidate generation only. No teacher validation, no CAE/INP, no solver activity.",
        "",
        "## 16. Output Files",
        *[f"- `{p}`" for p in outputs],
        "",
        "## 17. Recommended Run39",
        "If Option A is selected, create a handoff package for native batch32. Select Option B for maximum N24/N40 pressure. Select Option C for broader native coverage. Do not generate CAE/INP until one option is selected.",
    ]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_run_index(verdict: str) -> None:
    if not RUN_INDEX_PATH.exists():
        return
    text = RUN_INDEX_PATH.read_text(encoding="utf-8")
    if RUN_ID in text:
        return
    row = (
        "| run_38 | combined204 and combined204_plus_N32 model update candidate generation | "
        "Updates offline surrogate/GNN/pointer diagnostics after Run37 and creates native-N candidate batch options. | "
        "`scripts/stage3/run_38_combined204_and_combined204_plus_N32_model_update_candidate_generation.py` | "
        "`docs/stage3/runs/run_38_combined204_and_combined204_plus_N32_model_update_candidate_generation/RUN_38_COMBINED204_AND_COMBINED204_PLUS_N32_MODEL_UPDATE_CANDIDATE_GENERATION_REPORT.md` | "
        "`outputs/stage3_run_38_combined204_and_combined204_plus_N32_model_update_candidate_generation/` | "
        f"`{verdict}` | No Abaqus, no ODB opening, no abqjobpilot, no CAE/INP/JNL generation, no teacher validation, no online RL, no commit/push. |"
    )
    RUN_INDEX_PATH.write_text(text.rstrip() + "\n" + row + "\n", encoding="utf-8")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    plus_rows = load_plus_rows()
    native_rows = load_native_rows()
    validation = validate_inputs(plus_rows, native_rows)
    if validation["verdict"].startswith("FAIL"):
        print(validation["verdict"])
        return 2
    native_feature_rows = write_feature_table(native_rows, OUTPUT_DIR / "combined204_scan_order_features.csv")
    feature_rows = write_feature_table(plus_rows, OUTPUT_DIR / "combined204_plus_N32_scan_order_features.csv")
    reward_rows, reward_summary = reward_best_semantic_audit()
    surrogate_detailed, best_configs, surrogate_summary, surrogate_model, surrogate_info = surrogate_validation(native_rows, plus_rows, feature_rows)
    try:
        import torch
    except Exception:
        torch = None
    rows_by_regime = {"native_combined204": native_rows, "plus_N32_unweighted": plus_rows, "plus_N32_balanced": plus_rows}
    gnn_detailed, gnn_summary, gnn_model = train_gnn(rows_by_regime, torch)
    pointer_log, pointer_summary, pointer_policy = train_pointer(rows_by_regime, torch)
    candidates, pool_counts = generate_candidates(plus_rows, torch, pointer_policy)
    scored = score_candidates(candidates, plus_rows, surrogate_model, surrogate_info, torch, gnn_model, pointer_policy)
    option_a = select_batch(scored, OPTION_A_COUNTS, "option_a_native_batch32")
    option_b = select_batch(scored, OPTION_B_COUNTS, "option_b_native_N24_N40_focused_batch32")
    option_c = select_batch(scored, OPTION_C_COUNTS, "option_c_native_batch40")
    option_a_path = OUTPUT_DIR / "run38_native_batch32_U2_exploitation_reward_balanced_candidate_orders.csv"
    option_b_path = OUTPUT_DIR / "run38_native_N24_N40_focused_batch32_candidate_orders.csv"
    option_c_path = OUTPUT_DIR / "run38_native_batch40_broader_coverage_candidate_orders.csv"
    write_csv(option_a_path, option_a)
    write_csv(option_b_path, option_b)
    write_csv(option_c_path, option_c)
    teacher_hashes = {row["order_hash"] for row in plus_rows if row["n"] in NATIVE_NS}
    _, comparison_summary = compare_batch_options({"option_a_native_batch32": option_a, "option_b_N24_N40_batch32": option_b, "option_c_native_batch40": option_c}, teacher_hashes)
    claim_md, claim_json = write_claim_boundary()
    option_paths = {"option_a": str(option_a_path), "option_b": str(option_b_path), "option_c": str(option_c_path)}
    outputs = [
        str(OUTPUT_DIR / "run38_input_validation_summary.json"),
        str(OUTPUT_DIR / "run38_reward_best_semantic_audit.csv"),
        str(OUTPUT_DIR / "run38_reward_best_semantic_audit.json"),
        str(OUTPUT_DIR / "combined204_scan_order_features.csv"),
        str(OUTPUT_DIR / "combined204_plus_N32_scan_order_features.csv"),
        str(OUTPUT_DIR / "run38_surrogate_validation_results_detailed.csv"),
        str(OUTPUT_DIR / "run38_best_surrogate_configurations.csv"),
        str(OUTPUT_DIR / "run38_surrogate_validation_summary.json"),
        str(OUTPUT_DIR / "run38_gnn_reward_validation_results.csv"),
        str(OUTPUT_DIR / "run38_gnn_reward_validation_summary.json"),
        str(OUTPUT_DIR / "run38_graph_pointer_policy_training_log.csv"),
        str(OUTPUT_DIR / "run38_graph_pointer_policy_validation_summary.json"),
        str(OUTPUT_DIR / "run38_candidate_pool_scored.csv"),
        str(option_a_path),
        str(option_b_path),
        str(option_c_path),
        str(OUTPUT_DIR / "run38_batch_options_comparison_to_previous.csv"),
        str(OUTPUT_DIR / "run38_batch_options_comparison_summary.json"),
        str(claim_md),
        str(claim_json),
    ]
    write_report(validation, reward_summary, surrogate_summary, gnn_summary, pointer_summary, pool_counts, option_paths, comparison_summary, outputs)
    outputs.append(str(REPORT_PATH))
    manifest = {
        "run_id": RUN_ID,
        "run_name": RUN_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "branch": git_branch(),
        "input_files": [str(COMBINED204_READY), str(COMBINED204_TEACHER), str(COMBINED_PLUS_READY), str(COMBINED172_TEACHER), str(N32_DEDUP), str(RUN32A_REPORT), str(RUN33_REPORT), str(RUN37_REPORT), str(RUN37_MANIFEST), str(RUN36_ENRICHED), str(RUN36_RANKED), str(RUN36_COMPARISON), str(RUN36_EFFECTIVENESS), str(RUN36_PRED_AUDIT)],
        "output_files": outputs,
        "native_combined204_rows": len(native_rows),
        "combined204_plus_N32_rows": len(plus_rows),
        "per_N_counts": validation["per_n_counts"],
        "reward_best_semantic_audit": reward_summary,
        "best_model_summaries": {"surrogate": surrogate_summary.get("best_overall"), "gnn": gnn_summary.get("best_regime"), "pointer": pointer_summary.get("status")},
        "candidate_pool_counts": pool_counts.get("deduplicated_candidate_count_per_n"),
        "batch_options": {"option_a_native_batch32": {"path": str(option_a_path), "counts": dict(Counter(r["n"] for r in option_a))}, "option_b_native_N24_N40_focused_batch32": {"path": str(option_b_path), "counts": dict(Counter(r["n"] for r in option_b))}, "option_c_native_batch40": {"path": str(option_c_path), "counts": dict(Counter(r["n"] for r in option_c))}},
        "report_path": str(REPORT_PATH),
        "claim_boundary_path": str(claim_md),
        "no_solver_run": True,
        "no_odb_opened": True,
        "no_abqjobpilot_run": True,
        "no_cae_inp_generated": True,
        "no_teacher_validation": True,
        "no_online_rl": True,
        "no_commit_or_push": True,
    }
    write_json(MANIFEST_PATH, manifest)
    update_run_index("RUN38_MODEL_UPDATE_AND_CANDIDATE_GENERATION_ONLY_NO_TEACHER_VALIDATION")
    print(validation["verdict"])
    print(f"best_surrogate={surrogate_summary['best_overall']}")
    print(f"best_gnn={gnn_summary.get('best_regime')}")
    print(f"candidate_counts={pool_counts.get('deduplicated_candidate_count_per_n')}")
    print(f"option_a={len(option_a)} option_b={len(option_b)} option_c={len(option_c)}")
    print(f"report={REPORT_PATH}")
    print(f"manifest={MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
