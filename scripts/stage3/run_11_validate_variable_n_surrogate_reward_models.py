from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
RUN_ID = "run_11_variable_n_surrogate_reward_model_validation"
RUN_NAME = "lightweight variable-N surrogate reward model validation"

SURROGATE_TABLE = ROOT / "outputs" / "stage3_run_10_variable_n_normalized_reward_surrogate_dataset" / "probe60_surrogate_pretraining_table.csv"
REWARD_DATASET = ROOT / "outputs" / "stage3_run_10_variable_n_normalized_reward_surrogate_dataset" / "probe60_variable_n_reward_dataset.csv"
FEATURE_TABLE = ROOT / "outputs" / "stage3_run_10_variable_n_normalized_reward_surrogate_dataset" / "probe60_scan_order_features.csv"
PAIRWISE_DATASET = ROOT / "outputs" / "stage3_run_10_variable_n_normalized_reward_surrogate_dataset" / "probe60_pairwise_preference_dataset.csv"
SPLITS_JSON = ROOT / "outputs" / "stage3_run_10_variable_n_normalized_reward_surrogate_dataset" / "probe60_dataset_splits.json"
RUN10_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_10_variable_n_normalized_reward_surrogate_dataset" / "RUN_10_VARIABLE_N_NORMALIZED_REWARD_SURROGATE_DATASET_REPORT.md"
RUN10_MANIFEST = ROOT / "artifacts" / "manifests" / "stage3_run_10_manifest.json"

OUTPUT_DIR = ROOT / "outputs" / "stage3_run_11_variable_n_surrogate_reward_model_validation"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = ROOT / "docs" / "stage3" / "runs" / RUN_ID
REPORT_PATH = REPORT_DIR / "RUN_11_VARIABLE_N_SURROGATE_REWARD_MODEL_VALIDATION_REPORT.md"
MANIFEST_PATH = ROOT / "artifacts" / "manifests" / "stage3_run_11_manifest.json"
RUN_INDEX_PATH = ROOT / "docs" / "stage3" / "STAGE3_RUN_INDEX.md"

EXPECTED_N = [12, 16, 24, 40]
EXPECTED_TOTAL = 60
EXPECTED_PER_N = 15

TARGETS = [
    "target_reward_mean_all",
    "target_reward_v01_u2_primary",
    "target_reward_v02_safety_weighted",
    "target_reward_v04_penalized",
    "target_reward_v05_lexicographic",
    "target_u2_score_rank",
    "target_peeq_score_rank",
    "target_surfaceT_score_rank",
]
PRIMARY_TARGET = "target_reward_mean_all"

F01_BASIC = [
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
        if not fieldnames:
            fieldnames = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")


def write_table_json(path: Path, rows: list[dict[str, Any]]) -> None:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    payload = {"schema": "columns_and_rows", "columns": columns, "rows": [[row.get(col) for col in columns] for row in rows]}
    write_json(path, payload)


def parse_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def parse_int(value: Any) -> int:
    text = str(value).strip()
    if text.upper().startswith("N"):
        text = text[1:]
    return int(text)


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def safe_divide(numerator: float, denominator: float, default: float = math.nan) -> float:
    return numerator / denominator if denominator else default


def rank_desc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def rank_asc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return math.nan
    return pearson(rank_asc(x), rank_asc(y))


def ndcg_at_k(true_values: np.ndarray, predicted_values: np.ndarray, k: int = 5) -> float:
    if len(true_values) == 0:
        return math.nan
    k = min(k, len(true_values))
    pred_order = np.argsort(-predicted_values)[:k]
    ideal_order = np.argsort(-true_values)[:k]

    def dcg(order: np.ndarray) -> float:
        return float(sum((2 ** true_values[idx] - 1) / math.log2(rank + 2) for rank, idx in enumerate(order)))

    ideal = dcg(ideal_order)
    return safe_divide(dcg(pred_order), ideal, default=math.nan)


def validate_inputs(rows: list[dict[str, str]], splits: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    counts = Counter()
    seen_by_n: dict[int, set[str]] = defaultdict(set)
    required_features = [
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
    headers = set(rows[0].keys()) if rows else set()
    for column in ["n", "strategy_name", *TARGETS, *required_features]:
        if column not in headers:
            errors.append(f"Missing required column: {column}")
    for row in rows:
        try:
            n = parse_int(row.get("n"))
        except (TypeError, ValueError):
            errors.append(f"Invalid N value: {row.get('n')}")
            continue
        counts[n] += 1
        name = row.get("strategy_name", "")
        if not name:
            errors.append("Missing strategy_name")
        if name in seen_by_n[n]:
            errors.append(f"Duplicate strategy_name within N{n}: {name}")
        seen_by_n[n].add(name)
        for target in TARGETS:
            if not math.isfinite(parse_float(row.get(target))):
                errors.append(f"Missing/invalid target {target} for {name}")
        for feature in required_features:
            if not math.isfinite(parse_float(row.get(feature))):
                errors.append(f"Missing/invalid core feature {feature} for {name}")
    if len(rows) != EXPECTED_TOTAL:
        errors.append(f"Expected {EXPECTED_TOTAL} rows, found {len(rows)}")
    if sorted(counts) != EXPECTED_N:
        errors.append(f"Expected N values {EXPECTED_N}, found {sorted(counts)}")
    for n in EXPECTED_N:
        if counts[n] != EXPECTED_PER_N:
            errors.append(f"Expected {EXPECTED_PER_N} rows for N{n}, found {counts[n]}")
    if not splits:
        errors.append(f"Dataset split JSON is empty or missing: {SPLITS_JSON}")
    verdict = "PASS_RUN11_INPUTS_READY_60_ROWS_SURROGATE_TABLE" if not errors else "FAIL_RUN11_INPUTS_INVALID"
    return {"verdict": verdict, "errors": errors, "warnings": warnings, "total_rows": len(rows), "per_n_counts": dict(sorted(counts.items()))}


def is_numeric_column(rows: list[dict[str, str]], column: str) -> bool:
    values = [parse_float(row.get(column)) for row in rows]
    return any(math.isfinite(value) for value in values)


def define_feature_sets(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    exclude_terms = [
        "target",
        "reward",
        "rank",
        "u2",
        "peeq",
        "surface",
        "mises",
        "z_within",
        "percentile",
        "score",
        "pareto",
    ]
    identifiers = {
        "strategy_name",
        "job_name",
        "strategy_id",
        "scan_order_compact",
        "scan_order_json",
        "order_json",
        "policy_source",
        "strategy_family",
        "candidate_group",
        "feature_strategy_name",
        "feature_strategy_family",
        "feature_candidate_group",
        "teacher_validated",
        "trained_policy_used",
    }
    numeric_cols = []
    for column in rows[0]:
        lower = column.lower()
        if column in identifiers:
            continue
        if any(term in lower for term in exclude_terms):
            continue
        if is_numeric_column(rows, column):
            numeric_cols.append(column)
    heuristic_flags = [col for col in numeric_cols if col.startswith("is_")]
    f02 = sorted(set(numeric_cols))
    f04 = sorted(set(col for col in f02 if col not in heuristic_flags and col not in {"feature_n"}))
    f05 = sorted(set(col for col in f04 if col not in {"n", "feature_n", "order_length"}))
    return {
        "F01_basic_order": {"numeric": [col for col in F01_BASIC if col in rows[0]], "categorical": []},
        "F02_full_handcrafted": {"numeric": f02, "categorical": []},
        "F03_family_plus_features": {"numeric": f02, "categorical": ["strategy_family", "candidate_group", "policy_source"]},
        "F04_no_family_generalization": {"numeric": f04, "categorical": []},
        "F05_n_agnostic": {"numeric": f05, "categorical": []},
    }


def row_value(row: dict[str, str], column: str) -> float:
    value = row.get(column)
    if str(value).strip().lower() in {"true", "false"}:
        return 1.0 if parse_bool(value) else 0.0
    return parse_float(value, default=0.0)


def build_matrix(train_rows: list[dict[str, str]], test_rows: list[dict[str, str]], spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    numeric = spec["numeric"]
    categorical = spec["categorical"]
    train_parts = [[row_value(row, col) for col in numeric] for row in train_rows]
    test_parts = [[row_value(row, col) for col in numeric] for row in test_rows]
    names = list(numeric)
    for col in categorical:
        categories = sorted({row.get(col, "") for row in train_rows})
        for cat in categories:
            names.append(f"{col}={cat}")
            for idx, row in enumerate(train_rows):
                train_parts[idx].append(1.0 if row.get(col, "") == cat else 0.0)
            for idx, row in enumerate(test_rows):
                test_parts[idx].append(1.0 if row.get(col, "") == cat else 0.0)
    return np.asarray(train_parts, dtype=float), np.asarray(test_parts, dtype=float), names


class MeanBaseline:
    def fit(self, x: np.ndarray, y: np.ndarray) -> "MeanBaseline":
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(x.shape[0], self.mean_, dtype=float)


def get_regressors() -> dict[str, Any]:
    from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import ElasticNet, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    models: dict[str, Any] = {
        "MeanBaseline": MeanBaseline(),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "ElasticNet": make_pipeline(StandardScaler(), ElasticNet(alpha=0.01, l1_ratio=0.25, max_iter=10000, random_state=42)),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=80, max_depth=4, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "ExtraTreesRegressor": ExtraTreesRegressor(n_estimators=80, max_depth=4, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=60, max_depth=2, learning_rate=0.05, random_state=42),
    }
    return models


def clone_model(model: Any) -> Any:
    if isinstance(model, MeanBaseline):
        return MeanBaseline()
    from sklearn.base import clone

    return clone(model)


def build_protocols(rows: list[dict[str, str]], splits: dict[str, Any]) -> list[dict[str, Any]]:
    by_name = {row["strategy_name"]: row for row in rows}
    protocols: list[dict[str, Any]] = []
    for n in EXPECTED_N:
        test = [row for row in rows if parse_int(row["n"]) == n]
        train = [row for row in rows if parse_int(row["n"]) != n]
        protocols.append({"protocol": "P01_leave_N_out", "split_name": f"test_N{n}", "train_rows": train, "test_rows": test})
    protocols.append({"protocol": "P02_core_generalization", "split_name": "train_N12_N16_N24_test_N40", "train_rows": [r for r in rows if parse_int(r["n"]) in {12, 16, 24}], "test_rows": [r for r in rows if parse_int(r["n"]) == 40]})
    protocols.append({"protocol": "P03_large_N_generalization", "split_name": "train_N12_N16_test_N24_N40", "train_rows": [r for r in rows if parse_int(r["n"]) in {12, 16}], "test_rows": [r for r in rows if parse_int(r["n"]) in {24, 40}]})
    protocols.append({"protocol": "P05_train_small_test_large", "split_name": "train_N12_N16_test_N24_N40", "train_rows": [r for r in rows if parse_int(r["n"]) in {12, 16}], "test_rows": [r for r in rows if parse_int(r["n"]) in {24, 40}]})
    protocols.append({"protocol": "P06_train_large_test_small", "split_name": "train_N24_N40_test_N12_N16", "train_rows": [r for r in rows if parse_int(r["n"]) in {24, 40}], "test_rows": [r for r in rows if parse_int(r["n"]) in {12, 16}]})
    folds = splits.get("random_stratified_5fold", {})
    for name, spec in folds.items():
        test_names = set(spec.get("test_cases", []))
        train_names = set(spec.get("train_cases", []))
        protocols.append({"protocol": "P04_within_N_5fold", "split_name": name, "train_rows": [by_name[n] for n in train_names], "test_rows": [by_name[n] for n in test_names]})
    return protocols


def ranking_metrics(test_rows: list[dict[str, str]], y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    names = [row["strategy_name"] for row in test_rows]
    true_top = names[int(np.argmax(y_true))]
    pred_top = names[int(np.argmax(y_pred))]
    true_order = np.argsort(-y_true)
    pred_order = np.argsort(-y_pred)
    true_top3 = {names[idx] for idx in true_order[: min(3, len(names))]}
    pred_top3 = {names[idx] for idx in pred_order[: min(3, len(names))]}
    true_top5 = {names[idx] for idx in true_order[: min(5, len(names))]}
    pred_top5 = {names[idx] for idx in pred_order[: min(5, len(names))]}
    true_rank = rank_desc(y_true)
    pred_rank = rank_desc(y_pred)
    return {
        "top1_hit": true_top == pred_top,
        "top3_overlap": len(true_top3 & pred_top3),
        "top5_overlap": len(true_top5 & pred_top5),
        "ndcg_at_5": ndcg_at_k(y_true, y_pred, 5),
        "mean_abs_rank_error": float(np.mean(np.abs(true_rank - pred_rank))),
    }


def eval_metrics(test_rows: list[dict[str, str]], y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    residual = y_pred - y_true
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    metrics = {
        "spearman": spearman(y_pred, y_true),
        "pearson": pearson(y_pred, y_true),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(math.sqrt(np.mean(residual ** 2))),
        "r2": 1.0 - safe_divide(ss_res, ss_tot, default=math.nan),
    }
    metrics.update(ranking_metrics(test_rows, y_true, y_pred))
    return metrics


def train_eval(
    rows: list[dict[str, str]],
    feature_sets: dict[str, dict[str, Any]],
    protocols: list[dict[str, Any]],
    targets: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    regressors = get_regressors()
    detailed: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    importances: list[dict[str, Any]] = []
    best_models: dict[str, Any] = {}

    for target in targets:
        for feature_set_name, spec in feature_sets.items():
            for model_name, base_model in regressors.items():
                for protocol_spec in protocols:
                    train_rows = protocol_spec["train_rows"]
                    test_rows = protocol_spec["test_rows"]
                    if not train_rows or not test_rows:
                        continue
                    x_train, x_test, feature_names = build_matrix(train_rows, test_rows, spec)
                    y_train = np.asarray([parse_float(row[target]) for row in train_rows], dtype=float)
                    y_test = np.asarray([parse_float(row[target]) for row in test_rows], dtype=float)
                    model = clone_model(base_model)
                    try:
                        model.fit(x_train, y_train)
                        y_pred = np.asarray(model.predict(x_test), dtype=float)
                    except Exception as exc:
                        detailed.append(
                            {
                                "target": target,
                                "feature_set": feature_set_name,
                                "model_name": model_name,
                                "protocol": protocol_spec["protocol"],
                                "split_name": protocol_spec["split_name"],
                                "test_n": "ALL",
                                "status": "MODEL_FAILED",
                                "warning": repr(exc),
                            }
                        )
                        continue
                    metrics = eval_metrics(test_rows, y_test, y_pred)
                    test_ns = sorted({parse_int(row["n"]) for row in test_rows})
                    detailed.append(
                        {
                            "target": target,
                            "feature_set": feature_set_name,
                            "model_name": model_name,
                            "protocol": protocol_spec["protocol"],
                            "split_name": protocol_spec["split_name"],
                            "test_n": "ALL" if len(test_ns) > 1 else f"N{test_ns[0]}",
                            "test_count": len(test_rows),
                            "status": "OK",
                            **metrics,
                        }
                    )
                    if target == PRIMARY_TARGET and protocol_spec["protocol"] == "P01_leave_N_out":
                        best_models[(feature_set_name, model_name, protocol_spec["split_name"])] = (model, feature_names)
                        importances.extend(extract_importance(model, feature_names, target, feature_set_name, model_name, protocol_spec["protocol"], protocol_spec["split_name"]))
    return detailed, predictions, importances, best_models


def extract_importance(model: Any, feature_names: list[str], target: str, feature_set: str, model_name: str, protocol: str, split_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    estimator = model
    if hasattr(model, "named_steps"):
        estimator = list(model.named_steps.values())[-1]
    values = None
    kind = ""
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
        kind = "tree_feature_importance"
    elif hasattr(estimator, "coef_"):
        values = np.ravel(estimator.coef_)
        kind = "standardized_linear_coefficient"
    if values is None:
        return rows
    for name, value in zip(feature_names, values):
        rows.append(
            {
                "target": target,
                "feature_set": feature_set,
                "model_name": model_name,
                "protocol": protocol,
                "split_name": split_name,
                "importance_type": kind,
                "feature_name": name,
                "value": float(value),
                "abs_value": float(abs(value)),
            }
        )
    return rows


def add_leave_n_macro_rows(detailed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in detailed:
        if row.get("protocol") == "P01_leave_N_out" and row.get("status") == "OK":
            grouped[(row["target"], row["feature_set"], row["model_name"])].append(row)
    macro_rows: list[dict[str, Any]] = []
    metric_cols = ["spearman", "pearson", "mae", "rmse", "r2", "top1_hit", "top3_overlap", "top5_overlap", "ndcg_at_5", "mean_abs_rank_error"]
    for (target, feature_set, model_name), rows in grouped.items():
        if len(rows) != 4:
            continue
        macro = {
            "target": target,
            "feature_set": feature_set,
            "model_name": model_name,
            "protocol": "P01_leave_N_out_macro",
            "split_name": "macro_over_N12_N16_N24_N40",
            "test_n": "MACRO",
            "test_count": sum(int(row["test_count"]) for row in rows),
            "status": "OK",
        }
        for col in metric_cols:
            values = []
            for row in rows:
                value = row.get(col)
                if isinstance(value, bool):
                    values.append(float(value))
                elif isinstance(value, (int, float)) and math.isfinite(float(value)):
                    values.append(float(value))
            macro[col] = float(np.mean(values)) if values else math.nan
        macro_rows.append(macro)
    return detailed + macro_rows


def select_best_configs(detailed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [row for row in detailed if row.get("protocol") == "P01_leave_N_out_macro" and row.get("status") == "OK"]
    best_rows: list[dict[str, Any]] = []
    for target in TARGETS:
        target_rows = [row for row in candidates if row["target"] == target]
        if not target_rows:
            continue
        target_rows.sort(
            key=lambda row: (
                -(-1e9 if not math.isfinite(float(row.get("spearman", math.nan))) else float(row["spearman"])),
                -float(row.get("top5_overlap", 0.0)),
                float(row.get("mae", 1e9)),
                simplicity_rank(row["model_name"]),
            )
        )
        best = target_rows[0].copy()
        best["selection_criterion"] = "leave-N-out macro Spearman, then top5 overlap, MAE, model simplicity"
        best_rows.append(best)
    return best_rows


def build_predictions_for_config(
    feature_sets: dict[str, dict[str, Any]],
    protocols: list[dict[str, Any]],
    feature_set_name: str,
    model_name: str,
) -> list[dict[str, Any]]:
    regressors = get_regressors()
    predictions: list[dict[str, Any]] = []
    spec = feature_sets[feature_set_name]
    for protocol_spec in protocols:
        train_rows = protocol_spec["train_rows"]
        test_rows = protocol_spec["test_rows"]
        x_train, x_test, _feature_names = build_matrix(train_rows, test_rows, spec)
        y_train = np.asarray([parse_float(row[PRIMARY_TARGET]) for row in train_rows], dtype=float)
        y_test = np.asarray([parse_float(row[PRIMARY_TARGET]) for row in test_rows], dtype=float)
        model = clone_model(regressors[model_name])
        model.fit(x_train, y_train)
        y_pred = np.asarray(model.predict(x_test), dtype=float)
        true_rank = rank_desc(y_test)
        pred_rank = rank_desc(y_pred)
        for idx, row in enumerate(test_rows):
            predictions.append(
                {
                    "protocol": protocol_spec["protocol"],
                    "split_name": protocol_spec["split_name"],
                    "model_name": model_name,
                    "feature_set": feature_set_name,
                    "test_n": row["n"],
                    "strategy_name": row["strategy_name"],
                    "true_target_reward_mean_all": y_test[idx],
                    "pred_target_reward_mean_all": y_pred[idx],
                    "true_rank_within_test_n": true_rank[idx],
                    "pred_rank_within_test_n": pred_rank[idx],
                    "rank_error": abs(true_rank[idx] - pred_rank[idx]),
                    "is_true_top5": true_rank[idx] <= 5,
                    "is_pred_top5": pred_rank[idx] <= 5,
                }
            )
    return predictions


def simplicity_rank(model_name: str) -> int:
    order = {
        "MeanBaseline": 0,
        "Ridge": 1,
        "ElasticNet": 2,
        "KNeighborsRegressor": 3,
        "GradientBoostingRegressor": 4,
        "HistGradientBoostingRegressor": 5,
        "RandomForestRegressor": 6,
        "ExtraTreesRegressor": 7,
    }
    return order.get(model_name, 99)


def pairwise_validation(rows: list[dict[str, str]], pairwise_rows: list[dict[str, str]], feature_sets: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, roc_auc_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        counts = Counter(row.get("preferred_by_reward_mean", "") for row in pairwise_rows)
        return ([{"status": "WARNING_PAIRWISE_MODEL_SKIPPED", "reason": repr(exc), "pair_count": len(pairwise_rows), **counts}], {"status": "WARNING_PAIRWISE_MODEL_SKIPPED"})

    row_by_name = {row["strategy_name"]: row for row in rows}
    spec = feature_sets["F05_n_agnostic"]
    results: list[dict[str, Any]] = []
    class_counts = Counter(row.get("preferred_by_reward_mean", "") for row in pairwise_rows)
    results.append({"status": "PAIRWISE_CLASS_BALANCE", "pair_count": len(pairwise_rows), **dict(class_counts)})

    def pair_features(pair_rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
        xs = []
        ys = []
        for pair in pair_rows:
            if pair.get("preferred_by_reward_mean") == "tie":
                continue
            left = row_by_name[pair["case_i"]]
            right = row_by_name[pair["case_j"]]
            x_left, _, names = build_matrix([left], [left], spec)
            x_right, _, _ = build_matrix([right], [right], spec)
            xs.append((x_left[0] - x_right[0]).tolist())
            ys.append(1 if pair.get("preferred_by_reward_mean") == "i" else 0)
        return np.asarray(xs, dtype=float), np.asarray(ys, dtype=int)

    protocols = []
    for n in EXPECTED_N:
        protocols.append((f"leave_N_out_test_N{n}", [p for p in pairwise_rows if parse_int(p["n"]) != n], [p for p in pairwise_rows if parse_int(p["n"]) == n]))
    classifiers = {
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=2, random_state=42),
    }
    for split_name, train_pairs, test_pairs in protocols:
        x_train, y_train = pair_features(train_pairs)
        x_test, y_test = pair_features(test_pairs)
        if len(np.unique(y_train)) < 2 or len(y_test) == 0:
            continue
        for model_name, model in classifiers.items():
            clf = clone_model(model)
            clf.fit(x_train, y_train)
            pred = clf.predict(x_test)
            if hasattr(clf, "predict_proba"):
                score = clf.predict_proba(x_test)[:, 1]
            else:
                score = pred
            try:
                auc = roc_auc_score(y_test, score) if len(np.unique(y_test)) > 1 else math.nan
            except Exception:
                auc = math.nan
            results.append(
                {
                    "status": "OK",
                    "protocol": "pairwise_leave_N_out",
                    "split_name": split_name,
                    "model_name": model_name,
                    "pair_count": len(y_test),
                    "accuracy": float(accuracy_score(y_test, pred)),
                    "roc_auc": auc,
                }
            )
    ok_rows = [row for row in results if row.get("status") == "OK"]
    summary = {
        "status": "PAIRWISE_MODEL_VALIDATED" if ok_rows else "WARNING_PAIRWISE_MODEL_SKIPPED",
        "pair_count": len(pairwise_rows),
        "class_balance": dict(class_counts),
        "best_accuracy": max([row["accuracy"] for row in ok_rows], default=math.nan),
        "best_auc": max([row["roc_auc"] for row in ok_rows if math.isfinite(row["roc_auc"])], default=math.nan),
    }
    return results, summary


def maybe_plot(detailed: list[dict[str, Any]], predictions: list[dict[str, Any]], importances: list[dict[str, Any]], best_primary: dict[str, Any] | None) -> list[str]:
    written: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return [f"PLOTTING_SKIPPED: {exc}"]
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    if best_primary:
        pred_rows = [
            row
            for row in predictions
            if row["protocol"] == "P01_leave_N_out"
            and row["model_name"] == best_primary["model_name"]
            and row["feature_set"] == best_primary["feature_set"]
        ]
        if pred_rows:
            plt.figure(figsize=(5, 5))
            plt.scatter([float(row["true_target_reward_mean_all"]) for row in pred_rows], [float(row["pred_target_reward_mean_all"]) for row in pred_rows], s=28)
            plt.xlabel("true reward_mean_all")
            plt.ylabel("predicted reward_mean_all")
            plt.title("Best Leave-N-Out Surrogate")
            path = FIGURE_DIR / "best_leave_N_out_predicted_vs_true_reward_mean_all.png"
            plt.tight_layout()
            plt.savefig(path, dpi=160)
            plt.close()
            written.append(str(path))

            plt.figure(figsize=(6, 4))
            plt.scatter([float(row["true_rank_within_test_n"]) for row in pred_rows], [float(row["pred_rank_within_test_n"]) for row in pred_rows], s=28)
            plt.xlabel("true rank within test fold")
            plt.ylabel("predicted rank within test fold")
            plt.title("Predicted vs True Rank")
            path = FIGURE_DIR / "best_leave_N_out_predicted_rank_vs_true_rank.png"
            plt.tight_layout()
            plt.savefig(path, dpi=160)
            plt.close()
            written.append(str(path))
    imp_rows = sorted([row for row in importances if row.get("target") == PRIMARY_TARGET], key=lambda row: float(row.get("abs_value", 0.0)), reverse=True)[:15]
    if imp_rows:
        plt.figure(figsize=(8, 4))
        plt.bar([row["feature_name"] for row in imp_rows], [float(row["value"]) for row in imp_rows])
        plt.xticks(rotation=65, ha="right", fontsize=8)
        plt.ylabel("importance / coefficient")
        plt.title("Top Diagnostic Feature Importances")
        path = FIGURE_DIR / "top_feature_importance_bar.png"
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        written.append(str(path))
    lno = [row for row in detailed if row.get("target") == PRIMARY_TARGET and row.get("protocol") == "P01_leave_N_out_macro" and row.get("status") == "OK"]
    if lno:
        top = sorted(lno, key=lambda row: float(row.get("spearman", -999)), reverse=True)[:20]
        plt.figure(figsize=(8, 5))
        labels = [f"{row['model_name']}\n{row['feature_set'].replace('_', ' ')}" for row in top]
        plt.bar(range(len(top)), [float(row["spearman"]) for row in top])
        plt.xticks(range(len(top)), labels, rotation=75, ha="right", fontsize=6)
        plt.ylabel("macro Spearman")
        plt.title("Leave-N-Out Spearman by Model")
        path = FIGURE_DIR / "leave_N_out_spearman_by_model.png"
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        written.append(str(path))
        plt.figure(figsize=(8, 5))
        plt.bar(range(len(top)), [float(row["top5_overlap"]) for row in top])
        plt.xticks(range(len(top)), labels, rotation=75, ha="right", fontsize=6)
        plt.ylabel("macro top5 overlap")
        plt.title("Leave-N-Out Top5 Overlap")
        path = FIGURE_DIR / "leave_N_out_top5_overlap_by_model.png"
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        written.append(str(path))
    return written


def write_claim_boundary(path_md: Path, path_json: Path) -> None:
    safe = [
        "Run11 evaluates lightweight feature-based surrogates on the 60 teacher-labelled variable-N dataset.",
        "Leave-N-out validation provides preliminary diagnostic evidence of whether scan-order features transfer across N.",
        "Positive results only indicate learnable signal in simple handcrafted features for within-N normalized reward.",
        "Weak results motivate active learning or richer graph/sequence models.",
        "Pairwise preference data can support later ranking-policy training.",
    ]
    unsafe = [
        "Do not claim trained variable-N RL policy superiority.",
        "Do not claim final surrogate accuracy.",
        "Do not claim arbitrary-N generalization.",
        "Do not claim a physical optimum.",
        "Do not claim readiness to deploy.",
        "Do not claim feature importances are causal.",
        "Do not claim proxy/fallback policy is equivalent to trained RL.",
    ]
    lines = ["# Run 11 Claim Boundary", "", "## Safe Claims", ""]
    lines.extend(f"- {item}" for item in safe)
    lines.extend(["", "## Unsafe Claims", ""])
    lines.extend(f"- {item}" for item in unsafe)
    path_md.parent.mkdir(parents=True, exist_ok=True)
    path_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(path_json, {"verdict": "RUN11_SURROGATE_FEASIBILITY_VALIDATION_ONLY_NO_RL_POLICY_TRAINING", "safe_claims": safe, "unsafe_claims": unsafe})


def update_run_index(verdict: str) -> None:
    if not RUN_INDEX_PATH.exists():
        return
    entry = (
        "| run_11 | Variable-N surrogate reward model validation | Validate lightweight feature-based surrogate models on within-N normalized reward targets with leave-N-out and small split diagnostics. | "
        "`scripts/stage3/run_11_validate_variable_n_surrogate_reward_models.py` | "
        "`docs/stage3/runs/run_11_variable_n_surrogate_reward_model_validation/RUN_11_VARIABLE_N_SURROGATE_REWARD_MODEL_VALIDATION_REPORT.md` | "
        "`outputs/stage3_run_11_variable_n_surrogate_reward_model_validation/` | "
        f"`{verdict}` | No Abaqus, no datacheck, no ODB opening, no abqjobpilot, no final RL policy training, no commit/push. Next: run12 selected from diagnostic outcome. |"
    )
    lines = RUN_INDEX_PATH.read_text(encoding="utf-8").splitlines()
    for idx, line in enumerate(lines):
        if line.startswith("| run_11 | Variable-N surrogate reward model validation |"):
            lines[idx] = entry
            break
    else:
        lines.append(entry)
    RUN_INDEX_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(
    validation: dict[str, Any],
    feature_sets: dict[str, dict[str, Any]],
    detailed: list[dict[str, Any]],
    best_configs: list[dict[str, Any]],
    pairwise_summary: dict[str, Any],
    best_primary: dict[str, Any] | None,
    output_files: list[str],
) -> None:
    best_text = best_primary or {}
    p01_rows = [row for row in detailed if row.get("target") == PRIMARY_TARGET and row.get("protocol") == "P01_leave_N_out" and row.get("feature_set") == best_text.get("feature_set") and row.get("model_name") == best_text.get("model_name")]
    lines = [
        "# Stage 3 Run 11 - Variable-N Surrogate Reward Model Validation",
        "",
        "## Purpose",
        "Evaluate whether lightweight feature-based surrogate models can predict within-N normalized rewards and ranks from scan-order features and metadata.",
        "",
        "## Inputs",
        f"- `{SURROGATE_TABLE}`",
        f"- `{REWARD_DATASET}`",
        f"- `{FEATURE_TABLE}`",
        f"- `{PAIRWISE_DATASET}`",
        f"- `{SPLITS_JSON}`",
        f"- `{RUN10_REPORT}`",
        f"- `{RUN10_MANIFEST}`",
        "",
        "## Validation Status",
        f"- `{validation['verdict']}`",
        f"- Rows: {validation['total_rows']}",
        f"- Per-N counts: {validation['per_n_counts']}",
        "",
        "## Feature Sets",
    ]
    for name, spec in feature_sets.items():
        lines.append(f"- `{name}`: {len(spec['numeric'])} numeric, {len(spec['categorical'])} categorical features.")
    lines += [
        "",
        "## Targets",
        "- `target_reward_mean_all` is the primary target.",
        "- Secondary targets include reward variants and U2/PEEQ/SurfaceT rank scores.",
        "",
        "## Models",
        "- MeanBaseline, Ridge, ElasticNet, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor when available, and KNeighborsRegressor.",
        "",
        "## Validation Protocols",
        "- P01 leave-N-out, P02 core generalization, P03/P05 small-to-large, P06 large-to-small, and P04 run10 stratified five-fold.",
        "",
        "## Main Leave-N-Out Results",
        f"- Best diagnostic primary configuration: `{best_text.get('model_name', 'NA')}` with `{best_text.get('feature_set', 'NA')}`.",
        f"- Macro Spearman: {best_text.get('spearman', 'NA')}",
        f"- Macro top5 overlap: {best_text.get('top5_overlap', 'NA')}",
        "",
        "### Held-Out N Results For Best Primary Configuration",
    ]
    for row in p01_rows:
        lines.append(f"- {row['split_name']}: Spearman={row.get('spearman')}, top5_overlap={row.get('top5_overlap')}, MAE={row.get('mae')}")
    lines += [
        "",
        "## Best Diagnostic Surrogate Configurations",
    ]
    for row in best_configs:
        lines.append(f"- `{row['target']}`: `{row['model_name']}` / `{row['feature_set']}`, macro Spearman={row.get('spearman')}, top5={row.get('top5_overlap')}")
    lines += [
        "",
        "## Prediction / Ranking Diagnostics",
        "- Prediction tables are saved for the primary target across all protocols.",
        "",
        "## Feature Importance Diagnostics",
        "- Tree feature importances and standardized linear coefficients are reported as diagnostic only, not causal.",
        "",
        "## Pairwise Preference Baseline",
        f"- Status: `{pairwise_summary.get('status')}`",
        f"- Pair count: {pairwise_summary.get('pair_count')}",
        f"- Best accuracy: {pairwise_summary.get('best_accuracy')}",
        f"- Best AUC: {pairwise_summary.get('best_auc')}",
        "",
        "## Failure Modes and Limitations",
        "- Only 60 teacher-labelled cases are available.",
        "- Leave-N-out test folds have 15 cases each, so R2 and rank metrics can be unstable.",
        "- Handcrafted features may miss sequence/graph interactions.",
        "- Family labels can inflate interpolation-style diagnostics and should not be treated as physical mechanisms.",
        "",
        "## Claim Boundary",
        "- This is surrogate feasibility validation only.",
        "- It does not train the final RL policy.",
        "- It does not prove arbitrary-N generalization, physical optimality, or deployment readiness.",
        "",
        "## Output Files",
    ]
    lines.extend(f"- `{path}`" for path in output_files)
    recommendation = "run12 should perform active-learning design: propose a small additional teacher batch per N to improve surrogate coverage."
    if best_primary and math.isfinite(float(best_primary.get("spearman", math.nan))) and float(best_primary["spearman"]) > 0.25 and float(best_primary.get("top5_overlap", 0.0)) >= 2.0:
        recommendation = "run12 should generate candidate scan orders using the best diagnostic surrogate as an offline screening model, but still not run Abaqus."
    lines += ["", "## Recommended Run12", recommendation]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def git_branch() -> str:
    try:
        result = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception:
        return "UNKNOWN"


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = read_csv(SURROGATE_TABLE)
    splits = json.loads(SPLITS_JSON.read_text(encoding="utf-8")) if SPLITS_JSON.exists() else {}
    validation = validate_inputs(rows, splits)
    validation_path = OUTPUT_DIR / "run11_input_validation_summary.json"
    write_json(validation_path, validation)
    if validation["verdict"].startswith("FAIL"):
        print(validation["verdict"])
        print(json.dumps(validation, indent=2))
        return 2

    feature_sets = define_feature_sets(rows)
    feature_defs_path = OUTPUT_DIR / "run11_feature_set_definitions.json"
    write_json(feature_defs_path, feature_sets)
    protocols = build_protocols(rows, splits)
    detailed, predictions, importances, _best_models = train_eval(rows, feature_sets, protocols, TARGETS)
    detailed = add_leave_n_macro_rows(detailed)
    best_configs = select_best_configs(detailed)
    best_primary = next((row for row in best_configs if row["target"] == PRIMARY_TARGET), None)
    if best_primary:
        predictions = build_predictions_for_config(feature_sets, protocols, best_primary["feature_set"], best_primary["model_name"])

    pairwise_rows = read_csv(PAIRWISE_DATASET)
    pairwise_results, pairwise_summary = pairwise_validation(rows, pairwise_rows, feature_sets)

    plots = maybe_plot(detailed, predictions, importances, best_primary)
    claim_md = OUTPUT_DIR / "run11_claim_boundary.md"
    claim_json = OUTPUT_DIR / "run11_claim_boundary.json"
    write_claim_boundary(claim_md, claim_json)

    detailed_csv = OUTPUT_DIR / "surrogate_validation_results_detailed.csv"
    detailed_json = OUTPUT_DIR / "surrogate_validation_results_detailed.json"
    best_csv = OUTPUT_DIR / "best_surrogate_configurations.csv"
    best_json = OUTPUT_DIR / "best_surrogate_configurations.json"
    pred_csv = OUTPUT_DIR / "surrogate_predictions_target_reward_mean_all.csv"
    importance_csv = OUTPUT_DIR / "feature_importance_summary.csv"
    importance_json = OUTPUT_DIR / "feature_importance_summary.json"
    pairwise_csv = OUTPUT_DIR / "pairwise_preference_validation_summary.csv"
    pairwise_json = OUTPUT_DIR / "pairwise_preference_validation_summary.json"

    write_csv(detailed_csv, detailed)
    write_table_json(detailed_json, detailed)
    write_csv(best_csv, best_configs)
    write_json(best_json, best_configs)
    write_csv(pred_csv, predictions)
    write_csv(importance_csv, importances)
    write_json(importance_json, importances)
    write_csv(pairwise_csv, pairwise_results)
    write_json(pairwise_json, {"summary": pairwise_summary, "results": pairwise_results})

    output_files = [
        str(validation_path),
        str(feature_defs_path),
        str(detailed_csv),
        str(detailed_json),
        str(best_csv),
        str(best_json),
        str(pred_csv),
        str(importance_csv),
        str(importance_json),
        str(pairwise_csv),
        str(pairwise_json),
        str(claim_md),
        str(claim_json),
        *[path for path in plots if not path.startswith("PLOTTING_SKIPPED")],
        str(REPORT_PATH),
    ]
    write_report(validation, feature_sets, detailed, best_configs, pairwise_summary, best_primary, output_files)
    update_run_index(validation["verdict"])
    manifest = {
        "run_id": RUN_ID,
        "run_name": RUN_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "branch": git_branch(),
        "script_path": str(Path(__file__).resolve()),
        "input_files": [
            str(SURROGATE_TABLE),
            str(REWARD_DATASET),
            str(FEATURE_TABLE),
            str(PAIRWISE_DATASET),
            str(SPLITS_JSON),
            str(RUN10_REPORT),
            str(RUN10_MANIFEST),
        ],
        "output_files": output_files,
        "report_path": str(REPORT_PATH),
        "claim_boundary_path": str(claim_md),
        "validation_verdict": validation["verdict"],
        "best_model_summary_path": str(best_csv),
        "best_primary_configuration": best_primary,
        "pairwise_summary": pairwise_summary,
        "no_solver_run": True,
        "no_odb_opened": True,
        "no_abqjobpilot_run": True,
        "no_rl_policy_training": True,
        "no_commit_or_push": True,
    }
    write_json(MANIFEST_PATH, manifest)

    print(validation["verdict"])
    print(f"rows={validation['total_rows']}")
    print(f"per_n_counts={validation['per_n_counts']}")
    if best_primary:
        print(f"best_primary={best_primary['model_name']}|{best_primary['feature_set']}|spearman={best_primary.get('spearman')}|top5={best_primary.get('top5_overlap')}")
    print(f"pairwise_status={pairwise_summary.get('status')}")
    print(f"report={REPORT_PATH}")
    print(f"manifest={MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
