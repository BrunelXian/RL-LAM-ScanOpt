from __future__ import annotations

import csv
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
RUN_ID = "run_17_combined80_surrogate_reward_model_validation_update"
RUN_NAME = "combined80 lightweight variable-N surrogate validation update"

COMBINED80_READY = ROOT / "outputs" / "stage3_run_16_batch20_teacher_metrics_ingestion_and_combined80_ranking" / "combined80_RL_ready_dataset.csv"
COMBINED80_TEACHER = ROOT / "outputs" / "stage3_run_16_batch20_teacher_metrics_ingestion_and_combined80_ranking" / "combined80_teacher_dataset.csv"
COMBINED80_LEADERBOARD = ROOT / "outputs" / "stage3_run_16_batch20_teacher_metrics_ingestion_and_combined80_ranking" / "combined80_per_N_leaderboard.csv"
RUN16_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_16_batch20_teacher_metrics_ingestion_and_combined80_ranking" / "RUN_16_BATCH20_TEACHER_METRICS_INGESTION_AND_COMBINED80_RANKING_REPORT.md"
RUN11_DETAILED = ROOT / "outputs" / "stage3_run_11_variable_n_surrogate_reward_model_validation" / "surrogate_validation_results_detailed.csv"
RUN11_BEST = ROOT / "outputs" / "stage3_run_11_variable_n_surrogate_reward_model_validation" / "best_surrogate_configurations.csv"
RUN11_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_11_variable_n_surrogate_reward_model_validation" / "RUN_11_VARIABLE_N_SURROGATE_REWARD_MODEL_VALIDATION_REPORT.md"
RUN10_FEATURES = ROOT / "outputs" / "stage3_run_10_variable_n_normalized_reward_surrogate_dataset" / "probe60_scan_order_features.csv"
RUN13_HANDOFF = ROOT / "outputs" / "stage3_run_13_batch20_surrogate_screened_teacher_handoff" / "stage3_run13_batch20_candidate_orders.csv"
RUN12_SCORED = ROOT / "outputs" / "stage3_run_12_offline_surrogate_screened_candidate_generation" / "run12_candidate_pool_scored.csv"

OUTPUT_DIR = ROOT / "outputs" / "stage3_run_17_combined80_surrogate_reward_model_validation_update"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = ROOT / "docs" / "stage3" / "runs" / RUN_ID
REPORT_PATH = REPORT_DIR / "RUN_17_COMBINED80_SURROGATE_REWARD_MODEL_VALIDATION_UPDATE_REPORT.md"
MANIFEST_PATH = ROOT / "artifacts" / "manifests" / "stage3_run_17_manifest.json"
RUN_INDEX_PATH = ROOT / "docs" / "stage3" / "STAGE3_RUN_INDEX.md"

EXPECTED_N = [12, 16, 24, 40]
PRIMARY_TARGET = "target_reward_combined80_u2_primary"
TARGETS = [
    PRIMARY_TARGET,
    "target_u2_score_combined80_rank",
    "target_peeq_score_combined80_rank",
    "target_surfaceT_score_combined80_rank",
    "target_mises_score_combined80_rank",
]
RUN11_BEST_SPEARMAN = 0.7678571428571428
RUN11_BEST_TOP5 = 3.75


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
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")


def write_table_json(path: Path, rows: list[dict[str, Any]]) -> None:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    write_json(path, {"schema": "columns_and_rows", "columns": columns, "rows": [[row.get(col) for col in columns] for row in rows]})


def parse_int(value: Any) -> int:
    text = str(value).strip()
    if text.upper().startswith("N"):
        text = text[1:]
    return int(text)


def parse_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def safe_divide(num: float, den: float, default: float = 0.0) -> float:
    return num / den if den else default


def mean(vals: list[float], default: float = 0.0) -> float:
    clean = [x for x in vals if math.isfinite(x)]
    return statistics.fmean(clean) if clean else default


def std(vals: list[float]) -> float:
    clean = [x for x in vals if math.isfinite(x)]
    return statistics.pstdev(clean) if len(clean) > 1 else 0.0


def median(vals: list[float], default: float = 0.0) -> float:
    clean = [x for x in vals if math.isfinite(x)]
    return statistics.median(clean) if clean else default


def entropy(vals: list[int]) -> float:
    if not vals:
        return 0.0
    counts = Counter(vals)
    total = len(vals)
    return -sum((count / total) * math.log(count / total, 2) for count in counts.values())


def parse_order(text: str) -> list[int] | None:
    try:
        val = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(val, list):
        return None
    try:
        return [int(x) for x in val]
    except (TypeError, ValueError):
        return None


def max_unvisited_gap(order: list[int], n: int) -> tuple[float, float]:
    visited: set[int] = set()
    gaps = []
    full = set(range(n))
    for track in order:
        visited.add(track)
        unvisited = sorted(full - visited)
        if not unvisited:
            gaps.append(0)
            continue
        cur = 1
        mx = 1
        for a, b in zip(unvisited, unvisited[1:]):
            if b == a + 1:
                cur += 1
                mx = max(mx, cur)
            else:
                cur = 1
        gaps.append(mx)
    return mean([g / n for g in gaps]), max([g / n for g in gaps], default=0.0)


def order_features(row: dict[str, str]) -> dict[str, Any]:
    n = parse_int(row["n"])
    order = parse_order(row.get("order_json", ""))
    if order is None:
        raise ValueError(f"Missing/invalid order_json for {row.get('strategy_name')}")
    center = (n - 1) / 2.0
    jumps = [abs(b - a) for a, b in zip(order, order[1:])]
    signed = [b - a for a, b in zip(order, order[1:])]
    q = max(1, math.ceil(0.25 * n))
    early = order[:q]
    late = order[-q:]
    outer = set(range(0, max(1, n // 4))) | set(range(n - max(1, n // 4), n))
    center_tracks = set(range(n // 4, n - n // 4))
    parity_switches = sum(1 for a, b in zip(order, order[1:]) if (a % 2) != (b % 2))
    same_sign = sum(1 for a, b in zip(signed, signed[1:]) if a and b and (a > 0) == (b > 0))
    reversals = sum(1 for a, b in zip(signed, signed[1:]) if a and b and (a > 0) != (b > 0))
    gap_mean, gap_max = max_unvisited_gap(order, n)
    return {
        "n": n,
        "strategy_name": row["strategy_name"],
        "dataset_source": row.get("dataset_source", ""),
        "candidate_family": row.get("candidate_family", ""),
        "selection_bucket": row.get("selection_bucket", ""),
        "first_track": order[0],
        "last_track": order[-1],
        "center_track_index": center,
        "first_track_norm": safe_divide(order[0], n - 1),
        "last_track_norm": safe_divide(order[-1], n - 1),
        "mean_jump": mean(jumps),
        "median_jump": median(jumps),
        "max_jump": max(jumps, default=0),
        "min_jump": min(jumps, default=0),
        "std_jump": std([float(j) for j in jumps]),
        "total_jump": sum(jumps),
        "normalized_mean_jump": safe_divide(mean(jumps), n - 1),
        "normalized_max_jump": safe_divide(max(jumps, default=0), n - 1),
        "adjacent_jump_count": sum(1 for jump in jumps if jump == 1),
        "long_jump_count": sum(1 for jump in jumps if jump >= n / 2),
        "jump_entropy": entropy(jumps),
        "running_center_distance_mean": mean([abs(track - center) / max(1.0, center) for track in order]),
        "early_center_bias": mean([abs(track - center) / max(1.0, center) for track in early]),
        "late_center_bias": mean([abs(track - center) / max(1.0, center) for track in late]),
        "edge_early_count": sum(1 for track in early if track in outer),
        "center_early_count": sum(1 for track in early if track in center_tracks),
        "odd_even_transition_count": parity_switches,
        "parity_switch_rate": safe_divide(parity_switches, max(1, n - 1)),
        "monotonicity_fraction": safe_divide(same_sign, max(1, len(signed) - 1)),
        "direction_reversal_count": reversals,
        "max_unvisited_gap_proxy_mean_norm": gap_mean,
        "max_unvisited_gap_proxy_max_norm": gap_max,
    }


def validate_inputs(rows: list[dict[str, str]]) -> dict[str, Any]:
    errors: list[str] = []
    counts = Counter()
    by_n_names: dict[int, set[str]] = defaultdict(set)
    required = ["u2_range", "peeq_max", "surface_t_proxy", "mises_max", *TARGETS]
    for row in rows:
        try:
            n = parse_int(row["n"])
        except Exception:
            errors.append(f"Invalid N: {row.get('n')}")
            continue
        counts[n] += 1
        name = row.get("strategy_name", "")
        if name in by_n_names[n]:
            errors.append(f"Duplicate strategy_name within N{n}: {name}")
        by_n_names[n].add(name)
        for col in required:
            if not math.isfinite(parse_float(row.get(col))):
                errors.append(f"{name}: missing/invalid {col}")
        order = parse_order(row.get("order_json", ""))
        if order is None or len(order) != n or set(order) != set(range(n)):
            errors.append(f"{name}: invalid order_json")
    sources = set(row.get("dataset_source", "") for row in rows)
    if len(rows) != 80:
        errors.append(f"Expected 80 rows, found {len(rows)}")
    if sorted(counts) != EXPECTED_N:
        errors.append(f"Expected N values {EXPECTED_N}, found {sorted(counts)}")
    for n in EXPECTED_N:
        if counts[n] != 20:
            errors.append(f"Expected 20 rows for N{n}, found {counts[n]}")
    if not {"probe60_run08", "batch20_run14"}.issubset(sources):
        errors.append(f"Missing expected dataset sources; found {sorted(sources)}")
    verdict = "PASS_RUN17_INPUTS_READY_80_ROWS_COMBINED_SURROGATE_TABLE" if not errors else "FAIL_RUN17_INPUTS_INVALID"
    return {"verdict": verdict, "errors": errors, "total_rows": len(rows), "per_n_counts": dict(sorted(counts.items())), "dataset_sources": sorted(sources)}


def build_model_rows(rows: list[dict[str, str]], features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name = {f["strategy_name"]: f for f in features}
    out = []
    for row in rows:
        merged = dict(row)
        merged.update(by_name[row["strategy_name"]])
        out.append(merged)
    return out


def define_feature_sets() -> dict[str, dict[str, Any]]:
    numeric_all = [
        "adjacent_jump_count", "center_early_count", "center_track_index", "direction_reversal_count",
        "early_center_bias", "edge_early_count", "first_track", "first_track_norm", "jump_entropy",
        "last_track", "last_track_norm", "late_center_bias", "long_jump_count", "max_jump",
        "max_unvisited_gap_proxy_max_norm", "max_unvisited_gap_proxy_mean_norm", "mean_jump",
        "median_jump", "min_jump", "monotonicity_fraction", "n", "normalized_max_jump",
        "normalized_mean_jump", "odd_even_transition_count", "parity_switch_rate",
        "running_center_distance_mean", "std_jump", "total_jump",
    ]
    f01 = ["n", "first_track_norm", "last_track_norm", "normalized_mean_jump", "normalized_max_jump", "adjacent_jump_count", "long_jump_count", "parity_switch_rate", "monotonicity_fraction", "direction_reversal_count"]
    f05 = [c for c in numeric_all if c != "n"]
    return {
        "F01_basic_order": {"numeric": f01, "categorical": []},
        "F02_full_handcrafted": {"numeric": numeric_all, "categorical": []},
        "F03_family_plus_features": {"numeric": numeric_all, "categorical": ["candidate_family", "selection_bucket", "dataset_source"]},
        "F04_no_family_generalization": {"numeric": numeric_all, "categorical": []},
        "F05_n_agnostic": {"numeric": f05, "categorical": []},
        "F06_no_dataset_source": {"numeric": numeric_all, "categorical": ["candidate_family", "selection_bucket"]},
    }


def row_float(row: dict[str, Any], col: str) -> float:
    return parse_float(row.get(col), default=0.0)


def build_matrix(train: list[dict[str, Any]], test: list[dict[str, Any]], spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    numeric = spec["numeric"]
    categorical = spec["categorical"]
    xtr = [[row_float(row, c) for c in numeric] for row in train]
    xte = [[row_float(row, c) for c in numeric] for row in test]
    names = list(numeric)
    for col in categorical:
        cats = sorted({str(row.get(col, "")) for row in train})
        for cat in cats:
            names.append(f"{col}={cat}")
            for i, row in enumerate(train):
                xtr[i].append(1.0 if str(row.get(col, "")) == cat else 0.0)
            for i, row in enumerate(test):
                xte[i].append(1.0 if str(row.get(col, "")) == cat else 0.0)
    return np.asarray(xtr, dtype=float), np.asarray(xte, dtype=float), names


class MeanBaseline:
    def fit(self, x: np.ndarray, y: np.ndarray) -> "MeanBaseline":
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(x.shape[0], self.mean_, dtype=float)


def models() -> dict[str, Any]:
    from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import ElasticNet, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "MeanBaseline": MeanBaseline(),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "ElasticNet": make_pipeline(StandardScaler(), ElasticNet(alpha=0.01, l1_ratio=0.25, max_iter=10000, random_state=42)),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=70, max_depth=4, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "ExtraTreesRegressor": ExtraTreesRegressor(n_estimators=70, max_depth=4, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=50, max_depth=2, learning_rate=0.05, random_state=42),
    }


def clone_model(model: Any) -> Any:
    if isinstance(model, MeanBaseline):
        return MeanBaseline()
    from sklearn.base import clone

    return clone(model)


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


def ndcg_at_5(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    k = min(5, len(y_true))
    if k == 0:
        return math.nan
    pred_order = np.argsort(-y_pred)[:k]
    ideal_order = np.argsort(-y_true)[:k]
    def dcg(order: np.ndarray) -> float:
        return float(sum((2 ** y_true[idx] - 1) / math.log2(pos + 2) for pos, idx in enumerate(order)))
    ideal = dcg(ideal_order)
    return safe_divide(dcg(pred_order), ideal, default=math.nan)


def metric_rows(test: list[dict[str, Any]], y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    residual = y_pred - y_true
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    true_rank = rank_desc(y_true)
    pred_rank = rank_desc(y_pred)
    names = [r["strategy_name"] for r in test]
    true_top1 = names[int(np.argmax(y_true))]
    pred_top1 = names[int(np.argmax(y_pred))]
    true_top3 = {names[i] for i in np.argsort(-y_true)[: min(3, len(names))]}
    pred_top3 = {names[i] for i in np.argsort(-y_pred)[: min(3, len(names))]}
    true_top5 = {names[i] for i in np.argsort(-y_true)[: min(5, len(names))]}
    pred_top5 = {names[i] for i in np.argsort(-y_pred)[: min(5, len(names))]}
    return {
        "spearman": spearman(y_pred, y_true),
        "pearson": pearson(y_pred, y_true),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(math.sqrt(np.mean(residual ** 2))),
        "r2": 1.0 - safe_divide(ss_res, ss_tot, default=math.nan),
        "top1_hit": pred_top1 == true_top1,
        "top3_overlap": len(true_top3 & pred_top3),
        "top5_overlap": len(true_top5 & pred_top5),
        "mean_abs_rank_error": float(np.mean(np.abs(true_rank - pred_rank))),
        "ndcg_at_5": ndcg_at_5(y_true, y_pred),
    }


def protocols(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for n in EXPECTED_N:
        out.append({"protocol": "P01_leave_N_out", "split_name": f"test_N{n}", "train": [r for r in rows if parse_int(r["n"]) != n], "test": [r for r in rows if parse_int(r["n"]) == n]})
    out.append({"protocol": "P02_core_generalization", "split_name": "train_N12_N16_N24_test_N40", "train": [r for r in rows if parse_int(r["n"]) in {12, 16, 24}], "test": [r for r in rows if parse_int(r["n"]) == 40]})
    out.append({"protocol": "P03_small_to_large", "split_name": "train_N12_N16_test_N24_N40", "train": [r for r in rows if parse_int(r["n"]) in {12, 16}], "test": [r for r in rows if parse_int(r["n"]) in {24, 40}]})
    out.append({"protocol": "P04_large_to_small", "split_name": "train_N24_N40_test_N12_N16", "train": [r for r in rows if parse_int(r["n"]) in {24, 40}], "test": [r for r in rows if parse_int(r["n"]) in {12, 16}]})
    rng = random.Random(417)
    fold_buckets = {i: [] for i in range(5)}
    for n in EXPECTED_N:
        for source in ["probe60_run08", "batch20_run14"]:
            group = [r for r in rows if parse_int(r["n"]) == n and r["dataset_source"] == source]
            rng.shuffle(group)
            for i, row in enumerate(group):
                fold_buckets[i % 5].append(row)
    all_set = set(r["strategy_name"] for r in rows)
    by_name = {r["strategy_name"]: r for r in rows}
    for fold, test_rows in fold_buckets.items():
        test_names = {r["strategy_name"] for r in test_rows}
        out.append({"protocol": "P05_stratified_5fold", "split_name": f"fold_{fold+1}", "train": [by_name[n] for n in sorted(all_set - test_names)], "test": test_rows})
    out.append({"protocol": "P06_train_probe60_test_batch20", "split_name": "probe60_to_batch20", "train": [r for r in rows if r["dataset_source"] == "probe60_run08"], "test": [r for r in rows if r["dataset_source"] == "batch20_run14"]})
    batch20 = [r for r in rows if r["dataset_source"] == "batch20_run14"]
    for fold, test_rows in fold_buckets.items():
        test_batch = [r for r in test_rows if r["dataset_source"] == "batch20_run14"]
        test_names = {r["strategy_name"] for r in test_batch}
        train = [r for r in rows if r["dataset_source"] == "probe60_run08"] + [r for r in batch20 if r["strategy_name"] not in test_names]
        out.append({"protocol": "P07_probe60_plus_some_batch20", "split_name": f"batch20_fold_{fold+1}", "train": train, "test": test_batch})
    return out


def extract_importance(model: Any, names: list[str], target: str, feature_set: str, model_name: str, protocol: str, split: str) -> list[dict[str, Any]]:
    est = model
    if hasattr(model, "named_steps"):
        est = list(model.named_steps.values())[-1]
    vals = None
    typ = ""
    if hasattr(est, "feature_importances_"):
        vals = est.feature_importances_
        typ = "tree_feature_importance"
    elif hasattr(est, "coef_"):
        vals = np.ravel(est.coef_)
        typ = "standardized_linear_coefficient"
    if vals is None:
        return []
    return [{"target": target, "feature_set": feature_set, "model_name": model_name, "protocol": protocol, "split_name": split, "importance_type": typ, "feature_name": n, "value": float(v), "abs_value": float(abs(v))} for n, v in zip(names, vals)]


def evaluate(rows: list[dict[str, Any]], fsets: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    regs = models()
    protos = protocols(rows)
    detailed = []
    importances = []
    for target in TARGETS:
        for fs_name, fs in fsets.items():
            for model_name, base in regs.items():
                for proto in protos:
                    if not proto["test"] or not proto["train"]:
                        continue
                    xtr, xte, names = build_matrix(proto["train"], proto["test"], fs)
                    ytr = np.asarray([parse_float(r[target]) for r in proto["train"]], dtype=float)
                    yte = np.asarray([parse_float(r[target]) for r in proto["test"]], dtype=float)
                    model = clone_model(base)
                    try:
                        model.fit(xtr, ytr)
                        pred = np.asarray(model.predict(xte), dtype=float)
                    except Exception as exc:
                        detailed.append({"target": target, "feature_set": fs_name, "model_name": model_name, "protocol": proto["protocol"], "split_name": proto["split_name"], "test_n": "ALL", "status": "MODEL_FAILED", "warning": repr(exc)})
                        continue
                    groups: list[tuple[str, list[int]]] = [("ALL", list(range(len(proto["test"]))))]
                    if proto["protocol"] in {"P03_small_to_large", "P04_large_to_small", "P06_train_probe60_test_batch20"}:
                        for n in EXPECTED_N:
                            idxs = [i for i, r in enumerate(proto["test"]) if parse_int(r["n"]) == n]
                            if idxs:
                                groups.append((f"N{n}", idxs))
                    for test_n, idxs in groups:
                        sub_test = [proto["test"][i] for i in idxs]
                        m = metric_rows(sub_test, yte[idxs], pred[idxs])
                        detailed.append({"target": target, "feature_set": fs_name, "model_name": model_name, "protocol": proto["protocol"], "split_name": proto["split_name"], "test_n": test_n, "test_count": len(idxs), "status": "OK", **m})
                    if target == PRIMARY_TARGET and proto["protocol"] == "P01_leave_N_out":
                        importances.extend(extract_importance(model, names, target, fs_name, model_name, proto["protocol"], proto["split_name"]))
    return add_macro_rows(detailed), importances


def add_macro_rows(detailed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in detailed:
        if row.get("protocol") == "P01_leave_N_out" and row.get("status") == "OK":
            grouped[(row["target"], row["feature_set"], row["model_name"])].append(row)
    macro_rows = []
    metrics = ["spearman", "pearson", "mae", "rmse", "r2", "top1_hit", "top3_overlap", "top5_overlap", "mean_abs_rank_error", "ndcg_at_5"]
    for (target, fs, model), rows in grouped.items():
        if len(rows) != 4:
            continue
        out = {"target": target, "feature_set": fs, "model_name": model, "protocol": "P01_leave_N_out_macro", "split_name": "macro_over_N12_N16_N24_N40", "test_n": "MACRO", "test_count": sum(int(r["test_count"]) for r in rows), "status": "OK"}
        for m in metrics:
            vals = []
            for row in rows:
                v = row.get(m)
                if isinstance(v, bool):
                    vals.append(float(v))
                elif isinstance(v, (float, int)) and math.isfinite(float(v)):
                    vals.append(float(v))
            out[m] = mean(vals, default=math.nan)
        macro_rows.append(out)
    return detailed + macro_rows


def simplicity(model: str) -> int:
    return {"MeanBaseline": 0, "Ridge": 1, "ElasticNet": 2, "GradientBoostingRegressor": 3, "RandomForestRegressor": 4, "ExtraTreesRegressor": 5}.get(model, 99)


def best_configs(detailed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [r for r in detailed if r.get("protocol") == "P01_leave_N_out_macro" and r.get("status") == "OK"]
    out = []
    for target in TARGETS:
        target_rows = [r for r in rows if r["target"] == target]
        if not target_rows:
            continue
        target_rows.sort(key=lambda r: (-(parse_float(r.get("spearman"), -999)), -parse_float(r.get("top5_overlap"), 0), parse_float(r.get("mae"), 999), simplicity(r["model_name"])))
        best = dict(target_rows[0])
        best["selection_criterion"] = "leave-N-out macro Spearman, then top5 overlap, MAE, model simplicity"
        out.append(best)
    primary = [r for r in rows if r["target"] == PRIMARY_TARGET]
    for label, filt, key in [
        ("best_top5_overlap_primary", lambda r: True, lambda r: (-parse_float(r["top5_overlap"]), -parse_float(r["spearman"]))),
        ("best_no_family_primary", lambda r: r["feature_set"] == "F04_no_family_generalization", lambda r: (-parse_float(r["spearman"]), -parse_float(r["top5_overlap"]))),
        ("best_n_agnostic_primary", lambda r: r["feature_set"] == "F05_n_agnostic", lambda r: (-parse_float(r["spearman"]), -parse_float(r["top5_overlap"]))),
    ]:
        candidates = [r for r in primary if filt(r)]
        if candidates:
            candidates.sort(key=key)
            item = dict(candidates[0])
            item["target"] = f"{PRIMARY_TARGET}::{label}"
            out.append(item)
    p06 = [r for r in detailed if r["target"] == PRIMARY_TARGET and r["protocol"] == "P06_train_probe60_test_batch20" and r["test_n"] == "ALL" and r["status"] == "OK"]
    if p06:
        p06.sort(key=lambda r: (-parse_float(r["spearman"]), -parse_float(r["top5_overlap"]), parse_float(r["mae"])))
        item = dict(p06[0])
        item["target"] = f"{PRIMARY_TARGET}::best_probe60_to_batch20"
        out.append(item)
    return out


def predictions_for(rows: list[dict[str, Any]], fsets: dict[str, dict[str, Any]], best: dict[str, Any]) -> list[dict[str, Any]]:
    regs = models()
    fs = fsets[best["feature_set"]]
    reg = regs[best["model_name"]]
    out = []
    for proto in protocols(rows):
        xtr, xte, _names = build_matrix(proto["train"], proto["test"], fs)
        ytr = np.asarray([parse_float(r[PRIMARY_TARGET]) for r in proto["train"]], dtype=float)
        yte = np.asarray([parse_float(r[PRIMARY_TARGET]) for r in proto["test"]], dtype=float)
        model = clone_model(reg).fit(xtr, ytr)
        pred = np.asarray(model.predict(xte), dtype=float)
        true_rank = rank_desc(yte)
        pred_rank = rank_desc(pred)
        for i, row in enumerate(proto["test"]):
            out.append({
                "protocol": proto["protocol"], "split_name": proto["split_name"], "model_name": best["model_name"], "feature_set": best["feature_set"],
                "test_n": row["n"], "dataset_source": row["dataset_source"], "strategy_name": row["strategy_name"],
                "true_target": yte[i], "predicted_target": pred[i], "true_rank_within_test_group": true_rank[i],
                "pred_rank_within_test_group": pred_rank[i], "rank_error": abs(true_rank[i] - pred_rank[i]),
                "is_true_top5": true_rank[i] <= 5, "is_pred_top5": pred_rank[i] <= 5,
                "is_batch20": row["dataset_source"] == "batch20_run14", "is_probe60": row["dataset_source"] == "probe60_run08",
            })
    return out


def pairwise_dataset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for n in EXPECTED_N:
        group = [r for r in rows if parse_int(r["n"]) == n]
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                rew_a, rew_b = parse_float(a[PRIMARY_TARGET]), parse_float(b[PRIMARY_TARGET])
                u2_a, u2_b = parse_float(a["target_u2_score_combined80_rank"]), parse_float(b["target_u2_score_combined80_rank"])
                peeq_a, peeq_b = parse_float(a["target_peeq_score_combined80_rank"]), parse_float(b["target_peeq_score_combined80_rank"])
                out.append({"n": n, "case_i": a["strategy_name"], "case_j": b["strategy_name"], "preferred_by_reward": "i" if rew_a > rew_b else "j" if rew_b > rew_a else "tie", "preferred_by_u2": "i" if u2_a > u2_b else "j" if u2_b > u2_a else "tie", "preferred_by_peeq": "i" if peeq_a > peeq_b else "j" if peeq_b > peeq_a else "tie", "reward_margin": abs(rew_a - rew_b), "u2_margin": abs(u2_a - u2_b), "peeq_margin": abs(peeq_a - peeq_b)})
    return out


def pairwise_validate(rows: list[dict[str, Any]], pairs: list[dict[str, Any]], fsets: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    by_name = {r["strategy_name"]: r for r in rows}
    spec = fsets["F05_n_agnostic"]
    clfs = {
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=80, max_depth=4, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "ExtraTreesClassifier": ExtraTreesClassifier(n_estimators=80, max_depth=4, min_samples_leaf=2, random_state=42, n_jobs=-1),
    }

    def build(pair_rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for p in pair_rows:
            if p["preferred_by_reward"] == "tie":
                continue
            a, b = by_name[p["case_i"]], by_name[p["case_j"]]
            xa, _, _ = build_matrix([a], [a], spec)
            xb, _, _ = build_matrix([b], [b], spec)
            xs.append((xa[0] - xb[0]).tolist())
            ys.append(1 if p["preferred_by_reward"] == "i" else 0)
        return np.asarray(xs, dtype=float), np.asarray(ys, dtype=int)

    protocols_pw = []
    for n in EXPECTED_N:
        protocols_pw.append((f"leave_N_out_test_N{n}", [p for p in pairs if p["n"] != n], [p for p in pairs if p["n"] == n]))
    rng = random.Random(517)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    folds = [shuffled[i::5] for i in range(5)]
    all_ids = set(range(len(pairs)))
    # Preserve simple 5-fold by list identity through index map.
    for fold_idx, test in enumerate(folds):
        test_set = {id(p) for p in test}
        protocols_pw.append((f"stratified_fold_{fold_idx+1}", [p for p in pairs if id(p) not in test_set], test))
    results = [{"status": "PAIRWISE_DATASET_SUMMARY", "pair_count": len(pairs), **dict(Counter(p["preferred_by_reward"] for p in pairs))}]
    for split, train_pairs, test_pairs in protocols_pw:
        xtr, ytr = build(train_pairs)
        xte, yte = build(test_pairs)
        if len(set(ytr.tolist())) < 2 or len(yte) == 0:
            continue
        for name, clf0 in clfs.items():
            clf = clone_model(clf0).fit(xtr, ytr)
            pred = clf.predict(xte)
            score = clf.predict_proba(xte)[:, 1] if hasattr(clf, "predict_proba") else pred
            auc = roc_auc_score(yte, score) if len(set(yte.tolist())) > 1 else math.nan
            results.append({"status": "OK", "protocol": split, "model_name": name, "pair_count": len(yte), "accuracy": float(accuracy_score(yte, pred)), "balanced_accuracy": float(balanced_accuracy_score(yte, pred)), "auc": auc})
    ok = [r for r in results if r.get("status") == "OK"]
    summary = {"status": "PAIRWISE_MODEL_VALIDATED" if ok else "PAIRWISE_MODEL_SKIPPED", "pairwise_rows": len(pairs), "best_accuracy": max([r["accuracy"] for r in ok], default=math.nan), "best_auc": max([r["auc"] for r in ok if math.isfinite(r["auc"])], default=math.nan)}
    return results, summary


def run11_comparison(best_primary: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"metric": "leave_N_out_macro_spearman", "run11_value": RUN11_BEST_SPEARMAN, "run17_value": parse_float(best_primary.get("spearman")), "delta_run17_minus_run11": parse_float(best_primary.get("spearman")) - RUN11_BEST_SPEARMAN, "interpretation": "improved" if parse_float(best_primary.get("spearman")) > RUN11_BEST_SPEARMAN else "degraded_or_similar"},
        {"metric": "leave_N_out_macro_top5_overlap", "run11_value": RUN11_BEST_TOP5, "run17_value": parse_float(best_primary.get("top5_overlap")), "delta_run17_minus_run11": parse_float(best_primary.get("top5_overlap")) - RUN11_BEST_TOP5, "interpretation": "improved" if parse_float(best_primary.get("top5_overlap")) > RUN11_BEST_TOP5 else "degraded_or_similar"},
    ]


def maybe_plots(detailed: list[dict[str, Any]], predictions: list[dict[str, Any]], importances: list[dict[str, Any]], comparison: list[dict[str, Any]]) -> list[str]:
    paths = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return [f"PLOTTING_SKIPPED: {exc}"]
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4))
    for row in comparison:
        plt.bar(row["metric"], row["run11_value"], label="run11" if not paths else None, alpha=0.5)
        plt.bar(row["metric"], row["run17_value"], label="run17" if not paths else None, alpha=0.8)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("metric value")
    plt.legend()
    path = FIGURE_DIR / "run11_vs_run17_macro_metrics.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    paths.append(str(path))
    lno = [r for r in predictions if r["protocol"] == "P01_leave_N_out"]
    if lno:
        plt.figure(figsize=(5, 5))
        plt.scatter([r["true_target"] for r in lno], [r["predicted_target"] for r in lno], s=25)
        plt.xlabel("true combined80 reward")
        plt.ylabel("predicted combined80 reward")
        path = FIGURE_DIR / "leave_N_out_predicted_vs_true.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        paths.append(str(path))
    p06 = [r for r in predictions if r["protocol"] == "P06_train_probe60_test_batch20"]
    if p06:
        plt.figure(figsize=(5, 5))
        plt.scatter([r["true_target"] for r in p06], [r["predicted_target"] for r in p06], s=30)
        plt.xlabel("true batch20 reward")
        plt.ylabel("probe60-trained prediction")
        path = FIGURE_DIR / "probe60_train_batch20_test_predicted_vs_true.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        paths.append(str(path))
    top_imp = sorted([r for r in importances if r["target"] == PRIMARY_TARGET], key=lambda r: parse_float(r["abs_value"]), reverse=True)[:12]
    if top_imp:
        plt.figure(figsize=(8, 4))
        plt.bar([r["feature_name"] for r in top_imp], [parse_float(r["value"]) for r in top_imp])
        plt.xticks(rotation=60, ha="right", fontsize=8)
        path = FIGURE_DIR / "top_feature_importance.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        paths.append(str(path))
    return paths


def write_claim(md: Path, js: Path) -> None:
    safe = [
        "Run17 updates lightweight surrogate validation using the expanded 80-case teacher-labelled variable-N dataset.",
        "Combined80 provides 20 cases per N for N12/N16/N24/N40.",
        "Run17 compares surrogate validation against run11 and evaluates whether batch20 improves surrogate stability.",
        "Pairwise preference data are expanded to 760 within-N pairs.",
        "Results can guide whether the next stage should be surrogate-screened candidate generation, active learning, or richer graph/sequence models.",
    ]
    unsafe = [
        "Do not claim trained variable-N RL policy superiority.",
        "Do not claim final surrogate accuracy.",
        "Do not claim arbitrary-N generalization.",
        "Do not claim physical optimum.",
        "Do not claim readiness to deploy.",
        "Do not claim feature importances are causal.",
        "Do not claim batch20 success proves surrogate is perfect, especially because run16 showed only moderate prediction-realization correlation.",
    ]
    md.write_text("# Run 17 Claim Boundary\n\n## Safe Claims\n" + "\n".join(f"- {x}" for x in safe) + "\n\n## Unsafe Claims\n" + "\n".join(f"- {x}" for x in unsafe) + "\n", encoding="utf-8")
    write_json(js, {"verdict": "RUN17_COMBINED80_SURROGATE_VALIDATION_UPDATE_ONLY_NO_RL_POLICY_TRAINING", "safe_claims": safe, "unsafe_claims": unsafe})


def update_run_index(verdict: str) -> None:
    if not RUN_INDEX_PATH.exists():
        return
    entry = (
        "| run_17 | Combined80 surrogate reward model validation update | Re-run lightweight surrogate validation on the expanded 80-case teacher-labelled dataset and compare against run11. | "
        "`scripts/stage3/run_17_validate_combined80_surrogate_reward_models.py` | "
        "`docs/stage3/runs/run_17_combined80_surrogate_reward_model_validation_update/RUN_17_COMBINED80_SURROGATE_REWARD_MODEL_VALIDATION_UPDATE_REPORT.md` | "
        "`outputs/stage3_run_17_combined80_surrogate_reward_model_validation_update/` | "
        f"`{verdict}` | No Abaqus, no ODB, no abqjobpilot, no CAE/INP/JNL generation, no final RL policy training, no commit/push. |"
    )
    lines = RUN_INDEX_PATH.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if line.startswith("| run_17 | Combined80 surrogate reward model validation update |"):
            lines[i] = entry
            break
    else:
        lines.append(entry)
    RUN_INDEX_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(validation: dict[str, Any], best: dict[str, Any], p06: dict[str, Any], comp: list[dict[str, Any]], pair_summary: dict[str, Any], output_files: list[str], recommendation: str) -> None:
    lines = [
        "# Stage 3 Run 17 - Combined80 Surrogate Reward Model Validation Update",
        "",
        "## Purpose",
        "Update lightweight surrogate validation using the expanded 80-case combined teacher-labelled dataset.",
        "",
        "## Inputs",
        f"- `{COMBINED80_READY}`",
        f"- `{COMBINED80_TEACHER}`",
        f"- `{RUN16_REPORT}`",
        f"- `{RUN11_DETAILED}`",
        "",
        "## Combined80 Validation Status",
        f"- `{validation['verdict']}`",
        f"- Rows: {validation['total_rows']}",
        f"- Per-N counts: {validation['per_n_counts']}",
        "",
        "## Feature Reconstruction",
        "Scan-order features were reconstructed from `order_json` for all 80 rows using the run10-style feature logic.",
        "",
        "## Targets and Feature Sets",
        "- Primary target: `target_reward_combined80_u2_primary`.",
        "- Feature sets F01-F06 were evaluated, including family/source diagnostics and N-agnostic features.",
        "",
        "## Models and Validation Protocols",
        "- MeanBaseline, Ridge, ElasticNet, RandomForestRegressor, ExtraTreesRegressor, and GradientBoostingRegressor.",
        "- Leave-N-out, small/large transfer, stratified folds, probe60-to-batch20, and probe60-plus-some-batch20 protocols.",
        "",
        "## Main Leave-N-Out Results",
        f"- Best primary config: `{best['model_name']}` / `{best['feature_set']}`.",
        f"- Macro Spearman: `{best['spearman']}`.",
        f"- Macro top5 overlap: `{best['top5_overlap']}`.",
        "",
        "## Train Probe60 Test Batch20 Analysis",
        f"- Best P06 config: `{p06.get('model_name')}` / `{p06.get('feature_set')}`.",
        f"- Spearman: `{p06.get('spearman')}`; top5 overlap: `{p06.get('top5_overlap')}`.",
        "",
        "## Comparison to Run11",
    ]
    for row in comp:
        lines.append(f"- {row['metric']}: run11={row['run11_value']}, run17={row['run17_value']}, delta={row['delta_run17_minus_run11']}.")
    lines += [
        "",
        "## Pairwise Preference Update",
        f"- Pairwise rows: {pair_summary['pairwise_rows']}.",
        f"- Best AUC: {pair_summary.get('best_auc')}; best accuracy: {pair_summary.get('best_accuracy')}.",
        "",
        "## Feature Importance Diagnostics",
        "- Feature importances are diagnostic only, not causal.",
        "",
        "## Limitations",
        "- Combined80 is still small, and batch20 was actively selected rather than randomly sampled.",
        "- Dataset-source features can be diagnostic but may encode generation-protocol differences.",
        "",
        "## Claim Boundary",
        "- No trained RL policy superiority, arbitrary-N generalization, or deployment readiness is claimed.",
        "",
        "## Output Files",
        *[f"- `{p}`" for p in output_files],
        "",
        "## Recommended Run18",
        recommendation,
    ]
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
    raw_rows = read_csv(COMBINED80_READY)
    validation = validate_inputs(raw_rows)
    validation_path = OUTPUT_DIR / "run17_input_validation_summary.json"
    write_json(validation_path, validation)
    if validation["verdict"].startswith("FAIL"):
        print(validation["verdict"])
        print(json.dumps(validation, indent=2))
        return 2
    feature_rows = [order_features(r) for r in raw_rows]
    feature_path = OUTPUT_DIR / "combined80_scan_order_features.csv"
    write_csv(feature_path, feature_rows)
    rows = build_model_rows(raw_rows, feature_rows)
    fsets = define_feature_sets()
    fset_path = OUTPUT_DIR / "run17_feature_set_definitions.json"
    write_json(fset_path, fsets)
    detailed, importances = evaluate(rows, fsets)
    best = best_configs(detailed)
    primary_best = next(r for r in best if r["target"] == PRIMARY_TARGET)
    p06_best = next(r for r in best if r["target"] == f"{PRIMARY_TARGET}::best_probe60_to_batch20")
    preds = predictions_for(rows, fsets, primary_best)
    pairs = pairwise_dataset(rows)
    pair_results, pair_summary = pairwise_validate(rows, pairs, fsets)
    comp = run11_comparison(primary_best)
    plots = maybe_plots(detailed, preds, importances, comp)
    claim_md = OUTPUT_DIR / "run17_claim_boundary.md"
    claim_json = OUTPUT_DIR / "run17_claim_boundary.json"
    write_claim(claim_md, claim_json)
    detailed_csv = OUTPUT_DIR / "combined80_surrogate_validation_results_detailed.csv"
    detailed_json = OUTPUT_DIR / "combined80_surrogate_validation_results_detailed.json"
    best_csv = OUTPUT_DIR / "combined80_best_surrogate_configurations.csv"
    best_json = OUTPUT_DIR / "combined80_best_surrogate_configurations.json"
    pred_csv = OUTPUT_DIR / "combined80_predictions_target_reward_u2_primary.csv"
    imp_csv = OUTPUT_DIR / "combined80_feature_importance_summary.csv"
    imp_json = OUTPUT_DIR / "combined80_feature_importance_summary.json"
    pair_csv = OUTPUT_DIR / "combined80_pairwise_preference_dataset.csv"
    pair_val_csv = OUTPUT_DIR / "combined80_pairwise_preference_validation_summary.csv"
    pair_val_json = OUTPUT_DIR / "combined80_pairwise_preference_validation_summary.json"
    comp_csv = OUTPUT_DIR / "run11_vs_run17_surrogate_comparison.csv"
    summary_json = OUTPUT_DIR / "run17_diagnostic_summary.json"
    write_csv(detailed_csv, detailed)
    write_table_json(detailed_json, detailed)
    write_csv(best_csv, best)
    write_json(best_json, best)
    write_csv(pred_csv, preds)
    write_csv(imp_csv, importances)
    write_table_json(imp_json, importances)
    write_csv(pair_csv, pairs)
    write_csv(pair_val_csv, pair_results)
    write_json(pair_val_json, {"summary": pair_summary, "results": pair_results})
    write_csv(comp_csv, comp)
    rec = "Use active-learning design rather than pure exploitation, with special attention to N40 calibration and uncertainty coverage."
    if parse_float(primary_best["spearman"]) >= RUN11_BEST_SPEARMAN - 0.03 and parse_float(primary_best["top5_overlap"]) >= RUN11_BEST_TOP5:
        rec = "Generate a second offline candidate batch using the combined80-updated surrogate, biased toward N24/N40 and including calibration/sentinel candidates."
    write_json(summary_json, {"primary_best": primary_best, "probe60_to_batch20_best": p06_best, "run11_vs_run17": comp, "pairwise_summary": pair_summary, "recommended_run18": rec})
    outputs = [str(validation_path), str(feature_path), str(fset_path), str(detailed_csv), str(detailed_json), str(best_csv), str(best_json), str(pred_csv), str(imp_csv), str(imp_json), str(pair_csv), str(pair_val_csv), str(pair_val_json), str(comp_csv), str(summary_json), str(claim_md), str(claim_json), *[p for p in plots if not p.startswith("PLOTTING_SKIPPED")], str(REPORT_PATH)]
    write_report(validation, primary_best, p06_best, comp, pair_summary, outputs, rec)
    update_run_index(validation["verdict"])
    manifest = {"run_id": RUN_ID, "run_name": RUN_NAME, "timestamp": datetime.now(timezone.utc).isoformat(), "branch": git_branch(), "script_path": str(Path(__file__).resolve()), "input_files": [str(COMBINED80_READY), str(COMBINED80_TEACHER), str(COMBINED80_LEADERBOARD), str(RUN16_REPORT), str(RUN11_DETAILED), str(RUN11_BEST), str(RUN11_REPORT), str(RUN10_FEATURES), str(RUN13_HANDOFF), str(RUN12_SCORED)], "output_files": outputs, "report_path": str(REPORT_PATH), "claim_boundary_path": str(claim_md), "validation_verdict": validation["verdict"], "total_rows": 80, "per_N_rows": validation["per_n_counts"], "pairwise_rows": len(pairs), "no_solver_run": True, "no_odb_opened": True, "no_abqjobpilot_run": True, "no_cae_inp_generated": True, "no_rl_policy_training": True, "no_commit_or_push": True}
    write_json(MANIFEST_PATH, manifest)
    print(validation["verdict"])
    print(f"rows=80 per_n={validation['per_n_counts']}")
    print(f"best_primary={primary_best['model_name']}|{primary_best['feature_set']}|spearman={primary_best['spearman']}|top5={primary_best['top5_overlap']}")
    print(f"p06_best={p06_best['model_name']}|{p06_best['feature_set']}|spearman={p06_best['spearman']}|top5={p06_best['top5_overlap']}")
    print(f"pairwise_rows={len(pairs)} best_auc={pair_summary.get('best_auc')}")
    print(f"report={REPORT_PATH}")
    print(f"manifest={MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
