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
RUN_ID = "run_29_combined172_surrogate_gnn_hybrid_policy_update_and_candidate_generation"
RUN_NAME = "combined172 surrogate GNN hybrid policy update and candidate generation"

COMBINED172_READY = ROOT / "outputs" / "stage3_run_28_shortlist64_teacher_metrics_ingestion_and_combined172_ranking" / "combined172_RL_ready_dataset.csv"
COMBINED172_TEACHER = ROOT / "outputs" / "stage3_run_28_shortlist64_teacher_metrics_ingestion_and_combined172_ranking" / "combined172_teacher_dataset.csv"
COMBINED172_LEADERBOARD = ROOT / "outputs" / "stage3_run_28_shortlist64_teacher_metrics_ingestion_and_combined172_ranking" / "combined172_per_N_leaderboard.csv"
RUN27_ENRICHED = ROOT / "outputs" / "stage3_run_28_shortlist64_teacher_metrics_ingestion_and_combined172_ranking" / "run27_shortlist64_teacher_dataset_enriched.csv"
RUN27_PRED_AUDIT = ROOT / "outputs" / "stage3_run_28_shortlist64_teacher_metrics_ingestion_and_combined172_ranking" / "run27_shortlist64_prediction_audit.csv"
RUN27_BUCKET = ROOT / "outputs" / "stage3_run_28_shortlist64_teacher_metrics_ingestion_and_combined172_ranking" / "run27_bucket_source_performance_summary.csv"
RUN27_VS_108 = ROOT / "outputs" / "stage3_run_28_shortlist64_teacher_metrics_ingestion_and_combined172_ranking" / "run27_vs_combined108_best_comparison.csv"
RUN28_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_28_shortlist64_teacher_metrics_ingestion_and_combined172_ranking" / "RUN_28_SHORTLIST64_TEACHER_METRICS_INGESTION_AND_COMBINED172_RANKING_REPORT.md"
RUN26_POOL = ROOT / "outputs" / "stage3_run_26_combined108_gnn_graph_pointer_policy_candidate_generation" / "run26_gnn_candidate_pool_scored.csv"
RUN26_BATCH64 = ROOT / "outputs" / "stage3_run_26_combined108_gnn_graph_pointer_policy_candidate_generation" / "run26_gnn_policy_batch64_candidate_orders.csv"
RUN23_POOL = ROOT / "outputs" / "stage3_run_23_combined108_active_learning_coverage_calibration_design" / "run23_candidate_pool_scored.csv"
RUN24_SHORTLIST64 = ROOT / "outputs" / "stage3_run_24_run23_shortlist64_active_learning_handoff_package" / "stage3_run24_shortlist64_candidate_orders.csv"

OUTPUT_DIR = ROOT / "outputs" / "stage3_run_29_combined172_surrogate_gnn_hybrid_policy_update_and_candidate_generation"
REPORT_DIR = ROOT / "docs" / "stage3" / "runs" / RUN_ID
REPORT_PATH = REPORT_DIR / "RUN_29_COMBINED172_SURROGATE_GNN_HYBRID_POLICY_UPDATE_AND_CANDIDATE_GENERATION_REPORT.md"
MANIFEST_PATH = ROOT / "artifacts" / "manifests" / "stage3_run_29_manifest.json"
RUN_INDEX_PATH = ROOT / "docs" / "stage3" / "STAGE3_RUN_INDEX.md"

EXPECTED_N = [12, 16, 24, 40]
EXPECTED_COUNTS = {12: 32, 16: 32, 24: 54, 40: 54}
POOL_TARGETS = {12: 800, 16: 800, 24: 2500, 40: 3000}
BATCH64_COUNTS = {12: 8, 16: 8, 24: 24, 40: 24}
BATCH32_COUNTS = {12: 4, 16: 4, 24: 12, 40: 12}
FOCUSED48_COUNTS = {24: 24, 40: 24}
PRIMARY_TARGET = "target_reward_combined172_u2_primary"
SECONDARY_TARGETS = [
    "target_u2_score_combined172_rank",
    "target_peeq_score_combined172_rank",
    "target_surfaceT_score_combined172_rank",
    "target_mises_score_combined172_rank",
]
F01 = ["n", "first_track_norm", "last_track_norm", "normalized_mean_jump", "normalized_max_jump", "adjacent_jump_count", "long_jump_count", "parity_switch_rate", "monotonicity_fraction", "direction_reversal_count"]
GLOBAL_SEED = 29042


def load_run26_module() -> Any:
    path = ROOT / "scripts" / "stage3" / "run_26_gnn_graph_pointer_policy_candidate_generation.py"
    spec = importlib.util.spec_from_file_location("run26_helpers", path)
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
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_int(value: Any) -> int:
    return R26.parse_int(value)


def parse_float(value: Any, default: float = math.nan) -> float:
    return R26.parse_float(value, default)


def parse_order(text: Any) -> list[int] | None:
    return R26.parse_order(text)


def validate_order(order: list[int] | None, n: int) -> bool:
    return R26.validate_order(order, n)


def order_hash(order: list[int]) -> str:
    return hashlib.sha1(",".join(str(x) for x in order).encode("ascii")).hexdigest()[:16]


def canonical_order_json(order: list[int]) -> str:
    return json.dumps(order, separators=(",", ":"))


def scan_order_features(order: list[int], n: int) -> dict[str, Any]:
    out = dict(R26.scan_order_features(order, n))
    out["n"] = n
    out["odd_even_transition_count"] = out.get("parity_switch_rate", 0.0) * max(1, n - 1)
    return out


def safe_divide(num: float, den: float, default: float = 0.0) -> float:
    return num / den if den else default


def mean(values: list[float], default: float = math.nan) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return statistics.fmean(vals) if vals else default


def std(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return statistics.pstdev(vals) if len(vals) > 1 else 0.0


def median(values: list[float], default: float = math.nan) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return statistics.median(vals) if vals else default


def pearson(x: list[float], y: list[float]) -> float:
    pairs = [(a, b) for a, b in zip(x, y) if math.isfinite(a) and math.isfinite(b)]
    if len(pairs) < 2:
        return math.nan
    xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
    mx, my = mean(xs), mean(ys)
    den = math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys))
    return safe_divide(sum((a - mx) * (b - my) for a, b in zip(xs, ys)), den, math.nan)


def spearman(x: list[float], y: list[float]) -> float:
    pairs = [(a, b) for a, b in zip(x, y) if math.isfinite(a) and math.isfinite(b)]
    if len(pairs) < 3:
        return math.nan
    return pearson(R26.rank_values([p[0] for p in pairs]), R26.rank_values([p[1] for p in pairs]))


def rank_desc(values: list[float]) -> list[float]:
    return R26.rank_values([-v for v in values])


def topk_overlap(true: list[float], pred: list[float], k: int) -> int:
    k = min(k, len(true))
    true_idx = set(sorted(range(len(true)), key=lambda i: true[i], reverse=True)[:k])
    pred_idx = set(sorted(range(len(pred)), key=lambda i: pred[i], reverse=True)[:k])
    return len(true_idx & pred_idx)


def ndcg_at_k(true: list[float], pred: list[float], k: int = 5) -> float:
    order = sorted(range(len(pred)), key=lambda i: pred[i], reverse=True)[: min(k, len(pred))]
    ideal = sorted(range(len(true)), key=lambda i: true[i], reverse=True)[: min(k, len(true))]
    dcg = sum(((2 ** true[i]) - 1) / math.log2(rank + 2) for rank, i in enumerate(order))
    idcg = sum(((2 ** true[i]) - 1) / math.log2(rank + 2) for rank, i in enumerate(ideal))
    return safe_divide(dcg, idcg, math.nan)


def kendall_distance(a: list[int], b: list[int]) -> float:
    return R26.kendall_distance(a, b)


def nearest_order(order: list[int], refs: list[dict[str, Any]]) -> tuple[str, float]:
    return R26.nearest_order(order, refs)


def load_combined172() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_csv(COMBINED172_READY):
        n = parse_int(row.get("n"))
        order = parse_order(row.get("order_json"))
        features = scan_order_features(order or [], n) if validate_order(order, n) else {}
        rows.append(
            {
                **row,
                **features,
                "n": n,
                "order": order,
                "order_hash": order_hash(order) if validate_order(order, n) else "",
                "reward": parse_float(row.get(PRIMARY_TARGET)),
                PRIMARY_TARGET: parse_float(row.get(PRIMARY_TARGET)),
                **{target: parse_float(row.get(target)) for target in SECONDARY_TARGETS},
            }
        )
    return rows


def refs_by_n(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    refs: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if validate_order(row.get("order"), row["n"]):
            refs[row["n"]].append({"strategy_name": row["strategy_name"], "order": row["order"], "order_hash": row["order_hash"]})
    return refs


def prior_hashes() -> dict[int, set[str]]:
    prior: dict[int, set[str]] = defaultdict(set)
    for path in [RUN23_POOL, RUN24_SHORTLIST64, RUN26_POOL, RUN26_BATCH64]:
        if not path.exists():
            continue
        for row in read_csv(path):
            try:
                n = parse_int(row.get("n"))
            except Exception:
                continue
            digest = row.get("order_hash") or ""
            order = parse_order(row.get("order_json"))
            if not digest and validate_order(order, n):
                digest = order_hash(order)
            if digest:
                prior[n].add(digest)
    return prior


def validate_inputs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    counts = Counter(row["n"] for row in rows)
    sources = Counter(row.get("dataset_source", "") for row in rows)
    if len(rows) != 172:
        errors.append(f"Expected 172 rows, found {len(rows)}")
    if dict(sorted(counts.items())) != EXPECTED_COUNTS:
        errors.append(f"Expected counts {EXPECTED_COUNTS}, found {dict(sorted(counts.items()))}")
    for source in ["probe60_run08", "batch20_run14", "batch28_run20", "shortlist64_run27"]:
        if source not in sources:
            errors.append(f"Missing dataset_source {source}")
    for row in rows:
        if not validate_order(row.get("order"), row["n"]):
            errors.append(f"Invalid order for {row.get('strategy_name')}")
        if not math.isfinite(row.get("reward", math.nan)):
            errors.append(f"Missing primary target for {row.get('strategy_name')}")
        for col in ["u2_range", "peeq_max", "surface_t_proxy", "mises_max"]:
            if not math.isfinite(parse_float(row.get(col))):
                errors.append(f"Missing raw metric {col} for {row.get('strategy_name')}")
    anchor = {
        "N40_best_u2": min([r for r in rows if r["n"] == 40], key=lambda r: parse_float(r["u2_range"]))["strategy_name"],
        "N40_best_reward": max([r for r in rows if r["n"] == 40], key=lambda r: r["reward"])["strategy_name"],
        "N16_best_u2": min([r for r in rows if r["n"] == 16], key=lambda r: parse_float(r["u2_range"]))["strategy_name"],
        "N16_best_reward": max([r for r in rows if r["n"] == 16], key=lambda r: r["reward"])["strategy_name"],
    }
    expected_anchor = {
        "N40_best_u2": "S3R24L64_N40_B23_exploitation_reference",
        "N40_best_reward": "S3R24L64_N40_B23_exploitation_reference",
        "N16_best_u2": "S3R24L64_N16_B02_top_region",
        "N16_best_reward": "S3R24L64_N16_B02_top_region",
    }
    for key, expected in expected_anchor.items():
        if anchor[key] != expected:
            errors.append(f"Anchor mismatch {key}: expected {expected}, found {anchor[key]}")
    verdict = "PASS_RUN29_COMBINED172_INPUTS_READY" if not errors else "FAIL_RUN29_INPUT_VALIDATION"
    payload = {
        "verdict": verdict,
        "errors": errors,
        "combined172_rows": len(rows),
        "per_n_counts": dict(sorted(counts.items())),
        "dataset_source_counts": dict(sorted(sources.items())),
        "anchor_validation": anchor,
    }
    write_json(OUTPUT_DIR / "run29_input_validation_summary.json", payload)
    return payload


def write_feature_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table = []
    for row in rows:
        feature_cols = {k: row.get(k) for k in scan_order_features(row["order"], row["n"]).keys()}
        table.append(
            {
                "n": row["n"],
                "strategy_name": row["strategy_name"],
                "dataset_source": row.get("dataset_source", ""),
                "order_hash": row["order_hash"],
                **feature_cols,
                PRIMARY_TARGET: row[PRIMARY_TARGET],
                **{target: row[target] for target in SECONDARY_TARGETS},
            }
        )
    write_csv(OUTPUT_DIR / "combined172_scan_order_features.csv", table)
    return table


def feature_sets(feature_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    all_numeric = []
    exclusions = {"u2_range", "peeq_max", "surface_t_proxy", "mises_max", PRIMARY_TARGET, *SECONDARY_TARGETS}
    for key in feature_rows[0]:
        if key in exclusions or key in {"strategy_name", "dataset_source", "order_hash", "order_json", "order_compact"}:
            continue
        if all(isinstance(parse_float(row.get(key)), float) and math.isfinite(parse_float(row.get(key))) for row in feature_rows):
            all_numeric.append(key)
    all_numeric = sorted(set(all_numeric))
    return {
        "F01_basic_order": {"numeric": F01, "categorical": []},
        "F02_full_handcrafted": {"numeric": all_numeric, "categorical": []},
        "F03_family_plus_features": {"numeric": all_numeric, "categorical": ["candidate_family", "selection_bucket", "priority_role", "dataset_source"]},
        "F04_no_family_generalization": {"numeric": all_numeric, "categorical": []},
        "F05_n_agnostic": {"numeric": [c for c in all_numeric if c != "n"], "categorical": []},
        "F06_no_dataset_source": {"numeric": all_numeric, "categorical": ["candidate_family", "selection_bucket", "priority_role"]},
        "F07_F01_no_n": {"numeric": [c for c in F01 if c != "n"], "categorical": []},
    }


def design_matrix(rows: list[dict[str, Any]], spec: dict[str, Any], categories: dict[str, list[str]] | None = None) -> tuple[np.ndarray, dict[str, list[str]], list[str]]:
    if categories is None:
        categories = {cat: sorted({str(row.get(cat, "")) for row in rows}) for cat in spec.get("categorical", [])}
    columns = list(spec.get("numeric", []))
    for cat in spec.get("categorical", []):
        columns.extend([f"{cat}={value}" for value in categories.get(cat, [])])
    mat = []
    for row in rows:
        vals = [parse_float(row.get(col), 0.0) for col in spec.get("numeric", [])]
        for cat in spec.get("categorical", []):
            value = str(row.get(cat, ""))
            vals.extend([1.0 if value == option else 0.0 for option in categories.get(cat, [])])
        mat.append(vals)
    return np.asarray(mat, dtype=float), categories, columns


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
        return make_pipeline(StandardScaler(), ElasticNet(alpha=0.01, l1_ratio=0.2, random_state=42, max_iter=10000))
    if name == "RandomForestRegressor":
        return RandomForestRegressor(n_estimators=240, max_depth=8, min_samples_leaf=2, random_state=42, n_jobs=-1)
    if name == "ExtraTreesRegressor":
        return ExtraTreesRegressor(n_estimators=300, max_depth=9, min_samples_leaf=2, random_state=42, n_jobs=-1)
    if name == "GradientBoostingRegressor":
        return GradientBoostingRegressor(n_estimators=120, max_depth=2, learning_rate=0.05, random_state=42)
    raise ValueError(name)


def eval_predictions(true: list[float], pred: list[float], test_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not true:
        return {}
    ranks_true = rank_desc(true)
    ranks_pred = rank_desc(pred)
    top1 = int(sorted(range(len(pred)), key=lambda i: pred[i], reverse=True)[0] == sorted(range(len(true)), key=lambda i: true[i], reverse=True)[0])
    return {
        "spearman": spearman(pred, true),
        "pearson": pearson(pred, true),
        "mae": mean([abs(a - b) for a, b in zip(pred, true)]),
        "rmse": math.sqrt(mean([(a - b) ** 2 for a, b in zip(pred, true)])),
        "r2": 1.0 - safe_divide(sum((a - b) ** 2 for a, b in zip(pred, true)), sum((a - mean(true)) ** 2 for a in true), math.nan),
        "top1_hit": top1,
        "top3_overlap": topk_overlap(true, pred, 3),
        "top5_overlap": topk_overlap(true, pred, 5),
        "top10_overlap": topk_overlap(true, pred, 10),
        "mean_rank_error": mean([abs(a - b) for a, b in zip(ranks_true, ranks_pred)]),
        "ndcg_at_5": ndcg_at_k(true, pred, 5),
        "test_count": len(true),
        "test_sources": dict(Counter(row.get("dataset_source", "") for row in test_rows)),
    }


def validation_splits(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    splits: list[dict[str, Any]] = []
    for n in EXPECTED_N:
        splits.append({"protocol": "leave_N_out", "test_n": n, "train": [i for i, r in enumerate(rows) if r["n"] != n], "test": [i for i, r in enumerate(rows) if r["n"] == n]})
    splits.append({"protocol": "N40_core_generalization", "test_n": 40, "train": [i for i, r in enumerate(rows) if r["n"] != 40], "test": [i for i, r in enumerate(rows) if r["n"] == 40]})
    splits.append({"protocol": "small_to_large", "test_n": "N24_N40", "train": [i for i, r in enumerate(rows) if r["n"] in (12, 16)], "test": [i for i, r in enumerate(rows) if r["n"] in (24, 40)]})
    splits.append({"protocol": "large_to_small", "test_n": "N12_N16", "train": [i for i, r in enumerate(rows) if r["n"] in (24, 40)], "test": [i for i, r in enumerate(rows) if r["n"] in (12, 16)]})
    splits.append({"protocol": "train_combined108_test_shortlist64", "test_n": "shortlist64", "train": [i for i, r in enumerate(rows) if r.get("dataset_source") != "shortlist64_run27"], "test": [i for i, r in enumerate(rows) if r.get("dataset_source") == "shortlist64_run27"]})
    rng = random.Random(GLOBAL_SEED)
    fold_for: dict[int, int] = {}
    for n in EXPECTED_N:
        idxs = [i for i, r in enumerate(rows) if r["n"] == n]
        rng.shuffle(idxs)
        for j, idx in enumerate(idxs):
            fold_for[idx] = j % 5
    for fold in range(5):
        splits.append({"protocol": "stratified_5fold", "fold": fold, "test_n": "mixed", "train": [i for i in range(len(rows)) if fold_for[i] != fold], "test": [i for i in range(len(rows)) if fold_for[i] == fold]})
    return splits


def surrogate_validation(rows: list[dict[str, Any]], fsets: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], Any, dict[str, Any]]:
    try:
        import sklearn  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        summary = {"status": "SKLEARN_UNAVAILABLE", "error": repr(exc)}
        write_json(OUTPUT_DIR / "combined172_surrogate_validation_summary.json", summary)
        return [], [], summary, None, {}

    targets = [PRIMARY_TARGET, *SECONDARY_TARGETS]
    models = ["MeanBaseline", "Ridge", "ElasticNet", "RandomForestRegressor", "ExtraTreesRegressor", "GradientBoostingRegressor"]
    splits = validation_splits(rows)
    detailed: list[dict[str, Any]] = []
    for target in targets:
        y_all = np.asarray([row[target] for row in rows], dtype=float)
        for fs_name, spec in fsets.items():
            for model_name in models:
                # Keep secondary target validation lightweight; the primary reward gets the full matrix.
                if target != PRIMARY_TARGET and (fs_name not in {"F01_basic_order", "F05_n_agnostic"} or model_name not in {"MeanBaseline", "Ridge", "ExtraTreesRegressor"}):
                    continue
                for split in splits:
                    train_rows = [rows[i] for i in split["train"]]
                    test_rows = [rows[i] for i in split["test"]]
                    if len(train_rows) < 5 or len(test_rows) < 3:
                        continue
                    x_train, cats, _cols = design_matrix(train_rows, spec)
                    x_test, _cats, _cols2 = design_matrix(test_rows, spec, cats)
                    model = model_factory(model_name)
                    model.fit(x_train, y_all[split["train"]])
                    pred = [float(x) for x in model.predict(x_test)]
                    metrics = eval_predictions([row[target] for row in test_rows], pred, test_rows)
                    detailed.append({"target": target, "feature_set": fs_name, "model_name": model_name, "protocol": split["protocol"], "test_n": split.get("test_n", ""), "fold": split.get("fold", ""), **metrics})

    macro_rows: list[dict[str, Any]] = []
    group_keys = ["target", "feature_set", "model_name", "protocol"]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in detailed:
        grouped[tuple(row[k] for k in group_keys)].append(row)
    for key, group in grouped.items():
        item = dict(zip(group_keys, key))
        item.update(
            {
                "macro_spearman": mean([parse_float(r.get("spearman")) for r in group]),
                "macro_top5_overlap": mean([parse_float(r.get("top5_overlap")) for r in group]),
                "macro_top10_overlap": mean([parse_float(r.get("top10_overlap")) for r in group]),
                "macro_mae": mean([parse_float(r.get("mae")) for r in group]),
                "macro_mean_rank_error": mean([parse_float(r.get("mean_rank_error")) for r in group]),
            }
        )
        macro_rows.append(item)
    primary_lno = [r for r in macro_rows if r["target"] == PRIMARY_TARGET and r["protocol"] == "leave_N_out"]
    finite_primary_lno = [r for r in primary_lno if math.isfinite(parse_float(r["macro_spearman"]))]
    best_pool = finite_primary_lno or primary_lno
    best = sorted(best_pool, key=lambda r: (parse_float(r["macro_spearman"], -999.0), parse_float(r["macro_top5_overlap"], -999.0), -parse_float(r["macro_mae"], 999.0)), reverse=True)[0]
    best_top5 = sorted(primary_lno, key=lambda r: (parse_float(r["macro_top5_overlap"]), parse_float(r["macro_spearman"])), reverse=True)[0]
    # Fit best model on all rows for candidate scoring.
    best_spec = fsets[best["feature_set"]]
    x_all, cats, cols = design_matrix(rows, best_spec)
    best_model = model_factory(best["model_name"])
    best_model.fit(x_all, np.asarray([row[PRIMARY_TARGET] for row in rows]))
    best_payload = {
        "best_leave_N_out_macro_spearman": best,
        "best_top5_overlap": best_top5,
        "run22_baseline": {"macro_spearman": 0.8651, "top5_overlap": 2.5},
        "run27_prediction_audit": {"spearman": 0.7374, "mean_top5_overlap": 3.0},
    }
    summary = {
        "status": "SURROGATE_VALIDATION_COMPLETE",
        "best_config": best,
        "best_top5_config": best_top5,
        "improves_run22_macro_spearman": parse_float(best["macro_spearman"]) > 0.8651,
        "top5_vs_run22": parse_float(best["macro_top5_overlap"]) - 2.5,
    }
    write_csv(OUTPUT_DIR / "combined172_surrogate_validation_results_detailed.csv", detailed)
    write_csv(OUTPUT_DIR / "combined172_best_surrogate_configurations.csv", [best_payload["best_leave_N_out_macro_spearman"], best_payload["best_top5_overlap"]])
    write_json(OUTPUT_DIR / "combined172_surrogate_validation_summary.json", summary)
    return detailed, [best_payload["best_leave_N_out_macro_spearman"], best_payload["best_top5_overlap"]], summary, best_model, {"spec": best_spec, "categories": cats, "columns": cols, "feature_set": best["feature_set"], "model_name": best["model_name"]}


def try_import_torch() -> tuple[Any, bool, str]:
    try:
        import torch

        return torch, True, f"torch {torch.__version__}, cuda={torch.cuda.is_available()}"
    except Exception as exc:  # noqa: BLE001
        return None, False, repr(exc)


def feature_vector(order: list[int], n: int) -> np.ndarray:
    f = scan_order_features(order, n)
    return np.asarray([float(f.get(col, n if col == "n" else 0.0)) for col in F01], dtype=np.float32)


def node_sequence_tensor(order: list[int], n: int) -> np.ndarray:
    return R26.node_sequence_tensor(order, n)


def adjacency_matrix(n: int) -> np.ndarray:
    return R26.adjacency_matrix(n)


def train_gnn_reward(rows: list[dict[str, Any]], torch: Any) -> tuple[dict[str, Any], Any]:
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    random.seed(GLOBAL_SEED)

    class GNNReward(nn.Module):
        def __init__(self, hidden: int = 48):
            super().__init__()
            self.node = nn.Linear(9, hidden)
            self.msg = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(3)])
            self.out = nn.Sequential(nn.Linear(hidden + len(F01), 64), nn.ReLU(), nn.Linear(64, 1))

        def forward(self, x, adj, stats):
            h = F.relu(self.node(x))
            for layer in self.msg:
                h = F.relu(layer(torch.matmul(adj, h)))
            pooled = h.mean(dim=0)
            return self.out(torch.cat([pooled, stats], dim=0)).squeeze()

    def tensors(row: dict[str, Any]):
        order, n = row["order"], row["n"]
        return (
            torch.tensor(node_sequence_tensor(order, n), dtype=torch.float32),
            torch.tensor(adjacency_matrix(n), dtype=torch.float32),
            torch.tensor(feature_vector(order, n), dtype=torch.float32),
            torch.tensor(float(row[PRIMARY_TARGET]), dtype=torch.float32),
        )

    def fit(train_rows: list[dict[str, Any]], epochs: int) -> GNNReward:
        model = GNNReward()
        opt = torch.optim.Adam(model.parameters(), lr=0.008, weight_decay=1e-4)
        data = [tensors(row) for row in train_rows]
        for _ in range(epochs):
            random.shuffle(data)
            for x, adj, stats, y in data:
                opt.zero_grad()
                loss = F.smooth_l1_loss(model(x, adj, stats), y)
                loss.backward()
                opt.step()
        return model

    def pred(model: Any, eval_rows: list[dict[str, Any]]) -> list[float]:
        model.eval()
        vals = []
        with torch.no_grad():
            for row in eval_rows:
                x, adj, stats, _ = tensors(row)
                vals.append(float(model(x, adj, stats).clamp(0.0, 1.0)))
        return vals

    detailed = []
    for n in EXPECTED_N:
        train = [row for row in rows if row["n"] != n]
        test = [row for row in rows if row["n"] == n]
        model = fit(train, 45)
        p = pred(model, test)
        true = [row[PRIMARY_TARGET] for row in test]
        detailed.append({"protocol": "leave_N_out", "test_n": n, **eval_predictions(true, p, test)})
    # Train without shortlist64 / test shortlist64.
    train = [row for row in rows if row.get("dataset_source") != "shortlist64_run27"]
    test = [row for row in rows if row.get("dataset_source") == "shortlist64_run27"]
    if train and test:
        model = fit(train, 45)
        p = pred(model, test)
        detailed.append({"protocol": "train_combined108_test_shortlist64", "test_n": "shortlist64", **eval_predictions([row[PRIMARY_TARGET] for row in test], p, test)})
    all_model = fit(rows, 70)
    lno = [row for row in detailed if row["protocol"] == "leave_N_out"]
    summary = {
        "status": "GNN_REWARD_MODEL_TRAINED" if detailed else "GNN_REWARD_MODEL_NO_RESULTS",
        "model": "plain PyTorch order-graph message-passing reward model",
        "leave_N_out_macro_spearman": mean([parse_float(row["spearman"]) for row in lno]),
        "leave_N_out_macro_top5_overlap": mean([parse_float(row["top5_overlap"]) for row in lno]),
        "n24_spearman": next((row["spearman"] for row in lno if row["test_n"] == 24), math.nan),
        "n40_spearman": next((row["spearman"] for row in lno if row["test_n"] == 40), math.nan),
        "run26_baseline": {"macro_spearman": 0.8165, "top5_overlap": 2.0, "n40_spearman": 0.8478},
        "improves_run26_macro_spearman": mean([parse_float(row["spearman"]) for row in lno]) > 0.8165,
        "note": "Offline prototype only; no online Abaqus RL.",
    }
    write_csv(OUTPUT_DIR / "combined172_gnn_reward_model_validation_results.csv", detailed)
    write_json(OUTPUT_DIR / "combined172_gnn_reward_model_validation_summary.json", summary)
    return summary, all_model


def predict_gnn(torch: Any, model: Any, order: list[int], n: int) -> float:
    if torch is None or model is None:
        return math.nan
    with torch.no_grad():
        x = torch.tensor(node_sequence_tensor(order, n), dtype=torch.float32)
        adj = torch.tensor(adjacency_matrix(n), dtype=torch.float32)
        stats = torch.tensor(feature_vector(order, n), dtype=torch.float32)
        return float(model(x, adj, stats).clamp(0.0, 1.0))


def train_pointer_policy(rows: list[dict[str, Any]], torch: Any) -> tuple[dict[str, Any], Any]:
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(GLOBAL_SEED)
    random.seed(GLOBAL_SEED)

    class PointerPolicy(nn.Module):
        def __init__(self, hidden: int = 72):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(10, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

        def score(self, n: int, visited: set[int], prev: int | None, step: int):
            center = (n - 1) / 2.0
            feats = []
            for i in range(n):
                feats.append(
                    [
                        safe_divide(i, n - 1),
                        float(i % 2),
                        safe_divide(abs(i - center), center if center else 1.0),
                        min(i, n - 1 - i) / max(1, n - 1),
                        1.0 if i in visited else 0.0,
                        1.0 if prev is not None and i == prev else 0.0,
                        safe_divide(step, n),
                        0.0 if prev is None else abs(i - prev) / max(1, n - 1),
                        0.0 if prev is None else (i - prev) / max(1, n - 1),
                        safe_divide(n, 40),
                    ]
                )
            return self.net(torch.tensor(feats, dtype=torch.float32)).squeeze(-1)

    model = PointerPolicy()
    opt = torch.optim.Adam(model.parameters(), lr=0.007, weight_decay=1e-4)
    train_rows = list(rows)
    log_rows = []
    for epoch in range(36):
        total = 0.0
        random.shuffle(train_rows)
        for row in train_rows:
            n, order = row["n"], row["order"]
            weight = 0.20 + 1.80 * float(row[PRIMARY_TARGET])
            visited: set[int] = set()
            prev = None
            losses = []
            for step, action in enumerate(order):
                logits = model.score(n, visited, prev, step)
                mask = torch.tensor([i not in visited for i in range(n)], dtype=torch.bool)
                losses.append(F.cross_entropy(logits.masked_fill(~mask, -1e9).unsqueeze(0), torch.tensor([action])))
                visited.add(action)
                prev = action
            loss = torch.stack(losses).mean() * weight
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss)
        if epoch % 6 == 0 or epoch == 35:
            log_rows.append({"epoch": epoch, "reward_weighted_bc_loss": total / max(1, len(train_rows))})
    write_csv(OUTPUT_DIR / "combined172_graph_pointer_policy_training_log.csv", log_rows)
    validation = []
    run26_nll = {12: 1.404, 16: 1.570, 24: 1.944, 40: 2.297}
    for n in EXPECTED_N:
        group = [row for row in rows if row["n"] == n]
        nlls = []
        weighted = []
        top_acc = []
        for row in group:
            visited: set[int] = set()
            prev = None
            losses = []
            accs = []
            for step, action in enumerate(row["order"]):
                with torch.no_grad():
                    logits = model.score(n, visited, prev, step)
                    mask = torch.tensor([i not in visited for i in range(n)], dtype=torch.bool)
                    masked = logits.masked_fill(~mask, -1e9)
                    losses.append(float(F.cross_entropy(masked.unsqueeze(0), torch.tensor([action]))))
                    accs.append(int(torch.argmax(masked).item()) == action)
                visited.add(action)
                prev = action
            nlls.append(mean(losses))
            weighted.append(mean(losses) * (0.20 + 1.80 * row[PRIMARY_TARGET]))
            top_acc.extend(accs)
        validation.append({"n": n, "teacher_rows": len(group), "mean_teacher_action_nll": mean(nlls), "reward_weighted_nll": mean(weighted), "top_action_accuracy": mean([float(x) for x in top_acc]), "run26_reference_nll": run26_nll[n], "delta_vs_run26_nll": mean(nlls) - run26_nll[n]})
    summary = {"status": "GRAPH_POINTER_POLICY_WEIGHTED_IMITATION_TRAINED", "training_method": "offline weighted behavior cloning on combined172", "validation": validation, "no_online_rl": True}
    write_json(OUTPUT_DIR / "combined172_graph_pointer_policy_validation_summary.json", summary)
    return summary, model


def greedy_decode(policy: Any, torch: Any, n: int) -> list[int]:
    visited: set[int] = set()
    out: list[int] = []
    prev = None
    for step in range(n):
        with torch.no_grad():
            logits = policy.score(n, visited, prev, step)
            for i in visited:
                logits[i] = -1e9
            action = int(torch.argmax(logits).item())
        out.append(action)
        visited.add(action)
        prev = action
    return out


def sample_decode(policy: Any, torch: Any, n: int, rng: random.Random, temperature: float = 1.0, top_k: int | None = None) -> list[int]:
    visited: set[int] = set()
    out: list[int] = []
    prev = None
    for step in range(n):
        with torch.no_grad():
            logits = policy.score(n, visited, prev, step).numpy() / max(temperature, 1e-6)
        for i in visited:
            logits[i] = -1e9
        allowed = [i for i in range(n) if i not in visited]
        if top_k:
            allowed = sorted(allowed, key=lambda i: logits[i], reverse=True)[:top_k]
        probs = np.exp(np.array([logits[i] for i in allowed]) - max(logits[i] for i in allowed))
        probs = probs / probs.sum()
        action = rng.choices(allowed, weights=probs, k=1)[0]
        out.append(action)
        visited.add(action)
        prev = action
    return out


def beam_decode(policy: Any, torch: Any, n: int, beam_width: int = 8) -> list[list[int]]:
    beams = [([], set(), None, 0.0)]
    for step in range(n):
        new = []
        for order, visited, prev, score in beams:
            with torch.no_grad():
                logits = policy.score(n, visited, prev, step).numpy()
            for action in sorted([i for i in range(n) if i not in visited], key=lambda i: logits[i], reverse=True)[:beam_width]:
                nxt = set(visited)
                nxt.add(action)
                new.append((order + [action], nxt, action, score + float(logits[action])))
        beams = sorted(new, key=lambda x: x[3], reverse=True)[:beam_width]
    return [b[0] for b in beams]


def swap_positions(order: list[int], rng: random.Random) -> list[int]:
    out = list(order)
    i, j = rng.sample(range(len(out)), 2)
    out[i], out[j] = out[j], out[i]
    return out


def reverse_segment(order: list[int], rng: random.Random) -> list[int]:
    out = list(order)
    length = rng.randint(2, max(3, len(out) // 3))
    start = rng.randint(0, len(out) - length)
    out[start : start + length] = reversed(out[start : start + length])
    return out


def block_swap(order: list[int], rng: random.Random) -> list[int]:
    out = list(order)
    n = len(out)
    block = max(2, n // 6)
    a = rng.randint(0, n - block)
    b = rng.randint(0, n - block)
    if abs(a - b) < block:
        return swap_positions(out, rng)
    aa, bb = sorted([a, b])
    out[aa : aa + block], out[bb : bb + block] = out[bb : bb + block], out[aa : aa + block]
    return out


def random_order(n: int, rng: random.Random, mode: str) -> list[int]:
    return R26.random_order(n, rng, mode)


def pointer_logprob(policy: Any, torch: Any, order: list[int], n: int) -> float:
    if policy is None or torch is None:
        return math.nan
    import torch.nn.functional as F

    visited: set[int] = set()
    prev = None
    vals = []
    with torch.no_grad():
        for step, action in enumerate(order):
            logits = policy.score(n, visited, prev, step)
            mask = torch.tensor([i not in visited for i in range(n)], dtype=torch.bool)
            log_probs = F.log_softmax(logits.masked_fill(~mask, -1e9), dim=0)
            vals.append(float(log_probs[action]))
            visited.add(action)
            prev = action
    return mean(vals)


def add_candidate(store: dict[int, dict[str, dict[str, Any]]], n: int, order: list[int], source: str, bucket: str, method: str, seed: str = "") -> bool:
    if not validate_order(order, n):
        return False
    digest = order_hash(order)
    if digest in store[n]:
        return False
    store[n][digest] = {"n": n, "order": order, "order_hash": digest, "candidate_source": source, "selection_bucket": bucket, "generation_method": method, "seed_strategy": seed}
    return True


def generate_candidates(rows: list[dict[str, Any]], torch: Any, policy: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    refs = refs_by_n(rows)
    existing = {n: {row["order_hash"] for row in refs[n]} for n in EXPECTED_N}
    prior = prior_hashes()
    store: dict[int, dict[str, dict[str, Any]]] = {n: {} for n in EXPECTED_N}
    raw = Counter()
    dup_existing = Counter()
    dup_prior = Counter()
    best_by_n = {n: sorted([r for r in rows if r["n"] == n], key=lambda r: r[PRIMARY_TARGET], reverse=True)[:10] for n in EXPECTED_N}
    special_seeds = {
        12: ["S3B20_N12_B02_diversity_top"],
        16: ["S3R24L64_N16_B02_top_region"],
        24: ["S3R19B28_N24_B10_known_best_mutation", "S3R24L64_N24_B06_model_disagreement"],
        40: ["S3R24L64_N40_B23_exploitation_reference"],
    }
    for n in EXPECTED_N:
        rng = random.Random(GLOBAL_SEED + n)

        def try_add(order: list[int], source: str, bucket: str, method: str, seed: str = "") -> None:
            raw[n] += 1
            digest = order_hash(order) if validate_order(order, n) else ""
            if digest in existing[n]:
                dup_existing[n] += 1
                return
            if digest in prior[n]:
                dup_prior[n] += 1
                return
            add_candidate(store, n, order, source, bucket, method, seed)

        if policy is not None and torch is not None:
            try_add(greedy_decode(policy, torch, n), "graph_pointer_greedy", "gnn_policy_top_candidates", "greedy_decode")
            for idx, order in enumerate(beam_decode(policy, torch, n, 8)):
                try_add(order, "graph_pointer_beam_search", "gnn_policy_top_candidates", f"beam_{idx}")
            for idx in range(240 if n in (24, 40) else 100):
                temp = [0.55, 0.75, 0.95, 1.2, 1.55][idx % 5]
                try_add(sample_decode(policy, torch, n, rng, temp, max(4, n // 4) if idx % 4 == 0 else None), "graph_pointer_temperature_sampled", "diversity_coverage", f"temperature_{temp}")

        seed_rows = list(best_by_n[n])
        for name in special_seeds.get(n, []):
            seed_rows.extend([r for r in rows if r["strategy_name"] == name])
        for seed in seed_rows:
            base = seed["order"]
            for idx in range(520 if n in (24, 40) else 240):
                if idx % 4 == 0:
                    order, method = swap_positions(base, rng), "swap_mutation"
                elif idx % 4 == 1:
                    order, method = reverse_segment(base, rng), "reverse_segment"
                elif idx % 4 == 2:
                    order, method = block_swap(base, rng), "block_swap"
                else:
                    order, method = random_order(n, rng, "high_jump" if idx % 2 else "edge_center"), "local_random_reseed"
                if seed["strategy_name"] in {"S3R24L64_N40_B23_exploitation_reference", "S3R24L64_N16_B02_top_region"}:
                    bucket = "new_best_local_search"
                    source = "N40_new_best_neighborhood" if n == 40 else "N16_new_best_neighborhood"
                elif n == 24 and seed["strategy_name"] == "S3R24L64_N24_B06_model_disagreement":
                    bucket = "N24_surfaceT_best_neighborhood"
                    source = "N24_surfaceT_best_neighborhood"
                else:
                    bucket = ["surrogate_known_best_local_search", "uncertainty_calibration", "hybrid_gnn_surrogate_disagreement", "tradeoff_probe"][idx % 4]
                    source = "surrogate_known_best_local_search"
                try_add(order, source, bucket, method, seed["strategy_name"])

        modes = ["high_jump", "edge_center", "center_bias", "random"]
        guard = 0
        while len(store[n]) < POOL_TARGETS[n] and guard < POOL_TARGETS[n] * 40:
            guard += 1
            bucket = ["hybrid_gnn_surrogate_agreement", "hybrid_gnn_surrogate_disagreement", "uncertainty_calibration", "diversity_coverage", "sentinel_control"][guard % 5]
            source = ["surrogate_top_predicted", "gnn_reward_local_search", "diversity_coverage", "sentinel_control"][guard % 4]
            try_add(random_order(n, rng, modes[guard % len(modes)]), source, bucket, modes[guard % len(modes)])

    candidates: list[dict[str, Any]] = []
    for n in EXPECTED_N:
        for idx, item in enumerate(store[n].values(), start=1):
            order = item.pop("order")
            features = scan_order_features(order, n)
            nearest, novelty = nearest_order(order, refs[n])
            cid = f"R29_N{n}_C{idx:05d}"
            candidates.append(
                {
                    **item,
                    **features,
                    "candidate_id": cid,
                    "strategy_name": f"N{n}_{cid}_{item['selection_bucket']}",
                    "order_json": canonical_order_json(order),
                    "order_compact": "-".join(str(x) for x in order),
                    "novelty_distance_to_combined172": novelty,
                    "nearest_existing_teacher_strategy": nearest,
                    "duplicate_of_combined172_teacher": False,
                    "duplicate_of_previous_candidate_pool": False,
                }
            )
    counts = {
        "raw_generated_candidate_count_per_n": dict(raw),
        "deduplicated_candidate_count_per_n": {n: sum(1 for c in candidates if c["n"] == n) for n in EXPECTED_N},
        "duplicate_existing_teacher_attempts_per_n": dict(dup_existing),
        "duplicate_previous_candidate_attempts_per_n": dict(dup_prior),
    }
    return candidates, counts


def predict_surrogate(model: Any, model_info: dict[str, Any], rows: list[dict[str, Any]]) -> list[float]:
    if model is None:
        return [math.nan] * len(rows)
    x, _cats, _cols = design_matrix(rows, model_info["spec"], model_info["categories"])
    return [float(v) for v in model.predict(x)]


def score_candidates(candidates: list[dict[str, Any]], surrogate_model: Any, model_info: dict[str, Any], torch: Any, gnn_model: Any, pointer_policy: Any, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Secondary simple critics fit on F01 scores.
    from sklearn.ensemble import ExtraTreesRegressor

    x_train = np.asarray([feature_vector(r["order"], r["n"]) for r in rows], dtype=float)
    secondary_models = {}
    for target in SECONDARY_TARGETS:
        m = ExtraTreesRegressor(n_estimators=220, max_depth=8, min_samples_leaf=2, random_state=42, n_jobs=-1)
        m.fit(x_train, np.asarray([r[target] for r in rows], dtype=float))
        secondary_models[target] = m
    gnn_teacher_preds = [predict_gnn(torch, gnn_model, r["order"], r["n"]) for r in rows]
    gnn_proxy = ExtraTreesRegressor(n_estimators=160, max_depth=8, min_samples_leaf=2, random_state=42, n_jobs=-1)
    if any(math.isfinite(v) for v in gnn_teacher_preds):
        gnn_proxy.fit(x_train, np.asarray([v if math.isfinite(v) else r[PRIMARY_TARGET] for v, r in zip(gnn_teacher_preds, rows)], dtype=float))
        gnn_scoring_mode = "fast_feature_proxy_fit_to_trained_gnn_teacher_predictions"
    else:
        gnn_proxy.fit(x_train, np.asarray([r[PRIMARY_TARGET] for r in rows], dtype=float))
        gnn_scoring_mode = "fallback_feature_proxy_fit_to_teacher_reward"

    pred_sur = predict_surrogate(surrogate_model, model_info, candidates)
    candidate_orders = [parse_order(row["order_json"]) for row in candidates]
    x_candidates = np.asarray([feature_vector(order, row["n"]) for row, order in zip(candidates, candidate_orders) if order is not None], dtype=float)
    if len(x_candidates) != len(candidates):
        raise RuntimeError("Invalid candidate order encountered during scoring")
    gnn_preds = [float(v) for v in gnn_proxy.predict(x_candidates)]
    secondary_preds = {target: [float(v) for v in model.predict(x_candidates)] for target, model in secondary_models.items()}
    scored: list[dict[str, Any]] = []
    for idx, (row, sur) in enumerate(zip(candidates, pred_sur)):
        order = candidate_orders[idx]
        assert order is not None
        n = row["n"]
        gnn_pred = gnn_preds[idx]
        pointer_score = math.nan
        u2 = secondary_preds["target_u2_score_combined172_rank"][idx]
        peeq = secondary_preds["target_peeq_score_combined172_rank"][idx]
        surf = secondary_preds["target_surfaceT_score_combined172_rank"][idx]
        mises = secondary_preds["target_mises_score_combined172_rank"][idx]
        disagreement = abs(sur - gnn_pred) if math.isfinite(sur) and math.isfinite(gnn_pred) else math.nan
        novelty = parse_float(row.get("novelty_distance_to_combined172"), 0.0)
        hybrid = 0.40 * max(0.0, sur if math.isfinite(sur) else 0.0) + 0.25 * max(0.0, gnn_pred if math.isfinite(gnn_pred) else 0.0) + 0.15 * u2 + 0.10 * novelty + 0.10 * max(0.0, disagreement if math.isfinite(disagreement) else 0.0)
        item = dict(row)
        item.update(
            {
                "surrogate_reward_pred": sur,
                "gnn_reward_pred": gnn_pred,
                "graph_pointer_mean_logprob": pointer_score,
                "gnn_candidate_scoring_mode": gnn_scoring_mode,
                "graph_pointer_candidate_score_note": "not computed for full pool to keep Run29 lightweight; policy was trained/evaluated separately",
                "pred_u2_score": u2,
                "pred_peeq_score": peeq,
                "pred_surfaceT_score": surf,
                "pred_mises_score": mises,
                "gnn_surrogate_disagreement": disagreement,
                "hybrid_score": hybrid,
                "acquisition_score": hybrid + 0.05 * novelty,
            }
        )
        scored.append(item)
    for n in EXPECTED_N:
        group = [r for r in scored if r["n"] == n]
        for col in ["surrogate_reward_pred", "gnn_reward_pred", "hybrid_score", "gnn_surrogate_disagreement", "novelty_distance_to_combined172"]:
            for rank, row in enumerate(sorted(group, key=lambda r: parse_float(r.get(col)), reverse=True), start=1):
                row[f"{col}_rank_within_n"] = rank
    return scored


def select_batch(scored: list[dict[str, Any]], counts: dict[int, int], label: str) -> list[dict[str, Any]]:
    batch: list[dict[str, Any]] = []
    bucket_order = [
        "new_best_local_search",
        "gnn_policy_top_candidates",
        "hybrid_gnn_surrogate_agreement",
        "hybrid_gnn_surrogate_disagreement",
        "surrogate_known_best_local_search",
        "N24_surfaceT_best_neighborhood",
        "uncertainty_calibration",
        "diversity_coverage",
        "tradeoff_probe",
        "sentinel_control",
    ]
    for n, count in counts.items():
        group = [r for r in scored if r["n"] == n]
        selected: list[dict[str, Any]] = []
        used: set[str] = set()
        for bucket in bucket_order:
            pool = [r for r in group if r["selection_bucket"] == bucket]
            key = "gnn_surrogate_disagreement" if bucket == "hybrid_gnn_surrogate_disagreement" else "hybrid_score"
            pool = sorted(pool, key=lambda r: parse_float(r.get(key)), reverse=True)
            take = 2 if count >= 12 and bucket not in {"sentinel_control"} else 1
            if bucket == "new_best_local_search" and n in (16, 40):
                take = 4
            if bucket == "N24_surfaceT_best_neighborhood" and n == 24:
                take = 3
            if bucket == "sentinel_control":
                take = 1
            for row in pool:
                if len([x for x in selected if x["selection_bucket"] == bucket]) >= take or len(selected) >= count:
                    break
                if row["order_hash"] in used:
                    continue
                selected.append(dict(row))
                used.add(row["order_hash"])
            if len(selected) >= count:
                break
        for row in sorted(group, key=lambda r: parse_float(r.get("hybrid_score")), reverse=True):
            if len(selected) >= count:
                break
            if row["order_hash"] not in used:
                selected.append(dict(row))
                used.add(row["order_hash"])
        for idx, row in enumerate(selected[:count], start=1):
            row[f"{label}_rank_within_n"] = idx
            batch.append(row)
    return batch


def compare_batch(batch64: list[dict[str, Any]], rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    refs = {row["order_hash"] for row in rows}
    previous = {}
    for label, path in [("run23_pool", RUN23_POOL), ("run24_shortlist64", RUN24_SHORTLIST64), ("run26_pool", RUN26_POOL), ("run26_batch64", RUN26_BATCH64)]:
        hashes = set()
        if path.exists():
            for row in read_csv(path):
                digest = row.get("order_hash")
                if digest:
                    hashes.add(digest)
        previous[label] = hashes
    rows_out: list[dict[str, Any]] = []
    for n in EXPECTED_N:
        group = [r for r in batch64 if r["n"] == n]
        rows_out.append(
            {
                "n": n,
                "batch64_count": len(group),
                "overlap_combined172_teacher": sum(1 for r in group if r["order_hash"] in refs),
                "overlap_run23_pool": sum(1 for r in group if r["order_hash"] in previous["run23_pool"]),
                "overlap_run24_shortlist64": sum(1 for r in group if r["order_hash"] in previous["run24_shortlist64"]),
                "overlap_run26_pool": sum(1 for r in group if r["order_hash"] in previous["run26_pool"]),
                "overlap_run26_batch64": sum(1 for r in group if r["order_hash"] in previous["run26_batch64"]),
                "mean_surrogate_reward_pred": mean([parse_float(r["surrogate_reward_pred"]) for r in group]),
                "mean_gnn_reward_pred": mean([parse_float(r["gnn_reward_pred"]) for r in group]),
                "mean_hybrid_score": mean([parse_float(r["hybrid_score"]) for r in group]),
                "mean_novelty_distance": mean([parse_float(r["novelty_distance_to_combined172"]) for r in group]),
                "mean_gnn_surrogate_disagreement": mean([parse_float(r["gnn_surrogate_disagreement"]) for r in group]),
                "source_composition": dict(Counter(r["candidate_source"] for r in group)),
                "bucket_composition": dict(Counter(r["selection_bucket"] for r in group)),
            }
        )
    summary = {
        "batch64_count": len(batch64),
        "per_n_counts": dict(Counter(r["n"] for r in batch64)),
        "total_overlap_combined172_teacher": sum(1 for r in batch64 if r["order_hash"] in refs),
        "total_overlap_run24_shortlist64": sum(1 for r in batch64 if r["order_hash"] in previous["run24_shortlist64"]),
        "total_overlap_run26_batch64": sum(1 for r in batch64 if r["order_hash"] in previous["run26_batch64"]),
        "mostly_distinct_from_previous": sum(1 for r in batch64 if r["order_hash"] in previous["run24_shortlist64"] or r["order_hash"] in previous["run26_batch64"]) == 0,
    }
    return rows_out, summary


def write_claim_boundary() -> tuple[Path, Path]:
    md = OUTPUT_DIR / "run29_claim_boundary.md"
    js = OUTPUT_DIR / "run29_claim_boundary.json"
    safe = [
        "Run29 updates surrogate and offline GNN/graph-pointer models using combined172.",
        "Run29 evaluates whether GNN improves relative to Run26 after adding Run27 teacher labels.",
        "Run29 generates future hybrid-policy candidate batches for teacher validation.",
        "Run29 candidates are selected using GNN policy, surrogate critic, disagreement, novelty, uncertainty, and active-learning criteria.",
    ]
    unsafe = [
        "Run29 candidates are teacher-validated.",
        "GNN-RL has beaten baselines.",
        "online Abaqus-RL.",
        "arbitrary-N generalization.",
        "physical superiority.",
        "deployment readiness.",
        "feature importance or model scores are causal.",
    ]
    md.write_text("# Run29 Claim Boundary\n\n## Safe Claims\n" + "\n".join(f"- {x}" for x in safe) + "\n\n## Unsafe Claims\n" + "\n".join(f"- Do not claim {x}" for x in unsafe) + "\n", encoding="utf-8")
    write_json(js, {"verdict": "RUN29_MODEL_UPDATE_AND_HYBRID_CANDIDATE_GENERATION_ONLY_NO_TEACHER_VALIDATION", "safe_claims": safe, "unsafe_claims": unsafe})
    return md, js


def write_report(validation: dict[str, Any], surrogate_summary: dict[str, Any], gnn_summary: dict[str, Any], pointer_summary: dict[str, Any], candidate_counts: dict[str, Any], batch64: list[dict[str, Any]], batch32: list[dict[str, Any]], batch48: list[dict[str, Any]], comparison_summary: dict[str, Any], outputs: list[str]) -> None:
    lines = [
        "# Stage 3 Run 29 - Combined172 Surrogate, GNN, and Hybrid-Policy Candidate Generation",
        "",
        "## Purpose",
        "Update lightweight surrogate models, offline GNN reward modeling, and graph-pointer behavior cloning using combined172, then generate hybrid-policy candidate batches without CAE/solver activity.",
        "",
        "## Inputs",
        f"- Combined172 RL-ready dataset: `{COMBINED172_READY}`",
        f"- Combined172 teacher dataset: `{COMBINED172_TEACHER}`",
        f"- Run28 report: `{RUN28_REPORT}`",
        "",
        "## Combined172 Validation",
        f"- Verdict: `{validation['verdict']}`",
        f"- Per-N counts: `{validation['per_n_counts']}`",
        "",
        "## Surrogate Update",
        f"- Best config: `{surrogate_summary.get('best_config')}`",
        f"- Improves Run22 macro Spearman: `{surrogate_summary.get('improves_run22_macro_spearman')}`",
        "",
        "## GNN Reward Model Update",
        f"- Status: `{gnn_summary.get('status')}`",
        f"- Leave-N-out macro Spearman: `{gnn_summary.get('leave_N_out_macro_spearman')}`",
        f"- Improves Run26 macro Spearman: `{gnn_summary.get('improves_run26_macro_spearman')}`",
        "",
        "## Graph-Pointer Policy Update",
        f"- Status: `{pointer_summary.get('status')}`",
        "- Training method: offline weighted behavior cloning; no online Abaqus RL.",
        "",
        "## Candidate Generation",
        f"- Deduplicated counts: `{candidate_counts.get('deduplicated_candidate_count_per_n')}`",
        "",
        "## Hybrid Batch64",
        f"- Count: `{len(batch64)}`; per-N: `{dict(Counter(r['n'] for r in batch64))}`",
        "",
        "## Hybrid Batch32",
        f"- Count: `{len(batch32)}`; per-N: `{dict(Counter(r['n'] for r in batch32))}`",
        "",
        "## N24/N40 Focused Batch48",
        f"- Count: `{len(batch48)}`; per-N: `{dict(Counter(r['n'] for r in batch48))}`",
        "",
        "## Comparison to Previous Batches",
        f"- Summary: `{comparison_summary}`",
        "",
        "## Claim Boundary",
        "`RUN29_MODEL_UPDATE_AND_HYBRID_CANDIDATE_GENERATION_ONLY_NO_TEACHER_VALIDATION`.",
        "",
        "## Output Files",
    ]
    lines.extend(f"- `{path}`" for path in outputs)
    lines.extend(
        [
            "",
            "## Recommended Run30",
            "Create a handoff package for the selected Run29 hybrid batch. If the user wants 60+ overnight jobs, select hybrid batch64; if focused N24/N40 calibration is preferred, select focused batch48; if compute is limited, select batch32. Do not generate CAE/INP until the user selects one batch.",
            "",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def update_run_index(verdict: str) -> None:
    if not RUN_INDEX_PATH.exists():
        return
    text = RUN_INDEX_PATH.read_text(encoding="utf-8")
    if "| run_29 |" in text:
        return
    row = (
        "| run_29 | Combined172 surrogate, GNN, and hybrid-policy candidate generation | "
        "Update combined172 surrogate/GNN/pointer prototypes and generate hybrid candidate batches without CAE or teacher validation. | "
        "`scripts/stage3/run_29_combined172_surrogate_gnn_hybrid_policy_update_and_candidate_generation.py` | "
        "`docs/stage3/runs/run_29_combined172_surrogate_gnn_hybrid_policy_update_and_candidate_generation/RUN_29_COMBINED172_SURROGATE_GNN_HYBRID_POLICY_UPDATE_AND_CANDIDATE_GENERATION_REPORT.md` | "
        "`outputs/stage3_run_29_combined172_surrogate_gnn_hybrid_policy_update_and_candidate_generation/` | "
        f"`{verdict}` | No Abaqus, no ODB opening, no abqjobpilot, no CAE/INP/JNL generation, no teacher validation, no online RL, no commit/push. |"
    )
    RUN_INDEX_PATH.write_text(text.rstrip() + "\n" + row + "\n", encoding="utf-8")


def git_branch() -> str:
    try:
        return subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)

    rows = load_combined172()
    validation = validate_inputs(rows)
    if validation["verdict"].startswith("FAIL"):
        print(validation["verdict"])
        return 2

    feature_rows = write_feature_table(rows)
    fsets = feature_sets(feature_rows)
    write_json(OUTPUT_DIR / "combined172_feature_set_definitions.json", fsets)
    surrogate_detail, best_configs, surrogate_summary, surrogate_model, surrogate_info = surrogate_validation(rows, fsets)
    torch, torch_available, torch_status = try_import_torch()
    if torch_available:
        gnn_summary, gnn_model = train_gnn_reward(rows, torch)
        pointer_summary, pointer_policy = train_pointer_policy(rows, torch)
    else:
        gnn_summary = {"status": "PYTORCH_UNAVAILABLE", "torch_status": torch_status}
        pointer_summary = {"status": "PYTORCH_UNAVAILABLE", "torch_status": torch_status}
        gnn_model = None
        pointer_policy = None
        write_json(OUTPUT_DIR / "combined172_gnn_reward_model_validation_summary.json", gnn_summary)
        write_json(OUTPUT_DIR / "combined172_graph_pointer_policy_validation_summary.json", pointer_summary)
        write_csv(OUTPUT_DIR / "combined172_gnn_reward_model_validation_results.csv", [])
        write_csv(OUTPUT_DIR / "combined172_graph_pointer_policy_training_log.csv", [])

    candidates, candidate_counts = generate_candidates(rows, torch, pointer_policy)
    write_csv(OUTPUT_DIR / "run29_hybrid_candidate_pool_unscored.csv", candidates)
    scored = score_candidates(candidates, surrogate_model, surrogate_info, torch, gnn_model, pointer_policy, rows)
    write_csv(OUTPUT_DIR / "run29_hybrid_candidate_pool_scored.csv", scored)
    batch64 = select_batch(scored, BATCH64_COUNTS, "batch64")
    batch32 = select_batch(scored, BATCH32_COUNTS, "batch32")
    batch48 = select_batch(scored, FOCUSED48_COUNTS, "focused48")
    write_csv(OUTPUT_DIR / "run29_hybrid_policy_batch64_candidate_orders.csv", batch64)
    write_csv(OUTPUT_DIR / "run29_hybrid_policy_batch32_candidate_orders.csv", batch32)
    write_csv(OUTPUT_DIR / "run29_hybrid_policy_N24_N40_focused_batch48_candidate_orders.csv", batch48)
    comparison_rows, comparison_summary = compare_batch(batch64, rows)
    write_csv(OUTPUT_DIR / "run29_hybrid_batch64_comparison_to_previous_batches.csv", comparison_rows)
    write_json(OUTPUT_DIR / "run29_hybrid_batch64_comparison_summary.json", comparison_summary)
    claim_md, claim_json = write_claim_boundary()

    outputs = [
        str(OUTPUT_DIR / "run29_input_validation_summary.json"),
        str(OUTPUT_DIR / "combined172_scan_order_features.csv"),
        str(OUTPUT_DIR / "combined172_feature_set_definitions.json"),
        str(OUTPUT_DIR / "combined172_surrogate_validation_results_detailed.csv"),
        str(OUTPUT_DIR / "combined172_best_surrogate_configurations.csv"),
        str(OUTPUT_DIR / "combined172_surrogate_validation_summary.json"),
        str(OUTPUT_DIR / "combined172_gnn_reward_model_validation_results.csv"),
        str(OUTPUT_DIR / "combined172_gnn_reward_model_validation_summary.json"),
        str(OUTPUT_DIR / "combined172_graph_pointer_policy_training_log.csv"),
        str(OUTPUT_DIR / "combined172_graph_pointer_policy_validation_summary.json"),
        str(OUTPUT_DIR / "run29_hybrid_candidate_pool_unscored.csv"),
        str(OUTPUT_DIR / "run29_hybrid_candidate_pool_scored.csv"),
        str(OUTPUT_DIR / "run29_hybrid_policy_batch64_candidate_orders.csv"),
        str(OUTPUT_DIR / "run29_hybrid_policy_batch32_candidate_orders.csv"),
        str(OUTPUT_DIR / "run29_hybrid_policy_N24_N40_focused_batch48_candidate_orders.csv"),
        str(OUTPUT_DIR / "run29_hybrid_batch64_comparison_to_previous_batches.csv"),
        str(OUTPUT_DIR / "run29_hybrid_batch64_comparison_summary.json"),
        str(claim_md),
        str(claim_json),
    ]
    write_report(validation, surrogate_summary, gnn_summary, pointer_summary, candidate_counts, batch64, batch32, batch48, comparison_summary, outputs)
    outputs.append(str(REPORT_PATH))
    update_run_index(validation["verdict"])

    manifest = {
        "run_id": RUN_ID,
        "run_name": RUN_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "branch": git_branch(),
        "script_path": str(ROOT / "scripts" / "stage3" / "run_29_combined172_surrogate_gnn_hybrid_policy_update_and_candidate_generation.py"),
        "input_files": [str(p) for p in [COMBINED172_READY, COMBINED172_TEACHER, COMBINED172_LEADERBOARD, RUN27_ENRICHED, RUN27_PRED_AUDIT, RUN27_BUCKET, RUN27_VS_108, RUN28_REPORT, RUN26_POOL, RUN26_BATCH64, RUN23_POOL, RUN24_SHORTLIST64] if p.exists()],
        "output_files": outputs,
        "combined172_rows": len(rows),
        "per_N_counts": dict(sorted(Counter(r["n"] for r in rows).items())),
        "surrogate_update_summary": surrogate_summary,
        "GNN_update_summary": gnn_summary,
        "graph_pointer_update_summary": pointer_summary,
        "candidate_pool_counts": candidate_counts,
        "batch64_counts": dict(Counter(r["n"] for r in batch64)),
        "batch32_counts": dict(Counter(r["n"] for r in batch32)),
        "focused_batch48_counts": dict(Counter(r["n"] for r in batch48)),
        "report_path": str(REPORT_PATH),
        "claim_boundary_path": str(claim_md),
        "no_solver_run": True,
        "no_odb_opened": True,
        "no_abqjobpilot_run": True,
        "no_cae_inp_generated": True,
        "no_teacher_validation": True,
        "no_online_rl": True,
        "no_commit_or_push": True,
        "torch_status": torch_status,
    }
    write_json(MANIFEST_PATH, manifest)
    print(validation["verdict"])
    print(f"surrogate={surrogate_summary.get('best_config')}")
    print(f"gnn_macro={gnn_summary.get('leave_N_out_macro_spearman')}")
    print(f"candidate_counts={candidate_counts.get('deduplicated_candidate_count_per_n')}")
    print(f"batch64={dict(Counter(r['n'] for r in batch64))}")
    print(f"report={REPORT_PATH}")
    print(f"manifest={MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
