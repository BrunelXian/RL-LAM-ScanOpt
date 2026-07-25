from __future__ import annotations

import csv
import hashlib
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
RUN_ID = "run_26_combined108_gnn_graph_pointer_policy_candidate_generation"
RUN_NAME = "combined108 GNN / graph-pointer policy prototype and candidate generation"

COMBINED108_READY = ROOT / "outputs" / "stage3_run_21_batch28_teacher_metrics_ingestion_and_combined108_ranking" / "combined108_RL_ready_dataset.csv"
COMBINED108_TEACHER = ROOT / "outputs" / "stage3_run_21_batch28_teacher_metrics_ingestion_and_combined108_ranking" / "combined108_teacher_dataset.csv"
COMBINED108_LEADERBOARD = ROOT / "outputs" / "stage3_run_21_batch28_teacher_metrics_ingestion_and_combined108_ranking" / "combined108_per_N_leaderboard.csv"
RUN22_FEATURES = ROOT / "outputs" / "stage3_run_22_combined108_surrogate_reward_model_validation_update" / "combined108_scan_order_features.csv"
RUN22_BEST = ROOT / "outputs" / "stage3_run_22_combined108_surrogate_reward_model_validation_update" / "combined108_best_surrogate_configurations.csv"
RUN22_DETAILED = ROOT / "outputs" / "stage3_run_22_combined108_surrogate_reward_model_validation_update" / "combined108_surrogate_validation_results_detailed.csv"
RUN23_SCORED = ROOT / "outputs" / "stage3_run_23_combined108_active_learning_coverage_calibration_design" / "run23_candidate_pool_scored.csv"
RUN24_SHORTLIST64 = ROOT / "outputs" / "stage3_run_24_run23_shortlist64_active_learning_handoff_package" / "stage3_run24_shortlist64_candidate_orders.csv"
RUN24_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_24_run23_shortlist64_active_learning_handoff_package" / "RUN_24_RUN23_SHORTLIST64_ACTIVE_LEARNING_HANDOFF_PACKAGE_REPORT.md"

OUTPUT_DIR = ROOT / "outputs" / "stage3_run_26_combined108_gnn_graph_pointer_policy_candidate_generation"
REPORT_DIR = ROOT / "docs" / "stage3" / "runs" / RUN_ID
REPORT_PATH = REPORT_DIR / "RUN_26_COMBINED108_GNN_GRAPH_POINTER_POLICY_CANDIDATE_GENERATION_REPORT.md"
MANIFEST_PATH = ROOT / "artifacts" / "manifests" / "stage3_run_26_manifest.json"
RUN_INDEX_PATH = ROOT / "docs" / "stage3" / "STAGE3_RUN_INDEX.md"

EXPECTED_N = [12, 16, 24, 40]
EXPECTED_COUNTS = {12: 24, 16: 24, 24: 30, 40: 30}
POOL_TARGETS = {12: 500, 16: 500, 24: 1500, 40: 2000}
BATCH64_COUNTS = {12: 8, 16: 8, 24: 24, 40: 24}
BATCH32_COUNTS = {12: 4, 16: 4, 24: 12, 40: 12}
PRIMARY_TARGET = "target_reward_combined108_u2_primary"
SCORE_TARGETS = [
    "target_u2_score_combined108_rank",
    "target_peeq_score_combined108_rank",
    "target_surfaceT_score_combined108_rank",
    "target_mises_score_combined108_rank",
]
GLOBAL_SEED = 26042


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
    return int(float(text))


def parse_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def parse_order(text: Any) -> list[int] | None:
    try:
        value = json.loads(str(text))
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(value, list):
        return None
    try:
        return [int(x) for x in value]
    except (TypeError, ValueError):
        return None


def order_hash(order: list[int]) -> str:
    return hashlib.sha1(",".join(str(x) for x in order).encode("ascii")).hexdigest()[:16]


def validate_order(order: list[int] | None, n: int) -> bool:
    return order is not None and len(order) == n and set(order) == set(range(n)) and len(set(order)) == n


def canonical_order_json(order: list[int]) -> str:
    return json.dumps(order, separators=(",", ":"))


def mean(values: list[float], default: float = math.nan) -> float:
    clean = [x for x in values if math.isfinite(x)]
    return statistics.fmean(clean) if clean else default


def std(values: list[float]) -> float:
    clean = [x for x in values if math.isfinite(x)]
    return statistics.pstdev(clean) if len(clean) > 1 else 0.0


def safe_divide(num: float, den: float, default: float = 0.0) -> float:
    return num / den if den else default


def spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return math.nan
    rx = rank_values(x)
    ry = rank_values(y)
    return pearson(rx, ry)


def pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return math.nan
    mx, my = mean(x), mean(y)
    vx = sum((a - mx) ** 2 for a in x)
    vy = sum((b - my) ** 2 for b in y)
    if vx <= 0 or vy <= 0:
        return math.nan
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(vx * vy)


def rank_values(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def kendall_distance(a: list[int], b: list[int]) -> float:
    if len(a) != len(b):
        return 1.0
    pos_b = {v: i for i, v in enumerate(b)}
    mapped = [pos_b[v] for v in a]
    inv = 0
    total = len(mapped) * (len(mapped) - 1) // 2
    for i in range(len(mapped)):
        for j in range(i + 1, len(mapped)):
            inv += mapped[i] > mapped[j]
    return safe_divide(inv, total)


def mutation_distance(a: list[int], b: list[int]) -> float:
    return sum(1 for x, y in zip(a, b) if x != y) / max(1, len(a))


def nearest_order(order: list[int], refs: list[dict[str, Any]]) -> tuple[str, float]:
    if not refs:
        return "", math.nan
    best_name = ""
    best_dist = math.inf
    for ref in refs:
        dist = kendall_distance(order, ref["order"])
        if dist < best_dist:
            best_name = ref["strategy_name"]
            best_dist = dist
    return best_name, best_dist


def scan_order_features(order: list[int], n: int) -> dict[str, Any]:
    jumps = [abs(order[i + 1] - order[i]) for i in range(len(order) - 1)]
    signed = [order[i + 1] - order[i] for i in range(len(order) - 1)]
    center = (n - 1) / 2.0
    quarter = max(1, math.ceil(n / 4))
    early = order[: max(1, math.ceil(n * 0.25))]
    late = order[-max(1, math.ceil(n * 0.25)) :]
    reversals = sum(1 for i in range(len(signed) - 1) if signed[i] and signed[i + 1] and (signed[i] > 0) != (signed[i + 1] > 0))
    monotonic = safe_divide(sum(1 for i in range(len(signed) - 1) if signed[i] and signed[i + 1] and (signed[i] > 0) == (signed[i + 1] > 0)), max(1, len(signed) - 1))
    parity_switch = safe_divide(sum(1 for i in range(len(order) - 1) if (order[i] % 2) != (order[i + 1] % 2)), max(1, len(order) - 1))
    counts = Counter(jumps)
    entropy = 0.0
    for count in counts.values():
        p = count / max(1, len(jumps))
        entropy -= p * math.log(p + 1e-12)
    visited: set[int] = set()
    gaps: list[int] = []
    for x in order:
        visited.add(x)
        unseen = [i for i in range(n) if i not in visited]
        if not unseen:
            gaps.append(0)
            continue
        runs: list[int] = []
        current = 0
        for i in range(n):
            if i in unseen:
                current += 1
            elif current:
                runs.append(current)
                current = 0
        if current:
            runs.append(current)
        gaps.append(max(runs) if runs else 0)
    return {
        "n": n,
        "first_track_norm": safe_divide(order[0], n - 1),
        "last_track_norm": safe_divide(order[-1], n - 1),
        "normalized_mean_jump": safe_divide(mean(jumps, 0.0), n - 1),
        "normalized_max_jump": safe_divide(max(jumps) if jumps else 0, n - 1),
        "adjacent_jump_count": sum(1 for j in jumps if j == 1),
        "long_jump_count": sum(1 for j in jumps if j >= n / 2),
        "parity_switch_rate": parity_switch,
        "monotonicity_fraction": monotonic,
        "direction_reversal_count": reversals,
        "jump_entropy": entropy,
        "early_center_bias": mean([abs(x - center) / max(1.0, center) for x in early], 0.0),
        "late_center_bias": mean([abs(x - center) / max(1.0, center) for x in late], 0.0),
        "edge_early_count": sum(1 for x in early if x < quarter or x >= n - quarter),
        "center_early_count": sum(1 for x in early if quarter <= x < n - quarter),
        "max_unvisited_gap_proxy_mean_norm": safe_divide(mean(gaps, 0.0), n),
        "max_unvisited_gap_proxy_max_norm": safe_divide(max(gaps) if gaps else 0, n),
    }


def load_combined108() -> list[dict[str, Any]]:
    rows = []
    for row in read_csv(COMBINED108_READY):
        n = parse_int(row["n"])
        order = parse_order(row.get("order_json"))
        item = dict(row)
        item["n"] = n
        item["order"] = order
        item["order_hash"] = order_hash(order) if order else ""
        item["reward"] = parse_float(row.get(PRIMARY_TARGET))
        rows.append(item)
    return rows


def prior_hashes() -> dict[int, set[str]]:
    hashes: dict[int, set[str]] = defaultdict(set)
    for path in [RUN23_SCORED, RUN24_SHORTLIST64]:
        if not path.exists():
            continue
        for row in read_csv(path):
            try:
                n = parse_int(row.get("n"))
            except (TypeError, ValueError):
                continue
            digest = str(row.get("order_hash", "")).strip()
            if not digest:
                order = parse_order(row.get("order_json"))
                if order:
                    digest = order_hash(order)
            if digest:
                hashes[n].add(digest)
    return hashes


def refs_by_n(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    refs: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        refs[row["n"]].append({"strategy_name": row["strategy_name"], "order": row["order"], "order_hash": row["order_hash"], "reward": row["reward"]})
    return refs


def validate_inputs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    counts = Counter(row["n"] for row in rows)
    if len(rows) != 108:
        errors.append(f"Expected 108 rows, found {len(rows)}")
    for n, count in EXPECTED_COUNTS.items():
        if counts[n] != count:
            errors.append(f"Expected N{n} count {count}, found {counts[n]}")
    for col in [PRIMARY_TARGET, *SCORE_TARGETS, "order_json"]:
        if any(str(row.get(col, "")).strip() == "" for row in rows):
            errors.append(f"Missing values in {col}")
    seen: dict[int, set[str]] = defaultdict(set)
    duplicates: list[str] = []
    for row in rows:
        if not validate_order(row.get("order"), row["n"]):
            errors.append(f"Invalid order for {row.get('strategy_name')}")
            continue
        if row["order_hash"] in seen[row["n"]]:
            duplicates.append(str(row.get("strategy_name")))
        seen[row["n"]].add(row["order_hash"])
    if duplicates:
        errors.append(f"Duplicate order within same N: {duplicates[:5]}")
    summary = {
        "verdict": "PASS_RUN26_GNN_INPUTS_READY_108_ROWS" if not errors else "FAIL_RUN26_GNN_INPUTS_INVALID",
        "row_count": len(rows),
        "per_n_counts": dict(sorted(counts.items())),
        "run23_candidate_pool_exists": RUN23_SCORED.exists(),
        "run24_shortlist64_exists": RUN24_SHORTLIST64.exists(),
        "errors": errors,
    }
    write_json(OUTPUT_DIR / "run26_gnn_input_validation_summary.json", summary)
    return summary


def write_run25_suspended_note() -> Path:
    path = OUTPUT_DIR / "run25_suspended_status_note.md"
    text = "\n".join(
        [
            "# Run25 Suspended Status",
            "",
            "- Run25 shortlist64 CAE/INP generation is suspended by user decision.",
            "- No Run25 Abaqus/CAE/INP/solver activity should be executed before Run26.",
            "- Run23/Run24 shortlist64 remains available as comparison/control.",
            "- Run26 creates GNN-policy candidates before choosing the next teacher-validation batch.",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")
    return path


def build_training_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table = []
    for row in rows:
        features = scan_order_features(row["order"], row["n"])
        table.append(
            {
                "n": row["n"],
                "strategy_name": row.get("strategy_name", ""),
                "dataset_source": row.get("dataset_source", ""),
                "order_json": canonical_order_json(row["order"]),
                "order_hash": row["order_hash"],
                "target_reward_combined108_u2_primary": row["reward"],
                "target_u2_score_combined108_rank": row.get("target_u2_score_combined108_rank", ""),
                "target_peeq_score_combined108_rank": row.get("target_peeq_score_combined108_rank", ""),
                "target_surfaceT_score_combined108_rank": row.get("target_surfaceT_score_combined108_rank", ""),
                "target_mises_score_combined108_rank": row.get("target_mises_score_combined108_rank", ""),
                "combined108_constrained_rank_within_n": row.get("target_combined108_constrained_rank_within_n", ""),
                "candidate_family": row.get("candidate_family", ""),
                "graph_node_count": row["n"],
                "graph_edge_count_line_adjacent_directed": 2 * (row["n"] - 1),
                "sequence_length": row["n"],
                "feature_summary_json": json.dumps(features, sort_keys=True),
                **features,
            }
        )
    return table


def split_definitions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rng = random.Random(GLOBAL_SEED)
    by_n: dict[int, list[str]] = defaultdict(list)
    for row in rows:
        by_n[row["n"]].append(row["order_hash"])
    folds = []
    for fold in range(5):
        test: list[str] = []
        train: list[str] = []
        for n, hashes in by_n.items():
            shuffled = list(hashes)
            rng.shuffle(shuffled)
            for idx, digest in enumerate(shuffled):
                (test if idx % 5 == fold else train).append(digest)
        folds.append({"fold": fold, "train_hashes": train, "test_hashes": test})
    return {
        "leave_N_out": [{"test_n": n, "train_n": [m for m in EXPECTED_N if m != n]} for n in EXPECTED_N],
        "stratified_5fold": folds,
        "train_combined80_test_batch28": {
            "train_sources": ["probe60_run08", "batch20_run14"],
            "test_sources": ["batch28_run20"],
        },
        "all_combined108_fit": {"train_rows": 108, "purpose": "candidate generation"},
    }


def try_import_torch() -> tuple[Any, bool, str]:
    try:
        import torch
        return torch, True, f"torch {torch.__version__}, cuda={torch.cuda.is_available()}"
    except Exception as exc:  # noqa: BLE001
        return None, False, repr(exc)


def node_sequence_tensor(order: list[int], n: int) -> np.ndarray:
    pos = {track: idx for idx, track in enumerate(order)}
    center = (n - 1) / 2.0
    arr = []
    for i in range(n):
        i_norm = safe_divide(i, n - 1)
        parity = float(i % 2)
        center_dist = safe_divide(abs(i - center), center if center else 1.0)
        edge_dist = min(i, n - 1 - i) / max(1, n - 1)
        visit_pos = safe_divide(pos[i], n - 1)
        first_flag = 1.0 if pos[i] == 0 else 0.0
        last_flag = 1.0 if pos[i] == n - 1 else 0.0
        prev_jump = 0.0 if pos[i] == 0 else abs(i - order[pos[i] - 1]) / max(1, n - 1)
        next_jump = 0.0 if pos[i] == n - 1 else abs(i - order[pos[i] + 1]) / max(1, n - 1)
        arr.append([i_norm, parity, center_dist, edge_dist, visit_pos, first_flag, last_flag, prev_jump, next_jump])
    return np.asarray(arr, dtype=np.float32)


def adjacency_matrix(n: int) -> np.ndarray:
    adj = np.eye(n, dtype=np.float32)
    for i in range(n):
        if i > 0:
            adj[i, i - 1] = 1.0
        if i + 1 < n:
            adj[i, i + 1] = 1.0
    deg = adj.sum(axis=1, keepdims=True)
    return adj / np.maximum(deg, 1.0)


def train_gnn_reward(rows: list[dict[str, Any]], torch: Any) -> tuple[dict[str, Any], Any]:
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)

    class GNNRewardModel(nn.Module):
        def __init__(self, hidden: int = 48):
            super().__init__()
            self.node = nn.Linear(9, hidden)
            self.msg = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(3)])
            self.out = nn.Sequential(nn.Linear(hidden + 10, 64), nn.ReLU(), nn.Linear(64, 1))

        def forward(self, x, adj, stats):
            h = F.relu(self.node(x))
            for layer in self.msg:
                h = F.relu(layer(torch.matmul(adj, h)))
            pooled = h.mean(dim=0)
            return self.out(torch.cat([pooled, stats], dim=0)).squeeze()

    def row_tensors(row: dict[str, Any]):
        n = row["n"]
        order = row["order"]
        x = torch.tensor(node_sequence_tensor(order, n), dtype=torch.float32)
        adj = torch.tensor(adjacency_matrix(n), dtype=torch.float32)
        f = scan_order_features(order, n)
        stats = torch.tensor(
            [
                safe_divide(n, 40),
                f["first_track_norm"],
                f["last_track_norm"],
                f["normalized_mean_jump"],
                f["normalized_max_jump"],
                safe_divide(f["adjacent_jump_count"], max(1, n - 1)),
                safe_divide(f["long_jump_count"], max(1, n - 1)),
                f["parity_switch_rate"],
                f["monotonicity_fraction"],
                safe_divide(f["direction_reversal_count"], max(1, n - 2)),
            ],
            dtype=torch.float32,
        )
        y = torch.tensor(row["reward"], dtype=torch.float32)
        return x, adj, stats, y

    def fit(train_rows: list[dict[str, Any]], epochs: int = 40) -> GNNRewardModel:
        model = GNNRewardModel()
        opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
        train_data = [row_tensors(row) for row in train_rows]
        for _ in range(epochs):
            random.shuffle(train_data)
            for x, adj, stats, y in train_data:
                opt.zero_grad()
                pred = model(x, adj, stats)
                loss = F.smooth_l1_loss(pred, y)
                loss.backward()
                opt.step()
        return model

    def predict(model: Any, eval_rows: list[dict[str, Any]]) -> list[float]:
        model.eval()
        preds = []
        with torch.no_grad():
            for row in eval_rows:
                x, adj, stats, _ = row_tensors(row)
                preds.append(float(model(x, adj, stats).clamp(0.0, 1.0)))
        return preds

    detailed = []
    for test_n in EXPECTED_N:
        train_rows = [r for r in rows if r["n"] != test_n]
        test_rows = [r for r in rows if r["n"] == test_n]
        model = fit(train_rows, epochs=40)
        preds = predict(model, test_rows)
        true = [r["reward"] for r in test_rows]
        ranked_pred = sorted(range(len(preds)), key=lambda i: preds[i], reverse=True)[:5]
        ranked_true = sorted(range(len(true)), key=lambda i: true[i], reverse=True)[:5]
        detailed.append(
            {
                "protocol": "leave_N_out",
                "test_n": test_n,
                "test_count": len(test_rows),
                "spearman": spearman(preds, true),
                "mae": mean([abs(a - b) for a, b in zip(preds, true)]),
                "rmse": math.sqrt(mean([(a - b) ** 2 for a, b in zip(preds, true)])),
                "top5_overlap": len(set(ranked_pred) & set(ranked_true)),
            }
        )
    all_model = fit(rows, epochs=60)
    macro_s = mean([parse_float(r["spearman"]) for r in detailed])
    macro_top5 = mean([parse_float(r["top5_overlap"]) for r in detailed])
    summary = {
        "status": "GNN_REWARD_MODEL_TRAINED",
        "model": "plain PyTorch line-graph message passing reward model",
        "leave_N_out_macro_spearman": macro_s,
        "leave_N_out_macro_top5_overlap": macro_top5,
        "n40_result": next((r for r in detailed if r["test_n"] == 40), {}),
        "underperforms_run22_extra_trees_macro_spearman": macro_s < 0.8651 if math.isfinite(macro_s) else True,
        "note": "Small offline prototype; not a final surrogate.",
    }
    write_csv(OUTPUT_DIR / "gnn_reward_model_validation_results.csv", detailed)
    write_json(OUTPUT_DIR / "gnn_reward_model_validation_summary.json", summary)
    return summary, all_model


def train_extra_trees_baseline(rows: list[dict[str, Any]]) -> tuple[Any, dict[str, Any]]:
    try:
        from sklearn.ensemble import ExtraTreesRegressor
    except Exception as exc:  # noqa: BLE001
        return None, {"status": "SKLEARN_UNAVAILABLE", "error": repr(exc)}
    features = ["n", "first_track_norm", "last_track_norm", "normalized_mean_jump", "normalized_max_jump", "adjacent_jump_count", "long_jump_count", "parity_switch_rate", "monotonicity_fraction", "direction_reversal_count"]
    x = []
    y = []
    for row in rows:
        f = scan_order_features(row["order"], row["n"])
        x.append([float(f.get(col, row["n"] if col == "n" else 0.0)) for col in features])
        y.append(row["reward"])
    model = ExtraTreesRegressor(n_estimators=260, max_depth=6, min_samples_leaf=2, random_state=42, n_jobs=-1)
    model.fit(np.asarray(x), np.asarray(y))
    return model, {"status": "EXTRA_TREES_F01_BASELINE_FIT", "features": features}


def et_features(order: list[int], n: int) -> list[float]:
    f = scan_order_features(order, n)
    return [float(f.get(col, n if col == "n" else 0.0)) for col in ["n", "first_track_norm", "last_track_norm", "normalized_mean_jump", "normalized_max_jump", "adjacent_jump_count", "long_jump_count", "parity_switch_rate", "monotonicity_fraction", "direction_reversal_count"]]


def train_pointer_policy(rows: list[dict[str, Any]], torch: Any) -> tuple[dict[str, Any], Any]:
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(GLOBAL_SEED)

    class PointerPolicy(nn.Module):
        def __init__(self, hidden: int = 64):
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
    opt = torch.optim.Adam(model.parameters(), lr=0.008, weight_decay=1e-4)
    log_rows = []
    weighted_rows = sorted(rows, key=lambda r: r["reward"], reverse=True)
    for epoch in range(30):
        total = 0.0
        random.shuffle(weighted_rows)
        for row in weighted_rows:
            n = row["n"]
            order = row["order"]
            weight = 0.25 + 1.75 * row["reward"]
            visited: set[int] = set()
            prev = None
            loss_terms = []
            for step, action in enumerate(order):
                logits = model.score(n, visited, prev, step)
                mask = torch.tensor([i not in visited for i in range(n)], dtype=torch.bool)
                masked = logits.masked_fill(~mask, -1e9)
                loss_terms.append(F.cross_entropy(masked.unsqueeze(0), torch.tensor([action])))
                visited.add(action)
                prev = action
            loss = torch.stack(loss_terms).mean() * weight
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss)
        if epoch % 10 == 0 or epoch == 79:
            log_rows.append({"epoch": epoch, "weighted_bc_loss": total / max(1, len(rows))})
    write_csv(OUTPUT_DIR / "graph_pointer_policy_training_log.csv", log_rows)

    validation = []
    for n in EXPECTED_N:
        top_rows = [r for r in rows if r["n"] == n]
        nll = []
        for row in top_rows:
            visited: set[int] = set()
            prev = None
            losses = []
            for step, action in enumerate(row["order"]):
                with torch.no_grad():
                    logits = model.score(n, visited, prev, step)
                    mask = torch.tensor([i not in visited for i in range(n)], dtype=torch.bool)
                    masked = logits.masked_fill(~mask, -1e9)
                    losses.append(float(F.cross_entropy(masked.unsqueeze(0), torch.tensor([action]))))
                visited.add(action)
                prev = action
            nll.append(mean(losses))
        validation.append({"n": n, "mean_teacher_action_nll": mean(nll), "teacher_rows": len(top_rows)})
    summary = {
        "status": "GRAPH_POINTER_POLICY_WEIGHTED_IMITATION_TRAINED",
        "training_method": "offline weighted behavior cloning on combined108 teacher sequences",
        "epochs": 30,
        "validation": validation,
        "note": "No online Abaqus RL or policy-gradient teacher interaction.",
    }
    write_json(OUTPUT_DIR / "graph_pointer_policy_validation_summary.json", summary)
    return summary, model


def greedy_decode_pointer(policy: Any, torch: Any, n: int) -> list[int]:
    visited: set[int] = set()
    order = []
    prev = None
    for step in range(n):
        with torch.no_grad():
            logits = policy.score(n, visited, prev, step)
            for i in visited:
                logits[i] = -1e9
            action = int(torch.argmax(logits).item())
        order.append(action)
        visited.add(action)
        prev = action
    return order


def sample_decode_pointer(policy: Any, torch: Any, n: int, rng: random.Random, temperature: float = 1.0, top_k: int | None = None) -> list[int]:
    visited: set[int] = set()
    order = []
    prev = None
    for step in range(n):
        with torch.no_grad():
            logits = policy.score(n, visited, prev, step).numpy() / max(temperature, 1e-6)
        for i in visited:
            logits[i] = -1e9
        if top_k:
            allowed = sorted([i for i in range(n) if i not in visited], key=lambda i: logits[i], reverse=True)[:top_k]
        else:
            allowed = [i for i in range(n) if i not in visited]
        probs = np.exp(np.array([logits[i] for i in allowed]) - max(logits[i] for i in allowed))
        probs = probs / probs.sum()
        action = rng.choices(allowed, weights=probs, k=1)[0]
        order.append(action)
        visited.add(action)
        prev = action
    return order


def beam_decode_pointer(policy: Any, torch: Any, n: int, beam_width: int = 6) -> list[list[int]]:
    beams = [([], set(), None, 0.0)]
    for step in range(n):
        new_beams = []
        for order, visited, prev, score in beams:
            with torch.no_grad():
                logits = policy.score(n, visited, prev, step).numpy()
            choices = sorted([i for i in range(n) if i not in visited], key=lambda i: logits[i], reverse=True)[:beam_width]
            for action in choices:
                new_order = order + [action]
                new_visited = set(visited)
                new_visited.add(action)
                new_beams.append((new_order, new_visited, action, score + float(logits[action])))
        beams = sorted(new_beams, key=lambda x: x[3], reverse=True)[:beam_width]
    return [b[0] for b in beams]


def swap_positions(order: list[int], rng: random.Random) -> list[int]:
    out = list(order)
    i, j = rng.sample(range(len(out)), 2)
    out[i], out[j] = out[j], out[i]
    return out


def reverse_segment(order: list[int], rng: random.Random) -> list[int]:
    out = list(order)
    n = len(out)
    length = rng.randint(2, max(3, n // 3))
    start = rng.randint(0, n - length)
    out[start : start + length] = reversed(out[start : start + length])
    return out


def random_order(n: int, rng: random.Random, mode: str) -> list[int]:
    tracks = list(range(n))
    center = (n - 1) / 2
    if mode == "edge_center":
        edges = [i for i in tracks if i < n / 4 or i >= 3 * n / 4]
        centers = [i for i in tracks if i not in edges]
        rng.shuffle(edges)
        rng.shuffle(centers)
        order = []
        while edges or centers:
            if edges:
                order.append(edges.pop())
            if centers:
                order.append(centers.pop())
        return order
    if mode == "high_jump":
        order = [rng.choice(tracks)]
        unused = set(tracks) - {order[0]}
        while unused:
            current = order[-1]
            nxt = max(unused, key=lambda x: (abs(x - current), rng.random()))
            order.append(nxt)
            unused.remove(nxt)
        return order
    if mode == "center_bias":
        tracks.sort(key=lambda x: (abs(x - center), rng.random()))
        return tracks
    rng.shuffle(tracks)
    return tracks


def add_candidate(store: dict[int, dict[str, dict[str, Any]]], n: int, order: list[int], source: str, bucket: str, method: str, seed: str = "") -> bool:
    if not validate_order(order, n):
        return False
    digest = order_hash(order)
    if digest in store[n]:
        return False
    store[n][digest] = {
        "n": n,
        "order": order,
        "order_hash": digest,
        "candidate_source": source,
        "selection_bucket": bucket,
        "generation_method": method,
        "seed_strategy": seed,
    }
    return True


def generate_candidates(rows: list[dict[str, Any]], torch: Any, policy: Any, gnn_model: Any, et_model: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    refs = refs_by_n(rows)
    existing = {n: {r["order_hash"] for r in refs[n]} for n in EXPECTED_N}
    prior = prior_hashes()
    store: dict[int, dict[str, dict[str, Any]]] = {n: {} for n in EXPECTED_N}
    raw_counts = Counter()
    duplicate_existing = Counter()
    duplicate_prior = Counter()
    best_rows = {n: sorted([r for r in rows if r["n"] == n], key=lambda r: r["reward"], reverse=True)[:8] for n in EXPECTED_N}
    for n in EXPECTED_N:
        rng = random.Random(GLOBAL_SEED + n)
        def try_add(order: list[int], source: str, bucket: str, method: str, seed: str = "") -> None:
            raw_counts[n] += 1
            digest = order_hash(order) if validate_order(order, n) else ""
            if digest in existing[n]:
                duplicate_existing[n] += 1
                return
            if digest in prior[n]:
                duplicate_prior[n] += 1
                return
            add_candidate(store, n, order, source, bucket, method, seed)

        if policy is not None and torch is not None:
            try_add(greedy_decode_pointer(policy, torch, n), "graph_pointer_greedy", "gnn_policy_top_predicted", "greedy_decode")
            for idx, order in enumerate(beam_decode_pointer(policy, torch, n, beam_width=8)):
                try_add(order, "graph_pointer_beam_search", "gnn_policy_beam_search", f"beam_{idx}")
            for idx in range(180 if n in (24, 40) else 80):
                temp = [0.65, 0.85, 1.0, 1.25, 1.6][idx % 5]
                top_k = None if idx % 3 else max(4, n // 4)
                try_add(sample_decode_pointer(policy, torch, n, rng, temp, top_k), "graph_pointer_temperature_sampled", "gnn_policy_temperature_diverse", f"temp_{temp}_topk_{top_k}")

        for seed in best_rows[n]:
            base = seed["order"]
            for idx in range(500 if n in (24, 40) else 200):
                if idx % 3 == 0:
                    order = swap_positions(base, rng)
                    method = "swap_mutation"
                elif idx % 3 == 1:
                    order = reverse_segment(base, rng)
                    method = "reverse_segment"
                else:
                    order = random_order(n, rng, "high_jump" if idx % 2 else "edge_center")
                    method = "policy_neighborhood_random"
                bucket = "known_best_neighborhood" if idx % 4 else "gnn_reward_local_search"
                try_add(order, "gnn_policy_known_best_mutation", bucket, method, seed["strategy_name"])

        modes = ["high_jump", "edge_center", "center_bias", "random"]
        guard = 0
        while len(store[n]) < POOL_TARGETS[n] and guard < POOL_TARGETS[n] * 20:
            guard += 1
            mode = modes[guard % len(modes)]
            bucket = ["gnn_vs_ET_disagreement", "gnn_uncertainty_probe", "sentinel_control", "gnn_reward_local_search"][guard % 4]
            try_add(random_order(n, rng, mode), f"gnn_policy_{mode}", bucket, mode)

    candidates = []
    for n in EXPECTED_N:
        for idx, item in enumerate(store[n].values(), start=1):
            order = item.pop("order")
            features = scan_order_features(order, n)
            nearest_name, novelty = nearest_order(order, refs[n])
            nearest_run24_name, novelty_run24 = nearest_order(order, load_run24_refs(n))
            candidate_id = f"R26_N{n}_C{idx:05d}"
            candidates.append(
                {
                    **item,
                    **features,
                    "candidate_id": candidate_id,
                    "strategy_name": f"N{n}_{candidate_id}_{item['selection_bucket']}",
                    "order_json": canonical_order_json(order),
                    "order_compact": "-".join(str(x) for x in order),
                    "novelty_distance_to_combined108": novelty,
                    "nearest_combined108_strategy": nearest_name,
                    "novelty_distance_to_run24_shortlist64": novelty_run24,
                    "nearest_run24_shortlist64_strategy": nearest_run24_name,
                    "duplicate_of_combined108_teacher": False,
                    "duplicate_of_run23_or_run24": False,
                }
            )
    counts = {
        "raw_generated_candidate_count_per_n": dict(raw_counts),
        "deduplicated_candidate_count_per_n": {n: sum(1 for c in candidates if c["n"] == n) for n in EXPECTED_N},
        "duplicate_existing_teacher_attempts_per_n": dict(duplicate_existing),
        "duplicate_run23_run24_attempts_per_n": dict(duplicate_prior),
    }
    return candidates, counts


def load_run24_refs(n: int) -> list[dict[str, Any]]:
    refs = []
    if not RUN24_SHORTLIST64.exists():
        return refs
    for row in read_csv(RUN24_SHORTLIST64):
        try:
            rn = parse_int(row.get("n"))
        except (TypeError, ValueError):
            continue
        if rn != n:
            continue
        order = parse_order(row.get("order_json"))
        if order:
            refs.append({"strategy_name": row.get("handoff_strategy_name", row.get("strategy_name", "")), "order": order})
    return refs


def predict_gnn_reward(torch: Any, model: Any, order: list[int], n: int) -> float:
    if torch is None or model is None:
        return math.nan
    x = torch.tensor(node_sequence_tensor(order, n), dtype=torch.float32)
    adj = torch.tensor(adjacency_matrix(n), dtype=torch.float32)
    f = scan_order_features(order, n)
    stats = torch.tensor(
        [
            safe_divide(n, 40),
            f["first_track_norm"],
            f["last_track_norm"],
            f["normalized_mean_jump"],
            f["normalized_max_jump"],
            safe_divide(f["adjacent_jump_count"], max(1, n - 1)),
            safe_divide(f["long_jump_count"], max(1, n - 1)),
            f["parity_switch_rate"],
            f["monotonicity_fraction"],
            safe_divide(f["direction_reversal_count"], max(1, n - 2)),
        ],
        dtype=torch.float32,
    )
    with torch.no_grad():
        return float(model(x, adj, stats).clamp(0.0, 1.0))


def score_candidates(candidates: list[dict[str, Any]], torch: Any, gnn_model: Any, et_model: Any) -> list[dict[str, Any]]:
    scored = []
    et_values = []
    for row in candidates:
        n = row["n"]
        order = parse_order(row["order_json"])
        assert order is not None
        gnn_pred = predict_gnn_reward(torch, gnn_model, order, n)
        et_pred = float(et_model.predict(np.asarray([et_features(order, n)]))[0]) if et_model is not None else math.nan
        item = dict(row)
        item["gnn_reward_pred"] = gnn_pred
        item["extra_trees_F01_pred"] = et_pred
        item["gnn_vs_ET_disagreement"] = abs(gnn_pred - et_pred) if math.isfinite(gnn_pred) and math.isfinite(et_pred) else math.nan
        item["selection_score"] = (0.45 * (gnn_pred if math.isfinite(gnn_pred) else 0.0)) + (0.25 * (et_pred if math.isfinite(et_pred) else 0.0)) + 0.20 * parse_float(item["novelty_distance_to_combined108"], 0.0) + 0.10 * parse_float(item["gnn_vs_ET_disagreement"], 0.0)
        scored.append(item)
        et_values.append(et_pred)
    for n in EXPECTED_N:
        group = [r for r in scored if r["n"] == n]
        for rank, row in enumerate(sorted(group, key=lambda r: parse_float(r["selection_score"]), reverse=True), start=1):
            row["selection_rank_within_n"] = rank
        for rank, row in enumerate(sorted(group, key=lambda r: parse_float(r["gnn_reward_pred"]), reverse=True), start=1):
            row["gnn_reward_rank_within_n"] = rank
    return scored


def select_batch(scored: list[dict[str, Any]], counts: dict[int, int], label: str) -> list[dict[str, Any]]:
    batch = []
    bucket_order = [
        "gnn_policy_top_predicted",
        "gnn_policy_beam_search",
        "gnn_policy_temperature_diverse",
        "gnn_reward_local_search",
        "gnn_vs_ET_disagreement",
        "gnn_uncertainty_probe",
        "known_best_neighborhood",
        "sentinel_control",
    ]
    for n in EXPECTED_N:
        group = [r for r in scored if r["n"] == n]
        selected = []
        used = set()
        for bucket in bucket_order:
            pool = [r for r in group if r["selection_bucket"] == bucket]
            key = "gnn_vs_ET_disagreement" if bucket == "gnn_vs_ET_disagreement" else "selection_score"
            pool = sorted(pool, key=lambda r: parse_float(r.get(key)), reverse=True)
            take = 1 if counts[n] <= 8 else (2 if bucket in {"gnn_policy_temperature_diverse", "gnn_vs_ET_disagreement", "known_best_neighborhood"} else 1)
            for row in pool:
                if len([x for x in selected if x["selection_bucket"] == bucket]) >= take:
                    break
                if row["order_hash"] in used:
                    continue
                selected.append(dict(row))
                used.add(row["order_hash"])
                if len(selected) >= counts[n]:
                    break
            if len(selected) >= counts[n]:
                break
        for row in sorted(group, key=lambda r: parse_float(r["selection_score"]), reverse=True):
            if len(selected) >= counts[n]:
                break
            if row["order_hash"] not in used:
                selected.append(dict(row))
                used.add(row["order_hash"])
        for idx, row in enumerate(selected[: counts[n]], start=1):
            row[f"{label}_rank_within_n"] = idx
            batch.append(row)
    return batch


def compare_to_run24(batch64: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run24_hashes: dict[int, set[str]] = defaultdict(set)
    if RUN24_SHORTLIST64.exists():
        for row in read_csv(RUN24_SHORTLIST64):
            try:
                n = parse_int(row.get("n"))
            except (TypeError, ValueError):
                continue
            digest = row.get("order_hash", "")
            if digest:
                run24_hashes[n].add(digest)
    rows = []
    for n in EXPECTED_N:
        group = [r for r in batch64 if r["n"] == n]
        overlap = sum(1 for r in group if r["order_hash"] in run24_hashes[n])
        rows.append(
            {
                "n": n,
                "gnn_batch64_count": len(group),
                "run24_shortlist64_overlap_count": overlap,
                "distinct_from_run24_count": len(group) - overlap,
                "bucket_composition_json": json.dumps(dict(Counter(r["selection_bucket"] for r in group)), sort_keys=True),
                "mean_gnn_reward_pred": mean([parse_float(r["gnn_reward_pred"]) for r in group]),
                "mean_extra_trees_F01_pred": mean([parse_float(r["extra_trees_F01_pred"]) for r in group]),
                "mean_novelty_to_combined108": mean([parse_float(r["novelty_distance_to_combined108"]) for r in group]),
                "mean_novelty_to_run24": mean([parse_float(r["novelty_distance_to_run24_shortlist64"]) for r in group]),
                "mean_gnn_vs_ET_disagreement": mean([parse_float(r["gnn_vs_ET_disagreement"]) for r in group]),
            }
        )
    summary = {
        "total_overlap_with_run24_shortlist64": sum(row["run24_shortlist64_overlap_count"] for row in rows),
        "total_distinct_from_run24_shortlist64": sum(row["distinct_from_run24_count"] for row in rows),
        "mostly_distinct_from_run23_run24_shortlist64": sum(row["run24_shortlist64_overlap_count"] for row in rows) <= 3,
        "n_distribution": dict(Counter(r["n"] for r in batch64)),
        "bucket_composition": dict(Counter(r["selection_bucket"] for r in batch64)),
        "note": "Exact duplicate orders against run23/run24 were avoided during generation; comparison is retained for audit.",
    }
    return rows, summary


def write_claim_boundary() -> tuple[Path, Path]:
    md_path = OUTPUT_DIR / "run26_gnn_claim_boundary.md"
    json_path = OUTPUT_DIR / "run26_gnn_claim_boundary.json"
    md = "\n".join(
        [
            "# Run26 GNN Claim Boundary",
            "",
            "## Safe claims",
            "- Run26 implements an offline GNN / graph-pointer policy prototype.",
            "- Run26 uses combined108 teacher-labelled data for policy/reward modelling.",
            "- Run26 generates a future GNN-policy candidate batch for teacher validation.",
            "- Run26 compares GNN-policy candidates with the previous surrogate active-learning shortlist64.",
            "- Run26 does not perform teacher validation.",
            "",
            "## Unsafe claims",
            "- Do not claim the GNN-policy candidates are physically better.",
            "- Do not claim GNN-RL has beaten baselines.",
            "- Do not claim online RL with Abaqus.",
            "- Do not claim arbitrary-N generalization.",
            "- Do not claim deployment-ready policy.",
            "- Do not claim ODB results exist.",
            "",
            "Verdict: RUN26_OFFLINE_GNN_POLICY_PROTOTYPE_NO_TEACHER_VALIDATION",
            "",
        ]
    )
    md_path.write_text(md, encoding="utf-8")
    write_json(
        json_path,
        {
            "verdict": "RUN26_OFFLINE_GNN_POLICY_PROTOTYPE_NO_TEACHER_VALIDATION",
            "safe_claims": [
                "offline GNN / graph-pointer policy prototype implemented",
                "combined108 teacher-labelled data used for policy/reward modelling",
                "future GNN-policy candidate batch generated for possible teacher validation",
                "comparison with run23/run24 shortlist64 created",
            ],
            "unsafe_claims": [
                "physical superiority",
                "GNN-RL beat baselines",
                "online RL with Abaqus",
                "arbitrary-N generalization",
                "deployment-ready policy",
                "ODB results exist",
            ],
        },
    )
    return md_path, json_path


def write_report(
    validation: dict[str, Any],
    torch_available: bool,
    torch_status: str,
    gnn_summary: dict[str, Any],
    pointer_summary: dict[str, Any],
    counts: dict[str, Any],
    batch64: list[dict[str, Any]],
    batch32: list[dict[str, Any]],
    comparison_summary: dict[str, Any],
    outputs: list[str],
) -> None:
    lines = [
        "# Stage 3 Run 26 - Combined108 GNN / Graph-Pointer Policy Candidate Generation",
        "",
        "## Purpose",
        "Insert an offline GNN / graph-pointer policy step before spending new solver time, so the next candidate batch can be positioned as GNN-policy-generated rather than purely surrogate active-learning generated.",
        "",
        "## Run25 Suspended Status",
        "Run25 shortlist64 CAE/INP generation is suspended by user decision. No Run25 Abaqus/CAE/INP/solver activity should be executed before Run26.",
        "",
        "## Why Run26 Is Inserted Before New CAE/Solver Validation",
        "Run23/Run24 shortlist64 remains a valid surrogate active-learning control batch, but Run26 creates a graph-policy batch for paper-mainline consideration before committing the next 60+ teacher-validation jobs.",
        "",
        "## Inputs",
        f"- Combined108 RL-ready dataset: `{COMBINED108_READY}`",
        f"- Run22 surrogate diagnostics: `{RUN22_BEST}`",
        f"- Run23 candidate pool: `{RUN23_SCORED}`",
        f"- Run24 shortlist64: `{RUN24_SHORTLIST64}`",
        "",
        "## Graph Formulation",
        "Each track is a graph node. The prototype uses normalized track position, parity, center/edge distance, visit position, first/last flags, and adjacent line-graph message passing.",
        "",
        "## Reward Definition",
        "`R = 0.65*S_U2 + 0.20*S_PEEQ + 0.10*S_SurfaceT + 0.05*S_Mises`, using combined108 within-N normalized/ranked scores.",
        "",
        "## GNN Reward Model",
        f"- PyTorch available: `{torch_available}` ({torch_status})",
        f"- Status: `{gnn_summary.get('status')}`",
        f"- Leave-N-out macro Spearman: `{gnn_summary.get('leave_N_out_macro_spearman')}`",
        f"- Leave-N-out macro top5 overlap: `{gnn_summary.get('leave_N_out_macro_top5_overlap')}`",
        f"- N40 result: `{gnn_summary.get('n40_result')}`",
        "",
        "## Graph-Pointer Policy",
        f"- Status: `{pointer_summary.get('status')}`",
        f"- Training method: `{pointer_summary.get('training_method')}`",
        "- Visited nodes are masked during decoding.",
        "",
        "## Candidate Generation",
        f"- Deduplicated candidate counts per N: `{counts.get('deduplicated_candidate_count_per_n')}`",
        "- Candidate sources include graph-pointer greedy, beam search, temperature sampling, GNN reward local search, known-best mutation, disagreement probes, uncertainty probes, and sentinels.",
        "",
        "## GNN-Policy Batch64",
        f"- Counts: `{dict(Counter(row['n'] for row in batch64))}`",
        f"- Bucket composition: `{dict(Counter(row['selection_bucket'] for row in batch64))}`",
        "",
        "## GNN-Policy Batch32",
        f"- Counts: `{dict(Counter(row['n'] for row in batch32))}`",
        "",
        "## Comparison With Run23/Run24 Shortlist64",
        f"- Overlap with run24 shortlist64: `{comparison_summary.get('total_overlap_with_run24_shortlist64')}`",
        f"- Distinct from run24 shortlist64: `{comparison_summary.get('total_distinct_from_run24_shortlist64')}`",
        f"- Mostly distinct: `{comparison_summary.get('mostly_distinct_from_run23_run24_shortlist64')}`",
        "",
        "## Limitations and Claim Boundary",
        "Run26 is offline policy/reward modelling only. It does not prove physical superiority, GNN-RL baseline superiority, arbitrary-N generalization, or deployment readiness.",
        "",
        "## Output Files",
    ]
    lines.extend(f"- `{path}`" for path in outputs)
    lines.extend(
        [
            "",
            "## Recommended Run27",
            "Create a handoff package for the selected GNN-policy batch64. Do not generate CAE until user approval. If the user approves GNN as mainline, use GNN-policy batch64 rather than run23 shortlist64.",
            "",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def update_run_index() -> None:
    row = "| run_26 | Combined108 GNN / graph-pointer policy candidate generation | Build an offline GNN reward model and graph-pointer policy prototype from combined108, generate GNN-policy candidates, and compare against run23/run24 shortlist64 without CAE/solver execution. | `scripts/stage3/run_26_gnn_graph_pointer_policy_candidate_generation.py` | `docs/stage3/runs/run_26_combined108_gnn_graph_pointer_policy_candidate_generation/RUN_26_COMBINED108_GNN_GRAPH_POINTER_POLICY_CANDIDATE_GENERATION_REPORT.md` | `outputs/stage3_run_26_combined108_gnn_graph_pointer_policy_candidate_generation/` | `PASS_RUN26_GNN_INPUTS_READY_108_ROWS` | Run25 suspended; no Abaqus, no ODB, no abqjobpilot, no CAE/INP/JNL generation, no teacher validation, no online RL, no commit/push. Next: run27 GNN-policy batch64 handoff after user approval. |"
    if RUN_INDEX_PATH.exists():
        text = RUN_INDEX_PATH.read_text(encoding="utf-8")
        if "| run_26 |" not in text:
            RUN_INDEX_PATH.write_text(text.rstrip() + "\n" + row + "\n", encoding="utf-8")


def git_branch() -> str:
    try:
        result = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception:
        return ""


def main() -> int:
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    outputs.append(str(write_run25_suspended_note()))
    rows = load_combined108()
    validation = validate_inputs(rows)
    outputs.append(str(OUTPUT_DIR / "run26_gnn_input_validation_summary.json"))
    if validation["verdict"].startswith("FAIL"):
        print(validation["verdict"])
        return 2

    training_table = build_training_table(rows)
    write_csv(OUTPUT_DIR / "combined108_graph_policy_training_table.csv", training_table)
    outputs.append(str(OUTPUT_DIR / "combined108_graph_policy_training_table.csv"))
    write_json(OUTPUT_DIR / "combined108_graph_policy_split_definitions.json", split_definitions(rows))
    outputs.append(str(OUTPUT_DIR / "combined108_graph_policy_split_definitions.json"))

    torch, torch_available, torch_status = try_import_torch()
    et_model, et_meta = train_extra_trees_baseline(rows)
    write_json(OUTPUT_DIR / "run26_extra_trees_baseline_metadata.json", et_meta)
    outputs.append(str(OUTPUT_DIR / "run26_extra_trees_baseline_metadata.json"))

    if torch_available:
        gnn_summary, gnn_model = train_gnn_reward(rows, torch)
        pointer_summary, pointer_model = train_pointer_policy(rows, torch)
    else:
        gnn_summary = {"status": "GNN_REWARD_MODEL_SKIPPED_TORCH_UNAVAILABLE", "torch_status": torch_status}
        pointer_summary = {"status": "GRAPH_POINTER_POLICY_SKIPPED_TORCH_UNAVAILABLE", "torch_status": torch_status}
        write_csv(OUTPUT_DIR / "gnn_reward_model_validation_results.csv", [])
        write_json(OUTPUT_DIR / "gnn_reward_model_validation_summary.json", gnn_summary)
        write_csv(OUTPUT_DIR / "graph_pointer_policy_training_log.csv", [])
        write_json(OUTPUT_DIR / "graph_pointer_policy_validation_summary.json", pointer_summary)
        gnn_model = None
        pointer_model = None
    outputs.extend(
        [
            str(OUTPUT_DIR / "gnn_reward_model_validation_results.csv"),
            str(OUTPUT_DIR / "gnn_reward_model_validation_summary.json"),
            str(OUTPUT_DIR / "graph_pointer_policy_training_log.csv"),
            str(OUTPUT_DIR / "graph_pointer_policy_validation_summary.json"),
        ]
    )

    candidates, counts = generate_candidates(rows, torch if torch_available else None, pointer_model, gnn_model, et_model)
    scored = score_candidates(candidates, torch if torch_available else None, gnn_model, et_model)
    batch64 = select_batch(scored, BATCH64_COUNTS, "batch64")
    batch32 = select_batch(scored, BATCH32_COUNTS, "batch32")
    comparison_rows, comparison_summary = compare_to_run24(batch64)

    write_csv(OUTPUT_DIR / "run26_gnn_candidate_pool_unscored.csv", candidates)
    write_csv(OUTPUT_DIR / "run26_gnn_candidate_pool_scored.csv", scored)
    write_csv(OUTPUT_DIR / "run26_gnn_policy_batch64_candidate_orders.csv", batch64)
    write_csv(OUTPUT_DIR / "run26_gnn_policy_batch32_candidate_orders.csv", batch32)
    write_csv(OUTPUT_DIR / "run26_gnn_vs_run23_shortlist64_comparison.csv", comparison_rows)
    write_json(OUTPUT_DIR / "run26_gnn_vs_run23_shortlist64_comparison_summary.json", comparison_summary)
    outputs.extend(
        [
            str(OUTPUT_DIR / "run26_gnn_candidate_pool_unscored.csv"),
            str(OUTPUT_DIR / "run26_gnn_candidate_pool_scored.csv"),
            str(OUTPUT_DIR / "run26_gnn_policy_batch64_candidate_orders.csv"),
            str(OUTPUT_DIR / "run26_gnn_policy_batch32_candidate_orders.csv"),
            str(OUTPUT_DIR / "run26_gnn_vs_run23_shortlist64_comparison.csv"),
            str(OUTPUT_DIR / "run26_gnn_vs_run23_shortlist64_comparison_summary.json"),
        ]
    )
    claim_md, claim_json = write_claim_boundary()
    outputs.extend([str(claim_md), str(claim_json)])
    write_report(validation, torch_available, torch_status, gnn_summary, pointer_summary, counts, batch64, batch32, comparison_summary, outputs)
    outputs.append(str(REPORT_PATH))
    update_run_index()

    manifest = {
        "run_id": RUN_ID,
        "run_name": RUN_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "branch": git_branch(),
        "script_path": str(ROOT / "scripts" / "stage3" / "run_26_gnn_graph_pointer_policy_candidate_generation.py"),
        "input_files": [str(p) for p in [COMBINED108_READY, COMBINED108_TEACHER, COMBINED108_LEADERBOARD, RUN22_FEATURES, RUN22_BEST, RUN22_DETAILED, RUN23_SCORED, RUN24_SHORTLIST64, RUN24_REPORT] if p.exists()],
        "output_files": outputs,
        "model_type": "offline GNN / graph-pointer policy prototype",
        "teacher_data_rows": len(rows),
        "validation_verdict": validation["verdict"],
        "pytorch_available": torch_available,
        "pytorch_status": torch_status,
        "generated_candidate_counts_per_n": counts["deduplicated_candidate_count_per_n"],
        "selected_gnn_policy_batch64_counts": dict(Counter(row["n"] for row in batch64)),
        "selected_gnn_policy_batch32_counts": dict(Counter(row["n"] for row in batch32)),
        "report_path": str(REPORT_PATH),
        "claim_boundary_path": str(claim_md),
        "run25_suspended": True,
        "no_solver_run": True,
        "no_odb_opened": True,
        "no_abqjobpilot_run": True,
        "no_cae_inp_generated": True,
        "no_teacher_validation": True,
        "no_online_rl": True,
        "no_commit_or_push": True,
    }
    write_json(MANIFEST_PATH, manifest)

    print(validation["verdict"])
    print(f"torch_available={torch_available} {torch_status}")
    print(f"gnn_reward_status={gnn_summary.get('status')} macro_spearman={gnn_summary.get('leave_N_out_macro_spearman')}")
    print(f"pointer_status={pointer_summary.get('status')}")
    print(f"candidate_counts={counts['deduplicated_candidate_count_per_n']}")
    print(f"batch64={len(batch64)} per_n={dict(Counter(row['n'] for row in batch64))}")
    print(f"batch32={len(batch32)} per_n={dict(Counter(row['n'] for row in batch32))}")
    print(f"run24_overlap={comparison_summary['total_overlap_with_run24_shortlist64']}")
    print(f"report={REPORT_PATH}")
    print(f"manifest={MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
