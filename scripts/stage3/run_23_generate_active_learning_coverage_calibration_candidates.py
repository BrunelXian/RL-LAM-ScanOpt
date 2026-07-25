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
RUN_ID = "run_23_combined108_active_learning_coverage_calibration_design"
RUN_NAME = "combined108 active-learning coverage and calibration design"

COMBINED108_READY = ROOT / "outputs" / "stage3_run_21_batch28_teacher_metrics_ingestion_and_combined108_ranking" / "combined108_RL_ready_dataset.csv"
COMBINED108_TEACHER = ROOT / "outputs" / "stage3_run_21_batch28_teacher_metrics_ingestion_and_combined108_ranking" / "combined108_teacher_dataset.csv"
COMBINED108_LEADERBOARD = ROOT / "outputs" / "stage3_run_21_batch28_teacher_metrics_ingestion_and_combined108_ranking" / "combined108_per_N_leaderboard.csv"
RUN22_FEATURES = ROOT / "outputs" / "stage3_run_22_combined108_surrogate_reward_model_validation_update" / "combined108_scan_order_features.csv"
RUN22_DETAILED = ROOT / "outputs" / "stage3_run_22_combined108_surrogate_reward_model_validation_update" / "combined108_surrogate_validation_results_detailed.csv"
RUN22_BEST = ROOT / "outputs" / "stage3_run_22_combined108_surrogate_reward_model_validation_update" / "combined108_best_surrogate_configurations.csv"
RUN22_PREDICTIONS = ROOT / "outputs" / "stage3_run_22_combined108_surrogate_reward_model_validation_update" / "combined108_predictions_target_reward_u2_primary.csv"
RUN22_FEATURE_DEFS = ROOT / "outputs" / "stage3_run_22_combined108_surrogate_reward_model_validation_update" / "run22_feature_set_definitions.json"
RUN22_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_22_combined108_surrogate_reward_model_validation_update" / "RUN_22_COMBINED108_SURROGATE_REWARD_MODEL_VALIDATION_UPDATE_REPORT.md"
RUN18_SCORED = ROOT / "outputs" / "stage3_run_18_combined80_surrogate_screened_candidate_generation" / "run18_candidate_pool_scored.csv"
RUN19_BATCH28 = ROOT / "outputs" / "stage3_run_19_run18_candidate_handoff_review_package" / "batch28" / "stage3_run19_batch28_candidate_orders.csv"

OUTPUT_DIR = ROOT / "outputs" / "stage3_run_23_combined108_active_learning_coverage_calibration_design"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = ROOT / "docs" / "stage3" / "runs" / RUN_ID
REPORT_PATH = REPORT_DIR / "RUN_23_COMBINED108_ACTIVE_LEARNING_COVERAGE_CALIBRATION_DESIGN_REPORT.md"
MANIFEST_PATH = ROOT / "artifacts" / "manifests" / "stage3_run_23_manifest.json"
RUN_INDEX_PATH = ROOT / "docs" / "stage3" / "STAGE3_RUN_INDEX.md"

EXPECTED_N = [12, 16, 24, 40]
EXPECTED_COUNTS = {12: 24, 16: 24, 24: 30, 40: 30}
TARGET_DEDUP_PER_N = {12: 800, 16: 800, 24: 2500, 40: 3000}
MIN_DEDUP_PER_N = {12: 800, 16: 800, 24: 2000, 40: 2500}
SHORTLIST_COUNTS = {12: 8, 16: 8, 24: 24, 40: 24}
BATCH32_COUNTS = {12: 4, 16: 4, 24: 12, 40: 12}
BATCH24_COUNTS = {12: 2, 16: 2, 24: 10, 40: 10}
GLOBAL_SEED = 42
PRIMARY_TARGET = "target_reward_combined108_u2_primary"
SECONDARY_TARGETS = [
    "target_u2_score_combined108_rank",
    "target_peeq_score_combined108_rank",
    "target_surfaceT_score_combined108_rank",
    "target_mises_score_combined108_rank",
]
KNOWN_BEST_SEEDS = {
    12: "S3B20_N12_B02_diversity_top",
    16: "S3R19B28_N16_B04_method_c_inspired",
    24: "S3R19B28_N24_B10_known_best_mutation",
    40: "S3R19B28_N40_B01_surrogate_top",
}
F01_FEATURES = [
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
    return int(float(text))


def parse_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def mean(values: list[float], default: float = 0.0) -> float:
    clean = [v for v in values if math.isfinite(v)]
    return statistics.fmean(clean) if clean else default


def std(values: list[float]) -> float:
    clean = [v for v in values if math.isfinite(v)]
    return statistics.pstdev(clean) if len(clean) > 1 else 0.0


def median(values: list[float], default: float = 0.0) -> float:
    clean = [v for v in values if math.isfinite(v)]
    return statistics.median(clean) if clean else default


def safe_divide(num: float, den: float, default: float = 0.0) -> float:
    return num / den if den else default


def order_hash(order: list[int]) -> str:
    return hashlib.sha1(",".join(str(x) for x in order).encode("ascii")).hexdigest()[:16]


def parse_order(text: Any) -> list[int] | None:
    if isinstance(text, list):
        try:
            return [int(x) for x in text]
        except (TypeError, ValueError):
            return None
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


def validate_order(order: list[int], n: int) -> bool:
    return len(order) == n and len(set(order)) == n and set(order) == set(range(n))


def canonical_order_json(order: list[int]) -> str:
    return json.dumps(order, separators=(",", ":"))


def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)


def kendall_distance(a: list[int], b: list[int]) -> float:
    pos = {track: idx for idx, track in enumerate(b)}
    mapped = [pos[x] for x in a]
    inv = 0
    total = len(mapped) * (len(mapped) - 1) // 2
    for i in range(len(mapped)):
        ai = mapped[i]
        inv += sum(1 for j in range(i + 1, len(mapped)) if ai > mapped[j])
    return safe_divide(inv, total)


def mutation_distance(a: list[int], b: list[int]) -> float:
    if len(a) != len(b):
        return 1.0
    return sum(1 for x, y in zip(a, b) if x != y) / len(a)


def nearest_order(order: list[int], refs: list[dict[str, Any]]) -> tuple[str, float]:
    best_name = ""
    best_distance = 1.0
    for ref in refs:
        dist = kendall_distance(order, ref["order"])
        if dist < best_distance:
            best_name = ref["strategy_name"]
            best_distance = dist
    return best_name, best_distance


def entropy(values: list[int]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = len(values)
    return -sum((count / total) * math.log(count / total, 2) for count in counts.values())


def scan_order_features(order: list[int], n: int) -> dict[str, Any]:
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
    return {
        "n": n,
        "first_track": order[0],
        "last_track": order[-1],
        "center_track_index": center,
        "first_track_norm": safe_divide(order[0], n - 1),
        "last_track_norm": safe_divide(order[-1], n - 1),
        "mean_jump": mean(jumps),
        "median_jump": median(jumps),
        "max_jump": max(jumps) if jumps else 0,
        "min_jump": min(jumps) if jumps else 0,
        "std_jump": std([float(j) for j in jumps]),
        "total_jump": sum(jumps),
        "normalized_mean_jump": safe_divide(mean(jumps), n - 1),
        "normalized_max_jump": safe_divide(max(jumps) if jumps else 0, n - 1),
        "adjacent_jump_count": sum(1 for j in jumps if j == 1),
        "long_jump_count": sum(1 for j in jumps if j >= n / 2),
        "jump_entropy": entropy(jumps),
        "running_center_distance_mean": mean([abs(x - center) / max(1.0, center) for x in order]),
        "early_center_bias": mean([abs(x - center) / max(1.0, center) for x in early]),
        "late_center_bias": mean([abs(x - center) / max(1.0, center) for x in late]),
        "edge_early_count": sum(1 for x in early if x in outer),
        "center_early_count": sum(1 for x in early if x in center_tracks),
        "odd_even_transition_count": parity_switches,
        "parity_switch_rate": safe_divide(parity_switches, max(1, n - 1)),
        "monotonicity_fraction": safe_divide(same_sign, max(1, len(signed) - 1)),
        "direction_reversal_count": reversals,
    }


def raster(n: int, reverse: bool = False) -> list[int]:
    order = list(range(n))
    return list(reversed(order)) if reverse else order


def odd_even(n: int) -> list[int]:
    return list(range(0, n, 2)) + list(range(1, n, 2))


def edge_in(n: int) -> list[int]:
    order: list[int] = []
    left, right = 0, n - 1
    while left <= right:
        order.append(left)
        if left != right:
            order.append(right)
        left += 1
        right -= 1
    return order


def center_out(n: int) -> list[int]:
    order: list[int] = []
    left = (n - 1) // 2
    right = left + 1
    while left >= 0 or right < n:
        if left >= 0:
            order.append(left)
            left -= 1
        if right < n:
            order.append(right)
            right += 1
    return order


def greedy_maximin(n: int) -> list[int]:
    order = [0, n - 1]
    remaining = set(range(1, n - 1))
    while remaining:
        best = max(remaining, key=lambda x: (min(abs(x - y) for y in order), -x))
        order.append(best)
        remaining.remove(best)
    return order


def block_interleaved(n: int) -> list[int]:
    blocks = [list(range(round(i * n / 4), round((i + 1) * n / 4))) for i in range(4)]
    max_len = max(len(b) for b in blocks)
    order: list[int] = []
    for row in zip(*[block + [None] * (max_len - len(block)) for block in blocks]):
        for item in row:
            if item is not None:
                order.append(item)
    return order


def center_edge_alternating(n: int) -> list[int]:
    c = center_out(n)
    e = edge_in(n)
    order: list[int] = []
    seen: set[int] = set()
    for a, b in zip(c, e):
        for item in (a, b):
            if item not in seen:
                order.append(item)
                seen.add(item)
    order.extend(x for x in range(n) if x not in seen)
    return order


def regular_jump(n: int, start: int, jump: int, direction: int = 1) -> list[int]:
    order: list[int] = []
    seen: set[int] = set()
    current = start
    for _ in range(n):
        if current in seen:
            break
        order.append(current)
        seen.add(current)
        current = (current + direction * jump) % n
    order.extend(x for x in range(n) if x not in seen)
    return order


def method_c_inspired(n: int, flavor: str = "u2") -> list[int]:
    if flavor == "peeq":
        base = greedy_maximin(n)
        return interleave_orders(base, center_out(n), take=2)
    if flavor == "surface":
        return interleave_orders(center_out(n), block_interleaved(n), take=2)
    if flavor == "diversity":
        return interleave_orders(edge_in(n), greedy_maximin(n), take=1)
    if flavor == "regular":
        jump = best_coprime_jump(n)
        return interleave_orders(regular_jump(n, 0, jump), greedy_maximin(n), take=3)
    return interleave_orders(greedy_maximin(n), edge_in(n), take=2)


def best_coprime_jump(n: int) -> int:
    candidates = [j for j in range(2, n) if gcd(j, n) == 1]
    return max(candidates, key=lambda j: min(j, n - j)) if candidates else 1


def interleave_orders(a: list[int], b: list[int], take: int = 1) -> list[int]:
    order: list[int] = []
    seen: set[int] = set()
    ia = ib = 0
    use_a = True
    while len(order) < len(a):
        source = a if use_a else b
        idx = ia if use_a else ib
        added = 0
        while idx < len(source) and added < take:
            item = source[idx]
            idx += 1
            if item not in seen:
                order.append(item)
                seen.add(item)
                added += 1
        if use_a:
            ia = idx
        else:
            ib = idx
        use_a = not use_a
        if ia >= len(a) and ib >= len(b):
            break
    order.extend(x for x in a if x not in seen)
    return order


def swap_positions(order: list[int], i: int, j: int) -> list[int]:
    new = list(order)
    new[i], new[j] = new[j], new[i]
    return new


def reverse_segment(order: list[int], start: int, length: int) -> list[int]:
    new = list(order)
    end = min(len(order), start + length)
    new[start:end] = reversed(new[start:end])
    return new


def rotate_order(order: list[int], k: int) -> list[int]:
    k %= len(order)
    return order[k:] + order[:k]


def swap_blocks(order: list[int], start_a: int, start_b: int, length: int) -> list[int]:
    new = list(order)
    end_a = min(len(order), start_a + length)
    end_b = min(len(order), start_b + length)
    if end_a - start_a != end_b - start_b or end_a > start_b:
        return new
    block_a = new[start_a:end_a]
    block_b = new[start_b:end_b]
    new[start_a:end_a] = block_b
    new[start_b:end_b] = block_a
    return new


def parity_preserving_swap(order: list[int], rng: random.Random) -> list[int]:
    even_pos = [i for i, x in enumerate(order) if x % 2 == 0]
    odd_pos = [i for i, x in enumerate(order) if x % 2 == 1]
    choices = [p for p in (even_pos, odd_pos) if len(p) >= 2]
    if not choices:
        return list(order)
    positions = rng.choice(choices)
    i, j = rng.sample(positions, 2)
    return swap_positions(order, i, j)


def random_biased_order(n: int, rng: random.Random, mode: str) -> list[int]:
    tracks = list(range(n))
    center = (n - 1) / 2.0
    if mode == "edge_bias":
        tracks.sort(key=lambda x: (-abs(x - center), rng.random()))
    elif mode == "center_bias":
        tracks.sort(key=lambda x: (abs(x - center), rng.random()))
    elif mode == "high_jump":
        order = [rng.randrange(n)]
        remaining = set(range(n)) - {order[0]}
        while remaining:
            current = order[-1]
            high = sorted(remaining, key=lambda x: abs(x - current), reverse=True)[: max(2, min(8, len(remaining)))]
            nxt = rng.choice(high)
            order.append(nxt)
            remaining.remove(nxt)
        return order
    elif mode == "low_adjacent":
        order = [rng.randrange(n)]
        remaining = set(range(n)) - {order[0]}
        while remaining:
            current = order[-1]
            non_adj = [x for x in remaining if abs(x - current) > 1]
            pool = non_adj if non_adj else list(remaining)
            nxt = rng.choice(pool)
            order.append(nxt)
            remaining.remove(nxt)
        return order
    elif mode == "parity_balanced":
        evens = list(range(0, n, 2))
        odds = list(range(1, n, 2))
        rng.shuffle(evens)
        rng.shuffle(odds)
        order = []
        while evens or odds:
            if evens:
                order.append(evens.pop())
            if odds:
                order.append(odds.pop())
        if rng.random() < 0.5:
            order = order[1:] + order[:1]
        return order
    elif mode == "edge_center_alternating":
        edge = edge_in(n)
        center = center_out(n)
        return interleave_orders(edge, center, take=1)
    else:
        rng.shuffle(tracks)
    return tracks


def crossover(a: list[int], b: list[int], cut: int) -> list[int]:
    order = list(a[:cut])
    seen = set(order)
    order.extend(x for x in b if x not in seen)
    return order


def alternating_block_crossover(a: list[int], b: list[int], block: int) -> list[int]:
    order: list[int] = []
    seen: set[int] = set()
    use_a = True
    idx = 0
    while len(order) < len(a):
        source = a if use_a else b
        for item in source[idx: idx + block]:
            if item not in seen:
                order.append(item)
                seen.add(item)
        idx += block
        use_a = not use_a
        if idx >= len(a):
            idx = 0
            if len(order) == len(seen):
                for item in a + b:
                    if item not in seen:
                        order.append(item)
                        seen.add(item)
    return order[: len(a)]


def add_candidate(
    by_n: dict[int, dict[str, dict[str, Any]]],
    n: int,
    order: list[int],
    family: str,
    source: str,
    method: str,
    seed_strategy: str = "",
    mutation_type: str = "",
    mutation_distance_value: float | None = None,
    priority_role: str = "exploitation",
) -> bool:
    if not validate_order(order, n):
        return False
    digest = order_hash(order)
    if digest in by_n[n]:
        return False
    by_n[n][digest] = {
        "n": n,
        "candidate_family": family,
        "candidate_source": source,
        "generation_method": method,
        "seed_strategy": seed_strategy,
        "mutation_type": mutation_type,
        "mutation_distance": "" if mutation_distance_value is None else mutation_distance_value,
        "order": order,
        "order_hash": digest,
        "priority_role": priority_role,
    }
    return True


def load_combined108() -> list[dict[str, Any]]:
    rows = read_csv(COMBINED108_READY)
    parsed: list[dict[str, Any]] = []
    for row in rows:
        n = parse_int(row.get("n"))
        order = parse_order(row.get("order_json"))
        item = dict(row)
        item["n"] = n
        item["order"] = order
        item["order_hash"] = order_hash(order) if order else ""
        item["target"] = parse_float(row.get(PRIMARY_TARGET))
        parsed.append(item)
    return parsed


def validate_inputs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    counts = Counter(row["n"] for row in rows)
    if len(rows) != 108:
        errors.append(f"Expected 108 rows, found {len(rows)}")
    if sorted(counts) != EXPECTED_N:
        errors.append(f"Unexpected N values: {sorted(counts)}")
    for n in EXPECTED_N:
        if counts.get(n) != EXPECTED_COUNTS[n]:
            errors.append(f"Expected {EXPECTED_COUNTS[n]} rows for N{n}, found {counts.get(n, 0)}")
    required = [PRIMARY_TARGET, *SECONDARY_TARGETS, "order_json"]
    for col in required:
        if any(str(row.get(col, "")).strip() == "" for row in rows):
            errors.append(f"Missing required values in {col}")
    for row in rows:
        n = row["n"]
        order = row.get("order")
        if order is None or not validate_order(order, n):
            errors.append(f"Invalid scan order for {row.get('strategy_name')}")
    if RUN22_BEST.exists():
        best_rows = read_csv(RUN22_BEST)
        primary = next((r for r in best_rows if r.get("target") == PRIMARY_TARGET), None)
        if not primary:
            errors.append("Run22 primary best config row was not found")
        else:
            if primary.get("model_name") != "ExtraTreesRegressor":
                errors.append(f"Unexpected run22 model: {primary.get('model_name')}")
            if primary.get("feature_set") != "F01_basic_order":
                errors.append(f"Unexpected run22 feature set: {primary.get('feature_set')}")
            if parse_float(primary.get("spearman"), 0.0) < 0.85:
                warnings.append(f"Run22 macro Spearman below 0.85: {primary.get('spearman')}")
            if parse_float(primary.get("top5_overlap"), 10.0) > 3.0:
                warnings.append(f"Run22 top5 overlap did not show the expected top-region issue: {primary.get('top5_overlap')}")
    else:
        errors.append(f"Missing run22 best configurations: {RUN22_BEST}")
    verdict = "PASS_RUN23_INPUTS_READY_FOR_ACTIVE_LEARNING_DESIGN" if not errors else "FAIL_RUN23_INPUTS_INVALID"
    summary = {
        "verdict": verdict,
        "row_count": len(rows),
        "per_n_counts": dict(sorted(counts.items())),
        "required_features": F01_FEATURES,
        "errors": errors,
        "warnings": warnings,
    }
    write_json(OUTPUT_DIR / "run23_input_validation_summary.json", summary)
    return summary


def train_surrogates(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.linear_model import Ridge

    train_rows: list[dict[str, Any]] = []
    for row in rows:
        feature_row = {**row, **scan_order_features(row["order"], row["n"])}
        feature_row["candidate_family"] = row.get("candidate_family", "")
        feature_row["selection_bucket"] = row.get("selection_bucket", "")
        feature_row["priority_role"] = row.get("priority_role", "")
        feature_row["dataset_source"] = row.get("dataset_source", "")
        train_rows.append(feature_row)

    def numeric_all() -> list[str]:
        exclude_fragments = ["target_", "reward_", "rank_", "cost_", "u2_", "peeq_", "surface", "mises"]
        keep: list[str] = []
        for key, value in train_rows[0].items():
            if key in {"strategy_name", "dataset_source", "order_json", "order_compact", "order_hash", "candidate_family", "selection_bucket", "priority_role", "teacher_validation_status", "order"}:
                continue
            lower = key.lower()
            if any(fragment in lower for fragment in exclude_fragments):
                continue
            try:
                float(value)
            except (TypeError, ValueError):
                continue
            keep.append(key)
        for feature in F01_FEATURES:
            if feature not in keep:
                keep.append(feature)
        return keep

    full_numeric = numeric_all()
    feature_specs = {
        "F01_basic_order": {"numeric": F01_FEATURES, "categorical": []},
        "F03_family_plus_features": {"numeric": full_numeric, "categorical": ["candidate_family", "selection_bucket", "priority_role", "dataset_source"]},
        "F04_no_family_generalization": {"numeric": full_numeric, "categorical": []},
        "F05_n_agnostic": {"numeric": [f for f in full_numeric if f != "n"], "categorical": []},
        "F06_no_dataset_source": {"numeric": full_numeric, "categorical": ["candidate_family", "selection_bucket", "priority_role"]},
    }

    encoders: dict[str, dict[str, list[str]]] = {}

    def fit_encoder(feature_set: str) -> None:
        cats: dict[str, list[str]] = {}
        for col in feature_specs[feature_set]["categorical"]:
            cats[col] = sorted({str(row.get(col, "")) for row in train_rows})
        encoders[feature_set] = cats

    def build_matrix(data_rows: list[dict[str, Any]], feature_set: str) -> np.ndarray:
        spec = feature_specs[feature_set]
        rows_out: list[list[float]] = []
        for row in data_rows:
            values = [parse_float(row.get(col), 0.0) for col in spec["numeric"]]
            for col, categories in encoders.get(feature_set, {}).items():
                current = str(row.get(col, ""))
                values.extend(1.0 if current == category else 0.0 for category in categories)
            rows_out.append(values)
        return np.array(rows_out, dtype=float)

    for feature_set in feature_specs:
        fit_encoder(feature_set)

    models: dict[str, Any] = {}
    y_primary = np.array([parse_float(row.get(PRIMARY_TARGET)) for row in train_rows], dtype=float)

    def fit_extra(name: str, feature_set: str, target: str, n_estimators: int = 260) -> None:
        y = np.array([parse_float(row.get(target)) for row in train_rows], dtype=float)
        model = ExtraTreesRegressor(n_estimators=n_estimators, max_depth=6, min_samples_leaf=2, random_state=42, n_jobs=-1)
        model.fit(build_matrix(train_rows, feature_set), y)
        models[name] = {"model": model, "feature_set": feature_set, "target": target}

    def fit_ridge(name: str, feature_set: str) -> None:
        model = Ridge(alpha=1.0)
        model.fit(build_matrix(train_rows, feature_set), y_primary)
        models[name] = {"model": model, "feature_set": feature_set, "target": PRIMARY_TARGET}

    fit_extra("ET_F01_reward", "F01_basic_order", PRIMARY_TARGET)
    fit_ridge("Ridge_F03_reward", "F03_family_plus_features")
    fit_ridge("Ridge_F06_reward", "F06_no_dataset_source")
    fit_extra("ET_F04_reward", "F04_no_family_generalization", PRIMARY_TARGET)
    fit_extra("ET_F05_reward", "F05_n_agnostic", PRIMARY_TARGET)
    fit_extra("ET_F01_u2", "F01_basic_order", "target_u2_score_combined108_rank")
    fit_extra("ET_F01_peeq", "F01_basic_order", "target_peeq_score_combined108_rank")
    fit_extra("ET_F01_surfaceT", "F01_basic_order", "target_surfaceT_score_combined108_rank")
    fit_extra("ET_F01_mises", "F01_basic_order", "target_mises_score_combined108_rank")
    models["_feature_specs"] = feature_specs
    models["_encoders"] = encoders
    models["_build_matrix"] = build_matrix
    metadata = {
        "model_ensemble": {
            "primary_exploitation_model": "ExtraTreesRegressor/F01_basic_order",
            "n40_stability_comparison_models": ["Ridge/F03_family_plus_features", "Ridge/F06_no_dataset_source"],
            "robustness_models": ["ExtraTreesRegressor/F04_no_family_generalization", "ExtraTreesRegressor/F05_n_agnostic"],
            "secondary_metric_models": ["U2", "PEEQ", "SurfaceT", "Mises"],
        },
        "feature_sets": feature_specs,
        "primary_target": PRIMARY_TARGET,
        "secondary_targets": SECONDARY_TARGETS,
        "training_rows": len(rows),
        "random_state": 42,
        "n_estimators": 260,
        "max_depth": 6,
        "min_samples_leaf": 2,
        "model_label": "combined108 offline active-learning diagnostic ensemble",
        "not_final_or_deployed": True,
    }
    write_json(OUTPUT_DIR / "run23_surrogate_model_metadata.json", metadata)
    return models, metadata


def combined108_refs(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    refs: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        refs[row["n"]].append({"strategy_name": row.get("strategy_name", ""), "order": row["order"], "order_hash": row["order_hash"], "target": row["target"]})
    return refs


def prior_candidate_hashes() -> dict[int, set[str]]:
    result: dict[int, set[str]] = defaultdict(set)
    for path in [RUN18_SCORED, RUN19_BATCH28]:
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
                result[n].add(digest)
    return result


def seed_orders_from_combined108(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    seeds: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for n in EXPECTED_N:
        group = [row for row in rows if row["n"] == n]
        by_reward = sorted(group, key=lambda r: parse_float(r.get("target"), -1.0), reverse=True)[:5]
        by_u2 = sorted(group, key=lambda r: parse_float(r.get("target_u2_score_combined108_rank"), -1.0), reverse=True)[:5]
        named = [r for r in group if str(r.get("strategy_name", "")) == KNOWN_BEST_SEEDS[n] or "method_c" in str(r.get("strategy_name", "")).lower() or "surrogate_top" in str(r.get("strategy_name", "")).lower()]
        seen: set[str] = set()
        for row in by_reward + by_u2 + named:
            digest = row["order_hash"]
            if digest not in seen:
                seeds[n].append(row)
                seen.add(digest)
    return seeds


def generate_candidates(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng_global = random.Random(GLOBAL_SEED)
    refs = combined108_refs(rows)
    existing_hashes = {n: {ref["order_hash"] for ref in group} for n, group in refs.items()}
    prior_hashes = prior_candidate_hashes()
    seeds = seed_orders_from_combined108(rows)
    by_n: dict[int, dict[str, dict[str, Any]]] = {n: {} for n in EXPECTED_N}
    raw_attempts: Counter[str] = Counter()
    duplicate_existing_count: Counter[int] = Counter()
    duplicate_prior_count: Counter[int] = Counter()

    for n in EXPECTED_N:
        rng = random.Random(23000 + n)
        target = TARGET_DEDUP_PER_N[n]

        deterministic_orders = [
            ("raster", "geometry_baseline", "raster_left_to_right", raster(n), "sentinel"),
            ("raster", "geometry_baseline", "raster_right_to_left", raster(n, True), "sentinel"),
            ("odd_even", "geometry_baseline", "odd_even_interlaced", odd_even(n), "calibration"),
            ("center_out", "geometry_baseline", "center_out", center_out(n), "calibration"),
            ("edge_in", "geometry_baseline", "edge_in_alternating", edge_in(n), "calibration"),
            ("maximin", "geometry_baseline", "greedy_maximin_distance", greedy_maximin(n), "diversity"),
            ("block_interleaved", "geometry_baseline", "block_interleaved_quarters", block_interleaved(n), "calibration"),
            ("center_edge", "geometry_baseline", "center_edge_alternating", center_edge_alternating(n), "diversity"),
            ("method_c_inspired", "known_best_neighborhood", "method_c_u2_first_inspired", method_c_inspired(n, "u2"), "top_region_local_search"),
            ("method_c_inspired", "known_best_neighborhood", "method_c_peeq_safety_inspired", method_c_inspired(n, "peeq"), "tradeoff"),
            ("method_c_inspired", "known_best_neighborhood", "method_c_surfaceT_aware_inspired", method_c_inspired(n, "surface"), "tradeoff"),
            ("method_c_inspired", "known_best_inspired", "method_c_diversity_inspired", method_c_inspired(n, "diversity"), "diversity"),
            ("method_c_inspired", "known_best_inspired", "method_c_regular_seed_inspired", method_c_inspired(n, "regular"), "top_region_local_search"),
        ]
        for family, source, method, order, role in deterministic_orders:
            raw_attempts[f"N{n}"] += 1
            digest = order_hash(order)
            if digest in existing_hashes[n]:
                duplicate_existing_count[n] += 1
            elif digest in prior_hashes[n]:
                duplicate_prior_count[n] += 1
            add_candidate(by_n, n, order, family, source, method, priority_role=role)

        for jump in range(1, n):
            for start in range(n):
                for direction in (1, -1):
                    order = regular_jump(n, start, jump, direction)
                    family = "regular_jump_coprime" if gcd(jump, n) == 1 else "regular_jump_non_coprime"
                    role = "top_region_local_search" if gcd(jump, n) == 1 and n in (24, 40) else "sentinel"
                    raw_attempts[f"N{n}"] += 1
                    digest = order_hash(order)
                    if digest in existing_hashes[n]:
                        duplicate_existing_count[n] += 1
                        continue
                    if digest in prior_hashes[n]:
                        duplicate_prior_count[n] += 1
                        continue
                    add_candidate(by_n, n, order, family, "regular_jump_sweep", f"regular_jump_start{start}_jump{jump}_dir{direction}", priority_role=role)

        for seed in seeds[n]:
            base = seed["order"]
            name = str(seed.get("strategy_name", "combined108_seed"))
            for idx in range(520 if n in (24, 40) else 260):
                mode = idx % 8
                if mode == 0:
                    i, j = rng.sample(range(n), 2)
                    order = swap_positions(base, i, j)
                    mtype = "swap_two_positions"
                elif mode == 1:
                    length = rng.randint(2, max(3, min(n, n // 3)))
                    start = rng.randint(0, n - length)
                    order = reverse_segment(base, start, length)
                    mtype = "reverse_segment"
                elif mode == 2:
                    order = rotate_order(base, rng.randint(1, n - 1))
                    mtype = "rotate_sequence"
                elif mode == 3:
                    length = max(1, n // 6)
                    a = rng.randint(0, max(0, n // 2 - length))
                    b = rng.randint(max(a + length, n // 2), n - length)
                    order = swap_blocks(base, a, b, length)
                    mtype = "swap_early_late_blocks"
                elif mode == 4:
                    order = parity_preserving_swap(base, rng)
                    mtype = "parity_preserving_swap"
                elif mode == 5:
                    order = crossover(base, rng.choice(seeds[n])["order"], rng.randint(2, n - 2))
                    mtype = "seed_crossover"
                elif mode == 6:
                    order = alternating_block_crossover(base, rng.choice(seeds[n])["order"], max(1, n // 6))
                    mtype = "alternating_block_crossover"
                else:
                    order = random_biased_order(n, rng, rng.choice(["high_jump", "parity_balanced", "edge_bias", "center_bias"]))
                    mtype = "biased_random_perturbation"
                raw_attempts[f"N{n}"] += 1
                digest = order_hash(order)
                if digest in existing_hashes[n]:
                    duplicate_existing_count[n] += 1
                    continue
                if digest in prior_hashes[n]:
                    duplicate_prior_count[n] += 1
                    continue
                family = "teacher_best_local_search" if name == KNOWN_BEST_SEEDS[n] else ("known_best_mutation" if "S3" in name or "A04" in name else "combined108_seed_mutation")
                role = ["top_region_local_search", "model_disagreement", "uncertainty", "diversity", "tradeoff"][idx % 5]
                add_candidate(
                    by_n,
                    n,
                    order,
                    family,
                    "combined108_known_best_mutation",
                    f"{mtype}_{idx:03d}",
                    seed_strategy=name,
                    mutation_type=mtype,
                    mutation_distance_value=mutation_distance(base, order),
                    priority_role=role,
                )

        modes = [
            "high_jump",
            "parity_balanced",
            "edge_bias",
            "center_bias",
            "low_adjacent",
            "edge_center_alternating",
            "random",
        ]
        guard = 0
        while len(by_n[n]) < target and guard < target * 25:
            guard += 1
            mode = modes[guard % len(modes)]
            raw_attempts[f"N{n}"] += 1
            if guard % 11 == 0:
                a, b = rng.sample([s["order"] for s in seeds[n]], 2)
                order = crossover(a, b, rng.randint(2, n - 2))
                family = "top_region_crossover"
                source = "crossover"
                method = "seed_crossover_random_cut"
                role = "model_disagreement" if guard % 22 == 0 else "top_region_local_search"
            else:
                order = random_biased_order(n, rng, mode)
                family = {
                    "high_jump": "geometry_signal_high_jump",
                    "parity_balanced": "geometry_signal_parity_switch",
                    "edge_bias": "geometry_signal_edge_bias",
                    "center_bias": "geometry_signal_center_bias",
                    "low_adjacent": "negative_control_low_adjacent",
                    "edge_center_alternating": "geometry_signal_edge_center",
                    "random": "diversity_random",
                }[mode]
                source = "geometry_first_random_sweep"
                method = f"{mode}_seeded"
                role = "sentinel" if mode == "low_adjacent" else ("diversity" if mode == "random" else "calibration")
            digest = order_hash(order)
            if digest in existing_hashes[n]:
                duplicate_existing_count[n] += 1
                continue
            if digest in prior_hashes[n]:
                duplicate_prior_count[n] += 1
                continue
            add_candidate(by_n, n, order, family, source, method, priority_role=role)

        if len(by_n[n]) < MIN_DEDUP_PER_N[n]:
            raise RuntimeError(f"Could not generate minimum candidate count for N{n}: {len(by_n[n])}")

    flat: list[dict[str, Any]] = []
    for n in EXPECTED_N:
        for idx, row in enumerate(by_n[n].values(), start=1):
            order = row.pop("order")
            nearest_name, novelty = nearest_order(order, refs[n])
            features = scan_order_features(order, n)
            candidate_id = f"R23_N{n}_C{idx:05d}"
            strategy_name = f"N{n}_{candidate_id}_{row['candidate_family']}"
            flat.append(
                {
                    **row,
                    **features,
                    "candidate_id": candidate_id,
                    "strategy_name": strategy_name[:120],
                    "order_json": canonical_order_json(order),
                    "order_compact": "-".join(str(x) for x in order),
                    "duplicate_of_existing_teacher": False,
                    "is_existing_teacher_order": False,
                    "duplicate_of_run18": row["order_hash"] in prior_hashes[n],
                    "nearest_existing_teacher_strategy": nearest_name,
                    "novelty_distance_to_combined108": novelty,
                    "nearest_existing_teacher_distance": novelty,
                }
            )
    counts = {
        "raw_generated_attempts_per_n": {int(k[1:]): v for k, v in raw_attempts.items()},
        "deduplicated_candidate_count_per_n": {n: sum(1 for row in flat if row["n"] == n) for n in EXPECTED_N},
        "duplicate_existing_teacher_attempts_per_n": dict(duplicate_existing_count),
        "duplicate_prior_candidate_attempts_per_n": dict(duplicate_prior_count),
    }
    return flat, counts


def score_candidates(candidates: list[dict[str, Any]], models: dict[str, Any]) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    build_matrix = models["_build_matrix"]
    x_f01 = build_matrix(candidates, "F01_basic_order")
    pred_et_f01 = models["ET_F01_reward"]["model"].predict(x_f01)
    pred_ridge_f03 = models["Ridge_F03_reward"]["model"].predict(build_matrix(candidates, "F03_family_plus_features"))
    pred_ridge_f06 = models["Ridge_F06_reward"]["model"].predict(build_matrix(candidates, "F06_no_dataset_source"))
    pred_et_f04 = models["ET_F04_reward"]["model"].predict(build_matrix(candidates, "F04_no_family_generalization"))
    pred_et_f05 = models["ET_F05_reward"]["model"].predict(build_matrix(candidates, "F05_n_agnostic"))
    pred_u2 = models["ET_F01_u2"]["model"].predict(x_f01)
    pred_peeq = models["ET_F01_peeq"]["model"].predict(x_f01)
    pred_surface = models["ET_F01_surfaceT"]["model"].predict(x_f01)
    pred_mises = models["ET_F01_mises"]["model"].predict(x_f01)
    primary_model = models["ET_F01_reward"]["model"]
    tree_preds = np.vstack([tree.predict(x_f01) for tree in primary_model.estimators_])
    uncertainty = tree_preds.std(axis=0)
    for idx, row in enumerate(candidates):
        item = dict(row)
        model_values = [float(pred_et_f01[idx]), float(pred_ridge_f03[idx]), float(pred_ridge_f06[idx]), float(pred_et_f04[idx]), float(pred_et_f05[idx])]
        item["pred_reward_ET_F01"] = model_values[0]
        item["pred_reward_Ridge_F03"] = model_values[1]
        item["pred_reward_Ridge_F06"] = model_values[2]
        item["pred_reward_ET_F04"] = model_values[3]
        item["pred_reward_ET_F05"] = model_values[4]
        item["pred_u2_score"] = float(pred_u2[idx])
        item["pred_peeq_score"] = float(pred_peeq[idx])
        item["pred_surfaceT_score"] = float(pred_surface[idx])
        item["pred_mises_score"] = float(pred_mises[idx])
        item["pred_uncertainty_ET_F01_std"] = float(uncertainty[idx])
        item["model_prediction_mean"] = mean(model_values)
        item["model_prediction_std"] = std(model_values)
        item["disagreement_ET_F01_vs_Ridge_F03"] = abs(model_values[0] - model_values[1])
        item["disagreement_ET_F01_vs_Ridge_F06"] = abs(model_values[0] - model_values[2])
        scored.append(item)

    for n in EXPECTED_N:
        group = [row for row in scored if row["n"] == n]
        ranked_et = sorted(group, key=lambda r: parse_float(r["pred_reward_ET_F01"]), reverse=True)
        ranked_mean = sorted(group, key=lambda r: parse_float(r["model_prediction_mean"]), reverse=True)
        max_rank = max(1, len(ranked_et) - 1)
        reward_vals = [parse_float(r["pred_reward_ET_F01"]) for r in group]
        mean_vals = [parse_float(r["model_prediction_mean"]) for r in group]
        novelty_vals = [parse_float(r["novelty_distance_to_combined108"]) for r in group]
        uncertainty_vals = [parse_float(r["pred_uncertainty_ET_F01_std"]) for r in group]
        disagreement_vals = [parse_float(r["model_prediction_std"]) for r in group]
        rmin, rmax = min(reward_vals), max(reward_vals)
        mmin, mmax = min(mean_vals), max(mean_vals)
        nmin, nmax = min(novelty_vals), max(novelty_vals)
        umin, umax = min(uncertainty_vals), max(uncertainty_vals)
        dmin, dmax = min(disagreement_vals), max(disagreement_vals)
        for rank, row in enumerate(ranked_et, start=1):
            row["pred_rank_ET_F01_within_n"] = rank
            row["pred_percentile_ET_F01_within_n"] = 1.0 - (rank - 1) / max_rank
        for rank, row in enumerate(ranked_mean, start=1):
            row["pred_rank_mean_within_n"] = rank
            row["pred_percentile_mean_within_n"] = 1.0 - (rank - 1) / max_rank
        rank_values_by_hash = defaultdict(list)
        for key in ["pred_reward_ET_F01", "pred_reward_Ridge_F03", "pred_reward_Ridge_F06", "pred_reward_ET_F04", "pred_reward_ET_F05"]:
            for rank, row in enumerate(sorted(group, key=lambda r: parse_float(r[key]), reverse=True), start=1):
                rank_values_by_hash[row["order_hash"]].append(rank)
        for row in group:
            pred_norm = safe_divide(parse_float(row["pred_reward_ET_F01"]) - rmin, rmax - rmin)
            mean_norm = safe_divide(parse_float(row["model_prediction_mean"]) - mmin, mmax - mmin)
            novelty_norm = safe_divide(parse_float(row["novelty_distance_to_combined108"]) - nmin, nmax - nmin)
            uncertainty_norm = safe_divide(parse_float(row["pred_uncertainty_ET_F01_std"]) - umin, umax - umin)
            disagreement_norm = safe_divide(parse_float(row["model_prediction_std"]) - dmin, dmax - dmin)
            row["normalized_pred_reward_within_n"] = pred_norm
            row["normalized_model_mean_within_n"] = mean_norm
            row["normalized_novelty_within_n"] = novelty_norm
            row["normalized_uncertainty_within_n"] = uncertainty_norm
            row["normalized_disagreement_within_n"] = disagreement_norm
            row["feature_space_coverage_score"] = 0.5 * novelty_norm + 0.5 * disagreement_norm
            row["disagreement_rank_std"] = std([float(v) for v in rank_values_by_hash[row["order_hash"]]])
            row["exploitation_score"] = pred_norm
            row["exploration_score"] = 0.40 * mean_norm + 0.25 * novelty_norm + 0.20 * uncertainty_norm + 0.15 * disagreement_norm
            row["combined_selection_score"] = 0.35 * mean_norm + 0.25 * novelty_norm + 0.20 * uncertainty_norm + 0.20 * disagreement_norm
    return scored


def take_unique(
    selected: list[dict[str, Any]],
    pool: list[dict[str, Any]],
    count: int,
    bucket: str,
    reason: str,
) -> None:
    used = {row["order_hash"] for row in selected}
    for row in pool:
        if len([r for r in selected if r.get("selection_bucket") == bucket]) >= count:
            break
        if row["order_hash"] in used:
            continue
        item = dict(row)
        item["selection_bucket"] = bucket
        item["selection_reason"] = reason
        selected.append(item)
        used.add(item["order_hash"])


def select_for_n(group: list[dict[str, Any]], n: int, target_count: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    fresh_group = [row for row in group if not row.get("duplicate_of_run18")]
    if len(fresh_group) >= target_count:
        group = fresh_group
    top_region = sorted(
        [r for r in group if "teacher_best" in r.get("candidate_family", "") or r.get("priority_role") == "top_region_local_search"],
        key=lambda r: (parse_float(r["model_prediction_mean"]), parse_float(r["novelty_distance_to_combined108"])),
        reverse=True,
    )
    disagreement = sorted(group, key=lambda r: (parse_float(r["model_prediction_std"]), parse_float(r["disagreement_rank_std"]), parse_float(r["model_prediction_mean"])), reverse=True)
    uncertainty = sorted(group, key=lambda r: (parse_float(r["pred_uncertainty_ET_F01_std"]), parse_float(r["model_prediction_mean"])), reverse=True)
    diversity = sorted(
        [r for r in group if parse_float(r["model_prediction_mean"]) >= np.quantile([parse_float(x["model_prediction_mean"]) for x in group], 0.35)],
        key=lambda r: (parse_float(r["novelty_distance_to_combined108"]), parse_float(r["feature_space_coverage_score"])),
        reverse=True,
    )
    tradeoff = sorted(
        group,
        key=lambda r: (
            abs(parse_float(r["pred_u2_score"]) - parse_float(r["pred_peeq_score"]))
            + abs(parse_float(r["pred_u2_score"]) - parse_float(r["pred_surfaceT_score"])),
            parse_float(r["model_prediction_mean"]),
        ),
        reverse=True,
    )
    sentinel = sorted(
        group,
        key=lambda r: (
            parse_float(r["novelty_distance_to_combined108"]),
            -parse_float(r["model_prediction_mean"]),
            parse_float(r["adjacent_jump_count"]),
        ),
        reverse=True,
    )
    exploitation = sorted(group, key=lambda r: parse_float(r["pred_reward_ET_F01"]), reverse=True)
    top_u2 = sorted(group, key=lambda r: parse_float(r["pred_u2_score"]), reverse=True)
    geometry = sorted(
        group,
        key=lambda r: (
            parse_float(r["parity_switch_rate"]),
            parse_float(r["normalized_mean_jump"]),
            -parse_float(r["adjacent_jump_count"]),
            parse_float(r["model_prediction_mean"]),
        ),
        reverse=True,
    )

    bucket_counts = {
        "top_region_local_search": max(1, round(target_count * 0.22)),
        "model_disagreement": max(1, round(target_count * 0.18)),
        "uncertainty_calibration": max(1, round(target_count * 0.17)),
        "diversity_coverage": max(1, round(target_count * 0.17)),
        "tradeoff_probe": max(1, round(target_count * 0.12)),
        "sentinel_control": 1 if target_count <= 8 else 2,
        "exploitation_reference": max(1, round(target_count * 0.08)),
    }
    take_unique(selected, top_region or geometry, bucket_counts["top_region_local_search"], "top_region_local_search", "Near current teacher-best regions with controlled topology perturbations.")
    take_unique(selected, disagreement, bucket_counts["model_disagreement"], "model_disagreement", "High disagreement among ET_F01 and Ridge F03/F06 reward models.")
    take_unique(selected, uncertainty, bucket_counts["uncertainty_calibration"], "uncertainty_calibration", "High tree uncertainty with moderate/high predicted reward.")
    take_unique(selected, diversity, bucket_counts["diversity_coverage"], "diversity_coverage", "High novelty and feature-space coverage relative to combined108.")
    take_unique(selected, tradeoff, bucket_counts["tradeoff_probe"], "tradeoff_probe", "Potential U2/PEEQ/SurfaceT/Mises tradeoff probe.")
    take_unique(selected, sentinel, bucket_counts["sentinel_control"], "sentinel_control", "Structurally unusual or lower-predicted calibration sentinel.")
    take_unique(selected, exploitation, bucket_counts["exploitation_reference"], "exploitation_reference", "Small exploitation reference bucket, not dominant.")
    if len(selected) < target_count:
        take_unique(selected, top_u2, max(1, target_count - len(selected)), "u2_reference_fill", "Fill from U2-score reference candidates.")
    if len(selected) < target_count:
        take_unique(selected, sorted(group, key=lambda r: parse_float(r["combined_selection_score"]), reverse=True), target_count - len(selected), "combined_selection_fill", "Fill by combined prediction, novelty, and uncertainty score.")
    selected = selected[:target_count]
    for idx, row in enumerate(selected, start=1):
        row["shortlist_rank_within_n"] = idx
        row["shortlist_name"] = f"R23_N{n}_S{idx:02d}_{row['selection_bucket']}"
    return selected


def select_shortlists(scored: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    shortlist64: list[dict[str, Any]] = []
    for n in EXPECTED_N:
        group = [row for row in scored if row["n"] == n]
        shortlist64.extend(select_for_n(group, n, SHORTLIST_COUNTS[n]))

    def derive_batch(shortlist: list[dict[str, Any]], counts: dict[int, int], label: str) -> list[dict[str, Any]]:
        batch: list[dict[str, Any]] = []
        for n in EXPECTED_N:
            group = [row for row in shortlist if row["n"] == n]
            priority_buckets = [
                "top_region_local_search",
                "model_disagreement",
                "uncertainty_calibration",
                "diversity_coverage",
                "tradeoff_probe",
                "sentinel_control",
                "exploitation_reference",
                "u2_reference_fill",
                "combined_selection_fill",
            ]
            selected: list[dict[str, Any]] = []
            for bucket in priority_buckets:
                take_unique(selected, [r for r in group if r.get("selection_bucket") == bucket], 1, bucket, f"{label} balanced bucket selection.")
                if len(selected) >= counts[n]:
                    break
            if len(selected) < counts[n]:
                take_unique(selected, sorted(group, key=lambda r: parse_float(r["combined_selection_score"]), reverse=True), counts[n] - len(selected), "batch_fill", f"{label} fill by combined selection score.")
            for idx, row in enumerate(selected[: counts[n]], start=1):
                item = dict(row)
                item[f"{label}_rank_within_n"] = idx
                batch.append(item)
        return batch

    return shortlist64, derive_batch(shortlist64, BATCH32_COUNTS, "batch32"), derive_batch(shortlist64, BATCH24_COUNTS, "batch24")


def existing_best_by_n(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    best: dict[int, dict[str, Any]] = {}
    for n in EXPECTED_N:
        group = [r for r in rows if r["n"] == n]
        best_reward = max(group, key=lambda r: parse_float(r.get(PRIMARY_TARGET)))
        best_u2 = max(group, key=lambda r: parse_float(r.get("target_u2_score_combined108_rank")))
        best[n] = {
            "combined108_best_reward_strategy": best_reward.get("strategy_name", ""),
            "combined108_best_reward": parse_float(best_reward.get(PRIMARY_TARGET)),
            "combined108_best_u2_strategy": best_u2.get("strategy_name", ""),
            "combined108_best_u2_score": parse_float(best_u2.get("target_u2_score_combined108_rank")),
        }
    return best


def predicted_improvement(scored: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_existing = existing_best_by_n(rows)
    out: list[dict[str, Any]] = []
    for n in EXPECTED_N:
        group = [row for row in scored if row["n"] == n]
        top_et = max(group, key=lambda r: parse_float(r["pred_reward_ET_F01"]))
        top_mean = max(group, key=lambda r: parse_float(r["model_prediction_mean"]))
        top_disagree = max(group, key=lambda r: parse_float(r["model_prediction_std"]))
        top_uncertainty = max(group, key=lambda r: parse_float(r["pred_uncertainty_ET_F01_std"]))
        base = best_existing[n]
        gap = parse_float(top_et["pred_reward_ET_F01"]) - parse_float(base["combined108_best_reward"])
        out.append(
            {
                "n": n,
                **base,
                "top_predicted_run23_candidate_by_ET_F01": top_et["strategy_name"],
                "top_predicted_ET_F01_candidate_id": top_et["candidate_id"],
                "top_predicted_ET_F01_family": top_et["candidate_family"],
                "top_predicted_ET_F01_reward": top_et["pred_reward_ET_F01"],
                "top_predicted_run23_candidate_by_model_mean": top_mean["strategy_name"],
                "top_predicted_model_mean_reward": top_mean["model_prediction_mean"],
                "top_model_disagreement_candidate": top_disagree["strategy_name"],
                "top_model_disagreement_std": top_disagree["model_prediction_std"],
                "top_uncertainty_candidate": top_uncertainty["strategy_name"],
                "top_uncertainty_std": top_uncertainty["pred_uncertainty_ET_F01_std"],
                "predicted_gap_vs_combined108_best_reward": gap,
                "predicted_exceeds_combined108_best_reward_surrogate_only": gap > 0,
                "top_predicted_novelty_distance": top_et["novelty_distance_to_combined108"],
                "nearest_existing_teacher_strategy": top_et["nearest_existing_teacher_strategy"],
                "note": "Surrogate-only prediction; teacher validation required before any physical claim.",
            }
        )
    return out


def diagnostics(candidates: list[dict[str, Any]], scored: list[dict[str, Any]], shortlist: list[dict[str, Any]], batch32: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n in EXPECTED_N:
        for label, data in [("unscored", candidates), ("scored", scored), ("shortlist64", shortlist), ("batch32", batch32)]:
            group = [r for r in data if r["n"] == n]
            rewards = [parse_float(r.get("pred_reward_ET_F01")) for r in group if "pred_reward_ET_F01" in r]
            means = [parse_float(r.get("model_prediction_mean")) for r in group if "model_prediction_mean" in r]
            novelty = [parse_float(r.get("novelty_distance_to_combined108")) for r in group if "novelty_distance_to_combined108" in r]
            uncertainty = [parse_float(r.get("pred_uncertainty_ET_F01_std")) for r in group if "pred_uncertainty_ET_F01_std" in r]
            disagreement = [parse_float(r.get("model_prediction_std")) for r in group if "model_prediction_std" in r]
            rows.append(
                {
                    "n": n,
                    "dataset": label,
                    "count": len(group),
                    "family_counts_json": json.dumps(dict(Counter(r.get("candidate_family", "") for r in group)), sort_keys=True),
                    "bucket_counts_json": json.dumps(dict(Counter(r.get("selection_bucket", "") for r in group)), sort_keys=True),
                    "pred_reward_ET_F01_mean": mean(rewards, math.nan),
                    "pred_reward_model_mean": mean(means, math.nan),
                    "pred_reward_max": max(rewards) if rewards else "",
                    "pred_reward_min": min(rewards) if rewards else "",
                    "novelty_mean": mean(novelty, math.nan),
                    "uncertainty_mean": mean(uncertainty, math.nan),
                    "disagreement_mean": mean(disagreement, math.nan),
                }
            )
    return rows


def write_claim_boundary() -> None:
    md = "\n".join(
        [
            "# Run23 Claim Boundary",
            "",
            "## Safe claims",
            "- Run23 creates an active-learning candidate design based on combined108 surrogate diagnostics.",
            "- Run23 focuses on improving top-region calibration and N24/N40 coverage.",
            "- Run23 candidates include top-region local search, model-disagreement, uncertainty, diversity, tradeoff, and sentinel buckets.",
            "- Run23 produces recommended future teacher-validation batches for human review.",
            "",
            "## Unsafe claims",
            "- Do not claim teacher validation.",
            "- Do not claim physical superiority.",
            "- Do not claim trained variable-N RL policy success.",
            "- Do not claim arbitrary-N generalization.",
            "- Do not claim surrogate predictions are ground truth.",
            "- Do not claim feature importance is causal.",
            "- Do not claim CAE/INP files exist.",
            "",
            "Verdict: RUN23_ACTIVE_LEARNING_DESIGN_ONLY_NO_TEACHER_VALIDATION",
            "",
        ]
    )
    (OUTPUT_DIR / "run23_claim_boundary.md").write_text(md, encoding="utf-8")
    write_json(
        OUTPUT_DIR / "run23_claim_boundary.json",
        {
            "verdict": "RUN23_ACTIVE_LEARNING_DESIGN_ONLY_NO_TEACHER_VALIDATION",
            "safe_claims": [
                "active-learning candidate design based on combined108 surrogate diagnostics",
                "top-region calibration and N24/N40 coverage focus",
                "top-region local search, model-disagreement, uncertainty, diversity, tradeoff, and sentinel buckets used",
                "future teacher-validation batches produced for human review",
            ],
            "unsafe_claims": [
                "teacher validation",
                "physical superiority",
                "trained variable-N RL policy success",
                "arbitrary-N generalization",
                "surrogate predictions as ground truth",
                "CAE/INP files exist",
            ],
        },
    )


def write_report(
    validation: dict[str, Any],
    model_meta: dict[str, Any],
    counts: dict[str, Any],
    shortlist: list[dict[str, Any]],
    batch32: list[dict[str, Any]],
    batch24: list[dict[str, Any]],
    improvement: list[dict[str, Any]],
    outputs: list[str],
) -> None:
    lines = [
        "# Stage 3 Run 23 - Combined108 Active-Learning Coverage and Calibration Design",
        "",
        "## Purpose",
        "Generate an offline active-learning candidate design using combined108 diagnostics, focused on top-region calibration, model disagreement, uncertainty, and N24/N40 coverage rather than pure exploitation.",
        "",
        "## Inputs",
        f"- Combined108 RL-ready dataset: `{COMBINED108_READY}`",
        f"- Combined108 teacher dataset: `{COMBINED108_TEACHER}`",
        f"- Run22 best configurations: `{RUN22_BEST}`",
        f"- Run22 feature definitions: `{RUN22_FEATURE_DEFS}`",
        f"- Previous run18 candidate pool for duplicate avoidance: `{RUN18_SCORED}`",
        "",
        "## Run22 Motivation",
        "- Run22 improved leave-N-out macro Spearman to 0.8651 but weakened top5 retrieval to 2.5/5.",
        "- Run23 therefore prioritizes calibration, coverage, disagreement, and top-region exploration instead of pure predicted reward maximization.",
        "",
        "## Model Ensemble Used For Design",
        f"- Ensemble: `{model_meta['model_ensemble']}`",
        f"- Primary target: `{model_meta['primary_target']}`",
        f"- Training rows used for offline scoring: `{model_meta['training_rows']}`",
        "- These are combined108 offline diagnostic surrogates, not final or deployed models.",
        "",
        "## Candidate Generation Scope",
        f"- N values: `{EXPECTED_N}`",
        f"- Deduplicated candidates per N: `{counts['deduplicated_candidate_count_per_n']}`",
        "- Selection is intentionally biased toward N24/N40.",
        "",
        "## Candidate Generation Methods",
        "- Top-region local search around known combined108 teacher-best cases.",
        "- N24/N40 top5-retrieval calibration sweeps in parity, jump, first/last track, and direction-reversal features.",
        "- Model-disagreement and uncertainty candidates from ET_F01 versus Ridge F03/F06.",
        "- Tradeoff candidates for U2/PEEQ/SurfaceT/Mises tension.",
        "- Regular-jump, Method-C, and known-best neighborhoods.",
        "- Diversity and sentinel/control candidates.",
        "- No candidate is labelled as trained RL output.",
        "",
        "## Candidate Validation and Deduplication",
        "- All generated orders were validated as legal permutations of 0..N-1.",
        "- Exact duplicates of combined108 teacher orders were removed.",
        "- Exact duplicates of the prior run18/run19 candidate pools were avoided where detected.",
        "",
        "## Multi-Model Scoring",
        "- Candidates were scored by ET_F01 reward, Ridge F03/F06 reward, ET F04/F05 robustness models, and secondary U2/PEEQ/SurfaceT/Mises models.",
        "- Tree-wise standard deviation, model prediction standard deviation, rank disagreement, novelty, and feature-space coverage are reported as active-learning diagnostics.",
        "",
        "## Selection Policy",
        f"- Shortlist64 counts: `{dict(Counter(row['n'] for row in shortlist))}`",
        f"- Recommended batch32 counts: `{dict(Counter(row['n'] for row in batch32))}`",
        f"- Alternative batch24 counts: `{dict(Counter(row['n'] for row in batch24))}`",
        "- Selection buckets include top-region local search, model disagreement, uncertainty calibration, diversity coverage, tradeoff probes, sentinel controls, and a small exploitation reference bucket.",
        "",
        "## Predicted Comparison to Combined108 Best",
    ]
    for row in improvement:
        lines.append(
            f"- N{row['n']}: ET_F01 top `{row['top_predicted_run23_candidate_by_ET_F01']}` predicted reward `{float(row['top_predicted_ET_F01_reward']):.4f}`, "
            f"gap vs existing best `{float(row['predicted_gap_vs_combined108_best_reward']):.4f}` surrogate-only."
        )
    lines.extend(
        [
            "",
            "## N24/N40 Coverage Analysis",
            "- The shortlist and batch outputs deliberately allocate most capacity to N24/N40.",
            "- Coverage uses novelty to combined108, model disagreement, uncertainty, and top-region local search around known teacher-best cases.",
            "",
            "## Predicted Improvement vs Combined108 Best",
            "Predicted improvements are surrogate-only. They require future teacher validation before any physical claim.",
            "",
            "## Diagnostics",
            "- Candidate distributions, novelty, uncertainty, family composition, and duplicate-removal summaries are written to the diagnostics files.",
            "",
            "## Claim Boundary",
            "RUN23_ACTIVE_LEARNING_DESIGN_ONLY_NO_TEACHER_VALIDATION. No teacher validation, physical superiority, trained RL success, arbitrary-N generalization, or CAE/INP existence is claimed.",
            "",
            "## Output Files",
        ]
    )
    lines.extend(f"- `{path}`" for path in outputs)
    lines.extend(
        [
            "",
            "## Recommended Run24",
            "Human review and handoff packaging for either batch32 or batch24. Do not generate CAE/INP until the user selects a batch. Choose batch24 to control compute cost; choose batch32 for stronger N24/N40 calibration.",
            "",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def update_run_index() -> None:
    row = "| run_23 | Combined108 active-learning coverage calibration design | Generate N24/N40-focused offline active-learning candidates using combined108 surrogate diagnostics, emphasizing top-region calibration, disagreement, uncertainty, diversity, tradeoff, and sentinel buckets. | `scripts/stage3/run_23_generate_active_learning_coverage_calibration_candidates.py` | `docs/stage3/runs/run_23_combined108_active_learning_coverage_calibration_design/RUN_23_COMBINED108_ACTIVE_LEARNING_COVERAGE_CALIBRATION_DESIGN_REPORT.md` | `outputs/stage3_run_23_combined108_active_learning_coverage_calibration_design/` | `PASS_RUN23_INPUTS_READY_FOR_ACTIVE_LEARNING_DESIGN` | No Abaqus, no ODB, no abqjobpilot, no CAE/INP/JNL generation, no teacher validation, no final RL policy training, no commit/push. Next: run24 handoff packaging after user selects batch32 or batch24. |"
    if RUN_INDEX_PATH.exists():
        text = RUN_INDEX_PATH.read_text(encoding="utf-8")
        if "| run_23 |" not in text:
            RUN_INDEX_PATH.write_text(text.rstrip() + "\n" + row + "\n", encoding="utf-8")


def write_plots(scored: list[dict[str, Any]], batch32: list[dict[str, Any]]) -> list[str]:
    paths: list[str] = []
    try:
        import matplotlib.pyplot as plt

        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        for n in EXPECTED_N:
            vals = [parse_float(r["pred_reward_ET_F01"]) for r in scored if r["n"] == n]
            ax.hist(vals, bins=30, alpha=0.45, label=f"N{n}")
        ax.set_title("Run23 ET_F01 predicted reward distribution")
        ax.set_xlabel("Predicted combined108 U2-primary reward")
        ax.set_ylabel("Candidate count")
        ax.legend()
        fig.tight_layout()
        path = FIGURE_DIR / "run23_predicted_reward_histogram.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(str(path))

        fig, ax = plt.subplots(figsize=(7, 5))
        x = [parse_float(r["pred_uncertainty_ET_F01_std"]) for r in scored]
        y = [parse_float(r["pred_reward_ET_F01"]) for r in scored]
        ax.scatter(x, y, s=4, alpha=0.35)
        ax.set_title("Run23 predicted reward vs uncertainty")
        ax.set_xlabel("ET_F01 tree prediction std")
        ax.set_ylabel("Predicted reward")
        fig.tight_layout()
        path = FIGURE_DIR / "run23_predicted_reward_vs_uncertainty.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(str(path))

        fig, ax = plt.subplots(figsize=(7, 5))
        x = [parse_float(r["model_prediction_std"]) for r in scored]
        y = [parse_float(r["pred_reward_ET_F01"]) for r in scored]
        ax.scatter(x, y, s=4, alpha=0.35)
        ax.set_title("Run23 predicted reward vs model disagreement")
        ax.set_xlabel("Model prediction std")
        ax.set_ylabel("Predicted reward")
        fig.tight_layout()
        path = FIGURE_DIR / "run23_predicted_reward_vs_disagreement.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(str(path))

        bucket_counts = Counter(r.get("selection_bucket", "") for r in batch32)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(list(bucket_counts), list(bucket_counts.values()))
        ax.set_title("Run23 batch32 bucket composition")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        path = FIGURE_DIR / "run23_batch32_bucket_composition.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(str(path))
    except Exception as exc:  # noqa: BLE001
        write_json(OUTPUT_DIR / "run23_plotting_warning.json", {"plotting_warning": str(exc)})
    return paths


def git_branch() -> str:
    try:
        result = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception:
        return ""


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_combined108()
    validation = validate_inputs(rows)
    if validation["verdict"].startswith("FAIL"):
        print(validation["verdict"])
        return 2

    models, model_meta = train_surrogates(rows)
    candidates, counts = generate_candidates(rows)
    scored = score_candidates(candidates, models)
    shortlist64, batch32, batch24 = select_shortlists(scored)
    improvement = predicted_improvement(scored, rows)
    diag_rows = diagnostics(candidates, scored, shortlist64, batch32)

    outputs: list[str] = []
    for filename, data, writer in [
        ("run23_candidate_pool_unscored.csv", candidates, write_csv),
        ("run23_candidate_pool_scored.csv", scored, write_csv),
        ("run23_candidate_shortlist64.csv", shortlist64, write_csv),
        ("run23_recommended_active_learning_batch32.csv", batch32, write_csv),
        ("run23_conservative_active_learning_batch24.csv", batch24, write_csv),
        ("run23_predicted_comparison_vs_combined108.csv", improvement, write_csv),
        ("run23_candidate_generation_diagnostics.csv", diag_rows, write_csv),
    ]:
        path = OUTPUT_DIR / filename
        writer(path, data)
        outputs.append(str(path))
    for filename, data in [
        ("run23_candidate_pool_unscored.json", candidates),
        ("run23_candidate_pool_scored.json", scored),
        ("run23_candidate_shortlist64.json", shortlist64),
        ("run23_recommended_active_learning_batch32.json", batch32),
        ("run23_conservative_active_learning_batch24.json", batch24),
        ("run23_predicted_comparison_vs_combined108.json", improvement),
        ("run23_candidate_generation_diagnostics.json", diag_rows),
    ]:
        path = OUTPUT_DIR / filename
        write_table_json(path, data)
        outputs.append(str(path))

    write_claim_boundary()
    outputs.extend([str(OUTPUT_DIR / "run23_claim_boundary.md"), str(OUTPUT_DIR / "run23_claim_boundary.json"), str(OUTPUT_DIR / "run23_input_validation_summary.json"), str(OUTPUT_DIR / "run23_surrogate_model_metadata.json")])
    plot_paths = write_plots(scored, batch32)
    outputs.extend(plot_paths)
    write_report(validation, model_meta, counts, shortlist64, batch32, batch24, improvement, outputs)
    outputs.append(str(REPORT_PATH))
    update_run_index()

    manifest = {
        "run_id": RUN_ID,
        "run_name": RUN_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "branch": git_branch(),
        "script_path": str(ROOT / "scripts" / "stage3" / "run_23_generate_active_learning_coverage_calibration_candidates.py"),
        "input_files": [str(p) for p in [COMBINED108_READY, COMBINED108_TEACHER, COMBINED108_LEADERBOARD, RUN22_FEATURES, RUN22_DETAILED, RUN22_BEST, RUN22_PREDICTIONS, RUN22_FEATURE_DEFS, RUN22_REPORT, RUN18_SCORED, RUN19_BATCH28] if p.exists()],
        "output_files": outputs,
        "validation_verdict": validation["verdict"],
        "candidate_pool_count": len(scored),
        "candidate_pool_count_per_n": dict(Counter(row["n"] for row in scored)),
        "raw_generated_candidate_count_per_n": counts["raw_generated_attempts_per_n"],
        "deduplicated_candidate_count_per_n": counts["deduplicated_candidate_count_per_n"],
        "shortlist64_count": len(shortlist64),
        "shortlist64_count_per_n": dict(Counter(row["n"] for row in shortlist64)),
        "batch32_count": len(batch32),
        "batch32_count_per_n": dict(Counter(row["n"] for row in batch32)),
        "batch24_count": len(batch24),
        "batch24_count_per_n": dict(Counter(row["n"] for row in batch24)),
        "report_path": str(REPORT_PATH),
        "claim_boundary_path": str(OUTPUT_DIR / "run23_claim_boundary.md"),
        "no_solver_run": True,
        "no_odb_opened": True,
        "no_abqjobpilot_run": True,
        "no_cae_inp_generated": True,
        "no_teacher_validation": True,
        "no_rl_policy_training": True,
        "no_commit_or_push": True,
    }
    write_json(MANIFEST_PATH, manifest)

    top_by_n = {n: max([r for r in scored if r["n"] == n], key=lambda r: parse_float(r["pred_reward_ET_F01"]))["strategy_name"] for n in EXPECTED_N}
    print(validation["verdict"])
    print(f"candidate_pool={len(scored)} per_n={dict(Counter(row['n'] for row in scored))}")
    print(f"shortlist64={len(shortlist64)} per_n={dict(Counter(row['n'] for row in shortlist64))}")
    print(f"batch32={len(batch32)} per_n={dict(Counter(row['n'] for row in batch32))}")
    print(f"batch24={len(batch24)} per_n={dict(Counter(row['n'] for row in batch24))}")
    print(f"top_by_n={top_by_n}")
    print(f"report={REPORT_PATH}")
    print(f"manifest={MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
