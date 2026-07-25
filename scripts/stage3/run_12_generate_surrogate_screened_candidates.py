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
RUN_ID = "run_12_offline_surrogate_screened_candidate_generation"
RUN_NAME = "offline surrogate-screened variable-N candidate generation"

SURROGATE_TABLE = ROOT / "outputs" / "stage3_run_10_variable_n_normalized_reward_surrogate_dataset" / "probe60_surrogate_pretraining_table.csv"
FEATURE_TABLE = ROOT / "outputs" / "stage3_run_10_variable_n_normalized_reward_surrogate_dataset" / "probe60_scan_order_features.csv"
REWARD_DATASET = ROOT / "outputs" / "stage3_run_10_variable_n_normalized_reward_surrogate_dataset" / "probe60_variable_n_reward_dataset.csv"
RUN11_DETAILED = ROOT / "outputs" / "stage3_run_11_variable_n_surrogate_reward_model_validation" / "surrogate_validation_results_detailed.csv"
RUN11_BEST = ROOT / "outputs" / "stage3_run_11_variable_n_surrogate_reward_model_validation" / "best_surrogate_configurations.csv"
RUN11_PREDICTIONS = ROOT / "outputs" / "stage3_run_11_variable_n_surrogate_reward_model_validation" / "surrogate_predictions_target_reward_mean_all.csv"
RUN11_FEATURE_DEFS = ROOT / "outputs" / "stage3_run_11_variable_n_surrogate_reward_model_validation" / "run11_feature_set_definitions.json"
RUN11_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_11_variable_n_surrogate_reward_model_validation" / "RUN_11_VARIABLE_N_SURROGATE_REWARD_MODEL_VALIDATION_REPORT.md"
RUN06_CANDIDATES = ROOT / "outputs" / "stage3_run_06_variable_n_probe60_candidate_order_generation" / "variable_N_probe60_candidate_orders.csv"

OUTPUT_DIR = ROOT / "outputs" / "stage3_run_12_offline_surrogate_screened_candidate_generation"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = ROOT / "docs" / "stage3" / "runs" / RUN_ID
REPORT_PATH = REPORT_DIR / "RUN_12_OFFLINE_SURROGATE_SCREENED_CANDIDATE_GENERATION_REPORT.md"
MANIFEST_PATH = ROOT / "artifacts" / "manifests" / "stage3_run_12_manifest.json"
RUN_INDEX_PATH = ROOT / "docs" / "stage3" / "STAGE3_RUN_INDEX.md"

EXPECTED_N = [12, 16, 24, 40]
EXPECTED_ROWS_PER_N = 15
TARGET_RAW_PER_N = 1200
MIN_RAW_PER_N = 1000
MAX_DEDUP_NEW_PER_N = 1250
GLOBAL_SEED = 42
PRIMARY_TARGET = "target_reward_mean_all"
SECONDARY_TARGETS = [
    "target_reward_v01_u2_primary",
    "target_u2_score_rank",
    "target_peeq_score_rank",
    "target_surfaceT_score_rank",
]
F03 = "F03_family_plus_features"


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


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    return numerator / denominator if denominator else default


def mean(values: list[float], default: float = 0.0) -> float:
    return statistics.fmean(values) if values else default


def std(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def median(values: list[float], default: float = 0.0) -> float:
    return statistics.median(values) if values else default


def entropy(values: list[int]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = len(values)
    return -sum((count / total) * math.log(count / total, 2) for count in counts.values())


def order_hash(order: list[int]) -> str:
    return hashlib.sha1(",".join(str(x) for x in order).encode("ascii")).hexdigest()[:16]


def validate_order(order: list[int], n: int) -> bool:
    return len(order) == n and set(order) == set(range(n)) and len(order) == len(set(order))


def parse_order(text: str) -> list[int]:
    value = json.loads(text)
    return [int(x) for x in value]


def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)


def raster(n: int, reverse: bool = False) -> list[int]:
    order = list(range(n))
    return list(reversed(order)) if reverse else order


def odd_even(n: int) -> list[int]:
    return list(range(0, n, 2)) + list(range(1, n, 2))


def center_out(n: int) -> list[int]:
    left = (n - 1) // 2
    right = left + 1
    order: list[int] = []
    while left >= 0 or right < n:
        if left >= 0:
            order.append(left)
            left -= 1
        if right < n:
            order.append(right)
            right += 1
    return order


def edge_in(n: int) -> list[int]:
    left, right = 0, n - 1
    order: list[int] = []
    while left <= right:
        order.append(left)
        if left != right:
            order.append(right)
        left += 1
        right -= 1
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
    order: list[int] = []
    for rows in zip(*[block + [None] * (max(map(len, blocks)) - len(block)) for block in blocks]):
        for item in rows:
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


def method_c_like(n: int, mode: str) -> list[int]:
    base = greedy_maximin(n)
    if mode == "u2_first":
        return base
    if mode == "peeq_aware":
        return interleave_orders(base, odd_even(n), 2)
    if mode == "surfaceT_aware":
        return interleave_orders(center_edge_alternating(n), base, 3)
    if mode == "diversity_preserved":
        return interleave_orders(base, edge_in(n), 4)
    if mode == "regular_jump_seed":
        jump = max(2, n // 3)
        while gcd(jump, n) != 1 and jump < n:
            jump += 1
        return interleave_orders(regular_jump(n, 0, min(jump, n - 1)), base, 2)
    if mode == "maximin_seed":
        return interleave_orders(base, center_out(n), 1)
    return mutate_order(base, random.Random(1000 + n), "reverse_short_segment", n)


def interleave_orders(a: list[int], b: list[int], block: int) -> list[int]:
    order: list[int] = []
    seen: set[int] = set()
    toggle = True
    ia = ib = 0
    while len(order) < len(a):
        source = a if toggle else b
        idx = ia if toggle else ib
        added = 0
        while idx < len(source) and added < block:
            item = source[idx]
            idx += 1
            if item not in seen:
                order.append(item)
                seen.add(item)
                added += 1
        if toggle:
            ia = idx
        else:
            ib = idx
        toggle = not toggle
        if ia >= len(a) and ib >= len(b):
            break
    order.extend(x for x in a if x not in seen)
    return order


def mutate_order(order: list[int], rng: random.Random, mutation: str, n: int) -> list[int]:
    mutated = order[:]
    if mutation == "swap_two_positions":
        a, b = rng.sample(range(n), 2)
        mutated[a], mutated[b] = mutated[b], mutated[a]
    elif mutation == "reverse_short_segment":
        length = rng.randint(2, max(2, min(6, n // 3)))
        start = rng.randint(0, n - length)
        mutated[start : start + length] = reversed(mutated[start : start + length])
    elif mutation == "rotate_sequence":
        k = rng.randint(1, n - 1)
        mutated = mutated[k:] + mutated[:k]
    elif mutation == "swap_early_late_blocks":
        block = max(2, n // 5)
        a = rng.randint(0, max(0, n // 2 - block))
        b = rng.randint(max(n // 2, block), n - block)
        mutated[a : a + block], mutated[b : b + block] = mutated[b : b + block], mutated[a : a + block]
    elif mutation == "perturb_first_quarter":
        q = max(2, n // 4)
        segment = mutated[:q]
        rng.shuffle(segment)
        mutated[:q] = segment
    elif mutation == "perturb_last_quarter":
        q = max(2, n // 4)
        segment = mutated[-q:]
        rng.shuffle(segment)
        mutated[-q:] = segment
    elif mutation == "parity_preserving_swap":
        parity = rng.choice([0, 1])
        positions = [idx for idx, value in enumerate(mutated) if value % 2 == parity]
        if len(positions) >= 2:
            a, b = rng.sample(positions, 2)
            mutated[a], mutated[b] = mutated[b], mutated[a]
    elif mutation == "edge_center_swap":
        edge_tracks = set(range(0, max(1, n // 4))) | set(range(n - max(1, n // 4), n))
        center_tracks = set(range(n // 4, n - n // 4))
        ep = [idx for idx, value in enumerate(mutated) if value in edge_tracks]
        cp = [idx for idx, value in enumerate(mutated) if value in center_tracks]
        if ep and cp:
            a = rng.choice(ep)
            b = rng.choice(cp)
            mutated[a], mutated[b] = mutated[b], mutated[a]
    return mutated


def crossover(a: list[int], b: list[int], mode: str) -> list[int]:
    n = len(a)
    if mode == "half":
        prefix = a[: n // 2]
    elif mode == "quarter":
        prefix = a[: max(1, n // 4)]
    else:
        prefix = []
        blocks = max(1, n // 5)
        use_a = True
        idx = 0
        while idx < n:
            prefix.extend((a if use_a else b)[idx : idx + blocks])
            use_a = not use_a
            idx += blocks
        prefix = prefix[: n // 2]
    seen = set()
    order = []
    for item in prefix + b:
        if item not in seen:
            order.append(item)
            seen.add(item)
    order.extend(x for x in range(n) if x not in seen)
    return order


def random_family_order(n: int, rng: random.Random, family: str) -> list[int]:
    order = list(range(n))
    if family == "random":
        rng.shuffle(order)
        return order
    if family == "early_edge_bias":
        edges = list(range(0, max(1, n // 4))) + list(range(n - max(1, n // 4), n))
        middle = [x for x in order if x not in set(edges)]
        rng.shuffle(edges)
        rng.shuffle(middle)
        return edges + middle
    if family == "early_center_bias":
        center = list(range(n // 4, n - n // 4))
        outside = [x for x in order if x not in set(center)]
        rng.shuffle(center)
        rng.shuffle(outside)
        return center + outside
    if family == "alternating_edge_center":
        e = edge_in(n)
        c = center_out(n)
        return interleave_orders(e, c, 1)
    if family == "parity_balanced":
        evens = list(range(0, n, 2))
        odds = list(range(1, n, 2))
        rng.shuffle(evens)
        rng.shuffle(odds)
        return interleave_orders(evens, odds, 1)
    if family == "high_jump":
        return regular_jump(n, rng.randrange(n), max(1, n // 2 - 1), rng.choice([-1, 1]))
    if family == "low_jump":
        start = rng.randrange(n)
        return list(range(start, n)) + list(range(0, start))
    rng.shuffle(order)
    return order


def max_unvisited_gap_stats(order: list[int], n: int) -> tuple[float, float]:
    visited: set[int] = set()
    gaps: list[int] = []
    full = set(range(n))
    for track in order:
        visited.add(track)
        unvisited = sorted(full - visited)
        if not unvisited:
            gaps.append(0)
            continue
        current = 1
        max_gap = 1
        for prev, curr in zip(unvisited, unvisited[1:]):
            if curr == prev + 1:
                current += 1
                max_gap = max(max_gap, current)
            else:
                current = 1
        gaps.append(max_gap)
    return mean([gap / n for gap in gaps]), max([gap / n for gap in gaps], default=0.0)


def order_features(order: list[int], n: int) -> dict[str, Any]:
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
    gap_mean, gap_max = max_unvisited_gap_stats(order, n)
    return {
        "n": n,
        "feature_n": n,
        "first_track": order[0],
        "last_track": order[-1],
        "center_track_index": center,
        "first_track_norm": safe_divide(order[0], n - 1),
        "last_track_norm": safe_divide(order[-1], n - 1),
        "mean_jump": mean(jumps),
        "median_jump": median(jumps),
        "max_jump": max(jumps, default=0),
        "min_jump": min(jumps, default=0),
        "std_jump": std([float(x) for x in jumps]),
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


def load_inputs() -> tuple[list[dict[str, str]], dict[str, Any], list[dict[str, str]]]:
    rows = read_csv(SURROGATE_TABLE)
    feature_defs = json.loads(RUN11_FEATURE_DEFS.read_text(encoding="utf-8"))
    best = read_csv(RUN11_BEST)
    return rows, feature_defs, best


def validate_inputs(rows: list[dict[str, str]], best_rows: list[dict[str, str]], feature_defs: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    counts = Counter(parse_int(row.get("n")) for row in rows)
    required = ["target_reward_mean_all", "target_reward_v01_u2_primary", "target_u2_score_rank", "strategy_name", "scan_order_json"]
    for col in required:
        if rows and col not in rows[0]:
            errors.append(f"Missing required column: {col}")
    for row in rows:
        if not math.isfinite(parse_float(row.get("target_reward_mean_all"))):
            errors.append(f"Missing target_reward_mean_all for {row.get('strategy_name')}")
    best_primary = next((row for row in best_rows if row.get("target") == PRIMARY_TARGET), None)
    if not best_primary:
        errors.append("Missing run11 best primary configuration")
    else:
        if best_primary.get("model_name") != "ExtraTreesRegressor" or best_primary.get("feature_set") != F03:
            errors.append(f"Unexpected best config: {best_primary}")
        if parse_float(best_primary.get("spearman")) <= 0:
            errors.append("Best leave-N-out macro Spearman is not positive")
        if parse_float(best_primary.get("top5_overlap")) < 2:
            errors.append("Best top5 overlap is not useful")
    if F03 not in feature_defs:
        errors.append("F03_family_plus_features missing from feature definitions")
    if len(rows) != 60:
        errors.append(f"Expected 60 rows, found {len(rows)}")
    if sorted(counts) != EXPECTED_N:
        errors.append(f"Expected N values {EXPECTED_N}, found {sorted(counts)}")
    for n in EXPECTED_N:
        if counts[n] != EXPECTED_ROWS_PER_N:
            errors.append(f"Expected {EXPECTED_ROWS_PER_N} rows for N{n}, found {counts[n]}")
    return {
        "verdict": "PASS_RUN12_INPUTS_READY_FOR_OFFLINE_SURROGATE_SCREENING" if not errors else "FAIL_RUN12_INPUTS_INVALID",
        "errors": errors,
        "total_rows": len(rows),
        "per_n_counts": dict(sorted(counts.items())),
        "best_run11_config": best_primary,
    }


def build_matrix(train_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]], spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, list[str]]]:
    numeric = spec["numeric"]
    categorical = spec["categorical"]
    train_parts = [[parse_float(row.get(col), default=0.0) for col in numeric] for row in train_rows]
    cand_parts = [[parse_float(row.get(col), default=0.0) for col in numeric] for row in candidate_rows]
    names = list(numeric)
    categories_by_col: dict[str, list[str]] = {}
    for col in categorical:
        cats = sorted({str(row.get(col, "")) for row in train_rows})
        categories_by_col[col] = cats
        for cat in cats:
            names.append(f"{col}={cat}")
            for idx, row in enumerate(train_rows):
                train_parts[idx].append(1.0 if str(row.get(col, "")) == cat else 0.0)
            for idx, row in enumerate(candidate_rows):
                cand_parts[idx].append(1.0 if str(row.get(col, "")) == cat else 0.0)
    return np.asarray(train_parts, dtype=float), np.asarray(cand_parts, dtype=float), names, categories_by_col


def train_surrogates(rows: list[dict[str, str]], feature_defs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[str], dict[str, list[str]]]:
    from sklearn.ensemble import ExtraTreesRegressor

    spec = feature_defs[F03]
    train_rows: list[dict[str, Any]] = [dict(row) for row in rows]
    x_train, _, feature_names, categories = build_matrix(train_rows, train_rows, spec)
    models: dict[str, Any] = {}
    for target in [PRIMARY_TARGET, *SECONDARY_TARGETS]:
        y = np.asarray([parse_float(row[target]) for row in train_rows], dtype=float)
        model = ExtraTreesRegressor(n_estimators=120, max_depth=4, min_samples_leaf=2, random_state=42, n_jobs=-1)
        model.fit(x_train, y)
        models[target] = model
    return models, spec, feature_names, categories


def generate_candidates_for_n(n: int, teacher_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], int]:
    rng = random.Random(GLOBAL_SEED + n)
    raw: list[dict[str, Any]] = []
    priority = {12: "sanity_small_N_diagnostic", 16: "support_interpolation_diagnostic", 24: "core_unseen_intermediate_N_target", 40: "core_large_N_target"}[n]

    def add(order: list[int], family: str, source: str, method: str, seed: str = "", mutation: str = "", mutation_distance: int | str = "") -> None:
        if validate_order(order, n):
            raw.append(
                {
                    "n": n,
                    "candidate_family": family,
                    "candidate_source": source,
                    "generation_method": method,
                    "seed_strategy": seed,
                    "mutation_type": mutation,
                    "mutation_distance": mutation_distance,
                    "order": order,
                    "priority_n_role": priority,
                    "is_existing_teacher_order": False,
                }
            )

    baselines = [
        ("raster", "raster_left_to_right", raster(n)),
        ("raster", "raster_right_to_left", raster(n, True)),
        ("odd_even", "odd_even_interlaced", odd_even(n)),
        ("center_out", "center_out", center_out(n)),
        ("edge_in", "edge_in_alternating", edge_in(n)),
        ("maximin", "greedy_maximin_distance", greedy_maximin(n)),
        ("block_interleaved", "block_interleaved_quarters", block_interleaved(n)),
        ("center_edge", "center_edge_alternating", center_edge_alternating(n)),
    ]
    for family, method, order in baselines:
        add(order, family, "engineering_baseline", method)
    for jump in range(1, n):
        for start in range(n):
            for direction in (1, -1):
                fam = "regular_jump" if gcd(jump, n) == 1 else "regular_jump_sentinel"
                method = "regular_jump_coprime" if gcd(jump, n) == 1 else "regular_jump_non_coprime_sentinel"
                add(regular_jump(n, start, jump, direction), fam, "engineering_baseline", method)
    for mode in ["u2_first", "peeq_aware", "surfaceT_aware", "diversity_preserved", "regular_jump_seed", "maximin_seed", "mutated_topology"]:
        add(method_c_like(n, mode), "method_c_inspired", "method_c_inspired", f"method_c_{mode}_like")

    sorted_teacher = sorted(teacher_rows, key=lambda row: parse_float(row.get("target_reward_mean_all")), reverse=True)
    seed_orders = [(row["strategy_name"], parse_order(row["scan_order_json"])) for row in sorted_teacher[:8]]
    mutations = [
        "swap_two_positions",
        "reverse_short_segment",
        "rotate_sequence",
        "swap_early_late_blocks",
        "perturb_first_quarter",
        "perturb_last_quarter",
        "parity_preserving_swap",
        "edge_center_swap",
    ]
    for seed_name, seed_order in seed_orders:
        for mutation in mutations:
            for _ in range(24):
                mutated = seed_order[:]
                steps = rng.randint(1, 3)
                for _step in range(steps):
                    mutated = mutate_order(mutated, rng, mutation, n)
                add(mutated, "mutation", "surrogate_mutation", f"mutate_{mutation}", seed_name, mutation, steps)
    for (name_a, order_a), (name_b, order_b) in zip(seed_orders, reversed(seed_orders)):
        for mode in ("half", "quarter", "alternating_blocks"):
            add(crossover(order_a, order_b, mode), "crossover", "surrogate_crossover", f"crossover_{mode}", f"{name_a}+{name_b}")
            add(crossover(order_b, order_a, mode), "crossover", "surrogate_crossover", f"crossover_{mode}", f"{name_b}+{name_a}")
    for family in ["anti_odd_even_novelty_like", "zero_shot_or_proxy_best_like", "graph_pointer_diversity_like", "graph_pointer_high_dispersion_like", "graph_pointer_surfaceT_like"]:
        for _ in range(80):
            if "anti_odd_even" in family:
                order = mutate_order(edge_in(n), rng, "parity_preserving_swap", n)
            elif "zero_shot" in family:
                order = mutate_order(method_c_like(n, "u2_first"), rng, "reverse_short_segment", n)
            elif "high_dispersion" in family:
                order = regular_jump(n, rng.randrange(n), rng.randrange(max(2, n // 3), n), rng.choice([-1, 1]))
            elif "surfaceT" in family:
                order = mutate_order(center_edge_alternating(n), rng, "edge_center_swap", n)
            else:
                order = random_family_order(n, rng, "parity_balanced")
            add(order, "graph_pointer_inspired_proxy", "proxy_inspired_not_trained_rl", family)
    random_families = ["random", "early_edge_bias", "early_center_bias", "alternating_edge_center", "parity_balanced", "high_jump", "low_jump"]
    while len(raw) < TARGET_RAW_PER_N:
        fam = rng.choice(random_families)
        add(random_family_order(n, rng, fam), "random_diversity", "random_quasi_random", fam)
    if len(raw) < MIN_RAW_PER_N:
        raise RuntimeError(f"Only generated {len(raw)} raw candidates for N{n}")
    return raw, len(raw)


def dedup_candidates(raw: list[dict[str, Any]], teacher_hashes: dict[int, set[str]], teacher_rows_by_n: dict[int, list[dict[str, str]]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: dict[int, set[str]] = defaultdict(set)
    for row in raw:
        n = row["n"]
        h = order_hash(row["order"])
        if h in seen[n]:
            continue
        seen[n].add(h)
        row = dict(row)
        row["order_hash"] = h
        row["is_existing_teacher_order"] = h in teacher_hashes[n]
        if row["is_existing_teacher_order"]:
            continue
        deduped.append(row)
    for n, rows in teacher_rows_by_n.items():
        for teacher in rows:
            order = parse_order(teacher["scan_order_json"])
            deduped.append(
                {
                    "n": n,
                    "candidate_family": teacher.get("strategy_family", "existing_reference"),
                    "candidate_source": "existing_reference",
                    "generation_method": "existing_teacher_labelled_reference",
                    "seed_strategy": teacher["strategy_name"],
                    "mutation_type": "",
                    "mutation_distance": "",
                    "order": order,
                    "priority_n_role": {12: "sanity_small_N_diagnostic", 16: "support_interpolation_diagnostic", 24: "core_unseen_intermediate_N_target", 40: "core_large_N_target"}[n],
                    "is_existing_teacher_order": True,
                    "order_hash": order_hash(order),
                    "existing_reward_mean_all": parse_float(teacher["target_reward_mean_all"]),
                }
            )
    for idx, row in enumerate(deduped, start=1):
        n = row["n"]
        row["candidate_id"] = f"RUN12_N{n}_C{idx:05d}"
        row["strategy_name"] = f"N{n}_{row['candidate_id']}_{row['generation_method']}"
        row["order_json"] = json.dumps(row["order"], separators=(",", ":"))
        row["order_compact"] = ",".join(str(x) for x in row["order"])
        row.update(order_features(row["order"], n))
        row["strategy_family"] = row["candidate_family"]
        row["candidate_group"] = "engineering_baseline" if row["candidate_source"] in {"engineering_baseline", "method_c_inspired"} else "proxy_fallback_policy"
        row["policy_source"] = "engineering_baseline" if row["candidate_group"] == "engineering_baseline" else "proxy_policy"
    return deduped


def score_candidates(
    train_rows: list[dict[str, str]],
    candidates: list[dict[str, Any]],
    models: dict[str, Any],
    spec: dict[str, Any],
) -> list[dict[str, Any]]:
    _, x_cand, _names, _categories = build_matrix([dict(row) for row in train_rows], candidates, spec)
    for target, model in models.items():
        pred = np.asarray(model.predict(x_cand), dtype=float)
        short = {
            PRIMARY_TARGET: "pred_reward_mean_all",
            "target_reward_v01_u2_primary": "pred_reward_v01_u2_primary",
            "target_u2_score_rank": "pred_u2_score_rank",
            "target_peeq_score_rank": "pred_peeq_score_rank",
            "target_surfaceT_score_rank": "pred_surfaceT_score_rank",
        }[target]
        for row, value in zip(candidates, pred):
            row[short] = float(value)
        if target == PRIMARY_TARGET and hasattr(model, "estimators_"):
            tree_preds = np.vstack([tree.predict(x_cand) for tree in model.estimators_])
            means = tree_preds.mean(axis=0)
            stds = tree_preds.std(axis=0)
            for row, m, s in zip(candidates, means, stds):
                row["tree_prediction_mean"] = float(m)
                row["pred_uncertainty_std"] = float(s)
    teacher_orders_by_n = defaultdict(list)
    for row in train_rows:
        teacher_orders_by_n[parse_int(row["n"])].append((row["strategy_name"], parse_order(row["scan_order_json"])))
    for n in EXPECTED_N:
        group = [row for row in candidates if row["n"] == n]
        sorted_group = sorted(group, key=lambda row: row["pred_reward_mean_all"], reverse=True)
        denom = max(1, len(sorted_group) - 1)
        uncertainties = sorted([row.get("pred_uncertainty_std", 0.0) for row in group])
        uq = uncertainties[int(math.ceil(0.75 * len(uncertainties))) - 1] if uncertainties else 0.0
        for rank, row in enumerate(sorted_group, start=1):
            row["pred_rank_within_n"] = rank
            row["pred_percentile_within_n"] = 1.0 - safe_divide(rank - 1, denom)
            row["ensemble_disagreement_flag"] = row.get("pred_uncertainty_std", 0.0) >= uq
            nearest_name, nearest_distance = nearest_teacher(row["order"], teacher_orders_by_n[n])
            row["nearest_existing_strategy"] = nearest_name
            row["novelty_distance_to_nearest_existing"] = nearest_distance
            row["diversity_score"] = nearest_distance
    return candidates


def kendall_distance(a: list[int], b: list[int]) -> float:
    pos = {value: idx for idx, value in enumerate(b)}
    mapped = [pos[value] for value in a]
    inv = 0
    total = len(mapped) * (len(mapped) - 1) // 2
    for i in range(len(mapped)):
        for j in range(i + 1, len(mapped)):
            if mapped[i] > mapped[j]:
                inv += 1
    return safe_divide(inv, total)


def nearest_teacher(order: list[int], teachers: list[tuple[str, list[int]]]) -> tuple[str, float]:
    best_name = ""
    best_dist = math.inf
    for name, teacher_order in teachers:
        dist = kendall_distance(order, teacher_order)
        if dist < best_dist:
            best_name, best_dist = name, dist
    return best_name, best_dist


def select_shortlists(scored: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shortlist: list[dict[str, Any]] = []
    for n in EXPECTED_N:
        group = [row for row in scored if row["n"] == n and not row["is_existing_teacher_order"]]
        selected: dict[str, dict[str, Any]] = {}

        def add_bucket(bucket: str, rows: list[dict[str, Any]], limit: int) -> None:
            for row in rows:
                if len([r for r in selected.values() if r["selection_bucket"] == bucket]) >= limit:
                    break
                if row["order_hash"] in selected:
                    continue
                new = dict(row)
                new["selection_bucket"] = bucket
                selected[row["order_hash"]] = new

        add_bucket("surrogate_top", sorted(group, key=lambda r: r["pred_reward_mean_all"], reverse=True), 10)
        add_bucket("U2_primary_top", sorted(group, key=lambda r: (r.get("pred_u2_score_rank", 0), r.get("pred_reward_v01_u2_primary", 0)), reverse=True), 5)
        diverse = sorted([r for r in group if r["novelty_distance_to_nearest_existing"] >= 0.25], key=lambda r: (r["pred_reward_mean_all"], r["diversity_score"]), reverse=True)
        add_bucket("diversity_top", diverse, 5)
        method_c = sorted([r for r in group if r["candidate_family"] == "method_c_inspired"], key=lambda r: r["pred_reward_mean_all"], reverse=True)
        add_bucket("method_c_inspired", method_c, 5)
        uncertainty = sorted([r for r in group if r["pred_rank_within_n"] <= max(200, len(group) // 4)], key=lambda r: r.get("pred_uncertainty_std", 0.0), reverse=True)
        add_bucket("uncertainty_sentinel", uncertainty, 3)
        controls = sorted([r for r in group if r["pred_rank_within_n"] > len(group) * 0.75 and r["novelty_distance_to_nearest_existing"] >= 0.25], key=lambda r: (r["diversity_score"], -r["pred_reward_mean_all"]), reverse=True)
        add_bucket("negative_control_sentinel", controls, 2)
        rows = sorted(selected.values(), key=lambda r: (bucket_order(r["selection_bucket"]), r["pred_rank_within_n"]))
        for rank, row in enumerate(rows[:24], start=1):
            row["shortlist_rank_within_n"] = rank
            shortlist.append(row)
    batch20: list[dict[str, Any]] = []
    for n in EXPECTED_N:
        rows = [row for row in shortlist if row["n"] == n]
        preferred = []
        for bucket in ["surrogate_top", "diversity_top", "method_c_inspired", "uncertainty_sentinel", "negative_control_sentinel"]:
            bucket_rows = sorted([row for row in rows if row["selection_bucket"] == bucket], key=lambda r: r["pred_rank_within_n"])
            if bucket_rows:
                preferred.append(bucket_rows[0])
        fill = [row for row in sorted(rows, key=lambda r: r["pred_rank_within_n"]) if row["order_hash"] not in {p["order_hash"] for p in preferred}]
        chosen = (preferred + fill)[:5]
        for rank, row in enumerate(chosen, start=1):
            out = dict(row)
            out["teacher_batch20_rank_within_n"] = rank
            batch20.append(out)
    return shortlist, batch20


def bucket_order(bucket: str) -> int:
    return {
        "surrogate_top": 0,
        "U2_primary_top": 1,
        "diversity_top": 2,
        "method_c_inspired": 3,
        "uncertainty_sentinel": 4,
        "negative_control_sentinel": 5,
    }.get(bucket, 99)


def predicted_improvement(scored: list[dict[str, Any]], teacher_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n in EXPECTED_N:
        teacher_group = [row for row in teacher_rows if parse_int(row["n"]) == n]
        best_teacher = max(teacher_group, key=lambda row: parse_float(row["target_reward_mean_all"]))
        best_new = max([row for row in scored if row["n"] == n and not row["is_existing_teacher_order"]], key=lambda row: row["pred_reward_mean_all"])
        rows.append(
            {
                "n": n,
                "best_existing_strategy": best_teacher["strategy_name"],
                "best_existing_teacher_reward_mean_all": parse_float(best_teacher["target_reward_mean_all"]),
                "top_predicted_new_candidate_id": best_new["candidate_id"],
                "top_predicted_new_family": best_new["candidate_family"],
                "top_predicted_new_reward_mean_all": best_new["pred_reward_mean_all"],
                "surrogate_predicted_exceeds_existing_teacher_reward": best_new["pred_reward_mean_all"] > parse_float(best_teacher["target_reward_mean_all"]),
                "nearest_existing_strategy": best_new["nearest_existing_strategy"],
                "novelty_distance": best_new["novelty_distance_to_nearest_existing"],
                "note": "surrogate-only hypothetical; teacher validation required",
            }
        )
    return rows


def diagnostics(raw_counts: dict[int, int], dedup_counts: dict[int, int], scored: list[dict[str, Any]], shortlist: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n in EXPECTED_N:
        group = [r for r in scored if r["n"] == n]
        new = [r for r in group if not r["is_existing_teacher_order"]]
        rows.append({"diagnostic": "raw_generated_count", "n": n, "value": raw_counts[n]})
        rows.append({"diagnostic": "deduplicated_new_count", "n": n, "value": dedup_counts[n]})
        rows.append({"diagnostic": "scored_count_including_references", "n": n, "value": len(group)})
        rows.append({"diagnostic": "pred_reward_mean_all_mean", "n": n, "value": mean([r["pred_reward_mean_all"] for r in new])})
        rows.append({"diagnostic": "pred_uncertainty_std_mean", "n": n, "value": mean([r.get("pred_uncertainty_std", 0.0) for r in new])})
        rows.append({"diagnostic": "novelty_distance_mean", "n": n, "value": mean([r["novelty_distance_to_nearest_existing"] for r in new])})
        for family, count in Counter(r["candidate_family"] for r in new).items():
            rows.append({"diagnostic": "candidate_count_by_family", "n": n, "category": family, "value": count})
        for bucket, count in Counter(r["selection_bucket"] for r in shortlist if r["n"] == n).items():
            rows.append({"diagnostic": "shortlist_count_by_bucket", "n": n, "category": bucket, "value": count})
    return rows


def cap_deduped_pool(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    capped: list[dict[str, Any]] = []
    for n in EXPECTED_N:
        refs = [row for row in rows if row["n"] == n and row["is_existing_teacher_order"]]
        new = [row for row in rows if row["n"] == n and not row["is_existing_teacher_order"]]
        priority = {"method_c_inspired": 0, "mutation": 1, "crossover": 2, "graph_pointer_inspired_proxy": 3, "regular_jump": 4, "random_diversity": 5, "regular_jump_sentinel": 6}
        new_sorted = sorted(new, key=lambda row: (priority.get(row["candidate_family"], 9), row["candidate_id"]))
        capped.extend(new_sorted[:MAX_DEDUP_NEW_PER_N])
        capped.extend(refs)
    return capped


def maybe_plot(scored: list[dict[str, Any]], shortlist: list[dict[str, Any]]) -> list[str]:
    written: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return [f"PLOTTING_SKIPPED: {exc}"]
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for n in EXPECTED_N:
        rows = [r for r in scored if r["n"] == n and not r["is_existing_teacher_order"]]
        plt.figure(figsize=(6, 4))
        plt.hist([r["pred_reward_mean_all"] for r in rows], bins=30)
        plt.title(f"N{n} predicted reward distribution")
        plt.xlabel("pred_reward_mean_all")
        plt.ylabel("count")
        path = FIGURE_DIR / f"N{n}_predicted_reward_histogram.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        written.append(str(path))
    plt.figure(figsize=(6, 4))
    plt.scatter([r.get("pred_uncertainty_std", 0.0) for r in scored if not r["is_existing_teacher_order"]], [r["pred_reward_mean_all"] for r in scored if not r["is_existing_teacher_order"]], s=4)
    plt.xlabel("pred_uncertainty_std")
    plt.ylabel("pred_reward_mean_all")
    path = FIGURE_DIR / "predicted_reward_vs_uncertainty.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    written.append(str(path))
    plt.figure(figsize=(8, 4))
    counts = Counter(r["candidate_family"] for r in shortlist)
    plt.bar(counts.keys(), counts.values())
    plt.xticks(rotation=55, ha="right")
    plt.ylabel("shortlist count")
    path = FIGURE_DIR / "shortlist_family_composition.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    written.append(str(path))
    return written


def project_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    drop = {"order"}
    return [{k: v for k, v in row.items() if k not in drop} for row in rows]


def pool_projection(rows: list[dict[str, Any]], scored: bool = False, selection: bool = False) -> list[dict[str, Any]]:
    base_cols = [
        "n",
        "candidate_id",
        "strategy_name",
        "candidate_family",
        "candidate_source",
        "generation_method",
        "seed_strategy",
        "mutation_type",
        "mutation_distance",
        "order_json",
        "order_compact",
        "order_hash",
        "is_existing_teacher_order",
        "priority_n_role",
        "existing_reward_mean_all",
    ]
    scored_cols = [
        "pred_reward_mean_all",
        "pred_reward_v01_u2_primary",
        "pred_u2_score_rank",
        "pred_peeq_score_rank",
        "pred_surfaceT_score_rank",
        "tree_prediction_mean",
        "pred_uncertainty_std",
        "pred_rank_within_n",
        "pred_percentile_within_n",
        "ensemble_disagreement_flag",
        "nearest_existing_strategy",
        "novelty_distance_to_nearest_existing",
        "diversity_score",
    ]
    selection_cols = ["selection_bucket", "shortlist_rank_within_n", "teacher_batch20_rank_within_n"]
    cols = base_cols + (scored_cols if scored else []) + (selection_cols if selection else [])
    return [{col: row.get(col, "") for col in cols} for row in rows]


def write_claim_boundary(md: Path, js: Path) -> None:
    safe = [
        "Run12 generated and scored offline variable-N scan-order candidates using the run11 diagnostic surrogate.",
        "Run12 produced deduplicated, diversity-preserved candidate pools and future teacher-validation shortlists.",
        "Candidates are prioritized using within-N predicted normalized reward.",
        "Run12 supports human review and future active-learning/teacher-validation planning.",
    ]
    unsafe = [
        "Do not claim candidate physical superiority.",
        "Do not claim teacher validation.",
        "Do not claim trained variable-N RL policy success.",
        "Do not claim arbitrary-N generalization.",
        "Do not claim surrogate predictions are ground truth.",
        "Do not claim Method-C or graph-pointer-inspired candidates are final optima.",
        "Do not claim generated graph-pointer-inspired candidates are trained RL outputs.",
    ]
    lines = ["# Run 12 Claim Boundary", "", "## Safe Claims", *[f"- {x}" for x in safe], "", "## Unsafe Claims", *[f"- {x}" for x in unsafe]]
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(js, {"verdict": "RUN12_OFFLINE_SURROGATE_SCREENING_ONLY_NO_TEACHER_VALIDATION", "safe_claims": safe, "unsafe_claims": unsafe})


def update_run_index(verdict: str) -> None:
    if not RUN_INDEX_PATH.exists():
        return
    entry = (
        "| run_12 | Offline surrogate-screened candidate generation | Generate, score, deduplicate, and shortlist offline variable-N scan-order candidates with the run11 diagnostic surrogate. | "
        "`scripts/stage3/run_12_generate_surrogate_screened_candidates.py` | "
        "`docs/stage3/runs/run_12_offline_surrogate_screened_candidate_generation/RUN_12_OFFLINE_SURROGATE_SCREENED_CANDIDATE_GENERATION_REPORT.md` | "
        "`outputs/stage3_run_12_offline_surrogate_screened_candidate_generation/` | "
        f"`{verdict}` | No Abaqus, no ODB, no CAE/INP/JNL generation, no abqjobpilot, no final RL policy training, no commit/push. Next: run13 small teacher-validation handoff only after user approval. |"
    )
    lines = RUN_INDEX_PATH.read_text(encoding="utf-8").splitlines()
    for idx, line in enumerate(lines):
        if line.startswith("| run_12 | Offline surrogate-screened candidate generation |"):
            lines[idx] = entry
            break
    else:
        lines.append(entry)
    RUN_INDEX_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(validation: dict[str, Any], model_meta: dict[str, Any], counts: dict[str, Any], top_by_n: dict[str, Any], outputs: list[str]) -> None:
    lines = [
        "# Stage 3 Run 12 - Offline Surrogate-Screened Candidate Generation",
        "",
        "## Purpose",
        "Generate and screen new variable-N scan-order candidates offline using the run11 diagnostic surrogate.",
        "",
        "## Inputs",
        f"- `{SURROGATE_TABLE}`",
        f"- `{RUN11_BEST}`",
        f"- `{RUN11_FEATURE_DEFS}`",
        f"- `{RUN06_CANDIDATES}`",
        "",
        "## Run11 Surrogate Basis",
        f"- Model: `{model_meta['primary_model']}`",
        f"- Feature set: `{model_meta['feature_set']}`",
        f"- Target: `{model_meta['primary_target']}`",
        "- Refit on all 60 run10 rows for offline scoring only.",
        "",
        "## Candidate Generation Methods",
        "- Engineering baselines, regular-jump sweeps, Method-C-inspired heuristics, seed mutations, crossovers, random/quasi-random diversity, and graph-pointer-inspired proxy candidates.",
        "- Graph-pointer-inspired candidates are proxy/inspired orders, not trained RL outputs.",
        "",
        "## Candidate Validation and Deduplication",
        f"- Raw generated counts: {counts['raw_generated_per_n']}",
        f"- Deduplicated new counts: {counts['deduplicated_new_per_n']}",
        "- Existing teacher-labelled orders are retained as reference calibration rows only.",
        "",
        "## Surrogate Scoring Method",
        "- ExtraTrees predictions are within-N ranked by predicted normalized reward.",
        "- Tree prediction standard deviation is used as an uncertainty proxy.",
        "- Kendall distance to the nearest teacher-labelled order is used as a novelty/diversity proxy.",
        "",
        "## Candidate Pool Summary",
        f"- Scored candidate count including references: {counts['scored_total']}",
        "",
        "## Final Shortlist Per N",
        f"- Shortlist counts: {counts['shortlist_per_n']}",
        "",
        "## Recommended Future Teacher Batch20",
        f"- Batch20 counts: {counts['batch20_per_n']}",
        "",
        "## Predicted Improvement vs Existing Teacher-Labelled Cases",
    ]
    for n, row in top_by_n.items():
        lines.append(f"- N{n}: top surrogate candidate `{row['top_predicted_new_candidate_id']}` predicted {row['top_predicted_new_reward_mean_all']:.4f}; existing best `{row['best_existing_strategy']}` teacher reward {row['best_existing_teacher_reward_mean_all']:.4f}; exceeds flag `{row['surrogate_predicted_exceeds_existing_teacher_reward']}`. Surrogate-only.")
    lines += [
        "",
        "## Diversity and Uncertainty Diagnostics",
        "- Shortlist includes surrogate-top, U2-primary, diversity, Method-C-inspired, uncertainty sentinel, and negative/control sentinel buckets.",
        "",
        "## Claim Boundary",
        "- Candidate rankings are predictions only.",
        "- No physical validation, Abaqus execution, CAE/INP/JNL generation, or final RL training occurred.",
        "",
        "## Output Files",
        *[f"- `{path}`" for path in outputs],
        "",
        "## Recommended Run13",
        "Do not immediately run all run12 candidates. First create a small controlled teacher-validation handoff package, likely 20 cases total: 5 per N, or 4 for N12/N16 and 6 for N24/N40 if prioritizing core larger N. Run13 should create CAE handoff artifacts only after user approval.",
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
    train_rows, feature_defs, best_rows = load_inputs()
    validation = validate_inputs(train_rows, best_rows, feature_defs)
    validation_path = OUTPUT_DIR / "run12_input_validation_summary.json"
    write_json(validation_path, validation)
    if validation["verdict"].startswith("FAIL"):
        print(validation["verdict"])
        print(json.dumps(validation, indent=2))
        return 2
    models, spec, feature_names, categories = train_surrogates(train_rows, feature_defs)
    model_meta = {
        "primary_model": "ExtraTreesRegressor",
        "secondary_models": list(SECONDARY_TARGETS),
        "feature_set": F03,
        "primary_target": PRIMARY_TARGET,
        "random_state": 42,
        "fit_rows": len(train_rows),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "categorical_training_levels": categories,
        "model_pickle_saved": False,
    }
    model_meta_path = OUTPUT_DIR / "run12_surrogate_model_metadata.json"
    write_json(model_meta_path, model_meta)

    teacher_by_n = {n: [row for row in train_rows if parse_int(row["n"]) == n] for n in EXPECTED_N}
    teacher_hashes = {n: {order_hash(parse_order(row["scan_order_json"])) for row in rows} for n, rows in teacher_by_n.items()}
    raw_counts: dict[int, int] = {}
    all_raw: list[dict[str, Any]] = []
    for n in EXPECTED_N:
        raw_n, raw_count = generate_candidates_for_n(n, teacher_by_n[n])
        raw_counts[n] = raw_count
        all_raw.extend(raw_n)
    deduped = dedup_candidates(all_raw, teacher_hashes, teacher_by_n)
    dedup_new_total_counts = {n: sum(1 for row in deduped if row["n"] == n and not row["is_existing_teacher_order"]) for n in EXPECTED_N}
    deduped = cap_deduped_pool(deduped)
    dedup_new_counts = {n: sum(1 for row in deduped if row["n"] == n and not row["is_existing_teacher_order"]) for n in EXPECTED_N}
    scored = score_candidates(train_rows, deduped, models, spec)
    shortlist, batch20 = select_shortlists(scored)
    improvement = predicted_improvement(scored, train_rows)
    diag = diagnostics(raw_counts, dedup_new_counts, scored, shortlist)
    plots = maybe_plot(scored, shortlist)
    claim_md = OUTPUT_DIR / "run12_claim_boundary.md"
    claim_json = OUTPUT_DIR / "run12_claim_boundary.json"
    write_claim_boundary(claim_md, claim_json)

    unscored_csv = OUTPUT_DIR / "run12_candidate_pool_unscored.csv"
    unscored_json = OUTPUT_DIR / "run12_candidate_pool_unscored.json"
    scored_csv = OUTPUT_DIR / "run12_candidate_pool_scored.csv"
    scored_json = OUTPUT_DIR / "run12_candidate_pool_scored.json"
    shortlist_csv = OUTPUT_DIR / "run12_candidate_shortlist_per_N.csv"
    shortlist_json = OUTPUT_DIR / "run12_candidate_shortlist_per_N.json"
    batch_csv = OUTPUT_DIR / "run12_recommended_future_teacher_batch20.csv"
    batch_json = OUTPUT_DIR / "run12_recommended_future_teacher_batch20.json"
    improvement_csv = OUTPUT_DIR / "run12_predicted_improvement_vs_existing.csv"
    improvement_json = OUTPUT_DIR / "run12_predicted_improvement_vs_existing.json"
    diag_csv = OUTPUT_DIR / "run12_candidate_generation_diagnostics.csv"
    diag_json = OUTPUT_DIR / "run12_candidate_generation_diagnostics.json"

    unscored = pool_projection(deduped, scored=False)
    scored_projected = pool_projection(scored, scored=True)
    write_csv(unscored_csv, unscored)
    write_table_json(unscored_json, unscored)
    write_csv(scored_csv, scored_projected)
    write_table_json(scored_json, scored_projected)
    write_csv(shortlist_csv, pool_projection(shortlist, scored=True, selection=True))
    write_json(shortlist_json, pool_projection(shortlist, scored=True, selection=True))
    write_csv(batch_csv, pool_projection(batch20, scored=True, selection=True))
    write_json(batch_json, pool_projection(batch20, scored=True, selection=True))
    write_csv(improvement_csv, improvement)
    write_json(improvement_json, improvement)
    write_csv(diag_csv, diag)
    write_json(diag_json, diag)

    outputs = [
        str(validation_path),
        str(model_meta_path),
        str(unscored_csv),
        str(unscored_json),
        str(scored_csv),
        str(scored_json),
        str(shortlist_csv),
        str(shortlist_json),
        str(batch_csv),
        str(batch_json),
        str(improvement_csv),
        str(improvement_json),
        str(diag_csv),
        str(diag_json),
        str(claim_md),
        str(claim_json),
        *[path for path in plots if not path.startswith("PLOTTING_SKIPPED")],
        str(REPORT_PATH),
    ]
    counts = {
        "raw_generated_per_n": raw_counts,
        "deduplicated_new_total_before_cap_per_n": dedup_new_total_counts,
        "deduplicated_new_per_n": dedup_new_counts,
        "scored_total": len(scored),
        "shortlist_per_n": dict(Counter(row["n"] for row in shortlist)),
        "batch20_per_n": dict(Counter(row["n"] for row in batch20)),
    }
    top_by_n = {row["n"]: row for row in improvement}
    write_report(validation, model_meta, counts, top_by_n, outputs)
    update_run_index(validation["verdict"])
    manifest = {
        "run_id": RUN_ID,
        "run_name": RUN_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "branch": git_branch(),
        "script_path": str(Path(__file__).resolve()),
        "input_files": [str(SURROGATE_TABLE), str(FEATURE_TABLE), str(REWARD_DATASET), str(RUN11_DETAILED), str(RUN11_BEST), str(RUN11_PREDICTIONS), str(RUN11_FEATURE_DEFS), str(RUN11_REPORT), str(RUN06_CANDIDATES)],
        "output_files": outputs,
        "candidate_pool_count": len(scored),
        "candidate_pool_new_count": sum(1 for row in scored if not row["is_existing_teacher_order"]),
        "shortlist_count": len(shortlist),
        "teacher_batch20_count": len(batch20),
        "report_path": str(REPORT_PATH),
        "claim_boundary_path": str(claim_md),
        "validation_verdict": validation["verdict"],
        "no_solver_run": True,
        "no_odb_opened": True,
        "no_abqjobpilot_run": True,
        "no_cae_inp_generated": True,
        "no_rl_policy_training": True,
        "no_commit_or_push": True,
    }
    write_json(MANIFEST_PATH, manifest)

    top_pred = {f"N{row['n']}": row["top_predicted_new_candidate_id"] for row in improvement}
    print(validation["verdict"])
    print(f"raw_generated_per_n={raw_counts}")
    print(f"deduplicated_new_per_n={dedup_new_counts}")
    print(f"shortlist_per_n={counts['shortlist_per_n']}")
    print(f"teacher_batch20_count={len(batch20)}")
    print(f"top_predicted_candidate_per_n={top_pred}")
    print(f"report={REPORT_PATH}")
    print(f"manifest={MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
