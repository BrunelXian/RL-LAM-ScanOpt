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
RUN_ID = "run_18_combined80_surrogate_screened_candidate_generation"
RUN_NAME = "combined80-updated offline surrogate-screened candidate generation"

COMBINED80_READY = ROOT / "outputs" / "stage3_run_16_batch20_teacher_metrics_ingestion_and_combined80_ranking" / "combined80_RL_ready_dataset.csv"
COMBINED80_TEACHER = ROOT / "outputs" / "stage3_run_16_batch20_teacher_metrics_ingestion_and_combined80_ranking" / "combined80_teacher_dataset.csv"
RUN17_FEATURES = ROOT / "outputs" / "stage3_run_17_combined80_surrogate_reward_model_validation_update" / "combined80_scan_order_features.csv"
RUN17_DETAILED = ROOT / "outputs" / "stage3_run_17_combined80_surrogate_reward_model_validation_update" / "combined80_surrogate_validation_results_detailed.csv"
RUN17_BEST = ROOT / "outputs" / "stage3_run_17_combined80_surrogate_reward_model_validation_update" / "combined80_best_surrogate_configurations.csv"
RUN17_FEATURE_DEFS = ROOT / "outputs" / "stage3_run_17_combined80_surrogate_reward_model_validation_update" / "run17_feature_set_definitions.json"
RUN17_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_17_combined80_surrogate_reward_model_validation_update" / "RUN_17_COMBINED80_SURROGATE_REWARD_MODEL_VALIDATION_UPDATE_REPORT.md"
RUN12_SCORED = ROOT / "outputs" / "stage3_run_12_offline_surrogate_screened_candidate_generation" / "run12_candidate_pool_scored.csv"
RUN12_BATCH20 = ROOT / "outputs" / "stage3_run_12_offline_surrogate_screened_candidate_generation" / "run12_recommended_future_teacher_batch20.csv"
RUN16_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_16_batch20_teacher_metrics_ingestion_and_combined80_ranking" / "RUN_16_BATCH20_TEACHER_METRICS_INGESTION_AND_COMBINED80_RANKING_REPORT.md"

OUTPUT_DIR = ROOT / "outputs" / "stage3_run_18_combined80_surrogate_screened_candidate_generation"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = ROOT / "docs" / "stage3" / "runs" / RUN_ID
REPORT_PATH = REPORT_DIR / "RUN_18_COMBINED80_SURROGATE_SCREENED_CANDIDATE_GENERATION_REPORT.md"
MANIFEST_PATH = ROOT / "artifacts" / "manifests" / "stage3_run_18_manifest.json"
RUN_INDEX_PATH = ROOT / "docs" / "stage3" / "STAGE3_RUN_INDEX.md"

EXPECTED_N = [12, 16, 24, 40]
TARGET_DEDUP_PER_N = {12: 1000, 16: 1000, 24: 2000, 40: 2500}
MIN_DEDUP_PER_N = {12: 800, 16: 800, 24: 1500, 40: 1500}
SHORTLIST_COUNTS = {12: 8, 16: 8, 24: 16, 40: 16}
BATCH28_COUNTS = {12: 4, 16: 4, 24: 10, 40: 10}
BATCH24_COUNTS = {12: 3, 16: 3, 24: 9, 40: 9}
GLOBAL_SEED = 42
PRIMARY_TARGET = "target_reward_combined80_u2_primary"
SECONDARY_TARGETS = [
    "target_u2_score_combined80_rank",
    "target_peeq_score_combined80_rank",
    "target_surfaceT_score_combined80_rank",
    "target_mises_score_combined80_rank",
]
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


def load_combined80() -> list[dict[str, Any]]:
    rows = read_csv(COMBINED80_READY)
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
    if len(rows) != 80:
        errors.append(f"Expected 80 rows, found {len(rows)}")
    if sorted(counts) != EXPECTED_N:
        errors.append(f"Unexpected N values: {sorted(counts)}")
    for n in EXPECTED_N:
        if counts.get(n) != 20:
            errors.append(f"Expected 20 rows for N{n}, found {counts.get(n, 0)}")
    required = [PRIMARY_TARGET, "target_u2_score_combined80_rank", "order_json"]
    for col in required:
        if any(str(row.get(col, "")).strip() == "" for row in rows):
            errors.append(f"Missing required values in {col}")
    for row in rows:
        n = row["n"]
        order = row.get("order")
        if order is None or not validate_order(order, n):
            errors.append(f"Invalid scan order for {row.get('strategy_name')}")
    if RUN17_BEST.exists():
        best_rows = read_csv(RUN17_BEST)
        primary = next((r for r in best_rows if r.get("target") == PRIMARY_TARGET), None)
        if not primary:
            errors.append("Run17 primary best config row was not found")
        else:
            if primary.get("model_name") != "ExtraTreesRegressor":
                errors.append(f"Unexpected run17 model: {primary.get('model_name')}")
            if primary.get("feature_set") != "F01_basic_order":
                errors.append(f"Unexpected run17 feature set: {primary.get('feature_set')}")
            if parse_float(primary.get("spearman"), 0.0) < 0.8:
                warnings.append(f"Run17 macro Spearman below 0.8: {primary.get('spearman')}")
    else:
        errors.append(f"Missing run17 best configurations: {RUN17_BEST}")
    verdict = "PASS_RUN18_INPUTS_READY_FOR_COMBINED80_SURROGATE_SCREENING" if not errors else "FAIL_RUN18_INPUTS_INVALID"
    summary = {
        "verdict": verdict,
        "row_count": len(rows),
        "per_n_counts": dict(sorted(counts.items())),
        "required_features": F01_FEATURES,
        "errors": errors,
        "warnings": warnings,
    }
    write_json(OUTPUT_DIR / "run18_input_validation_summary.json", summary)
    return summary


def train_surrogates(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    from sklearn.ensemble import ExtraTreesRegressor

    x = np.array([[float(scan_order_features(row["order"], row["n"])[feature]) for feature in F01_FEATURES] for row in rows], dtype=float)
    models: dict[str, Any] = {}
    for target in [PRIMARY_TARGET] + SECONDARY_TARGETS:
        y = np.array([parse_float(row.get(target)) for row in rows], dtype=float)
        model = ExtraTreesRegressor(
            n_estimators=240,
            max_depth=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(x, y)
        models[target] = model
    metadata = {
        "model_family": "ExtraTreesRegressor",
        "feature_set": "F01_basic_order",
        "feature_names": F01_FEATURES,
        "primary_target": PRIMARY_TARGET,
        "secondary_targets": SECONDARY_TARGETS,
        "training_rows": len(rows),
        "random_state": 42,
        "n_estimators": 240,
        "max_depth": 5,
        "min_samples_leaf": 2,
        "model_label": "combined80 offline diagnostic surrogate",
        "not_final_or_deployed": True,
    }
    write_json(OUTPUT_DIR / "run18_surrogate_model_metadata.json", metadata)
    return models, metadata


def combined80_refs(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    refs: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        refs[row["n"]].append({"strategy_name": row.get("strategy_name", ""), "order": row["order"], "order_hash": row["order_hash"], "target": row["target"]})
    return refs


def run12_hashes() -> dict[int, set[str]]:
    result: dict[int, set[str]] = defaultdict(set)
    for path in [RUN12_SCORED, RUN12_BATCH20]:
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


def seed_orders_from_combined80(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    seeds: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for n in EXPECTED_N:
        group = [row for row in rows if row["n"] == n]
        by_reward = sorted(group, key=lambda r: parse_float(r.get("target"), -1.0), reverse=True)[:5]
        by_u2 = sorted(group, key=lambda r: parse_float(r.get("target_u2_score_combined80_rank"), -1.0), reverse=True)[:4]
        named = [r for r in group if "method_c" in str(r.get("strategy_name", "")).lower() or "diversity_top" in str(r.get("strategy_name", "")).lower()]
        seen: set[str] = set()
        for row in by_reward + by_u2 + named:
            digest = row["order_hash"]
            if digest not in seen:
                seeds[n].append(row)
                seen.add(digest)
    return seeds


def generate_candidates(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng_global = random.Random(GLOBAL_SEED)
    refs = combined80_refs(rows)
    existing_hashes = {n: {ref["order_hash"] for ref in group} for n, group in refs.items()}
    prior_run12_hashes = run12_hashes()
    seeds = seed_orders_from_combined80(rows)
    by_n: dict[int, dict[str, dict[str, Any]]] = {n: {} for n in EXPECTED_N}
    raw_attempts: Counter[str] = Counter()
    duplicate_existing_count: Counter[int] = Counter()
    duplicate_run12_count: Counter[int] = Counter()

    for n in EXPECTED_N:
        rng = random.Random(42000 + n)
        target = TARGET_DEDUP_PER_N[n]

        deterministic_orders = [
            ("raster", "geometry_baseline", "raster_left_to_right", raster(n), "sentinel"),
            ("raster", "geometry_baseline", "raster_right_to_left", raster(n, True), "sentinel"),
            ("odd_even", "geometry_baseline", "odd_even_interlaced", odd_even(n), "calibration"),
            ("center_out", "geometry_baseline", "center_out", center_out(n), "calibration"),
            ("edge_in", "geometry_baseline", "edge_in_alternating", edge_in(n), "calibration"),
            ("maximin", "geometry_baseline", "greedy_maximin_distance", greedy_maximin(n), "exploitation"),
            ("block_interleaved", "geometry_baseline", "block_interleaved_quarters", block_interleaved(n), "calibration"),
            ("center_edge", "geometry_baseline", "center_edge_alternating", center_edge_alternating(n), "diversity"),
            ("method_c_inspired", "known_best_inspired", "method_c_u2_first_inspired", method_c_inspired(n, "u2"), "exploitation"),
            ("method_c_inspired", "known_best_inspired", "method_c_peeq_safety_inspired", method_c_inspired(n, "peeq"), "exploitation"),
            ("method_c_inspired", "known_best_inspired", "method_c_surfaceT_aware_inspired", method_c_inspired(n, "surface"), "calibration"),
            ("method_c_inspired", "known_best_inspired", "method_c_diversity_inspired", method_c_inspired(n, "diversity"), "diversity"),
            ("method_c_inspired", "known_best_inspired", "method_c_regular_seed_inspired", method_c_inspired(n, "regular"), "exploitation"),
        ]
        for family, source, method, order, role in deterministic_orders:
            raw_attempts[f"N{n}"] += 1
            digest = order_hash(order)
            if digest in existing_hashes[n]:
                duplicate_existing_count[n] += 1
            elif digest in prior_run12_hashes[n]:
                duplicate_run12_count[n] += 1
            add_candidate(by_n, n, order, family, source, method, priority_role=role)

        for jump in range(1, n):
            for start in range(n):
                for direction in (1, -1):
                    order = regular_jump(n, start, jump, direction)
                    family = "regular_jump_coprime" if gcd(jump, n) == 1 else "regular_jump_non_coprime"
                    role = "exploitation" if gcd(jump, n) == 1 else "sentinel"
                    raw_attempts[f"N{n}"] += 1
                    digest = order_hash(order)
                    if digest in existing_hashes[n]:
                        duplicate_existing_count[n] += 1
                        continue
                    if digest in prior_run12_hashes[n]:
                        duplicate_run12_count[n] += 1
                    add_candidate(by_n, n, order, family, "regular_jump_sweep", f"regular_jump_start{start}_jump{jump}_dir{direction}", priority_role=role)

        for seed in seeds[n]:
            base = seed["order"]
            name = str(seed.get("strategy_name", "combined80_seed"))
            for idx in range(350 if n in (24, 40) else 220):
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
                if digest in prior_run12_hashes[n]:
                    duplicate_run12_count[n] += 1
                    continue
                family = "known_best_mutation" if "S3B20" in name or "A04" in name else "combined80_seed_mutation"
                role = "exploitation" if idx % 4 in (0, 1) else "diversity"
                add_candidate(
                    by_n,
                    n,
                    order,
                    family,
                    "combined80_known_best_mutation",
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
                family = "surrogate_topology_crossover"
                source = "crossover"
                method = "seed_crossover_random_cut"
                role = "exploitation"
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
                role = "sentinel" if mode == "low_adjacent" else ("diversity" if mode == "random" else "exploitation")
            digest = order_hash(order)
            if digest in existing_hashes[n]:
                duplicate_existing_count[n] += 1
                continue
            if digest in prior_run12_hashes[n]:
                duplicate_run12_count[n] += 1
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
            candidate_id = f"R18_N{n}_C{idx:05d}"
            strategy_name = f"N{n}_{candidate_id}_{row['candidate_family']}"
            flat.append(
                {
                    **row,
                    **features,
                    "candidate_id": candidate_id,
                    "strategy_name": strategy_name[:120],
                    "order_json": canonical_order_json(order),
                    "order_compact": "-".join(str(x) for x in order),
                    "is_existing_teacher_order": False,
                    "duplicate_of_run12": row["order_hash"] in prior_run12_hashes[n],
                    "nearest_existing_teacher_strategy": nearest_name,
                    "novelty_distance_to_combined80": novelty,
                }
            )
    counts = {
        "raw_generated_attempts_per_n": {int(k[1:]): v for k, v in raw_attempts.items()},
        "deduplicated_candidate_count_per_n": {n: sum(1 for row in flat if row["n"] == n) for n in EXPECTED_N},
        "duplicate_existing_teacher_attempts_per_n": dict(duplicate_existing_count),
        "duplicate_run12_attempts_per_n": dict(duplicate_run12_count),
    }
    return flat, counts


def score_candidates(candidates: list[dict[str, Any]], models: dict[str, Any]) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    feature_matrix = np.array([[float(row[f]) for f in F01_FEATURES] for row in candidates], dtype=float)
    primary_model = models[PRIMARY_TARGET]
    preds: dict[str, np.ndarray] = {PRIMARY_TARGET: primary_model.predict(feature_matrix)}
    for target in SECONDARY_TARGETS:
        preds[target] = models[target].predict(feature_matrix)
    tree_preds = np.vstack([tree.predict(feature_matrix) for tree in primary_model.estimators_])
    uncertainty = tree_preds.std(axis=0)
    for idx, row in enumerate(candidates):
        item = dict(row)
        item["pred_reward_combined80_u2_primary"] = float(preds[PRIMARY_TARGET][idx])
        item["pred_u2_score_combined80_rank"] = float(preds["target_u2_score_combined80_rank"][idx])
        item["pred_peeq_score_combined80_rank"] = float(preds["target_peeq_score_combined80_rank"][idx])
        item["pred_surfaceT_score_combined80_rank"] = float(preds["target_surfaceT_score_combined80_rank"][idx])
        item["pred_mises_score_combined80_rank"] = float(preds["target_mises_score_combined80_rank"][idx])
        item["pred_tree_mean"] = float(tree_preds[:, idx].mean())
        item["pred_uncertainty_std"] = float(uncertainty[idx])
        scored.append(item)

    for n in EXPECTED_N:
        group = [row for row in scored if row["n"] == n]
        ranked = sorted(group, key=lambda r: parse_float(r["pred_reward_combined80_u2_primary"]), reverse=True)
        max_rank = max(1, len(ranked) - 1)
        reward_vals = [parse_float(r["pred_reward_combined80_u2_primary"]) for r in group]
        novelty_vals = [parse_float(r["novelty_distance_to_combined80"]) for r in group]
        uncertainty_vals = [parse_float(r["pred_uncertainty_std"]) for r in group]
        rmin, rmax = min(reward_vals), max(reward_vals)
        nmin, nmax = min(novelty_vals), max(novelty_vals)
        umin, umax = min(uncertainty_vals), max(uncertainty_vals)
        for rank, row in enumerate(ranked, start=1):
            row["pred_rank_within_n"] = rank
            row["pred_percentile_within_n"] = 1.0 - (rank - 1) / max_rank
            pred_norm = safe_divide(parse_float(row["pred_reward_combined80_u2_primary"]) - rmin, rmax - rmin)
            novelty_norm = safe_divide(parse_float(row["novelty_distance_to_combined80"]) - nmin, nmax - nmin)
            uncertainty_norm = safe_divide(parse_float(row["pred_uncertainty_std"]) - umin, umax - umin)
            row["normalized_pred_reward_within_n"] = pred_norm
            row["normalized_novelty_within_n"] = novelty_norm
            row["normalized_uncertainty_within_n"] = uncertainty_norm
            row["exploitation_score"] = pred_norm
            row["exploration_score"] = 0.55 * pred_norm + 0.25 * novelty_norm + 0.20 * uncertainty_norm
            row["combined_selection_score"] = 0.70 * pred_norm + 0.15 * novelty_norm + 0.15 * uncertainty_norm
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
    fresh_group = [row for row in group if not row.get("duplicate_of_run12")]
    if len(fresh_group) >= target_count:
        group = fresh_group
    top_reward = sorted(group, key=lambda r: parse_float(r["pred_reward_combined80_u2_primary"]), reverse=True)
    top_u2 = sorted(group, key=lambda r: parse_float(r["pred_u2_score_combined80_rank"]), reverse=True)
    geometry = sorted(
        group,
        key=lambda r: (
            parse_float(r["parity_switch_rate"]),
            parse_float(r["normalized_mean_jump"]),
            -parse_float(r["adjacent_jump_count"]),
            parse_float(r["pred_reward_combined80_u2_primary"]),
        ),
        reverse=True,
    )
    method_c = sorted([r for r in group if "method_c" in r["candidate_family"] or "known_best" in r["candidate_family"]], key=lambda r: parse_float(r["pred_reward_combined80_u2_primary"]), reverse=True)
    diversity = sorted([r for r in group if parse_float(r["pred_reward_combined80_u2_primary"]) >= np.quantile([parse_float(x["pred_reward_combined80_u2_primary"]) for x in group], 0.50)], key=lambda r: (parse_float(r["novelty_distance_to_combined80"]), parse_float(r["pred_reward_combined80_u2_primary"])), reverse=True)
    uncertainty = sorted([r for r in group if parse_float(r["pred_reward_combined80_u2_primary"]) >= np.quantile([parse_float(x["pred_reward_combined80_u2_primary"]) for x in group], 0.40)], key=lambda r: (parse_float(r["pred_uncertainty_std"]), parse_float(r["pred_reward_combined80_u2_primary"])), reverse=True)
    sentinel = sorted(group, key=lambda r: (parse_float(r["novelty_distance_to_combined80"]), -parse_float(r["pred_reward_combined80_u2_primary"])), reverse=True)

    bucket_counts = {
        "surrogate_top": max(2, round(target_count * 0.25)),
        "U2_primary_top": max(1, round(target_count * 0.15)),
        "geometry_signal_top": max(1, round(target_count * 0.18)),
        "method_c_or_known_best_inspired": max(1, round(target_count * 0.15)),
        "diversity_top": max(1, round(target_count * 0.12)),
        "uncertainty_calibration": max(1, round(target_count * 0.10)),
        "negative_control_sentinel": 1 if target_count <= 8 else 2,
    }
    take_unique(selected, top_reward, bucket_counts["surrogate_top"], "surrogate_top", "Highest predicted combined80 U2-primary reward.")
    take_unique(selected, top_u2, bucket_counts["U2_primary_top"], "U2_primary_top", "Highest predicted U2 rank score.")
    take_unique(selected, geometry, bucket_counts["geometry_signal_top"], "geometry_signal_top", "Strong F01 geometry signal with parity/jump structure.")
    take_unique(selected, method_c, bucket_counts["method_c_or_known_best_inspired"], "method_c_or_known_best_inspired", "Known-best or Method-C-inspired local topology.")
    take_unique(selected, diversity, bucket_counts["diversity_top"], "diversity_top", "High novelty to combined80 teacher orders with acceptable predicted reward.")
    take_unique(selected, uncertainty, bucket_counts["uncertainty_calibration"], "uncertainty_calibration", "High uncertainty with moderate-to-high predicted reward for calibration.")
    take_unique(selected, sentinel, bucket_counts["negative_control_sentinel"], "negative_control_sentinel", "Structurally distinct lower/mid predicted control sentinel.")
    if len(selected) < target_count:
        take_unique(selected, sorted(group, key=lambda r: parse_float(r["combined_selection_score"]), reverse=True), target_count - len(selected), "combined_selection_fill", "Fill by combined prediction, novelty, and uncertainty score.")
    selected = selected[:target_count]
    for idx, row in enumerate(selected, start=1):
        row["shortlist_rank_within_n"] = idx
        row["shortlist_name"] = f"R18_N{n}_S{idx:02d}_{row['selection_bucket']}"
    return selected


def select_shortlists(scored: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    shortlist48: list[dict[str, Any]] = []
    for n in EXPECTED_N:
        group = [row for row in scored if row["n"] == n]
        shortlist48.extend(select_for_n(group, n, SHORTLIST_COUNTS[n]))

    def derive_batch(shortlist: list[dict[str, Any]], counts: dict[int, int], label: str) -> list[dict[str, Any]]:
        batch: list[dict[str, Any]] = []
        for n in EXPECTED_N:
            group = [row for row in shortlist if row["n"] == n]
            priority_buckets = [
                "surrogate_top",
                "U2_primary_top",
                "geometry_signal_top",
                "method_c_or_known_best_inspired",
                "diversity_top",
                "uncertainty_calibration",
                "negative_control_sentinel",
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

    return shortlist48, derive_batch(shortlist48, BATCH28_COUNTS, "batch28"), derive_batch(shortlist48, BATCH24_COUNTS, "batch24")


def existing_best_by_n(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    best: dict[int, dict[str, Any]] = {}
    for n in EXPECTED_N:
        group = [r for r in rows if r["n"] == n]
        best_reward = max(group, key=lambda r: parse_float(r.get(PRIMARY_TARGET)))
        best_u2 = max(group, key=lambda r: parse_float(r.get("target_u2_score_combined80_rank")))
        best[n] = {
            "combined80_best_reward_strategy": best_reward.get("strategy_name", ""),
            "combined80_best_reward": parse_float(best_reward.get(PRIMARY_TARGET)),
            "combined80_best_u2_strategy": best_u2.get("strategy_name", ""),
            "combined80_best_u2_score": parse_float(best_u2.get("target_u2_score_combined80_rank")),
        }
    return best


def predicted_improvement(scored: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_existing = existing_best_by_n(rows)
    out: list[dict[str, Any]] = []
    for n in EXPECTED_N:
        top = max([row for row in scored if row["n"] == n], key=lambda r: parse_float(r["pred_reward_combined80_u2_primary"]))
        base = best_existing[n]
        gap = parse_float(top["pred_reward_combined80_u2_primary"]) - parse_float(base["combined80_best_reward"])
        out.append(
            {
                "n": n,
                **base,
                "top_predicted_run18_candidate": top["strategy_name"],
                "top_predicted_candidate_id": top["candidate_id"],
                "top_predicted_family": top["candidate_family"],
                "top_predicted_reward": top["pred_reward_combined80_u2_primary"],
                "predicted_gap_vs_combined80_best_reward": gap,
                "predicted_exceeds_combined80_best_reward_surrogate_only": gap > 0,
                "top_predicted_novelty_distance": top["novelty_distance_to_combined80"],
                "nearest_existing_teacher_strategy": top["nearest_existing_teacher_strategy"],
                "note": "Surrogate-only prediction; teacher validation required before any physical claim.",
            }
        )
    return out


def diagnostics(candidates: list[dict[str, Any]], scored: list[dict[str, Any]], shortlist: list[dict[str, Any]], batch28: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n in EXPECTED_N:
        for label, data in [("unscored", candidates), ("scored", scored), ("shortlist48", shortlist), ("batch28", batch28)]:
            group = [r for r in data if r["n"] == n]
            rewards = [parse_float(r.get("pred_reward_combined80_u2_primary")) for r in group if "pred_reward_combined80_u2_primary" in r]
            novelty = [parse_float(r.get("novelty_distance_to_combined80")) for r in group if "novelty_distance_to_combined80" in r]
            uncertainty = [parse_float(r.get("pred_uncertainty_std")) for r in group if "pred_uncertainty_std" in r]
            rows.append(
                {
                    "n": n,
                    "dataset": label,
                    "count": len(group),
                    "family_counts_json": json.dumps(dict(Counter(r.get("candidate_family", "") for r in group)), sort_keys=True),
                    "bucket_counts_json": json.dumps(dict(Counter(r.get("selection_bucket", "") for r in group)), sort_keys=True),
                    "pred_reward_mean": mean(rewards, math.nan),
                    "pred_reward_max": max(rewards) if rewards else "",
                    "pred_reward_min": min(rewards) if rewards else "",
                    "novelty_mean": mean(novelty, math.nan),
                    "uncertainty_mean": mean(uncertainty, math.nan),
                }
            )
    return rows


def write_claim_boundary() -> None:
    md = "\n".join(
        [
            "# Run18 Claim Boundary",
            "",
            "## Safe claims",
            "- Run18 generated offline candidates using the combined80-updated diagnostic surrogate.",
            "- Run18 produced N24/N40-biased candidate shortlists and recommended future teacher-validation batches.",
            "- Candidate selection used within-N predicted normalized reward, novelty, uncertainty, and calibration buckets.",
            "- Run18 candidates are ready for human review and possible handoff packaging.",
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
            "Verdict: RUN18_OFFLINE_SURROGATE_SCREENING_ONLY_NO_TEACHER_VALIDATION",
            "",
        ]
    )
    (OUTPUT_DIR / "run18_claim_boundary.md").write_text(md, encoding="utf-8")
    write_json(
        OUTPUT_DIR / "run18_claim_boundary.json",
        {
            "verdict": "RUN18_OFFLINE_SURROGATE_SCREENING_ONLY_NO_TEACHER_VALIDATION",
            "safe_claims": [
                "offline candidates generated with the combined80-updated diagnostic surrogate",
                "N24/N40-biased shortlists and future teacher-validation batch options produced",
                "within-N predicted normalized reward, novelty, uncertainty, and calibration buckets used",
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
    batch28: list[dict[str, Any]],
    batch24: list[dict[str, Any]],
    improvement: list[dict[str, Any]],
    outputs: list[str],
) -> None:
    lines = [
        "# Stage 3 Run 18 - Combined80 Surrogate-Screened Candidate Generation",
        "",
        "## Purpose",
        "Generate a second offline candidate pool using the combined80-updated diagnostic surrogate, biased toward N24/N40 while retaining N12/N16 calibration and sentinel cases.",
        "",
        "## Inputs",
        f"- Combined80 RL-ready dataset: `{COMBINED80_READY}`",
        f"- Run17 best configurations: `{RUN17_BEST}`",
        f"- Run17 feature definitions: `{RUN17_FEATURE_DEFS}`",
        f"- Previous run12 candidate pool for duplicate avoidance: `{RUN12_SCORED}`",
        "",
        "## Run17 Surrogate Basis",
        f"- Model: `{model_meta['model_family']}`",
        f"- Feature set: `{model_meta['feature_set']}`",
        f"- Primary target: `{model_meta['primary_target']}`",
        f"- Training rows used for offline scoring: `{model_meta['training_rows']}`",
        "- This is a combined80 offline diagnostic surrogate, not a final or deployed model.",
        "",
        "## Candidate Generation Scope",
        f"- N values: `{EXPECTED_N}`",
        f"- Deduplicated candidates per N: `{counts['deduplicated_candidate_count_per_n']}`",
        "- Selection is intentionally biased toward N24/N40.",
        "",
        "## Candidate Generation Methods",
        "- Geometry-first candidates based on F01 signal: parity switching, jump statistics, direction reversals, and first/last track placement.",
        "- Method-C and known-best inspired variants using mutation, crossover, segment reversal, and block swaps.",
        "- Regular jump and coprime sweeps, especially for N24/N40.",
        "- Random and quasi-random diversity/calibration/sentinel candidates.",
        "- No candidate is labelled as trained RL output.",
        "",
        "## Candidate Validation and Deduplication",
        "- All generated orders were validated as legal permutations of 0..N-1.",
        "- Exact duplicates of combined80 teacher orders were removed.",
        "- Exact duplicates of the prior run12 pool were avoided where detected.",
        "",
        "## Surrogate Scoring",
        "- Candidates were scored by predicted within-N normalized combined80 U2-primary reward.",
        "- Secondary diagnostic targets include U2, PEEQ, SurfaceT, and Mises rank scores.",
        "- ExtraTrees tree-wise standard deviation is reported as an uncertainty proxy.",
        "",
        "## N24/N40-Biased Shortlist Policy",
        f"- Shortlist48 counts: `{dict(Counter(row['n'] for row in shortlist))}`",
        f"- Recommended batch28 counts: `{dict(Counter(row['n'] for row in batch28))}`",
        f"- Alternative batch24 counts: `{dict(Counter(row['n'] for row in batch24))}`",
        "- Selection buckets include surrogate top, U2-primary top, geometry signal top, Method-C/known-best inspired, diversity, uncertainty calibration, and negative/control sentinels.",
        "",
        "## Top Predicted Candidates",
    ]
    for row in improvement:
        lines.append(
            f"- N{row['n']}: `{row['top_predicted_run18_candidate']}` predicted reward `{float(row['top_predicted_reward']):.4f}`, "
            f"gap vs existing best `{float(row['predicted_gap_vs_combined80_best_reward']):.4f}` surrogate-only."
        )
    lines.extend(
        [
            "",
            "## Predicted Improvement vs Combined80 Best",
            "Predicted improvements are surrogate-only. They require future teacher validation before any physical claim.",
            "",
            "## Diagnostics",
            "- Candidate distributions, novelty, uncertainty, family composition, and duplicate-removal summaries are written to the diagnostics files.",
            "",
            "## Claim Boundary",
            "RUN18_OFFLINE_SURROGATE_SCREENING_ONLY_NO_TEACHER_VALIDATION. No teacher validation, physical superiority, trained RL success, arbitrary-N generalization, or CAE/INP existence is claimed.",
            "",
            "## Output Files",
        ]
    )
    lines.extend(f"- `{path}`" for path in outputs)
    lines.extend(
        [
            "",
            "## Recommended Run19",
            "Human review and handoff packaging for either batch24 or batch28. Do not generate CAE/INP until the user selects a batch size. If approved, run19 should create a handoff-only package similar to run13, not directly run Abaqus.",
            "",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def update_run_index() -> None:
    row = "| run_18 | Combined80 surrogate-screened candidate generation | Generate an N24/N40-biased offline candidate pool using the combined80-updated diagnostic surrogate and prepare shortlist/batch options for human review. | `scripts/stage3/run_18_generate_combined80_surrogate_screened_candidates.py` | `docs/stage3/runs/run_18_combined80_surrogate_screened_candidate_generation/RUN_18_COMBINED80_SURROGATE_SCREENED_CANDIDATE_GENERATION_REPORT.md` | `outputs/stage3_run_18_combined80_surrogate_screened_candidate_generation/` | `PASS_RUN18_INPUTS_READY_FOR_COMBINED80_SURROGATE_SCREENING` | No Abaqus, no ODB, no abqjobpilot, no CAE/INP/JNL generation, no teacher validation, no final RL policy training, no commit/push. Next: run19 handoff packaging after user selects batch24 or batch28. |"
    if RUN_INDEX_PATH.exists():
        text = RUN_INDEX_PATH.read_text(encoding="utf-8")
        if "| run_18 |" not in text:
            RUN_INDEX_PATH.write_text(text.rstrip() + "\n" + row + "\n", encoding="utf-8")


def write_plots(scored: list[dict[str, Any]], batch28: list[dict[str, Any]]) -> list[str]:
    paths: list[str] = []
    try:
        import matplotlib.pyplot as plt

        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        for n in EXPECTED_N:
            vals = [parse_float(r["pred_reward_combined80_u2_primary"]) for r in scored if r["n"] == n]
            ax.hist(vals, bins=30, alpha=0.45, label=f"N{n}")
        ax.set_title("Run18 predicted reward distribution")
        ax.set_xlabel("Predicted combined80 U2-primary reward")
        ax.set_ylabel("Candidate count")
        ax.legend()
        fig.tight_layout()
        path = FIGURE_DIR / "run18_predicted_reward_histogram.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(str(path))

        fig, ax = plt.subplots(figsize=(7, 5))
        x = [parse_float(r["novelty_distance_to_combined80"]) for r in scored]
        y = [parse_float(r["pred_reward_combined80_u2_primary"]) for r in scored]
        ax.scatter(x, y, s=4, alpha=0.35)
        ax.set_title("Run18 predicted reward vs novelty")
        ax.set_xlabel("Kendall novelty to nearest combined80 teacher order")
        ax.set_ylabel("Predicted reward")
        fig.tight_layout()
        path = FIGURE_DIR / "run18_predicted_reward_vs_novelty.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(str(path))

        bucket_counts = Counter(r.get("selection_bucket", "") for r in batch28)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(list(bucket_counts), list(bucket_counts.values()))
        ax.set_title("Run18 batch28 bucket composition")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        path = FIGURE_DIR / "run18_batch28_bucket_composition.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(str(path))
    except Exception as exc:  # noqa: BLE001
        write_json(OUTPUT_DIR / "run18_plotting_warning.json", {"plotting_warning": str(exc)})
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
    rows = load_combined80()
    validation = validate_inputs(rows)
    if validation["verdict"].startswith("FAIL"):
        print(validation["verdict"])
        return 2

    models, model_meta = train_surrogates(rows)
    candidates, counts = generate_candidates(rows)
    scored = score_candidates(candidates, models)
    shortlist48, batch28, batch24 = select_shortlists(scored)
    improvement = predicted_improvement(scored, rows)
    diag_rows = diagnostics(candidates, scored, shortlist48, batch28)

    outputs: list[str] = []
    for filename, data, writer in [
        ("run18_candidate_pool_unscored.csv", candidates, write_csv),
        ("run18_candidate_pool_scored.csv", scored, write_csv),
        ("run18_candidate_shortlist48.csv", shortlist48, write_csv),
        ("run18_recommended_future_teacher_batch28.csv", batch28, write_csv),
        ("run18_recommended_future_teacher_batch24.csv", batch24, write_csv),
        ("run18_predicted_improvement_vs_combined80.csv", improvement, write_csv),
        ("run18_candidate_generation_diagnostics.csv", diag_rows, write_csv),
    ]:
        path = OUTPUT_DIR / filename
        writer(path, data)
        outputs.append(str(path))
    for filename, data in [
        ("run18_candidate_pool_unscored.json", candidates),
        ("run18_candidate_pool_scored.json", scored),
        ("run18_candidate_shortlist48.json", shortlist48),
        ("run18_recommended_future_teacher_batch28.json", batch28),
        ("run18_recommended_future_teacher_batch24.json", batch24),
        ("run18_predicted_improvement_vs_combined80.json", improvement),
        ("run18_candidate_generation_diagnostics.json", diag_rows),
    ]:
        path = OUTPUT_DIR / filename
        write_table_json(path, data)
        outputs.append(str(path))

    write_claim_boundary()
    outputs.extend([str(OUTPUT_DIR / "run18_claim_boundary.md"), str(OUTPUT_DIR / "run18_claim_boundary.json"), str(OUTPUT_DIR / "run18_input_validation_summary.json"), str(OUTPUT_DIR / "run18_surrogate_model_metadata.json")])
    plot_paths = write_plots(scored, batch28)
    outputs.extend(plot_paths)
    write_report(validation, model_meta, counts, shortlist48, batch28, batch24, improvement, outputs)
    outputs.append(str(REPORT_PATH))
    update_run_index()

    manifest = {
        "run_id": RUN_ID,
        "run_name": RUN_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "branch": git_branch(),
        "script_path": str(ROOT / "scripts" / "stage3" / "run_18_generate_combined80_surrogate_screened_candidates.py"),
        "input_files": [str(p) for p in [COMBINED80_READY, COMBINED80_TEACHER, RUN17_FEATURES, RUN17_DETAILED, RUN17_BEST, RUN17_FEATURE_DEFS, RUN17_REPORT, RUN12_SCORED, RUN12_BATCH20, RUN16_REPORT] if p.exists()],
        "output_files": outputs,
        "validation_verdict": validation["verdict"],
        "candidate_pool_count": len(scored),
        "candidate_pool_count_per_n": dict(Counter(row["n"] for row in scored)),
        "raw_generated_candidate_count_per_n": counts["raw_generated_attempts_per_n"],
        "deduplicated_candidate_count_per_n": counts["deduplicated_candidate_count_per_n"],
        "shortlist48_count": len(shortlist48),
        "shortlist48_count_per_n": dict(Counter(row["n"] for row in shortlist48)),
        "batch28_count": len(batch28),
        "batch28_count_per_n": dict(Counter(row["n"] for row in batch28)),
        "batch24_count": len(batch24),
        "batch24_count_per_n": dict(Counter(row["n"] for row in batch24)),
        "report_path": str(REPORT_PATH),
        "claim_boundary_path": str(OUTPUT_DIR / "run18_claim_boundary.md"),
        "no_solver_run": True,
        "no_odb_opened": True,
        "no_abqjobpilot_run": True,
        "no_cae_inp_generated": True,
        "no_teacher_validation": True,
        "no_rl_policy_training": True,
        "no_commit_or_push": True,
    }
    write_json(MANIFEST_PATH, manifest)

    top_by_n = {n: max([r for r in scored if r["n"] == n], key=lambda r: parse_float(r["pred_reward_combined80_u2_primary"]))["strategy_name"] for n in EXPECTED_N}
    print(validation["verdict"])
    print(f"candidate_pool={len(scored)} per_n={dict(Counter(row['n'] for row in scored))}")
    print(f"shortlist48={len(shortlist48)} per_n={dict(Counter(row['n'] for row in shortlist48))}")
    print(f"batch28={len(batch28)} per_n={dict(Counter(row['n'] for row in batch28))}")
    print(f"batch24={len(batch24)} per_n={dict(Counter(row['n'] for row in batch24))}")
    print(f"top_by_n={top_by_n}")
    print(f"report={REPORT_PATH}")
    print(f"manifest={MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
