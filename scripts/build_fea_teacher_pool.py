"""Build the first FEA-teacher candidate pool for the LDED 32-track benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.geometry import LDEDCouponBenchmark, build_lded_coupon_32track_baselines, build_lded_coupon_32track_v1
from core.reward import build_sequence_objective_weights, compute_sequence_objective


BENCHMARK_NAME = "lded_coupon_32track_v1"
OUTPUT_DIR = PROJECT_ROOT / "assets" / "fea_teacher_pool_lded_32track"
SEQUENCE_DIR = OUTPUT_DIR / "sequences"
MANIFEST_CSV = OUTPUT_DIR / "fea_teacher_pool_manifest.csv"
MANIFEST_JSON = OUTPUT_DIR / "fea_teacher_pool_manifest.json"
SUMMARY_TXT = OUTPUT_DIR / "fea_teacher_pool_summary.txt"

LEGACY_POOL_DIR = PROJECT_ROOT / "assets" / "fea_teacher_pool"
LEGACY_TOP_JSON = PROJECT_ROOT / "assets" / "models" / "top_10_sequences_twi_64x64.json"
SELECTION_PATH = PROJECT_ROOT / "assets" / "models" / "reward_calibration_selection.json"

LINE_DECAY = 0.95
LINE_DEPOSIT = 1.0
GRID_RESOLUTION_MM = 1.0
RANDOM_CANDIDATE_SEEDS = list(range(256))
TARGET_POOL_SIZE = 46
MIN_POOL_SIZE = 30
MAX_POOL_SIZE = 50
EARLY_FRACTION = 0.20
ADJACENCY_HISTORY_WINDOW = 5
ADJACENCY_TRACK_RADIUS_MM = 3.05


@dataclass(frozen=True)
class TrackCandidate:
    """One fully specified candidate track-order trajectory."""

    name: str
    track_order: list[int]
    provenance: dict[str, Any]


def parse_args() -> argparse.Namespace:
    """Parse the minimal benchmark-selection CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        default=BENCHMARK_NAME,
        choices=[BENCHMARK_NAME],
        help="Line-order benchmark to build the FEA-teacher pool for.",
    )
    return parser.parse_args()


def load_variant1_weights() -> dict[str, float]:
    """Load the calibrated Stage A reward weights for sequence-objective reuse."""
    payload = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    return {key: float(value) for key, value in payload["variants"]["variant_1"].items()}


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV with stable field order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def write_json(path: Path, payload: Any) -> None:
    """Write UTF-8 JSON with deterministic formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, lines: list[str]) -> None:
    """Write a plain-text summary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sequence_key(track_order: Iterable[int]) -> tuple[int, ...]:
    """Return an immutable deduplication key for a track permutation."""
    return tuple(int(track_id) for track_id in track_order)


def is_valid_track_permutation(track_order: list[int], track_count: int) -> bool:
    """Return whether a track order is a full permutation of 0..track_count-1."""
    return len(track_order) == track_count and set(track_order) == set(range(track_count))


def build_line_masks(benchmark: LDEDCouponBenchmark) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return target mask and one boolean region mask per track on a 1 mm grid."""
    height = int(round(benchmark.plane_height_mm / GRID_RESOLUTION_MM))
    width = int(round(benchmark.plane_width_mm / GRID_RESOLUTION_MM))
    target_mask = np.zeros((height, width), dtype=bool)
    row_start = int(round(benchmark.patch_y_min_mm / GRID_RESOLUTION_MM))
    row_end = int(round(benchmark.patch_y_max_mm / GRID_RESOLUTION_MM))
    col_start = int(round(benchmark.patch_x_min_mm / GRID_RESOLUTION_MM))
    col_end = int(round(benchmark.patch_x_max_mm / GRID_RESOLUTION_MM))
    target_mask[row_start:row_end, col_start:col_end] = True

    track_masks: list[np.ndarray] = []
    for track in benchmark.tracks:
        mask = np.zeros_like(target_mask)
        track_col_start = int(round(track.x_start_mm / GRID_RESOLUTION_MM))
        track_col_end = int(round(track.x_end_mm / GRID_RESOLUTION_MM))
        track_row_start = int(round(track.y_start_mm / GRID_RESOLUTION_MM))
        track_row_end = int(round(track.y_end_mm / GRID_RESOLUTION_MM))
        mask[track_row_start:track_row_end, track_col_start:track_col_end] = True
        track_masks.append(mask)
    return target_mask, track_masks


def normalized_track_jump(benchmark: LDEDCouponBenchmark, previous_track_id: int | None, current_track_id: int) -> float:
    """Return normalized center-to-center jump between two tracks."""
    if previous_track_id is None:
        return 0.0
    previous = benchmark.tracks[int(previous_track_id)]
    current = benchmark.tracks[int(current_track_id)]
    max_distance = max(benchmark.tracks[-1].x_center_mm - benchmark.tracks[0].x_center_mm, 1e-9)
    return float(abs(current.x_center_mm - previous.x_center_mm) / max_distance)


def adjacency_ratio_track_order(benchmark: LDEDCouponBenchmark, track_order: list[int]) -> float:
    """Return how often the current track stays near recent tracks."""
    if len(track_order) <= 1:
        return 0.0
    adjacent = 0
    considered = 0
    for index, track_id in enumerate(track_order):
        if index == 0:
            continue
        current_center = benchmark.tracks[int(track_id)].x_center_mm
        recent_ids = track_order[max(0, index - ADJACENCY_HISTORY_WINDOW) : index]
        considered += 1
        if any(abs(current_center - benchmark.tracks[int(recent_id)].x_center_mm) <= ADJACENCY_TRACK_RADIUS_MM for recent_id in recent_ids):
            adjacent += 1
    return adjacent / max(considered, 1)


def evaluate_track_order(
    benchmark: LDEDCouponBenchmark,
    track_order: list[int],
    reward_weights: dict[str, float],
    target_mask: np.ndarray,
    track_masks: list[np.ndarray],
) -> dict[str, Any]:
    """Evaluate one 32-track permutation with a simple line-based thermal proxy."""
    if not is_valid_track_permutation(track_order, benchmark.track_count):
        raise ValueError("track_order must be a full 32-track permutation")

    heat_map = np.zeros_like(target_mask, dtype=np.float32)
    scanned_tracks = np.zeros(benchmark.track_count, dtype=bool)
    peak_values: list[float] = []
    variance_values: list[float] = []
    reheat_values: list[float] = []
    jump_values: list[float] = []
    target_mean_values: list[float] = []
    previous_track_id: int | None = None

    for track_id in track_order:
        track_id = int(track_id)
        if scanned_tracks[track_id]:
            raise ValueError("track_order contains duplicate track ids")
        scanned_tracks[track_id] = True

        heat_map *= LINE_DECAY
        local_preheat = float(np.mean(heat_map[track_masks[track_id]]))
        heat_map[track_masks[track_id]] += LINE_DEPOSIT

        target_values = heat_map[target_mask]
        peak_values.append(float(np.max(target_values)))
        variance_values.append(float(np.var(target_values)))
        reheat_values.append(local_preheat)
        jump_values.append(normalized_track_jump(benchmark, previous_track_id, track_id))
        target_mean_values.append(float(np.mean(target_values)))
        previous_track_id = track_id

    peak_max = max(peak_values, default=0.0)
    var_agg = float(np.sum(variance_values))
    reheat_sum = float(np.sum(reheat_values))
    jump_sum = float(np.sum(jump_values))
    sequence_objective = compute_sequence_objective(
        peak_max=peak_max,
        var_agg=var_agg,
        reheat_sum=reheat_sum,
        jump_sum=jump_sum,
        sequence_weights=build_sequence_objective_weights(reward_weights),
    )
    final_coverage = float(np.mean(scanned_tracks))
    early_stop = max(1, math.ceil(len(track_order) * EARLY_FRACTION))
    adjacency_ratio = float(adjacency_ratio_track_order(benchmark, track_order))
    early_clustering_metric = float(adjacency_ratio_track_order(benchmark, track_order[:early_stop]))
    total_jump = float(np.sum(jump_values))
    mean_jump = float(np.mean(jump_values[1:])) if len(jump_values) > 1 else 0.0
    max_jump = float(np.max(jump_values)) if jump_values else 0.0
    cumulative_heat = float(np.sum(target_mean_values))
    feature_vector = [
        float(-sequence_objective),
        float(peak_max),
        float(variance_values[-1] if variance_values else 0.0),
        float(reheat_sum),
        total_jump,
        mean_jump,
        max_jump,
        adjacency_ratio,
        early_clustering_metric,
    ]
    return {
        "final_coverage": final_coverage,
        "proxy_score": float(-sequence_objective),
        "sequence_objective": float(sequence_objective),
        "peak_heat": float(peak_max),
        "heat_variance": float(variance_values[-1] if variance_values else 0.0),
        "cumulative_heat": cumulative_heat,
        "reheat_sum": reheat_sum,
        "total_jump": total_jump,
        "mean_jump": mean_jump,
        "max_jump": max_jump,
        "adjacency_ratio": adjacency_ratio,
        "early_clustering_metric": early_clustering_metric,
        "feature_vector": feature_vector,
    }


def perturb_local_reversal(track_order: list[int], block_size: int, seed: int) -> list[int]:
    """Reverse one local block while preserving permutation validity."""
    if len(track_order) <= block_size:
        return [int(track_id) for track_id in track_order]
    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, len(track_order) - block_size + 1))
    mutated = list(track_order)
    mutated[start : start + block_size] = list(reversed(mutated[start : start + block_size]))
    return [int(track_id) for track_id in mutated]


def perturb_block_swap(track_order: list[int], block_size: int, seed: int) -> list[int]:
    """Swap two equal-size blocks inside a track permutation."""
    if len(track_order) <= 2 * block_size:
        return [int(track_id) for track_id in track_order]
    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, len(track_order) - 2 * block_size + 1))
    second = int(rng.integers(first + block_size, len(track_order) - block_size + 1))
    mutated = list(track_order)
    block_a = mutated[first : first + block_size]
    block_b = mutated[second : second + block_size]
    mutated[first : first + block_size] = block_b
    mutated[second : second + block_size] = block_a
    return [int(track_id) for track_id in mutated]


def perturb_stride_shuffle(track_order: list[int], stride: int, seed: int) -> list[int]:
    """Shuffle a sparse stride subset while keeping a valid permutation."""
    if len(track_order) <= stride + 2:
        return [int(track_id) for track_id in track_order]
    rng = np.random.default_rng(seed)
    offset = int(rng.integers(0, stride))
    indices = list(range(offset, len(track_order), stride))
    if len(indices) <= 2:
        return [int(track_id) for track_id in track_order]
    mutated = list(track_order)
    values = [mutated[index] for index in indices]
    rng.shuffle(values)
    for index, value in zip(indices, values, strict=True):
        mutated[index] = int(value)
    return [int(track_id) for track_id in mutated]


def perturb_partial_random_insertion(track_order: list[int], moves: int, seed: int) -> list[int]:
    """Move a few tracks to random new positions while preserving the permutation."""
    rng = np.random.default_rng(seed)
    mutated = list(track_order)
    for _ in range(max(moves, 1)):
        source = int(rng.integers(0, len(mutated)))
        item = mutated.pop(source)
        destination = int(rng.integers(0, len(mutated) + 1))
        mutated.insert(destination, item)
    return [int(track_id) for track_id in mutated]


def build_base_candidates(benchmark: LDEDCouponBenchmark) -> list[TrackCandidate]:
    """Build deterministic anchor and random candidates for the benchmark."""
    baselines = build_lded_coupon_32track_baselines(random_seeds=(0,))
    candidates = [
        TrackCandidate(
            name=name,
            track_order=[int(track_id) for track_id in sequence],
            provenance={"generator": "anchor_baseline", "source_file": "core.geometry.build_lded_coupon_32track_baselines"},
        )
        for name, sequence in baselines.items()
    ]

    indices = list(range(benchmark.track_count))
    for seed in RANDOM_CANDIDATE_SEEDS:
        rng = np.random.default_rng(int(seed))
        candidates.append(
            TrackCandidate(
                name=f"random_seed_{int(seed)}",
                track_order=[int(track_id) for track_id in rng.permutation(indices)],
                provenance={"generator": "random_candidate", "seed": int(seed)},
            )
        )
    return candidates


def evaluate_candidates(
    benchmark: LDEDCouponBenchmark,
    candidates: list[TrackCandidate],
    reward_weights: dict[str, float],
    target_mask: np.ndarray,
    track_masks: list[np.ndarray],
) -> tuple[list[dict[str, Any]], int]:
    """Evaluate unique candidates and count exact duplicate permutations removed."""
    evaluated: list[dict[str, Any]] = []
    seen_keys: set[tuple[int, ...]] = set()
    duplicate_removed = 0

    for candidate in candidates:
        key = sequence_key(candidate.track_order)
        if key in seen_keys:
            duplicate_removed += 1
            continue
        seen_keys.add(key)
        metrics = evaluate_track_order(benchmark, candidate.track_order, reward_weights, target_mask, track_masks)
        evaluated.append(
            {
                "candidate_name": candidate.name,
                "track_order": [int(track_id) for track_id in candidate.track_order],
                "sequence_key": key,
                "provenance": dict(candidate.provenance),
                "sequence_length": len(candidate.track_order),
                **metrics,
            }
        )
    return evaluated, duplicate_removed


def feature_distance_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    """Return a normalized pairwise feature distance matrix."""
    matrix = np.asarray([row["feature_vector"] for row in rows], dtype=np.float64)
    if matrix.size == 0:
        return np.zeros((0, 0), dtype=np.float64)
    minima = matrix.min(axis=0)
    spans = np.maximum(matrix.max(axis=0) - minima, 1e-9)
    normalized = (matrix - minima) / spans
    diff = normalized[:, None, :] - normalized[None, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))


def build_perturbation_candidates(
    evaluated_base: list[dict[str, Any]],
) -> list[TrackCandidate]:
    """Build deterministic perturbations from anchors and top proxy trajectories."""
    lookup = {row["candidate_name"]: row for row in evaluated_base}
    top_base_rows = sorted(evaluated_base, key=lambda row: float(row["proxy_score"]), reverse=True)[:2]
    base_names = [
        "raster_left_to_right",
        "center_out",
        "odd_even_interlaced",
        "even_odd_interlaced",
    ]
    perturb_bases = [lookup[name] for name in base_names if name in lookup]
    perturb_bases.extend(top_base_rows)

    perturbations = [
        ("local_reversal", lambda sequence, seed: perturb_local_reversal(sequence, block_size=4, seed=seed)),
        ("block_swap", lambda sequence, seed: perturb_block_swap(sequence, block_size=4, seed=seed)),
        ("stride_shuffle", lambda sequence, seed: perturb_stride_shuffle(sequence, stride=3, seed=seed)),
        ("partial_random_insertion", lambda sequence, seed: perturb_partial_random_insertion(sequence, moves=2, seed=seed)),
    ]

    candidates: list[TrackCandidate] = []
    for base_index, base in enumerate(perturb_bases):
        for perturb_index, (name, builder) in enumerate(perturbations):
            seed = 20_000 + base_index * 101 + perturb_index
            candidates.append(
                TrackCandidate(
                    name=f"{base['candidate_name']}_{name}",
                    track_order=builder(base["track_order"], seed),
                    provenance={
                        "generator": "perturbation",
                        "base_name": base["candidate_name"],
                        "perturbation": name,
                        "seed": seed,
                    },
                )
            )
    return candidates


def add_selected(
    selected: list[dict[str, Any]],
    selected_ids: set[str],
    row: dict[str, Any],
    *,
    source_type: str,
    selection_reason: str,
    source_file: str = "",
    seed: int | str | None = None,
) -> None:
    """Append one row to the selected pool with manifest metadata."""
    trajectory_id = str(row["candidate_name"])
    if trajectory_id in selected_ids:
        return
    selected_ids.add(trajectory_id)
    selected.append(
        {
            **row,
            "trajectory_id": trajectory_id,
            "source_type": source_type,
            "selection_reason": selection_reason,
            "source_file": source_file,
            "seed": "" if seed is None else seed,
        }
    )


def select_anchor_baselines(selected: list[dict[str, Any]], selected_ids: set[str], evaluated: list[dict[str, Any]]) -> None:
    """Select the required anchor baselines for the line benchmark."""
    lookup = {row["candidate_name"]: row for row in evaluated}
    anchor_names = [
        "raster_left_to_right",
        "raster_right_to_left",
        "center_out",
        "edge_in",
        "odd_even_interlaced",
        "even_odd_interlaced",
        "random_seed_0",
    ]
    for name in anchor_names:
        row = lookup.get(name)
        if row is None:
            continue
        add_selected(
            selected,
            selected_ids,
            row,
            source_type="anchor_baseline",
            selection_reason=f"anchor_{name}",
            source_file=str(row["provenance"].get("source_file", "")),
            seed=row["provenance"].get("seed"),
        )


def select_proxy_best(selected: list[dict[str, Any]], selected_ids: set[str], evaluated: list[dict[str, Any]]) -> None:
    """Select the top-scoring line-order candidates."""
    rows = [
        row
        for row in evaluated
        if row["candidate_name"] not in selected_ids
        and str(row["provenance"].get("generator")) == "random_candidate"
    ]
    rows.sort(key=lambda row: float(row["proxy_score"]), reverse=True)
    for rank, row in enumerate(rows[:10], start=1):
        add_selected(
            selected,
            selected_ids,
            row,
            source_type="proxy_best",
            selection_reason=f"best_proxy_rank_{rank}",
            source_file="generated_random_candidates",
            seed=row["provenance"].get("seed"),
        )


def select_proxy_worst(selected: list[dict[str, Any]], selected_ids: set[str], evaluated: list[dict[str, Any]]) -> None:
    """Select the weakest or hottest candidate tails."""
    rows = [
        row
        for row in evaluated
        if row["candidate_name"] not in selected_ids
    ]
    rows.sort(key=lambda row: (float(row["proxy_score"]), -float(row["peak_heat"])))
    for row in rows[:5]:
        add_selected(
            selected,
            selected_ids,
            row,
            source_type="proxy_worst",
            selection_reason="worst_proxy_tail",
            seed=row["provenance"].get("seed"),
            source_file="generated_lined_order_candidates",
        )


def select_random_diverse(selected: list[dict[str, Any]], selected_ids: set[str], evaluated: list[dict[str, Any]]) -> None:
    """Select random trajectories spanning low/mid/high proxy quantiles."""
    rows = [
        row
        for row in evaluated
        if str(row["provenance"].get("generator")) == "random_candidate"
        and row["candidate_name"] not in selected_ids
    ]
    rows.sort(key=lambda row: float(row["proxy_score"]))
    if not rows:
        return
    positions = np.linspace(0, len(rows) - 1, num=min(8, len(rows)))
    used_indices: set[int] = set()
    for rank, position in enumerate(positions):
        index = int(round(float(position)))
        while index in used_indices and index + 1 < len(rows):
            index += 1
        if index in used_indices:
            continue
        used_indices.add(index)
        row = rows[index]
        quantile = rank / max(len(positions) - 1, 1)
        if quantile <= 0.25:
            selection_reason = "random_quantile_low"
        elif quantile >= 0.75:
            selection_reason = "random_quantile_high"
        else:
            selection_reason = "random_quantile_mid"
        add_selected(
            selected,
            selected_ids,
            row,
            source_type="random_diverse",
            selection_reason=selection_reason,
            seed=row["provenance"].get("seed"),
            source_file="generated_random_candidates",
        )


def select_proxy_ambiguous(selected: list[dict[str, Any]], selected_ids: set[str], evaluated: list[dict[str, Any]]) -> int:
    """Select near-score but far-feature ambiguous track-order pairs."""
    remaining = [
        row
        for row in evaluated
        if row["candidate_name"] not in selected_ids
    ]
    if len(remaining) < 2:
        return 0
    distances = feature_distance_matrix(remaining)
    score_values = np.asarray([float(row["proxy_score"]) for row in remaining], dtype=np.float64)
    score_range = float(score_values.max() - score_values.min()) if score_values.size else 0.0
    chosen_pairs = 0
    used_names: set[str] = set()
    score_multipliers = (0.015, 0.03, 0.05, 0.07)
    feature_quantiles = (0.80, 0.70, 0.60, 0.50)

    for score_multiplier, feature_quantile in zip(score_multipliers, feature_quantiles, strict=True):
        score_threshold = max(0.5, score_multiplier * score_range)
        upper_triangle = distances[np.triu_indices(len(remaining), 1)]
        feature_threshold = float(np.quantile(upper_triangle, feature_quantile)) if upper_triangle.size else 0.0
        pairs: list[tuple[float, float, int, int]] = []
        for first in range(len(remaining)):
            for second in range(first + 1, len(remaining)):
                score_gap = abs(float(remaining[first]["proxy_score"]) - float(remaining[second]["proxy_score"]))
                feature_gap = float(distances[first, second])
                if score_gap <= score_threshold and feature_gap >= feature_threshold:
                    pairs.append((score_gap, -feature_gap, first, second))
        pairs.sort()
        for score_gap, neg_feature_gap, first, second in pairs:
            row_a = remaining[first]
            row_b = remaining[second]
            if row_a["candidate_name"] in used_names or row_b["candidate_name"] in used_names:
                continue
            for row in (row_a, row_b):
                add_selected(
                    selected,
                    selected_ids,
                    row,
                    source_type="proxy_ambiguous",
                    selection_reason="proxy_ambiguous",
                    seed=row["provenance"].get("seed"),
                    source_file="generated_lined_order_candidates",
                )
                used_names.add(str(row["candidate_name"]))
            chosen_pairs += 1
            if chosen_pairs >= 4:
                break
        if chosen_pairs >= 4:
            break
    return chosen_pairs


def select_perturbed_or_mixed(selected: list[dict[str, Any]], selected_ids: set[str], evaluated: list[dict[str, Any]]) -> int:
    """Select a compact set of perturbed trajectories."""
    rows = [
        row
        for row in evaluated
        if str(row["provenance"].get("generator")) == "perturbation"
        and row["candidate_name"] not in selected_ids
    ]
    rows.sort(
        key=lambda row: (
            str(row["provenance"].get("base_name", "")),
            float(row["proxy_score"]),
        ),
        reverse=True,
    )
    count = 0
    for row in rows[:8]:
        add_selected(
            selected,
            selected_ids,
            row,
            source_type="perturbed_or_mixed",
            selection_reason=f"perturbed_{row['provenance'].get('perturbation', 'unknown')}",
            seed=row["provenance"].get("seed"),
            source_file=str(row["provenance"].get("base_name", "")),
        )
        count += 1
    return count


def fill_pool_if_needed(selected: list[dict[str, Any]], selected_ids: set[str], evaluated: list[dict[str, Any]]) -> None:
    """Fill the pool to target size using remaining high-diversity candidates."""
    if len(selected) >= TARGET_POOL_SIZE:
        return
    remaining = [
        row
        for row in evaluated
        if row["candidate_name"] not in selected_ids
    ]
    if not remaining:
        return

    selected_vectors = np.asarray([row["feature_vector"] for row in selected], dtype=np.float64) if selected else np.zeros((0, 9), dtype=np.float64)
    remaining.sort(key=lambda row: float(row["proxy_score"]), reverse=True)
    while len(selected) < TARGET_POOL_SIZE and remaining:
        if selected_vectors.size == 0:
            chosen_index = 0
        else:
            distances = []
            for row in remaining:
                vector = np.asarray(row["feature_vector"], dtype=np.float64)
                diff = selected_vectors - vector
                nearest = float(np.min(np.sqrt(np.sum(diff**2, axis=1))))
                distances.append(nearest)
            chosen_index = int(np.argmax(np.asarray(distances)))
        row = remaining.pop(chosen_index)
        add_selected(
            selected,
            selected_ids,
            row,
            source_type="perturbed_or_mixed",
            selection_reason="diversity_fill",
            seed=row["provenance"].get("seed"),
            source_file="generated_lined_order_candidates",
        )
        selected_vectors = np.asarray([item["feature_vector"] for item in selected], dtype=np.float64)


def save_sequence_payloads(benchmark: LDEDCouponBenchmark, selected: list[dict[str, Any]]) -> None:
    """Save one line-based sequence JSON per selected trajectory."""
    SEQUENCE_DIR.mkdir(parents=True, exist_ok=True)
    for existing_file in SEQUENCE_DIR.glob("*.json"):
        existing_file.unlink()
    for row in selected:
        payload = {
            "trajectory_id": row["trajectory_id"],
            "benchmark_name": benchmark.benchmark_name,
            "trajectory_type": "track_order",
            "track_order": [int(track_id) for track_id in row["track_order"]],
            "num_tracks": benchmark.track_count,
            "fixed_track_direction": "bottom_to_top",
            "substrate_size_mm": [benchmark.plane_width_mm, benchmark.plane_height_mm],
            "deposited_patch_size_mm": [
                benchmark.patch_x_max_mm - benchmark.patch_x_min_mm,
                benchmark.patch_y_max_mm - benchmark.patch_y_min_mm,
            ],
            "margin_mm": benchmark.margin_left_mm,
            "track_width_mm": benchmark.track_width_mm,
            "track_length_mm": benchmark.track_length_mm,
            "track_pitch_mm": benchmark.track_pitch_mm,
            "source_type": row["source_type"],
            "selection_reason": row["selection_reason"],
            "source_file": row["source_file"],
            "seed": row["seed"],
            "metrics": {
                "sequence_length": int(row["sequence_length"]),
                "final_coverage": float(row["final_coverage"]),
                "proxy_score": float(row["proxy_score"]),
                "sequence_objective": float(row["sequence_objective"]),
                "peak_heat": float(row["peak_heat"]),
                "heat_variance": float(row["heat_variance"]),
                "cumulative_heat": float(row["cumulative_heat"]),
                "reheat_sum": float(row["reheat_sum"]),
                "total_jump": float(row["total_jump"]),
                "mean_jump": float(row["mean_jump"]),
                "max_jump": float(row["max_jump"]),
                "adjacency_ratio": float(row["adjacency_ratio"]),
                "early_clustering_metric": float(row["early_clustering_metric"]),
            },
            "feature_vector": [float(value) for value in row["feature_vector"]],
            "provenance": row["provenance"],
        }
        sequence_path = SEQUENCE_DIR / f"{row['trajectory_id']}.json"
        write_json(sequence_path, payload)
        row["sequence_file"] = str(sequence_path)


def build_manifest_rows(benchmark: LDEDCouponBenchmark, selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert selected rows to manifest rows."""
    rows: list[dict[str, Any]] = []
    for row in selected:
        rows.append(
            {
                "trajectory_id": row["trajectory_id"],
                "benchmark_name": benchmark.benchmark_name,
                "trajectory_type": "track_order",
                "source_type": row["source_type"],
                "selection_reason": row["selection_reason"],
                "seed": row["seed"],
                "source_file": row["source_file"],
                "sequence_length": int(row["sequence_length"]),
                "num_tracks": benchmark.track_count,
                "fixed_track_direction": "bottom_to_top",
                "final_coverage": float(row["final_coverage"]),
                "proxy_score": float(row["proxy_score"]),
                "sequence_objective": float(row["sequence_objective"]),
                "peak_heat": float(row["peak_heat"]),
                "heat_variance": float(row["heat_variance"]),
                "cumulative_heat": float(row["cumulative_heat"]),
                "reheat_sum": float(row["reheat_sum"]),
                "total_jump": float(row["total_jump"]),
                "mean_jump": float(row["mean_jump"]),
                "max_jump": float(row["max_jump"]),
                "adjacency_ratio": float(row["adjacency_ratio"]),
                "early_clustering_metric": float(row["early_clustering_metric"]),
                "feature_vector": json.dumps([float(value) for value in row["feature_vector"]]),
                "sequence_file": row["sequence_file"],
            }
        )
    return rows


def build_summary(
    benchmark: LDEDCouponBenchmark,
    selected: list[dict[str, Any]],
    *,
    ambiguous_pair_count: int,
    perturbation_count: int,
    duplicate_removed_count: int,
) -> list[str]:
    """Build the required plain-text pool summary."""
    source_counts = Counter(str(row["source_type"]) for row in selected)
    reason_counts = Counter(str(row["selection_reason"]) for row in selected)

    def range_text(values: list[float]) -> str:
        return f"{min(values):.6f} .. {max(values):.6f}" if values else "n/a"

    proxy_scores = [float(row["proxy_score"]) for row in selected]
    peaks = [float(row["peak_heat"]) for row in selected]
    variances = [float(row["heat_variance"]) for row in selected]
    jumps = [float(row["total_jump"]) for row in selected]
    adjacencies = [float(row["adjacency_ratio"]) for row in selected]
    early_clusterings = [float(row["early_clustering_metric"]) for row in selected]
    coverages = [float(row["final_coverage"]) for row in selected]

    return [
        f"selected trajectory count: {len(selected)}",
        f"benchmark_name: {benchmark.benchmark_name}",
        "trajectory_type: track_order",
        "source_type distribution: " + ", ".join(f"{key}={value}" for key, value in sorted(source_counts.items())),
        "selection_reason distribution: " + ", ".join(f"{key}={value}" for key, value in sorted(reason_counts.items())),
        f"proxy_score range: {range_text(proxy_scores)}",
        f"peak_heat range: {range_text(peaks)}",
        f"heat_variance range: {range_text(variances)}",
        f"jump distance range: {range_text(jumps)}",
        f"adjacency_ratio range: {range_text(adjacencies)}",
        f"early_clustering_metric range: {range_text(early_clusterings)}",
        f"coverage range: {range_text(coverages)}",
        f"ambiguous pair count: {ambiguous_pair_count}",
        f"perturbation count: {perturbation_count}",
        f"duplicate permutation count removed: {duplicate_removed_count}",
        "twi_64x64 top sequences were not used as primary source: YES",
        f"legacy twi pool preserved at: {LEGACY_POOL_DIR}",
        f"legacy twi top-sequence asset exists: {'YES' if LEGACY_TOP_JSON.exists() else 'NO'}",
        f"30-50 target satisfied: {'YES' if MIN_POOL_SIZE <= len(selected) <= MAX_POOL_SIZE else 'NO'}",
    ]


def main() -> None:
    """Build the first line-based FEA-teacher trajectory pool."""
    args = parse_args()
    if args.benchmark != BENCHMARK_NAME:
        raise ValueError(f"Unsupported benchmark: {args.benchmark}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SEQUENCE_DIR.mkdir(parents=True, exist_ok=True)

    benchmark = build_lded_coupon_32track_v1()
    reward_weights = load_variant1_weights()
    target_mask, track_masks = build_line_masks(benchmark)

    base_candidates = build_base_candidates(benchmark)
    evaluated_base, duplicate_removed_base = evaluate_candidates(benchmark, base_candidates, reward_weights, target_mask, track_masks)
    perturbation_candidates = build_perturbation_candidates(evaluated_base)
    evaluated_perturbations, duplicate_removed_perturb = evaluate_candidates(
        benchmark,
        perturbation_candidates,
        reward_weights,
        target_mask,
        track_masks,
    )

    combined_lookup: dict[tuple[int, ...], dict[str, Any]] = {}
    for row in evaluated_base + evaluated_perturbations:
        combined_lookup.setdefault(sequence_key(row["track_order"]), row)
    evaluated = list(combined_lookup.values())
    duplicate_removed_count = duplicate_removed_base + duplicate_removed_perturb + (len(evaluated_base) + len(evaluated_perturbations) - len(evaluated))

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    select_anchor_baselines(selected, selected_ids, evaluated)
    select_proxy_best(selected, selected_ids, evaluated)
    select_proxy_worst(selected, selected_ids, evaluated)
    select_random_diverse(selected, selected_ids, evaluated)
    ambiguous_pair_count = select_proxy_ambiguous(selected, selected_ids, evaluated)
    perturbation_count = select_perturbed_or_mixed(selected, selected_ids, evaluated)
    fill_pool_if_needed(selected, selected_ids, evaluated)

    if len(selected) > MAX_POOL_SIZE:
        selected = selected[:MAX_POOL_SIZE]

    save_sequence_payloads(benchmark, selected)
    manifest_rows = build_manifest_rows(benchmark, selected)

    write_csv(
        MANIFEST_CSV,
        [
            "trajectory_id",
            "benchmark_name",
            "trajectory_type",
            "source_type",
            "selection_reason",
            "seed",
            "source_file",
            "sequence_length",
            "num_tracks",
            "fixed_track_direction",
            "final_coverage",
            "proxy_score",
            "sequence_objective",
            "peak_heat",
            "heat_variance",
            "cumulative_heat",
            "reheat_sum",
            "total_jump",
            "mean_jump",
            "max_jump",
            "adjacency_ratio",
            "early_clustering_metric",
            "feature_vector",
            "sequence_file",
        ],
        manifest_rows,
    )
    write_json(
        MANIFEST_JSON,
        {
            "benchmark_name": benchmark.benchmark_name,
            "trajectory_type": "track_order",
            "selected_count": len(selected),
            "num_tracks": benchmark.track_count,
            "fixed_track_direction": "bottom_to_top",
            "legacy_twi_pool_preserved": str(LEGACY_POOL_DIR),
            "trajectories": manifest_rows,
        },
    )
    write_text(
        SUMMARY_TXT,
        build_summary(
            benchmark,
            selected,
            ambiguous_pair_count=ambiguous_pair_count,
            perturbation_count=perturbation_count,
            duplicate_removed_count=duplicate_removed_count,
        ),
    )

    print("LDED 32-track FEA teacher pool build complete.")
    print(f"Saved manifest CSV to: {MANIFEST_CSV}")
    print(f"Saved manifest JSON to: {MANIFEST_JSON}")
    print(f"Saved summary to: {SUMMARY_TXT}")


if __name__ == "__main__":
    main()
