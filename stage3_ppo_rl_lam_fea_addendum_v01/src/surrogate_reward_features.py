"""Deterministic scan-order feature builder for PPO surrogate reward models."""

from __future__ import annotations

import ast
import json
import re
from typing import Iterable, Mapping, Sequence

import numpy as np


MAX_N = 40
ORDER_COLUMNS = ("order_json", "order_compact", "scan_order", "order", "scan_order_json")


def parse_order(row: Mapping[str, object]) -> list[int]:
    """Parse a scan order from a dataset row.

    Accepted columns include ``order_json``, ``order_compact``, ``scan_order``,
    and close equivalents. JSON/Python lists and separator-delimited integer
    strings are supported.
    """

    for column in ORDER_COLUMNS:
        raw = row.get(column)
        if raw is None:
            continue
        if isinstance(raw, float) and np.isnan(raw):
            continue
        text = str(raw).strip()
        if not text:
            continue
        return _parse_order_text(text)
    raise ValueError(f"No supported order column found. Tried: {ORDER_COLUMNS}")


def _parse_order_text(text: str) -> list[int]:
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(text)
        return [int(value) for value in parsed]

    tokens = re.findall(r"-?\d+", text)
    if not tokens:
        raise ValueError(f"Could not parse scan order from text: {text[:80]}")
    return [int(token) for token in tokens]


def validate_order(n: int, order: Sequence[int]) -> bool:
    """Return true when order is exactly a legal permutation of ``0..n-1``."""

    n = int(n)
    order_list = [int(item) for item in order]
    return len(order_list) == n and sorted(order_list) == list(range(n))


def feature_names(max_n: int = MAX_N) -> list[str]:
    names = [
        "n",
        "n_norm_40",
        "first_track_norm",
        "last_track_norm",
        "mean_track_index_norm",
        "std_track_index_norm",
        "early_mean_track_index_norm",
        "late_mean_track_index_norm",
        "early_center_distance_mean",
        "late_center_distance_mean",
        "early_edge_distance_mean",
        "late_edge_distance_mean",
        "mean_abs_jump_norm",
        "max_abs_jump_norm",
        "std_abs_jump_norm",
        "adjacent_jump_fraction",
        "long_jump_fraction",
        "signed_jump_mean_norm",
        "signed_jump_std_norm",
        "direction_reversal_count_norm",
        "monotonicity_fraction",
        "parity_switch_fraction",
        "odd_even_transition_count_norm",
        "early_odd_fraction",
        "early_even_fraction",
        "q1_center_edge_balance",
        "q2_center_edge_balance",
        "q3_center_edge_balance",
        "q4_center_edge_balance",
        "max_unvisited_gap_early_prefix",
        "early_spatial_spread_proxy",
    ]
    for track in range(max_n):
        names.append(f"track_{track:02d}_rank_position_norm")
        names.append(f"track_{track:02d}_valid_flag")
    return names


def feature_schema(max_n: int = MAX_N) -> dict[str, object]:
    names = feature_names(max_n=max_n)
    return {
        "schema_name": "ppo_surrogate_scan_order_features_v01",
        "max_n": max_n,
        "feature_count": len(names),
        "feature_names": names,
        "order_columns": list(ORDER_COLUMNS),
        "deterministic": True,
        "groups": {
            "global_n": ["n", "n_norm_40"],
            "position_normalized_sequence": names[2:12],
            "jump_features": names[12:21],
            "parity_and_interleaving": names[21:25],
            "coverage_dispersion": names[25:31],
            "fixed_40_rank_encoding": names[31:],
        },
    }


def order_to_features(n: int, order: Sequence[int]) -> np.ndarray:
    """Convert a legal scan order into a fixed-length feature vector."""

    n = int(n)
    order_arr = np.asarray([int(item) for item in order], dtype=np.float64)
    if not validate_order(n, order_arr.astype(int).tolist()):
        raise ValueError(f"Illegal scan order for n={n}: {order}")
    if n > MAX_N:
        raise ValueError(f"n={n} exceeds max_n={MAX_N}")

    denom = max(1.0, float(n - 1))
    center = (n - 1) / 2.0
    center_denom = max(1.0, center)
    norm_order = order_arr / denom
    early = order_arr[: max(1, n // 4)]
    late = order_arr[-max(1, n // 4) :]

    center_dist = np.abs(order_arr - center) / center_denom
    edge_dist = np.minimum(order_arr, n - 1 - order_arr) / center_denom
    early_center = center_dist[: len(early)]
    late_center = center_dist[-len(late) :]
    early_edge = edge_dist[: len(early)]
    late_edge = edge_dist[-len(late) :]

    jumps = np.diff(order_arr)
    abs_jumps = np.abs(jumps)
    signs = np.sign(jumps)
    nonzero_signs = signs[signs != 0]
    direction_reversals = int(np.sum(nonzero_signs[1:] != nonzero_signs[:-1])) if len(nonzero_signs) > 1 else 0
    positive_fraction = float(np.mean(jumps > 0)) if len(jumps) else 0.0
    negative_fraction = float(np.mean(jumps < 0)) if len(jumps) else 0.0
    monotonicity_fraction = max(positive_fraction, negative_fraction)

    parity = order_arr.astype(int) % 2
    parity_switches = int(np.sum(parity[1:] != parity[:-1])) if n > 1 else 0
    early_parity = parity[: len(early)]

    quarter_balances = []
    for quarter in np.array_split(order_arr, 4):
        if len(quarter) == 0:
            quarter_balances.append(0.0)
            continue
        q_center = np.abs(quarter - center) / center_denom
        q_edge = np.minimum(quarter, n - 1 - quarter) / center_denom
        quarter_balances.append(float(np.mean(q_center) - np.mean(q_edge)))

    prefix_len = max(1, n // 4)
    prefix = set(int(item) for item in order_arr[:prefix_len])
    unvisited = [idx for idx in range(n) if idx not in prefix]
    max_gap = 0
    if unvisited:
        runs: list[int] = []
        current = 1
        for left, right in zip(unvisited[:-1], unvisited[1:]):
            if right == left + 1:
                current += 1
            else:
                runs.append(current)
                current = 1
        runs.append(current)
        max_gap = max(runs)
    early_spread = float(np.ptp(early) / denom) if len(early) > 1 else 0.0

    features: list[float] = [
        float(n),
        float(n / MAX_N),
        float(norm_order[0]),
        float(norm_order[-1]),
        float(np.mean(norm_order)),
        float(np.std(norm_order)),
        float(np.mean(early / denom)),
        float(np.mean(late / denom)),
        float(np.mean(early_center)),
        float(np.mean(late_center)),
        float(np.mean(early_edge)),
        float(np.mean(late_edge)),
        float(np.mean(abs_jumps / denom)) if len(abs_jumps) else 0.0,
        float(np.max(abs_jumps / denom)) if len(abs_jumps) else 0.0,
        float(np.std(abs_jumps / denom)) if len(abs_jumps) else 0.0,
        float(np.mean(abs_jumps == 1)) if len(abs_jumps) else 0.0,
        float(np.mean(abs_jumps >= max(2.0, n / 3.0))) if len(abs_jumps) else 0.0,
        float(np.mean(jumps / denom)) if len(jumps) else 0.0,
        float(np.std(jumps / denom)) if len(jumps) else 0.0,
        float(direction_reversals / max(1, n - 2)),
        float(monotonicity_fraction),
        float(parity_switches / max(1, n - 1)),
        float(parity_switches / max(1, n - 1)),
        float(np.mean(early_parity == 1)),
        float(np.mean(early_parity == 0)),
        *quarter_balances,
        float(max_gap / max(1, n)),
        early_spread,
    ]

    rank_position = np.full(MAX_N, -1.0, dtype=np.float64)
    for pos, track in enumerate(order_arr.astype(int).tolist()):
        rank_position[track] = pos / denom
    for track in range(MAX_N):
        if track < n:
            features.append(float(rank_position[track]))
            features.append(1.0)
        else:
            features.append(0.0)
            features.append(0.0)

    vector = np.asarray(features, dtype=np.float64)
    expected = len(feature_names())
    if vector.shape != (expected,):
        raise RuntimeError(f"Feature vector shape {vector.shape} != {(expected,)}")
    return vector


def schema_markdown(schema: dict[str, object]) -> str:
    lines = [
        "# PPO Surrogate Feature Schema",
        "",
        f"- Schema: `{schema['schema_name']}`",
        f"- Max N: `{schema['max_n']}`",
        f"- Feature count: `{schema['feature_count']}`",
        f"- Deterministic: `{schema['deterministic']}`",
        "",
        "## Groups",
        "",
    ]
    groups = schema["groups"]
    assert isinstance(groups, dict)
    for group_name, group_features in groups.items():
        lines.append(f"### {group_name}")
        lines.append("")
        for feature in group_features:
            lines.append(f"- `{feature}`")
        lines.append("")
    return "\n".join(lines)
