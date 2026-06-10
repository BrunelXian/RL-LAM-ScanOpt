from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

from src.feature_builders.normalisation import (
    clip01,
    log_n_norm,
    normalise_distance,
    normalise_index,
    normalise_remaining,
    normalise_step,
    safe_divide,
    validate_n,
)


N_VALUES = [16, 24, 32, 40]
NODE_FEATURE_NAMES = [
    "x_norm",
    "distance_to_left_edge_norm",
    "distance_to_right_edge_norm",
    "is_edge_track",
    "is_center_track",
    "current_scanned_flag",
    "current_available_flag",
    "step_fraction",
    "remaining_fraction",
    "time_since_left_neighbor_scanned_norm",
    "time_since_right_neighbor_scanned_norm",
    "neighbor_scanned_count_norm",
    "recent_neighbor_scanned_flag",
    "local_heat_proxy_norm",
]
EDGE_FEATURE_NAMES = [
    "distance_ij_norm",
    "relative_position_ij_norm",
    "is_adjacent",
    "is_within_thermal_radius",
    "left_or_right_relation",
    "edge_type_code",
]
GLOBAL_FEATURE_NAMES = [
    "log_N_norm",
    "step_fraction",
    "remaining_fraction",
    "current_center_of_mass_scanned_norm",
    "left_right_heat_balance_norm",
    "edge_center_heat_balance_norm",
    "mean_jump_so_far_norm",
    "max_jump_so_far_norm",
    "last_jump_length_norm",
    "dispersion_score_norm",
]
SIGNED_FEATURES = {
    "relative_position_ij_norm",
    "left_or_right_relation",
    "left_right_heat_balance_norm",
    "edge_center_heat_balance_norm",
}


def _clean_prefix(n: int, scanned_order_prefix: list[int] | None) -> list[int]:
    validate_n(n)
    prefix = list(scanned_order_prefix or [])
    if any(not isinstance(x, int) or isinstance(x, bool) for x in prefix):
        raise ValueError("scanned_order_prefix must contain integer track indices")
    if any(x < 0 or x >= n for x in prefix):
        raise ValueError("scanned_order_prefix contains out-of-range track index")
    if len(set(prefix)) != len(prefix):
        raise ValueError("scanned_order_prefix contains duplicate track index")
    return prefix


def _clean_available(n: int, available_tracks: list[int] | None) -> list[int]:
    validate_n(n)
    if available_tracks is None:
        return list(range(n))
    available = list(available_tracks)
    if any(not isinstance(x, int) or isinstance(x, bool) for x in available):
        raise ValueError("available_tracks must contain integer track indices")
    if any(x < 0 or x >= n for x in available):
        raise ValueError("available_tracks contains out-of-range track index")
    if len(set(available)) != len(available):
        raise ValueError("available_tracks contains duplicate track index")
    return available


def _center_tracks(n: int) -> set[int]:
    lower = (n - 1) / 2.0
    return {i for i in range(n) if abs(i - lower) <= 1.0}


def _edge_tracks(n: int) -> set[int]:
    edge_width = min(4, max(1, n // 8))
    return set(range(edge_width)) | set(range(n - edge_width, n))


def _heat_proxy(n: int, prefix: list[int]) -> dict[int, float]:
    if not prefix:
        return {i: 0.0 for i in range(n)}
    current_step = len(prefix)
    tau_steps = max(1.0, 0.25 * n)
    epsilon = 0.05
    scanned_step = {track: step for step, track in enumerate(prefix)}
    raw: dict[int, float] = {}
    for i in range(n):
        total = 0.0
        for j, step_j in scanned_step.items():
            delta_t = current_step - step_j
            distance_norm = normalise_distance(i - j, n)
            total += math.exp(-delta_t / tau_steps) / (distance_norm + epsilon)
        raw[i] = total
    min_value = min(raw.values())
    max_value = max(raw.values())
    span = max_value - min_value
    return {i: clip01(safe_divide(value - min_value, span)) for i, value in raw.items()}


def _time_since_neighbor(n: int, prefix: list[int], neighbor: int | None) -> float:
    if neighbor is None or neighbor not in prefix:
        return 0.0
    scanned_step = {track: step for step, track in enumerate(prefix)}
    return normalise_step(len(prefix) - scanned_step[neighbor], n)


def build_track_nodes(n: int, scanned_order_prefix: list[int] | None = None) -> list[dict[str, Any]]:
    prefix = _clean_prefix(n, scanned_order_prefix)
    scanned = set(prefix)
    available = set(range(n))
    edge_tracks = _edge_tracks(n)
    center_tracks = _center_tracks(n)
    heat = _heat_proxy(n, prefix)
    remaining = n - len(scanned)
    nodes: list[dict[str, Any]] = []
    for i in range(n):
        left = i - 1 if i > 0 else None
        right = i + 1 if i < n - 1 else None
        neighbor_scanned_count = int(left in scanned) + int(right in scanned)
        recent_neighbor_scanned = False
        if prefix:
            recent = set(prefix[-min(2, len(prefix)) :])
            recent_neighbor_scanned = (left in recent) or (right in recent)
        features = {
            "x_norm": normalise_index(i, n),
            "distance_to_left_edge_norm": normalise_distance(i, n),
            "distance_to_right_edge_norm": normalise_distance(n - 1 - i, n),
            "is_edge_track": float(i in edge_tracks),
            "is_center_track": float(i in center_tracks),
            "current_scanned_flag": float(i in scanned),
            "current_available_flag": float(i in available),
            "step_fraction": normalise_step(len(prefix), n),
            "remaining_fraction": normalise_remaining(remaining, n),
            "time_since_left_neighbor_scanned_norm": _time_since_neighbor(n, prefix, left),
            "time_since_right_neighbor_scanned_norm": _time_since_neighbor(n, prefix, right),
            "neighbor_scanned_count_norm": neighbor_scanned_count / 2.0,
            "recent_neighbor_scanned_flag": float(recent_neighbor_scanned),
            "local_heat_proxy_norm": heat[i],
        }
        nodes.append({"track_index": i, "features": features})
    return nodes


def build_edges(
    n: int,
    mode: str = "adjacent_thermal",
    thermal_radius_norm: float = 0.25,
    k_nearest: int = 2,
) -> list[dict[str, Any]]:
    validate_n(n)
    if mode != "adjacent_thermal":
        raise ValueError("only adjacent_thermal mode is supported in run_03")
    directed: dict[tuple[int, int], str] = {}
    max_distance = max(1, math.ceil(thermal_radius_norm * (n - 1)))
    for i in range(n):
        for delta in range(1, max(k_nearest, max_distance) + 1):
            j = i + delta
            if j >= n:
                continue
            edge_type = "thermal_radius" if delta <= max_distance else "k_nearest"
            if delta == 1:
                edge_type = "adjacent"
            if delta <= k_nearest or delta <= max_distance:
                directed[(i, j)] = edge_type
                directed[(j, i)] = edge_type
    edge_type_code = {"adjacent": 1.0, "k_nearest": 0.5, "thermal_radius": 0.75}
    edges: list[dict[str, Any]] = []
    for source, target in sorted(directed):
        diff = target - source
        distance_norm = normalise_distance(diff, n)
        edge_type = directed[(source, target)]
        features = {
            "distance_ij_norm": distance_norm,
            "relative_position_ij_norm": diff / (n - 1),
            "is_adjacent": float(abs(diff) == 1),
            "is_within_thermal_radius": float(distance_norm <= thermal_radius_norm),
            "left_or_right_relation": -1.0 if diff < 0 else 1.0,
            "edge_type_code": edge_type_code[edge_type],
        }
        edges.append(
            {
                "source_index": source,
                "target_index": target,
                "edge_type": edge_type,
                "features": features,
            }
        )
    return edges


def build_global_context(n: int, scanned_order_prefix: list[int] | None = None) -> dict[str, Any]:
    prefix = _clean_prefix(n, scanned_order_prefix)
    scanned_count = len(prefix)
    remaining_count = n - scanned_count
    scanned_set = set(prefix)
    left = sum(1 for x in prefix if x < n / 2)
    right = scanned_count - left
    edge_tracks = _edge_tracks(n)
    center_tracks = _center_tracks(n)
    edge_count = sum(1 for x in prefix if x in edge_tracks)
    center_count = sum(1 for x in prefix if x in center_tracks)
    jumps = [abs(prefix[i + 1] - prefix[i]) for i in range(len(prefix) - 1)]
    if scanned_count:
        center_of_mass = statistics.fmean(prefix)
        center_of_mass_norm = normalise_index(int(round(center_of_mass)), n)
    else:
        center_of_mass_norm = 0.0
    if len(scanned_set) > 1:
        pair_distances = [
            normalise_distance(a - b, n)
            for idx, a in enumerate(prefix)
            for b in prefix[idx + 1 :]
        ]
        dispersion = statistics.fmean(pair_distances)
    else:
        dispersion = 0.0
    return {
        "N": n,
        "scanned_count": scanned_count,
        "remaining_count": remaining_count,
        "features": {
            "log_N_norm": log_n_norm(n, reference_n=max(N_VALUES)),
            "step_fraction": normalise_step(scanned_count, n),
            "remaining_fraction": normalise_remaining(remaining_count, n),
            "current_center_of_mass_scanned_norm": center_of_mass_norm,
            "left_right_heat_balance_norm": safe_divide(left - right, max(1, scanned_count)),
            "edge_center_heat_balance_norm": safe_divide(edge_count - center_count, max(1, scanned_count)),
            "mean_jump_so_far_norm": normalise_distance(statistics.fmean(jumps), n) if jumps else 0.0,
            "max_jump_so_far_norm": normalise_distance(max(jumps), n) if jumps else 0.0,
            "last_jump_length_norm": normalise_distance(jumps[-1], n) if jumps else 0.0,
            "dispersion_score_norm": clip01(dispersion),
        },
    }


def build_masks(
    n: int,
    scanned_order_prefix: list[int] | None = None,
    available_tracks: list[int] | None = None,
) -> dict[str, list[bool]]:
    prefix = _clean_prefix(n, scanned_order_prefix)
    available = set(_clean_available(n, available_tracks))
    scanned = set(prefix)
    scanned_mask = [i in scanned for i in range(n)]
    available_mask = [i in available for i in range(n)]
    pointer_legal_mask = [available_mask[i] and not scanned_mask[i] for i in range(n)]
    return {
        "scanned_mask": scanned_mask,
        "available_mask": available_mask,
        "pointer_legal_mask": pointer_legal_mask,
    }


def build_graph_state(
    n: int,
    scanned_order_prefix: list[int] | None = None,
    available_tracks: list[int] | None = None,
) -> dict[str, Any]:
    prefix = _clean_prefix(n, scanned_order_prefix)
    graph = {
        "metadata": {
            "n": n,
            "scanned_order_prefix": prefix,
            "representation": "variable_n_track_graph",
            "fixed_id_embedding_used": False,
            "raw_track_id_model_feature_used": False,
        },
        "nodes": build_track_nodes(n, prefix),
        "edges": build_edges(n),
        "global_context": build_global_context(n, prefix),
        "masks": build_masks(n, prefix, available_tracks),
        "feature_names": {
            "node": NODE_FEATURE_NAMES,
            "edge": EDGE_FEATURE_NAMES,
            "global": GLOBAL_FEATURE_NAMES,
        },
    }
    graph["validation"] = validate_graph_state(graph)
    return graph


def _finite_values(features: dict[str, float], names: list[str]) -> list[float]:
    return [float(features[name]) for name in names]


def _in_expected_range(name: str, value: float) -> bool:
    if not math.isfinite(value):
        return False
    if name in SIGNED_FEATURES:
        return -1.0 <= value <= 1.0
    return 0.0 <= value <= 1.0


def validate_graph_state(graph: dict[str, Any]) -> dict[str, Any]:
    n = graph["metadata"]["n"]
    prefix = graph["metadata"]["scanned_order_prefix"]
    nodes = graph["nodes"]
    edges = graph["edges"]
    masks = graph["masks"]
    duplicate_edges = len(edges) - len({(e["source_index"], e["target_index"]) for e in edges})
    mask_rule = all(
        masks["pointer_legal_mask"][i] == (masks["available_mask"][i] and not masks["scanned_mask"][i])
        for i in range(n)
    )
    all_values: list[tuple[str, float]] = []
    for node in nodes:
        all_values.extend((name, value) for name, value in node["features"].items())
    for edge in edges:
        all_values.extend((name, value) for name, value in edge["features"].items())
    all_values.extend((name, value) for name, value in graph["global_context"]["features"].items())
    all_finite = all(math.isfinite(float(value)) for _, value in all_values)
    bounds_pass = all(_in_expected_range(name, float(value)) for name, value in all_values)
    pass_status = (
        len(nodes) == n
        and len(masks["scanned_mask"]) == n
        and len(masks["available_mask"]) == n
        and len(masks["pointer_legal_mask"]) == n
        and len(prefix) == len(set(prefix))
        and all(0 <= x < n for x in prefix)
        and bool(edges)
        and duplicate_edges == 0
        and mask_rule
        and all_finite
        and bounds_pass
    )
    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "scanned_count": len(prefix),
        "available_count": sum(masks["available_mask"]),
        "legal_action_count": sum(masks["pointer_legal_mask"]),
        "duplicate_edge_count": duplicate_edges,
        "all_finite": all_finite,
        "normalized_bounds_pass": bounds_pass,
        "mask_legality_pass": mask_rule,
        "pass": pass_status,
        "verdict": "PASS" if pass_status else "FAIL",
    }


def summarise_graph_features(graph: dict[str, Any]) -> dict[str, dict[str, dict[str, float | int | str | bool]]]:
    def summarize(values: list[float], signed: bool) -> dict[str, float | int | str | bool]:
        finite = [v for v in values if math.isfinite(v)]
        expected = "[-1,1]" if signed else "[0,1]"
        passed = bool(finite) and all((-1.0 <= v <= 1.0) if signed else (0.0 <= v <= 1.0) for v in finite)
        return {
            "min": min(finite) if finite else "",
            "max": max(finite) if finite else "",
            "mean": statistics.fmean(finite) if finite else "",
            "finite_count": len(finite),
            "expected_range": expected,
            "pass": passed,
        }

    node_summary = {
        name: summarize([float(node["features"][name]) for node in graph["nodes"]], name in SIGNED_FEATURES)
        for name in NODE_FEATURE_NAMES
    }
    edge_summary = {
        name: summarize([float(edge["features"][name]) for edge in graph["edges"]], name in SIGNED_FEATURES)
        for name in EDGE_FEATURE_NAMES
    }
    global_summary = {
        name: {
            "value": float(graph["global_context"]["features"][name]),
            "expected_range": "[-1,1]" if name in SIGNED_FEATURES else "[0,1]",
            "pass": _in_expected_range(name, float(graph["global_context"]["features"][name])),
        }
        for name in GLOBAL_FEATURE_NAMES
    }
    return {"node": node_summary, "edge": edge_summary, "global": global_summary}


def sample_prefixes(n: int) -> dict[str, list[int]]:
    center_left = (n - 1) // 2
    center_right = center_left + 1 if center_left + 1 < n else center_left
    center_out: list[int] = []
    for left, right in zip(range(center_left, -1, -1), range(center_right, n)):
        for track in (left, right):
            if track not in center_out:
                center_out.append(track)
        if len(center_out) >= 4:
            break
    maximin_seed = []
    for track in [0, n - 1, (n - 1) // 2, (n - 1) // 4]:
        if track not in maximin_seed:
            maximin_seed.append(track)
    return {
        "empty": [],
        "raster_like_early": list(range(min(4, n))),
        "odd_even_like_early": [x for x in [0, 2, 4, 6] if x < n],
        "center_out_like_early": center_out[:4],
        "maximin_like_early": maximin_seed[:4],
    }


def build_sample_graphs(n_values: list[int] | None = None) -> list[dict[str, Any]]:
    graphs: list[dict[str, Any]] = []
    for n in n_values or N_VALUES:
        for prefix_name, prefix in sample_prefixes(n).items():
            graph = build_graph_state(n, scanned_order_prefix=prefix)
            graph["metadata"]["prefix_name"] = prefix_name
            graphs.append(graph)
    return graphs


def compact_graph(graph: dict[str, Any]) -> dict[str, Any]:
    return {
        "metadata": graph["metadata"],
        "nodes_first_5": graph["nodes"][:5],
        "edges_first_20": graph["edges"][:20],
        "global_context": graph["global_context"],
        "masks": graph["masks"],
        "validation": graph["validation"],
    }


def export_sample_graphs(output_path: str | Path) -> list[dict[str, Any]]:
    graphs = build_sample_graphs()
    payload = [compact_graph(graph) for graph in graphs]
    Path(output_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return graphs
