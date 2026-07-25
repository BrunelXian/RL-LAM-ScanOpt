from __future__ import annotations

import json
import math
import statistics
from typing import Iterable


def validate_order(order: list[int], n: int) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if len(order) != n:
        reasons.append("wrong_length")
    if not all(isinstance(x, int) and not isinstance(x, bool) for x in order):
        reasons.append("non_integer")
    if len(set(order)) != len(order):
        reasons.append("duplicate_tracks")
    if any((not isinstance(x, int)) or x < 0 or x >= n for x in order):
        reasons.append("out_of_range_tracks")
    if set(order) != set(range(n)):
        missing = sorted(set(range(n)) - set(x for x in order if isinstance(x, int)))
        if missing:
            reasons.append("missing_tracks")
    return not reasons, reasons


def unique_in_order(values: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def fill_remaining(prefix: list[int], n: int, mode: str = "ascending") -> list[int]:
    used = set(prefix)
    remaining = [i for i in range(n) if i not in used]
    if mode == "descending":
        remaining.reverse()
    return prefix + remaining


def coprime_step(n: int, preferred: int | None = None) -> int:
    candidates = []
    if preferred is not None:
        candidates.append(preferred)
    candidates.extend([max(1, n // 3), max(1, n // 2 - 1), 5, 7, 3])
    for step in candidates:
        step = step % n
        if step and math.gcd(step, n) == 1:
            return step
    return 1


def jump_stats(order: list[int], n: int) -> dict[str, float | int]:
    jumps = [abs(order[i + 1] - order[i]) for i in range(len(order) - 1)]
    if not jumps:
        jumps = [0]
    denom = max(1, n - 1)
    return {
        "mean_jump_norm": statistics.fmean(jumps) / denom,
        "median_jump_norm": statistics.median(jumps) / denom,
        "max_jump_norm": max(jumps) / denom,
        "min_jump_norm": min(jumps) / denom,
        "std_jump_norm": statistics.pstdev(jumps) / denom,
        "count_jump_1": sum(1 for x in jumps if x == 1),
        "count_jump_le_2": sum(1 for x in jumps if x <= 2),
        "count_jump_ge_0_25N": sum(1 for x in jumps if x >= 0.25 * n),
        "count_jump_ge_0_50N": sum(1 for x in jumps if x >= 0.50 * n),
    }


def edge_center_stats(order: list[int], n: int) -> dict[str, float]:
    edge_width = min(4, max(1, n // 8))
    edges = set(range(edge_width)) | set(range(n - edge_width, n))
    center_mid = (n - 1) / 2
    centers = {i for i in range(n) if abs(i - center_mid) <= 1}
    pos = {track: idx for idx, track in enumerate(order)}
    edge_steps = [pos[x] for x in edges]
    center_steps = [pos[x] for x in centers]
    denom = max(1, n - 1)
    edge_mean = statistics.fmean(edge_steps) / denom
    center_mean = statistics.fmean(center_steps) / denom
    return {
        "edge_mean_step_norm": edge_mean,
        "center_mean_step_norm": center_mean,
        "edge_before_center_score": center_mean - edge_mean,
    }


def lr_balance_stats(order: list[int], n: int) -> dict[str, float]:
    left = 0
    right = 0
    imbalances: list[int] = []
    for track in order:
        if track < n / 2:
            left += 1
        else:
            right += 1
        imbalances.append(abs(left - right))
    return {
        "left_right_mean_abs_imbalance": statistics.fmean(imbalances),
        "left_right_max_abs_imbalance": max(imbalances),
    }


def dispersion_score(order: list[int], n: int) -> float:
    if len(order) < 2:
        return 0.0
    distances = [
        abs(a - b) / max(1, n - 1)
        for idx, a in enumerate(order)
        for b in order[idx + 1 :]
    ]
    return statistics.fmean(distances)


def structural_summary(order: list[int], n: int) -> dict[str, float | int]:
    jumps = jump_stats(order, n)
    edge_center = edge_center_stats(order, n)
    lr = lr_balance_stats(order, n)
    adjacent_transition_count = sum(1 for i in range(len(order) - 1) if abs(order[i + 1] - order[i]) == 1)
    return {
        **jumps,
        **edge_center,
        **lr,
        "dispersion_score_norm": dispersion_score(order, n),
        "adjacent_transition_count": adjacent_transition_count,
        "repeated_neighbor_transition_rate": adjacent_transition_count / max(1, len(order) - 1),
    }


def directed_overlap(a: list[int], b: list[int]) -> float:
    pa = {(a[i], a[i + 1]) for i in range(len(a) - 1)}
    pb = {(b[i], b[i + 1]) for i in range(len(b) - 1)}
    return len(pa & pb) / max(1, len(a) - 1)


def undirected_overlap(a: list[int], b: list[int]) -> float:
    pa = {tuple(sorted((a[i], a[i + 1]))) for i in range(len(a) - 1)}
    pb = {tuple(sorted((b[i], b[i + 1]))) for i in range(len(b) - 1)}
    return len(pa & pb) / max(1, len(a) - 1)


def kendall_distance(a: list[int], b: list[int]) -> float:
    pos_b = {track: idx for idx, track in enumerate(b)}
    seq = [pos_b[x] for x in a]
    inv = 0
    for i in range(len(seq)):
        for j in range(i + 1, len(seq)):
            if seq[i] > seq[j]:
                inv += 1
    return inv / max(1, len(seq) * (len(seq) - 1) / 2)


def spearman_distance(a: list[int], b: list[int]) -> float:
    n = len(a)
    pos_a = {track: idx for idx, track in enumerate(a)}
    pos_b = {track: idx for idx, track in enumerate(b)}
    ss = sum((pos_a[i] - pos_b[i]) ** 2 for i in range(n))
    return ss / max(1, n * (n**2 - 1) / 3)


def jump_profile_distance(a: list[int], b: list[int], n: int) -> float:
    ja = [abs(a[i + 1] - a[i]) / max(1, n - 1) for i in range(n - 1)]
    jb = [abs(b[i + 1] - b[i]) / max(1, n - 1) for i in range(n - 1)]
    return statistics.fmean(abs(x - y) for x, y in zip(ja, jb))


def order_json(order: list[int]) -> str:
    return json.dumps(order, separators=(",", ":"))
