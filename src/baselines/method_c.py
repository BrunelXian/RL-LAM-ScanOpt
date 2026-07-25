from __future__ import annotations

from src.baselines.baseline_utils import fill_remaining, unique_in_order


def method_c_u2_first_engineering(n: int) -> list[int]:
    evens_left = list(range(0, max(1, n // 2), 2))
    evens_right = list(range(n // 2, n, 2))
    center = [n // 2 - 1, n // 2]
    odds_left_back = list(range(max(0, n // 2 - 1), -1, -2))
    odds_right = list(range(n // 2 + 1, n, 2))
    tail = list(range(n - 1, -1, -2))
    prefix = unique_in_order(evens_left + evens_right + center + odds_left_back + odds_right + tail)
    return fill_remaining(prefix, n)
