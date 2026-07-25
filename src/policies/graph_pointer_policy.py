from __future__ import annotations

from src.baselines.baseline_utils import fill_remaining, unique_in_order
from src.baselines.center_out import center_out
from src.baselines.edge_in import edge_in_alternating
from src.baselines.maximin import greedy_maximin_distance
from src.baselines.odd_even import odd_even_interlaced
from src.baselines.regular_jump import regular_jump_coprime


class GraphPointerPolicyPrototype:
    """Deterministic proxy pointer policy; no teacher validation or RL training."""

    def __init__(self, mode: str = "balanced_dispersion_proxy") -> None:
        self.mode = mode

    def decode(self, n: int) -> list[int]:
        if self.mode == "proxy_best":
            return self._blend(n, greedy_maximin_distance(n), odd_even_interlaced(n))
        if self.mode == "diverse_01":
            return self._blend(n, regular_jump_coprime(n), edge_in_alternating(n))
        if self.mode == "diverse_02":
            return self._rotate(center_out(n), max(1, n // 5))
        if self.mode == "anti_odd_even_novelty":
            return self._anti_odd_even(n)
        if self.mode == "u2first_proxy":
            return self._blend(n, odd_even_interlaced(n), center_out(n))
        return self._blend(n, greedy_maximin_distance(n), center_out(n))

    @staticmethod
    def _rotate(order: list[int], k: int) -> list[int]:
        k = k % len(order)
        return order[k:] + order[:k]

    @staticmethod
    def _blend(n: int, a: list[int], b: list[int]) -> list[int]:
        merged: list[int] = []
        for x, y in zip(a, b):
            merged.extend([x, y])
        return fill_remaining(unique_in_order(merged), n)

    @staticmethod
    def _anti_odd_even(n: int) -> list[int]:
        order: list[int] = []
        left = list(range(1, n, 2))
        right = list(range(n - 2 if n % 2 == 0 else n - 1, -1, -2))
        for x, y in zip(left, right):
            order.extend([x, y])
        order.extend(left[len(right) :])
        order.extend(right[len(left) :])
        return fill_remaining(unique_in_order(order), n)
