from __future__ import annotations

from src.baselines.baseline_utils import coprime_step


def regular_jump_coprime(n: int) -> list[int]:
    step = coprime_step(n)
    order = []
    current = 0
    seen: set[int] = set()
    while len(order) < n:
        if current not in seen:
            seen.add(current)
            order.append(current)
        current = (current + step) % n
    return order
