from __future__ import annotations


def greedy_maximin_distance(n: int) -> list[int]:
    order = [0, n - 1]
    while len(order) < n:
        remaining = [i for i in range(n) if i not in order]
        best = max(remaining, key=lambda x: (min(abs(x - y) for y in order), -x))
        order.append(best)
    return order
