from __future__ import annotations


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


def center_edge_alternating(n: int) -> list[int]:
    center = center_out(n)
    edges = []
    for i in range((n + 1) // 2):
        edges.append(i)
        j = n - 1 - i
        if j != i:
            edges.append(j)
    order: list[int] = []
    for a, b in zip(center, edges):
        for x in (a, b):
            if x not in order:
                order.append(x)
    return order + [x for x in range(n) if x not in order]
