from __future__ import annotations


def edge_in_alternating(n: int) -> list[int]:
    order: list[int] = []
    left = 0
    right = n - 1
    while left <= right:
        order.append(left)
        if right != left:
            order.append(right)
        left += 1
        right -= 1
    return order
