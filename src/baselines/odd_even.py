from __future__ import annotations


def odd_even_interlaced(n: int) -> list[int]:
    return list(range(0, n, 2)) + list(range(1, n, 2))
