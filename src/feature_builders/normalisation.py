from __future__ import annotations

import math


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0:
        return default
    value = numerator / denominator
    return value if math.isfinite(value) else default


def validate_n(n: int) -> int:
    if not isinstance(n, int) or isinstance(n, bool):
        raise ValueError("n must be an integer")
    if n < 2:
        raise ValueError("n must be >= 2")
    return n


def normalise_index(i: int, n: int) -> float:
    validate_n(n)
    return i / (n - 1) if n > 1 else 0.0


def normalise_distance(d: float, n: int) -> float:
    validate_n(n)
    return abs(d) / (n - 1) if n > 1 else 0.0


def normalise_step(t: int, n: int) -> float:
    return t / n if n > 0 else 0.0


def normalise_remaining(remaining_count: int, n: int) -> float:
    return remaining_count / n if n > 0 else 0.0


def clip01(x: float) -> float:
    if not math.isfinite(x):
        return 0.0
    return max(0.0, min(1.0, x))


def log_n_norm(n: int, reference_n: int = 32) -> float:
    validate_n(n)
    validate_n(reference_n)
    return math.log(n) / math.log(reference_n)
