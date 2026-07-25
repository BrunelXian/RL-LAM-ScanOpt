"""Rank-derived 1D score and deterministic 2D score lift.

This module intentionally contains no scan-order generation. It only converts a
final Stage 3 one-dimensional strategy order into score vectors and score
matrices.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence


def _is_missing(raw: object) -> bool:
    if raw is None:
        return True
    if isinstance(raw, str):
        text = raw.strip()
        return text == "" or text.lower() in {"nan", "none", "null"}
    return False


def _coerce_int_list(values: Sequence[object], n: int) -> list[int]:
    order: list[int] = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError("Order entries must be integer track ids, not bools.")
        if isinstance(value, int):
            order.append(value)
            continue
        if isinstance(value, str) and value.strip().isdigit():
            order.append(int(value.strip()))
            continue
        raise ValueError(f"Order entry {value!r} is not an integer track id.")

    expected = list(range(n))
    if len(order) != n:
        raise ValueError(f"Order length {len(order)} does not match n={n}.")
    if sorted(order) != expected:
        missing = sorted(set(expected) - set(order))
        extra = sorted(set(order) - set(expected))
        duplicates = sorted({x for x in order if order.count(x) > 1})
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        if duplicates:
            details.append(f"duplicates={duplicates}")
        raise ValueError("Order is not a legal permutation of 0..n-1: " + ", ".join(details))
    return order


def parse_order(raw: object, n: int) -> list[int]:
    """Parse JSON-list or compact-string final order and validate it."""

    if n < 1:
        raise ValueError("n must be positive.")
    if _is_missing(raw):
        raise ValueError("Order value is missing.")

    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        return _coerce_int_list(raw, n)

    text = str(raw).strip()
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Order JSON could not be decoded: {exc}") from exc
        if not isinstance(parsed, list):
            raise ValueError("Order JSON must decode to a list.")
        return _coerce_int_list(parsed, n)

    if "-" in text:
        parts = [part.strip() for part in text.split("-")]
    else:
        parts = [part.strip() for part in text.split(",")]
    if any(part == "" for part in parts):
        raise ValueError(f"Compact order contains an empty token: {text!r}")
    return _coerce_int_list(parts, n)


def derive_rank_score_from_order(order: Sequence[int], eps: float = 1e-6) -> list[float]:
    """Derive s(i) from a final order, returned by track index i."""

    n = len(order)
    if n < 2:
        raise ValueError("At least two tracks are required for rank normalization.")
    if not 0.0 < eps < 0.5:
        raise ValueError("eps must be in (0, 0.5).")

    legal_order = parse_order(list(order), n)
    rank_by_track = [0] * n
    for rank, track_id in enumerate(legal_order):
        rank_by_track[track_id] = rank

    scale = 1.0 - 2.0 * eps
    denom = float(n - 1)
    scores = [eps + scale * (1.0 - rank / denom) for rank in rank_by_track]
    if not all(0.0 < value < 1.0 for value in scores):
        raise ValueError("Derived score values must be strictly in (0, 1).")
    return scores


def _validate_score_vector(s: Sequence[float]) -> list[float]:
    scores = [float(value) for value in s]
    if not scores:
        raise ValueError("Score vector must not be empty.")
    if not all(0.0 < value < 1.0 for value in scores):
        raise ValueError("Score vector values must be strictly in (0, 1).")
    return scores


def lift_score_1d_to_2d_unit(s: Sequence[float]) -> list[list[float]]:
    """Compute M[i,j] = sqrt((s[i]^2 + s[j]^2) / 2)."""

    scores = _validate_score_vector(s)
    return [
        [math.sqrt((scores[i] ** 2 + scores[j] ** 2) / 2.0) for j in range(len(scores))]
        for i in range(len(scores))
    ]


def lift_score_1d_to_2d_raw(s: Sequence[float]) -> list[list[float]]:
    """Compute M_raw[i,j] = sqrt(s[i]^2 + s[j]^2)."""

    scores = _validate_score_vector(s)
    return [
        [math.sqrt(scores[i] ** 2 + scores[j] ** 2) for j in range(len(scores))]
        for i in range(len(scores))
    ]


def score_cell(s: Sequence[float], i: int, j: int, unit_normalized: bool = True) -> float:
    """Return the lifted score for one matrix cell."""

    scores = _validate_score_vector(s)
    n = len(scores)
    if not 0 <= i < n or not 0 <= j < n:
        raise IndexError(f"Cell indices ({i}, {j}) are outside score vector length {n}.")
    if unit_normalized:
        return math.sqrt((scores[i] ** 2 + scores[j] ** 2) / 2.0)
    return math.sqrt(scores[i] ** 2 + scores[j] ** 2)
