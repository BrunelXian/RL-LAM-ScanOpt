"""Structured checkerboard-like planner for irregular target masks."""

from __future__ import annotations

import numpy as np


def plan_checkerboard(mask: np.ndarray) -> list[tuple[int, int]]:
    """Scan even-parity target cells first, then odd-parity cells, both in row-major order."""
    valid_mask = np.asarray(mask, dtype=bool)
    if valid_mask.ndim != 2:
        raise ValueError("mask must be a 2D array")

    even_phase: list[tuple[int, int]] = []
    odd_phase: list[tuple[int, int]] = []
    for row in range(valid_mask.shape[0]):
        for col in range(valid_mask.shape[1]):
            if not valid_mask[row, col]:
                continue
            if (row + col) % 2 == 0:
                even_phase.append((row, col))
            else:
                odd_phase.append((row, col))
    return even_phase + odd_phase
