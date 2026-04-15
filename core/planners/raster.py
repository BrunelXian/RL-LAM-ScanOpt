"""Row-major raster planner for valid scan cells."""

from __future__ import annotations

import numpy as np


def plan_raster(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return valid scan cells in row-major order."""
    valid_mask = np.asarray(mask, dtype=bool)
    if valid_mask.ndim != 2:
        raise ValueError("mask must be a 2D array")

    actions: list[tuple[int, int]] = []
    for row in range(valid_mask.shape[0]):
        for col in range(valid_mask.shape[1]):
            if valid_mask[row, col]:
                actions.append((row, col))
    return actions
