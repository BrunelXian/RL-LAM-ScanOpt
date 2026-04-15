"""Random baseline planner for valid scan cells."""

from __future__ import annotations

import numpy as np


def plan_random(mask: np.ndarray, seed: int = 42) -> list[tuple[int, int]]:
    """Return valid scan cells in a reproducibly shuffled order."""
    valid_positions = np.argwhere(np.asarray(mask, dtype=bool))
    rng = np.random.default_rng(seed)
    shuffled = valid_positions[rng.permutation(len(valid_positions))]
    return [(int(row), int(col)) for row, col in shuffled]
