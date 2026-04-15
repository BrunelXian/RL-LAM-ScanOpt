"""Greedy planner that prefers currently cooler scan cells."""

from __future__ import annotations

import numpy as np

from core.thermal import create_empty_thermal_field, update_thermal_field


def _local_cost(field: np.ndarray, row: int, col: int) -> float:
    """Estimate local thermal burden around a candidate cell."""
    r0 = max(0, row - 1)
    r1 = min(field.shape[0], row + 2)
    c0 = max(0, col - 1)
    c1 = min(field.shape[1], col + 2)
    neighborhood = field[r0:r1, c0:c1]
    return float(0.7 * field[row, col] + 0.3 * neighborhood.mean())


def plan_greedy_cool_first(
    mask: np.ndarray,
    field: np.ndarray | None = None,
    deposit_strength: float = 1.0,
    diffusion: float = 0.08,
    decay: float = 0.96,
) -> list[tuple[int, int]]:
    """Build a sequential plan by repeatedly choosing the coolest remaining cell."""
    valid_mask = np.asarray(mask, dtype=bool)
    if valid_mask.ndim != 2:
        raise ValueError("mask must be a 2D array")

    working_field = (
        np.asarray(field, dtype=np.float32).copy()
        if field is not None
        else create_empty_thermal_field(grid_size=valid_mask.shape[0])
    )
    remaining = {(int(row), int(col)) for row, col in np.argwhere(valid_mask)}
    actions: list[tuple[int, int]] = []

    while remaining:
        chosen = min(
            remaining,
            key=lambda pos: (_local_cost(working_field, pos[0], pos[1]), pos[0], pos[1]),
        )
        actions.append(chosen)
        remaining.remove(chosen)
        working_field = update_thermal_field(
            field=working_field,
            action=chosen,
            deposit_strength=deposit_strength,
            diffusion=diffusion,
            decay=decay,
        )

    return actions
