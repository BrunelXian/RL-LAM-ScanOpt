"""Deterministic planner that always selects the coolest remaining target cell."""

from __future__ import annotations

import numpy as np

from core.thermal import create_empty_thermal_field, update_thermal_field


def plan_cool_first(
    mask: np.ndarray,
    field: np.ndarray | None = None,
    deposit_strength: float = 1.0,
    diffusion: float = 0.08,
    decay: float = 0.96,
) -> list[tuple[int, int]]:
    """Build a plan by repeatedly choosing the lowest-heat remaining cell."""
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
            key=lambda pos: (float(working_field[pos[0], pos[1]]), pos[0], pos[1]),
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
