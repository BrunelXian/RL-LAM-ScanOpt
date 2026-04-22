"""Cool-first planner with an explicit jump-distance trade-off."""

from __future__ import annotations

import numpy as np

from core.thermal import create_empty_thermal_field, update_thermal_field


def plan_distance_aware_cool_first(
    mask: np.ndarray,
    field: np.ndarray | None = None,
    deposit_strength: float = 1.0,
    diffusion: float = 0.08,
    decay: float = 0.96,
    alpha: float = 1.0,
    beta: float = 0.35,
) -> list[tuple[int, int]]:
    """Balance current heat and travel distance when choosing the next cell."""
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
    previous: tuple[int, int] | None = None
    max_grid_distance = float(np.hypot(valid_mask.shape[0] - 1, valid_mask.shape[1] - 1))

    while remaining:
        candidates = sorted(remaining)
        heat_values = np.array([float(working_field[row, col]) for row, col in candidates], dtype=np.float32)
        heat_min = float(heat_values.min()) if heat_values.size else 0.0
        heat_span = float(heat_values.max() - heat_min) if heat_values.size else 0.0

        scored_candidates: list[tuple[float, int, int]] = []
        for row, col in candidates:
            if heat_span > 1e-9:
                normalized_heat = (float(working_field[row, col]) - heat_min) / heat_span
            else:
                normalized_heat = 0.0

            if previous is None or max_grid_distance <= 0.0:
                normalized_distance = 0.0
            else:
                normalized_distance = float(np.hypot(row - previous[0], col - previous[1]) / max_grid_distance)

            score = alpha * normalized_heat + beta * normalized_distance
            scored_candidates.append((score, row, col))

        _, chosen_row, chosen_col = min(scored_candidates)
        chosen = (chosen_row, chosen_col)
        actions.append(chosen)
        remaining.remove(chosen)
        working_field = update_thermal_field(
            field=working_field,
            action=chosen,
            deposit_strength=deposit_strength,
            diffusion=diffusion,
            decay=decay,
        )
        previous = chosen

    return actions
