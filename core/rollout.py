"""Plan execution utilities for Phase 1/2 baseline thermal simulations."""

from __future__ import annotations

from typing import Any

import numpy as np

from core.metrics import summarise_run
from core.thermal import create_empty_thermal_field, update_thermal_field


def run_plan(
    mask: np.ndarray,
    actions: list[tuple[int, int]],
    deposit_strength: float = 1.0,
    diffusion: float = 0.08,
    decay: float = 0.96,
    record_history: bool = False,
    history_stride: int = 1,
) -> dict[str, Any]:
    """Execute a scan plan and return the final masks, field, and metrics."""
    target_mask = np.asarray(mask, dtype=bool)
    if target_mask.ndim != 2:
        raise ValueError("mask must be a 2D array")
    if history_stride <= 0:
        raise ValueError("history_stride must be positive")

    scanned_mask = np.zeros_like(target_mask, dtype=bool)
    order_map = np.full(target_mask.shape, fill_value=-1, dtype=np.int32)
    field = create_empty_thermal_field(grid_size=target_mask.shape[0])
    executed_actions: list[tuple[int, int]] = []
    scanned_history: list[np.ndarray] = []
    thermal_history: list[np.ndarray] = []

    for step_idx, action in enumerate(actions):
        row, col = action
        if not (0 <= row < target_mask.shape[0] and 0 <= col < target_mask.shape[1]):
            continue
        if not target_mask[row, col]:
            continue
        if scanned_mask[row, col]:
            continue

        scanned_mask[row, col] = True
        order_map[row, col] = step_idx
        field = update_thermal_field(
            field=field,
            action=(row, col),
            deposit_strength=deposit_strength,
            diffusion=diffusion,
            decay=decay,
        )
        executed_actions.append((row, col))
        if record_history and (
            len(executed_actions) == 1
            or len(executed_actions) % history_stride == 0
            or len(executed_actions) == len(actions)
        ):
            scanned_history.append(scanned_mask.copy())
            thermal_history.append(field.copy())

    metrics = summarise_run(
        target_mask=target_mask,
        scanned_mask=scanned_mask,
        final_thermal=field,
        actions=executed_actions,
    )

    result = {
        "target_mask": target_mask,
        "scanned_mask": scanned_mask,
        "order_map": order_map,
        "final_thermal": field,
        "metrics": metrics,
        "actions": executed_actions,
    }
    if record_history:
        if not scanned_history:
            scanned_history.append(scanned_mask.copy())
            thermal_history.append(field.copy())
        result["scanned_history"] = scanned_history
        result["thermal_history"] = thermal_history
        result["history_stride"] = history_stride
    return result
