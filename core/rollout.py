"""Plan execution utilities for Phase 1/2 baseline thermal simulations."""

from __future__ import annotations

from typing import Any

import numpy as np

from core.metrics import summarise_run
from core.reward import (
    DEFAULT_REHEAT_WINDOW_SIZE,
    DEFAULT_REWARD_WEIGHTS,
    DEFAULT_USE_SUPPORT_RISK,
    REWARD_TERM_KEYS,
    build_reward_weights,
    compute_reward_statistics,
    compute_reward_terms,
    target_heat_peak,
    target_heat_variance,
)
from core.thermal import create_empty_thermal_field, update_thermal_field


def run_plan(
    mask: np.ndarray,
    actions: list[tuple[int, int]],
    deposit_strength: float = 1.0,
    diffusion: float = 0.08,
    decay: float = 0.96,
    record_history: bool = False,
    history_stride: int = 1,
    reward_weights: dict[str, float] | None = None,
    reheat_window_size: int = DEFAULT_REHEAT_WINDOW_SIZE,
    use_support_risk: bool = DEFAULT_USE_SUPPORT_RISK,
) -> dict[str, Any]:
    """Execute a scan plan and return the final masks, field, metrics, and reward breakdown."""
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
    reward_history: list[dict[str, float]] = []
    reward_totals = {key: 0.0 for key in REWARD_TERM_KEYS}
    weights = build_reward_weights(reward_weights or DEFAULT_REWARD_WEIGHTS)
    previous_valid_location: tuple[int, int] | None = None

    for step_idx, action in enumerate(actions):
        row, col = action
        pre_update_heat = field.copy()
        if not (0 <= row < target_mask.shape[0] and 0 <= col < target_mask.shape[1]):
            reward_terms = compute_reward_terms(
                valid_action=False,
                coverage_event=False,
                episode_complete=False,
                peak_value=0.0,
                variance_value=0.0,
                local_preheat=0.0,
                jump_distance=0.0,
                reward_weights=weights,
                use_support_risk=use_support_risk,
            )
            reward_history.append(reward_terms)
            for key in REWARD_TERM_KEYS:
                reward_totals[key] += reward_terms.get(key, 0.0)
            continue
        if not target_mask[row, col]:
            reward_terms = compute_reward_terms(
                valid_action=False,
                coverage_event=False,
                episode_complete=False,
                peak_value=0.0,
                variance_value=0.0,
                local_preheat=0.0,
                jump_distance=0.0,
                reward_weights=weights,
                use_support_risk=use_support_risk,
            )
            reward_history.append(reward_terms)
            for key in REWARD_TERM_KEYS:
                reward_totals[key] += reward_terms.get(key, 0.0)
            continue
        if scanned_mask[row, col]:
            reward_terms = compute_reward_terms(
                valid_action=False,
                coverage_event=False,
                episode_complete=False,
                peak_value=0.0,
                variance_value=0.0,
                local_preheat=0.0,
                jump_distance=0.0,
                reward_weights=weights,
                use_support_risk=use_support_risk,
            )
            reward_history.append(reward_terms)
            for key in REWARD_TERM_KEYS:
                reward_totals[key] += reward_terms.get(key, 0.0)
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
        reward_stats = compute_reward_statistics(
            pre_update_heat=pre_update_heat,
            post_update_heat=field,
            target_mask=target_mask,
            current_location=(row, col),
            previous_location=previous_valid_location,
            reheat_window_size=reheat_window_size,
        )
        reward_terms = compute_reward_terms(
            valid_action=True,
            coverage_event=True,
            episode_complete=bool(scanned_mask[target_mask].all()),
            peak_value=reward_stats["peak_heat"],
            variance_value=reward_stats["heat_variance"],
            local_preheat=reward_stats["local_preheat"],
            jump_distance=reward_stats["jump_distance"],
            reward_weights=weights,
            use_support_risk=use_support_risk,
        )
        reward_history.append(reward_terms)
        for key in REWARD_TERM_KEYS:
            reward_totals[key] += reward_terms.get(key, 0.0)
        previous_valid_location = (row, col)
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
        "reward_history": reward_history,
        "reward_terms_total": reward_totals,
        "reward_breakdown": {
            "total_reward": reward_totals["total"],
            "coverage": reward_totals["coverage"],
            "invalid": reward_totals["invalid"],
            "peak": reward_totals["peak"],
            "variance": reward_totals["variance"],
            "reheat": reward_totals["reheat"],
            "jump": reward_totals["jump"],
            "completion_bonus": reward_totals["completion_bonus"],
            "support_risk": reward_totals["support_risk"],
            "steps": metrics["steps"],
            "coverage_ratio": metrics["coverage_ratio"],
            "peak_heat_final": target_heat_peak(field, target_mask),
            "heat_variance_final": target_heat_variance(field, target_mask),
        },
    }
    if record_history:
        if not scanned_history:
            scanned_history.append(scanned_mask.copy())
            thermal_history.append(field.copy())
        result["scanned_history"] = scanned_history
        result["thermal_history"] = thermal_history
        result["history_stride"] = history_stride
    return result
