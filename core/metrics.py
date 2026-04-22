"""Simple thermal-related summary metrics for baseline runs."""

from __future__ import annotations

from typing import Any

import numpy as np

from core.reward import target_heat_mean, target_heat_peak, target_heat_variance


def coverage_ratio(target_mask: np.ndarray, scanned_mask: np.ndarray) -> float:
    """Return the fraction of target cells that were scanned at least once."""
    target = np.asarray(target_mask, dtype=bool)
    scanned = np.asarray(scanned_mask, dtype=bool)

    target_count = int(target.sum())
    if target_count == 0:
        return 0.0
    return float(np.logical_and(target, scanned).sum() / target_count)


def thermal_variance(field: np.ndarray, target_mask: np.ndarray | None = None) -> float:
    """Return the target-masked thermal variance of the final proxy field."""
    if target_mask is None:
        return float(np.var(np.asarray(field, dtype=np.float32)))
    return target_heat_variance(field, target_mask)


def thermal_peak(field: np.ndarray, target_mask: np.ndarray | None = None) -> float:
    """Return the maximum target-masked proxy temperature in the field."""
    if target_mask is None:
        return float(np.max(np.asarray(field, dtype=np.float32)))
    return target_heat_peak(field, target_mask)


def thermal_mean(field: np.ndarray, target_mask: np.ndarray | None = None) -> float:
    """Return the mean target-masked proxy temperature in the field."""
    if target_mask is None:
        return float(np.mean(np.asarray(field, dtype=np.float32)))
    return target_heat_mean(field, target_mask)


def summarise_run(
    target_mask: np.ndarray,
    scanned_mask: np.ndarray,
    final_thermal: np.ndarray,
    actions: list[tuple[int, int]],
) -> dict[str, Any]:
    """Summarise coverage and final thermal field statistics."""
    target = np.asarray(target_mask, dtype=bool)
    scanned = np.asarray(scanned_mask, dtype=bool)

    summary: dict[str, Any] = {
        "target_cells": int(target.sum()),
        "scanned_cells": int(scanned.sum()),
        "steps": len(actions),
        "coverage_ratio": coverage_ratio(target, scanned),
        "thermal_mean": thermal_mean(final_thermal, target),
        "thermal_peak": thermal_peak(final_thermal, target),
        "thermal_variance": thermal_variance(final_thermal, target),
    }
    return summary
