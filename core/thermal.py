"""Gaussian heat-source plus simple diffusion/cooling proxy updates for scan simulation."""

from __future__ import annotations

import numpy as np


def create_empty_thermal_field(grid_size: int) -> np.ndarray:
    """Create a square zero-initialised thermal field."""
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    return np.zeros((grid_size, grid_size), dtype=np.float32)


def _gaussian_heat_map(shape: tuple[int, int], action: tuple[int, int], P: float, sigma: float) -> np.ndarray:
    """Create a full-field Gaussian heat map centered at the scan action."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if P < 0:
        raise ValueError("P must be non-negative")

    row, col = action
    yy, xx = np.indices(shape, dtype=np.float32)
    squared_dist = (yy - float(row)) ** 2 + (xx - float(col)) ** 2
    amplitude = float(P) / (np.pi * sigma**2)
    heat_map = amplitude * np.exp(-squared_dist / (sigma**2))
    return heat_map.astype(np.float32)


def apply_heat_source(
    field: np.ndarray,
    action: tuple[int, int],
    P: float = 200.0,
    sigma: float = 1.0,
) -> np.ndarray:
    """Apply a Gaussian heat source centered at the selected scan cell."""
    if field.ndim != 2:
        raise ValueError("field must be a 2D array")

    row, col = action
    if not (0 <= row < field.shape[0] and 0 <= col < field.shape[1]):
        raise ValueError("action is out of bounds for the thermal field")

    updated = np.asarray(field, dtype=np.float32).copy()
    updated += _gaussian_heat_map(shape=updated.shape, action=action, P=P, sigma=sigma)
    return updated


def diffuse_and_decay(
    field: np.ndarray,
    diffusion_rate: float = 0.08,
    decay_rate: float = 0.96,
) -> np.ndarray:
    """Apply local-neighborhood diffusion followed by global cooling."""
    if field.ndim != 2:
        raise ValueError("field must be a 2D array")
    if not 0.0 <= diffusion_rate <= 1.0:
        raise ValueError("diffusion_rate must be between 0 and 1")
    if not 0.0 <= decay_rate <= 1.0:
        raise ValueError("decay_rate must be between 0 and 1")

    base = np.asarray(field, dtype=np.float32)
    padded = np.pad(base, pad_width=1, mode="constant", constant_values=0.0)
    neighborhood_sum = (
        padded[0:-2, 0:-2]
        + padded[0:-2, 1:-1]
        + padded[0:-2, 2:]
        + padded[1:-1, 0:-2]
        + padded[1:-1, 1:-1]
        + padded[1:-1, 2:]
        + padded[2:, 0:-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    )
    neighborhood_mean = neighborhood_sum / 9.0
    diffused = (1.0 - diffusion_rate) * base + diffusion_rate * neighborhood_mean
    cooled = diffused * decay_rate
    return np.clip(cooled, a_min=0.0, a_max=None).astype(np.float32)


def update_thermal_field(
    field: np.ndarray,
    action: tuple[int, int],
    deposit_strength: float = 200.0,
    diffusion: float = 0.08,
    decay: float = 0.96,
    sigma: float = 1.0,
) -> np.ndarray:
    """Apply Gaussian heat input, then diffuse and cool the result."""
    heated = apply_heat_source(field=field, action=action, P=deposit_strength, sigma=sigma)
    return diffuse_and_decay(field=heated, diffusion_rate=diffusion, decay_rate=decay)
