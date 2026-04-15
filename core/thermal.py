"""Gaussian heat-source plus diffusion/cooling proxy updates for scan simulation."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def create_empty_thermal_field(grid_size: int) -> np.ndarray:
    """Create a square zero-initialised thermal field."""
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    return np.zeros((grid_size, grid_size), dtype=np.float32)


def _gaussian_kernel(kernel_size: int, sigma: float) -> np.ndarray:
    """Create a small normalised Gaussian-like kernel."""
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    radius = kernel_size // 2
    coords = np.arange(-radius, radius + 1, dtype=np.float32)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    kernel = np.exp(-0.5 * (xx**2 + yy**2) / (sigma**2))
    kernel /= kernel.max()
    return kernel.astype(np.float32)


def apply_heat_source(
    field: np.ndarray,
    action: tuple[int, int],
    deposit_strength: float = 1.0,
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> np.ndarray:
    """Add a local Gaussian heat-source deposit at the chosen scan cell."""
    if field.ndim != 2:
        raise ValueError("field must be a 2D array")

    row, col = action
    if not (0 <= row < field.shape[0] and 0 <= col < field.shape[1]):
        raise ValueError("action is out of bounds for the thermal field")

    updated = np.asarray(field, dtype=np.float32).copy()
    kernel = _gaussian_kernel(kernel_size=kernel_size, sigma=sigma) * float(deposit_strength)
    radius = kernel_size // 2

    r0 = max(0, row - radius)
    r1 = min(updated.shape[0], row + radius + 1)
    c0 = max(0, col - radius)
    c1 = min(updated.shape[1], col + radius + 1)

    kr0 = radius - (row - r0)
    kr1 = kr0 + (r1 - r0)
    kc0 = radius - (col - c0)
    kc1 = kc0 + (c1 - c0)

    updated[r0:r1, c0:c1] += kernel[kr0:kr1, kc0:kc1]
    return updated


def diffuse_and_decay(field: np.ndarray, diffusion: float = 0.08, decay: float = 0.96) -> np.ndarray:
    """Apply simple diffusion-like smoothing and multiplicative cooling."""
    if field.ndim != 2:
        raise ValueError("field must be a 2D array")
    if not 0.0 <= diffusion <= 1.0:
        raise ValueError("diffusion must be between 0 and 1")
    if not 0.0 <= decay <= 1.0:
        raise ValueError("decay must be between 0 and 1")

    base = np.asarray(field, dtype=np.float32)
    if diffusion == 0.0:
        smoothed = base
    else:
        sigma = max(0.5, diffusion * 3.0)
        blurred = gaussian_filter(base, sigma=sigma, mode="nearest")
        smoothed = (1.0 - diffusion) * base + diffusion * blurred

    cooled = smoothed * decay
    return np.clip(cooled, a_min=0.0, a_max=None).astype(np.float32)


def update_thermal_field(
    field: np.ndarray,
    action: tuple[int, int],
    deposit_strength: float = 1.0,
    diffusion: float = 0.08,
    decay: float = 0.96,
) -> np.ndarray:
    """Apply a heat deposit, then diffuse and cool the result."""
    heated = apply_heat_source(field=field, action=action, deposit_strength=deposit_strength)
    return diffuse_and_decay(field=heated, diffusion=diffusion, decay=decay)
