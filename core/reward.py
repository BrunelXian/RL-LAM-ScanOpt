"""Shared Stage A composite reward helpers for environment and baseline rollouts."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

DEFAULT_REWARD_WEIGHTS: dict[str, float] = {
    "coverage": 1.0,
    "invalid": 5.0,
    "peak": 0.8,
    "variance": 1.0,
    "reheat": 1.2,
    "jump": 0.3,
    "completion_bonus": 8.0,
}
DEFAULT_REHEAT_WINDOW_SIZE = 3
DEFAULT_USE_SUPPORT_RISK = False
REWARD_TERM_KEYS = (
    "coverage",
    "invalid",
    "peak",
    "variance",
    "reheat",
    "jump",
    "completion_bonus",
    "support_risk",
    "total",
)


def build_reward_weights(overrides: Mapping[str, float] | None = None) -> dict[str, float]:
    """Return a copy of the default Stage A reward weights."""
    weights = DEFAULT_REWARD_WEIGHTS.copy()
    if overrides:
        weights.update({key: float(value) for key, value in overrides.items()})
    return weights


def target_heat_values(field: np.ndarray, target_mask: np.ndarray) -> np.ndarray:
    """Return thermal values restricted to legal target cells."""
    target = np.asarray(target_mask, dtype=bool)
    values = np.asarray(field, dtype=np.float32)[target]
    if values.size == 0:
        return np.zeros(1, dtype=np.float32)
    return values


def target_heat_peak(field: np.ndarray, target_mask: np.ndarray) -> float:
    """Return the maximum target-masked thermal value."""
    return float(np.max(target_heat_values(field, target_mask)))


def target_heat_variance(field: np.ndarray, target_mask: np.ndarray) -> float:
    """Return the variance of the target-masked thermal values."""
    return float(np.var(target_heat_values(field, target_mask)))


def target_heat_mean(field: np.ndarray, target_mask: np.ndarray) -> float:
    """Return the mean of the target-masked thermal values."""
    return float(np.mean(target_heat_values(field, target_mask)))


def representative_location(region_mask: np.ndarray) -> tuple[int, int] | None:
    """Return a stable center-like location for the selected region."""
    cells = np.argwhere(np.asarray(region_mask, dtype=bool))
    if len(cells) == 0:
        return None
    center = np.rint(cells.mean(axis=0)).astype(int)
    return int(center[0]), int(center[1])


def local_target_preheat(
    pre_update_heat: np.ndarray,
    target_mask: np.ndarray,
    center: tuple[int, int] | None,
    window_size: int = DEFAULT_REHEAT_WINDOW_SIZE,
) -> float:
    """Return mean pre-deposit heat in a local target-masked window."""
    if center is None:
        return 0.0

    row, col = center
    radius = max(int(window_size), 1) // 2
    row_start = max(0, row - radius)
    row_end = min(pre_update_heat.shape[0], row + radius + 1)
    col_start = max(0, col - radius)
    col_end = min(pre_update_heat.shape[1], col + radius + 1)

    local_mask = np.asarray(target_mask, dtype=bool)[row_start:row_end, col_start:col_end]
    if not np.any(local_mask):
        return 0.0

    local_heat = np.asarray(pre_update_heat, dtype=np.float32)[row_start:row_end, col_start:col_end]
    return float(np.mean(local_heat[local_mask]))


def normalised_jump_distance(
    previous_location: tuple[int, int] | None,
    current_location: tuple[int, int] | None,
    shape: tuple[int, int],
) -> float:
    """Return normalized Euclidean distance between valid scan locations."""
    if previous_location is None or current_location is None:
        return 0.0

    max_distance = float(np.hypot(shape[0] - 1, shape[1] - 1))
    if max_distance <= 0.0:
        return 0.0

    jump_distance = float(
        np.hypot(
            current_location[0] - previous_location[0],
            current_location[1] - previous_location[1],
        )
    )
    return jump_distance / max_distance


def compute_reward_statistics(
    pre_update_heat: np.ndarray,
    post_update_heat: np.ndarray,
    target_mask: np.ndarray,
    current_location: tuple[int, int] | None,
    previous_location: tuple[int, int] | None,
    reheat_window_size: int = DEFAULT_REHEAT_WINDOW_SIZE,
) -> dict[str, float]:
    """Compute the target-masked thermal summaries used by the composite reward."""
    return {
        "peak_heat": target_heat_peak(post_update_heat, target_mask),
        "heat_variance": target_heat_variance(post_update_heat, target_mask),
        "local_preheat": local_target_preheat(
            pre_update_heat=pre_update_heat,
            target_mask=target_mask,
            center=current_location,
            window_size=reheat_window_size,
        ),
        "jump_distance": normalised_jump_distance(
            previous_location=previous_location,
            current_location=current_location,
            shape=np.asarray(target_mask).shape,
        ),
    }


def combine_reward_terms(reward_terms: Mapping[str, float]) -> float:
    """Combine per-term components into one scalar reward."""
    return float(
        reward_terms.get("coverage", 0.0)
        + reward_terms.get("invalid", 0.0)
        + reward_terms.get("peak", 0.0)
        + reward_terms.get("variance", 0.0)
        + reward_terms.get("reheat", 0.0)
        + reward_terms.get("jump", 0.0)
        + reward_terms.get("completion_bonus", 0.0)
        + reward_terms.get("support_risk", 0.0)
    )


def compute_reward_terms(
    *,
    valid_action: bool,
    coverage_event: bool,
    episode_complete: bool,
    peak_value: float,
    variance_value: float,
    local_preheat: float,
    jump_distance: float,
    reward_weights: Mapping[str, float] | None = None,
    use_support_risk: bool = DEFAULT_USE_SUPPORT_RISK,
) -> dict[str, float]:
    """Build the Stage A composite reward dictionary."""
    weights = build_reward_weights(reward_weights)
    reward_terms = {
        "coverage": 0.0,
        "invalid": 0.0,
        "peak": 0.0,
        "variance": 0.0,
        "reheat": 0.0,
        "jump": 0.0,
        "completion_bonus": 0.0,
        "support_risk": 0.0,
    }

    if not valid_action:
        reward_terms["invalid"] = -weights["invalid"]
        reward_terms["total"] = combine_reward_terms(reward_terms)
        return reward_terms

    if coverage_event:
        reward_terms["coverage"] = weights["coverage"]
    if episode_complete:
        reward_terms["completion_bonus"] = weights["completion_bonus"]

    reward_terms["peak"] = -weights["peak"] * float(peak_value)
    reward_terms["variance"] = -weights["variance"] * float(variance_value)
    reward_terms["reheat"] = -weights["reheat"] * float(local_preheat)
    reward_terms["jump"] = -weights["jump"] * float(jump_distance)

    if use_support_risk:
        # TODO: keep this reserved for future constrained-geometry / overhang-aware extensions.
        reward_terms["support_risk"] = 0.0

    reward_terms["total"] = combine_reward_terms(reward_terms)
    return reward_terms
