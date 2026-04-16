"""Gymnasium environment for stripe-based scan planning inside letter-shaped regions."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from core.geometry import downsample_mask, generate_stripe_segments, generate_text_mask
from core.metrics import summarise_run
from core.thermal import create_empty_thermal_field, update_thermal_field


def build_twi_mask(grid_size: int = 64, canvas_size: int = 1024, text: str = "TWI") -> np.ndarray:
    """Create the default text-based planning mask used in training and evaluation."""
    high_res_mask = generate_text_mask(text=text, canvas_size=canvas_size)
    return downsample_mask(high_res_mask, grid_size=grid_size, threshold=0.2)


class ScanPlanningEnv(gym.Env[np.ndarray, int]):
    """Stripe-based scan-planning environment with invalid-action masking."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        planning_mask: np.ndarray | None = None,
        grid_size: int = 64,
        text: str = "TWI",
        canvas_size: int = 1024,
        stripe_width: int = 1,
        deposit_strength: float = 1.0,
        diffusion: float = 0.08,
        decay: float = 0.96,
        variance_penalty_coef: float = 0.5,
        peak_penalty_coef: float = 0.1,
        temp_diff_bonus_coef: float = 0.25,
        hotspot_dispersion_bonus_coef: float = 0.2,
        coverage_bonus_coef: float = 1.0,
        invalid_action_penalty: float = 100.0,
        invalid_action_limit: int = 5,
        max_steps: int | None = None,
        thermal_clip: float = 3.0,
    ) -> None:
        super().__init__()
        self.grid_size = int(grid_size)
        self.text = text
        self.canvas_size = int(canvas_size)
        self.stripe_width = int(stripe_width)
        self.deposit_strength = float(deposit_strength)
        self.diffusion = float(diffusion)
        self.decay = float(decay)
        self.variance_penalty_coef = float(variance_penalty_coef)
        self.peak_penalty_coef = float(peak_penalty_coef)
        self.temp_diff_bonus_coef = float(temp_diff_bonus_coef)
        self.hotspot_dispersion_bonus_coef = float(hotspot_dispersion_bonus_coef)
        self.coverage_bonus_coef = float(coverage_bonus_coef)
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.invalid_action_limit = int(invalid_action_limit)
        self.thermal_clip = float(thermal_clip)

        self._provided_mask = None if planning_mask is None else self._prepare_mask(planning_mask)
        self.target_mask = self._initial_mask()
        self.target_cell_count = int(self.target_mask.sum())
        self.stripes = self._build_stripes(self.target_mask)
        self.max_steps = int(max_steps) if max_steps is not None else max(len(self.stripes) * 2, 1)

        self.action_space = spaces.Discrete(max(len(self.stripes), 1))
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(3, self.grid_size, self.grid_size),
            dtype=np.uint8,
        )

        self.scanned_mask = np.zeros_like(self.target_mask, dtype=bool)
        self.thermal_field = create_empty_thermal_field(self.grid_size)
        self.scanned_stripes = np.zeros(len(self.stripes), dtype=bool)
        self.executed_actions: list[tuple[int, int]] = []
        self.executed_stripes: list[int] = []
        self.steps_taken = 0
        self.invalid_action_count = 0

    def _prepare_mask(self, mask: np.ndarray) -> np.ndarray:
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.ndim != 2:
            raise ValueError("planning_mask must be a 2D array")
        if mask_array.shape != (self.grid_size, self.grid_size):
            return downsample_mask(mask_array, grid_size=self.grid_size, threshold=0.2)
        return mask_array

    def _initial_mask(self) -> np.ndarray:
        if self._provided_mask is not None:
            return self._provided_mask.copy()
        return build_twi_mask(grid_size=self.grid_size, canvas_size=self.canvas_size, text=self.text)

    def _build_stripes(self, mask: np.ndarray) -> list[np.ndarray]:
        stripes = generate_stripe_segments(mask, grid_size=self.grid_size, stripe_width=self.stripe_width)
        legal_stripes = [stripe for stripe in stripes if np.logical_and(stripe, ~mask).sum() == 0 and stripe.any()]
        if not legal_stripes:
            raise ValueError("No legal stripe segments were generated for the planning mask")
        return legal_stripes

    def _normalised_thermal_channel(self) -> np.ndarray:
        scaled = np.clip(self.thermal_field / max(self.thermal_clip, 1e-6), 0.0, 1.0)
        return (scaled * 255.0).astype(np.uint8)

    def get_observation(self) -> np.ndarray:
        """Return the current observation tensor."""
        channels = [
            self.target_mask.astype(np.uint8) * 255,
            self.scanned_mask.astype(np.uint8) * 255,
            self._normalised_thermal_channel(),
        ]
        return np.stack(channels, axis=0)

    def _get_obs(self) -> np.ndarray:
        return self.get_observation()

    def _stripe_cells(self, stripe: np.ndarray) -> list[tuple[int, int]]:
        rows, cols = np.nonzero(stripe)
        order = np.lexsort((rows, cols))
        return [(int(rows[idx]), int(cols[idx])) for idx in order]

    def _local_temperature_difference(self, stripe_mask: np.ndarray) -> float:
        cells = np.argwhere(stripe_mask)
        if len(cells) == 0:
            return 0.0

        diffs: list[float] = []
        for row, col in cells:
            r0 = max(0, int(row) - 1)
            r1 = min(self.grid_size, int(row) + 2)
            c0 = max(0, int(col) - 1)
            c1 = min(self.grid_size, int(col) + 2)
            neighborhood = self.thermal_field[r0:r1, c0:c1]
            diffs.append(float(np.mean(np.abs(neighborhood - self.thermal_field[int(row), int(col)]))))
        return float(np.mean(diffs))

    def _hotspot_cluster_penalty(self) -> float:
        peak = float(np.max(self.thermal_field))
        if peak <= 0.0:
            return 0.0

        threshold = 0.8 * peak
        hotspot_mask = self.thermal_field >= threshold
        hotspot_positions = np.argwhere(hotspot_mask)
        if len(hotspot_positions) <= 1:
            return 0.0

        neighbor_links = 0.0
        for row, col in hotspot_positions:
            r0 = max(0, int(row) - 1)
            r1 = min(self.grid_size, int(row) + 2)
            c0 = max(0, int(col) - 1)
            c1 = min(self.grid_size, int(col) + 2)
            neighbor_links += float(hotspot_mask[r0:r1, c0:c1].sum() - 1)
        return neighbor_links / max(float(len(hotspot_positions)), 1.0)

    def _stripe_is_scanned(self, stripe_index: int) -> bool:
        return bool(self.scanned_stripes[stripe_index])

    def is_valid_action(self, action: int) -> bool:
        """Return True only when the action points to an unscanned legal stripe."""
        if not 0 <= int(action) < len(self.stripes):
            return False
        stripe = self.stripes[int(action)]
        return bool(not self._stripe_is_scanned(int(action)) and np.logical_and(stripe, self.target_mask).any())

    def action_masks(self) -> np.ndarray:
        """Return a flat mask with True only for unscanned legal stripes."""
        valid_actions = np.zeros(len(self.stripes), dtype=bool)
        for stripe_index, stripe in enumerate(self.stripes):
            if self._stripe_is_scanned(stripe_index):
                continue
            if np.logical_and(stripe, self.target_mask).any():
                valid_actions[stripe_index] = True
        return valid_actions

    def _terminal_info(self) -> dict[str, Any]:
        return summarise_run(
            target_mask=self.target_mask,
            scanned_mask=self.scanned_mask,
            final_thermal=self.thermal_field,
            actions=self.executed_actions,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if options and "mask" in options:
            self.target_mask = self._prepare_mask(options["mask"])
        else:
            self.target_mask = self._initial_mask()

        self.target_cell_count = int(self.target_mask.sum())
        self.stripes = self._build_stripes(self.target_mask)
        self.action_space = spaces.Discrete(max(len(self.stripes), 1))
        self.max_steps = max(self.max_steps, len(self.stripes))
        self.scanned_mask = np.zeros_like(self.target_mask, dtype=bool)
        self.thermal_field = create_empty_thermal_field(self.grid_size)
        self.scanned_stripes = np.zeros(len(self.stripes), dtype=bool)
        self.executed_actions = []
        self.executed_stripes = []
        self.steps_taken = 0
        self.invalid_action_count = 0

        info = {
            "target_cells": self.target_cell_count,
            "stripe_count": len(self.stripes),
            "valid_actions": int(self.action_masks().sum()),
        }
        return self.get_observation(), info

    def _calculate_reward(self, stripe_mask: np.ndarray, invalid_scan: bool) -> float:
        thermal_variance = float(np.var(self.thermal_field))
        thermal_peak = float(np.max(self.thermal_field))
        coverage_ratio = float(self.scanned_mask.sum() / max(self.target_cell_count, 1))
        local_temp_diff = self._local_temperature_difference(stripe_mask)
        hotspot_penalty = self._hotspot_cluster_penalty()

        reward = (
            self.coverage_bonus_coef * coverage_ratio
            - self.variance_penalty_coef * thermal_variance
            - self.peak_penalty_coef * thermal_peak
            + self.temp_diff_bonus_coef / (1.0 + local_temp_diff)
            + self.hotspot_dispersion_bonus_coef / (1.0 + hotspot_penalty)
        )
        if invalid_scan:
            reward -= self.invalid_action_penalty
        return float(reward)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.steps_taken += 1
        is_valid = self.is_valid_action(action)
        terminated = False
        truncated = False

        if not is_valid:
            self.invalid_action_count += 1
            reward = -self.invalid_action_penalty
            stripe_mask = np.zeros_like(self.target_mask, dtype=bool)
            if self.invalid_action_count >= self.invalid_action_limit:
                terminated = True
        else:
            stripe_index = int(action)
            stripe_mask = self.stripes[stripe_index]
            self.scanned_stripes[stripe_index] = True
            self.executed_stripes.append(stripe_index)

            for row, col in self._stripe_cells(stripe_mask):
                self.scanned_mask[row, col] = True
                self.executed_actions.append((row, col))
                self.thermal_field = update_thermal_field(
                    field=self.thermal_field,
                    action=(row, col),
                    deposit_strength=self.deposit_strength,
                    diffusion=self.diffusion,
                    decay=self.decay,
                )

            invalid_scan = bool(np.logical_and(self.scanned_mask, ~self.target_mask).any())
            reward = self._calculate_reward(stripe_mask, invalid_scan=invalid_scan)

        if not self.action_masks().any():
            terminated = True
        if self.steps_taken >= self.max_steps and not terminated:
            truncated = True

        info: dict[str, Any] = {
            "valid_action": is_valid,
            "invalid_action_count": self.invalid_action_count,
            "steps_taken": self.steps_taken,
            "remaining_valid_actions": int(self.action_masks().sum()),
            "coverage_ratio": float(self.scanned_mask.sum() / max(self.target_cell_count, 1)),
            "thermal_mean": float(np.mean(self.thermal_field)),
            "thermal_peak": float(np.max(self.thermal_field)),
            "thermal_variance": float(np.var(self.thermal_field)),
            "stripe_count": len(self.stripes),
            "executed_stripes": len(self.executed_stripes),
        }
        if terminated or truncated:
            info["metrics"] = self._terminal_info()

        return self.get_observation(), float(reward), terminated, truncated, info
