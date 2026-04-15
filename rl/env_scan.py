"""Gymnasium environment for masked scan-planning inside letter-shaped regions."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from core.geometry import downsample_mask, generate_text_mask
from core.metrics import summarise_run
from core.thermal import create_empty_thermal_field, update_thermal_field


def build_twi_mask(grid_size: int = 64, canvas_size: int = 1024, text: str = "TWI") -> np.ndarray:
    """Create the default text-based planning mask used in training and evaluation."""
    high_res_mask = generate_text_mask(text=text, canvas_size=canvas_size)
    return downsample_mask(high_res_mask, grid_size=grid_size, threshold=0.2)


class ScanPlanningEnv(gym.Env[np.ndarray, int]):
    """Minimal scan-planning environment with invalid-action masking."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        planning_mask: np.ndarray | None = None,
        grid_size: int = 64,
        text: str = "TWI",
        canvas_size: int = 1024,
        deposit_strength: float = 1.0,
        diffusion: float = 0.08,
        decay: float = 0.96,
        newly_scanned_bonus: float = 1.0,
        coverage_bonus_coef: float = 0.5,
        variance_penalty_coef: float = 0.5,
        peak_penalty_coef: float = 0.1,
        local_temp_diff_penalty_coef: float = 0.25,
        hotspot_penalty_coef: float = 0.2,
        invalid_action_penalty: float = 1.0,
        invalid_action_limit: int = 10,
        max_steps: int | None = None,
        thermal_clip: float = 3.0,
    ) -> None:
        super().__init__()
        self.grid_size = int(grid_size)
        self.text = text
        self.canvas_size = int(canvas_size)
        self.deposit_strength = float(deposit_strength)
        self.diffusion = float(diffusion)
        self.decay = float(decay)
        self.newly_scanned_bonus = float(newly_scanned_bonus)
        self.coverage_bonus_coef = float(coverage_bonus_coef)
        self.variance_penalty_coef = float(variance_penalty_coef)
        self.peak_penalty_coef = float(peak_penalty_coef)
        self.local_temp_diff_penalty_coef = float(local_temp_diff_penalty_coef)
        self.hotspot_penalty_coef = float(hotspot_penalty_coef)
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.invalid_action_limit = int(invalid_action_limit)
        self.thermal_clip = float(thermal_clip)

        self._provided_mask = None if planning_mask is None else self._prepare_mask(planning_mask)

        self.action_space = spaces.Discrete(self.grid_size * self.grid_size)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(3, self.grid_size, self.grid_size),
            dtype=np.uint8,
        )

        self.target_mask = self._initial_mask()
        self.target_cell_count = int(self.target_mask.sum())
        self.max_steps = int(max_steps) if max_steps is not None else max(self.target_cell_count * 2, 1)

        self.scanned_mask = np.zeros_like(self.target_mask, dtype=bool)
        self.thermal_field = create_empty_thermal_field(self.grid_size)
        self.executed_actions: list[tuple[int, int]] = []
        self.steps_taken = 0
        self.invalid_action_count = 0

    def _local_temperature_difference(self, row: int, col: int) -> float:
        r0 = max(0, row - 1)
        r1 = min(self.grid_size, row + 2)
        c0 = max(0, col - 1)
        c1 = min(self.grid_size, col + 2)
        neighborhood = self.thermal_field[r0:r1, c0:c1]
        return float(np.mean(np.abs(neighborhood - self.thermal_field[row, col])))

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
            r0 = max(0, row - 1)
            r1 = min(self.grid_size, row + 2)
            c0 = max(0, col - 1)
            c1 = min(self.grid_size, col + 2)
            neighbor_links += float(hotspot_mask[r0:r1, c0:c1].sum() - 1)
        return neighbor_links / max(float(len(hotspot_positions)), 1.0)

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

    def _encode_action(self, row: int, col: int) -> int:
        return row * self.grid_size + col

    def _decode_action(self, action: int) -> tuple[int, int]:
        row, col = divmod(int(action), self.grid_size)
        return row, col

    def _normalised_thermal_channel(self) -> np.ndarray:
        scaled = np.clip(self.thermal_field / max(self.thermal_clip, 1e-6), 0.0, 1.0)
        return (scaled * 255.0).astype(np.uint8)

    def _get_obs(self) -> np.ndarray:
        channels = [
            self.target_mask.astype(np.uint8) * 255,
            self.scanned_mask.astype(np.uint8) * 255,
            self._normalised_thermal_channel(),
        ]
        return np.stack(channels, axis=0)

    def action_masks(self) -> np.ndarray:
        """Return a flat mask with True only inside unscanned letter cells."""
        valid_mask = np.logical_and(self.target_mask, ~self.scanned_mask)
        return valid_mask.reshape(-1).astype(bool)

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
        self.max_steps = max(self.max_steps, self.target_cell_count)
        self.scanned_mask = np.zeros_like(self.target_mask, dtype=bool)
        self.thermal_field = create_empty_thermal_field(self.grid_size)
        self.executed_actions = []
        self.steps_taken = 0
        self.invalid_action_count = 0

        info = {
            "target_cells": self.target_cell_count,
            "valid_actions": int(self.action_masks().sum()),
        }
        return self._get_obs(), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.steps_taken += 1
        row, col = self._decode_action(action)
        valid_mask = self.action_masks()
        is_valid = bool(valid_mask[int(action)])

        reward: float
        terminated = False
        truncated = False

        if not is_valid:
            self.invalid_action_count += 1
            reward = -self.invalid_action_penalty
            if self.invalid_action_count >= self.invalid_action_limit:
                terminated = True
        else:
            self.scanned_mask[row, col] = True
            self.executed_actions.append((row, col))
            self.thermal_field = update_thermal_field(
                field=self.thermal_field,
                action=(row, col),
                deposit_strength=self.deposit_strength,
                diffusion=self.diffusion,
                decay=self.decay,
            )
            coverage_value = float(self.scanned_mask.sum() / max(self.target_cell_count, 1))
            variance_value = float(np.var(self.thermal_field))
            peak_value = float(np.max(self.thermal_field))
            local_temp_diff = self._local_temperature_difference(row, col)
            hotspot_penalty = self._hotspot_cluster_penalty()
            reward = (
                self.newly_scanned_bonus
                + self.coverage_bonus_coef * coverage_value
                - self.variance_penalty_coef * variance_value
                - self.peak_penalty_coef * peak_value
                - self.local_temp_diff_penalty_coef * local_temp_diff
                - self.hotspot_penalty_coef * hotspot_penalty
            )

        if bool(np.all(~self.action_masks())):
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
            "local_temp_diff": float(self._local_temperature_difference(row, col)) if is_valid else 0.0,
            "hotspot_penalty": float(self._hotspot_cluster_penalty()),
        }
        if terminated or truncated:
            info["metrics"] = self._terminal_info()

        return self._get_obs(), float(reward), terminated, truncated, info
