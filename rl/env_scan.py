"""Gymnasium environment for stripe-based scan planning inside letter-shaped regions."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from core.geometry import downsample_mask, generate_stripe_segments, generate_text_mask
from core.metrics import summarise_run
from core.reward import (
    DEFAULT_REHEAT_WINDOW_SIZE,
    DEFAULT_REWARD_WEIGHTS,
    DEFAULT_USE_SUPPORT_RISK,
    build_reward_weights,
    compute_reward_statistics,
    compute_reward_terms,
    representative_location,
    target_heat_mean,
    target_heat_peak,
    target_heat_variance,
)
from core.thermal import create_empty_thermal_field, update_thermal_field


def build_twi_mask(grid_size: int = 64, canvas_size: int = 1024, text: str = "TWI") -> np.ndarray:
    """Create the default text-based planning mask used in training and evaluation."""
    high_res_mask = generate_text_mask(text=text, canvas_size=canvas_size)
    return downsample_mask(high_res_mask, grid_size=grid_size, threshold=0.2)


class ScanPlanningEnv(gym.Env[np.ndarray, int]):
    """Stripe-based scan-planning environment with Stage A composite reward logging."""

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
        reward_weights: dict[str, float] | None = None,
        reheat_window_size: int = DEFAULT_REHEAT_WINDOW_SIZE,
        use_support_risk: bool = DEFAULT_USE_SUPPORT_RISK,
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
        self.reward_weights = build_reward_weights(reward_weights or DEFAULT_REWARD_WEIGHTS)
        self.reheat_window_size = int(reheat_window_size)
        self.use_support_risk = bool(use_support_risk)
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
        self.previous_valid_location: tuple[int, int] | None = None

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

    def _coverage_ratio(self) -> float:
        return float(np.logical_and(self.scanned_mask, self.target_mask).sum() / max(self.target_cell_count, 1))

    def _terminal_info(self) -> dict[str, Any]:
        return summarise_run(
            target_mask=self.target_mask,
            scanned_mask=self.scanned_mask,
            final_thermal=self.thermal_field,
            actions=self.executed_actions,
        )

    def _base_info(
        self,
        *,
        reward_terms: dict[str, float],
        jump_distance: float,
    ) -> dict[str, Any]:
        peak_heat = target_heat_peak(self.thermal_field, self.target_mask)
        heat_variance = target_heat_variance(self.thermal_field, self.target_mask)
        return {
            "reward_terms": reward_terms,
            "valid_action": reward_terms["invalid"] == 0.0,
            "invalid_action_count": self.invalid_action_count,
            "steps_taken": self.steps_taken,
            "remaining_valid_actions": int(self.action_masks().sum()),
            "coverage_ratio": self._coverage_ratio(),
            "peak_heat": peak_heat,
            "heat_variance": heat_variance,
            "jump_distance": float(jump_distance),
            "thermal_mean": target_heat_mean(self.thermal_field, self.target_mask),
            "thermal_peak": peak_heat,
            "thermal_variance": heat_variance,
            "stripe_count": len(self.stripes),
            "executed_stripes": len(self.executed_stripes),
        }

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
        self.previous_valid_location = None

        info = {
            "target_cells": self.target_cell_count,
            "stripe_count": len(self.stripes),
            "valid_actions": int(self.action_masks().sum()),
        }
        return self.get_observation(), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply one stripe action and return the Stage A composite reward."""
        self.steps_taken += 1
        is_valid = self.is_valid_action(action)
        terminated = False
        truncated = False
        jump_distance = 0.0

        if not is_valid:
            self.invalid_action_count += 1
            reward_terms = compute_reward_terms(
                valid_action=False,
                coverage_event=False,
                episode_complete=False,
                peak_value=0.0,
                variance_value=0.0,
                local_preheat=0.0,
                jump_distance=0.0,
                reward_weights=self.reward_weights,
                use_support_risk=self.use_support_risk,
            )
            if self.invalid_action_count >= self.invalid_action_limit:
                terminated = True
        else:
            stripe_index = int(action)
            stripe_mask = self.stripes[stripe_index]
            new_cells_mask = np.logical_and(stripe_mask, self.target_mask & ~self.scanned_mask)
            stripe_location = representative_location(new_cells_mask if new_cells_mask.any() else stripe_mask)
            pre_update_heat = self.thermal_field.copy()

            self.scanned_stripes[stripe_index] = True
            self.executed_stripes.append(stripe_index)

            for row, col in self._stripe_cells(new_cells_mask):
                self.scanned_mask[row, col] = True
                self.executed_actions.append((row, col))
                self.thermal_field = update_thermal_field(
                    field=self.thermal_field,
                    action=(row, col),
                    deposit_strength=self.deposit_strength,
                    diffusion=self.diffusion,
                    decay=self.decay,
                )

            reward_stats = compute_reward_statistics(
                pre_update_heat=pre_update_heat,
                post_update_heat=self.thermal_field,
                target_mask=self.target_mask,
                current_location=stripe_location,
                previous_location=self.previous_valid_location,
                reheat_window_size=self.reheat_window_size,
            )
            jump_distance = reward_stats["jump_distance"]
            episode_complete = bool(self.scanned_mask[self.target_mask].all())
            reward_terms = compute_reward_terms(
                valid_action=True,
                coverage_event=bool(new_cells_mask.any()),
                episode_complete=episode_complete,
                peak_value=reward_stats["peak_heat"],
                variance_value=reward_stats["heat_variance"],
                local_preheat=reward_stats["local_preheat"],
                jump_distance=reward_stats["jump_distance"],
                reward_weights=self.reward_weights,
                use_support_risk=self.use_support_risk,
            )
            if new_cells_mask.any():
                self.previous_valid_location = stripe_location

        if not self.action_masks().any():
            terminated = True
        if self.steps_taken >= self.max_steps and not terminated:
            truncated = True

        info = self._base_info(reward_terms=reward_terms, jump_distance=jump_distance)
        if terminated or truncated:
            info["metrics"] = self._terminal_info()

        return self.get_observation(), float(reward_terms["total"]), terminated, truncated, info
