"""Gymnasium environment for strict 2-cell local primitive scan planning."""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

from core.reward import (
    compute_reward_statistics,
    compute_reward_terms,
    representative_location,
    target_heat_mean,
    target_heat_peak,
    target_heat_variance,
)
from core.thermal import update_thermal_field
from rl.env_scan import ScanPlanningEnv


class ScanPlanningLocalPrimitiveEnv(ScanPlanningEnv):
    """Local primitive variant using strict 2-cell scan actions.

    One action is defined by:
    - anchor cell (row, col), restricted to target cells
    - direction: right or down

    The executed cells are the target-masked cells covered by:
    - right: (row, col), (row, col + 1)
    - down: (row, col), (row + 1, col)
    """

    DIRECTION_NAMES = ("right", "down")

    def __init__(
        self,
        *args: Any,
        max_steps: int | None = None,
        **kwargs: Any,
    ) -> None:
        self._requested_max_steps = max_steps
        super().__init__(*args, max_steps=max_steps, **kwargs)
        self._rebuild_local_actions()

    def _build_action_cells(
        self,
        anchor: tuple[int, int],
        direction_index: int,
    ) -> list[tuple[int, int]]:
        """Return target-masked cells covered by one 2-cell primitive."""
        row, col = anchor
        candidates = [(row, col)]
        if direction_index == 0:
            candidates.append((row, col + 1))
        else:
            candidates.append((row + 1, col))

        cells: list[tuple[int, int]] = []
        for new_row, new_col in candidates:
            if not (0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size):
                continue
            if not self.target_mask[new_row, new_col]:
                continue
            cells.append((int(new_row), int(new_col)))
        return cells

    def _rebuild_local_actions(self) -> None:
        """Recompute the local primitive action catalog."""
        self.action_cells: list[list[tuple[int, int]]] = []
        self.action_masks_catalog: list[np.ndarray] = []
        self.action_specs: list[dict[str, int | str]] = []
        self.action_lookup: dict[tuple[int, int, int], int] = {}
        seen_static_masks: set[tuple[tuple[int, int], ...]] = set()

        target_rows, target_cols = np.nonzero(self.target_mask)
        for row, col in zip(target_rows, target_cols):
            anchor = (int(row), int(col))
            for direction_index, direction_name in enumerate(self.DIRECTION_NAMES):
                cells = self._build_action_cells(anchor, direction_index)
                if not cells:
                    continue
                mask_key = tuple(sorted(cells))
                if mask_key in seen_static_masks:
                    continue
                seen_static_masks.add(mask_key)

                action_mask = np.zeros_like(self.target_mask, dtype=bool)
                for cell_row, cell_col in cells:
                    action_mask[cell_row, cell_col] = True
                action_index = len(self.action_cells)
                self.action_cells.append(cells)
                self.action_masks_catalog.append(action_mask)
                self.action_specs.append(
                    {
                        "anchor_row": int(row),
                        "anchor_col": int(col),
                        "direction_index": int(direction_index),
                        "direction": direction_name,
                    }
                )
                self.action_lookup[(int(row), int(col), int(direction_index))] = action_index

        self.completed_actions = np.zeros(len(self.action_cells), dtype=bool)
        self.executed_local_actions: list[int] = []
        self.action_space = spaces.Discrete(max(len(self.action_cells), 1))
        if self._requested_max_steps is None:
            self.max_steps = max(int(self.target_mask.sum()) * 2, 1)
        else:
            self.max_steps = int(self._requested_max_steps)

    def _remaining_cells_for_action(self, action: int) -> list[tuple[int, int]]:
        """Return currently executable cells for one local primitive action."""
        return [
            cell
            for cell in self.action_cells[int(action)]
            if self.target_mask[cell[0], cell[1]] and not self.scanned_mask[cell[0], cell[1]]
        ]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment and rebuild the local primitive catalog."""
        observation, info = super().reset(seed=seed, options=options)
        self._rebuild_local_actions()
        info.update(
            {
                "action_mode": "local_primitive",
                "primitive_size": 2,
                "local_action_count": len(self.action_cells),
                "valid_actions": int(self.action_masks().sum()),
            }
        )
        return observation, info

    def decode_action(self, action: int) -> tuple[int, int, int]:
        """Return ``(anchor_row, anchor_col, direction_index)`` for one action."""
        spec = self.action_specs[int(action)]
        return int(spec["anchor_row"]), int(spec["anchor_col"]), int(spec["direction_index"])

    def is_valid_action(self, action: int) -> bool:
        """Return True only when the primitive can still affect unscanned target cells."""
        if not 0 <= int(action) < len(self.action_cells):
            return False
        remaining_cells = self._remaining_cells_for_action(int(action))
        self.completed_actions[int(action)] = not bool(remaining_cells)
        return bool(remaining_cells)

    def action_masks(self) -> np.ndarray:
        """Return a flat mask over the local primitive action catalog."""
        masks = np.zeros(len(self.action_cells), dtype=bool)
        for index in range(len(self.action_cells)):
            masks[index] = self.is_valid_action(index)
        return masks

    def _base_info(
        self,
        *,
        reward_terms: dict[str, float],
        jump_distance: float,
    ) -> dict[str, Any]:
        """Return Stage A logging fields plus local primitive metadata."""
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
            "action_mode": "local_primitive",
            "primitive_size": 2,
            "local_action_count": len(self.action_cells),
            "executed_local_actions": len(self.executed_local_actions),
        }

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply one 2-cell local primitive and return the Stage A composite reward."""
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
            action_index = int(action)
            local_cells = self._remaining_cells_for_action(action_index)
            local_mask = np.zeros_like(self.target_mask, dtype=bool)
            for row, col in local_cells:
                local_mask[row, col] = True
            fallback_mask = self.action_masks_catalog[action_index]
            local_location = representative_location(local_mask if local_mask.any() else fallback_mask)
            pre_update_heat = self.thermal_field.copy()

            self.executed_local_actions.append(action_index)
            for row, col in local_cells:
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
                current_location=local_location,
                previous_location=self.previous_valid_location,
                reheat_window_size=self.reheat_window_size,
            )
            jump_distance = reward_stats["jump_distance"]
            episode_complete = bool(self.scanned_mask[self.target_mask].all())
            reward_terms = compute_reward_terms(
                valid_action=True,
                coverage_event=bool(local_cells),
                episode_complete=episode_complete,
                peak_value=reward_stats["peak_heat"],
                variance_value=reward_stats["heat_variance"],
                local_preheat=reward_stats["local_preheat"],
                jump_distance=reward_stats["jump_distance"],
                reward_weights=self.reward_weights,
                use_support_risk=self.use_support_risk,
            )
            if local_cells:
                self.previous_valid_location = local_location

        if not self.action_masks().any():
            terminated = True
        if self.steps_taken >= self.max_steps and not terminated:
            truncated = True

        info = self._base_info(reward_terms=reward_terms, jump_distance=jump_distance)
        if terminated or truncated:
            info["metrics"] = self._terminal_info()
        return self.get_observation(), float(reward_terms["total"]), terminated, truncated, info
