"""Gymnasium environments for segment-based scan planning inside letter-shaped regions."""

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


class ScanPlanningSegmentEnv(ScanPlanningEnv):
    """Segment-based variant of the stripe environment.

    The environment now supports two action-catalog modes:
    - ``fixed``: the historical balanced split into ``segments_per_stripe`` parts
    - ``variable_length``: each action is ``(stripe_id, start_cell, length)``
      where ``length`` is chosen from ``min_segment_length`` to
      ``max_segment_length`` and the executed cells are the currently unscanned
      cells that fall inside that stripe window.
    """

    def __init__(
        self,
        *args: Any,
        segments_per_stripe: int = 6,
        action_mode: str = "fixed",
        min_segment_length: int = 2,
        max_segment_length: int = 8,
        max_steps: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.segments_per_stripe = int(segments_per_stripe)
        self.action_mode = str(action_mode)
        self.min_segment_length = int(min_segment_length)
        self.max_segment_length = int(max_segment_length)
        self._requested_max_steps = max_steps
        super().__init__(*args, max_steps=max_steps, **kwargs)
        self._rebuild_segments()

    def _validate_mode(self) -> None:
        """Validate segment-mode parameters early."""
        if self.action_mode not in {"fixed", "variable_length"}:
            raise ValueError("action_mode must be 'fixed' or 'variable_length'")
        if self.segments_per_stripe <= 0:
            raise ValueError("segments_per_stripe must be positive")
        if self.min_segment_length <= 0 or self.max_segment_length <= 0:
            raise ValueError("segment lengths must be positive")
        if self.min_segment_length > self.max_segment_length:
            raise ValueError("min_segment_length must be <= max_segment_length")

    def _split_stripe_cells(self, cells: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
        """Split one stripe cell list into balanced fixed segments."""
        cell_groups = np.array_split(np.asarray(cells, dtype=np.int32), self.segments_per_stripe)
        segments: list[list[tuple[int, int]]] = []
        for group in cell_groups:
            if len(group) == 0:
                segments.append([])
            else:
                segments.append([(int(row), int(col)) for row, col in group])
        return segments

    def _build_fixed_actions(self, ordered_cells: list[tuple[int, int]], stripe_index: int) -> None:
        """Populate the catalog for the historical fixed-segment formulation."""
        segment_groups = self._split_stripe_cells(ordered_cells)
        for segment_index, segment_cells in enumerate(segment_groups):
            self._append_action(
                stripe_index=stripe_index,
                start_cell_index=segment_index,
                requested_length=len(segment_cells),
                segment_cells=segment_cells,
                fixed_segment_index=segment_index,
            )

    def _build_variable_actions(self, ordered_cells: list[tuple[int, int]], stripe_index: int) -> None:
        """Populate the catalog for variable-length stripe windows."""
        for start_index in range(len(ordered_cells)):
            for requested_length in range(self.min_segment_length, self.max_segment_length + 1):
                segment_cells = ordered_cells[start_index : start_index + requested_length]
                self._append_action(
                    stripe_index=stripe_index,
                    start_cell_index=start_index,
                    requested_length=requested_length,
                    segment_cells=segment_cells,
                    fixed_segment_index=None,
                )

    def _append_action(
        self,
        *,
        stripe_index: int,
        start_cell_index: int,
        requested_length: int,
        segment_cells: list[tuple[int, int]],
        fixed_segment_index: int | None,
    ) -> None:
        """Append one action definition to the segment-action catalog."""
        segment_mask = np.zeros_like(self.target_mask, dtype=bool)
        for row, col in segment_cells:
            segment_mask[row, col] = True
        action_index = len(self.segment_cells)
        self.segment_cells.append(segment_cells)
        self.segment_masks.append(segment_mask)
        self.segment_specs.append(
            {
                "stripe_id": int(stripe_index),
                "start_cell": int(start_cell_index),
                "length": int(requested_length),
                "fixed_segment_index": fixed_segment_index,
            }
        )
        self.segment_to_stripe.append((int(stripe_index), int(start_cell_index)))
        if self.action_mode == "fixed":
            for cell in segment_cells:
                self.cell_to_segment_action[cell] = action_index
        else:
            self.action_lookup[(int(stripe_index), int(start_cell_index), int(requested_length))] = action_index

    def _estimate_default_max_steps(self) -> int:
        """Estimate a practical default max-step budget for the active action mode."""
        target_cells = int(self.target_mask.sum())
        if self.action_mode == "fixed":
            valid_action_count = int(sum(1 for cells in self.segment_cells if cells))
            return max(valid_action_count * 2, 1)
        min_effective_length = max(1, self.min_segment_length)
        lower_bound_actions = int(np.ceil(target_cells / min_effective_length))
        return max(lower_bound_actions * 2, 1)

    def _rebuild_segments(self) -> None:
        """Recompute segment actions and their action-index mappings."""
        self._validate_mode()
        self.segment_cells: list[list[tuple[int, int]]] = []
        self.segment_masks: list[np.ndarray] = []
        self.segment_specs: list[dict[str, int | None]] = []
        self.segment_to_stripe: list[tuple[int, int]] = []
        self.cell_to_segment_action: dict[tuple[int, int], int] = {}
        self.action_lookup: dict[tuple[int, int, int], int] = {}
        self.stripe_cell_orders: list[list[tuple[int, int]]] = []
        self.cell_to_stripe_position: dict[tuple[int, int], tuple[int, int]] = {}

        for stripe_index, stripe in enumerate(self.stripes):
            ordered_cells = self._stripe_cells(stripe)
            self.stripe_cell_orders.append(ordered_cells)
            for position, cell in enumerate(ordered_cells):
                self.cell_to_stripe_position[cell] = (stripe_index, position)
            if self.action_mode == "fixed":
                self._build_fixed_actions(ordered_cells, stripe_index)
            else:
                self._build_variable_actions(ordered_cells, stripe_index)

        self.completed_segments = np.zeros(len(self.segment_cells), dtype=bool)
        self.executed_segments: list[int] = []
        self.action_space = spaces.Discrete(max(len(self.segment_cells), 1))
        if self._requested_max_steps is None:
            self.max_steps = self._estimate_default_max_steps()
        else:
            self.max_steps = int(self._requested_max_steps)

    def _remaining_cells_for_action(self, action: int) -> list[tuple[int, int]]:
        """Return the currently executable cells for one action."""
        return [
            cell
            for cell in self.segment_cells[int(action)]
            if self.target_mask[cell[0], cell[1]] and not self.scanned_mask[cell[0], cell[1]]
        ]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the segment environment and rebuild the action catalog."""
        observation, info = super().reset(seed=seed, options=options)
        self._rebuild_segments()
        info.update(
            {
                "segment_count": len(self.segment_cells),
                "segments_per_stripe": self.segments_per_stripe,
                "action_mode": self.action_mode,
                "min_segment_length": self.min_segment_length,
                "max_segment_length": self.max_segment_length,
                "valid_actions": int(self.action_masks().sum()),
            }
        )
        return observation, info

    def decode_action(self, action: int) -> tuple[int, int, int]:
        """Return ``(stripe_id, start_cell, length)`` for a flat action index."""
        spec = self.segment_specs[int(action)]
        return int(spec["stripe_id"]), int(spec["start_cell"]), int(spec["length"])

    def is_valid_action(self, action: int) -> bool:
        """Return True only when the action points to a non-empty unfinished window."""
        if not 0 <= int(action) < len(self.segment_cells):
            return False
        remaining_cells = self._remaining_cells_for_action(int(action))
        self.completed_segments[int(action)] = not bool(remaining_cells)
        return bool(remaining_cells)

    def action_masks(self) -> np.ndarray:
        """Return a flat mask with True only for executable actions."""
        masks = np.zeros(len(self.segment_cells), dtype=bool)
        for index in range(len(self.segment_cells)):
            masks[index] = self.is_valid_action(index)
        return masks

    def _base_info(
        self,
        *,
        reward_terms: dict[str, float],
        jump_distance: float,
    ) -> dict[str, Any]:
        """Return logging info aligned with the stripe environment plus action metadata."""
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
            "segment_count": len(self.segment_cells),
            "segments_per_stripe": self.segments_per_stripe,
            "action_mode": self.action_mode,
            "min_segment_length": self.min_segment_length,
            "max_segment_length": self.max_segment_length,
            "executed_stripes": len({self.segment_specs[index]["stripe_id"] for index in self.executed_segments}),
            "executed_segments": len(self.executed_segments),
        }

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply one action-catalog window and return the Stage A composite reward."""
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
            segment_cells = self._remaining_cells_for_action(action_index)
            segment_mask = np.zeros_like(self.target_mask, dtype=bool)
            for row, col in segment_cells:
                segment_mask[row, col] = True
            fallback_mask = self.segment_masks[action_index]
            segment_location = representative_location(segment_mask if segment_mask.any() else fallback_mask)
            pre_update_heat = self.thermal_field.copy()

            self.executed_segments.append(action_index)
            self.executed_stripes.append(int(self.segment_specs[action_index]["stripe_id"]))

            for row, col in segment_cells:
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
                current_location=segment_location,
                previous_location=self.previous_valid_location,
                reheat_window_size=self.reheat_window_size,
            )
            jump_distance = reward_stats["jump_distance"]
            episode_complete = bool(self.scanned_mask[self.target_mask].all())
            reward_terms = compute_reward_terms(
                valid_action=True,
                coverage_event=bool(segment_cells),
                episode_complete=episode_complete,
                peak_value=reward_stats["peak_heat"],
                variance_value=reward_stats["heat_variance"],
                local_preheat=reward_stats["local_preheat"],
                jump_distance=reward_stats["jump_distance"],
                reward_weights=self.reward_weights,
                use_support_risk=self.use_support_risk,
            )
            if segment_cells:
                self.previous_valid_location = segment_location

        if not self.action_masks().any():
            terminated = True
        if self.steps_taken >= self.max_steps and not terminated:
            truncated = True

        info = self._base_info(reward_terms=reward_terms, jump_distance=jump_distance)
        if terminated or truncated:
            info["metrics"] = self._terminal_info()

        return self.get_observation(), float(reward_terms["total"]), terminated, truncated, info


class ScanPlanningVariableSegmentEnv(ScanPlanningSegmentEnv):
    """Convenience wrapper for the variable-length action formulation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("action_mode", "variable_length")
        super().__init__(*args, **kwargs)
