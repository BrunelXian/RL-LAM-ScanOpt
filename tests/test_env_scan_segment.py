"""Sanity checks for segment-based scan-planning environments."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.env_scan_segment import ScanPlanningSegmentEnv, ScanPlanningVariableSegmentEnv


class EnvScanSegmentTests(unittest.TestCase):
    """Basic behavior tests for fixed and variable segment environments."""

    def setUp(self) -> None:
        self.mask = np.zeros((8, 8), dtype=bool)
        self.mask[1, 1] = True
        self.mask[1, 2] = True
        self.mask[1, 3] = True
        self.mask[2, 2] = True
        self.fixed_env = ScanPlanningSegmentEnv(
            planning_mask=self.mask,
            grid_size=8,
            segments_per_stripe=6,
            invalid_action_limit=3,
            max_steps=20,
        )
        self.variable_env = ScanPlanningVariableSegmentEnv(
            planning_mask=self.mask,
            grid_size=8,
            min_segment_length=2,
            max_segment_length=4,
            invalid_action_limit=3,
            max_steps=20,
        )

    def tearDown(self) -> None:
        self.fixed_env.close()
        self.variable_env.close()

    def test_reset_reports_segment_metadata(self) -> None:
        obs, info = self.fixed_env.reset(seed=123)
        self.assertEqual(obs.shape, (3, 8, 8))
        self.assertIn("segment_count", info)
        self.assertEqual(info["segments_per_stripe"], 6)
        self.assertEqual(info["action_mode"], "fixed")

    def test_empty_segments_are_masked(self) -> None:
        self.fixed_env.reset()
        masks = self.fixed_env.action_masks()
        self.assertEqual(masks.shape, (len(self.fixed_env.segment_cells),))
        self.assertGreater(int((~masks).sum()), 0)

    def test_valid_segment_action_updates_state(self) -> None:
        self.fixed_env.reset()
        action = int(np.flatnonzero(self.fixed_env.action_masks())[0])
        _, _, terminated, truncated, info = self.fixed_env.step(action)
        self.assertGreater(int(self.fixed_env.scanned_mask.sum()), 0)
        self.assertTrue(info["valid_action"])
        self.assertIn("reward_terms", info)
        self.assertEqual(len(self.fixed_env.executed_segments), 1)
        self.assertFalse(terminated)
        self.assertFalse(truncated)

    def test_multiple_segment_counts_reset_cleanly(self) -> None:
        for segment_count in (4, 6, 8):
            env = ScanPlanningSegmentEnv(
                planning_mask=self.mask,
                grid_size=8,
                segments_per_stripe=segment_count,
                invalid_action_limit=3,
                max_steps=20,
            )
            obs, info = env.reset(seed=123)
            self.assertEqual(obs.shape, (3, 8, 8))
            self.assertEqual(info["segments_per_stripe"], segment_count)
            self.assertEqual(env.action_masks().shape, (len(env.segment_cells),))
            env.close()

    def test_variable_env_reports_action_mode(self) -> None:
        obs, info = self.variable_env.reset(seed=123)
        self.assertEqual(obs.shape, (3, 8, 8))
        self.assertEqual(info["action_mode"], "variable_length")
        self.assertEqual(info["min_segment_length"], 2)
        self.assertEqual(info["max_segment_length"], 4)

    def test_variable_env_masks_and_decodes_actions(self) -> None:
        self.variable_env.reset(seed=123)
        masks = self.variable_env.action_masks()
        self.assertEqual(masks.shape, (len(self.variable_env.segment_cells),))
        action = int(np.flatnonzero(masks)[0])
        stripe_id, start_cell, length = self.variable_env.decode_action(action)
        self.assertGreaterEqual(length, 2)
        self.assertGreaterEqual(start_cell, 0)
        self.assertGreaterEqual(stripe_id, 0)

    def test_variable_segment_action_updates_state(self) -> None:
        self.variable_env.reset(seed=123)
        action = int(np.flatnonzero(self.variable_env.action_masks())[0])
        _, _, terminated, truncated, info = self.variable_env.step(action)
        self.assertGreater(int(self.variable_env.scanned_mask.sum()), 0)
        self.assertTrue(info["valid_action"])
        self.assertIn("reward_terms", info)
        self.assertEqual(len(self.variable_env.executed_segments), 1)
        self.assertFalse(terminated)
        self.assertFalse(truncated)


if __name__ == "__main__":
    unittest.main()
