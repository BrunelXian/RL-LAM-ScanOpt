"""Sanity checks for the masked scan-planning RL environment."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.env_scan import ScanPlanningEnv


class EnvScanTests(unittest.TestCase):
    """Basic environment behavior tests."""

    def setUp(self) -> None:
        self.mask = np.zeros((8, 8), dtype=bool)
        self.mask[1, 1] = True
        self.mask[1, 2] = True
        self.mask[2, 2] = True
        self.env = ScanPlanningEnv(planning_mask=self.mask, grid_size=8, invalid_action_limit=3, max_steps=10)

    def tearDown(self) -> None:
        self.env.close()

    def test_reset_returns_expected_observation_shape(self) -> None:
        obs, info = self.env.reset(seed=123)
        self.assertEqual(obs.shape, (3, 8, 8))
        self.assertEqual(obs.dtype, np.uint8)
        self.assertEqual(info["target_cells"], 3)
        self.assertGreater(info["stripe_count"], 0)

    def test_action_mask_has_expected_shape(self) -> None:
        self.env.reset()
        mask = self.env.action_masks()
        self.assertEqual(mask.shape, (len(self.env.stripes),))
        self.assertEqual(int(mask.sum()), len(self.env.stripes))

    def test_valid_action_updates_state(self) -> None:
        self.env.reset()
        action = 0
        _, reward, terminated, truncated, info = self.env.step(action)
        self.assertGreater(int(self.env.scanned_mask.sum()), 0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["valid_action"])
        self.assertEqual(len(self.env.executed_stripes), 1)

    def test_invalid_action_is_penalised(self) -> None:
        self.env.reset()
        invalid_action = len(self.env.stripes) + 1
        _, reward, terminated, truncated, info = self.env.step(invalid_action)
        self.assertLess(reward, 0.0)
        self.assertFalse(info["valid_action"])
        self.assertEqual(info["invalid_action_count"], 1)
        self.assertFalse(terminated)
        self.assertFalse(truncated)


if __name__ == "__main__":
    unittest.main()
