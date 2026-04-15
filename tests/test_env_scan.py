"""Sanity checks for the masked scan-planning RL environment."""

from __future__ import annotations

import unittest

import numpy as np

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

    def test_action_mask_has_expected_shape(self) -> None:
        self.env.reset()
        mask = self.env.action_masks()
        self.assertEqual(mask.shape, (64,))
        self.assertEqual(int(mask.sum()), 3)

    def test_valid_action_updates_state(self) -> None:
        self.env.reset()
        action = 1 * 8 + 1
        _, reward, terminated, truncated, info = self.env.step(action)
        self.assertTrue(self.env.scanned_mask[1, 1])
        self.assertGreater(reward, -1.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["valid_action"])

    def test_invalid_action_is_penalised(self) -> None:
        self.env.reset()
        invalid_action = 0
        _, reward, terminated, truncated, info = self.env.step(invalid_action)
        self.assertLess(reward, 0.0)
        self.assertFalse(info["valid_action"])
        self.assertEqual(info["invalid_action_count"], 1)
        self.assertFalse(terminated)
        self.assertFalse(truncated)


if __name__ == "__main__":
    unittest.main()
