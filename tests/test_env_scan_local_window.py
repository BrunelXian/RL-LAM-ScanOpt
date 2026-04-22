"""Sanity checks for the local-window scan-planning environment."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.env_scan_local_window import ScanPlanningLocalWindowEnv


class EnvScanLocalWindowTests(unittest.TestCase):
    """Basic behavior tests for the local-window environment."""

    def setUp(self) -> None:
        self.mask = np.zeros((8, 8), dtype=bool)
        self.mask[2, 2] = True
        self.mask[2, 3] = True
        self.mask[2, 4] = True
        self.mask[3, 3] = True
        self.mask[4, 3] = True
        self.env = ScanPlanningLocalWindowEnv(
            planning_mask=self.mask,
            grid_size=8,
            window_size=3,
            invalid_action_limit=3,
            max_steps=20,
        )

    def tearDown(self) -> None:
        self.env.close()

    def test_reset_reports_local_window_metadata(self) -> None:
        obs, info = self.env.reset(seed=123)
        self.assertEqual(obs.shape, (3, 8, 8))
        self.assertEqual(info["action_mode"], "local_window")
        self.assertEqual(info["window_size"], 3)
        self.assertIn("local_action_count", info)

    def test_action_catalog_is_maskable(self) -> None:
        self.env.reset(seed=123)
        masks = self.env.action_masks()
        self.assertEqual(masks.shape, (len(self.env.action_cells),))
        self.assertGreater(int(masks.sum()), 0)

    def test_decode_and_step_work(self) -> None:
        self.env.reset(seed=123)
        action = int(np.flatnonzero(self.env.action_masks())[0])
        row, col, orientation = self.env.decode_action(action)
        self.assertGreaterEqual(row, 0)
        self.assertGreaterEqual(col, 0)
        self.assertIn(orientation, (0, 1))

        _, _, terminated, truncated, info = self.env.step(action)
        self.assertIn("reward_terms", info)
        self.assertTrue(info["valid_action"])
        self.assertGreater(int(self.env.scanned_mask.sum()), 0)
        self.assertEqual(len(self.env.executed_local_actions), 1)
        self.assertFalse(terminated)
        self.assertFalse(truncated)


if __name__ == "__main__":
    unittest.main()
