"""Sanity checks for the baseline rollout pipeline."""

from __future__ import annotations

import unittest

from core.geometry import downsample_mask, generate_text_mask
from core.planners.raster import plan_raster
from core.rollout import run_plan


class RolloutTests(unittest.TestCase):
    """Basic rollout result tests."""

    def test_run_plan_returns_expected_keys(self) -> None:
        mask = downsample_mask(generate_text_mask("TWI", canvas_size=256), grid_size=32)
        actions = plan_raster(mask)
        result = run_plan(mask, actions)
        expected_keys = {"target_mask", "scanned_mask", "order_map", "final_thermal", "metrics", "actions"}
        self.assertTrue(expected_keys.issubset(result.keys()))
        self.assertEqual(result["target_mask"].shape, (32, 32))
        self.assertGreaterEqual(result["metrics"]["coverage_ratio"], 0.0)


if __name__ == "__main__":
    unittest.main()
