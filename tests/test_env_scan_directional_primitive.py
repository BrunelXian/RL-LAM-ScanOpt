"""Tests for the directional primitive scan-planning environment."""

from __future__ import annotations

import unittest

from core.geometry import downsample_mask, generate_text_mask
from rl.env_scan_directional_primitive import ScanPlanningDirectionalPrimitiveEnv


def build_mask(grid_size: int = 16):
    """Create a compact test mask for directional primitive validation."""
    return downsample_mask(generate_text_mask("TWI", canvas_size=512), grid_size=grid_size, threshold=0.2)


class DirectionalPrimitiveEnvTests(unittest.TestCase):
    """Sanity tests for the explicit directional primitive action space."""

    def test_reset_exposes_directional_metadata(self) -> None:
        env = ScanPlanningDirectionalPrimitiveEnv(planning_mask=build_mask(), grid_size=16)
        _, info = env.reset(seed=7)
        self.assertEqual(info["action_mode"], "directional_primitive")
        self.assertEqual(info["primitive_size"], 2)
        self.assertGreater(info["directional_action_count"], 0)
        env.close()

    def test_action_masks_are_non_empty(self) -> None:
        env = ScanPlanningDirectionalPrimitiveEnv(planning_mask=build_mask(), grid_size=16)
        env.reset(seed=11)
        masks = env.action_masks()
        self.assertEqual(masks.dtype, bool)
        self.assertTrue(masks.any())
        env.close()

    def test_valid_action_updates_state_and_logs_reward_terms(self) -> None:
        env = ScanPlanningDirectionalPrimitiveEnv(planning_mask=build_mask(), grid_size=16)
        env.reset(seed=19)
        action = int(env.action_masks().nonzero()[0][0])
        _, reward, terminated, truncated, info = env.step(action)
        self.assertIsInstance(reward, float)
        self.assertFalse(terminated and truncated)
        self.assertIn("reward_terms", info)
        self.assertGreater(info["coverage_ratio"], 0.0)
        self.assertGreater(len(env.executed_actions), 0)
        env.close()


if __name__ == "__main__":
    unittest.main()
