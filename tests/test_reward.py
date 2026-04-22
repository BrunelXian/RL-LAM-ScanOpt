"""Sanity checks for the Stage A composite reward helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.reward import (
    REWARD_TERM_KEYS,
    compute_reward_statistics,
    compute_reward_terms,
    local_target_preheat,
    target_heat_peak,
)


class RewardHelperTests(unittest.TestCase):
    """Small helper-level tests for target-masked reward behavior."""

    def test_target_masked_peak_ignores_background(self) -> None:
        field = np.zeros((4, 4), dtype=np.float32)
        mask = np.zeros((4, 4), dtype=bool)
        mask[1, 1] = True
        field[1, 1] = 1.0
        field[3, 3] = 99.0
        self.assertEqual(target_heat_peak(field, mask), 1.0)

    def test_reheat_handles_empty_neighborhood(self) -> None:
        field = np.zeros((4, 4), dtype=np.float32)
        mask = np.zeros((4, 4), dtype=bool)
        mask[0, 0] = True
        self.assertEqual(local_target_preheat(field, mask, center=(3, 3), window_size=1), 0.0)

    def test_reward_terms_include_required_keys_and_total(self) -> None:
        terms = compute_reward_terms(
            valid_action=True,
            coverage_event=True,
            episode_complete=False,
            peak_value=0.5,
            variance_value=0.25,
            local_preheat=0.1,
            jump_distance=0.2,
        )
        self.assertTrue(set(REWARD_TERM_KEYS).issubset(terms.keys()))
        self.assertAlmostEqual(
            terms["total"],
            sum(value for key, value in terms.items() if key != "total"),
            places=6,
        )

    def test_reward_statistics_do_not_crash_for_sparse_target(self) -> None:
        pre_heat = np.zeros((5, 5), dtype=np.float32)
        post_heat = np.zeros((5, 5), dtype=np.float32)
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = True
        stats = compute_reward_statistics(
            pre_update_heat=pre_heat,
            post_update_heat=post_heat,
            target_mask=mask,
            current_location=(2, 2),
            previous_location=None,
            reheat_window_size=3,
        )
        self.assertIn("peak_heat", stats)
        self.assertIn("heat_variance", stats)
        self.assertEqual(stats["jump_distance"], 0.0)


if __name__ == "__main__":
    unittest.main()
