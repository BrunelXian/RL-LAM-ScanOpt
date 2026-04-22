"""Sanity checks for stronger deterministic baseline planners."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.planners.checkerboard import plan_checkerboard
from core.planners.cool_first import plan_cool_first
from core.planners.distance_aware_cool_first import plan_distance_aware_cool_first


class PlannerTests(unittest.TestCase):
    """Planner-level checks for determinism and mask handling."""

    def setUp(self) -> None:
        self.mask = np.zeros((6, 6), dtype=bool)
        self.mask[1, 1:4] = True
        self.mask[2, 2] = True
        self.mask[3, 1] = True
        self.mask[4, 3:5] = True

    def test_cool_first_returns_valid_full_coverage_plan(self) -> None:
        actions = plan_cool_first(self.mask)
        self.assertEqual(len(actions), int(self.mask.sum()))
        self.assertEqual(set(actions), {(int(r), int(c)) for r, c in np.argwhere(self.mask)})

    def test_checkerboard_handles_irregular_mask(self) -> None:
        actions = plan_checkerboard(self.mask)
        self.assertEqual(len(actions), int(self.mask.sum()))
        even_prefix_length = sum(1 for row, col in actions if (row + col) % 2 == 0)
        self.assertTrue(all((row + col) % 2 == 0 for row, col in actions[:even_prefix_length]))

    def test_cool_first_is_deterministic(self) -> None:
        self.assertEqual(plan_cool_first(self.mask), plan_cool_first(self.mask))

    def test_distance_aware_cool_first_is_deterministic(self) -> None:
        self.assertEqual(
            plan_distance_aware_cool_first(self.mask),
            plan_distance_aware_cool_first(self.mask),
        )


if __name__ == "__main__":
    unittest.main()
