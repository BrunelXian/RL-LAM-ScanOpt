"""Sanity checks for thermal field updates."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.thermal import create_empty_thermal_field, update_thermal_field


class ThermalTests(unittest.TestCase):
    """Basic thermal update tests."""

    def test_update_preserves_shape_and_finite_values(self) -> None:
        field = create_empty_thermal_field(64)
        updated = update_thermal_field(field, action=(32, 32))
        self.assertEqual(updated.shape, field.shape)
        self.assertTrue(np.isfinite(updated).all())
        self.assertGreater(float(updated.max()), 0.0)


if __name__ == "__main__":
    unittest.main()
