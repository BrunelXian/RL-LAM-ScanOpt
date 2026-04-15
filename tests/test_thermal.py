"""Sanity checks for thermal field updates."""

from __future__ import annotations

import unittest

import numpy as np

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
