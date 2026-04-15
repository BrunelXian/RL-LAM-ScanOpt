"""Sanity checks for mask generation and downsampling."""

from __future__ import annotations

import unittest

from core.geometry import downsample_mask, generate_text_mask


class GeometryTests(unittest.TestCase):
    """Basic geometry creation tests."""

    def test_generate_text_mask_is_non_empty(self) -> None:
        mask = generate_text_mask("TWI", canvas_size=256)
        self.assertGreater(int(mask.sum()), 0)

    def test_downsample_mask_has_expected_shape(self) -> None:
        mask = generate_text_mask("TWI", canvas_size=256)
        coarse = downsample_mask(mask, grid_size=64)
        self.assertEqual(coarse.shape, (64, 64))
        self.assertGreater(int(coarse.sum()), 0)


if __name__ == "__main__":
    unittest.main()
