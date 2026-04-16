"""Sanity checks for mask generation and downsampling."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.geometry import downsample_mask, generate_stripe_segments, generate_text_mask


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

    def test_generate_stripe_segments_stays_inside_mask(self) -> None:
        mask = generate_text_mask("TWI", canvas_size=256)
        coarse = downsample_mask(mask, grid_size=32)
        stripes = generate_stripe_segments(coarse, grid_size=32, stripe_width=1)
        self.assertGreater(len(stripes), 0)
        combined = np.zeros_like(coarse, dtype=bool)
        for stripe in stripes:
            self.assertTrue(np.logical_and(stripe, ~coarse).sum() == 0)
            combined |= stripe
        self.assertTrue(np.array_equal(combined, coarse))


if __name__ == "__main__":
    unittest.main()
