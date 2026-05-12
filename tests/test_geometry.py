"""Sanity checks for mask generation and downsampling."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.geometry import (
    build_lded_coupon_32track_baselines,
    build_lded_coupon_32track_v1,
    downsample_mask,
    generate_stripe_segments,
    generate_text_mask,
)


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

    def test_build_lded_coupon_32track_v1_has_expected_geometry(self) -> None:
        benchmark = build_lded_coupon_32track_v1()
        self.assertEqual(benchmark.benchmark_name, "lded_coupon_32track_v1")
        self.assertEqual(benchmark.track_count, 32)
        self.assertEqual(len(benchmark.tracks), 32)
        self.assertAlmostEqual(benchmark.plane_width_mm, 100.0)
        self.assertAlmostEqual(benchmark.plane_height_mm, 40.0)
        self.assertAlmostEqual(benchmark.patch_x_min_mm, 2.0)
        self.assertAlmostEqual(benchmark.patch_x_max_mm, 98.0)
        self.assertAlmostEqual(benchmark.patch_y_min_mm, 2.0)
        self.assertAlmostEqual(benchmark.patch_y_max_mm, 38.0)
        self.assertAlmostEqual(benchmark.tracks[0].x_start_mm, 2.0)
        self.assertAlmostEqual(benchmark.tracks[-1].x_end_mm, 98.0)

    def test_lded_coupon_baselines_are_valid_track_permutations(self) -> None:
        baselines = build_lded_coupon_32track_baselines(random_seeds=(0, 7))
        expected_set = set(range(32))
        self.assertIn("raster_left_to_right", baselines)
        self.assertIn("center_out", baselines)
        self.assertIn("random_seed_0", baselines)
        for sequence in baselines.values():
            self.assertEqual(len(sequence), 32)
            self.assertEqual(set(sequence), expected_set)


if __name__ == "__main__":
    unittest.main()
