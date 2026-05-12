"""Tests for the LDED 32-track FEA-teacher pool builder helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.geometry import build_lded_coupon_32track_v1
from scripts.build_fea_teacher_pool import (
    build_line_masks,
    is_valid_track_permutation,
    load_variant1_weights,
    perturb_block_swap,
    perturb_local_reversal,
    perturb_partial_random_insertion,
    perturb_stride_shuffle,
    sequence_key,
    evaluate_track_order,
)


class BuildFeaTeacherPoolLDEDTests(unittest.TestCase):
    """Sanity checks for the line-based teacher-pool migration."""

    def test_is_valid_track_permutation_accepts_full_length_order(self) -> None:
        self.assertTrue(is_valid_track_permutation(list(range(32)), 32))
        self.assertFalse(is_valid_track_permutation(list(range(31)), 32))
        self.assertFalse(is_valid_track_permutation(list(range(31)) + [30], 32))

    def test_line_based_evaluation_is_deterministic(self) -> None:
        benchmark = build_lded_coupon_32track_v1()
        reward_weights = load_variant1_weights()
        target_mask, track_masks = build_line_masks(benchmark)
        track_order = list(range(32))
        result_a = evaluate_track_order(benchmark, track_order, reward_weights, target_mask, track_masks)
        result_b = evaluate_track_order(benchmark, track_order, reward_weights, target_mask, track_masks)
        self.assertEqual(result_a["sequence_length"] if "sequence_length" in result_a else len(track_order), 32)
        self.assertAlmostEqual(result_a["proxy_score"], result_b["proxy_score"])
        self.assertAlmostEqual(result_a["peak_heat"], result_b["peak_heat"])
        self.assertAlmostEqual(result_a["total_jump"], result_b["total_jump"])

    def test_permutation_perturbations_preserve_track_set(self) -> None:
        baseline = list(range(32))
        expected = set(range(32))
        mutations = [
            perturb_local_reversal(baseline, block_size=4, seed=1),
            perturb_block_swap(baseline, block_size=4, seed=2),
            perturb_stride_shuffle(baseline, stride=3, seed=3),
            perturb_partial_random_insertion(baseline, moves=2, seed=4),
        ]
        for mutation in mutations:
            self.assertEqual(len(mutation), 32)
            self.assertEqual(set(mutation), expected)
            self.assertNotEqual(sequence_key(mutation), ())


if __name__ == "__main__":
    unittest.main()
