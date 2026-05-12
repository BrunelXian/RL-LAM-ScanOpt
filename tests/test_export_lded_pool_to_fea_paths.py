"""Tests for exporting the LDED 32-track teacher pool into FEA scan paths."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.geometry import build_lded_coupon_32track_v1
from scripts.export_lded_pool_to_fea_paths import (
    ExportSettings,
    build_scan_path_rows,
    export_pool,
    load_pool_manifest,
    track_center_x_mm,
)


class ExportLDEDPoolToFeaPathsTests(unittest.TestCase):
    """Sanity checks for the first LDED FEA export adapter."""

    def test_single_track_order_exports_32_rows(self) -> None:
        benchmark = build_lded_coupon_32track_v1()
        rows = build_scan_path_rows(
            trajectory_id="raster_left_to_right",
            track_order=list(range(32)),
            benchmark=benchmark,
            settings=ExportSettings(),
            source_sequence_file="dummy.json",
        )
        self.assertEqual(len(rows), 32)
        self.assertTrue(all(row["event_type"] == "deposition_track" for row in rows))

    def test_raster_left_to_right_centers_increase(self) -> None:
        benchmark = build_lded_coupon_32track_v1()
        rows = build_scan_path_rows(
            trajectory_id="raster_left_to_right",
            track_order=list(range(32)),
            benchmark=benchmark,
            settings=ExportSettings(),
            source_sequence_file="dummy.json",
        )
        centers = [float(row["x_start_mm"]) for row in rows]
        self.assertEqual(centers, sorted(centers))

    def test_raster_right_to_left_centers_decrease(self) -> None:
        benchmark = build_lded_coupon_32track_v1()
        rows = build_scan_path_rows(
            trajectory_id="raster_right_to_left",
            track_order=list(range(31, -1, -1)),
            benchmark=benchmark,
            settings=ExportSettings(),
            source_sequence_file="dummy.json",
        )
        centers = [float(row["x_start_mm"]) for row in rows]
        self.assertEqual(centers, sorted(centers, reverse=True))

    def test_exported_coordinates_stay_inside_plane_and_patch(self) -> None:
        benchmark = build_lded_coupon_32track_v1()
        rows = build_scan_path_rows(
            trajectory_id="center_out",
            track_order=[15, 16] + [track_id for track_id in range(32) if track_id not in {15, 16}],
            benchmark=benchmark,
            settings=ExportSettings(),
            source_sequence_file="dummy.json",
        )
        for row in rows:
            for x in (float(row["x_start_mm"]), float(row["x_end_mm"])):
                self.assertGreaterEqual(x, 0.0)
                self.assertLessEqual(x, benchmark.plane_width_mm)
                self.assertGreaterEqual(x, benchmark.patch_x_min_mm)
                self.assertLessEqual(x, benchmark.patch_x_max_mm)
            for y in (float(row["y_start_mm"]), float(row["y_end_mm"])):
                self.assertGreaterEqual(y, 0.0)
                self.assertLessEqual(y, benchmark.plane_height_mm)
                self.assertGreaterEqual(y, benchmark.patch_y_min_mm)
                self.assertLessEqual(y, benchmark.patch_y_max_mm)

    def test_export_pool_manifest_valid_export_all_true(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            export_dir = Path(temp_dir)
            exported_rows, failed_exports = export_pool(export_dir=export_dir)
            self.assertEqual(len(failed_exports), 0)
            self.assertGreater(len(exported_rows), 0)
            manifest_csv = export_dir / "fea_export_manifest.csv"
            with manifest_csv.open("r", encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertTrue(all(row["valid_export"] == "True" for row in rows))
            sample_scan_path = Path(rows[0]["scan_path_csv"])
            self.assertTrue(sample_scan_path.exists())
            with sample_scan_path.open("r", encoding="utf-8", newline="") as file:
                scan_rows = list(csv.DictReader(file))
            self.assertEqual(len(scan_rows), 32)


if __name__ == "__main__":
    unittest.main()
