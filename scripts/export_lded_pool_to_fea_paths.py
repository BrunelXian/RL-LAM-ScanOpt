"""Export the LDED 32-track teacher pool into FEA-friendly scan-path files."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.geometry import LDEDCouponBenchmark, build_lded_coupon_32track_v1


POOL_DIR = PROJECT_ROOT / "assets" / "fea_teacher_pool_lded_32track"
POOL_MANIFEST_JSON = POOL_DIR / "fea_teacher_pool_manifest.json"
SEQUENCE_DIR = POOL_DIR / "sequences"
EXPORT_DIR = POOL_DIR / "fea_exports"
EXPORT_MANIFEST_CSV = EXPORT_DIR / "fea_export_manifest.csv"
EXPORT_MANIFEST_JSON = EXPORT_DIR / "fea_export_manifest.json"
EXPORT_SUMMARY_TXT = EXPORT_DIR / "fea_export_summary.txt"

SCAN_SPEED_MM_S = 10.0
NOMINAL_POWER_W = 1000.0
DWELL_TIME_S_BETWEEN_TRACKS = 0.0
Z_LEVEL_MM = 0.0


@dataclass(frozen=True)
class ExportSettings:
    """Placeholder process parameters for the first FEA export version."""

    scan_speed_mm_s: float = SCAN_SPEED_MM_S
    nominal_power_w: float = NOMINAL_POWER_W
    dwell_time_s_between_tracks: float = DWELL_TIME_S_BETWEEN_TRACKS
    z_level_mm: float = Z_LEVEL_MM

    @property
    def track_duration_s(self) -> float:
        """Return deposition duration for one 36 mm track."""
        benchmark = build_lded_coupon_32track_v1()
        return benchmark.track_length_mm / self.scan_speed_mm_s


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV with stable field order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def write_json(path: Path, payload: Any) -> None:
    """Write JSON payload with UTF-8 formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, lines: list[str]) -> None:
    """Write plain-text summary lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_pool_manifest(path: Path = POOL_MANIFEST_JSON) -> dict[str, Any]:
    """Load the line-based teacher pool manifest."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_sequence_payload(sequence_file: str | Path) -> dict[str, Any]:
    """Load one track-order payload JSON."""
    path = Path(sequence_file)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return json.loads(path.read_text(encoding="utf-8"))


def is_valid_track_permutation(track_order: list[int], track_count: int) -> bool:
    """Return whether a track order is a full permutation."""
    return len(track_order) == track_count and set(int(track_id) for track_id in track_order) == set(range(track_count))


def track_center_x_mm(track_id: int, benchmark: LDEDCouponBenchmark) -> float:
    """Return the center x coordinate for one vertical track."""
    return float(benchmark.patch_x_min_mm + 0.5 * benchmark.track_width_mm + benchmark.track_pitch_mm * int(track_id))


def build_scan_path_rows(
    trajectory_id: str,
    track_order: list[int],
    benchmark: LDEDCouponBenchmark,
    settings: ExportSettings,
    source_sequence_file: str,
) -> list[dict[str, Any]]:
    """Convert one 32-track order into a deposition-only scan path."""
    if not is_valid_track_permutation(track_order, benchmark.track_count):
        raise ValueError("track_order must be a length-32 permutation")

    rows: list[dict[str, Any]] = []
    current_time_s = 0.0
    duration_s = benchmark.track_length_mm / settings.scan_speed_mm_s

    for order_index, track_id in enumerate(track_order):
        center_x = track_center_x_mm(int(track_id), benchmark)
        t_start_s = current_time_s
        t_end_s = t_start_s + duration_s
        rows.append(
            {
                "trajectory_id": trajectory_id,
                "segment_id": f"track_{int(track_id):02d}",
                "order_index": int(order_index),
                "track_id": int(track_id),
                "event_type": "deposition_track",
                "x_start_mm": center_x,
                "y_start_mm": float(benchmark.patch_y_min_mm),
                "z_start_mm": float(settings.z_level_mm),
                "x_end_mm": center_x,
                "y_end_mm": float(benchmark.patch_y_max_mm),
                "z_end_mm": float(settings.z_level_mm),
                "track_width_mm": float(benchmark.track_width_mm),
                "track_length_mm": float(benchmark.track_length_mm),
                "t_start_s": float(t_start_s),
                "t_end_s": float(t_end_s),
                "duration_s": float(duration_s),
                "laser_on": 1,
                "scan_speed_mm_s": float(settings.scan_speed_mm_s),
                "nominal_power_w": float(settings.nominal_power_w),
                "source_sequence_file": str(source_sequence_file),
            }
        )
        current_time_s = t_end_s + settings.dwell_time_s_between_tracks

    return rows


def export_one_trajectory(
    trajectory_entry: dict[str, Any],
    benchmark: LDEDCouponBenchmark,
    settings: ExportSettings,
    export_dir: Path,
) -> dict[str, Any]:
    """Export one teacher-pool trajectory into CSV and metadata JSON."""
    sequence_file = trajectory_entry["sequence_file"]
    payload = load_sequence_payload(sequence_file)
    track_order = [int(track_id) for track_id in payload["track_order"]]
    trajectory_id = str(payload["trajectory_id"])

    if not is_valid_track_permutation(track_order, benchmark.track_count):
        raise ValueError(f"{trajectory_id} is not a valid 32-track permutation")

    scan_path_rows = build_scan_path_rows(
        trajectory_id=trajectory_id,
        track_order=track_order,
        benchmark=benchmark,
        settings=settings,
        source_sequence_file=str(sequence_file),
    )
    scan_path_csv = export_dir / f"{trajectory_id}_scan_path.csv"
    metadata_json = export_dir / f"{trajectory_id}_fea_metadata.json"

    write_csv(
        scan_path_csv,
        [
            "trajectory_id",
            "segment_id",
            "order_index",
            "track_id",
            "event_type",
            "x_start_mm",
            "y_start_mm",
            "z_start_mm",
            "x_end_mm",
            "y_end_mm",
            "z_end_mm",
            "track_width_mm",
            "track_length_mm",
            "t_start_s",
            "t_end_s",
            "duration_s",
            "laser_on",
            "scan_speed_mm_s",
            "nominal_power_w",
            "source_sequence_file",
        ],
        scan_path_rows,
    )

    total_deposition_time_s = float(scan_path_rows[-1]["t_end_s"]) if scan_path_rows else 0.0
    metadata_payload = {
        "trajectory_id": trajectory_id,
        "benchmark_name": benchmark.benchmark_name,
        "trajectory_type": "track_order",
        "track_order": track_order,
        "num_tracks": benchmark.track_count,
        "substrate_size_mm": [benchmark.plane_width_mm, benchmark.plane_height_mm],
        "deposited_patch_size_mm": [
            benchmark.patch_x_max_mm - benchmark.patch_x_min_mm,
            benchmark.patch_y_max_mm - benchmark.patch_y_min_mm,
        ],
        "margin_mm": float(benchmark.margin_left_mm),
        "track_width_mm": float(benchmark.track_width_mm),
        "track_pitch_mm": float(benchmark.track_pitch_mm),
        "track_length_mm": float(benchmark.track_length_mm),
        "fixed_track_direction": "bottom_to_top",
        "scan_speed_mm_s": float(settings.scan_speed_mm_s),
        "nominal_power_w": float(settings.nominal_power_w),
        "dwell_time_s_between_tracks": float(settings.dwell_time_s_between_tracks),
        "total_deposition_time_s": total_deposition_time_s,
        "scan_path_csv": str(scan_path_csv),
        "source_sequence_file": str(sequence_file),
    }
    write_json(metadata_json, metadata_payload)

    return {
        "trajectory_id": trajectory_id,
        "source_type": trajectory_entry["source_type"],
        "selection_reason": trajectory_entry["selection_reason"],
        "sequence_file": str(sequence_file),
        "scan_path_csv": str(scan_path_csv),
        "fea_metadata_json": str(metadata_json),
        "num_tracks": benchmark.track_count,
        "total_deposition_time_s": total_deposition_time_s,
        "scan_speed_mm_s": float(settings.scan_speed_mm_s),
        "nominal_power_w": float(settings.nominal_power_w),
        "valid_export": True,
    }


def clean_export_dir(export_dir: Path) -> None:
    """Remove stale export files before writing a fresh export set."""
    export_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("*_scan_path.csv", "*_fea_metadata.json", "fea_export_manifest.csv", "fea_export_manifest.json", "fea_export_summary.txt"):
        for path in export_dir.glob(pattern):
            path.unlink()


def build_summary(
    input_pool_dir: Path,
    export_dir: Path,
    benchmark: LDEDCouponBenchmark,
    settings: ExportSettings,
    exported_rows: list[dict[str, Any]],
    failed_exports: list[str],
) -> list[str]:
    """Build the required export summary."""
    total_times = [float(row["total_deposition_time_s"]) for row in exported_rows]
    x_coords: list[float] = []
    y_coords: list[float] = []
    z_coords: list[float] = []
    permutation_ok = True

    for row in exported_rows:
        metadata = json.loads(Path(row["fea_metadata_json"]).read_text(encoding="utf-8"))
        order = metadata["track_order"]
        if not is_valid_track_permutation(order, benchmark.track_count):
            permutation_ok = False
        with Path(row["scan_path_csv"]).open("r", encoding="utf-8", newline="") as file:
            scan_rows = list(csv.DictReader(file))
        for scan_row in scan_rows:
            x_coords.extend([float(scan_row["x_start_mm"]), float(scan_row["x_end_mm"])])
            y_coords.extend([float(scan_row["y_start_mm"]), float(scan_row["y_end_mm"])])
            z_coords.extend([float(scan_row["z_start_mm"]), float(scan_row["z_end_mm"])])

    def range_text(values: list[float]) -> str:
        return f"{min(values):.6f} .. {max(values):.6f}" if values else "n/a"

    return [
        f"input pool directory: {input_pool_dir}",
        f"output export directory: {export_dir}",
        f"number of trajectories found: {len(exported_rows) + len(failed_exports)}",
        f"number of trajectories exported: {len(exported_rows)}",
        f"number of failed exports: {len(failed_exports)}",
        f"scan_speed_mm_s: {settings.scan_speed_mm_s:.6f}",
        f"nominal_power_w: {settings.nominal_power_w:.6f}",
        f"track duration s: {settings.track_duration_s:.6f}",
        f"total deposition time range s: {range_text(total_times)}",
        f"coordinate range x mm: {range_text(x_coords)}",
        f"coordinate range y mm: {range_text(y_coords)}",
        f"coordinate range z mm: {range_text(z_coords)}",
        f"all exported trajectories are length-32 permutations: {'YES' if permutation_ok else 'NO'}",
        "no Abaqus job was run: YES",
        "old TWI/grid pool was not modified: YES",
        "failed export details:",
        *([f"- {item}" for item in failed_exports] if failed_exports else ["- none"]),
    ]


def export_pool(
    manifest_path: Path = POOL_MANIFEST_JSON,
    export_dir: Path = EXPORT_DIR,
    settings: ExportSettings | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Export the whole LDED teacher pool into FEA scan-path payloads."""
    benchmark = build_lded_coupon_32track_v1()
    settings = settings or ExportSettings()
    manifest = load_pool_manifest(manifest_path)
    trajectories = list(manifest.get("trajectories", []))
    clean_export_dir(export_dir)
    export_manifest_csv = export_dir / "fea_export_manifest.csv"
    export_manifest_json = export_dir / "fea_export_manifest.json"
    export_summary_txt = export_dir / "fea_export_summary.txt"

    exported_rows: list[dict[str, Any]] = []
    failed_exports: list[str] = []

    for trajectory_entry in trajectories:
        try:
            exported_rows.append(export_one_trajectory(trajectory_entry, benchmark, settings, export_dir))
        except Exception as exc:
            failed_exports.append(f"{trajectory_entry.get('trajectory_id', 'unknown')}: {exc}")

    write_csv(
        export_manifest_csv,
        [
            "trajectory_id",
            "source_type",
            "selection_reason",
            "sequence_file",
            "scan_path_csv",
            "fea_metadata_json",
            "num_tracks",
            "total_deposition_time_s",
            "scan_speed_mm_s",
            "nominal_power_w",
            "valid_export",
        ],
        exported_rows,
    )
    write_json(
        export_manifest_json,
        {
            "benchmark_name": benchmark.benchmark_name,
            "trajectory_type": "track_order",
            "input_pool_manifest": str(manifest_path),
            "exported_count": len(exported_rows),
            "failed_count": len(failed_exports),
            "exports": exported_rows,
        },
    )
    write_text(
        export_summary_txt,
        build_summary(
            input_pool_dir=POOL_DIR,
            export_dir=export_dir,
            benchmark=benchmark,
            settings=settings,
            exported_rows=exported_rows,
            failed_exports=failed_exports,
        ),
    )
    return exported_rows, failed_exports


def main() -> None:
    """Export the current 32-track teacher pool into FEA-ready path files."""
    exported_rows, failed_exports = export_pool()
    print("LDED 32-track FEA export complete.")
    print(f"Exported trajectories: {len(exported_rows)}")
    print(f"Failed exports: {len(failed_exports)}")
    print(f"Saved export manifest CSV to: {EXPORT_MANIFEST_CSV}")
    print(f"Saved export manifest JSON to: {EXPORT_MANIFEST_JSON}")
    print(f"Saved export summary to: {EXPORT_SUMMARY_TXT}")


if __name__ == "__main__":
    main()
