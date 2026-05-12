"""Preview the first LDED line-order benchmark and export baseline trajectories."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.geometry import build_lded_coupon_32track_baselines, build_lded_coupon_32track_v1

FIGURES_DIR = PROJECT_ROOT / "assets" / "figures"
DATA_DIR = PROJECT_ROOT / "assets" / "data"


def _ensure_output_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def render_layout_preview(with_track_ids: bool = False) -> Path:
    """Render the 100x40 coupon layout with the deposited patch and tracks."""
    benchmark = build_lded_coupon_32track_v1()
    figure, axis = plt.subplots(figsize=(14, 6))

    plane = Rectangle(
        (0.0, 0.0),
        benchmark.plane_width_mm,
        benchmark.plane_height_mm,
        facecolor="#f7f7f7",
        edgecolor="#444444",
        linewidth=2.0,
    )
    axis.add_patch(plane)

    patch = Rectangle(
        (benchmark.patch_x_min_mm, benchmark.patch_y_min_mm),
        benchmark.patch_x_max_mm - benchmark.patch_x_min_mm,
        benchmark.patch_y_max_mm - benchmark.patch_y_min_mm,
        facecolor="#d9edf7",
        edgecolor="#2b6c8f",
        linewidth=1.5,
        alpha=0.75,
    )
    axis.add_patch(patch)

    for track in benchmark.tracks:
        track_rect = Rectangle(
            (track.x_start_mm, track.y_start_mm),
            track.width_mm,
            track.length_mm,
            facecolor="#8fc4de",
            edgecolor="#114b5f",
            linewidth=0.6,
            alpha=0.95,
        )
        axis.add_patch(track_rect)
        if with_track_ids:
            axis.text(
                track.x_center_mm,
                benchmark.patch_y_max_mm + 0.65,
                str(track.track_id),
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

    axis.annotate(
        "96 mm deposited patch",
        xy=(benchmark.patch_x_min_mm + 48.0, benchmark.patch_y_max_mm + 0.4),
        xytext=(benchmark.patch_x_min_mm + 48.0, benchmark.patch_y_max_mm + 3.5),
        ha="center",
        arrowprops={"arrowstyle": "-|>", "lw": 1.0, "color": "#444444"},
        fontsize=10,
    )
    axis.annotate(
        "2 mm margin",
        xy=(1.0, 20.0),
        xytext=(8.0, 26.0),
        arrowprops={"arrowstyle": "-|>", "lw": 1.0, "color": "#444444"},
        fontsize=10,
    )
    axis.annotate(
        "32 vertical tracks\n(3 mm width / 3 mm pitch)",
        xy=(benchmark.patch_x_min_mm + 24.0, benchmark.patch_y_min_mm + 18.0),
        xytext=(benchmark.patch_x_min_mm + 10.0, benchmark.patch_y_min_mm - 6.0),
        arrowprops={"arrowstyle": "-|>", "lw": 1.0, "color": "#444444"},
        fontsize=10,
        ha="left",
    )

    axis.set_title("LDED Coupon Benchmark: lded_coupon_32track_v1", fontsize=13)
    axis.set_xlim(-1.5, benchmark.plane_width_mm + 1.5)
    axis.set_ylim(-2.0, benchmark.plane_height_mm + 6.0)
    axis.set_aspect("equal")
    axis.set_xlabel("X (mm)")
    axis.set_ylabel("Y (mm)")
    axis.grid(False)

    filename = "lded_coupon_32track_layout_with_ids.png" if with_track_ids else "lded_coupon_32track_layout.png"
    output_path = FIGURES_DIR / filename
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path


def render_baseline_preview() -> Path:
    """Render a compact comparison of a few track-order baselines."""
    benchmark = build_lded_coupon_32track_v1()
    baselines = build_lded_coupon_32track_baselines()
    preview_names = ["raster_left_to_right", "center_out", "edge_in", "odd_even_interlaced"]
    figure, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)

    for axis, name in zip(axes.ravel(), preview_names):
        sequence = baselines[name]
        axis.add_patch(
            Rectangle(
                (benchmark.patch_x_min_mm, benchmark.patch_y_min_mm),
                benchmark.patch_x_max_mm - benchmark.patch_x_min_mm,
                benchmark.patch_y_max_mm - benchmark.patch_y_min_mm,
                facecolor="#f5f9fc",
                edgecolor="#b0c4d4",
                linewidth=1.0,
            )
        )
        for order_idx, track_id in enumerate(sequence):
            track = benchmark.tracks[track_id]
            color = plt.cm.viridis(order_idx / max(len(sequence) - 1, 1))
            axis.add_patch(
                Rectangle(
                    (track.x_start_mm, track.y_start_mm),
                    track.width_mm,
                    track.length_mm,
                    facecolor=color,
                    edgecolor="none",
                )
            )
        axis.set_title(name, fontsize=11)
        axis.set_aspect("equal")
        axis.set_xlim(benchmark.patch_x_min_mm - 1.0, benchmark.patch_x_max_mm + 1.0)
        axis.set_ylim(benchmark.patch_y_min_mm - 1.0, benchmark.patch_y_max_mm + 1.0)
        axis.set_xticks([])
        axis.set_yticks([])

    figure.tight_layout()
    output_path = FIGURES_DIR / "lded_coupon_32track_baseline_previews.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def save_benchmark_payload() -> Path:
    """Save benchmark geometry and baseline track trajectories for later reuse."""
    benchmark = build_lded_coupon_32track_v1()
    baselines = build_lded_coupon_32track_baselines()
    output_path = DATA_DIR / "lded_coupon_32track_baselines.json"
    payload = {
        "benchmark": benchmark.to_dict(),
        "baseline_trajectories": baselines,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    _ensure_output_dirs()
    layout_path = render_layout_preview(with_track_ids=False)
    layout_ids_path = render_layout_preview(with_track_ids=True)
    baseline_preview_path = render_baseline_preview()
    baseline_json_path = save_benchmark_payload()
    print(f"Saved layout preview to: {layout_path}")
    print(f"Saved track-id preview to: {layout_ids_path}")
    print(f"Saved baseline preview to: {baseline_preview_path}")
    print(f"Saved baseline trajectories to: {baseline_json_path}")


if __name__ == "__main__":
    main()
