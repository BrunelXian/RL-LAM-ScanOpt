"""Synthetic N32 score-derived scan strategy preview.

Stage 3 final native evidence does not include N32. This preview derives a
synthetic N32 1D order by 2x upsampling the final N16 best_u2_primary order:
each N16 track k becomes N32 tracks 2k, 2k+1. It then computes the Stage 3.5
score lift and sorts the 32x32 score field into a coordinate-order preview.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import matplotlib.pyplot as plt


STAGE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = STAGE_DIR.parents[0]
SRC_DIR = STAGE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from final_strategy_score_lift_2d import (  # noqa: E402
    derive_rank_score_from_order,
    lift_score_1d_to_2d_unit,
    parse_order,
)


SOURCE_N = 16
TARGET_N = 32
OBJECTIVE = "best_u2_primary"
MODULE_NAME = "stage3_5_final_strategy_2d_score_lift_v01"
INPUT_TABLE = (
    PROJECT_ROOT
    / "outputs"
    / "stage3_run_78_final_evidence_freeze_package"
    / "stage3_final_native_best_strategy_table.csv"
)
OUT_DIR = PROJECT_ROOT / "outputs" / MODULE_NAME / "n32_synthetic_preview"
VECTOR_DIR = OUT_DIR / "score_vectors"
MATRIX_DIR = OUT_DIR / "score_matrices"
PLOT_DIR = OUT_DIR / "plots"
REPORT_DIR = OUT_DIR / "reports"


def _branch() -> str:
    head_path = PROJECT_ROOT / ".git" / "HEAD"
    try:
        text = head_path.read_text(encoding="utf-8").strip()
    except OSError:
        return "UNKNOWN"
    if text.startswith("ref: "):
        return text.rsplit("/", 1)[-1]
    return text[:12] if text else "UNKNOWN"


def _ensure_dirs() -> None:
    for path in [OUT_DIR, VECTOR_DIR, MATRIX_DIR, PLOT_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def _read_source_row() -> dict[str, str]:
    with INPUT_TABLE.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if row.get("n") == str(SOURCE_N)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one N{SOURCE_N} row, found {len(matches)}")
    return matches[0]


def _select_source_order(row: dict[str, str]) -> tuple[str, str]:
    for suffix in ("order_json", "order_compact"):
        column = f"{OBJECTIVE}_{suffix}"
        value = row.get(column, "")
        if value and value.strip():
            return column, value
    raise RuntimeError(f"No order_json/order_compact for {OBJECTIVE}")


def _upsample_n16_order_to_n32(order16: list[int]) -> list[int]:
    order32: list[int] = []
    for track in order16:
        order32.extend([2 * track, 2 * track + 1])
    return parse_order(order32, TARGET_N)


def _rank_by_track(order: list[int]) -> list[int]:
    ranks = [0] * len(order)
    for rank, track in enumerate(order):
        ranks[track] = rank
    return ranks


def _write_vector(path: Path, order: list[int], scores: list[float]) -> None:
    ranks = _rank_by_track(order)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["track_index", "rank", "s"])
        writer.writeheader()
        for track_index, score in enumerate(scores):
            writer.writerow(
                {
                    "track_index": track_index,
                    "rank": ranks[track_index],
                    "s": f"{score:.12g}",
                }
            )


def _write_matrix(path: Path, matrix: list[list[float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for row in matrix:
            writer.writerow([f"{value:.12g}" for value in row])


def _derive_coordinate_order(matrix: list[list[float]], ranks: list[int]) -> list[dict[str, object]]:
    cells = []
    for i in range(TARGET_N):
        for j in range(TARGET_N):
            cells.append(
                {
                    "step": 0,
                    "i": i,
                    "j": j,
                    "coord": [i, j],
                    "s_new": matrix[i][j],
                    "rank_i": ranks[i],
                    "rank_j": ranks[j],
                }
            )
    cells.sort(
        key=lambda item: (
            -float(item["s_new"]),
            int(item["rank_i"]),
            int(item["rank_j"]),
            int(item["i"]),
            int(item["j"]),
        )
    )
    for step, cell in enumerate(cells, start=1):
        cell["step"] = step
    return cells


def _write_coordinate_csv(path: Path, cells: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "i", "j", "s_new", "rank_i", "rank_j"])
        writer.writeheader()
        for cell in cells:
            writer.writerow(
                {
                    "step": cell["step"],
                    "i": cell["i"],
                    "j": cell["j"],
                    "s_new": f"{float(cell['s_new']):.12g}",
                    "rank_i": cell["rank_i"],
                    "rank_j": cell["rank_j"],
                }
            )


def _step_grid(cells: list[dict[str, object]]) -> list[list[int]]:
    grid = [[0 for _ in range(TARGET_N)] for _ in range(TARGET_N)]
    for cell in cells:
        grid[int(cell["i"])][int(cell["j"])] = int(cell["step"])
    return grid


def _plot_score_heatmap(path: Path, matrix: list[list[float]]) -> None:
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    image = ax.imshow(matrix, cmap="viridis", origin="lower", vmin=0.0, vmax=1.0)
    ax.set_title("Synthetic N32 best_u2_primary lifted score field")
    ax.set_xlabel("j / column index")
    ax.set_ylabel("i / row index")
    ax.set_xticks(range(0, TARGET_N, 2))
    ax.set_yticks(range(0, TARGET_N, 2))
    ax.tick_params(labelsize=7)
    fig.colorbar(image, ax=ax, label="s_new(i,j)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_step_map(path: Path, cells: list[dict[str, object]]) -> None:
    grid = _step_grid(cells)
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    image = ax.imshow(grid, cmap="magma_r", origin="lower", vmin=1, vmax=TARGET_N * TARGET_N)
    ax.set_title("Synthetic N32 score-sorted coordinate order")
    ax.set_xlabel("j / column index")
    ax.set_ylabel("i / row index")
    ax.set_xticks(range(0, TARGET_N, 2))
    ax.set_yticks(range(0, TARGET_N, 2))
    ax.tick_params(labelsize=7)
    fig.colorbar(image, ax=ax, label="step")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_polyline(path: Path, cells: list[dict[str, object]]) -> None:
    xs = [int(cell["j"]) for cell in cells]
    ys = [int(cell["i"]) for cell in cells]
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    ax.plot(xs, ys, linewidth=0.4, color="#2f5597", alpha=0.45)
    ax.scatter(xs, ys, c=list(range(1, len(cells) + 1)), cmap="magma_r", s=7)
    ax.set_title("Synthetic N32 score-sorted polyline preview")
    ax.set_xlabel("j / column index")
    ax.set_ylabel("i / row index")
    ax.set_xticks(range(0, TARGET_N, 2))
    ax.set_yticks(range(0, TARGET_N, 2))
    ax.grid(alpha=0.2)
    ax.set_xlim(-0.5, TARGET_N - 0.5)
    ax.set_ylim(-0.5, TARGET_N - 0.5)
    ax.set_aspect("equal")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    _ensure_dirs()
    row = _read_source_row()
    order_column, raw_order = _select_source_order(row)
    order16 = parse_order(raw_order, SOURCE_N)
    order32 = _upsample_n16_order_to_n32(order16)
    ranks32 = _rank_by_track(order32)
    scores32 = derive_rank_score_from_order(order32)
    matrix32 = lift_score_1d_to_2d_unit(scores32)
    cells = _derive_coordinate_order(matrix32, ranks32)
    coord_array = [cell["coord"] for cell in cells]

    vector_path = VECTOR_DIR / "N32_synthetic_from_N16_best_u2_primary_s_1d.csv"
    matrix_path = MATRIX_DIR / "N32_synthetic_from_N16_best_u2_primary_snew_2d_unit_32x32.csv"
    order_csv_path = OUT_DIR / "N32_synthetic_from_N16_best_u2_primary_score_sorted_coordinate_order.csv"
    order_json_path = OUT_DIR / "N32_synthetic_from_N16_best_u2_primary_score_sorted_coordinate_order.json"
    order_txt_path = OUT_DIR / "N32_synthetic_from_N16_best_u2_primary_score_sorted_coordinate_order_array.txt"
    score_heatmap_path = PLOT_DIR / "N32_synthetic_from_N16_best_u2_primary_snew_2d_heatmap.png"
    step_map_path = PLOT_DIR / "N32_synthetic_from_N16_best_u2_primary_score_sorted_step_map.png"
    polyline_path = PLOT_DIR / "N32_synthetic_from_N16_best_u2_primary_score_sorted_polyline_preview.png"
    report_path = REPORT_DIR / "N32_SYNTHETIC_FROM_N16_BEST_U2_PRIMARY_SCAN_PREVIEW.md"
    manifest_path = OUT_DIR / "n32_synthetic_score_derived_scan_preview_manifest.json"

    _write_vector(vector_path, order32, scores32)
    _write_matrix(matrix_path, matrix32)
    _write_coordinate_csv(order_csv_path, cells)
    order_json_path.write_text(
        json.dumps(
            {
                "n": TARGET_N,
                "synthetic": True,
                "source_n": SOURCE_N,
                "objective": OBJECTIVE,
                "source_final_strategy_name": row.get(f"{OBJECTIVE}_strategy", ""),
                "source_order_column": order_column,
                "source_order_n16": order16,
                "synthetic_order_n32": order32,
                "synthetic_rule": "N16 track k -> N32 tracks 2k, 2k+1, preserving N16 final order",
                "cell_sort_rule": "descending s_new, then rank_i, rank_j, i, j",
                "coordinate_order": coord_array,
                "cells": cells,
                "teacher_validated": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    order_txt_path.write_text(str([tuple(coord) for coord in coord_array]), encoding="utf-8")
    _plot_score_heatmap(score_heatmap_path, matrix32)
    _plot_step_map(step_map_path, cells)
    _plot_polyline(polyline_path, cells)

    top40 = ", ".join(f"({coord[0]},{coord[1]})" for coord in coord_array[:40])
    report_path.write_text(
        f"""# Synthetic N32 Score-Derived Scan Preview

## What This Is

This is a 32 by 32 coordinate-order preview derived from a synthetic N32 score field.

Stage 3 final native evidence does not include N32. The N32 one-dimensional order is derived from the N16 `{OBJECTIVE}` final order by this deterministic rule:

```text
N16 track k -> N32 tracks 2k, 2k+1
```

The 1024 cells are then sorted by:

```text
descending s_new(i,j), then rank_i, rank_j, i, j
```

## Source

- Input table: `{INPUT_TABLE}`
- Source N: {SOURCE_N}
- Target N: {TARGET_N}
- Objective: `{OBJECTIVE}`
- Source final strategy: `{row.get(f"{OBJECTIVE}_strategy", "")}`
- Source order column: `{order_column}`
- Source N16 order: `{order16}`
- Synthetic N32 order: `{order32}`

## First 40 Coordinates

```text
{top40}
```

## Files

- Score vector: `{vector_path}`
- Score matrix: `{matrix_path}`
- Full coordinate CSV: `{order_csv_path}`
- Full coordinate JSON: `{order_json_path}`
- Python tuple-array text: `{order_txt_path}`
- Score heatmap: `{score_heatmap_path}`
- Step map: `{step_map_path}`
- Polyline preview: `{polyline_path}`

## Claim Boundary

- This is synthetic N32, not native Stage 3 evidence.
- This is a score-derived coordinate-order preview.
- It is not teacher validated.
- It does not run Abaqus, ODB extraction, solver, CAE, INP, JNL, or training.
- It does not modify frozen Stage 3 evidence.
- It does not claim physical performance improvement.
""",
        encoding="utf-8",
    )

    manifest_path.write_text(
        json.dumps(
            {
                "branch": _branch(),
                "input_table": str(INPUT_TABLE),
                "source_n": SOURCE_N,
                "target_n": TARGET_N,
                "objective": OBJECTIVE,
                "source_final_strategy_name": row.get(f"{OBJECTIVE}_strategy", ""),
                "source_order_column": order_column,
                "source_order_n16": order16,
                "synthetic_order_n32": order32,
                "synthetic_rule": "N16 track k -> N32 tracks 2k, 2k+1, preserving N16 final order",
                "coordinate_count": len(coord_array),
                "output_files": [
                    str(vector_path),
                    str(matrix_path),
                    str(order_csv_path),
                    str(order_json_path),
                    str(order_txt_path),
                    str(report_path),
                ],
                "plots": [str(score_heatmap_path), str(step_map_path), str(polyline_path)],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "native_N32_evidence": False,
                "synthetic_N32_preview": True,
                "scan_order_generated_by_user_request": True,
                "score_sorted_cells": True,
                "png_cell_values_annotated": False,
                "teacher_validated": False,
                "no_Abaqus": True,
                "no_ODB": True,
                "no_solver": True,
                "no_CAE_INP_JNL": True,
                "no_training": True,
                "no_commit_or_push": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Generated synthetic N32 preview with {len(coord_array)} coordinates.")
    print(f"First 40: {top40}")
    print(f"Coordinate CSV: {order_csv_path}")
    print(f"Coordinate JSON: {order_json_path}")
    print(f"Score heatmap: {score_heatmap_path}")
    print(f"Step map: {step_map_path}")
    print(f"Polyline: {polyline_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
