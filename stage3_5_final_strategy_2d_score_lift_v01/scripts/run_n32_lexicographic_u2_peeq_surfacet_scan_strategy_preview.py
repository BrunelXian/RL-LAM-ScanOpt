"""Synthetic N32 preview from the U2 -> PEEQ -> SurfaceT lexicographic winner.

The final native best-strategy table does not carry the V05 lexicographic
objective as its own column, so this script reads the frozen native combined552
table and selects the N16 row by ascending:
    u2_rank_combined552_within_n,
    peeq_rank_combined552_within_n,
    surfaceT_rank_combined552_within_n.

It then upsamples that N16 order to N32 and builds a score-derived coordinate
order preview. This is not teacher validation.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


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
OBJECTIVE = "lexicographic_u2_peeq_surfacet"
MODULE_NAME = "stage3_5_final_strategy_2d_score_lift_v01"
INPUT_TABLE = (
    PROJECT_ROOT
    / "outputs"
    / "stage3_run_78_final_evidence_freeze_package"
    / "FROZEN_stage3_native_combined552_RL_ready_dataset.csv"
)
OUT_DIR = PROJECT_ROOT / "outputs" / MODULE_NAME / "n32_lexicographic_u2_peeq_surfacet_preview"
VECTOR_DIR = OUT_DIR / "score_vectors"
MATRIX_DIR = OUT_DIR / "score_matrices"
PLOT_DIR = OUT_DIR / "plots"
REPORT_DIR = OUT_DIR / "reports"

nature_score_cmap = LinearSegmentedColormap.from_list(
    "nature_score_bluegrey",
    [
        "#f7f7f7",  # near white
        "#e3e8ec",  # very light blue-grey
        "#c9d4dc",  # pale blue-grey
        "#9fb5c5",  # muted blue-grey
        "#6f91a8",  # medium blue-grey
        "#496f89",  # dark blue-grey
        "#2f4b63",  # deep blue-grey
    ],
    N=256,
)


def _branch() -> str:
    head_path = PROJECT_ROOT / ".git" / "HEAD"
    try:
        text = head_path.read_text(encoding="utf-8").strip()
    except OSError:
        return "UNKNOWN"
    if text.startswith("ref: "):
        return text.rsplit("/", 1)[-1]
    return text[:12] if text else "UNKNOWN"


def _num(row: dict[str, str], column: str) -> float:
    try:
        return float(row[column])
    except Exception:
        return float("inf")


def _ensure_dirs() -> None:
    for path in [OUT_DIR, VECTOR_DIR, MATRIX_DIR, PLOT_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def _select_lexicographic_n16_row() -> dict[str, str]:
    with INPUT_TABLE.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("n") == str(SOURCE_N)]
    if not rows:
        raise RuntimeError(f"No N{SOURCE_N} rows found in {INPUT_TABLE}")
    rows.sort(
        key=lambda row: (
            _num(row, "u2_rank_combined552_within_n"),
            _num(row, "peeq_rank_combined552_within_n"),
            _num(row, "surfaceT_rank_combined552_within_n"),
            row.get("strategy_name", ""),
        )
    )
    return rows[0]


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


def _plot_score_heatmap(path: Path, matrix: list[list[float]]) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 6.2), constrained_layout=True, facecolor="white")
    ax.set_facecolor("white")
    image = ax.imshow(
        matrix,
        cmap=nature_score_cmap,
        origin="lower",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="equal",
    )
    ax.set_title("Synthetic N32 priority field", fontsize=11, color="black", pad=8)
    ax.set_xlabel("j / column index")
    ax.set_ylabel("i / row index")
    ax.set_xticks(range(0, TARGET_N, 2))
    ax.set_yticks(range(0, TARGET_N, 2))
    ax.tick_params(axis="both", colors="black", labelsize=7, width=0.8, length=3)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.8)
    ax.set_aspect("equal")
    ax.grid(False)
    cbar = fig.colorbar(
        image,
        ax=ax,
        label="Synthetic priority score, s_new(i,j)",
        pad=0.02,
        fraction=0.046,
    )
    cbar.ax.tick_params(colors="black", labelsize=7, width=0.8, length=3)
    cbar.outline.set_edgecolor("black")
    cbar.outline.set_linewidth(0.8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_step_map(path: Path, cells: list[dict[str, object]]) -> None:
    grid = [[0 for _ in range(TARGET_N)] for _ in range(TARGET_N)]
    for cell in cells:
        grid[int(cell["i"])][int(cell["j"])] = int(cell["step"])

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    image = ax.imshow(grid, cmap="magma_r", origin="lower", vmin=1, vmax=TARGET_N * TARGET_N)
    ax.set_title("Synthetic N32 lexicographic score-sorted coordinate order")
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
    steps = list(range(1, len(cells) + 1))
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    ax.plot(xs, ys, linewidth=0.4, color="#2f5597", alpha=0.45)
    scatter = ax.scatter(xs, ys, c=steps, cmap="magma_r", s=7)
    ax.set_title("Synthetic N32 lexicographic score-sorted polyline preview")
    ax.set_xlabel("j / column index")
    ax.set_ylabel("i / row index")
    ax.set_xticks(range(0, TARGET_N, 2))
    ax.set_yticks(range(0, TARGET_N, 2))
    ax.grid(alpha=0.2)
    ax.set_xlim(-0.5, TARGET_N - 0.5)
    ax.set_ylim(-0.5, TARGET_N - 0.5)
    ax.set_aspect("equal")
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("scan step order")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    _ensure_dirs()
    row = _select_lexicographic_n16_row()
    order16 = parse_order(row["order_json"] or row["order_compact"], SOURCE_N)
    order32 = _upsample_n16_order_to_n32(order16)
    ranks32 = _rank_by_track(order32)
    scores32 = derive_rank_score_from_order(order32)
    matrix32 = lift_score_1d_to_2d_unit(scores32)
    cells = _derive_coordinate_order(matrix32, ranks32)
    coord_array = [cell["coord"] for cell in cells]

    prefix = "N32_synthetic_from_N16_lexicographic_u2_peeq_surfacet"
    vector_path = VECTOR_DIR / f"{prefix}_s_1d.csv"
    matrix_path = MATRIX_DIR / f"{prefix}_snew_2d_unit_32x32.csv"
    order_csv_path = OUT_DIR / f"{prefix}_score_sorted_coordinate_order.csv"
    order_json_path = OUT_DIR / f"{prefix}_score_sorted_coordinate_order.json"
    order_txt_path = OUT_DIR / f"{prefix}_score_sorted_coordinate_order_array.txt"
    score_heatmap_path = PLOT_DIR / f"{prefix}_snew_2d_heatmap.png"
    step_map_path = PLOT_DIR / f"{prefix}_score_sorted_step_map.png"
    polyline_path = PLOT_DIR / f"{prefix}_score_sorted_polyline_preview.png"
    report_path = REPORT_DIR / "N32_SYNTHETIC_FROM_N16_LEXICOGRAPHIC_U2_PEEQ_SURFACET_SCAN_PREVIEW.md"
    manifest_path = OUT_DIR / "n32_lexicographic_u2_peeq_surfacet_scan_preview_manifest.json"

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
                "source_table": str(INPUT_TABLE),
                "source_strategy_name": row.get("strategy_name", ""),
                "source_order_n16": order16,
                "synthetic_order_n32": order32,
                "source_metrics": {
                    "u2_rank_combined552_within_n": row.get("u2_rank_combined552_within_n"),
                    "peeq_rank_combined552_within_n": row.get("peeq_rank_combined552_within_n"),
                    "surfaceT_rank_combined552_within_n": row.get("surfaceT_rank_combined552_within_n"),
                    "u2_range": row.get("u2_range"),
                    "peeq_max": row.get("peeq_max"),
                    "surface_t_proxy": row.get("surface_t_proxy"),
                },
                "selection_rule": "minimize lexicographic tuple: U2 rank, then PEEQ rank, then SurfaceT rank",
                "synthetic_rule": "N16 track k -> N32 tracks 2k, 2k+1, preserving selected N16 order",
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
        f"""# Synthetic N32 From N16 Lexicographic U2-PEEQ-SurfaceT Scan Preview

## What This Is

This is a 32 by 32 coordinate-order preview derived from the N16 winner under the V05-style lexicographic priority:

```text
U2 first, then PEEQ, then SurfaceT
```

The N16 row is selected from frozen native combined552 by ascending:

```text
u2_rank_combined552_within_n,
peeq_rank_combined552_within_n,
surfaceT_rank_combined552_within_n
```

Stage 3 final native evidence does not include native N32. The synthetic N32 one-dimensional order is derived from the selected N16 order by:

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
- Source strategy: `{row.get("strategy_name", "")}`
- Source N16 order: `{order16}`
- Synthetic N32 order: `{order32}`
- U2 rank: `{row.get("u2_rank_combined552_within_n")}`
- PEEQ rank: `{row.get("peeq_rank_combined552_within_n")}`
- SurfaceT rank: `{row.get("surfaceT_rank_combined552_within_n")}`
- U2 raw: `{row.get("u2_range")}`
- PEEQ raw: `{row.get("peeq_max")}`
- SurfaceT raw: `{row.get("surface_t_proxy")}`

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
- This uses the lexicographic U2 -> PEEQ -> SurfaceT selected N16 order.
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
                "source_strategy_name": row.get("strategy_name", ""),
                "source_order_n16": order16,
                "synthetic_order_n32": order32,
                "selection_rule": "ascending U2 rank, then PEEQ rank, then SurfaceT rank",
                "source_metrics": {
                    "u2_rank_combined552_within_n": row.get("u2_rank_combined552_within_n"),
                    "peeq_rank_combined552_within_n": row.get("peeq_rank_combined552_within_n"),
                    "surfaceT_rank_combined552_within_n": row.get("surfaceT_rank_combined552_within_n"),
                    "u2_range": row.get("u2_range"),
                    "peeq_max": row.get("peeq_max"),
                    "surface_t_proxy": row.get("surface_t_proxy"),
                },
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

    print(f"Selected N16 strategy: {row.get('strategy_name', '')}")
    print(f"Selected N16 order: {'-'.join(str(x) for x in order16)}")
    print(
        "Ranks: "
        f"U2={row.get('u2_rank_combined552_within_n')}, "
        f"PEEQ={row.get('peeq_rank_combined552_within_n')}, "
        f"SurfaceT={row.get('surfaceT_rank_combined552_within_n')}"
    )
    print(f"Generated synthetic N32 lexicographic preview with {len(coord_array)} coordinates.")
    print(f"First 40: {top40}")
    print(f"Coordinate CSV: {order_csv_path}")
    print(f"Coordinate JSON: {order_json_path}")
    print(f"Score heatmap: {score_heatmap_path}")
    print(f"Step map: {step_map_path}")
    print(f"Polyline: {polyline_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
