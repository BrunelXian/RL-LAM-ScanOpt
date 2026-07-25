"""N16 score-derived scan strategy preview.

This script intentionally creates a coordinate-order preview from the Stage 3.5
score field because the user requested a visible 16x16 plane strategy. It is
not teacher validation and does not run Abaqus.
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


N = 16
OBJECTIVE = "best_u2_primary"
MODULE_NAME = "stage3_5_final_strategy_2d_score_lift_v01"
INPUT_TABLE = (
    PROJECT_ROOT
    / "outputs"
    / "stage3_run_78_final_evidence_freeze_package"
    / "stage3_final_native_best_strategy_table.csv"
)
OUT_DIR = PROJECT_ROOT / "outputs" / MODULE_NAME / "n16_focused_visual_check" / "scan_strategy_preview"
REPORT_DIR = OUT_DIR / "reports"
PLOT_DIR = OUT_DIR / "plots"


def _branch() -> str:
    head = PROJECT_ROOT / ".git" / "HEAD"
    try:
        text = head.read_text(encoding="utf-8").strip()
    except OSError:
        return "UNKNOWN"
    if text.startswith("ref: "):
        return text.rsplit("/", 1)[-1]
    return text[:12] if text else "UNKNOWN"


def _read_n16_row() -> dict[str, str]:
    with INPUT_TABLE.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if row.get("n") == str(N)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one N16 row, found {len(matches)}")
    return matches[0]


def _select_order(row: dict[str, str]) -> tuple[str, str]:
    for suffix in ("order_json", "order_compact"):
        column = f"{OBJECTIVE}_{suffix}"
        value = row.get(column, "")
        if value and value.strip():
            return column, value
    raise RuntimeError(f"No order_json/order_compact for {OBJECTIVE}")


def _rank_by_track(order: list[int]) -> list[int]:
    ranks = [0] * len(order)
    for rank, track in enumerate(order):
        ranks[track] = rank
    return ranks


def _derive_coordinate_order(matrix: list[list[float]], ranks: list[int]) -> list[dict[str, object]]:
    cells = []
    for i in range(N):
        for j in range(N):
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

    # Descending score, then deterministic rank/coordinate tie-break.
    cells.sort(key=lambda item: (-float(item["s_new"]), int(item["rank_i"]), int(item["rank_j"]), int(item["i"]), int(item["j"])))
    for step, item in enumerate(cells, start=1):
        item["step"] = step
    return cells


def _write_csv(path: Path, cells: list[dict[str, object]]) -> None:
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


def _plot_step_map(path: Path, cells: list[dict[str, object]]) -> None:
    step_by_cell = [[0 for _ in range(N)] for _ in range(N)]
    for cell in cells:
        step_by_cell[int(cell["i"])][int(cell["j"])] = int(cell["step"])

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    image = ax.imshow(step_by_cell, cmap="magma_r", origin="lower", vmin=1, vmax=N * N)
    ax.set_title("N16 best_u2_primary score-derived coordinate order preview")
    ax.set_xlabel("j / column index")
    ax.set_ylabel("i / row index")
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.tick_params(labelsize=7)
    for i in range(N):
        for j in range(N):
            ax.text(j, i, str(step_by_cell[i][j]), ha="center", va="center", fontsize=5, color="white")
    fig.colorbar(image, ax=ax, label="step")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_polyline(path: Path, cells: list[dict[str, object]]) -> None:
    xs = [int(cell["j"]) for cell in cells]
    ys = [int(cell["i"]) for cell in cells]
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    ax.plot(xs, ys, linewidth=0.7, color="#2f5597", alpha=0.75)
    ax.scatter(xs, ys, c=list(range(1, len(cells) + 1)), cmap="magma_r", s=18)
    ax.set_title("N16 best_u2_primary score-derived preview path")
    ax.set_xlabel("j / column index")
    ax.set_ylabel("i / row index")
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.grid(alpha=0.25)
    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(-0.5, N - 0.5)
    ax.set_aspect("equal")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    row = _read_n16_row()
    order_column, raw_order = _select_order(row)
    order = parse_order(raw_order, N)
    ranks = _rank_by_track(order)
    scores = derive_rank_score_from_order(order)
    matrix = lift_score_1d_to_2d_unit(scores)
    cells = _derive_coordinate_order(matrix, ranks)
    coord_array = [cell["coord"] for cell in cells]

    csv_path = OUT_DIR / "N16_best_u2_primary_score_sorted_coordinate_order.csv"
    json_path = OUT_DIR / "N16_best_u2_primary_score_sorted_coordinate_order.json"
    txt_path = OUT_DIR / "N16_best_u2_primary_score_sorted_coordinate_order_array.txt"
    step_map_path = PLOT_DIR / "N16_best_u2_primary_score_sorted_step_map.png"
    path_plot_path = PLOT_DIR / "N16_best_u2_primary_score_sorted_polyline_preview.png"
    report_path = REPORT_DIR / "N16_BEST_U2_PRIMARY_SCORE_DERIVED_SCAN_PREVIEW.md"
    manifest_path = OUT_DIR / "n16_score_derived_scan_preview_manifest.json"

    _write_csv(csv_path, cells)
    json_path.write_text(
        json.dumps(
            {
                "n": N,
                "objective": OBJECTIVE,
                "final_strategy_name": row.get(f"{OBJECTIVE}_strategy", ""),
                "source_order_column": order_column,
                "final_order": order,
                "tie_break": "descending s_new, then rank_i, rank_j, i, j",
                "coordinate_order": coord_array,
                "cells": cells,
                "teacher_validated": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    txt_path.write_text(str([tuple(coord) for coord in coord_array]), encoding="utf-8")
    _plot_step_map(step_map_path, cells)
    _plot_polyline(path_plot_path, cells)

    top20 = ", ".join(f"({coord[0]},{coord[1]})" for coord in coord_array[:20])
    report_path.write_text(
        f"""# N16 best_u2_primary Score-Derived Scan Preview

## What This Is

This is a 16 by 16 coordinate-order preview derived from the Stage 3.5 lifted score field.

It sorts all 256 cells by:

```text
descending s_new(i,j), then rank_i, rank_j, i, j
```

## Source

- Input table: `{INPUT_TABLE}`
- Objective: `{OBJECTIVE}`
- Final strategy: `{row.get(f"{OBJECTIVE}_strategy", "")}`
- Source order column: `{order_column}`
- Final 1D order: `{order}`

## First 20 Coordinates

```text
{top20}
```

## Files

- Full CSV order: `{csv_path}`
- Full JSON order: `{json_path}`
- Python tuple-array text: `{txt_path}`
- Step map: `{step_map_path}`
- Polyline preview: `{path_plot_path}`

## Claim Boundary

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
                "n": N,
                "objective": OBJECTIVE,
                "final_strategy_name": row.get(f"{OBJECTIVE}_strategy", ""),
                "source_order_column": order_column,
                "final_order": order,
                "coordinate_count": len(coord_array),
                "output_files": [str(csv_path), str(json_path), str(txt_path), str(report_path)],
                "plots": [str(step_map_path), str(path_plot_path)],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "scan_order_generated_by_user_request": True,
                "score_sorted_cells": True,
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

    print(f"Generated {len(coord_array)} score-sorted N16 coordinates.")
    print(f"First 20: {top20}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
