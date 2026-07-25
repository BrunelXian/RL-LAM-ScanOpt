"""Plot Stage 3.5 score-matrix heatmaps.

These are score-matrix visualizations only. They do not construct or display
cell traversal paths.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


STAGE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = STAGE_DIR.parents[0]
MODULE_NAME = "stage3_5_final_strategy_2d_score_lift_v01"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / MODULE_NAME
MATRIX_DIR = OUTPUT_DIR / "score_matrices"
PLOT_DIR = OUTPUT_DIR / "plots"


def read_matrix(path: Path) -> list[list[float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [[float(value) for value in row] for row in csv.reader(handle) if row]


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    matrix_paths = sorted(MATRIX_DIR.glob("N*_snew_2d_unit.csv"))
    if not matrix_paths:
        raise RuntimeError(f"No unit score matrices found in {MATRIX_DIR}")

    for matrix_path in matrix_paths:
        matrix = read_matrix(matrix_path)
        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
        image = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0, origin="lower")
        ax.set_title(matrix_path.stem)
        ax.set_xlabel("j")
        ax.set_ylabel("i")
        fig.colorbar(image, ax=ax, label="unit score")
        plot_path = PLOT_DIR / f"{matrix_path.stem}_heatmap.png"
        fig.savefig(plot_path, dpi=180)
        plt.close(fig)
        print(plot_path)


if __name__ == "__main__":
    main()
