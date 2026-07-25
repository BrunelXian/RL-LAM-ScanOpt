"""Focused N16 visualization check for the Stage 3.5 score lift."""

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
MODULE_NAME = "stage3_5_final_strategy_2d_score_lift_v01"
INPUT_TABLE = (
    PROJECT_ROOT
    / "outputs"
    / "stage3_run_78_final_evidence_freeze_package"
    / "stage3_final_native_best_strategy_table.csv"
)
FOCUSED_DIR = PROJECT_ROOT / "outputs" / MODULE_NAME / "n16_focused_visual_check"
VECTOR_DIR = FOCUSED_DIR / "score_vectors"
MATRIX_DIR = FOCUSED_DIR / "score_matrices"
PLOT_DIR = FOCUSED_DIR / "plots"
REPORT_DIR = FOCUSED_DIR / "reports"

OBJECTIVES = [
    "best_u2_primary",
    "best_constrained_reward",
    "best_strict_penalty_guard",
    "best_penalty_repair",
    "best_U2",
]


def _read_branch() -> str:
    head_path = PROJECT_ROOT / ".git" / "HEAD"
    try:
        head_text = head_path.read_text(encoding="utf-8").strip()
    except OSError:
        return "UNKNOWN"
    if head_text.startswith("ref: "):
        return head_text.rsplit("/", 1)[-1]
    return head_text[:12] if head_text else "UNKNOWN"


def _ensure_dirs() -> None:
    for path in [FOCUSED_DIR, VECTOR_DIR, MATRIX_DIR, PLOT_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def _read_n16_row() -> dict[str, str]:
    with INPUT_TABLE.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if row.get("n") == str(N)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one n={N} row in {INPUT_TABLE}, found {len(matches)}.")
    return matches[0]


def _select_order(row: dict[str, str], objective: str) -> tuple[str, str]:
    for suffix in ("order_json", "order_compact"):
        column = f"{objective}_{suffix}"
        value = row.get(column, "")
        if value and value.strip():
            return column, value
    raise ValueError(f"No order_json/order_compact found for {objective}.")


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


def _annotate_order(fig: plt.Figure, order: list[int]) -> None:
    order_text = "Final 1D order: " + "-".join(str(item) for item in order)
    fig.text(0.5, 0.015, order_text, ha="center", va="bottom", fontsize=8)


def _plot_bar(path: Path, objective: str, order: list[int], scores: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=False)
    ax.bar(range(N), scores, color="#4069a8")
    ax.set_title(f"N16 {objective} rank-derived 1D score")
    ax.set_xlabel("track index i")
    ax.set_ylabel("s(i)")
    ax.set_xticks(range(N))
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    _annotate_order(fig, order)
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_rank_bar(path: Path, objective: str, order: list[int]) -> None:
    ranks = _rank_by_track(order)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.bar(range(N), ranks, color="#7a5c99")
    ax.set_title(f"N16 {objective} final 1D order rank by track")
    ax.set_xlabel("track index i")
    ax.set_ylabel("rank(i)")
    ax.set_xticks(range(N))
    ax.invert_yaxis()
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _draw_heatmap(ax: plt.Axes, matrix: list[list[float]], objective: str) -> object:
    image = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0, origin="lower")
    ax.set_title(f"N16 {objective} 2D lifted score field")
    ax.set_xlabel("j / column index")
    ax.set_ylabel("i / row index")
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.tick_params(labelsize=7)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=5, color="white")
    return image


def _plot_heatmap(path: Path, objective: str, matrix: list[list[float]]) -> None:
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    image = _draw_heatmap(ax, matrix, objective)
    fig.colorbar(image, ax=ax, label="s_new(i,j)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_combined(
    path: Path,
    objective: str,
    order: list[int],
    scores: list[float],
    matrix: list[list[float]],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=False)
    axes[0].bar(range(N), scores, color="#4069a8")
    axes[0].set_title(f"N16 {objective} rank-derived 1D score")
    axes[0].set_xlabel("track index i")
    axes[0].set_ylabel("s(i)")
    axes[0].set_xticks(range(N))
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(axis="y", alpha=0.25)

    image = _draw_heatmap(axes[1], matrix, objective)
    fig.colorbar(image, ax=axes[1], label="s_new(i,j)")
    _annotate_order(fig, order)
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _markdown_table(rows: list[dict[str, object]]) -> str:
    headers = ["objective", "final_strategy_name", "source_order_column", "s_min", "s_max", "matrix_min", "matrix_max"]
    lines = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        values = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("|" + "|".join(values) + "|")
    return "\n".join(lines)


def _write_report(rows: list[dict[str, object]]) -> Path:
    report_path = REPORT_DIR / "N16_FOCUSED_SCORE_LIFT_VISUAL_CHECK.md"
    objective_list = "\n".join(f"- {row['objective']}" for row in rows)
    details = []
    for row in rows:
        details.append(
            "\n".join(
                [
                    f"### {row['objective']}",
                    f"- Final strategy: `{row['final_strategy_name']}`",
                    f"- Source order column: `{row['source_order_column']}`",
                    f"- Final order: `{row['final_order_compact']}`",
                    f"- Score vector: `{row['score_vector_path']}`",
                    f"- Score matrix: `{row['score_matrix_path']}`",
                    f"- Bar plot: `{row['bar_plot_path']}`",
                    f"- Heatmap: `{row['heatmap_path']}`",
                    f"- Combined plot: `{row['combined_plot_path']}`",
                    f"- Rank bar: `{row['rank_bar_path']}`",
                ]
            )
        )

    text = f"""# N16 Focused Score-Lift Visual Check

## Purpose

This report checks the N16 Stage 3.5 final-strategy score lift visually and numerically. It derives a one-dimensional score from the frozen final Stage 3 native best order, then lifts it to a 16 by 16 score matrix.

## Source

- Source table: `{INPUT_TABLE}`
- N: 16

## Objectives Processed

{objective_list}

## Formula

```text
rank(i) = position of track i in final order
s(i) = eps + (1 - 2*eps) * (1 - rank(i)/(N-1))
eps = 1e-6
s_new(i,j) = sqrt((s(i)^2 + s(j)^2) / 2)
```

The score vector `s(i)` was rank-derived from final best `order_json` or `order_compact`. The lifted `s_new(i,j)` output is a score matrix only.

## Summary Table

{_markdown_table(rows)}

## Objective Details And Figures

{chr(10).join(details)}

## Claim Boundary

- No scan order generated.
- No teacher validation.
- No Abaqus.
- No physical performance claim.
- No 256-point path generated.
- No ODB, solver, CAE, INP, JNL, or training action.
"""
    report_path.write_text(text, encoding="utf-8")
    return report_path


def main() -> None:
    _ensure_dirs()
    row = _read_n16_row()
    results: list[dict[str, object]] = []

    for objective in OBJECTIVES:
        try:
            order_column, order_raw = _select_order(row, objective)
            order = parse_order(order_raw, N)
        except ValueError:
            continue

        scores = derive_rank_score_from_order(order)
        matrix = lift_score_1d_to_2d_unit(scores)

        vector_path = VECTOR_DIR / f"N16_{objective}_s_1d.csv"
        matrix_path = MATRIX_DIR / f"N16_{objective}_snew_2d_unit_16x16.csv"
        bar_path = PLOT_DIR / f"N16_{objective}_s_1d_bar.png"
        heatmap_path = PLOT_DIR / f"N16_{objective}_snew_2d_heatmap.png"
        combined_path = PLOT_DIR / f"N16_{objective}_combined_score_lift.png"
        rank_bar_path = PLOT_DIR / f"N16_{objective}_order_rank_bar.png"

        _write_vector(vector_path, order, scores)
        _write_matrix(matrix_path, matrix)
        _plot_bar(bar_path, objective, order, scores)
        _plot_heatmap(heatmap_path, objective, matrix)
        _plot_combined(combined_path, objective, order, scores, matrix)
        _plot_rank_bar(rank_bar_path, objective, order)

        results.append(
            {
                "n": N,
                "objective": objective,
                "final_strategy_name": row.get(f"{objective}_strategy", ""),
                "source_order_column": order_column,
                "final_order_json_or_compact": order_raw,
                "final_order_compact": "-".join(str(item) for item in order),
                "order_length": len(order),
                "order_valid": True,
                "s_min": min(scores),
                "s_max": max(scores),
                "matrix_min": min(min(values) for values in matrix),
                "matrix_max": max(max(values) for values in matrix),
                "diagonal_check_pass": all(abs(matrix[i][i] - scores[i]) <= 1e-12 for i in range(N)),
                "symmetry_check_pass": all(
                    abs(matrix[i][j] - matrix[j][i]) <= 1e-12 for i in range(N) for j in range(N)
                ),
                "score_vector_path": str(vector_path),
                "score_matrix_path": str(matrix_path),
                "bar_plot_path": str(bar_path),
                "heatmap_path": str(heatmap_path),
                "combined_plot_path": str(combined_path),
                "rank_bar_path": str(rank_bar_path),
                "no_scan_order_generated": True,
                "no_teacher_validation": True,
            }
        )

    if not results:
        raise RuntimeError("No N16 objectives with usable final order data were processed.")

    summary_csv_path = REPORT_DIR / "n16_focused_score_lift_summary.csv"
    with summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    summary_json_path = REPORT_DIR / "n16_focused_score_lift_summary.json"
    summary_json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    report_path = _write_report(results)

    manifest = {
        "branch": _read_branch(),
        "input_table": str(INPUT_TABLE),
        "N_processed": N,
        "objectives_processed": [str(row["objective"]) for row in results],
        "output_files": {
            "score_vectors": [str(row["score_vector_path"]) for row in results],
            "score_matrices": [str(row["score_matrix_path"]) for row in results],
            "summary_csv": str(summary_csv_path),
            "summary_json": str(summary_json_path),
            "markdown_report": str(report_path),
        },
        "plots_generated": {
            "bar_plots": [str(row["bar_plot_path"]) for row in results],
            "heatmaps": [str(row["heatmap_path"]) for row in results],
            "combined_plots": [str(row["combined_plot_path"]) for row in results],
            "rank_bars": [str(row["rank_bar_path"]) for row in results],
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "no_scan_order_generated": True,
        "no_256_path_generated": True,
        "no_Abaqus": True,
        "no_ODB": True,
        "no_solver": True,
        "no_CAE_INP_JNL": True,
        "no_training": True,
        "no_teacher_validation": True,
        "no_commit_or_push": True,
    }
    manifest_path = FOCUSED_DIR / "n16_focused_score_lift_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Processed {len(results)} N16 objectives.")
    print(f"Summary CSV: {summary_csv_path}")
    print(f"Summary JSON: {summary_json_path}")
    print(f"Report: {report_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
