"""Run the Stage 3.5 final-strategy score lift.

Reads the frozen Stage 3 final native best strategy table and writes only score
vectors, score matrices, summary files, and a manifest.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import sys


STAGE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = STAGE_DIR.parents[0]
SRC_DIR = STAGE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from final_strategy_score_lift_2d import (  # noqa: E402
    derive_rank_score_from_order,
    lift_score_1d_to_2d_raw,
    lift_score_1d_to_2d_unit,
    parse_order,
)


MODULE_NAME = "stage3_5_final_strategy_2d_score_lift_v01"
INPUT_DIR = PROJECT_ROOT / "outputs" / "stage3_run_78_final_evidence_freeze_package"
INPUT_CSV = INPUT_DIR / "stage3_final_native_best_strategy_table.csv"
INPUT_MD = INPUT_DIR / "stage3_final_native_best_strategy_table.md"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / MODULE_NAME
DOCS_DIR = PROJECT_ROOT / "docs" / MODULE_NAME

SCORE_VECTOR_DIR = OUTPUT_DIR / "score_vectors"
SCORE_MATRIX_DIR = OUTPUT_DIR / "score_matrices"
REPORT_DIR = OUTPUT_DIR / "reports"
PLOT_DIR = OUTPUT_DIR / "plots"

TARGET_N_VALUES = [12, 16, 24, 40]
OBJECTIVES = [
    "best_u2_primary",
    "best_constrained_reward",
    "best_strict_penalty_guard",
    "best_penalty_repair",
    "best_U2",
]


def _read_git_branch(project_root: Path) -> str:
    head_path = project_root / ".git" / "HEAD"
    try:
        text = head_path.read_text(encoding="utf-8").strip()
    except OSError:
        return "UNKNOWN"
    if text.startswith("ref: "):
        return text.rsplit("/", 1)[-1]
    return text[:12] if text else "UNKNOWN"


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_markdown_table_rows(path: Path) -> list[dict[str, str]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    table_lines = [line for line in lines if line.startswith("|") and line.endswith("|")]
    if len(table_lines) < 3:
        raise RuntimeError(f"Markdown fallback has no readable table: {path}")

    header = [cell.strip() for cell in table_lines[0].strip("|").split("|")]
    rows: list[dict[str, str]] = []
    for line in table_lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != len(header):
            raise RuntimeError(f"Markdown fallback row has {len(cells)} cells, expected {len(header)}.")
        rows.append(dict(zip(header, cells)))
    if not rows:
        raise RuntimeError(f"Markdown fallback has no data rows: {path}")
    return rows


def read_source_rows() -> tuple[list[dict[str, str]], Path]:
    if INPUT_CSV.exists():
        return _read_csv_rows(INPUT_CSV), INPUT_CSV
    if INPUT_MD.exists():
        return _read_markdown_table_rows(INPUT_MD), INPUT_MD
    raise FileNotFoundError(f"No final best strategy table found at {INPUT_CSV} or {INPUT_MD}.")


def select_order_value(row: dict[str, str], objective: str) -> tuple[str, str]:
    for suffix in ("order_json", "order_compact"):
        column = f"{objective}_{suffix}"
        value = row.get(column, "")
        if value and value.strip():
            return column, value
    raise ValueError(f"No order_json/order_compact available for objective {objective}.")


def write_score_vector(path: Path, order: list[int], scores: list[float]) -> None:
    rank_by_track = [0] * len(order)
    for rank, track_id in enumerate(order):
        rank_by_track[track_id] = rank

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["track_index", "rank", "s"])
        writer.writeheader()
        for track_index, score in enumerate(scores):
            writer.writerow(
                {
                    "track_index": track_index,
                    "rank": rank_by_track[track_index],
                    "s": f"{score:.12g}",
                }
            )


def write_matrix(path: Path, matrix: list[list[float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for row in matrix:
            writer.writerow([f"{value:.12g}" for value in row])


def matrix_is_symmetric(matrix: list[list[float]], tol: float = 1e-12) -> bool:
    n = len(matrix)
    return all(abs(matrix[i][j] - matrix[j][i]) <= tol for i in range(n) for j in range(n))


def diagonal_equals_scores(matrix: list[list[float]], scores: list[float], tol: float = 1e-12) -> bool:
    return all(abs(matrix[i][i] - scores[i]) <= tol for i in range(len(scores)))


def ensure_directories() -> list[Path]:
    folders = [
        STAGE_DIR,
        SRC_DIR,
        STAGE_DIR / "scripts",
        STAGE_DIR / "tests",
        STAGE_DIR / "configs",
        OUTPUT_DIR,
        SCORE_VECTOR_DIR,
        SCORE_MATRIX_DIR,
        REPORT_DIR,
        PLOT_DIR,
        DOCS_DIR,
    ]
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
    return folders


def main() -> None:
    created_folders = ensure_directories()
    rows, input_path = read_source_rows()
    rows_by_n = {int(row["n"]): row for row in rows if row.get("n", "").strip()}

    summaries: list[dict[str, object]] = []
    generated_vectors: list[str] = []
    generated_matrices: list[str] = []

    for n in TARGET_N_VALUES:
        if n not in rows_by_n:
            continue
        row = rows_by_n[n]
        for objective in OBJECTIVES:
            try:
                order_column, order_raw = select_order_value(row, objective)
                order = parse_order(order_raw, n)
            except ValueError:
                continue

            scores = derive_rank_score_from_order(order)
            unit_matrix = lift_score_1d_to_2d_unit(scores)
            raw_matrix = lift_score_1d_to_2d_raw(scores)

            vector_path = SCORE_VECTOR_DIR / f"N{n}_{objective}_s_1d_from_final_order.csv"
            unit_path = SCORE_MATRIX_DIR / f"N{n}_{objective}_snew_2d_unit.csv"
            raw_path = SCORE_MATRIX_DIR / f"N{n}_{objective}_snew_2d_raw.csv"

            write_score_vector(vector_path, order, scores)
            write_matrix(unit_path, unit_matrix)
            write_matrix(raw_path, raw_matrix)

            generated_vectors.append(str(vector_path))
            generated_matrices.extend([str(unit_path), str(raw_path)])

            summaries.append(
                {
                    "n": n,
                    "objective": objective,
                    "final_strategy_name": row.get(f"{objective}_strategy", ""),
                    "source_order_column": order_column,
                    "order_length": len(order),
                    "order_valid": True,
                    "s_min": min(scores),
                    "s_max": max(scores),
                    "snew_unit_min": min(min(values) for values in unit_matrix),
                    "snew_unit_max": max(max(values) for values in unit_matrix),
                    "snew_raw_min": min(min(values) for values in raw_matrix),
                    "snew_raw_max": max(max(values) for values in raw_matrix),
                    "diagonal_check_pass": diagonal_equals_scores(unit_matrix, scores),
                    "symmetry_check_pass": matrix_is_symmetric(unit_matrix),
                    "no_scan_order_generated": True,
                    "no_teacher_validation": True,
                }
            )

    if not summaries:
        raise RuntimeError("No valid final strategy orders were processed.")

    summary_csv_path = REPORT_DIR / "final_strategy_score_lift_summary.csv"
    with summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(summaries[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    summary_json_path = REPORT_DIR / "final_strategy_score_lift_summary.json"
    summary_json_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    manifest = {
        "project_name": MODULE_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "branch": _read_git_branch(PROJECT_ROOT),
        "input_final_best_table_path": str(input_path),
        "created_folders": [str(path) for path in created_folders],
        "created_scripts": [
            str(STAGE_DIR / "src" / "final_strategy_score_lift_2d.py"),
            str(STAGE_DIR / "scripts" / "run_final_best_strategy_score_lift.py"),
            str(STAGE_DIR / "scripts" / "plot_final_strategy_score_lift_heatmaps.py"),
        ],
        "created_docs": [
            str(DOCS_DIR / "STAGE3_5_FINAL_STRATEGY_2D_SCORE_LIFT_NOTE.md"),
            str(DOCS_DIR / "STAGE3_5_FINAL_STRATEGY_2D_SCORE_LIFT_CLAIM_BOUNDARY.md"),
        ],
        "generated_score_vectors": generated_vectors,
        "generated_score_matrices": generated_matrices,
        "summary_outputs": [str(summary_csv_path), str(summary_json_path)],
        "objectives_processed": sorted({str(item["objective"]) for item in summaries}),
        "N_values_processed": sorted({int(item["n"]) for item in summaries}),
        "no_scan_order_generated": True,
        "no_1024_path_generated": True,
        "no_Abaqus": True,
        "no_ODB": True,
        "no_solver": True,
        "no_CAE_INP_JNL": True,
        "no_training": True,
        "no_teacher_validation": True,
        "no_commit_or_push": True,
    }
    manifest_path = OUTPUT_DIR / "stage3_5_final_strategy_score_lift_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Processed {len(summaries)} objective/N score lifts.")
    print(f"Summary: {summary_csv_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
