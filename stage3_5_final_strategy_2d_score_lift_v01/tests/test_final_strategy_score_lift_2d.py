from __future__ import annotations

import csv
import inspect
import json
import math
from pathlib import Path
import sys

import pytest


STAGE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = STAGE_DIR.parents[0]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage3_5_final_strategy_2d_score_lift_v01"
sys.path.insert(0, str(STAGE_DIR / "src"))

import final_strategy_score_lift_2d as lift  # noqa: E402


def test_parse_order_accepts_json_list() -> None:
    assert lift.parse_order("[2,0,1,3]", 4) == [2, 0, 1, 3]


def test_parse_order_accepts_compact_string() -> None:
    assert lift.parse_order("2-0-1-3", 4) == [2, 0, 1, 3]


def test_illegal_duplicate_order_rejected() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        lift.parse_order("[2,0,1,1]", 4)


def test_missing_track_rejected() -> None:
    with pytest.raises(ValueError, match="missing"):
        lift.parse_order("[2,0,1,4]", 4)


def test_rank_score_values_are_in_unit_interval() -> None:
    scores = lift.derive_rank_score_from_order([2, 0, 1, 3])
    assert all(0.0 < score < 1.0 for score in scores)


def test_earlier_track_has_higher_score_than_later_track() -> None:
    order = [2, 0, 1, 3]
    scores = lift.derive_rank_score_from_order(order)
    assert scores[2] > scores[0] > scores[1] > scores[3]


def test_unit_matrix_shape_is_n_by_n() -> None:
    scores = lift.derive_rank_score_from_order([2, 0, 1, 3])
    matrix = lift.lift_score_1d_to_2d_unit(scores)
    assert len(matrix) == 4
    assert all(len(row) == 4 for row in matrix)


def test_unit_matrix_symmetric() -> None:
    scores = lift.derive_rank_score_from_order([2, 0, 1, 3])
    matrix = lift.lift_score_1d_to_2d_unit(scores)
    assert all(matrix[i][j] == pytest.approx(matrix[j][i]) for i in range(4) for j in range(4))


def test_unit_matrix_diagonal_equals_scores() -> None:
    scores = lift.derive_rank_score_from_order([2, 0, 1, 3])
    matrix = lift.lift_score_1d_to_2d_unit(scores)
    assert all(matrix[i][i] == pytest.approx(scores[i]) for i in range(4))


def test_unit_matrix_values_are_in_unit_interval() -> None:
    scores = lift.derive_rank_score_from_order([2, 0, 1, 3])
    matrix = lift.lift_score_1d_to_2d_unit(scores)
    assert all(0.0 < value < 1.0 for row in matrix for value in row)


def test_raw_matrix_values_are_in_zero_sqrt_two_interval() -> None:
    scores = lift.derive_rank_score_from_order([2, 0, 1, 3])
    matrix = lift.lift_score_1d_to_2d_raw(scores)
    assert all(0.0 < value < math.sqrt(2.0) for row in matrix for value in row)


def test_no_generate_scan_order_function_exists() -> None:
    assert "generate_scan_order" not in {
        name for name, obj in inspect.getmembers(lift) if inspect.isfunction(obj)
    }


def _contains_coordinate_pair_collection(value: object) -> bool:
    if isinstance(value, list):
        if len(value) >= 100 and all(
            isinstance(item, list)
            and len(item) == 2
            and all(isinstance(axis, int) for axis in item)
            for item in value
        ):
            return True
        return any(_contains_coordinate_pair_collection(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_coordinate_pair_collection(item) for item in value.values())
    return False


def test_no_output_file_contains_sorted_cell_coordinate_list() -> None:
    if not OUTPUT_DIR.exists():
        pytest.skip("Stage 3.5 output folder has not been generated yet.")

    forbidden_terms = {
        "scan_sequence",
        "cell_order",
        "cell_coordinates",
        "row_column_path",
        "sorted_cells",
        "traversal_order",
    }
    for path in OUTPUT_DIR.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".csv", ".json", ".md"}:
            continue
        text = path.read_text(encoding="utf-8")
        lowered = text.lower()
        assert not any(term in lowered for term in forbidden_terms), path
        if path.suffix.lower() == ".json":
            data = json.loads(text)
            assert not _contains_coordinate_pair_collection(data), path
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                header = next(csv.reader(handle), [])
            assert not any(term in ",".join(header).lower() for term in forbidden_terms), path
