# Synthetic N32 Score-Derived Scan Preview

## What This Is

This is a 32 by 32 coordinate-order preview derived from a synthetic N32 score field.

Stage 3 final native evidence does not include N32. The N32 one-dimensional order is derived from the N16 `best_u2_primary` final order by this deterministic rule:

```text
N16 track k -> N32 tracks 2k, 2k+1
```

The 1024 cells are then sorted by:

```text
descending s_new(i,j), then rank_i, rank_j, i, j
```

## Source

- Input table: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\stage3_final_native_best_strategy_table.csv`
- Source N: 16
- Target N: 32
- Objective: `best_u2_primary`
- Source final strategy: `S3R74FSD_N16_B01_n16_penalty`
- Source order column: `best_u2_primary_order_json`
- Source N16 order: `[14, 6, 10, 12, 8, 4, 2, 0, 1, 3, 5, 7, 9, 11, 13, 15]`
- Synthetic N32 order: `[28, 29, 12, 13, 20, 21, 24, 25, 16, 17, 8, 9, 4, 5, 0, 1, 2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31]`

## First 40 Coordinates

```text
(28,28), (28,29), (29,28), (28,12), (12,28), (29,29), (28,13), (13,28), (29,12), (12,29), (28,20), (20,28), (29,13), (13,29), (12,12), (28,21), (21,28), (29,20), (20,29), (12,13), (13,12), (28,24), (24,28), (29,21), (21,29), (12,20), (20,12), (13,13), (28,25), (25,28), (29,24), (24,29), (12,21), (21,12), (13,20), (20,13), (28,16), (16,28), (29,25), (25,29)
```

## Files

- Score vector: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_5_final_strategy_2d_score_lift_v01\n32_synthetic_preview\score_vectors\N32_synthetic_from_N16_best_u2_primary_s_1d.csv`
- Score matrix: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_5_final_strategy_2d_score_lift_v01\n32_synthetic_preview\score_matrices\N32_synthetic_from_N16_best_u2_primary_snew_2d_unit_32x32.csv`
- Full coordinate CSV: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_5_final_strategy_2d_score_lift_v01\n32_synthetic_preview\N32_synthetic_from_N16_best_u2_primary_score_sorted_coordinate_order.csv`
- Full coordinate JSON: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_5_final_strategy_2d_score_lift_v01\n32_synthetic_preview\N32_synthetic_from_N16_best_u2_primary_score_sorted_coordinate_order.json`
- Python tuple-array text: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_5_final_strategy_2d_score_lift_v01\n32_synthetic_preview\N32_synthetic_from_N16_best_u2_primary_score_sorted_coordinate_order_array.txt`
- Score heatmap: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_5_final_strategy_2d_score_lift_v01\n32_synthetic_preview\plots\N32_synthetic_from_N16_best_u2_primary_snew_2d_heatmap.png`
- Step map: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_5_final_strategy_2d_score_lift_v01\n32_synthetic_preview\plots\N32_synthetic_from_N16_best_u2_primary_score_sorted_step_map.png`
- Polyline preview: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_5_final_strategy_2d_score_lift_v01\n32_synthetic_preview\plots\N32_synthetic_from_N16_best_u2_primary_score_sorted_polyline_preview.png`

## Claim Boundary

- This is synthetic N32, not native Stage 3 evidence.
- This is a score-derived coordinate-order preview.
- It is not teacher validated.
- It does not run Abaqus, ODB extraction, solver, CAE, INP, JNL, or training.
- It does not modify frozen Stage 3 evidence.
- It does not claim physical performance improvement.
