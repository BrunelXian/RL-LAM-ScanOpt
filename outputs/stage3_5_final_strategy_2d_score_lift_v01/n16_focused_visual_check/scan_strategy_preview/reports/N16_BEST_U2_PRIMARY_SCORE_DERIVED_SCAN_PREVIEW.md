# N16 best_u2_primary Score-Derived Scan Preview

## What This Is

This is a 16 by 16 coordinate-order preview derived from the Stage 3.5 lifted score field.

It sorts all 256 cells by:

```text
descending s_new(i,j), then rank_i, rank_j, i, j
```

## Source

- Input table: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\stage3_final_native_best_strategy_table.csv`
- Objective: `best_u2_primary`
- Final strategy: `S3R74FSD_N16_B01_n16_penalty`
- Source order column: `best_u2_primary_order_json`
- Final 1D order: `[14, 6, 10, 12, 8, 4, 2, 0, 1, 3, 5, 7, 9, 11, 13, 15]`

## First 20 Coordinates

```text
(14,14), (14,6), (6,14), (14,10), (10,14), (6,6), (14,12), (12,14), (6,10), (10,6), (14,8), (8,14), (6,12), (12,6), (10,10), (14,4), (4,14), (6,8), (8,6), (10,12)
```

## Files

- Full CSV order: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_5_final_strategy_2d_score_lift_v01\n16_focused_visual_check\scan_strategy_preview\N16_best_u2_primary_score_sorted_coordinate_order.csv`
- Full JSON order: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_5_final_strategy_2d_score_lift_v01\n16_focused_visual_check\scan_strategy_preview\N16_best_u2_primary_score_sorted_coordinate_order.json`
- Python tuple-array text: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_5_final_strategy_2d_score_lift_v01\n16_focused_visual_check\scan_strategy_preview\N16_best_u2_primary_score_sorted_coordinate_order_array.txt`
- Step map: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_5_final_strategy_2d_score_lift_v01\n16_focused_visual_check\scan_strategy_preview\plots\N16_best_u2_primary_score_sorted_step_map.png`
- Polyline preview: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_5_final_strategy_2d_score_lift_v01\n16_focused_visual_check\scan_strategy_preview\plots\N16_best_u2_primary_score_sorted_polyline_preview.png`

## Claim Boundary

- This is a score-derived coordinate-order preview.
- It is not teacher validated.
- It does not run Abaqus, ODB extraction, solver, CAE, INP, JNL, or training.
- It does not modify frozen Stage 3 evidence.
- It does not claim physical performance improvement.
