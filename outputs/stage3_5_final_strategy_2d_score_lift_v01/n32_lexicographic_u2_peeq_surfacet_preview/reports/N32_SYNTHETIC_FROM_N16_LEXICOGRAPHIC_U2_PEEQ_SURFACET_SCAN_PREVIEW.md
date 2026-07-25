# Synthetic N32 From N16 Lexicographic U2-PEEQ-SurfaceT Scan Preview

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

- Input table: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\FROZEN_stage3_native_combined552_RL_ready_dataset.csv`
- Source N: 16
- Target N: 32
- Objective: `lexicographic_u2_peeq_surfacet`
- Source strategy: `S3R74FSD_N16_B13_n16_run71_local`
- Source N16 order: `[14, 12, 8, 4, 6, 10, 2, 0, 1, 3, 5, 7, 9, 11, 13, 15]`
- Synthetic N32 order: `[28, 29, 24, 25, 16, 17, 8, 9, 12, 13, 20, 21, 4, 5, 0, 1, 2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31]`
- U2 rank: `1.0`
- PEEQ rank: `13.0`
- SurfaceT rank: `4.0`
- U2 raw: `2.926077831943985e-05`
- PEEQ raw: `0.1494684070348739`
- SurfaceT raw: `580903232.0`

## First 40 Coordinates

```text
(28,28), (28,29), (29,28), (28,24), (24,28), (29,29), (28,25), (25,28), (29,24), (24,29), (28,16), (16,28), (29,25), (25,29), (24,24), (28,17), (17,28), (29,16), (16,29), (24,25), (25,24), (28,8), (8,28), (29,17), (17,29), (24,16), (16,24), (25,25), (28,9), (9,28), (29,8), (8,29), (24,17), (17,24), (25,16), (16,25), (28,12), (12,28), (29,9), (9,29)
```

## Files

- Score vector: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_5_final_strategy_2d_score_lift_v01\n32_lexicographic_u2_peeq_surfacet_preview\score_vectors\N32_synthetic_from_N16_lexicographic_u2_peeq_surfacet_s_1d.csv`
- Score matrix: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_5_final_strategy_2d_score_lift_v01\n32_lexicographic_u2_peeq_surfacet_preview\score_matrices\N32_synthetic_from_N16_lexicographic_u2_peeq_surfacet_snew_2d_unit_32x32.csv`
- Full coordinate CSV: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_5_final_strategy_2d_score_lift_v01\n32_lexicographic_u2_peeq_surfacet_preview\N32_synthetic_from_N16_lexicographic_u2_peeq_surfacet_score_sorted_coordinate_order.csv`
- Full coordinate JSON: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_5_final_strategy_2d_score_lift_v01\n32_lexicographic_u2_peeq_surfacet_preview\N32_synthetic_from_N16_lexicographic_u2_peeq_surfacet_score_sorted_coordinate_order.json`
- Python tuple-array text: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_5_final_strategy_2d_score_lift_v01\n32_lexicographic_u2_peeq_surfacet_preview\N32_synthetic_from_N16_lexicographic_u2_peeq_surfacet_score_sorted_coordinate_order_array.txt`
- Score heatmap: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_5_final_strategy_2d_score_lift_v01\n32_lexicographic_u2_peeq_surfacet_preview\plots\N32_synthetic_from_N16_lexicographic_u2_peeq_surfacet_snew_2d_heatmap.png`
- Step map: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_5_final_strategy_2d_score_lift_v01\n32_lexicographic_u2_peeq_surfacet_preview\plots\N32_synthetic_from_N16_lexicographic_u2_peeq_surfacet_score_sorted_step_map.png`
- Polyline preview: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_5_final_strategy_2d_score_lift_v01\n32_lexicographic_u2_peeq_surfacet_preview\plots\N32_synthetic_from_N16_lexicographic_u2_peeq_surfacet_score_sorted_polyline_preview.png`

## Claim Boundary

- This is synthetic N32, not native Stage 3 evidence.
- This uses the lexicographic U2 -> PEEQ -> SurfaceT selected N16 order.
- This is a score-derived coordinate-order preview.
- It is not teacher validated.
- It does not run Abaqus, ODB extraction, solver, CAE, INP, JNL, or training.
- It does not modify frozen Stage 3 evidence.
- It does not claim physical performance improvement.
