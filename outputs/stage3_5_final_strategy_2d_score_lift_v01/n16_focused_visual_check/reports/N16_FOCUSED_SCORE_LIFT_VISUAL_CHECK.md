# N16 Focused Score-Lift Visual Check

## Purpose

This report checks the N16 Stage 3.5 final-strategy score lift visually and numerically. It derives a one-dimensional score from the frozen final Stage 3 native best order, then lifts it to a 16 by 16 score matrix.

## Source

- Source table: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\stage3_final_native_best_strategy_table.csv`
- N: 16

## Objectives Processed

- best_u2_primary
- best_constrained_reward
- best_strict_penalty_guard
- best_penalty_repair
- best_U2

## Formula

```text
rank(i) = position of track i in final order
s(i) = eps + (1 - 2*eps) * (1 - rank(i)/(N-1))
eps = 1e-6
s_new(i,j) = sqrt((s(i)^2 + s(j)^2) / 2)
```

The score vector `s(i)` was rank-derived from final best `order_json` or `order_compact`. The lifted `s_new(i,j)` output is a score matrix only.

## Summary Table

|objective|final_strategy_name|source_order_column|s_min|s_max|matrix_min|matrix_max|
|---|---|---|---|---|---|---|
|best_u2_primary|S3R74FSD_N16_B01_n16_penalty|best_u2_primary_order_json|1e-06|0.999999|1e-06|0.999999|
|best_constrained_reward|S3R74FSD_N16_B01_n16_penalty|best_constrained_reward_order_compact|1e-06|0.999999|1e-06|0.999999|
|best_strict_penalty_guard|S3R74FSD_N16_B01_n16_penalty|best_strict_penalty_guard_order_compact|1e-06|0.999999|1e-06|0.999999|
|best_penalty_repair|S3R74FSD_N16_B01_n16_penalty|best_penalty_repair_order_json|1e-06|0.999999|1e-06|0.999999|
|best_U2|S3R74FSD_N16_B13_n16_run71_local|best_U2_order_json|1e-06|0.999999|1e-06|0.999999|

## Objective Details And Figures

### best_u2_primary
- Final strategy: `S3R74FSD_N16_B01_n16_penalty`
- Source order column: `best_u2_primary_order_json`
- Final order: `14-6-10-12-8-4-2-0-1-3-5-7-9-11-13-15`
- Score vector: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_5_final_strategy_2d_score_lift_v01\n16_focused_visual_check\score_vectors\N16_best_u2_primary_s_1d.csv`
- Score matrix: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_5_final_strategy_2d_score_lift_v01\n16_focused_visual_check\score_matrices\N16_best_u2_primary_snew_2d_unit_16x16.csv`
- Bar plot: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_5_final_strategy_2d_score_lift_v01\n16_focused_visual_check\plots\N16_best_u2_primary_s_1d_bar.png`
- Heatmap: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_5_final_strategy_2d_score_lift_v01\n16_focused_visual_check\plots\N16_best_u2_primary_snew_2d_heatmap.png`
- Combined plot: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_5_final_strategy_2d_score_lift_v01\n16_focused_visual_check\plots\N16_best_u2_primary_combined_score_lift.png`
- Rank bar: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_5_final_strategy_2d_score_lift_v01\n16_focused_visual_check\plots\N16_best_u2_primary_order_rank_bar.png`
### best_constrained_reward
- Final strategy: `S3R74FSD_N16_B01_n16_penalty`
- Source order column: `best_constrained_reward_order_compact`
- Final order: `14-6-10-12-8-4-2-0-1-3-5-7-9-11-13-15`
- Score vector: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_5_final_strategy_2d_score_lift_v01\n16_focused_visual_check\score_vectors\N16_best_constrained_reward_s_1d.csv`
- Score matrix: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_5_final_strategy_2d_score_lift_v01\n16_focused_visual_check\score_matrices\N16_best_constrained_reward_snew_2d_unit_16x16.csv`
- Bar plot: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_5_final_strategy_2d_score_lift_v01\n16_focused_visual_check\plots\N16_best_constrained_reward_s_1d_bar.png`
- Heatmap: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_5_final_strategy_2d_score_lift_v01\n16_focused_visual_check\plots\N16_best_constrained_reward_snew_2d_heatmap.png`
- Combined plot: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_5_final_strategy_2d_score_lift_v01\n16_focused_visual_check\plots\N16_best_constrained_reward_combined_score_lift.png`
- Rank bar: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_5_final_strategy_2d_score_lift_v01\n16_focused_visual_check\plots\N16_best_constrained_reward_order_rank_bar.png`
### best_strict_penalty_guard
- Final strategy: `S3R74FSD_N16_B01_n16_penalty`
- Source order column: `best_strict_penalty_guard_order_compact`
- Final order: `14-6-10-12-8-4-2-0-1-3-5-7-9-11-13-15`
- Score vector: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_5_final_strategy_2d_score_lift_v01\n16_focused_visual_check\score_vectors\N16_best_strict_penalty_guard_s_1d.csv`
- Score matrix: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_5_final_strategy_2d_score_lift_v01\n16_focused_visual_check\score_matrices\N16_best_strict_penalty_guard_snew_2d_unit_16x16.csv`
- Bar plot: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_5_final_strategy_2d_score_lift_v01\n16_focused_visual_check\plots\N16_best_strict_penalty_guard_s_1d_bar.png`
- Heatmap: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_5_final_strategy_2d_score_lift_v01\n16_focused_visual_check\plots\N16_best_strict_penalty_guard_snew_2d_heatmap.png`
- Combined plot: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_5_final_strategy_2d_score_lift_v01\n16_focused_visual_check\plots\N16_best_strict_penalty_guard_combined_score_lift.png`
- Rank bar: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_5_final_strategy_2d_score_lift_v01\n16_focused_visual_check\plots\N16_best_strict_penalty_guard_order_rank_bar.png`
### best_penalty_repair
- Final strategy: `S3R74FSD_N16_B01_n16_penalty`
- Source order column: `best_penalty_repair_order_json`
- Final order: `14-6-10-12-8-4-2-0-1-3-5-7-9-11-13-15`
- Score vector: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_5_final_strategy_2d_score_lift_v01\n16_focused_visual_check\score_vectors\N16_best_penalty_repair_s_1d.csv`
- Score matrix: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_5_final_strategy_2d_score_lift_v01\n16_focused_visual_check\score_matrices\N16_best_penalty_repair_snew_2d_unit_16x16.csv`
- Bar plot: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_5_final_strategy_2d_score_lift_v01\n16_focused_visual_check\plots\N16_best_penalty_repair_s_1d_bar.png`
- Heatmap: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_5_final_strategy_2d_score_lift_v01\n16_focused_visual_check\plots\N16_best_penalty_repair_snew_2d_heatmap.png`
- Combined plot: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_5_final_strategy_2d_score_lift_v01\n16_focused_visual_check\plots\N16_best_penalty_repair_combined_score_lift.png`
- Rank bar: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_5_final_strategy_2d_score_lift_v01\n16_focused_visual_check\plots\N16_best_penalty_repair_order_rank_bar.png`
### best_U2
- Final strategy: `S3R74FSD_N16_B13_n16_run71_local`
- Source order column: `best_U2_order_json`
- Final order: `14-12-8-4-6-10-2-0-1-3-5-7-9-11-13-15`
- Score vector: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_5_final_strategy_2d_score_lift_v01\n16_focused_visual_check\score_vectors\N16_best_U2_s_1d.csv`
- Score matrix: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_5_final_strategy_2d_score_lift_v01\n16_focused_visual_check\score_matrices\N16_best_U2_snew_2d_unit_16x16.csv`
- Bar plot: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_5_final_strategy_2d_score_lift_v01\n16_focused_visual_check\plots\N16_best_U2_s_1d_bar.png`
- Heatmap: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_5_final_strategy_2d_score_lift_v01\n16_focused_visual_check\plots\N16_best_U2_snew_2d_heatmap.png`
- Combined plot: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_5_final_strategy_2d_score_lift_v01\n16_focused_visual_check\plots\N16_best_U2_combined_score_lift.png`
- Rank bar: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_5_final_strategy_2d_score_lift_v01\n16_focused_visual_check\plots\N16_best_U2_order_rank_bar.png`

## Claim Boundary

- No scan order generated.
- No teacher validation.
- No Abaqus.
- No physical performance claim.
- No 256-point path generated.
- No ODB, solver, CAE, INP, JNL, or training action.
