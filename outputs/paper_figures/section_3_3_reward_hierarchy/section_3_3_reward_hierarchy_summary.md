# Section 3.3 Physically Constrained Reward Hierarchy Summary

## Source Files Used

- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\FROZEN_stage3_native_combined552_teacher_dataset.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\FROZEN_stage3_native_combined552_RL_ready_dataset.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\FROZEN_stage3_native_combined552_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\stage3_final_native_best_strategy_table.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\stage3_final_native_top5_by_N_and_objective.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\stage3_final_native_top10_by_N_and_objective.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\stage3_final_metric_reward_record_timeline.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\stage3_final_claim_evidence_map.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\stage3_final_claim_boundary.md`

## Exact Columns Used

- N: `n`
- Strategy identifier: `strategy_name`
- Family label: `candidate_family`
- Candidate source: `candidate_source`
- U2: `u2_range`
- PEEQ: `peeq_max`
- SurfaceT: `surface_t_proxy`
- Mises: `mises_max`
- Within-N ranks/norms: recomputed from native rows for transparent Section 3.3 analysis
- Constrained reward retained when available: `target_reward_combined552_constrained_u2_reward_balanced`

## Guard Column Treatment

Existing reward/rank columns were found, including `target_reward_combined552_constrained_u2_reward_balanced`, but no explicit binary U2/PEEQ feasibility guard column was identified. The U2/PEEQ gate statuses in this section therefore use the requested within-N analysis thresholds for visualisation.


Analysis thresholds used for visualisation only: `U2_safe = u2_norm_within_N <= 0.25`; `U2_near_safe = u2_norm_within_N <= 0.50`; `PEEQ_safe = peeq_norm_within_N <= 0.75`; `combined_safe = U2_safe AND PEEQ_safe`; `near_admissible = U2_near_safe AND PEEQ_safe`. These are not physical experimental limits.


## Native Counts And Gate Counts

| N          |   row_count |   number_of_U2_safe_cases |   number_of_PEEQ_safe_cases |   number_of_combined_safe_cases |   number_of_near_admissible_cases |   count_of_low_SurfaceT_but_high_U2_cases |   count_of_low_U2_but_high_SurfaceT_cases |
|:-----------|------------:|--------------------------:|----------------------------:|--------------------------------:|----------------------------------:|------------------------------------------:|------------------------------------------:|
| all_native |         552 |                       406 |                         532 |                             405 |                               478 |                                        15 |                                         6 |
| N12        |          78 |                        60 |                          74 |                              60 |                                69 |                                         2 |                                         3 |
| N16        |          78 |                        61 |                          73 |                              60 |                                65 |                                         3 |                                         0 |
| N24        |         190 |                       138 |                         185 |                             138 |                               166 |                                         7 |                                         1 |
| N40        |         206 |                       147 |                         200 |                             147 |                               178 |                                         3 |                                         2 |


## Metric Availability

| N   |   rows |   u2_range_valid |   peeq_max_valid |   surface_t_proxy_valid |   mises_max_valid |
|:----|-------:|-----------------:|-----------------:|------------------------:|------------------:|
| N12 |     78 |               78 |               78 |                      78 |                78 |
| N16 |     78 |               78 |               78 |                      78 |                78 |
| N24 |    190 |              190 |              190 |                     190 |               190 |
| N40 |    206 |              206 |              206 |                     206 |               206 |


## U2 vs SurfaceT Correlation Summary

| N          |   spearman_u2_vs_surface_t_raw |   pearson_u2_vs_surface_t_raw |   spearman_u2_norm_vs_surface_t_norm |   pearson_u2_norm_vs_surface_t_norm |
|:-----------|-------------------------------:|------------------------------:|-------------------------------------:|------------------------------------:|
| all_native |                         0.5414 |                        0.4703 |                               0.3963 |                              0.3044 |
| N12        |                         0.1690 |                        0.1031 |                               0.1690 |                              0.1031 |
| N16        |                         0.4686 |                        0.2411 |                               0.4686 |                              0.2411 |
| N24        |                         0.2136 |                        0.3053 |                               0.2136 |                              0.3053 |
| N40        |                         0.6577 |                        0.4324 |                               0.6577 |                              0.4324 |


## Relationship Interpretation

- Across all native rows, Spearman correlation between raw U2 and SurfaceT is `0.5414`; within-N normalized Spearman correlation is `0.3963`.
- Low SurfaceT does not guarantee low U2: `15` native rows have low SurfaceT but high U2 under the within-N definitions.
- Low U2 can coexist with high SurfaceT: `6` native rows have low U2 but high SurfaceT.
- Residual-stress-first optimisation is therefore unsafe as the primary reward hierarchy for this evidence set.
- PEEQ acts as a safety filter after U2 admissibility.
- SurfaceT remains useful as a secondary objective inside the admissible or near-admissible U2+PEEQ region.
- Mises is retained as a diagnostic rather than a final primary reward component.

## SurfaceT Ranges By Gate Region

| N          |   SurfaceT_min_inside_combined_safe |   SurfaceT_median_inside_combined_safe |   SurfaceT_range_inside_combined_safe |   SurfaceT_min_inside_near_admissible |   SurfaceT_median_inside_near_admissible |   SurfaceT_range_inside_near_admissible |   SurfaceT_min_outside_U2_safe |   SurfaceT_median_outside_U2_safe |   SurfaceT_range_outside_U2_safe |
|:-----------|------------------------------------:|---------------------------------------:|--------------------------------------:|--------------------------------------:|-----------------------------------------:|----------------------------------------:|-------------------------------:|----------------------------------:|---------------------------------:|
| all_native |                         5.8084e+08  |                            5.82097e+08 |                           1.76951e+07 |                           5.8084e+08  |                              5.82185e+08 |                             1.76951e+07 |                    5.80753e+08 |                       5.83401e+08 |                      1.48297e+07 |
| N12        |                         5.8084e+08  |                            5.81178e+08 |                           5.07482e+06 |                           5.8084e+08  |                              5.81279e+08 |                             5.07482e+06 |                    5.80753e+08 |                       5.81855e+08 |                      5.17786e+06 |
| N16        |                         5.80887e+08 |                            5.81441e+08 |                           5.02829e+06 |                           5.80887e+08 |                              5.81444e+08 |                             5.02829e+06 |                    5.809e+08   |                       5.82182e+08 |                      6.94202e+06 |
| N24        |                         5.80876e+08 |                            5.82202e+08 |                           1.11594e+07 |                           5.80876e+08 |                              5.82387e+08 |                             1.11594e+07 |                    5.81205e+08 |                       5.8324e+08  |                      9.3865e+06  |
| N40        |                         5.81119e+08 |                            5.82293e+08 |                           1.74163e+07 |                           5.81119e+08 |                              5.82639e+08 |                             1.74163e+07 |                    5.81542e+08 |                       5.84743e+08 |                      1.40407e+07 |


## Generated Outputs

- `section_3_3_reward_hierarchy_analysis_native.csv`
- `section_3_3_u2_surfacet_relationship_summary.csv`
- `table_3_3_reward_hierarchy_evidence.csv`
- `figure_3_6_u2_surfacet_feasible_window_map.png/pdf/svg`
- `figure_3_7_reward_hierarchy_schematic.png/pdf/svg`

## Claim Boundary

- Native N only: N12, N16, N24, and N40.
- N32 is excluded from Section 3.3 main figures and tables.
- No global optimum claim.
- No arbitrary-N generalisation claim.
- No physical experiment validation claim.
- No claim that SurfaceT-first optimisation solved the residual-stress problem.