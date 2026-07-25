# Section 3.4 Multi-N Scan-Order Optimisation Outcomes Summary

## Source Files Used

- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\FROZEN_stage3_native_combined552_teacher_dataset.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\FROZEN_stage3_native_combined552_RL_ready_dataset.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\FROZEN_stage3_native_combined552_per_N_leaderboard.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\FROZEN_stage3_native_combined552_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\stage3_final_native_best_strategy_table.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\stage3_final_native_top5_by_N_and_objective.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\stage3_final_native_top10_by_N_and_objective.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\stage3_final_native_topk_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\stage3_final_claim_evidence_map.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\stage3_final_claim_boundary.md`

## Native Counts By N

| N   |   teacher_label_count |
|:----|----------------------:|
| N12 |                    78 |
| N16 |                    78 |
| N24 |                   190 |
| N40 |                   206 |


## Best Teacher-Labelled Scan Orders By N


### U2

| N   | best_strategy                     |   best_value | candidate_family   | candidate_source                   | source_run   |
|:----|:----------------------------------|-------------:|:-------------------|:-----------------------------------|:-------------|
| N12 | S3R69SNR_N12_B16_n12_uncertainty  |  2.201e-05   | uncertainty        | N12_recovery_uncertainty           | Run71        |
| N16 | S3R74FSD_N16_B13_n16_run71_local  |  2.92608e-05 | recovery_anchor    | N16_local_search_around_Run71_best | Run76        |
| N24 | S3R49SCN_N24_B09_median_guard     |  2.95231e-05 | penalty_repair     | no_penalty_worse_than_median       | Run51        |
| N40 | S3R64VNR_N40_B15_n40_u2ret_anchor |  4.57793e-05 | surrogate_guided   | N40_followup_u2_reward_retention   | Run66        |

### u2_primary

| N   | best_strategy                       |   best_value | candidate_family   | candidate_source             | source_run   |
|:----|:------------------------------------|-------------:|:-------------------|:-----------------------------|:-------------|
| N12 | S3R74FSD_N12_B02_n12_penalty        |     0.897727 | penalty_repair     | N12_final_diag_penalty_aware | Run76        |
| N16 | S3R74FSD_N16_B01_n16_penalty        |     0.923377 | penalty_repair     | N16_final_diag_penalty_aware | Run76        |
| N24 | S3R44CNS_N24_B16_uncertainty        |     0.871825 | penalty_repair     | uncertainty_calibration      | Run46        |
| N40 | S3R64VNR_N40_B03_n40_penalty_anchor |     0.939512 | penalty_repair     | N40_followup_penalty_repair  | Run66        |

### constrained_reward

| N   | best_strategy                       |   best_value | candidate_family   | candidate_source             | source_run   |
|:----|:------------------------------------|-------------:|:-------------------|:-----------------------------|:-------------|
| N12 | S3R74FSD_N12_B02_n12_penalty        |     0.859091 | penalty_repair     | N12_final_diag_penalty_aware | Run76        |
| N16 | S3R74FSD_N16_B01_n16_penalty        |     0.880519 | penalty_repair     | N16_final_diag_penalty_aware | Run76        |
| N24 | S3R44CNS_N24_B16_uncertainty        |     0.85582  | penalty_repair     | uncertainty_calibration      | Run46        |
| N40 | S3R64VNR_N40_B03_n40_penalty_anchor |     0.909756 | penalty_repair     | N40_followup_penalty_repair  | Run66        |

### strict_penalty_guard

| N   | best_strategy                       |   best_value | candidate_family   | candidate_source             | source_run   |
|:----|:------------------------------------|-------------:|:-------------------|:-----------------------------|:-------------|
| N12 | S3R74FSD_N12_B02_n12_penalty        |     0.853896 | penalty_repair     | N12_final_diag_penalty_aware | Run76        |
| N16 | S3R74FSD_N16_B01_n16_penalty        |     0.881818 | penalty_repair     | N16_final_diag_penalty_aware | Run76        |
| N24 | S3R44CNS_N24_B16_uncertainty        |     0.850529 | penalty_repair     | uncertainty_calibration      | Run46        |
| N40 | S3R64VNR_N40_B03_n40_penalty_anchor |     0.898537 | penalty_repair     | N40_followup_penalty_repair  | Run66        |

### penalty_repair

| N   | best_strategy                       |   best_value | candidate_family   | candidate_source             | source_run   |
|:----|:------------------------------------|-------------:|:-------------------|:-----------------------------|:-------------|
| N12 | S3R74FSD_N12_B01_n12_penalty        |     0.827597 | penalty_repair     | N12_final_diag_penalty_aware | Run76        |
| N16 | S3R74FSD_N16_B01_n16_penalty        |     0.837013 | penalty_repair     | N16_final_diag_penalty_aware | Run76        |
| N24 | S3R69SNR_N24_B04_n24_uncertainty    |     0.846693 | uncertainty        | N24_uncertainty_anchor       | Run71        |
| N40 | S3R64VNR_N40_B03_n40_penalty_anchor |     0.876098 | penalty_repair     | N40_followup_penalty_repair  | Run66        |

### PEEQ

| N   | best_strategy                                      |   best_value | candidate_family    | candidate_source             | source_run   |
|:----|:---------------------------------------------------|-------------:|:--------------------|:-----------------------------|:-------------|
| N12 | N12_A13_graph_pointer_policy_anti_odd_even_novelty |     0.140322 | graph_pointer_proxy | probe60_run08                | Run08        |
| N16 | S3R74FSD_N16_B01_n16_penalty                       |     0.145787 | penalty_repair      | N16_final_diag_penalty_aware | Run76        |
| N24 | S3R59N40PR40_N24_B09_n24_diversity                 |     0.149217 | diversity           | N24_diversity_coverage       | Run61        |
| N40 | S3R39N2440B60_N40_B10_surrogate_top                |     0.144025 | surrogate_guided    | surrogate_top_predicted      | Run41        |

### SurfaceT

| N   | best_strategy                            |   best_value | candidate_family         | candidate_source                | source_run   |
|:----|:-----------------------------------------|-------------:|:-------------------------|:--------------------------------|:-------------|
| N12 | N12_A09_center_edge_alternating          |  5.80753e+08 | center_edge              | probe60_run08                   | Run08        |
| N16 | S3R74FSD_N16_B03_n16_reward_bal          |  5.80887e+08 | penalty_repair           | N16_final_diag_reward_balanced  | Run76        |
| N24 | S3R24L64_N24_B06_model_disagreement      |  5.80876e+08 | known_best_mutation      | combined108_known_best_mutation | Run27        |
| N40 | S3R19B28_N40_B06_uncertainty_calibration |  5.81119e+08 | regular_jump_non_coprime | regular_jump_sweep              | Run20        |

### Mises

| N   | best_strategy                |   best_value | candidate_family   | candidate_source   | source_run   |
|:----|:-----------------------------|-------------:|:-------------------|:-------------------|:-------------|
| N12 | N12_A06_edge_in_alternating  |  5.79938e+08 | edge_in            | probe60_run08      | Run08        |
| N16 | N16_A07_regular_jump_coprime |  5.79947e+08 | regular_jump       | probe60_run08      | Run08        |
| N24 | N24_A06_edge_in_alternating  |  5.79943e+08 | edge_in            | probe60_run08      | Run08        |
| N40 | N40_A07_regular_jump_coprime |  5.79924e+08 | regular_jump       | probe60_run08      | Run08        |


## U2-Best Versus Reward-View Agreement

| N   | reward_view          | same_as_U2_best   | U2_best_strategy                  | reward_best_strategy                |
|:----|:---------------------|:------------------|:----------------------------------|:------------------------------------|
| N12 | u2_primary           | False             | S3R69SNR_N12_B16_n12_uncertainty  | S3R74FSD_N12_B02_n12_penalty        |
| N12 | constrained_reward   | False             | S3R69SNR_N12_B16_n12_uncertainty  | S3R74FSD_N12_B02_n12_penalty        |
| N12 | strict_penalty_guard | False             | S3R69SNR_N12_B16_n12_uncertainty  | S3R74FSD_N12_B02_n12_penalty        |
| N12 | penalty_repair       | False             | S3R69SNR_N12_B16_n12_uncertainty  | S3R74FSD_N12_B01_n12_penalty        |
| N16 | u2_primary           | False             | S3R74FSD_N16_B13_n16_run71_local  | S3R74FSD_N16_B01_n16_penalty        |
| N16 | constrained_reward   | False             | S3R74FSD_N16_B13_n16_run71_local  | S3R74FSD_N16_B01_n16_penalty        |
| N16 | strict_penalty_guard | False             | S3R74FSD_N16_B13_n16_run71_local  | S3R74FSD_N16_B01_n16_penalty        |
| N16 | penalty_repair       | False             | S3R74FSD_N16_B13_n16_run71_local  | S3R74FSD_N16_B01_n16_penalty        |
| N24 | u2_primary           | False             | S3R49SCN_N24_B09_median_guard     | S3R44CNS_N24_B16_uncertainty        |
| N24 | constrained_reward   | False             | S3R49SCN_N24_B09_median_guard     | S3R44CNS_N24_B16_uncertainty        |
| N24 | strict_penalty_guard | False             | S3R49SCN_N24_B09_median_guard     | S3R44CNS_N24_B16_uncertainty        |
| N24 | penalty_repair       | False             | S3R49SCN_N24_B09_median_guard     | S3R69SNR_N24_B04_n24_uncertainty    |
| N40 | u2_primary           | False             | S3R64VNR_N40_B15_n40_u2ret_anchor | S3R64VNR_N40_B03_n40_penalty_anchor |
| N40 | constrained_reward   | False             | S3R64VNR_N40_B15_n40_u2ret_anchor | S3R64VNR_N40_B03_n40_penalty_anchor |
| N40 | strict_penalty_guard | False             | S3R64VNR_N40_B15_n40_u2ret_anchor | S3R64VNR_N40_B03_n40_penalty_anchor |
| N40 | penalty_repair       | False             | S3R64VNR_N40_B15_n40_u2ret_anchor | S3R64VNR_N40_B03_n40_penalty_anchor |


## Short Interpretation By N

- `N12`: U2 best is `S3R69SNR_N12_B16_n12_uncertainty`. The constrained-reward best is `S3R74FSD_N12_B02_n12_penalty` and penalty-repair best is `S3R74FSD_N12_B01_n12_penalty`; 0 of 4 reward views select the same strategy as the U2 metric winner.
- `N16`: U2 best is `S3R74FSD_N16_B13_n16_run71_local`. The constrained-reward best is `S3R74FSD_N16_B01_n16_penalty` and penalty-repair best is `S3R74FSD_N16_B01_n16_penalty`; 0 of 4 reward views select the same strategy as the U2 metric winner.
- `N24`: U2 best is `S3R49SCN_N24_B09_median_guard`. The constrained-reward best is `S3R44CNS_N24_B16_uncertainty` and penalty-repair best is `S3R69SNR_N24_B04_n24_uncertainty`; 0 of 4 reward views select the same strategy as the U2 metric winner.
- `N40`: U2 best is `S3R64VNR_N40_B15_n40_u2ret_anchor`. The constrained-reward best is `S3R64VNR_N40_B03_n40_penalty_anchor` and penalty-repair best is `S3R64VNR_N40_B03_n40_penalty_anchor`; 0 of 4 reward views select the same strategy as the U2 metric winner.

## Top-K Family Occupancy Summary

| family_label            |   Mises |   PEEQ |   SurfaceT |   U2 |   constrained_reward |   penalty_repair |   strict_penalty_guard |   u2_primary |
|:------------------------|--------:|-------:|-----------:|-----:|---------------------:|-----------------:|-----------------------:|-------------:|
| engineering_baseline    |      13 |      3 |          3 |    0 |                    0 |                0 |                      0 |            0 |
| penalty_repair          |       0 |      8 |         16 |   19 |                   22 |               23 |                     23 |           20 |
| recovery_anchor         |       5 |      5 |          6 |    7 |                    6 |                5 |                      5 |            6 |
| uncertainty             |       1 |      2 |          5 |    3 |                    4 |                5 |                      5 |            2 |
| diversity               |       6 |      7 |          0 |    2 |                    2 |                1 |                      2 |            3 |
| surrogate_guided        |       3 |     10 |          4 |    9 |                    6 |                6 |                      5 |            9 |
| graph_pointer_or_gnn_rl |      10 |      3 |          3 |    0 |                    0 |                0 |                      0 |            0 |
| method_c                |       1 |      0 |          0 |    0 |                    0 |                0 |                      0 |            0 |
| other                   |       1 |      2 |          3 |    0 |                    0 |                0 |                      0 |            0 |


Broad-family inference rules used for top-k occupancy: method_c keywords -> `method_c`; penalty/repair/guard/constrained/reward-balanced keywords -> `penalty_repair`; graph-pointer/GNN keywords -> `graph_pointer_or_gnn_rl`; surrogate/model-disagreement/hybrid prediction keywords -> `surrogate_guided`; uncertainty/calibration keywords -> `uncertainty`; diversity/novelty/coverage/maximin keywords -> `diversity`; recovery/anchor/local/known-best mutation keywords -> `recovery_anchor`; raster/odd-even/center/edge/regular-jump/block/geometry engineering patterns -> `engineering_baseline`; otherwise `other`.


## Generated Outputs

- `table_3_4_best_teacher_labelled_scan_orders_by_N.csv`
- `figure_3_8_per_N_leaderboard.csv/png/pdf/svg`
- `figure_3_9_topk_family_occupancy.csv/png/pdf/svg`
- `figure_3_10_best_scan_order_patterns_across_N.png/pdf/svg`
- `section_3_4_claim_boundary_note.md`

## Claim Boundary

- Main native claim scope only: N12, N16, N24, and N40.
- N32 is auxiliary fixed-N32 context only if mentioned and is not merged into the native multi-N outcomes.
- Within-N rankings only; raw cross-N objective values are not treated as directly comparable.
- No global optimum claim.
- No arbitrary-N generalisation claim.
- No claim that the framework beats all possible scan strategies.
- No online RL control or physical experiment validation claim.