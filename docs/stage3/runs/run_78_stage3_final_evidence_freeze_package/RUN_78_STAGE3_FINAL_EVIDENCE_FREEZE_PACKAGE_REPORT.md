# Stage 3 Run 78 - Final Evidence Freeze Package

## 1. Purpose
Run78 freezes the final Stage 3 evidence package for reporting, paper writing, and ARA-style evidence archiving. It is not a new optimization run.

## 2. Inputs
- Native combined552 teacher dataset: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_77_final_smallN_diagnostic_batch32_teacher_metrics_ingestion_and_combined552_final_evidence_readiness\combined552_teacher_dataset.csv`
- Native combined552 RL-ready dataset: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_77_final_smallN_diagnostic_batch32_teacher_metrics_ingestion_and_combined552_final_evidence_readiness\combined552_RL_ready_dataset.csv`
- combined552_plus_N32 RL-ready dataset: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_77_final_smallN_diagnostic_batch32_teacher_metrics_ingestion_and_combined552_final_evidence_readiness\combined552_plus_N32_RL_ready_dataset.csv`
- Run77 maturity audit: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_77_final_smallN_diagnostic_batch32_teacher_metrics_ingestion_and_combined552_final_evidence_readiness\stage3_final_maturity_and_evidence_freeze_readiness_audit.md`
- Run77 claim boundary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_77_final_smallN_diagnostic_batch32_teacher_metrics_ingestion_and_combined552_final_evidence_readiness\run77_final_claim_boundary.md`

## 3. Run77 Final Readiness Verdict
`RUN77_STAGE3_FINAL_EVIDENCE_FREEZE_READY_WITH_BOUNDED_NATIVE_N_CLAIMS`

## 4. Frozen Datasets and File Hashes
- File hashes: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\FROZEN_stage3_file_hashes.csv`
- Frozen file count: 10

## 5. Final Native combined552 Summary
- Rows: 552
- Per-N counts: {12: 78, 16: 78, 24: 190, 40: 206}

## 6. combined552_plus_N32 Auxiliary Summary
- Rows: 884
- Per-N counts: {12: 78, 16: 78, 24: 190, 32: 332, 40: 206}
- N32 is auxiliary legacy-compatible context, not native Stage 3 teacher validation.

## 7. Final Best Strategy Table
| n | best_U2_strategy | best_U2_value | best_U2_dataset_source | best_U2_source_run | best_U2_teacher_validation_status | best_U2_order_hash | best_U2_order_compact | best_U2_order_json | best_PEEQ_strategy | best_PEEQ_value | best_PEEQ_dataset_source | best_PEEQ_source_run | best_PEEQ_teacher_validation_status | best_PEEQ_order_hash | best_PEEQ_order_compact | best_SurfaceT_strategy | best_SurfaceT_value | best_SurfaceT_dataset_source | best_SurfaceT_source_run | best_SurfaceT_teacher_validation_status | best_SurfaceT_order_hash | best_SurfaceT_order_compact | best_Mises_strategy | best_Mises_value | best_Mises_dataset_source | best_Mises_source_run | best_Mises_teacher_validation_status | best_Mises_order_hash | best_Mises_order_compact | best_u2_primary_strategy | best_u2_primary_value | best_u2_primary_dataset_source | best_u2_primary_source_run | best_u2_primary_teacher_validation_status | best_u2_primary_order_hash | best_u2_primary_order_compact | best_u2_primary_order_json | best_constrained_reward_strategy | best_constrained_reward_value | best_constrained_reward_dataset_source | best_constrained_reward_source_run | best_constrained_reward_teacher_validation_status | best_constrained_reward_order_hash | best_constrained_reward_order_compact | best_strict_penalty_guard_strategy | best_strict_penalty_guard_value | best_strict_penalty_guard_dataset_source | best_strict_penalty_guard_source_run | best_strict_penalty_guard_teacher_validation_status | best_strict_penalty_guard_order_hash | best_strict_penalty_guard_order_compact | best_penalty_repair_strategy | best_penalty_repair_value | best_penalty_repair_dataset_source | best_penalty_repair_source_run | best_penalty_repair_teacher_validation_status | best_penalty_repair_order_hash | best_penalty_repair_order_compact | best_penalty_repair_order_json |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 12 | S3R69SNR_N12_B16_n12_uncertainty | 2.20100214392005e-05 | run71_smallN_recovery_focused_batch40 | Run71 | PASS_TEACHER_FIELDS_EXTRACTED | 2bda5fb0f837e0b2 | 11-9-5-7-3-1-0-2-4-6-8-10 | [11,9,5,7,3,1,0,2,4,6,8,10] | N12_A13_graph_pointer_policy_anti_odd_even_novelty | 0.1403219699859619 | probe60_run08 | Run08 | PASS_ODB_EXTRACTED |  |  | N12_A09_center_edge_alternating | 580752832.0 | probe60_run08 | Run08 | PASS_ODB_EXTRACTED |  |  | N12_A06_edge_in_alternating | 579937984.0 | probe60_run08 | Run08 | PASS_ODB_EXTRACTED |  |  | S3R74FSD_N12_B02_n12_penalty | 0.8977272727272728 | run76_final_smallN_diagnostic_batch32 | Run76 | PASS_TEACHER_FIELDS_EXTRACTED | 386d1b07eed1cd5a | 9-11-7-5-3-1-0-2-4-6-8-10 | [9,11,7,5,3,1,0,2,4,6,8,10] | S3R74FSD_N12_B02_n12_penalty | 0.859090909090909 | run76_final_smallN_diagnostic_batch32 | Run76 | PASS_TEACHER_FIELDS_EXTRACTED | 386d1b07eed1cd5a | 9-11-7-5-3-1-0-2-4-6-8-10 | S3R74FSD_N12_B02_n12_penalty | 0.8538961038961039 | run76_final_smallN_diagnostic_batch32 | Run76 | PASS_TEACHER_FIELDS_EXTRACTED | 386d1b07eed1cd5a | 9-11-7-5-3-1-0-2-4-6-8-10 | S3R74FSD_N12_B01_n12_penalty | 0.8275974025974026 | run76_final_smallN_diagnostic_batch32 | Run76 | PASS_TEACHER_FIELDS_EXTRACTED | 7e02169f5069d99d | 11-7-9-5-3-1-0-2-4-6-8-10 | [11,7,9,5,3,1,0,2,4,6,8,10] |
| 16 | S3R74FSD_N16_B13_n16_run71_local | 2.926077831943985e-05 | run76_final_smallN_diagnostic_batch32 | Run76 | PASS_TEACHER_FIELDS_EXTRACTED | cb0a72a06b7d08c9 | 14-12-8-4-6-10-2-0-1-3-5-7-9-11-13-15 | [14,12,8,4,6,10,2,0,1,3,5,7,9,11,13,15] | S3R74FSD_N16_B01_n16_penalty | 0.1457871496677398 | run76_final_smallN_diagnostic_batch32 | Run76 | PASS_TEACHER_FIELDS_EXTRACTED | 5e0129a21f85cd34 | 14-6-10-12-8-4-2-0-1-3-5-7-9-11-13-15 | S3R74FSD_N16_B03_n16_reward_bal | 580887040.0 | run76_final_smallN_diagnostic_batch32 | Run76 | PASS_TEACHER_FIELDS_EXTRACTED | 543b47f89dc534e4 | 14-12-10-8-6-4-2-0-1-3-5-7-9-11-13-15 | N16_A07_regular_jump_coprime | 579946752.0 | probe60_run08 | Run08 | PASS_ODB_EXTRACTED |  |  | S3R74FSD_N16_B01_n16_penalty | 0.9233766233766234 | run76_final_smallN_diagnostic_batch32 | Run76 | PASS_TEACHER_FIELDS_EXTRACTED | 5e0129a21f85cd34 | 14-6-10-12-8-4-2-0-1-3-5-7-9-11-13-15 | [14,6,10,12,8,4,2,0,1,3,5,7,9,11,13,15] | S3R74FSD_N16_B01_n16_penalty | 0.8805194805194806 | run76_final_smallN_diagnostic_batch32 | Run76 | PASS_TEACHER_FIELDS_EXTRACTED | 5e0129a21f85cd34 | 14-6-10-12-8-4-2-0-1-3-5-7-9-11-13-15 | S3R74FSD_N16_B01_n16_penalty | 0.8818181818181817 | run76_final_smallN_diagnostic_batch32 | Run76 | PASS_TEACHER_FIELDS_EXTRACTED | 5e0129a21f85cd34 | 14-6-10-12-8-4-2-0-1-3-5-7-9-11-13-15 | S3R74FSD_N16_B01_n16_penalty | 0.837012987012987 | run76_final_smallN_diagnostic_batch32 | Run76 | PASS_TEACHER_FIELDS_EXTRACTED | 5e0129a21f85cd34 | 14-6-10-12-8-4-2-0-1-3-5-7-9-11-13-15 | [14,6,10,12,8,4,2,0,1,3,5,7,9,11,13,15] |
| 24 | S3R49SCN_N24_B09_median_guard | 2.9523077955673216e-05 | run51_stricter_constrained_N24_N40_batch32 | Run51 | PASS_TEACHER_FIELDS_EXTRACTED | 22fe0ffd756973ac | 22-21-19-11-1-3-5-7-9-13-15-17-23-0-2-4-6-8-10-12-14-16-18-20 | [22,21,19,11,1,3,5,7,9,13,15,17,23,0,2,4,6,8,10,12,14,16,18,20] | S3R59N40PR40_N24_B09_n24_diversity | 0.1492166519165039 | run61_custom_N40_focused_batch40 | Run61 | PASS_TEACHER_FIELDS_EXTRACTED | c8e65c9c58288448 | 21-14-7-0-17-10-3-20-13-6-23-16-9-2-19-12-5-22-15-8-1-18-11-4 | S3R24L64_N24_B06_model_disagreement | 580875968.0 | shortlist64_run27 | Run27 | PASS_TEACHER_FIELDS_EXTRACTED | 5e8f606e284ef168 | 11-21-7-17-9-13-23-3-19-5-15-1-0-2-4-6-8-10-12-14-16-18-20-22 | N24_A06_edge_in_alternating | 579942784.0 | probe60_run08 | Run08 | PASS_ODB_EXTRACTED |  |  | S3R44CNS_N24_B16_uncertainty | 0.8718253968253968 | run46_constrained_N24_N40_batch32 | Run46 | PASS_TEACHER_FIELDS_EXTRACTED | e72abd003436348f | 22-17-15-13-9-7-5-3-1-11-19-21-23-0-2-4-6-8-10-12-14-16-18-20 | [22,17,15,13,9,7,5,3,1,11,19,21,23,0,2,4,6,8,10,12,14,16,18,20] | S3R44CNS_N24_B16_uncertainty | 0.8558201058201058 | run46_constrained_N24_N40_batch32 | Run46 | PASS_TEACHER_FIELDS_EXTRACTED | e72abd003436348f | 22-17-15-13-9-7-5-3-1-11-19-21-23-0-2-4-6-8-10-12-14-16-18-20 | S3R44CNS_N24_B16_uncertainty | 0.8505291005291007 | run46_constrained_N24_N40_batch32 | Run46 | PASS_TEACHER_FIELDS_EXTRACTED | e72abd003436348f | 22-17-15-13-9-7-5-3-1-11-19-21-23-0-2-4-6-8-10-12-14-16-18-20 | S3R69SNR_N24_B04_n24_uncertainty | 0.8466931216931217 | run71_smallN_recovery_focused_batch40 | Run71 | PASS_TEACHER_FIELDS_EXTRACTED | 03078ca2a17672da | 22-17-15-13-9-5-7-3-1-11-19-21-23-0-6-4-2-8-10-12-14-16-18-20 | [22,17,15,13,9,5,7,3,1,11,19,21,23,0,6,4,2,8,10,12,14,16,18,20] |
| 40 | S3R64VNR_N40_B15_n40_u2ret_anchor | 4.5779268475598656e-05 | run66_variable_N_recovery_anchor_batch48 | Run66 | PASS_TEACHER_FIELDS_EXTRACTED | d9c22149ede815b6 | 0-2-4-6-9-11-13-15-17-19-34-36-38-32-30-28-26-24-22-20-18-16-14-12-10-8-7-5-3... | [0,2,4,6,9,11,13,15,17,19,34,36,38,32,30,28,26,24,22,20,18,16,14,12,10,8,7,5,... | S3R39N2440B60_N40_B10_surrogate_top | 0.1440253555774688 | run41_native_N24_N40_focused_batch60 | Run41 | PASS_TEACHER_FIELDS_EXTRACTED | ea73009f75c06ec9 | 14-30-20-8-2-24-32-28-26-0-34-18-16-36-12-4-10-22-38-6-19-23-29-31-37-13-7-27... | S3R19B28_N40_B06_uncertainty_calibration | 581119040.0 | batch28_run20 | Run20 | PASS_TEACHER_FIELDS_EXTRACTED |  |  | N40_A07_regular_jump_coprime | 579924032.0 | probe60_run08 | Run08 | PASS_ODB_EXTRACTED |  |  | S3R64VNR_N40_B03_n40_penalty_anchor | 0.9395121951219512 | run66_variable_N_recovery_anchor_batch48 | Run66 | PASS_TEACHER_FIELDS_EXTRACTED | 520a2adf7b065203 | 0-2-4-7-9-11-13-15-17-19-34-36-38-32-30-28-26-24-22-20-18-16-14-12-10-8-6-5-3... | [0,2,4,7,9,11,13,15,17,19,34,36,38,32,30,28,26,24,22,20,18,16,14,12,10,8,6,5,... | S3R64VNR_N40_B03_n40_penalty_anchor | 0.9097560975609756 | run66_variable_N_recovery_anchor_batch48 | Run66 | PASS_TEACHER_FIELDS_EXTRACTED | 520a2adf7b065203 | 0-2-4-7-9-11-13-15-17-19-34-36-38-32-30-28-26-24-22-20-18-16-14-12-10-8-6-5-3... | S3R64VNR_N40_B03_n40_penalty_anchor | 0.8985365853658538 | run66_variable_N_recovery_anchor_batch48 | Run66 | PASS_TEACHER_FIELDS_EXTRACTED | 520a2adf7b065203 | 0-2-4-7-9-11-13-15-17-19-34-36-38-32-30-28-26-24-22-20-18-16-14-12-10-8-6-5-3... | S3R64VNR_N40_B03_n40_penalty_anchor | 0.8760975609756098 | run66_variable_N_recovery_anchor_batch48 | Run66 | PASS_TEACHER_FIELDS_EXTRACTED | 520a2adf7b065203 | 0-2-4-7-9-11-13-15-17-19-34-36-38-32-30-28-26-24-22-20-18-16-14-12-10-8-6-5-3... | [0,2,4,7,9,11,13,15,17,19,34,36,38,32,30,28,26,24,22,20,18,16,14,12,10,8,6,5,... |

## 8. Final Top-k Tables
- Top5: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\stage3_final_native_top5_by_N_and_objective.csv`
- Top10: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\stage3_final_native_top10_by_N_and_objective.csv`

## 9. Run-by-Run Evidence Ledger
`E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\stage3_final_run_by_run_evidence_ledger.md`

## 10. Metric/Reward Record Timeline
`E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\stage3_final_metric_reward_record_timeline.md`

## 11. Claim-Evidence Map
`E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\stage3_final_claim_evidence_map.md`

## 12. Final Claim Boundary
`E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\stage3_final_claim_boundary.md`

## 13. Paper-Safe Conclusions
`E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\stage3_final_paper_safe_conclusions.md`

## 14. N32 Legacy-Semantic Separation
`E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\stage3_final_N32_legacy_semantic_separation_memo.md`

## 15. ARA-Style Evidence Index
`E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\ARA_STAGE3_FINAL_EVIDENCE_INDEX.md`

## 16. Final Freeze Summary
`E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\STAGE3_FINAL_EVIDENCE_FREEZE_SUMMARY.md`

## 17. Output Files
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\FROZEN_stage3_native_combined552_teacher_dataset.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\FROZEN_stage3_native_combined552_RL_ready_dataset.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\FROZEN_stage3_combined552_plus_N32_teacher_dataset.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\FROZEN_stage3_combined552_plus_N32_RL_ready_dataset.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\FROZEN_stage3_native_combined552_per_N_leaderboard.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\FROZEN_stage3_native_combined552_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\FROZEN_run77_final_claim_boundary.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\FROZEN_run77_final_claim_boundary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\FROZEN_stage3_final_maturity_and_evidence_freeze_readiness_audit.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\FROZEN_stage3_final_maturity_and_evidence_freeze_readiness_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\FROZEN_stage3_file_hashes.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\FROZEN_stage3_file_hashes.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\stage3_final_native_best_strategy_table.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\stage3_final_native_best_strategy_table.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\stage3_final_native_best_strategy_table.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\stage3_final_native_top5_by_N_and_objective.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\stage3_final_native_top10_by_N_and_objective.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\stage3_final_native_topk_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\stage3_final_run_by_run_evidence_ledger.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\stage3_final_metric_reward_record_timeline.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\stage3_final_claim_evidence_map.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\stage3_final_claim_boundary.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\stage3_final_claim_boundary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\stage3_final_paper_safe_conclusions.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\stage3_final_N32_legacy_semantic_separation_memo.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\stage3_final_N32_legacy_semantic_separation_memo.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\ARA_STAGE3_FINAL_EVIDENCE_INDEX.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\ARA_STAGE3_FINAL_EVIDENCE_INDEX.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\STAGE3_FINAL_EVIDENCE_FREEZE_SUMMARY.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\STAGE3_FINAL_EVIDENCE_FREEZE_SUMMARY.json`

## 18. Recommended Next Action
Do not generate more Stage 3 candidates by default. Use this evidence-freeze package to prepare the final Stage 3 write-up, paper methods/results sections, figures, and GitHub ARA-style evidence archive.

## Run78 Safety Boundary
Run78 did not run Abaqus, open ODB files, run solver/datacheck/abqjobpilot/enqueue, generate CAE/INP/JNL, train models, or generate candidates.
