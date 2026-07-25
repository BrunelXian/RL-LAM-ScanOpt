# Stage 3 Run 63 - Combined432 Model Update, N24/N40 Evidence Freeze, and N12/N16 Recovery Candidate Generation

## 1. Purpose
Run63 updates offline diagnostics using native combined432 and combined432_plus_N32, performs the final N24/N40 active-learning evidence-freeze audit, and creates N12/N16 recovery-anchor candidate batches. It is model update and candidate generation only.

## 2. Inputs
- Native combined432: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_62_custom_N40_focused_batch40_teacher_metrics_ingestion_and_combined432_ranking\combined432_RL_ready_dataset.csv`
- combined432_plus_N32: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_62_custom_N40_focused_batch40_teacher_metrics_ingestion_and_combined432_ranking\combined432_plus_N32_RL_ready_dataset.csv`
- Run61 prediction audit: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_62_custom_N40_focused_batch40_teacher_metrics_ingestion_and_combined432_ranking\run61_prediction_audit_for_run58_custom_batch40_summary.json`
- Run62 maturity audit: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_62_custom_N40_focused_batch40_teacher_metrics_ingestion_and_combined432_ranking\n24_n40_updated_maturity_and_claim_boundary_summary.json`

## 3. Run62/Run61 Context
Run61 added N24 PEEQ and N40 U2/reward-family records versus combined392, but prediction calibration remained modest. N24/N40 now have dense native teacher evidence, while N12/N16 remain at 36 rows each.

## 4. Target Reward Definition Audit
Run63 target definitions prioritize N12/N16 recovery anchors, freeze N24/N40 evidence, and retain N40 follow-up only as a secondary option while preserving N32 metric-semantic warnings.

## 5. Feature Reconstruction
Run63 wrote Run22/Run29/Run33/Run38/Run43/Run48/Run53/Run58-compatible handcrafted order features for native combined432 and combined432_plus_N32.

## 6. Surrogate Update
Best surrogate overall: `{'regime': 'native_combined432', 'target': 'target_reward_combined432_constrained_u2_reward_balanced', 'feature_set': 'F07_F01_no_n', 'model': 'ExtraTreesRegressor', 'macro_spearman': 0.8075271720802665, 'macro_top5_overlap': 1.5, 'macro_top10_overlap': 5.25, 'mean_mae': 0.10117582428894487, 'protocols': 4}`.

Best native surrogate: `{'regime': 'native_combined432', 'target': 'target_reward_combined432_constrained_u2_reward_balanced', 'feature_set': 'F07_F01_no_n', 'model': 'ExtraTreesRegressor', 'macro_spearman': 0.8075271720802665, 'macro_top5_overlap': 1.5, 'macro_top10_overlap': 5.25, 'mean_mae': 0.10117582428894487, 'protocols': 4}`.

Best plus-N32 surrogate: `{'regime': 'plus_N32_balanced', 'target': 'target_reward_combined432_plus_N32_mapped_u2_primary', 'feature_set': 'F07_F01_no_n', 'model': 'GradientBoostingRegressor', 'macro_spearman': 0.7689994531218576, 'macro_top5_overlap': 0.8, 'macro_top10_overlap': 3.2, 'mean_mae': 0.1306527718547829, 'protocols': 5}`.

Best N12 recovery: `{'regime': 'native_combined432', 'target': 'target_reward_combined432_N12_recovery', 'feature_set': 'F01_basic_order', 'model': 'ExtraTreesRegressor', 'macro_spearman': 0.7594623814283538, 'macro_top5_overlap': 0.75, 'macro_top10_overlap': 3.75, 'mean_mae': 0.0992512448413739, 'protocols': 4}`.

Best N16 recovery: `{'regime': 'native_combined432', 'target': 'target_reward_combined432_N16_recovery', 'feature_set': 'F07_F01_no_n', 'model': 'ExtraTreesRegressor', 'macro_spearman': 0.7849932484065073, 'macro_top5_overlap': 1.0, 'macro_top10_overlap': 5.0, 'mean_mae': 0.10126613852954561, 'protocols': 4}`.

Best variable-N recovery: `{'regime': 'native_combined432', 'target': 'target_reward_combined432_variable_N_recovery', 'feature_set': 'F07_F01_no_n', 'model': 'ExtraTreesRegressor', 'macro_spearman': 0.7575300751456656, 'macro_top5_overlap': 1.5, 'macro_top10_overlap': 4.25, 'mean_mae': 0.09955955611952681, 'protocols': 4}`.

## 7. GNN and Graph-Pointer Diagnostics
GNN diagnostic is an offline order-graph MLP proxy; it is not online RL or teacher validation. Best GNN diagnostic: `{'regime': 'plus_N32_balanced', 'macro_spearman': 0.6044577762016339, 'macro_top5_overlap': 0.2, 'protocols': 5}`.
Offline transition-frequency graph-pointer diagnostic fitted with reward-weighted teacher sequences; no online RL was run. Mean NLL by regime: `{'native_combined432': 1.4081103989265285, 'plus_N32_balanced': 1.5311151073156215, 'plus_N32_unweighted': 1.5311151073156215}`.

## 8. Final N24/N40 Evidence-Freeze Audit
N24/N40 active-learning evidence is mature enough to freeze at N24=176 and N40=184 native teacher rows; full variable-N RL remains limited by N12/N16 at 36 rows each.

## 9. N12/N16 Recovery Candidate Generation
Candidate pool counts: `{12: 3000, 16: 3000, 24: 1500, 40: 2500}`. N12/N16 meet the >=3000 candidate minimums, with N24 frozen references and optional N40 follow-up candidates included for comparison.

## 10. Option A - Variable-N Recovery Anchor Batch48
Path: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_63_combined432_model_update_N24_N40_evidence_freeze_and_N12_N16_recovery_candidate_generation\run63_variable_N_recovery_anchor_batch48_candidate_orders.csv`. Counts: `{12: 12, 16: 12, 24: 8, 40: 16}`.

## 11. Option B - Small-N Recovery Batch32
Path: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_63_combined432_model_update_N24_N40_evidence_freeze_and_N12_N16_recovery_candidate_generation\run63_smallN_recovery_batch32_candidate_orders.csv`. Counts: `{12: 12, 16: 12, 24: 4, 40: 4}`.

## 12. Option C - Optional N40 Follow-Up Batch32
Path: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_63_combined432_model_update_N24_N40_evidence_freeze_and_N12_N16_recovery_candidate_generation\run63_optional_N40_followup_batch32_candidate_orders.csv`. Counts: `{24: 8, 40: 24}`.

## 13. Comparison to Previous Batches
Run63 batch options were checked for exact order overlap, small-N recovery score, optional N40 follow-up score, novelty, and source diversity against combined432, Run61, Run56, Run51, Run46, Run41, Run36, Run27, and superseded Run31.

## 14. Claim Boundary
Verdict: `RUN63_MODEL_UPDATE_EVIDENCE_FREEZE_AND_CANDIDATE_GENERATION_ONLY_NO_TEACHER_VALIDATION`.

## 15. Output Files
- Candidate pool: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_63_combined432_model_update_N24_N40_evidence_freeze_and_N12_N16_recovery_candidate_generation\run63_candidate_pool_scored.csv`
- Surrogate summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_63_combined432_model_update_N24_N40_evidence_freeze_and_N12_N16_recovery_candidate_generation\run63_surrogate_validation_summary.json`
- GNN summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_63_combined432_model_update_N24_N40_evidence_freeze_and_N12_N16_recovery_candidate_generation\run63_gnn_reward_validation_summary.json`
- Pointer summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_63_combined432_model_update_N24_N40_evidence_freeze_and_N12_N16_recovery_candidate_generation\run63_graph_pointer_policy_validation_summary.json`
- Evidence freeze: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_63_combined432_model_update_N24_N40_evidence_freeze_and_N12_N16_recovery_candidate_generation\n24_n40_final_active_learning_rl_evidence_freeze.md`
- Batch options comparison: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_63_combined432_model_update_N24_N40_evidence_freeze_and_N12_N16_recovery_candidate_generation\run63_batch_options_comparison_summary.json`
- Manifest: `E:\Projects\RL-LAM-ScanOpt\artifacts\manifests\stage3_run_63_manifest.json`

## 16. Recommended Run64
Create a handoff package for Option A (`run63_variable_N_recovery_anchor_batch48_candidate_orders.csv`) unless the user explicitly selects Option B or Option C. Do not generate CAE/INP until a Run63 option is selected and handed off.
