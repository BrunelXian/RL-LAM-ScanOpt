# Stage 3 Run 73 - Combined520 Model Update and Final Small-N Diagnostic Candidate Generation

## 1. Purpose
Run73 updates offline diagnostics using native combined520 and combined520_plus_N32, updates the Stage 3 evidence-freeze readiness boundary after Run72, and creates final N12/N16 diagnostic candidate batches. It is model update and candidate generation only.

## 2. Inputs
- Native combined520: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_72_smallN_recovery_focused_batch40_teacher_metrics_ingestion_and_combined520_ranking\combined520_RL_ready_dataset.csv`
- combined520_plus_N32: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_72_smallN_recovery_focused_batch40_teacher_metrics_ingestion_and_combined520_ranking\combined520_plus_N32_RL_ready_dataset.csv`
- Run71 prediction audit: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_72_smallN_recovery_focused_batch40_teacher_metrics_ingestion_and_combined520_ranking\run71_prediction_audit_for_run68_batch40_summary.json`
- Run72 maturity audit: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_72_smallN_recovery_focused_batch40_teacher_metrics_ingestion_and_combined520_ranking\full_variable_N_updated_maturity_and_claim_boundary_summary.json`

## 3. Run72/Run71 Context
Run71 created 13 new metric/reward records versus combined480. N12/N16 now have 64 native teacher rows each, while N24/N40 remain mature anchor groups with 188 and 204 native teacher rows respectively.

## 4. Target Reward Definition Audit
Run73 target definitions prioritize a final N12/N16 diagnostic, keep N24/N40 as low-count anchors, and preserve N32 metric-semantic warnings for plus_N32 diagnostics.

## 5. Feature Reconstruction
Run73 wrote Run22/Run29/Run33/Run38/Run43/Run48/Run53/Run58-compatible handcrafted order features for native combined520 and combined520_plus_N32.

## 6. Surrogate Update
Best surrogate overall: `{'regime': 'plus_N32_balanced', 'target': 'target_reward_combined520_plus_N32_N12_final_diagnostic', 'feature_set': 'F07_F01_no_n', 'model': 'ExtraTreesRegressor', 'macro_spearman': 0.749023722101642, 'macro_top5_overlap': 1.2, 'macro_top10_overlap': 2.2, 'mean_mae': 0.12586970558540278, 'protocols': 5}`.

Best native surrogate: `{'regime': 'native_combined520', 'target': 'target_reward_combined520_variable_N_bounded', 'feature_set': 'F01_basic_order', 'model': 'ExtraTreesRegressor', 'macro_spearman': 0.7370454245203283, 'macro_top5_overlap': 1.75, 'macro_top10_overlap': 3.0, 'mean_mae': 0.08902802737817876, 'protocols': 4}`.

Best plus-N32 surrogate: `{'regime': 'plus_N32_balanced', 'target': 'target_reward_combined520_plus_N32_N12_final_diagnostic', 'feature_set': 'F07_F01_no_n', 'model': 'ExtraTreesRegressor', 'macro_spearman': 0.749023722101642, 'macro_top5_overlap': 1.2, 'macro_top10_overlap': 2.2, 'mean_mae': 0.12586970558540278, 'protocols': 5}`.

Best N12 final diagnostic model: `{'regime': 'native_combined520', 'target': 'target_reward_combined520_N12_final_diagnostic', 'feature_set': 'F07_F01_no_n', 'model': 'Ridge', 'macro_spearman': 0.7134747049753849, 'macro_top5_overlap': 1.25, 'macro_top10_overlap': 2.25, 'mean_mae': 0.13085149925211448, 'protocols': 4}`.

Best N16 final diagnostic model: `{'regime': 'native_combined520', 'target': 'target_reward_combined520_N16_final_diagnostic', 'feature_set': 'F07_F01_no_n', 'model': 'Ridge', 'macro_spearman': 0.7134747049753849, 'macro_top5_overlap': 1.25, 'macro_top10_overlap': 2.25, 'mean_mae': 0.13085149925211448, 'protocols': 4}`.

Best small-N final diagnostic model: `{'regime': 'native_combined520', 'target': 'target_reward_combined520_smallN_final_diagnostic', 'feature_set': 'F07_F01_no_n', 'model': 'Ridge', 'macro_spearman': 0.6858602570563503, 'macro_top5_overlap': 1.25, 'macro_top10_overlap': 2.75, 'mean_mae': 0.13104370081385963, 'protocols': 4}`.

Best variable-N bounded model: `{'regime': 'native_combined520', 'target': 'target_reward_combined520_variable_N_bounded', 'feature_set': 'F01_basic_order', 'model': 'ExtraTreesRegressor', 'macro_spearman': 0.7370454245203283, 'macro_top5_overlap': 1.75, 'macro_top10_overlap': 3.0, 'mean_mae': 0.08902802737817876, 'protocols': 4}`.

## 7. GNN and Graph-Pointer Diagnostics
GNN diagnostic is an offline order-graph MLP proxy; it is not online RL or teacher validation. Best GNN diagnostic: `{'regime': 'plus_N32_balanced', 'macro_spearman': 0.3690334897160547, 'macro_top5_overlap': 0.0, 'protocols': 5}`.
Offline transition-frequency graph-pointer diagnostic fitted with reward-weighted teacher sequences; no online RL was run. Mean NLL by regime: `{'native_combined520': 1.3308394424241872, 'plus_N32_balanced': 1.4692983421137487, 'plus_N32_unweighted': 1.4692983421137487}`.

## 8. Stage 3 Evidence-Freeze Readiness After Run72
Run72/Run71 strengthened full variable-N evidence to N12=64 and N16=64 while N24=188 and N40=204 remain mature anchors; Run73 recommends a final small-N diagnostic loop before declaring an evidence freeze, with stop-and-freeze kept as an explicit option.

## 9. Final Small-N Diagnostic Candidate Generation
Candidate pool counts: `{12: 4000, 16: 4000, 24: 500, 40: 500}`. N12/N16 meet the >=4000 candidate minimums, with N24/N40 frozen-anchor references included for comparison.

## 10. Option A - Final Small-N Diagnostic Batch32
Path: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_73_combined520_model_update_final_smallN_diagnostic_candidate_generation\run73_final_smallN_diagnostic_batch32_candidate_orders.csv`. Counts: `{12: 14, 16: 14, 24: 2, 40: 2}`.

## 11. Option B - Final Small-N Diagnostic Batch24
Path: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_73_combined520_model_update_final_smallN_diagnostic_candidate_generation\run73_final_smallN_diagnostic_batch24_candidate_orders.csv`. Counts: `{12: 10, 16: 10, 24: 2, 40: 2}`.

## 12. Option C - Stop and Freeze Evidence Package Only
Path: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_73_combined520_model_update_final_smallN_diagnostic_candidate_generation\run73_optionC_stop_and_freeze_evidence_package_only.json`. Counts: `{}`.

## 13. Comparison to Previous Batches
Run73 batch options were checked for exact order overlap, final small-N diagnostic score, optional N40 follow-up score, novelty, and source diversity against combined520, Run71, Run66, Run61, Run56, Run51, Run46, Run41, Run36, Run27, and superseded Run31.

## 14. Claim Boundary
Verdict: `RUN73_MODEL_UPDATE_AND_FINAL_SMALLN_DIAGNOSTIC_CANDIDATE_GENERATION_ONLY_NO_TEACHER_VALIDATION`.

## 15. Output Files
- Candidate pool: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_73_combined520_model_update_final_smallN_diagnostic_candidate_generation\run73_candidate_pool_scored.csv`
- Surrogate summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_73_combined520_model_update_final_smallN_diagnostic_candidate_generation\run73_surrogate_validation_summary.json`
- GNN summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_73_combined520_model_update_final_smallN_diagnostic_candidate_generation\run73_gnn_reward_validation_summary.json`
- Pointer summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_73_combined520_model_update_final_smallN_diagnostic_candidate_generation\run73_graph_pointer_policy_validation_summary.json`
- Evidence update: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_73_combined520_model_update_final_smallN_diagnostic_candidate_generation\stage3_evidence_freeze_readiness_after_run72.md`
- Batch options comparison: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_73_combined520_model_update_final_smallN_diagnostic_candidate_generation\run73_batch_options_comparison_summary.json`
- Manifest: `E:\Projects\RL-LAM-ScanOpt\artifacts\manifests\stage3_run_73_manifest.json`

## 16. Recommended Run74
Create a handoff package for Option A (`run73_final_smallN_diagnostic_batch32_candidate_orders.csv`) if the user wants one final validation loop. Select Option B only for lower compute, or Option C if the user wants to stop validation and freeze the Stage 3 evidence package now. Do not generate CAE/INP until a Run73 option is selected and handed off.
