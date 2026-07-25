# Stage 3 Run 68 - Combined480 Model Update and Small-N Recovery Candidate Generation

## 1. Purpose
Run68 updates offline diagnostics using native combined480 and combined480_plus_N32, updates the full variable-N evidence boundary after Run67, and creates N12/N16 recovery-focused candidate batches. It is model update and candidate generation only.

## 2. Inputs
- Native combined480: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_67_variable_N_recovery_anchor_batch48_teacher_metrics_ingestion_and_combined480_ranking\combined480_RL_ready_dataset.csv`
- combined480_plus_N32: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_67_variable_N_recovery_anchor_batch48_teacher_metrics_ingestion_and_combined480_ranking\combined480_plus_N32_RL_ready_dataset.csv`
- Run66 prediction audit: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_67_variable_N_recovery_anchor_batch48_teacher_metrics_ingestion_and_combined480_ranking\run66_prediction_audit_for_run63_batch48_summary.json`
- Run67 maturity audit: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_67_variable_N_recovery_anchor_batch48_teacher_metrics_ingestion_and_combined480_ranking\full_variable_N_updated_maturity_and_claim_boundary_summary.json`

## 3. Run67/Run66 Context
Run66 created 14 new metric/reward records versus combined432. N16 was the clearest small-N winner, N12 improved reward-family records, N40 produced useful U2/reward anchor records, and N24 acted as an anchor. N12/N16 now have 48 rows each but remain much less dense than N24/N40.

## 4. Target Reward Definition Audit
Run68 target definitions prioritize N12/N16 recovery, keep N24/N40 as low-count anchors, and preserve N32 metric-semantic warnings for plus_N32 diagnostics.

## 5. Feature Reconstruction
Run68 wrote Run22/Run29/Run33/Run38/Run43/Run48/Run53/Run58-compatible handcrafted order features for native combined480 and combined480_plus_N32.

## 6. Surrogate Update
Best surrogate overall: `{'regime': 'plus_N32_unweighted', 'target': 'target_reward_combined480_plus_N32_mapped_u2_primary', 'feature_set': 'F07_F01_no_n', 'model': 'ExtraTreesRegressor', 'macro_spearman': 0.7496305137388312, 'macro_top5_overlap': 0.4, 'macro_top10_overlap': 1.0, 'mean_mae': 0.13625589953529016, 'protocols': 5}`.

Best native surrogate: `{'regime': 'native_combined480', 'target': 'target_reward_combined480_N12_recovery', 'feature_set': 'F07_F01_no_n', 'model': 'ExtraTreesRegressor', 'macro_spearman': 0.7385046390495479, 'macro_top5_overlap': 0.75, 'macro_top10_overlap': 1.5, 'mean_mae': 0.10923270999496071, 'protocols': 4}`.

Best plus-N32 surrogate: `{'regime': 'plus_N32_unweighted', 'target': 'target_reward_combined480_plus_N32_mapped_u2_primary', 'feature_set': 'F07_F01_no_n', 'model': 'ExtraTreesRegressor', 'macro_spearman': 0.7496305137388312, 'macro_top5_overlap': 0.4, 'macro_top10_overlap': 1.0, 'mean_mae': 0.13625589953529016, 'protocols': 5}`.

Best N12 recovery: `{'regime': 'native_combined480', 'target': 'target_reward_combined480_N12_recovery', 'feature_set': 'F07_F01_no_n', 'model': 'ExtraTreesRegressor', 'macro_spearman': 0.7385046390495479, 'macro_top5_overlap': 0.75, 'macro_top10_overlap': 1.5, 'mean_mae': 0.10923270999496071, 'protocols': 4}`.

Best N16 recovery: `{'regime': 'native_combined480', 'target': 'target_reward_combined480_N16_recovery', 'feature_set': 'F07_F01_no_n', 'model': 'ExtraTreesRegressor', 'macro_spearman': 0.7385046390495479, 'macro_top5_overlap': 0.75, 'macro_top10_overlap': 1.5, 'mean_mae': 0.10923270999496071, 'protocols': 4}`.

Best small-N recovery: `{'regime': 'native_combined480', 'target': 'target_reward_combined480_smallN_recovery', 'feature_set': 'F07_F01_no_n', 'model': 'ExtraTreesRegressor', 'macro_spearman': 0.7182280611932164, 'macro_top5_overlap': 0.75, 'macro_top10_overlap': 3.0, 'mean_mae': 0.11086909423293581, 'protocols': 4}`.

Best variable-N balanced: `{'regime': 'native_combined480', 'target': 'target_reward_combined480_variable_N_balanced', 'feature_set': 'F07_F01_no_n', 'model': 'ExtraTreesRegressor', 'macro_spearman': 0.6945115029211236, 'macro_top5_overlap': 0.25, 'macro_top10_overlap': 2.0, 'mean_mae': 0.12858660670740588, 'protocols': 4}`.

## 7. GNN and Graph-Pointer Diagnostics
GNN diagnostic is an offline order-graph MLP proxy; it is not online RL or teacher validation. Best GNN diagnostic: `{'regime': 'plus_N32_balanced', 'macro_spearman': 0.3398693726736609, 'macro_top5_overlap': 0.2, 'protocols': 5}`.
Offline transition-frequency graph-pointer diagnostic fitted with reward-weighted teacher sequences; no online RL was run. Mean NLL by regime: `{'native_combined480': 1.3685939187030691, 'plus_N32_balanced': 1.499501923136854, 'plus_N32_unweighted': 1.499501923136854}`.

## 8. Full Variable-N Evidence Update After Run67
Run67/Run66 strengthened full variable-N evidence to N12=48 and N16=48 while N24=184 and N40=200 remain mature anchors; the next validation should continue small-N recovery rather than broad N24/N40 exploitation.

## 9. N12/N16 Recovery Candidate Generation
Candidate pool counts: `{12: 5000, 16: 5000, 24: 1000, 40: 1000}`. N12/N16 meet the >=5000 candidate minimums, with N24/N40 frozen-anchor references included for comparison.

## 10. Option A - Small-N Recovery-Focused Batch40
Path: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_68_combined480_model_update_smallN_recovery_candidate_generation\run68_smallN_recovery_focused_batch40_candidate_orders.csv`. Counts: `{12: 16, 16: 16, 24: 4, 40: 4}`.

## 11. Option B - Small-N Recovery Batch32
Path: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_68_combined480_model_update_smallN_recovery_candidate_generation\run68_smallN_recovery_batch32_candidate_orders.csv`. Counts: `{12: 14, 16: 14, 24: 2, 40: 2}`.

## 12. Option C - Final Diagnostic Batch24
Path: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_68_combined480_model_update_smallN_recovery_candidate_generation\run68_final_diagnostic_batch24_candidate_orders.csv`. Counts: `{12: 8, 16: 8, 24: 4, 40: 4}`.

## 13. Comparison to Previous Batches
Run68 batch options were checked for exact order overlap, small-N recovery score, optional N40 follow-up score, novelty, and source diversity against combined480, Run66, Run56, Run51, Run46, Run41, Run36, Run27, and superseded Run31.

## 14. Claim Boundary
Verdict: `RUN68_MODEL_UPDATE_AND_SMALLN_RECOVERY_CANDIDATE_GENERATION_ONLY_NO_TEACHER_VALIDATION`.

## 15. Output Files
- Candidate pool: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_68_combined480_model_update_smallN_recovery_candidate_generation\run68_candidate_pool_scored.csv`
- Surrogate summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_68_combined480_model_update_smallN_recovery_candidate_generation\run68_surrogate_validation_summary.json`
- GNN summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_68_combined480_model_update_smallN_recovery_candidate_generation\run68_gnn_reward_validation_summary.json`
- Pointer summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_68_combined480_model_update_smallN_recovery_candidate_generation\run68_graph_pointer_policy_validation_summary.json`
- Evidence update: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_68_combined480_model_update_smallN_recovery_candidate_generation\full_variable_N_evidence_update_after_run67.md`
- Batch options comparison: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_68_combined480_model_update_smallN_recovery_candidate_generation\run68_batch_options_comparison_summary.json`
- Manifest: `E:\Projects\RL-LAM-ScanOpt\artifacts\manifests\stage3_run_68_manifest.json`

## 16. Recommended Run69
Create a handoff package for Option A (`run68_smallN_recovery_focused_batch40_candidate_orders.csv`) unless the user explicitly selects Option B or Option C. Do not generate CAE/INP until a Run68 option is selected and handed off.
