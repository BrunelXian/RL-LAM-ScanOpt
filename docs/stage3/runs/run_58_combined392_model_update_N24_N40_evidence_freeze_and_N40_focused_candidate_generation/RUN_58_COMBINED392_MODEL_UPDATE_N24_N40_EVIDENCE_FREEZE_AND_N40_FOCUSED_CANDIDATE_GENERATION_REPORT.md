# Stage 3 Run 58 - Combined392 Model Update, N24/N40 Evidence Freeze, and N40-Focused Candidate Generation

## 1. Purpose
Run58 updates offline diagnostics using native combined392 and combined392_plus_N32, freezes the N24/N40 active-learning evidence summary, and creates N40-focused calibrated candidate batches. It is model update and candidate generation only.

## 2. Inputs
- Native combined392: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_57_calibrated_N24_N40_batch64_teacher_metrics_ingestion_and_combined392_ranking\combined392_RL_ready_dataset.csv`
- combined392_plus_N32: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_57_calibrated_N24_N40_batch64_teacher_metrics_ingestion_and_combined392_ranking\combined392_plus_N32_RL_ready_dataset.csv`
- Run56 prediction audit: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_57_calibrated_N24_N40_batch64_teacher_metrics_ingestion_and_combined392_ranking\run56_prediction_audit_for_run53_batch64_summary.json`

## 3. Run57/Run56 Context
Run56 created five new records versus combined328, all in N40 U2/reward-family metrics. N24 produced useful top-density but no new combined328 bests. Prediction calibration was moderate, so Run58 uses N40-focused diversification rather than a naive top-only batch.

## 4. N24/N40 Evidence Freeze
N24/N40 active-learning evidence is mature enough to freeze at 160 native teacher rows each; full variable-N RL remains limited by N12/N16 at 36 rows each.

## 5. Target Reward Definition Audit
Run58 target definitions prioritize N40 U2/reward retention, N40 penalty repair, and N24 maintenance while preserving N32 metric-semantic warnings.

## 6. Feature Reconstruction
Run58 wrote Run22/Run29/Run33/Run38-compatible handcrafted order features for native combined392 and combined392_plus_N32.

## 7. Surrogate Update and Calibration
Best surrogate overall: `{'regime': 'plus_N32_unweighted', 'target': 'target_reward_combined392_plus_N32_mapped_u2_primary', 'feature_set': 'F07_F01_no_n', 'model': 'ExtraTreesRegressor', 'macro_spearman': 0.8015235406959332, 'macro_top5_overlap': 1.0, 'macro_top10_overlap': 3.4, 'mean_mae': 0.12336767818908598, 'protocols': 5}`.

Best native surrogate: `{'regime': 'native_combined392', 'target': 'target_reward_combined392_u2_primary', 'feature_set': 'F01_basic_order', 'model': 'ExtraTreesRegressor', 'macro_spearman': 0.8006031299400806, 'macro_top5_overlap': 1.25, 'macro_top10_overlap': 3.75, 'mean_mae': 0.10507856095565274, 'protocols': 4}`.

Best plus-N32 surrogate: `{'regime': 'plus_N32_unweighted', 'target': 'target_reward_combined392_plus_N32_mapped_u2_primary', 'feature_set': 'F07_F01_no_n', 'model': 'ExtraTreesRegressor', 'macro_spearman': 0.8015235406959332, 'macro_top5_overlap': 1.0, 'macro_top10_overlap': 3.4, 'mean_mae': 0.12336767818908598, 'protocols': 5}`.

## 8. GNN and Graph-Pointer Diagnostics
GNN diagnostic is an offline order-graph MLP proxy; it is not online RL or teacher validation. Best GNN diagnostic: `{'regime': 'plus_N32_balanced', 'macro_spearman': 0.479670379106346, 'macro_top5_overlap': 0.2, 'protocols': 5}`.
Offline transition-frequency graph-pointer diagnostic fitted with reward-weighted teacher sequences; no online RL was run. Mean NLL by regime: `{'native_combined392': 1.4367819359957088, 'plus_N32_balanced': 1.5540523369709658, 'plus_N32_unweighted': 1.5540523369709658}`.

## 9. N40-Focused Candidate Pool Generation
Candidate pool counts: `{12: 500, 16: 500, 24: 3000, 40: 7000}`. N24 meets the >=3000 candidate minimum and N40 meets the >=7000 candidate minimum.

## 10. Option A - N40-Focused Calibrated Penalty-Repair Batch32
Path: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_58_combined392_model_update_N24_N40_evidence_freeze_and_N40_focused_candidate_generation\run58_N40_focused_calibrated_penalty_repair_batch32_candidate_orders.csv`. Counts: `{24: 8, 40: 24}`.

## 11. Option B - N40-Focused Calibrated Batch64
Path: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_58_combined392_model_update_N24_N40_evidence_freeze_and_N40_focused_candidate_generation\run58_N40_focused_calibrated_batch64_candidate_orders.csv`. Counts: `{24: 16, 40: 48}`.

## 12. Option C - Variable-N Recovery Anchor Batch48
Path: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_58_combined392_model_update_N24_N40_evidence_freeze_and_N40_focused_candidate_generation\run58_variable_N_recovery_anchor_batch48_candidate_orders.csv`. Counts: `{12: 8, 16: 8, 24: 16, 40: 16}`.

## 13. Comparison to Previous Batches
Run58 batch options were checked for exact order overlap, N40-focused score distribution, penalty-repair score, novelty, and source diversity against combined392, Run56, Run51, Run46, Run41, Run36, Run27, and superseded Run31.

## 14. Claim Boundary
Verdict: `RUN58_MODEL_UPDATE_EVIDENCE_FREEZE_AND_CANDIDATE_GENERATION_ONLY_NO_TEACHER_VALIDATION`.

## 15. Output Files
- Candidate pool: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_58_combined392_model_update_N24_N40_evidence_freeze_and_N40_focused_candidate_generation\run58_candidate_pool_scored.csv`
- Surrogate summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_58_combined392_model_update_N24_N40_evidence_freeze_and_N40_focused_candidate_generation\run58_surrogate_validation_summary.json`
- GNN summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_58_combined392_model_update_N24_N40_evidence_freeze_and_N40_focused_candidate_generation\run58_gnn_reward_validation_summary.json`
- Pointer summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_58_combined392_model_update_N24_N40_evidence_freeze_and_N40_focused_candidate_generation\run58_graph_pointer_policy_validation_summary.json`
- Evidence freeze: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_58_combined392_model_update_N24_N40_evidence_freeze_and_N40_focused_candidate_generation\n24_n40_active_learning_rl_evidence_freeze.md`
- Batch options comparison: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_58_combined392_model_update_N24_N40_evidence_freeze_and_N40_focused_candidate_generation\run58_batch_options_comparison_summary.json`
- Manifest: `E:\Projects\RL-LAM-ScanOpt\artifacts\manifests\stage3_run_58_manifest.json`

## 16. Recommended Run59
Select Option A for a quick N40-focused penalty-repair validation loop unless the user explicitly wants another overnight batch64. Do not generate CAE/INP until the selected Run58 option is handed off.
