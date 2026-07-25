# Stage 3 Run 38 - Combined204 and Combined204 Plus N32 Model Update Candidate Generation

## 1. Purpose
Update offline surrogate, GNN reward, and graph-pointer diagnostics after Run37 combined204 ingestion, compare native-only and N32-augmented regimes, and generate native-N candidate batch options.

## 2. Inputs
- native combined204 RL-ready: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_37_N32_informed_native_batch32_teacher_metrics_ingestion_and_combined204_ranking\combined204_RL_ready_dataset.csv`
- combined204_plus_N32 RL-ready: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_37_N32_informed_native_batch32_teacher_metrics_ingestion_and_combined204_ranking\combined204_plus_N32_RL_ready_dataset.csv`
- N32 dedup training table: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_32a_stage2_n32_legacy_teacher_label_ingestion_for_stage3\n32_legacy_teacher_dataset_dedup_training_332.csv`

## 3. Run37/Run36 Context
Run36 validated N32-informed native-N candidates only. It contained no N32 cases. Run37 built native combined204 and combined204_plus_N32.

## 4. Reward-Best Semantic Audit
- Status: `RUN38_REWARD_BEST_SEMANTIC_AUDIT_COMPLETE`
- Headline: The N24 Run36 reward-best appears as the recomputed combined204 rank-based reward best, while Run37's previous-comparison table did not mark it as beating the old combined172 reward best.
- N24 interpretation: `recomputed_combined204_rank_reward_best_after_rank_normalization_shift`

## 5. Feature Reconstruction
- Reconstructed Run22/Run29/Run33-compatible scan-order descriptors for native combined204 and combined204_plus_N32.

## 6. Surrogate Update
- Verdict: `PASS_RUN38_COMBINED204_AND_PLUS_N32_INPUTS_READY`
- Rows: `536`
- Per-N counts: `{12: 36, 16: 36, 24: 66, 32: 332, 40: 66}`
- Best config: `native_combined204 / ExtraTreesRegressor / F07_F01_no_n`
- Macro Spearman: `0.8680565944557831`
- Macro top5 overlap: `2.0`

## 7. GNN Reward Update
- Status: `RUN38_GNN_REWARD_MODEL_TRAINED`
- Best regime: `{'regime': 'plus_N32_unweighted', 'macro_spearman': 0.8077513974106804, 'macro_top5_overlap': 1.4, 'n40_spearman': 0.8514316172059149}`

## 8. Graph-Pointer Update
- Status: `RUN38_GRAPH_POINTER_POLICY_WEIGHTED_BC_TRAINED`
- Training is offline weighted behavior cloning only; no online RL was run.

## 9. N32 Augmentation Diagnostic
- Native Stage 3 prediction effect: `improved_or_similar`
- N32 rows remain legacy-compatible; PEEQ/Mises semantic warnings are preserved.

## 10. Candidate Generation
- Deduplicated candidate counts: `{12: 800, 16: 800, 24: 3000, 32: 1352, 40: 3000}`

## 11. Option A Native Batch32
- Path: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_38_combined204_and_combined204_plus_N32_model_update_candidate_generation\run38_native_batch32_U2_exploitation_reward_balanced_candidate_orders.csv`
- N12=4, N16=4, N24=12, N40=12.

## 12. Option B N24/N40 Focused Batch32
- Path: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_38_combined204_and_combined204_plus_N32_model_update_candidate_generation\run38_native_N24_N40_focused_batch32_candidate_orders.csv`
- N24=16, N40=16.

## 13. Option C Native Batch40
- Path: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_38_combined204_and_combined204_plus_N32_model_update_candidate_generation\run38_native_batch40_broader_coverage_candidate_orders.csv`
- N12=4, N16=4, N24=16, N40=16.

## 14. Comparison To Previous Batches
- Headline: Run38 options are checked for exact overlap against combined204 teachers, Run36, Run27, and the superseded old Run31 batch; selected options remain candidate orders only.

## 15. Claim Boundary
- Run38 is model update and candidate generation only. No teacher validation, no CAE/INP, no solver activity.

## 16. Output Files
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_38_combined204_and_combined204_plus_N32_model_update_candidate_generation\run38_input_validation_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_38_combined204_and_combined204_plus_N32_model_update_candidate_generation\run38_reward_best_semantic_audit.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_38_combined204_and_combined204_plus_N32_model_update_candidate_generation\run38_reward_best_semantic_audit.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_38_combined204_and_combined204_plus_N32_model_update_candidate_generation\combined204_scan_order_features.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_38_combined204_and_combined204_plus_N32_model_update_candidate_generation\combined204_plus_N32_scan_order_features.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_38_combined204_and_combined204_plus_N32_model_update_candidate_generation\run38_surrogate_validation_results_detailed.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_38_combined204_and_combined204_plus_N32_model_update_candidate_generation\run38_best_surrogate_configurations.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_38_combined204_and_combined204_plus_N32_model_update_candidate_generation\run38_surrogate_validation_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_38_combined204_and_combined204_plus_N32_model_update_candidate_generation\run38_gnn_reward_validation_results.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_38_combined204_and_combined204_plus_N32_model_update_candidate_generation\run38_gnn_reward_validation_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_38_combined204_and_combined204_plus_N32_model_update_candidate_generation\run38_graph_pointer_policy_training_log.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_38_combined204_and_combined204_plus_N32_model_update_candidate_generation\run38_graph_pointer_policy_validation_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_38_combined204_and_combined204_plus_N32_model_update_candidate_generation\run38_candidate_pool_scored.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_38_combined204_and_combined204_plus_N32_model_update_candidate_generation\run38_native_batch32_U2_exploitation_reward_balanced_candidate_orders.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_38_combined204_and_combined204_plus_N32_model_update_candidate_generation\run38_native_N24_N40_focused_batch32_candidate_orders.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_38_combined204_and_combined204_plus_N32_model_update_candidate_generation\run38_native_batch40_broader_coverage_candidate_orders.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_38_combined204_and_combined204_plus_N32_model_update_candidate_generation\run38_batch_options_comparison_to_previous.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_38_combined204_and_combined204_plus_N32_model_update_candidate_generation\run38_batch_options_comparison_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_38_combined204_and_combined204_plus_N32_model_update_candidate_generation\run38_claim_boundary.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_38_combined204_and_combined204_plus_N32_model_update_candidate_generation\run38_claim_boundary.json`

## 17. Recommended Run39
If Option A is selected, create a handoff package for native batch32. Select Option B for maximum N24/N40 pressure. Select Option C for broader native coverage. Do not generate CAE/INP until one option is selected.
