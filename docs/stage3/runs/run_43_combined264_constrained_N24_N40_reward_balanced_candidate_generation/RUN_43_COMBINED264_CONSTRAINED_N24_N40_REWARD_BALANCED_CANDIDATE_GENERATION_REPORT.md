# Stage 3 Run 43 - Combined264 Constrained N24/N40 Reward-Balanced Candidate Generation

## 1. Purpose
Run43 updates offline diagnostics using native combined264 and combined264_plus_N32, then creates constrained N24/N40 reward-balanced candidate batches. It is model update and candidate generation only.

## 2. Inputs
- Native combined264: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_42_native_N24_N40_focused_batch60_teacher_metrics_ingestion_and_combined264_ranking\combined264_RL_ready_dataset.csv`
- combined264_plus_N32: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_42_native_N24_N40_focused_batch60_teacher_metrics_ingestion_and_combined264_ranking\combined264_plus_N32_RL_ready_dataset.csv`
- Run41 prediction audit: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_42_native_N24_N40_focused_batch60_teacher_metrics_ingestion_and_combined264_ranking\run41_prediction_audit_for_run38_batch60_summary.json`

## 3. Run42/Run41 Context
Run41 produced no new U2 bests versus combined204, but created a new N40 PEEQ best and strong top5/top10 density. Run38 prediction calibration on Run41 was strong, so Run43 shifts from pure U2-neighborhood exploitation toward constrained U2 plus reward-balanced selection.

## 4. Target Reward Definition Audit
Constrained targets were created to shift away from pure U2-near exploitation toward U2-primary reward balance with explicit penalty guards.

## 5. Feature Reconstruction
Run43 wrote Run22/Run29/Run33/Run38-compatible handcrafted order features for native combined264 and combined264_plus_N32.

## 6. Surrogate Update
Best surrogate overall: `{'regime': 'native_combined264', 'target': 'target_reward_combined264_u2_primary', 'feature_set': 'F07_F01_no_n', 'model': 'ExtraTreesRegressor', 'macro_spearman': 0.879199270871463, 'macro_top5_overlap': 1.5, 'macro_top10_overlap': 6.5, 'mean_mae': 0.08030496050102474, 'protocols': 4}`.

Best native surrogate: `{'regime': 'native_combined264', 'target': 'target_reward_combined264_u2_primary', 'feature_set': 'F07_F01_no_n', 'model': 'ExtraTreesRegressor', 'macro_spearman': 0.879199270871463, 'macro_top5_overlap': 1.5, 'macro_top10_overlap': 6.5, 'mean_mae': 0.08030496050102474, 'protocols': 4}`.

Best plus-N32 surrogate: `{'regime': 'plus_N32_unweighted', 'target': 'target_reward_combined264_plus_N32_mapped_u2_primary', 'feature_set': 'F07_F01_no_n', 'model': 'ExtraTreesRegressor', 'macro_spearman': 0.8580032000575384, 'macro_top5_overlap': 1.2, 'macro_top10_overlap': 4.2, 'mean_mae': 0.09352907219860668, 'protocols': 5}`.

## 7. GNN Reward Update
GNN diagnostic is an offline order-graph MLP proxy; it is not online RL or teacher validation. Best GNN diagnostic: `{'regime': 'native_combined264', 'macro_spearman': 0.6618648749910644, 'macro_top5_overlap': 1.0, 'protocols': 4}`.

## 8. Graph-Pointer Update
Offline transition-frequency graph-pointer diagnostic fitted with reward-weighted teacher sequences; no online RL was run. Mean NLL by regime: `{'native_combined264': 1.625317299882875, 'plus_N32_balanced': 1.7048806280806985, 'plus_N32_unweighted': 1.7048806280806985}`.

## 9. Constrained N24/N40 Candidate Generation
Candidate pool counts: `{12: 500, 16: 500, 24: 4000, 40: 4000}`. N24 and N40 each meet the >=4000 candidate minimum.

## 10. Option A Constrained N24/N40 Batch32
Path: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_43_combined264_constrained_N24_N40_reward_balanced_candidate_generation\run43_constrained_N24_N40_batch32_candidate_orders.csv`. Counts: `{24: 16, 40: 16}`.

## 11. Option B Constrained N24/N40 Batch60
Path: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_43_combined264_constrained_N24_N40_reward_balanced_candidate_generation\run43_constrained_N24_N40_batch60_candidate_orders.csv`. Counts: `{24: 30, 40: 30}`.

## 12. Option C Native Recovery Batch40 With Anchors
Path: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_43_combined264_constrained_N24_N40_reward_balanced_candidate_generation\run43_native_recovery_batch40_with_anchors_candidate_orders.csv`. Counts: `{12: 4, 16: 4, 24: 16, 40: 16}`.

## 13. Comparison to Previous Batches
All Run43 options were checked for exact order overlap and constrained-score/novelty composition against combined264, Run41, Run36, Run27, and superseded Run31.

## 14. Claim Boundary
Verdict: `RUN43_MODEL_UPDATE_AND_CANDIDATE_GENERATION_ONLY_NO_TEACHER_VALIDATION`.

## 15. Output Files
- Candidate pool: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_43_combined264_constrained_N24_N40_reward_balanced_candidate_generation\run43_candidate_pool_scored.csv`
- Surrogate summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_43_combined264_constrained_N24_N40_reward_balanced_candidate_generation\run43_surrogate_validation_summary.json`
- GNN summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_43_combined264_constrained_N24_N40_reward_balanced_candidate_generation\run43_gnn_reward_validation_summary.json`
- Pointer summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_43_combined264_constrained_N24_N40_reward_balanced_candidate_generation\run43_graph_pointer_policy_validation_summary.json`
- Batch comparison: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_43_combined264_constrained_N24_N40_reward_balanced_candidate_generation\run43_batch_options_comparison_summary.json`
- Manifest: `E:\Projects\RL-LAM-ScanOpt\artifacts\manifests\stage3_run_43_manifest.json`

## 16. Recommended Run44
If the user wants a quick validation loop, select Option A. If the user wants another overnight large batch, select Option B. If the user wants to restore native-N balance with smaller-N anchors, select Option C. Do not generate CAE/INP until one option is explicitly selected.
