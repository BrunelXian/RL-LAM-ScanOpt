# Stage 3 Run 53 - Combined328 Calibrated N24/N40 Batch64 Candidate Generation

## 1. Purpose
Run53 updates offline diagnostics using native combined328 and combined328_plus_N32, then creates the user-selected calibrated N24/N40 overnight batch64. It is model update and candidate generation only.

## 2. Inputs
- Native combined328: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_52_stricter_constrained_N24_N40_batch32_teacher_metrics_ingestion_and_combined328_ranking\combined328_RL_ready_dataset.csv`
- combined328_plus_N32: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_52_stricter_constrained_N24_N40_batch32_teacher_metrics_ingestion_and_combined328_ranking\combined328_plus_N32_RL_ready_dataset.csv`
- Run51 prediction audit: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_52_stricter_constrained_N24_N40_batch32_teacher_metrics_ingestion_and_combined328_ranking\run51_prediction_audit_for_run48_batch32_summary.json`

## 3. Run52/Run51 Context
Run51 produced an N24 U2 best and N40 strict/reward gains versus combined296, but did not create raw PEEQ, SurfaceT, or Mises bests. Run51 prediction calibration was moderate, so Run53 uses calibrated diversification instead of a naive top64.

## 4. User Decision to Select Overnight Batch64
The primary selected batch is `calibrated_N24_N40_batch64` with N24=32 and N40=32. N12, N16, and N32 are not selected for the primary batch.

## 5. Target Reward Definition Audit
Calibrated targets combine N24 U2 retention, N40 strict/reward retention, penalty repair, and two-stage penalty repair after Run51 improved U2/reward without raw PEEQ/SurfaceT/Mises records.

## 6. Feature Reconstruction
Run53 wrote Run22/Run29/Run33/Run38-compatible handcrafted order features for native combined328 and combined328_plus_N32.

## 7. Surrogate Update and Calibration
Best surrogate overall: `{'regime': 'native_combined328', 'target': 'target_reward_combined328_u2_primary', 'feature_set': 'F01_basic_order', 'model': 'ExtraTreesRegressor', 'macro_spearman': 0.8349863849292928, 'macro_top5_overlap': 1.5, 'macro_top10_overlap': 4.0, 'mean_mae': 0.09893450888761607, 'protocols': 4}`.

Best native surrogate: `{'regime': 'native_combined328', 'target': 'target_reward_combined328_u2_primary', 'feature_set': 'F01_basic_order', 'model': 'ExtraTreesRegressor', 'macro_spearman': 0.8349863849292928, 'macro_top5_overlap': 1.5, 'macro_top10_overlap': 4.0, 'mean_mae': 0.09893450888761607, 'protocols': 4}`.

Best plus-N32 surrogate: `{'regime': 'plus_N32_balanced', 'target': 'target_reward_combined328_plus_N32_mapped_u2_primary', 'feature_set': 'F07_F01_no_n', 'model': 'ExtraTreesRegressor', 'macro_spearman': 0.8286129655470909, 'macro_top5_overlap': 1.0, 'macro_top10_overlap': 4.0, 'mean_mae': 0.10742575614613477, 'protocols': 5}`.

## 8. GNN and Graph-Pointer Diagnostics
GNN diagnostic is an offline order-graph MLP proxy; it is not online RL or teacher validation. Best GNN diagnostic: `{'regime': 'plus_N32_balanced', 'macro_spearman': 0.5284103881189569, 'macro_top5_overlap': 0.2, 'protocols': 5}`.
Offline transition-frequency graph-pointer diagnostic fitted with reward-weighted teacher sequences; no online RL was run. Mean NLL by regime: `{'native_combined328': 1.50955209473979, 'plus_N32_balanced': 1.6122684639662306, 'plus_N32_unweighted': 1.6122684639662306}`.

## 9. Calibrated N24/N40 Candidate Pool Generation
Candidate pool counts: `{12: 500, 16: 500, 24: 6000, 40: 6000}`. N24 and N40 each meet the >=6000 candidate minimum.

## 10. Primary Selected Batch64
Path: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_53_combined328_calibrated_N24_N40_batch64_candidate_generation\run53_calibrated_N24_N40_batch64_candidate_orders.csv`. Counts: `{24: 32, 40: 32}`.

## 11. Reference Batch32 and Recovery Batch40
Reference batch32 path: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_53_combined328_calibrated_N24_N40_batch64_candidate_generation\run53_calibrated_N24_N40_batch32_REFERENCE_candidate_orders.csv`. Counts: `{24: 16, 40: 16}`.

Recovery batch40 path: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_53_combined328_calibrated_N24_N40_batch64_candidate_generation\run53_native_recovery_batch40_with_anchors_REFERENCE_candidate_orders.csv`. Counts: `{12: 4, 16: 4, 24: 16, 40: 16}`.

## 12. Comparison to Previous Batches
Primary Run53 batch64 and reference batches were checked for exact order overlap, calibrated score distribution, penalty-repair score, novelty, and source diversity against combined328, Run51, Run46, Run41, Run36, Run27, and superseded Run31.

## 13. Claim Boundary
Verdict: `RUN53_MODEL_UPDATE_AND_PRIMARY_BATCH64_CANDIDATE_GENERATION_ONLY_NO_TEACHER_VALIDATION`.

## 14. Output Files
- Candidate pool: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_53_combined328_calibrated_N24_N40_batch64_candidate_generation\run53_candidate_pool_scored.csv`
- Surrogate summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_53_combined328_calibrated_N24_N40_batch64_candidate_generation\run53_surrogate_validation_summary.json`
- GNN summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_53_combined328_calibrated_N24_N40_batch64_candidate_generation\run53_gnn_reward_validation_summary.json`
- Pointer summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_53_combined328_calibrated_N24_N40_batch64_candidate_generation\run53_graph_pointer_policy_validation_summary.json`
- Batch64 comparison: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_53_combined328_calibrated_N24_N40_batch64_candidate_generation\run53_batch64_comparison_summary.json`
- Manifest: `E:\Projects\RL-LAM-ScanOpt\artifacts\manifests\stage3_run_53_manifest.json`

## 15. Recommended Run54
Create a handoff package for the primary selected batch64: `run53_calibrated_N24_N40_batch64_candidate_orders.csv`. Do not generate CAE/INP until Run54 handoff is approved.
