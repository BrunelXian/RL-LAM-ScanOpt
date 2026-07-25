# Stage 3 Run 48 - Combined296 Stricter Constrained N24/N40 Candidate Generation

## 1. Purpose
Run48 updates offline diagnostics using native combined296 and combined296_plus_N32, then creates stricter constrained N24/N40 candidate batches. It is model update and candidate generation only.

## 2. Inputs
- Native combined296: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_47_constrained_N24_N40_batch32_teacher_metrics_ingestion_and_combined296_ranking\combined296_RL_ready_dataset.csv`
- combined296_plus_N32: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_47_constrained_N24_N40_batch32_teacher_metrics_ingestion_and_combined296_ranking\combined296_plus_N32_RL_ready_dataset.csv`
- Run46 prediction audit: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_47_constrained_N24_N40_batch32_teacher_metrics_ingestion_and_combined296_ranking\run46_prediction_audit_for_run43_batch32_summary.json`

## 3. Run47/Run46 Context
Run46 produced N24 U2/reward and N40 reward gains versus combined264, but did not create raw PEEQ, SurfaceT, or Mises bests. Run43 prediction calibration on Run46 was weak, so Run48 retrains on combined296 and tightens penalty guards before scaling.

## 4. Target Reward Definition Audit
Stricter constrained targets were created to shift away from pure U2-near exploitation toward U2-primary reward balance with explicit PEEQ, SurfaceT, and Mises guards.

## 5. Feature Reconstruction
Run48 wrote Run22/Run29/Run33/Run38-compatible handcrafted order features for native combined296 and combined296_plus_N32.

## 6. Surrogate Update
Best surrogate overall: `{'regime': 'native_combined296', 'target': 'target_reward_combined296_u2_primary', 'feature_set': 'F07_F01_no_n', 'model': 'ExtraTreesRegressor', 'macro_spearman': 0.8489555530549906, 'macro_top5_overlap': 1.5, 'macro_top10_overlap': 4.5, 'mean_mae': 0.09100724613108938, 'protocols': 4}`.

Best native surrogate: `{'regime': 'native_combined296', 'target': 'target_reward_combined296_u2_primary', 'feature_set': 'F07_F01_no_n', 'model': 'ExtraTreesRegressor', 'macro_spearman': 0.8489555530549906, 'macro_top5_overlap': 1.5, 'macro_top10_overlap': 4.5, 'mean_mae': 0.09100724613108938, 'protocols': 4}`.

Best plus-N32 surrogate: `{'regime': 'plus_N32_balanced', 'target': 'target_reward_combined296_plus_N32_mapped_u2_primary', 'feature_set': 'F01_basic_order', 'model': 'ExtraTreesRegressor', 'macro_spearman': 0.841132715590802, 'macro_top5_overlap': 1.2, 'macro_top10_overlap': 3.2, 'mean_mae': 0.1014057064147853, 'protocols': 5}`.

## 7. GNN Reward Update
GNN diagnostic is an offline order-graph MLP proxy; it is not online RL or teacher validation. Best GNN diagnostic: `{'regime': 'native_combined296', 'macro_spearman': 0.5237339625369488, 'macro_top5_overlap': 0.25, 'protocols': 4}`.

## 8. Graph-Pointer Update
Offline transition-frequency graph-pointer diagnostic fitted with reward-weighted teacher sequences; no online RL was run. Mean NLL by regime: `{'native_combined296': 1.5611628303334495, 'plus_N32_balanced': 1.6535570524411582, 'plus_N32_unweighted': 1.6535570524411582}`.

## 9. Stricter Constrained N24/N40 Candidate Generation
Candidate pool counts: `{12: 500, 16: 500, 24: 5000, 40: 5000}`. N24 and N40 each meet the >=5000 candidate minimum.

## 10. Option A Stricter Constrained N24/N40 Batch32
Path: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_48_combined296_stricter_constrained_N24_N40_candidate_generation\run48_stricter_constrained_N24_N40_batch32_candidate_orders.csv`. Counts: `{24: 16, 40: 16}`.

## 11. Option B Stricter Constrained N24/N40 Batch60
Path: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_48_combined296_stricter_constrained_N24_N40_candidate_generation\run48_stricter_constrained_N24_N40_batch60_candidate_orders.csv`. Counts: `{24: 30, 40: 30}`.

## 12. Option C Native Recovery Batch40 With Anchors
Path: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_48_combined296_stricter_constrained_N24_N40_candidate_generation\run48_native_recovery_batch40_with_anchors_candidate_orders.csv`. Counts: `{12: 4, 16: 4, 24: 16, 40: 16}`.

## 13. Comparison to Previous Batches
All Run48 options were checked for exact order overlap and strict-guard/novelty composition against combined296, Run46, Run41, Run36, Run27, and superseded Run31.

## 14. Claim Boundary
Verdict: `RUN48_MODEL_UPDATE_AND_CANDIDATE_GENERATION_ONLY_NO_TEACHER_VALIDATION`.

## 15. Output Files
- Candidate pool: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_48_combined296_stricter_constrained_N24_N40_candidate_generation\run48_candidate_pool_scored.csv`
- Surrogate summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_48_combined296_stricter_constrained_N24_N40_candidate_generation\run48_surrogate_validation_summary.json`
- GNN summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_48_combined296_stricter_constrained_N24_N40_candidate_generation\run48_gnn_reward_validation_summary.json`
- Pointer summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_48_combined296_stricter_constrained_N24_N40_candidate_generation\run48_graph_pointer_policy_validation_summary.json`
- Batch comparison: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_48_combined296_stricter_constrained_N24_N40_candidate_generation\run48_batch_options_comparison_summary.json`
- Manifest: `E:\Projects\RL-LAM-ScanOpt\artifacts\manifests\stage3_run_48_manifest.json`

## 16. Recommended Run49
If diagnostics are stable, select Option A or B depending on compute budget. Because Run43 prediction calibration on Run46 was weak, Option A is the safer next handoff unless the user explicitly wants overnight scale. Do not generate CAE/INP until one option is explicitly selected.
