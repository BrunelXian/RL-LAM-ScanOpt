# Stage 3 Run 33 - Combined172 Plus N32 Balanced Surrogate/GNN Candidate Generation

## Purpose
Update offline surrogate, GNN reward, and graph-pointer models using combined172_plus_N32, while controlling for N32 imbalance and legacy metric semantics, then generate new candidate batch options.

## Inputs
- combined172_plus_N32 RL-ready: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_32a_stage2_n32_legacy_teacher_label_ingestion_for_stage3\combined172_plus_N32_RL_ready_dataset.csv`
- native combined172 RL-ready: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_28_shortlist64_teacher_metrics_ingestion_and_combined172_ranking\combined172_RL_ready_dataset.csv`
- N32 dedup training table: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_32a_stage2_n32_legacy_teacher_label_ingestion_for_stage3\n32_legacy_teacher_dataset_dedup_training_332.csv`

## N32 Ingestion Context
N32 is legacy-compatible Stage 2 teacher data, not native Stage 3 teacher validation. PEEQ and Mises columns are mapped proxies with semantic warnings.

## Input Validation
- Verdict: `PASS_RUN33_COMBINED172_PLUS_N32_INPUT_READY_WITH_LEGACY_WARNINGS`
- Rows: `504`
- Per-N counts: `{12: 32, 16: 32, 24: 54, 32: 332, 40: 54}`

## Feature Reconstruction
- Reconstructed Run22/Run29-compatible scan-order descriptors including F01 order features, odd/even transition count, and jump descriptors.

## Surrogate Update
- Best config: `native_only_combined172 / MeanBaseline / F01_basic_order`
- Macro Spearman: `nan`
- Macro top5 overlap: `0.5`

## GNN Reward Update
- Status: `RUN33_GNN_REWARD_MODEL_TRAINED`
- Best regime: `{'regime': 'plus_N32_unweighted', 'macro_spearman': 0.7850523406848685, 'macro_top5_overlap': 1.6, 'n40_spearman': 0.8321515760427128}`

## Graph-Pointer Update
- Status: `RUN33_GRAPH_POINTER_POLICY_WEIGHTED_BC_TRAINED`
- Training is offline weighted behavior cloning only; no online RL was run.

## Effect of N32 Augmentation
- Native Stage 3 prediction effect: `degraded`
- Balanced regime is preferred for interpretation because N32 has 332 rows.

## Candidate Generation
- Deduplicated candidate counts: `{12: 800, 16: 800, 24: 2500, 32: 2500, 40: 3000}`

## Batch Option A: N32-informed Native Batch32
- Path: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_33_combined172_plus_N32_balanced_surrogate_gnn_candidate_generation\run33_N32_informed_native_batch32_candidate_orders.csv`

## Batch Option B: Variable-N Batch40 With N32
- Path: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_33_combined172_plus_N32_balanced_surrogate_gnn_candidate_generation\run33_N32_included_variableN_batch40_candidate_orders.csv`

## Batch Option C: Focused N24/N32/N40 Batch36
- Path: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_33_combined172_plus_N32_balanced_surrogate_gnn_candidate_generation\run33_focused_N24_N32_N40_batch36_candidate_orders.csv`

## Comparison To Superseded Run31
- Exact overlap count: `0`
- Headline: Option A is a fresh N32-informed candidate set replacing the superseded Run31 batch; exact overlap is reported, not reused as approval.

## Claim Boundary
- Run33 is model update and candidate generation only. No teacher validation, no CAE/INP, no solver activity.

## Output Files
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_33_combined172_plus_N32_balanced_surrogate_gnn_candidate_generation\run33_input_validation_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_33_combined172_plus_N32_balanced_surrogate_gnn_candidate_generation\combined172_plus_N32_scan_order_features.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_33_combined172_plus_N32_balanced_surrogate_gnn_candidate_generation\run33_surrogate_validation_results_detailed.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_33_combined172_plus_N32_balanced_surrogate_gnn_candidate_generation\run33_best_surrogate_configurations.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_33_combined172_plus_N32_balanced_surrogate_gnn_candidate_generation\run33_surrogate_validation_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_33_combined172_plus_N32_balanced_surrogate_gnn_candidate_generation\run33_gnn_reward_validation_results.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_33_combined172_plus_N32_balanced_surrogate_gnn_candidate_generation\run33_gnn_reward_validation_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_33_combined172_plus_N32_balanced_surrogate_gnn_candidate_generation\run33_graph_pointer_policy_training_log.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_33_combined172_plus_N32_balanced_surrogate_gnn_candidate_generation\run33_graph_pointer_policy_validation_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_33_combined172_plus_N32_balanced_surrogate_gnn_candidate_generation\run33_candidate_pool_scored.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_33_combined172_plus_N32_balanced_surrogate_gnn_candidate_generation\run33_N32_informed_native_batch32_candidate_orders.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_33_combined172_plus_N32_balanced_surrogate_gnn_candidate_generation\run33_N32_included_variableN_batch40_candidate_orders.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_33_combined172_plus_N32_balanced_surrogate_gnn_candidate_generation\run33_focused_N24_N32_N40_batch36_candidate_orders.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_33_combined172_plus_N32_balanced_surrogate_gnn_candidate_generation\run33_comparison_to_superseded_run31_batch32.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_33_combined172_plus_N32_balanced_surrogate_gnn_candidate_generation\run33_comparison_to_superseded_run31_batch32_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_33_combined172_plus_N32_balanced_surrogate_gnn_candidate_generation\run33_claim_boundary.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_33_combined172_plus_N32_balanced_surrogate_gnn_candidate_generation\run33_claim_boundary.json`

## Recommended Run34
Select one candidate batch and create a handoff package. Choose Option A for the cleanest replacement for abandoned Run31; Option B to include N32 in the next teacher-validation plan; Option C for large/intermediate-N calibration.
