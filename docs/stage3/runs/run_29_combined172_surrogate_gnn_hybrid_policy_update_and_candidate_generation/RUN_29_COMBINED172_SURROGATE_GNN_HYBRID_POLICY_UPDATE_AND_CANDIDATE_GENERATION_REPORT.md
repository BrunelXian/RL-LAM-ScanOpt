# Stage 3 Run 29 - Combined172 Surrogate, GNN, and Hybrid-Policy Candidate Generation

## Purpose
Update lightweight surrogate models, offline GNN reward modeling, and graph-pointer behavior cloning using combined172, then generate hybrid-policy candidate batches without CAE/solver activity.

## Inputs
- Combined172 RL-ready dataset: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_28_shortlist64_teacher_metrics_ingestion_and_combined172_ranking\combined172_RL_ready_dataset.csv`
- Combined172 teacher dataset: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_28_shortlist64_teacher_metrics_ingestion_and_combined172_ranking\combined172_teacher_dataset.csv`
- Run28 report: `E:\Projects\RL-LAM-ScanOpt\docs\stage3\runs\run_28_shortlist64_teacher_metrics_ingestion_and_combined172_ranking\RUN_28_SHORTLIST64_TEACHER_METRICS_INGESTION_AND_COMBINED172_RANKING_REPORT.md`

## Combined172 Validation
- Verdict: `PASS_RUN29_COMBINED172_INPUTS_READY`
- Per-N counts: `{12: 32, 16: 32, 24: 54, 40: 54}`

## Surrogate Update
- Best config: `{'target': 'target_reward_combined172_u2_primary', 'feature_set': 'F07_F01_no_n', 'model_name': 'ExtraTreesRegressor', 'protocol': 'leave_N_out', 'macro_spearman': 0.8451076789185723, 'macro_top5_overlap': 2.0, 'macro_top10_overlap': 6.0, 'macro_mae': 0.09250616011426938, 'macro_mean_rank_error': 5.1186342592592595}`
- Improves Run22 macro Spearman: `False`

## GNN Reward Model Update
- Status: `GNN_REWARD_MODEL_TRAINED`
- Leave-N-out macro Spearman: `0.7445167376916271`
- Improves Run26 macro Spearman: `False`

## Graph-Pointer Policy Update
- Status: `GRAPH_POINTER_POLICY_WEIGHTED_IMITATION_TRAINED`
- Training method: offline weighted behavior cloning; no online Abaqus RL.

## Candidate Generation
- Deduplicated counts: `{12: 800, 16: 840, 24: 2500, 40: 3133}`

## Hybrid Batch64
- Count: `64`; per-N: `{12: 8, 16: 8, 24: 24, 40: 24}`

## Hybrid Batch32
- Count: `32`; per-N: `{12: 4, 16: 4, 24: 12, 40: 12}`

## N24/N40 Focused Batch48
- Count: `48`; per-N: `{24: 24, 40: 24}`

## Comparison to Previous Batches
- Summary: `{'batch64_count': 64, 'per_n_counts': {12: 8, 16: 8, 24: 24, 40: 24}, 'total_overlap_combined172_teacher': 0, 'total_overlap_run24_shortlist64': 0, 'total_overlap_run26_batch64': 0, 'mostly_distinct_from_previous': True}`

## Claim Boundary
`RUN29_MODEL_UPDATE_AND_HYBRID_CANDIDATE_GENERATION_ONLY_NO_TEACHER_VALIDATION`.

## Output Files
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_29_combined172_surrogate_gnn_hybrid_policy_update_and_candidate_generation\run29_input_validation_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_29_combined172_surrogate_gnn_hybrid_policy_update_and_candidate_generation\combined172_scan_order_features.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_29_combined172_surrogate_gnn_hybrid_policy_update_and_candidate_generation\combined172_feature_set_definitions.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_29_combined172_surrogate_gnn_hybrid_policy_update_and_candidate_generation\combined172_surrogate_validation_results_detailed.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_29_combined172_surrogate_gnn_hybrid_policy_update_and_candidate_generation\combined172_best_surrogate_configurations.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_29_combined172_surrogate_gnn_hybrid_policy_update_and_candidate_generation\combined172_surrogate_validation_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_29_combined172_surrogate_gnn_hybrid_policy_update_and_candidate_generation\combined172_gnn_reward_model_validation_results.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_29_combined172_surrogate_gnn_hybrid_policy_update_and_candidate_generation\combined172_gnn_reward_model_validation_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_29_combined172_surrogate_gnn_hybrid_policy_update_and_candidate_generation\combined172_graph_pointer_policy_training_log.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_29_combined172_surrogate_gnn_hybrid_policy_update_and_candidate_generation\combined172_graph_pointer_policy_validation_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_29_combined172_surrogate_gnn_hybrid_policy_update_and_candidate_generation\run29_hybrid_candidate_pool_unscored.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_29_combined172_surrogate_gnn_hybrid_policy_update_and_candidate_generation\run29_hybrid_candidate_pool_scored.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_29_combined172_surrogate_gnn_hybrid_policy_update_and_candidate_generation\run29_hybrid_policy_batch64_candidate_orders.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_29_combined172_surrogate_gnn_hybrid_policy_update_and_candidate_generation\run29_hybrid_policy_batch32_candidate_orders.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_29_combined172_surrogate_gnn_hybrid_policy_update_and_candidate_generation\run29_hybrid_policy_N24_N40_focused_batch48_candidate_orders.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_29_combined172_surrogate_gnn_hybrid_policy_update_and_candidate_generation\run29_hybrid_batch64_comparison_to_previous_batches.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_29_combined172_surrogate_gnn_hybrid_policy_update_and_candidate_generation\run29_hybrid_batch64_comparison_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_29_combined172_surrogate_gnn_hybrid_policy_update_and_candidate_generation\run29_claim_boundary.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_29_combined172_surrogate_gnn_hybrid_policy_update_and_candidate_generation\run29_claim_boundary.json`

## Recommended Run30
Create a handoff package for the selected Run29 hybrid batch. If the user wants 60+ overnight jobs, select hybrid batch64; if focused N24/N40 calibration is preferred, select focused batch48; if compute is limited, select batch32. Do not generate CAE/INP until the user selects one batch.
