# Stage 3 Run 10 - Variable-N Normalized Reward and Surrogate Dataset

## Purpose
Build a clean within-N normalized reward dataset, scan-order feature table, surrogate-pretraining table, and pairwise preference dataset from the 60 teacher-labelled variable-N probe cases.

## Inputs
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_09_variable_n_probe60_teacher_ranking_analysis\probe60_teacher_ranked_canonical.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_06_variable_n_probe60_candidate_order_generation\variable_N_probe60_candidate_orders.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_09_variable_n_probe60_teacher_ranking_analysis\pareto_front_cases.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_09_variable_n_probe60_teacher_ranking_analysis\run09_claim_boundary.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_08_probe60_odb_teacher_validation\probe60_odb_teacher_labels.csv`

## Validation Verdict
- `PASS_RUN10_INPUTS_READY_60_TEACHER_LABELS_WITH_SCAN_ORDERS`
- Total rows: 60
- Per-N counts: {12: 15, 16: 15, 24: 15, 40: 15}

## Objective Hierarchy
- Primary: U2 / warpage.
- Safety: PEEQ.
- Secondary diagnostic / tie-breaker: SurfaceT proxy.
- All ranking and reward normalization are within N.

## Normalization Strategy
- Rank scores map best within-N rank to 1.0 and worst within-N rank to 0.0.
- Min-max scores are computed within each N.
- Z-score desirability negates within-N z scores because lower raw metrics are better.
- Raw objective magnitudes are preserved but are not used as direct cross-N reward scales.

## Reward Variants V01-V05
- V01: U2-primary rank reward, 70/20/10 for U2/PEEQ/SurfaceT.
- V02: safety-weighted rank reward, 60/30/10.
- V03: SurfaceT-aware diagnostic reward, 55/25/20.
- V04: constrained penalty reward with penalties for weak U2, PEEQ, or SurfaceT ranks.
- V05: lexicographic constrained score sorted by U2, then PEEQ, then SurfaceT.

## Canonical Reward Dataset Summary
- Rows: 60
- Top reward_mean_all per N: {'N12': 'N12_A08_block_interleaved_quarters', 'N16': 'N16_A03_greedy_maximin_distance', 'N24': 'N24_A04_method_c_u2_first_engineering', 'N40': 'N40_A04_method_c_u2_first_engineering'}

## Scan-Order Feature Table Summary
- Rows: 60
- Features include normalized jump, edge/center timing, parity, monotonicity, direction reversal, and unvisited-gap proxy summaries.

## Surrogate-Pretraining Table Summary
- Rows: 60
- Includes raw metrics, within-N normalized scores, reward targets, scan-order features, candidate labels, and Pareto flags.

## Pairwise Preference Dataset Summary
- Rows: 420
- Pairwise preferences are generated within each N only: 15 choose 2 per N, 420 total.

## Dataset Split Definitions
- Split definitions: 11 total split rows.
- Includes leave-N-out, core N40 generalization, larger-N generalization, and 5 random stratified folds.

## Reward Diagnostics
- Strongest reward variant correlation: {'diagnostic_type': 'reward_variant_spearman', 'item_a': 'reward_v01_u2_primary', 'item_b': 'reward_mean_all', 'value': 0.9987912134680874}
- Strongest feature-reward correlation: {'feature_name': 'normalized_mean_jump', 'target': 'reward_mean_all', 'spearman': -0.38767230014674814, 'abs_spearman': 0.38767230014674814}
- Pareto but not high-reward cases: 7
- High-reward but not Pareto cases: 9

## Claim Boundary
- Run10 constructs a dataset and candidate reward formulations only.
- It does not train the final RL policy.
- It does not prove surrogate accuracy or variable-N RL policy superiority.
- It does not transfer fixed-32 absolute U2 guards to variable-N.

## Output Files
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_10_variable_n_normalized_reward_surrogate_dataset\run10_input_validation_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_10_variable_n_normalized_reward_surrogate_dataset\probe60_variable_n_reward_dataset.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_10_variable_n_normalized_reward_surrogate_dataset\probe60_variable_n_reward_dataset.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_10_variable_n_normalized_reward_surrogate_dataset\probe60_scan_order_features.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_10_variable_n_normalized_reward_surrogate_dataset\probe60_scan_order_features.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_10_variable_n_normalized_reward_surrogate_dataset\probe60_surrogate_pretraining_table.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_10_variable_n_normalized_reward_surrogate_dataset\probe60_surrogate_pretraining_table.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_10_variable_n_normalized_reward_surrogate_dataset\probe60_pairwise_preference_dataset.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_10_variable_n_normalized_reward_surrogate_dataset\probe60_pairwise_preference_dataset.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_10_variable_n_normalized_reward_surrogate_dataset\probe60_dataset_splits.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_10_variable_n_normalized_reward_surrogate_dataset\probe60_dataset_splits_summary.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_10_variable_n_normalized_reward_surrogate_dataset\reward_diagnostics_summary.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_10_variable_n_normalized_reward_surrogate_dataset\reward_diagnostics_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_10_variable_n_normalized_reward_surrogate_dataset\feature_reward_correlation_summary.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_10_variable_n_normalized_reward_surrogate_dataset\run10_claim_boundary.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_10_variable_n_normalized_reward_surrogate_dataset\run10_claim_boundary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_10_variable_n_normalized_reward_surrogate_dataset\figures\reward_mean_all_distribution_per_N.png`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_10_variable_n_normalized_reward_surrogate_dataset\figures\reward_variant_correlation_heatmap.png`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_10_variable_n_normalized_reward_surrogate_dataset\figures\u2_rank_vs_reward_mean_all_per_N.png`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_10_variable_n_normalized_reward_surrogate_dataset\figures\peeq_rank_vs_reward_mean_all_per_N.png`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_10_variable_n_normalized_reward_surrogate_dataset\figures\surfaceT_rank_vs_reward_mean_all_per_N.png`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_10_variable_n_normalized_reward_surrogate_dataset\figures\top_feature_reward_correlations.png`
- `E:\Projects\RL-LAM-ScanOpt\docs\stage3\runs\run_10_variable_n_normalized_reward_surrogate_dataset\RUN_10_VARIABLE_N_NORMALIZED_REWARD_SURROGATE_DATASET_REPORT.md`

## Recommended Run11
Build the first lightweight variable-N surrogate / reward model using the run10 table. Use leave-N-out validation and report whether simple feature-based models can predict within-N normalized reward/ranks. Do not train the final RL policy yet.
