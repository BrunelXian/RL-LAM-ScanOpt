# Stage 3 Run 12 - Offline Surrogate-Screened Candidate Generation

## Purpose
Generate and screen new variable-N scan-order candidates offline using the run11 diagnostic surrogate.

## Inputs
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_10_variable_n_normalized_reward_surrogate_dataset\probe60_surrogate_pretraining_table.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_11_variable_n_surrogate_reward_model_validation\best_surrogate_configurations.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_11_variable_n_surrogate_reward_model_validation\run11_feature_set_definitions.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_06_variable_n_probe60_candidate_order_generation\variable_N_probe60_candidate_orders.csv`

## Run11 Surrogate Basis
- Model: `ExtraTreesRegressor`
- Feature set: `F03_family_plus_features`
- Target: `target_reward_mean_all`
- Refit on all 60 run10 rows for offline scoring only.

## Candidate Generation Methods
- Engineering baselines, regular-jump sweeps, Method-C-inspired heuristics, seed mutations, crossovers, random/quasi-random diversity, and graph-pointer-inspired proxy candidates.
- Graph-pointer-inspired candidates are proxy/inspired orders, not trained RL outputs.

## Candidate Validation and Deduplication
- Raw generated counts: {12: 2183, 16: 2399, 24: 3023, 40: 5039}
- Deduplicated new counts: {12: 1245, 16: 1250, 24: 1250, 40: 1250}
- Existing teacher-labelled orders are retained as reference calibration rows only.

## Surrogate Scoring Method
- ExtraTrees predictions are within-N ranked by predicted normalized reward.
- Tree prediction standard deviation is used as an uncertainty proxy.
- Kendall distance to the nearest teacher-labelled order is used as a novelty/diversity proxy.

## Candidate Pool Summary
- Scored candidate count including references: 5055

## Final Shortlist Per N
- Shortlist counts: {12: 24, 16: 24, 24: 24, 40: 24}

## Recommended Future Teacher Batch20
- Batch20 counts: {12: 5, 16: 5, 24: 5, 40: 5}

## Predicted Improvement vs Existing Teacher-Labelled Cases
- N12: top surrogate candidate `RUN12_N12_C00417` predicted 0.7767; existing best `N12_A08_block_interleaved_quarters` teacher reward 0.9043; exceeds flag `False`. Surrogate-only.
- N16: top surrogate candidate `RUN12_N16_C01673` predicted 0.8065; existing best `N16_A03_greedy_maximin_distance` teacher reward 0.8614; exceeds flag `False`. Surrogate-only.
- N24: top surrogate candidate `RUN12_N24_C03770` predicted 0.8347; existing best `N24_A04_method_c_u2_first_engineering` teacher reward 0.9786; exceeds flag `False`. Surrogate-only.
- N40: top surrogate candidate `RUN12_N40_C06776` predicted 0.8395; existing best `N40_A04_method_c_u2_first_engineering` teacher reward 0.9929; exceeds flag `False`. Surrogate-only.

## Diversity and Uncertainty Diagnostics
- Shortlist includes surrogate-top, U2-primary, diversity, Method-C-inspired, uncertainty sentinel, and negative/control sentinel buckets.

## Claim Boundary
- Candidate rankings are predictions only.
- No physical validation, Abaqus execution, CAE/INP/JNL generation, or final RL training occurred.

## Output Files
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_12_offline_surrogate_screened_candidate_generation\run12_input_validation_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_12_offline_surrogate_screened_candidate_generation\run12_surrogate_model_metadata.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_12_offline_surrogate_screened_candidate_generation\run12_candidate_pool_unscored.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_12_offline_surrogate_screened_candidate_generation\run12_candidate_pool_unscored.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_12_offline_surrogate_screened_candidate_generation\run12_candidate_pool_scored.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_12_offline_surrogate_screened_candidate_generation\run12_candidate_pool_scored.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_12_offline_surrogate_screened_candidate_generation\run12_candidate_shortlist_per_N.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_12_offline_surrogate_screened_candidate_generation\run12_candidate_shortlist_per_N.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_12_offline_surrogate_screened_candidate_generation\run12_recommended_future_teacher_batch20.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_12_offline_surrogate_screened_candidate_generation\run12_recommended_future_teacher_batch20.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_12_offline_surrogate_screened_candidate_generation\run12_predicted_improvement_vs_existing.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_12_offline_surrogate_screened_candidate_generation\run12_predicted_improvement_vs_existing.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_12_offline_surrogate_screened_candidate_generation\run12_candidate_generation_diagnostics.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_12_offline_surrogate_screened_candidate_generation\run12_candidate_generation_diagnostics.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_12_offline_surrogate_screened_candidate_generation\run12_claim_boundary.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_12_offline_surrogate_screened_candidate_generation\run12_claim_boundary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_12_offline_surrogate_screened_candidate_generation\figures\N12_predicted_reward_histogram.png`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_12_offline_surrogate_screened_candidate_generation\figures\N16_predicted_reward_histogram.png`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_12_offline_surrogate_screened_candidate_generation\figures\N24_predicted_reward_histogram.png`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_12_offline_surrogate_screened_candidate_generation\figures\N40_predicted_reward_histogram.png`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_12_offline_surrogate_screened_candidate_generation\figures\predicted_reward_vs_uncertainty.png`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_12_offline_surrogate_screened_candidate_generation\figures\shortlist_family_composition.png`
- `E:\Projects\RL-LAM-ScanOpt\docs\stage3\runs\run_12_offline_surrogate_screened_candidate_generation\RUN_12_OFFLINE_SURROGATE_SCREENED_CANDIDATE_GENERATION_REPORT.md`

## Recommended Run13
Do not immediately run all run12 candidates. First create a small controlled teacher-validation handoff package, likely 20 cases total: 5 per N, or 4 for N12/N16 and 6 for N24/N40 if prioritizing core larger N. Run13 should create CAE handoff artifacts only after user approval.
