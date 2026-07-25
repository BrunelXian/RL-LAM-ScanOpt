# Stage 3 Run 11 - Variable-N Surrogate Reward Model Validation

## Purpose
Evaluate whether lightweight feature-based surrogate models can predict within-N normalized rewards and ranks from scan-order features and metadata.

## Inputs
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_10_variable_n_normalized_reward_surrogate_dataset\probe60_surrogate_pretraining_table.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_10_variable_n_normalized_reward_surrogate_dataset\probe60_variable_n_reward_dataset.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_10_variable_n_normalized_reward_surrogate_dataset\probe60_scan_order_features.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_10_variable_n_normalized_reward_surrogate_dataset\probe60_pairwise_preference_dataset.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_10_variable_n_normalized_reward_surrogate_dataset\probe60_dataset_splits.json`
- `E:\Projects\RL-LAM-ScanOpt\docs\stage3\runs\run_10_variable_n_normalized_reward_surrogate_dataset\RUN_10_VARIABLE_N_NORMALIZED_REWARD_SURROGATE_DATASET_REPORT.md`
- `E:\Projects\RL-LAM-ScanOpt\artifacts\manifests\stage3_run_10_manifest.json`

## Validation Status
- `PASS_RUN11_INPUTS_READY_60_ROWS_SURROGATE_TABLE`
- Rows: 60
- Per-N counts: {12: 15, 16: 15, 24: 15, 40: 15}

## Feature Sets
- `F01_basic_order`: 10 numeric, 0 categorical features.
- `F02_full_handcrafted`: 30 numeric, 0 categorical features.
- `F03_family_plus_features`: 30 numeric, 3 categorical features.
- `F04_no_family_generalization`: 29 numeric, 0 categorical features.
- `F05_n_agnostic`: 27 numeric, 0 categorical features.

## Targets
- `target_reward_mean_all` is the primary target.
- Secondary targets include reward variants and U2/PEEQ/SurfaceT rank scores.

## Models
- MeanBaseline, Ridge, ElasticNet, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor when available, and KNeighborsRegressor.

## Validation Protocols
- P01 leave-N-out, P02 core generalization, P03/P05 small-to-large, P06 large-to-small, and P04 run10 stratified five-fold.

## Main Leave-N-Out Results
- Best diagnostic primary configuration: `ExtraTreesRegressor` with `F03_family_plus_features`.
- Macro Spearman: 0.7678571428571428
- Macro top5 overlap: 3.75

### Held-Out N Results For Best Primary Configuration
- test_N12: Spearman=0.6607142857142856, top5_overlap=3, MAE=0.13580647194822656
- test_N16: Spearman=0.8857142857142855, top5_overlap=5, MAE=0.0983358475999434
- test_N24: Spearman=0.7964285714285713, top5_overlap=3, MAE=0.12933988502702565
- test_N40: Spearman=0.7285714285714285, top5_overlap=4, MAE=0.15323450710090966

## Best Diagnostic Surrogate Configurations
- `target_reward_mean_all`: `ExtraTreesRegressor` / `F03_family_plus_features`, macro Spearman=0.7678571428571428, top5=3.75
- `target_reward_v01_u2_primary`: `ExtraTreesRegressor` / `F03_family_plus_features`, macro Spearman=0.7705357142857141, top5=3.75
- `target_reward_v02_safety_weighted`: `ExtraTreesRegressor` / `F03_family_plus_features`, macro Spearman=0.7794642857142856, top5=3.75
- `target_reward_v04_penalized`: `ExtraTreesRegressor` / `F05_n_agnostic`, macro Spearman=0.788392857142857, top5=3.75
- `target_reward_v05_lexicographic`: `Ridge` / `F05_n_agnostic`, macro Spearman=0.7607142857142856, top5=3.75
- `target_u2_score_rank`: `Ridge` / `F05_n_agnostic`, macro Spearman=0.7607142857142856, top5=3.75
- `target_peeq_score_rank`: `ExtraTreesRegressor` / `F04_no_family_generalization`, macro Spearman=0.8267857142857142, top5=3.75
- `target_surfaceT_score_rank`: `ElasticNet` / `F03_family_plus_features`, macro Spearman=0.6375, top5=3.5

## Prediction / Ranking Diagnostics
- Prediction tables are saved for the primary target across all protocols.

## Feature Importance Diagnostics
- Tree feature importances and standardized linear coefficients are reported as diagnostic only, not causal.

## Pairwise Preference Baseline
- Status: `PAIRWISE_MODEL_VALIDATED`
- Pair count: 420
- Best accuracy: 0.8285714285714286
- Best AUC: 0.9130272489734975

## Failure Modes and Limitations
- Only 60 teacher-labelled cases are available.
- Leave-N-out test folds have 15 cases each, so R2 and rank metrics can be unstable.
- Handcrafted features may miss sequence/graph interactions.
- Family labels can inflate interpolation-style diagnostics and should not be treated as physical mechanisms.

## Claim Boundary
- This is surrogate feasibility validation only.
- It does not train the final RL policy.
- It does not prove arbitrary-N generalization, physical optimality, or deployment readiness.

## Output Files
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_11_variable_n_surrogate_reward_model_validation\run11_input_validation_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_11_variable_n_surrogate_reward_model_validation\run11_feature_set_definitions.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_11_variable_n_surrogate_reward_model_validation\surrogate_validation_results_detailed.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_11_variable_n_surrogate_reward_model_validation\surrogate_validation_results_detailed.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_11_variable_n_surrogate_reward_model_validation\best_surrogate_configurations.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_11_variable_n_surrogate_reward_model_validation\best_surrogate_configurations.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_11_variable_n_surrogate_reward_model_validation\surrogate_predictions_target_reward_mean_all.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_11_variable_n_surrogate_reward_model_validation\feature_importance_summary.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_11_variable_n_surrogate_reward_model_validation\feature_importance_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_11_variable_n_surrogate_reward_model_validation\pairwise_preference_validation_summary.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_11_variable_n_surrogate_reward_model_validation\pairwise_preference_validation_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_11_variable_n_surrogate_reward_model_validation\run11_claim_boundary.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_11_variable_n_surrogate_reward_model_validation\run11_claim_boundary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_11_variable_n_surrogate_reward_model_validation\figures\best_leave_N_out_predicted_vs_true_reward_mean_all.png`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_11_variable_n_surrogate_reward_model_validation\figures\best_leave_N_out_predicted_rank_vs_true_rank.png`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_11_variable_n_surrogate_reward_model_validation\figures\top_feature_importance_bar.png`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_11_variable_n_surrogate_reward_model_validation\figures\leave_N_out_spearman_by_model.png`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_11_variable_n_surrogate_reward_model_validation\figures\leave_N_out_top5_overlap_by_model.png`
- `E:\Projects\RL-LAM-ScanOpt\docs\stage3\runs\run_11_variable_n_surrogate_reward_model_validation\RUN_11_VARIABLE_N_SURROGATE_REWARD_MODEL_VALIDATION_REPORT.md`

## Recommended Run12
run12 should generate candidate scan orders using the best diagnostic surrogate as an offline screening model, but still not run Abaqus.
