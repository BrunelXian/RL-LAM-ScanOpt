# Stage 3 Run 17 - Combined80 Surrogate Reward Model Validation Update

## Purpose
Update lightweight surrogate validation using the expanded 80-case combined teacher-labelled dataset.

## Inputs
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_16_batch20_teacher_metrics_ingestion_and_combined80_ranking\combined80_RL_ready_dataset.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_16_batch20_teacher_metrics_ingestion_and_combined80_ranking\combined80_teacher_dataset.csv`
- `E:\Projects\RL-LAM-ScanOpt\docs\stage3\runs\run_16_batch20_teacher_metrics_ingestion_and_combined80_ranking\RUN_16_BATCH20_TEACHER_METRICS_INGESTION_AND_COMBINED80_RANKING_REPORT.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_11_variable_n_surrogate_reward_model_validation\surrogate_validation_results_detailed.csv`

## Combined80 Validation Status
- `PASS_RUN17_INPUTS_READY_80_ROWS_COMBINED_SURROGATE_TABLE`
- Rows: 80
- Per-N counts: {12: 20, 16: 20, 24: 20, 40: 20}

## Feature Reconstruction
Scan-order features were reconstructed from `order_json` for all 80 rows using the run10-style feature logic.

## Targets and Feature Sets
- Primary target: `target_reward_combined80_u2_primary`.
- Feature sets F01-F06 were evaluated, including family/source diagnostics and N-agnostic features.

## Models and Validation Protocols
- MeanBaseline, Ridge, ElasticNet, RandomForestRegressor, ExtraTreesRegressor, and GradientBoostingRegressor.
- Leave-N-out, small/large transfer, stratified folds, probe60-to-batch20, and probe60-plus-some-batch20 protocols.

## Main Leave-N-Out Results
- Best primary config: `ExtraTreesRegressor` / `F01_basic_order`.
- Macro Spearman: `0.8349624060150375`.
- Macro top5 overlap: `3.75`.

## Train Probe60 Test Batch20 Analysis
- Best P06 config: `ElasticNet` / `F01_basic_order`.
- Spearman: `0.5699248120300752`; top5 overlap: `1`.

## Comparison to Run11
- leave_N_out_macro_spearman: run11=0.7678571428571428, run17=0.8349624060150375, delta=0.06710526315789467.
- leave_N_out_macro_top5_overlap: run11=3.75, run17=3.75, delta=0.0.

## Pairwise Preference Update
- Pairwise rows: 760.
- Best AUC: 0.9503634475597091; best accuracy: 0.875.

## Feature Importance Diagnostics
- Feature importances are diagnostic only, not causal.

## Limitations
- Combined80 is still small, and batch20 was actively selected rather than randomly sampled.
- Dataset-source features can be diagnostic but may encode generation-protocol differences.

## Claim Boundary
- No trained RL policy superiority, arbitrary-N generalization, or deployment readiness is claimed.

## Output Files
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_17_combined80_surrogate_reward_model_validation_update\run17_input_validation_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_17_combined80_surrogate_reward_model_validation_update\combined80_scan_order_features.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_17_combined80_surrogate_reward_model_validation_update\run17_feature_set_definitions.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_17_combined80_surrogate_reward_model_validation_update\combined80_surrogate_validation_results_detailed.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_17_combined80_surrogate_reward_model_validation_update\combined80_surrogate_validation_results_detailed.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_17_combined80_surrogate_reward_model_validation_update\combined80_best_surrogate_configurations.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_17_combined80_surrogate_reward_model_validation_update\combined80_best_surrogate_configurations.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_17_combined80_surrogate_reward_model_validation_update\combined80_predictions_target_reward_u2_primary.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_17_combined80_surrogate_reward_model_validation_update\combined80_feature_importance_summary.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_17_combined80_surrogate_reward_model_validation_update\combined80_feature_importance_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_17_combined80_surrogate_reward_model_validation_update\combined80_pairwise_preference_dataset.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_17_combined80_surrogate_reward_model_validation_update\combined80_pairwise_preference_validation_summary.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_17_combined80_surrogate_reward_model_validation_update\combined80_pairwise_preference_validation_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_17_combined80_surrogate_reward_model_validation_update\run11_vs_run17_surrogate_comparison.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_17_combined80_surrogate_reward_model_validation_update\run17_diagnostic_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_17_combined80_surrogate_reward_model_validation_update\run17_claim_boundary.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_17_combined80_surrogate_reward_model_validation_update\run17_claim_boundary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_17_combined80_surrogate_reward_model_validation_update\figures\run11_vs_run17_macro_metrics.png`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_17_combined80_surrogate_reward_model_validation_update\figures\leave_N_out_predicted_vs_true.png`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_17_combined80_surrogate_reward_model_validation_update\figures\probe60_train_batch20_test_predicted_vs_true.png`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_17_combined80_surrogate_reward_model_validation_update\figures\top_feature_importance.png`
- `E:\Projects\RL-LAM-ScanOpt\docs\stage3\runs\run_17_combined80_surrogate_reward_model_validation_update\RUN_17_COMBINED80_SURROGATE_REWARD_MODEL_VALIDATION_UPDATE_REPORT.md`

## Recommended Run18
Generate a second offline candidate batch using the combined80-updated surrogate, biased toward N24/N40 and including calibration/sentinel candidates.
