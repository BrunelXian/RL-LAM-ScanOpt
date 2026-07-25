# Stage 3 Run 22 - Combined108 Surrogate Reward Model Validation Update

## Purpose
Update lightweight surrogate validation using the expanded 108-case combined teacher-labelled dataset.

## Inputs
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_21_batch28_teacher_metrics_ingestion_and_combined108_ranking\combined108_RL_ready_dataset.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_21_batch28_teacher_metrics_ingestion_and_combined108_ranking\combined108_teacher_dataset.csv`
- `E:\Projects\RL-LAM-ScanOpt\docs\stage3\runs\run_21_batch28_teacher_metrics_ingestion_and_combined108_ranking\RUN_21_BATCH28_TEACHER_METRICS_INGESTION_AND_COMBINED108_RANKING_REPORT.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_17_combined80_surrogate_reward_model_validation_update\combined80_surrogate_validation_results_detailed.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_11_variable_n_surrogate_reward_model_validation\surrogate_validation_results_detailed.csv`

## Combined108 Validation Status
- `PASS_RUN22_INPUTS_READY_108_ROWS_COMBINED_SURROGATE_TABLE`
- Rows: 108
- Per-N counts: {12: 24, 16: 24, 24: 30, 40: 30}

## Feature Reconstruction
Scan-order features were reconstructed from `order_json` for all 108 rows using the run10/run17-style feature logic.

## Targets and Feature Sets
- Primary target: `target_reward_combined108_u2_primary`.
- Feature sets F01-F07 were evaluated, including family/source diagnostics, N-agnostic features, and F01 without raw N.

## Models and Validation Protocols
- MeanBaseline, Ridge, ElasticNet, RandomForestRegressor, ExtraTreesRegressor, and GradientBoostingRegressor.
- Leave-N-out, small/large transfer, stratified folds, combined80-to-batch28, probe60-to-active-learning, and incremental calibration protocols.

## Main Leave-N-Out Results
- Best primary config: `ExtraTreesRegressor` / `F01_basic_order`.
- Macro Spearman: `0.865143154229337`.
- Macro top5 overlap: `2.5`.

## Train Combined80 Test Batch28 Analysis
- Best P06 config: `ExtraTreesRegressor` / `F05_n_agnostic`.
- Spearman: `0.6239737274220033`; top5 overlap: `1`.

## N40 Stability Analysis
- `Ridge` / `F03_family_plus_features` N40 Spearman `0.9136818687430479`, top5 `3`.
- `Ridge` / `F06_no_dataset_source` N40 Spearman `0.9136818687430479`, top5 `3`.
- `ElasticNet` / `F02_full_handcrafted` N40 Spearman `0.9052280311457174`, top5 `3`.
- `ElasticNet` / `F04_no_family_generalization` N40 Spearman `0.9052280311457174`, top5 `3`.
- `ElasticNet` / `F05_n_agnostic` N40 Spearman `0.9052280311457174`, top5 `3`.

## Comparison to Run11 and Run17
- leave_N_out_macro_spearman: run11=0.7678571428571428, run17=0.8349624060150375, run22=0.865143154229337, delta22-17=0.030180748214299502.
- leave_N_out_macro_top5_overlap: run11=3.75, run17=3.75, run22=2.5, delta22-17=-1.25.
- best_top5_overlap_config: run11=nan, run17=4.25, run22=2.5, delta22-17=-1.75.
- pairwise_best_auc: run11=0.913, run17=0.9504, run22=0.9438654129375779, delta22-17=-0.006534587062422159.

## Pairwise Preference Update
- Pairwise rows: 1422.
- Best AUC: 0.9438654129375779; best accuracy: 0.8697183098591549.

## Feature Importance Diagnostics
- Feature importances are diagnostic only, not causal.

## Limitations
- Combined108 remains small, and batch20/batch28 were actively selected rather than randomly sampled.
- Dataset-source features can be diagnostic but may encode generation-protocol differences.

## Claim Boundary
- No trained RL policy superiority, arbitrary-N generalization, or deployment readiness is claimed.

## Output Files
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_22_combined108_surrogate_reward_model_validation_update\run22_input_validation_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_22_combined108_surrogate_reward_model_validation_update\combined108_scan_order_features.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_22_combined108_surrogate_reward_model_validation_update\run22_feature_set_definitions.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_22_combined108_surrogate_reward_model_validation_update\combined108_surrogate_validation_results_detailed.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_22_combined108_surrogate_reward_model_validation_update\combined108_surrogate_validation_results_detailed.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_22_combined108_surrogate_reward_model_validation_update\combined108_best_surrogate_configurations.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_22_combined108_surrogate_reward_model_validation_update\combined108_best_surrogate_configurations.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_22_combined108_surrogate_reward_model_validation_update\combined108_predictions_target_reward_u2_primary.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_22_combined108_surrogate_reward_model_validation_update\combined108_feature_importance_summary.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_22_combined108_surrogate_reward_model_validation_update\combined108_feature_importance_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_22_combined108_surrogate_reward_model_validation_update\combined108_pairwise_preference_dataset.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_22_combined108_surrogate_reward_model_validation_update\combined108_pairwise_preference_validation_summary.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_22_combined108_surrogate_reward_model_validation_update\combined108_pairwise_preference_validation_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_22_combined108_surrogate_reward_model_validation_update\run11_run17_run22_surrogate_comparison.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_22_combined108_surrogate_reward_model_validation_update\run22_diagnostic_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_22_combined108_surrogate_reward_model_validation_update\run22_claim_boundary.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_22_combined108_surrogate_reward_model_validation_update\run22_claim_boundary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_22_combined108_surrogate_reward_model_validation_update\figures\run11_run17_run22_macro_metrics.png`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_22_combined108_surrogate_reward_model_validation_update\figures\leave_N_out_predicted_vs_true.png`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_22_combined108_surrogate_reward_model_validation_update\figures\combined80_train_batch28_test_predicted_vs_true.png`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_22_combined108_surrogate_reward_model_validation_update\figures\top_feature_importance.png`
- `E:\Projects\RL-LAM-ScanOpt\docs\stage3\runs\run_22_combined108_surrogate_reward_model_validation_update\RUN_22_COMBINED108_SURROGATE_REWARD_MODEL_VALIDATION_UPDATE_REPORT.md`

## Recommended Run23
Stop exploitation and create an active-learning coverage design, with special attention to N40 calibration and uncertainty coverage.
