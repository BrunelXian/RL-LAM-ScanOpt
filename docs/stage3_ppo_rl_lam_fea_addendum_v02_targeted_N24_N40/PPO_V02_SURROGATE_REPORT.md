# PPO v02 N-Specific Surrogate Report

## Purpose

Train targeted N24/N40 reward surrogates using combined552 plus teacher-validated PPO v01 N24/N40 rows.

## Dataset

Input: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v02_targeted_N24_N40\data\v02_targeted_N24_N40_teacher_dataset.csv`. N24/N40 only; no N12/N16/N32.

## Models

Model families: HistGradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor, Ridge baseline.

## Conservative Reward

The saved payloads include ensemble models and use `mean_pred_reward - 0.5*std_pred_reward` as the default conservative reward for PPO training and candidate scoring.

## Best Validation Metrics

- N24: HistGradientBoostingRegressor, Spearman 0.9011, Pearson 0.9086, MAE 0.0958, RMSE 0.1203, top25 hit quality 0.8182
- N40: ExtraTreesRegressor, Spearman 0.9463, Pearson 0.9453, MAE 0.0649, RMSE 0.0956, top25 hit quality 0.9167

## Verdict

`PASS_V02_SURROGATES_READY_FOR_FIXED_N_PPO`
