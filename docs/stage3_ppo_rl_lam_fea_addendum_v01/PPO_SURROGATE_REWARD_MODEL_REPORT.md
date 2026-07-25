# PPO Surrogate Reward Model Report

## 1. Purpose

Train and freeze a supervised terminal reward emulator from the FEA teacher-labelled native combined552 dataset. This surrogate is for later PPO environment rewards; it is not a PPO policy and it is not the physical teacher.

## 2. Dataset

- Dataset: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\FROZEN_stage3_native_combined552_teacher_dataset.csv`
- Rows used: `552`
- Row counts by N: `{'12': 78, '16': 78, '24': 190, '40': 206}`
- N32 usage: not used

## 3. Feature Schema

- Feature schema JSON: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\surrogate_reward_model\models\ppo_surrogate_feature_schema.json`
- Feature count: `111`

## 4. Target Schema

- Target schema JSON: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\surrogate_reward_model\models\ppo_surrogate_target_schema.json`
- Primary target: `reward_lex_u2_peeq_surfacet`
- Reward direction: larger is better
- Mises role: diagnostic only

## 5. Train/Validation Split

- Split random state: `20260623`
- Split type: deterministic stratified split by N
- Train rows: `414`
- Validation rows: `138`

## 6. Model Candidates

- ExtraTreesRegressor
- RandomForestRegressor
- GradientBoostingRegressor
- HistGradientBoostingRegressor
- Ridge baseline
- ElasticNet baseline

## 7. Selected Model

- Selected primary model: `HistGradientBoostingRegressor`
- Model artifact: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\surrogate_reward_model\models\ppo_surrogate_reward_model_best.joblib`

## 8. Validation Metrics

- Spearman: `0.878558039393863`
- Pearson: `0.8863107612337054`
- MAE: `0.09416472314732778`
- RMSE: `0.13792435602768863`

## 9. Per-N Stability

| n | validation_rows | spearman | pearson | mae | rmse |
| --- | --- | --- | --- | --- | --- |
| 12 | 20 | 0.816403 | 0.825921 | 0.135663 | 0.175176 |
| 16 | 20 | 0.793985 | 0.85497 | 0.131751 | 0.181925 |
| 24 | 47 | 0.844935 | 0.882834 | 0.0941225 | 0.142786 |
| 40 | 51 | 0.953573 | 0.948951 | 0.06319 | 0.0875925 |

## 10. Top-k Quality

| n | validation_rows | k | topk_hit_fraction | predicted_topk_teacher_reward_mean | predicted_topk_teacher_reward_best | observed_validation_best_reward | observed_validation_mean_reward | predicted_topk_lift_fraction_vs_mean_to_best |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 12 | 20 | 5 | 0.4 | 0.771377 | 1.01143 | 1.05604 | 0.576948 | 0.405828 |
| 16 | 20 | 5 | 0.8 | 0.874169 | 1.07052 | 1.07052 | 0.556886 | 0.617722 |
| 24 | 47 | 5 | 0.2 | 0.817556 | 1.02709 | 1.03217 | 0.550429 | 0.554503 |
| 40 | 51 | 5 | 0.4 | 0.927234 | 1.0379 | 1.0379 | 0.512728 | 0.789273 |

## 11. Known Limitations

- The model is trained on 552 teacher-labelled native-N examples, so PPO exploration outside the teacher-labelled distribution can be mis-scored.
- The reward is teacher-derived and surrogate-predicted; it is not an Abaqus solve.
- No physical feasibility threshold was invented for strict penalty guards.
- PPO candidates produced later must still be independently validated by Abaqus.

## 12. Whether Suitable For PPO Training

The model is suitable as a first surrogate terminal reward for PPO training if used with the claim boundary and later Abaqus teacher validation. PPO training should log all surrogate versions and freeze the reward model artifact used for each policy run.

## 13. Claim Boundary

- This surrogate is a terminal reward emulator trained on FEA teacher-labelled scan-order data.
- It is not the physical teacher.
- It is not a PPO policy.
- PPO candidates must still be independently validated by Abaqus.

## 14. Verdict

`PASS_PPO_SURROGATE_REWARD_MODEL_READY_FOR_PPO_TRAINING`
