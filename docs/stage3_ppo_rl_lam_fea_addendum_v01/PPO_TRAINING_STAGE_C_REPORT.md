# PPO Training Stage C Report

## 1. Purpose

Train a MaskablePPO policy in `LamScanOrderPPOEnv` using the frozen FEA-teacher-trained surrogate reward model as sparse terminal reward. This is not online Abaqus PPO.

## 2. Inputs

- PPO config: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\ppo_training\ppo_training_config.json`
- Surrogate reward model: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\surrogate_reward_model\models\ppo_surrogate_reward_model_best.joblib`

## 3. Surrogate Reward Model Used

The Stage B frozen `HistGradientBoostingRegressor` reward emulator was loaded through `PPOSurrogateRewardModel` and used only for terminal rewards.

## 4. PPO Environment Definition

- Environment: `LamScanOrderSurrogateRewardEnv` wrapping `LamScanOrderPPOEnv`
- Supported N: `[12, 16, 24, 40]`
- N sampling: balanced random N during training
- Reward: sparse terminal surrogate reward
- Intermediate valid rewards: 0

## 5. Action Mask Verification

- Action mask audit: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\ppo_training\tables\ppo_action_mask_verification.csv`
- Action masks verified: `True`

## 6. PPO Config

- Algorithm: `MaskablePPO`
- Policy: `MlpPolicy`
- Timesteps completed: `200352`
- n_envs: `1`
- Vectorization status: `deferred_for_first_action_mask_audit_run`

## 7. Training Status

- Episode count logged: `4394`
- Mean terminal surrogate reward during training: `0.7296862377357832`
- Training elapsed seconds: `697.9653370380402`

## 8. Checkpoint Path

`E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\ppo_training\checkpoints\maskable_ppo_lam_scan_order_final.zip`

## 9. Parameter Count

- Total policy parameters: `72937`
- Trainable policy parameters: `72937`

## 10. Internal Surrogate Evaluation Results

| n | eval_episodes | legal_orders | illegal_orders | mean_surrogate_reward | max_surrogate_reward | min_surrogate_reward | unique_orders | duplicate_rate | mean_jump | max_jump_mean | adjacent_fraction_mean | parity_switch_fraction_mean | center_edge_early_bias_mean | simple_baseline_best_surrogate_reward | simple_baseline_mean_surrogate_reward | ppo_minus_baseline_best | ppo_minus_baseline_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 12 | 40 | 40 | 0 | 0.853606 | 0.873723 | 0.850597 | 3 | 0.925 | 3.36364 | 9.7 | 0 | 0.0909091 | 0.515152 | 0.559081 | 0.178659 | 0.294525 | 0.674947 |
| 16 | 40 | 40 | 0 | 0.887583 | 0.908438 | 0.535694 | 7 | 0.825 | 3.59333 | 12.55 | 0 | 0.07 | 0.466667 | 0.494622 | 0.139107 | 0.392962 | 0.748477 |
| 24 | 40 | 40 | 0 | 0.822741 | 0.865851 | 0.71814 | 21 | 0.475 | 4.42391 | 13.5 | 0.00108696 | 0.0543478 | 0.26087 | 0.603116 | 0.180606 | 0.219626 | 0.642136 |
| 40 | 40 | 40 | 0 | 0.679143 | 0.744067 | 0.515316 | 21 | 0.475 | 7.63974 | 22.275 | 0.000641026 | 0.0782051 | 0.0174359 | 0.514706 | 0.140579 | 0.164437 | 0.538564 |

## 11. Legality Audit

| n | eval_rows | legal_rows | unique_orders | illegal_rows | all_legal |
| --- | --- | --- | --- | --- | --- |
| 12 | 40 | 40 | 3 | 0 | True |
| 16 | 40 | 40 | 7 | 0 | True |
| 24 | 40 | 40 | 21 | 0 | True |
| 40 | 40 | 40 | 21 | 0 | True |

## 12. Baseline Comparison In Surrogate Environment

These comparisons are surrogate-environment only, not Abaqus teacher validation.

| index | n | mean | max |
| --- | --- | --- | --- |
| 0 | 12 | 0.178659 | 0.559081 |
| 1 | 16 | 0.139107 | 0.494622 |
| 2 | 24 | 0.180606 | 0.603116 |
| 3 | 40 | 0.140579 | 0.514706 |

## 13. Limitations

- Internal evaluation orders are not final Stage D candidate orders.
- Rewards are surrogate predictions, not Abaqus teacher metrics.
- No physical superiority claim can be made until Stage E teacher validation.
- Vectorized action-mask training was deferred for the first audit run.

## 14. Claim Boundary

Safe claim: a MaskablePPO policy was trained in a surrogate reward environment derived from FEA teacher-labelled scan-order data.

Not safe: teacher validation, physical improvement, global optimisation, or final optimiser claims.

## 15. Ready For Stage D PPO-Only Candidate Generation

`True`

## 16. Verdict

`PASS_PPO_TRAINING_READY_FOR_POLICY_ONLY_CANDIDATE_GENERATION`
