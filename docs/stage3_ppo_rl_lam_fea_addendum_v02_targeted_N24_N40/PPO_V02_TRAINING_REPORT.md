# PPO v02 Fixed-N Training Report

## Training Scope

One seed per N was trained for the initial targeted v02 pass. The optional three-seed ensemble was deferred and is not claimed.

## Checkpoints

- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v02_targeted_N24_N40\ppo_training_v02\checkpoints\N24_seed20260624_maskable_ppo_v02.zip`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v02_targeted_N24_N40\ppo_training_v02\checkpoints\N40_seed20260624_maskable_ppo_v02.zip`

## Internal Surrogate Evaluation

| n | mode | count | mean_reward | max_reward | unique_orders | legal_all |
| --- | --- | --- | --- | --- | --- | --- |
| 24 | deterministic | 1 | 0.6379732442250573 | 0.6379732442250573 | 1 | True |
| 24 | stochastic | 200 | 0.6263256976517908 | 0.6390767891815406 | 188 | True |
| 40 | deterministic | 1 | 0.4173152009736466 | 0.4173152009736466 | 1 | True |
| 40 | stochastic | 200 | 0.3389241144729307 | 0.41357704093899944 | 200 | True |

## Verdict

`WARNING_V02_FIXED_N_PPO_PARTIAL_REVIEW`
