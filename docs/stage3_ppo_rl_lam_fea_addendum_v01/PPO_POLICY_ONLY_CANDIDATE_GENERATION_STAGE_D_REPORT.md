# PPO Policy-Only Candidate Generation Stage D Report

## 1. Purpose

Generate a clean 32-case PPO-only candidate batch for later Abaqus teacher validation. This stage generates candidate orders only.

## 2. PPO Checkpoint Used

`E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\ppo_training\checkpoints\maskable_ppo_lam_scan_order_final.zip`

## 3. Surrogate Reward Model Used For Ranking Only

`E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\surrogate_reward_model\models\ppo_surrogate_reward_model_best.joblib`

## 4. Candidate-Source Boundary

Every selected candidate has `candidate_source = PPO_checkpoint_inference`, `teacher_validated = false`, and `abaqus_validated = false`.

## 5. Rollout Pool Generation Method

For each N, the script ran at least 16 deterministic and 512 stochastic PPO checkpoint rollouts with action masks. Illegal orders were not admitted to the pool.

| n | rollout_rows | unique_orders | deterministic_rows | stochastic_rows |
| --- | --- | --- | --- | --- |
| 12 | 528 | 20 | 16 | 512 |
| 16 | 528 | 62 | 16 | 512 |
| 24 | 528 | 499 | 16 | 512 |
| 40 | 528 | 513 | 16 | 512 |

## 6. Selection Rule

For each N: include one deterministic candidate, add top surrogate-reward PPO-generated candidates, add diversity candidates from the upper half of the PPO-predicted reward pool by maximin Hamming distance, add a novelty-favored upper-half candidate when available, then fill remaining slots from high-reward stochastic PPO-generated orders. No order is repaired, mutated, or hand-designed.

## 7. Selected Batch32 Summary

| n | selected_count |
| --- | --- |
| 12 | 8 |
| 16 | 8 |
| 24 | 8 |
| 40 | 8 |

## 8. Legality Audit

- All legal: `True`
- Duplicate selected hashes: `0`

## 9. Novelty Audit Against Combined552

| n | selected_count | duplicate_combined552_count | min_hamming_mean | min_hamming_min |
| --- | --- | --- | --- | --- |
| 12 | 8 | 1 | 2.5 | 0 |
| 16 | 8 | 0 | 8 | 6 |
| 24 | 8 | 0 | 17.375 | 13 |
| 40 | 8 | 0 | 36.125 | 35 |

## 10. Predicted Surrogate Reward Summary

| n | selected_count | reward_mean | reward_min | reward_max | rollout_pool_rows | rollout_pool_unique_orders |
| --- | --- | --- | --- | --- | --- | --- |
| 12 | 8 | 0.81098 | 0.73704 | 0.873723 | 528 | 20 |
| 16 | 8 | 0.899077 | 0.8355 | 0.935066 | 528 | 62 |
| 24 | 8 | 0.902806 | 0.824662 | 1.00914 | 528 | 499 |
| 40 | 8 | 0.762503 | 0.687247 | 0.792521 | 528 | 513 |

## 11. Order Descriptor Summary

| n | selected_count | mean_abs_jump_mean | max_abs_jump_mean | adjacent_jump_fraction_mean | parity_switch_fraction_mean | early_center_bias_mean | early_edge_bias_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 12 | 8 | 3.47727 | 9.25 | 0 | 0.0909091 | 0.757576 | 0.242424 |
| 16 | 8 | 3.66667 | 12 | 0 | 0.0666667 | 0.7 | 0.3 |
| 24 | 8 | 4.67391 | 13.25 | 0 | 0.0869565 | 0.634058 | 0.365942 |
| 40 | 8 | 7.17949 | 23.125 | 0 | 0.0769231 | 0.560256 | 0.439744 |

## 12. Stage E Abaqus Validation Handoff Preview

`E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\ppo_candidate_generation\selected_batch32\ppo_stageE_candidate_handoff_preview.csv`

## 13. Limitations

- Surrogate scores are predictions only.
- Candidate orders are not physically validated yet.
- Stage D does not generate CAE/INP/JNL files.
- Stage D does not run Abaqus, solver, datacheck, enqueue, or ODB extraction.

## 14. Claim Boundary

Safe: a trained PPO policy generated legal scan-order candidates selected from PPO checkpoint inference only. Not safe: physical improvement or teacher-validation claims.

## 15. Ready For Stage E CAE/INP Generation And Abaqus Teacher Validation

`True`

## 16. Verdict

`PASS_PPO_POLICY_ONLY_BATCH32_READY_FOR_STAGE_E_TEACHER_VALIDATION_HANDOFF`
