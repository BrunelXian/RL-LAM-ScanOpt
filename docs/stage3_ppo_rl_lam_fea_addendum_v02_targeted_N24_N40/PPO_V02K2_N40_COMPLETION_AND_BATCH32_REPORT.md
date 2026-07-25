# PPO v02K2 N40 Completion And Batch32 Report

## Scope

Stage K2 continued N40 PPO v02 training in the surrogate environment, refreshed only N40 candidates, and merged them with frozen Stage K N24 candidates. No Abaqus/ODB/solver/CAE/INP/JNL occurred.

## N40 Training

`WARNING_STAGEK2_N40_COMPLETION_SINGLE_SEED_ONLY`

## N40 Internal Evaluation

Rollouts: 501; unique orders: 501; mean reward: 0.584360873921502; max reward: 0.6336436014802017.

## Refreshed N40 Generation

Rollout pool: 3001; unique orders: 2998; selected: 16.

## Merged K2 Batch32

Path: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v02_targeted_N24_N40\stageK2_n40_completion\selected_batch32_K2\v02K2_ppo_targeted_N24_N40_candidate_batch32.csv`

Counts: N24=16 retained from Stage K; N40=16 refreshed in K2.

## Audits

Legality audit: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v02_targeted_N24_N40\stageK2_n40_completion\tables\stageK2_candidate_legality_audit.csv`
Novelty audit: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v02_targeted_N24_N40\stageK2_n40_completion\tables\stageK2_candidate_novelty_audit.csv`
Score summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v02_targeted_N24_N40\stageK2_n40_completion\tables\stageK2_candidate_score_summary_by_N.csv`

## Verdict

`PASS_PPO_V02K2_BATCH32_READY_FOR_CAE_INP_HANDOFF`
