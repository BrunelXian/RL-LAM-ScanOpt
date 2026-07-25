# PPO v02 Targeted Candidate Generation Report

## Scope

Generated N24/N40 PPO v02 candidates from fixed-N PPO checkpoints only. No CAE/INP/JNL, Abaqus, ODB, solver, datacheck, enqueue, surrogate retraining, or candidate mutation occurred in candidate generation.

## Rollout Pool

Feasible initial pool: 512 stochastic attempts plus one deterministic attempt per N/checkpoint. The requested 2000-attempt pool was deferred because it exceeded local runtime limits.

| n | rollout_rows | unique_orders |
| --- | --- | --- |
| 24 | 513 | 430 |
| 40 | 513 | 513 |

## Selected Batch

N24=16, N40=16, total=32.

## Audits

- Legality audit: `PASS`
- Novelty audit: `PASS`

## Selected CSV

`E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v02_targeted_N24_N40\candidate_generation_v02\selected_batch32\v02_ppo_targeted_N24_N40_candidate_batch32.csv`

## Verdict

`PASS_PPO_V02_TARGETED_BATCH32_READY_FOR_CAE_INP_HANDOFF`
