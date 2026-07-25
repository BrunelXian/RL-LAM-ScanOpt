# PPO Final Expansion 224 Candidate Generation Report

## Purpose

Stage T creates a fixed-budget 224-case PPO-generated expansion set so that future teacher validation can bring the cumulative PPO pool from 96 cases to 320 cases.

## Current and Target Pool

- Current PPO teacher-validated pool: 96
- Target PPO teacher-validated pool: 320
- Remaining candidate target: 224

## Why Fixed-Budget Expansion

Stage S showed that v03 did not improve U2/lex physical performance and had weak surrogate-to-teacher alignment. This stage therefore stops open-ended reward redesign and uses existing PPO checkpoint inference to create a broad, auditable evidence pool.

## Rollout Pool Summary

- Rollout pool path: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_expansion_224_to_320\rollout_pools\ppo_final_expansion_rollout_pool.csv`
- Total pool rows: 5435
- Total unique order hashes: 5435

## Selected Candidate Summary

- Selected total: 224
- Counts by N: {12: 32, 16: 32, 24: 80, 40: 80}
- Batch count: 7

## Bucket Distribution

{
  "quality": 78,
  "diversity": 56,
  "efficiency": 44,
  "baseline_proximity": 24,
  "novelty": 22
}

## Legality Audit

- All legal: True
- Candidate source OK: True
- All unvalidated: True

## Novelty Audit

- Duplicate within selected: 0
- Duplicate vs combined552/v01/v02K2/v03 references: 0

## Industrial-Efficiency Proxy Summary

These are sequence descriptors only, not teacher metrics or physically validated efficiency measures.

```text
   mean_abs_jump                  max_abs_jump            total_travel_proxy               adjacent_fraction              path_complexity_score                
            mean     min      max         mean  min   max               mean    min    max              mean  min     max                  mean     min     max
n                                                                                                                                                              
12        3.5511  3.3636   4.0909       9.2500  8.0  10.0            39.0625   37.0   45.0            0.0455  0.0  0.1818                0.3481  0.3138  0.4017
16        3.8250  3.5333   4.8667      12.8125  8.0  14.0            57.3750   53.0   73.0            0.0333  0.0  0.1333                0.2844  0.2439  0.3699
24        4.2380  2.1739   7.0000      12.5375  4.0  23.0            97.4750   50.0  161.0            0.0766  0.0  0.4783                0.1952  0.0925  0.3487
40        6.8019  3.1282  12.6923      21.0375  6.0  38.0           265.2750  122.0  495.0            0.0269  0.0  0.1282                0.1879  0.0800  0.3752
```

## Handoff Preview

Handoff preview: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_expansion_224_to_320\handoff_preview\PPO_FINAL_EXPANSION_224_CAE_INP_HANDOFF_PREVIEW.csv`

## Claim Boundary

Stage T supports only candidate-generation and handoff-readiness claims. It does not support physical improvement, teacher validation, or industrial efficiency claims.

## Verdict

PASS_PPO_FINAL_EXPANSION_224_READY_FOR_BATCHED_CAE_INP_HANDOFF
