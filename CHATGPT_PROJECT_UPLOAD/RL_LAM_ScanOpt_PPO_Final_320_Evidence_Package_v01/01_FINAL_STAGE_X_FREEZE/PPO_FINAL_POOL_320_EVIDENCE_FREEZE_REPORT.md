# PPO Final Pool 320 Evidence Freeze Report

## Purpose

Stage X freezes the final 320-case PPO teacher-metric evidence package and creates manuscript-facing summary materials.

## Final Evidence Chain

PPO v01, PPO v02K2, PPO v03 and the final expansion together form a 320-case teacher-metric-extracted PPO pool.

## Final PPO Pool Composition

- Total: 320
- N12: 40
- N16: 40
- N24: 120
- N40: 120
- v01: 32
- v02K2: 32
- v03: 32
- final expansion: 224

## Input Integrity

Stage X preflight passed and confirmed Stage W verdict `PASS_STAGEW_PPO_FINAL_POOL_320_BOUNDED_NO_NEW_RECORDS`.

## Frozen Table List

- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_pool_320_evidence_freeze\frozen_tables\FROZEN_PPO_final_pool_320_teacher_metrics.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_pool_320_evidence_freeze\frozen_tables\FROZEN_combined552_plus_PPO_final_pool_320_analysis_dataset.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_pool_320_evidence_freeze\frozen_tables\FROZEN_PPO_final_pool_320_teacher_metric_ranking_full.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_pool_320_evidence_freeze\frozen_tables\FROZEN_PPO_final_pool_320_claim_decision_table.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_pool_320_evidence_freeze\frozen_tables\FROZEN_ppo_final_pool_320_topk_competitive_candidates.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_pool_320_evidence_freeze\frozen_tables\FROZEN_ppo_final_pool_320_topk_summary_by_N.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_pool_320_evidence_freeze\frozen_tables\FROZEN_ppo_final_pool_320_topk_summary_by_version.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_pool_320_evidence_freeze\frozen_tables\FROZEN_ppo_final_pool_320_new_record_candidates.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_pool_320_evidence_freeze\frozen_tables\FROZEN_ppo_final_pool_320_vs_bootstrap_random_reference_by_N.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_pool_320_evidence_freeze\frozen_tables\FROZEN_ppo_final_pool_320_vs_bootstrap_random_reference_global.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_pool_320_evidence_freeze\frozen_tables\FROZEN_ppo_final_pool_320_vs_identified_baseline_families.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_pool_320_evidence_freeze\frozen_tables\FROZEN_ppo_final_pool_320_identified_baseline_family_inventory.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_pool_320_evidence_freeze\frozen_tables\FROZEN_ppo_final_pool_320_industrial_efficiency_proxy_summary.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_pool_320_evidence_freeze\frozen_tables\FROZEN_ppo_final_pool_320_efficiency_proxy_vs_teacher_metrics.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_pool_320_evidence_freeze\frozen_tables\FROZEN_ppo_final_pool_320_version_summary.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_pool_320_evidence_freeze\frozen_tables\FROZEN_ppo_final_pool_320_best_candidates_by_N.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_pool_320_evidence_freeze\frozen_tables\FROZEN_final_expansion_vs_prior_ppo_by_N.csv`

## Hash Table

`E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_pool_320_evidence_freeze\hashes\FROZEN_PPO_final_pool_320_file_hashes.csv`

## Main Stage W Results

- New records vs combined552: 0
- Primary top25-any count: 106
- Equal-budget bootstrap primary top25-any mean: 163.106
- Bootstrap q05/q95: 154 / 173
- Bootstrap interpretation: weak
- SurfaceT top10 enrichment was observed but is not U2/lex dominance.

## New-Record Audit

No PPO candidate produced a new combined552 record.

## Top-k Competitiveness

Top-k evidence is bounded. The strongest lexicographic PPO ranks are in N12 and N16.

## Bootstrap Interpretation

The 320-case PPO pool is weak relative to equal-budget random-reference draws for overall primary top25 enrichment.

## Baseline-Family Comparison

PPO can be compared to identified conventional baseline labels where available, but label-derived baseline comparisons should not be overclaimed.

## Industrial-Efficiency Proxy Caveat

Industrial-efficiency fields are sequence descriptors only and are not physically validated efficiency measurements.

## Claim Boundary

See `E:\Projects\RL-LAM-ScanOpt\docs\stage3_ppo_final_pool_320_evidence_freeze\PPO_FINAL_POOL_320_FINAL_CLAIM_BOUNDARY.md`.

## Manuscript Integration Summary

Use PPO as a large-scale teacher-metric-extracted policy-generation addendum, not as the strongest optimiser.

## Final Verdict

PASS_STAGEX_PPO_FINAL_POOL_320_EVIDENCE_FREEZE_BOUNDED_NO_NEW_RECORDS
