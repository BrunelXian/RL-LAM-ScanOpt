# PPO Final Pool 320 Evidence Index

## Evidence Chain Overview

- PPO v01: 32 teacher-metric-extracted cases.
- PPO v02K2: 32 teacher-metric-extracted targeted N24/N40 cases.
- PPO v03: 32 teacher-metric-extracted lex-primary N24/N40 cases.
- Final expansion: 224 teacher-metric-extracted cases.
- Stage W ranking: final 320-case comparison against native combined552.
- Stage X freeze: final frozen evidence, hashes, manuscript tables, and claim boundary.

## Frozen Tables

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

## Manuscript Tables

- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_pool_320_evidence_freeze\manuscript_tables\PPO_final_pool_320_composition_for_manuscript.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_pool_320_evidence_freeze\manuscript_tables\PPO_final_pool_320_main_results_for_manuscript.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_pool_320_evidence_freeze\manuscript_tables\PPO_final_pool_320_safe_claims_for_manuscript.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_pool_320_evidence_freeze\manuscript_tables\PPO_final_pool_320_claim_decision_for_manuscript.csv`

## Hash Table

- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_pool_320_evidence_freeze\hashes\FROZEN_PPO_final_pool_320_file_hashes.csv`

## Reports and Claim Boundaries

- Stage W report: `E:\Projects\RL-LAM-ScanOpt\docs\stage3_ppo_final_pool_320_analysis\PPO_FINAL_POOL_320_STAGEW_RANKING_AND_COMPARISON_REPORT.md`
- Stage W claim boundary: `E:\Projects\RL-LAM-ScanOpt\docs\stage3_ppo_final_pool_320_analysis\PPO_FINAL_POOL_320_STAGEW_CLAIM_BOUNDARY.md`
- Stage X report: `E:\Projects\RL-LAM-ScanOpt\docs\stage3_ppo_final_pool_320_evidence_freeze\PPO_FINAL_POOL_320_EVIDENCE_FREEZE_REPORT.md`
- Stage X final claim boundary: `E:\Projects\RL-LAM-ScanOpt\docs\stage3_ppo_final_pool_320_evidence_freeze\PPO_FINAL_POOL_320_FINAL_CLAIM_BOUNDARY.md`

## Final Verdict

`PASS_STAGEX_PPO_FINAL_POOL_320_EVIDENCE_FREEZE_BOUNDED_NO_NEW_RECORDS`

## Exact Safe Claim Paragraph

A 320-case PPO-generated scan-order pool was independently teacher-metric extracted using Abaqus. The pool demonstrates legal, executable and teacher-evaluable policy-generated scan orders with bounded small-N competitiveness and SurfaceT-related signals, but it produced no new combined552 records and did not outperform the mature surrogate-assisted optimisation reference.
