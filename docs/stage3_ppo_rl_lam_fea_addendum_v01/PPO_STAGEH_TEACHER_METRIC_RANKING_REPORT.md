# PPO Stage H Teacher-Metric Ranking Report

## 1. Purpose

Stage H ranks the 32 Abaqus teacher-metric-extracted PPO candidates against the native combined552 Stage 3 reference dataset. This is analysis only.

## 2. Inputs

- Stage G teacher metrics: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageG_odb_teacher_metrics\tables\stageG_ppo_batch32_teacher_metrics.csv`
- Stage D PPO selected batch: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\ppo_candidate_generation\selected_batch32\ppo_policy_only_candidate_batch32.csv`
- Native combined552 reference: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\FROZEN_stage3_native_combined552_teacher_dataset.csv`
- N32 is not used in the primary ranking.

## 3. Stage G Extraction Status

Stage G teacher metrics were available for 32/32 PPO cases. No failed or incomplete cases were present in the Stage H input.

## 4. Input Integrity Audit

Input verdict: `PASS_STAGEH_INPUTS_READY`

- Audit CSV: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageH_teacher_metric_ranking\checks\stageH_input_integrity_audit.csv`
- Audit summary JSON: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageH_teacher_metric_ranking\checks\stageH_input_integrity_audit_summary.json`

## 5. Analysis Dataset

Analysis dataset: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageH_teacher_metric_ranking\tables\combined552_plus_ppo32_analysis_dataset.csv`

The dataset contains 584 rows: 552 native combined reference rows plus 32 PPO batch rows. It is an analysis artifact, not a frozen replacement for Run78 evidence.

## 6. PPO Batch32 Ranking Against Native Combined552

- Full PPO ranking table: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageH_teacher_metric_ranking\tables\ppo_batch32_teacher_metric_ranking_full.csv`
- Summary by N: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageH_teacher_metric_ranking\tables\ppo_batch32_summary_by_N.csv`
- Global summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageH_teacher_metric_ranking\tables\ppo_batch32_global_summary.csv`

Best PPO candidates by lexicographic U2 -> PEEQ -> SurfaceT:

| N | Strategy | Combined lex rank | Ref lex position | U2 range | PEEQ max | SurfaceT proxy |
|---:|---|---:|---:|---:|---:|---:|
| 12 | `PPOV01_N12_B08_stochastic_highreward` | 6 | 6 | 2.51481e-05 | 0.149027 | 5.83066e+08 |
| 16 | `PPOV01_N16_B02_surrogate_top` | 2 | 2 | 2.95212e-05 | 0.153918 | 5.84205e+08 |
| 24 | `PPOV01_N24_B08_stochastic_highreward` | 134 | 134 | 0.000128648 | 0.161806 | 5.85439e+08 |
| 40 | `PPOV01_N40_B07_novelty_tophalf` | 147 | 147 | 0.000474217 | 0.154445 | 5.83565e+08 |

## 7. Metric-Wise Comparison By N

| N | Best PPO U2 / Ref Best | Best PPO PEEQ / Ref Best | Best PPO SurfaceT / Ref Best | Best PPO Mises / Ref Best | Top25 Lex Count |
|---:|---:|---:|---:|---:|---:|
| 12 | 1.143 | 1.041 | 1.001 | 1 | 4 |
| 16 | 1.009 | 1.045 | 1.003 | 1 | 4 |
| 24 | 4.358 | 1.014 | 1.006 | 1 | 0 |
| 40 | 10.36 | 1.068 | 1.003 | 1 | 0 |

Ratios below 1.0 indicate that the best PPO candidate beats the prior combined552 best for that metric. Ratios at or above 1.0 do not support a new-record claim for that metric.

## 8. Lexicographic U2->PEEQ->SurfaceT Comparison

Lexicographic ranking is computed within each N using U2 range first, then PEEQ max, then SurfaceT proxy, all smaller-is-better.

- PPO candidates beating the prior combined552 lexicographic best: 0
- PPO candidates in reference top10pct lex: 5
- PPO candidates in reference top25pct lex: 8

## 9. New-Record Audit

New-record table: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageH_teacher_metric_ranking\tables\ppo_batch32_new_record_candidates.csv`

New-record candidate count: 0

A new-record row is included only when a PPO candidate beats the prior combined552 best in at least one primary metric, Mises diagnostic metric, or lexicographic ranking.

## 10. Top-K Audit

Top-k table: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageH_teacher_metric_ranking\tables\ppo_batch32_topk_candidates.csv`

Top-k candidate count: 12

A top-k row is included when a PPO candidate falls in the reference top10pct or top25pct by a primary metric or lexicographic ranking.

## 11. Surrogate-Vs-Teacher Alignment

- Alignment table: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageH_teacher_metric_ranking\tables\ppo_surrogate_vs_teacher_alignment.csv`
- Alignment summary JSON: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageH_teacher_metric_ranking\tables\ppo_surrogate_vs_teacher_alignment_summary.json`
- Overall Spearman: 0.279
- Overall Pearson: 0.2092
- False-positive count: 1
- True-positive count: 2

## 12. Recovery-Anchor Duplicate Audit

Recovery-anchor audit: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageH_teacher_metric_ranking\tables\ppo_recovery_anchor_duplicate_audit.csv`

- `PPOV01_N12_B02_surrogate_top`: PPO recovered a known teacher-validated strategy; not a new novel PPO discovery
- Metrics match source row within tolerance: True

## 13. Claim Boundary

Safe claims after Stage H are limited to teacher validation and rankings demonstrated by these tables. New-record or top-k claims are allowed only where the tables prove them.

Unsafe claims remain: PPO globally optimised scan order, PPO solved arbitrary-N optimisation, online Abaqus PPO was performed, PPO is experimentally validated, or PPO is first in the world.

## 14. Manuscript Implication

The PPO addendum can report that the PPO-generated batch was independently Abaqus teacher-evaluated and achieved bounded top-k competitiveness in the specific rankings listed in the top-k audit. It must not claim broad superiority.

## 15. Plots

- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageH_teacher_metric_ranking\plots\ppo_vs_combined552_u2_distribution_by_N.png`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageH_teacher_metric_ranking\plots\ppo_vs_combined552_peeq_distribution_by_N.png`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageH_teacher_metric_ranking\plots\ppo_vs_combined552_surfaceT_distribution_by_N.png`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageH_teacher_metric_ranking\plots\ppo_vs_combined552_mises_distribution_by_N.png`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageH_teacher_metric_ranking\plots\ppo_lexicographic_rank_percentile_by_N.png`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageH_teacher_metric_ranking\plots\surrogate_predicted_reward_vs_teacher_reward.png`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageH_teacher_metric_ranking\plots\ppo_candidate_topk_status_summary.png`

## 16. Verdict

`PASS_STAGEH_PPO_BATCH32_TEACHER_VALIDATED_AND_COMPETITIVE`
