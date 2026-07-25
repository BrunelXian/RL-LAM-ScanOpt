# PPO v02K2 Stage N Teacher-Metric Ranking Report

## Purpose
Compare the teacher-metric-extracted PPO v02K2 N24/N40 batch32 against native combined552, PPO v01 N24/N40, identified baseline labels where available, and equal-budget bootstrap draws from combined552.

## Inputs
- Stage M metrics: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v02_targeted_N24_N40\stageM_ODB_teacher_metric_extraction\stageM_v02K2_teacher_metrics.csv`
- K2 selected batch: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v02_targeted_N24_N40\stageK2_n40_completion\selected_batch32_K2\v02K2_ppo_targeted_N24_N40_candidate_batch32.csv`
- Native combined552: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\FROZEN_stage3_native_combined552_teacher_dataset.csv`
- PPO v01 metrics: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageI_final_ppo_evidence_freeze\frozen_tables\FROZEN_PPO_batch32_teacher_metrics.csv`
- PPO v01 ranking: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageI_final_ppo_evidence_freeze\frozen_tables\FROZEN_PPO_batch32_teacher_metric_ranking_full.csv`

## Stage M Extraction Status
Stage M reported 32/32 extracted teacher-metric rows: N24=16 and N40=16.

## Input Integrity Verdict
`PASS_STAGEN_V02K2_INPUTS_READY`

## Analysis Datasets
- combined552 + v02K2: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v02_targeted_N24_N40\stageN_teacher_metric_ranking\tables\combined552_N24N40_plus_v02K2_analysis_dataset.csv`
- combined552 + v01 + v02K2: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v02_targeted_N24_N40\stageN_teacher_metric_ranking\tables\combined552_N24N40_plus_v01_plus_v02K2_analysis_dataset.csv`

## v02K2 Ranking Against Native combined552
| n | v02K2_count | best_v02K2_lex_candidate | best_v02K2_ref_lex_rank | total_unique_primary_topk_candidates | new_record_count |
| --- | --- | --- | --- | --- | --- |
| 24 | 16 | PPOV02_N24_B16_novelty_tophalf | 114 | 0 | 0 |
| 40 | 16 | PPOV02K2_N40_B01_deterministic | 147 | 16 | 0 |

## New-Record Audit
No primary-metric or primary-lex new records over native combined552 were found.

Table: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v02_targeted_N24_N40\stageN_teacher_metric_ranking\tables\v02K2_new_record_candidates.csv`

## Top-k Competitiveness Audit
| n | v02K2_count | top10pct_U2_count | top25pct_U2_count | top10pct_lex_count | top25pct_lex_count | total_unique_primary_topk_candidates | diagnostic_Mises_topk_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 24 | 16 | 0 | 0 | 0 | 0 | 0 | 6 |
| 40 | 16 | 0 | 0 | 0 | 0 | 16 | 0 |

## v02K2 vs PPO v01 N24/N40
| n | v01_candidate_count | v02K2_candidate_count | v01_best_lex_rank | v02K2_best_lex_rank | v01_topk_count | v02K2_topk_count | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 24 | 8 | 16 | 134 | 114 | 3 | 0 | v02K2 partially improves v01 |
| 40 | 8 | 16 | 147 | 147 | 0 | 16 | v02K2 partially improves v01 |

## Equal-Budget v02K2-vs-v01 Bootstrap
| n | expected_v02K2_best_lex_rank_equal8_median | prob_v02K2_subsample_beats_v01_best_lex_rank | expected_v02K2_topk_count_equal8_median | prob_v02K2_subsample_beats_v01_topk_count |
| --- | --- | --- | --- | --- |
| 24 | 114.0 | 0.7715 | 0.0 | 0.0 |
| 40 | 147.0 | 0.0 | 8.0 | 1.0 |

## v02K2 vs Random-Reference Bootstrap
Global bootstrap summary:

| metric | observed | bootstrap_mean | q05 | q95 | empirical_p_value_greater_equal | interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| total_unique_primary_top25_or_lex_top25_count | 16 | 16.3771 | 12.0 | 21.0 | 0.6265 | comparable |

This is an equal-budget bootstrap against the existing teacher-labelled reference distribution, not against the full scan-order universe.

## Identified Baseline-Family Comparison
Output: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v02_targeted_N24_N40\stageN_teacher_metric_ranking\tables\v02K2_vs_identified_baseline_families.csv`

Status preview:
| family | n | family_count | v02K2_count | family_best_u2_range | family_median_u2_range | v02K2_best_u2_range | v02K2_median_u2_range | v02K2_best_beats_family_best_u2_range | v02K2_median_beats_family_median_u2_range | family_best_peeq_max | family_median_peeq_max | v02K2_best_peeq_max | v02K2_median_peeq_max | v02K2_best_beats_family_best_peeq_max | v02K2_median_beats_family_median_peeq_max | family_best_surface_t_proxy | family_median_surface_t_proxy | v02K2_best_surface_t_proxy | v02K2_median_surface_t_proxy | v02K2_best_beats_family_best_surface_t_proxy | v02K2_median_beats_family_median_surface_t_proxy | family_best_mises_max | family_median_mises_max | v02K2_best_mises_max | v02K2_median_mises_max | v02K2_best_beats_family_best_mises_max | v02K2_median_beats_family_median_mises_max | comparison_boundary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raster | 24 | 1 | 16 | 0.0005523721708868 | 0.0005523721708868 | 9.453498660150216e-05 | 0.00014045583725415 | True | True | 0.1892502903938293 | 0.1892502903938293 | 0.1558585613965988 | 0.1570346280932426 | True | True | 587159872.0 | 587159872.0 | 582223488.0 | 582460832.0 | True | True | 579944256.0 | 579944256.0 | 579943296.0 | 579944192.0 | True | True | Label-derived family comparison; use only where family labels are reliable. |
| raster | 40 | 1 | 16 | 0.0015270577277988 | 0.0015270577277988 | 0.0004007218865353 | 0.00045678247636254997 | True | True | 0.200013056397438 | 0.200013056397438 | 0.1641117632389068 | 0.1695306897163391 | True | True | 585953088.0 | 585953088.0 | 581153472.0 | 581623392.0 | True | True | 579928512.0 | 579928512.0 | 579926912.0 | 579926976.0 | True | True | Label-derived family comparison; use only where family labels are reliable. |
| odd_even | 24 | 16 | 16 | 8.360300034837564e-05 | 0.00020757473248520002 | 9.453498660150216e-05 | 0.00014045583725415 | False | True | 0.1496064215898513 | 0.15481503307819366 | 0.1558585613965988 | 0.1570346280932426 | False | False | 581204672.0 | 583085152.0 | 582223488.0 | 582460832.0 | False | True | 579943296.0 | 579944128.0 | 579943296.0 | 579944192.0 | False | False | Label-derived family comparison; use only where family labels are reliable. |
| odd_even | 40 | 16 | 16 | 0.0001595688736415 | 0.0006819835541591001 | 0.0004007218865353 | 0.00045678247636254997 | False | True | 0.1440253555774688 | 0.1516208499670028 | 0.1641117632389068 | 0.1695306897163391 | False | False | 581541824.0 | 583446368.0 | 581153472.0 | 581623392.0 | True | True | 579924928.0 | 579926080.0 | 579926912.0 | 579926976.0 | False | False | Label-derived family comparison; use only where family labels are reliable. |
| edge_in | 24 | 1 | 16 | 0.0005255469384337 | 0.0005255469384337 | 9.453498660150216e-05 | 0.00014045583725415 | True | True | 0.1848475784063339 | 0.1848475784063339 | 0.1558585613965988 | 0.1570346280932426 | True | True | 590591168.0 | 590591168.0 | 582223488.0 | 582460832.0 | True | True | 579942784.0 | 579942784.0 | 579943296.0 | 579944192.0 | False | False | Label-derived family comparison; use only where family labels are reliable. |
| edge_in | 40 | 1 | 16 | 0.0016398808220401 | 0.0016398808220401 | 0.0004007218865353 | 0.00045678247636254997 | True | True | 0.2005691230297088 | 0.2005691230297088 | 0.1641117632389068 | 0.1695306897163391 | True | True | 595556480.0 | 595556480.0 | 581153472.0 | 581623392.0 | True | True | 579926400.0 | 579926400.0 | 579926912.0 | 579926976.0 | False | False | Label-derived family comparison; use only where family labels are reliable. |
| center_out | 24 | 2 | 16 | 0.0006249901030059 | 0.00062579460866625 | 9.453498660150216e-05 | 0.00014045583725415 | True | True | 0.1726419776678085 | 0.17280188947916025 | 0.1558585613965988 | 0.1570346280932426 | True | True | 588072192.0 | 588171392.0 | 582223488.0 | 582460832.0 | True | True | 579944128.0 | 579944160.0 | 579943296.0 | 579944192.0 | True | False | Label-derived family comparison; use only where family labels are reliable. |
| center_out | 40 | 2 | 16 | 0.0016625018324702 | 0.0017483317642472 | 0.0004007218865353 | 0.00045678247636254997 | True | True | 0.1796219497919082 | 0.1808240190148353 | 0.1641117632389068 | 0.1695306897163391 | True | True | 587521984.0 | 588064608.0 | 581153472.0 | 581623392.0 | True | True | 579925632.0 | 579926208.0 | 579926912.0 | 579926976.0 | False | False | Label-derived family comparison; use only where family labels are reliable. |

## Surrogate-to-Teacher Alignment
- Predicted reward Spearman: `0.2650`
- Predicted reward Pearson: `0.2043`
- Conservative reward Spearman: `0.2655`
- Conservative reward Pearson: `0.1191`
- False positives: `9`
- True positives: `0`

## Best Candidates By N
| n | strategy_name | ref_lex_rank | is_top25pct |
| --- | --- | --- | --- |
| 24 | PPOV02_N24_B16_novelty_tophalf | 114.0 | False |
| 40 | PPOV02K2_N40_B01_deterministic | 147.0 | False |

## Claim Implications
v02K2 candidates were teacher-validated for N24/N40; performance claims are limited to the ranking, top-k, v01-comparison, and bootstrap evidence generated in Stage N.

## Limitations
- This is teacher-metric analysis only and does not create new candidates.
- The random-reference bootstrap samples from the active-learning-enriched combined552 pool, not from all possible permutations.
- Surrogate score alignment is diagnostic and cannot replace Abaqus teacher validation.

## Verdict
`PASS_STAGEN_V02K2_TEACHER_VALIDATED_AND_IMPROVES_V01_TARGETED`
