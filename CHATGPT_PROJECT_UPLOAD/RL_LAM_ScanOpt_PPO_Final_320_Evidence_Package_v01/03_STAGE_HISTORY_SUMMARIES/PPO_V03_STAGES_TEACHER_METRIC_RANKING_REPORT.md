# PPO v03 Stage S Teacher-Metric Ranking Report

## Purpose
Rank PPO v03 teacher metrics against native combined552, PPO v01, PPO v02K2, conventional baseline families, and the cumulative PPO teacher-validated pool.

## Inputs
- Stage R metrics: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40\stageR_ODB_teacher_metric_extraction\stageR_v03_teacher_metrics.csv`
- v03 selected batch: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40\candidate_generation_v03\selected_batch32\v03_ppo_lex_primary_N24_N40_candidate_batch32.csv`
- combined552: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\FROZEN_stage3_native_combined552_teacher_dataset.csv`
- v01 metrics: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageI_final_ppo_evidence_freeze\frozen_tables\FROZEN_PPO_batch32_teacher_metrics.csv`
- v02K2 metrics: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v02_targeted_N24_N40\stageM_ODB_teacher_metric_extraction\stageM_v02K2_teacher_metrics.csv`

## Stage R Extraction Status
Stage R extracted 32/32 PPO v03 teacher-metric rows: N24=16 and N40=16. No failed cases are included.

## v03 Partial-Training Caveat
The Stage R rows preserve the caveat that N24 used a 100000-step interrupted checkpoint and N40 used 61440 timesteps. This limits claim strength.

## Input Integrity Verdict
`PASS_STAGES_V03_INPUTS_READY`

## Analysis Datasets
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40\stageS_teacher_metric_ranking\tables\combined552_N24N40_plus_v03_analysis_dataset.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40\stageS_teacher_metric_ranking\tables\combined552_N24N40_plus_ppo_v01_v02K2_v03_analysis_dataset.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40\stageS_teacher_metric_ranking\tables\ppo_teacher_validated_pool_v01_v02K2_v03_96cases.csv`

## v03 Ranking Against Native combined552
| n | v03_count | best_v03_lex_candidate | best_v03_ref_lex_rank | total_unique_primary_topk_candidates | new_record_count |
| --- | --- | --- | --- | --- | --- |
| 24 | 16 | PPOV03_N24_B05_top_v03_score | 139 | 5 | 0 |
| 40 | 16 | PPOV03_N40_B02_top_v03_score | 165 | 0 | 0 |

## New-Record Audit
New-record rows: `0`; primary new-record rows: `0`; diagnostic rows: `0`.

Table: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40\stageS_teacher_metric_ranking\tables\v03_new_record_candidates.csv`

## Top-k Competitiveness Audit
| n | v03_count | top10pct_U2_count | top25pct_U2_count | top10pct_lex_count | top25pct_lex_count | total_unique_primary_topk_candidates | diagnostic_Mises_topk_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 24 | 16 | 0 | 0 | 0 | 0 | 5 | 0 |
| 40 | 16 | 0 | 0 | 0 | 0 | 0 | 6 |

## v03 vs v01/v02K2
| n | v01_best_lex_rank | v02K2_best_lex_rank | v03_best_lex_rank | v01_topk_count | v02K2_topk_count | v03_topk_count | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 24 | 134 | 114 | 139 | 3 | 0 | 5 | v03 improves prior PPO on at least one primary targeted criterion |
| 40 | 147 | 147 | 165 | 0 | 16 | 0 | v03 is weak under primary targeted criteria |

## Equal-Budget v03-vs-Prior PPO Bootstrap
| n | prob_v03_equal8_beats_v01_best_lex | prob_v03_equal8_beats_v02K2_best_lex | prob_v03_equal8_beats_v01_topk | prob_v03_equal8_beats_v02K2_topk | v03_direct16_beats_v02K2_best_lex | v03_direct16_beats_v02K2_topk |
| --- | --- | --- | --- | --- | --- | --- |
| 24 | 0.0 | 0.0 | 0.1385 | 0.9875 | False | True |
| 40 | 0.0 | 0.0 | 0.0 | 0.0 | False | False |

## v03 vs Random-Reference Bootstrap
| metric | observed | bootstrap_mean | q05 | q95 | empirical_p_value_greater_equal | interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| total_unique_primary_top25_or_lex_top25_count | 5 | 16.434 | 12.0 | 21.0 | 1.0 | weak |

This bootstrap samples the existing teacher-labelled reference distribution, not the full scan-order universe.

## Conventional Baseline-Family Comparison
Inventory: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40\stageS_teacher_metric_ranking\tables\v03_identified_baseline_family_inventory.csv`. v03 comparison: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40\stageS_teacher_metric_ranking\tables\v03_vs_identified_baseline_families.csv`. Cumulative PPO comparison: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40\stageS_teacher_metric_ranking\tables\cumulative_ppo_pool_vs_identified_baseline_families.csv`.

Preview:
| family | n | family_count | v03_count | family_best_u2_range | family_median_u2_range | v03_best_u2_range | v03_median_u2_range | v03_best_beats_family_best_u2_range | v03_median_beats_family_median_u2_range | family_best_peeq_max | family_median_peeq_max | v03_best_peeq_max | v03_median_peeq_max | v03_best_beats_family_best_peeq_max | v03_median_beats_family_median_peeq_max | family_best_surface_t_proxy | family_median_surface_t_proxy | v03_best_surface_t_proxy | v03_median_surface_t_proxy | v03_best_beats_family_best_surface_t_proxy | v03_median_beats_family_median_surface_t_proxy | family_best_mises_max | family_median_mises_max | v03_best_mises_max | v03_median_mises_max | v03_best_beats_family_best_mises_max | v03_median_beats_family_median_mises_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raster | 24 | 1 | 16 | 0.0005523721708868 | 0.0005523721708868 | 0.0001702423680853 | 0.00027177957622365 | True | True | 0.1892502903938293 | 0.1892502903938293 | 0.1666601449251175 | 0.17386179417371744 | True | True | 587159872.0 | 587159872.0 | 581186176.0 | 582332512.0 | True | True | 579944256.0 | 579944256.0 | 579943936.0 | 579944832.0 | True | False |
| raster | 40 | 1 | 16 | 0.0015270577277988 | 0.0015270577277988 | 0.0007193772387381 | 0.0008975243688382 | True | True | 0.200013056397438 | 0.200013056397438 | 0.1566403955221176 | 0.16961321979761118 | True | True | 585953088.0 | 585953088.0 | 582840064.0 | 584022432.0 | True | True | 579928512.0 | 579928512.0 | 579925824.0 | 579926496.0 | True | True |
| odd_even | 24 | 16 | 16 | 8.360300034837564e-05 | 0.00020757473248520002 | 0.0001702423680853 | 0.00027177957622365 | False | False | 0.1496064215898513 | 0.15481503307819366 | 0.1666601449251175 | 0.17386179417371744 | False | False | 581204672.0 | 583085152.0 | 581186176.0 | 582332512.0 | True | True | 579943296.0 | 579944128.0 | 579943936.0 | 579944832.0 | False | False |
| odd_even | 40 | 16 | 16 | 0.0001595688736415 | 0.0006819835541591001 | 0.0007193772387381 | 0.0008975243688382 | False | False | 0.1440253555774688 | 0.1516208499670028 | 0.1566403955221176 | 0.16961321979761118 | False | False | 581541824.0 | 583446368.0 | 582840064.0 | 584022432.0 | False | False | 579924928.0 | 579926080.0 | 579925824.0 | 579926496.0 | False | False |
| edge_in | 24 | 1 | 16 | 0.0005255469384337 | 0.0005255469384337 | 0.0001702423680853 | 0.00027177957622365 | True | True | 0.1848475784063339 | 0.1848475784063339 | 0.1666601449251175 | 0.17386179417371744 | True | True | 590591168.0 | 590591168.0 | 581186176.0 | 582332512.0 | True | True | 579942784.0 | 579942784.0 | 579943936.0 | 579944832.0 | False | False |
| edge_in | 40 | 1 | 16 | 0.0016398808220401 | 0.0016398808220401 | 0.0007193772387381 | 0.0008975243688382 | True | True | 0.2005691230297088 | 0.2005691230297088 | 0.1566403955221176 | 0.16961321979761118 | True | True | 595556480.0 | 595556480.0 | 582840064.0 | 584022432.0 | True | True | 579926400.0 | 579926400.0 | 579925824.0 | 579926496.0 | True | False |
| center_out | 24 | 2 | 16 | 0.0006249901030059 | 0.00062579460866625 | 0.0001702423680853 | 0.00027177957622365 | True | True | 0.1726419776678085 | 0.17280188947916025 | 0.1666601449251175 | 0.17386179417371744 | True | False | 588072192.0 | 588171392.0 | 581186176.0 | 582332512.0 | True | True | 579944128.0 | 579944160.0 | 579943936.0 | 579944832.0 | True | False |
| center_out | 40 | 2 | 16 | 0.0016625018324702 | 0.0017483317642472 | 0.0007193772387381 | 0.0008975243688382 | True | True | 0.1796219497919082 | 0.1808240190148353 | 0.1566403955221176 | 0.16961321979761118 | True | True | 587521984.0 | 588064608.0 | 582840064.0 | 584022432.0 | True | True | 579925632.0 | 579926208.0 | 579925824.0 | 579926496.0 | False | False |

## Surrogate-to-Teacher Alignment
- Final v03 score Spearman: `-0.2939`
- Final v03 score Pearson: `0.0398`
- Lex-primary score Spearman: `-0.2711`
- U2-guarded score Spearman: `-0.1130`
- False positives: `8`
- True positives: `1`
- Teacher SurfaceT-only false positives: `5`

## Best v03 Candidates By N
| n | strategy_name | ref_lex_rank | is_top25pct |
| --- | --- | --- | --- |
| 24 | PPOV03_N24_B05_top_v03_score | 139.0 | False |
| 40 | PPOV03_N40_B02_top_v03_score | 165.0 | False |

## Cumulative PPO Pool 96-Case Summary
| n | total_ppo_count | count_by_ppo_version | primary_topk_count | new_record_count | comparison_against_combined552_best |
| --- | --- | --- | --- | --- | --- |
| 12 | 8 | {"v01": 8} |  |  |  |
| 16 | 8 | {"v01": 8} |  |  |  |
| 24 | 40 | {"v02K2": 16, "v03": 16, "v01": 8} | 8.0 | 0.0 | no combined552 best beat |
| 40 | 40 | {"v02K2": 16, "v03": 16, "v01": 8} | 16.0 | 0.0 | no combined552 best beat |

## Progress Toward 320-Case PPO Target
| stage | teacher_validated_count |
| --- | --- |
| v01 | 32 |
| v02K2 | 32 |
| v03 | 32 |
| current_total | 96 |
| target_total | 320 |
| remaining | 224 |

## Claim Implications
Use v03 claims only where supported by these tables. The cumulative PPO teacher-validated pool now contains 96 cases, but v03 retains the partial-training caveat.

## Limitations
- Stage S is ranking/analysis only.
- Bootstrap is against combined552, an active-learning-enriched reference distribution, not the universe of scan orders.
- Partial v03 training limits claims about policy convergence.

## Verdict
`PASS_STAGES_V03_TEACHER_VALIDATED_AND_IMPROVES_PRIOR_PPO`
