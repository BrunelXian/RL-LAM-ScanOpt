# PPO Final Pool 320 Stage W Ranking and Comparison Report

## Purpose

Stage W integrates PPO v01, PPO v02K2, PPO v03, and the final 224-case PPO expansion into a 320-case teacher-metric-extracted PPO evidence pool, then compares it against native combined552 and related baselines.

## Evidence Pool Composition

- PPO v01: 32
- PPO v02K2: 32
- PPO v03: 32
- Final expansion: 224
- Final PPO pool: 320
- By N: {'12': 40, '16': 40, '24': 120, '40': 120}

## Stage V Extraction Status

Stage V final expansion teacher metrics were read from `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_expansion_224_to_320\stageV_ODB_teacher_metric_extraction\stageV_ppo_final_expansion_224_ODB_metrics.csv`. Stage V warning/nonfatal status is preserved in the input audit; Stage W does not open ODB files or extract metrics.

## Input Integrity Verdict

PASS_STAGEW_FINAL_PPO_POOL_INPUTS_READY

## Final PPO Pool Dataset

- PPO final pool: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_pool_320_analysis\tables\ppo_final_pool_320_teacher_metrics.csv`
- Combined552 + PPO analysis dataset: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_pool_320_analysis\tables\combined552_plus_ppo_final_pool_320_analysis_dataset.csv`
- Full ranking table: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_pool_320_analysis\tables\ppo_final_pool_320_teacher_metric_ranking_full.csv`

## Comparison Against Combined552

PPO candidates were ranked within each N against the native combined552 teacher-labelled reference. Metrics are smaller-is-better. SurfaceT is compared in MPa units after explicit mapping.

## New-Record Audit

- Primary new-record rows: 0
- New-record table: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_pool_320_analysis\tables\ppo_final_pool_320_new_record_candidates.csv`

## Top-k Competitiveness Audit

- Unique PPO candidates in at least one primary top25 region: 106
- Top-k by N table: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_pool_320_analysis\tables\ppo_final_pool_320_topk_summary_by_N.csv`
- Top-k by version table: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_pool_320_analysis\tables\ppo_final_pool_320_topk_summary_by_version.csv`

## Final Expansion vs Prior PPO Stages

`E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_pool_320_analysis\tables\final_expansion_vs_prior_ppo_by_N.csv`

```text
 N  v01_count  v01_best_lex_rank  v01_best_U2_rank  v01_best_PEEQ_rank  v01_best_SurfaceT_rank  v01_top25_any_primary_count  v02K2_count  v02K2_best_lex_rank  v02K2_top25_any_primary_count  v03_count  v03_best_lex_rank  v03_top25_any_primary_count  final_expansion_count  final_expansion_best_lex_rank  final_expansion_best_U2_rank  final_expansion_best_PEEQ_rank  final_expansion_best_SurfaceT_rank  final_expansion_top25_any_primary_count  final_expansion_improves_prior_best_lex  final_expansion_improves_prior_topk_count                                                  interpretation  v02K2_best_U2_rank  v02K2_best_PEEQ_rank  v02K2_best_SurfaceT_rank  v03_best_U2_rank  v03_best_PEEQ_rank  v03_best_SurfaceT_rank
12          8                  6                 6                  17                      46                            5            0                  NaN                              0          0                NaN                            0                     32                              7                             7                              17                                  36                                       13                                    False                                       True final expansion improves prior PPO top-k count by larger budget                 NaN                   NaN                       NaN               NaN                 NaN                     NaN
16          8                  2                 2                  50                      68                            4            0                  NaN                              0          0                NaN                            0                     32                              2                             2                              13                                  53                                       10                                    False                                       True final expansion improves prior PPO top-k count by larger budget                 NaN                   NaN                       NaN               NaN                 NaN                     NaN
24          8                134               134                   9                     155                            3           16                114.0                              0         16              139.0                            5                     80                            123                           123                              11                                  13                                       21                                    False                                       True final expansion improves prior PPO top-k count by larger budget               114.0                  70.0                      83.0             139.0               172.0                    10.0
40          8                147               147                  65                      95                            0           16                147.0                             16         16              165.0                            0                     80                            147                           147                              10                                   2                                       29                                    False                                       True final expansion improves prior PPO top-k count by larger budget               147.0                 168.0                       2.0             165.0               100.0                    97.0
```

## Random-Reference Bootstrap

Equal-budget bootstrap against existing teacher-labelled reference distribution:

observed=106, bootstrap mean=163.11, q05=154.00, q95=173.00, interpretation=weak

## Conventional Baseline-Family Comparison

Baseline-family comparison is label-derived from combined552 where labels are available. It should not be overclaimed when family labels are sparse or broad.

## Industrial-Efficiency Proxy Analysis

Industrial-efficiency descriptors are sequence proxies only. They are not physical teacher metrics and do not establish experimentally validated industrial efficiency.

## Claim-Decision Table

`E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_pool_320_analysis\tables\ppo_final_pool_320_claim_decision_table.csv`

## Main Scientific Interpretation

The final 320-case pool is a large-scale teacher-metric-extracted evidence base for surrogate-trained policy-gradient scan-order generation. Final physical claims must follow the new-record, top-k, bootstrap, and baseline-family tables rather than surrogate scores.

## Limitations

- The random-reference bootstrap samples from the existing teacher-labelled combined552 distribution, not the full scan-order universe.
- Baseline-family extraction depends on available labels.
- Industrial-efficiency fields are proxies only.
- Stage W performs analysis only; it does not generate new physical evidence.

## Recommended Manuscript Wording

Use bounded wording: "A 320-case PPO-generated scan-order pool was teacher-metric extracted and ranked against the native combined552 reference. The PPO evidence is reported by new-record, top-k, bootstrap, and baseline-family audits."

## Verdict

PASS_STAGEW_PPO_FINAL_POOL_320_BOUNDED_NO_NEW_RECORDS
