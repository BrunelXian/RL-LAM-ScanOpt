# Stage 3 Run 57 - Calibrated N24/N40 Batch64 Teacher Metrics Ingestion and Combined392 Ranking

## 1. Purpose
Run57 ingests the completed Run56 user-selected overnight calibrated N24/N40 batch64 teacher metrics, merges them with Run54/Run53 candidate metadata, recomputes within-N rankings, and builds native combined392 plus combined392_plus_N32 datasets.

## 2. Inputs
- Run56 teacher metrics: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_56_calibrated_N24_N40_batch64_odb_teacher_validation\run56_calibrated_N24_N40_batch64_teacher_metrics.csv`
- Run54 handoff metadata: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_54_run53_calibrated_N24_N40_batch64_handoff_package\stage3_run54_calibrated_N24_N40_batch64_candidate_orders.csv`
- Native combined328 RL-ready dataset: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_52_stricter_constrained_N24_N40_batch32_teacher_metrics_ingestion_and_combined328_ranking\combined328_RL_ready_dataset.csv`
- combined328_plus_N32 RL-ready dataset: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_52_stricter_constrained_N24_N40_batch32_teacher_metrics_ingestion_and_combined328_ranking\combined328_plus_N32_RL_ready_dataset.csv`
- N32 deduplicated legacy-compatible table: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_32a_stage2_n32_legacy_teacher_label_ingestion_for_stage3\n32_legacy_teacher_dataset_dedup_training_332.csv`

## 3. Run56 Teacher-Extraction Status
Run56 was complete for 64/64 teacher-validated cases: N24=32 and N40=32. It contains no N12, N16, or N32 cases. Run56 is native calibrated N24/N40 teacher validation, not N32 teacher validation.

## 4. Input Validation
Verdict: `PASS_RUN57_CALIBRATED_N24_N40_BATCH64_TEACHER_METRICS_64_OF_64_READY`.

Run56 counts: N24=32, N40=32. Native combined328 input counts: N12=36, N16=36, N24=128, N40=128. combined328_plus_N32 input counts: N12=36, N16=36, N24=128, N32=332, N40=128.

## 5. Run56 Enriched Teacher Dataset
Run57 produced an enriched Run56 teacher dataset with handoff names, Run53/Run54 prediction metadata, candidate-source metadata, scan orders, hashes, raw teacher metrics, extraction status, and nonfatal-warning flags.

## 6. Run56 Within-Batch Ranking
Within Run56, lower raw metric values are better for U2, PEEQ, SurfaceT, and Mises. The U2-primary reward uses 0.65 U2, 0.20 PEEQ, 0.10 SurfaceT, and 0.05 Mises rank scores.

Run56 best U2 by N:
| n | best_u2_strategy | best_u2_value |
| --- | --- | --- |
| 24 | S3R54CAL64_N24_B04_n24_u2ret_top | 3.356376646479475e-05 |
| 40 | S3R54CAL64_N40_B12_penalty_repair | 5.201993008085992e-05 |

Run56 best combined reward by N:
| n | best_reward_strategy | best_reward_value |
| --- | --- | --- |
| 24 | S3R54CAL64_N24_B04_n24_u2ret_top | 0.7927419354838711 |
| 40 | S3R54CAL64_N40_B25_peeq_repair | 0.9072580645161292 |

## 7. Native Combined392 Construction
Native combined392 rows: 392. Counts: N12=36, N16=36, N24=160, N40=160. There are no N32 rows in native combined392.

## 8. combined392_plus_N32 Construction
combined392_plus_N32 rows: 724. Counts: N12=36, N16=36, N24=160, N32=332, N40=160. N32 rows preserve the legacy metric semantic warnings from Run32A.

## 9. Run56 vs Combined328 Best Comparison
Run56 was compared against the native combined328 best records for N24 and N40 across U2, PEEQ, SurfaceT, Mises, U2-primary reward, constrained reward, strict penalty guard, and penalty-repair reward.

| n | metric | run56_beats_baseline | baseline_best_strategy | run56_best_strategy | absolute_improvement |
| --- | --- | --- | --- | --- | --- |
| 24 | U2 | False | S3R49SCN_N24_B09_median_guard | S3R54CAL64_N24_B04_n24_u2ret_top | -4.040688509121537e-06 |
| 24 | PEEQ | False | S3R19B28_N24_B07_control_sentinel | S3R54CAL64_N24_B20_median_guard | -0.0024325400590896884 |
| 24 | SurfaceT | False | S3R24L64_N24_B06_model_disagreement | S3R54CAL64_N24_B10_penalty_repair | -96256.0 |
| 24 | Mises | False | N24_A06_edge_in_alternating | S3R54CAL64_N24_B14_penalty_repair | -704.0 |
| 24 | combined_reward | False | S3R44CNS_N24_B16_uncertainty | S3R54CAL64_N24_B04_n24_u2ret_top | -0.09879349758699507 |
| 40 | U2 | True | S3R34N32INF_N40_B08_n40_best_near | S3R54CAL64_N40_B12_penalty_repair | 4.946132321492769e-06 |
| 40 | PEEQ | False | S3R39N2440B60_N40_B10_surrogate_top | S3R54CAL64_N40_B32_diversity | -0.004046991467475919 |
| 40 | SurfaceT | False | S3R19B28_N40_B06_uncertainty_calibration | S3R54CAL64_N40_B14_penalty_repair | -884352.0 |
| 40 | Mises | False | N40_A07_regular_jump_coprime | S3R54CAL64_N40_B21_median_guard | -1856.0 |
| 40 | combined_reward | True | S3R49SCN_N40_B09_median_guard | S3R54CAL64_N40_B25_peeq_repair | 0.023990347980695992 |
| 24 | constrained_u2_reward_balanced | False | S3R44CNS_N24_B16_uncertainty | S3R54CAL64_N24_B04_n24_u2ret_top | -0.16784353568707122 |
| 40 | constrained_u2_reward_balanced | True | S3R49SCN_N40_B09_median_guard | S3R54CAL64_N40_B11_penalty_repair | 0.04688849377698756 |
| 24 | strict_penalty_guard | False | S3R44CNS_N24_B16_uncertainty | S3R54CAL64_N24_B09_penalty_repair | -0.1526162052324106 |
| 40 | strict_penalty_guard | True | S3R49SCN_N40_B09_median_guard | S3R54CAL64_N40_B11_penalty_repair | 0.06203962407924801 |
| 24 | penalty_repair | False | S3R44CNS_N24_B16_uncertainty | S3R54CAL64_N24_B09_penalty_repair | -0.14064643129286258 |
| 40 | penalty_repair | True | S3R49SCN_N40_B09_median_guard | S3R54CAL64_N40_B11_penalty_repair | 0.08707772415544857 |

## 10. Run56 vs Prior Key Records
Run56 was compared against combined328, Run46, Run41, Run36, Run27, and earlier baselines where available. Summary: Run56 is compared as a native N24/N40 teacher-validation batch against earlier Run36 and Run27 sources through combined392 ranks.

## 11. Calibrated N24/N40 Batch64 Effectiveness Audit
Run56 created 5 new best metric-level records versus combined328 and contributed top5/top10 density in combined392.

Top-entry counts by N:
```json
{
  "N24": {
    "top5_u2_entries": 1,
    "top10_u2_entries": 3,
    "top5_reward_entries": 4,
    "top10_reward_entries": 7
  },
  "N40": {
    "top5_u2_entries": 2,
    "top10_u2_entries": 5,
    "top5_reward_entries": 3,
    "top10_reward_entries": 3
  }
}
```

## 12. Prediction Audit for Run53 Batch64
Prediction column used for reward audit: `calibrated_reward_prediction`. Overall reward Spearman: `0.36598957753742745`. Top1 hit: `0/2`. Mean top5 overlap: `2.0`.

Per-N prediction audit:
```json
{
  "N24": {
    "count": 32,
    "reward_spearman": 0.3420287515669539,
    "constrained_reward_spearman": 0.2670163026707567,
    "strict_penalty_guard_spearman": 0.13776668296198463,
    "penalty_repair_spearman": 0.0323292482684124,
    "top5_overlap": 2,
    "predicted_top1": "S3R54CAL64_N24_B01_n24_u2ret_top",
    "realized_top1": "S3R54CAL64_N24_B04_n24_u2ret_top",
    "u2_spearman": 0.44190161129074285,
    "peeq_spearman": -0.1382682488078973,
    "surfaceT_spearman": -0.22475550242464915,
    "mises_spearman": 0.23466831313688336
  },
  "N40": {
    "count": 32,
    "reward_spearman": 0.6928807615202631,
    "constrained_reward_spearman": 0.6862884526525841,
    "strict_penalty_guard_spearman": 0.6527646662027878,
    "penalty_repair_spearman": 0.6222616444176109,
    "top5_overlap": 2,
    "predicted_top1": "S3R54CAL64_N40_B24_peeq_repair",
    "realized_top1": "S3R54CAL64_N40_B25_peeq_repair",
    "u2_spearman": 0.690061823726965,
    "peeq_spearman": 0.47508852629704545,
    "surfaceT_spearman": 0.6171675910687677,
    "mises_spearman": 0.06084825588374554
  }
}
```

## 13. U2 Gain Versus Penalty Analysis
Run56 U2-best candidates were audited against PEEQ, SurfaceT, Mises, and reward ranks to identify any safety or balance penalties.

## 14. N24/N40 Maturity and RL-Readiness Audit
N24/N40 now have 160 native teacher rows each, enough to support a stronger offline active-learning/RL evidence freeze for those N values, while N12/N16 remain under-sampled anchors.

N24 rows: 160. N40 rows: 160. N32 legacy rows: 332. N12/N16 remain at 36 and 36.

## 15. Metric Semantic Boundary for N32
combined392_plus_N32 includes N32 legacy-compatible rows. These rows are not native Stage 3 teacher validation. PEEQ is mapped from Stage 2 `peeq_guard`, and Mises is mapped from `mises_P95_top_band`; they are proxy-compatible fields with warnings, not literal native Stage 3 metric identities.

## 16. Claim Boundary
Claim boundary verdict: `RUN57_INGESTION_AND_COMBINED392_RANKING_ONLY_NO_SOLVER_OR_TRAINING`.

## 17. Output Files
- Run56 enriched dataset: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_57_calibrated_N24_N40_batch64_teacher_metrics_ingestion_and_combined392_ranking\run56_calibrated_N24_N40_batch64_teacher_dataset_enriched.csv`
- Run56 ranked within batch: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_57_calibrated_N24_N40_batch64_teacher_metrics_ingestion_and_combined392_ranking\run56_calibrated_N24_N40_batch64_ranked_within_batch.csv`
- Native combined392 RL-ready dataset: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_57_calibrated_N24_N40_batch64_teacher_metrics_ingestion_and_combined392_ranking\combined392_RL_ready_dataset.csv`
- combined392_plus_N32 RL-ready dataset: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_57_calibrated_N24_N40_batch64_teacher_metrics_ingestion_and_combined392_ranking\combined392_plus_N32_RL_ready_dataset.csv`
- Run56 vs combined328 comparison: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_57_calibrated_N24_N40_batch64_teacher_metrics_ingestion_and_combined392_ranking\run56_vs_combined328_best_comparison.csv`
- Effectiveness audit: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_57_calibrated_N24_N40_batch64_teacher_metrics_ingestion_and_combined392_ranking\run56_calibrated_batch64_effectiveness_audit.csv`
- Prediction audit: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_57_calibrated_N24_N40_batch64_teacher_metrics_ingestion_and_combined392_ranking\run56_prediction_audit_for_run53_batch64.csv`
- Maturity audit: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_57_calibrated_N24_N40_batch64_teacher_metrics_ingestion_and_combined392_ranking\n24_n40_maturity_and_rl_readiness_audit.md`
- Manifest: `E:\Projects\RL-LAM-ScanOpt\artifacts\manifests\stage3_run_57_manifest.json`

## 18. Recommended Run58
Run58 should update models with native combined392 and combined392_plus_N32, then decide whether to run targeted penalty-repair generation or freeze N24/N40 policy-learning evidence depending on prediction calibration.
