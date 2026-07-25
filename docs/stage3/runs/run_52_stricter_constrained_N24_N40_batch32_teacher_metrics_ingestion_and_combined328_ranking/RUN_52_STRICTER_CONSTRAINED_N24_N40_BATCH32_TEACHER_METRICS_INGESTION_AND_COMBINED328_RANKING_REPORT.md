# Stage 3 Run 52 - Stricter Constrained N24/N40 Batch32 Teacher Metrics Ingestion and Combined328 Ranking

## 1. Purpose
Run52 ingests the completed Run51 native constrained N24/N40 batch32 teacher metrics, merges them with Run49/Run48 candidate metadata, recomputes within-N rankings, and builds native combined328 plus combined328_plus_N32 datasets.

## 2. Inputs
- Run51 teacher metrics: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_51_stricter_constrained_N24_N40_batch32_odb_teacher_validation\run51_stricter_constrained_N24_N40_batch32_teacher_metrics.csv`
- Run49 handoff metadata: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_49_run48_stricter_constrained_N24_N40_batch32_handoff_package\stage3_run49_stricter_constrained_N24_N40_batch32_candidate_orders.csv`
- Native combined296 RL-ready dataset: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_47_constrained_N24_N40_batch32_teacher_metrics_ingestion_and_combined296_ranking\combined296_RL_ready_dataset.csv`
- combined296_plus_N32 RL-ready dataset: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_47_constrained_N24_N40_batch32_teacher_metrics_ingestion_and_combined296_ranking\combined296_plus_N32_RL_ready_dataset.csv`
- N32 deduplicated legacy-compatible table: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_32a_stage2_n32_legacy_teacher_label_ingestion_for_stage3\n32_legacy_teacher_dataset_dedup_training_332.csv`

## 3. Run51 Teacher-Extraction Status
Run51 was complete for 32/32 teacher-validated cases: N24=16 and N40=16. It contains no N12, N16, or N32 cases. Run51 is native stricter constrained N24/N40 teacher validation, not N32 teacher validation.

## 4. Input Validation
Verdict: `PASS_RUN52_STRICTER_CONSTRAINED_N24_N40_BATCH32_TEACHER_METRICS_32_OF_32_READY`.

Run51 counts: N24=16, N40=16. Native combined296 input counts: N12=36, N16=36, N24=112, N40=112. combined296_plus_N32 input counts: N12=36, N16=36, N24=112, N32=332, N40=112.

## 5. Run51 Enriched Teacher Dataset
Run52 produced an enriched Run51 teacher dataset with handoff names, Run48/Run49 prediction metadata, candidate-source metadata, scan orders, hashes, raw teacher metrics, extraction status, and nonfatal-warning flags.

## 6. Run51 Within-Batch Ranking
Within Run51, lower raw metric values are better for U2, PEEQ, SurfaceT, and Mises. The U2-primary reward uses 0.65 U2, 0.20 PEEQ, 0.10 SurfaceT, and 0.05 Mises rank scores.

Run51 best U2 by N:
| n | best_u2_strategy | best_u2_value |
| --- | --- | --- |
| 24 | S3R49SCN_N24_B09_median_guard | 2.9523077955673216e-05 |
| 40 | S3R49SCN_N40_B04_strict_guard | 6.170961933094077e-05 |

Run51 best combined reward by N:
| n | best_reward_strategy | best_reward_value |
| --- | --- | --- |
| 24 | S3R49SCN_N24_B09_median_guard | 0.7983333333333332 |
| 40 | S3R49SCN_N40_B04_strict_guard | 0.8750000000000001 |

## 7. Native Combined328 Construction
Native combined328 rows: 328. Counts: N12=36, N16=36, N24=128, N40=128. There are no N32 rows in native combined328.

## 8. combined328_plus_N32 Construction
combined328_plus_N32 rows: 660. Counts: N12=36, N16=36, N24=128, N32=332, N40=128. N32 rows preserve the legacy metric semantic warnings from Run32A.

## 9. Run51 vs Combined296 Best Comparison
Run51 was compared against the native combined296 best records for N24 and N40 across U2, PEEQ, SurfaceT, Mises, U2-primary reward, constrained reward, and strict penalty guard reward.

| n | metric | run51_beats_baseline | baseline_best_strategy | run51_best_strategy | absolute_improvement |
| --- | --- | --- | --- | --- | --- |
| 24 | U2 | True | S3R44CNS_N24_B14_hybrid_disagree | S3R49SCN_N24_B09_median_guard | 2.140329343092162e-06 |
| 24 | PEEQ | False | S3R19B28_N24_B07_control_sentinel | S3R49SCN_N24_B07_two_stage | -0.004698440432548495 |
| 24 | SurfaceT | False | S3R24L64_N24_B06_model_disagreement | S3R49SCN_N24_B03_strict_guard | -989376.0 |
| 24 | Mises | False | N24_A06_edge_in_alternating | S3R49SCN_N24_B15_mises_repair | -704.0 |
| 24 | combined_reward | False | S3R44CNS_N24_B16_uncertainty | S3R49SCN_N24_B09_median_guard | -0.10999999999999999 |
| 40 | U2 | False | S3R34N32INF_N40_B08_n40_best_near | S3R49SCN_N40_B04_strict_guard | -4.7435569285880774e-06 |
| 40 | PEEQ | False | S3R39N2440B60_N40_B10_surrogate_top | S3R49SCN_N40_B09_median_guard | -0.009407103061676109 |
| 40 | SurfaceT | False | S3R19B28_N40_B06_uncertainty_calibration | S3R49SCN_N40_B05_two_stage | -861632.0 |
| 40 | Mises | False | N40_A07_regular_jump_coprime | S3R49SCN_N40_B11_peeq_repair | -2496.0 |
| 40 | combined_reward | False | S3R44CNS_N40_B01_reward_balanced | S3R49SCN_N40_B04_strict_guard | -0.007882882882882747 |
| 24 | constrained_u2_reward_balanced | False | S3R44CNS_N24_B16_uncertainty | S3R49SCN_N24_B09_median_guard | -0.16720720720720705 |
| 40 | constrained_u2_reward_balanced | True | S3R39N2440B60_N40_B02_n40_u2_near | S3R49SCN_N40_B09_median_guard | 0.04189189189189191 |
| 24 | strict_penalty_guard | False | S3R44CNS_N24_B16_uncertainty | S3R49SCN_N24_B09_median_guard | -0.22486486486486468 |
| 40 | strict_penalty_guard | True | S3R39N2440B60_N40_B02_n40_u2_near | S3R49SCN_N40_B09_median_guard | 0.06306306306306309 |

## 10. Run51 vs Prior Key Records
Run51 was compared against combined296, Run46, Run41, Run36, Run27, and earlier baselines where available. Summary: Run51 is compared as a native N24/N40 teacher-validation batch against earlier Run36 and Run27 sources through combined328 ranks.

## 11. Stricter Constrained N24/N40 Batch32 Effectiveness Audit
Run51 created 3 new best metric-level records versus combined296 and contributed top5/top10 density in combined328.

Top-entry counts by N:
```json
{
  "N24": {
    "top5_u2_entries": 3,
    "top10_u2_entries": 5,
    "top5_reward_entries": 3,
    "top10_reward_entries": 3
  },
  "N40": {
    "top5_u2_entries": 1,
    "top10_u2_entries": 4,
    "top5_reward_entries": 1,
    "top10_reward_entries": 1
  }
}
```

## 12. Prediction Audit for Run48 Batch32
Prediction column used for reward audit: `strict_penalty_guard_prediction`. Overall reward Spearman: `0.39989742648661625`. Top1 hit: `1/2`. Mean top5 overlap: `2.5`.

Per-N prediction audit:
```json
{
  "N24": {
    "count": 16,
    "reward_spearman": 0.4520616665135417,
    "constrained_reward_spearman": 0.42130917219289254,
    "strict_penalty_guard_spearman": 0.3469896067523393,
    "top5_overlap": 2,
    "predicted_top1": "S3R49SCN_N24_B09_median_guard",
    "realized_top1": "S3R49SCN_N24_B09_median_guard",
    "u2_spearman": 0.42130917219289254,
    "peeq_spearman": 0.04920399091303855,
    "surfaceT_spearman": -0.22295558382470593,
    "mises_spearman": 0.0332838928630014
  },
  "N40": {
    "count": 16,
    "reward_spearman": 0.51150699812112,
    "constrained_reward_spearman": 0.5174375140413648,
    "strict_penalty_guard_spearman": 0.5204027720014873,
    "top5_overlap": 3,
    "predicted_top1": "S3R49SCN_N40_B11_peeq_repair",
    "realized_top1": "S3R49SCN_N40_B09_median_guard",
    "u2_spearman": 0.44280744277004763,
    "peeq_spearman": 0.49371545036038544,
    "surfaceT_spearman": 0.5040938532208139,
    "mises_spearman": 0.7052915156234785
  }
}
```

## 13. U2 Gain Versus Penalty Analysis
Run51 U2-best candidates were audited against PEEQ, SurfaceT, Mises, and reward ranks to identify any safety or balance penalties.

## 14. Metric Semantic Boundary for N32
combined328_plus_N32 includes N32 legacy-compatible rows. These rows are not native Stage 3 teacher validation. PEEQ is mapped from Stage 2 `peeq_guard`, and Mises is mapped from `mises_P95_top_band`; they are proxy-compatible fields with warnings, not literal native Stage 3 metric identities.

## 15. Claim Boundary
Claim boundary verdict: `RUN52_INGESTION_AND_COMBINED328_RANKING_ONLY_NO_SOLVER_OR_TRAINING`.

## 16. Output Files
- Run51 enriched dataset: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_52_stricter_constrained_N24_N40_batch32_teacher_metrics_ingestion_and_combined328_ranking\run51_stricter_constrained_N24_N40_batch32_teacher_dataset_enriched.csv`
- Run51 ranked within batch: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_52_stricter_constrained_N24_N40_batch32_teacher_metrics_ingestion_and_combined328_ranking\run51_stricter_constrained_N24_N40_batch32_ranked_within_batch.csv`
- Native combined328 RL-ready dataset: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_52_stricter_constrained_N24_N40_batch32_teacher_metrics_ingestion_and_combined328_ranking\combined328_RL_ready_dataset.csv`
- combined328_plus_N32 RL-ready dataset: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_52_stricter_constrained_N24_N40_batch32_teacher_metrics_ingestion_and_combined328_ranking\combined328_plus_N32_RL_ready_dataset.csv`
- Run51 vs combined296 comparison: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_52_stricter_constrained_N24_N40_batch32_teacher_metrics_ingestion_and_combined328_ranking\run51_vs_combined296_best_comparison.csv`
- Effectiveness audit: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_52_stricter_constrained_N24_N40_batch32_teacher_metrics_ingestion_and_combined328_ranking\run51_stricter_constrained_batch32_effectiveness_audit.csv`
- Prediction audit: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_52_stricter_constrained_N24_N40_batch32_teacher_metrics_ingestion_and_combined328_ranking\run51_prediction_audit_for_run48_batch32.csv`
- Manifest: `E:\Projects\RL-LAM-ScanOpt\artifacts\manifests\stage3_run_52_manifest.json`

## 17. Recommended Run53
Run53 should update models with native combined328 and combined328_plus_N32, then consider another small stricter-guard batch if prediction calibration is credible; scale only if strict-guard gains and top-density are stable.
