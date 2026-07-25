# Stage 3 Run 42 - Native N24/N40 Focused Batch60 Teacher Metrics Ingestion and Combined264 Ranking

## 1. Purpose
Run42 ingests the completed Run41 native N24/N40 focused batch60 teacher metrics, merges them with Run39/Run38 candidate metadata, recomputes within-N rankings, and builds native combined264 plus combined264_plus_N32 datasets.

## 2. Inputs
- Run41 teacher metrics: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_41_native_N24_N40_focused_batch60_odb_teacher_extraction\run41_native_N24_N40_focused_batch60_teacher_metrics.csv`
- Run39 handoff metadata: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_39_run38_native_N24_N40_focused_batch60_handoff_package\stage3_run39_native_N24_N40_focused_batch60_candidate_orders.csv`
- Native combined204 RL-ready dataset: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_37_N32_informed_native_batch32_teacher_metrics_ingestion_and_combined204_ranking\combined204_RL_ready_dataset.csv`
- combined204_plus_N32 RL-ready dataset: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_37_N32_informed_native_batch32_teacher_metrics_ingestion_and_combined204_ranking\combined204_plus_N32_RL_ready_dataset.csv`
- N32 deduplicated legacy-compatible table: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_32a_stage2_n32_legacy_teacher_label_ingestion_for_stage3\n32_legacy_teacher_dataset_dedup_training_332.csv`

## 3. Run41 Teacher-Extraction Status
Run41 was complete for 60/60 teacher-extracted cases: N24=30 and N40=30. It contains no N12, N16, or N32 cases. Run41 is native N24/N40 teacher validation, not N32 teacher validation.

## 4. Input Validation
Verdict: `PASS_RUN42_NATIVE_N24_N40_FOCUSED_BATCH60_TEACHER_METRICS_60_OF_60_READY`.

Run41 counts: N24=30, N40=30. Native combined204 counts: N12=36, N16=36, N24=66, N40=66. combined204_plus_N32 counts: N12=36, N16=36, N24=66, N32=332, N40=66.

## 5. Run41 Enriched Teacher Dataset
Run42 produced an enriched Run41 teacher dataset with handoff names, Run38/Run39 prediction metadata, candidate-source metadata, scan orders, hashes, raw teacher metrics, extraction status, and nonfatal-warning flags.

## 6. Run41 Within-Batch Ranking
Within Run41, lower raw metric values are better for U2, PEEQ, SurfaceT, and Mises. The U2-primary reward uses 0.65 U2, 0.20 PEEQ, 0.10 SurfaceT, and 0.05 Mises rank scores.

Run41 best U2 by N:
| n | best_u2_strategy | best_u2_value |
| --- | --- | --- |
| 24 | S3R39N2440B60_N24_B08_n24_u2_near | 3.7988751273587695e-05 |
| 40 | S3R39N2440B60_N40_B04_n40_u2_near | 6.792713247705251e-05 |

Run41 best combined reward by N:
| n | best_reward_strategy | best_reward_value |
| --- | --- | --- |
| 24 | S3R39N2440B60_N24_B07_n24_u2_near | 0.8379310344827586 |
| 40 | S3R39N2440B60_N40_B04_n40_u2_near | 0.85 |

## 7. Native Combined264 Construction
Native combined264 rows: 264. Counts: N12=36, N16=36, N24=96, N40=96. There are no N32 rows in native combined264.

## 8. combined264_plus_N32 Construction
combined264_plus_N32 rows: 596. Counts: N12=36, N16=36, N24=96, N32=332, N40=96. N32 rows preserve the legacy metric semantic warnings from Run32A.

## 9. Run41 vs Combined204 Best Comparison
Run41 was compared against the native combined204 best records for N24 and N40 across U2, PEEQ, SurfaceT, Mises, and recomputed combined reward.

| n | metric | run41_beats_baseline | baseline_best_strategy | run41_best_strategy | absolute_improvement |
| --- | --- | --- | --- | --- | --- |
| 24 | U2 | False | S3R34N32INF_N24_B10_n24_calibration | S3R39N2440B60_N24_B08_n24_u2_near | -5.2493421662802575e-06 |
| 24 | PEEQ | False | S3R19B28_N24_B07_control_sentinel | S3R39N2440B60_N24_B12_surrogate_top | -0.00031788647174829654 |
| 24 | SurfaceT | False | S3R24L64_N24_B06_model_disagreement | S3R39N2440B60_N24_B17_diversity | -461824.0 |
| 24 | Mises | False | N24_A06_edge_in_alternating | S3R39N2440B60_N24_B17_diversity | -64.0 |
| 24 | combined_reward | False | S3R34N32INF_N24_B11_n24_calibration | S3R39N2440B60_N24_B07_n24_u2_near | -0.0036074270557028276 |
| 40 | U2 | False | S3R34N32INF_N40_B08_n40_best_near | S3R39N2440B60_N40_B04_n40_u2_near | -1.0961070074699819e-05 |
| 40 | PEEQ | True | S3R19B28_N40_B05_diversity_top | S3R39N2440B60_N40_B10_surrogate_top | 0.00019846856594091244 |
| 40 | SurfaceT | False | S3R19B28_N40_B06_uncertainty_calibration | S3R39N2440B60_N40_B01_n40_u2_near | -497792.0 |
| 40 | Mises | False | N40_A07_regular_jump_coprime | S3R39N2440B60_N40_B25_diversity | -320.0 |
| 40 | combined_reward | False | S3R24L64_N40_B23_exploitation_reference | S3R39N2440B60_N40_B04_n40_u2_near | -0.0492307692307693 |

## 10. Run41 vs Prior Key Records
Run41 was compared against combined204, Run36, Run27, and the earlier combined172 baseline where available. Summary: Run41 is compared as a native N24/N40 teacher-validation batch against earlier Run36 and Run27 sources through combined264 ranks.

## 11. N24/N40 Focused Batch60 Effectiveness Audit
Run41 created 1 new best metric-level records versus combined204 and contributed top5/top10 density in combined264.

Top-entry counts by N:
```json
{
  "N24": {
    "top5_u2_entries": 1,
    "top10_u2_entries": 4,
    "top5_reward_entries": 2,
    "top10_reward_entries": 3
  },
  "N40": {
    "top5_u2_entries": 2,
    "top10_u2_entries": 4,
    "top5_reward_entries": 2,
    "top10_reward_entries": 5
  }
}
```

## 12. Prediction Audit for Run38 Batch60
Prediction column used for reward audit: `hybrid_score`. Overall reward Spearman: `0.9123898969129455`. Top1 hit: `0/2`. Mean top5 overlap: `4.0`.

Per-N prediction audit:
```json
{
  "N24": {
    "count": 30,
    "reward_spearman": 0.9269106743306323,
    "top5_overlap": 4,
    "predicted_top1": "S3R39N2440B60_N24_B01_n24_u2_near",
    "realized_top1": "S3R39N2440B60_N24_B07_n24_u2_near",
    "u2_spearman": 0.8663922625644461
  },
  "N40": {
    "count": 30,
    "reward_spearman": 0.8803114571746384,
    "top5_overlap": 4,
    "predicted_top1": "S3R39N2440B60_N40_B01_n40_u2_near",
    "realized_top1": "S3R39N2440B60_N40_B04_n40_u2_near",
    "u2_spearman": 0.8364899430074236
  }
}
```

## 13. U2 Gain Versus Penalty Analysis
Run41 did not create new U2 bests versus combined204, so no U2-gain penalty rows were generated.

## 14. Metric Semantic Boundary for N32
combined264_plus_N32 includes N32 legacy-compatible rows. These rows are not native Stage 3 teacher validation. PEEQ is mapped from Stage 2 `peeq_guard`, and Mises is mapped from `mises_P95_top_band`; they are proxy-compatible fields with warnings, not literal native Stage 3 metric identities.

## 15. Claim Boundary
Claim boundary verdict: `RUN42_INGESTION_AND_COMBINED264_RANKING_ONLY_NO_SOLVER_OR_TRAINING`.

## 16. Output Files
- Run41 enriched dataset: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_42_native_N24_N40_focused_batch60_teacher_metrics_ingestion_and_combined264_ranking\run41_native_N24_N40_focused_batch60_teacher_dataset_enriched.csv`
- Run41 ranked within batch: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_42_native_N24_N40_focused_batch60_teacher_metrics_ingestion_and_combined264_ranking\run41_native_N24_N40_focused_batch60_ranked_within_batch.csv`
- Native combined264 RL-ready dataset: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_42_native_N24_N40_focused_batch60_teacher_metrics_ingestion_and_combined264_ranking\combined264_RL_ready_dataset.csv`
- combined264_plus_N32 RL-ready dataset: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_42_native_N24_N40_focused_batch60_teacher_metrics_ingestion_and_combined264_ranking\combined264_plus_N32_RL_ready_dataset.csv`
- Run41 vs combined204 comparison: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_42_native_N24_N40_focused_batch60_teacher_metrics_ingestion_and_combined264_ranking\run41_vs_combined204_best_comparison.csv`
- Effectiveness audit: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_42_native_N24_N40_focused_batch60_teacher_metrics_ingestion_and_combined264_ranking\run41_N24_N40_focused_batch60_effectiveness_audit.csv`
- Prediction audit: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_42_native_N24_N40_focused_batch60_teacher_metrics_ingestion_and_combined264_ranking\run41_prediction_audit_for_run38_batch60.csv`
- Manifest: `E:\Projects\RL-LAM-ScanOpt\artifacts\manifests\stage3_run_42_manifest.json`

## 17. Recommended Run43
Run43 should update models with combined264 and continue local calibration around high-density top5/top10 regions rather than declaring the focused search exhausted.
