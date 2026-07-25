# Stage 3 Run 42 - Constrained N24/N40 Batch32 Teacher Metrics Ingestion and Combined296 Ranking

## 1. Purpose
Run47 ingests the completed Run46 native constrained N24/N40 batch32 teacher metrics, merges them with Run44/Run43 candidate metadata, recomputes within-N rankings, and builds native combined296 plus combined296_plus_N32 datasets.

## 2. Inputs
- Run46 teacher metrics: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_46_constrained_N24_N40_batch32_odb_teacher_validation\run46_constrained_N24_N40_batch32_teacher_metrics.csv`
- Run44 handoff metadata: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_44_run43_constrained_N24_N40_batch32_handoff_package\stage3_run44_constrained_N24_N40_batch32_candidate_orders.csv`
- Native combined264 RL-ready dataset: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_42_native_N24_N40_focused_batch60_teacher_metrics_ingestion_and_combined264_ranking\combined264_RL_ready_dataset.csv`
- combined264_plus_N32 RL-ready dataset: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_42_native_N24_N40_focused_batch60_teacher_metrics_ingestion_and_combined264_ranking\combined264_plus_N32_RL_ready_dataset.csv`
- N32 deduplicated legacy-compatible table: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_32a_stage2_n32_legacy_teacher_label_ingestion_for_stage3\n32_legacy_teacher_dataset_dedup_training_332.csv`

## 3. Run46 Teacher-Extraction Status
Run46 was complete for 60/60 teacher-extracted cases: N24=30 and N40=30. It contains no N12, N16, or N32 cases. Run46 is native N24/N40 teacher validation, not N32 teacher validation.

## 4. Input Validation
Verdict: `PASS_RUN47_CONSTRAINED_N24_N40_BATCH32_TEACHER_METRICS_32_OF_32_READY`.

Run46 counts: N24=16, N40=16. Native combined264 counts: N12=36, N16=36, N24=96, N40=96. combined264_plus_N32 counts: N12=36, N16=36, N24=96, N32=332, N40=96.

## 5. Run46 Enriched Teacher Dataset
Run47 produced an enriched Run46 teacher dataset with handoff names, Run43/Run44 prediction metadata, candidate-source metadata, scan orders, hashes, raw teacher metrics, extraction status, and nonfatal-warning flags.

## 6. Run46 Within-Batch Ranking
Within Run46, lower raw metric values are better for U2, PEEQ, SurfaceT, and Mises. The U2-primary reward uses 0.65 U2, 0.20 PEEQ, 0.10 SurfaceT, and 0.05 Mises rank scores.

Run46 best U2 by N:
| n | best_u2_strategy | best_u2_value |
| --- | --- | --- |
| 24 | S3R44CNS_N24_B14_hybrid_disagree | 3.166340729876538e-05 |
| 40 | S3R44CNS_N40_B01_reward_balanced | 5.7689914683578536e-05 |

Run46 best combined reward by N:
| n | best_reward_strategy | best_reward_value |
| --- | --- | --- |
| 24 | S3R44CNS_N24_B16_uncertainty | 0.8783333333333334 |
| 40 | S3R44CNS_N40_B01_reward_balanced | 0.9033333333333333 |

## 7. Native Combined296 Construction
Native combined296 rows: 296. Counts: N12=36, N16=36, N24=112, N40=112. There are no N32 rows in native combined296.

## 8. combined296_plus_N32 Construction
combined296_plus_N32 rows: 628. Counts: N12=36, N16=36, N24=112, N32=332, N40=112. N32 rows preserve the legacy metric semantic warnings from Run32A.

## 9. Run46 vs Combined264 Best Comparison
Run46 was compared against the native combined264 best records for N24 and N40 across U2, PEEQ, SurfaceT, Mises, and recomputed combined reward.

| n | metric | run46_beats_baseline | baseline_best_strategy | run46_best_strategy | absolute_improvement |
| --- | --- | --- | --- | --- | --- |
| 24 | U2 | True | S3R34N32INF_N24_B10_n24_calibration | S3R44CNS_N24_B14_hybrid_disagree | 1.0760018085420597e-06 |
| 24 | PEEQ | False | S3R19B28_N24_B07_control_sentinel | S3R44CNS_N24_B16_uncertainty | -0.0035199970006943027 |
| 24 | SurfaceT | False | S3R24L64_N24_B06_model_disagreement | S3R44CNS_N24_B07_surfacet_guarded | -621760.0 |
| 24 | Mises | False | N24_A06_edge_in_alternating | S3R44CNS_N24_B03_constr_surrogate | -576.0 |
| 24 | combined_reward | True | S3R34N32INF_N24_B11_n24_calibration | S3R44CNS_N24_B16_uncertainty | 0.03964912280701771 |
| 40 | U2 | False | S3R34N32INF_N40_B08_n40_best_near | S3R44CNS_N40_B01_reward_balanced | -7.238522812258452e-07 |
| 40 | PEEQ | False | S3R39N2440B60_N40_B10_surrogate_top | S3R44CNS_N40_B08_surfacet_guarded | -0.009398207068443298 |
| 40 | SurfaceT | False | S3R19B28_N40_B06_uncertainty_calibration | S3R44CNS_N40_B05_peeq_guarded | -802688.0 |
| 40 | Mises | False | N40_A07_regular_jump_coprime | S3R44CNS_N40_B04_constr_surrogate | -2496.0 |
| 40 | combined_reward | True | S3R24L64_N40_B23_exploitation_reference | S3R44CNS_N40_B01_reward_balanced | 0.02359649122807017 |
| 24 | constrained_u2_reward_balanced | True | S3R34N32INF_N24_B11_n24_calibration | S3R44CNS_N24_B16_uncertainty | 0.07824561403508778 |
| 40 | constrained_u2_reward_balanced | True | S3R39N2440B60_N40_B02_n40_u2_near | S3R44CNS_N40_B01_reward_balanced | 0.008245614035087723 |

## 10. Run46 vs Prior Key Records
Run46 was compared against combined264, Run41, Run36, Run27, and earlier baselines where available. Summary: Run46 is compared as a native N24/N40 teacher-validation batch against earlier Run36 and Run27 sources through combined296 ranks.

## 11. Constrained N24/N40 Batch32 Effectiveness Audit
Run46 created 5 new best metric-level records versus combined264 and contributed top5/top10 density in combined296.

Top-entry counts by N:
```json
{
  "N24": {
    "top5_u2_entries": 1,
    "top10_u2_entries": 3,
    "top5_reward_entries": 4,
    "top10_reward_entries": 5
  },
  "N40": {
    "top5_u2_entries": 2,
    "top10_u2_entries": 4,
    "top5_reward_entries": 3,
    "top10_reward_entries": 4
  }
}
```

## 12. Prediction Audit for Run43 Batch60
Prediction column used for reward audit: `constrained_reward_prediction`. Overall reward Spearman: `-0.11305355146557049`. Top1 hit: `1/2`. Mean top5 overlap: `2.5`.

Per-N prediction audit:
```json
{
  "N24": {
    "count": 16,
    "reward_spearman": -0.09533273239401217,
    "constrained_reward_spearman": 0.03536536846874646,
    "top5_overlap": 2,
    "predicted_top1": "S3R44CNS_N24_B01_reward_balanced",
    "realized_top1": "S3R44CNS_N24_B16_uncertainty",
    "u2_spearman": -0.02116130293527289
  },
  "N40": {
    "count": 16,
    "reward_spearman": -0.3675385261144965,
    "constrained_reward_spearman": -0.3617352862284781,
    "top5_overlap": 3,
    "predicted_top1": "S3R44CNS_N40_B01_reward_balanced",
    "realized_top1": "S3R44CNS_N40_B01_reward_balanced",
    "u2_spearman": -0.08401680504168059
  }
}
```

## 13. U2 Gain Versus Penalty Analysis
Run46 U2-best candidates were audited against PEEQ, SurfaceT, Mises, and reward ranks to identify any safety or balance penalties.

## 14. Metric Semantic Boundary for N32
combined296_plus_N32 includes N32 legacy-compatible rows. These rows are not native Stage 3 teacher validation. PEEQ is mapped from Stage 2 `peeq_guard`, and Mises is mapped from `mises_P95_top_band`; they are proxy-compatible fields with warnings, not literal native Stage 3 metric identities.

## 15. Claim Boundary
Claim boundary verdict: `RUN42_INGESTION_AND_COMBINED296_RANKING_ONLY_NO_SOLVER_OR_TRAINING`.

## 16. Output Files
- Run46 enriched dataset: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_47_constrained_N24_N40_batch32_teacher_metrics_ingestion_and_combined296_ranking\run46_constrained_N24_N40_batch32_teacher_dataset_enriched.csv`
- Run46 ranked within batch: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_47_constrained_N24_N40_batch32_teacher_metrics_ingestion_and_combined296_ranking\run46_constrained_N24_N40_batch32_ranked_within_batch.csv`
- Native combined296 RL-ready dataset: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_47_constrained_N24_N40_batch32_teacher_metrics_ingestion_and_combined296_ranking\combined296_RL_ready_dataset.csv`
- combined296_plus_N32 RL-ready dataset: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_47_constrained_N24_N40_batch32_teacher_metrics_ingestion_and_combined296_ranking\combined296_plus_N32_RL_ready_dataset.csv`
- Run46 vs combined264 comparison: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_47_constrained_N24_N40_batch32_teacher_metrics_ingestion_and_combined296_ranking\run46_vs_combined264_best_comparison.csv`
- Effectiveness audit: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_47_constrained_N24_N40_batch32_teacher_metrics_ingestion_and_combined296_ranking\run46_N24_N40_focused_batch32_effectiveness_audit.csv`
- Prediction audit: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_47_constrained_N24_N40_batch32_teacher_metrics_ingestion_and_combined296_ranking\run46_prediction_audit_for_run43_batch32.csv`
- Manifest: `E:\Projects\RL-LAM-ScanOpt\artifacts\manifests\stage3_run_47_manifest.json`

## 17. Recommended Run43
Run48 should update models with native combined296 and combined296_plus_N32, then generate a constrained N24/N40 candidate design that exploits U2 gains while explicitly guarding PEEQ and SurfaceT.
