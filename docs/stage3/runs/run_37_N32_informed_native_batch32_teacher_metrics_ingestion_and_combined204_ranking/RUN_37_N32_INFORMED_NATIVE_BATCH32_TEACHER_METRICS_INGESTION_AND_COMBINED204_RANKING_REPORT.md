# Stage 3 Run 37 - N32-Informed Native Batch32 Teacher Metrics Ingestion and Combined204 Ranking

## 1. Purpose
Run37 ingests the 32 teacher metrics from Run36, ranks the N32-informed native batch32, builds native combined204 and combined204_plus_N32 datasets, and audits whether the N32-informed candidate route improved native Stage 3 performance.

## 2. Inputs
- Run36 teacher metrics: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_36_N32_informed_native_batch32_odb_teacher_validation\run36_N32_informed_native_batch32_teacher_metrics.csv`
- Run34 handoff metadata: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_34_run33_N32_informed_native_batch32_handoff_package\stage3_run34_N32_informed_native_batch32_candidate_orders.csv`
- Native combined172 teacher dataset: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_28_shortlist64_teacher_metrics_ingestion_and_combined172_ranking\combined172_teacher_dataset.csv`
- N32 dedup training dataset: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_32a_stage2_n32_legacy_teacher_label_ingestion_for_stage3\n32_legacy_teacher_dataset_dedup_training_332.csv`

## 3. Run36 Teacher-Validation Status
Run36 completed 32/32 teacher extraction with N12=4, N16=4, N24=12, N40=12. No N32 cases are present in Run36.

## 4. Input Validation
Verdict: `PASS_RUN37_N32_INFORMED_NATIVE_BATCH32_TEACHER_METRICS_32_OF_32_READY`.

## 5. Run36 Enriched Teacher Dataset
Run36 rows were merged with Run34 handoff metadata and written to `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_37_N32_informed_native_batch32_teacher_metrics_ingestion_and_combined204_ranking\run36_N32_informed_native_batch32_teacher_dataset_enriched.csv`.

## 6. Run36 Within-Batch Ranking
Within each native N, Run36 rankings were recomputed for U2, PEEQ, SurfaceT, Mises, and the U2-primary reward.

## 7. Native Combined204 Construction
Native combined204 contains 204 rows with counts {12: 36, 16: 36, 24: 66, 40: 66} and no N32 rows.

## 8. combined204_plus_N32 Construction
combined204_plus_N32 contains 536 rows with counts {12: 36, 16: 36, 24: 66, 32: 332, 40: 66}. N32 remains legacy-compatible and carries metric semantic warnings.

## 9. Run36 vs Combined172 Best Comparison
Run36 produced 2 metric-level records beating the previous combined172 best.
- N24 U2: `S3R34N32INF_N24_B10_n24_calibration` improved by 7.27056e-06.
- N40 U2: `S3R34N32INF_N40_B08_n40_best_near` improved by 2.61482e-06.

## 10. N32-Informed Candidate Effectiveness Audit
Run36 candidates entering combined204 top5 U2: 5; top10 U2: 8; top5 reward: 5; top10 reward: 8.

## 11. Prediction Audit for Run33 Option A
Primary prediction column: `hybrid_score`. Overall Spearman vs combined204 reward: 0.7225.

## 12. N24/N40 Focus Analysis
N24 top10 reward entries from Run36: 3. N40 top10 reward entries from Run36: 4.

## 13. Metric Semantic Boundary for N32
N32 rows in combined204_plus_N32 come from the legacy-compatible Stage 2 ingestion. Run36 contains no N32 rows, and Run37 performs no new N32 teacher validation.

## 14. Claim Boundary
Claim boundary files: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_37_N32_informed_native_batch32_teacher_metrics_ingestion_and_combined204_ranking\run37_claim_boundary.md` and `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_37_N32_informed_native_batch32_teacher_metrics_ingestion_and_combined204_ranking\run37_claim_boundary.json`.

## 15. Output Files
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_37_N32_informed_native_batch32_teacher_metrics_ingestion_and_combined204_ranking\run37_input_validation_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_37_N32_informed_native_batch32_teacher_metrics_ingestion_and_combined204_ranking\run36_N32_informed_native_batch32_teacher_dataset_enriched.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_37_N32_informed_native_batch32_teacher_metrics_ingestion_and_combined204_ranking\run36_N32_informed_native_batch32_ranked_within_batch.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_37_N32_informed_native_batch32_teacher_metrics_ingestion_and_combined204_ranking\run36_N32_informed_native_batch32_per_N_leaderboard.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_37_N32_informed_native_batch32_teacher_metrics_ingestion_and_combined204_ranking\combined204_teacher_dataset.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_37_N32_informed_native_batch32_teacher_metrics_ingestion_and_combined204_ranking\combined204_RL_ready_dataset.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_37_N32_informed_native_batch32_teacher_metrics_ingestion_and_combined204_ranking\combined204_per_N_leaderboard.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_37_N32_informed_native_batch32_teacher_metrics_ingestion_and_combined204_ranking\combined204_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_37_N32_informed_native_batch32_teacher_metrics_ingestion_and_combined204_ranking\combined204_plus_N32_teacher_dataset.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_37_N32_informed_native_batch32_teacher_metrics_ingestion_and_combined204_ranking\combined204_plus_N32_RL_ready_dataset.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_37_N32_informed_native_batch32_teacher_metrics_ingestion_and_combined204_ranking\combined204_plus_N32_per_N_leaderboard.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_37_N32_informed_native_batch32_teacher_metrics_ingestion_and_combined204_ranking\combined204_plus_N32_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_37_N32_informed_native_batch32_teacher_metrics_ingestion_and_combined204_ranking\run36_vs_combined172_best_comparison.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_37_N32_informed_native_batch32_teacher_metrics_ingestion_and_combined204_ranking\run36_N32_informed_candidate_effectiveness_audit.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_37_N32_informed_native_batch32_teacher_metrics_ingestion_and_combined204_ranking\run36_N32_informed_candidate_effectiveness_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_37_N32_informed_native_batch32_teacher_metrics_ingestion_and_combined204_ranking\run36_prediction_audit_for_run33_optionA.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_37_N32_informed_native_batch32_teacher_metrics_ingestion_and_combined204_ranking\run36_prediction_audit_for_run33_optionA_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_37_N32_informed_native_batch32_teacher_metrics_ingestion_and_combined204_ranking\run37_claim_boundary.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_37_N32_informed_native_batch32_teacher_metrics_ingestion_and_combined204_ranking\run37_claim_boundary.json`
- `E:\Projects\RL-LAM-ScanOpt\docs\stage3\runs\run_37_N32_informed_native_batch32_teacher_metrics_ingestion_and_combined204_ranking\RUN_37_N32_INFORMED_NATIVE_BATCH32_TEACHER_METRICS_INGESTION_AND_COMBINED204_RANKING_REPORT.md`
- `E:\Projects\RL-LAM-ScanOpt\artifacts\manifests\stage3_run_37_manifest.json`

## 16. Recommended Run38
Update models using combined204 and combined204_plus_N32, then generate the next candidate batch with explicit native-only versus N32-augmented diagnostics.
