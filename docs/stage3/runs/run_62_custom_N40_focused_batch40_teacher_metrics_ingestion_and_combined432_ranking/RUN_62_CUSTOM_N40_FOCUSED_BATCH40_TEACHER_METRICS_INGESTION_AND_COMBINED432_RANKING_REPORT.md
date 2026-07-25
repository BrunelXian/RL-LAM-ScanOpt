# Stage 3 Run 62 - Custom N40-Focused Batch40 Teacher Metrics Ingestion and Combined432 Ranking

## 1. Purpose
Run62 ingests Run61 teacher metrics, builds native combined432 and combined432_plus_N32 datasets, and evaluates whether the custom N40-focused calibrated penalty-repair batch40 improved native Stage 3 teacher metrics.

## 2. Inputs
- Run61 teacher metrics: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_61_custom_N40_focused_calibrated_penalty_repair_batch40_odb_teacher_validation\run61_custom_N40_focused_batch40_teacher_metrics.csv`
- Run59 handoff metadata: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_59_run58_N40_focused_calibrated_penalty_repair_batch40_handoff_package\stage3_run59_N40_focused_calibrated_penalty_repair_batch40_candidate_orders.csv`
- Native combined392: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_57_calibrated_N24_N40_batch64_teacher_metrics_ingestion_and_combined392_ranking\combined392_RL_ready_dataset.csv`
- combined392_plus_N32: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_57_calibrated_N24_N40_batch64_teacher_metrics_ingestion_and_combined392_ranking\combined392_plus_N32_RL_ready_dataset.csv`

## 3. Run61 Teacher-Validation Status
Run61 completed 40/40 with N24=16 and N40=24. Nonfatal warnings were present, with no failed or incomplete cases.

## 4. Input Validation
Verdict: `PASS_RUN62_CUSTOM_N40_FOCUSED_BATCH40_TEACHER_METRICS_40_OF_40_READY`.

## 5. Run61 Enriched Teacher Dataset
Output: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_62_custom_N40_focused_batch40_teacher_metrics_ingestion_and_combined432_ranking\run61_custom_N40_focused_batch40_teacher_dataset_enriched.csv`

## 6. Run61 Within-Batch Ranking
Output: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_62_custom_N40_focused_batch40_teacher_metrics_ingestion_and_combined432_ranking\run61_custom_N40_focused_batch40_ranked_within_batch.csv`

## 7. Native Combined432 Construction
Native combined432 counts: `{12: 36, 16: 36, 24: 176, 40: 184}`.

## 8. combined432_plus_N32 Construction
combined432_plus_N32 counts: `{12: 36, 16: 36, 24: 176, 32: 332, 40: 184}`. N32 metric semantic warnings are preserved.

## 9. Run61 vs Combined392 Best Comparison
Run61 created 4 new metric/reward records versus combined392.

## 10. Run61 vs Prior Key Records
Run61 was compared with Run56, Run51, Run46, Run41, Run36, and combined best records using recomputed combined432 ranks.

## 11. Custom N40-Focused Batch40 Effectiveness Audit
Run61 produced 4 new metric/reward records versus combined392; N40 remains the primary focus while the expanded N24 rows provide maintenance/diagnostic coverage.

## 12. Prediction Audit for Run58/Run59 Custom Batch40
Run58/Run59 prediction calibration was evaluated on realized Run61 labels; use these diagnostics as calibration evidence, not teacher-validation claims for future candidates.

## 13. N40 Focus Versus N24 Maintenance Analysis
N40 remains the main active improvement direction. The extra N24 rows evaluate maintenance and diagnostic coverage after N24 reached dense evidence status.

## 14. Updated N24/N40 Maturity and Claim-Boundary Audit
After Run61, N24 has 176 and N40 has 184 native teacher rows; N40 remains the main active improvement direction, N24 is in maintenance/evidence-freeze status, and full variable-N maturity remains limited by N12/N16 at 36 rows each.

## 15. Metric Semantic Boundary for N32
N32 rows are legacy-compatible auxiliary data. They are not native Stage 3 teacher validation, and PEEQ/Mises mappings must retain semantic warnings.

## 16. Claim Boundary
Verdict: `RUN62_INGESTION_AND_ANALYSIS_ONLY_NO_SOLVER_OR_CANDIDATE_GENERATION`.

## 17. Output Files
- combined432 RL-ready: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_62_custom_N40_focused_batch40_teacher_metrics_ingestion_and_combined432_ranking\combined432_RL_ready_dataset.csv`
- combined432_plus_N32 RL-ready: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_62_custom_N40_focused_batch40_teacher_metrics_ingestion_and_combined432_ranking\combined432_plus_N32_RL_ready_dataset.csv`
- Run61 comparison: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_62_custom_N40_focused_batch40_teacher_metrics_ingestion_and_combined432_ranking\run61_vs_combined392_best_comparison.csv`
- Updated maturity audit: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_62_custom_N40_focused_batch40_teacher_metrics_ingestion_and_combined432_ranking\n24_n40_updated_maturity_and_claim_boundary_audit.md`
- Manifest: `E:\Projects\RL-LAM-ScanOpt\artifacts\manifests\stage3_run_62_manifest.json`

## 18. Recommended Run63
Update models with combined432 and decide whether to freeze N24/N40 evidence or run a small N40-focused diagnostic follow-up; do not claim full variable-N maturity.
