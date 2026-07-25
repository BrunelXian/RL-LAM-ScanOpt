# Stage 3 Run 72 - Small-N Recovery-Focused Batch40 Teacher Metrics Ingestion and Combined520 Ranking

## 1. Purpose
Run72 ingests Run71 teacher metrics, builds native combined520 and combined520_plus_N32 datasets, and evaluates whether the small-N recovery-focused batch40 improved N12/N16 recovery while preserving mature N24/N40 anchors.

## 2. Inputs
- Run71 teacher metrics: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_71_smallN_recovery_focused_batch40_odb_teacher_validation\run71_smallN_recovery_focused_batch40_teacher_metrics.csv`
- Run69 handoff metadata: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_69_run68_smallN_recovery_focused_batch40_handoff_package\stage3_run69_smallN_recovery_focused_batch40_candidate_orders.csv`
- Native combined480: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_67_variable_N_recovery_anchor_batch48_teacher_metrics_ingestion_and_combined480_ranking\combined480_RL_ready_dataset.csv`
- combined480_plus_N32: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_67_variable_N_recovery_anchor_batch48_teacher_metrics_ingestion_and_combined480_ranking\combined480_plus_N32_RL_ready_dataset.csv`

## 3. Run71 Teacher-Validation Status
Run71 completed 40/40 with N12=16, N16=16, N24=4, and N40=4. Nonfatal warnings were present, with no failed or incomplete cases.

## 4. Input Validation
Verdict: `PASS_RUN72_SMALLN_RECOVERY_FOCUSED_BATCH40_TEACHER_METRICS_40_OF_40_READY`.

## 5. Run71 Enriched Teacher Dataset
Output: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_72_smallN_recovery_focused_batch40_teacher_metrics_ingestion_and_combined520_ranking\run71_smallN_recovery_focused_batch40_teacher_dataset_enriched.csv`

## 6. Run71 Within-Batch Ranking
Output: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_72_smallN_recovery_focused_batch40_teacher_metrics_ingestion_and_combined520_ranking\run71_smallN_recovery_focused_batch40_ranked_within_batch.csv`

## 7. Native Combined520 Construction
Native combined520 counts: `{12: 64, 16: 64, 24: 188, 40: 204}`.

## 8. combined520_plus_N32 Construction
combined520_plus_N32 counts: `{12: 64, 16: 64, 24: 188, 32: 332, 40: 204}`. N32 metric semantic warnings are preserved.

## 9. Run71 vs Combined480 Best Comparison
Run71 created 13 new metric/reward records versus combined480.

## 10. Run71 vs Prior Key Records
Run71 was compared with Run61, Run56, Run51, Run46, Run41, Run36, Run27, and recomputed combined520 best records.

## 11. Small-N Recovery-Focused Batch40 Effectiveness Audit
Run71 produced 13 new metric/reward records versus combined480; the batch directly tests N12/N16 recovery while preserving N24/N40 frozen-anchor coverage.

## 12. Prediction Audit for Run68/Run69 Batch40
Run68/Run69 prediction calibration was evaluated on realized Run71 variable-N labels; use these diagnostics as calibration evidence, not teacher-validation claims for future candidates.

## 13. Small-N Recovery Versus N24/N40 Anchors
N12/N16 recovery is the primary purpose. N24/N40 are retained as mature anchors rather than fresh exploitation targets.

## 14. Updated Full Variable-N Maturity and Claim-Boundary Audit
After Run71, combined520 has N12=64, N16=64, N24=188, and N40=204; full variable-N evidence is stronger, but N12/N16 remain the limiting under-sampled regimes relative to mature N24/N40 anchors.

## 15. Metric Semantic Boundary for N32
N32 rows are legacy-compatible auxiliary data. They are not native Stage 3 teacher validation, and PEEQ/Mises mappings must retain semantic warnings.

## 16. Claim Boundary
Verdict: `RUN72_INGESTION_AND_ANALYSIS_ONLY_NO_SOLVER_OR_CANDIDATE_GENERATION`.

## 17. Output Files
- combined520 RL-ready: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_72_smallN_recovery_focused_batch40_teacher_metrics_ingestion_and_combined520_ranking\combined520_RL_ready_dataset.csv`
- combined520_plus_N32 RL-ready: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_72_smallN_recovery_focused_batch40_teacher_metrics_ingestion_and_combined520_ranking\combined520_plus_N32_RL_ready_dataset.csv`
- Run71 comparison: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_72_smallN_recovery_focused_batch40_teacher_metrics_ingestion_and_combined520_ranking\run71_vs_combined480_best_comparison.csv`
- Updated maturity audit: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_72_smallN_recovery_focused_batch40_teacher_metrics_ingestion_and_combined520_ranking\full_variable_N_updated_maturity_and_claim_boundary_audit.md`
- Manifest: `E:\Projects\RL-LAM-ScanOpt\artifacts\manifests\stage3_run_72_manifest.json`

## 18. Recommended Run73
Update models with combined520 and generate the next N12/N16 recovery-focused batch, keeping N24/N40 as frozen anchors.
