# Stage 3 Run 67 - Variable-N Recovery Anchor Batch48 Teacher Metrics Ingestion and Combined480 Ranking

## 1. Purpose
Run67 ingests Run66 teacher metrics, builds native combined480 and combined480_plus_N32 datasets, and evaluates whether the variable-N recovery anchor batch48 improved small-N evidence while preserving mature N24/N40 anchors.

## 2. Inputs
- Run66 teacher metrics: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_66_variable_N_recovery_anchor_batch48_odb_teacher_validation\run66_variable_N_recovery_anchor_batch48_teacher_metrics.csv`
- Run64 handoff metadata: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_64_run63_variable_N_recovery_anchor_batch48_handoff_package\stage3_run64_variable_N_recovery_anchor_batch48_candidate_orders.csv`
- Native combined432: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_62_custom_N40_focused_batch40_teacher_metrics_ingestion_and_combined432_ranking\combined432_RL_ready_dataset.csv`
- combined432_plus_N32: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_62_custom_N40_focused_batch40_teacher_metrics_ingestion_and_combined432_ranking\combined432_plus_N32_RL_ready_dataset.csv`

## 3. Run66 Teacher-Validation Status
Run66 completed 48/48 with N12=12, N16=12, N24=8, and N40=16. Nonfatal warnings were present, with no failed or incomplete cases.

## 4. Input Validation
Verdict: `PASS_RUN67_VARIABLE_N_RECOVERY_ANCHOR_BATCH48_TEACHER_METRICS_48_OF_48_READY`.

## 5. Run66 Enriched Teacher Dataset
Output: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_67_variable_N_recovery_anchor_batch48_teacher_metrics_ingestion_and_combined480_ranking\run66_variable_N_recovery_anchor_batch48_teacher_dataset_enriched.csv`

## 6. Run66 Within-Batch Ranking
Output: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_67_variable_N_recovery_anchor_batch48_teacher_metrics_ingestion_and_combined480_ranking\run66_variable_N_recovery_anchor_batch48_ranked_within_batch.csv`

## 7. Native Combined480 Construction
Native combined480 counts: `{12: 48, 16: 48, 24: 184, 40: 200}`.

## 8. combined480_plus_N32 Construction
combined480_plus_N32 counts: `{12: 48, 16: 48, 24: 184, 32: 332, 40: 200}`. N32 metric semantic warnings are preserved.

## 9. Run66 vs Combined432 Best Comparison
Run66 created 14 new metric/reward records versus combined432.

## 10. Run66 vs Prior Key Records
Run66 was compared with Run61, Run56, Run51, Run46, Run41, Run36, Run27, and recomputed combined480 best records.

## 11. Variable-N Recovery Anchor Batch48 Effectiveness Audit
Run66 produced 14 new metric/reward records versus combined432; the batch directly tests N12/N16 recovery while preserving N24/N40 frozen-anchor coverage.

## 12. Prediction Audit for Run63/Run64 Variable-N Batch48
Run63/Run64 prediction calibration was evaluated on realized Run66 variable-N labels; use these diagnostics as calibration evidence, not teacher-validation claims for future candidates.

## 13. Small-N Recovery Versus N24/N40 Anchors
N12/N16 recovery is the primary purpose. N24/N40 are retained as mature anchors rather than fresh exploitation targets.

## 14. Updated Full Variable-N Maturity and Claim-Boundary Audit
After Run66, combined480 has N12=48, N16=48, N24=184, and N40=200; full variable-N evidence is stronger, but N12/N16 remain the limiting under-sampled regimes relative to mature N24/N40 anchors.

## 15. Metric Semantic Boundary for N32
N32 rows are legacy-compatible auxiliary data. They are not native Stage 3 teacher validation, and PEEQ/Mises mappings must retain semantic warnings.

## 16. Claim Boundary
Verdict: `RUN67_INGESTION_AND_ANALYSIS_ONLY_NO_SOLVER_OR_CANDIDATE_GENERATION`.

## 17. Output Files
- combined480 RL-ready: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_67_variable_N_recovery_anchor_batch48_teacher_metrics_ingestion_and_combined480_ranking\combined480_RL_ready_dataset.csv`
- combined480_plus_N32 RL-ready: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_67_variable_N_recovery_anchor_batch48_teacher_metrics_ingestion_and_combined480_ranking\combined480_plus_N32_RL_ready_dataset.csv`
- Run66 comparison: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_67_variable_N_recovery_anchor_batch48_teacher_metrics_ingestion_and_combined480_ranking\run66_vs_combined432_best_comparison.csv`
- Updated maturity audit: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_67_variable_N_recovery_anchor_batch48_teacher_metrics_ingestion_and_combined480_ranking\full_variable_N_updated_maturity_and_claim_boundary_audit.md`
- Manifest: `E:\Projects\RL-LAM-ScanOpt\artifacts\manifests\stage3_run_67_manifest.json`

## 18. Recommended Run68
Update models with combined480 and generate the next N12/N16 recovery-focused batch, keeping N24/N40 as frozen anchors.
