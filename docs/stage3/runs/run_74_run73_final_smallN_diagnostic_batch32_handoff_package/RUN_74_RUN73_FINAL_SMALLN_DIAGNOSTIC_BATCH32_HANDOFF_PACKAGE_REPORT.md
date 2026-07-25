# Stage 3 Run 74 - Run73 Final Small-N Diagnostic Batch32 Handoff Package

## 1. Purpose
Run74 packages the selected Run73 Option A final small-N diagnostic batch32 for human review and future CAE generation. It is handoff packaging only.

## 2. Inputs
- Selected Run73 Option A: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_73_combined520_model_update_final_smallN_diagnostic_candidate_generation\run73_final_smallN_diagnostic_batch32_candidate_orders.csv`
- Run73 candidate pool: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_73_combined520_model_update_final_smallN_diagnostic_candidate_generation\run73_candidate_pool_scored.csv`
- Run73 comparison summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_73_combined520_model_update_final_smallN_diagnostic_candidate_generation\run73_batch_options_comparison_summary.json`
- Run73 evidence update: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_73_combined520_model_update_final_smallN_diagnostic_candidate_generation\stage3_evidence_freeze_readiness_after_run72.md`
- Native combined520 RL-ready dataset: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_72_smallN_recovery_focused_batch40_teacher_metrics_ingestion_and_combined520_ranking\combined520_RL_ready_dataset.csv`

## 3. Selected Option A
Selected batch: `run73_final_smallN_diagnostic_batch32`.

Counts: `{12: 14, 16: 14, 24: 2, 40: 2}`.

## 4. Why Option A Was Selected
Option A is the recommended final small-N diagnostic before Stage 3 evidence freeze. It tests N12/N16 stability with enough density while keeping N24/N40 as minimal anchors.

## 5. Why This Is the Final Small-N Diagnostic Before Evidence Freeze
Run73 found Stage 3 evidence close to freeze-ready, but recommended one final small-N diagnostic loop before freezing. Option C remains available later if the user elects to stop validation and freeze immediately.

## 6. Why N12/N16 Remain the Focus
N12/N16 each have 64 native teacher rows, while N24/N40 have much denser mature anchor evidence at 188 and 204 rows respectively.

## 7. Why N24/N40 Are Minimal Anchors
N24/N40 are included only as minimal anchors to preserve continuity with the mature teacher regions, not as the main exploitation target for this handoff.

## 8. Validation Status
Verdict: `PASS_RUN74_FINAL_SMALLN_DIAGNOSTIC_BATCH32_INPUT_READY`.

No N32 rows are included, and exact overlap counts are `{'combined520': 0, 'run71': 0, 'run66': 0, 'run61': 0, 'run56': 0, 'run51': 0, 'run46': 0, 'run41': 0, 'run36': 0, 'run27': 0, 'superseded_run31': 0}`.

## 9. Stable Naming Convention
`S3R74FSD_N{N}_B{index:02d}_{short_bucket_or_family}`

## 10. Candidate-Order Handoff Package
Candidate handoff CSV: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_74_run73_final_smallN_diagnostic_batch32_handoff_package\stage3_run74_final_smallN_diagnostic_batch32_candidate_orders.csv`.

## 11. Per-Candidate Scan-Order JSON Outputs
Scan-order JSON directory: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_74_run73_final_smallN_diagnostic_batch32_handoff_package\scan_orders`.

## 12. Future CAE Handoff Template
Future CAE manifest template: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_74_run73_final_smallN_diagnostic_batch32_handoff_package\stage3_run74_final_smallN_diagnostic_batch32_future_cae_handoff_manifest_TEMPLATE.csv`.

Expected future case root: `E:\Projects\RL-LAM-ScanOpt\cae_model\stage3_run74_final_smallN_diagnostic_batch32_v01`.

## 13. Future abqjobpilot Command Template
Future command template: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_74_run73_final_smallN_diagnostic_batch32_handoff_package\stage3_run74_final_smallN_diagnostic_batch32_abqjobpilot_commands_TEMPLATE.txt`.

This command file is a template only. INPs do not exist yet and the file is not ready to run until CAE/INP generation has completed and passed checks.

## 14. Review Summary
Run74 packages the recommended final small-N diagnostic before Stage 3 evidence freeze: N12/N16 are the focus while N24/N40 are retained only as minimal mature anchors.

## 15. Claim Boundary
Verdict: `RUN74_HANDOFF_ONLY_FINAL_SMALLN_DIAGNOSTIC_BATCH32_NO_TEACHER_VALIDATION`.

## 16. Output Files
- Validation summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_74_run73_final_smallN_diagnostic_batch32_handoff_package\run74_input_validation_summary.json`
- Handoff CSV: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_74_run73_final_smallN_diagnostic_batch32_handoff_package\stage3_run74_final_smallN_diagnostic_batch32_candidate_orders.csv`
- Scan orders: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_74_run73_final_smallN_diagnostic_batch32_handoff_package\scan_orders`
- Future CAE template: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_74_run73_final_smallN_diagnostic_batch32_handoff_package\stage3_run74_final_smallN_diagnostic_batch32_future_cae_handoff_manifest_TEMPLATE.csv`
- Future abqjobpilot template: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_74_run73_final_smallN_diagnostic_batch32_handoff_package\stage3_run74_final_smallN_diagnostic_batch32_abqjobpilot_commands_TEMPLATE.txt`
- Review summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_74_run73_final_smallN_diagnostic_batch32_handoff_package\final_smallN_diagnostic_batch32_review_summary.md`
- Claim boundary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_74_run73_final_smallN_diagnostic_batch32_handoff_package\run74_claim_boundary.md`
- Manifest: `E:\Projects\RL-LAM-ScanOpt\artifacts\manifests\stage3_run_74_manifest.json`

## 17. Recommended Run75
CAE module should generate CAE/INP/JNL for selected Run74 final small-N diagnostic batch32 only. Do not run solver. Do not execute abqjobpilot. Do not generate Run73 Option B batch24. Do not perform Option C stop-and-freeze unless explicitly selected later.
