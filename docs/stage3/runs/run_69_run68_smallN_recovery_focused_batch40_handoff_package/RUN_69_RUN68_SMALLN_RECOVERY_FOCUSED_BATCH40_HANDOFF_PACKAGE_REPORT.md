# Stage 3 Run 69 - Run68 Small-N Recovery-Focused Batch40 Handoff Package

## 1. Purpose
Run69 packages the selected Run68 Option A small-N recovery-focused batch40 for human review and future CAE generation. It is handoff packaging only.

## 2. Inputs
- Selected Run68 Option A: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_68_combined480_model_update_smallN_recovery_candidate_generation\run68_smallN_recovery_focused_batch40_candidate_orders.csv`
- Run68 candidate pool: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_68_combined480_model_update_smallN_recovery_candidate_generation\run68_candidate_pool_scored.csv`
- Run68 comparison summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_68_combined480_model_update_smallN_recovery_candidate_generation\run68_batch_options_comparison_summary.json`
- Run68 evidence update: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_68_combined480_model_update_smallN_recovery_candidate_generation\full_variable_N_evidence_update_after_run67.md`
- Native combined480 RL-ready dataset: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_67_variable_N_recovery_anchor_batch48_teacher_metrics_ingestion_and_combined480_ranking\combined480_RL_ready_dataset.csv`

## 3. Selected Option A
Selected batch: `run68_smallN_recovery_focused_batch40`.

Counts: `{12: 16, 16: 16, 24: 4, 40: 4}`.

## 4. Why Option A Was Selected
Option A most directly repairs the remaining N12/N16 evidence gap while keeping N24/N40 as mature anchors.

## 5. Why N12/N16 Recovery Remains Primary
Run68 identified small-N recovery as the primary next objective. N12/N16 each have 48 native teacher rows, while N24/N40 have much denser mature anchor evidence.

## 6. Why N24/N40 Are Anchors Rather Than Exploitation Targets
N24/N40 are included only as minimal anchors to preserve continuity with the mature teacher regions, not as the main exploitation target for this handoff.

## 7. Validation Status
Verdict: `PASS_RUN69_SMALLN_RECOVERY_FOCUSED_BATCH40_INPUT_READY`.

No N32 rows are included, and exact overlap counts are `{'combined480': 0, 'run66': 0, 'run61': 0, 'run56': 0, 'run51': 0, 'run46': 0, 'run41': 0, 'run36': 0, 'run27': 0, 'superseded_run31': 0}`.

## 8. Stable Naming Convention
`S3R69SNR_N{N}_B{index:02d}_{short_bucket_or_family}`

## 9. Candidate-Order Handoff Package
Candidate handoff CSV: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_69_run68_smallN_recovery_focused_batch40_handoff_package\stage3_run69_smallN_recovery_focused_batch40_candidate_orders.csv`.

## 10. Per-Candidate Scan-Order JSON Outputs
Scan-order JSON directory: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_69_run68_smallN_recovery_focused_batch40_handoff_package\scan_orders`.

## 11. Future CAE Handoff Template
Future CAE manifest template: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_69_run68_smallN_recovery_focused_batch40_handoff_package\stage3_run69_smallN_recovery_focused_batch40_future_cae_handoff_manifest_TEMPLATE.csv`.

Expected future case root: `E:\Projects\RL-LAM-ScanOpt\cae_model\stage3_run69_smallN_recovery_focused_batch40_v01`.

## 12. Future abqjobpilot Command Template
Future command template: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_69_run68_smallN_recovery_focused_batch40_handoff_package\stage3_run69_smallN_recovery_focused_batch40_abqjobpilot_commands_TEMPLATE.txt`.

This command file is a template only. INPs do not exist yet and the file is not ready to run until CAE/INP generation has completed and passed checks.

## 13. Review Summary
Run69 packages the recommended small-N recovery follow-up after Run67/Run68: N12/N16 recovery is primary while N24/N40 are retained only as mature anchors.

## 14. Claim Boundary
Verdict: `RUN69_HANDOFF_ONLY_SMALLN_RECOVERY_FOCUSED_BATCH40_NO_TEACHER_VALIDATION`.

## 15. Output Files
- Validation summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_69_run68_smallN_recovery_focused_batch40_handoff_package\run69_input_validation_summary.json`
- Handoff CSV: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_69_run68_smallN_recovery_focused_batch40_handoff_package\stage3_run69_smallN_recovery_focused_batch40_candidate_orders.csv`
- Scan orders: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_69_run68_smallN_recovery_focused_batch40_handoff_package\scan_orders`
- Future CAE template: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_69_run68_smallN_recovery_focused_batch40_handoff_package\stage3_run69_smallN_recovery_focused_batch40_future_cae_handoff_manifest_TEMPLATE.csv`
- Future abqjobpilot template: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_69_run68_smallN_recovery_focused_batch40_handoff_package\stage3_run69_smallN_recovery_focused_batch40_abqjobpilot_commands_TEMPLATE.txt`
- Review summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_69_run68_smallN_recovery_focused_batch40_handoff_package\smallN_recovery_focused_batch40_review_summary.md`
- Claim boundary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_69_run68_smallN_recovery_focused_batch40_handoff_package\run69_claim_boundary.md`
- Manifest: `E:\Projects\RL-LAM-ScanOpt\artifacts\manifests\stage3_run_69_manifest.json`

## 16. Recommended Run70
CAE module should generate CAE/INP/JNL for selected Run69 small-N recovery-focused batch40 only. Do not run solver. Do not execute abqjobpilot. Do not generate Run68 Option B or Option C unless explicitly selected later.
