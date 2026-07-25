# Stage 3 Run 64 - Run63 Variable-N Recovery Anchor Batch48 Handoff Package

## 1. Purpose
Run64 packages the selected Run63 Option A variable-N recovery anchor batch48 for human review and future CAE generation. It is handoff packaging only.

## 2. Inputs
- Selected Run63 Option A: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_63_combined432_model_update_N24_N40_evidence_freeze_and_N12_N16_recovery_candidate_generation\run63_variable_N_recovery_anchor_batch48_candidate_orders.csv`
- Run63 candidate pool: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_63_combined432_model_update_N24_N40_evidence_freeze_and_N12_N16_recovery_candidate_generation\run63_candidate_pool_scored.csv`
- Run63 comparison summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_63_combined432_model_update_N24_N40_evidence_freeze_and_N12_N16_recovery_candidate_generation\run63_batch_options_comparison_summary.json`
- Run63 evidence freeze: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_63_combined432_model_update_N24_N40_evidence_freeze_and_N12_N16_recovery_candidate_generation\n24_n40_final_active_learning_rl_evidence_freeze.md`
- Native combined432 RL-ready dataset: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_62_custom_N40_focused_batch40_teacher_metrics_ingestion_and_combined432_ranking\combined432_RL_ready_dataset.csv`

## 3. Selected Option A
Selected batch: `run63_variable_N_recovery_anchor_batch48`.

Counts: `{12: 12, 16: 12, 24: 8, 40: 16}`.

## 4. Why Option A Was Selected
Option A directly addresses the full variable-N claim boundary by adding N12/N16 recovery evidence while retaining mature N24/N40 anchors.

## 5. Why N12/N16 Recovery Is Now Primary
Run63 froze N24/N40 evidence at dense native teacher counts, while N12/N16 remain at 36 rows each. N12/N16 recovery is therefore the active bottleneck for stronger variable-N claims.

## 6. Why N24/N40 Are Anchors Rather Than Exploitation Targets
N24/N40 are included as anchors to maintain continuity with mature evidence, not as the main exploitation target for this handoff.

## 7. Validation Status
Verdict: `PASS_RUN64_VARIABLE_N_RECOVERY_ANCHOR_BATCH48_INPUT_READY`.

No N32 rows are included, and exact overlap counts are `{'combined432': 0, 'run61': 0, 'run56': 0, 'run51': 0, 'run46': 0, 'run41': 0, 'run36': 0, 'run27': 0, 'superseded_run31': 0}`.

## 8. Stable Naming Convention
`S3R64VNR_N{N}_B{index:02d}_{short_bucket_or_family}`

## 9. Candidate-Order Handoff Package
Candidate handoff CSV: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_64_run63_variable_N_recovery_anchor_batch48_handoff_package\stage3_run64_variable_N_recovery_anchor_batch48_candidate_orders.csv`.

## 10. Per-Candidate Scan-Order JSON Outputs
Scan-order JSON directory: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_64_run63_variable_N_recovery_anchor_batch48_handoff_package\scan_orders`.

## 11. Future CAE Handoff Template
Future CAE manifest template: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_64_run63_variable_N_recovery_anchor_batch48_handoff_package\stage3_run64_variable_N_recovery_anchor_batch48_future_cae_handoff_manifest_TEMPLATE.csv`.

Expected future case root: `E:\Projects\RL-LAM-ScanOpt\cae_model\stage3_run64_variable_N_recovery_anchor_batch48_v01`.

## 12. Future abqjobpilot Command Template
Future command template: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_64_run63_variable_N_recovery_anchor_batch48_handoff_package\stage3_run64_variable_N_recovery_anchor_batch48_abqjobpilot_commands_TEMPLATE.txt`.

This command file is a template only. INPs do not exist yet and the file is not ready to run until CAE/INP generation has completed and passed checks.

## 13. Review Summary
Run64 packages the recommended full variable-N recovery follow-up after N24/N40 evidence freeze: N12/N16 recovery is primary while N24/N40 mature anchors are preserved.

## 14. Claim Boundary
Verdict: `RUN64_HANDOFF_ONLY_VARIABLE_N_RECOVERY_ANCHOR_BATCH48_NO_TEACHER_VALIDATION`.

## 15. Output Files
- Validation summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_64_run63_variable_N_recovery_anchor_batch48_handoff_package\run64_input_validation_summary.json`
- Handoff CSV: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_64_run63_variable_N_recovery_anchor_batch48_handoff_package\stage3_run64_variable_N_recovery_anchor_batch48_candidate_orders.csv`
- Scan orders: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_64_run63_variable_N_recovery_anchor_batch48_handoff_package\scan_orders`
- Future CAE template: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_64_run63_variable_N_recovery_anchor_batch48_handoff_package\stage3_run64_variable_N_recovery_anchor_batch48_future_cae_handoff_manifest_TEMPLATE.csv`
- Future abqjobpilot template: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_64_run63_variable_N_recovery_anchor_batch48_handoff_package\stage3_run64_variable_N_recovery_anchor_batch48_abqjobpilot_commands_TEMPLATE.txt`
- Review summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_64_run63_variable_N_recovery_anchor_batch48_handoff_package\variable_N_recovery_anchor_batch48_review_summary.md`
- Claim boundary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_64_run63_variable_N_recovery_anchor_batch48_handoff_package\run64_claim_boundary.md`
- Manifest: `E:\Projects\RL-LAM-ScanOpt\artifacts\manifests\stage3_run_64_manifest.json`

## 16. Recommended Run65
CAE module should generate CAE/INP/JNL for selected Run64 variable-N recovery anchor batch48 only. Do not run solver. Do not execute abqjobpilot. Do not generate Run63 Option B or Option C unless explicitly selected later.
