# Stage 3 Run 44 - Run43 Constrained N24/N40 Batch32 Handoff Package

## 1. Purpose
Run44 packages the selected Run43 Option A constrained N24/N40 batch32 for human review and future CAE generation.

## 2. Inputs
- Selected Option A candidate orders: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_43_combined264_constrained_N24_N40_reward_balanced_candidate_generation\run43_constrained_N24_N40_batch32_candidate_orders.csv`
- Run43 candidate pool: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_43_combined264_constrained_N24_N40_reward_balanced_candidate_generation\run43_candidate_pool_scored.csv`
- Run43 comparison table: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_43_combined264_constrained_N24_N40_reward_balanced_candidate_generation\run43_batch_options_comparison_to_previous.csv`
- Run43 report: `E:\Projects\RL-LAM-ScanOpt\docs\stage3\runs\run_43_combined264_constrained_N24_N40_reward_balanced_candidate_generation\RUN_43_COMBINED264_CONSTRAINED_N24_N40_REWARD_BALANCED_CANDIDATE_GENERATION_REPORT.md`

## 3. User-Selected Batch
Selected batch: `constrained_N24_N40_batch32`. The batch contains 32 candidates: N24=16 and N40=16.

## 4. Why Option A Was Selected
Option A is the quick validation loop for the constrained selection rule. It tests U2-primary but reward-balanced candidates after the previous pure U2-near N24/N40 batch60 did not extend U2 bests.

## 5. Validation Status
Verdict: `PASS_RUN44_CONSTRAINED_N24_N40_BATCH32_INPUT_READY`. Only N24/N40 are present; no N12, N16, or N32 candidates are included.

## 6. Stable Naming Convention
Stable handoff names use `S3R44CNS_N{N}_B{index:02d}_{short_bucket}`.

## 7. Candidate-Order Handoff Package
Candidate order CSV: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_44_run43_constrained_N24_N40_batch32_handoff_package\stage3_run44_constrained_N24_N40_batch32_candidate_orders.csv`.

## 8. Per-Candidate Scan-Order JSON Outputs
Scan-order JSON directory: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_44_run43_constrained_N24_N40_batch32_handoff_package\scan_orders`.

## 9. Future CAE Handoff Template
Future CAE manifest template: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_44_run43_constrained_N24_N40_batch32_handoff_package\stage3_run44_constrained_N24_N40_batch32_future_cae_handoff_manifest_TEMPLATE.csv`. Run44 did not create CAE case directories.

## 10. Future abqjobpilot Command Template
Future abqjobpilot template: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_44_run43_constrained_N24_N40_batch32_handoff_package\stage3_run44_constrained_N24_N40_batch32_abqjobpilot_commands_TEMPLATE.txt`. It is a template only and must not be executed until INPs exist and pass checks.

## 11. Review Summary
Run44 packages a quick constrained U2 plus reward-balanced N24/N40 validation loop after pure U2-near batch60 saturation.

Candidate-source composition: `{'reward_balanced_local_search': 4, 'constrained_surrogate_top': 4, 'PEEQ_guarded_candidates': 4, 'SurfaceT_guarded_candidates': 4, 'U2_guarded_local_search': 4, 'hybrid_agreement': 4, 'hybrid_disagreement': 4, 'uncertainty_calibration': 4}`.

Selection-bucket composition: `{'reward_balanced_local_search': 4, 'constrained_surrogate_top': 4, 'PEEQ_guarded_candidates': 4, 'SurfaceT_guarded_candidates': 4, 'U2_guarded_local_search': 4, 'hybrid_agreement': 4, 'hybrid_disagreement': 4, 'uncertainty_calibration': 4}`.

## 12. Claim Boundary
Claim boundary verdict: `RUN44_HANDOFF_ONLY_CONSTRAINED_N24_N40_BATCH32_NO_TEACHER_VALIDATION`.

## 13. Output Files
- Handoff CSV: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_44_run43_constrained_N24_N40_batch32_handoff_package\stage3_run44_constrained_N24_N40_batch32_candidate_orders.csv`
- Scan-order JSON directory: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_44_run43_constrained_N24_N40_batch32_handoff_package\scan_orders`
- Future CAE template: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_44_run43_constrained_N24_N40_batch32_handoff_package\stage3_run44_constrained_N24_N40_batch32_future_cae_handoff_manifest_TEMPLATE.csv`
- Future abqjobpilot template: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_44_run43_constrained_N24_N40_batch32_handoff_package\stage3_run44_constrained_N24_N40_batch32_abqjobpilot_commands_TEMPLATE.txt`
- Review summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_44_run43_constrained_N24_N40_batch32_handoff_package\constrained_N24_N40_batch32_review_summary.md`
- Manifest: `E:\Projects\RL-LAM-ScanOpt\artifacts\manifests\stage3_run_44_manifest.json`

## 14. Recommended Run45
CAE module should generate CAE/INP/JNL for selected Run44 constrained N24/N40 batch32 only. Do not run solver. Do not execute abqjobpilot. Do not generate Option B or Option C unless explicitly selected later.
