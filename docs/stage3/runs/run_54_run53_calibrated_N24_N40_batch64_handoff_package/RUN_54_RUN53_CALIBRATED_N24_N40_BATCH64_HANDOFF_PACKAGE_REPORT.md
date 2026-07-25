# Stage 3 Run 54 - Run53 Calibrated N24/N40 Batch64 Handoff Package

## 1. Purpose
Run54 packages the selected Run53 primary calibrated N24/N40 batch64 for human review and future CAE generation.

## 2. Inputs
- Selected primary batch64 candidate orders: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_53_combined328_calibrated_N24_N40_batch64_candidate_generation\run53_calibrated_N24_N40_batch64_candidate_orders.csv`
- Run53 candidate pool: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_53_combined328_calibrated_N24_N40_batch64_candidate_generation\run53_candidate_pool_scored.csv`
- Run53 comparison table: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_53_combined328_calibrated_N24_N40_batch64_candidate_generation\run53_batch64_comparison_to_previous.csv`
- Run53 report: `E:\Projects\RL-LAM-ScanOpt\docs\stage3\runs\run_53_combined328_calibrated_N24_N40_batch64_candidate_generation\RUN_53_COMBINED328_CALIBRATED_N24_N40_BATCH64_CANDIDATE_GENERATION_REPORT.md`

## 3. User-Selected Primary Batch64
Selected batch: `calibrated_N24_N40_batch64`. The batch contains 64 candidates: N24=32 and N40=32.

## 4. Why Batch64 Was Selected
The user explicitly selected an overnight batch64. The batch expands Run53 calibrated candidate generation after Run51 improved N24 U2 and N40 strict/reward behavior but still did not create raw PEEQ, SurfaceT, or Mises records.

## 5. Validation Status
Verdict: `PASS_RUN54_CALIBRATED_N24_N40_BATCH64_INPUT_READY`. Only N24/N40 are present; no N12, N16, or N32 candidates are included.

## 6. Stable Naming Convention
Stable handoff names use `S3R54CAL64_N{N}_B{index:02d}_{short_bucket}`.

## 7. Candidate-Order Handoff Package
Candidate order CSV: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_54_run53_calibrated_N24_N40_batch64_handoff_package\stage3_run54_calibrated_N24_N40_batch64_candidate_orders.csv`.

## 8. Per-Candidate Scan-Order JSON Outputs
Scan-order JSON directory: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_54_run53_calibrated_N24_N40_batch64_handoff_package\scan_orders`.

## 9. Future CAE Handoff Template
Future CAE manifest template: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_54_run53_calibrated_N24_N40_batch64_handoff_package\stage3_run54_calibrated_N24_N40_batch64_future_cae_handoff_manifest_TEMPLATE.csv`. Run54 did not create CAE case directories.

## 10. Future abqjobpilot Command Template
Future abqjobpilot template: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_54_run53_calibrated_N24_N40_batch64_handoff_package\stage3_run54_calibrated_N24_N40_batch64_abqjobpilot_commands_TEMPLATE.txt`. It is a template only and must not be executed until INPs exist and pass checks.

## 11. Review Summary
Run54 packages the user-selected overnight calibrated N24/N40 batch64 with N24 U2-retention, N40 strict/reward-retention, penalty-repair, uncertainty, diversity, and sentinel/control coverage.

Candidate-source composition: `{'two_stage_penalty_repair': 10, 'penalty_repair_local_search': 6, 'penalty_repair_top': 6, 'PEEQ_repair_candidates': 6, 'N24_u2_retention_local_repair': 4, 'median_guard_repair': 4, 'N24_u2_retention_top': 4, 'no_penalty_worse_than_median': 4, 'SurfaceT_repair_candidates': 4, 'Mises_repair_candidates': 4, 'N40_strict_reward_retention_top': 4, 'N40_strict_reward_local_repair': 4, 'uncertainty_calibration': 2, 'sentinel_control': 1, 'diversity_coverage': 1}`.

Selection-bucket composition: `{'two_stage_penalty_repair': 10, 'penalty_repair_local_search': 6, 'penalty_repair_top': 6, 'PEEQ_repair_candidates': 6, 'N24_u2_retention_local_repair': 4, 'median_guard_repair': 4, 'N24_u2_retention_top': 4, 'no_penalty_worse_than_median': 4, 'SurfaceT_repair_candidates': 4, 'Mises_repair_candidates': 4, 'N40_strict_reward_retention_top': 4, 'N40_strict_reward_local_repair': 4, 'uncertainty_calibration': 2, 'sentinel_control': 1, 'diversity_coverage': 1}`.

## 12. Claim Boundary
Claim boundary verdict: `RUN54_HANDOFF_ONLY_CALIBRATED_N24_N40_BATCH64_NO_TEACHER_VALIDATION`.

## 13. Output Files
- Handoff CSV: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_54_run53_calibrated_N24_N40_batch64_handoff_package\stage3_run54_calibrated_N24_N40_batch64_candidate_orders.csv`
- Scan-order JSON directory: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_54_run53_calibrated_N24_N40_batch64_handoff_package\scan_orders`
- Future CAE template: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_54_run53_calibrated_N24_N40_batch64_handoff_package\stage3_run54_calibrated_N24_N40_batch64_future_cae_handoff_manifest_TEMPLATE.csv`
- Future abqjobpilot template: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_54_run53_calibrated_N24_N40_batch64_handoff_package\stage3_run54_calibrated_N24_N40_batch64_abqjobpilot_commands_TEMPLATE.txt`
- Review summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_54_run53_calibrated_N24_N40_batch64_handoff_package\calibrated_N24_N40_batch64_review_summary.md`
- Manifest: `E:\Projects\RL-LAM-ScanOpt\artifacts\manifests\stage3_run_54_manifest.json`

## 14. Recommended Run55
CAE module should generate CAE/INP/JNL for selected Run54 calibrated N24/N40 batch64 only. Do not run solver. Do not execute abqjobpilot. Do not generate the Run53 reference batch32 or reference recovery batch40 unless explicitly selected later.
