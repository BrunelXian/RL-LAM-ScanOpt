# Stage 3 Run 59 - Run58 N40-Focused Calibrated Penalty-Repair Batch40 Handoff Package

## 1. Purpose
Run59 creates a handoff package for a custom Run58-derived N40-focused calibrated penalty-repair batch40. It is handoff packaging only.

## 2. Inputs
- Run58 Option A: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_58_combined392_model_update_N24_N40_evidence_freeze_and_N40_focused_candidate_generation\run58_N40_focused_calibrated_penalty_repair_batch32_candidate_orders.csv`
- Run58 candidate pool: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_58_combined392_model_update_N24_N40_evidence_freeze_and_N40_focused_candidate_generation\run58_candidate_pool_scored.csv`
- Run58 evidence freeze: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_58_combined392_model_update_N24_N40_evidence_freeze_and_N40_focused_candidate_generation\n24_n40_active_learning_rl_evidence_freeze.md`

## 3. Custom Selected Batch40
The custom batch contains N24=16 and N40=24, for 40 total candidates. It preserves all valid Run58 Option A rows and adds 8 extra N24 candidates from the existing Run58 candidate pool.

## 4. Why N24 Was Increased From 8 to 16
N24 has 160 native teacher rows and useful diagnostic density, but Run56 did not create new N24 combined328 bests. Increasing N24 from 8 to 16 preserves the N40-heavy plan while giving N24 enough maintenance and diagnostic coverage.

## 5. Why N40 Remains the Majority
Run56 produced the strongest recent signal in N40, including U2 and reward-family improvements. N40 remains the majority target at 24/40 cases.

## 6. Validation Status
Verdict: `PASS_RUN59_N40_FOCUSED_CALIBRATED_PENALTY_REPAIR_BATCH40_INPUT_READY`. Counts: `{24: 16, 40: 24}`. Overlap status: `{'combined392': 0, 'run56': 0, 'run51': 0, 'run46': 0, 'run41': 0, 'run36': 0, 'run27': 0, 'superseded_run31': 0}`.

## 7. Stable Naming Convention
Handoff names use `S3R59N40PR40_N{N}_B{index:02d}_{short_bucket_or_family}`.

## 8. Candidate-Order Handoff Package
CSV: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_59_run58_N40_focused_calibrated_penalty_repair_batch40_handoff_package\stage3_run59_N40_focused_calibrated_penalty_repair_batch40_candidate_orders.csv`

## 9. Per-Candidate Scan-Order JSON Outputs
Directory: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_59_run58_N40_focused_calibrated_penalty_repair_batch40_handoff_package\scan_orders`

## 10. Future CAE Handoff Template
Template: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_59_run58_N40_focused_calibrated_penalty_repair_batch40_handoff_package\stage3_run59_N40_focused_calibrated_penalty_repair_batch40_future_cae_handoff_manifest_TEMPLATE.csv`. Run59 did not create CAE case directories.

## 11. Future abqjobpilot Command Template
Template: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_59_run58_N40_focused_calibrated_penalty_repair_batch40_handoff_package\stage3_run59_N40_focused_calibrated_penalty_repair_batch40_abqjobpilot_commands_TEMPLATE.txt`. It is not ready to execute until future CAE/INP generation has completed and passed checks.

## 12. Review Summary
Custom batch40 preserves all 32 Run58 Option A candidates and adds 8 complementary N24 candidates, yielding N24=16 and N40=24 with zero overlap against combined392 and prior tracked batches.

## 13. Claim Boundary
Verdict: `RUN59_HANDOFF_ONLY_CUSTOM_N40_FOCUSED_BATCH40_NO_TEACHER_VALIDATION`.

## 14. Output Files
- Pre-handoff selected CSV: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_59_run58_N40_focused_calibrated_penalty_repair_batch40_handoff_package\run59_custom_N40_focused_calibrated_penalty_repair_batch40_candidate_orders_PRE_HANDOFF.csv`
- Handoff CSV: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_59_run58_N40_focused_calibrated_penalty_repair_batch40_handoff_package\stage3_run59_N40_focused_calibrated_penalty_repair_batch40_candidate_orders.csv`
- Review summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_59_run58_N40_focused_calibrated_penalty_repair_batch40_handoff_package\N40_focused_calibrated_penalty_repair_batch40_review_summary.md`
- Claim boundary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_59_run58_N40_focused_calibrated_penalty_repair_batch40_handoff_package\run59_claim_boundary.md`
- Manifest: `E:\Projects\RL-LAM-ScanOpt\artifacts\manifests\stage3_run_59_manifest.json`

## 15. Recommended Run60
CAE module should generate CAE/INP/JNL for selected Run59 custom N40-focused calibrated penalty-repair batch40 only. Do not run solver. Do not execute abqjobpilot. Do not generate Run58 Option B batch64 or Option C variable-N recovery anchor batch48 unless explicitly selected later.
