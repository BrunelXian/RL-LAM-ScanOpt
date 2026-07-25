# Stage 3 Run 49 - Run48 Stricter Constrained N24/N40 Batch32 Handoff Package

## 1. Purpose
Run49 packages the selected Run48 Option A stricter constrained N24/N40 batch32 for human review and future CAE generation.

## 2. Inputs
- Selected Option A candidate orders: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_48_combined296_stricter_constrained_N24_N40_candidate_generation\run48_stricter_constrained_N24_N40_batch32_candidate_orders.csv`
- Run48 candidate pool: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_48_combined296_stricter_constrained_N24_N40_candidate_generation\run48_candidate_pool_scored.csv`
- Run48 comparison table: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_48_combined296_stricter_constrained_N24_N40_candidate_generation\run48_batch_options_comparison_to_previous.csv`
- Run48 report: `E:\Projects\RL-LAM-ScanOpt\docs\stage3\runs\run_48_combined296_stricter_constrained_N24_N40_candidate_generation\RUN_48_COMBINED296_STRICTER_CONSTRAINED_N24_N40_CANDIDATE_GENERATION_REPORT.md`

## 3. User-Selected Batch
Selected batch: `stricter_constrained_N24_N40_batch32`. The batch contains 32 candidates: N24=16 and N40=16.

## 4. Why Option A Was Selected
Option A is the quick validation loop for the stricter penalty-guard rule. It tests candidates that keep U2 near the top region while pushing PEEQ, SurfaceT, and Mises guards harder after Run46 improved U2/reward but did not create raw penalty-metric records.

## 5. Validation Status
Verdict: `PASS_RUN49_STRICTER_CONSTRAINED_N24_N40_BATCH32_INPUT_READY`. Only N24/N40 are present; no N12, N16, or N32 candidates are included.

## 6. Stable Naming Convention
Stable handoff names use `S3R49SCN_N{N}_B{index:02d}_{short_bucket}`.

## 7. Candidate-Order Handoff Package
Candidate order CSV: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_49_run48_stricter_constrained_N24_N40_batch32_handoff_package\stage3_run49_stricter_constrained_N24_N40_batch32_candidate_orders.csv`.

## 8. Per-Candidate Scan-Order JSON Outputs
Scan-order JSON directory: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_49_run48_stricter_constrained_N24_N40_batch32_handoff_package\scan_orders`.

## 9. Future CAE Handoff Template
Future CAE manifest template: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_49_run48_stricter_constrained_N24_N40_batch32_handoff_package\stage3_run49_stricter_constrained_N24_N40_batch32_future_cae_handoff_manifest_TEMPLATE.csv`. Run49 did not create CAE case directories.

## 10. Future abqjobpilot Command Template
Future abqjobpilot template: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_49_run48_stricter_constrained_N24_N40_batch32_handoff_package\stage3_run49_stricter_constrained_N24_N40_batch32_abqjobpilot_commands_TEMPLATE.txt`. It is a template only and must not be executed until INPs exist and pass checks.

## 11. Review Summary
Run49 packages a quick stricter penalty-guard N24/N40 validation loop after Run46 improved U2/reward but did not create raw PEEQ/SurfaceT/Mises records.

Candidate-source composition: `{'strict_penalty_guard_top': 4, 'strict_penalty_guard_local_search': 4, 'two_stage_guarded_top': 4, 'two_stage_guarded_local_search': 4, 'no_penalty_worse_than_median': 4, 'PEEQ_repair_candidates': 4, 'SurfaceT_repair_candidates': 4, 'Mises_repair_candidates': 4}`.

Selection-bucket composition: `{'strict_penalty_guard_top': 4, 'strict_penalty_guard_local_search': 4, 'two_stage_guarded_top': 4, 'two_stage_guarded_local_search': 4, 'no_penalty_worse_than_median': 4, 'PEEQ_repair_candidates': 4, 'SurfaceT_repair_candidates': 4, 'Mises_repair_candidates': 4}`.

## 12. Claim Boundary
Claim boundary verdict: `RUN49_HANDOFF_ONLY_STRICTER_CONSTRAINED_N24_N40_BATCH32_NO_TEACHER_VALIDATION`.

## 13. Output Files
- Handoff CSV: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_49_run48_stricter_constrained_N24_N40_batch32_handoff_package\stage3_run49_stricter_constrained_N24_N40_batch32_candidate_orders.csv`
- Scan-order JSON directory: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_49_run48_stricter_constrained_N24_N40_batch32_handoff_package\scan_orders`
- Future CAE template: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_49_run48_stricter_constrained_N24_N40_batch32_handoff_package\stage3_run49_stricter_constrained_N24_N40_batch32_future_cae_handoff_manifest_TEMPLATE.csv`
- Future abqjobpilot template: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_49_run48_stricter_constrained_N24_N40_batch32_handoff_package\stage3_run49_stricter_constrained_N24_N40_batch32_abqjobpilot_commands_TEMPLATE.txt`
- Review summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_49_run48_stricter_constrained_N24_N40_batch32_handoff_package\stricter_constrained_N24_N40_batch32_review_summary.md`
- Manifest: `E:\Projects\RL-LAM-ScanOpt\artifacts\manifests\stage3_run_49_manifest.json`

## 14. Recommended Run50
CAE module should generate CAE/INP/JNL for selected Run49 stricter constrained N24/N40 batch32 only. Do not run solver. Do not execute abqjobpilot. Do not generate Option B batch60 or Option C recovery batch40 unless explicitly selected later.
