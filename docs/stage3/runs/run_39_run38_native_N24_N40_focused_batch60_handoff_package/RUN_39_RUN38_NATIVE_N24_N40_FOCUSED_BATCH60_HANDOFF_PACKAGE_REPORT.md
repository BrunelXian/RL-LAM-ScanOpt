# Stage 3 Run 39 - Run38 Native N24/N40 Focused Batch60 Handoff Package

## 1. Purpose
Package a Run38-derived native N24/N40 focused batch60 for human review and future CAE generation.

## 2. Inputs
- Run38 scored pool: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_38_combined204_and_combined204_plus_N32_model_update_candidate_generation\run38_candidate_pool_scored.csv`
- Run38 Option B reference: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_38_combined204_and_combined204_plus_N32_model_update_candidate_generation\run38_native_N24_N40_focused_batch32_candidate_orders.csv`
- Native combined204 teacher dataset: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_37_N32_informed_native_batch32_teacher_metrics_ingestion_and_combined204_ranking\combined204_teacher_dataset.csv`

## 3. User-Selected Batch60
- Batch name: `stage3_run39_native_N24_N40_focused_batch60_v01`
- Selected counts: N24=30, N40=30.

## 4. Why Batch60 Was Selected
The user is running overnight and wants more coverage than the smaller Run38 batch32. Run36 refreshed N24/N40 U2 bests, so Run39 presses those two N values directly.

## 5. Why Only N24/N40 Are Included
Native combined204 remains the strongest Run38 surrogate regime, and the current scientific pressure point is N24/N40 U2 exploitation and calibration. N12, N16, and N32 are excluded from this selected batch.

## 6. Validation Status
- Verdict: `PASS_RUN39_NATIVE_N24_N40_FOCUSED_BATCH60_POOL_READY`
- Pool counts: `{12: 800, 16: 800, 24: 3000, 32: 1352, 40: 3000}`

## 7. Stable Naming Convention
`S3R39N2440B60_N{N}_B{index:02d}_{short_bucket}`

## 8. Candidate-Order Handoff Package
- Candidate CSV: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_39_run38_native_N24_N40_focused_batch60_handoff_package\stage3_run39_native_N24_N40_focused_batch60_candidate_orders.csv`

## 9. Per-Candidate Scan-Order JSON Outputs
- Directory: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_39_run38_native_N24_N40_focused_batch60_handoff_package\scan_orders`

## 10. Future CAE Handoff Template
- Template: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_39_run38_native_N24_N40_focused_batch60_handoff_package\stage3_run39_native_N24_N40_focused_batch60_future_cae_handoff_manifest_TEMPLATE.csv`

## 11. Future abqjobpilot Command Template
- Template: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_39_run38_native_N24_N40_focused_batch60_handoff_package\stage3_run39_native_N24_N40_focused_batch60_abqjobpilot_commands_TEMPLATE.txt`

## 12. Review Summary
- Selected batch60 focuses directly on N24/N40 U2 gains from Run36 and has no exact overlap with the checked prior/reference sets.

## 13. Claim Boundary
- Run39 is handoff packaging only. No CAE/INP, no solver activity, and no teacher validation were performed.

## 14. Output Files
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_39_run38_native_N24_N40_focused_batch60_handoff_package\run39_input_validation_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_39_run38_native_N24_N40_focused_batch60_handoff_package\stage3_run39_native_N24_N40_focused_batch60_candidate_orders.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_39_run38_native_N24_N40_focused_batch60_handoff_package\stage3_run39_native_N24_N40_focused_batch60_future_cae_handoff_manifest_TEMPLATE.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_39_run38_native_N24_N40_focused_batch60_handoff_package\stage3_run39_native_N24_N40_focused_batch60_abqjobpilot_commands_TEMPLATE.txt`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_39_run38_native_N24_N40_focused_batch60_handoff_package\native_N24_N40_focused_batch60_review_summary.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_39_run38_native_N24_N40_focused_batch60_handoff_package\native_N24_N40_focused_batch60_review_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_39_run38_native_N24_N40_focused_batch60_handoff_package\native_N24_N40_focused_batch60_review_summary.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_39_run38_native_N24_N40_focused_batch60_handoff_package\run39_claim_boundary.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_39_run38_native_N24_N40_focused_batch60_handoff_package\run39_claim_boundary.json`
- `E:\Projects\RL-LAM-ScanOpt\docs\stage3\runs\run_39_run38_native_N24_N40_focused_batch60_handoff_package\RUN_39_RUN38_NATIVE_N24_N40_FOCUSED_BATCH60_HANDOFF_PACKAGE_REPORT.md`
- `E:\Projects\RL-LAM-ScanOpt\artifacts\manifests\stage3_run_39_manifest.json`

## 15. Recommended Run40
CAE module should generate CAE/INP/JNL for selected Run39 native N24/N40 focused batch60 only. Do not run solver. Do not execute abqjobpilot. Do not generate smaller batch32 unless explicitly selected later.
