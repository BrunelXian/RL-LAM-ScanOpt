# Stage 3 Run 34 - Run33 N32-Informed Native Batch32 Handoff Package

## Purpose
Create a clean handoff package for selected Run33 Option A, suitable for future CAE generation after user approval.

## Inputs
- Selected Option A candidate orders: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_33_combined172_plus_N32_balanced_surrogate_gnn_candidate_generation\run33_N32_informed_native_batch32_candidate_orders.csv`
- Run33 candidate pool: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_33_combined172_plus_N32_balanced_surrogate_gnn_candidate_generation\run33_candidate_pool_scored.csv`
- Run33 report: `E:\Projects\RL-LAM-ScanOpt\docs\stage3\runs\run_33_combined172_plus_N32_balanced_surrogate_gnn_candidate_generation\RUN_33_COMBINED172_PLUS_N32_BALANCED_SURROGATE_GNN_CANDIDATE_GENERATION_REPORT.md`
- Run32A report: `E:\Projects\RL-LAM-ScanOpt\docs\stage3\runs\run_32a_stage2_n32_legacy_teacher_label_ingestion_for_stage3\RUN_32A_STAGE2_N32_LEGACY_TEACHER_LABEL_INGESTION_FOR_STAGE3_REPORT.md`

## User-Selected Batch
- Selected batch: `Run33 Option A - N32-informed native batch32`.
- Option B and Option C are not packaged as the selected batch.

## Why Option A Was Selected
Run33 showed N32 augmentation improved GNN relative to Run29 but did not clearly improve native Stage 3 surrogate performance overall. Option A uses N32-informed modelling while testing only native Stage 3 N values.

## Validation Status
- Verdict: `PASS_RUN34_N32_INFORMED_NATIVE_BATCH32_INPUT_READY`
- Counts: `{12: 4, 16: 4, 24: 12, 40: 12}`
- No N32 candidates are included.

## Stable Naming Convention
- `S3R34N32INF_N{N}_B{index:02d}_{short_bucket_or_family}`

## Candidate-Order Handoff Package
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_34_run33_N32_informed_native_batch32_handoff_package\stage3_run34_N32_informed_native_batch32_candidate_orders.csv`
- Rows: `32`

## Per-Candidate Scan-Order JSON Outputs
- Directory: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_34_run33_N32_informed_native_batch32_handoff_package\scan_orders`

## Future CAE Handoff Template
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_34_run33_N32_informed_native_batch32_handoff_package\stage3_run34_N32_informed_native_batch32_future_cae_handoff_manifest_TEMPLATE.csv`
- Template only; no CAE directories or INP files were created.

## Future abqjobpilot Command Template
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_34_run33_N32_informed_native_batch32_handoff_package\stage3_run34_N32_informed_native_batch32_abqjobpilot_commands_TEMPLATE.txt`
- Template only; not ready to execute until future CAE/INP generation has completed and passed checks.

## Review Summary
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_34_run33_N32_informed_native_batch32_handoff_package\N32_informed_native_batch32_review_summary.md`

## Superseded Run31 Note
- Superseded note exists: `True`
- The old Run31 READY_TO_RUN commands remain superseded and must not be enqueued unless explicitly re-approved later.

## Claim Boundary
`RUN34_N32_INFORMED_NATIVE_BATCH32_HANDOFF_ONLY_NO_CAE_NO_TEACHER_VALIDATION`.

## Output Files
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_34_run33_N32_informed_native_batch32_handoff_package\run34_input_validation_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_34_run33_N32_informed_native_batch32_handoff_package\stage3_run34_N32_informed_native_batch32_candidate_orders.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_34_run33_N32_informed_native_batch32_handoff_package\scan_orders\scan_order_S3R34N32INF_N12_B01_graph_pointer.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_34_run33_N32_informed_native_batch32_handoff_package\scan_orders\scan_order_S3R34N32INF_N12_B02_surrogate_top.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_34_run33_N32_informed_native_batch32_handoff_package\scan_orders\scan_order_S3R34N32INF_N12_B03_hybrid_agree.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_34_run33_N32_informed_native_batch32_handoff_package\scan_orders\scan_order_S3R34N32INF_N12_B04_hybrid_disagree.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_34_run33_N32_informed_native_batch32_handoff_package\scan_orders\scan_order_S3R34N32INF_N16_B01_graph_pointer.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_34_run33_N32_informed_native_batch32_handoff_package\scan_orders\scan_order_S3R34N32INF_N16_B02_surrogate_top.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_34_run33_N32_informed_native_batch32_handoff_package\scan_orders\scan_order_S3R34N32INF_N16_B03_hybrid_agree.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_34_run33_N32_informed_native_batch32_handoff_package\scan_orders\scan_order_S3R34N32INF_N16_B04_hybrid_disagree.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_34_run33_N32_informed_native_batch32_handoff_package\scan_orders\scan_order_S3R34N32INF_N24_B01_graph_pointer.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_34_run33_N32_informed_native_batch32_handoff_package\scan_orders\scan_order_S3R34N32INF_N24_B02_surrogate_top.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_34_run33_N32_informed_native_batch32_handoff_package\scan_orders\scan_order_S3R34N32INF_N24_B03_hybrid_agree.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_34_run33_N32_informed_native_batch32_handoff_package\scan_orders\scan_order_S3R34N32INF_N24_B04_hybrid_disagree.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_34_run33_N32_informed_native_batch32_handoff_package\scan_orders\scan_order_S3R34N32INF_N24_B05_uncertainty.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_34_run33_N32_informed_native_batch32_handoff_package\scan_orders\scan_order_S3R34N32INF_N24_B06_diversity.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_34_run33_N32_informed_native_batch32_handoff_package\scan_orders\scan_order_S3R34N32INF_N24_B07_sentinel.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_34_run33_N32_informed_native_batch32_handoff_package\scan_orders\scan_order_S3R34N32INF_N24_B08_n24_calibration.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_34_run33_N32_informed_native_batch32_handoff_package\scan_orders\scan_order_S3R34N32INF_N24_B09_n24_calibration.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_34_run33_N32_informed_native_batch32_handoff_package\scan_orders\scan_order_S3R34N32INF_N24_B10_n24_calibration.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_34_run33_N32_informed_native_batch32_handoff_package\scan_orders\scan_order_S3R34N32INF_N24_B11_n24_calibration.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_34_run33_N32_informed_native_batch32_handoff_package\scan_orders\scan_order_S3R34N32INF_N24_B12_n24_calibration.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_34_run33_N32_informed_native_batch32_handoff_package\scan_orders\scan_order_S3R34N32INF_N40_B01_graph_pointer.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_34_run33_N32_informed_native_batch32_handoff_package\scan_orders\scan_order_S3R34N32INF_N40_B02_surrogate_top.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_34_run33_N32_informed_native_batch32_handoff_package\scan_orders\scan_order_S3R34N32INF_N40_B03_hybrid_agree.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_34_run33_N32_informed_native_batch32_handoff_package\scan_orders\scan_order_S3R34N32INF_N40_B04_hybrid_disagree.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_34_run33_N32_informed_native_batch32_handoff_package\scan_orders\scan_order_S3R34N32INF_N40_B05_uncertainty.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_34_run33_N32_informed_native_batch32_handoff_package\scan_orders\scan_order_S3R34N32INF_N40_B06_diversity.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_34_run33_N32_informed_native_batch32_handoff_package\scan_orders\scan_order_S3R34N32INF_N40_B07_sentinel.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_34_run33_N32_informed_native_batch32_handoff_package\scan_orders\scan_order_S3R34N32INF_N40_B08_n40_best_near.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_34_run33_N32_informed_native_batch32_handoff_package\scan_orders\scan_order_S3R34N32INF_N40_B09_n40_best_near.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_34_run33_N32_informed_native_batch32_handoff_package\scan_orders\scan_order_S3R34N32INF_N40_B10_n40_best_near.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_34_run33_N32_informed_native_batch32_handoff_package\scan_orders\scan_order_S3R34N32INF_N40_B11_n40_best_near.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_34_run33_N32_informed_native_batch32_handoff_package\scan_orders\scan_order_S3R34N32INF_N40_B12_n40_best_near.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_34_run33_N32_informed_native_batch32_handoff_package\stage3_run34_N32_informed_native_batch32_future_cae_handoff_manifest_TEMPLATE.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_34_run33_N32_informed_native_batch32_handoff_package\stage3_run34_N32_informed_native_batch32_abqjobpilot_commands_TEMPLATE.txt`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_34_run33_N32_informed_native_batch32_handoff_package\N32_informed_native_batch32_review_summary.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_34_run33_N32_informed_native_batch32_handoff_package\N32_informed_native_batch32_review_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_34_run33_N32_informed_native_batch32_handoff_package\N32_informed_native_batch32_review_summary.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_34_run33_N32_informed_native_batch32_handoff_package\run34_claim_boundary.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_34_run33_N32_informed_native_batch32_handoff_package\run34_claim_boundary.json`

## Recommended Run35
CAE module should generate CAE/INP/JNL for selected Run34 N32-informed native batch32 only. Do not run solver, do not execute abqjobpilot, and do not generate Option B or Option C unless explicitly selected later.
