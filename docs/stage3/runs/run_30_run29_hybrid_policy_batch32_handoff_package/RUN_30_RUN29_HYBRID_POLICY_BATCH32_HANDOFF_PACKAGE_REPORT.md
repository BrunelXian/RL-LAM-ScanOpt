# Stage 3 Run 30 - Run29 Hybrid-Policy Batch32 Handoff Package

## Purpose
Create a clean handoff package for the selected Run29 hybrid-policy batch32, suitable for future CAE generation after user approval.

## Inputs
- Selected Run29 batch32: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_29_combined172_surrogate_gnn_hybrid_policy_update_and_candidate_generation\run29_hybrid_policy_batch32_candidate_orders.csv`
- Run29 candidate pool: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_29_combined172_surrogate_gnn_hybrid_policy_update_and_candidate_generation\run29_hybrid_candidate_pool_scored.csv`
- Run29 report: `E:\Projects\RL-LAM-ScanOpt\docs\stage3\runs\run_29_combined172_surrogate_gnn_hybrid_policy_update_and_candidate_generation\RUN_29_COMBINED172_SURROGATE_GNN_HYBRID_POLICY_UPDATE_AND_CANDIDATE_GENERATION_REPORT.md`

## User-Selected Batch
- Selected batch: `run29_hybrid_policy_batch32`
- Batch64 and focused batch48 are reference-only and were not packaged as the selected batch.

## Validation Status
- Verdict: `PASS_RUN30_HYBRID_BATCH32_INPUT_READY`
- Per-N counts: `{12: 4, 16: 4, 24: 12, 40: 12}`

## Stable Naming Convention
- Format: `S3R30H32_N{N}_B{index:02d}_{short_bucket_or_family}`
- Batch name: `stage3_run30_hybrid_policy_batch32_v01`

## Hybrid-Policy Batch32 Handoff Package
- Candidate-order CSV rows: `32`
- Metadata preserves Run29 candidate IDs, source, generation method, selection bucket, surrogate/GNN/hybrid scores, disagreement, novelty, and nearest-teacher information where available.

## Per-Candidate Scan-Order JSON Outputs
- Directory: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_30_run29_hybrid_policy_batch32_handoff_package\scan_orders`

## Future CAE Handoff Template
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_30_run29_hybrid_policy_batch32_handoff_package\stage3_run30_hybrid_policy_batch32_future_cae_handoff_manifest_TEMPLATE.csv`
- Template only; no CAE directories or INP files were created.

## Future abqjobpilot Command Template
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_30_run29_hybrid_policy_batch32_handoff_package\stage3_run30_hybrid_policy_batch32_abqjobpilot_commands_TEMPLATE.txt`
- Template only; commands are not executable until INPs exist and pass checks.

## Hybrid Batch32 Review Summary
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_30_run29_hybrid_policy_batch32_handoff_package\hybrid_batch32_review_summary.md`

## Claim Boundary
`RUN30_HYBRID_BATCH32_HANDOFF_ONLY_NO_CAE_NO_TEACHER_VALIDATION`.

## Output Files
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_30_run29_hybrid_policy_batch32_handoff_package\run30_input_validation_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_30_run29_hybrid_policy_batch32_handoff_package\stage3_run30_hybrid_policy_batch32_candidate_orders.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_30_run29_hybrid_policy_batch32_handoff_package\scan_orders\scan_order_S3R30H32_N12_B01_gnn_policy_top.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_30_run29_hybrid_policy_batch32_handoff_package\scan_orders\scan_order_S3R30H32_N12_B02_hybrid_agreement.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_30_run29_hybrid_policy_batch32_handoff_package\scan_orders\scan_order_S3R30H32_N12_B03_hybrid_disagreement.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_30_run29_hybrid_policy_batch32_handoff_package\scan_orders\scan_order_S3R30H32_N12_B04_surrogate_local.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_30_run29_hybrid_policy_batch32_handoff_package\scan_orders\scan_order_S3R30H32_N16_B01_new_best_local.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_30_run29_hybrid_policy_batch32_handoff_package\scan_orders\scan_order_S3R30H32_N16_B02_new_best_local.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_30_run29_hybrid_policy_batch32_handoff_package\scan_orders\scan_order_S3R30H32_N16_B03_new_best_local.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_30_run29_hybrid_policy_batch32_handoff_package\scan_orders\scan_order_S3R30H32_N16_B04_new_best_local.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_30_run29_hybrid_policy_batch32_handoff_package\scan_orders\scan_order_S3R30H32_N24_B01_gnn_policy_top.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_30_run29_hybrid_policy_batch32_handoff_package\scan_orders\scan_order_S3R30H32_N24_B02_gnn_policy_top.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_30_run29_hybrid_policy_batch32_handoff_package\scan_orders\scan_order_S3R30H32_N24_B03_hybrid_agreement.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_30_run29_hybrid_policy_batch32_handoff_package\scan_orders\scan_order_S3R30H32_N24_B04_hybrid_agreement.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_30_run29_hybrid_policy_batch32_handoff_package\scan_orders\scan_order_S3R30H32_N24_B05_hybrid_disagreement.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_30_run29_hybrid_policy_batch32_handoff_package\scan_orders\scan_order_S3R30H32_N24_B06_hybrid_disagreement.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_30_run29_hybrid_policy_batch32_handoff_package\scan_orders\scan_order_S3R30H32_N24_B07_surrogate_local.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_30_run29_hybrid_policy_batch32_handoff_package\scan_orders\scan_order_S3R30H32_N24_B08_surrogate_local.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_30_run29_hybrid_policy_batch32_handoff_package\scan_orders\scan_order_S3R30H32_N24_B09_n24_surfacet.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_30_run29_hybrid_policy_batch32_handoff_package\scan_orders\scan_order_S3R30H32_N24_B10_n24_surfacet.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_30_run29_hybrid_policy_batch32_handoff_package\scan_orders\scan_order_S3R30H32_N24_B11_n24_surfacet.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_30_run29_hybrid_policy_batch32_handoff_package\scan_orders\scan_order_S3R30H32_N24_B12_uncertainty.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_30_run29_hybrid_policy_batch32_handoff_package\scan_orders\scan_order_S3R30H32_N40_B01_new_best_local.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_30_run29_hybrid_policy_batch32_handoff_package\scan_orders\scan_order_S3R30H32_N40_B02_new_best_local.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_30_run29_hybrid_policy_batch32_handoff_package\scan_orders\scan_order_S3R30H32_N40_B03_new_best_local.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_30_run29_hybrid_policy_batch32_handoff_package\scan_orders\scan_order_S3R30H32_N40_B04_new_best_local.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_30_run29_hybrid_policy_batch32_handoff_package\scan_orders\scan_order_S3R30H32_N40_B05_gnn_policy_top.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_30_run29_hybrid_policy_batch32_handoff_package\scan_orders\scan_order_S3R30H32_N40_B06_gnn_policy_top.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_30_run29_hybrid_policy_batch32_handoff_package\scan_orders\scan_order_S3R30H32_N40_B07_hybrid_disagreement.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_30_run29_hybrid_policy_batch32_handoff_package\scan_orders\scan_order_S3R30H32_N40_B08_hybrid_disagreement.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_30_run29_hybrid_policy_batch32_handoff_package\scan_orders\scan_order_S3R30H32_N40_B09_surrogate_local.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_30_run29_hybrid_policy_batch32_handoff_package\scan_orders\scan_order_S3R30H32_N40_B10_surrogate_local.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_30_run29_hybrid_policy_batch32_handoff_package\scan_orders\scan_order_S3R30H32_N40_B11_uncertainty.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_30_run29_hybrid_policy_batch32_handoff_package\scan_orders\scan_order_S3R30H32_N40_B12_uncertainty.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_30_run29_hybrid_policy_batch32_handoff_package\stage3_run30_hybrid_policy_batch32_future_cae_handoff_manifest_TEMPLATE.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_30_run29_hybrid_policy_batch32_handoff_package\stage3_run30_hybrid_policy_batch32_abqjobpilot_commands_TEMPLATE.txt`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_30_run29_hybrid_policy_batch32_handoff_package\hybrid_batch32_review_summary.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_30_run29_hybrid_policy_batch32_handoff_package\hybrid_batch32_review_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_30_run29_hybrid_policy_batch32_handoff_package\hybrid_batch32_review_summary.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_30_run29_hybrid_policy_batch32_handoff_package\run30_claim_boundary.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_30_run29_hybrid_policy_batch32_handoff_package\run30_claim_boundary.json`

## Recommended Run31
CAE module should generate CAE/INP/JNL for selected hybrid-policy batch32 only. Do not run solver, do not execute abqjobpilot, and do not generate hybrid batch64 or focused batch48 unless explicitly selected later.
