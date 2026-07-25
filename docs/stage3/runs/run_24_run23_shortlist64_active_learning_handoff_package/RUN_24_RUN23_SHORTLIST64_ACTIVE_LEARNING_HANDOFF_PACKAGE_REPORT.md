# Stage 3 Run 24 - Run23 Shortlist64 Active-Learning Handoff Package

## Purpose
Package the user-selected run23 shortlist64 active-learning candidates for future CAE generation. This run is handoff-only.

## Inputs
- Run23 scored candidate pool: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_23_combined108_active_learning_coverage_calibration_design\run23_candidate_pool_scored.csv`
- Run23 selected shortlist64: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_23_combined108_active_learning_coverage_calibration_design\run23_candidate_shortlist64.csv`
- Run23 batch32 and batch24 were read only as reference inputs and were not packaged as the selected batch.
- Combined108 teacher dataset: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_21_batch28_teacher_metrics_ingestion_and_combined108_ranking\combined108_teacher_dataset.csv`

## User-Selected Batch
- Selected batch: `shortlist64`
- Batch name: `stage3_run24_shortlist64_active_learning_calibration_v01`
- Batch24 and batch32 are explicitly not selected.

## Validation Status
- Verdict: `PASS_RUN24_SHORTLIST64_INPUT_READY`
- Per-N counts: `{12: 8, 16: 8, 24: 24, 40: 24}`
- N24/N40 count: `48`

## Stable Naming Convention
- Format: `S3R24L64_N{N}_B{index:02d}_{short_bucket_or_family}`
- Names are filesystem-safe and preserve original run23 candidate IDs in metadata.

## Shortlist64 Handoff Package
- Candidate order CSV contains stable handoff names, run23 provenance, active-learning bucket metadata, predictions, uncertainty, disagreement, novelty, scan orders, and NOT_RUN teacher status.

## Per-Candidate Scan-Order JSON Outputs
- JSON directory: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders`
- Each JSON is handoff metadata only and is not teacher-validated.

## Future CAE Handoff Template
- The future CAE manifest template lists expected paths only. It does not create CAE case directories and does not generate CAE/INP/JNL files.

## Future abqjobpilot Command Template
- The command template is not executable yet because INP files do not exist.
- Commands must not be run until CAE/INP generation has completed and passed checks.

## Shortlist64 Review Summary
- Total count: `64`
- Per-N counts: `{12: 8, 16: 8, 24: 24, 40: 24}`
- N24/N40 share: `48/64`
- Bucket composition: `{'top_region_local_search': 14, 'model_disagreement': 10, 'uncertainty_calibration': 10, 'diversity_coverage': 10, 'tradeoff_probe': 8, 'sentinel_control': 6, 'exploitation_reference': 6}`
- Expected Abaqus cost: `64 jobs total, with 48 jobs from N24/N40`

## Claim Boundary
RUN24_SHORTLIST64_HANDOFF_ONLY_NO_TEACHER_VALIDATION. No CAE/INP generation, solver execution, abqjobpilot execution, teacher validation, or physical superiority is claimed.

## Output Files
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\stage3_run24_shortlist64_candidate_orders.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N12_B01_top_region.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N12_B02_top_region.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N12_B03_model_disagreement.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N12_B04_uncertainty_calibration.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N12_B05_diversity_coverage.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N12_B06_tradeoff_probe.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N12_B07_sentinel_control.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N12_B08_exploitation_reference.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N16_B01_top_region.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N16_B02_top_region.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N16_B03_model_disagreement.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N16_B04_uncertainty_calibration.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N16_B05_diversity_coverage.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N16_B06_tradeoff_probe.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N16_B07_sentinel_control.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N16_B08_exploitation_reference.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N24_B01_top_region.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N24_B02_top_region.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N24_B03_top_region.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N24_B04_top_region.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N24_B05_top_region.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N24_B06_model_disagreement.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N24_B07_model_disagreement.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N24_B08_model_disagreement.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N24_B09_model_disagreement.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N24_B10_uncertainty_calibration.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N24_B11_uncertainty_calibration.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N24_B12_uncertainty_calibration.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N24_B13_uncertainty_calibration.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N24_B14_diversity_coverage.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N24_B15_diversity_coverage.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N24_B16_diversity_coverage.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N24_B17_diversity_coverage.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N24_B18_tradeoff_probe.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N24_B19_tradeoff_probe.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N24_B20_tradeoff_probe.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N24_B21_sentinel_control.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N24_B22_sentinel_control.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N24_B23_exploitation_reference.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N24_B24_exploitation_reference.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N40_B01_top_region.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N40_B02_top_region.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N40_B03_top_region.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N40_B04_top_region.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N40_B05_top_region.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N40_B06_model_disagreement.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N40_B07_model_disagreement.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N40_B08_model_disagreement.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N40_B09_model_disagreement.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N40_B10_uncertainty_calibration.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N40_B11_uncertainty_calibration.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N40_B12_uncertainty_calibration.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N40_B13_uncertainty_calibration.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N40_B14_diversity_coverage.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N40_B15_diversity_coverage.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N40_B16_diversity_coverage.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N40_B17_diversity_coverage.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N40_B18_tradeoff_probe.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N40_B19_tradeoff_probe.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N40_B20_tradeoff_probe.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N40_B21_sentinel_control.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N40_B22_sentinel_control.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N40_B23_exploitation_reference.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders\scan_order_S3R24L64_N40_B24_exploitation_reference.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\stage3_run24_shortlist64_future_cae_handoff_manifest_TEMPLATE.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\stage3_run24_shortlist64_abqjobpilot_commands_TEMPLATE.txt`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\shortlist64_review_summary.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\shortlist64_review_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\shortlist64_review_summary.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\README_FOR_FUTURE_CAE_GENERATION.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\run24_claim_boundary.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\run24_claim_boundary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\run24_input_validation_summary.json`

## Recommended Run25
CAE module should generate CAE/INP/JNL for selected shortlist64 only. Do not run solver, do not execute abqjobpilot, and do not generate batch24 or batch32.
