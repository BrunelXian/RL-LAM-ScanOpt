# Stage 3 Run 19 - Run18 Candidate Handoff Review Package

## Purpose
Create human-review handoff packages for both run18 batch24 and batch28 options without generating CAE/INP/JNL files or running any solver/job tooling.

## Inputs
- Run18 scored candidate pool: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_18_combined80_surrogate_screened_candidate_generation\run18_candidate_pool_scored.csv`
- Run18 shortlist48: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_18_combined80_surrogate_screened_candidate_generation\run18_candidate_shortlist48.csv`
- Run18 batch24: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_18_combined80_surrogate_screened_candidate_generation\run18_recommended_future_teacher_batch24.csv`
- Run18 batch28: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_18_combined80_surrogate_screened_candidate_generation\run18_recommended_future_teacher_batch28.csv`
- Combined80 RL-ready dataset: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_16_batch20_teacher_metrics_ingestion_and_combined80_ranking\combined80_RL_ready_dataset.csv`

## Validation Status
- Verdict: `PASS_RUN19_BATCH24_BATCH28_INPUTS_READY`
- Batch24 counts: `{12: 3, 16: 3, 24: 9, 40: 9}`
- Batch28 counts: `{12: 4, 16: 4, 24: 10, 40: 10}`

## Batch24 Handoff Package
- Candidate order table: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch24\stage3_run19_batch24_candidate_orders.csv`
- Scan order JSON directory: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch24\scan_orders`
- Future CAE handoff template: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch24\stage3_run19_batch24_future_cae_handoff_manifest_TEMPLATE.csv`
- Future abqjobpilot template: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch24\stage3_run19_batch24_abqjobpilot_commands_TEMPLATE.txt`

## Batch28 Handoff Package
- Candidate order table: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch28\stage3_run19_batch28_candidate_orders.csv`
- Scan order JSON directory: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch28\scan_orders`
- Future CAE handoff template: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch28\stage3_run19_batch28_future_cae_handoff_manifest_TEMPLATE.csv`
- Future abqjobpilot template: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch28\stage3_run19_batch28_abqjobpilot_commands_TEMPLATE.txt`

## Batch24 vs Batch28 Review Summary
- Batch24 is cheaper and more conservative.
- Batch28 gives slightly broader N24/N40 coverage at the cost of 4 additional future jobs.
- Neither option is universally superior; both remain unvalidated until future teacher validation.

## Future CAE Handoff Templates
The CAE manifest templates list expected future paths only. Run19 did not create case directories and did not generate CAE/INP/JNL files.

## Future Abqjobpilot Command Templates
Command templates are not executable yet because INP files do not exist. They must not be run until CAE/INP generation has completed and passed checks.

## Claim Boundary
RUN19_HANDOFF_REVIEW_PACKAGE_ONLY_NO_TEACHER_VALIDATION. No physical superiority, teacher validation, trained RL success, arbitrary-N generalization, or executable job readiness is claimed.

## Output Files
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch24\stage3_run19_batch24_candidate_orders.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch28\stage3_run19_batch28_candidate_orders.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch24\scan_orders\scan_order_S3R19B24_N12_B01_surrogate_top.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch24\scan_orders\scan_order_S3R19B24_N12_B02_u2_primary_top.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch24\scan_orders\scan_order_S3R19B24_N12_B03_geometry_signal_top.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch24\scan_orders\scan_order_S3R19B24_N16_B01_surrogate_top.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch24\scan_orders\scan_order_S3R19B24_N16_B02_u2_primary_top.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch24\scan_orders\scan_order_S3R19B24_N16_B03_geometry_signal_top.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch24\scan_orders\scan_order_S3R19B24_N24_B01_surrogate_top.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch24\scan_orders\scan_order_S3R19B24_N24_B02_u2_primary_top.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch24\scan_orders\scan_order_S3R19B24_N24_B03_geometry_signal_top.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch24\scan_orders\scan_order_S3R19B24_N24_B04_method_c_inspired.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch24\scan_orders\scan_order_S3R19B24_N24_B05_diversity_top.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch24\scan_orders\scan_order_S3R19B24_N24_B06_uncertainty_calibration.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch24\scan_orders\scan_order_S3R19B24_N24_B07_control_sentinel.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch24\scan_orders\scan_order_S3R19B24_N24_B08_known_best_mutation.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch24\scan_orders\scan_order_S3R19B24_N24_B09_known_best_mutation.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch24\scan_orders\scan_order_S3R19B24_N40_B01_surrogate_top.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch24\scan_orders\scan_order_S3R19B24_N40_B02_u2_primary_top.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch24\scan_orders\scan_order_S3R19B24_N40_B03_geometry_signal_top.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch24\scan_orders\scan_order_S3R19B24_N40_B04_method_c_inspired.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch24\scan_orders\scan_order_S3R19B24_N40_B05_diversity_top.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch24\scan_orders\scan_order_S3R19B24_N40_B06_uncertainty_calibration.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch24\scan_orders\scan_order_S3R19B24_N40_B07_control_sentinel.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch24\scan_orders\scan_order_S3R19B24_N40_B08_known_best_mutation.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch24\scan_orders\scan_order_S3R19B24_N40_B09_known_best_mutation.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch28\scan_orders\scan_order_S3R19B28_N12_B01_surrogate_top.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch28\scan_orders\scan_order_S3R19B28_N12_B02_u2_primary_top.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch28\scan_orders\scan_order_S3R19B28_N12_B03_geometry_signal_top.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch28\scan_orders\scan_order_S3R19B28_N12_B04_method_c_inspired.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch28\scan_orders\scan_order_S3R19B28_N16_B01_surrogate_top.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch28\scan_orders\scan_order_S3R19B28_N16_B02_u2_primary_top.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch28\scan_orders\scan_order_S3R19B28_N16_B03_geometry_signal_top.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch28\scan_orders\scan_order_S3R19B28_N16_B04_method_c_inspired.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch28\scan_orders\scan_order_S3R19B28_N24_B01_surrogate_top.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch28\scan_orders\scan_order_S3R19B28_N24_B02_u2_primary_top.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch28\scan_orders\scan_order_S3R19B28_N24_B03_geometry_signal_top.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch28\scan_orders\scan_order_S3R19B28_N24_B04_method_c_inspired.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch28\scan_orders\scan_order_S3R19B28_N24_B05_diversity_top.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch28\scan_orders\scan_order_S3R19B28_N24_B06_uncertainty_calibration.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch28\scan_orders\scan_order_S3R19B28_N24_B07_control_sentinel.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch28\scan_orders\scan_order_S3R19B28_N24_B08_known_best_mutation.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch28\scan_orders\scan_order_S3R19B28_N24_B09_known_best_mutation.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch28\scan_orders\scan_order_S3R19B28_N24_B10_known_best_mutation.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch28\scan_orders\scan_order_S3R19B28_N40_B01_surrogate_top.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch28\scan_orders\scan_order_S3R19B28_N40_B02_u2_primary_top.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch28\scan_orders\scan_order_S3R19B28_N40_B03_geometry_signal_top.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch28\scan_orders\scan_order_S3R19B28_N40_B04_method_c_inspired.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch28\scan_orders\scan_order_S3R19B28_N40_B05_diversity_top.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch28\scan_orders\scan_order_S3R19B28_N40_B06_uncertainty_calibration.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch28\scan_orders\scan_order_S3R19B28_N40_B07_control_sentinel.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch28\scan_orders\scan_order_S3R19B28_N40_B08_known_best_mutation.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch28\scan_orders\scan_order_S3R19B28_N40_B09_known_best_mutation.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch28\scan_orders\scan_order_S3R19B28_N40_B10_known_best_mutation.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch24\stage3_run19_batch24_future_cae_handoff_manifest_TEMPLATE.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch28\stage3_run19_batch28_future_cae_handoff_manifest_TEMPLATE.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch24\stage3_run19_batch24_abqjobpilot_commands_TEMPLATE.txt`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch28\stage3_run19_batch28_abqjobpilot_commands_TEMPLATE.txt`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch24_vs_batch28_review_summary.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch24_vs_batch28_review_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch24_vs_batch28_review_summary.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\README_FOR_FUTURE_CAE_GENERATION.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\run19_claim_boundary.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\run19_claim_boundary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\run19_input_validation_summary.json`

## Recommended Next Step
User reviews batch24 vs batch28. If batch24 is selected, the CAE module should create run20 batch24 CAE/INP generation only. If batch28 is selected, the CAE module should create run20 batch28 CAE/INP generation only. Do not generate both unless explicitly requested.
