# Stage 3 Run 13 - Batch20 Surrogate-Screened Teacher Handoff

## Purpose
Create a clean handoff package for 20 run12 surrogate-screened candidates. This is active-learning/diversity validation packaging only.

## Inputs
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_12_offline_surrogate_screened_candidate_generation\run12_recommended_future_teacher_batch20.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_12_offline_surrogate_screened_candidate_generation\run12_candidate_pool_scored.csv`
- `E:\Projects\RL-LAM-ScanOpt\docs\stage3\runs\run_12_offline_surrogate_screened_candidate_generation\RUN_12_OFFLINE_SURROGATE_SCREENED_CANDIDATE_GENERATION_REPORT.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_09_variable_n_probe60_teacher_ranking_analysis\probe60_teacher_ranked_canonical.csv`

## Validation Status
- `PASS_RUN13_BATCH20_INPUT_READY`
- Candidate count: 20
- Per-N counts: {12: 5, 16: 5, 24: 5, 40: 5}

## Batch Composition
- Batch name: `stage3_run13_batch20_surrogate_screened_v01`
- Selection buckets: {'surrogate_top': 12, 'diversity_top': 4, 'method_c_inspired': 4}

## Naming Convention
- `S3B20_N{N}_B{index:02d}_{short_family_or_bucket}`
- Names preserve run12 candidate IDs and order hashes in metadata.

## Candidate Order Handoff Table
- See `stage3_run13_batch20_candidate_orders.csv`.

## Per-Candidate Scan_Order JSON Outputs
- Directory: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_13_batch20_surrogate_screened_teacher_handoff\scan_orders`

## Future CAE Handoff Manifest Template
- Template only. No CAE/INP/JNL/ODB files were generated.

## Future Abqjobpilot Command Template
- Template only. Commands must not be run until INPs exist and are checked.

## Comparison To Existing Teacher Best
- N12: top batch20 `S3B20_N12_B01_surrogate_top` predicted 0.7767; existing best `N12_A08_block_interleaved_quarters` reward 0.9043; exceeds `False`. Surrogate-only.
- N16: top batch20 `S3B20_N16_B01_surrogate_top` predicted 0.8065; existing best `N16_A03_greedy_maximin_distance` reward 0.8614; exceeds `False`. Surrogate-only.
- N24: top batch20 `S3B20_N24_B01_surrogate_top` predicted 0.8347; existing best `N24_A04_method_c_u2_first_engineering` reward 0.9786; exceeds `False`. Surrogate-only.
- N40: top batch20 `S3B20_N40_B01_surrogate_top` predicted 0.8395; existing best `N40_A04_method_c_u2_first_engineering` reward 0.9929; exceeds `False`. Surrogate-only.

## Claim Boundary
- Run13 does not perform teacher validation.
- Run13 does not prove physical superiority.
- Future CAE/INP paths and abqjobpilot commands are templates only.

## Output Files
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_13_batch20_surrogate_screened_teacher_handoff\run13_input_validation_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_13_batch20_surrogate_screened_teacher_handoff\stage3_run13_batch20_candidate_orders.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_13_batch20_surrogate_screened_teacher_handoff\scan_orders\scan_order_S3B20_N12_B01_surrogate_top.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_13_batch20_surrogate_screened_teacher_handoff\scan_orders\scan_order_S3B20_N12_B02_diversity_top.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_13_batch20_surrogate_screened_teacher_handoff\scan_orders\scan_order_S3B20_N12_B03_method_c_inspired.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_13_batch20_surrogate_screened_teacher_handoff\scan_orders\scan_order_S3B20_N12_B04_surrogate_top.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_13_batch20_surrogate_screened_teacher_handoff\scan_orders\scan_order_S3B20_N12_B05_surrogate_top.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_13_batch20_surrogate_screened_teacher_handoff\scan_orders\scan_order_S3B20_N16_B01_surrogate_top.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_13_batch20_surrogate_screened_teacher_handoff\scan_orders\scan_order_S3B20_N16_B02_diversity_top.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_13_batch20_surrogate_screened_teacher_handoff\scan_orders\scan_order_S3B20_N16_B03_method_c_inspired.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_13_batch20_surrogate_screened_teacher_handoff\scan_orders\scan_order_S3B20_N16_B04_surrogate_top.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_13_batch20_surrogate_screened_teacher_handoff\scan_orders\scan_order_S3B20_N16_B05_surrogate_top.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_13_batch20_surrogate_screened_teacher_handoff\scan_orders\scan_order_S3B20_N24_B01_surrogate_top.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_13_batch20_surrogate_screened_teacher_handoff\scan_orders\scan_order_S3B20_N24_B02_diversity_top.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_13_batch20_surrogate_screened_teacher_handoff\scan_orders\scan_order_S3B20_N24_B03_method_c_inspired.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_13_batch20_surrogate_screened_teacher_handoff\scan_orders\scan_order_S3B20_N24_B04_surrogate_top.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_13_batch20_surrogate_screened_teacher_handoff\scan_orders\scan_order_S3B20_N24_B05_surrogate_top.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_13_batch20_surrogate_screened_teacher_handoff\scan_orders\scan_order_S3B20_N40_B01_surrogate_top.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_13_batch20_surrogate_screened_teacher_handoff\scan_orders\scan_order_S3B20_N40_B02_diversity_top.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_13_batch20_surrogate_screened_teacher_handoff\scan_orders\scan_order_S3B20_N40_B03_method_c_inspired.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_13_batch20_surrogate_screened_teacher_handoff\scan_orders\scan_order_S3B20_N40_B04_surrogate_top.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_13_batch20_surrogate_screened_teacher_handoff\scan_orders\scan_order_S3B20_N40_B05_surrogate_top.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_13_batch20_surrogate_screened_teacher_handoff\stage3_run13_batch20_future_cae_handoff_manifest_TEMPLATE.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_13_batch20_surrogate_screened_teacher_handoff\stage3_run13_batch20_abqjobpilot_commands_TEMPLATE.txt`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_13_batch20_surrogate_screened_teacher_handoff\README_FOR_CAE_GENERATION.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_13_batch20_surrogate_screened_teacher_handoff\batch20_vs_existing_teacher_best_summary.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_13_batch20_surrogate_screened_teacher_handoff\batch20_vs_existing_teacher_best_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_13_batch20_surrogate_screened_teacher_handoff\run13_claim_boundary.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_13_batch20_surrogate_screened_teacher_handoff\run13_claim_boundary.json`
- `E:\Projects\RL-LAM-ScanOpt\docs\stage3\runs\run_13_batch20_surrogate_screened_teacher_handoff\RUN_13_BATCH20_SURROGATE_SCREENED_TEACHER_HANDOFF_REPORT.md`

## Recommended Next Step
Human review of the 20 candidates. If approved, the CAE module should create a separate Stage 3 run13 batch20 CAE-generation workflow, generate CAE/INP only, validate mesh and final cooling controls, and only then prepare executable abqjobpilot commands.
