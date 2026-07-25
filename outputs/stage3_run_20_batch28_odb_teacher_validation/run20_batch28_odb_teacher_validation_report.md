# Stage 3 Run20 Batch28 ODB Teacher Validation Report

## Verdict

`PASS_RUN20_BATCH28_ODB_TEACHER_VALIDATION_28_OF_28`

## Solver Completion Audit

- total: `28/28`
- warning: `28`
- lck present: `0`

| N | expected | complete | warning | failed_or_incomplete |
|---|---:|---:|---:|---:|
| N12 | 4 | 4 | 4 | 0 |
| N16 | 4 | 4 | 4 | 0 |
| N24 | 10 | 10 | 10 | 0 |
| N40 | 10 | 10 | 10 | 0 |

## Incomplete Cases

None.

## Warning Cases

- `S3R19B28_N12_B01_surrogate_top`: `WARNING_SUCCESS_WITH_WARNINGS`; complete_with_nonfatal_warnings
- `S3R19B28_N12_B02_u2_primary_top`: `WARNING_SUCCESS_WITH_WARNINGS`; complete_with_nonfatal_warnings
- `S3R19B28_N12_B03_geometry_signal_top`: `WARNING_SUCCESS_WITH_WARNINGS`; complete_with_nonfatal_warnings
- `S3R19B28_N12_B04_method_c_inspired`: `WARNING_SUCCESS_WITH_WARNINGS`; complete_with_nonfatal_warnings
- `S3R19B28_N16_B01_surrogate_top`: `WARNING_SUCCESS_WITH_WARNINGS`; complete_with_nonfatal_warnings
- `S3R19B28_N16_B02_u2_primary_top`: `WARNING_SUCCESS_WITH_WARNINGS`; complete_with_nonfatal_warnings
- `S3R19B28_N16_B03_geometry_signal_top`: `WARNING_SUCCESS_WITH_WARNINGS`; complete_with_nonfatal_warnings
- `S3R19B28_N16_B04_method_c_inspired`: `WARNING_SUCCESS_WITH_WARNINGS`; complete_with_nonfatal_warnings
- `S3R19B28_N24_B01_surrogate_top`: `WARNING_SUCCESS_WITH_WARNINGS`; complete_with_nonfatal_warnings
- `S3R19B28_N24_B02_u2_primary_top`: `WARNING_SUCCESS_WITH_WARNINGS`; complete_with_nonfatal_warnings
- `S3R19B28_N24_B03_geometry_signal_top`: `WARNING_SUCCESS_WITH_WARNINGS`; complete_with_nonfatal_warnings
- `S3R19B28_N24_B04_method_c_inspired`: `WARNING_SUCCESS_WITH_WARNINGS`; complete_with_nonfatal_warnings
- `S3R19B28_N24_B05_diversity_top`: `WARNING_SUCCESS_WITH_WARNINGS`; complete_with_nonfatal_warnings
- `S3R19B28_N24_B06_uncertainty_calibration`: `WARNING_SUCCESS_WITH_WARNINGS`; complete_with_nonfatal_warnings
- `S3R19B28_N24_B07_control_sentinel`: `WARNING_SUCCESS_WITH_WARNINGS`; complete_with_nonfatal_warnings
- `S3R19B28_N24_B08_known_best_mutation`: `WARNING_SUCCESS_WITH_WARNINGS`; complete_with_nonfatal_warnings
- `S3R19B28_N24_B09_known_best_mutation`: `WARNING_SUCCESS_WITH_WARNINGS`; complete_with_nonfatal_warnings
- `S3R19B28_N24_B10_known_best_mutation`: `WARNING_SUCCESS_WITH_WARNINGS`; complete_with_nonfatal_warnings
- `S3R19B28_N40_B01_surrogate_top`: `WARNING_SUCCESS_WITH_WARNINGS`; complete_with_nonfatal_warnings
- `S3R19B28_N40_B02_u2_primary_top`: `WARNING_SUCCESS_WITH_WARNINGS`; complete_with_nonfatal_warnings
- `S3R19B28_N40_B03_geometry_signal_top`: `WARNING_SUCCESS_WITH_WARNINGS`; complete_with_nonfatal_warnings
- `S3R19B28_N40_B04_method_c_inspired`: `WARNING_SUCCESS_WITH_WARNINGS`; complete_with_nonfatal_warnings
- `S3R19B28_N40_B05_diversity_top`: `WARNING_SUCCESS_WITH_WARNINGS`; complete_with_nonfatal_warnings
- `S3R19B28_N40_B06_uncertainty_calibration`: `WARNING_SUCCESS_WITH_WARNINGS`; complete_with_nonfatal_warnings
- `S3R19B28_N40_B07_control_sentinel`: `WARNING_SUCCESS_WITH_WARNINGS`; complete_with_nonfatal_warnings
- `S3R19B28_N40_B08_known_best_mutation`: `WARNING_SUCCESS_WITH_WARNINGS`; complete_with_nonfatal_warnings
- `S3R19B28_N40_B09_known_best_mutation`: `WARNING_SUCCESS_WITH_WARNINGS`; complete_with_nonfatal_warnings
- `S3R19B28_N40_B10_known_best_mutation`: `WARNING_SUCCESS_WITH_WARNINGS`; complete_with_nonfatal_warnings

## ODB Extraction Summary

- total: `28/28`
- final step: `step_final_cooling`
- required fields: `U;PEEQ;S;NT11`

| N | expected | extracted | failed |
|---|---:|---:|---:|
| N12 | 4 | 4 | 0 |
| N16 | 4 | 4 | 0 |
| N24 | 10 | 10 | 0 |
| N40 | 10 | 10 | 0 |

## Failed Extractions

None.

## Outputs

- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_20_batch28_odb_teacher_validation\run20_batch28_solver_completion_audit.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_20_batch28_odb_teacher_validation\run20_batch28_solver_completion_audit.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_20_batch28_odb_teacher_validation\run20_batch28_odb_extraction_summary.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_20_batch28_odb_teacher_validation\run20_batch28_teacher_metrics.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_20_batch28_odb_teacher_validation\run20_batch28_odb_teacher_validation_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_20_batch28_odb_teacher_validation\run20_batch28_odb_teacher_validation_report.md`

## Scientific Boundary

These batch28 candidates are surrogate-screened only until ODB teacher metrics are extracted and validated. This report does not claim arbitrary-N generalisation, masked/variable-N scaling, or physical superiority.

## Guardrails

- ODB files were opened read-only only after 28/28 solver completion.
- No Abaqus solver job was run.
- No datacheck was run.
- No abqjobpilot/enqueue command was run.
- No CAE/INP/JNL/ODB files were modified.
- Batch24 was not processed.
- Results were not mixed with probe60/run08-run16 outputs.
