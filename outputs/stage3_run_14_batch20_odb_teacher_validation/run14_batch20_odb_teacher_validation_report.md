# Stage 3 Run14 Batch20 ODB Teacher Validation Report

## Verdict

`PASS_RUN14_BATCH20_ODB_TEACHER_VALIDATION_20_OF_20`

## Solver Completion Audit

- total: `20/20`
- lck present: `0`

| N | expected | complete | failed_or_incomplete |
|---|---:|---:|---:|
| N12 | 5 | 5 | 0 |
| N16 | 5 | 5 | 0 |
| N24 | 5 | 5 | 0 |
| N40 | 5 | 5 | 0 |

## Incomplete Cases

None.

## Warning Cases

- `S3B20_N12_B01_surrogate_top`: `WARNING_SUCCESS_WITH_WARNINGS`; complete_with_nonfatal_warnings
- `S3B20_N12_B02_diversity_top`: `WARNING_SUCCESS_WITH_WARNINGS`; complete_with_nonfatal_warnings
- `S3B20_N12_B03_method_c_inspired`: `WARNING_SUCCESS_WITH_WARNINGS`; complete_with_nonfatal_warnings
- `S3B20_N12_B04_surrogate_top`: `WARNING_SUCCESS_WITH_WARNINGS`; complete_with_nonfatal_warnings
- `S3B20_N12_B05_surrogate_top`: `WARNING_SUCCESS_WITH_WARNINGS`; complete_with_nonfatal_warnings
- `S3B20_N16_B01_surrogate_top`: `WARNING_SUCCESS_WITH_WARNINGS`; complete_with_nonfatal_warnings
- `S3B20_N16_B02_diversity_top`: `WARNING_SUCCESS_WITH_WARNINGS`; complete_with_nonfatal_warnings
- `S3B20_N16_B03_method_c_inspired`: `WARNING_SUCCESS_WITH_WARNINGS`; complete_with_nonfatal_warnings
- `S3B20_N16_B04_surrogate_top`: `WARNING_SUCCESS_WITH_WARNINGS`; complete_with_nonfatal_warnings
- `S3B20_N16_B05_surrogate_top`: `WARNING_SUCCESS_WITH_WARNINGS`; complete_with_nonfatal_warnings
- `S3B20_N24_B01_surrogate_top`: `WARNING_SUCCESS_WITH_WARNINGS`; complete_with_nonfatal_warnings
- `S3B20_N24_B02_diversity_top`: `WARNING_SUCCESS_WITH_WARNINGS`; complete_with_nonfatal_warnings
- `S3B20_N24_B03_method_c_inspired`: `WARNING_SUCCESS_WITH_WARNINGS`; complete_with_nonfatal_warnings
- `S3B20_N24_B04_surrogate_top`: `WARNING_SUCCESS_WITH_WARNINGS`; complete_with_nonfatal_warnings
- `S3B20_N24_B05_surrogate_top`: `WARNING_SUCCESS_WITH_WARNINGS`; complete_with_nonfatal_warnings
- `S3B20_N40_B01_surrogate_top`: `WARNING_SUCCESS_WITH_WARNINGS`; complete_with_nonfatal_warnings
- `S3B20_N40_B02_diversity_top`: `WARNING_SUCCESS_WITH_WARNINGS`; complete_with_nonfatal_warnings
- `S3B20_N40_B03_method_c_inspired`: `WARNING_SUCCESS_WITH_WARNINGS`; complete_with_nonfatal_warnings
- `S3B20_N40_B04_surrogate_top`: `WARNING_SUCCESS_WITH_WARNINGS`; complete_with_nonfatal_warnings
- `S3B20_N40_B05_surrogate_top`: `WARNING_SUCCESS_WITH_WARNINGS`; complete_with_nonfatal_warnings

## ODB Extraction Summary

- total: `20/20`
- final step: `step_final_cooling`
- required fields: `U;PEEQ;S;NT11`

| N | expected | extracted | failed |
|---|---:|---:|---:|
| N12 | 5 | 5 | 0 |
| N16 | 5 | 5 | 0 |
| N24 | 5 | 5 | 0 |
| N40 | 5 | 5 | 0 |

## Failed Extractions

None.

## Outputs

- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_14_batch20_odb_teacher_validation\run14_batch20_solver_completion_audit.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_14_batch20_odb_teacher_validation\run14_batch20_solver_completion_audit.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_14_batch20_odb_teacher_validation\run14_batch20_odb_extraction_summary.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_14_batch20_odb_teacher_validation\run14_batch20_teacher_metrics.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_14_batch20_odb_teacher_validation\run14_batch20_odb_teacher_validation_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_14_batch20_odb_teacher_validation\run14_batch20_odb_teacher_validation_report.md`

## Scientific Boundary

These run13/run14 candidates are surrogate-screened only. This report records ODB extraction status and final-frame teacher quantities; it does not claim physical superiority or validated improvement.

## Guardrails

- ODB files were opened read-only only after 20/20 solver completion.
- No Abaqus solver job was run.
- No datacheck was run.
- No abqjobpilot/enqueue command was run.
- No CAE/INP/JNL files were modified.
- Results were not mixed with probe60/run08-run12 outputs.
