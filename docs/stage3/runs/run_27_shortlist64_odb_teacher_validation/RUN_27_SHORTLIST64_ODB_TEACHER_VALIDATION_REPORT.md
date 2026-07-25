# Stage 3 Run27 Shortlist64 ODB Teacher Validation Report

## Verdict

`PASS_RUN27_SHORTLIST64_ODB_TEACHER_VALIDATION_64_OF_64`

## Solver Completion Audit

- total: `64/64`
- warning: `64`
- lck present: `0`

| N | expected | complete | warning | failed_or_incomplete |
|---|---:|---:|---:|---:|
| N12 | 8 | 8 | 8 | 0 |
| N16 | 8 | 8 | 8 | 0 |
| N24 | 24 | 24 | 24 | 0 |
| N40 | 24 | 24 | 24 | 0 |

## Patched N40 Cases

- `S3R24L64_N40_B02_top_region`: `WARNING_SUCCESS_WITH_WARNINGS`; complete_with_nonfatal_warnings
- `S3R24L64_N40_B03_top_region`: `WARNING_SUCCESS_WITH_WARNINGS`; complete_with_nonfatal_warnings
- `S3R24L64_N40_B04_top_region`: `WARNING_SUCCESS_WITH_WARNINGS`; complete_with_nonfatal_warnings
- `S3R24L64_N40_B05_top_region`: `WARNING_SUCCESS_WITH_WARNINGS`; complete_with_nonfatal_warnings

## Incomplete Cases

None.

## ODB Extraction Summary

- total: `64/64`
- final step: `step_final_cooling`
- required fields: `U;PEEQ;S;NT11`

| N | expected | extracted | failed |
|---|---:|---:|---:|
| N12 | 8 | 8 | 0 |
| N16 | 8 | 8 | 0 |
| N24 | 24 | 24 | 0 |
| N40 | 24 | 24 | 0 |

## Failed Extractions

None.

## Outputs

- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_27_shortlist64_odb_teacher_validation\run27_shortlist64_solver_completion_audit.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_27_shortlist64_odb_teacher_validation\run27_shortlist64_solver_completion_audit.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_27_shortlist64_odb_teacher_validation\run27_shortlist64_odb_extraction_summary.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_27_shortlist64_odb_teacher_validation\run27_shortlist64_teacher_metrics.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_27_shortlist64_odb_teacher_validation\run27_shortlist64_odb_teacher_validation_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\docs\stage3\runs\run_27_shortlist64_odb_teacher_validation\RUN_27_SHORTLIST64_ODB_TEACHER_VALIDATION_REPORT.md`
- `E:\Projects\RL-LAM-ScanOpt\artifacts\manifests\stage3_run_27_manifest.json`

## Scientific Boundary

This report records computed teacher metrics for shortlist64. It does not claim physical superiority beyond computed teacher metrics and does not claim arbitrary-N generalisation.

## Guardrails

- ODB files were opened read-only only after 64/64 solver completion.
- No Abaqus solver job was run.
- No datacheck was run.
- No abqjobpilot/enqueue command was run.
- No CAE/INP/JNL or base sanity files were modified.
- No commit or push was made.
