# Stage 3 Run36 N32-Informed Native Batch32 ODB Teacher Validation Report

## Verdict

`PASS_RUN36_N32_INFORMED_NATIVE_BATCH32_ODB_TEACHER_VALIDATION_32_OF_32`

## Solver Completion Audit

- total: `32/32`
- warning: `32`
- lck present: `0`

| N | expected | complete | warning | failed_or_incomplete |
|---|---:|---:|---:|---:|
| N12 | 4 | 4 | 4 | 0 |
| N16 | 4 | 4 | 4 | 0 |
| N24 | 12 | 12 | 12 | 0 |
| N40 | 12 | 12 | 12 | 0 |

## Run35 Generation Details

- No N32 cases were generated.
- Run35 supersedes old Run31 as the active teacher-validation batch.
- N40 cases were generated with `step_cool_XX` initialInc = `0.001`.

## Patched/Rerun Cases

None.

## Incomplete Cases

None.

## ODB Extraction Summary

- total: `32/32`
- final step: `step_final_cooling`
- required fields: `U;PEEQ;S;NT11`

| N | expected | extracted | failed |
|---|---:|---:|---:|
| N12 | 4 | 4 | 0 |
| N16 | 4 | 4 | 0 |
| N24 | 12 | 12 | 0 |
| N40 | 12 | 12 | 0 |

## Failed Extractions

None.

## Outputs

- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_36_N32_informed_native_batch32_odb_teacher_validation\run36_N32_informed_native_batch32_solver_completion_audit.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_36_N32_informed_native_batch32_odb_teacher_validation\run36_N32_informed_native_batch32_solver_completion_audit.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_36_N32_informed_native_batch32_odb_teacher_validation\run36_N32_informed_native_batch32_odb_extraction_summary.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_36_N32_informed_native_batch32_odb_teacher_validation\run36_N32_informed_native_batch32_teacher_metrics.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_36_N32_informed_native_batch32_odb_teacher_validation\run36_N32_informed_native_batch32_odb_teacher_validation_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\docs\stage3\runs\run_36_N32_informed_native_batch32_odb_teacher_validation\RUN_36_N32_INFORMED_NATIVE_BATCH32_ODB_TEACHER_VALIDATION_REPORT.md`
- `E:\Projects\RL-LAM-ScanOpt\artifacts\manifests\stage3_run_36_manifest.json`

## Scientific Boundary

This report records computed teacher metrics for Run35 N32-informed native batch32. It does not claim physical superiority beyond computed teacher metrics and does not claim arbitrary-N generalisation.

## Guardrails

- ODB files were opened read-only only after 32/32 solver completion.
- No Abaqus solver job was run.
- No datacheck was run.
- No abqjobpilot/enqueue command was run.
- No CAE/INP/JNL or base sanity files were modified.
- No commit or push was made.
