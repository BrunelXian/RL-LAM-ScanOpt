# Stage 3 Run56 Calibrated N24/N40 Batch64 ODB Teacher Validation Report

## Verdict

`PASS_RUN56_CALIBRATED_N24_N40_BATCH64_ODB_TEACHER_VALIDATION_64_OF_64`

## Solver Completion Audit

- total: `64/64`
- warning: `64`
- lck present: `0`

| N | expected | complete | warning | failed_or_incomplete |
|---|---:|---:|---:|---:|
| N24 | 32 | 32 | 32 | 0 |
| N40 | 32 | 32 | 32 | 0 |

## Run55 Generation Details

- No N12, N16, or N32 cases were generated.
- N24 and N40 only: 32 cases each.
- Metrics are extracted from the final frame of `step_final_cooling`.
- N40 cases were generated with `step_cool_XX` initialInc = `0.001`.

## Patched/Rerun Cases

None.

## Incomplete Cases

None.

## ODB Extraction Summary

- total: `64/64`
- final step: `step_final_cooling`
- required fields: `U;PEEQ;S;NT11`

| N | expected | extracted | failed |
|---|---:|---:|---:|
| N24 | 32 | 32 | 0 |
| N40 | 32 | 32 | 0 |

## Failed Extractions

None.

## Outputs

- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_56_calibrated_N24_N40_batch64_odb_teacher_validation\run56_calibrated_N24_N40_batch64_solver_completion_audit.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_56_calibrated_N24_N40_batch64_odb_teacher_validation\run56_calibrated_N24_N40_batch64_solver_completion_audit.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_56_calibrated_N24_N40_batch64_odb_teacher_validation\run56_calibrated_N24_N40_batch64_solver_completion_audit.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_56_calibrated_N24_N40_batch64_odb_teacher_validation\run56_calibrated_N24_N40_batch64_odb_extraction_summary.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_56_calibrated_N24_N40_batch64_odb_teacher_validation\run56_calibrated_N24_N40_batch64_odb_extraction_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_56_calibrated_N24_N40_batch64_odb_teacher_validation\run56_calibrated_N24_N40_batch64_teacher_metrics.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_56_calibrated_N24_N40_batch64_odb_teacher_validation\run56_calibrated_N24_N40_batch64_teacher_metrics_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_56_calibrated_N24_N40_batch64_odb_teacher_validation\run56_calibrated_N24_N40_batch64_odb_teacher_validation_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\docs\stage3\runs\run_56_calibrated_N24_N40_batch64_odb_teacher_validation\RUN_56_CALIBRATED_N24_N40_BATCH64_ODB_TEACHER_VALIDATION_REPORT.md`
- `E:\Projects\RL-LAM-ScanOpt\artifacts\manifests\stage3_run_56_manifest.json`

## Scientific Boundary

This report records computed teacher metrics for Run55 calibrated N24/N40 batch64. It does not claim physical superiority beyond computed teacher metrics, RL/GNN success, or arbitrary-N generalisation.

## Guardrails

- ODB files were opened read-only only after 64/64 solver completion.
- No Abaqus solver job was run.
- No datacheck was run.
- No abqjobpilot/enqueue command was run.
- No CAE/INP/JNL or base sanity files were modified.
- No commit or push was made.
