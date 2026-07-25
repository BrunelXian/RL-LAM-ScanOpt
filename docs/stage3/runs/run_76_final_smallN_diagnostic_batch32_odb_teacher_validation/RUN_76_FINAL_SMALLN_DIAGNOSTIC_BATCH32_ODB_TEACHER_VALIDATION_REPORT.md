# Stage 3 Run76 Final Small-N Diagnostic Batch32 ODB Teacher Validation Report

## Verdict

`PASS_RUN76_FINAL_SMALLN_DIAGNOSTIC_BATCH32_ODB_TEACHER_VALIDATION_32_OF_32`

## Solver Completion Audit

- total: `32/32`
- warning: `32`
- lck present: `0`

| N | expected | complete | warning | failed_or_incomplete |
|---|---:|---:|---:|---:|
| N12 | 14 | 14 | 14 | 0 |
| N16 | 14 | 14 | 14 | 0 |
| N24 | 2 | 2 | 2 | 0 |
| N40 | 2 | 2 | 2 | 0 |

## Run75 Generation Details

- N12, N16, N24, and N40 cases were generated; no N32 cases were generated.
- Per-N counts: N12=14, N16=14, N24=2, N40=2.
- Metrics are extracted from the final frame of `step_final_cooling`.
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
| N12 | 14 | 14 | 0 |
| N16 | 14 | 14 | 0 |
| N24 | 2 | 2 | 0 |
| N40 | 2 | 2 | 0 |

## Failed Extractions

None.

## Outputs

- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_76_final_smallN_diagnostic_batch32_odb_teacher_validation\run76_final_smallN_diagnostic_batch32_solver_completion_audit.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_76_final_smallN_diagnostic_batch32_odb_teacher_validation\run76_final_smallN_diagnostic_batch32_solver_completion_audit.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_76_final_smallN_diagnostic_batch32_odb_teacher_validation\run76_final_smallN_diagnostic_batch32_solver_completion_audit.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_76_final_smallN_diagnostic_batch32_odb_teacher_validation\run76_final_smallN_diagnostic_batch32_odb_extraction_summary.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_76_final_smallN_diagnostic_batch32_odb_teacher_validation\run76_final_smallN_diagnostic_batch32_odb_extraction_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_76_final_smallN_diagnostic_batch32_odb_teacher_validation\run76_final_smallN_diagnostic_batch32_teacher_metrics.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_76_final_smallN_diagnostic_batch32_odb_teacher_validation\run76_final_smallN_diagnostic_batch32_teacher_metrics_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_76_final_smallN_diagnostic_batch32_odb_teacher_validation\run76_final_smallN_diagnostic_batch32_odb_teacher_validation_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\docs\stage3\runs\run_76_final_smallN_diagnostic_batch32_odb_teacher_validation\RUN_76_FINAL_SMALLN_DIAGNOSTIC_BATCH32_ODB_TEACHER_VALIDATION_REPORT.md`
- `E:\Projects\RL-LAM-ScanOpt\artifacts\manifests\stage3_run_76_manifest.json`

## Scientific Boundary

This report records computed teacher metrics for Run75 final small-N diagnostic batch32. It does not claim physical superiority beyond computed teacher metrics, RL/GNN success, or arbitrary-N generalisation.

## Guardrails

- ODB files were opened read-only only after 32/32 solver completion.
- No Abaqus solver job was run.
- No datacheck was run.
- No abqjobpilot/enqueue command was run.
- No CAE/INP/JNL or base sanity files were modified.
- No commit or push was made.
