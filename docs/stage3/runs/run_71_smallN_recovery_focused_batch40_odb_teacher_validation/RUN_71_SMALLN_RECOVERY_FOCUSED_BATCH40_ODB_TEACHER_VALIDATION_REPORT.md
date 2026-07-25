# Stage 3 Run71 Small-N Recovery-Focused Batch40 ODB Teacher Validation Report

## Verdict

`PASS_RUN71_SMALLN_RECOVERY_FOCUSED_BATCH40_ODB_TEACHER_VALIDATION_40_OF_40`

## Solver Completion Audit

- total: `40/40`
- warning: `40`
- lck present: `0`

| N | expected | complete | warning | failed_or_incomplete |
|---|---:|---:|---:|---:|
| N12 | 16 | 16 | 16 | 0 |
| N16 | 16 | 16 | 16 | 0 |
| N24 | 4 | 4 | 4 | 0 |
| N40 | 4 | 4 | 4 | 0 |

## Run70 Generation Details

- N12, N16, N24, and N40 cases were generated; no N32 cases were generated.
- Per-N counts: N12=16, N16=16, N24=4, N40=4.
- Metrics are extracted from the final frame of `step_final_cooling`.
- N40 cases were generated with `step_cool_XX` initialInc = `0.001`.

## Patched/Rerun Cases

None.

## Incomplete Cases

None.

## ODB Extraction Summary

- total: `40/40`
- final step: `step_final_cooling`
- required fields: `U;PEEQ;S;NT11`

| N | expected | extracted | failed |
|---|---:|---:|---:|
| N12 | 16 | 16 | 0 |
| N16 | 16 | 16 | 0 |
| N24 | 4 | 4 | 0 |
| N40 | 4 | 4 | 0 |

## Failed Extractions

None.

## Outputs

- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_71_smallN_recovery_focused_batch40_odb_teacher_validation\run71_smallN_recovery_focused_batch40_solver_completion_audit.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_71_smallN_recovery_focused_batch40_odb_teacher_validation\run71_smallN_recovery_focused_batch40_solver_completion_audit.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_71_smallN_recovery_focused_batch40_odb_teacher_validation\run71_smallN_recovery_focused_batch40_solver_completion_audit.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_71_smallN_recovery_focused_batch40_odb_teacher_validation\run71_smallN_recovery_focused_batch40_odb_extraction_summary.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_71_smallN_recovery_focused_batch40_odb_teacher_validation\run71_smallN_recovery_focused_batch40_odb_extraction_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_71_smallN_recovery_focused_batch40_odb_teacher_validation\run71_smallN_recovery_focused_batch40_teacher_metrics.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_71_smallN_recovery_focused_batch40_odb_teacher_validation\run71_smallN_recovery_focused_batch40_teacher_metrics_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_71_smallN_recovery_focused_batch40_odb_teacher_validation\run71_smallN_recovery_focused_batch40_odb_teacher_validation_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\docs\stage3\runs\run_71_smallN_recovery_focused_batch40_odb_teacher_validation\RUN_71_SMALLN_RECOVERY_FOCUSED_BATCH40_ODB_TEACHER_VALIDATION_REPORT.md`
- `E:\Projects\RL-LAM-ScanOpt\artifacts\manifests\stage3_run_71_manifest.json`

## Scientific Boundary

This report records computed teacher metrics for Run70 small-N recovery-focused batch40. It does not claim physical superiority beyond computed teacher metrics, RL/GNN success, or arbitrary-N generalisation.

## Guardrails

- ODB files were opened read-only only after 40/40 solver completion.
- No Abaqus solver job was run.
- No datacheck was run.
- No abqjobpilot/enqueue command was run.
- No CAE/INP/JNL or base sanity files were modified.
- No commit or push was made.
