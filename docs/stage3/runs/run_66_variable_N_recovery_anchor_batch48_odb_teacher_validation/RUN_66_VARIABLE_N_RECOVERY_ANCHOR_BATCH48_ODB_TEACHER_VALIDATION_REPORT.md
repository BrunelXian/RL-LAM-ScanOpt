# Stage 3 Run66 Variable-N Recovery Anchor Batch48 ODB Teacher Validation Report

## Verdict

`PASS_RUN66_VARIABLE_N_RECOVERY_ANCHOR_BATCH48_ODB_TEACHER_VALIDATION_48_OF_48`

## Solver Completion Audit

- total: `48/48`
- warning: `48`
- lck present: `0`

| N | expected | complete | warning | failed_or_incomplete |
|---|---:|---:|---:|---:|
| N12 | 12 | 12 | 12 | 0 |
| N16 | 12 | 12 | 12 | 0 |
| N24 | 8 | 8 | 8 | 0 |
| N40 | 16 | 16 | 16 | 0 |

## Run65 Generation Details

- N12, N16, N24, and N40 cases were generated; no N32 cases were generated.
- Per-N counts: N12=12, N16=12, N24=8, N40=16.
- Metrics are extracted from the final frame of `step_final_cooling`.
- N40 cases were generated with `step_cool_XX` initialInc = `0.001`.

## Patched/Rerun Cases

None.

## Incomplete Cases

None.

## ODB Extraction Summary

- total: `48/48`
- final step: `step_final_cooling`
- required fields: `U;PEEQ;S;NT11`

| N | expected | extracted | failed |
|---|---:|---:|---:|
| N12 | 12 | 12 | 0 |
| N16 | 12 | 12 | 0 |
| N24 | 8 | 8 | 0 |
| N40 | 16 | 16 | 0 |

## Failed Extractions

None.

## Outputs

- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_66_variable_N_recovery_anchor_batch48_odb_teacher_validation\run66_variable_N_recovery_anchor_batch48_solver_completion_audit.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_66_variable_N_recovery_anchor_batch48_odb_teacher_validation\run66_variable_N_recovery_anchor_batch48_solver_completion_audit.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_66_variable_N_recovery_anchor_batch48_odb_teacher_validation\run66_variable_N_recovery_anchor_batch48_solver_completion_audit.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_66_variable_N_recovery_anchor_batch48_odb_teacher_validation\run66_variable_N_recovery_anchor_batch48_odb_extraction_summary.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_66_variable_N_recovery_anchor_batch48_odb_teacher_validation\run66_variable_N_recovery_anchor_batch48_odb_extraction_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_66_variable_N_recovery_anchor_batch48_odb_teacher_validation\run66_variable_N_recovery_anchor_batch48_teacher_metrics.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_66_variable_N_recovery_anchor_batch48_odb_teacher_validation\run66_variable_N_recovery_anchor_batch48_teacher_metrics_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_66_variable_N_recovery_anchor_batch48_odb_teacher_validation\run66_variable_N_recovery_anchor_batch48_odb_teacher_validation_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\docs\stage3\runs\run_66_variable_N_recovery_anchor_batch48_odb_teacher_validation\RUN_66_VARIABLE_N_RECOVERY_ANCHOR_BATCH48_ODB_TEACHER_VALIDATION_REPORT.md`
- `E:\Projects\RL-LAM-ScanOpt\artifacts\manifests\stage3_run_66_manifest.json`

## Scientific Boundary

This report records computed teacher metrics for Run65 variable-N recovery anchor batch48. It does not claim physical superiority beyond computed teacher metrics, RL/GNN success, or arbitrary-N generalisation.

## Guardrails

- ODB files were opened read-only only after 48/48 solver completion.
- No Abaqus solver job was run.
- No datacheck was run.
- No abqjobpilot/enqueue command was run.
- No CAE/INP/JNL or base sanity files were modified.
- No commit or push was made.
