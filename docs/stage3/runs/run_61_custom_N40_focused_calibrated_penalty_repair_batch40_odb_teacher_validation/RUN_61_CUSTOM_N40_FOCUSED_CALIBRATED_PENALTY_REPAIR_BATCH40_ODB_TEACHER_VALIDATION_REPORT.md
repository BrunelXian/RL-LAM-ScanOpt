# Stage 3 Run61 Custom N40-Focused Calibrated Penalty-Repair Batch40 ODB Teacher Validation Report

## Verdict

`PASS_RUN61_CUSTOM_N40_FOCUSED_BATCH40_ODB_TEACHER_VALIDATION_40_OF_40`

## Solver Completion Audit

- total: `40/40`
- warning: `40`
- lck present: `0`

| N | expected | complete | warning | failed_or_incomplete |
|---|---:|---:|---:|---:|
| N24 | 16 | 16 | 16 | 0 |
| N40 | 24 | 24 | 24 | 0 |

## Run60 Generation Details

- No N12, N16, or N32 cases were generated.
- N24 and N40 only: N24=16, N40=24.
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
| N24 | 16 | 16 | 0 |
| N40 | 24 | 24 | 0 |

## Failed Extractions

None.

## Outputs

- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_61_custom_N40_focused_calibrated_penalty_repair_batch40_odb_teacher_validation\run61_custom_N40_focused_batch40_solver_completion_audit.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_61_custom_N40_focused_calibrated_penalty_repair_batch40_odb_teacher_validation\run61_custom_N40_focused_batch40_solver_completion_audit.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_61_custom_N40_focused_calibrated_penalty_repair_batch40_odb_teacher_validation\run61_custom_N40_focused_batch40_solver_completion_audit.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_61_custom_N40_focused_calibrated_penalty_repair_batch40_odb_teacher_validation\run61_custom_N40_focused_batch40_odb_extraction_summary.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_61_custom_N40_focused_calibrated_penalty_repair_batch40_odb_teacher_validation\run61_custom_N40_focused_batch40_odb_extraction_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_61_custom_N40_focused_calibrated_penalty_repair_batch40_odb_teacher_validation\run61_custom_N40_focused_batch40_teacher_metrics.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_61_custom_N40_focused_calibrated_penalty_repair_batch40_odb_teacher_validation\run61_custom_N40_focused_batch40_teacher_metrics_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_61_custom_N40_focused_calibrated_penalty_repair_batch40_odb_teacher_validation\run61_custom_N40_focused_batch40_odb_teacher_validation_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\docs\stage3\runs\run_61_custom_N40_focused_calibrated_penalty_repair_batch40_odb_teacher_validation\RUN_61_CUSTOM_N40_FOCUSED_CALIBRATED_PENALTY_REPAIR_BATCH40_ODB_TEACHER_VALIDATION_REPORT.md`
- `E:\Projects\RL-LAM-ScanOpt\artifacts\manifests\stage3_run_61_manifest.json`

## Scientific Boundary

This report records computed teacher metrics for Run60 custom N40-focused calibrated penalty-repair batch40. It does not claim physical superiority beyond computed teacher metrics, RL/GNN success, or arbitrary-N generalisation.

## Guardrails

- ODB files were opened read-only only after 40/40 solver completion.
- No Abaqus solver job was run.
- No datacheck was run.
- No abqjobpilot/enqueue command was run.
- No CAE/INP/JNL or base sanity files were modified.
- No commit or push was made.
