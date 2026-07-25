# Stage 3 Run41 Native N24/N40 Focused Batch60 ODB Teacher Extraction Report

## Verdict

`PASS_RUN41_NATIVE_N24_N40_FOCUSED_BATCH60_ODB_TEACHER_EXTRACTION_60_OF_60`

## Solver Completion Audit

- total: `60/60`
- warning: `60`
- lck present: `0`

| N | expected | complete | warning | failed_or_incomplete |
|---|---:|---:|---:|---:|
| N24 | 30 | 30 | 30 | 0 |
| N40 | 30 | 30 | 30 | 0 |

## Run40 Generation Details

- No N12, N16, or N32 cases were generated.
- N24 and N40 only: 30 cases each.
- Metrics are extracted from the final frame of `step_final_cooling`.
- N40 cases were generated with `step_cool_XX` initialInc = `0.001`.

## Patched/Rerun Cases

None.

## Incomplete Cases

None.

## ODB Extraction Summary

- total: `60/60`
- final step: `step_final_cooling`
- required fields: `U;PEEQ;S;NT11`

| N | expected | extracted | failed |
|---|---:|---:|---:|
| N24 | 30 | 30 | 0 |
| N40 | 30 | 30 | 0 |

## Failed Extractions

None.

## Outputs

- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_41_native_N24_N40_focused_batch60_odb_teacher_extraction\run41_native_N24_N40_focused_batch60_solver_completion_audit.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_41_native_N24_N40_focused_batch60_odb_teacher_extraction\run41_native_N24_N40_focused_batch60_solver_completion_audit.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_41_native_N24_N40_focused_batch60_odb_teacher_extraction\run41_native_N24_N40_focused_batch60_odb_extraction_summary.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_41_native_N24_N40_focused_batch60_odb_teacher_extraction\run41_native_N24_N40_focused_batch60_teacher_metrics.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_41_native_N24_N40_focused_batch60_odb_teacher_extraction\run41_native_N24_N40_focused_batch60_odb_teacher_extraction_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\docs\stage3\runs\run_41_native_N24_N40_focused_batch60_odb_teacher_extraction\RUN_41_NATIVE_N24_N40_FOCUSED_BATCH60_ODB_TEACHER_EXTRACTION_REPORT.md`
- `E:\Projects\RL-LAM-ScanOpt\artifacts\manifests\stage3_run_41_manifest.json`

## Scientific Boundary

This report records computed teacher metrics for Run40 native N24/N40 focused batch60. It does not claim physical superiority beyond computed teacher metrics and does not claim arbitrary-N generalisation.

## Guardrails

- ODB files were opened read-only only after 60/60 solver completion.
- No Abaqus solver job was run.
- No datacheck was run.
- No abqjobpilot/enqueue command was run.
- No CAE/INP/JNL or base sanity files were modified.
- No commit or push was made.
