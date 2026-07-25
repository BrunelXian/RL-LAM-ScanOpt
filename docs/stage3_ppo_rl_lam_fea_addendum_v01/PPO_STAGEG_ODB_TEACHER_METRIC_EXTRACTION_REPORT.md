# Stage G PPO Batch32 ODB Teacher Metric Extraction Report

## Verdict

`PASS_STAGEG_PPO_BATCH32_ODB_TEACHER_METRIC_EXTRACTION_32_OF_32`

## Solver Completion Audit

- total: `32/32`
- warning: `32`
- lck present: `0`

| N | expected | complete | warning | failed_or_incomplete |
|---|---:|---:|---:|---:|
| N12 | 8 | 8 | 8 | 0 |
| N16 | 8 | 8 | 8 | 0 |
| N24 | 8 | 8 | 8 | 0 |
| N40 | 8 | 8 | 8 | 0 |

## Stage E/F Context

- N12, N16, N24, and N40 cases are included; no N32 cases are included.
- Per-N counts: N12=8, N16=8, N24=8, N40=8.
- Metrics are extracted from the final frame of `step_final_cooling`.
- `PPOV01_N12_B02_surrogate_top` is preserved as the known duplicate/recovery-anchor case, which is not a failure.

## Recovery Anchor Case

- `PPOV01_N12_B02_surrogate_top`: `WARNING_SUCCESS_WITH_WARNINGS`; complete_with_nonfatal_warnings

## Incomplete Cases

None.

## ODB Extraction Summary

- total: `32/32`
- final step: `step_final_cooling`
- required fields: `U;PEEQ;S;NT11`

| N | expected | extracted | failed |
|---|---:|---:|---:|
| N12 | 8 | 8 | 0 |
| N16 | 8 | 8 | 0 |
| N24 | 8 | 8 | 0 |
| N40 | 8 | 8 | 0 |

## Failed Extractions

None.

## Outputs

- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageG_odb_teacher_metrics\checks\stageG_ppo_batch32_solver_completion_audit.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageG_odb_teacher_metrics\checks\stageG_ppo_batch32_solver_completion_audit.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageG_odb_teacher_metrics\checks\stageG_ppo_batch32_solver_completion_audit.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageG_odb_teacher_metrics\tables\stageG_ppo_batch32_odb_extraction_summary.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageG_odb_teacher_metrics\tables\stageG_ppo_batch32_odb_extraction_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageG_odb_teacher_metrics\tables\stageG_ppo_batch32_teacher_metrics.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageG_odb_teacher_metrics\tables\stageG_ppo_batch32_teacher_metrics_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageG_odb_teacher_metrics\stageG_ppo_batch32_odb_teacher_metric_extraction_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\docs\stage3_ppo_rl_lam_fea_addendum_v01\PPO_STAGEG_ODB_TEACHER_METRIC_EXTRACTION_REPORT.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageG_odb_teacher_metrics\stageG_ppo_batch32_odb_teacher_metrics_manifest.json`

## Scientific Boundary

This report records PPO Stage G ODB-extracted teacher metrics only. It does not claim PPO candidate superiority, RL/GNN success, or arbitrary-N generalisation.

## Guardrails

- ODB files were opened read-only only after 32/32 solver completion.
- No Abaqus solver job was run.
- No datacheck was run.
- No abqjobpilot/enqueue command was run.
- No CAE/INP/JNL or base sanity files were modified.
- No commit or push was made.
