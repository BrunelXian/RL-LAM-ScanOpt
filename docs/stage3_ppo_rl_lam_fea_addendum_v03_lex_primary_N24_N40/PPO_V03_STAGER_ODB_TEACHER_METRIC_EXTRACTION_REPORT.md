# Stage R v03 ODB Teacher Metric Extraction Report

## Verdict

`PASS_STAGER_V03_ODB_TEACHER_METRIC_EXTRACTION_32_OF_32`

## Solver Completion Audit

- total: `32/32`
- warning: `32`
- lck present: `0`

| N | expected | complete | warning | failed_or_incomplete |
|---|---:|---:|---:|---:|
| N24 | 16 | 16 | 16 | 0 |
| N40 | 16 | 16 | 16 | 0 |

## Stage Q/R Context

- N24 and N40 cases are included; no N12, N16, or N32 cases are included.
- Per-N counts: N24=16, N40=16.
- Metrics are extracted from the final frame of `step_final_cooling`.
- PPO v03 partial-training caveat is preserved in the audit and teacher metric rows.

## Selection Tags

- `diverse_upper_quartile`: `4` cases
- `fill_next_best`: `2` cases
- `lex_primary_novel`: `6` cases
- `record_seeking`: `2` cases
- `top_v03_score`: `10` cases
- `u2_guarded`: `8` cases

## Incomplete Cases

None.

## ODB Extraction Summary

- total: `32/32`
- final step: `step_final_cooling`
- required fields: `U;PEEQ;S;NT11`

| N | expected | extracted | failed |
|---|---:|---:|---:|
| N24 | 16 | 16 | 0 |
| N40 | 16 | 16 | 0 |

## Failed Extractions

None.

## Outputs

- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40\stageR_ODB_teacher_metric_extraction\stageR_v03_solver_completion_audit.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40\stageR_ODB_teacher_metric_extraction\stageR_v03_solver_completion_audit_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40\stageR_ODB_teacher_metric_extraction\stageR_v03_solver_completion_audit.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40\stageR_ODB_teacher_metric_extraction\stageR_v03_odb_extraction_summary.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40\stageR_ODB_teacher_metric_extraction\stageR_v03_odb_extraction_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40\stageR_ODB_teacher_metric_extraction\stageR_v03_teacher_metrics.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40\stageR_ODB_teacher_metric_extraction\stageR_v03_teacher_metrics.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40\stageR_ODB_teacher_metric_extraction\stageR_v03_extraction_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\docs\stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40\PPO_V03_STAGER_ODB_TEACHER_METRIC_EXTRACTION_REPORT.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40\stageR_ODB_teacher_metric_extraction\stageR_v03_odb_teacher_metric_extraction_manifest.json`

## Scientific Boundary

This report records Stage R v03 ODB-extracted teacher metrics only. It preserves the partial-training caveat and does not claim PPO candidate superiority, RL/GNN success, or arbitrary-N generalisation.

## Guardrails

- ODB files were opened read-only only after 32/32 solver completion.
- No Abaqus solver job was run.
- No datacheck was run.
- No abqjobpilot/enqueue command was run.
- No CAE/INP/JNL or base sanity files were modified.
- No commit or push was made.
