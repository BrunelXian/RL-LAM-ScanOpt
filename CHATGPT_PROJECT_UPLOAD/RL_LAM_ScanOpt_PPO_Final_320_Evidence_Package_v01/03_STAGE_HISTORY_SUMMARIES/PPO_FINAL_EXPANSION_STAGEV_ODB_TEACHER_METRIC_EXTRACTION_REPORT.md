# Stage V PPO Final Expansion 224 ODB Teacher Metric Extraction Report

## Verdict

`PASS_STAGEV_FINAL_EXPANSION_224_ODB_TEACHER_METRIC_EXTRACTION_224_OF_224`

## Solver Completion Audit

- total: `224/224`
- warning: `224`
- lck present: `0`

| N | expected | complete | warning | failed_or_incomplete |
|---|---:|---:|---:|---:|
| N12 | 32 | 32 | 32 | 0 |
| N16 | 32 | 32 | 32 | 0 |
| N24 | 80 | 80 | 80 | 0 |
| N40 | 80 | 80 | 80 | 0 |

## Stage U/V Context

- N12, N16, N24, and N40 cases are included; no N32 cases are included.
- Per-N counts: N12=32, N16=32, N24=80, N40=80.
- Seven final expansion batches are included, each with 32 cases.
- Metrics are extracted from the final frame of `step_final_cooling`.

## Batch Summary

| final_expansion_batch | expected | complete | warning | failed_or_incomplete |
|---|---:|---:|---:|---:|
| final_expansion_batch01 | 32 | 32 | 32 | 0 |
| final_expansion_batch02 | 32 | 32 | 32 | 0 |
| final_expansion_batch03 | 32 | 32 | 32 | 0 |
| final_expansion_batch04 | 32 | 32 | 32 | 0 |
| final_expansion_batch05 | 32 | 32 | 32 | 0 |
| final_expansion_batch06 | 32 | 32 | 32 | 0 |
| final_expansion_batch07 | 32 | 32 | 32 | 0 |

## Incomplete Cases

None.

## ODB Extraction Summary

- total: `224/224`
- final step: `step_final_cooling`
- required fields: `U;PEEQ;S;NT11`

| N | expected | extracted | failed |
|---|---:|---:|---:|
| N12 | 32 | 32 | 0 |
| N16 | 32 | 32 | 0 |
| N24 | 80 | 80 | 0 |
| N40 | 80 | 80 | 0 |

| final_expansion_batch | expected | extracted | failed |
|---|---:|---:|---:|
| final_expansion_batch01 | 32 | 32 | 0 |
| final_expansion_batch02 | 32 | 32 | 0 |
| final_expansion_batch03 | 32 | 32 | 0 |
| final_expansion_batch04 | 32 | 32 | 0 |
| final_expansion_batch05 | 32 | 32 | 0 |
| final_expansion_batch06 | 32 | 32 | 0 |
| final_expansion_batch07 | 32 | 32 | 0 |

## Failed Extractions

None.

## Outputs

- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_expansion_224_to_320\stageV_ODB_teacher_metric_extraction\stageV_completion_gate.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_expansion_224_to_320\stageV_ODB_teacher_metric_extraction\stageV_completion_gate_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_expansion_224_to_320\stageV_ODB_teacher_metric_extraction\stageV_completion_gate.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_expansion_224_to_320\stageV_ODB_teacher_metric_extraction\stageV_ppo_final_expansion_224_ODB_extraction_summary.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_expansion_224_to_320\stageV_ODB_teacher_metric_extraction\stageV_ppo_final_expansion_224_ODB_extraction_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_expansion_224_to_320\stageV_ODB_teacher_metric_extraction\stageV_ppo_final_expansion_224_ODB_metrics.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_expansion_224_to_320\stageV_ODB_teacher_metric_extraction\stageV_ppo_final_expansion_224_ODB_metrics_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_expansion_224_to_320\stageV_ODB_teacher_metric_extraction\stageV_ppo_final_expansion_224_extraction_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\docs\stage3_ppo_final_expansion_224_to_320\PPO_FINAL_EXPANSION_STAGEV_ODB_TEACHER_METRIC_EXTRACTION_REPORT.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_expansion_224_to_320\stageV_ODB_teacher_metric_extraction\stageV_ppo_final_expansion_224_extraction_manifest.json`

## Scientific Boundary

This report records Stage V read-only ODB-extracted teacher metrics only. It does not claim PPO candidate physical superiority, RL/GNN success, or arbitrary-N generalisation.

## Guardrails

- ODB files were opened read-only only after 224/224 solver completion.
- No Abaqus solver job was run.
- No datacheck was run.
- No abqjobpilot/enqueue command was run.
- No CAE/INP/JNL or base sanity files were modified.
- No commit or push was made.
