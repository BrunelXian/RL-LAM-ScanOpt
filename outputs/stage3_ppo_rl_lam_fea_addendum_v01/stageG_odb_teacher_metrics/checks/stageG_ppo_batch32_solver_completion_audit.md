# Stage G PPO Batch32 Solver Completion Audit

Verdict: `WARNING_STAGEG_PPO_BATCH32_SOLVER_COMPLETION_WITH_NONFATAL_WARNINGS`

- total: `32/32`
- warning: `32`
- lck present: `0`

| N | expected | complete | warning | failed_or_incomplete |
|---|---:|---:|---:|---:|
| N12 | 8 | 8 | 8 | 0 |
| N16 | 8 | 8 | 8 | 0 |
| N24 | 8 | 8 | 8 | 0 |
| N40 | 8 | 8 | 8 | 0 |

## Failed / Incomplete Cases

None.

## Gate

ODB extraction may proceed only when failed_or_incomplete is 0 and all expected cases are complete.
