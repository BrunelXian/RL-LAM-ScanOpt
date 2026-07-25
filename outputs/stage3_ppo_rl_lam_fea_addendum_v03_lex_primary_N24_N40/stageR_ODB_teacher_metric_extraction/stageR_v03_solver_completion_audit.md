# Stage R v03 Solver Completion Audit

Verdict: `WARNING_STAGER_V03_SOLVER_COMPLETION_WITH_NONFATAL_WARNINGS`

- total: `32/32`
- warning: `32`
- lck present: `0`

| N | expected | complete | warning | failed_or_incomplete |
|---|---:|---:|---:|---:|
| N24 | 16 | 16 | 16 | 0 |
| N40 | 16 | 16 | 16 | 0 |

## Failed / Incomplete Cases

None.

## Gate

ODB extraction may proceed only when failed_or_incomplete is 0 and all expected cases are complete.
