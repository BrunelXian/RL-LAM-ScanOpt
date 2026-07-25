# Run49 Stricter Constrained N24/N40 Batch32 Review Summary

- Total count: 32
- Per-N counts: N24=16, N40=16
- Included N values: N24/N40 only; no N12, N16, or N32 candidates.
- Purpose: test stricter penalty guards after Run46 improved U2/reward but did not create raw PEEQ/SurfaceT/Mises records.
- Expected Abaqus cost: 32 jobs total, with 16 N24 and 16 N40.
- Teacher validation status: NOT_RUN. Run49 did not create CAE/INP files.

## Candidate Source Composition

- strict_penalty_guard_top: 4
- strict_penalty_guard_local_search: 4
- two_stage_guarded_top: 4
- two_stage_guarded_local_search: 4
- no_penalty_worse_than_median: 4
- PEEQ_repair_candidates: 4
- SurfaceT_repair_candidates: 4
- Mises_repair_candidates: 4

## Selection Bucket Composition

- strict_penalty_guard_top: 4
- strict_penalty_guard_local_search: 4
- two_stage_guarded_top: 4
- two_stage_guarded_local_search: 4
- no_penalty_worse_than_median: 4
- PEEQ_repair_candidates: 4
- SurfaceT_repair_candidates: 4
- Mises_repair_candidates: 4
