# Stage 3.5 Final Strategy 2D Score Lift Claim Boundary

## Safe Claims

- Stage 3.5 derives `s(i)` from final Stage 3 native best `order_json` or `order_compact`.
- Stage 3.5 computes a two-dimensional lifted score matrix `s_new(i,j)`.
- The score lift is deterministic and reproducible.
- The unit-normalized form remains in `(0,1)`.
- This can serve as a future area-level scan-priority prior.

## Unsafe Claims

- It is teacher-validated.
- It generates a 1024-point scan strategy.
- It improves U2, PEEQ, SurfaceT, or Mises.
- It solves area scan-path planning.
- It extends Stage 3 final evidence.
- It proves arbitrary-area or arbitrary-N generalization.
- It uses N32 as native evidence.

## Boundary Statement

This Stage 3.5 module is a score-function lift only. It is not a new Stage 3 evidence run, not a teacher-validated strategy, and not a scan-path planner.
