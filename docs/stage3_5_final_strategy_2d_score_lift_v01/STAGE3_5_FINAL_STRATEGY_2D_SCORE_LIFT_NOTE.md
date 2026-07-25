# Stage 3.5 Final Strategy 2D Score Lift Note

## Purpose

Stage 3.5 creates an isolated score-function lift from the frozen Stage 3 final native best strategy table. It derives a one-dimensional score vector from final native best track orders, then lifts that vector to a two-dimensional score matrix.

This module does not generate a cell traversal sequence, does not create a new scan strategy, and does not modify Stage 3 frozen evidence.

## Source Of s(i)

The one-dimensional score `s(i)` is derived from final Stage 3 native best strategy `order_json` or, when JSON is unavailable, `order_compact`.

The intended native N values are:

- N12
- N16
- N24
- N40

N32 is legacy auxiliary evidence and is not used as native evidence here.

## Rank-Based Score Derivation

For each legal final order, `rank(i)` is the position of track `i` in that final order. The score vector is indexed by track id:

```text
s(i) = eps + (1 - 2*eps) * (1 - rank(i)/(N-1))
eps = 1e-6
```

This maps earlier tracks in the final order to larger scores while keeping every score strictly inside `(0,1)`.

## Unit-Normalized 2D Lift

The preferred lifted score matrix is:

```text
s_new(i,j) = sqrt((s(i)^2 + s(j)^2) / 2)
```

The unit-normalized form is preferred because its values remain strictly inside `(0,1)` when `s(i)` is strictly inside `(0,1)`. It also preserves the diagonal identity:

```text
s_new(i,i) = s(i)
```

## Raw 2D Lift

The raw lifted score matrix is also saved for comparison:

```text
s_new_raw(i,j) = sqrt(s(i)^2 + s(j)^2)
```

The raw form can range up to but not including `sqrt(2)`, so it is not unit-normalized.

## No Cell Traversal Sequence

A 32 by 32 area contains 1024 cells, but this Stage 3.5 module does not rank or sort area cells into a traversal sequence. It only writes score vectors and score matrices. The matrices can be inspected as area-level score fields, but they are not paths.

## Not Teacher Validation

This stage performs no Abaqus run, no ODB extraction, no solver call, and no teacher-metric validation. It is a deterministic post-processing lift from already frozen final Stage 3 strategy orders.

## Future Use

The lifted matrix could later serve as an area-level scan-priority prior. Turning it into a validated area scan planner would require a separate method for path construction, constraints, evaluation, and teacher validation.
