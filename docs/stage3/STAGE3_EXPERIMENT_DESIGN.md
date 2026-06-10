# Stage 3 Experiment Design

Stage 3 tests whether fixed-32 GNN/RL scan-order policy principles can transfer across track counts through a Variable-N Graph Pointer RL Policy.

## Track Counts

- Training: `N_train = {16, 32}`.
- Testing: `N_test = {24, 40}`.

## Metrics

- Primary: U2 / warpage.
- Safety: PEEQ.
- Secondary: SurfaceT inside the U2/PEEQ feasible or near-feasible region.
- Diagnostics: Gradient, Mises, and internal tensile stress.

## Ranking Protocol

Use within-N ranking and normalized improvement. Each N should have its own feasible or near-feasible comparison context. A full-32 U2 guard must not be reused as a universal cross-N acceptance rule.

## Initial Run Discipline

Runs 01-07 are documentation, audit, feature, baseline, dry-run, and handoff preparation only. Run 08 may read ODB evidence only after external Abaqus completion. This initialization performs no Abaqus execution, no datacheck, no ODB opening, and no model training.
