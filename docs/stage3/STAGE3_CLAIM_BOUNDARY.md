# Stage 3 Claim Boundary

## Allowed Claims

- Stage 3 proposes and tests Variable-N Graph Pointer RL Policy feasibility.
- Stage 3 evaluates N-track transfer from `N_train = {16, 32}` to `N_test = {24, 40}`.
- Stage 3 uses within-N ranking and normalized improvement for cross-N evidence.

## Disallowed Claims

- Arbitrary-N generalisation is solved.
- GNN/RL is the final physical optimiser.
- The same full-32 U2 guard transfers to all N.
- Masked transfer is solved.
- SurfaceT optimisation is solved.

## Interpretation Rule

Any positive result is evidence for feasibility under the tested N values and evaluation protocol only. It should be framed as a transfer test, not as a universal scan-order optimiser.
