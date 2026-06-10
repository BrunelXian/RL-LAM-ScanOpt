# Stage 3 Variable-N Policy Design

Stage 3 replaces the fixed-32 action interface with a Variable-N Graph Pointer RL Policy. The policy should operate on a graph whose nodes represent tracks and whose decoding process selects an ordered scan sequence.

## Design Requirements

- Accept variable track counts for the planned study: `N = 16, 24, 32, 40`.
- Train on `N_train = {16, 32}`.
- Test transfer on `N_test = {24, 40}`.
- Use masks only to enforce valid unvisited-track selection during pointer decoding.
- Keep fixed-32 policy artifacts frozen as Stage 2 reference evidence.

## Feature Scope

The first feature builder should stay lightweight and deterministic. Candidate graph features may include track index, coordinates, neighborhood relations, scan adjacency candidates, and normalized geometric descriptors. Physical labels remain teacher-derived evidence, not assumptions baked into the policy.

## Evaluation Interface

Policy outputs should be compared within each N using ranking and normalized improvement. Cross-N interpretation must avoid a single full-32 absolute U2 threshold.
