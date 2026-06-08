# Stage 2 Claim Boundary

## Safe Claims
- The 32-track search space is `32! ≈ 2.63 × 10^35`.
- Stage 2 should not be framed as brute-force search.
- The final objective hierarchy is U2 primary, PEEQ safety, SurfaceT secondary, Gradient/Mises/internal diagnostics.
- GNN/RL is successful policy-learning / agent-feasibility evidence.
- GNN/RL has teacher-validated advantage over 9/10 labelled early full-32 baselines.
- SurfaceT has diagnostic ranking signal.
- Gradient remains weak/diagnostic.
- Transformer did not improve over ExtraTrees under current teacher data.
- Masked probe40 exposes a masked generalisation boundary.

## Conditional Claims
- GNN/RL exceeds all earliest 10 baselines only if `smartscan_proxy_variance` is teacher-validated or excluded with explicit justification.
- SurfaceT-guided generation can be claimed only as diagnostic unless U2/PEEQ feasibility and teacher SurfaceT improvement are confirmed.
- Masked generalisation can be claimed only after per-mask U2/PEEQ guard calibration.

## Unsafe Claims
- Global optimum found.
- SurfaceT optimisation solved.
- GNN/RL is the final physical optimiser.
- Transformer is superior to ExtraTrees.
- Masked transfer success is proven.
- Variable-N generalisation is already solved.

## Missing Evidence
- `smartscan_proxy_variance` teacher label is missing from the earliest 10 baseline comparison.
- SurfaceT improvement over the best existing reference is not demonstrated by UST probe10.
- Masked feasible SurfaceT region is not established under the full-32 absolute guard.

## Suggested Paper Wording
“Stage 2 demonstrates a teacher-guided learning framework for scan-order optimisation in a 32-track search space of approximately `2.63 × 10^35` permutations. The final evidence supports a U2-first, PEEQ-safe, SurfaceT-secondary objective hierarchy. GNN/RL models provide policy-learning and candidate-generation evidence, while physical claims remain bounded by teacher validation.”

## Overclaims To Avoid
- “The optimiser solved scan-ordering.”
- “GNN/RL beats all baselines.”
- “SurfaceT optimisation is solved.”
- “Masked generalisation is proven.”
- “Transformer is the best model.”
