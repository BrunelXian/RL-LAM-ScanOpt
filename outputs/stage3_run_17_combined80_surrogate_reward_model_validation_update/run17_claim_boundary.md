# Run 17 Claim Boundary

## Safe Claims
- Run17 updates lightweight surrogate validation using the expanded 80-case teacher-labelled variable-N dataset.
- Combined80 provides 20 cases per N for N12/N16/N24/N40.
- Run17 compares surrogate validation against run11 and evaluates whether batch20 improves surrogate stability.
- Pairwise preference data are expanded to 760 within-N pairs.
- Results can guide whether the next stage should be surrogate-screened candidate generation, active learning, or richer graph/sequence models.

## Unsafe Claims
- Do not claim trained variable-N RL policy superiority.
- Do not claim final surrogate accuracy.
- Do not claim arbitrary-N generalization.
- Do not claim physical optimum.
- Do not claim readiness to deploy.
- Do not claim feature importances are causal.
- Do not claim batch20 success proves surrogate is perfect, especially because run16 showed only moderate prediction-realization correlation.
