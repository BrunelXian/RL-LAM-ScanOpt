# Run26 GNN Claim Boundary

## Safe claims
- Run26 implements an offline GNN / graph-pointer policy prototype.
- Run26 uses combined108 teacher-labelled data for policy/reward modelling.
- Run26 generates a future GNN-policy candidate batch for teacher validation.
- Run26 compares GNN-policy candidates with the previous surrogate active-learning shortlist64.
- Run26 does not perform teacher validation.

## Unsafe claims
- Do not claim the GNN-policy candidates are physically better.
- Do not claim GNN-RL has beaten baselines.
- Do not claim online RL with Abaqus.
- Do not claim arbitrary-N generalization.
- Do not claim deployment-ready policy.
- Do not claim ODB results exist.

Verdict: RUN26_OFFLINE_GNN_POLICY_PROTOTYPE_NO_TEACHER_VALIDATION
