# Run68 Claim Boundary

## Safe claims
- Run68 updates offline models using native combined480 and combined480_plus_N32.
- Run68 generates candidate-order batches for future teacher validation.
- Run68 identifies small-N recovery, especially N12/N16, as the main follow-up direction.
- N12/N16 now have 48 native teacher rows each.
- N24/N40 remain mature teacher-labelled anchor groups with 184 and 200 native rows respectively.
- Full variable-N evidence is strengthened but still bounded by N12/N16 density.
- Run68 does not include teacher validation for new candidates.

## Unsafe claims
- Do not claim new candidates are teacher-validated.
- Do not claim N32 itself was newly validated.
- Do not claim N32 caused improvement.
- Do not claim GNN-RL superiority unless supported.
- Do not claim online RL.
- Do not claim arbitrary-N generalization.
- Do not claim physical optimum.
- Do not claim Abaqus was run.
