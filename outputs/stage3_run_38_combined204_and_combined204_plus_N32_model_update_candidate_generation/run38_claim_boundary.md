# Run38 Claim Boundary

## Safe Claims
- Run38 updates offline models using native combined204 and combined204_plus_N32.
- Run38 evaluates whether N32-augmented training helps native-N prediction after Run36 validation.
- Run38 generates candidate-order batches for future teacher validation.
- Run38 does not include teacher validation for new candidates.

## Unsafe Claims
- Do not claim new candidates are teacher-validated.
- Do not claim N32 itself was newly validated.
- Do not claim N32 caused improvement unless supported by diagnostic metrics.
- Do not claim GNN-RL superiority.
- Do not claim online RL.
- Do not claim arbitrary-N generalization.
- Do not claim physical superiority.
- Do not claim Abaqus was run.
