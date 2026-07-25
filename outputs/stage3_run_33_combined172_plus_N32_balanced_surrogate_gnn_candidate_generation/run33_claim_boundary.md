# Run33 Claim Boundary

## Safe Claims
- Run33 updates offline surrogate, GNN reward, and graph-pointer models using combined172_plus_N32.
- Run33 explicitly evaluates the effect of adding N32 legacy-compatible teacher labels.
- Run33 uses per-N balancing to avoid N32 dominance.
- Run33 generates new N32-informed candidate batches for future teacher validation.

## Unsafe Claims
- Do not claim Run33 candidates are teacher-validated.
- Do not claim N32 is native Stage 3 teacher validation.
- Do not claim PEEQ/Mises are semantically identical across Stage 2 and Stage 3.
- Do not claim GNN-RL superiority.
- Do not claim online RL.
- Do not claim arbitrary-N generalization.
- Do not claim physical superiority.
