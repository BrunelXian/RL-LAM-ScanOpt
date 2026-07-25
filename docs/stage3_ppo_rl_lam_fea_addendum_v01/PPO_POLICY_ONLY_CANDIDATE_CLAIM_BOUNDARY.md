# PPO Policy-Only Candidate Claim Boundary

## Safe After Stage D

- A trained MaskablePPO policy generated legal scan-order candidates.
- Candidate orders were selected from PPO checkpoint inference only.
- Candidate orders are ready for future Abaqus teacher validation handoff.
- Surrogate scores are predictions only.

## Not Safe After Stage D

- PPO candidates are physically validated.
- PPO improves U2/PEEQ/SurfaceT under Abaqus.
- PPO outperforms teacher-validated baselines.
- PPO is final physical optimiser.

## Stage E Required

- Generate CAE/INP from these 32 candidates.
- Run Abaqus.
- Extract U/PEEQ/S/NT11 teacher metrics.
- Compare against combined552 and baselines.
