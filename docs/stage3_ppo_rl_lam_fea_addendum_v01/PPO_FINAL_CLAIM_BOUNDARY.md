# PPO Final Claim Boundary

## Safe Claims

- A MaskablePPO policy was trained in a surrogate terminal-reward environment derived from FEA teacher-labelled LDED scan-order data.
- PPO-only scan-order candidates were generated from the frozen PPO checkpoint.
- Thirty-two PPO-generated scan orders were converted into Abaqus CAE/INP cases.
- All 32 PPO-generated cases completed solver execution and yielded nonzero ODB files.
- Teacher metrics were extracted for 32/32 PPO cases.
- PPO batch32 was teacher-validated and small-N/top-k competitive.
- PPO achieved N12/N16 top-k competitiveness, with best lexicographic reference ranks 6 and 2 respectively.
- PPO produced no new combined552 records.
- N40 PPO performance was not competitive under primary metrics; only diagnostic Mises top-k hits were observed.
- PPO was trained in a surrogate environment, not online Abaqus.

## Unsafe Claims

- PPO outperformed all known scan strategies.
- PPO produced a new global best.
- PPO dominated all native N.
- PPO solved arbitrary-N scan-order optimisation.
- PPO was trained online in Abaqus.
- PPO is experimentally validated.
- PPO is first in the world.

Any "first PPO/RL+LAM+FEA" claim requires a separate literature-priority audit.
