# Run57 Claim Boundary

## Safe claims

- Run57 ingests 64/64 teacher-validated Run56 calibrated N24/N40 batch64 cases.
- Run57 builds native combined392 with N12=36, N16=36, N24=160, N40=160.
- Run57 builds combined392_plus_N32 with N12=36, N16=36, N24=160, N32=332, N40=160.
- Run57 evaluates whether calibrated batch64 candidate selection improved native Stage 3 teacher metrics.
- Run56 is teacher validation of native N24/N40 calibrated candidates, not N32 cases.
- Run57 audits whether N24/N40 are approaching a mature teacher-label density for offline active-learning / offline-RL evidence.

## Unsafe claims

- Do not claim N32 itself was newly teacher-validated in Run56.
- Do not claim N32 caused Run56 improvements.
- Do not claim GNN-RL superiority.
- Do not claim online RL.
- Do not claim arbitrary-N generalization.
- Do not claim physical optimum.
- Do not claim solver/ODB extraction happened in Run57.
- Do not claim full variable-N RL maturity if N12/N16 remain under-sampled.
