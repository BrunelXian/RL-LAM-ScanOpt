# PPO Training Claim Boundary

## Safe After Stage C

- A MaskablePPO policy was trained in a surrogate terminal-reward environment derived from FEA teacher-labelled scan orders.
- The trained PPO policy can generate legal scan-order permutations in internal surrogate-environment evaluation.
- The model checkpoint, config, logs, and parameter count are frozen.

## Not Safe After Stage C

- PPO-generated candidates are Abaqus validated.
- PPO improves physical U2/PEEQ/SurfaceT.
- PPO outperforms teacher-validated baselines.
- PPO is final physical optimiser.

Those claims require Stage D candidate generation and Stage E Abaqus teacher validation.
