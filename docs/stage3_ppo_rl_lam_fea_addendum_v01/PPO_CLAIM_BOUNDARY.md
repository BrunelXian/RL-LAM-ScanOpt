# PPO Claim Boundary

## Safe Claims After Foundation Only

- A PPO addendum environment has been specified.
- An action-masked scan-order environment has been implemented.
- No PPO training has been performed yet.
- No FEA validation has been performed yet.

## Safe Claims After Future PPO Training But Before Abaqus

- A PPO policy was trained in a surrogate reward environment.
- PPO generated legal scan-order candidates.

These claims require PPO-specific artifacts such as a policy checkpoint, training log, PPO config, environment config, reward-model config, parameter count, seed manifest, inference candidate orders, and source audit.

## Safe Claims Only After Future Abaqus Validation

- PPO-generated scan orders were independently evaluated by Abaqus teacher simulations.
- The PPO policy produced teacher-validated candidate scan orders.

## Unsafe Claims Unless Specifically Proven

- Online Abaqus PPO was performed.
- PPO solved arbitrary-N scan-order optimisation.
- PPO globally optimised LDED scan order.
- PPO outperformed all known strategies.
- PPO-generated paths are experimentally validated.
- PPO is first in the world.

## Literature-Priority Note

Any "first" or "first true RL+LAM+FEA" claim requires a separate literature-priority audit before manuscript submission.

## Current Stage 3 Boundary

The existing Stage 3 final evidence package does not support claiming that final results came from a deployed PPO or GNN-pointer RL policy. PPO claims must be based only on future PPO-specific artifacts and PPO-generated candidates.
