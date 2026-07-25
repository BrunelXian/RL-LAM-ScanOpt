# Run 10 Claim Boundary

## Safe Claims

- Run10 creates a within-N normalized reward and surrogate-pretraining dataset from 60/60 teacher-labelled variable-N cases.
- The dataset supports future variable-N surrogate, reward-model, and preference-model experiments.
- The reward is explicitly U2-primary, PEEQ-safety-aware, and SurfaceT-secondary.
- Pairwise preference data are generated within each N only.
- Cross-N generalization splits are defined but not yet evaluated.

## Unsafe Claims

- Do not claim trained variable-N RL policy superiority.
- Do not claim surrogate accuracy.
- Do not claim arbitrary-N generalization.
- Do not claim a physical optimum.
- Do not claim fixed-32 absolute guard transfer.
- Do not claim reward V01-V05 are physically final; they are candidate reward formulations for later ablation.
