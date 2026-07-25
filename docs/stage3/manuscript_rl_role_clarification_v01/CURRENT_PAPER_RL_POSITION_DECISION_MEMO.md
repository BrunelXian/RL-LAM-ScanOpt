# Current Paper RL Position Decision Memo

## Decision

Use Option B.

## Meaning

- Keep RL as formulation and auxiliary policy-learning module.
- Do not claim PPO or online RL.
- Do not claim a deployed GNN-pointer RL policy as the final optimizer.
- Current paper main algorithmic frame: RL-style sequential formulation plus teacher-guided offline active-learning optimization.
- Future paper: PPO / online or surrogate-environment RL policy training.

## Why This Avoids Overclaiming

The frozen Stage 3 evidence package states that GNN and graph-pointer diagnostics are auxiliary, not the primary evidence for final claims. Run26 contains an offline GNN reward regressor and a separate MLP pointer-style action scorer trained by reward-weighted behavior cloning, not PPO or online actor-critic training. Later selected candidate batches came from hybrid active-learning / surrogate-guided model-update loops and Abaqus teacher validation, not from a deployed online RL controller.

Option B preserves the honest value of the RL framing while aligning the manuscript with the actual evidence.

## How To Revise Title

Prefer phrases such as:

- Reinforcement-learning formulation and teacher-guided offline scan-order optimization
- Mask-aware sequential decision formulation for teacher-validated scan-order optimization
- Surrogate-guided active learning with offline policy-learning diagnostics

Avoid:

- PPO-based scan-order optimization
- Online RL with Abaqus
- Deployed GNN-pointer RL controller

## How To Revise Methods

Methods 2.2 should define the MDP and legality mask, then immediately state the implementation boundary: the paper uses offline teacher-guided optimization rather than online Abaqus-in-the-loop RL.

Run26 should be described as an offline prototype with a GNN reward regressor and a separate reward-weighted behavior-cloning pointer-style scorer. Later GNN/pointer modules should be described as diagnostics.

## How To Revise Results

Tie final performance claims to teacher-validated selected scan orders and the frozen Stage 3 evidence package. Report GNN/pointer results only as auxiliary diagnostics unless a separate RL-only validation is added in a future study.

## How To Revise Discussion

State that direct online RL and PPO training are outside the current paper. Discuss them as future work that would require a surrogate or reduced-order environment and subsequent Abaqus teacher validation.

## Whether New Experiments Are Required

No new Abaqus or RL training is required for Option B.

No PPO training, new RL policy training, new surrogate training, or new candidate generation is required.
