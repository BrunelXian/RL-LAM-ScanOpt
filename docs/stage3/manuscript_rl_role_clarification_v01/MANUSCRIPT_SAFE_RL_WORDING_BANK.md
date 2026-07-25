# Manuscript-Safe RL Wording Bank

## A. Safe Title Phrases

- Reinforcement-learning formulation
- Teacher-guided offline policy optimization
- Offline policy-learning diagnostics
- Surrogate-guided active-learning scan-order generation
- Mask-aware sequential decision formulation
- Teacher-validated variable-N scan-order optimization

## B. Unsafe Title Phrases

- Online reinforcement learning with Abaqus
- PPO-based scan-order optimization
- Deployed GNN-pointer RL controller
- RL solved arbitrary-N scan-order optimization
- End-to-end GNN-RL control of deposition tracks

## C. Safe Contribution Wording

- We formulate scan-order construction as a mask-aware sequential decision problem.
- We implement a teacher-guided offline optimization loop that combines surrogate models, active-learning selection, policy-learning diagnostics, and Abaqus teacher validation.
- We report GNN and pointer-style components as auxiliary diagnostics that inform representation and sequence-learning behavior.
- The final evidence supports bounded variable-N optimization across tested native N values and the tested strategy space.

## D. Unsafe Contribution Wording

- We train a deployed GNN-pointer RL policy as the final optimizer.
- We use PPO as the main scan-order algorithm in the current paper.
- We solve arbitrary-N scan-order optimization.
- We train online reinforcement learning directly against Abaqus.
- GNN-RL outperforms all baselines.

## E. Safe Methods Wording

- The action at each step is the selection of the next unvisited track, with a feasibility mask preventing duplicate selections.
- Rewards are derived from teacher-labelled full scan orders rather than from an online step-wise Abaqus environment.
- Run26 includes a GNN reward regressor and a separate MLP pointer-style action scorer trained by reward-weighted behavior cloning.
- Later graph and pointer diagnostics are used as auxiliary evidence within the offline model-update workflow.
- Abaqus teacher simulations are used to evaluate selected scan orders and update the evidence base.

## F. Safe Results Wording

- The final teacher-validated evidence demonstrates bounded improvement within the tested native N values and strategy families.
- The strongest paper-facing claims should be tied to the frozen combined552 native teacher-labelled evidence package.
- GNN/pointer diagnostics support interpretation of sequence-learning behavior but are not the primary final result.
- The final teacher validation evaluates selected scan orders, not an online RL controller.

## G. Safe Discussion / Limitations Wording

- Direct online RL with Abaqus was outside the computational scope of this study.
- The final evidence does not establish global optimality or arbitrary-N generalization.
- GNN and pointer-style modules remain auxiliary policy-learning diagnostics in the current paper.
- Future work should evaluate whether a fully trained policy-gradient or actor-critic agent can improve the offline loop under a surrogate or reduced-order environment.

## H. Future-Work Wording for PPO

A future study may train a PPO or other policy-gradient agent in a surrogate or reduced-order environment and then validate the learned policy with Abaqus teacher simulations. This is outside the current paper.
