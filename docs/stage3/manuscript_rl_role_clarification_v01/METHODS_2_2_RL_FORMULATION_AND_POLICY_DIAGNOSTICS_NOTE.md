# Reinforcement-learning formulation and offline policy-learning diagnostics

## Recommended Methods 2.2 Title

Reinforcement-learning formulation and offline policy-learning diagnostics

## MDP Definition

State: a partial scan-order prefix, the set of already visited tracks, track index/position features, optional summary features of the current prefix, and the fixed build geometry for the selected native N.

Action: select the next unvisited track from the legal action set.

Transition: append the selected track to the prefix and update the visited mask. The transition is deterministic in the offline sequence-construction problem.

Episode: one complete scan order for a fixed N, ending after all tracks have been selected exactly once.

Reward: teacher-derived or teacher-normalized scan-order quality after the full episode, using the paper's bounded reward definitions over U2/warpage and secondary physical metrics. Intermediate rewards are not required for the current offline formulation.

Legality mask: at each decoding step, tracks already present in the prefix are masked out. The action space therefore shrinks until all tracks have been assigned.

## Implementation Boundary

The scan-order problem is formulated as an RL-style sequential decision problem, but the final implementation is not online reinforcement learning with Abaqus in the loop.

Direct online Abaqus-in-the-loop reinforcement learning is computationally prohibitive for this paper because every action sequence would require expensive teacher simulation before a reliable terminal reward is available.

The implemented optimization loop is teacher-guided and offline. It combines surrogate models, policy-learning diagnostics, novelty/diversity and uncertainty-aware selection, and Abaqus teacher validation of selected scan orders.

The final Stage 3 evidence package supports bounded variable-N scan-order optimization over tested N values and the tested strategy space. It does not support describing the main result as a single deployed GNN-encoded graph-pointer RL policy.

## Run26 Prototype

Run26 is best described as an offline policy-learning prototype and diagnostic.

- It includes a GNN reward regressor over an adjacent-track graph / line-graph representation.
- It includes a separate MLP pointer-style action scorer.
- The pointer-style scorer is trained by reward-weighted behavior cloning on teacher-labelled sequences.
- Decoding uses a feasibility mask so already selected tracks cannot be selected again.
- It is not PPO.
- It is not actor-critic learning.
- It is not online reinforcement learning.
- It is not Abaqus-in-the-loop policy-gradient training.
- It is not the final deployed optimizer claimed by the frozen Stage 3 evidence package.

Core evidence:

- `scripts/stage3/run_26_gnn_graph_pointer_policy_candidate_generation.py`
- `outputs/stage3_run_26_combined108_gnn_graph_pointer_policy_candidate_generation/gnn_reward_model_validation_summary.json`
- `outputs/stage3_run_26_combined108_gnn_graph_pointer_policy_candidate_generation/graph_pointer_policy_validation_summary.json`
- `docs/stage3/runs/run_26_combined108_gnn_graph_pointer_policy_candidate_generation/RUN_26_COMBINED108_GNN_GRAPH_POINTER_POLICY_CANDIDATE_GENERATION_REPORT.md`

## Later Diagnostics

Later model-update runs contain surrogate, GNN, graph-pointer, and related order diagnostics. These are useful as auxiliary evidence about representation learning, candidate ranking, and sequence structure, but they should not be promoted to a single final deployed controller.

OrderGraphMLP diagnostic: report, if used, as an auxiliary graph/order model diagnostic within the offline model-update loop.

Transition-frequency pointer diagnostic: report, if used, as an auxiliary pointer-style sequence diagnostic or empirical transition-preference diagnostic.

The frozen final package explicitly states that GNN and graph-pointer diagnostics are auxiliary, not primary evidence for final claims.

## Manuscript-Safe Paragraph Bullets

Safe wording:

- We formulate scan-order optimization as a finite-horizon sequential decision problem with legal-action masking.
- Because direct Abaqus-in-the-loop online RL would be computationally prohibitive, the implementation uses teacher-guided offline optimization.
- The offline loop combines surrogate-guided active learning, policy-learning diagnostics, novelty/diversity selection, and Abaqus teacher validation.
- Run26 provides an offline GNN reward-regression diagnostic and a separate reward-weighted behavior-cloning pointer-style scorer.
- GNN/pointer components are reported as auxiliary diagnostics and policy-learning evidence, not as the sole deployed final optimizer.

Unsafe wording:

- We trained PPO with Abaqus in the loop.
- The final Stage 3 optimizer is a deployed GNN-pointer RL controller.
- A GNN-RL policy outperformed all baselines.
- The method solves arbitrary-N scan-order optimization.
- The final teacher validation evaluates an online RL controller.

Suggested figure/table references:

- Table 2.3: use "Audited policy-learning and graph-diagnostic components."
- Results claim boundary: cite the final Stage 3 evidence package and claim map.
- Supplementary table: include the RL role separation table to separate formulation, diagnostics, active learning, and teacher validation.
