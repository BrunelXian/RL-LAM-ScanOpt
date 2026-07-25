# Stage 3 Manuscript RL Role Clarification Report

## 1. Purpose

Create an audit-backed manuscript support package that preserves reinforcement learning as an important formulation and policy-learning component while preventing unsupported claims about a deployed final GNN-pointer RL controller.

## 2. Inputs Inspected

- `scripts/stage3/run_26_gnn_graph_pointer_policy_candidate_generation.py`
- `docs/stage3/runs/run_26_combined108_gnn_graph_pointer_policy_candidate_generation/RUN_26_COMBINED108_GNN_GRAPH_POINTER_POLICY_CANDIDATE_GENERATION_REPORT.md`
- `src/policies/graph_pointer_policy.py`
- `src/policies/pointer_decoder.py`
- `docs/stage3/STAGE3_CLAIM_BOUNDARY.md`
- `outputs/stage3_run_78_final_evidence_freeze_package/stage3_final_claim_boundary.md`
- `outputs/stage3_run_78_final_evidence_freeze_package/stage3_final_claim_evidence_map.md`
- `outputs/stage3_run_78_final_evidence_freeze_package/stage3_final_paper_safe_conclusions.md`
- Later model-update output directories for Run43, Run48, Run53, Run58, Run63, Run68, and Run73.
- Old PPO files under `rl/` and older PPO assets.

## 3. Final Decision: Option B

Use Option B.

RL remains in the paper as the sequential decision formulation and as auxiliary offline policy-learning diagnostics. The current paper should not claim PPO, online RL, or a deployed GNN-pointer RL controller as the final optimizer.

## 4. What Can Be Claimed About RL

- The scan-order problem is formulated as a reinforcement-learning-style finite-horizon sequential decision problem.
- The action space uses a legality mask to prevent repeated track selection.
- Run26 explored offline policy-learning diagnostics, including a GNN reward regressor and a separate MLP pointer-style action scorer trained by reward-weighted behavior cloning.
- Later GNN/pointer components can be reported as auxiliary diagnostics.
- The implemented optimization loop is teacher-guided and offline, combining surrogate models, active-learning selection, policy diagnostics, and Abaqus teacher validation.

## 5. What Cannot Be Claimed About RL

- The current paper cannot claim a deployed online RL controller.
- The current paper cannot claim PPO as the main algorithm.
- The current paper cannot claim that the final Stage 3 result is a single GNN-encoded graph-pointer RL policy.
- The current paper cannot claim GNN-RL superiority over all baselines.
- The current paper cannot claim arbitrary-N scan-order optimization is solved.

## 6. Corrected Methods 2.2 Structure

Recommended title: "Reinforcement-learning formulation and offline policy-learning diagnostics."

Recommended structure:

- Define state, action, transition, episode, reward, and legality mask.
- State the implementation boundary: no online Abaqus-in-the-loop RL.
- Explain that direct online Abaqus RL is computationally prohibitive.
- Describe Run26 as an offline GNN reward-regression and pointer-style behavior-cloning prototype.
- Describe later GNN/pointer modules as auxiliary diagnostics.
- Point readers to teacher-guided active learning and Abaqus validation for the main implemented loop.

## 7. Corrected Table 2.3 Structure

Use the draft title: "Audited policy-learning and graph-diagnostic components."

The table should separate:

- MDP formulation
- Track graph representation
- Run26 GNN reward regressor
- Run26 MLP pointer-style scorer
- Feasibility mask
- Training objective
- Candidate decoding / inference
- Later GNN diagnostics
- Later graph-pointer diagnostics
- Final candidate selection loop
- Excluded PPO component

The table should explicitly state parameter count = NOT_FOUND, edge features = NOT_FOUND / none found, checkpoint metadata = NOT_FOUND, online RL = not used, and PPO = future work / excluded from current claim unless a reproducibility artifact is later added.

## 8. Current Paper Title Guidance

Safe title direction:

- Reinforcement-learning formulation and teacher-guided offline scan-order optimization.
- Surrogate-guided active learning with offline policy-learning diagnostics.
- Mask-aware sequential decision formulation for teacher-validated scan-order optimization.

Avoid title language implying online RL, PPO, or a deployed GNN-pointer controller.

## 9. PPO Future-Work Boundary

PPO should be reserved for a future paper. A future study may train a PPO or other policy-gradient agent in a surrogate or reduced-order environment and then validate the learned policy with Abaqus teacher simulations. This is outside the current paper.

## 10. Required Manuscript Revisions

- Replace any claim that the final optimizer is a deployed GNN-pointer RL controller.
- Replace PPO/current-algorithm phrasing with future-work wording.
- Revise Methods 2.2 to distinguish RL formulation from offline implementation.
- Revise Table 2.3 to list audited components rather than a unified architecture.
- Tie Results claims to the frozen teacher-validated evidence package.
- Add limitations language for online RL, PPO, arbitrary-N generalization, and global optimality.

## 11. Whether New Experiments Are Required

No new experiments are required for Option B.

No new Abaqus simulations, ODB extraction, solver runs, PPO training, RL/GNN/surrogate training, or candidate generation are required.

## 12. Verdict

PASS_MANUSCRIPT_RL_ROLE_CLARIFIED_OPTION_B_NO_NEW_ABAQUS_REQUIRED
