# Table 2.3 Draft: Audited policy-learning and graph-diagnostic components

This replacement avoids presenting a fake unified GNN-pointer architecture. It separates formulation, reward modelling, pointer-style scoring, masks, diagnostics, candidate selection, and excluded PPO code.

| component | implementation | role in current paper | evidence | caveat |
|---|---|---|---|---|
| MDP formulation | Finite-horizon sequential decision problem; each step selects the next unvisited track. | Conceptual RL-style framing for scan-order construction. | `docs/stage3/STAGE3_CLAIM_BOUNDARY.md`; final claim boundary. | Formulation only; not proof of online RL training. |
| Track graph representation for Run26 GNN reward model | Adjacent-track graph / line-graph style message passing over track nodes and node/order features. | Offline reward-regression diagnostic. | `scripts/stage3/run_26_gnn_graph_pointer_policy_candidate_generation.py`; Run26 report. | Edge features = NOT_FOUND / none found as a reproducibility artifact. |
| Run26 GNN reward regressor | Plain PyTorch message-passing reward model trained on combined108 teacher-labelled rows. | Auxiliary evidence that graph/order representations can regress teacher-normalized reward. | `gnn_reward_model_validation_summary.json`; `gnn_reward_model_validation_results.csv`. | Parameter count = NOT_FOUND unless a reproducibility artifact exists; checkpoint metadata = NOT_FOUND. |
| Run26 MLP pointer-style action scorer | MLP action scorer over candidate-track features with masked decoding. | Offline policy-learning diagnostic. | `graph_pointer_policy_validation_summary.json`; `graph_pointer_policy_training_log.csv`. | Separate from the GNN reward regressor; not a GNN pointer policy. |
| Feasibility mask | Already visited tracks are masked at each decoding step. | Enforces legal permutations. | Run26 script lines around pointer loss/decoding; `src/policies/pointer_decoder.py`. | Masking is a legality mechanism, not evidence of RL training. |
| Training objective | Reward-weighted behavior cloning / weighted imitation on teacher-labelled sequences. | Offline imitation-style policy diagnostic. | Run26 report: training method is offline weighted behavior cloning. | Not PPO, not policy gradient, not actor-critic. |
| Candidate decoding / inference | Greedy/beam/temperature-style and local-search candidate generation in Run26; later selection driven by model updates and active-learning criteria. | Historical candidate-generation component, not final proof of deployed controller. | Run26 candidate files; later model-update run candidate pools. | The current clarification package does not generate candidates. |
| Later GNN diagnostic | GNN reward diagnostics in Run43, Run48, Run53, Run58, Run63, Run68, and Run73 artifacts. | Auxiliary model-update diagnostics. | Later `*_gnn_reward_validation_summary.json` files. | Not primary final evidence; final package says GNN/pointer diagnostics are auxiliary. |
| Later graph-pointer diagnostic | Pointer-style training logs/validation summaries in later model-update artifacts. | Auxiliary policy-learning or sequence diagnostic. | Later `*_graph_pointer_policy_validation_summary.json` files. | Online RL = not used. |
| Final candidate selection loop | Teacher-guided offline active-learning and surrogate-guided model updates, followed by Abaqus teacher validation of selected batches. | Main algorithmic frame for current paper. | Run23/29/43/48/53/58/63/68/73 generation artifacts; Run76/77/78 validation/freeze artifacts. | Final teacher validation evaluates selected scan orders, not an online RL controller. |
| Excluded PPO component | Legacy Maskable PPO scripts under `rl/` and older assets. | Future-work context only. | `rl/train_maskable_ppo.py`; `rl/eval_policy.py`; older PPO assets. | PPO = future work / excluded from current claim. |

Required metadata statements:

- Parameter count = NOT_FOUND unless a reproducibility artifact exists.
- Edge features = NOT_FOUND / none found as a reproducibility artifact.
- Checkpoint metadata = NOT_FOUND.
- Online RL = not used.
- PPO = future work / excluded from current claim.
