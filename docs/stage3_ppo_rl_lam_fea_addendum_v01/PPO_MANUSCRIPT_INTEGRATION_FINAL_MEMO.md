# PPO Manuscript Integration Final Memo

## 1. How This Changes The Current Paper

The PPO addendum upgrades the reinforcement-learning evidence from environment/planning language to a completed, teacher-validated PPO evidence chain: surrogate reward model, MaskablePPO policy training, checkpoint inference, Abaqus case conversion, solver completion, ODB metric extraction, and comparison against native combined552.

## 2. Recommended Title Direction

Use bounded language such as: "Surrogate-trained PPO policy generation with Abaqus teacher validation for LDED scan-order design." Avoid title wording that implies global optimisation or online Abaqus RL.

## 3. Recommended Abstract Wording Boundary

Safe abstract wording: "We trained a MaskablePPO policy in a surrogate terminal-reward environment derived from FEA teacher-labelled scan-order data, generated 32 PPO-only scan-order candidates, and independently evaluated them using Abaqus teacher simulations. The PPO batch achieved bounded small-N top-k competitiveness but produced no new combined552 records."

## 4. Recommended Methods Additions

- PPO surrogate reward environment: describe the terminal sparse reward emulator trained from native combined552 only.
- MaskablePPO policy training: report MaskablePPO + MlpPolicy, 200352 timesteps, and 72937 parameters.
- PPO-only candidate generation: state candidates came from frozen checkpoint inference and were not hand-mutated.
- Abaqus teacher validation: describe CAE/INP generation, solver execution, and ODB metric extraction for 32/32 cases.

## 5. Recommended Results Subsection

Suggested subsection title: "PPO-generated scan orders under Abaqus teacher validation."

Report: 32/32 evaluated, 0 new records, 12 top-k candidates, with N12/N16 competitiveness and limited N40 primary-metric performance.

## 6. Recommended Discussion Wording

The PPO result validates RL policy generation feasibility in this workflow. It does not establish global superiority. The surrogate-to-teacher alignment was weak positive, so independent teacher validation remains necessary. The strongest evidence is small-N top-k competitiveness; N40 remains limited.

## 7. Suggested Figure/Table List

- PPO evidence chain schematic.
- PPO performance summary by N.
- PPO vs combined552 metric distributions.
- Surrogate vs teacher alignment.
- Claim boundary table.

## 8. How To Avoid Overclaiming

Do not claim online Abaqus RL, experimental validation, new global bests, dominance over all native N, arbitrary-N optimisation, or first-in-world status. Use the final claim boundary: `E:\Projects\RL-LAM-ScanOpt\docs\stage3_ppo_rl_lam_fea_addendum_v01\PPO_FINAL_CLAIM_BOUNDARY.md`.
