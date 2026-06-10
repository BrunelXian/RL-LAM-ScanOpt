# Run 02 Fixed-32 Policy Novelty Audit Report

## Executive Verdict

PASS_FIXED32_POLICY_NOVELTY_AUDIT_READY

## What Was Audited

- Legal fixed-32 engineering baseline orders: `60`
- Legal fixed-32 GNN/RL orders: `120`
- Legal fixed-32 orders total: `180`
- Source files scanned: `2247`
- Rejected/non-legal order candidates: `45276`
- Rejection audit rows written: `1000` of max `1000`
- Teacher metrics linked: `True`

## Guardrails

- No Abaqus jobs.
- No datacheck.
- No ODB opened.
- No CAE/INP/JNL generated.
- No model training.
- Stage 2 source was read-only.
- No new physical candidates were generated for CAE.

## Core Findings

- RL/GNN vs engineering baseline directed adjacent-pair overlap: `min=0.0000, median=0.0645, max=0.9032`
- RL/GNN vs engineering baseline Kendall distance: `min=0.0020, median=0.5000, max=0.8810`
- Closest baseline family mode across learned candidates: `odd_even`
- Structural interpretation: The learned fixed-32 candidates are structurally distinct from the audited engineering baselines under adjacent-transition and permutation-distance metrics.

## Teacher U2 Linkage

Teacher U2 metrics were linked for at least one labelled baseline and one labelled learned candidate.

- Best labelled engineering baseline by U2: `{'strategy_name': 'mc_extra_017', 'family': 'method_c', 'U2': 8.741542e-05}`
- Best labelled GNN/RL candidate by U2: `{'strategy_name': 'RLU2M_A28_V01', 'family': 'RLU2M', 'U2': 4.7092337e-05}`

Run_02 remains a structural audit only; it is not new teacher validation.

## Claim Boundary

This audit does not prove variable-N generalisation. It does not prove arbitrary-N optimisation. It does not prove masked transfer. It does not prove SurfaceT optimisation. It does not replace Abaqus teacher validation.

## Outputs

- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_02_policy_novelty_audit\extraction_rejection_audit.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_02_policy_novelty_audit\fixed32_orders_extracted.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_02_policy_novelty_audit\adjacent_pair_overlap_matrix.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_02_policy_novelty_audit\undirected_adjacent_pair_overlap_matrix.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_02_policy_novelty_audit\kendall_distance_matrix.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_02_policy_novelty_audit\spearman_rank_distance_matrix.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_02_policy_novelty_audit\jump_length_summary.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_02_policy_novelty_audit\edge_center_timing_profile.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_02_policy_novelty_audit\left_right_balance_summary.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_02_policy_novelty_audit\rl_vs_baseline_novelty_summary.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_02_policy_novelty_audit\policy_novelty_summary.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_02_policy_novelty_audit\adjacent_pair_overlap_heatmap.png`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_02_policy_novelty_audit\kendall_distance_heatmap.png`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_02_policy_novelty_audit\jump_length_distribution.png`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_02_policy_novelty_audit\edge_center_timing_profile.png`
- `E:\Projects\RL-LAM-ScanOpt\artifacts\manifests\stage3_run_02_manifest.json`

Plotting status: `plots written`

## Recommended Next Run

`run_03_variable_n_graph_feature_builder`

## Final Verdict

PASS_FIXED32_POLICY_NOVELTY_AUDIT_READY
