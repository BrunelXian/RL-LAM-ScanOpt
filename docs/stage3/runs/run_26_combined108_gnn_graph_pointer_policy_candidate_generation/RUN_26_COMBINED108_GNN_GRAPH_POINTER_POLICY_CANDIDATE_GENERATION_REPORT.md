# Stage 3 Run 26 - Combined108 GNN / Graph-Pointer Policy Candidate Generation

## Purpose
Insert an offline GNN / graph-pointer policy step before spending new solver time, so the next candidate batch can be positioned as GNN-policy-generated rather than purely surrogate active-learning generated.

## Run25 Suspended Status
Run25 shortlist64 CAE/INP generation is suspended by user decision. No Run25 Abaqus/CAE/INP/solver activity should be executed before Run26.

## Why Run26 Is Inserted Before New CAE/Solver Validation
Run23/Run24 shortlist64 remains a valid surrogate active-learning control batch, but Run26 creates a graph-policy batch for paper-mainline consideration before committing the next 60+ teacher-validation jobs.

## Inputs
- Combined108 RL-ready dataset: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_21_batch28_teacher_metrics_ingestion_and_combined108_ranking\combined108_RL_ready_dataset.csv`
- Run22 surrogate diagnostics: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_22_combined108_surrogate_reward_model_validation_update\combined108_best_surrogate_configurations.csv`
- Run23 candidate pool: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_23_combined108_active_learning_coverage_calibration_design\run23_candidate_pool_scored.csv`
- Run24 shortlist64: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\stage3_run24_shortlist64_candidate_orders.csv`

## Graph Formulation
Each track is a graph node. The prototype uses normalized track position, parity, center/edge distance, visit position, first/last flags, and adjacent line-graph message passing.

## Reward Definition
`R = 0.65*S_U2 + 0.20*S_PEEQ + 0.10*S_SurfaceT + 0.05*S_Mises`, using combined108 within-N normalized/ranked scores.

## GNN Reward Model
- PyTorch available: `True` (torch 2.6.0+cu124, cuda=False)
- Status: `GNN_REWARD_MODEL_TRAINED`
- Leave-N-out macro Spearman: `0.816510227523396`
- Leave-N-out macro top5 overlap: `2.0`
- N40 result: `{'protocol': 'leave_N_out', 'test_n': 40, 'test_count': 30, 'spearman': 0.8477632711708692, 'mae': 0.12509402835848688, 'rmse': 0.14236957391815425, 'top5_overlap': 2}`

## Graph-Pointer Policy
- Status: `GRAPH_POINTER_POLICY_WEIGHTED_IMITATION_TRAINED`
- Training method: `offline weighted behavior cloning on combined108 teacher sequences`
- Visited nodes are masked during decoding.

## Candidate Generation
- Deduplicated candidate counts per N: `{12: 603, 16: 755, 24: 2080, 40: 2839}`
- Candidate sources include graph-pointer greedy, beam search, temperature sampling, GNN reward local search, known-best mutation, disagreement probes, uncertainty probes, and sentinels.

## GNN-Policy Batch64
- Counts: `{12: 8, 16: 8, 24: 24, 40: 24}`
- Bucket composition: `{'gnn_policy_top_predicted': 1, 'gnn_policy_beam_search': 4, 'gnn_policy_temperature_diverse': 6, 'gnn_reward_local_search': 9, 'known_best_neighborhood': 44}`

## GNN-Policy Batch32
- Counts: `{12: 4, 16: 4, 24: 12, 40: 12}`

## Comparison With Run23/Run24 Shortlist64
- Overlap with run24 shortlist64: `0`
- Distinct from run24 shortlist64: `64`
- Mostly distinct: `True`

## Limitations and Claim Boundary
Run26 is offline policy/reward modelling only. It does not prove physical superiority, GNN-RL baseline superiority, arbitrary-N generalization, or deployment readiness.

## Output Files
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_26_combined108_gnn_graph_pointer_policy_candidate_generation\run25_suspended_status_note.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_26_combined108_gnn_graph_pointer_policy_candidate_generation\run26_gnn_input_validation_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_26_combined108_gnn_graph_pointer_policy_candidate_generation\combined108_graph_policy_training_table.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_26_combined108_gnn_graph_pointer_policy_candidate_generation\combined108_graph_policy_split_definitions.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_26_combined108_gnn_graph_pointer_policy_candidate_generation\run26_extra_trees_baseline_metadata.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_26_combined108_gnn_graph_pointer_policy_candidate_generation\gnn_reward_model_validation_results.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_26_combined108_gnn_graph_pointer_policy_candidate_generation\gnn_reward_model_validation_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_26_combined108_gnn_graph_pointer_policy_candidate_generation\graph_pointer_policy_training_log.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_26_combined108_gnn_graph_pointer_policy_candidate_generation\graph_pointer_policy_validation_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_26_combined108_gnn_graph_pointer_policy_candidate_generation\run26_gnn_candidate_pool_unscored.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_26_combined108_gnn_graph_pointer_policy_candidate_generation\run26_gnn_candidate_pool_scored.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_26_combined108_gnn_graph_pointer_policy_candidate_generation\run26_gnn_policy_batch64_candidate_orders.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_26_combined108_gnn_graph_pointer_policy_candidate_generation\run26_gnn_policy_batch32_candidate_orders.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_26_combined108_gnn_graph_pointer_policy_candidate_generation\run26_gnn_vs_run23_shortlist64_comparison.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_26_combined108_gnn_graph_pointer_policy_candidate_generation\run26_gnn_vs_run23_shortlist64_comparison_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_26_combined108_gnn_graph_pointer_policy_candidate_generation\run26_gnn_claim_boundary.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_26_combined108_gnn_graph_pointer_policy_candidate_generation\run26_gnn_claim_boundary.json`

## Recommended Run27
Create a handoff package for the selected GNN-policy batch64. Do not generate CAE until user approval. If the user approves GNN as mainline, use GNN-policy batch64 rather than run23 shortlist64.
