# Stage 3 Run 09 — Variable-N Probe60 Teacher Ranking Analysis

## Purpose
Analyze the completed run_08 teacher labels using within-N rankings for U2, PEEQ, and SurfaceT proxy.

## Inputs
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_08_probe60_odb_teacher_validation\probe60_odb_teacher_labels.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_08_probe60_odb_teacher_validation\probe60_odb_teacher_validation_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_06_variable_n_probe60_candidate_order_generation\variable_N_probe60_candidate_orders.csv`

## Validation Status
- `PASS_RUN09_INPUT_TEACHER_LABELS_60_OF_60_READY`
- Total rows: `60`
- Per-N counts: `{12: 15, 16: 15, 24: 15, 40: 15}`

## Objective Hierarchy
U2 / warpage is primary. PEEQ is safety/plasticity. SurfaceT proxy is a secondary residual-stress diagnostic. Lower values are treated as better for all three metrics in this run.

## Per-N Ranking Results
- Best U2 per N: `{'N12': 'N12_A08_block_interleaved_quarters', 'N16': 'N16_A03_greedy_maximin_distance', 'N24': 'N24_A04_method_c_u2_first_engineering', 'N40': 'N40_A04_method_c_u2_first_engineering'}`
- Best PEEQ per N: `{'N12': 'N12_A13_graph_pointer_policy_anti_odd_even_novelty', 'N16': 'N16_A08_block_interleaved_quarters', 'N24': 'N24_A10_graph_pointer_policy_zero_shot_or_proxy_best', 'N40': 'N40_A04_method_c_u2_first_engineering'}`
- Best SurfaceT proxy per N: `{'N12': 'N12_A09_center_edge_alternating', 'N16': 'N16_A09_center_edge_alternating', 'N24': 'N24_A13_graph_pointer_policy_anti_odd_even_novelty', 'N40': 'N40_A13_graph_pointer_policy_anti_odd_even_novelty'}`

## Best Candidates Per N
- Best constrained-rank per N: `{'N12': 'N12_A08_block_interleaved_quarters', 'N16': 'N16_A03_greedy_maximin_distance', 'N24': 'N24_A04_method_c_u2_first_engineering', 'N40': 'N40_A04_method_c_u2_first_engineering'}`

## Strategy Family Analysis
- Best mean simple-rank family: `method_c` with mean rank `2.667`

## Candidate Group Comparison
proxy_fallback_policy: mean U2 rank 7.33; engineering_baseline: mean U2 rank 8.44

## Metric Interaction and Tradeoff Analysis
- Overall Spearman U2/PEEQ: `0.614`
- Overall Spearman U2/SurfaceT: `0.514`
- Overall top-5 U2/PEEQ overlap: `16`

## Pareto Front Summary
- Pareto rows written: `27`

## N24_A07 Resolution Note
N24_A07_regular_jump_coprime was previously incomplete during solver completion audit but was rerun/reprocessed successfully and is included as a valid teacher-labelled case in run08/run09.

## Safe Claims
- True variable-N teacher validation completed for N=12/16/24/40 with 60/60 ODB-extracted labels.
- Within-N ranking is now possible for U2, PEEQ, and SurfaceT proxy.
- Cross-N diagnostic comparison is now possible using ranks, percentiles, and normalized scores.
- N-specific ranking is required because raw objective magnitudes are N-dependent.
- The run provides a teacher-labelled variable-N benchmark dataset for later variable-N policy training/evaluation.

## Claim Boundaries
- Do not claim trained variable-N RL policy superiority.
- Do not claim arbitrary-N generalization.
- Do not claim a physical optimum.
- Do not claim fixed-32 U2 guard transfer to variable-N.
- Do not claim SurfaceT optimization outside U2/PEEQ feasible or near-feasible regions unless supported.
- Do not claim proxy/fallback policy is equivalent to trained RL.

## Outputs
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_09_variable_n_probe60_teacher_ranking_analysis\run09_input_validation_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_09_variable_n_probe60_teacher_ranking_analysis\probe60_teacher_ranked_canonical.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_09_variable_n_probe60_teacher_ranking_analysis\probe60_teacher_ranked_canonical.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_09_variable_n_probe60_teacher_ranking_analysis\per_N_top_bottom_leaderboards.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_09_variable_n_probe60_teacher_ranking_analysis\per_N_top_bottom_leaderboards.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_09_variable_n_probe60_teacher_ranking_analysis\per_N_leaderboards_summary.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_09_variable_n_probe60_teacher_ranking_analysis\strategy_family_summary.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_09_variable_n_probe60_teacher_ranking_analysis\strategy_family_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_09_variable_n_probe60_teacher_ranking_analysis\candidate_group_comparison.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_09_variable_n_probe60_teacher_ranking_analysis\candidate_group_comparison.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_09_variable_n_probe60_teacher_ranking_analysis\metric_interaction_summary.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_09_variable_n_probe60_teacher_ranking_analysis\metric_interaction_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_09_variable_n_probe60_teacher_ranking_analysis\pareto_front_cases.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_09_variable_n_probe60_teacher_ranking_analysis\run09_claim_boundary.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_09_variable_n_probe60_teacher_ranking_analysis\run09_claim_boundary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_09_variable_n_probe60_teacher_ranking_analysis\figures\per_N_u2_vs_peeq.png`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_09_variable_n_probe60_teacher_ranking_analysis\figures\per_N_u2_vs_surfaceT.png`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_09_variable_n_probe60_teacher_ranking_analysis\figures\N12_top5_u2.png`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_09_variable_n_probe60_teacher_ranking_analysis\figures\N16_top5_u2.png`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_09_variable_n_probe60_teacher_ranking_analysis\figures\N24_top5_u2.png`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_09_variable_n_probe60_teacher_ranking_analysis\figures\N40_top5_u2.png`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_09_variable_n_probe60_teacher_ranking_analysis\figures\family_mean_rank_comparison.png`

Plotting status: `plots written`

## Recommended Next Step
Stage 3 run_10 should use the 60 teacher-labelled variable-N cases to build the first variable-N surrogate / normalized reward dataset, using within-N normalized ranks rather than raw cross-N objective magnitudes.
