# Run 03 Variable-N Graph Feature Builder Report

## Executive Verdict

PASS_VARIABLE_N_GRAPH_FEATURE_BUILDER_READY

## What Was Built

- N values covered: `[16, 24, 32, 40]`
- Graph states generated: `20`
- Node feature count: `14`
- Edge feature count: `6`
- Global feature count: `10`

## Guardrails

- No Abaqus jobs.
- No datacheck.
- No ODB opened.
- No CAE/INP/JNL generated.
- No model training.
- No RL candidate generation.
- No teacher validation.
- D-drive source was not modified.

## Scientific Positioning

Run 03 creates the variable-N representation foundation for later graph pointer policy work. It is not yet an RL result and does not prove variable-N generalisation.

## Feature Schema

### Node Features

- `x_norm`
- `distance_to_left_edge_norm`
- `distance_to_right_edge_norm`
- `is_edge_track`
- `is_center_track`
- `current_scanned_flag`
- `current_available_flag`
- `step_fraction`
- `remaining_fraction`
- `time_since_left_neighbor_scanned_norm`
- `time_since_right_neighbor_scanned_norm`
- `neighbor_scanned_count_norm`
- `recent_neighbor_scanned_flag`
- `local_heat_proxy_norm`

### Edge Features

- `distance_ij_norm`
- `relative_position_ij_norm`
- `is_adjacent`
- `is_within_thermal_radius`
- `left_or_right_relation`
- `edge_type_code`

### Global Features

- `log_N_norm`
- `step_fraction`
- `remaining_fraction`
- `current_center_of_mass_scanned_norm`
- `left_right_heat_balance_norm`
- `edge_center_heat_balance_norm`
- `mean_jump_so_far_norm`
- `max_jump_so_far_norm`
- `last_jump_length_norm`
- `dispersion_score_norm`

### Masks

- `scanned_mask`
- `available_mask`
- `pointer_legal_mask`

## Normalisation Audit

All generated graph states passed finite-value, normalized-bounds, mask-legality, and duplicate-edge checks.

No raw track ID, fixed 32-dimensional representation, fixed track ID embedding, absolute step index, or raw unnormalized jump length is used as a model feature. `track_index`, `source_index`, and `target_index` appear only as metadata.

## Local Heat Proxy Note

`local_heat_proxy_norm` is a geometry/time-decay proxy normalized within each graph state. It is not a teacher metric and is not Abaqus validation evidence.

## Claim Boundary

Allowed: variable-N graph feature representation is implemented and validated on N=16/24/32/40 sample states.

Not allowed: RL generalises to variable N; GNN/RL solves variable-N optimisation; the same full-32 U2 guard transfers to all N; teacher-validated variable-N improvement exists.

## Outputs

- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_03_variable_n_graph_feature_builder\variable_N_graph_feature_schema.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_03_variable_n_graph_feature_builder\sample_graphs_N16_N24_N32_N40.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_03_variable_n_graph_feature_builder\feature_normalisation_audit.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_03_variable_n_graph_feature_builder\node_feature_summary.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_03_variable_n_graph_feature_builder\edge_feature_summary.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_03_variable_n_graph_feature_builder\global_feature_summary.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_03_variable_n_graph_feature_builder\mask_legality_audit.csv`
- `E:\Projects\RL-LAM-ScanOpt\artifacts\manifests\stage3_run_03_manifest.json`

## Recommended Next Run

`run_04_variable_n_baseline_generator`

## Final Verdict

PASS_VARIABLE_N_GRAPH_FEATURE_BUILDER_READY
