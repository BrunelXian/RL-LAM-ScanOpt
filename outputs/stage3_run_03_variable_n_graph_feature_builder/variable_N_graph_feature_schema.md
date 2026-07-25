# Variable-N Graph Feature Schema

Run 03 defines a variable-N graph representation for `N = 16, 24, 32, 40`.

Each track is a graph node. Directed edges encode adjacent, k-nearest, and thermal-radius spatial relations. Global context stores normalized scan-progress and balance descriptors. Masks expose future pointer-decoder legality, but no policy is trained in this run.

## Node Features

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

`track_index` is retained only as metadata and is not a model feature.

## Edge Features

- `distance_ij_norm`
- `relative_position_ij_norm`
- `is_adjacent`
- `is_within_thermal_radius`
- `left_or_right_relation`
- `edge_type_code`

`source_index`, `target_index`, and string `edge_type` are metadata. The numeric edge feature for edge type is `edge_type_code`.

## Global Features

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

## Signed Normalized Features

- `edge_center_heat_balance_norm`
- `left_or_right_relation`
- `left_right_heat_balance_norm`
- `relative_position_ij_norm`

All other numeric model features are expected in `[0, 1]`. Signed relation and balance features are expected in `[-1, 1]`.

## Local Heat Proxy

`local_heat_proxy_norm` is a geometry/time-decay proxy computed from prior scanned tracks and normalized within each graph state. It is not a teacher metric and must not be interpreted as Abaqus evidence.
