# PPO Surrogate Feature Schema

- Schema: `ppo_surrogate_scan_order_features_v01`
- Max N: `40`
- Feature count: `111`
- Deterministic: `True`

## Groups

### global_n

- `n`
- `n_norm_40`

### position_normalized_sequence

- `first_track_norm`
- `last_track_norm`
- `mean_track_index_norm`
- `std_track_index_norm`
- `early_mean_track_index_norm`
- `late_mean_track_index_norm`
- `early_center_distance_mean`
- `late_center_distance_mean`
- `early_edge_distance_mean`
- `late_edge_distance_mean`

### jump_features

- `mean_abs_jump_norm`
- `max_abs_jump_norm`
- `std_abs_jump_norm`
- `adjacent_jump_fraction`
- `long_jump_fraction`
- `signed_jump_mean_norm`
- `signed_jump_std_norm`
- `direction_reversal_count_norm`
- `monotonicity_fraction`

### parity_and_interleaving

- `parity_switch_fraction`
- `odd_even_transition_count_norm`
- `early_odd_fraction`
- `early_even_fraction`

### coverage_dispersion

- `q1_center_edge_balance`
- `q2_center_edge_balance`
- `q3_center_edge_balance`
- `q4_center_edge_balance`
- `max_unvisited_gap_early_prefix`
- `early_spatial_spread_proxy`

### fixed_40_rank_encoding

- `track_00_rank_position_norm`
- `track_00_valid_flag`
- `track_01_rank_position_norm`
- `track_01_valid_flag`
- `track_02_rank_position_norm`
- `track_02_valid_flag`
- `track_03_rank_position_norm`
- `track_03_valid_flag`
- `track_04_rank_position_norm`
- `track_04_valid_flag`
- `track_05_rank_position_norm`
- `track_05_valid_flag`
- `track_06_rank_position_norm`
- `track_06_valid_flag`
- `track_07_rank_position_norm`
- `track_07_valid_flag`
- `track_08_rank_position_norm`
- `track_08_valid_flag`
- `track_09_rank_position_norm`
- `track_09_valid_flag`
- `track_10_rank_position_norm`
- `track_10_valid_flag`
- `track_11_rank_position_norm`
- `track_11_valid_flag`
- `track_12_rank_position_norm`
- `track_12_valid_flag`
- `track_13_rank_position_norm`
- `track_13_valid_flag`
- `track_14_rank_position_norm`
- `track_14_valid_flag`
- `track_15_rank_position_norm`
- `track_15_valid_flag`
- `track_16_rank_position_norm`
- `track_16_valid_flag`
- `track_17_rank_position_norm`
- `track_17_valid_flag`
- `track_18_rank_position_norm`
- `track_18_valid_flag`
- `track_19_rank_position_norm`
- `track_19_valid_flag`
- `track_20_rank_position_norm`
- `track_20_valid_flag`
- `track_21_rank_position_norm`
- `track_21_valid_flag`
- `track_22_rank_position_norm`
- `track_22_valid_flag`
- `track_23_rank_position_norm`
- `track_23_valid_flag`
- `track_24_rank_position_norm`
- `track_24_valid_flag`
- `track_25_rank_position_norm`
- `track_25_valid_flag`
- `track_26_rank_position_norm`
- `track_26_valid_flag`
- `track_27_rank_position_norm`
- `track_27_valid_flag`
- `track_28_rank_position_norm`
- `track_28_valid_flag`
- `track_29_rank_position_norm`
- `track_29_valid_flag`
- `track_30_rank_position_norm`
- `track_30_valid_flag`
- `track_31_rank_position_norm`
- `track_31_valid_flag`
- `track_32_rank_position_norm`
- `track_32_valid_flag`
- `track_33_rank_position_norm`
- `track_33_valid_flag`
- `track_34_rank_position_norm`
- `track_34_valid_flag`
- `track_35_rank_position_norm`
- `track_35_valid_flag`
- `track_36_rank_position_norm`
- `track_36_valid_flag`
- `track_37_rank_position_norm`
- `track_37_valid_flag`
- `track_38_rank_position_norm`
- `track_38_valid_flag`
- `track_39_rank_position_norm`
- `track_39_valid_flag`

