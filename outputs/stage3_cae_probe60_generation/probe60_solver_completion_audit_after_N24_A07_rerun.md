# Stage 3 Probe60 Solver Completion Audit After N24_A07 Rerun

## Verdict

`WARNING_PROBE60_COMPLETION_WITH_NONFATAL_WARNINGS`

## Summary Table

| N | expected | solver_success | odb_present | lck_present | failed_or_incomplete |
|---|---:|---:|---:|---:|---:|
| N12 | 15 | 15 | 15 | 0 | 0 |
| N16 | 15 | 15 | 15 | 0 | 0 |
| N24 | 15 | 15 | 15 | 0 | 0 |
| N40 | 15 | 15 | 15 | 0 | 0 |

## Special Case Check: N24_A07_regular_jump_coprime

- STA success marker: true
- ODB exists and size: true, 191163772 bytes
- LCK absence: true
- fatal markers: sta=false, dat=false, msg=false, log=false
- final status: `WARNING_SUCCESS_WITH_WARNINGS`
- notes: sta_success_and_odb_present_with_nonfatal_warnings

## Failed / Incomplete Cases

None.

## Warning Cases

- `N12_A01_raster_left_to_right` (N12): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N12_A02_odd_even_interlaced` (N12): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N12_A03_greedy_maximin_distance` (N12): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N12_A04_method_c_u2_first_engineering` (N12): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N12_A05_center_out` (N12): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N12_A06_edge_in_alternating` (N12): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N12_A07_regular_jump_coprime` (N12): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N12_A08_block_interleaved_quarters` (N12): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N12_A09_center_edge_alternating` (N12): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N12_A10_graph_pointer_policy_zero_shot_or_proxy_best` (N12): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N12_A11_graph_pointer_policy_diverse_01` (N12): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N12_A12_graph_pointer_policy_diverse_02` (N12): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N12_A13_graph_pointer_policy_anti_odd_even_novelty` (N12): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N12_A14_graph_pointer_policy_u2first_proxy` (N12): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N12_A15_graph_pointer_policy_balanced_dispersion_proxy` (N12): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N16_A01_raster_left_to_right` (N16): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N16_A02_odd_even_interlaced` (N16): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N16_A03_greedy_maximin_distance` (N16): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N16_A04_method_c_u2_first_engineering` (N16): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N16_A05_center_out` (N16): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N16_A06_edge_in_alternating` (N16): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N16_A07_regular_jump_coprime` (N16): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N16_A08_block_interleaved_quarters` (N16): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N16_A09_center_edge_alternating` (N16): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N16_A10_graph_pointer_policy_zero_shot_or_proxy_best` (N16): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N16_A11_graph_pointer_policy_diverse_01` (N16): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N16_A12_graph_pointer_policy_diverse_02` (N16): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N16_A13_graph_pointer_policy_anti_odd_even_novelty` (N16): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N16_A14_graph_pointer_policy_u2first_proxy` (N16): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N16_A15_graph_pointer_policy_balanced_dispersion_proxy` (N16): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N24_A01_raster_left_to_right` (N24): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N24_A02_odd_even_interlaced` (N24): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N24_A03_greedy_maximin_distance` (N24): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N24_A04_method_c_u2_first_engineering` (N24): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N24_A05_center_out` (N24): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N24_A06_edge_in_alternating` (N24): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N24_A07_regular_jump_coprime` (N24): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N24_A08_block_interleaved_quarters` (N24): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N24_A09_center_edge_alternating` (N24): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N24_A10_graph_pointer_policy_zero_shot_or_proxy_best` (N24): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N24_A11_graph_pointer_policy_diverse_01` (N24): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N24_A12_graph_pointer_policy_diverse_02` (N24): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N24_A13_graph_pointer_policy_anti_odd_even_novelty` (N24): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N24_A14_graph_pointer_policy_u2first_proxy` (N24): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N24_A15_graph_pointer_policy_balanced_dispersion_proxy` (N24): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N40_A01_raster_left_to_right` (N40): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N40_A02_odd_even_interlaced` (N40): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N40_A03_greedy_maximin_distance` (N40): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N40_A04_method_c_u2_first_engineering` (N40): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N40_A05_center_out` (N40): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N40_A06_edge_in_alternating` (N40): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N40_A07_regular_jump_coprime` (N40): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N40_A08_block_interleaved_quarters` (N40): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N40_A09_center_edge_alternating` (N40): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N40_A10_graph_pointer_policy_zero_shot_or_proxy_best` (N40): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N40_A11_graph_pointer_policy_diverse_01` (N40): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N40_A12_graph_pointer_policy_diverse_02` (N40): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N40_A13_graph_pointer_policy_anti_odd_even_novelty` (N40): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N40_A14_graph_pointer_policy_u2first_proxy` (N40): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings
- `N40_A15_graph_pointer_policy_balanced_dispersion_proxy` (N40): `WARNING_SUCCESS_WITH_WARNINGS`; sta_success_and_odb_present_with_nonfatal_warnings

## Next-Step Gate

`Do not start ODB postprocessing. Resolve failed/incomplete cases first.`

## Audit Guardrails

- No Abaqus command was run.
- No solver, datacheck, or queue command was run.
- No ODB was opened; only path existence, size, and timestamp metadata were checked.
- No solver output files were modified, moved, renamed, archived, or deleted.
