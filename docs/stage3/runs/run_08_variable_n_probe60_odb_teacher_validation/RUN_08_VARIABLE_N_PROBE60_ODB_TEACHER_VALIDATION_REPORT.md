# Run 08 Variable-N Probe60 ODB Teacher Validation Report

## Verdict

`PASS_STAGE3_RUN08_PROBE60_ODB_TEACHER_VALIDATION_60_OF_60`

## Scope

Read-only ODB teacher-metric extraction for the Stage 3 true variable-N Probe60 batch after external Abaqus completion.

## Summary Table

| N | cases | extracted | warnings | failed | best U2 range | best PEEQ max | best SurfaceT proxy |
|---|---:|---:|---:|---:|---|---|---|
| N12 | 15 | 15 | 0 | 0 | `N12_A08_block_interleaved_quarters` (3.68265e-05) | `N12_A13_graph_pointer_policy_anti_odd_even_novelty` (0.140322) | `N12_A09_center_edge_alternating` (5.80753e+08) |
| N16 | 15 | 15 | 0 | 0 | `N16_A03_greedy_maximin_distance` (4.60719e-05) | `N16_A08_block_interleaved_quarters` (0.150555) | `N16_A09_center_edge_alternating` (5.809e+08) |
| N24 | 15 | 15 | 0 | 0 | `N24_A04_method_c_u2_first_engineering` (5.62117e-05) | `N24_A10_graph_pointer_policy_zero_shot_or_proxy_best` (0.154159) | `N24_A13_graph_pointer_policy_anti_odd_even_novelty` (5.81205e+08) |
| N40 | 15 | 15 | 0 | 0 | `N40_A04_method_c_u2_first_engineering` (0.000110819) | `N40_A04_method_c_u2_first_engineering` (0.153951) | `N40_A13_graph_pointer_policy_anti_odd_even_novelty` (5.81542e+08) |

## Metric Contract

- Final frame: last frame of `step_final_cooling`.
- U2: nodal `U` component `U2`; lower `u2_range` is better for within-N warpage ranking.
- PEEQ: final-frame integration-point `PEEQ`; threshold fraction uses `0.002`.
- SurfaceT proxy: final-frame maximum positive principal/component tensile stress over available integration points.
- Stress diagnostic: final-frame `S` Mises maximum.
- Temperature diagnostic: final-frame nodal `NT11` summary.
- No shared full-32 U2 guard is applied; rankings are within each N.

## Failed / Partial Cases

None.

## Outputs

- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_08_probe60_odb_teacher_validation\probe60_odb_teacher_labels.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_08_probe60_odb_teacher_validation\probe60_odb_teacher_labels.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_08_probe60_odb_teacher_validation\probe60_odb_teacher_validation_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\docs\stage3\runs\run_08_variable_n_probe60_odb_teacher_validation\RUN_08_VARIABLE_N_PROBE60_ODB_TEACHER_VALIDATION_REPORT.md`
- `E:\Projects\RL-LAM-ScanOpt\artifacts\manifests\stage3_run_08_manifest.json`

## Guardrails

- ODB files were opened read-only.
- No Abaqus solver job was run.
- No datacheck or job submission was run.
- No solver output files were modified.
- No physical superiority claim is made by this extraction report.
