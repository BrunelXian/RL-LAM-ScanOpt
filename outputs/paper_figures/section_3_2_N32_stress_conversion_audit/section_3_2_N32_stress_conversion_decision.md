# Section 3.2 N32 Stress Conversion Decision

Final verdict: `WARNING_N32_STRESS_CONVERSION_NOT_ALLOWED_LEGACY_SEMANTIC_MISMATCH`

Decision code: `CONVERSION_NOT_ALLOWED_LEGACY_SEMANTIC_MISMATCH`

## Source Files Inspected

- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\FROZEN_stage3_combined552_plus_N32_teacher_dataset.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\stage3_final_N32_legacy_semantic_separation_memo.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\stage3_final_N32_legacy_semantic_separation_memo.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\FROZEN_stage3_combined552_plus_N32_teacher_dataset.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\stage3_final_N32_legacy_semantic_separation_memo.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\stage3_final_N32_legacy_semantic_separation_memo.json`
- `D:\Projects\RL-LAM-ScanOpt\rl-training\v01\outputs\surface_tensile_gradient_full_field_label_backfill_v03\canonical_full_field_surface_gradient_teacher_labels_v03.csv`
- `D:\Projects\RL-LAM-ScanOpt\rl-training\v01\outputs\surface_tensile_gradient_full_field_label_backfill_v03\combined_backfill_results_v03.csv`
- `D:\Projects\RL-LAM-ScanOpt\rl-training\v01\outputs\surface_tensile_gradient_full_field_label_backfill_v03\partial_backfill_results_batch_005_v03.csv`
- `D:\Projects\RL-LAM-ScanOpt\rl-training\v01\outputs\GNN_RL_vs_stage1_full32_baseline_audit_v01\GNN_RL_teacher_metric_table_v01.csv`
- `D:\Projects\RL-LAM-ScanOpt\rl-training\v01\outputs\GNN_RL_vs_stage1_full32_baseline_audit_v01\stage1_full32_baseline10_teacher_metric_table_v01.csv`
- `D:\Projects\RL-LAM-ScanOpt\rl-training\v01\outputs\RLU2_medium40_focused_ODB_teacher_validation_v01\RLU2_medium40_focused_teacher_metrics_v01.csv`
- `D:\Projects\RL-LAM-ScanOpt\rl-training\v01\outputs\RLU2_medium40_focused_ODB_teacher_validation_v01\RLU2_medium40_focused_teacher_labels_v01.csv`
- `D:\Projects\RL-LAM-ScanOpt\rl-training\v01\outputs\UST_probe10_ODB_teacher_validation_v01\UST_probe10_teacher_metrics_v01.csv`
- `D:\Projects\RL-LAM-ScanOpt\LDED_2D_CAE_Framework\cae_models\post-odb\teacher_module_v01\outputs\teacher_labels\full48_teacher_labels_with_orders.json`
- `D:\Projects\RL-LAM-ScanOpt\LDED_2D_CAE_Framework\cae_models\post-odb\teacher_module_v01\outputs\teacher_labels\full48_teacher_labels.json`
- `D:\Projects\RL-LAM-ScanOpt\LDED_2D_CAE_Framework\cae_models\post-odb\teacher_module_v01\outputs\teacher_labels\full20_teacher_labels.json`
- `D:\Projects\RL-LAM-ScanOpt\LDED_2D_CAE_Framework\cae_models\post-odb\method_c_extra20_teacher_eval_v01\outputs\teacher_labels\method_c_extra20_teacher_labels.json`
- `D:\Projects\RL-LAM-ScanOpt\LDED_2D_CAE_Framework\cae_models\post-odb\method_c_extra20_teacher_eval_v01\outputs\combined_full20_extra20\combined_full40_teacher_ranking.json`
- `D:\Projects\RL-LAM-ScanOpt\LDED_2D_CAE_Framework\cae_models\post-odb\method_c_extra20_teacher_eval_v01\outputs\combined_full20_extra20\combined_full40_teacher_ranking.csv`
- `D:\Projects\RL-LAM-ScanOpt\rl-training\v01\outputs\surface_tensile_gradient_full_field_label_backfill_v03\canonical_duplicate_label_audit_v03.csv`
- `D:\Projects\RL-LAM-ScanOpt\rl-training\v01\outputs\surface_tensile_gradient_full_field_label_backfill_v02\canonical_full_field_surface_gradient_teacher_labels_v02.json`
- `D:\Projects\RL-LAM-ScanOpt\rl-training\v01\outputs\surface_tensile_gradient_full_field_label_backfill_v02\canonical_full_field_surface_gradient_teacher_labels_v02.csv`


## Exact N32 Stress Columns Found

In the frozen plus-N32 table:

- `surface_t_proxy`: populated for 332 N32 rows; legacy raw/proxy scale.
- `surface_t_proxy_mpa`: populated for 332 N32 rows, but computed from the legacy proxy and not a validated native MPa stress definition.
- `mises_max`: populated for 332 N32 rows; mapped from legacy `mises_P95_top_band`/proxy semantics per the frozen memo.
- `surface_t_proxy_max_tensile_pa`, `surface_t_proxy_max_tensile_mpa`, `mises_mean`, `max_principal_stress_max`, `max_principal_stress_mean`: not populated for N32 rows in the plus-N32 table.

In Stage 2 source files:

- `surface_tensile_strength`, `surface_tensile_primary`, `teacher_SurfaceT`
- `mises_P95_top_band`, `teacher_Mises`, `masked_xge4_final_Mises_P99_diagnostic`
- S11/S22/S33 diagnostic percentile columns in ODB teacher-validation tables

## Exact Native Stress Columns Found

- Native SurfaceT: `surface_t_proxy`, with `surface_t_proxy_mpa` and native-only extracted stress fields such as `surface_t_proxy_max_tensile_pa` where available.
- Native Mises: `mises_max`, with native-only fields such as `mises_mean` where available.

## Unit And Definition Check

Units are explicit for some native columns (`surface_t_proxy_mpa`, `_pa` fields) and some Stage 2 diagnostic fields appear Pa-scale by name/value. However, the extraction definitions do not match: Stage 2 N32 Mises is P95/top-band/P99 diagnostic/proxy style, while native Stage 3 uses `mises_max`; Stage 2 N32 SurfaceT is `surface_tensile_primary`/`teacher_SurfaceT` proxy style, not proven equivalent to native `surface_t_proxy`.

## Recommendation

Recommended main manuscript figure: `figure_3_4_reward_metric_distributions_native_only_recommended.*`.

Recommended companion/context figure: `figure_3_4_reward_metric_distributions_U2_PEEQ_with_N32_only.*`.

Recommended caption sentence: "Native stress panels use N12/N16/N24/N40 only; auxiliary fixed-N32 stress proxies were excluded because source-unit and extraction-definition compatibility with native Stage 3 SurfaceT/Mises was not established."

## Claim Boundary

N12/N16/N24/N40 are native Stage 3 claim scope. N32 is auxiliary fixed-N32 context only. No global optimum, arbitrary-N generalisation, or physical experiment validation is claimed.
