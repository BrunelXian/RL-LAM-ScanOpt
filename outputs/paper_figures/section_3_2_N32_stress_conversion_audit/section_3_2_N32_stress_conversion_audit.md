# Section 3.2 N32 Stress Conversion Audit

Final verdict: `WARNING_N32_STRESS_CONVERSION_NOT_ALLOWED_LEGACY_SEMANTIC_MISMATCH`

Decision code: `CONVERSION_NOT_ALLOWED_LEGACY_SEMANTIC_MISMATCH`

## Why Conversion Is Not Allowed

A same-unit N32 stress conversion is not allowed because the available N32 stress-related columns fail the required two-condition rule:

1. The selected frozen plus-N32 table stores N32 `surface_t_proxy`/`surface_t_proxy_mpa` as legacy raw/proxy values and N32 `mises_max` as a legacy mapped value. The source rows are marked `stage2_n32_legacy`, `metric_semantic_warning=True`, and `LEGACY_COMPATIBLE_WITH_WARNINGS`.
2. The Stage 2 source evidence maps N32 Mises through `mises_P95_top_band` or diagnostic P99/top-band variants, not the native Stage 3 `mises_max` extraction definition. Surface metrics are represented as `surface_tensile_primary`, `teacher_SurfaceT`, or related proxy/diagnostic quantities, not proven equivalent to native Stage 3 `surface_t_proxy`.

No fitted or magnitude-only scale factor was applied.

## Plus-N32 Stress Schema Audit

| column | native non-null | N32 non-null | native min / median / max | N32 min / median / max | inferred unit/scale | compatibility class |
|---|---:|---:|---|---|---|---|
| `surface_t_proxy` | 552 | 332 | 580752832 / 582285312 / 598535360 | 0.0010144115886554 / 0.0070103019324473503 / 0.0542577414436596 | native:Pa; N32:legacy_raw_proxy | mixed_native_compatible_and_n32_legacy_proxy |
| `surface_t_proxy_mpa` | 552 | 332 | 580.75283200000001 / 582.28531199999998 / 598.53535999999997 | 1.0144115886554e-09 / 7.0103019324473505e-09 / 5.4257741443659601e-08 | MPa_if_physical_column | mixed_native_compatible_and_n32_legacy_proxy |
| `mises_max` | 552 | 332 | 579924032 / 579939712 / 579948672 | 1.9332757333333328 / 1.9332789546666664 / 1.933567361933568 | native:Pa; N32:legacy_raw_proxy | mixed_native_compatible_and_n32_legacy_proxy |
| `mises_mean` | 348 | 0 | 579920643.66666663 / 579938629.85714281 / 579947526.37037039 |  /  /  | source_numeric_scale | native_only_or_absent_for_n32 |
| `max_principal_stress_max` | 348 | 0 | 580840256 / 582140864 / 592035328 |  /  /  | source_numeric_scale | native_only_or_absent_for_n32 |
| `max_principal_stress_mean` | 348 | 0 | 579916072.09523809 / 579938119.92857146 / 579947057.37037039 |  /  /  | source_numeric_scale | native_only_or_absent_for_n32 |
| `surface_t_proxy_max_tensile_pa` | 348 | 0 | 580840256 / 582140864 / 592035328 |  /  /  | Pa_if_physical_column | native_only_or_absent_for_n32 |
| `surface_t_proxy_max_tensile_mpa` | 348 | 0 | 580.84025599999995 / 582.14086399999997 / 592.03532800000005 |  /  /  | MPa_if_physical_column | native_only_or_absent_for_n32 |
| `mises_metric_semantics` | 0 | 332 |  /  /  |  /  /  | metadata_or_derived | semantic_metadata |

Full schema audit is in `section_3_2_N32_stress_conversion_audit.csv`.

## Stage 2 Source Search Summary

| source file | row count | N32/full32 evidence | SurfaceT semantics | Mises semantics | usable for native scale conversion |
|---|---:|---|---|---|---|
| `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\FROZEN_stage3_combined552_plus_N32_teacher_dataset.csv` | 884 | YES_PATH | explicit_pa_or_mpa_column_present | mises_max_column_present_but_definition_must_be_checked | UNCERTAIN |
| `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\stage3_final_N32_legacy_semantic_separation_memo.md` | NA | YES_TEXT | not_detected | mises_max_column_present_but_definition_must_be_checked | NO |
| `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\stage3_final_N32_legacy_semantic_separation_memo.json` | NA | YES_TEXT | not_detected | mises_max_column_present_but_definition_must_be_checked | NO |
| `D:\Projects\RL-LAM-ScanOpt\rl-training\v01\outputs\surface_tensile_gradient_full_field_label_backfill_v03\canonical_full_field_surface_gradient_teacher_labels_v03.csv` | 387 | YES_CONTENT | legacy_surface_tensile_primary_or_teacher_SurfaceT_proxy | mises_P95_top_band_proxy | NO |
| `D:\Projects\RL-LAM-ScanOpt\rl-training\v01\outputs\surface_tensile_gradient_full_field_label_backfill_v03\combined_backfill_results_v03.csv` | 347 | YES_CONTENT | legacy_surface_tensile_primary_or_teacher_SurfaceT_proxy | mises_P95_top_band_proxy | NO |
| `D:\Projects\RL-LAM-ScanOpt\rl-training\v01\outputs\surface_tensile_gradient_full_field_label_backfill_v03\partial_backfill_results_batch_005_v03.csv` | 25 | YES_CONTENT | legacy_surface_tensile_primary_or_teacher_SurfaceT_proxy | mises_P95_top_band_proxy | NO |
| `D:\Projects\RL-LAM-ScanOpt\rl-training\v01\outputs\GNN_RL_vs_stage1_full32_baseline_audit_v01\GNN_RL_teacher_metric_table_v01.csv` | 234 | YES_PATH | legacy_surface_tensile_primary_or_teacher_SurfaceT_proxy | teacher_Mises_legacy_alias | NO |
| `D:\Projects\RL-LAM-ScanOpt\rl-training\v01\outputs\GNN_RL_vs_stage1_full32_baseline_audit_v01\stage1_full32_baseline10_teacher_metric_table_v01.csv` | 10 | YES_PATH | legacy_surface_tensile_primary_or_teacher_SurfaceT_proxy | teacher_Mises_legacy_alias | NO |
| `D:\Projects\RL-LAM-ScanOpt\rl-training\v01\outputs\RLU2_medium40_focused_ODB_teacher_validation_v01\RLU2_medium40_focused_teacher_metrics_v01.csv` | 40 | YES_CONTENT | stress_related_diagnostic_columns | mises_P99_diagnostic_proxy | NO |
| `D:\Projects\RL-LAM-ScanOpt\rl-training\v01\outputs\RLU2_medium40_focused_ODB_teacher_validation_v01\RLU2_medium40_focused_teacher_labels_v01.csv` | 40 | YES_CONTENT | stress_related_diagnostic_columns | mises_P99_diagnostic_proxy | NO |
| `D:\Projects\RL-LAM-ScanOpt\rl-training\v01\outputs\UST_probe10_ODB_teacher_validation_v01\UST_probe10_teacher_metrics_v01.csv` | 10 | YES_CONTENT | legacy_surface_tensile_primary_or_teacher_SurfaceT_proxy | mises_P95_top_band_proxy | NO |
| `D:\Projects\RL-LAM-ScanOpt\LDED_2D_CAE_Framework\cae_models\post-odb\teacher_module_v01\outputs\teacher_labels\full48_teacher_labels_with_orders.json` | NA | YES_TEXT | stress_related_diagnostic_columns | mises_max_column_present_but_definition_must_be_checked | UNCERTAIN |
| `D:\Projects\RL-LAM-ScanOpt\LDED_2D_CAE_Framework\cae_models\post-odb\teacher_module_v01\outputs\teacher_labels\full48_teacher_labels.json` | NA | YES_TEXT | stress_related_diagnostic_columns | mises_max_column_present_but_definition_must_be_checked | UNCERTAIN |
| `D:\Projects\RL-LAM-ScanOpt\LDED_2D_CAE_Framework\cae_models\post-odb\teacher_module_v01\outputs\teacher_labels\full20_teacher_labels.json` | NA | YES_TEXT | stress_related_diagnostic_columns | mises_max_column_present_but_definition_must_be_checked | NO |
| `D:\Projects\RL-LAM-ScanOpt\LDED_2D_CAE_Framework\cae_models\post-odb\method_c_extra20_teacher_eval_v01\outputs\teacher_labels\method_c_extra20_teacher_labels.json` | NA | YES_TEXT | stress_related_diagnostic_columns | mises_max_column_present_but_definition_must_be_checked | UNCERTAIN |
| `D:\Projects\RL-LAM-ScanOpt\LDED_2D_CAE_Framework\cae_models\post-odb\method_c_extra20_teacher_eval_v01\outputs\combined_full20_extra20\combined_full40_teacher_ranking.json` | NA | YES_TEXT | stress_related_diagnostic_columns | mises_max_column_present_but_definition_must_be_checked | NO |
| `D:\Projects\RL-LAM-ScanOpt\LDED_2D_CAE_Framework\cae_models\post-odb\method_c_extra20_teacher_eval_v01\outputs\combined_full20_extra20\combined_full40_teacher_ranking.csv` | 40 | YES_CONTENT | stress_related_diagnostic_columns | mises_max_column_present_but_definition_must_be_checked | UNCERTAIN |
| `D:\Projects\RL-LAM-ScanOpt\rl-training\v01\outputs\surface_tensile_gradient_full_field_label_backfill_v03\canonical_duplicate_label_audit_v03.csv` | 80 | YES_CONTENT | legacy_surface_tensile_primary_or_teacher_SurfaceT_proxy | mises_P95_top_band_proxy | NO |
| `D:\Projects\RL-LAM-ScanOpt\rl-training\v01\outputs\surface_tensile_gradient_full_field_label_backfill_v02\canonical_full_field_surface_gradient_teacher_labels_v02.json` | NA | YES_TEXT | legacy_surface_tensile_primary_or_teacher_SurfaceT_proxy | mises_P95_top_band_proxy | NO |
| `D:\Projects\RL-LAM-ScanOpt\rl-training\v01\outputs\surface_tensile_gradient_full_field_label_backfill_v02\canonical_full_field_surface_gradient_teacher_labels_v02.csv` | 40 | YES_CONTENT | legacy_surface_tensile_primary_or_teacher_SurfaceT_proxy | mises_P95_top_band_proxy | NO |

The strongest source evidence is `canonical_full_field_surface_gradient_teacher_labels_v03.csv` and `combined_backfill_results_v03.csv`, which contain `surface_tensile_primary`, `gradient_S11`, `gradient_S22`, and `mises_P95_top_band`. For example, `RLU2M_A28_V01` has `surface_tensile_primary=0.002383854107380318` and `mises_P95_top_band=1.9332782933333332`. These are useful legacy/proxy labels, but they are not the native Stage 3 SurfaceT/Mises definitions.

## Generated Artifacts

- `section_3_2_N32_stress_conversion_audit.csv`
- `section_3_2_N32_stress_conversion_audit.md`
- `section_3_2_N32_stress_conversion_decision.md`
- `figure_3_4_reward_metric_distributions_native_only_recommended.png/.pdf/.svg`
- `figure_3_4_reward_metric_distributions_U2_PEEQ_with_N32_only.png/.pdf/.svg`

## Recommended Main Manuscript Figure

Use `figure_3_4_reward_metric_distributions_native_only_recommended.*` as the main same-unit reward-response figure for Section 3.2. If N32 context is needed, use `figure_3_4_reward_metric_distributions_U2_PEEQ_with_N32_only.*` for U2/PEEQ only.

Recommended caption sentence: "Native stress panels use N12/N16/N24/N40 only; auxiliary fixed-N32 stress proxies were excluded because source-unit and extraction-definition compatibility with native Stage 3 SurfaceT/Mises was not established."

## Claim Boundary

N12/N16/N24/N40 remain the native Stage 3 multi-N claim scope. N32 remains auxiliary fixed-N32 / Stage 2 legacy-compatible context. This audit does not support global optimum claims, arbitrary-N generalisation, or physical experiment validation.
