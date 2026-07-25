# N32 Legacy Teacher Data Audit

Generated: 2026-06-13T13:41:44

## Purpose

Audit whether existing Stage 2 N32 / 32-track teacher-labelled CSV/JSON data can be reused as an N=32 group in the Stage 3 combined dataset.

This audit did not run Abaqus, did not open ODB files, did not train models, and did not generate candidates or CAE/INP/JNL files.

## Search Roots

- `D:\Projects\RL-LAM-ScanOpt`
- `E:\Projects\RL-LAM-ScanOpt`

## Candidate Files Found

- Candidate text/table files inventoried: `4247`
- CSV/JSON/Parquet schemas inspected: `3039`
- Tables with usable Stage 3 N32 mapping signal: `52`

## Best Teacher-Label Table Candidates

- `D:\Projects\RL-LAM-ScanOpt\rl-training\v01\outputs\surface_tensile_gradient_full_field_label_backfill_v03\canonical_full_field_surface_gradient_teacher_labels_v03.csv` rows=387 score=957
- `D:\Projects\RL-LAM-ScanOpt\rl-training\v01\outputs\surface_tensile_gradient_full_field_label_backfill_v03\canonical_full_field_surface_gradient_teacher_labels_v03.json` rows=387 score=957
- `D:\Projects\RL-LAM-ScanOpt\rl-training\v01\outputs\GNN_RL_vs_stage1_full32_baseline_audit_v01\GNN_RL_teacher_metric_table_v01.csv` rows=234 score=764
- `D:\Projects\RL-LAM-ScanOpt\rl-training\v01\outputs\surface_gradient_single_objective_surrogate_training_v01\gradient_surrogate_training_rows_v01.csv` rows=336 score=866
- `D:\Projects\RL-LAM-ScanOpt\rl-training\v01\outputs\surface_gradient_single_objective_surrogate_training_v01\surface_gradient_feature_table_v01.csv` rows=336 score=866
- `D:\Projects\RL-LAM-ScanOpt\rl-training\v01\outputs\surface_gradient_single_objective_surrogate_training_v01\surface_surrogate_training_rows_v01.csv` rows=336 score=866
- `D:\Projects\RL-LAM-ScanOpt\rl-training\v01\outputs\transformer_sequence_surrogate_ablation_v01\transformer_training_rows_v01.csv` rows=336 score=866
- `D:\Projects\RL-LAM-ScanOpt\rl-training\v01\outputs\surface_tensile_gradient_full_field_label_backfill_v03\combined_backfill_results_v03.csv` rows=347 score=917

## Selected Best Source

- Source file: `D:\Projects\RL-LAM-ScanOpt\rl-training\v01\outputs\surface_tensile_gradient_full_field_label_backfill_v03\canonical_full_field_surface_gradient_teacher_labels_v03.csv`
- Rows in source: `387`
- N32/legal order rows detected: `336`
- Strategy column: `strategy_id`
- Scan-order column: `scan_order`
- U2 column: `U2_range`
- PEEQ column: `peeq_guard`
- SurfaceT column: `surface_tensile_primary`
- Mises column: `mises_P95_top_band`

## Ingestion Preview

- Preview rows with legal N32 scan orders: `336`
- Rows skipped due missing/invalid N32 scan order: `51`
- Duplicate order-hash rows in preview: `4`

## Compatibility With Stage 3 combined172

- `n` <- `constant 32`: OK (N=32 inferred from legal full 0..31 scan_order and Stage 2 32track source)
- `strategy_name` <- `strategy_id`: OK
- `order_json` <- `scan_order`: OK
- `u2_range` <- `U2_range`: OK
- `peeq_max` <- `peeq_guard`: OK (May be peeq_guard if canonical table is used)
- `surface_t_proxy` <- `surface_tensile_primary`: OK
- `mises_max` <- `mises_P95_top_band`: OK (May be P95/top-band Mises diagnostic rather than global max)
- `dataset_source` <- `constant stage2_n32_gnn_rl_legacy/stage2_n32_full32_legacy`: OK
- `teacher_validation_status` <- `final_cooling_success`: OK
- `row_count` <- `336`: OK (Requires at least 100 reusable N32 rows for PASS)
- `duplicate_order_hash_status` <- `4`: NOT READY (Duplicates should be removed or documented before final Stage 3 merge)

## Teacher-Validation Evidence

The selected source is treated as teacher-labelled only when it contains explicit extraction/final-cooling/status columns or comes from canonical full-field ODB extraction outputs. The audit records inferred teacher status in the preview, but final ingestion should keep the source file and row index for traceability.

## Whether N32 Can Be Introduced As An Additional N Group

If the preview row count is sufficient and the mapped metrics are accepted, the discovered Stage 2 N32 table can be introduced as an additional N=32 group. The preview table is intentionally not merged into the Stage 3 combined dataset in this run.

## Risks And Missing Evidence

- Some Stage 2 fields are semantically close but not name-identical to Stage 3 fields, especially `peeq_guard` versus `peeq_max` and `mises_P95_top_band` versus `mises_max`.
- Final cooling / extracted-field metadata may be inferred from source provenance when explicit columns are absent.
- If a strict Stage 3 merge requires literal global `mises_max` or literal `peeq_max`, an additional mapping decision is needed.
- Duplicates, if any, must be removed or documented before the final combined dataset merge.

## Verdict

`WARNING_N32_LEGACY_TEACHER_DATA_PARTIAL`

## Recommended Next Action

Repair missing mappings or locate a stronger source table before merging N32 into Stage 3.
