# Stage 3 Run 32A - Stage 2 N32 Legacy Teacher Label Ingestion for Stage 3

## Purpose
Ingest audited Stage 2 N32 legacy teacher labels into a Stage 3-compatible schema, deduplicate by order hash, compute N32 ranks, and build combined172_plus_N32 datasets without model training or candidate generation.

## Source Audit Summary
- Audit verdict: `WARNING_N32_LEGACY_TEACHER_DATA_PARTIAL`.
- Audit report: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_n32_legacy_teacher_data_audit\N32_LEGACY_TEACHER_DATA_AUDIT_REPORT.md`

## Source Table and Row Counts
- Preview rows: `336`
- Complete legal N32 rows: `336`
- Deduplicated training rows: `332`

## Compatibility Decision
The N32 table is ingested as legacy-compatible with warnings. U2 and SurfaceT are the strongest mappings; PEEQ and Mises are proxy-compatible fields.
- Missing mapped Mises proxy rows: `40`
- Mises fill policy: `conservative_worst_observed_plus_epsilon_for_rank_reward_compatibility_only`

## Duplicate Handling
- Duplicate groups: `2`
- Representatives are selected by earliest source row, with aliases and metric spread recorded.

## Full Provenance N32 Dataset
- All 336 compatible rows are preserved with source file, source row, duplicate metadata, and semantic notes.

## Deduplicated N32 Training Dataset
- One row per unique order hash; duplicate aliases preserved.

## N32 Within-N Ranking and Leaderboard
- Best U2: `RLU2M_A28_V01`
- Best mapped reward: `RL20_A15_V01`

## combined172_plus_N32 Construction
- Total rows: `504`
- Per-N counts: `{12: 32, 16: 32, 24: 54, 32: 332, 40: 54}`

## Metric Semantic Boundary
- `peeq_guard` is mapped to `peeq_max` only as a proxy-compatible legacy field.
- `mises_P95_top_band` is mapped to `mises_max` only as a proxy-compatible legacy field.
- Rows with missing `mises_P95_top_band` are flagged and receive a conservative worst-observed proxy fill for rank/reward compatibility only.

## Per-N Imbalance Warning
N32 has 332 rows, much larger than N12/N16=32 and N24/N40=54. Future training should use per-N balancing or sample weighting.

## Training-Use Recommendations
Use N32 as a legacy-compatible intermediate N group and compare model updates with and without per-N balancing.

## Claim Boundary
`RUN32A_N32_LEGACY_INGESTION_ONLY_WITH_METRIC_SEMANTIC_WARNINGS`.

## Output Files
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_32a_stage2_n32_legacy_teacher_label_ingestion_for_stage3\n32_duplicate_order_hash_audit.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_32a_stage2_n32_legacy_teacher_label_ingestion_for_stage3\n32_legacy_teacher_dataset_full_provenance_336.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_32a_stage2_n32_legacy_teacher_label_ingestion_for_stage3\n32_legacy_teacher_dataset_dedup_training_332.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_32a_stage2_n32_legacy_teacher_label_ingestion_for_stage3\n32_legacy_teacher_dataset_dedup_ranked_332.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_32a_stage2_n32_legacy_teacher_label_ingestion_for_stage3\n32_legacy_per_metric_leaderboard.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_32a_stage2_n32_legacy_teacher_label_ingestion_for_stage3\combined172_plus_N32_teacher_dataset.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_32a_stage2_n32_legacy_teacher_label_ingestion_for_stage3\combined172_plus_N32_RL_ready_dataset.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_32a_stage2_n32_legacy_teacher_label_ingestion_for_stage3\n32_duplicate_order_hash_audit.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_32a_stage2_n32_legacy_teacher_label_ingestion_for_stage3\n32_legacy_teacher_dataset_full_provenance_336.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_32a_stage2_n32_legacy_teacher_label_ingestion_for_stage3\n32_legacy_teacher_dataset_dedup_training_332.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_32a_stage2_n32_legacy_teacher_label_ingestion_for_stage3\run32a_input_validation_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_32a_stage2_n32_legacy_teacher_label_ingestion_for_stage3\n32_legacy_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_32a_stage2_n32_legacy_teacher_label_ingestion_for_stage3\combined172_plus_N32_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_32a_stage2_n32_legacy_teacher_label_ingestion_for_stage3\n32_training_use_notes.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_32a_stage2_n32_legacy_teacher_label_ingestion_for_stage3\run32a_claim_boundary.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_32a_stage2_n32_legacy_teacher_label_ingestion_for_stage3\run32a_claim_boundary.json`

## Recommended Next Action
- If current hybrid batch32 has not been run yet, the user may still manually run it; this N32 ingestion does not conflict.
- After hybrid batch32 teacher validation, build combined204_plus_N32.
- Future model-update run should compare combined172 only, combined172_plus_N32 with per-N balancing, and combined204_plus_N32 after the 32 new jobs are teacher-validated.
