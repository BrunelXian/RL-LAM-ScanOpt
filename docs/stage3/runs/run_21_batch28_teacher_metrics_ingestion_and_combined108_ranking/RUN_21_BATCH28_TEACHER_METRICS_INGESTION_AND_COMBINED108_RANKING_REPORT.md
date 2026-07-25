# Stage 3 Run 21 - Batch28 Teacher Metrics Ingestion and Combined108 Ranking

## Purpose
Ingest official batch28 teacher metrics, audit surrogate predictions, compare against combined80, and build a combined108 RL-ready teacher-labelled dataset.

## Inputs
- Batch28 metrics: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_20_batch28_odb_teacher_validation\run20_batch28_teacher_metrics.csv`
- Run19 batch28 handoff: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_19_run18_candidate_handoff_review_package\batch28\stage3_run19_batch28_candidate_orders.csv`
- Previous combined80 dataset: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_16_batch20_teacher_metrics_ingestion_and_combined80_ranking\combined80_teacher_dataset.csv`

## Batch28 Validation Status
- Verdict: `PASS_RUN21_BATCH28_TEACHER_METRICS_28_OF_28_READY`
- Per-N counts: `{12: 4, 16: 4, 24: 10, 40: 10}`
- Solver completion verdict: `WARNING_RUN20_BATCH28_SOLVER_COMPLETION_WITH_NONFATAL_WARNINGS`
- Extraction verdict: `PASS_RUN20_BATCH28_ODB_TEACHER_VALIDATION_28_OF_28`

## Batch28 Within-N Ranking
- N12: best U2 `S3R19B28_N12_B04_method_c_inspired`, best reward `S3R19B28_N12_B04_method_c_inspired`.
- N16: best U2 `S3R19B28_N16_B04_method_c_inspired`, best reward `S3R19B28_N16_B04_method_c_inspired`.
- N24: best U2 `S3R19B28_N24_B10_known_best_mutation`, best reward `S3R19B28_N24_B10_known_best_mutation`.
- N40: best U2 `S3R19B28_N40_B01_surrogate_top`, best reward `S3R19B28_N40_B01_surrogate_top`.

## Surrogate Prediction Audit
- Overall Spearman predicted vs realized batch28 reward: `0.549226391678098`.
- This is diagnostic calibration evidence only.

## Batch28 vs Combined80 Best Comparison
- N12: batch28 beats combined80 best metrics: `[]`.
- N16: batch28 beats combined80 best metrics: `['u2']`.
- N24: batch28 beats combined80 best metrics: `['u2', 'peeq', 'surfaceT']`.
- N40: batch28 beats combined80 best metrics: `['u2', 'peeq', 'surfaceT']`.

## Combined108 Teacher Dataset Construction
- Total rows: `108`
- Per-N rows: `{12: 24, 16: 24, 24: 30, 40: 30}`

## Claim Boundary
RUN21_BATCH28_INGESTION_AND_COMBINED108_DATASET_ONLY_NO_RL_POLICY_TRAINING. Do not claim trained RL superiority, arbitrary-N generalization, or physical superiority beyond teacher-supported metric-level comparisons.

## Output Files
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_21_batch28_teacher_metrics_ingestion_and_combined108_ranking\batch28_teacher_metrics_canonical.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_21_batch28_teacher_metrics_ingestion_and_combined108_ranking\batch28_within_N_rank_table.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_21_batch28_teacher_metrics_ingestion_and_combined108_ranking\batch28_surrogate_prediction_audit.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_21_batch28_teacher_metrics_ingestion_and_combined108_ranking\batch28_vs_combined80_best_comparison.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_21_batch28_teacher_metrics_ingestion_and_combined108_ranking\combined108_teacher_dataset.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_21_batch28_teacher_metrics_ingestion_and_combined108_ranking\combined108_RL_ready_dataset.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_21_batch28_teacher_metrics_ingestion_and_combined108_ranking\combined108_per_N_leaderboard.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_21_batch28_teacher_metrics_ingestion_and_combined108_ranking\batch28_teacher_metrics_canonical.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_21_batch28_teacher_metrics_ingestion_and_combined108_ranking\batch28_within_N_rank_table.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_21_batch28_teacher_metrics_ingestion_and_combined108_ranking\batch28_surrogate_prediction_audit.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_21_batch28_teacher_metrics_ingestion_and_combined108_ranking\batch28_vs_combined80_best_comparison.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_21_batch28_teacher_metrics_ingestion_and_combined108_ranking\combined108_teacher_dataset.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_21_batch28_teacher_metrics_ingestion_and_combined108_ranking\combined108_RL_ready_dataset.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_21_batch28_teacher_metrics_ingestion_and_combined108_ranking\combined108_per_N_leaderboard.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_21_batch28_teacher_metrics_ingestion_and_combined108_ranking\run21_batch28_input_validation_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_21_batch28_teacher_metrics_ingestion_and_combined108_ranking\batch28_highlight_cases.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_21_batch28_teacher_metrics_ingestion_and_combined108_ranking\run21_claim_boundary.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_21_batch28_teacher_metrics_ingestion_and_combined108_ranking\run21_claim_boundary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_21_batch28_teacher_metrics_ingestion_and_combined108_ranking\figures\combined108_u2_vs_peeq_by_source.png`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_21_batch28_teacher_metrics_ingestion_and_combined108_ranking\figures\batch28_predicted_vs_realized_reward.png`

## Recommended Run22
Use `combined108_RL_ready_dataset.csv` to update lightweight surrogate validation again. Re-run leave-N-out validation and compare against run17, with special attention to N40 stability. Do not train final RL policy unless explicitly instructed.
