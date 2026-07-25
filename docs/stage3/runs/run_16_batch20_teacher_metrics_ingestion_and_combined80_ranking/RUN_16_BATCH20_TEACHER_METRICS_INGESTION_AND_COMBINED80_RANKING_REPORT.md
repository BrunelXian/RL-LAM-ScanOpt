# Stage 3 Run 16 - Batch20 Teacher Metrics Ingestion and Combined80 Ranking

## Purpose
Ingest 20 official batch20 teacher metrics, audit surrogate predictions, compare against probe60, and build a combined 80-case teacher-labelled dataset.

## Inputs
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_14_batch20_odb_teacher_validation\run14_batch20_teacher_metrics.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_13_batch20_surrogate_screened_teacher_handoff\stage3_run13_batch20_candidate_orders.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_09_variable_n_probe60_teacher_ranking_analysis\probe60_teacher_ranked_canonical.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_10_variable_n_normalized_reward_surrogate_dataset\probe60_variable_n_reward_dataset.csv`

## Batch20 Validation Status
- `PASS_RUN16_BATCH20_TEACHER_METRICS_20_OF_20_READY`
- Per-N counts: {12: 5, 16: 5, 24: 5, 40: 5}
- Solver completion verdict: `WARNING_RUN14_BATCH20_SOLVER_COMPLETION_WITH_NONFATAL_WARNINGS`
- Extraction verdict: `PASS_RUN14_BATCH20_ODB_TEACHER_VALIDATION_20_OF_20`

## Batch20 Within-N Ranking
Batch20 ranks, min-max costs, U2-primary reward, and Pareto flags were recomputed over the 5 candidates per N.

## Surrogate Prediction Audit
- Overall Spearman predicted vs realized batch20 reward: `0.43157894736842106`
- Top1/top2 within-N comparisons are recorded in `batch20_surrogate_prediction_audit.csv`.

## Batch20 vs Probe60 Best Comparison
- N12: beats probe60 best U2=True, PEEQ=False, SurfaceT=False, Mises=False.
- N16: beats probe60 best U2=False, PEEQ=True, SurfaceT=False, Mises=False.
- N24: beats probe60 best U2=False, PEEQ=True, SurfaceT=False, Mises=False.
- N40: beats probe60 best U2=True, PEEQ=False, SurfaceT=False, Mises=False.

## Combined80 Teacher Dataset Construction
- Total rows: 80.
- Per-N rows: 20 for N12/N16/N24/N40.

## Combined80 Within-N Rankings
Ranks, rank scores, min-max costs, U2-primary reward, and Pareto flags were recomputed over 20 cases per N.

## RL-Ready Dataset
- `combined80_RL_ready_dataset.csv` contains target columns for updated surrogate/RL analysis.

## Highlight Cases
- `S3B20_N40_B02_diversity_top` is included and teacher-valid.

## Claim Boundary
- No final RL policy superiority, arbitrary-N generalization, fixed-32 guard transfer, or final optimum is claimed.

## Output Files
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_16_batch20_teacher_metrics_ingestion_and_combined80_ranking\run16_batch20_input_validation_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_16_batch20_teacher_metrics_ingestion_and_combined80_ranking\batch20_teacher_metrics_canonical.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_16_batch20_teacher_metrics_ingestion_and_combined80_ranking\batch20_teacher_metrics_canonical.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_16_batch20_teacher_metrics_ingestion_and_combined80_ranking\batch20_within_N_rank_table.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_16_batch20_teacher_metrics_ingestion_and_combined80_ranking\batch20_within_N_rank_table.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_16_batch20_teacher_metrics_ingestion_and_combined80_ranking\batch20_surrogate_prediction_audit.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_16_batch20_teacher_metrics_ingestion_and_combined80_ranking\batch20_surrogate_prediction_audit.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_16_batch20_teacher_metrics_ingestion_and_combined80_ranking\batch20_vs_probe60_best_comparison.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_16_batch20_teacher_metrics_ingestion_and_combined80_ranking\batch20_vs_probe60_best_comparison.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_16_batch20_teacher_metrics_ingestion_and_combined80_ranking\combined80_teacher_dataset.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_16_batch20_teacher_metrics_ingestion_and_combined80_ranking\combined80_teacher_dataset.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_16_batch20_teacher_metrics_ingestion_and_combined80_ranking\combined80_RL_ready_dataset.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_16_batch20_teacher_metrics_ingestion_and_combined80_ranking\combined80_RL_ready_dataset.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_16_batch20_teacher_metrics_ingestion_and_combined80_ranking\combined80_per_N_leaderboard.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_16_batch20_teacher_metrics_ingestion_and_combined80_ranking\combined80_per_N_leaderboard.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_16_batch20_teacher_metrics_ingestion_and_combined80_ranking\batch20_highlight_cases.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_16_batch20_teacher_metrics_ingestion_and_combined80_ranking\run16_claim_boundary.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_16_batch20_teacher_metrics_ingestion_and_combined80_ranking\run16_claim_boundary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_16_batch20_teacher_metrics_ingestion_and_combined80_ranking\figures\u2_vs_peeq_by_source.png`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_16_batch20_teacher_metrics_ingestion_and_combined80_ranking\figures\u2_vs_surfaceT_by_source.png`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_16_batch20_teacher_metrics_ingestion_and_combined80_ranking\figures\batch20_predicted_vs_realized_reward.png`
- `E:\Projects\RL-LAM-ScanOpt\docs\stage3\runs\run_16_batch20_teacher_metrics_ingestion_and_combined80_ranking\RUN_16_BATCH20_TEACHER_METRICS_INGESTION_AND_COMBINED80_RANKING_REPORT.md`

## Recommended Run17
Use `combined80_RL_ready_dataset.csv` to update lightweight surrogate validation with the expanded 80-case dataset. Re-run leave-N-out validation and compare against run11. Do not train final RL policy yet unless explicitly instructed.
