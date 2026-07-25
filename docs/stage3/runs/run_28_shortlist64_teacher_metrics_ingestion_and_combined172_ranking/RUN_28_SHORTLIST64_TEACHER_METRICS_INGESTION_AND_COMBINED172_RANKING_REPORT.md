# Stage 3 Run 28 - Shortlist64 Teacher Metrics Ingestion and Combined172 Ranking

## Purpose
Ingest completed Run27 shortlist64 teacher metrics, merge active-learning metadata, recompute within-N rankings, build combined172, and audit Run23 calibration against realized teacher metrics.

## Inputs
- Run27 teacher metrics: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_27_shortlist64_odb_teacher_validation\run27_shortlist64_teacher_metrics.csv`
- Run24 shortlist64 handoff metadata: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\stage3_run24_shortlist64_candidate_orders.csv`
- Previous combined108 teacher dataset: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_21_batch28_teacher_metrics_ingestion_and_combined108_ranking\combined108_teacher_dataset.csv`
- Run26 GNN report used only for boundary context: `E:\Projects\RL-LAM-ScanOpt\docs\stage3\runs\run_26_combined108_gnn_graph_pointer_policy_candidate_generation\RUN_26_COMBINED108_GNN_GRAPH_POINTER_POLICY_CANDIDATE_GENERATION_REPORT.md`

## Run27 Teacher-Validation Status
- User-provided upstream verdict: `PASS_RUN27_SHORTLIST64_ODB_TEACHER_VALIDATION_64_OF_64`.
- Run28 did not open ODB files or perform solver/extraction work; it read the completed CSV/JSON outputs only.

## Input Validation
- Verdict: `PASS_RUN28_SHORTLIST64_TEACHER_METRICS_64_OF_64_READY`
- Run27 rows: `64`; per-N: `{12: 8, 16: 8, 24: 24, 40: 24}`
- Combined108 rows: `108`; per-N: `{12: 24, 16: 24, 24: 30, 40: 30}`
- Run24 metadata direct matches: `64`

## Run27 Enriched Teacher Dataset
The enriched table preserves handoff strategy names, Run23 candidate IDs, bucket metadata, prediction metadata, novelty/disagreement fields, and official teacher metrics.

## Run27 Within-Batch Ranking
- N12: best U2 `S3R24L64_N12_B05_diversity_coverage`, best reward `S3R24L64_N12_B01_top_region`.
- N16: best U2 `S3R24L64_N16_B02_top_region`, best reward `S3R24L64_N16_B02_top_region`.
- N24: best U2 `S3R24L64_N24_B20_tradeoff_probe`, best reward `S3R24L64_N24_B23_exploitation_reference`.
- N40: best U2 `S3R24L64_N40_B23_exploitation_reference`, best reward `S3R24L64_N40_B23_exploitation_reference`.

## Combined172 Construction
- Total rows: `172`
- Per-N rows: `{12: 32, 16: 32, 24: 54, 40: 54}`

## Run27 vs Combined108 Best Comparison
- N12: Run27 beats combined108 best metrics: `[]`.
- N16: Run27 beats combined108 best metrics: `['u2', 'combined_reward']`.
- N24: Run27 beats combined108 best metrics: `['surfaceT']`.
- N40: Run27 beats combined108 best metrics: `['u2', 'combined_reward']`.

## Prediction Audit for Run23 Active-Learning Design
- Overall Spearman predicted vs realized Run27 reward: `0.7373566246580882`
- Mean top5 overlap: `3.0 / 5`
- Top1 hits: `1 / 4`

## Bucket/Source Performance
- Best bucket by Run27 reward rank: `exploitation_reference` with best reward rank `1.0`.

## N24/N40 Focus Analysis
- N24: `24` Run27 active-learning cases; new-best metrics `['surfaceT']`.
- N40: `24` Run27 active-learning cases; new-best metrics `['u2', 'combined_reward']`.

## Context Note on Run26 GNN Prototype
Run27 teacher metrics are not GNN-policy validation. They validate the Run23/Run24 active-learning shortlist64 batch only. Combined172 can support a later update of both surrogate and offline GNN / graph-pointer policy models.

## Claim Boundary
`RUN28_SHORTLIST64_INGESTION_AND_COMBINED172_RANKING_ONLY_NOT_GNN_VALIDATION`.

## Output Files
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_28_shortlist64_teacher_metrics_ingestion_and_combined172_ranking\run27_shortlist64_teacher_dataset_enriched.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_28_shortlist64_teacher_metrics_ingestion_and_combined172_ranking\run27_shortlist64_ranked_within_batch.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_28_shortlist64_teacher_metrics_ingestion_and_combined172_ranking\run27_shortlist64_per_N_leaderboard.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_28_shortlist64_teacher_metrics_ingestion_and_combined172_ranking\run27_shortlist64_bucket_performance.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_28_shortlist64_teacher_metrics_ingestion_and_combined172_ranking\combined172_teacher_dataset.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_28_shortlist64_teacher_metrics_ingestion_and_combined172_ranking\combined172_RL_ready_dataset.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_28_shortlist64_teacher_metrics_ingestion_and_combined172_ranking\combined172_per_N_leaderboard.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_28_shortlist64_teacher_metrics_ingestion_and_combined172_ranking\run27_vs_combined108_best_comparison.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_28_shortlist64_teacher_metrics_ingestion_and_combined172_ranking\run27_shortlist64_prediction_audit.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_28_shortlist64_teacher_metrics_ingestion_and_combined172_ranking\run27_shortlist64_top_region_retrieval_audit.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_28_shortlist64_teacher_metrics_ingestion_and_combined172_ranking\run27_bucket_source_performance_summary.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_28_shortlist64_teacher_metrics_ingestion_and_combined172_ranking\run27_shortlist64_teacher_dataset_enriched.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_28_shortlist64_teacher_metrics_ingestion_and_combined172_ranking\combined172_teacher_dataset.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_28_shortlist64_teacher_metrics_ingestion_and_combined172_ranking\run27_vs_combined108_best_comparison.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_28_shortlist64_teacher_metrics_ingestion_and_combined172_ranking\run27_bucket_source_performance_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_28_shortlist64_teacher_metrics_ingestion_and_combined172_ranking\run28_input_validation_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_28_shortlist64_teacher_metrics_ingestion_and_combined172_ranking\combined172_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_28_shortlist64_teacher_metrics_ingestion_and_combined172_ranking\run27_shortlist64_prediction_audit_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_28_shortlist64_teacher_metrics_ingestion_and_combined172_ranking\run28_claim_boundary.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_28_shortlist64_teacher_metrics_ingestion_and_combined172_ranking\run28_claim_boundary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_28_shortlist64_teacher_metrics_ingestion_and_combined172_ranking\run28_gnn_context_note.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_28_shortlist64_teacher_metrics_ingestion_and_combined172_ranking\figures\combined172_u2_vs_peeq_by_source.png`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_28_shortlist64_teacher_metrics_ingestion_and_combined172_ranking\figures\run27_predicted_vs_realized_reward.png`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_28_shortlist64_teacher_metrics_ingestion_and_combined172_ranking\figures\run27_bucket_reward_rank_performance.png`

## Recommended Run29
Use the combined172 dataset to update surrogate and offline GNN / graph-pointer policy models. If Run27 produced strong new bests, prepare a refined hybrid-policy candidate batch; if it mainly improved calibration, focus Run29 on top-region model calibration before proposing more solver cases.
