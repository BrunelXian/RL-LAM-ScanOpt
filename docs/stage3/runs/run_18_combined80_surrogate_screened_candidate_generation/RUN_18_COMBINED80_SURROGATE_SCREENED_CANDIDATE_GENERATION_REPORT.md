# Stage 3 Run 18 - Combined80 Surrogate-Screened Candidate Generation

## Purpose
Generate a second offline candidate pool using the combined80-updated diagnostic surrogate, biased toward N24/N40 while retaining N12/N16 calibration and sentinel cases.

## Inputs
- Combined80 RL-ready dataset: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_16_batch20_teacher_metrics_ingestion_and_combined80_ranking\combined80_RL_ready_dataset.csv`
- Run17 best configurations: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_17_combined80_surrogate_reward_model_validation_update\combined80_best_surrogate_configurations.csv`
- Run17 feature definitions: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_17_combined80_surrogate_reward_model_validation_update\run17_feature_set_definitions.json`
- Previous run12 candidate pool for duplicate avoidance: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_12_offline_surrogate_screened_candidate_generation\run12_candidate_pool_scored.csv`

## Run17 Surrogate Basis
- Model: `ExtraTreesRegressor`
- Feature set: `F01_basic_order`
- Primary target: `target_reward_combined80_u2_primary`
- Training rows used for offline scoring: `80`
- This is a combined80 offline diagnostic surrogate, not a final or deployed model.

## Candidate Generation Scope
- N values: `[12, 16, 24, 40]`
- Deduplicated candidates per N: `{12: 1000, 16: 1166, 24: 2101, 40: 3036}`
- Selection is intentionally biased toward N24/N40.

## Candidate Generation Methods
- Geometry-first candidates based on F01 signal: parity switching, jump statistics, direction reversals, and first/last track placement.
- Method-C and known-best inspired variants using mutation, crossover, segment reversal, and block swaps.
- Regular jump and coprime sweeps, especially for N24/N40.
- Random and quasi-random diversity/calibration/sentinel candidates.
- No candidate is labelled as trained RL output.

## Candidate Validation and Deduplication
- All generated orders were validated as legal permutations of 0..N-1.
- Exact duplicates of combined80 teacher orders were removed.
- Exact duplicates of the prior run12 pool were avoided where detected.

## Surrogate Scoring
- Candidates were scored by predicted within-N normalized combined80 U2-primary reward.
- Secondary diagnostic targets include U2, PEEQ, SurfaceT, and Mises rank scores.
- ExtraTrees tree-wise standard deviation is reported as an uncertainty proxy.

## N24/N40-Biased Shortlist Policy
- Shortlist48 counts: `{12: 8, 16: 8, 24: 16, 40: 16}`
- Recommended batch28 counts: `{12: 4, 16: 4, 24: 10, 40: 10}`
- Alternative batch24 counts: `{12: 3, 16: 3, 24: 9, 40: 9}`
- Selection buckets include surrogate top, U2-primary top, geometry signal top, Method-C/known-best inspired, diversity, uncertainty calibration, and negative/control sentinels.

## Top Predicted Candidates
- N12: `N12_R18_N12_C00205_known_best_mutation` predicted reward `0.7768`, gap vs existing best `-0.0811` surrogate-only.
- N16: `N16_R18_N16_C00297_known_best_mutation` predicted reward `0.7838`, gap vs existing best `-0.1202` surrogate-only.
- N24: `N24_R18_N24_C00601_known_best_mutation` predicted reward `0.8347`, gap vs existing best `-0.0917` surrogate-only.
- N40: `N40_R18_N40_C01795_known_best_mutation` predicted reward `0.8515`, gap vs existing best `-0.0801` surrogate-only.

## Predicted Improvement vs Combined80 Best
Predicted improvements are surrogate-only. They require future teacher validation before any physical claim.

## Diagnostics
- Candidate distributions, novelty, uncertainty, family composition, and duplicate-removal summaries are written to the diagnostics files.

## Claim Boundary
RUN18_OFFLINE_SURROGATE_SCREENING_ONLY_NO_TEACHER_VALIDATION. No teacher validation, physical superiority, trained RL success, arbitrary-N generalization, or CAE/INP existence is claimed.

## Output Files
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_18_combined80_surrogate_screened_candidate_generation\run18_candidate_pool_unscored.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_18_combined80_surrogate_screened_candidate_generation\run18_candidate_pool_scored.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_18_combined80_surrogate_screened_candidate_generation\run18_candidate_shortlist48.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_18_combined80_surrogate_screened_candidate_generation\run18_recommended_future_teacher_batch28.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_18_combined80_surrogate_screened_candidate_generation\run18_recommended_future_teacher_batch24.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_18_combined80_surrogate_screened_candidate_generation\run18_predicted_improvement_vs_combined80.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_18_combined80_surrogate_screened_candidate_generation\run18_candidate_generation_diagnostics.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_18_combined80_surrogate_screened_candidate_generation\run18_candidate_pool_unscored.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_18_combined80_surrogate_screened_candidate_generation\run18_candidate_pool_scored.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_18_combined80_surrogate_screened_candidate_generation\run18_candidate_shortlist48.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_18_combined80_surrogate_screened_candidate_generation\run18_recommended_future_teacher_batch28.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_18_combined80_surrogate_screened_candidate_generation\run18_recommended_future_teacher_batch24.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_18_combined80_surrogate_screened_candidate_generation\run18_predicted_improvement_vs_combined80.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_18_combined80_surrogate_screened_candidate_generation\run18_candidate_generation_diagnostics.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_18_combined80_surrogate_screened_candidate_generation\run18_claim_boundary.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_18_combined80_surrogate_screened_candidate_generation\run18_claim_boundary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_18_combined80_surrogate_screened_candidate_generation\run18_input_validation_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_18_combined80_surrogate_screened_candidate_generation\run18_surrogate_model_metadata.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_18_combined80_surrogate_screened_candidate_generation\figures\run18_predicted_reward_histogram.png`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_18_combined80_surrogate_screened_candidate_generation\figures\run18_predicted_reward_vs_novelty.png`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_18_combined80_surrogate_screened_candidate_generation\figures\run18_batch28_bucket_composition.png`

## Recommended Run19
Human review and handoff packaging for either batch24 or batch28. Do not generate CAE/INP until the user selects a batch size. If approved, run19 should create a handoff-only package similar to run13, not directly run Abaqus.
