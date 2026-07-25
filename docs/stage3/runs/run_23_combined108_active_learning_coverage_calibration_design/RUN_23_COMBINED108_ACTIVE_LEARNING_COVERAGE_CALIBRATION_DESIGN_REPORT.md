# Stage 3 Run 23 - Combined108 Active-Learning Coverage and Calibration Design

## Purpose
Generate an offline active-learning candidate design using combined108 diagnostics, focused on top-region calibration, model disagreement, uncertainty, and N24/N40 coverage rather than pure exploitation.

## Inputs
- Combined108 RL-ready dataset: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_21_batch28_teacher_metrics_ingestion_and_combined108_ranking\combined108_RL_ready_dataset.csv`
- Combined108 teacher dataset: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_21_batch28_teacher_metrics_ingestion_and_combined108_ranking\combined108_teacher_dataset.csv`
- Run22 best configurations: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_22_combined108_surrogate_reward_model_validation_update\combined108_best_surrogate_configurations.csv`
- Run22 feature definitions: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_22_combined108_surrogate_reward_model_validation_update\run22_feature_set_definitions.json`
- Previous run18 candidate pool for duplicate avoidance: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_18_combined80_surrogate_screened_candidate_generation\run18_candidate_pool_scored.csv`

## Run22 Motivation
- Run22 improved leave-N-out macro Spearman to 0.8651 but weakened top5 retrieval to 2.5/5.
- Run23 therefore prioritizes calibration, coverage, disagreement, and top-region exploration instead of pure predicted reward maximization.

## Model Ensemble Used For Design
- Ensemble: `{'primary_exploitation_model': 'ExtraTreesRegressor/F01_basic_order', 'n40_stability_comparison_models': ['Ridge/F03_family_plus_features', 'Ridge/F06_no_dataset_source'], 'robustness_models': ['ExtraTreesRegressor/F04_no_family_generalization', 'ExtraTreesRegressor/F05_n_agnostic'], 'secondary_metric_models': ['U2', 'PEEQ', 'SurfaceT', 'Mises']}`
- Primary target: `target_reward_combined108_u2_primary`
- Training rows used for offline scoring: `108`
- These are combined108 offline diagnostic surrogates, not final or deployed models.

## Candidate Generation Scope
- N values: `[12, 16, 24, 40]`
- Deduplicated candidates per N: `{12: 800, 16: 958, 24: 2893, 40: 3000}`
- Selection is intentionally biased toward N24/N40.

## Candidate Generation Methods
- Top-region local search around known combined108 teacher-best cases.
- N24/N40 top5-retrieval calibration sweeps in parity, jump, first/last track, and direction-reversal features.
- Model-disagreement and uncertainty candidates from ET_F01 versus Ridge F03/F06.
- Tradeoff candidates for U2/PEEQ/SurfaceT/Mises tension.
- Regular-jump, Method-C, and known-best neighborhoods.
- Diversity and sentinel/control candidates.
- No candidate is labelled as trained RL output.

## Candidate Validation and Deduplication
- All generated orders were validated as legal permutations of 0..N-1.
- Exact duplicates of combined108 teacher orders were removed.
- Exact duplicates of the prior run18/run19 candidate pools were avoided where detected.

## Multi-Model Scoring
- Candidates were scored by ET_F01 reward, Ridge F03/F06 reward, ET F04/F05 robustness models, and secondary U2/PEEQ/SurfaceT/Mises models.
- Tree-wise standard deviation, model prediction standard deviation, rank disagreement, novelty, and feature-space coverage are reported as active-learning diagnostics.

## Selection Policy
- Shortlist64 counts: `{12: 8, 16: 8, 24: 24, 40: 24}`
- Recommended batch32 counts: `{12: 4, 16: 4, 24: 12, 40: 12}`
- Alternative batch24 counts: `{12: 2, 16: 2, 24: 10, 40: 10}`
- Selection buckets include top-region local search, model disagreement, uncertainty calibration, diversity coverage, tradeoff probes, sentinel controls, and a small exploitation reference bucket.

## Predicted Comparison to Combined108 Best
- N12: ET_F01 top `N12_R23_N12_C00144_known_best_mutation` predicted reward `0.7810`, gap vs existing best `-0.0700` surrogate-only.
- N16: ET_F01 top `N16_R23_N16_C00213_known_best_mutation` predicted reward `0.7737`, gap vs existing best `-0.0818` surrogate-only.
- N24: ET_F01 top `N24_R23_N24_C00189_known_best_mutation` predicted reward `0.7605`, gap vs existing best `-0.1076` surrogate-only.
- N40: ET_F01 top `N40_R23_N40_C00126_known_best_mutation` predicted reward `0.7793`, gap vs existing best `-0.1146` surrogate-only.

## N24/N40 Coverage Analysis
- The shortlist and batch outputs deliberately allocate most capacity to N24/N40.
- Coverage uses novelty to combined108, model disagreement, uncertainty, and top-region local search around known teacher-best cases.

## Predicted Improvement vs Combined108 Best
Predicted improvements are surrogate-only. They require future teacher validation before any physical claim.

## Diagnostics
- Candidate distributions, novelty, uncertainty, family composition, and duplicate-removal summaries are written to the diagnostics files.

## Claim Boundary
RUN23_ACTIVE_LEARNING_DESIGN_ONLY_NO_TEACHER_VALIDATION. No teacher validation, physical superiority, trained RL success, arbitrary-N generalization, or CAE/INP existence is claimed.

## Output Files
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_23_combined108_active_learning_coverage_calibration_design\run23_candidate_pool_unscored.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_23_combined108_active_learning_coverage_calibration_design\run23_candidate_pool_scored.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_23_combined108_active_learning_coverage_calibration_design\run23_candidate_shortlist64.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_23_combined108_active_learning_coverage_calibration_design\run23_recommended_active_learning_batch32.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_23_combined108_active_learning_coverage_calibration_design\run23_conservative_active_learning_batch24.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_23_combined108_active_learning_coverage_calibration_design\run23_predicted_comparison_vs_combined108.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_23_combined108_active_learning_coverage_calibration_design\run23_candidate_generation_diagnostics.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_23_combined108_active_learning_coverage_calibration_design\run23_candidate_pool_unscored.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_23_combined108_active_learning_coverage_calibration_design\run23_candidate_pool_scored.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_23_combined108_active_learning_coverage_calibration_design\run23_candidate_shortlist64.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_23_combined108_active_learning_coverage_calibration_design\run23_recommended_active_learning_batch32.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_23_combined108_active_learning_coverage_calibration_design\run23_conservative_active_learning_batch24.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_23_combined108_active_learning_coverage_calibration_design\run23_predicted_comparison_vs_combined108.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_23_combined108_active_learning_coverage_calibration_design\run23_candidate_generation_diagnostics.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_23_combined108_active_learning_coverage_calibration_design\run23_claim_boundary.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_23_combined108_active_learning_coverage_calibration_design\run23_claim_boundary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_23_combined108_active_learning_coverage_calibration_design\run23_input_validation_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_23_combined108_active_learning_coverage_calibration_design\run23_surrogate_model_metadata.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_23_combined108_active_learning_coverage_calibration_design\figures\run23_predicted_reward_histogram.png`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_23_combined108_active_learning_coverage_calibration_design\figures\run23_predicted_reward_vs_uncertainty.png`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_23_combined108_active_learning_coverage_calibration_design\figures\run23_predicted_reward_vs_disagreement.png`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_23_combined108_active_learning_coverage_calibration_design\figures\run23_batch32_bucket_composition.png`

## Recommended Run24
Human review and handoff packaging for either batch32 or batch24. Do not generate CAE/INP until the user selects a batch. Choose batch24 to control compute cost; choose batch32 for stronger N24/N40 calibration.
