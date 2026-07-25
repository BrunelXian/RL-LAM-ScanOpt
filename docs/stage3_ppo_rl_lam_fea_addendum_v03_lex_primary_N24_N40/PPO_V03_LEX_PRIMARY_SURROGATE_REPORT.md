# PPO v03 Lex-Primary Surrogate Report

## Purpose
Train N-specific v03 surrogate/ranking models that prioritize U2 and lexicographic U2->PEEQ->SurfaceT reward while penalizing SurfaceT-only false positives.

## Dataset
Input: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40\data\v03_N24_N40_teacher_dataset.csv`. N24/N40 only; combined552 + teacher-validated PPO v01 + teacher-validated PPO v02K2.

## Targets
- `reward_lex_primary_v03`
- `reward_u2_guarded_v03`
- `reward_record_seeking_v03`
- `reward_topk_classifier_v03`

## Best Lex-Primary Validation
- N24: ExtraTreesRegressor, Spearman 0.8977, Pearson 0.8317, MAE 0.1799, RMSE 0.2809, top25 hit quality 0.8333
- N40: ExtraTreesRegressor, Spearman 0.9289, Pearson 0.8938, MAE 0.1391, RMSE 0.2171, top25 hit quality 0.7692

## SurfaceT False-Positive Audit
Validation false-positive audit: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40\surrogate_v03\tables\v03_surrogate_false_positive_audit.csv`.

## Verdict
`PASS_V03_LEX_PRIMARY_SURROGATES_READY_FOR_PPO`
