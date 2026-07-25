# PPO v03 Lex-Primary Candidate Generation Report

## Purpose
Generate a PPO-only N24/N40 candidate batch that explicitly prioritizes U2-primary and lexicographic U2->PEEQ->SurfaceT performance while avoiding SurfaceT-only false positives.

## Why v03 Was Needed
PPO v01/v02K2 did not beat mature combined552 records. v02K2 improved N40 SurfaceT top-k counts, but not primary lexicographic ranking. v03 therefore targets lex-primary and U2-guarded rewards.

## Dataset Assembly
Input dataset: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40\data\v03_N24_N40_teacher_dataset.csv`. Rows: combined552 N24/N40 + PPO v01 N24/N40 + PPO v02K2 N24/N40.

## Surrogate/Ranking Model Results
Summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40\surrogate_v03\tables\v03_surrogate_model_selection_summary.json`.

## PPO Training Status
Summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40\ppo_training_v03\v03_ppo_training_summary.json`.

## Internal Surrogate Evaluation
Summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40\ppo_training_v03\tables\v03_internal_eval_summary.json`.

## Rollout Pool Size And Uniqueness
- Pool CSV: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40\candidate_generation_v03\rollout_pool\v03_ppo_rollout_pool.csv`
- Pool counts by N: {24: 121, 40: 161}
- Unique orders by N: {24: 121, 40: 161}

## Candidate Selection Logic
Per N: 5 top final v03 score, 4 top U2-guarded, 3 lex-primary with novelty, 2 diverse upper quartile, 1 deterministic, 1 record-seeking, with overlap filled by next eligible PPO-generated candidates.

## Audits
- Legality: `PASS` at `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40\candidate_generation_v03\tables\v03_candidate_legality_audit.csv`
- Novelty: `PASS` at `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40\candidate_generation_v03\tables\v03_candidate_novelty_audit.csv`
- SurfaceT false-positive screening: `PASS` at `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40\candidate_generation_v03\tables\v03_surfaceT_false_positive_screening.csv`
- Score summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40\candidate_generation_v03\tables\v03_candidate_score_summary_by_N.csv`

## Selected Batch
Selected path: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40\candidate_generation_v03\selected_batch32\v03_ppo_lex_primary_N24_N40_candidate_batch32.csv`
Counts by N: {24: 16, 40: 16}

## Claim Boundary
The v03 batch is not physically validated. Surrogate scores are candidate-generation/ranking signals only.

## Verdict
`PASS_PPO_V03_LEX_PRIMARY_BATCH32_READY_FOR_CAE_INP_HANDOFF`
