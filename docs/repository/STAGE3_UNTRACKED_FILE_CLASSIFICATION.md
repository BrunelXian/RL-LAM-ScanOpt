# Stage 3 Untracked File Classification

## Scope

This report classifies the files reported by git status --porcelain=v1 -uall before this audit generated its own report files. It does not rely on GitHub Desktop selection state.

No files were staged, committed, moved, copied, renamed, deleted, or opened as scientific data.

## Summary

- Total changed/untracked status entries before this audit report set: 7840
- Tracked/modified entries: 1
- Untracked entries: 7839
- Ignored Abaqus/safety-extension files observed separately: 7371

## Category Table

| category | file_count | total_size | largest_file | tracked | untracked | ignored | recommended_action |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| source_code | 233 | 4.56 MiB | scripts/stage3/run_23_generate_active_learning_coverage_calibration_candidates.py | 0 | 233 | 0 | COMMIT_CANDIDATE |
| documentation | 175 | 758.95 KiB | docs/stage3/cae_model_generation/PROBE60_CAE_GENERATION_REPORT.md | 0 | 175 | 0 | COMMIT_CANDIDATE |
| manifests_and_metadata | 84 | 679.81 KiB | artifacts/manifests/stage3_run_68_manifest.json | 1 | 83 | 0 | COMMIT_CANDIDATE |
| small_evidence | 23 | 5.11 MiB | CHATGPT_PROJECT_UPLOAD/RL_LAM_ScanOpt_PPO_Final_320_Evidence_Package_v01/04_CORE_REFERENCE_DATA/FROZEN_stage3_native_combined552_teacher_dataset.csv | 0 | 23 | 0 | REVIEW_MANUALLY |
| generated_outputs | 3071 | 454.78 MiB | outputs/stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40/surrogate_v03/models/N40_v03_lex_primary_surrogate.joblib | 0 | 3071 | 0 | REVIEW_MANUALLY |
| Abaqus_and_solver_files | 4223 | 95.09 MiB | cae_model/stage3_run59_N40_focused_calibrated_penalty_repair_batch40_v01/N40S3R59N40PR40_N40_B08_n40_penalty_repair/J2D_S3R59N40PR40_N40_B08_n40_penalty_repair.inp | 0 | 4223 | 7371 | DO_NOT_COMMIT |
| archives | 22 | 19.65 MiB | CHATGPT_PROJECT_UPLOAD/RL_LAM_ScanOpt_PPO_Final_320_Evidence_Package_v01.zip | 0 | 22 | 0 | REVIEW_MANUALLY |
| caches_and_environments | 0 | 0 B |  | 0 | 0 | 0 | KEEP_LOCAL_IGNORED |
| unknown | 9 | 424.19 KiB | CHATGPT_PROJECT_UPLOAD/RL_LAM_ScanOpt_PPO_Final_320_Evidence_Package_v01/09_OPTIONAL_PLOTS_INDEX_ONLY/plots/final_expansion_vs_prior_ppo_best_lex_rank.png | 0 | 9 | 0 | UNKNOWN_REQUIRES_REVIEW |

## Notes

- Abaqus_and_solver_files are not commit candidates in this plan.
- generated_outputs may include small evidence and repeated runtime outputs; review manually before staging.
- caches_and_environments should stay local/ignored.
- unknown requires manual review before any Git operation.

## Inventory

Detailed status inventory: docs/repository/stage3_untracked_file_inventory.csv