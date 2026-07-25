# Stage 3 GitHub Incremental Commit Plan

## Guardrails

Do not use git add . or git add -A for these batches. Use explicit audited paths or a reviewed pathspec file generated from the CSV inventory.

Do not commit CAE/ODB/solver outputs, environments, caches, temporary files, generated output trees, archives, or unknown files without manual review.

## Batch Summary

| batch | files | total_size | longest_path_len | over_100MB | abaqus_files | risk | message |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Batch 1 | 234 | 4.56 MiB | 116 | 0 | 0 | MEDIUM | stage3: add core source and tests |
| Batch 2 | 0 | 0 B | 0 | 0 | 0 | MEDIUM | chore(stage3): add configs and small tools |
| Batch 3 | 147 | 702.73 KiB | 238 | 0 | 0 | HIGH | docs(stage3): add method and run documentation |
| Batch 4 | 82 | 654.10 KiB | 152 | 0 | 0 | LOW | docs(stage3): add manifests and indices |
| Batch 5 | 52 | 5.19 MiB | 166 | 0 | 0 | MEDIUM | docs(stage3): add final evidence freeze core files |
| Batch 6 | 0 | 0 B | 0 | 0 | 0 | MEDIUM | docs(stage3): add reviewed small evidence files |
| Excluded | 7325 | 569.93 MiB | 231 | 0 | 4223 | HIGH | no commit |

## Batch 1

Goal: Core source code, experiments, scripts, tests, app/core/rl modules.

Candidate path rules: Explicit audited paths under src/, scripts/, tests/, core/, rl/, app/; no git add .

File count: 234

Total size: 4.56 MiB

Longest path: stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40/scripts/stageQ_generate_v03_CAE_INP_from_sanity_base_nogui.py

Longest path length: 116

Contains files over 100 MB: NO

Contains Abaqus files: NO

Suggested commit message: stage3: add core source and tests

Risk level: MEDIUM

## Batch 2

Goal: Configuration, schemas, and small utilities after source review.

Candidate path rules: Explicit audited paths under tools/ and config-like root files; no environments

File count: 0

Total size: 0 B

Longest path: 

Longest path length: 0

Contains files over 100 MB: NO

Contains Abaqus files: NO

Suggested commit message: chore(stage3): add configs and small tools

Risk level: MEDIUM

## Batch 3

Goal: Stage 3 method docs and run reports; handle long paths after core.longpaths.

Candidate path rules: Explicit audited docs/stage3* and docs/manuscript* paths

File count: 147

Total size: 702.73 KiB

Longest path: docs/stage3/runs/run_77_final_smallN_diagnostic_batch32_teacher_metrics_ingestion_and_combined552_final_evidence_readiness/RUN_77_FINAL_SMALLN_DIAGNOSTIC_BATCH32_TEACHER_METRICS_INGESTION_AND_COMBINED552_FINAL_EVIDENCE_READINESS_REPORT.md

Longest path length: 238

Contains files over 100 MB: NO

Contains Abaqus files: NO

Suggested commit message: docs(stage3): add method and run documentation

Risk level: HIGH

## Batch 4

Goal: Manifests, indices, hashes, checksums, and metadata.

Candidate path rules: Explicit audited artifacts/manifests and manifest/hash/index paths

File count: 82

Total size: 654.10 KiB

Longest path: CHATGPT_PROJECT_UPLOAD/RL_LAM_ScanOpt_PPO_Final_320_Evidence_Package_v01/01_FINAL_STAGE_X_FREEZE/stageX_PPO_final_pool_320_evidence_freeze_manifest.json

Longest path length: 152

Contains files over 100 MB: NO

Contains Abaqus files: NO

Suggested commit message: docs(stage3): add manifests and indices

Risk level: LOW

## Batch 5

Goal: Final evidence freeze core markdown/csv/json/txt under size bounds.

Candidate path rules: Explicit audited final/evidence/freeze files <=25MB, excluding Abaqus/solver paths

File count: 52

Total size: 5.19 MiB

Longest path: CHATGPT_PROJECT_UPLOAD/RL_LAM_ScanOpt_PPO_Final_320_Evidence_Package_v01/03_STAGE_HISTORY_SUMMARIES/PPO_FINAL_EXPANSION_STAGEV_ODB_TEACHER_METRIC_EXTRACTION_REPORT.md

Longest path length: 166

Contains files over 100 MB: NO

Contains Abaqus files: NO

Suggested commit message: docs(stage3): add final evidence freeze core files

Risk level: MEDIUM

## Batch 6

Goal: Other small reviewed evidence files.

Candidate path rules: Explicit audited small evidence/docs <=10MB, excluding generated outputs and Abaqus

File count: 0

Total size: 0 B

Longest path: 

Longest path length: 0

Contains files over 100 MB: NO

Contains Abaqus files: NO

Suggested commit message: docs(stage3): add reviewed small evidence files

Risk level: MEDIUM

## Excluded

Goal: Excluded from automated commit planning.

Candidate path rules: CAE/ODB/solver outputs, environments, caches, generated outputs, large outputs, archives, unknown files

File count: 7325

Total size: 569.93 MiB

Longest path: outputs/stage3_run_25_shortlist64_active_learning_cae_inp_generation/archived_failed_N40_B02_B03_B04_B05_solver_outputs_before_cool_initialInc_patch_20260613_085016/N40S3R24L64_N40_B05_top_region/J2D_S3R24L64_N40_B05_top_region.env

Longest path length: 231

Contains files over 100 MB: NO

Contains Abaqus files: YES

Suggested commit message: no commit

Risk level: HIGH

## Recommended Workflow

1. Regenerate or review docs/repository/stage3_untracked_file_inventory.csv.
2. For one batch at a time, create an explicit path list from reviewed rows only.
3. Stage with explicit paths, never git add . or git add -A.
4. Run git diff --cached --check, git diff --cached --name-only, and inspect for forbidden extensions before each commit.
5. Push only after a successful commit-level audit.