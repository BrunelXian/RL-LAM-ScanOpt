# Git Long Path Audit

## Verdict

WARNING_LONG_PATH_CONFIGURED_WITH_REMAINING_PATH_RISKS

## Repository State

- Repository root: E:/Projects/RL-LAM-ScanOpt
- Branch: stage3-variable-n-graph-pointer-init-v01
- Local HEAD: 55f08b28a5d81330457aa6db95de29e9eb975abf
- Remote HEAD: 55f08b28a5d81330457aa6db95de29e9eb975abf
- Failed Desktop commit created a commit: NO
- Staged files after recovery: none

## Long Path Configuration

- core.longpaths lookup after local setting: file:.git/config	true
- Local core.longpaths: true
- Scope changed: repository local .git/config only

This setting does not move, rename, copy, delete, or modify research files. It only changes how Git in this repository handles long paths.

## Known Failing Path

- Relative path length: 238
- Absolute path length: 265
- Path: docs/stage3/runs/run_77_final_smallN_diagnostic_batch32_teacher_metrics_ingestion_and_combined552_final_evidence_readiness/RUN_77_FINAL_SMALLN_DIAGNOSTIC_BATCH32_TEACHER_METRICS_INGESTION_AND_COMBINED552_FINAL_EVIDENCE_READINESS_REPORT.md

## Path Length Statistics

- Relative path > 180 characters: 432
- Relative path > 220 characters: 106
- Relative path > 240 characters: 0
- Absolute path > 240 characters: 107
- Absolute path > 260 characters: 25
- Longest relative path length: 238
- Longest absolute path length: 265

## Abaqus Path Handling

Abaqus-related files were not opened or modified. They are aggregated only.

- Abaqus/safety-extension files counted: 11594
- Abaqus/safety-extension relative paths > 240: 0
- Abaqus/safety-extension absolute paths > 260: 21

## Risk Classes

A. core.longpaths=true lets repository-local Git process paths that GitHub Desktop previously rejected.

B. Extremely long paths may still affect GitHub Desktop, Windows shell integrations, editors, backup tools, and archive tools even when command-line Git can process them.

C. Future shortening candidates include long Stage 3 run directory names and repeated all-caps report filenames. This audit does not rename anything.

D. Frozen evidence paths and run reports may encode provenance and should not be renamed without an explicit reproducibility decision.

## Longest Ordinary Text/Code Paths

| relative_len | absolute_len | size | extension | path |
| ---: | ---: | ---: | --- | --- |
| 238 | 265 | 5.42 KiB | .md | docs/stage3/runs/run_77_final_smallN_diagnostic_batch32_teacher_metrics_ingestion_and_combined552_final_evidence_readiness/RUN_77_FINAL_SMALLN_DIAGNOSTIC_BATCH32_TEACHER_METRICS_INGESTION_AND_COMBINED552_FINAL_EVIDENCE_READINESS_REPORT.md |
| 222 | 249 | 7.58 KiB | .md | docs/stage3/runs/run_63_combined432_model_update_N24_N40_evidence_freeze_and_N12_N16_recovery_candidate_generation/RUN_63_COMBINED432_MODEL_UPDATE_N24_N40_EVIDENCE_FREEZE_AND_N12_N16_RECOVERY_CANDIDATE_GENERATION_REPORT.md |
| 214 | 241 | 9.58 KiB | .md | docs/stage3/runs/run_52_stricter_constrained_N24_N40_batch32_teacher_metrics_ingestion_and_combined328_ranking/RUN_52_STRICTER_CONSTRAINED_N24_N40_BATCH32_TEACHER_METRICS_INGESTION_AND_COMBINED328_RANKING_REPORT.md |
| 212 | 239 | 6.27 KiB | .md | docs/stage3/runs/run_58_combined392_model_update_N24_N40_evidence_freeze_and_N40_focused_candidate_generation/RUN_58_COMBINED392_MODEL_UPDATE_N24_N40_EVIDENCE_FREEZE_AND_N40_FOCUSED_CANDIDATE_GENERATION_REPORT.md |
| 210 | 237 | 4.65 KiB | .md | docs/stage3/runs/run_67_variable_N_recovery_anchor_batch48_teacher_metrics_ingestion_and_combined480_ranking/RUN_67_VARIABLE_N_RECOVERY_ANCHOR_BATCH48_TEACHER_METRICS_INGESTION_AND_COMBINED480_RANKING_REPORT.md |
| 204 | 231 | 4.61 KiB | .md | docs/stage3/runs/run_72_smallN_recovery_focused_batch40_teacher_metrics_ingestion_and_combined520_ranking/RUN_72_SMALLN_RECOVERY_FOCUSED_BATCH40_TEACHER_METRICS_INGESTION_AND_COMBINED520_RANKING_REPORT.md |
| 202 | 229 | 8.37 KiB | .md | docs/stage3/runs/run_42_native_N24_N40_focused_batch60_teacher_metrics_ingestion_and_combined264_ranking/RUN_42_NATIVE_N24_N40_FOCUSED_BATCH60_TEACHER_METRICS_INGESTION_AND_COMBINED264_RANKING_REPORT.md |
| 196 | 223 | 8.79 KiB | .md | docs/stage3/runs/run_47_constrained_N24_N40_batch32_teacher_metrics_ingestion_and_combined296_ranking/RUN_47_CONSTRAINED_N24_N40_BATCH32_TEACHER_METRICS_INGESTION_AND_COMBINED296_RANKING_REPORT.md |
| 196 | 223 | 7.06 KiB | .md | docs/stage3/runs/run_37_N32_informed_native_batch32_teacher_metrics_ingestion_and_combined204_ranking/RUN_37_N32_INFORMED_NATIVE_BATCH32_TEACHER_METRICS_INGESTION_AND_COMBINED204_RANKING_REPORT.md |
| 194 | 221 | 4.67 KiB | .md | docs/stage3/runs/run_62_custom_N40_focused_batch40_teacher_metrics_ingestion_and_combined432_ranking/RUN_62_CUSTOM_N40_FOCUSED_BATCH40_TEACHER_METRICS_INGESTION_AND_COMBINED432_RANKING_REPORT.md |
| 194 | 221 | 10.33 KiB | .md | docs/stage3/runs/run_57_calibrated_N24_N40_batch64_teacher_metrics_ingestion_and_combined392_ranking/RUN_57_CALIBRATED_N24_N40_BATCH64_TEACHER_METRICS_INGESTION_AND_COMBINED392_RANKING_REPORT.md |
| 192 | 219 | 3.31 KiB | .md | docs/stage3/runs/run_61_custom_N40_focused_calibrated_penalty_repair_batch40_odb_teacher_validation/RUN_61_CUSTOM_N40_FOCUSED_CALIBRATED_PENALTY_REPAIR_BATCH40_ODB_TEACHER_VALIDATION_REPORT.md |
| 189 | 216 | 22.55 KiB | .csv | outputs/stage3_run_59_run58_N40_focused_calibrated_penalty_repair_batch40_handoff_package/stage3_run59_N40_focused_calibrated_penalty_repair_batch40_future_cae_handoff_manifest_TEMPLATE.csv |
| 188 | 215 | 176.23 KiB | .json | outputs/stage3_run_77_final_smallN_diagnostic_batch32_teacher_metrics_ingestion_and_combined552_final_evidence_readiness/run76_final_smallN_diagnostic_batch32_teacher_dataset_enriched.json |
| 187 | 214 | 71.69 KiB | .csv | outputs/stage3_run_77_final_smallN_diagnostic_batch32_teacher_metrics_ingestion_and_combined552_final_evidence_readiness/run76_final_smallN_diagnostic_batch32_teacher_dataset_enriched.csv |
| 186 | 213 | 9.38 KiB | .txt | outputs/stage3_5_final_strategy_2d_score_lift_v01/n32_lexicographic_u2_peeq_surfacet_preview/N32_synthetic_from_N16_lexicographic_u2_peeq_surfacet_score_sorted_coordinate_order_array.txt |
| 185 | 212 | 19.60 KiB | .json | outputs/stage3_run_77_final_smallN_diagnostic_batch32_teacher_metrics_ingestion_and_combined552_final_evidence_readiness/run76_final_smallN_diagnostic_batch32_effectiveness_summary.json |
| 185 | 212 | 16.15 KiB | .csv | outputs/stage3_run_25_shortlist64_active_learning_cae_inp_generation/archived_failed_N40_B02_B03_B04_B05_solver_outputs_before_cool_initialInc_patch_20260613_085016/ARCHIVE_MANIFEST.csv |
| 185 | 212 | 1.05 KiB | .json | outputs/stage3_run_77_final_smallN_diagnostic_batch32_teacher_metrics_ingestion_and_combined552_final_evidence_readiness/stage3_final_maturity_and_evidence_freeze_readiness_summary.json |
| 184 | 211 | 6.04 KiB | .md | docs/stage3/runs/run_60_custom_N40_focused_calibrated_penalty_repair_batch40_cae_inp_generation/RUN_60_CUSTOM_N40_FOCUSED_CALIBRATED_PENALTY_REPAIR_BATCH40_CAE_INP_GENERATION_REPORT.md |
| 184 | 211 | 5.89 KiB | .md | docs/stage3/runs/run_29_combined172_surrogate_gnn_hybrid_policy_update_and_candidate_generation/RUN_29_COMBINED172_SURROGATE_GNN_HYBRID_POLICY_UPDATE_AND_CANDIDATE_GENERATION_REPORT.md |
| 184 | 211 | 14.72 KiB | .csv | outputs/stage3_5_final_strategy_2d_score_lift_v01/n32_lexicographic_u2_peeq_surfacet_preview/score_matrices/N32_synthetic_from_N16_lexicographic_u2_peeq_surfacet_snew_2d_unit_32x32.csv |
| 182 | 209 | 7.20 KiB | .md | docs/stage3/runs/run_38_combined204_and_combined204_plus_N32_model_update_candidate_generation/RUN_38_COMBINED204_AND_COMBINED204_PLUS_N32_MODEL_UPDATE_CANDIDATE_GENERATION_REPORT.md |
| 182 | 209 | 807 B | .csv | outputs/stage3_run_77_final_smallN_diagnostic_batch32_teacher_metrics_ingestion_and_combined552_final_evidence_readiness/stage3_final_maturity_and_evidence_freeze_readiness_audit.csv |
| 182 | 209 | 12.85 KiB | .txt | outputs/stage3_run_59_run58_N40_focused_calibrated_penalty_repair_batch40_handoff_package/stage3_run59_N40_focused_calibrated_penalty_repair_batch40_abqjobpilot_commands_TEMPLATE.txt |
| 182 | 209 | 71.69 KiB | .csv | outputs/stage3_run_77_final_smallN_diagnostic_batch32_teacher_metrics_ingestion_and_combined552_final_evidence_readiness/run76_final_smallN_diagnostic_batch32_ranked_within_batch.csv |
| 182 | 209 | 1.21 KiB | .csv | outputs/stage3_run_77_final_smallN_diagnostic_batch32_teacher_metrics_ingestion_and_combined552_final_evidence_readiness/run76_final_smallN_diagnostic_batch32_effectiveness_audit.csv |
| 181 | 208 | 42.34 KiB | .csv | outputs/stage3_run_59_run58_N40_focused_calibrated_penalty_repair_batch40_handoff_package/run59_custom_N40_focused_calibrated_penalty_repair_batch40_candidate_orders_PRE_HANDOFF.csv |
| 181 | 208 | 163.80 KiB | .json | outputs/stage3_run_52_stricter_constrained_N24_N40_batch32_teacher_metrics_ingestion_and_combined328_ranking/run51_stricter_constrained_N24_N40_batch32_teacher_dataset_enriched.json |
| 181 | 208 | 231.84 KiB | .json | outputs/stage3_5_final_strategy_2d_score_lift_v01/n32_lexicographic_u2_peeq_surfacet_preview/N32_synthetic_from_N16_lexicographic_u2_peeq_surfacet_score_sorted_coordinate_order.json |
| 181 | 208 | 809 B | .md | outputs/stage3_run_77_final_smallN_diagnostic_batch32_teacher_metrics_ingestion_and_combined552_final_evidence_readiness/stage3_final_maturity_and_evidence_freeze_readiness_audit.md |
| 180 | 207 | 7.94 KiB | .md | docs/stage3/runs/run_73_combined520_model_update_final_smallN_diagnostic_candidate_generation/RUN_73_COMBINED520_MODEL_UPDATE_FINAL_SMALLN_DIAGNOSTIC_CANDIDATE_GENERATION_REPORT.md |
| 180 | 207 | 30.38 KiB | .csv | outputs/stage3_5_final_strategy_2d_score_lift_v01/n32_lexicographic_u2_peeq_surfacet_preview/N32_synthetic_from_N16_lexicographic_u2_peeq_surfacet_score_sorted_coordinate_order.csv |
| 180 | 207 | 72.48 KiB | .csv | outputs/stage3_run_52_stricter_constrained_N24_N40_batch32_teacher_metrics_ingestion_and_combined328_ranking/run51_stricter_constrained_N24_N40_batch32_teacher_dataset_enriched.csv |
| 180 | 207 | 3.60 KiB | .csv | outputs/stage3_run_77_final_smallN_diagnostic_batch32_teacher_metrics_ingestion_and_combined552_final_evidence_readiness/run76_final_smallN_diagnostic_batch32_per_N_leaderboard.csv |
| 180 | 207 | 34.30 KiB | .csv | outputs/stage3_run_58_combined392_model_update_N24_N40_evidence_freeze_and_N40_focused_candidate_generation/run58_N40_focused_calibrated_penalty_repair_batch32_candidate_orders.csv |
| 178 | 205 | 5.60 KiB | .md | docs/stage3/runs/run_43_combined264_constrained_N24_N40_reward_balanced_candidate_generation/RUN_43_COMBINED264_CONSTRAINED_N24_N40_REWARD_BALANCED_CANDIDATE_GENERATION_REPORT.md |
| 177 | 204 | 237.33 KiB | .json | outputs/stage3_run_67_variable_N_recovery_anchor_batch48_teacher_metrics_ingestion_and_combined480_ranking/run66_variable_N_recovery_anchor_batch48_teacher_dataset_enriched.json |
| 177 | 204 | 676 B | .json | outputs/stage3_run_60_custom_N40_focused_calibrated_penalty_repair_batch40_cae_inp_generation/stage3_run60_custom_N40_focused_batch40_abqjobpilot_command_validation_summary.json |
| 176 | 203 | 102.49 KiB | .csv | outputs/stage3_run_67_variable_N_recovery_anchor_batch48_teacher_metrics_ingestion_and_combined480_ranking/run66_variable_N_recovery_anchor_batch48_teacher_dataset_enriched.csv |
| 176 | 203 | 4.52 KiB | .md | docs/stage3/runs/run_59_run58_N40_focused_calibrated_penalty_repair_batch40_handoff_package/RUN_59_RUN58_N40_FOCUSED_CALIBRATED_PENALTY_REPAIR_BATCH40_HANDOFF_PACKAGE_REPORT.md |
| 175 | 202 | 79.49 KiB | .csv | outputs/stage3_run_52_stricter_constrained_N24_N40_batch32_teacher_metrics_ingestion_and_combined328_ranking/run51_stricter_constrained_N24_N40_batch32_ranked_within_batch.csv |
| 174 | 201 | 21.17 KiB | .json | outputs/stage3_run_67_variable_N_recovery_anchor_batch48_teacher_metrics_ingestion_and_combined480_ranking/run66_variable_N_recovery_anchor_batch48_effectiveness_summary.json |
| 174 | 201 | 5.85 KiB | .json | outputs/stage3_run_77_final_smallN_diagnostic_batch32_teacher_metrics_ingestion_and_combined552_final_evidence_readiness/run76_prediction_audit_for_run73_batch32_summary.json |
| 174 | 201 | 51.18 KiB | .csv | outputs/stage3_run_63_combined432_model_update_N24_N40_evidence_freeze_and_N12_N16_recovery_candidate_generation/run63_variable_N_recovery_anchor_batch48_candidate_orders.csv |
| 173 | 200 | 1.71 KiB | .csv | outputs/stage3_run_52_stricter_constrained_N24_N40_batch32_teacher_metrics_ingestion_and_combined328_ranking/run51_stricter_constrained_N24_N40_batch32_per_N_leaderboard.csv |
| 171 | 198 | 12.72 KiB | .txt | outputs/stage3_run_60_custom_N40_focused_calibrated_penalty_repair_batch40_cae_inp_generation/stage3_run60_custom_N40_focused_batch40_abqjobpilot_commands_READY_TO_RUN.txt |
| 171 | 198 | 664 B | .json | outputs/stage3_run_50_stricter_constrained_N24_N40_batch32_cae_inp_generation/stage3_run50_stricter_constrained_N24_N40_batch32_abqjobpilot_command_validation_summary.json |
| 171 | 198 | 196.64 KiB | .json | outputs/stage3_run_72_smallN_recovery_focused_batch40_teacher_metrics_ingestion_and_combined520_ranking/run71_smallN_recovery_focused_batch40_teacher_dataset_enriched.json |
| 171 | 198 | 34.77 KiB | .csv | outputs/stage3_run_49_run48_stricter_constrained_N24_N40_batch32_handoff_package/stage3_run49_stricter_constrained_N24_N40_batch32_future_cae_handoff_manifest_TEMPLATE.csv |
| 171 | 198 | 109.93 KiB | .csv | outputs/stage3_run_67_variable_N_recovery_anchor_batch48_teacher_metrics_ingestion_and_combined480_ranking/run66_variable_N_recovery_anchor_batch48_ranked_within_batch.csv |
| 171 | 198 | 8.55 KiB | .json | outputs/stage3_run_67_variable_N_recovery_anchor_batch48_teacher_metrics_ingestion_and_combined480_ranking/full_variable_N_updated_maturity_and_claim_boundary_summary.json |
| 171 | 198 | 1.25 KiB | .csv | outputs/stage3_run_67_variable_N_recovery_anchor_batch48_teacher_metrics_ingestion_and_combined480_ranking/run66_variable_N_recovery_anchor_batch48_effectiveness_audit.csv |
| 170 | 197 | 2.18 KiB | .json | outputs/stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40/candidate_generation_v03/selected_batch32/scan_orders/scan_order_PPOV03_N40_B13_diverse_upper_quartile.json |
| 170 | 197 | 4.00 KiB | .md | outputs/stage3_5_final_strategy_2d_score_lift_v01/n32_lexicographic_u2_peeq_surfacet_preview/reports/N32_SYNTHETIC_FROM_N16_LEXICOGRAPHIC_U2_PEEQ_SURFACET_SCAN_PREVIEW.md |
| 170 | 197 | 2.18 KiB | .json | outputs/stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40/candidate_generation_v03/selected_batch32/scan_orders/scan_order_PPOV03_N40_B14_diverse_upper_quartile.json |
| 170 | 197 | 6.12 KiB | .md | docs/stage3/runs/run_33_combined172_plus_N32_balanced_surrogate_gnn_candidate_generation/RUN_33_COMBINED172_PLUS_N32_BALANCED_SURROGATE_GNN_CANDIDATE_GENERATION_REPORT.md |
| 170 | 197 | 1.94 KiB | .json | outputs/stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40/candidate_generation_v03/selected_batch32/scan_orders/scan_order_PPOV03_N24_B14_diverse_upper_quartile.json |
| 170 | 197 | 1.94 KiB | .json | outputs/stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40/candidate_generation_v03/selected_batch32/scan_orders/scan_order_PPOV03_N24_B13_diverse_upper_quartile.json |
| 170 | 197 | 13.03 KiB | .json | outputs/stage3_run_52_stricter_constrained_N24_N40_batch32_teacher_metrics_ingestion_and_combined328_ranking/run51_stricter_constrained_batch32_effectiveness_summary.json |
| 170 | 197 | 83.51 KiB | .csv | outputs/stage3_run_72_smallN_recovery_focused_batch40_teacher_metrics_ingestion_and_combined520_ranking/run71_smallN_recovery_focused_batch40_teacher_dataset_enriched.csv |
| 169 | 196 | 3.60 KiB | .csv | outputs/stage3_run_67_variable_N_recovery_anchor_batch48_teacher_metrics_ingestion_and_combined480_ranking/run66_variable_N_recovery_anchor_batch48_per_N_leaderboard.csv |
| 169 | 196 | 47.86 KiB | .csv | outputs/stage3_run_58_combined392_model_update_N24_N40_evidence_freeze_and_N40_focused_candidate_generation/run58_variable_N_recovery_anchor_batch48_candidate_orders.csv |
| 169 | 196 | 273.42 KiB | .json | outputs/stage3_run_42_native_N24_N40_focused_batch60_teacher_metrics_ingestion_and_combined264_ranking/run41_native_N24_N40_focused_batch60_teacher_dataset_enriched.json |
| 169 | 196 | 509 B | .csv | outputs/stage3_5_final_strategy_2d_score_lift_v01/n32_lexicographic_u2_peeq_surfacet_preview/score_vectors/N32_synthetic_from_N16_lexicographic_u2_peeq_surfacet_s_1d.csv |
| 169 | 196 | 43.43 KiB | .csv | outputs/stage3_run_59_run58_N40_focused_calibrated_penalty_repair_batch40_handoff_package/stage3_run59_N40_focused_calibrated_penalty_repair_batch40_candidate_orders.csv |
| 169 | 196 | 36.70 KiB | .csv | outputs/stage3_run_63_combined432_model_update_N24_N40_evidence_freeze_and_N12_N16_recovery_candidate_generation/run63_optional_N40_followup_batch32_candidate_orders.csv |
| 168 | 195 | 8.17 KiB | .json | outputs/stage3_run_72_smallN_recovery_focused_batch40_teacher_metrics_ingestion_and_combined520_ranking/full_variable_N_updated_maturity_and_claim_boundary_summary.json |
| 168 | 195 | 20.46 KiB | .json | outputs/stage3_run_72_smallN_recovery_focused_batch40_teacher_metrics_ingestion_and_combined520_ranking/run71_smallN_recovery_focused_batch40_effectiveness_summary.json |
| 168 | 195 | 263 B | .csv | outputs/stage3_run_67_variable_N_recovery_anchor_batch48_teacher_metrics_ingestion_and_combined480_ranking/full_variable_N_updated_maturity_and_claim_boundary_audit.csv |
| 168 | 195 | 122.03 KiB | .csv | outputs/stage3_run_42_native_N24_N40_focused_batch60_teacher_metrics_ingestion_and_combined264_ranking/run41_native_N24_N40_focused_batch60_teacher_dataset_enriched.csv |
| 168 | 195 | 16.66 KiB | .csv | outputs/stage3_run_60_custom_N40_focused_calibrated_penalty_repair_batch40_cae_inp_generation/stage3_run60_custom_N40_focused_batch40_abqjobpilot_command_validation.csv |
| 167 | 194 | 27.49 KiB | .csv | outputs/stage3_run_64_run63_variable_N_recovery_anchor_batch48_handoff_package/stage3_run64_variable_N_recovery_anchor_batch48_future_cae_handoff_manifest_TEMPLATE.csv |
| 167 | 194 | 615 B | .md | outputs/stage3_run_67_variable_N_recovery_anchor_batch48_teacher_metrics_ingestion_and_combined480_ranking/full_variable_N_updated_maturity_and_claim_boundary_audit.md |
| 167 | 194 | 1.44 KiB | .json | outputs/stage3_ppo_rl_lam_fea_addendum_v01/stageE_teacher_validation_handoff/reports/archived_initial_dry_run/stageE_ppo_batch32_generator_failure_initial_attempt.json |
| 167 | 194 | 2.04 KiB | .csv | outputs/stage3_run_52_stricter_constrained_N24_N40_batch32_teacher_metrics_ingestion_and_combined328_ranking/run51_stricter_constrained_batch32_effectiveness_audit.csv |
| 167 | 194 | 685 B | .json | outputs/stage3_run_65_variable_N_recovery_anchor_batch48_cae_inp_generation/stage3_run65_variable_N_recovery_anchor_batch48_abqjobpilot_command_validation_summary.json |
| 166 | 193 | 3.94 KiB | .md | CHATGPT_PROJECT_UPLOAD/RL_LAM_ScanOpt_PPO_Final_320_Evidence_Package_v01/03_STAGE_HISTORY_SUMMARIES/PPO_FINAL_EXPANSION_STAGEV_ODB_TEACHER_METRIC_EXTRACTION_REPORT.md |
| 166 | 193 | 29.48 KiB | .csv | outputs/stage3_run_38_combined204_and_combined204_plus_N32_model_update_candidate_generation/run38_native_batch32_U2_exploitation_reward_balanced_candidate_orders.csv |
| 166 | 193 | 7.65 KiB | .json | outputs/stage3_run_61_custom_N40_focused_calibrated_penalty_repair_batch40_odb_teacher_validation/run61_custom_N40_focused_batch40_odb_teacher_validation_summary.json |
| 166 | 193 | 541 B | .json | outputs/stage3_run_60_custom_N40_focused_calibrated_penalty_repair_batch40_cae_inp_generation/stage3_run60_custom_N40_focused_batch40_generated_inp_check_summary.json |
| 166 | 193 | 10.37 KiB | .json | outputs/stage3_run_63_combined432_model_update_N24_N40_evidence_freeze_and_N12_N16_recovery_candidate_generation/n24_n40_final_active_learning_rl_evidence_freeze.json |
| 165 | 192 | 1.25 KiB | .csv | outputs/stage3_run_72_smallN_recovery_focused_batch40_teacher_metrics_ingestion_and_combined520_ranking/run71_smallN_recovery_focused_batch40_effectiveness_audit.csv |
| 165 | 192 | 2.17 KiB | .json | outputs/stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40/candidate_generation_v03/selected_batch32/scan_orders/scan_order_PPOV03_N40_B10_lex_primary_novel.json |
| 165 | 192 | 9.08 KiB | .txt | outputs/stage3_run_50_stricter_constrained_N24_N40_batch32_cae_inp_generation/stage3_run50_stricter_constrained_N24_N40_batch32_abqjobpilot_commands_READY_TO_RUN.txt |
| 165 | 192 | 951 B | .csv | outputs/stage3_run_63_combined432_model_update_N24_N40_evidence_freeze_and_N12_N16_recovery_candidate_generation/n24_n40_final_active_learning_rl_evidence_freeze.csv |
| 165 | 192 | 89.32 KiB | .csv | outputs/stage3_run_72_smallN_recovery_focused_batch40_teacher_metrics_ingestion_and_combined520_ranking/run71_smallN_recovery_focused_batch40_ranked_within_batch.csv |
| 165 | 192 | 1.93 KiB | .json | outputs/stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40/candidate_generation_v03/selected_batch32/scan_orders/scan_order_PPOV03_N24_B11_lex_primary_novel.json |
| 165 | 192 | 1.93 KiB | .json | outputs/stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40/candidate_generation_v03/selected_batch32/scan_orders/scan_order_PPOV03_N24_B10_lex_primary_novel.json |
| 165 | 192 | 67.04 KiB | .csv | outputs/stage3_run_58_combined392_model_update_N24_N40_evidence_freeze_and_N40_focused_candidate_generation/run58_N40_focused_calibrated_batch64_candidate_orders.csv |
| 165 | 192 | 1.93 KiB | .json | outputs/stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40/candidate_generation_v03/selected_batch32/scan_orders/scan_order_PPOV03_N24_B12_lex_primary_novel.json |
| 165 | 192 | 263 B | .csv | outputs/stage3_run_72_smallN_recovery_focused_batch40_teacher_metrics_ingestion_and_combined520_ranking/full_variable_N_updated_maturity_and_claim_boundary_audit.csv |
| 165 | 192 | 2.17 KiB | .json | outputs/stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40/candidate_generation_v03/selected_batch32/scan_orders/scan_order_PPOV03_N40_B12_lex_primary_novel.json |
| 165 | 192 | 2.17 KiB | .json | outputs/stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40/candidate_generation_v03/selected_batch32/scan_orders/scan_order_PPOV03_N40_B11_lex_primary_novel.json |
| 165 | 192 | 2.21 KiB | .csv | outputs/stage3_run_77_final_smallN_diagnostic_batch32_teacher_metrics_ingestion_and_combined552_final_evidence_readiness/run76_prediction_audit_for_run73_batch32.csv |
| 164 | 191 | 7.68 KiB | .md | docs/stage3/runs/run_68_combined480_model_update_smallN_recovery_candidate_generation/RUN_68_COMBINED480_MODEL_UPDATE_SMALLN_RECOVERY_CANDIDATE_GENERATION_REPORT.md |
| 164 | 191 | 615 B | .md | outputs/stage3_run_72_smallN_recovery_focused_batch40_teacher_metrics_ingestion_and_combined520_ranking/full_variable_N_updated_maturity_and_claim_boundary_audit.md |
| 164 | 191 | 9.23 KiB | .txt | outputs/stage3_run_49_run48_stricter_constrained_N24_N40_batch32_handoff_package/stage3_run49_stricter_constrained_N24_N40_batch32_abqjobpilot_commands_TEMPLATE.txt |
| 164 | 191 | 1.56 KiB | .md | outputs/stage3_run_63_combined432_model_update_N24_N40_evidence_freeze_and_N12_N16_recovery_candidate_generation/n24_n40_final_active_learning_rl_evidence_freeze.md |
| 164 | 191 | 7.38 KiB | .md | docs/stage3/runs/run_28_shortlist64_teacher_metrics_ingestion_and_combined172_ranking/RUN_28_SHORTLIST64_TEACHER_METRICS_INGESTION_AND_COMBINED172_RANKING_REPORT.md |

## Inventory

Detailed non-Abaqus long-path inventory: docs/repository/git_long_path_inventory.csv

## Line Ending Context

- core.autocrlf: file:E:/Program Files/Git/etc/gitconfig	true
- core.eol: 
- .gitattributes present: False

LF will be replaced by CRLF is not the cause of the Filename-too-long failure. No line-ending conversion was performed.