# Run 04 True Variable-N CAE Template Parameterisation Audit Report

## Executive Verdict

WARNING_TRUE_VARIABLE_N_CAE_TEMPLATE_ADAPTER_REQUIRED

## Final Feasibility Category

B_TRUE_VARIABLE_N_ADAPTER_REQUIRED_BUT_FEASIBLE

## What Was Audited

- Source roots scanned: `['E:\\Projects\\RL-LAM-ScanOpt', 'D:\\Projects\\RL-LAM-ScanOpt']`
- Files scanned: `3587`
- Candidate CAE generation files found: `1042`
- Most relevant files:

- `LDED_2D_CAE_Framework\abaqus_scripts\build_masked_v02_method_c_validation_caes_and_inputs.py`
- `LDED_2D_CAE_Framework\abaqus_scripts\build_masked_UST_probe40_v01_from_handoff.py`
- `LDED_2D_CAE_Framework\abaqus_scripts\build_32track_masked_v01_caes_and_inputs.py`
- `LDED_2D_CAE_Framework\abaqus_scripts\build_masked_v02_method_c_remaining24_unique_inps.py`
- `LDED_2D_CAE_Framework\abaqus_scripts\build_32track_strategy_caes_from_base.py`
- `LDED_2D_CAE_Framework\abaqus_scripts\build_full20_2d_strategy_caes_and_inputs.py`
- `LDED_2D_CAE_Framework\abaqus_scripts\build_stage1_full10_2d_strategy_caes.py`
- `LDED_2D_CAE_Framework\cae_models\32track_learned_hybrid_top5_v01\build_learned_hybrid_top5_fulltrack_inps.py`
- `LDED_2D_CAE_Framework\cae_models\32track_random20_v01\build_random20_fulltrack_inps.py`
- `LDED_2D_CAE_Framework\cae_models\64track_full\prototype_5strategy_v01\build_64track_prototype_5strategy.py`
- `LDED_2D_CAE_Framework\abaqus_scripts\generate_method_c_batch_cae_inp_v01.py`
- `LDED_2D_CAE_Framework\abaqus_scripts\complete_2d_32track_scan_steps_with_final_cooling.py`

## Guardrails

- No Abaqus jobs.
- No datacheck.
- No ODB opened.
- No CAE/INP/JNL generated.
- No model training.
- No teacher validation.
- D-drive source was not modified.

## Fixed-32 Hardcode Audit

The audit found fixed-32 assumptions in these categories:

- `32track_name`: `698`
- `domain_width_literal`: `1`
- `fixed_arrays_32`: `28`
- `fixed_track_positions`: `42`
- `postprocess_32`: `10`
- `range32`: `8`
- `scan_order_len32`: `2`
- `track_count`: `11`

The detailed hardcode table is capped at 800 rows to keep run_04 outputs small and GitHub-safe.

Key risk areas are track count, `range(32)` or 0..31 checks, scan-order length assumptions, 32track names, output paths, and postprocessing tables that carry fixed-32 naming or label contracts.

## True Variable-N Geometry Design

Preferred parameterisation for Option A:

- `track_count = N`, with N in `(16, 24, 32, 40)`.
- `track_pitch` remains physically constant unless the existing model documents another convention.
- `heat_source_width` is explicit and independent from N.
- `domain_width = margin_left + margin_right + heat_source_width + (N - 1) * track_pitch`.
- `domain_height` and thickness should remain physically consistent unless a documented coupon-scaling rule is introduced.
- `track_positions` are generated from N, pitch, and margins.
- `scan_order` must satisfy `len(scan_order) == N` and `set(scan_order) == set(range(N))`.
- `step_count` is derived from scan_order length N.
- `batch_name`, `strategy_name`, and `output_dir` encode N and `true_variable_n_geometry`.

## Fixed Pitch vs Fixed Domain Width

Recommendation: use fixed physical track pitch and N-dependent domain width. Fixed pitch means N=40 is physically wider than N=32, while fixed domain width would change track spacing with N. For scan-order principle transfer, fixed pitch is cleaner if the process track spacing is physically fixed.

## Required Template Changes

- Introduce a geometry/config object carrying N, pitch, heat source width, margins, derived domain width, mesh controls, and naming tokens.
- Replace `range(32)` and fixed 0..31 checks with `range(track_count)`.
- Generate track positions and sets/surfaces from N.
- Derive heat-load step loops and activation/cooling names from scan_order length.
- Make job/batch/output naming N-aware and distinguish true-N from masked/subset-N.
- Parameterize postprocessing and teacher label schemas by N.

## Risks

- N=40 may require larger geometry and mesh.
- N=16 may be less directly comparable due to a smaller domain.
- The full-32 U2 absolute guard cannot be shared across N.
- Mesh density must remain physically consistent.
- Heat source magnitude must remain physically consistent.
- Postprocessing must not assume 32 tracks.
- Computational cost may increase for N=40.

## Stop Point

The next step after run_04 is `run_05_true_variable_n_cae_generator_adapter_dryrun`.

## Claim Boundary

This run does not prove variable-N generalisation. It does not generate models. It does not validate physics. It only prepares the true variable-N teacher environment design.

## Outputs

- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_04_true_variable_n_cae_template_parameterisation_audit\cae_generation_source_inventory.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_04_true_variable_n_cae_template_parameterisation_audit\fixed32_hardcode_audit.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_04_true_variable_n_cae_template_parameterisation_audit\true_variable_n_parameterisation_plan.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_04_true_variable_n_cae_template_parameterisation_audit\required_template_changes.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_04_true_variable_n_cae_template_parameterisation_audit\true_variable_n_design_decisions.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_04_true_variable_n_cae_template_parameterisation_audit\variable_n_cae_feasibility_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\artifacts\manifests\stage3_run_04_manifest.json`

## Final Verdict

WARNING_TRUE_VARIABLE_N_CAE_TEMPLATE_ADAPTER_REQUIRED
