# Stage 3 Run60 - Custom N40-Focused Calibrated Penalty-Repair Batch40 CAE/INP Generation

## Purpose

Run60 generated CAE/INP/JNL handoff files for the selected Run59 custom N40-focused calibrated penalty-repair batch40. This run is model-generation only.

## Inputs

- Candidate orders: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_59_run58_N40_focused_calibrated_penalty_repair_batch40_handoff_package\stage3_run59_N40_focused_calibrated_penalty_repair_batch40_candidate_orders.csv`
- Scan orders: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_59_run58_N40_focused_calibrated_penalty_repair_batch40_handoff_package\scan_orders`
- N24 base CAE: `E:\Projects\RL-LAM-ScanOpt\cae_model\24track_full\sanity_base\24track_sanity_base.cae`
- N40 base CAE: `E:\Projects\RL-LAM-ScanOpt\cae_model\40track_full\sanity_base\40track_sanity_base.cae`
- Generated case root: `E:\Projects\RL-LAM-ScanOpt\cae_model\stage3_run59_N40_focused_calibrated_penalty_repair_batch40_v01`

## Selected Batch

- Selected batch: Run59 custom Run58-derived N40-focused calibrated penalty-repair batch40
- Batch name: `stage3_run59_N40_focused_calibrated_penalty_repair_batch40_v01`
- Expected cases: 40
- Per-N counts: N24 = 16, N40 = 24
- No N12, N16, or N32 cases were generated.

## Preflight Validation

- Verdict: `PASS_RUN60_CUSTOM_N40_FOCUSED_BATCH40_INPUT_READY_FOR_CAE_GENERATION`
- Summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_60_custom_N40_focused_calibrated_penalty_repair_batch40_cae_inp_generation\run60_preflight_summary.json`
- Cases CSV: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_60_custom_N40_focused_calibrated_penalty_repair_batch40_cae_inp_generation\run60_preflight_cases.csv`

## Future CAE Root Safety Audit

- Verdict: `PASS_RUN60_FUTURE_CAE_ROOT_SAFE`
- Case root existed before generation: false
- Existing solver-output count before generation: 0
- Summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_60_custom_N40_focused_calibrated_penalty_repair_batch40_cae_inp_generation\run60_future_cae_root_safety_audit.json`

## Base Mesh Audit

- Overall verdict: `PASS_RUN60_BASE_MESH_AUDIT_READY`
- N24: `PASS_BASE_MESH_READY`, nodes = 735, elements = 624, heat sets = 24
- N40: `PASS_BASE_MESH_READY`, nodes = 1183, elements = 1008, heat sets = 40
- Summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_60_custom_N40_focused_calibrated_penalty_repair_batch40_cae_inp_generation\run60_base_mesh_audit_summary.json`

## CAE/INP Generation

- Abaqus noGUI command executed:
  `abaqus cae noGUI="E:\Projects\RL-LAM-ScanOpt\scripts\stage3\run_60_generate_custom_N40_focused_batch40_from_sanity_base_nogui.py"`
- Generated CAE count: 40
- Generated INP count: 40
- Generated JNL count: 40
- Generated case logs: 40
- Generation manifest CSV: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_60_custom_N40_focused_calibrated_penalty_repair_batch40_cae_inp_generation\stage3_run60_custom_N40_focused_batch40_cae_generation_manifest.csv`
- Generation manifest JSON: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_60_custom_N40_focused_calibrated_penalty_repair_batch40_cae_inp_generation\stage3_run60_custom_N40_focused_batch40_cae_generation_manifest.json`

## Generated Counts By N

- N24 CAE/INP: 16 / 16
- N40 CAE/INP: 24 / 24
- N12/N16/N32 generated cases: 0

## INP Mesh And Final-Cooling Validation

- Verdict: `PASS_RUN60_CUSTOM_N40_FOCUSED_BATCH40_40_INPS_READY_FOR_USER_REVIEW`
- Checked INPs: 40
- Passed INPs: 40
- Solver output count under generated case root: 0
- Generated INP check CSV: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_60_custom_N40_focused_calibrated_penalty_repair_batch40_cae_inp_generation\stage3_run60_custom_N40_focused_batch40_generated_inp_check.csv`
- Generated INP check summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_60_custom_N40_focused_calibrated_penalty_repair_batch40_cae_inp_generation\stage3_run60_custom_N40_focused_batch40_generated_inp_check_summary.json`

## Final Cooling Controls

- `step_final_cooling` duration: 1200.0 seconds
- `step_final_cooling` initial increment: 0.01
- `step_final_cooling` maximum increment size: 60.0
- `maxInc = 60.0` is maximum increment size, not `maxNumInc = 60`.
- For generated N40 INPs, all `step_cool_XX` initial increments were verified as 0.001.

## abqjobpilot Command-File Validation

- Verdict: `PASS_RUN60_ABQJOBPILOT_COMMAND_FILE_VALID`
- Command count: 40
- Per-N command counts: N24 = 16, N40 = 24
- Ready-to-run command file: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_60_custom_N40_focused_calibrated_penalty_repair_batch40_cae_inp_generation\stage3_run60_custom_N40_focused_batch40_abqjobpilot_commands_READY_TO_RUN.txt`
- Validation summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_60_custom_N40_focused_calibrated_penalty_repair_batch40_cae_inp_generation\stage3_run60_custom_N40_focused_batch40_abqjobpilot_command_validation_summary.json`
- The custom abqjobpilot command file is ready for user-controlled enqueue, but was not executed by Codex.

## Claim Boundary

Run60 generated CAE/INP/JNL handoff files only. The generated candidates are not teacher-validated, no Abaqus solver was run, no datacheck was run, no ODB was opened, and no abqjobpilot/enqueue command was executed by Codex.

Claim boundary files:
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_60_custom_N40_focused_calibrated_penalty_repair_batch40_cae_inp_generation\run60_claim_boundary.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_60_custom_N40_focused_calibrated_penalty_repair_batch40_cae_inp_generation\run60_claim_boundary.json`

## Output Files

- Output directory: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_60_custom_N40_focused_calibrated_penalty_repair_batch40_cae_inp_generation`
- Main generation manifest: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_60_custom_N40_focused_calibrated_penalty_repair_batch40_cae_inp_generation\stage3_run60_custom_N40_focused_batch40_cae_generation_manifest.csv`
- Run manifest: `E:\Projects\RL-LAM-ScanOpt\artifacts\manifests\stage3_run_60_manifest.json`

## Exact Next User Action

Manually review the READY_TO_RUN command file and, if approved, enqueue the 40 jobs with the user's custom abqjobpilot workflow.
