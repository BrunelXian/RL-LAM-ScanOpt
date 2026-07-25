# Stage 3 Run65 - Variable-N Recovery Anchor Batch48 CAE/INP Generation

## Purpose

Run65 generated CAE/INP/JNL handoff files for the selected Run64 variable-N recovery anchor batch48. This run is model-generation only.

## Inputs

- Candidate orders: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_64_run63_variable_N_recovery_anchor_batch48_handoff_package\stage3_run64_variable_N_recovery_anchor_batch48_candidate_orders.csv`
- Scan orders: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_64_run63_variable_N_recovery_anchor_batch48_handoff_package\scan_orders`
- N12 base CAE: `E:\Projects\RL-LAM-ScanOpt\cae_model\12track_full\sanity_base\12track_sanity_base.cae`
- N16 base CAE: `E:\Projects\RL-LAM-ScanOpt\cae_model\16track_full\sanity_base\16track_sanity_base.cae`
- N24 base CAE: `E:\Projects\RL-LAM-ScanOpt\cae_model\24track_full\sanity_base\24track_sanity_base.cae`
- N40 base CAE: `E:\Projects\RL-LAM-ScanOpt\cae_model\40track_full\sanity_base\40track_sanity_base.cae`
- Generated case root: `E:\Projects\RL-LAM-ScanOpt\cae_model\stage3_run64_variable_N_recovery_anchor_batch48_v01`

## Selected Batch

- Selected batch: Run64 handoff package for Run63 Option A variable-N recovery anchor batch48
- Batch name: `stage3_run64_variable_N_recovery_anchor_batch48_v01`
- Expected cases: 48
- Per-N counts: N12 = 12, N16 = 12, N24 = 8, N40 = 16
- No N32 cases were generated.

## Preflight Validation

- Verdict: `PASS_RUN65_VARIABLE_N_RECOVERY_ANCHOR_BATCH48_INPUT_READY_FOR_CAE_GENERATION`
- Summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_65_variable_N_recovery_anchor_batch48_cae_inp_generation\run65_preflight_summary.json`
- Cases CSV: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_65_variable_N_recovery_anchor_batch48_cae_inp_generation\run65_preflight_cases.csv`

## Future CAE Root Safety Audit

- Verdict: `PASS_RUN65_FUTURE_CAE_ROOT_SAFE`
- Case root existed before generation: false
- Existing solver-output count before generation: 0
- Summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_65_variable_N_recovery_anchor_batch48_cae_inp_generation\run65_future_cae_root_safety_audit.json`

## Base Mesh Audit

- Overall verdict: `PASS_RUN65_BASE_MESH_AUDIT_READY`
- N12: `PASS_BASE_MESH_READY`, nodes = 399, elements = 336, heat sets = 12
- N16: `PASS_BASE_MESH_READY`, nodes = 511, elements = 432, heat sets = 16
- N24: `PASS_BASE_MESH_READY`, nodes = 735, elements = 624, heat sets = 24
- N40: `PASS_BASE_MESH_READY`, nodes = 1183, elements = 1008, heat sets = 40
- Summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_65_variable_N_recovery_anchor_batch48_cae_inp_generation\run65_base_mesh_audit_summary.json`

## CAE/INP Generation

- Abaqus noGUI command executed:
  `abaqus cae noGUI="E:\Projects\RL-LAM-ScanOpt\scripts\stage3\run_65_generate_variable_N_recovery_anchor_batch48_from_sanity_base_nogui.py"`
- Generated CAE count: 48
- Generated INP count: 48
- Generated JNL count: 48
- Generated case logs: 48
- Generation manifest CSV: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_65_variable_N_recovery_anchor_batch48_cae_inp_generation\stage3_run65_variable_N_recovery_anchor_batch48_cae_generation_manifest.csv`
- Generation manifest JSON: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_65_variable_N_recovery_anchor_batch48_cae_inp_generation\stage3_run65_variable_N_recovery_anchor_batch48_cae_generation_manifest.json`

## Generated Counts By N

- N12 CAE/INP: 12 / 12
- N16 CAE/INP: 12 / 12
- N24 CAE/INP: 8 / 8
- N40 CAE/INP: 16 / 16
- N32 generated cases: 0

## INP Mesh And Final-Cooling Validation

- Verdict: `PASS_RUN65_VARIABLE_N_RECOVERY_ANCHOR_BATCH48_48_INPS_READY_FOR_USER_REVIEW`
- Checked INPs: 48
- Passed INPs: 48
- Solver output count under generated case root: 0
- Generated INP check CSV: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_65_variable_N_recovery_anchor_batch48_cae_inp_generation\stage3_run65_variable_N_recovery_anchor_batch48_generated_inp_check.csv`
- Generated INP check summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_65_variable_N_recovery_anchor_batch48_cae_inp_generation\stage3_run65_variable_N_recovery_anchor_batch48_generated_inp_check_summary.json`

## Final Cooling Controls

- `step_final_cooling` duration: 1200.0 seconds
- `step_final_cooling` initial increment: 0.01
- `step_final_cooling` maximum increment size: 60.0
- `maxInc = 60.0` is maximum increment size, not `maxNumInc = 60`.
- For generated N40 INPs, all `step_cool_XX` initial increments were verified as 0.001.

## abqjobpilot Command-File Validation

- Verdict: `PASS_RUN65_ABQJOBPILOT_COMMAND_FILE_VALID`
- Command count: 48
- Per-N command counts: N12 = 12, N16 = 12, N24 = 8, N40 = 16
- Ready-to-run command file: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_65_variable_N_recovery_anchor_batch48_cae_inp_generation\stage3_run65_variable_N_recovery_anchor_batch48_abqjobpilot_commands_READY_TO_RUN.txt`
- Validation summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_65_variable_N_recovery_anchor_batch48_cae_inp_generation\stage3_run65_variable_N_recovery_anchor_batch48_abqjobpilot_command_validation_summary.json`
- The custom abqjobpilot command file is ready for user-controlled enqueue, but was not executed by Codex.

## Claim Boundary

Run65 generated CAE/INP/JNL handoff files only. The generated candidates are not teacher-validated, no Abaqus solver was run, no datacheck was run, no ODB was opened, and no abqjobpilot/enqueue command was executed by Codex.

Claim boundary files:
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_65_variable_N_recovery_anchor_batch48_cae_inp_generation\run65_claim_boundary.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_65_variable_N_recovery_anchor_batch48_cae_inp_generation\run65_claim_boundary.json`

## Output Files

- Output directory: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_65_variable_N_recovery_anchor_batch48_cae_inp_generation`
- Main generation manifest: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_65_variable_N_recovery_anchor_batch48_cae_inp_generation\stage3_run65_variable_N_recovery_anchor_batch48_cae_generation_manifest.csv`
- Run manifest: `E:\Projects\RL-LAM-ScanOpt\artifacts\manifests\stage3_run_65_manifest.json`

## Exact Next User Action

Manually review the READY_TO_RUN command file and, if approved, enqueue the 48 jobs with the user's custom abqjobpilot workflow.
