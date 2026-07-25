# Stage 3 Run50 - Stricter Constrained N24/N40 Batch32 CAE/INP Generation

## Purpose

Run50 generated CAE/INP/JNL handoff files for the selected Run49 stricter constrained N24/N40 batch32. This run is model-generation only.

## Inputs

- Candidate orders: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_49_run48_stricter_constrained_N24_N40_batch32_handoff_package\stage3_run49_stricter_constrained_N24_N40_batch32_candidate_orders.csv`
- Scan orders: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_49_run48_stricter_constrained_N24_N40_batch32_handoff_package\scan_orders`
- N24 base CAE: `E:\Projects\RL-LAM-ScanOpt\cae_model\24track_full\sanity_base\24track_sanity_base.cae`
- N40 base CAE: `E:\Projects\RL-LAM-ScanOpt\cae_model\40track_full\sanity_base\40track_sanity_base.cae`
- Generated case root: `E:\Projects\RL-LAM-ScanOpt\cae_model\stage3_run49_stricter_constrained_N24_N40_batch32_v01`

## Selected Batch

- Selected batch: Run48 Option A, `run48_stricter_constrained_N24_N40_batch32`
- Batch name: `stage3_run49_stricter_constrained_N24_N40_batch32_v01`
- Expected cases: 32
- Per-N counts: N24 = 16, N40 = 16
- No N12, N16, or N32 cases were generated.

## Preflight Validation

- Verdict: `PASS_RUN50_STRICTER_CONSTRAINED_N24_N40_BATCH32_INPUT_READY_FOR_CAE_GENERATION`
- Summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_50_stricter_constrained_N24_N40_batch32_cae_inp_generation\run50_preflight_summary.json`
- Cases CSV: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_50_stricter_constrained_N24_N40_batch32_cae_inp_generation\run50_preflight_cases.csv`

## Future CAE Root Safety Audit

- Verdict: `PASS_RUN50_FUTURE_CAE_ROOT_SAFE`
- Case root existed before generation: false
- Existing solver-output count before generation: 0
- Summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_50_stricter_constrained_N24_N40_batch32_cae_inp_generation\run50_future_cae_root_safety_audit.json`

## Base Mesh Audit

- Overall verdict: `PASS_RUN50_BASE_MESH_AUDIT_READY`
- N24: `PASS_BASE_MESH_READY`, nodes = 735, elements = 624, heat sets = 24
- N40: `PASS_BASE_MESH_READY`, nodes = 1183, elements = 1008, heat sets = 40
- Summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_50_stricter_constrained_N24_N40_batch32_cae_inp_generation\run50_base_mesh_audit_summary.json`

## CAE/INP Generation

- Abaqus noGUI command executed:
  `abaqus cae noGUI="E:\Projects\RL-LAM-ScanOpt\scripts\stage3\run_50_generate_stricter_constrained_N24_N40_batch32_from_sanity_base_nogui.py"`
- Generated CAE count: 32
- Generated INP count: 32
- Generated JNL count: 32
- Generated case logs: 32
- Generation manifest CSV: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_50_stricter_constrained_N24_N40_batch32_cae_inp_generation\stage3_run50_stricter_constrained_N24_N40_batch32_cae_generation_manifest.csv`
- Generation manifest JSON: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_50_stricter_constrained_N24_N40_batch32_cae_inp_generation\stage3_run50_stricter_constrained_N24_N40_batch32_cae_generation_manifest.json`

## Generated Counts By N

- N24 CAE/INP: 16 / 16
- N40 CAE/INP: 16 / 16
- N12/N16/N32 generated cases: 0

## INP Mesh And Final-Cooling Validation

- Verdict: `PASS_RUN50_STRICTER_CONSTRAINED_N24_N40_BATCH32_32_INPS_READY_FOR_USER_REVIEW`
- Checked INPs: 32
- Passed INPs: 32
- Solver output count under generated case root: 0
- Generated INP check CSV: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_50_stricter_constrained_N24_N40_batch32_cae_inp_generation\stage3_run50_stricter_constrained_N24_N40_batch32_generated_inp_check.csv`
- Generated INP check summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_50_stricter_constrained_N24_N40_batch32_cae_inp_generation\stage3_run50_stricter_constrained_N24_N40_batch32_generated_inp_check_summary.json`

## Final Cooling Controls

- `step_final_cooling` duration: 1200.0 seconds
- `step_final_cooling` initial increment: 0.01
- `step_final_cooling` maximum increment size: 60.0
- `maxInc = 60.0` is maximum increment size, not `maxNumInc = 60`.
- For generated N40 INPs, all `step_cool_XX` initial increments were verified as 0.001.

## abqjobpilot Command-File Validation

- Verdict: `PASS_RUN50_ABQJOBPILOT_COMMAND_FILE_VALID`
- Command count: 32
- Per-N command counts: N24 = 16, N40 = 16
- Ready-to-run command file: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_50_stricter_constrained_N24_N40_batch32_cae_inp_generation\stage3_run50_stricter_constrained_N24_N40_batch32_abqjobpilot_commands_READY_TO_RUN.txt`
- Validation summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_50_stricter_constrained_N24_N40_batch32_cae_inp_generation\stage3_run50_stricter_constrained_N24_N40_batch32_abqjobpilot_command_validation_summary.json`
- The custom abqjobpilot command file is ready for user-controlled enqueue, but was not executed by Codex.

## Claim Boundary

Run50 generated CAE/INP/JNL handoff files only. The generated candidates are not teacher-validated, no Abaqus solver was run, no datacheck was run, no ODB was opened, and no abqjobpilot/enqueue command was executed by Codex.

Claim boundary files:
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_50_stricter_constrained_N24_N40_batch32_cae_inp_generation\run50_claim_boundary.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_50_stricter_constrained_N24_N40_batch32_cae_inp_generation\run50_claim_boundary.json`

## Output Files

- Output directory: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_50_stricter_constrained_N24_N40_batch32_cae_inp_generation`
- Main generation manifest: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_50_stricter_constrained_N24_N40_batch32_cae_inp_generation\stage3_run50_stricter_constrained_N24_N40_batch32_cae_generation_manifest.csv`
- Run manifest: `E:\Projects\RL-LAM-ScanOpt\artifacts\manifests\stage3_run_50_manifest.json`

## Exact Next User Action

Manually review the READY_TO_RUN command file and, if approved, enqueue the 32 jobs with the user's custom abqjobpilot workflow.
