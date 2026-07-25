# Stage 3 Run 40 - Native N24/N40 Focused Batch60 CAE/INP Generation

## Purpose
Generate CAE/INP/JNL handoff artifacts only for the selected Run39 native N24/N40 focused batch60. No solver, datacheck, ODB extraction, teacher validation, abqjobpilot, or enqueue execution was performed.

## Inputs
- Candidate CSV: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_39_run38_native_N24_N40_focused_batch60_handoff_package\stage3_run39_native_N24_N40_focused_batch60_candidate_orders.csv`
- Scan-order JSON directory: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_39_run38_native_N24_N40_focused_batch60_handoff_package\scan_orders`
- Base CAEs used: N24 and N40 only.

## Selected Batch
- Selected batch: `run38_native_N24_N40_focused_batch60`
- Batch name: `stage3_run39_native_N24_N40_focused_batch60_v01`
- Expected cases: 60 total, N24=30 and N40=30.
- No N12, N16, or N32 cases were generated.

## Preflight Validation
- Verdict: `PASS_RUN40_NATIVE_N24_N40_FOCUSED_BATCH60_INPUT_READY_FOR_CAE_GENERATION`
- Future CAE root safety: `PASS_RUN40_FUTURE_CAE_ROOT_SAFE`

## Base Mesh Audit
- Overall verdict: `PASS_RUN40_BASE_MESH_AUDIT_READY`
- N24: PASS_BASE_MESH_READY nodes=735.0 elements=624.0 heat_sets=24
- N40: PASS_BASE_MESH_READY nodes=1183.0 elements=1008.0 heat_sets=40

## CAE/INP Generation
- Abaqus noGUI command: `abaqus cae noGUI="E:\Projects\RL-LAM-ScanOpt\scripts\stage3\run_40_generate_native_N24_N40_focused_batch60_from_sanity_base_nogui.py"`
- Generation verdict: `PASS_RUN40_NATIVE_N24_N40_FOCUSED_BATCH60_GENERATION_COMPLETE`
- Case root: `E:\Projects\RL-LAM-ScanOpt\cae_model\stage3_run39_native_N24_N40_focused_batch60_v01`

## Generated Counts By N
- CAE total: 60; N24=30, N40=30
- INP total: 60; N24=30, N40=30
- JNL total: 60; N24=30, N40=30
- Generation logs total: 60; N24=30, N40=30

## INP Mesh/Final-Cooling Validation
- Verdict: `PASS_RUN40_NATIVE_N24_N40_FOCUSED_BATCH60_60_INPS_READY_FOR_USER_REVIEW`
- Checked count: 60
- PASS count: 60
- Solver output count under Run40 case root: 0

## Final Cooling Controls
- `step_final_cooling`: `timePeriod=1200.0`, `initialInc=0.01`, `maxInc=60.0` where visible in exported INPs.
- N40 `step_cool_XX`: all generated N40 INPs verified with `initialInc=0.001` and `timePeriod=3.4`.

## abqjobpilot Command-File Validation
- Verdict: `PASS_RUN40_ABQJOBPILOT_COMMAND_FILE_VALID`
- Command file: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_40_native_N24_N40_focused_batch60_cae_inp_generation\stage3_run40_native_N24_N40_focused_batch60_abqjobpilot_commands_READY_TO_RUN.txt`
- The command file is ready for user-controlled enqueue, but was not executed by Codex.

## Claim Boundary
Run40 generated CAE/INP/JNL handoff artifacts only. The candidates are not teacher-validated. No physical superiority, solver completion, ODB result, RL training, or arbitrary-N generalization claim is made here.

## Output Files
- Preflight summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_40_native_N24_N40_focused_batch60_cae_inp_generation\run40_preflight_summary.json`
- Base mesh audit summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_40_native_N24_N40_focused_batch60_cae_inp_generation\run40_base_mesh_audit_summary.json`
- Generation manifest CSV: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_40_native_N24_N40_focused_batch60_cae_inp_generation\stage3_run40_native_N24_N40_focused_batch60_cae_generation_manifest.csv`
- Generated INP check summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_40_native_N24_N40_focused_batch60_cae_inp_generation\stage3_run40_native_N24_N40_focused_batch60_generated_inp_check_summary.json`
- abqjobpilot validation summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_40_native_N24_N40_focused_batch60_cae_inp_generation\stage3_run40_native_N24_N40_focused_batch60_abqjobpilot_command_validation_summary.json`
- Claim boundary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_40_native_N24_N40_focused_batch60_cae_inp_generation\run40_claim_boundary.md`
- Manifest: `E:\Projects\RL-LAM-ScanOpt\artifacts\manifests\stage3_run_40_manifest.json`

## Exact Next User Action
Manually review the READY_TO_RUN command file and, if approved, enqueue the 60 jobs with the custom abqjobpilot outside Codex.
