# Stage 3 Run 35 - N32-Informed Native Batch32 CAE/INP Generation

## Purpose
Generate CAE/INP/JNL handoff artifacts for the selected Run34 N32-informed native batch32 only. This run does not perform solver execution, ODB extraction, teacher validation, or RL/GNN training.

## Inputs
- Candidate order CSV: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_34_run33_N32_informed_native_batch32_handoff_package\stage3_run34_N32_informed_native_batch32_candidate_orders.csv`
- Scan-order JSON directory: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_34_run33_N32_informed_native_batch32_handoff_package\scan_orders`
- Run34 report: `E:\Projects\RL-LAM-ScanOpt\docs\stage3\runs\run_34_run33_N32_informed_native_batch32_handoff_package\RUN_34_RUN33_N32_INFORMED_NATIVE_BATCH32_HANDOFF_PACKAGE_REPORT.md`
- Case root: `E:\Projects\RL-LAM-ScanOpt\cae_model\stage3_run34_N32_informed_native_batch32_v01`

## Selected Batch
- Selected batch: run33_N32_informed_native_batch32
- Batch name: stage3_run34_N32_informed_native_batch32_v01
- Run33 Option B generated: no
- Run33 Option C generated: no
- Any N32 cases generated: no

## Superseded Old Run31 Warning
The older Run31 CAE/INP package for `stage3_run30_hybrid_policy_batch32_v01` is superseded by this Run35 package for active teacher-validation planning. Run35 did not delete or modify old Run31 files.

## Preflight Validation
- Verdict: `PASS_RUN35_N32_INFORMED_NATIVE_BATCH32_INPUT_READY_FOR_CAE_GENERATION`
- Row count: 32
- Per-N counts: {'12': 4, '16': 4, '24': 12, '40': 12}
- Generation plan: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_35_N32_informed_native_batch32_cae_inp_generation\run35_generation_plan.csv`

## Future CAE Root Safety Audit
- Verdict: `PASS_RUN35_FUTURE_CAE_ROOT_SAFE`
- Pre-existing solver outputs: 0
- Pre-existing CAE/INP/JNL artifacts: 0

## Base Mesh Audit
- Overall verdict: `PASS_RUN35_BASE_MESH_AUDIT_READY`
- N12: PASS_BASE_MESH_READY (399 nodes, 336 elements, 12 heat sets)
- N16: PASS_BASE_MESH_READY (511 nodes, 432 elements, 16 heat sets)
- N24: PASS_BASE_MESH_READY (735 nodes, 624 elements, 24 heat sets)
- N40: PASS_BASE_MESH_READY (1183 nodes, 1008 elements, 40 heat sets)

## CAE/INP Generation
- Abaqus noGUI command: `abaqus cae noGUI="E:\Projects\RL-LAM-ScanOpt\scripts\stage3\run_35_generate_N32_informed_native_batch32_from_sanity_base_nogui.py"`
- Generation verdict: `PASS_RUN35_N32_INFORMED_NATIVE_BATCH32_GENERATION_COMPLETE`
- Generated count: 32
- Generation manifest CSV: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_35_N32_informed_native_batch32_cae_inp_generation\stage3_run35_N32_informed_native_batch32_cae_generation_manifest.csv`
- Generation manifest JSON: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_35_N32_informed_native_batch32_cae_inp_generation\stage3_run35_N32_informed_native_batch32_cae_generation_manifest.json`

## Generated Counts By N
- N12: 4
- N16: 4
- N24: 12
- N40: 12

## INP Mesh/Final-Cooling Validation
- Verdict: `PASS_RUN35_N32_INFORMED_NATIVE_BATCH32_32_INPS_READY_FOR_USER_REVIEW`
- Checked INPs: 32
- Passing INPs: 32
- Total CAE count: 32
- Total INP count: 32
- N32 INP count: 0
- Solver-output count under case root: 0

## Final Cooling Controls
All generated cases use `step_final_cooling` after the last cooling step with:

- `timePeriod = 1200.0`
- `initialInc = 0.01`
- `maxInc = 60.0` maximum increment size, not `maxNumInc = 60`
- all scan-related heat loads inactive during final cooling

## N40 Cooling Controls
For all Run35 N40 cases, every `step_cool_XX` was generated with:

- `timePeriod = 3.4`
- `initialInc = 0.001`

This was added because prior N40 runs showed multiple cooling-step increment convergence failures.

## abqjobpilot Command-File Validation
- Verdict: `PASS_RUN35_ABQJOBPILOT_COMMAND_FILE_VALID`
- Command count: 32
- Command file: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_35_N32_informed_native_batch32_cae_inp_generation\stage3_run35_N32_informed_native_batch32_abqjobpilot_commands_READY_TO_RUN.txt`
- abqjobpilot executed by Codex: false
- enqueue executed by Codex: false

## Claim Boundary
Run35 generated CAE/INP/JNL handoff artifacts only for the selected Run34 N32-informed native batch32. The candidates are not teacher-validated, no solver jobs were run, no ODB was opened, and no physical-superiority, GNN-RL-superiority, or N32-causality claim is supported by this generation step.

## Output Files
- Preflight summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_35_N32_informed_native_batch32_cae_inp_generation\run35_preflight_summary.json`
- Safety audit: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_35_N32_informed_native_batch32_cae_inp_generation\run35_future_cae_root_safety_audit.json`
- Base mesh audit CSV: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_35_N32_informed_native_batch32_cae_inp_generation\run35_base_mesh_audit.csv`
- INP check summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_35_N32_informed_native_batch32_cae_inp_generation\stage3_run35_N32_informed_native_batch32_generated_inp_check_summary.json`
- Command validation summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_35_N32_informed_native_batch32_cae_inp_generation\stage3_run35_N32_informed_native_batch32_abqjobpilot_command_validation_summary.json`
- Claim boundary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_35_N32_informed_native_batch32_cae_inp_generation\run35_claim_boundary.md`

## Exact Next User Action
Manually review the READY_TO_RUN command file and, if approved, enqueue the 32 jobs using the user's custom abqjobpilot workflow. After solver completion, run a completion audit before any ODB extraction.
