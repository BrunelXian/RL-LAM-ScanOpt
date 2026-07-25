# Stage 3 Run 70 - Small-N Recovery-Focused Batch40 CAE/INP Generation

## Purpose

Run70 generated CAE/INP/JNL handoff files for the selected Run69 small-N recovery-focused batch40. The batch is intended for later user-controlled solver submission, with small-N recovery as the primary focus and N24/N40 retained as mature anchors.

## Inputs

- Candidate order CSV: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_69_run68_smallN_recovery_focused_batch40_handoff_package\stage3_run69_smallN_recovery_focused_batch40_candidate_orders.csv`
- Scan-order directory: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_69_run68_smallN_recovery_focused_batch40_handoff_package\scan_orders`
- Run69 report: `E:\Projects\RL-LAM-ScanOpt\docs\stage3\runs\run_69_run68_smallN_recovery_focused_batch40_handoff_package\RUN_69_RUN68_SMALLN_RECOVERY_FOCUSED_BATCH40_HANDOFF_PACKAGE_REPORT.md`
- Case root: `E:\Projects\RL-LAM-ScanOpt\cae_model\stage3_run69_smallN_recovery_focused_batch40_v01`

## Selected Batch

- Selected batch: `run68_smallN_recovery_focused_batch40`
- Batch name: `stage3_run69_smallN_recovery_focused_batch40_v01`
- Expected cases: 40
- Expected per-N counts: N12=16, N16=16, N24=4, N40=4
- N32 excluded.

## Preflight Validation

Preflight verdict: `PASS_RUN70_SMALLN_RECOVERY_FOCUSED_BATCH40_INPUT_READY_FOR_CAE_GENERATION`

The preflight confirmed the Run69 candidate CSV, scan-order JSON files, per-N counts, filesystem-safe strategy names, legal scan-order permutations, correct batch name, valid future CAE root schema, no doubled-N path schema, no N32 rows, and required base CAE paths.

Output files:

- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_70_smallN_recovery_focused_batch40_cae_inp_generation\run70_preflight_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_70_smallN_recovery_focused_batch40_cae_inp_generation\run70_preflight_cases.csv`

## Future CAE Root Safety Audit

Safety audit verdict: `PASS_RUN70_FUTURE_CAE_ROOT_SAFE`

The future CAE root did not contain existing solver outputs before generation. No deletion of solver outputs was performed.

Output file:

- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_70_smallN_recovery_focused_batch40_cae_inp_generation\run70_future_cae_root_safety_audit.json`

## Base Mesh Audit

Base mesh audit verdict: `PASS_RUN70_BASE_MESH_AUDIT_READY`

Per-N results:

- N12: `PASS_BASE_MESH_READY`, nodes=399, elements=336
- N16: `PASS_BASE_MESH_READY`, nodes=511, elements=432
- N24: `PASS_BASE_MESH_READY`, nodes=735, elements=624
- N40: `PASS_BASE_MESH_READY`, nodes=1183, elements=1008

The audit confirmed required heat sets, `step_scan_00`, `step_cool_00`, and `load_body_hflux_00` for all native N values.

Output files:

- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_70_smallN_recovery_focused_batch40_cae_inp_generation\run70_base_mesh_audit.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_70_smallN_recovery_focused_batch40_cae_inp_generation\run70_base_mesh_audit_summary.json`

## CAE/INP Generation

Abaqus CAE noGUI generation was executed for model generation only:

`abaqus cae noGUI="E:\Projects\RL-LAM-ScanOpt\scripts\stage3\run_70_generate_smallN_recovery_focused_batch40_from_sanity_base_nogui.py"`

No solver, datacheck, abqjobpilot, enqueue, ODB opening, teacher validation, or RL training was performed.

Generated counts:

- CAE total: 40
- INP total: 40
- JNL total: 40
- Generation logs: 40

Per-N generated counts:

- N12: 16
- N16: 16
- N24: 4
- N40: 4

Generation manifest:

- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_70_smallN_recovery_focused_batch40_cae_inp_generation\stage3_run70_smallN_recovery_focused_batch40_cae_generation_manifest.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_70_smallN_recovery_focused_batch40_cae_inp_generation\stage3_run70_smallN_recovery_focused_batch40_cae_generation_manifest.json`

## INP Mesh And Final-Cooling Validation

INP checker verdict: `PASS_RUN70_SMALLN_RECOVERY_FOCUSED_BATCH40_40_INPS_READY_FOR_USER_REVIEW`

The checker confirmed:

- Total INP count = 40
- Per-N INP counts: N12=16, N16=16, N24=4, N40=4
- Total CAE count = 40
- No N32 INPs
- No ODB/SIM/STA/DAT/MSG/LCK solver outputs under the case root
- Every INP contains `*Node` and `*Element`
- Every INP contains scan/cool steps, heat flux entries, and `step_final_cooling`
- Final cooling controls are visible: `initialInc = 0.01`, `timePeriod = 1200.0`, `maxInc = 60.0`
- Generated N40 INPs have all `step_cool_XX` initial increments set to `0.001`

Output files:

- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_70_smallN_recovery_focused_batch40_cae_inp_generation\stage3_run70_smallN_recovery_focused_batch40_generated_inp_check.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_70_smallN_recovery_focused_batch40_cae_inp_generation\stage3_run70_smallN_recovery_focused_batch40_generated_inp_check_summary.json`

## Final Cooling Controls

Final cooling step:

- Step name: `step_final_cooling`
- Duration: `1200.0`
- Initial increment: `0.01`
- Maximum increment size: `60.0`

For N40 scan/cool steps, `step_cool_XX` initial increment was verified as `0.001`.

## abqjobpilot Command-File Validation

Command validation verdict: `PASS_RUN70_ABQJOBPILOT_COMMAND_FILE_VALID`

The command file contains 40 commands:

- N12 = 16
- N16 = 16
- N24 = 4
- N40 = 4

Validation confirmed that every command starts with `enqueue --inp`, references an existing INP, includes `--cpus 14`, excludes `--gpus`, uses batch `stage3_run69_smallN_recovery_focused_batch40_v01`, matches the generated strategy folder, and contains no bad doubled-N path schema.

Command file:

- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_70_smallN_recovery_focused_batch40_cae_inp_generation\stage3_run70_smallN_recovery_focused_batch40_abqjobpilot_commands_READY_TO_RUN.txt`

The custom abqjobpilot command file was generated and validated for user-controlled execution, but was not executed by Codex.

## Claim Boundary

Run70 generated CAE/INP/JNL files only. The candidates remain not teacher-validated. No solver completion or physical improvement is claimed. No ODB results exist from Codex actions in this run.

Claim boundary files:

- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_70_smallN_recovery_focused_batch40_cae_inp_generation\run70_claim_boundary.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_70_smallN_recovery_focused_batch40_cae_inp_generation\run70_claim_boundary.json`

## Output Files

- Ready-to-run command file: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_70_smallN_recovery_focused_batch40_cae_inp_generation\stage3_run70_smallN_recovery_focused_batch40_abqjobpilot_commands_READY_TO_RUN.txt`
- CAE generation manifest: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_70_smallN_recovery_focused_batch40_cae_inp_generation\stage3_run70_smallN_recovery_focused_batch40_cae_generation_manifest.csv`
- INP checker summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_70_smallN_recovery_focused_batch40_cae_inp_generation\stage3_run70_smallN_recovery_focused_batch40_generated_inp_check_summary.json`
- Command validation summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_70_smallN_recovery_focused_batch40_cae_inp_generation\stage3_run70_smallN_recovery_focused_batch40_abqjobpilot_command_validation_summary.json`
- Run manifest: `E:\Projects\RL-LAM-ScanOpt\artifacts\manifests\stage3_run_70_manifest.json`

## Exact Next User Action

User may manually review the READY_TO_RUN command file and enqueue the 40 jobs with their custom abqjobpilot if desired.
