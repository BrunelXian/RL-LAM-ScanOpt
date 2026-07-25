# Stage 3 Run 25 - Shortlist64 Active-Learning CAE/INP Generation

## Purpose
Generate CAE/INP/JNL handoff artifacts for the selected run24 shortlist64 active-learning calibration batch only. This run does not perform solver execution, ODB extraction, teacher validation, or policy training.

## Inputs
- Candidate order CSV: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\stage3_run24_shortlist64_candidate_orders.csv`
- Scan-order JSON directory: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_24_run23_shortlist64_active_learning_handoff_package\scan_orders`
- Case root: `E:\Projects\RL-LAM-ScanOpt\cae_model\stage3_run24_shortlist64_active_learning_calibration_v01`

## Selected Batch
- Selected batch: shortlist64
- Batch name: stage3_run24_shortlist64_active_learning_calibration_v01
- Batch24 generated: no
- Batch32 generated: no

## Preflight Validation
- Verdict: `PASS_RUN25_SHORTLIST64_INPUT_READY_FOR_CAE_GENERATION`
- Row count: 64
- Per-N counts: {'12': 8, '16': 8, '24': 24, '40': 24}
- Generation plan: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_25_shortlist64_active_learning_cae_inp_generation\run25_generation_plan.csv`

## Future CAE Root Safety Audit
- Verdict: `PASS_RUN25_FUTURE_CAE_ROOT_SAFE`
- Pre-existing solver outputs: 0
- Pre-existing CAE/INP/JNL artifacts: 0

## Base Mesh Audit
- Overall verdict: `PASS_RUN25_BASE_MESH_AUDIT_READY`
- N12: PASS_BASE_MESH_READY (399 nodes, 336 elements, 12 heat sets)
- N16: PASS_BASE_MESH_READY (511 nodes, 432 elements, 16 heat sets)
- N24: PASS_BASE_MESH_READY (735 nodes, 624 elements, 24 heat sets)
- N40: PASS_BASE_MESH_READY (1183 nodes, 1008 elements, 40 heat sets)

## CAE/INP Generation
- Abaqus noGUI command: `abaqus cae noGUI="E:\Projects\RL-LAM-ScanOpt\scripts\stage3\run_25_generate_shortlist64_from_sanity_base_nogui.py"`
- Generation verdict: `PASS_RUN25_SHORTLIST64_GENERATION_COMPLETE`
- Generated count: 64
- Generation manifest CSV: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_25_shortlist64_active_learning_cae_inp_generation\stage3_run25_shortlist64_cae_generation_manifest.csv`
- Generation manifest JSON: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_25_shortlist64_active_learning_cae_inp_generation\stage3_run25_shortlist64_cae_generation_manifest.json`

## Generated Counts By N
- N12: 8
- N16: 8
- N24: 24
- N40: 24

## INP Mesh/Final-Cooling Validation
- Verdict: `PASS_RUN25_SHORTLIST64_64_INPS_READY_FOR_USER_REVIEW`
- Checked INPs: 64
- Passing INPs: 64
- Total CAE count: 64
- Total INP count: 64
- Solver-output count under case root: 0

## Final Cooling Controls
All generated cases use `step_final_cooling` after the last cooling step with:

- `timePeriod = 1200.0`
- `initialInc = 0.01`
- `maxInc = 60.0` maximum increment size, not `maxNumInc = 60`
- all scan-related heat loads inactive during final cooling

## abqjobpilot Command-File Validation
- Verdict: `PASS_RUN25_ABQJOBPILOT_COMMAND_FILE_VALID`
- Command count: 64
- Command file: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_25_shortlist64_active_learning_cae_inp_generation\stage3_run25_shortlist64_abqjobpilot_commands_READY_TO_RUN.txt`
- abqjobpilot executed by Codex: false
- enqueue executed by Codex: false

## Claim Boundary
Run25 generated CAE/INP/JNL handoff artifacts only for the selected shortlist64 batch. The candidates are not teacher-validated, no solver jobs were run, no ODB was opened, and no physical-superiority claim is supported by this generation step.

## Output Files
- Preflight summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_25_shortlist64_active_learning_cae_inp_generation\run25_preflight_summary.json`
- Safety audit: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_25_shortlist64_active_learning_cae_inp_generation\run25_future_cae_root_safety_audit.json`
- Base mesh audit CSV: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_25_shortlist64_active_learning_cae_inp_generation\run25_base_mesh_audit.csv`
- INP check summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_25_shortlist64_active_learning_cae_inp_generation\stage3_run25_shortlist64_generated_inp_check_summary.json`
- Command validation summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_25_shortlist64_active_learning_cae_inp_generation\stage3_run25_shortlist64_abqjobpilot_command_validation_summary.json`
- Claim boundary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_25_shortlist64_active_learning_cae_inp_generation\run25_claim_boundary.md`

## Exact Next User Action
Manually review the READY_TO_RUN command file and, if approved, enqueue the 64 jobs using the user's custom abqjobpilot workflow. After solver completion, run a completion audit before any ODB extraction.

## N40 B02-B05 Cooling Initial-Increment Patch

Four run25 shortlist64 N40 cases failed in solver execution due to increment convergence in cooling steps. Codex did not rerun solver, datacheck, abqjobpilot, enqueue, or ODB postprocessing. Existing failed solver outputs for the four cases were archived before rewriting model artifacts.

- Patched cases: `S3R24L64_N40_B02_top_region`, `S3R24L64_N40_B03_top_region`, `S3R24L64_N40_B04_top_region`, `S3R24L64_N40_B05_top_region`
- User-reported failing locations: B02 `step_cool_14`, B03 `step_cool_15`, B04 `step_cool_15`, B05 `step_cool_14`
- Patch applied: all `step_cool_00` through `step_cool_39` initial increments set to `0.01` in each of the four CAE models
- Final cooling retained: `step_final_cooling` with `timePeriod=1200.0`, `initialInc=0.01`, `maxInc=60.0`
- Patch verdict: `PASS_RUN25_N40_B02_B03_B04_B05_COOL_INITIAL_INC_PATCHED`
- INP verification verdict: `PASS_RUN25_N40_B02_B03_B04_B05_COOL_INITIAL_INC_INP_VERIFIED`
- Archived failed solver outputs: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_25_shortlist64_active_learning_cae_inp_generation\archived_failed_N40_B02_B03_B04_B05_solver_outputs_before_cool_initialInc_patch_20260613_085016`
- Archive manifest: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_25_shortlist64_active_learning_cae_inp_generation\archived_failed_N40_B02_B03_B04_B05_solver_outputs_before_cool_initialInc_patch_20260613_085016\ARCHIVE_MANIFEST.csv`
- Patch summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_25_shortlist64_active_learning_cae_inp_generation\run25_N40_B02_B03_B04_B05_cool_initialInc_patch_summary.json`
- INP verification summary: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_25_shortlist64_active_learning_cae_inp_generation\run25_N40_B02_B03_B04_B05_cool_initialInc_patch_inp_verification.json`
- Four-case READY_TO_RUN command file: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_25_shortlist64_active_learning_cae_inp_generation\stage3_run25_N40_B02_B03_B04_B05_abqjobpilot_commands_READY_TO_RUN.txt`

The four-case custom abqjobpilot command file is ready for user-controlled enqueue, but was not executed by Codex.
