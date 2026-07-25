# Stage 3 Run 14 - Batch20 CAE/INP Generation

## Purpose

Run 14 generated Abaqus CAE/INP/JNL model-generation artifacts for the 20
run13 surrogate-screened batch20 candidates.

This is a model-generation run only. It does not validate physical performance.

## Inputs

- candidate order CSV:
  `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_13_batch20_surrogate_screened_teacher_handoff\stage3_run13_batch20_candidate_orders.csv`
- scan-order JSON directory:
  `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_13_batch20_surrogate_screened_teacher_handoff\scan_orders`
- normalized run14 generation plan:
  `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_14_batch20_cae_inp_generation\run14_generation_plan.csv`

Input validation verdict:

`PASS_RUN14_BATCH20_INPUT_READY_FOR_CAE_GENERATION`

Validated candidate counts:

- N12: 5
- N16: 5
- N24: 5
- N40: 5
- total: 20

## Base Mesh Audit

Base mesh audit verdict:

`PASS_RUN14_BASE_MESH_AUDIT_READY`

Per-N base readiness:

- N12: `PASS_BASE_MESH_READY`, 399 nodes, 336 elements, 12 heat sets
- N16: `PASS_BASE_MESH_READY`, 511 nodes, 432 elements, 16 heat sets
- N24: `PASS_BASE_MESH_READY`, 735 nodes, 624 elements, 24 heat sets
- N40: `PASS_BASE_MESH_READY`, 1183 nodes, 1008 elements, 40 heat sets

Audit outputs:

- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_14_batch20_cae_inp_generation\run14_base_mesh_audit.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_14_batch20_cae_inp_generation\run14_base_mesh_audit_summary.json`

## Generation Method

Abaqus noGUI command run:

```powershell
abaqus cae noGUI="E:\Projects\RL-LAM-ScanOpt\scripts\stage3\run_14_generate_batch20_cae_inp_from_sanity_base_nogui.py"
```

Generation logic:

- opened the appropriate N-specific sanity-base CAE
- created scan/cool sequence from each run13 `scan_order`
- created one `BodyHeatFlux` load per scan step
- deactivated heat loads in cooling steps
- appended `step_final_cooling`
- wrote CAE, INP, JNL placeholder, scan-order copy, metadata, and generation log
- did not submit any job

## Final Cooling Controls

Each generated case uses:

- final step: `step_final_cooling`
- `timePeriod = 1200.0`
- `initialInc = 0.01`
- `maxInc = 60.0`

`maxInc = 60.0` is the maximum increment size, not `maxNumInc = 60`.

## Generated CAE/INP Counts

Case root:

`E:\Projects\RL-LAM-ScanOpt\cae_model\stage3_run13_batch20_surrogate_screened_v01`

Generated counts:

- CAE: 20
- INP: 20
- JNL: 20
- generation logs: 20

Per-N counts:

- N12: 5 CAE, 5 INP
- N16: 5 CAE, 5 INP
- N24: 5 CAE, 5 INP
- N40: 5 CAE, 5 INP

## INP Mesh/Step/Heat-Load Checks

INP checker verdict:

`PASS_RUN14_BATCH20_20_INPS_READY_FOR_USER_REVIEW`

The checker confirmed all 20 INPs:

- exist and are nonempty
- contain `*Node`
- contain `*Element`
- have nonzero node and element section entries
- contain expected scan and cool step sequences
- contain `step_final_cooling`
- show final cooling controls in the exported INP text
- contain body heat flux entries
- contain expected `set_body_heat_XX` names
- have text-verifiable heat-load order matching `scan_order`
- have no ODB/SIM/STA/DAT/MSG/LCK solver outputs in the run14 case folders

Checker outputs:

- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_14_batch20_cae_inp_generation\run14_generated_inp_check.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_14_batch20_cae_inp_generation\run14_generated_inp_check_summary.json`

## abqjobpilot Command File Validation

Custom abqjobpilot command-file validation:

`PASS_RUN14_ABQJOBPILOT_COMMAND_FILE_VALID`

Ready-to-run command file:

`E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_14_batch20_cae_inp_generation\stage3_run14_batch20_abqjobpilot_commands_READY_TO_RUN.txt`

The file contains 20 commands. Each command references an existing INP, uses
`--cpus 14`, does not include `--gpus`, uses batch
`stage3_run13_batch20_surrogate_screened_v01`, and matches the case strategy
name.

The commands were generated for user-controlled execution but were not executed
by Codex.

## Claim Boundary

- run14 generated CAE/INP/JNL model artifacts only
- the run13/run14 batch20 candidates are still not teacher-validated
- no Abaqus solver was run
- no datacheck was run
- no ODB was opened
- no ODB postprocessing was run
- no teacher validation was run
- no RL or GNN training was run
- no physical superiority or validated improvement is claimed

## Output Files

- input validation:
  `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_14_batch20_cae_inp_generation\run14_input_validation_summary.json`
- generation summary:
  `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_14_batch20_cae_inp_generation\run14_generation_summary.json`
- CAE generation manifest CSV:
  `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_14_batch20_cae_inp_generation\stage3_run14_batch20_cae_generation_manifest.csv`
- command validation:
  `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_14_batch20_cae_inp_generation\run14_abqjobpilot_command_validation.json`
- run manifest:
  `E:\Projects\RL-LAM-ScanOpt\artifacts\manifests\stage3_run_14_manifest.json`
- generator script:
  `E:\Projects\RL-LAM-ScanOpt\scripts\stage3\run_14_generate_batch20_cae_inp_from_sanity_base_nogui.py`
- checker script:
  `E:\Projects\RL-LAM-ScanOpt\scripts\stage3\check_run14_batch20_generated_inps.py`

## Recommended Next Step

The user should manually review one or more generated INPs/CAEs. If approved,
the user may manually submit the 20 commands via the custom abqjobpilot command
file. After solver completion, run a solver completion audit before any ODB
extraction or teacher validation.
