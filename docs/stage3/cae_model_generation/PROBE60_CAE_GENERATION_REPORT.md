# Probe60 CAE Generation Report

Date: 2026-06-11

## Current Status

Stage 3 true variable-N probe60 model-generation workflow was created and run
only in normal-Python check/dry-run mode. No Abaqus command was executed. No
CAE, INP, JNL, ODB, SIM, DAT, MSG, STA, or LCK solver artifacts were generated
by this conversation.

## Base Model Inventory

| N | Base CAE | Base JNL |
|---:|---|---|
| 12 | `E:\Projects\RL-LAM-ScanOpt\cae_model\12track_full\sanity_base\12track_sanity_base.cae` | `E:\Projects\RL-LAM-ScanOpt\cae_model\12track_full\sanity_base\12track_sanity_base.jnl` |
| 16 | `E:\Projects\RL-LAM-ScanOpt\cae_model\16track_full\sanity_base\16track_sanity_base.cae` | `E:\Projects\RL-LAM-ScanOpt\cae_model\16track_full\sanity_base\16track_sanity_base.jnl` |
| 24 | `E:\Projects\RL-LAM-ScanOpt\cae_model\24track_full\sanity_base\24track_sanity_base.cae` | `E:\Projects\RL-LAM-ScanOpt\cae_model\24track_full\sanity_base\24track_sanity_base.jnl` |
| 40 | `E:\Projects\RL-LAM-ScanOpt\cae_model\40track_full\sanity_base\40track_sanity_base.cae` | `E:\Projects\RL-LAM-ScanOpt\cae_model\40track_full\sanity_base\40track_sanity_base.jnl` |

## Preflight Verdict

`PASS_PROBE60_GENERATION_PREFLIGHT_READY`

- Manifest rows: 60
- Scan-order files found: 60
- Current INP count before generation: 0
- Correct N-folder schema was checked for `\N12\N12_`, `\N16\N16_`,
  `\N24\N24_`, and `\N40\N40_`.
- Concatenated path bugs such as `N12N12_` were checked and not found.

## JNL Object-Name Audit Summary

`WARNING_BASE_JNL_OBJECT_MAPPING_PARTIAL`

The four base journals expose consistent naming inventory:

- model: `Model-1`
- part: `part_plate`
- instance: `inst_plate`
- external surface: `surf_external_all`
- heat-region sets: `set_body_heat_XX`, with counts matching N
- recorded scan/cool steps: `step_scan_00`, `step_cool_00`
- recorded heat load: `load_body_hflux_00`

No `writeInput` or `saveAs` calls were found in the base journals.

## Heat-Load Mapping Status

`MANUAL_OBJECT_MAPPING_REQUIRED`

The base journals show N heat-region sets but only one recorded scan step and
one body heat flux load bound to `set_body_heat_00`. That is not enough evidence
to automatically create candidate-specific scan-order INPs for arbitrary probe60
orders.

Generic export without reorder is blocked by default because it would preserve
the base model's recorded heat-load mapping instead of producing true
candidate-specific scan-order models.

## Generator Readiness

The noGUI generator exists and defaults to:

- `MODE = "dry_run"`
- `ALLOW_GENERIC_EXPORT_WITHOUT_REORDER = False`
- `ALLOW_USER_CONFIRMED_HEAT_MAPPING = False`
- `ONLY_GENERATE_ONE_PILOT_CASE = True`
- `PILOT_N = 12`
- `PILOT_STRATEGY = "N12_A01_raster_left_to_right"`

Dry run completed with:

- case count: 60
- heat-load mapping status: `MANUAL_OBJECT_MAPPING_REQUIRED`
- generate mode safe: `False`

Generate mode is blocked until heat-load object mapping is manually specified
and implemented safely.

## Manual Heat-Load Mapping Resolver

JNL mapping extraction verdict:

`PASS_HEAT_MAPPING_JNL_PATTERNS_READY`

The detailed extractor created:

- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_cae_probe60_generation\heat_mapping_jnl_snippets.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_cae_probe60_generation\heat_mapping_jnl_snippets.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_cae_probe60_generation\heat_mapping_inferred_patterns.json`

Inferred object patterns:

- heat set pattern: `set_body_heat_{track:02d}`
- heat set counts: N12=12, N16=16, N24=24, N40=40
- scan step template: `step_scan_00`
- cooling step template: `step_cool_00`
- load template: `load_body_hflux_00`
- load type evidence: `BodyHeatFlux`
- load magnitude evidence: `80000000000.0`
- scan duration evidence: `0.2` seconds
- cooling duration evidence: `3.4` seconds

Missing / still-required evidence:

- Abaqus CAE model object inspection has not been run in this conversation.
- User confirmation is still required before Abaqus generation.
- The generator must validate at runtime that every `set_body_heat_XX` object
  exists in the loaded CAE model before writing any CAE or INP.

Generated heat mapping config:

`E:\Projects\RL-LAM-ScanOpt\outputs\stage3_cae_probe60_generation\probe60_heat_mapping_config_TEMPLATE.json`

The config remains conservative:

- `allow_generate_when_confirmed = false`
- final cooling step name: `step_final_cooling`
- final cooling duration: `1200.0` seconds
- final stress / downstream residual-field evaluation should use the end of
  `step_final_cooling`

The updated generator is fail-closed:

- It refuses generic export without scan-order remapping.
- It refuses generate mode while
  `ALLOW_USER_CONFIRMED_HEAT_MAPPING = False`.
- It defaults to pilot-only generation even after user confirmation.
- It appends `step_final_cooling` after the last scan/cool pair.
- It checks that scan loads were deactivated in cooling steps before writing.

Pilot-only generation plan:

1. Review `heat_mapping_jnl_snippets.md`.
2. Review and edit `probe60_heat_mapping_config_TEMPLATE.json` if needed.
3. Manually confirm the mapping assumptions.
4. Set `ALLOW_USER_CONFIRMED_HEAT_MAPPING = True` while keeping
   `ONLY_GENERATE_ONE_PILOT_CASE = True`.
5. Run Abaqus noGUI only for the pilot case
   `N12_A01_raster_left_to_right`.
6. Run the generated-file checker and pilot INP text checker.
7. Manually inspect the pilot INP before considering all-60 generation.

Pilot INP text-check status:

`FAIL_PILOT_INP_NOT_READY`

This is expected because no pilot INP has been generated yet.

Full-generation status:

`BLOCKED_FULL_60_GENERATION`

All-60 generation should remain blocked until the pilot INP is inspected and
the user explicitly chooses to enable full generation.

## Generated File Check

`FAIL_PROBE60_NO_INPS_FOUND`

- Current INP count: 0
- Current CAE count under probe60 case root: 0
- INP count per N: N12=0, N16=0, N24=0, N40=0
- CAE count per N: N12=0, N16=0, N24=0, N40=0

No ODB, SIM, DAT, MSG, STA, or LCK files were found under the probe60 case root.

## N12_A01 One-Case Pilot Generation

Exact command run:

```powershell
abaqus cae noGUI="E:\Projects\RL-LAM-ScanOpt\scripts\stage3\generate_probe60_from_sanity_base_nogui.py"
```

Abaqus noGUI executed: yes.

Preflight before pilot:

`PASS_PROBE60_GENERATION_PREFLIGHT_READY`

Pilot generation status:

`FAIL_PILOT_GENERATION_NO_CAE_OR_INP_WRITTEN`

Abaqus opened:

`E:\Projects\RL-LAM-ScanOpt\cae_model\12track_full\sanity_base\12track_sanity_base.cae`

Generator failure:

`cannot read template BodyHeatFlux magnitude`

Last generation log:

`E:\Projects\RL-LAM-ScanOpt\outputs\stage3_cae_probe60_generation\probe60_generator_failure.json`

Pilot CAE path:

`E:\Projects\RL-LAM-ScanOpt\cae_model\stage3_true_variable_N_probe60_v01\N12\N12_A01_raster_left_to_right\J2D_N12_N12_A01_raster_left_to_right.cae`

Pilot CAE exists: no.

Pilot INP path:

`E:\Projects\RL-LAM-ScanOpt\cae_model\stage3_true_variable_N_probe60_v01\N12\N12_A01_raster_left_to_right\J2D_N12_N12_A01_raster_left_to_right.inp`

Pilot INP exists: no.

Pilot checker verdict:

`FAIL_PILOT_INP_NOT_READY`

Final cooling verification:

Not reached. `step_final_cooling` was not written because generation stopped
before CAE/INP export.

Heat-load order verification status:

Not verified. The pilot INP does not exist. No comparison against
`scan_order_N12_A01_raster_left_to_right.json` was possible.

Solver/datacheck/job submission:

No solver job, datacheck, abqjobpilot, enqueue, or ODB postprocessing was run.
The Abaqus replay shows the script opened the base CAE and then failed closed.

Solver artifact check:

No ODB, SIM, DAT, MSG, STA, or LCK files were found under the probe60 case root.

Full 60 generation remains blocked:

Yes. No all-case generation was attempted.

Next recommended action:

Inspect the Abaqus `BodyHeatFlux` object API in noGUI with a read-only pilot
inspection script, or update the generator to derive the magnitude from the JNL
mapping config instead of reading `template_load.magnitude` directly. Do not
retry generation until that accessor is fixed and reviewed.

## N12_A01 Pilot Retry After Magnitude Resolver Fix

Prior failure diagnosis:

The first pilot attempt failed in
`create_or_copy_body_heat_flux_for_sequence()` because the Abaqus
`BodyHeatFlux` object did not expose magnitude as a simple `.magnitude`
attribute. The prior failure log only recorded:

`cannot read template BodyHeatFlux magnitude`

Confirmed pilot config:

`E:\Projects\RL-LAM-ScanOpt\outputs\stage3_cae_probe60_generation\probe60_heat_mapping_config_CONFIRMED_PILOT_N12.json`

Magnitude and duration sources used:

- Body heat flux magnitude: `80000000000.0`, source `confirmed_config`
- scan duration: `0.2`, source `confirmed_config`
- cool duration: `3.4`, source `confirmed_config`
- final cooling duration: `1200.0`, source `confirmed_config`
- Abaqus object magnitude readable: `false`

Exact Abaqus command run:

```powershell
abaqus cae noGUI="E:\Projects\RL-LAM-ScanOpt\scripts\stage3\generate_probe60_from_sanity_base_nogui.py"
```

Pilot generation result:

`GENERATED_PILOT`

Pilot CAE:

`E:\Projects\RL-LAM-ScanOpt\cae_model\stage3_true_variable_N_probe60_v01\N12\N12_A01_raster_left_to_right\J2D_N12_N12_A01_raster_left_to_right.cae`

Pilot INP:

`E:\Projects\RL-LAM-ScanOpt\cae_model\stage3_true_variable_N_probe60_v01\N12\N12_A01_raster_left_to_right\J2D_N12_N12_A01_raster_left_to_right.inp`

Generation log:

`E:\Projects\RL-LAM-ScanOpt\cae_model\stage3_true_variable_N_probe60_v01\N12\N12_A01_raster_left_to_right\J2D_N12_N12_A01_raster_left_to_right.generation_log.json`

Checker verdicts:

- pilot INP heat-order checker: `PASS_PILOT_INP_HEAT_ORDER_TEXT_CHECK`
- generated-file checker: `WARNING_PROBE60_PARTIAL_INPS_EXIST`

Heat-load order text verification:

Verified for the N12 raster pilot. The INP contains `step_scan_00` through
`step_scan_11`, `step_cool_00` through `step_cool_11`, `load_body_hflux_00`
through `load_body_hflux_11`, and `set_body_heat_00` through
`set_body_heat_11` in the expected raster order.

Final cooling verification:

Verified in pilot INP text:

- `step_final_cooling` appears after `step_cool_11`
- final cooling duration is exported as `1200.`
- no body heat flux load block appears in the final cooling step

Current generated-file status:

- total INP count under probe60 root: 1
- total CAE count under probe60 root: 1
- no ODB, SIM, DAT, MSG, STA, or LCK files found under the probe60 root

Full 60 generation remains blocked:

Yes. The generator config remains pilot-only and
`allow_full60_generation = false`.

abqjobpilot remains blocked:

Yes. The all-case checker is not enqueue-ready and reports only partial INPs.

## Full Probe60 CAE/INP Generation

Full-generation config:

`E:\Projects\RL-LAM-ScanOpt\outputs\stage3_cae_probe60_generation\probe60_heat_mapping_config_CONFIRMED_FULL60.json`

Exact Abaqus noGUI command run:

```powershell
abaqus cae noGUI="E:\Projects\RL-LAM-ScanOpt\scripts\stage3\generate_probe60_from_sanity_base_nogui.py"
```

Generation result:

`PASS_PROBE60_60_INPS_EXIST_READY_TO_ENQUEUE`

Generated file root:

`E:\Projects\RL-LAM-ScanOpt\cae_model\stage3_true_variable_N_probe60_v01`

Counts:

- total CAE count: 60
- total INP count: 60
- total JNL or generation-log count: 60
- per-N INP counts: N12=15, N16=15, N24=15, N40=15

Generated-file checker:

`PASS_PROBE60_60_INPS_EXIST_READY_TO_ENQUEUE`

Spot-check summary:

`PASS_FULL60_SPOT_CHECKS`

Spot-checked INPs:

- `N12_A01_raster_left_to_right`
- `N16_A01_raster_left_to_right`
- `N24_A01_raster_left_to_right`
- `N40_A01_raster_left_to_right`

For each spot-check:

- scan steps from `step_scan_00` through `step_scan_{N-1}` were found
- cool steps from `step_cool_00` through `step_cool_{N-1}` were found
- `step_final_cooling` was found after the last cool step
- final cooling duration was found as `1200.`
- body heat flux load text was found
- `set_body_heat_XX` names were preserved
- no body heat flux load block was found inside `step_final_cooling`

Custom abqjobpilot command-file validation:

`PASS_ABQJOBPILOT_COMMAND_FILE_VALID`

Validated command file:

`E:\Projects\RL-LAM-ScanOpt\outputs\stage3_manual_probe60_handoff\variable_N_probe60_abqjobpilot_commands_FIXED.txt`

Validation checks:

- command count: 60
- each command starts with `enqueue --inp`
- each command includes `--cpus 14`
- no command includes `--gpus`
- each command uses `--batch stage3_true_variable_N_probe60_v01`
- every command points to an existing INP
- strategy names match case folders

The user's custom abqjobpilot command file is ready for user-controlled
enqueue, but was not executed by Codex.

Forbidden-action confirmation:

- no solver analysis was run
- no datacheck was run
- no job submission was run
- no abqjobpilot command was executed
- no enqueue command was executed
- no ODB was opened
- no ODB, SIM, DAT, MSG, STA, or LCK files were produced under the probe60 root

## Full Probe60 Regeneration With Final Cooling Increment Control

Reason for regeneration:

The previous full60 CAE/INP generation was structurally acceptable, but one
generated model inspection showed unsuitable final cooling increment controls.
The regeneration changes only the terminal cooling step controls while
preserving the validated scan/cool/load mapping approach.

Pre-archive safety audit:

- CAE: 60
- INP: 60
- JNL: 60
- generation logs: 60
- scan-order JSON: 60
- case directories: 60
- ODB: 1
- SIM: 0
- STA: 1
- DAT: 1
- MSG: 1
- LCK: 0

Old solver outputs archived:

`E:\Projects\RL-LAM-ScanOpt\outputs\stage3_cae_probe60_generation\archived_old_solver_outputs_before_finalcool_regen_20260611_113729`

Archive manifest:

`E:\Projects\RL-LAM-ScanOpt\outputs\stage3_cae_probe60_generation\archived_old_solver_outputs_before_finalcool_regen_20260611_113729\ARCHIVE_MANIFEST.csv`

Archived solver-output count: 7

Deleted old generated model artifacts:

- CAE: 60
- INP: 60
- JNL: 60
- generation logs: 60

Preserved scan-order JSON count: 60

Updated full-generation config:

`E:\Projects\RL-LAM-ScanOpt\outputs\stage3_cae_probe60_generation\probe60_heat_mapping_config_CONFIRMED_FULL60.json`

Final cooling parameters:

- `timePeriod = 1200.0`
- `initialInc = 0.01`
- `maxInc = 60.0`
- `maxInc` is interpreted as maximum increment size, not `maxNumInc`

Exact Abaqus noGUI command run:

```powershell
abaqus cae noGUI="E:\Projects\RL-LAM-ScanOpt\scripts\stage3\generate_probe60_from_sanity_base_nogui.py"
```

Regeneration result:

`PASS_PROBE60_60_INPS_EXIST_READY_TO_ENQUEUE`

Regenerated counts:

- total CAE count: 60
- total INP count: 60
- total JNL or generation-log count: 60
- per-N INP counts: N12=15, N16=15, N24=15, N40=15

Generated-file checker verdict:

`PASS_PROBE60_60_INPS_EXIST_READY_TO_ENQUEUE`

Final cooling increment spot-check summary:

`PASS_FULL60_FINALCOOL_INCREMENT_SPOT_CHECKS`

Spot-checked INPs:

- `N12_A01_raster_left_to_right`
- `N16_A01_raster_left_to_right`
- `N24_A01_raster_left_to_right`
- `N40_A01_raster_left_to_right`

For each spot-check:

- `step_final_cooling` exists after the last cool step
- final cooling data line is `0.01, 1200., 3.4e-30, 60.`
- final cooling initial increment is `0.01`
- final cooling time period is `1200.0`
- final cooling maximum increment size is `60.0`
- generation logs verify `final_cooling_initialInc = 0.01` and
  `final_cooling_maxInc = 60.0`
- no body heat flux load block appears during `step_final_cooling`
- scan/cool/load/set naming remains present

Custom abqjobpilot command-file validation:

`PASS_ABQJOBPILOT_COMMAND_FILE_VALID`

Validated command file:

`E:\Projects\RL-LAM-ScanOpt\outputs\stage3_manual_probe60_handoff\variable_N_probe60_abqjobpilot_commands_FIXED.txt`

The user's custom abqjobpilot command file is ready for user-controlled
enqueue, but was not executed by Codex.

Forbidden-action confirmation:

- no solver analysis was run by Codex
- no datacheck was run
- no job submission was run
- no abqjobpilot command was executed
- no enqueue command was executed
- no ODB was opened
- no ODB, SIM, DAT, MSG, STA, or LCK files exist in the active probe60 root
  after regeneration

## Partial Regeneration of N16/N24/N40 After Corrected Mesh Base Models

Reason:

The previously generated N16/N24/N40 models were structurally generated before
the corresponding sanity-base CAE models contained mesh. The user manually
corrected the N16/N24/N40 sanity bases, and N12 was preserved because it had
already run successfully.

N12 preservation:

- N12 generated CAE/INP/JNL/log files were not deleted.
- N12 solver outputs were not moved.
- N12 was not regenerated.

Archived old N16/N24/N40 solver outputs:

- archive path: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_cae_probe60_generation\archived_old_N16_N24_N40_solver_outputs_before_mesh_regen_20260611_141853`
- manifest: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_cae_probe60_generation\archived_old_N16_N24_N40_solver_outputs_before_mesh_regen_20260611_141853\ARCHIVE_MANIFEST.csv`
- moved solver-output file count: 360
- active N16/N24/N40 ODB/SIM/STA/DAT/MSG/LCK count after archive: 0

Corrected base mesh audit:

- N16: `PASS_BASE_MESH_READY`, 511 nodes, 432 elements, 16 heat sets
- N24: `PASS_BASE_MESH_READY`, 735 nodes, 624 elements, 24 heat sets
- N40: `PASS_BASE_MESH_READY`, 1183 nodes, 1008 elements, 40 heat sets

Deleted stale generated artifacts under N16/N24/N40 only:

- N16: 15 CAE, 15 INP, 15 JNL, 15 generation logs
- N24: 15 CAE, 15 INP, 15 JNL, 15 generation logs
- N40: 15 CAE, 15 INP, 15 JNL, 15 generation logs
- preserved scan-order JSON files: 45

Regeneration:

- Abaqus noGUI command run:

```powershell
abaqus cae noGUI="E:\Projects\RL-LAM-ScanOpt\scripts\stage3\generate_probe60_from_sanity_base_nogui.py"
```

- regenerated target count: 45 cases
- N16 regenerated CAE/INP/JNL/log count: 15 each
- N24 regenerated CAE/INP/JNL/log count: 15 each
- N40 regenerated CAE/INP/JNL/log count: 15 each
- final total CAE count: 60
- final total INP count: 60

Final cooling controls retained:

- `timePeriod = 1200.0`
- `initialInc = 0.01`
- `maxInc = 60.0`
- `maxInc` is maximum increment size, not `maxNumInc = 60`.

Post-generation checks:

- `check_probe60_generated_inps.py`: `WARNING_PROBE60_PARTIAL_INPS_EXIST`
- warning reason: all 60 CAE/INP/JNL/log files exist, but N12 solver outputs are
  intentionally preserved in the active N12 folders.
- generated INP mesh-section checker:
  `PASS_N16_N24_N40_INP_MESH_SECTIONS_READY`
- N16/N24/N40 checked INPs: 45
- every regenerated N16/N24/N40 INP contains `*Node`, `*Element`, nonzero mesh
  entries, scan/cool steps, body heat flux entries, and visible final cooling
  controls.
- active N16/N24/N40 ODB/SIM/STA/DAT/MSG/LCK count after regeneration: 0

Custom abqjobpilot command files:

- full 60 command file validation: pass, 60 commands, all referenced INPs exist
- N16/N24/N40-only command file validation: pass, 45 commands, N12 excluded
- N16/N24/N40-only command file:
  `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_manual_probe60_handoff\variable_N_probe60_abqjobpilot_commands_N16_N24_N40_READY_TO_RUN.txt`

The user's custom abqjobpilot N16/N24/N40 command file is ready for
user-controlled enqueue, but was not executed by Codex.

Forbidden-action confirmation:

- no solver analysis was run by Codex
- no datacheck was run
- no job submission was run
- no abqjobpilot command was executed
- no enqueue command was executed
- no ODB was opened
- no new ODB/SIM/STA/DAT/MSG/LCK was produced under N16/N24/N40

## N24_A07 Incomplete Solver Case Isolation Before Single-Case Rerun

Previous solver completion audit:

- verdict: `FAIL_PROBE60_SOLVER_COMPLETION_INCOMPLETE`
- completion count: 59/60
- incomplete case: `N24_A07_regular_jump_coprime`
- target job: `J2D_N24_N24_A07_regular_jump_coprime`

Active process and lock audit:

- no target-case Abaqus solver/job process was found
- `ABAQUSLM.exe` license service was present, but no process command line
  referenced the N24_A07 case directory or job name
- `.lck` was present and treated as stale incomplete-output evidence

N24_A07 text-output diagnostic:

- `.sta` success marker present: no
- `.sta` termination/error marker present: no
- fatal/error marker present in `.sta/.msg/.dat`: no
- last visible `.sta` step/increment:
  `27    34   1     0     2     2  47.0       0.200      0.001111`

Diagnostic outputs:

- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_cae_probe60_generation\N24_A07_incomplete_case_diagnostic.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_cae_probe60_generation\N24_A07_incomplete_case_diagnostic.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_cae_probe60_generation\N24_A07_incomplete_case_diagnostic.md`

Archived incomplete N24_A07 solver outputs:

- archive path:
  `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_cae_probe60_generation\archived_incomplete_N24_A07_before_single_case_rerun_20260612_014308`
- archive manifest:
  `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_cae_probe60_generation\archived_incomplete_N24_A07_before_single_case_rerun_20260612_014308\ARCHIVE_MANIFEST.csv`
- moved file count: 11
- active N24_A07 ODB/SIM/STA/DAT/MSG/LCK count after archive: 0
- preserved files: INP, CAE, JNL, and scan-order JSON

One-case custom abqjobpilot rerun command:

- command file:
  `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_manual_probe60_handoff\N24_A07_regular_jump_coprime_SINGLE_RERUN_abqjobpilot_command.txt`
- validation: pass

ODB postprocessing and teacher validation remain blocked until N24_A07 completes
successfully.

The custom abqjobpilot rerun command was generated for user-controlled
execution but was not executed by Codex.

## Exact Manual Next Command

Review the object-name audit before attempting generation:

```powershell
notepad "E:\Projects\RL-LAM-ScanOpt\outputs\stage3_cae_probe60_generation\base_jnl_object_name_audit.md"
```

Do not run all-case Abaqus generation until the heat-load remapping from
`scan_order` entries to Abaqus step/load/set objects is explicitly implemented
and re-audited as `SAFE_AUTOMATIC_SCAN_ORDER_MAPPING_READY`.

## Guardrails

- No Abaqus solver jobs.
- No datacheck.
- No ODB opened.
- No abqjobpilot execution.
- No enqueue commands.
- No teacher validation.
- No claims of variable-N generalisation, trained-policy success, or physical
  superiority.

## What Was Not Done

- No CAE/INP/JNL model artifacts were generated.
- No Abaqus command was executed.
- No solver job was submitted.
- No abqjobpilot command was executed.
- No ODB postprocessing was performed.

## Outputs

- `E:\Projects\RL-LAM-ScanOpt\scripts\stage3\preflight_probe60_generation_inputs.py`
- `E:\Projects\RL-LAM-ScanOpt\scripts\stage3\audit_probe60_base_jnl_names.py`
- `E:\Projects\RL-LAM-ScanOpt\scripts\stage3\generate_probe60_from_sanity_base_nogui.py`
- `E:\Projects\RL-LAM-ScanOpt\scripts\stage3\check_probe60_generated_inps.py`
- `E:\Projects\RL-LAM-ScanOpt\scripts\stage3\extract_probe60_heat_mapping_from_jnl.py`
- `E:\Projects\RL-LAM-ScanOpt\scripts\stage3\check_probe60_pilot_inp_heat_order.py`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_cae_probe60_generation\probe60_generation_commands.ps1`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_cae_probe60_generation\README_STOP_BEFORE_ABQJOBPILOT.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_cae_probe60_generation\probe60_heat_mapping_config_TEMPLATE.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_cae_probe60_generation\probe60_PILOT_ONLY_generation_instructions.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_cae_probe60_generation\heat_mapping_jnl_snippets.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_cae_probe60_generation\heat_mapping_jnl_snippets.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_cae_probe60_generation\heat_mapping_inferred_patterns.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_cae_probe60_generation\pilot_inp_heat_order_check.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_cae_probe60_generation\pilot_inp_heat_order_check_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_cae_probe60_generation\probe60_generation_preflight.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_cae_probe60_generation\probe60_generation_preflight_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_cae_probe60_generation\base_jnl_object_name_audit.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_cae_probe60_generation\base_jnl_object_name_audit.md`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_cae_probe60_generation\base_jnl_object_name_audit_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_cae_probe60_generation\probe60_generator_dry_run_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_cae_probe60_generation\probe60_generator_dry_run_cases.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_cae_probe60_generation\probe60_generated_file_check.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_cae_probe60_generation\probe60_generated_file_check_summary.json`
