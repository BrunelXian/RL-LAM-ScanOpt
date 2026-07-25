# PPO Stage E CAE/INP Handoff Report

## 1. Purpose

Stage E converts the Stage D PPO-only batch32 scan-order candidates into Abaqus CAE/INP teacher-validation cases for later user-controlled solver execution.

This stage does not run Abaqus solver jobs, datacheck, abqjobpilot, enqueue, ODB extraction, or teacher-metric ingestion.

## 2. Input PPO Batch32

- Selected PPO batch: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\ppo_candidate_generation\selected_batch32\ppo_policy_only_candidate_batch32.csv`
- Scan-order JSON source: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\ppo_candidate_generation\selected_batch32\scan_orders`
- Batch name: `stage3_ppo_policy_only_batch32_v01`
- Case root: `E:\Projects\RL-LAM-ScanOpt\cae_model\stage3_ppo_policy_only_batch32_v01`

## 3. PPO Checkpoint Source

The selected orders trace to PPO checkpoint inference from:

`E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\ppo_training\checkpoints\maskable_ppo_lam_scan_order_final.zip`

## 4. Candidate-Source Boundary

All 32 cases were generated from the frozen Stage D selected PPO batch. No scan orders were modified, repaired, mutated, replaced, or hand-designed in Stage E.

Each selected candidate keeps:

- `candidate_source = PPO_checkpoint_inference`
- `teacher_validated = false`
- `abaqus_validated = false`

## 5. Preflight Verdict

Preflight verdict: `PASS_STAGEE_PPO_BATCH32_PREFLIGHT_READY`

Preflight outputs:

- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageE_teacher_validation_handoff\checks\stageE_preflight_ppo_batch32.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageE_teacher_validation_handoff\checks\stageE_preflight_ppo_batch32_summary.json`

## 6. Manifest Path

- Case manifest CSV: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageE_teacher_validation_handoff\manifest\stageE_ppo_batch32_case_manifest.csv`
- Case manifest JSON: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageE_teacher_validation_handoff\manifest\stageE_ppo_batch32_case_manifest.json`

## 7. CAE/INP Generation Method

A dedicated Stage E Abaqus CAE noGUI script generated the cases:

`E:\Projects\RL-LAM-ScanOpt\stage3_ppo_rl_lam_fea_addendum_v01\scripts\generate_stageE_ppo_batch32_from_sanity_base_nogui.py`

The script opened the appropriate sanity-base CAE by N, applied the frozen PPO scan order, generated scan/cool thermal step sequences, appended final cooling, saved CAE files, and wrote INPs. It did not submit jobs.

## 8. Counts By N

| N | CAE count | INP count |
|---:|---:|---:|
| 12 | 8 | 8 |
| 16 | 8 | 8 |
| 24 | 8 | 8 |
| 40 | 8 | 8 |
| Total | 32 | 32 |

## 9. Final Cooling Controls

Final cooling controls were verified in the generated INPs:

- `timePeriod = 1200.0`
- `initialInc = 0.01`
- `maxInc = 60.0`

`maxInc = 60.0` is maximum increment size, not `maxNumInc = 60`.

## 10. Output Field Request Status

The generated INP audit verified output availability for:

- `U`
- `PEEQ`
- `S`
- thermal nodal output as Abaqus `NT` / requested `NT11` equivalent for later temperature extraction

## 11. INP/CAE Audit Verdict

INP/CAE audit verdict: `PASS_STAGEE_PPO_BATCH32_CAE_INP_READY_FOR_USER_CONTROLLED_SOLVER`

Audit outputs:

- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageE_teacher_validation_handoff\checks\stageE_ppo_batch32_generated_inp_check.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageE_teacher_validation_handoff\checks\stageE_ppo_batch32_generated_inp_check_summary.json`

All 32 individual case checks passed. No solver output files were found under the case root.

## 12. Solver Command-File Path

Command file prepared for later user-controlled execution:

`E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageE_teacher_validation_handoff\commands\stageE_ppo_batch32_abqjobpilot_commands_READY_TO_RUN.txt`

The file contains 32 `enqueue --inp ... --cpus 14` commands. It was not executed.

## 13. Duplicate / Recovery Anchor Note

One selected PPO N12 order duplicates an existing combined552 order hash and is retained by instruction as a recovery anchor:

- `PPOV01_N12_B02_surrogate_top`
- `duplicate_vs_combined552 = true`
- `duplicate_role = recovery_anchor`

This is not a failure and no replacement was made.

## 14. Confirmations

- No solver run.
- No datacheck run.
- No ODB opened.
- No ODB metrics extracted.
- No abqjobpilot or enqueue command executed.
- No teacher validation claimed.
- No Stage D PPO selected orders were modified.

## 15. Stage F / User-Controlled Solver Execution Plan

Stage F should be a separate user-controlled Abaqus teacher-validation stage:

1. Review the generated command file.
2. Execute the 32 solver jobs under user control.
3. Confirm solver completion status for all 32 cases.
4. Extract teacher metrics from ODB outputs: `U`, `PEEQ`, `S`, and temperature output.
5. Ingest metrics and compare PPO batch32 against combined552 and relevant baselines.

## 16. Claim Boundary

Safe after Stage E:

- Thirty-two PPO-generated scan-order candidates were converted into Abaqus CAE/INP teacher-validation cases.
- These cases are ready for user-controlled solver execution.

Not safe after Stage E:

- PPO candidates are teacher validated.
- PPO improves U2/PEEQ/SurfaceT.
- PPO outperforms baselines.
- PPO is the final physical optimiser.

## 17. Verdict

`PASS_STAGEE_PPO_BATCH32_READY_FOR_USER_CONTROLLED_ABAQUS_TEACHER_VALIDATION`
