# PPO Final Expansion Stage U - CAE/INP Handoff Report

## Purpose

Stage U converted the repaired PPO final-expansion 224 selected candidates into Abaqus CAE/INP cases for later user-controlled solver submission. No physical validation is claimed.

## Input Stage T Selected Master CSV

E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_expansion_224_to_320\selected_candidates\PPO_FINAL_EXPANSION_224_SELECTED_MASTER.csv

## Stage T Repair Before Stage U

Same-N exact scan-order duplicates were not downgraded. A repair step replaced physical duplicates using existing Stage T rollout-pool candidates before Stage U generation.

- Duplicate groups found: 
9
- Replacement count: 
21
- Repair report: 
E:\Projects\RL-LAM-ScanOpt\docs\stage3_ppo_final_expansion_224_to_320\PPO_FINAL_EXPANSION_224_STAGE_T_REPAIR_REPORT.md
- Repair manifest: 
E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_expansion_224_to_320\stageT_repair_duplicate_orders_manifest.json

## Candidate Counts

- N12: 32
- N16: 32
- N24: 80
- N40: 80
- Total: 224

## Batch Allocation

- final_expansion_batch01: N12=16, N16=16
- final_expansion_batch02: N12=16, N16=16
- final_expansion_batch03: N24=32
- final_expansion_batch04: N24=32
- final_expansion_batch05: N24=16, N40=16
- final_expansion_batch06: N40=32
- final_expansion_batch07: N40=32

## Base CAE Files

- N12: 
E:\Projects\RL-LAM-ScanOpt\cae_model\12track_full\sanity_base\12track_sanity_base.cae
- N16: 
E:\Projects\RL-LAM-ScanOpt\cae_model\16track_full\sanity_base\16track_sanity_base.cae
- N24: 
E:\Projects\RL-LAM-ScanOpt\cae_model\24track_full\sanity_base\24track_sanity_base.cae
- N40: 
E:\Projects\RL-LAM-ScanOpt\cae_model\40track_full\sanity_base\40track_sanity_base.cae

## Case Root

E:\Projects\RL-LAM-ScanOpt\cae_model\stage3_ppo_final_expansion_224_to_320_batch224_v01

## Preflight Verdict

PASS_STAGEU_FINAL_EXPANSION_CAE_INP_PREFLIGHT_READY

Preflight was rerun after duplicate repair and passed with no blockers or warnings.

- Preflight CSV: 
E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_expansion_224_to_320\stageU_CAE_INP_handoff\checks\stageU_preflight_CAE_INP.csv
- Preflight summary: 
E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_expansion_224_to_320\stageU_CAE_INP_handoff\checks\stageU_preflight_CAE_INP_summary.json

## Case Manifest

E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_expansion_224_to_320\stageU_CAE_INP_handoff\manifest\stageU_final_expansion_case_manifest.csv

## CAE/INP Generation Summary

Abaqus CAE noGUI was executed for generation only.

- Generated CAE count: 224
- Generated INP count: 224
- N12: CAE=32, INP=32
- N16: CAE=32, INP=32
- N24: CAE=80, INP=80
- N40: CAE=80, INP=80

Generation summary: 
E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_expansion_224_to_320\stageU_CAE_INP_handoff\manifest\stageU_final_expansion_generation_summary.json

## CAE/INP Audit Verdict

PASS_STAGEU_FINAL_EXPANSION_CAE_INP_READY_FOR_USER_CONTROLLED_SOLVER

Audit confirmed 224/224 CAE/INP cases passed text checks, each batch has 32 INPs, final cooling controls are visible, output fields U/PEEQ/S/NT11 are visible, and no ODB/SIM/STA/DAT/MSG/LCK solver outputs exist.

- Audit CSV: 
E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_expansion_224_to_320\stageU_CAE_INP_handoff\checks\stageU_generated_CAE_INP_audit.csv
- Audit summary: 
E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_expansion_224_to_320\stageU_CAE_INP_handoff\checks\stageU_generated_CAE_INP_audit_summary.json

## Final Cooling Verification

Final cooling controls were verified in generated INPs: step_final_cooling, timePeriod 1200.0, initialInc 0.01, maxInc 60.0.

## Output Field Verification

Generated INPs include output requests for U, PEEQ, S, and NT/NT11 where detectable.

## Solver Command Files

- E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_expansion_224_to_320\stageU_CAE_INP_handoff\commands\stageU_final_expansion_batch01_abqjobpilot_commands_READY_TO_RUN.txt
- E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_expansion_224_to_320\stageU_CAE_INP_handoff\commands\stageU_final_expansion_batch02_abqjobpilot_commands_READY_TO_RUN.txt
- E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_expansion_224_to_320\stageU_CAE_INP_handoff\commands\stageU_final_expansion_batch03_abqjobpilot_commands_READY_TO_RUN.txt
- E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_expansion_224_to_320\stageU_CAE_INP_handoff\commands\stageU_final_expansion_batch04_abqjobpilot_commands_READY_TO_RUN.txt
- E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_expansion_224_to_320\stageU_CAE_INP_handoff\commands\stageU_final_expansion_batch05_abqjobpilot_commands_READY_TO_RUN.txt
- E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_expansion_224_to_320\stageU_CAE_INP_handoff\commands\stageU_final_expansion_batch06_abqjobpilot_commands_READY_TO_RUN.txt
- E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_expansion_224_to_320\stageU_CAE_INP_handoff\commands\stageU_final_expansion_batch07_abqjobpilot_commands_READY_TO_RUN.txt

All-batch command file:
E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_expansion_224_to_320\stageU_CAE_INP_handoff\commands\stageU_final_expansion_all224_abqjobpilot_commands_READY_TO_RUN.txt

## Confirmations

- No solver was run.
- No datacheck was run.
- No abqjobpilot/enqueue command was executed.
- No ODB was opened.
- No ODB extraction was run.
- No teacher validation is claimed.
- No scan-order JSON was manually mutated.
- No PPO training, surrogate training, or candidate generation was run in Stage U.

## Claim Boundary

Stage U generated and audited CAE/INP handoff files only. No physical teacher metrics are available yet for these 224 cases.

Claim boundary:
E:\Projects\RL-LAM-ScanOpt\docs\stage3_ppo_final_expansion_224_to_320\PPO_FINAL_EXPANSION_STAGEU_CLAIM_BOUNDARY.md

## Next Action

User may manually review the READY_TO_RUN command files and submit jobs batch-by-batch with custom abqjobpilot if desired.

## Verdict

PASS_STAGEU_FINAL_EXPANSION_CAE_INP_READY_FOR_USER_CONTROLLED_SOLVER
