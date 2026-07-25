# PPO Stage F Solver Execution Report

## Purpose

Stage F attempted user-approved submission of all 32 Stage E PPO-only Abaqus teacher-validation cases to the user's custom abqjobpilot queue.

## User Approval

The user explicitly approved the aggressive option: submit all 32 Stage E-generated PPO-only INP cases to the custom abqjobpilot queue.

## Approved Command File

`E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageE_teacher_validation_handoff\commands\stageE_ppo_batch32_abqjobpilot_commands_READY_TO_RUN.txt`

## Pre-Submit Audit

Verdict: `PASS_STAGEF_PPO_BATCH32_READY_TO_ENQUEUE`

The audit confirmed:

- 32 non-empty commands.
- N distribution: N12=8, N16=8, N24=8, N40=8.
- Every command starts with `enqueue --inp`.
- Every command includes `--cpus 14`.
- No command includes `--gpus`.
- Every command uses `--batch stage3_ppo_policy_only_batch32_v01`.
- Every referenced INP exists and is nonzero size.
- No N32 cases.
- No active `.lck` files under the case root.
- No pre-existing ODB/SIM/STA/DAT/MSG solver outputs under the case root.

Outputs:

- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageF_solver_execution\checks\stageF_pre_submit_audit.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageF_solver_execution\checks\stageF_pre_submit_audit_summary.json`

## Enqueue Execution Summary

Codex attempted to execute exactly the 32 commands from the approved command file. No commands were edited, added, removed, or retried.

Result:

- Commands attempted: 32
- Successful command returns: 0
- Failed command returns: 32
- Failure status: `SUBMISSION_EXCEPTION`

All failures had the same cause: the current Codex shell environment does not recognize `enqueue` as an available command. Therefore, no job was submitted to abqjobpilot from this environment.

Execution outputs:

- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageF_solver_execution\logs\stageF_enqueue_execution_log.txt`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageF_solver_execution\tables\stageF_enqueue_submission_status.csv`

## Submission Count By N

Commands attempted by N:

- N12 = 8
- N16 = 8
- N24 = 8
- N40 = 8

Successful submissions by N:

- N12 = 0
- N16 = 0
- N24 = 0
- N40 = 0

## File Snapshot By N

Immediate filesystem snapshot after the failed enqueue attempts:

- N12: INP=8, CAE=8, ODB=0, SIM=0, STA=0, DAT=0, MSG=0, LCK=0
- N16: INP=8, CAE=8, ODB=0, SIM=0, STA=0, DAT=0, MSG=0, LCK=0
- N24: INP=8, CAE=8, ODB=0, SIM=0, STA=0, DAT=0, MSG=0, LCK=0
- N40: INP=8, CAE=8, ODB=0, SIM=0, STA=0, DAT=0, MSG=0, LCK=0

Outputs:

- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageF_solver_execution\tables\stageF_solver_file_snapshot_by_N.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageF_solver_execution\checks\stageF_solver_monitoring_snapshot.json`

## Immediate Completion Audit

Verdict: `WARNING_STAGEF_PPO_BATCH32_SOLVER_RUNNING_OR_PARTIAL`

Status details:

- Completed: 0/32
- ODB total: 0
- Active `.lck` total: 0
- Detected solver failures: 0
- Status count: `NOT_STARTED_OR_QUEUED = 32`

Because `enqueue` was unavailable, this warning means the cases were not started from this environment, not that Abaqus solver failures occurred.

Outputs:

- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageF_solver_execution\checks\stageF_solver_completion_audit.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageF_solver_execution\checks\stageF_solver_completion_audit_summary.json`

## Confirmations

- No scan-order JSON was modified.
- No INP/CAE/JNL was modified.
- No ODB was opened.
- No ODB metric extraction was run.
- No teacher metrics are claimed.
- No manual Abaqus solver command was run outside enqueue.
- No datacheck was run by Codex.
- No commit or push was performed.

## Next Required Step

Run the approved 32 `enqueue` commands in an environment where the user's custom abqjobpilot/enqueue command is available. After submission and solver completion, run the Stage F completion audit again. Once 32/32 ODBs are available and completion markers pass, proceed to Stage G ODB teacher-metric extraction.

## Claim Boundary

Safe claim: Stage F pre-submit audit passed, and Codex attempted all 32 approved commands exactly as listed.

Not safe to claim: solver submission succeeded, solver completion, ODB results, PPO teacher metrics, PPO physical validation, or PPO outperforming baselines.

## Verdict

`FAIL_STAGEF_PPO_BATCH32_ENQUEUE_OR_SOLVER_FAILURE`

Reason: `enqueue` is not available in the current Codex shell environment, so all 32 approved command attempts failed before reaching abqjobpilot.
