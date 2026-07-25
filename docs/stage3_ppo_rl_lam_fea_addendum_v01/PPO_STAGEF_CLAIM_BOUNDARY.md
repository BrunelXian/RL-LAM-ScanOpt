# PPO Stage F Claim Boundary

## Safe Claims

- The Stage F pre-submit audit passed for all 32 PPO-only INP cases.
- Codex attempted to execute exactly the 32 user-approved `enqueue` commands from the approved command file.
- The current Codex shell environment did not have `enqueue` available, so no jobs were submitted from this environment.
- No scan-order JSON, INP, CAE, or JNL files were modified.
- No ODB was opened.
- No ODB metric extraction was run.
- No teacher metrics are claimed.

## Not Safe Until Stage G

- PPO candidates have teacher metrics.
- PPO improves U2, PEEQ, SurfaceT, or any physical metric.
- PPO outperforms baselines.
- PPO is physically validated.
- Solver completion is achieved.

## Required Next Stage

Run the 32 approved commands in the user's abqjobpilot-enabled environment, then confirm solver completion. After 32/32 cases complete successfully, Stage G should open ODBs read-only and extract U, PEEQ, S, and NT/NT11 metrics.
