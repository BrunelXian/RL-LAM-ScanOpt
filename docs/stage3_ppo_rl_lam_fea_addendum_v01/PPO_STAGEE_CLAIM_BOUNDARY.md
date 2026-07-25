# PPO Stage E Claim Boundary

## Safe After Stage E

- Thirty-two PPO-generated candidates were converted into Abaqus CAE/INP teacher-validation cases.
- These cases are ready for user-controlled solver execution because Stage E preflight and generated INP/CAE checks passed.
- The generated command file is a handoff artifact only.

## Not Safe After Stage E

- PPO candidates are teacher validated.
- PPO improves U2/PEEQ/SurfaceT.
- PPO outperforms baselines.
- PPO is the final physical optimiser.
- Any physical metric claim about the PPO batch32.

## Stage F Required

Stage F must be separate from this handoff stage:

- User executes solver jobs.
- Solver completion is confirmed for 32/32 cases or failures are documented.
- ODB metrics are extracted for `U`, `PEEQ`, `S`, and temperature output.
- Teacher metrics are ingested.
- PPO batch32 is compared against combined552 and baselines.

## Negative Controls

Stage E performed no solver, no datacheck, no ODB opening, no ODB metric extraction, no abqjobpilot execution, and no enqueue execution.
