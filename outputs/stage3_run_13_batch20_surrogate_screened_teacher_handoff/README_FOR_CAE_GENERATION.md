# Stage 3 Run13 Batch20 CAE Generation Handoff

Run13 created a 20-candidate handoff package from run12 surrogate-screened candidates.

These candidates are not teacher-validated. They are active-learning/diversity probes, not guaranteed improvements.

The CAE module should generate true variable-N models using the corrected N12/N16/N24/N40 `sanity_base` models. It must preserve final cooling settings:

- `step_final_cooling` duration = `1200.0`
- `initialInc = 0.01`
- `maxInc = 60.0`

The CAE module should not run the solver until the user approves.

The abqjobpilot commands in `stage3_run13_batch20_abqjobpilot_commands_TEMPLATE.txt` are template-only. Do not run them until CAE/INP generation has completed and the generated INPs have been checked.
