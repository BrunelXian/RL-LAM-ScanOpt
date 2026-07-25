# README For Future CAE Generation

Run24 created a handoff package for the selected shortlist64 active-learning candidates.

No CAE/INP files exist yet. No abqjobpilot command is executable yet.

The CAE module should generate true variable-N models using corrected sanity_base CAE files.

Established heat-load mapping:
- `set_body_heat_{track:02d}`
- `step_scan_{seq:02d}`
- `step_cool_{seq:02d}`
- `load_body_hflux_{seq:02d}`
- BodyHeatFlux magnitude `80000000000.0`

Final cooling controls must remain:
- `step_final_cooling` duration = `1200.0`
- `initialInc = 0.01`
- `maxInc = 60.0`

The CAE module should not run solver until the user approves. The future abqjobpilot command template must not be executed until INPs exist and pass checks.
