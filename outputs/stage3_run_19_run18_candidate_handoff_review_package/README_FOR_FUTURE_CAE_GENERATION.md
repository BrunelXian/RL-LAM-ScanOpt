# README For Future CAE Generation

Run19 created human-review handoff packages for batch24 and batch28.

- No CAE/INP files exist yet.
- No abqjobpilot command is executable yet.
- The user must choose either batch24 or batch28 before CAE generation.
- CAE generation should use true variable-N models and the corrected sanity_base CAE files.

Established heat-load mapping to preserve:

- `set_body_heat_{track:02d}`
- `step_scan_{seq:02d}`
- `step_cool_{seq:02d}`
- `load_body_hflux_{seq:02d}`
- `BodyHeatFlux` magnitude `80000000000.0`

Final cooling controls to preserve:

- `step_final_cooling` duration = `1200.0`
- `initialInc = 0.01`
- `maxInc = 60.0`

The CAE module should not run solver until the user approves. Future abqjobpilot command templates must not be executed until INPs exist and pass checks.
