# Stage 3 Run50 Claim Boundary

Run50 generated CAE/INP/JNL handoff files for the selected Run49 stricter constrained N24/N40 batch32 only.

Safe claims:
- Generated files correspond to N24 = 16 and N40 = 16.
- No N12, N16, or N32 cases were generated.
- Generated INPs passed text, mesh, scan/cool step, heat-load, final-cooling, and N40 cooling-increment checks.
- A validated custom abqjobpilot command file was created for user-controlled submission.
- No solver job was submitted by Codex.

Unsafe claims:
- Do not claim teacher validation.
- Do not claim physical metric improvement.
- Do not claim ODB results exist.
- Do not claim solver completion.
- Do not claim RL policy training.
- Do not claim GNN-RL superiority.
- Do not claim N32 caused improvement.
- Do not claim arbitrary-N generalization.
- Do not claim abqjobpilot was executed.

Operational boundary:
- No Abaqus solver, datacheck, abqjobpilot, enqueue, ODB opening, ODB postprocessing, teacher validation, or RL training was run by Codex.
