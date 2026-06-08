# Stage 2 Evidence Package

Stage 2 final conclusion:

A teacher-validated, RL-informed scan-order optimisation framework was developed for a 32-track LDED benchmark. GNN/RL showed clear teacher-validated advantage over 9/10 labelled early full-32 engineering baselines. The final recommended physical hierarchy is U2-first, PEEQ safety, and SurfaceT-secondary constrained search. Transformer was closed as a negative ablation. Masked probe40 was frozen as generalisation-boundary evidence and should not scale to 400 before per-mask guard calibration. Stage 3 should focus on Variable-N Graph Pointer RL Policy.

Use conservative wording:

- “teacher-validated advantage over 9/10 labelled early baselines”
- not “beats all baselines”
- not “global optimum”
- not “arbitrary-N generalisation solved”
- not “masked transfer succeeded”

## Key Documents

- [Stage 2 Final Summary](STAGE2_FINAL_SUMMARY.md)
- [Claim Boundary](STAGE2_CLAIM_BOUNDARY.md)
- [Run Index](STAGE2_RUN_INDEX.md)
- [Key Results Table](STAGE2_KEY_RESULTS_TABLE.csv)
- [Stage 3 Handoff](STAGE2_STAGE3_HANDOFF.md)

## GitHub Packaging Notes

This documentation package is intended for GitHub publication. It includes small Markdown and CSV summaries only. Abaqus ODB/CAE/SIM/STA/MSG/DAT and other heavy solver files should remain excluded.
