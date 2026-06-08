# Stage 2 Final Evidence Consolidation Report

Generated: `2026-06-08T14:59:09`

## Output Locations
- Public docs folder: `D:\Projects\RL-LAM-ScanOpt\docs\stage2`
- Evidence package folder: `D:\Projects\RL-LAM-ScanOpt\docs\stage2_final_evidence_package_v01`

## Files Created
- `docs/stage2/STAGE2_FINAL_SUMMARY.md`
- `docs/stage2/STAGE2_CLAIM_BOUNDARY.md`
- `docs/stage2/STAGE2_RUN_INDEX.md`
- `docs/stage2/STAGE2_EVIDENCE_MANIFEST.csv`
- `docs/stage2/STAGE2_KEY_RESULTS_TABLE.csv`
- `docs/stage2/STAGE2_GITHUB_README_DRAFT.md`
- `docs/stage2/STAGE2_STAGE3_HANDOFF.md`
- `docs/stage2_final_evidence_package_v01/stage2_final_evidence_consolidation_report.md`

## Missing Expected Reports
`none`

## Scientific Consolidation
The final Stage 2 package preserves these conclusions:

1. The full 32-track search space is `32! ≈ 2.63 × 10^35`.
2. Stage 2 is teacher-guided, not brute force.
3. Final objective hierarchy is U2 primary, PEEQ safety, SurfaceT secondary, Gradient/Mises/internal diagnostics.
4. Old multi-weight residual-stress composite was demoted.
5. Unconstrained SurfaceT-first is not the final route.
6. GNN/RL is successful policy-learning / agent-feasibility evidence.
7. GNN/RL has teacher-validated advantage over 9/10 labelled early full-32 baselines.
8. Do not claim all-10 baseline superiority until `smartscan_proxy_variance` is validated or excluded.
9. SurfaceT has diagnostic ranking signal; Gradient remains weak.
10. Transformer did not beat feature-based ExtraTrees under current teacher data.
11. Masked probe40 defines a boundary, not a scale-up success.
12. Stage 3 should move to Variable-N Graph Pointer RL Policy.

## GitHub Upload Recommendation
Include scripts, docs, small CSV summaries, and reports. Exclude ODB/CAE/large INP/Abaqus temporary outputs/cache folders. Existing `.gitignore` already covers most Abaqus files; verify `outputs/**/large_raw/` and `cae_models/**/large_raw/`.

## Guardrails
- Models trained: `False`
- Candidates generated: `False`
- CAE/INP/JNL generated: `False`
- Abaqus jobs submitted: `False`
- Datacheck run: `False`
- ODB files opened: `False`
- Teacher modules modified: `False`

## Final Verdict
`PASS_STAGE2_EVIDENCE_PACKAGE_CREATED`
