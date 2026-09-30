# CAE STATUS

## Final archive decision — 2026-09-30

ARCHIVED: tested CAE production route. Agent diagnostic workflow: positive result within the bounded experiment.
See `status/STAGE4_ARCHIVED.md` and `README.md`. No further model/solver/training work; Stage 5 not started.
Documentation Run: run_006; Archive commit 8e02d2c62494216f939df280c430151c592d21e3 pushed to stage4-cae-failure-agent-diagnostics; closure metadata follows on that branch.
Earlier entries below are retained as historical state and do not authorize resumed work.

Latest module run: `run_005` — stage4_cae / diagnose_n32_numerical_stability_ablation.
Execution: completed. Diagnostic validation: passed (evidence/isolation scope only).
Handoff: ready for diagnostic report review only. Formal model ready: false.
Final diagnostic verdict: NO_COMPLETE_NUMERICAL_RESCUE_IN_TESTED_MATRIX.
Summary: `run/run_005/summary.md`.
Results: `CAE/Reports/N32/numerical_stability/run_005/matrix_results.json`.

A baseline: FAILED, Step 40 / event 10 reactivation geometry check before equilibrium;
721/723/725 small/negative area. BASELINE_FAILURE_CONFIRMED.
B ALE: FAILED, Step 42 / event 10 cool, too many attempts; Step 40 passed.
ALE_GEOMETRY_IMPROVED_BUT_NOT_SOLVED.
C stabilization: FAILED, Step 6 increment 2. STABILIZATION_NO_MEANINGFUL_IMPROVEMENT.
D combined: FAILED, Step 6 increment 2. ALE_STAB_NO_MEANINGFUL_IMPROVEMENT.
C/D final converged ALLSD/ALLIE = 0.4874% / 0.1103%; neither reached Step 40.

ALE supported for actual Standard coupled/CPE4T procedure with region restrictions.
This run used 256 always-active pool elements, excluding removed/reactivated deposits.
B/D use identical ALE settings; C/D use identical 23-step stabilization settings.
One numerical configuration per mechanism; no parameter sweeps.
B attempt001 was invalidated and stopped due ALE step reset; valid B uses suffix _02.
A native .log absent; native MSG/DAT/STA and transcribed console evidence preserved.

Baseline input: `CAE/Models/N32/Generated/diagnostic/stage4_N32_diag_identity_order.cae`.
Derived models: `CAE/Models/N32/Generated/diagnostic/numerical_stability/run_005/`.
Jobs: `CAE/Jobs/run_005/`; scripts: `CAE/scripts/debug/`.
Baseline/master/frozen YAML SHA256 unchanged from run start. Stage 3 inventory unchanged.
RL_STATUS/ODB_STATUS unchanged. No real CASE, S4C000001 unallocated, case_ids=[].
Solver/ODB use was explicitly authorized for this numerical diagnostic; no teacher metrics.

Generator: `CAE/scripts/generate/stage4_generate_n32_deposition_model.py` — unchanged.
Audit: `CAE/scripts/audit/stage4_audit_n32_deposition_model.py` — unchanged.
Generator build/mapping/lifecycle status remains as validated in run_003 and run_004;
that validation does not establish solver convergence. Custom-order model from run_004
was not the input to this diagnostic. See those run summaries for build evidence.

Current blocker: no tested configuration completed the full 32-event analysis.
Next action: stopped after A/B/C/D; review the diagnostic report. No further numerical
repair, solver, training or research work is scheduled. No universal architecture
viability conclusion follows from this single restricted matrix.
Authority remains the deposition-event template, parameter register and sanity YAML.
