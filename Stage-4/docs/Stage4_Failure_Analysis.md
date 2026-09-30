# Stage 4 Failure Analysis

Archive date: 2026-09-30. Scope: the tested N32 deposition representation and run_005's fixed numerical matrix.
Sources: [run_003](../run/run_003/summary.md), [run_004](../run/run_004/summary.md), [run_005](../run/run_005/summary.md), [matrix](../CAE/Reports/N32/numerical_stability/run_005/matrix_results.json).

## CAE outcome and decision boundary

The tested deposition architecture did not achieve production-ready numerical robustness for high-throughput teacher generation. All four valid diagnostic arms failed. The project archives this route by user-directed engineering decision; this is not a universal falsification of every pre-meshed ModelChange/NLGEOM formulation.
Build/state-machine audits in run_003/run_004 passed; they did not certify solver convergence. Run_005 validation passed for execution, isolation and evidence only, with formal_model_ready=false and teacher_candidate_ready=false.

## Direct failure evidence

A failed in Step 40 / Step-10-react, increment 1, during reactivation geometry checking before any equilibrium convergence check. INST_PLATE.721/723/725 had very small/negative-area errors; ErrElemAreaSmallNegZero was reported. The final accepted step was 39. This is a specific geometry-phase failure rather than merely “the model did not converge.”

The native baseline MSG contains earlier large strain increments, plasticity difficulties, excessive distortion, negative eigenvalues and cutbacks. Reported angle ranges at Steps 24/28/32/36/40 are 40.778–44.752°, 31.135–37.688°, 20.307–27.013°, 8.7945–15.846°, and 0.27893–4.5306°. They refer to different groups, respectively 849/851/853; 817/819/821/823; 785/787/789/791; 753/755/757/759; 721/725/727. They are not a continuous history of one element. The last group differs from the area-error group.

Source: [A native MSG](../CAE/Jobs/run_005/A_baseline/stage4_diag_A_baseline_run_005.msg), [A_logs.json](../CAE/Reports/N32/numerical_stability/run_005/A_logs.json). Existing ODB diagnostic reports are referenced by run_005; no ODB was reopened for this document update.

## Hypotheses versus established facts

Pre-reactivation geometry deterioration is an evidence-supported working explanation for a major contribution to the fatal failure. Shared-node/dormant-node motion, its complete connectivity chain, and a unique upstream physical cause are not proved by the present synthesis. Initial-coordinate-plus-saved-U reconstruction is supporting evidence, not identical to the solver's internal geometry check.

## Controlled numerical test

B/C/D were independently copied from one run_005 baseline. A=baseline; B=ALE; C=automatic stabilization; D=ALE+stabilization. Input-difference and physical-snapshot audits preserved physics and isolated permitted numerical changes.
A failed at 40; B passed 40 but failed at 42 in cooling; C and D failed at 6. Verdict: **NO_COMPLETE_NUMERICAL_RESCUE_IN_TESTED_MATRIX**.
ALE used the legal 256-element, always-active pool band; it did not directly smooth removed/reactivated deposit elements. Stabilization used one fixed setting and 23 selected non-react steps. C/D never reached 40, so absence of area errors in them is not evidence of fixing that fault.

## Agent Diagnostic Performance

### Detection
The workflow identified a fatal reactivation geometry error, not just a failed process exit.

### Localization
Step 40, increment 1, before equilibrium iteration; affected area-error elements 721/723/725. This narrowed the direct failure phase.

### Evidence synthesis
Earlier distortion, strain/plasticity, cutback, eigenvalue and cross-stage angle evidence provided context. Different element groups and different warning/count definitions remained explicit.

### Hypothesis testing
The ALE/stabilization 2×2 controlled diagnostic experiment separated mesh-quality and convergence interventions. Independent baseline derivation and input-diff audits supported variable isolation.

### Falsification / refinement
ALE crossed the original Step 40 fault, supporting a geometry contribution, but still failed at Step 42. The workflow rejected “ALE solved the problem.” C/D early failures prevent strong claims about their effect on the original Step 40 geometry check. The evidence does not prove a universal architecture failure or a unique upstream physical root cause.

### Governance
One configuration per mechanism; no ALE, stabilization-factor, mesh or material sweep; no uncontrolled convergence tuning. Valid A/B/C/D were completed as a bounded experiment, then diagnostic execution stopped. The present archive decision is a traceable user-directed stop decision, not an autonomous scientific discovery claim.

### Traceability
Run summaries, manifests, actual logs, structured diagnostics, input audits and safety records preserve the evidence chain. Run_003's build PASS is not conflated with run_005 solver success. The current documentation is registered in run_006.

### Limitations and detected implementation error
B attempt001 inadvertently reset ALE after the first step. It was stopped, preserved and excluded; B_02 corrected lifecycle while retaining the same intended domain/settings. The original input diff alone had not ensured ALE persistence; the audit was strengthened. Actual Standard logs reported five mesh sweeps, not the API's one. A has no native launcher .log; its transcribed console is separately identified. These limitations belong in the record rather than being hidden by a success narrative.

## Provenance and archive scope

run_005 baseline SHA256: 0b612e7baac43a4f3aa076673bb057c8ecb41924d6b9786fb30beaf608a580c6.
run_003 handoff SHA256 differed; run_004 already recorded that historical difference. The baseline was immutable within run_005, not demonstrably across the entire project.
No new solver/ODB/teacher/RL work accompanies this archive. Numerical observations are not calibrated experimental truth. See [archive status](../status/STAGE4_ARCHIVED.md) and [handoff](Stage4_to_Stage5_Handoff.md).
