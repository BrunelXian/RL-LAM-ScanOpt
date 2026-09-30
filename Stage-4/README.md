# RL-LAM-ScanOpt Stage 4

## Status

**ARCHIVED — tested CAE route. Positive diagnostic-workflow outcome.**
Archive decision recorded on 2026-09-30; no Stage 5 work is started by this update.

**CAE outcome:** The tested Stage 4 deposition CAE architecture did not reach the numerical robustness required for high-throughput teacher generation.

**Agent outcome:** The Agent/Run workflow successfully detected, localized, tested, and documented the failure mechanism within the tested diagnostic scope. This is positive evidence for an engineering diagnostic workflow, not a claim of universal solver repair or independent proof of every upstream physical cause.

> Stage 4 was archived as a production CAE modelling route, while simultaneously providing positive evidence that the Agent workflow can detect, diagnose, test, and document CAE failure mechanisms in a controlled manner.

## What Stage 4 attempted

Stage 4 extended a single-drop sanity model into pre-meshed, progressive material activation using ModelChange and NLGEOM, with each deposition event following preheat → react → coheat → cool. The goal was an experiment-oriented FEA teacher model for later surrogate/RL work.

[run_003](run/run_003/summary.md) established the identity-order generator and validated step, load, surface and activation lifecycles; [run_004](run/run_004/summary.md) validated a custom-order build. These were successful **build/state audits**, not solver convergence validation. The subsequent [run_005](run/run_005/summary.md) numerical diagnostic did not yield a complete 32-event analysis. Stage 4 teacher production and RL/surrogate retraining were not reached.

## Why the CAE route was archived

The baseline failure was more specific than generic convergence difficulty: Step 40 (`Step-10-react`), increment 1, failed during element-reactivation geometry checking **before Newton equilibrium iteration**. `INST_PLATE.721`, `.723` and `.725` were reported as very small/negative-area elements, with `ErrElemAreaSmallNegZero`. Earlier distortion, material/plasticity and convergence warnings provided context rather than replacing this direct fatal classification.

No tested A/B/C/D configuration completed the analysis. On that bounded evidence, the project stops investing in this tested representation as its production teacher route. This is an engineering stop decision, not a proof that every pre-meshed ModelChange architecture, ALE domain or numerical setting must fail.

## What Stage 4 demonstrated about the Agent

Stage 4 did not validate the selected deposition CAE architecture, but it provided an important validation of the Agent workflow itself **within this engineering diagnostic task**. The Agent was not limited to launching Abaqus jobs or reporting solver failure. Through the project Run protocol, it supported a structured engineering-debugging loop from failure detection to evidence-based model rejection, within user-defined scope and stopping constraints.

The evidence chain supports:

failure detected → solver evidence collected → failure phase localized → direct cause separated from upstream hypotheses → controlled diagnostic variants created → single-factor interventions and a two-factor matrix executed → results compared → unsupported interpretations rejected → stop criterion reached → tested modelling route archived with traceable evidence.

The working hypothesis was that **pre-reactivation geometry deterioration was a major contributor to the fatal ModelChange failure**. ALE's limited success supports a geometric contribution; it does not prove a unique upstream physical mechanism. Shared-node or dormant-node movement remains a mechanism hypothesis unless directly established by connectivity and node-history evidence. The present archive does not claim that proof.

### The Agent Did Not Fail

The Stage 4 termination should not be interpreted as an Agent failure. Within this bounded task, the workflow successfully:

- preserved the run_005 diagnostic baseline and independently derived numerical variants;
- read and structured solver evidence, distinguishing fatal errors from upstream warnings;
- localized failure to a specific solver phase and formed testable hypotheses;
- maintained variable isolation across A/B/C/D and collected comparable diagnostics;
- detected an invalid ALE lifecycle implementation, stopped it, and retained its evidence;
- rejected incomplete numerical rescue as insufficient for production readiness;
- stopped after the specified matrix and supported the present traceable archive decision.

The Agent localized the direct failure mechanism, constructed evidence-supported hypotheses, and tested them through controlled numerical experiments. These records demonstrate practical failure diagnosis rather than merely job execution. They do not establish general autonomous competence or an ability to solve arbitrary Abaqus failures.

### Evidence-based termination

run_005 used one numerical configuration per mechanism: no ALE parameter sweep, stabilization-factor sweep, mesh sweep, material-property tuning or uncontrolled convergence tuning. The invalid first ALE attempt was corrected only to enforce the intended step lifecycle; it was excluded from the comparison and was not a new parameter configuration.

When all four valid arms failed to provide a complete production-capable solution, the diagnostic search stopped. This **bounded diagnostic search** is a governance result: failure did not trigger an indefinite sequence of unstructured reruns. The present user-directed archive decision follows that evidence; it is not retroactively presented as a fully autonomous scientific decision.

### From automation to a failure-aware CAE workflow

A minimal traditional automation pattern is:

input → run solver → solver failed → human investigates

The demonstrated Stage 4 workflow was:

input → controlled solver execution → failure detection → evidence extraction → failure classification → hypothesis generation → controlled diagnostic experiments → comparative evaluation → stop/continue decision → evidence archive

This is an **evidence-driven CAE diagnostic workflow**, demonstrated on this model and bounded test matrix. It complements engineering judgment rather than replacing all investigation.

## Controlled numerical diagnostic — run_005

B, C and D were independently derived from the same run_005 baseline. Mesh-quality intervention (ALE) and nonlinear-convergence intervention (automatic stabilization) were separated in a controlled 2 × 2 numerical diagnostic experiment.

| Arm | Intervention | Actual outcome | Interpretation |
|---|---|---|---|
| A | Baseline | FAILED at Step 40, before equilibrium iteration | Reactivation geometry failure confirmed |
| B | ALE only | Passed Step 40; FAILED at Step 42 (`Step-10-cool`) | Diagnostically informative improvement; no complete rescue |
| C | Automatic stabilization only | FAILED at Step 6 | No improvement in this implementation; never tested Step 40 |
| D | ALE + automatic stabilization | FAILED at Step 6 | No complete rescue; never tested Step 40 |

**NO_COMPLETE_NUMERICAL_RESCUE_IN_TESTED_MATRIX**.

ALE bypassed the original Step 40 negative-area fatal failure in the valid B run, supporting mesh/geometric deterioration as a genuine component of the baseline failure. B still had severe distortion and exhausted increment attempts at Step 42. It did not establish robust full-process behaviour or later reactivation success.

ALE was restricted to 256 always-active pool-band elements, excluding deposits that undergo removal/reactivation. Stabilization used the same predefined 23 steps in C/D. Their shorter runs and fewer total warnings cannot be interpreted as improvement. See the [matrix results](CAE/Reports/N32/numerical_stability/run_005/matrix_results.json) and [input-difference audits](run/run_005/artifacts/artifacts.json).

The first B attempt reset ALE after its first step. It was stopped and preserved; valid B uses `_02`. Input auditing was then strengthened to check the whole ALE lifecycle. Actual Standard logs report five smoothing sweeps, despite the API argument being one. These details limit the claim and show why inspecting actual solver execution mattered.

## Key evidence

[Published baseline log excerpts](outputs/diagnostics/stage4_run_006_baseline_log_excerpt.txt) retain the source path, hash and line numbers. Full native solver files remain local.

The [baseline native MSG](CAE/Jobs/run_005/A_baseline/stage4_diag_A_baseline_run_005.msg) and [structured A log report](CAE/Reports/N32/numerical_stability/run_005/A_logs.json) contain excessive distortion, large strain-increment/plasticity difficulties, negative eigenvalues and repeated cutbacks. Cross-step synthesis shows worsening reported isoparametric-line angles across successive reactivation stages:

| Solver step | Reported elements (INST_PLATE) | Reported angle range |
|---|---|---|
| 24 | 849, 851, 853 | 40.778–44.752° (about 40–45°) |
| 28 | 817, 819, 821, 823 | 31.135–37.688° (about 31–38°) |
| 32 | 785, 787, 789, 791 | 20.307–27.013° (about 20–27°) |
| 36 | 753, 755, 757, 759 | 8.7945–15.846° (about 9–16°) |
| 40 | 721, 725, 727 | 0.27893–4.5306° |

**These are different element groups across successive stages, not continuous tracking of the same element.** The Step 40 angle group is also not identical to the fatal area-error group (721/723/725). This is cross-step evidence synthesis, not an invented individual-element trajectory.

Direct evidence establishes the reactivation geometry fatal error and the different A/B/C/D outcomes. Upstream deformation, connectivity and physical-model explanations remain hypotheses to the extent not independently established. The surviving ALE run is evidence against both “geometry played no role” and “ALE solved the model.”

## What did not fail

| Component | Stage 4 outcome | Interpretation |
|---|---|---|
| Stage 3 RL foundation | Unchanged in the diagnostic work | Not implicated by these CAE results; RL was not tested here |
| Stage 4 RL retraining | Not reached | Teacher-model robustness blocked progression |
| Abaqus automation | Operational | Builds, jobs and diagnostics executed |
| Run protocol | Successful in this task | Preserved records, scopes and traceability |
| Failure detection | Successful | Fatal phase localized |
| Failure diagnosis | Successful within scope | Geometry failure distinguished from equilibrium convergence; upstream hypotheses bounded |
| Controlled numerical testing | A/B/C/D matrix completed | All valid arms compared; all solver runs failed |
| CAE deposition architecture | Archived | Tested implementation not robust enough for batch teacher use |
| Agent workflow | Positive result | Demonstrated failure-aware CAE diagnosis and bounded stopping |

## Lessons carried into Stage 5

Carry forward immutable-baseline discipline, explicit Run dependencies, solver-log evidence extraction, failure classification, controlled ablation, stop criteria and evidence freeze—not only RL code. Do not transfer the archived CAE representation as a validated production teacher. See [Stage 4 → Stage 5 handoff](docs/Stage4_to_Stage5_Handoff.md). This document records inheritance only; it does not launch Stage 5.

## Evidence and archive entry points

[Git archive scope](docs/Stage4_Git_Archive_Scope.md) distinguishes published text evidence from local-only CAE/ODB artifacts.

- [Archive decision and scope](status/STAGE4_ARCHIVED.md)
- [Failure analysis and Agent diagnostic performance](docs/Stage4_Failure_Analysis.md)
- [run_003: generator build/state audit](run/run_003/summary.md)
- [run_004: custom-order build and provenance notice](run/run_004/summary.md)
- [run_005: complete numerical diagnostic](run/run_005/summary.md)
- [run_005 matrix](CAE/Reports/N32/numerical_stability/run_005/matrix_results.json), [plan](CAE/Reports/N32/numerical_stability/run_005/plan.json), [safety audit](CAE/Reports/N32/numerical_stability/run_005/safety_after.json)
- [Run index](run/README.md), [archive documentation Run](run/run_006/summary.md)

Provenance boundary: run_005 used baseline SHA256 `0b612e7baac43a4f3aa076673bb057c8ecb41924d6b9786fb30beaf608a580c6`. It was preserved within that experiment, but it is not the original run_003 handoff hash; run_004 recorded the difference. “Immutable baseline” refers to the controlled experiment, not an assertion that the model never changed across all project history.

Root: `E:\Projects\Stage-4`. Stage 3: `E:\Projects\RL-LAM-ScanOpt` — READ ONLY.
`docs/` holds knowledge; `status/` current state; `run/` execution history; `CAE/`, `RL/`, `surrogate/`, `descriptors/` engineering entities; `outputs/` cross-module evidence. No engineering files were relocated by archiving.
Start from [AGENTS](AGENTS.md), [communication protocol](docs/Agent_File_Communication_Protocol.md), [architecture](docs/Stage4_Orchestration_Architecture.md) and [interface contract](docs/Stage4_Interface_Contract.md). RUN, CASE and JOB remain distinct. A/B/C/D are diagnostic arms, not allocated scientific CASEs.
Parameters remain in the [register](docs/Stage4_Parameter_Register.md) and [sanity YAML](configs/stage4_sanity.yaml), without claiming final IR calibration. Legacy Step-4 references are retained. No further solver or training work is scheduled.
