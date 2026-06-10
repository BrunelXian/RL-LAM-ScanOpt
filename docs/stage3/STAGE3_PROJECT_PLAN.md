# Stage 3 Project Plan

Stage 3 upgrades the fixed-32 GNN/RL scan-order policy evidence into a Variable-N Graph Pointer RL Policy framework. The purpose is to test whether fixed-32 policy principles transfer across track counts, not to continue fixed-32 leaderboard tuning.

## Scientific Goal

- Train/test design: `N_train = {16, 32}` and `N_test = {24, 40}`.
- Evaluation should use within-N ranking and normalized improvement.
- Do not use one absolute full-32 U2 guard across all N.
- Do not claim arbitrary-N generalisation.

## Physical Hierarchy

- Primary: U2 / warpage.
- Safety: PEEQ.
- Secondary: SurfaceT inside the U2/PEEQ feasible or near-feasible region.
- Diagnostics: Gradient, Mises, and internal tensile stress.

## First Runs

- run_01: Stage 2 handoff import.
- run_02: fixed-32 policy novelty audit.
- run_03: variable-N graph feature builder.
- run_04: variable-N baseline generator.
- run_05: Graph Pointer Policy prototype dry-run.
- run_06: N=24/N=40 candidate design.
- run_07: FEA handoff only, no submission.
- run_08: read-only ODB teacher validation after external Abaqus completion.
- run_09: evidence freeze and claim boundary.

## Operating Rules

Stage 3 starts with documentation, feature design, baselines, and dry-run policy mechanics. Initial work must remain small and GitHub-safe: no Abaqus jobs, no datacheck, no ODB opening, no CAE/INP/JNL generation, no model training, and no candidate generation beyond documentation placeholders.
