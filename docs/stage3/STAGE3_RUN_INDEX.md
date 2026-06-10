# Stage 3 Run Index

| Run | Name | Purpose | Script | Report | Output Path | Verdict | Guardrails |
| --- | --- | --- | --- | --- | --- | --- | --- |
| run_01 | Stage 2 handoff import | Import compact Stage 2 reference docs and write a manifest. | `src/experiments/run_01_import_stage2_handoff.py` | `docs/stage3/runs/run_01_handoff_import/RUN_01_HANDOFF_IMPORT_REPORT.md` | `artifacts/manifests/stage3_run_01_manifest.json` | `PASS_STAGE3_RUN01_HANDOFF_IMPORT_READY` | No Abaqus, no ODB, no training. |
| run_02 | Fixed-32 policy novelty audit | Read-only structural audit of fixed-32 GNN/RL candidates against engineering baselines. | `src/experiments/run_02_fixed32_policy_novelty_audit.py` | `docs/stage3/runs/run_02_fixed32_policy_novelty_audit/RUN_02_POLICY_NOVELTY_AUDIT_REPORT.md` | `outputs/stage3_run_02_policy_novelty_audit/` | `PENDING_RUN_02_VALIDATION` | No Abaqus, no datacheck, no ODB, no CAE/INP/JNL, no training, read-only Stage 2. |
| run_03 | Variable-N graph feature builder | Build and audit within-N normalized graph features, masks, and sample graph states for N=16/24/32/40. | `src/experiments/run_03_variable_n_graph_feature_builder.py` | `docs/stage3/runs/run_03_variable_n_graph_feature_builder/RUN_03_VARIABLE_N_GRAPH_FEATURE_BUILDER_REPORT.md` | `outputs/stage3_run_03_variable_n_graph_feature_builder/` | `PASS_VARIABLE_N_GRAPH_FEATURE_BUILDER_READY` | No Abaqus, no datacheck, no ODB, no CAE/INP/JNL, no training, no RL candidate generation, no teacher validation. Next: `run_04_variable_n_baseline_generator`. |
| run_04 | Variable-N baseline generator | Specify deterministic baselines for within-N comparison. | TBD | TBD | TBD | planned | Small metadata only. |
| run_05 | Graph Pointer Policy prototype dry-run | Validate variable-N pointer mechanics without training. | TBD | TBD | TBD | planned | No model training. |
| run_06 | N=24/N=40 candidate design | Design transfer-test candidate sets. | TBD | TBD | TBD | planned | Documentation placeholders only at initialization. |
| run_07 | FEA handoff only, no submission | Prepare external FEA handoff packet. | TBD | TBD | TBD | planned | No local Abaqus job submission. |
| run_08 | Read-only ODB teacher validation | Validate teacher outputs after external Abaqus completion. | TBD | TBD | TBD | planned | Read-only ODB after external completion only. |
| run_09 | Evidence freeze and claim boundary | Freeze evidence and final claim limits. | TBD | TBD | TBD | planned | Documentation and manifests. |
