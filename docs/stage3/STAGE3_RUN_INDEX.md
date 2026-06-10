# Stage 3 Run Index

| Run | Name | Purpose | Execution Boundary |
| --- | --- | --- | --- |
| run_01 | Stage 2 handoff import | Import compact Stage 2 reference docs and write a manifest. | No Abaqus, no ODB, no training. |
| run_02 | Fixed-32 policy novelty audit | Freeze what Stage 2 already showed and identify novelty needed for Stage 3. | Documentation/audit only. |
| run_03 | Variable-N graph feature builder | Define and dry-check graph features for N variation. | No candidate generation beyond placeholders. |
| run_04 | Variable-N baseline generator | Specify deterministic baselines for within-N comparison. | Small metadata only. |
| run_05 | Graph Pointer Policy prototype dry-run | Validate variable-N pointer mechanics without training. | No model training. |
| run_06 | N=24/N=40 candidate design | Design transfer-test candidate sets. | Documentation placeholders only at initialization. |
| run_07 | FEA handoff only, no submission | Prepare external FEA handoff packet. | No local Abaqus job submission. |
| run_08 | Read-only ODB teacher validation | Validate teacher outputs after external Abaqus completion. | Read-only ODB after external completion only. |
| run_09 | Evidence freeze and claim boundary | Freeze evidence and final claim limits. | Documentation and manifests. |
