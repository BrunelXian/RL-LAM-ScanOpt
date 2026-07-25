# Figure 3.1-3.3 Data Tables for Section 3.1

Generated from the frozen Stage 3 evidence package:
`E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package`.

## Output files

- `figure_3_1_dataset_composition.csv`: native teacher-label counts for N12/N16/N24/N40 plus the auxiliary N32 legacy-compatible context row.
- `figure_3_2_fixed32_baseline_vs_gnn_rl.csv`: fixed-N32 baseline versus GNN/RL values supplied for the manuscript context figure.
- `figure_3_3_topk_family_occupancy.csv`: zero-filled family occupancy counts for top5 and top10 within each native N and ranking view.

## Claim boundary

Native manuscript claims are limited to within-N comparisons for N12, N16, N24, and N40. N32 is retained only as auxiliary legacy-compatible context. These tables do not encode or imply a global optimum, an arbitrary-N solution, or superiority over all known strategies.

## Family-label inference rules

Family labels for Figure 3.3 are inferred by joining each top-k row back to `FROZEN_stage3_native_combined552_RL_ready_dataset.csv` using N plus strategy/handoff/job identifiers, then scanning `strategy_name`, `handoff_strategy_name`, `dataset_source`, `candidate_family`, `generation_method`, `selection_bucket`, `priority_role`, and available original-strategy fields.

Precedence is applied in this order: `method_c`, `graph_pointer_or_gnn_rl`, `uncertainty`, `diversity`, `surrogate_guided`, `penalty_repair`, `recovery_anchor`, `engineering_baseline`, `other`. This keeps explicit uncertainty/diversity/surrogate/penalty variants from being collapsed into broader recovery or anchor batches.

Keyword rules:

- `method_c`: explicit `method_c` or method-C-inspired metadata.
- `graph_pointer_or_gnn_rl`: `graph_pointer`, `gnn`, or `hybrid_gnn` metadata.
- `uncertainty`: `uncertainty`, `model_disagreement`, `disagreement`, or calibration-oriented metadata.
- `diversity`: `diversity`, `coverage`, or `novelty` metadata unless a higher-precedence graph/GNN rule applies.
- `surrogate_guided`: `surrogate`, model-prediction, predicted, or surrogate-top metadata.
- `penalty_repair`: `penalty`, `repair`, `guard`, or `safe` metadata.
- `recovery_anchor`: `recovery`, `anchor`, `known_best`, or local-search-around-best metadata.
- `engineering_baseline`: hand-engineered/probe baseline families such as raster, odd-even, regular jumps, maximin, center/edge patterns, block interleaving, and similar native probe60 designs.
- `other`: rows not matched by the above rules.

## Sanity checks

- Native counts: N12=78, N16=78, N24=190, N40=206.
- Auxiliary N32 count in the legacy-compatible file: 332.
- Figure 3.3 covers 4 native N values x 8 ranking views x top5/top10, with all missing family labels written as zero rows.
