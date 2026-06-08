# Stage 2 Final Summary

## Executive Summary
Stage 2 established a teacher-validated evidence chain for scan-order optimisation without pretending to brute-force the full search space. The full 32-track search space is `32! ≈ 2.63 × 10^35`, so the scientific contribution is not exhaustive enumeration. The final Stage 2 position is a constrained physical hierarchy: U2/warpage first, PEEQ safety required, SurfaceT as secondary performance inside the feasible region, and Gradient/Mises/internal tensile stress as diagnostics.

## Research Problem
Laser additive scan-order optimisation is a sequential combinatorial decision problem. The project asks whether learning-guided policies and teacher-validated surrogate analysis can find useful scan-order families while remaining physically honest.

## Teacher-Validation Workflow
Stage 2 uses Abaqus/ODB teacher validation as the arbiter for physical claims. Surrogates, GNNs, RL policies, and Transformer models are candidate generators or diagnostic rankers unless teacher validation supports a physical claim.

## Final Metric Hierarchy
1. Primary: `U2` / vertical in-plane warpage.
2. Safety: `PEEQ`.
3. Secondary: `SurfaceT` / surface tensile residual stress inside U2/PEEQ feasible or near-feasible candidates.
4. Diagnostics: Gradient, Mises, internal tensile stress.

The old multi-weight residual-stress composite was demoted because it was confounded and because unconstrained S/SurfaceT-first selection caused U2/PEEQ safety problems. Unconstrained SurfaceT-first is not the final route. The current best route is `U2-first + SurfaceT-secondary constrained search`.

## GNN/RL Evidence
Run80 freezes GNN/RL as successful policy-learning / agent-feasibility evidence. GNN/RL demonstrates that scan-ordering can be formulated as a sequential decision problem and that learned policies or graph/node scoring can produce legal and useful candidate families. It is not the final physical optimisation engine by itself.

Run84 adds a current-metric audit: GNN/RL has teacher-validated advantage over `9/10` labelled early full-32 baselines. Use that wording. Do not claim advantage over all 10 because `smartscan_proxy_variance` is missing teacher evidence.

## SurfaceT / Gradient Evidence
Run71c consolidated `387` canonical full-field labels with `336` training-ready SurfaceT rows and `336` training-ready Gradient rows. Run72 found SurfaceT ExtraTrees leave-family-out Spearman `0.5301` and top10 overlap `6.6`, supporting a diagnostic ranking signal. Gradient Spearman `0.3820` remains weak and diagnostic.

## U2-SurfaceT Relationship
Run76 found `387` valid U2+SurfaceT rows, global Spearman `0.4939`, Pearson `0.0302`, `97` U2-pass rows, and `57` Pareto candidates. The relationship is weak/nonlinear/family-dependent, not a simple global monotonic stress-release trade-off.

## UST Probe10
Run79 postprocessed `10/10` ODBs. Teacher U2 pass was `2/10`, PEEQ pass `10/10`, combined pass `2/10`, and SurfaceT proxy-vs-teacher Spearman was `0.9515`. The best UST SurfaceT did not beat the best existing reference, so SurfaceT improvement should not be claimed.

## Transformer Ablation
Run81 showed no clear Transformer improvement over feature-based ExtraTrees: Transformer SurfaceT Spearman `0.4479` and top10 `5.8` versus ExtraTrees SurfaceT Spearman `0.5301` and top10 `6.6`.

## Masked Generalisation Boundary
Run85 froze masked probe40 evidence. The full-32 absolute U2 guard `7.8362e-05` was too restrictive or uncalibrated for masked cases: U2 pass `0/40`, PEEQ pass `28/40`, combined feasible `0/40`; best masked U2 was `4.47x` guard and median masked U2 was `5.96x` guard. Masked should not scale to 400 without per-mask guard calibration.

## What Can Be Claimed
- Stage 2 is not brute force; it is teacher-guided evidence-driven search in a huge combinatorial space.
- U2-first + SurfaceT-secondary is the final Stage 2 physical direction.
- GNN/RL is successful policy-learning / agent-feasibility evidence.
- GNN/RL has teacher-validated advantage over 9/10 labelled early full-32 baselines under current metrics.
- SurfaceT has diagnostic ranking signal; Gradient remains weak.
- Transformer did not beat ExtraTrees under current teacher data.
- Masked transfer needs per-mask guard calibration.

## What Cannot Be Claimed
- Do not claim global optimum.
- Do not claim GNN/RL is the final physical optimiser.
- Do not claim superiority over all earliest 10 baselines unless `smartscan_proxy_variance` is validated or excluded with justification.
- Do not claim SurfaceT optimisation is solved.
- Do not claim masked transfer success or scale masked to 400 now.
- Do not claim arbitrary-N generalisation is already solved.

## Why Stage 2 Should Stop Here
Stage 2 has enough evidence to define the physical objective hierarchy, freeze GNN/RL as policy-learning evidence, demote confounded objectives, and identify masked/fixed-32 limits. Further fixed-32 tuning risks diminishing returns.

## Stage 3 Handoff
Stage 3 should move to Variable-N Graph Pointer RL Policy with per-instance feasibility guards, not more fixed-32 tuning.
