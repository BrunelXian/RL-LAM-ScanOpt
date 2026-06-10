# Stage 2 to Stage 3 Handoff

## Why Not Continue Fixed-32 Tuning
Fixed-32 evidence is mature enough to define the objective hierarchy and model boundaries. Additional fixed-32 tuning risks overfitting to known families.

## Why Masked Should Not Scale To 400 Now
Run85 shows the full-32 absolute U2 guard does not transfer directly to masked regimes: U2 pass `0/40`, combined feasible `0/40`, best masked U2 `4.47x` guard. Per-mask guard calibration is required first.

## Stage 3 Direction
Move to Variable-N Graph Pointer RL Policy.

## Proposed Proof of Concept
1. Variable-N graph representation for available tracks.
2. Pointer-style policy with legality masking.
3. Per-instance U2/PEEQ guard calibration.
4. SurfaceT secondary ranking only inside feasible or near-feasible region.
5. Teacher validation on small calibrated batches before scale-up.

## Claim Boundary
Stage 3 should aim to prove variable-N policy feasibility and calibrated constrained generation. It should not claim arbitrary-N physical superiority before teacher validation.
