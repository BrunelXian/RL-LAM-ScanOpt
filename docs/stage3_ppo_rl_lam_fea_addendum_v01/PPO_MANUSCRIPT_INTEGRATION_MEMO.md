# PPO Manuscript Integration Memo

## Purpose

This memo describes how the current paper would change if the PPO addendum succeeds. It does not change the current Stage 3 final claim boundary.

## Methods Additions

- Add the `LamScanOrderPPOEnv` PPO scan-order environment.
- Add PPO policy architecture and training details.
- Add the surrogate reward environment trained from FEA teacher-labelled scan-order data.
- Add PPO-only candidate generation protocol.
- Add the Abaqus teacher validation protocol for PPO-only candidates.

## Results Additions

Add a PPO-only validation section. Compare PPO-only candidates against:

- Initial heuristic baselines
- Combined552 final bests and top-k candidates
- Hybrid active-learning candidates

The PPO results section should separate surrogate-environment training metrics from independent Abaqus teacher-validation metrics.

## Discussion Additions

- State that PPO is trained in a surrogate environment, not online Abaqus.
- State that Abaqus is used as independent teacher validation.
- Keep the bounded native-N claim boundary.
- Avoid claiming arbitrary-N optimisation or global optimality.
- Avoid claiming experimental validation unless physical experiments are actually performed.

## Integration Rule

The addendum should be presented as a new PPO-specific evidence chain. It should not retroactively relabel the frozen Stage 3 final evidence package as PPO-generated evidence.
