# RL-LAM-ScanOpt Work Summary

This document is a practical summary of the work completed so far in `RL-LAM-ScanOpt`.

It is written as a project-progress memo rather than a formal paper. The goal is to make it easy to understand:

- what has already been built
- what is currently runnable
- what has been tested
- what still needs follow-up

## 1. Project Positioning

`RL-LAM-ScanOpt` is a lightweight research/demo project for **scan-path optimisation inside letter-shaped additive-manufacturing regions** such as `TWI`.

The current project scope is intentionally limited:

- no FEM
- no calibrated thermal solver
- no exact residual stress prediction
- no claim of physical fidelity beyond proxy-model experimentation

The project is built to answer a focused question:

> Can scan order inside a constrained letter geometry be improved using reinforcement learning and simple thermal-quality proxy objectives?

## 2. Repository Foundation

The repository was first scaffolded into a clean modular structure:

- `core/` for geometry, thermal logic, metrics, rollout, and visualisation
- `core/planners/` for hand-crafted baselines
- `rl/` for environment, training, and evaluation
- `scripts/` for runnable experiment entrypoints
- `assets/` for saved figures and models
- `tests/` for basic validation

The documentation baseline was also created:

- `README.md`
- `PROJECT_PLAN.md`
- `requirements.txt`

## 3. Phase 1 Work Completed

### Geometry

Implemented:

- high-resolution text-mask generation for `TWI`
- downsampling to a `64x64` coarse planning grid
- mask cropping and square padding to preserve irregular letter structure

### Thermal Proxy

Implemented:

- Gaussian-like heat-source deposition
- diffusion and cooling update logic
- stable proxy thermal field updates after each scan action

### Baseline Planning

Implemented:

- raster planner
- random planner
- greedy cool-first planner

### Rollout and Metrics

Implemented:

- execution of scan plans
- final thermal field tracking
- order map generation
- coverage / thermal mean / thermal peak / thermal variance metrics

### Visualisation

Implemented:

- target mask plots
- scan-order maps
- thermal maps
- metric comparison charts

### Outcome

The baseline pipeline is runnable end-to-end and can export figures into `assets/figures/`.

## 4. RL Pipeline Work Completed

### Initial RL Environment

The project first supported a masked RL environment using cell-wise actions over the legal letter mask.

### Updated RL Environment

The environment was later refactored into a **stripe-based scan planning environment**:

- the legal letter mask is decomposed into valid vertical stripe segments
- each action chooses the next stripe
- invalid actions are masked out
- thermal updates are applied sequentially over the cells in the chosen stripe

This made the action space more consistent with process-like scan-path reasoning.

### Observation

The current RL observation contains three channels:

- `target_mask`
- `scanned_mask`
- `thermal_field`

### Reward

The current reward mixes:

- coverage progress
- thermal variance penalty
- peak temperature penalty
- local temperature-difference shaping
- hotspot-dispersion shaping
- invalid-action penalty

This is a proxy objective, not a calibrated manufacturing objective.

## 5. Training Pipeline Work Completed

The training script `rl/train_maskable_ppo.py` was upgraded to:

- use `MaskablePPO`
- support CLI arguments
- support configurable timesteps
- support configurable device
- support configurable save path
- support configurable logging interval
- save model, training history, and training curves

The current recommended example is:

```powershell
python rl\train_maskable_ppo.py --timesteps 100000
```

## 6. Evaluation and Output Work Completed

The evaluation pipeline can:

- load a trained model
- run masked inference
- reconstruct a rollout-style result
- export RL order maps
- export RL thermal maps
- export GIF scan animations
- compare RL with raster, random, and greedy baselines
- write markdown result summaries

## 7. Search Pipeline Work Completed

A separate large-scale search script was also added:

- `scripts/search_top_sequences.py`

This script supports:

- many candidate-sequence evaluations
- GPU-assisted scoring
- top-k sequence tracking
- saving `Top 10` candidate plans
- visualising the best found sequence

This search pipeline is useful as a non-RL reference path to compare against learned strategies.

## 8. Tests Added or Updated

The test suite now includes sanity checks for:

- geometry generation
- stripe segmentation staying inside the mask
- thermal updates
- rollout outputs
- RL environment reset
- RL environment action masking
- valid action state updates
- invalid action penalties

This is still a lightweight test suite, but it is enough to validate that the project is runnable and internally consistent.

## 9. Documentation Work Completed

The README has been rewritten to make the repository much easier to understand on GitHub.

It now explains:

- what the project is
- what has been implemented
- how the RL setup works
- how to run training and evaluation
- what outputs to expect
- what the limitations are

## 10. Latest Local Training Status

A stripe-based training run of **100000 timesteps** has already been completed locally.

Latest saved files:

- `assets/models/maskable_ppo_twi_stripe.zip`
- `assets/models/maskable_ppo_twi_stripe_history.json`
- `assets/figures/maskable_ppo_twi_stripe_training_curves.png`

From the recorded local history snapshot:

- final coverage reached `1.0`
- final thermal mean was about `0.00586`
- final thermal peak was about `0.5169`
- final thermal variance was about `0.000866`

This confirms that the stripe-based pipeline can complete a full training run and save outputs correctly.

## 11. Latest Official Stripe Evaluation

The latest evaluation was run against the newest stripe checkpoint and exported a full result set:

- `assets/figures/order_map_rl_maskable_ppo_stripe.png`
- `assets/figures/thermal_map_rl_maskable_ppo_stripe.png`
- `assets/figures/scan_path_rl_maskable_ppo_stripe.gif`
- `assets/figures/order_map_comparison_grid_stripe.png`
- `assets/figures/thermal_map_comparison_grid_stripe.png`
- `assets/figures/metrics_comparison_with_rl_stripe.png`
- `training_results_stripe.md`

Latest comparison snapshot:

- raster: `peak=0.555`, `variance=0.001`
- random: `peak=0.292`, `variance=0.000`
- greedy_cool_first: `peak=0.326`, `variance=0.000`
- rl_maskable_ppo_stripe: `peak=0.512`, `variance=0.001`

Interpretation:

- the stripe-based RL run is already stronger than raster
- it is not yet stronger than the best handcrafted baselines
- the bottleneck is now reward tuning / training quality, not missing infrastructure
## 12. Current Strengths

The repository is now strong in the following ways:

- modular structure
- clear geometry-to-environment flow
- working baseline pipeline
- working stripe-based RL environment
- CLI-controllable training
- strong visual output potential
- practical demo value for explanation and presentation

## 13. Current Gaps / Follow-Up Work

The main gaps are now mostly on the experiment side rather than on the scaffolding side.

Still worth doing next:

1. Evaluate the latest stripe-trained model using the newest checkpoint rather than older legacy paths.
2. Export and compare the latest RL order map, thermal map, and GIF against baselines.
3. Tune reward weights so RL more reliably beats raster and heuristic baselines.
4. Train on more than one geometry to reduce overfitting to a single `TWI` mask.
5. Clean up local experiment artifacts and decide which generated figures/models should be versioned.

## 14. Overall Conclusion

This project is no longer just an idea or folder scaffold.

It is now a **working experimental platform** for:

- letter-shaped scan planning
- lightweight thermal proxy simulation
- baseline comparison
- stripe-based RL experimentation
- visual explanation of learned scan behaviour

The current state is already strong enough for:

- internal demos
- method discussions
- early presentation material
- continued reward and environment tuning

It should still be described honestly as a **proxy-model RL demo**, not a validated manufacturing simulator.
