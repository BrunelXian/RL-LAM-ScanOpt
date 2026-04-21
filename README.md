# RL-LAM-ScanOpt

**RL-LAM-ScanOpt** is a lightweight research/demo project for **reinforcement learning-based scan path optimisation inside letter-shaped regions** such as `TWI`.

Instead of treating the scan area as a full rectangular plate, this project uses the **interior of the letters themselves** as the legal scan region. The core question is simple:

> Can we learn a scan order that produces a more uniform thermal field and fewer concentrated hotspots than simple handcrafted baselines?

This repository is intentionally designed as a **fast, interpretable, engineering-friendly demo**. It does **not** attempt to be a calibrated process simulator.

## Latest Validation Round

The latest work in this repository focused on one narrow question:

> Is PPO still failing mainly because the action representation is too coarse for thermal credit assignment?

To answer that cleanly, the project went through three controlled validation steps without touching the thermal proxy or reward weights:

1. **Segment-count sweep (`4 / 6 / 8`)**
   - `segments_per_stripe = 6` remained the best trade-off.
   - It improved locality relative to the original stripe setup while keeping thermal signal strength and baseline ranking sensible.
2. **Strict PPO smoke test in `segment=6`**
   - PPO no longer collapsed into the severe early clustering seen in the old stripe setting.
   - However, it still failed the strict thermal and reward margins against `random_segment6`.
   - Current verdict: **NO-GO for scaling PPO**.
3. **Variable-length action experiment**
   - A new representation was added where one action selects:
     - `stripe_id`
     - `start_cell`
     - `length` in `[2, 8]`
   - This was tested with baselines and diagnostics only.
   - Result: it preserved planner discrimination, but it did **not** clearly improve thermal attribution and is **not** yet a better PPO candidate than fixed `segment=6`.

### Current Gate Decision

- Best action granularity tested so far: **fixed `segment=6`**
- Latest PPO gate result: **NO-GO**
- Recommended next step: **revisit action representation before larger PPO training**

### Key Numeric Takeaways

#### Segment-count sweep

| Setting | Action Locality (mean cells/action) | Thermal Share | Distance-aware Early Adjacency |
| --- | ---: | ---: | ---: |
| stripe | 15.89 | 0.253 | n/a |
| segment=4 | 3.97 | 0.263 | 0.030 |
| segment=6 | 2.67 | 0.262 | 0.010 |
| segment=8 | 2.14 | 0.255 | 0.065 |

Interpretation:

- `segment=4` is better than stripe, but still a bit coarse.
- `segment=8` is finer, but starts to fragment the task and worsens the early clustering proxy.
- `segment=6` is the best balance of locality, thermal signal, and stable baseline structure.

#### Strict PPO smoke test (`segment=6`)

| Policy | Total Reward | Coverage | Invalid Rate | Reheat | Peak | Early Adjacency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ppo_segment6` | 172.617 | 1.000 | 0.000% | -8.611 | -122.641 | 0.061 |
| `random_segment6` | 171.503 | 1.000 | 0.000% | -9.583 | -122.567 | 0.061 |
| `distance_aware_cool_first_segment6` | 200.047 | 1.000 | 0.000% | -8.894 | -119.771 | 0.010 |

Strict gate result:

- `coverage = 1.0`: **pass**
- `invalid < 1%`: **pass**
- `early adjacency < 0.5`: **pass**
- `total reward >= random + 15%`: **fail**
- `reheat >= 20% better than random`: **fail**
- `peak >= 20% better than random`: **fail**

This means PPO is no longer obviously collapsed, but it is still too close to `random_segment6` and not yet showing a strong thermo-aware advantage.

#### Variable-length action experiment

| Environment | Action Format | Action Count | Mean Cells/Action | Range |
| --- | --- | ---: | ---: | ---: |
| fixed `segment=6` | `(stripe_id, segment_index)` | 336 | 2.67 | 1..4 |
| variable-length | `(stripe_id, start_cell, length[2..8])` | 6230 | 4.25 | 1..8 |

Diagnostics summary:

- thermal share changed from `0.261` to `0.258`
- coverage+completion share changed from `0.668` to `0.672`
- distance-aware early adjacency worsened from `0.010` to `0.184`

Interpretation:

- the new variable-length representation did **not** strengthen the thermal learning signal
- it made the action catalog much larger
- and it made the best thermo-aware baseline look less locally structured in the early phase

So this representation is useful as a diagnostic experiment, but it is **not** yet the next PPO candidate.

### Current Defects / Open Issues

The current repository is strong as a diagnostic platform, but there are still important limitations:

1. **PPO still does not clearly beat `random_segment6`**
   - the latest smoke test only improved total reward by about `0.6%`
   - reheat improvement vs random was only about `10.1%`
   - peak did not improve meaningfully vs random
2. **Credit assignment is improved, but not solved**
   - `segment=6` reduced the old collapse problem
   - PPO still does not move convincingly toward the `distance_aware_cool_first` behavior
3. **Variable-length actions are not yet a clean solution**
   - they increase catalog size a lot
   - and do not make the thermal signal more dominant
4. **The project is still a proxy-model demo**
   - no FEM
   - no calibrated melt-pool model
   - no exact residual-stress prediction

### Useful Output Files

Recent outputs referenced above are saved locally in:

- `assets/models/segment_count_sweep_verdict.txt`
- `assets/models/action_granularity_sweep_comparison.csv`
- `assets/models/ppo_smoke_segment6_verdict.txt`
- `assets/models/ppo_vs_baselines_segment6.csv`
- `assets/models/action_space_comparison_variable_segment.csv`
- `assets/models/variable_segment_diagnostics_summary.txt`

Recent figures include:

- `assets/figures/maskable_ppo_smoke_segment6_training_curve.png`
- `assets/figures/ppo_smoke_segment6_reward_breakdown.png`
- `assets/figures/ppo_smoke_segment6_heatmap_comparison.png`
- `assets/figures/ppo_smoke_segment6_scan_order_comparison.png`
- `assets/figures/baseline_reward_breakdown_variable_segment.png`
- `assets/figures/baseline_heatmap_comparison_variable_segment.png`
- `assets/figures/baseline_scan_order_comparison_variable_segment.png`

## Why This Project Exists

In laser-based additive manufacturing, scan order matters. Even with a simplified thermal proxy, the order in which the beam visits different parts of a geometry can change:

- how heat accumulates
- how evenly temperature is distributed
- how strongly hotspots cluster
- how large stress-related proxy risk may become

RL-LAM-ScanOpt turns that intuition into a compact optimisation problem that is easy to visualise, easy to extend, and fast enough to iterate on.

## What Is Implemented Today

The current repository already supports an end-to-end demo pipeline:

- high-resolution text mask generation for targets such as `TWI`
- downsampling to a coarse `64x64` decision grid
- lightweight Gaussian heat-source plus diffusion/cooling proxy simulation
- baseline planners:
  - raster
  - random
  - greedy cool-first
- a **stripe-based** masked RL environment
- a **fixed segment-based** masked RL environment
- a **variable-length segment** diagnostic environment
- `MaskablePPO` training and evaluation
- reward-breakdown diagnostics and planner-level CSV outputs
- exported figures:
  - target masks
  - scan order maps
  - thermal maps
  - training curves
  - scan GIFs
  - comparison figures

In the current RL setup, the agent chooses the **next legal vertical stripe segment inside the letter region**, not an arbitrary out-of-mask cell.

## Current Status

The repository is now beyond the scaffold stage and already supports real experimentation:

- Phase 1 baseline simulation is implemented and runnable
- the geometry pipeline supports text masks and legal stripe segmentation
- the thermal proxy pipeline is active
- the stripe-based RL environment is implemented
- the fixed `segment=6` action environment is implemented and validated
- a variable-length action environment has been added for diagnostics
- `MaskablePPO` training is implemented with CLI control
- evaluation and visualisation scripts exist and export figures/GIFs
- a `100000`-timestep stripe-based training run has already been completed locally

Latest local training artifacts:

- model: `assets/models/maskable_ppo_twi_stripe.zip`
- history: `assets/models/maskable_ppo_twi_stripe_history.json`
- curves: `assets/figures/maskable_ppo_twi_stripe_training_curves.png`

At the moment, the repository is strongest as a **working experimental demo platform**, not yet as a polished benchmark suite.

## Latest Stripe Evaluation

The latest official local evaluation was run using the stripe-based checkpoint:

- model: `assets/models/maskable_ppo_twi_stripe.zip`
- summary: `training_results_stripe.md`

Current comparison snapshot:

| Planner | Coverage | Thermal Mean | Thermal Peak | Thermal Variance | Steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| raster | 1.000 | 0.006 | 0.555 | 0.001 | 890 |
| random | 1.000 | 0.006 | 0.292 | 0.000 | 890 |
| greedy_cool_first | 1.000 | 0.006 | 0.326 | 0.000 | 890 |
| rl_maskable_ppo_stripe | 1.000 | 0.006 | 0.512 | 0.001 | 890 |

Takeaway:

- the stripe-based RL model reaches full coverage
- it clearly improves over `raster`
- it is still behind `random` and `greedy_cool_first` on this simplified proxy task
- the project is now in the phase where reward tuning and evaluation quality matter more than scaffolding

### Latest Visuals

#### RL Stripe Order Map

![RL Stripe Order Map](assets/figures/order_map_rl_maskable_ppo_stripe.png)

#### RL Stripe Thermal Map

![RL Stripe Thermal Map](assets/figures/thermal_map_rl_maskable_ppo_stripe.png)

#### RL Stripe Scan GIF

![RL Stripe Scan GIF](assets/figures/scan_path_rl_maskable_ppo_stripe.gif)

#### RL vs Baselines

![Stripe Scan Order Comparison](assets/figures/order_map_comparison_grid_stripe.png)

![Stripe Thermal Comparison](assets/figures/thermal_map_comparison_grid_stripe.png)

![Stripe Training Curves](assets/figures/maskable_ppo_twi_stripe_training_curves.png)

## Project Scope

Included:

- letter-shaped scan regions such as `TWI`
- stripe-based or cell-wise scan-order reasoning on a coarse grid
- lightweight thermal proxy updates
- RL and baseline comparison
- visualisation-first experimentation

Explicitly out of scope:

- FEM
- exact melt-pool simulation
- exact residual stress prediction
- calibrated temperature prediction
- production-ready process planning

## Core Idea

The repository separates the problem into a few small, understandable parts:

1. **Geometry**
   Generate a text mask and turn it into a legal scan region.

2. **Thermal Proxy**
   Apply a Gaussian-like heat deposit, then diffuse and cool the field after each scan step.

3. **Planner / Policy**
   Choose the next scan action, either from a baseline heuristic or from an RL policy.

4. **Metrics**
   Compare final thermal variance, peak temperature, mean temperature, coverage, and visual scan structure.

5. **Visual Outputs**
   Turn the result into order maps, thermal maps, GIFs, and comparison charts that are easy to explain.

## Current RL Formulation

### Observation

The policy observes a channel-first tensor:

- `target_mask`
- `scanned_mask`
- `thermal_field`

This is designed to be CNN-friendly.

### Action Space

The current environment is **stripe-based**:

- the letter region is split into legal **vertical stripe segments**
- each action selects the **next stripe**
- stripes are constrained to stay inside the target letter geometry
- invalid stripe selections are masked out

### Reward

The reward is a simple thermal-quality proxy, built to encourage useful behaviour without pretending to be a physical truth model.

It includes:

- coverage progress
- penalty for high thermal variance
- penalty for high peak temperature
- local temperature-difference shaping
- hotspot-dispersion shaping
- invalid-action penalty

## Repository Structure

```text
app/
assets/
  figures/
  models/
core/
  evaluators/
  planners/
rl/
scripts/
tests/
```

### Important Files

| Path | Purpose |
| --- | --- |
| `core/geometry.py` | Text masks, coarse grids, stripe generation |
| `core/thermal.py` | Gaussian heat input and diffusion/cooling proxy |
| `core/rollout.py` | Executes scan sequences and records outputs |
| `core/viz.py` | Figures, comparison plots, GIF generation |
| `core/planners/` | Raster, random, and greedy baselines |
| `rl/env_scan.py` | Masked Gymnasium environment for stripe-based scan planning |
| `rl/train_maskable_ppo.py` | Train `MaskablePPO` on the stripe environment |
| `rl/eval_policy.py` | Evaluate RL against baselines and export visuals |
| `scripts/run_baselines.py` | Run the baseline planners |
| `scripts/search_top_sequences.py` | Large-scale candidate search for strong scan orders |

## How To Run

### 1. Install dependencies

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you already have a CUDA-ready Python environment, you can use that instead.

### 2. Generate baseline results

```powershell
python scripts\run_baselines.py
```

### 3. Train the stripe-based RL policy

```powershell
python rl\train_maskable_ppo.py --timesteps 100000
```

Useful CLI options:

- `--timesteps`
- `--device`
- `--save-path`
- `--log-interval`

Example:

```powershell
python rl\train_maskable_ppo.py --timesteps 100000 --device cuda --save-path assets/models/maskable_ppo_twi_stripe.zip
```

### 4. Evaluate the trained model

```powershell
python rl\eval_policy.py
```

## Generated Outputs

The project can generate:

- target mask images
- scan-order heatmaps
- final thermal maps
- RL scan-path GIFs
- baseline vs RL comparison figures
- training curves
- markdown summary reports

Typical output locations:

- `assets/figures/`
- `assets/models/`
- `training_results.md`
- `training_results_stripe.md`
- `WORK_SUMMARY.md`

## Recent Progress In This Repository

This repository has recently been upgraded from a simple cell-wise masked demo into a **stripe-based scan planning setup**:

- the geometry module now supports legal stripe segmentation inside letter masks
- the RL environment now acts on stripe segments rather than arbitrary grid cells
- training is driven by `MaskablePPO`
- evaluation exports order maps, thermal maps, comparison grids, and GIF animations
- the training script now supports CLI control for practical experimentation

That means the repo is no longer just a concept scaffold; it is now a runnable demonstration platform.

## Practical Notes

- Training is launched through `rl/train_maskable_ppo.py`
- the training script now supports CLI control for faster iteration
- the evaluation script still compares RL against raster, random, and greedy baselines
- the search script can also perform large-scale candidate-sequence search outside the RL loop

If you want to continue from the current state, the most useful next command is:

```powershell
python rl\train_maskable_ppo.py --timesteps 100000
```

## How To Judge Whether A Run Is Useful

The most practical evaluation questions are:

1. Does the RL scan order look more like a deliberate process strategy than a noisy jump pattern?
2. Is the final thermal map more uniform than raster?
3. Are thermal variance and peak temperature improving relative to at least one baseline?
4. Are the figures and GIFs clear enough to explain the result to a non-ML audience?

For this project, **explainability and comparison quality** are as important as raw optimisation performance.

## Limitations

Please read this repository as a **planning demo**, not as a validated manufacturing model.

- The thermal field is a proxy field.
- Stress-related outputs are qualitative or relative proxies.
- Results should be interpreted comparatively, not as absolute process guarantees.
- A policy that looks good here still requires more serious physics before any real manufacturing claim.

## Disclaimer

This repository uses simplified proxy models for thermal behaviour and stress-related risk.

- It does **not** predict exact physical temperature.
- It does **not** predict exact residual stress.
- It does **not** replace calibrated process simulation.

Its value is in showing how scan-sequence optimisation can be framed, trained, visualised, and compared in a compact, modular system.

## Next Logical Extensions

If we continue this work, the highest-value next steps are:

1. Improve reward tuning so RL can more consistently beat raster and heuristic strategies.
2. Add multiple geometries instead of training only on one fixed `TWI` mask.
3. Build a small interface layer to let users compare scan strategies interactively.
4. Add cleaner experiment/version tracking for multiple RL runs and result sets.

---

If you are reading this repo on GitHub, the shortest summary is:

**RL-LAM-ScanOpt is a modular demo for learning better scan orders inside letter-shaped additive manufacturing regions using a lightweight thermal proxy and strong visual outputs.**
