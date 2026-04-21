# RL-LAM-ScanOpt

**RL-LAM-ScanOpt** is a lightweight research/demo project for **reinforcement learning-based scan path optimisation inside letter-shaped regions** such as `TWI`.

Instead of treating the scan area as a full rectangular plate, this project uses the **interior of the letters themselves** as the legal scan region. The core question is simple:

> Can we learn a scan order that produces a more uniform thermal field and fewer concentrated hotspots than simple handcrafted baselines?

This repository is intentionally designed as a **fast, interpretable, engineering-friendly demo**. It does **not** attempt to be a calibrated process simulator.

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
- `MaskablePPO` training and evaluation
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
- `MaskablePPO` training is implemented with CLI control
- evaluation and visualisation scripts exist and export figures/GIFs
- a `100000`-timestep stripe-based training run has already been completed locally

Latest local training artifacts:

- model: `assets/models/maskable_ppo_twi_stripe.zip`
- history: `assets/models/maskable_ppo_twi_stripe_history.json`
- curves: `assets/figures/maskable_ppo_twi_stripe_training_curves.png`

At the moment, the repository is strongest as a **working experimental demo platform**, not yet as a polished benchmark suite.

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

1. Evaluate the new stripe-based model against baselines with the latest trained checkpoint.
2. Improve reward tuning so RL can more consistently beat raster and heuristic strategies.
3. Add multiple geometries instead of training only on one fixed `TWI` mask.
4. Build a small interface layer to let users compare scan strategies interactively.

---

If you are reading this repo on GitHub, the shortest summary is:

**RL-LAM-ScanOpt is a modular demo for learning better scan orders inside letter-shaped additive manufacturing regions using a lightweight thermal proxy and strong visual outputs.**
