# RL-LAM-ScanOpt

Minimal research/demo project for **reinforcement learning-based scan path optimisation inside letter-shaped regions** such as `TWI`.

The scan region is the **interior of the letters themselves**. The project explores whether RL can find scan orders that reduce thermal non-uniformity, hotspot concentration, and stress-related proxy risk compared with simple baselines.

> This repository is intentionally a lightweight proxy-model demo.
> It does **not** implement FEM, exact melt-pool physics, or exact residual stress prediction.

## Overview

In laser-based additive manufacturing, the order in which regions are scanned can change how heat accumulates over time. RL-LAM-ScanOpt turns that idea into a compact optimisation problem:

- generate a target mask for a letter geometry such as `TWI`
- restrict valid scan actions to cells inside that letter region
- update a lightweight thermal field after each scan action
- train an RL policy to choose scan order under thermal-quality objectives
- compare RL against raster, random, and heuristic baselines

The emphasis is on **fast iteration**, **clear visuals**, and **modular design**, not high-fidelity process simulation.

## Project Goal

Use RL to optimise scan paths so that temperature distribution inside a letter-shaped scan region stays as uniform as possible, hotspots are reduced, and residual-stress-related risk proxies are improved.

## Core Modules

| Module | Purpose |
| --- | --- |
| `core/geometry.py` | Generate text masks and convert them into coarse legal scan grids |
| `core/thermal.py` | Apply Gaussian heat-source deposits plus diffusion/cooling proxy updates |
| `core/planners/` | Provide raster, random, and greedy baseline planners |
| `core/rollout.py` | Execute scan plans and collect thermal fields, order maps, and metrics |
| `rl/env_scan.py` | Define the masked Gymnasium environment for scan planning |
| `rl/train_maskable_ppo.py` | Train a `MaskablePPO` policy and save model/training curves |
| `rl/eval_policy.py` | Run masked inference, compare RL with baselines, and export reports |

## State Representation

The RL observation uses a channel-first tensor for CNN compatibility:

- `target_mask`: binary mask of the target letter region
- `scanned_mask`: binary mask of already scanned cells
- `thermal_field`: current thermal proxy field

Only cells **inside the letter shape** are valid scan actions.

## Reward Design

The reward is based on simplified thermal-quality indicators:

- bonus for valid coverage progress
- penalty for high thermal variance
- penalty for large local temperature differences
- penalty for concentrated hotspots / poor hotspot distribution
- penalty for invalid actions

The current environment does **not** add a jump-path penalty, so the policy is free to explore non-local moves if they help the thermal objective.

## Thermal Model

The thermal model is deliberately simple and fast:

- local **Gaussian heat source** at the active scan cell
- lightweight **diffusion** using smoothing
- global **cooling/decay** after each step

This is a proxy thermal field for planning experiments, not a calibrated physical simulation.

## Current Workflow

1. Generate a high-resolution text mask for a target such as `TWI`
2. Downsample it to a `64x64` scan grid
3. Restrict legal actions to cells inside the letter geometry
4. Run baselines and RL in the same masked environment
5. Export scan-order maps, thermal maps, training curves, GIF animation, and markdown summaries

## Repository Layout

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

## Quick Start

### Windows setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Typical commands

```powershell
python scripts\run_baselines.py
python rl\train_maskable_ppo.py
python rl\eval_policy.py
```

## Outputs

The project can generate:

- target-mask figures
- scan-order maps
- thermal maps
- RL training curves
- RL scan-path GIF animation
- metric comparison charts
- markdown evaluation summary

## Scope and Limitations

Included:

- letter-shaped scan regions such as `TWI`
- coarse grid planning
- Gaussian heat-source proxy
- diffusion/cooling proxy updates
- Maskable PPO training and evaluation

Explicitly out of scope:

- FEM
- exact residual stress prediction
- exact thermal calibration
- production-ready process simulation

## Disclaimer

This repository uses **proxy models** for thermal behaviour and stress-related risk.

- Temperatures are simplified indicators rather than calibrated physical temperatures.
- Stress-related outputs are qualitative or relative proxies, not exact residual stress predictions.
- Results should be interpreted as scan-planning demonstrations rather than validated manufacturing guarantees.
