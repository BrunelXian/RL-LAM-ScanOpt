# RL-LAM-ScanOpt

**Current mainline:** a compact **LDED line-order benchmark** plus a **FEA-teacher candidate pipeline**.  
**Current benchmark:** `lded_coupon_32track_v1`  
**Current status:** `46` candidate trajectories exported to FEA scan paths, plus a `4`-trajectory Abaqus sanity-check package.

![LDED 32-track layout](assets/figures/lded_coupon_32track_layout.png)

## Overview

This repository no longer treats proxy-only RL as the main research path.

The active goal is:

1. define a simple, interpretable **32-track LDED benchmark**
2. build a **small, diverse trajectory pool**
3. export those trajectories into **Abaqus/FEA-consumable scan-path files**
4. use a **small sanity package first**, before any expensive batch FEA

What this repo is for now:

- line-order benchmark design
- trajectory generation and filtering
- FEA export preparation
- Abaqus-side interface sanity checking

What this repo is not doing on the active path:

- PPO training
- reward tuning
- ranking-model training
- Abaqus job submission
- real FEA solving
- surrogate modeling

## Why The Direction Changed

The earlier proxy-only learning route has already been systematically stress-tested and deprioritized.

Confirmed `NO-GO` branches:

- PPO on the cheap thermal proxy
- patch-based action family
- reward-only rewrites
- selector-state coupling
- offline partial-state ranking under the same proxy semantics

The working conclusion is simple:

> the cheap evaluator is not sensitive enough to scan-order history to serve as the main optimization target.

The old RL and TWI/grid assets remain in the repository as **legacy research artifacts**, but they are no longer the project front door.

For the detailed legacy record, see:

- [docs/legacy_proxy_rl.md](docs/legacy_proxy_rl.md)

## Active Benchmark

`lded_coupon_32track_v1` is a single-layer LDED coupon with **32 vertical tracks**.

| Field | Value |
| --- | --- |
| Working plane | `100 mm x 40 mm` |
| Deposited patch | `96 mm x 36 mm` |
| Margin | `2 mm` all around |
| Track count | `32` |
| Track width | `3 mm` |
| Track pitch | `3 mm` |
| Track length | `36 mm` |
| Track direction | `bottom_to_top` |
| Trajectory type | length-`32` permutation of track ids `0..31` |

Core implementation:

- [core/geometry.py](core/geometry.py)
- [scripts/preview_lded_coupon_32track.py](scripts/preview_lded_coupon_32track.py)

More detail:

- [docs/lded_fea_teacher_pipeline.md](docs/lded_fea_teacher_pipeline.md)

## Current Deliverables

### 1. Layout and baseline track orders

Outputs:

- `assets/figures/lded_coupon_32track_layout.png`
- `assets/figures/lded_coupon_32track_layout_with_ids.png`
- `assets/figures/lded_coupon_32track_baseline_previews.png`
- `assets/data/lded_coupon_32track_baselines.json`

### 2. FEA-teacher trajectory pool

Builder:

- [scripts/build_fea_teacher_pool.py](scripts/build_fea_teacher_pool.py)

Output directory:

- `assets/fea_teacher_pool_lded_32track/`

Current pool summary:

- selected trajectories: `46`
- source types:
  - `anchor_baseline = 7`
  - `proxy_best = 10`
  - `proxy_worst = 5`
  - `random_diverse = 8`
  - `proxy_ambiguous = 8`
  - `perturbed_or_mixed = 8`
- duplicate permutations removed: `1`
- target size `30-50`: `satisfied`

Key files:

- `assets/fea_teacher_pool_lded_32track/fea_teacher_pool_manifest.csv`
- `assets/fea_teacher_pool_lded_32track/fea_teacher_pool_manifest.json`
- `assets/fea_teacher_pool_lded_32track/fea_teacher_pool_summary.txt`
- `assets/fea_teacher_pool_lded_32track/sequences/*.json`

### 3. FEA export adapter

Exporter:

- [scripts/export_lded_pool_to_fea_paths.py](scripts/export_lded_pool_to_fea_paths.py)

Output directory:

- `assets/fea_teacher_pool_lded_32track/fea_exports/`

Current export summary:

- found trajectories: `46`
- exported trajectories: `46`
- failed exports: `0`
- track duration: `3.6 s`
- total deposition time per trajectory: `115.2 s`
- coordinate range:
  - `x = 3.5 .. 96.5 mm`
  - `y = 2.0 .. 38.0 mm`
  - `z = 0.0 mm`

Key files:

- `assets/fea_teacher_pool_lded_32track/fea_exports/fea_export_manifest.csv`
- `assets/fea_teacher_pool_lded_32track/fea_exports/fea_export_manifest.json`
- `assets/fea_teacher_pool_lded_32track/fea_exports/fea_export_summary.txt`

### 4. Abaqus sanity-check package

Preparer:

- [scripts/prepare_abaqus_sanity_check.py](scripts/prepare_abaqus_sanity_check.py)

Output directory:

- `assets/fea_teacher_pool_lded_32track/abaqus_sanity_check/`

Selected trajectories:

- `raster_left_to_right`
- `odd_even_interlaced`
- `random_seed_249` (`best_proxy_rank_1`)
- `center_out_local_reversal` (`worst_proxy_tail`)

Key files:

- `assets/fea_teacher_pool_lded_32track/abaqus_sanity_check/sanity_manifest.csv`
- `assets/fea_teacher_pool_lded_32track/abaqus_sanity_check/sanity_manifest.json`
- `assets/fea_teacher_pool_lded_32track/abaqus_sanity_check/sanity_summary.txt`
- `assets/fea_teacher_pool_lded_32track/abaqus_sanity_check/abaqus_read_scan_path_stub.py`

## Quick Start

### Preview the benchmark

```powershell
python scripts/preview_lded_coupon_32track.py
```

### Rebuild the LDED teacher pool

```powershell
python scripts/build_fea_teacher_pool.py --benchmark lded_coupon_32track_v1
```

### Re-export all trajectories to FEA scan paths

```powershell
python scripts/export_lded_pool_to_fea_paths.py
```

### Rebuild the Abaqus sanity package

```powershell
python scripts/prepare_abaqus_sanity_check.py
```

## Repo Map

```text
assets/
  data/                         benchmark baseline trajectories
  fea_teacher_pool/             legacy TWI/grid pool
  fea_teacher_pool_lded_32track/
    sequences/                  line-order trajectory payloads
    fea_exports/                exported scan paths and metadata
    abaqus_sanity_check/        4-trajectory sanity package
  figures/                      benchmark and preview figures
core/
  geometry.py                   LDED benchmark definition
scripts/
  preview_lded_coupon_32track.py
  build_fea_teacher_pool.py
  export_lded_pool_to_fea_paths.py
  prepare_abaqus_sanity_check.py
docs/
  current_status.md
  lded_fea_teacher_pipeline.md
  legacy_proxy_rl.md
```

## Legacy Assets

The repository still contains older TWI/grid and RL artifacts for traceability, including:

- `assets/fea_teacher_pool/`
- `assets/models/top_10_sequences_twi_64x64.json`
- selector and PPO diagnostics under `assets/models/` and `assets/figures/`

Those files are preserved, but they are **not** the active benchmark or decision path.

## Recommended Next Step

The next step is **not** new learning experiments.

The next step is:

- run a **manual Abaqus-side sanity check** on the `4` packaged trajectories and verify that the exported `scan_path.csv` files can drive the intended deposition-sequence setup.

## More Documentation

- [docs/current_status.md](docs/current_status.md)
- [docs/lded_fea_teacher_pipeline.md](docs/lded_fea_teacher_pipeline.md)
- [docs/legacy_proxy_rl.md](docs/legacy_proxy_rl.md)
