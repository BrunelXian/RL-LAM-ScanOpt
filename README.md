# RL-LAM-ScanOpt

`RL-LAM-ScanOpt` is now a **line-order benchmark and FEA-teacher preparation repo** for laser directed energy deposition (LDED).

The original proxy-only RL mainline in this repository has already been stress-tested and **deprioritized**. The current project direction is:

1. define a simple but physically interpretable **LDED line-order benchmark**
2. generate a **small, diverse candidate trajectory pool**
3. export those trajectories into **Abaqus/FEA-consumable scan-path files**
4. run sparse high-value FEA checks later, outside this repository's current scope

This repository does **not** currently run Abaqus jobs, solve FEA, train PPO, or train ranking models as part of the active mainline.

## Current Status

The active benchmark is:

- `benchmark_name = lded_coupon_32track_v1`

The active deliverables already implemented are:

- line-based coupon geometry and baseline trajectory generation
- a `46`-trajectory LDED teacher pool
- `46 / 46` FEA scan-path exports
- a `4`-trajectory Abaqus sanity-check package

The current next step is **manual Abaqus-side sanity checking on the selected 4 trajectories**, not more proxy-RL experimentation.

## Why The Project Direction Changed

The earlier mainline used cheap thermal proxy rewards to train or diagnose scan-order policies on grid-based targets.

That line of work has been pushed far enough to make a clear decision:

- PPO on the cheap proxy route: **NO-GO**
- patch-based action family: **NO-GO**
- reward-only rewrites: **NO-GO**
- selector-state coupling: **NO-GO**
- offline partial-state ranking under the same cheap proxy: **NO-GO**

The core finding was not "PPO is bad."

The core finding was:

> the current cheap evaluator is not sufficiently sensitive to scan-order history and cumulative path effects to serve as the main learning target.

Because of that, the repository has shifted away from "optimize path directly with proxy-RL" and toward:

- teacher-pool construction
- FEA export
- later sparse teacher evaluation in Abaqus or another high-fidelity tool

Legacy RL, selector, and TWI/grid assets are still kept in the repo as research artifacts, but they are **not** the current mainline.

## Active Benchmark: `lded_coupon_32track_v1`

This benchmark models a simple single-layer LDED coupon with **32 vertical tracks**.

### Geometry

| Field | Value |
| --- | --- |
| Working plane | `100 mm x 40 mm` |
| Deposited patch | `96 mm x 36 mm` |
| Margin | `2 mm` on all sides |
| Track count | `32` |
| Track width | `3 mm` |
| Track pitch | `3 mm` |
| Track length | `36 mm` |
| Layer count | `1` |
| Track direction | `bottom_to_top` |

Coordinate convention:

- `x`: across tracks
- `y`: along track length
- patch `x` range: `2 mm .. 98 mm`
- patch `y` range: `2 mm .. 38 mm`
- track `i` center:
  - `x_center = 2.0 + 1.5 + 3.0 * i`
- export version:
  - `z = 0.0`

### Action / Trajectory Semantics

At this stage, one trajectory is simply a **track-order permutation**:

- each action selects the next unscanned track
- one episode contains `32` actions
- each trajectory is a length-`32` permutation of track ids `0..31`
- no per-track direction choice
- no intra-track segmentation
- no multilayer logic yet

The benchmark definition lives in:

- [D:\Projects\RL-LAM-ScanOpt\core\geometry.py](D:/Projects/RL-LAM-ScanOpt/core/geometry.py:1)

## Layout Preview

Preview script:

- [D:\Projects\RL-LAM-ScanOpt\scripts\preview_lded_coupon_32track.py](D:/Projects/RL-LAM-ScanOpt/scripts/preview_lded_coupon_32track.py:1)

Run:

```powershell
python scripts/preview_lded_coupon_32track.py
```

Outputs:

- [D:\Projects\RL-LAM-ScanOpt\assets\figures\lded_coupon_32track_layout.png](D:/Projects/RL-LAM-ScanOpt/assets/figures/lded_coupon_32track_layout.png)
- [D:\Projects\RL-LAM-ScanOpt\assets\figures\lded_coupon_32track_layout_with_ids.png](D:/Projects/RL-LAM-ScanOpt/assets/figures/lded_coupon_32track_layout_with_ids.png)
- [D:\Projects\RL-LAM-ScanOpt\assets\figures\lded_coupon_32track_baseline_previews.png](D:/Projects/RL-LAM-ScanOpt/assets/figures/lded_coupon_32track_baseline_previews.png)
- [D:\Projects\RL-LAM-ScanOpt\assets\data\lded_coupon_32track_baselines.json](D:/Projects/RL-LAM-ScanOpt/assets/data/lded_coupon_32track_baselines.json:1)

Included baseline track orders:

- `raster_left_to_right`
- `raster_right_to_left`
- `center_out`
- `edge_in`
- `odd_even_interlaced`
- `even_odd_interlaced`
- multiple `random_seed_*`

## FEA-Teacher Trajectory Pool

Pool builder:

- [D:\Projects\RL-LAM-ScanOpt\scripts\build_fea_teacher_pool.py](D:/Projects/RL-LAM-ScanOpt/scripts/build_fea_teacher_pool.py:1)

Run:

```powershell
python scripts/build_fea_teacher_pool.py --benchmark lded_coupon_32track_v1
```

Output directory:

- [D:\Projects\RL-LAM-ScanOpt\assets\fea_teacher_pool_lded_32track](D:/Projects/RL-LAM-ScanOpt/assets/fea_teacher_pool_lded_32track)

Generated files:

- [D:\Projects\RL-LAM-ScanOpt\assets\fea_teacher_pool_lded_32track\fea_teacher_pool_manifest.csv](D:/Projects/RL-LAM-ScanOpt/assets/fea_teacher_pool_lded_32track/fea_teacher_pool_manifest.csv:1)
- [D:\Projects\RL-LAM-ScanOpt\assets\fea_teacher_pool_lded_32track\fea_teacher_pool_manifest.json](D:/Projects/RL-LAM-ScanOpt/assets/fea_teacher_pool_lded_32track/fea_teacher_pool_manifest.json:1)
- [D:\Projects\RL-LAM-ScanOpt\assets\fea_teacher_pool_lded_32track\fea_teacher_pool_summary.txt](D:/Projects/RL-LAM-ScanOpt/assets/fea_teacher_pool_lded_32track/fea_teacher_pool_summary.txt:1)
- [D:\Projects\RL-LAM-ScanOpt\assets\fea_teacher_pool_lded_32track\sequences](D:/Projects/RL-LAM-ScanOpt/assets/fea_teacher_pool_lded_32track/sequences)

### Current Pool Summary

The current pool contains `46` trajectories and satisfies the `30-50` target.

Source-type distribution:

- `anchor_baseline = 7`
- `proxy_best = 10`
- `proxy_worst = 5`
- `random_diverse = 8`
- `proxy_ambiguous = 8`
- `perturbed_or_mixed = 8`

Selection behavior reported in the summary:

- proxy-score range: `-11.389177 .. -4.639177`
- peak-heat range: `1.000000 .. 1.000000`
- heat-variance range: `0.054540 .. 0.054540`
- jump-distance range: `1.000000 .. 16.000000`
- adjacency-ratio range: `0.000000 .. 1.000000`
- early-clustering range: `0.000000 .. 1.000000`
- coverage range: `1.000000 .. 1.000000`
- ambiguous pair count: `4`
- perturbation count: `8`
- duplicate permutations removed: `1`

Important interpretation:

- in this simple line-based proxy, `peak_heat` and `heat_variance` are effectively constant across full 32-track permutations
- the remaining cheap-proxy separation comes mostly from:
  - `reheat`-like cumulative exposure
  - jump distance
  - adjacency / locality
  - early clustering

That is exactly why this pool should now be treated as a **candidate set for sparse FEA evaluation**, not as proof that the cheap proxy is sufficient.

## FEA Export Adapter

Export script:

- [D:\Projects\RL-LAM-ScanOpt\scripts\export_lded_pool_to_fea_paths.py](D:/Projects/RL-LAM-ScanOpt/scripts/export_lded_pool_to_fea_paths.py:1)

Run:

```powershell
python scripts/export_lded_pool_to_fea_paths.py
```

Output directory:

- [D:\Projects\RL-LAM-ScanOpt\assets\fea_teacher_pool_lded_32track\fea_exports](D:/Projects/RL-LAM-ScanOpt/assets/fea_teacher_pool_lded_32track/fea_exports)

Generated files:

- [D:\Projects\RL-LAM-ScanOpt\assets\fea_teacher_pool_lded_32track\fea_exports\fea_export_manifest.csv](D:/Projects/RL-LAM-ScanOpt/assets/fea_teacher_pool_lded_32track/fea_exports/fea_export_manifest.csv:1)
- [D:\Projects\RL-LAM-ScanOpt\assets\fea_teacher_pool_lded_32track\fea_exports\fea_export_manifest.json](D:/Projects/RL-LAM-ScanOpt/assets/fea_teacher_pool_lded_32track/fea_exports/fea_export_manifest.json:1)
- [D:\Projects\RL-LAM-ScanOpt\assets\fea_teacher_pool_lded_32track\fea_exports\fea_export_summary.txt](D:/Projects/RL-LAM-ScanOpt/assets/fea_teacher_pool_lded_32track/fea_exports/fea_export_summary.txt:1)
- `*_scan_path.csv`
- `*_fea_metadata.json`

### Export Conventions

Each exported track path uses:

- `scan_speed_mm_s = 10.0`
- `nominal_power_w = 1000.0`
- `dwell_time_s_between_tracks = 0.0`
- `duration per track = 36.0 / 10.0 = 3.6 s`

Current export summary:

- trajectories found: `46`
- trajectories exported: `46`
- failed exports: `0`
- total deposition time per trajectory: `115.2 s`
- x range: `3.5 .. 96.5 mm`
- y range: `2.0 .. 38.0 mm`
- z range: `0.0 .. 0.0 mm`
- all exported trajectories are valid length-32 permutations: `YES`
- no Abaqus job was run: `YES`

## Abaqus Sanity-Check Package

Sanity-package script:

- [D:\Projects\RL-LAM-ScanOpt\scripts\prepare_abaqus_sanity_check.py](D:/Projects/RL-LAM-ScanOpt/scripts/prepare_abaqus_sanity_check.py:1)

Run:

```powershell
python scripts/prepare_abaqus_sanity_check.py
```

Output directory:

- [D:\Projects\RL-LAM-ScanOpt\assets\fea_teacher_pool_lded_32track\abaqus_sanity_check](D:/Projects/RL-LAM-ScanOpt/assets/fea_teacher_pool_lded_32track/abaqus_sanity_check)

Generated files:

- [D:\Projects\RL-LAM-ScanOpt\assets\fea_teacher_pool_lded_32track\abaqus_sanity_check\sanity_manifest.csv](D:/Projects/RL-LAM-ScanOpt/assets/fea_teacher_pool_lded_32track/abaqus_sanity_check/sanity_manifest.csv:1)
- [D:\Projects\RL-LAM-ScanOpt\assets\fea_teacher_pool_lded_32track\abaqus_sanity_check\sanity_manifest.json](D:/Projects/RL-LAM-ScanOpt/assets/fea_teacher_pool_lded_32track/abaqus_sanity_check/sanity_manifest.json:1)
- [D:\Projects\RL-LAM-ScanOpt\assets\fea_teacher_pool_lded_32track\abaqus_sanity_check\sanity_summary.txt](D:/Projects/RL-LAM-ScanOpt/assets/fea_teacher_pool_lded_32track/abaqus_sanity_check/sanity_summary.txt:1)
- [D:\Projects\RL-LAM-ScanOpt\assets\fea_teacher_pool_lded_32track\abaqus_sanity_check\abaqus_read_scan_path_stub.py](D:/Projects/RL-LAM-ScanOpt/assets/fea_teacher_pool_lded_32track/abaqus_sanity_check/abaqus_read_scan_path_stub.py:1)
- copied `scan_path.csv` and `fea_metadata.json` files for `4` selected trajectories

Selected sanity trajectories:

- `raster_left_to_right`
- `odd_even_interlaced`
- `random_seed_249` (`best_proxy_rank_1`)
- `center_out_local_reversal` (`worst_proxy_tail`)

This package exists so Abaqus-side validation can start with a **small, representative set**, not all `46` trajectories at once.

The stub does **not** import Abaqus modules, call `mdb`, or submit jobs. It only validates that exported path files are readable and structurally correct.

## What Is Legacy Now

These assets remain in the repository, but they are **legacy / smoke-test artifacts**, not the current mainline:

- TWI / `64x64` grid geometry and associated figures
- PPO smoke-test results
- patch-based action family experiments
- selector-PPO and selector-preview experiments
- offline ranking under the old cheap proxy semantics
- old pool:
  - [D:\Projects\RL-LAM-ScanOpt\assets\fea_teacher_pool](D:/Projects/RL-LAM-ScanOpt/assets/fea_teacher_pool)
- old top-sequence files:
  - [D:\Projects\RL-LAM-ScanOpt\assets\models\top_10_sequences_twi_64x64.json](D:/Projects/RL-LAM-ScanOpt/assets/models/top_10_sequences_twi_64x64.json:1)
  - [D:\Projects\RL-LAM-ScanOpt\assets\models\top_10_sequences_twi_64x64.txt](D:/Projects/RL-LAM-ScanOpt/assets/models/top_10_sequences_twi_64x64.txt:1)

Those files are preserved for traceability, but they should not drive new decisions for the current LDED line-order branch.

## What This Repo Does Not Do Yet

The current codebase still does **not** implement:

- Abaqus job submission
- real FEA solving
- multilayer deposition
- per-track direction optimization
- per-track segmentation
- surrogate modeling
- ranking-model training
- imitation learning
- a new RL mainline

## Recommended Next Step

The single next step is:

- use the `4`-trajectory sanity package to verify that Abaqus-side preprocessing can read the exported `scan_path.csv` files and build the intended deposition sequence representation

Do **not** restart PPO, reward tuning, selector coupling, or proxy-only ranking work from this point.

## Minimal File Map

Current mainline files:

- [D:\Projects\RL-LAM-ScanOpt\core\geometry.py](D:/Projects/RL-LAM-ScanOpt/core/geometry.py:1)
- [D:\Projects\RL-LAM-ScanOpt\scripts\preview_lded_coupon_32track.py](D:/Projects/RL-LAM-ScanOpt/scripts/preview_lded_coupon_32track.py:1)
- [D:\Projects\RL-LAM-ScanOpt\scripts\build_fea_teacher_pool.py](D:/Projects/RL-LAM-ScanOpt/scripts/build_fea_teacher_pool.py:1)
- [D:\Projects\RL-LAM-ScanOpt\scripts\export_lded_pool_to_fea_paths.py](D:/Projects/RL-LAM-ScanOpt/scripts/export_lded_pool_to_fea_paths.py:1)
- [D:\Projects\RL-LAM-ScanOpt\scripts\prepare_abaqus_sanity_check.py](D:/Projects/RL-LAM-ScanOpt/scripts/prepare_abaqus_sanity_check.py:1)

Current main outputs:

- [D:\Projects\RL-LAM-ScanOpt\assets\fea_teacher_pool_lded_32track\fea_teacher_pool_summary.txt](D:/Projects/RL-LAM-ScanOpt/assets/fea_teacher_pool_lded_32track/fea_teacher_pool_summary.txt:1)
- [D:\Projects\RL-LAM-ScanOpt\assets\fea_teacher_pool_lded_32track\fea_exports\fea_export_summary.txt](D:/Projects/RL-LAM-ScanOpt/assets/fea_teacher_pool_lded_32track/fea_exports/fea_export_summary.txt:1)
- [D:\Projects\RL-LAM-ScanOpt\assets\fea_teacher_pool_lded_32track\abaqus_sanity_check\sanity_summary.txt](D:/Projects/RL-LAM-ScanOpt/assets/fea_teacher_pool_lded_32track/abaqus_sanity_check/sanity_summary.txt:1)

## Testing

Recently added tests for the active line-order branch include:

- [D:\Projects\RL-LAM-ScanOpt\tests\test_geometry.py](D:/Projects/RL-LAM-ScanOpt/tests/test_geometry.py:1)
- [D:\Projects\RL-LAM-ScanOpt\tests\test_build_fea_teacher_pool_lded.py](D:/Projects/RL-LAM-ScanOpt/tests/test_build_fea_teacher_pool_lded.py:1)
- [D:\Projects\RL-LAM-ScanOpt\tests\test_export_lded_pool_to_fea_paths.py](D:/Projects/RL-LAM-ScanOpt/tests/test_export_lded_pool_to_fea_paths.py:1)
- [D:\Projects\RL-LAM-ScanOpt\tests\test_prepare_abaqus_sanity_check.py](D:/Projects/RL-LAM-ScanOpt/tests/test_prepare_abaqus_sanity_check.py:1)

## License / Research Note

This repository is still a research prototype.

The active LDED branch is intentionally simple:

- single layer
- fixed bottom-to-top track direction
- no physical diffusion model
- no Abaqus coupling in-repo yet

That simplicity is deliberate. The current purpose is to produce a clean bridge from:

- interpretable line-order trajectories

to:

- sparse high-value FEA sanity checking and teacher evaluation.
