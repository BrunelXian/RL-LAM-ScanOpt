# Current Status

## Active Mainline

The active project branch is:

- `lded_coupon_32track_v1`
- single-layer
- line-order benchmark
- teacher-pool construction
- FEA scan-path export
- small Abaqus sanity package

The active output chain is:

1. define LDED geometry
2. generate baseline and candidate track orders
3. build a `46`-trajectory teacher pool
4. export all `46` trajectories into FEA-ready scan-path CSV files
5. reduce to a `4`-trajectory Abaqus sanity package

## Current Deliverables

### Benchmark geometry

- working plane: `100 mm x 40 mm`
- deposited patch: `96 mm x 36 mm`
- margin: `2 mm`
- tracks: `32`
- track width / pitch: `3 mm`
- track length: `36 mm`
- direction: `bottom_to_top`

Primary files:

- `core/geometry.py`
- `scripts/preview_lded_coupon_32track.py`

### Teacher pool

Output directory:

- `assets/fea_teacher_pool_lded_32track/`

Current pool:

- count: `46`
- target satisfied: `YES`
- duplicate permutations removed: `1`

Source-type distribution:

- `anchor_baseline = 7`
- `proxy_best = 10`
- `proxy_worst = 5`
- `random_diverse = 8`
- `proxy_ambiguous = 8`
- `perturbed_or_mixed = 8`

### FEA export

Output directory:

- `assets/fea_teacher_pool_lded_32track/fea_exports/`

Current export:

- found: `46`
- exported: `46`
- failed: `0`
- per-track duration: `3.6 s`
- per-trajectory deposition time: `115.2 s`

### Abaqus sanity package

Output directory:

- `assets/fea_teacher_pool_lded_32track/abaqus_sanity_check/`

Selected trajectories:

- `raster_left_to_right`
- `odd_even_interlaced`
- `random_seed_249`
- `center_out_local_reversal`

## What Is Not Active

The following are legacy or deprioritized:

- PPO training
- reward tuning
- selector-state coupling
- ranking-model training
- patch-based RL action exploration
- TWI/grid benchmark as the active path

## Immediate Next Step

Use the `4`-trajectory sanity package to validate Abaqus-side input parsing and deposition-sequence setup before running any real FEA jobs.
