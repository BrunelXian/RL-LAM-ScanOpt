# LDED FEA-Teacher Pipeline

## Goal

Prepare a small, diverse, reproducible set of LDED track-order trajectories for later sparse FEA evaluation.

This pipeline is intentionally lightweight:

- no Abaqus solve in-repo
- no PPO
- no ranking-model training
- no multilayer physics

## Benchmark

`benchmark_name = lded_coupon_32track_v1`

Geometry:

- working plane: `100 x 40 mm`
- deposited patch: `96 x 36 mm`
- margin: `2 mm`
- tracks: `32`
- track width / pitch: `3 mm`
- track length: `36 mm`
- direction: `bottom_to_top`

Trajectory format:

- one trajectory = one length-`32` permutation of track ids `0..31`

## Pipeline Stages

### 1. Geometry preview

Script:

- `scripts/preview_lded_coupon_32track.py`

Outputs:

- layout figure
- labeled track figure
- baseline preview figure
- baseline JSON payload

### 2. Teacher-pool construction

Script:

- `scripts/build_fea_teacher_pool.py --benchmark lded_coupon_32track_v1`

Selection philosophy:

- do not search for one "best" trajectory only
- instead collect a small stratified set:
  - anchors
  - proxy-best
  - proxy-worst
  - random-diverse
  - ambiguous cases
  - perturbed or mixed cases

Pool outputs:

- manifest CSV / JSON
- sequence payload JSON files
- summary text report

### 3. FEA path export

Script:

- `scripts/export_lded_pool_to_fea_paths.py`

Per trajectory:

- `*_scan_path.csv`
- `*_fea_metadata.json`

Export convention:

- scan speed: `10 mm/s`
- nominal power: `1000 W`
- dwell: `0 s`
- track duration: `3.6 s`

### 4. Abaqus sanity package

Script:

- `scripts/prepare_abaqus_sanity_check.py`

Output:

- small 4-trajectory package
- copied scan-path CSVs
- copied metadata JSONs
- Python reader stub without Abaqus dependencies

## Why This Pipeline Exists

The earlier proxy-only RL route showed that cheap thermal scoring was not strong enough to stably supervise scan-order learning.

The current pipeline changes the role of the cheap proxy:

- no longer the final optimization target
- now only a cheap filter for building a manageable FEA candidate set

## Current Constraints

Still intentionally missing:

- true thermal diffusion
- residual-stress modeling
- multilayer timing logic
- travel moves
- Abaqus job orchestration
- surrogate fitting

Those should only be added after the export and sanity-check path is stable.
