# Abaqus Teacher Production Run Notes

## 1. Night run automation

- Created `scripts/abaqus/run_remaining_6_abaqus_jobs.py`
- Created `scripts/abaqus/launch_remaining_6_abaqus_jobs.ps1`
- Goal: sequentially run the remaining `6` Abaqus jobs
- Execution mode:
  - `cpus=12`
  - no GUI
  - no GPU
  - stop immediately on failure

Confirmed run status:

- `formal_raster_left_to_right`: success
- `greedy_maximin_distance`: success
- `smartscan_proxy_variance`: success
- `multi_lag_regular_jump`: failed
- `block_interleaved_quarters`: pending / not submitted by runner
- `center_edge_alternating`: pending / not submitted by runner

Small run summaries were written to:

- `abaqus-models/night_run_remaining_6_summary/run_remaining_6_summary.csv`
- `abaqus-models/night_run_remaining_6_summary/run_remaining_6_summary.json`

## 2. `multi_lag_regular_jump` failure diagnosis

Failure location:

- `step_final_cooling`
- Step `33`, Increment `1`

Primary errors:

- `TOO MANY ATTEMPTS MADE FOR THIS INCREMENT`
- `FORCE EQUILIBRIUM NOT ACHIEVED`
- `HEAT FLUX EQUILIBRIUM NOT ACHIEVED`
- `STRAIN INCREMENT HAS EXCEEDED FIFTY TIMES...`

Conclusion:

- all `32` scan steps completed
- failure occurred at cooling start
- `label_valid = false`
- failure type = `cooling_start_convergence_failed`

## 3. Restart-cooling attempt

Created script:

- `scripts/abaqus/create_multilag_restart_cooling_v02.py`

This script checks for restart artifacts such as:

- `.res`
- `.stt`
- `.mdl`

Observed behavior:

- the original `v01` job did not contain the required `.res/.stt/.mdl` restart files
- the original input had:

```text
*Restart, write, frequency=0
```

Therefore:

- restart from `step_scan_31` was not possible
- current `v01` cannot be restarted
- future jobs must enable restart write before running

## 4. v02 full rerun cooling-stabilized setup

Created script:

- `scripts/abaqus/create_multilag_fullrerun_cooling_stabilized_v02.py`

Generated directory:

- `abaqus-models/lded_stripe_tm_32track_multilag_jump_v02_fullrerun_cooling_stabilized`

The original multi-lag scan order was preserved:

```text
25,31,21,15,26,30,22,11,2,6,16,10,20,29,24,14,5,1,9,19,28,23,13,4,0,8,18,12,3,7,17,27
```

Cooling-step changes:

- removed original `step_final_cooling`
- added:
  - `step_cooling_transition_10s`
  - `step_final_cooling_remaining`

Transition step parameters:

- `timePeriod=10.0`
- `initialInc=0.001`
- `minInc=1e-10`
- `maxInc=0.01`
- `deltmx=50`
- `maxNumInc=999999`

Final cooling remaining parameters:

- `timePeriod=1500.0`
- `initialInc=1.0`
- `minInc=1e-10`
- `maxInc=60.0`
- `deltmx=100`
- `maxNumInc=999999`

Cooling steps use:

- `*Dsflux, op=NEW` to disable scan heat flux
- original `*Sfilm`
- original `*Sradiate`

Status:

- datacheck passed
- formal run started with `cpus=12`

## 5. Restart write risk

The current `v02` script changes `34` occurrences of:

- `*Restart, write, frequency=0`

to:

- `*Restart, write, frequency=1`

Risk implication:

- `frequency=1` writes restart data at every increment

Latest live check snapshot:

- Step `6`, Increment `14`
- `.res = 1.59 MB`
- `.stt = 7681.13 MB`
- `.mdl = 27.66 MB`
- `.odb = 809.00 MB`
- `D:` free space = `225.03 GB`

Risk assessment:

- `HIGH RISK`

Recommendation:

- future versions should use:

```text
*Restart, write, frequency=999999, overlay
```

- do not keep global `frequency=1` as the default production choice

Important note:

- the current run was **not** killed during the live check
- no live restart/output files were modified during inspection

## 6. GitHub policy

- Do **not** commit Abaqus heavy outputs
- Commit:
  - scripts
  - small CSV/JSON summaries
  - Markdown notes
  - reproducibility metadata
- Keep:
  - ODB
  - CAE
  - restart files
  - full solver outputs
  local only or in institutional storage
