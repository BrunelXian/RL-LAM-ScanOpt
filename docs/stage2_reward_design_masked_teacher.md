# Stage 2 Reward Design for Masked 2D CAE Teacher Workflow

## 1. Core Architecture: Decoupled Step Reward and Terminal Teacher Reward

In Stage 2, the Agent makes `N` sequential masked track-selection decisions, while the 2D Abaqus teacher evaluates the completed trajectory only at the terminal stage. The reward architecture is therefore intentionally decoupled:

- dense step-level penalty for cheap geometric / path-proxy guidance
- sparse terminal CAE teacher reward for physical evaluation

The overall reward can be written as:

```text
R_total = sum_{t=1}^{N-1} r_step(s_t, a_t) + R_terminal(S_final)
```

The step reward must remain cheap and computable without FEA. It exists to shape the search behavior during sequential masked decision making, while the terminal teacher reward carries the physical evaluation signal from the completed scan order.

## 2. Dense Step Penalty

The dense step penalty is:

```text
r_step(s_t, a_t) =
- (lambda_1 * D_jump(a_{t-1}, a_t) / D_max + lambda_2 * P_thermal(s_t, a_t))
```

Where:

- `D_jump` is the physical distance between the previously selected track and the current selected track.
- `D_max` is a characteristic geometry length used for nondimensionalisation.
- The purpose of the jump term is to discourage excessive non-productive travel / jump distance.
- `P_thermal` is a cheap path-based thermal overlap proxy.
- `P_thermal` penalises selecting tracks close to recently scanned tracks, using a time-decayed contribution such as `exp(-Delta t_i)`.
- This is intended to discourage local heat accumulation in SS316L / S316L, whose thermal conductivity is relatively low compared with highly conductive light alloys.

Implementation note:

The exact `lambda` weights should not be assumed correct initially. They must be calibrated against baseline diagnostics before PPO or RL training.

## 3. Sparse Terminal CAE Teacher Reward

The terminal reward is defined as a negative weighted physical cost:

```text
R_terminal =
- (w_1 * Norm(U3) + w_2 * Norm(PEEQ) + w_3 * Norm(sigma_Mises) + w_4 * NormVar(T_preheat))
```

### Physical metrics used in the terminal teacher reward

| Term | Suggested raw form | Recommended nondimensional form | Meaning | Implementation note |
| --- | --- | --- | --- | --- |
| Warpage / out-of-plane deformation | `max(U3) - min(U3)` or `RMS(U3)` | `(max(U3) - min(U3)) / plate_thickness` | Main deformation metric. Important for SS316L / S316L because thermal expansion and thermal gradients can cause significant distortion. | Use a form that is robust under mesh refinement and reporting noise. |
| Accumulated plastic strain | `Area(PEEQ > PEEQ_threshold) / Area_total` | same ratio form is already normalised | Measures the spatial extent of irreversible plastic deformation. | Keep `PEEQ_threshold` configurable. Default threshold: `PEEQ_threshold = 0.002`. |
| Residual von Mises stress | `Mean(sigma_Mises)` | `Mean(sigma_Mises) / sigma_yield_material` | Measures the normalised residual stress level after cooling. | `sigma_yield_material` must be read from material configuration. Do not hard-code `300 MPa` as a permanent value. It may only be mentioned as an approximate SS316L example value. |
| Preheat temperature stability | `Var(T_preheat_1, ..., T_preheat_N)` | `Var(T_preheat) / T_melt^2` or coefficient of variation if more stable | Measures how consistent the thermal base state is before each track is activated. Lower variation implies more stable melt-pool conditions and less over-heating / under-heating risk. | Abaqus output needs a clear extraction rule for preheat temperature, e.g. sample the track centre region immediately before the track activation step. If the exact frame does not exist, use the closest previous output frame and record the extraction rule in metadata. |

## 4. Baseline Shaping for Masked Geometries

In random masked layouts, missing tracks / holes change stiffness, thermal mass, and jump-distance structure. Absolute terminal scores can therefore be biased by geometry difficulty rather than path quality.

The relative reward is:

```text
R_shared = R_terminal_agent - R_terminal_baseline
```

If `R_terminal` is a negative cost, this is equivalent to:

```text
cost_baseline - cost_agent
```

Therefore the Agent receives a positive relative score only when it outperforms the deterministic baseline under the same mask.

Deterministic masked baselines should include:

- `raster_filtered`
- `center_out_filtered`
- optionally `greedy_maximin_distance_filtered` or other deterministic baselines already available in the project

Engineering rule:

Baseline CAE results must be cached by mask hash. Do not rerun deterministic baseline FEA for an identical mask.

Suggested cache fields:

- `mask_hash`
- `baseline_name`
- `trajectory_id`
- `material_id`
- `geometry_id`
- `terminal_metrics`
- `R_terminal_baseline`
- `timestamp`
- `source_odb` or `source_result_path`

## 5. Implementation Guardrails

- This document is a design specification, not a request to immediately modify reward code.
- Do not hard-code material-specific constants into the environment.
- Keep SS316L / S316L as the current example material, but preserve material configurability.
- All terminal teacher metrics should be saved as raw values and normalised values.
- The reward scalar should not be the only output; store a full teacher-label row for analysis.
- Before RL training, run deterministic baseline diagnostics and weight-sensitivity checks.
- Do not run Abaqus inside standard RL `environment.step()`.
- FEA teacher calls should be terminal-stage batch evaluations, offline teacher-label generation, or cached evaluation calls.
