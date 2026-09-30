| CASE | Configuration | Solver | Last completed / fatal step | Area error | Distortion warnings | Cutbacks | Neg. eigen | Stabilization energy | Verdict |
|---|---|---|---|---|---:|---:|---:|---|---|
| A | baseline | FAILED | 39 / 40 | Yes: 721,723,725 | 5 | 59 | 9 | N/A | BASELINE_FAILURE_CONFIRMED |
| B | ALE | FAILED | 41 / 42 | Not observed | 17 | 105 | 15 | N/A | ALE_GEOMETRY_IMPROVED_BUT_NOT_SOLVED |
| C | stabilization | FAILED | 5 / 6 | Not observed | 0 | 5 | 0 | ALLSD/ALLIE = 0.4874% | STABILIZATION_NO_MEANINGFUL_IMPROVEMENT |
| D | ALE + stabilization | FAILED | 5 / 6 | Not observed | 1 | 5 | 0 | ALLSD/ALLIE = 0.1103% | ALE_STAB_NO_MEANINGFUL_IMPROVEMENT |
