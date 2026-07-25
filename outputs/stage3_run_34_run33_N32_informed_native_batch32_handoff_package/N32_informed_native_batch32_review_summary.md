# Run34 N32-Informed Native Batch32 Review Summary

- Selected batch: `Run33 Option A - N32-informed native batch32`
- Total candidates: `32`
- Per-N counts: `N12=4`, `N16=4`, `N24=12`, `N40=12`
- No N32 candidates are included.
- N24/N40 share: `24/32`
- Expected Abaqus cost: 32 jobs total, with 24 jobs from N24/N40.
- Exact overlap with superseded Run31/Run30 batch32: `0/32`.
- This batch uses N32 as training information but validates only native Stage 3 N values.
- This is the clean replacement for the abandoned/superseded Run31 batch.
- The batch remains unvalidated until future Abaqus teacher validation.
- Run34 did not create CAE/INP files.

## Candidate Source Composition
- `graph_pointer_greedy`: `4`
- `surrogate_top_predicted`: `4`
- `hybrid_gnn_surrogate_agreement`: `4`
- `hybrid_gnn_surrogate_disagreement`: `4`
- `uncertainty_calibration`: `2`
- `diversity_coverage`: `2`
- `sentinel_control`: `2`
- `N24_calibration_neighborhood`: `5`
- `N40_new_best_neighborhood`: `5`

## Selection Bucket Composition
- `graph_pointer_top`: `4`
- `surrogate_top_predicted`: `4`
- `hybrid_gnn_surrogate_agreement`: `4`
- `hybrid_gnn_surrogate_disagreement`: `4`
- `uncertainty_calibration`: `2`
- `diversity_coverage`: `2`
- `sentinel_control`: `2`
- `N24_calibration_neighborhood`: `5`
- `N40_new_best_neighborhood`: `5`
