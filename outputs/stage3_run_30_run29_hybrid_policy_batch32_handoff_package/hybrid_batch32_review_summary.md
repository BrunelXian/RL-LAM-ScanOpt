# Run30 Hybrid Batch32 Review Summary

- Selected batch: `run29_hybrid_policy_batch32`
- Total count: `32`
- Per-N counts: `N12=4`, `N16=4`, `N24=12`, `N40=12`
- N24/N40 share: `24/32`
- Candidate-source composition: `{'graph_pointer_beam_search': 5, 'gnn_reward_local_search': 2, 'surrogate_known_best_local_search': 13, 'N16_new_best_neighborhood': 4, 'sentinel_control': 1, 'N24_surfaceT_best_neighborhood': 3, 'N40_new_best_neighborhood': 4}`
- Selection-bucket composition: `{'gnn_policy_top_candidates': 5, 'hybrid_gnn_surrogate_agreement': 3, 'hybrid_gnn_surrogate_disagreement': 5, 'surrogate_known_best_local_search': 5, 'new_best_local_search': 8, 'N24_surfaceT_best_neighborhood': 3, 'uncertainty_calibration': 3}`
- Expected Abaqus cost: 32 jobs total, with 24 jobs from N24/N40.
- Hybrid batch32 is selected because the user wants a faster daytime validation loop.
- Hybrid batch32 remains unvalidated until future Abaqus teacher validation.
- Run30 did not create CAE/INP files.
