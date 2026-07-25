# Run32A Claim Boundary

## Safe Claims
- Run32A ingests audited Stage 2 N32 legacy teacher labels into a Stage 3-compatible schema.
- Run32A creates a deduplicated N32 training table with one row per unique order_hash.
- Run32A creates combined172_plus_N32 datasets with N32 as an additional N group.
- N32 provides a large intermediate-N teacher-labelled group for future surrogate/GNN/graph-pointer training.
- N32 compatibility has metric-semantics warnings.
- Rows with missing mapped Mises proxy values are explicitly flagged and use a conservative rank/reward compatibility fill.

## Unsafe Claims
- Do not claim N32 data is native Stage 3 teacher validation.
- Do not claim peeq_guard is exactly Stage 3 peeq_max.
- Do not claim mises_P95_top_band is exactly global Stage 3 mises_max.
- Do not claim conservative Mises fill values are literal teacher measurements.
- Do not claim no compatibility limitations.
- Do not claim model improvement.
- Do not claim GNN-RL superiority.
- Do not claim arbitrary-N generalization.
- Do not claim online RL.
- Do not claim any new Abaqus/ODB extraction was performed.
