# Run21 Claim Boundary

## Safe claims
- Batch28 ODB teacher metrics were ingested successfully for 28/28 cases.
- Combined teacher-labelled dataset now contains 108 cases.
- Combined108 contains N12=24, N16=24, N24=30, and N40=30.
- Batch28 can be compared against combined80 using within-N ranks.
- Combined108 rankings and normalized costs are ready for updated surrogate/RL analysis.
- `S3R19B28_N40_B01_surrogate_top` is teacher-valid and can be included in RL/analysis.
- Specific metric-level improvements can be claimed only if confirmed by combined108 comparison.

## Unsafe claims
- Do not claim trained variable-N RL policy superiority.
- Do not claim surrogate predictions are ground truth.
- Do not claim arbitrary-N generalization.
- Do not claim fixed-32 U2 guard transfer.
- Do not claim final optimum.
- Do not claim physical superiority except for explicitly supported metric-level teacher comparisons.

Verdict: RUN21_BATCH28_INGESTION_AND_COMBINED108_DATASET_ONLY_NO_RL_POLICY_TRAINING
