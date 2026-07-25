# Stage 3 Final Evidence Freeze Summary

Final verdict: `PASS_STAGE3_FINAL_EVIDENCE_FREEZE_READY_WITH_BOUNDED_NATIVE_N_CLAIMS`

## Final Datasets
- Native: combined552
- Auxiliary plus-N32: combined552_plus_N32
- Native counts: {12: 78, 16: 78, 24: 190, 40: 206}
- N32 legacy auxiliary count: 332

## Final Best Strategy Highlights
| n | U2 | u2_primary | constrained_reward | strict_penalty_guard | penalty_repair |
| --- | --- | --- | --- | --- | --- |
| 12 | S3R69SNR_N12_B16_n12_uncertainty | S3R74FSD_N12_B02_n12_penalty | S3R74FSD_N12_B02_n12_penalty | S3R74FSD_N12_B02_n12_penalty | S3R74FSD_N12_B01_n12_penalty |
| 16 | S3R74FSD_N16_B13_n16_run71_local | S3R74FSD_N16_B01_n16_penalty | S3R74FSD_N16_B01_n16_penalty | S3R74FSD_N16_B01_n16_penalty | S3R74FSD_N16_B01_n16_penalty |
| 24 | S3R49SCN_N24_B09_median_guard | S3R44CNS_N24_B16_uncertainty | S3R44CNS_N24_B16_uncertainty | S3R44CNS_N24_B16_uncertainty | S3R69SNR_N24_B04_n24_uncertainty |
| 40 | S3R64VNR_N40_B15_n40_u2ret_anchor | S3R64VNR_N40_B03_n40_penalty_anchor | S3R64VNR_N40_B03_n40_penalty_anchor | S3R64VNR_N40_B03_n40_penalty_anchor | S3R64VNR_N40_B03_n40_penalty_anchor |

## Interpretation
- N12/N16 reached 78 native teacher rows each after recovery and final diagnostic loops.
- N24/N40 remain mature anchors with 190 and 206 native teacher rows respectively.
- GNN and graph-pointer diagnostics remain auxiliary only.
- Bounded teacher-validated claims over tested native N values N12/N16/N24/N40 only.

## Recommended Next Step
Prepare paper/report/ARA package; do not generate more Stage 3 candidates by default.
