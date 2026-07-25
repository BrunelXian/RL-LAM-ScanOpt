# Section 3.2 Abaqus World-Model Reward Response Summary

## Source Files Used

- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\FROZEN_stage3_native_combined552_teacher_dataset.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\FROZEN_stage3_native_combined552_RL_ready_dataset.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\FROZEN_stage3_native_combined552_summary.json`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\stage3_final_native_best_strategy_table.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\stage3_final_metric_reward_record_timeline.csv`

No Abaqus run was launched and no ODB file was opened. The generated tables and figures use the frozen CSV evidence only.

## Exact Source Columns Used

- U2: `u2_range`
- PEEQ: `peeq_max`
- SurfaceT: `surface_t_proxy`
- Mises: `mises_max`
- Gradient: `NOT_PRESENT`
- internal tensile stress: `NOT_PRESENT`
- NT11 availability audit: `nt11_mean`

Additional tensile-proxy columns observed but not treated as internal tensile stress: `surface_t_proxy_max_tensile_pa`, `surface_t_proxy_max_tensile_mpa`.

## Native Counts By N

| N | teacher labels | valid U2 | valid PEEQ | valid SurfaceT | valid Mises | valid Gradient | valid internal tensile | valid NT11 | validation statuses |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| N12 | 78 | 78 | 78 | 78 | 78 | NOT_PRESENT | NOT_PRESENT | 42 | PASS_ODB_EXTRACTED=15; PASS_TEACHER_FIELDS_EXTRACTED=63 |
| N16 | 78 | 78 | 78 | 78 | 78 | NOT_PRESENT | NOT_PRESENT | 42 | PASS_ODB_EXTRACTED=15; PASS_TEACHER_FIELDS_EXTRACTED=63 |
| N24 | 190 | 190 | 190 | 190 | 190 | NOT_PRESENT | NOT_PRESENT | 124 | PASS_ODB_EXTRACTED=15; PASS_TEACHER_FIELDS_EXTRACTED=175 |
| N40 | 206 | 206 | 206 | 206 | 206 | NOT_PRESENT | NOT_PRESENT | 140 | PASS_ODB_EXTRACTED=15; PASS_TEACHER_FIELDS_EXTRACTED=191 |

## Distribution Ranges

- `U2` (`u2_range`): N12: 2.20100214392005e-05 to 0.00015835521639919999 (median 3.9525062220491236e-05); N16: 2.9260778319439851e-05 to 0.00027908391211890001 (median 4.9522939889357076e-05); N24: 2.9523077955673216e-05 to 0.00062659911432659998 (median 8.0960768855220522e-05); N40: 4.5779268475598656e-05 to 0.0020766726229339001 (median 0.0001498124456702).
- `PEEQ` (`peeq_max`): N12: 0.14032196998596189 to 0.175326257944107 (median 0.14851062744855875); N16: 0.14578714966773981 to 0.17975655198097229 (median 0.15198192000389094); N24: 0.14921665191650391 to 0.1892505139112472 (median 0.15729529410600662); N40: 0.14402535557746879 to 0.20056912302970881 (median 0.1574980020523071).
- `SurfaceT` (`surface_t_proxy`): N12: 580752832 to 585930688 (median 581293664); N16: 580887040 to 587841664 (median 581493984); N24: 580875968 to 592035328 (median 582566592); N40: 581119040 to 598535360 (median 583269440).
- `Mises` (`mises_max`): N12: 579937984 to 579939840 (median 579939520); N16: 579946752 to 579948672 (median 579948288); N24: 579942784 to 579944896 (median 579944128); N40: 579924032 to 579929728 (median 579927040).

The four required reward-response metrics are complete for all native rows: `True`. NT11 columns are present but partially populated; missing NT11 counts are {'N12': 36, 'N16': 36, 'N24': 66, 'N40': 66}. Gradient and explicitly named internal tensile stress columns are not present in the frozen native CSV.

## Generated Outputs

- `table_3_2_teacher_metric_availability.csv`
- `figure_3_4_reward_metric_distributions.csv`
- `figure_3_4_reward_metric_distribution_summary.csv`
- `figure_3_4_reward_metric_distributions.png`
- `figure_3_4_reward_metric_distributions.pdf`
- `figure_3_4_reward_metric_distributions.svg`
- `figure_3_5_within_N_normalisation_demo.png`
- `figure_3_5_within_N_normalisation_demo.pdf`
- `figure_3_5_within_N_normalisation_demo.svg`

## Interpretation Notes

Figure 3.4 uses separate panels for U2, PEEQ, SurfaceT, and Mises so metric scales are not mixed on a single y-axis. The CSV data retain source-column raw values; for plot legibility, SurfaceT and Mises are displayed in MPa. Separate panels avoid visually hiding the smaller-scale U2 and PEEQ responses. Optional Figure 3.5 shows that within-N min-max normalization preserves within-N ordering while avoiding cross-N range conflation, which is why native claims should remain within N.

## Claim Boundary

Native claims are restricted to N12, N16, N24, and N40 within-N comparisons. N32 is excluded from the native Section 3.2 tables and should be treated only as auxiliary legacy-compatible context if mentioned elsewhere. These artifacts do not provide physical experiment validation, do not identify a global optimum, and do not support arbitrary-N generalisation. Reward hierarchy claims are intentionally not made here.
