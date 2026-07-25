# Section 3.2 Reward Response Summary With Auxiliary N32

## Source Files Used

- Selected unified source: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\FROZEN_stage3_combined552_plus_N32_teacher_dataset.csv`
- Duplicate RL-ready view audited but not selected: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\FROZEN_stage3_combined552_plus_N32_RL_ready_dataset.csv`
- Native-only reference audited: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\FROZEN_stage3_native_combined552_teacher_dataset.csv`
- Native-only RL-ready reference audited: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\FROZEN_stage3_native_combined552_RL_ready_dataset.csv`
- Native summary reference: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\FROZEN_stage3_native_combined552_summary.json`
- Best-strategy reference: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\stage3_final_native_best_strategy_table.csv`
- Metric/reward timeline reference: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\stage3_final_metric_reward_record_timeline.csv`

No Abaqus run was launched, no ODB file was opened, no source CSV was modified, and no new simulation was generated.

## N32 Source Decision

N32 was selected from the frozen Stage 3 plus-N32 teacher package, not directly from a Stage 2 directory. The selected file contains 884 rows total: 552 native Stage 3 rows and 332 auxiliary fixed-N32 rows. The N32 rows are marked in the source as `stage2_n32_legacy` with `metric_semantic_warning=True`, so they are included as auxiliary fixed-N32 context and not merged into the native variable-N claim scope.

## Exact Columns Used

- U2: `u2_range`
- PEEQ: `peeq_max`
- SurfaceT: `surface_t_proxy`
- Mises: `mises_max`
- NT11: `nt11_mean` where populated
- Gradient: `NOT_PRESENT`
- internal tensile stress: `NOT_PRESENT`

SurfaceT and Mises display values are converted to MPa only for Pa-scale native Stage 3 rows. Auxiliary N32 legacy-scale SurfaceT and Mises values are retained as `legacy_aux_raw` display values and visually marked as auxiliary. Figure stress panels use log y-axes so both native MPa-scale values and N32 legacy-scale values remain visible without asserting cross-scope comparability.

## Counts And Availability

| dataset_scope | source_stage | N | teacher labels | valid U2 | valid PEEQ | valid SurfaceT | valid Mises | valid NT11 | validation statuses | claim role |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| native_stage3 | Stage3_native | N12 | 78 | 78 | 78 | 78 | 78 | 42 | PASS_ODB_EXTRACTED=15; PASS_TEACHER_FIELDS_EXTRACTED=63 | native_multi_N_claim_scope |
| native_stage3 | Stage3_native | N16 | 78 | 78 | 78 | 78 | 78 | 42 | PASS_ODB_EXTRACTED=15; PASS_TEACHER_FIELDS_EXTRACTED=63 | native_multi_N_claim_scope |
| native_stage3 | Stage3_native | N24 | 190 | 190 | 190 | 190 | 190 | 124 | PASS_ODB_EXTRACTED=15; PASS_TEACHER_FIELDS_EXTRACTED=175 | native_multi_N_claim_scope |
| auxiliary_fixed_N32 | Stage2_fixed32_auxiliary | N32 | 332 | 332 | 332 | 332 | 332 | 0 | LEGACY_TEACHER_LABEL_COMPATIBLE=332 | auxiliary_fixed_N32_context |
| native_stage3 | Stage3_native | N40 | 206 | 206 | 206 | 206 | 206 | 140 | PASS_ODB_EXTRACTED=15; PASS_TEACHER_FIELDS_EXTRACTED=191 | native_multi_N_claim_scope |

Required reward metric completeness by N: N12=True, N16=True, N24=True, N32=True, N40=True.

NT11 availability is partial or absent by N: N12=42/78, N16=42/78, N24=124/190, N32=0/332, N40=140/206. Gradient and explicitly named internal tensile stress columns are not present in the selected frozen plus-N32 CSV. Columns `surface_t_proxy_max_tensile_pa` and `surface_t_proxy_max_tensile_mpa` exist in the schema but are not treated as internal tensile stress.

## Distribution Ranges

- `U2`: N12: 2.20100214392005e-05 to 0.00015835521639919999 source_unit (median 3.9525062220491236e-05); N16: 2.9260778319439851e-05 to 0.00027908391211890001 source_unit (median 4.9522939889357076e-05); N24: 2.9523077955673216e-05 to 0.00062659911432659998 source_unit (median 8.0960768855220522e-05); N32: 4.7092336999999998e-05 to 0.0012124716999999999 source_unit (median 0.00042679008999999999); N40: 4.5779268475598656e-05 to 0.0020766726229339001 source_unit (median 0.0001498124456702).
- `PEEQ`: N12: 0.14032196998596189 to 0.175326257944107 source_unit (median 0.14851062744855875); N16: 0.14578714966773981 to 0.17975655198097229 source_unit (median 0.15198192000389094); N24: 0.14921665191650391 to 0.1892505139112472 source_unit (median 0.15729529410600662); N32: 0.1380848805606365 to 0.17761470019817349 source_unit (median 0.14516664888709779); N40: 0.14402535557746879 to 0.20056912302970881 source_unit (median 0.1574980020523071).
- `SurfaceT`: N12: 580.75283200000001 to 585.93068800000003 MPa (median 581.29366400000004); N16: 580.88703999999996 to 587.84166400000004 MPa (median 581.49398399999995); N24: 580.87596799999994 to 592.03532800000005 MPa (median 582.56659200000001); N32: 0.0010144115886554 to 0.0542577414436596 legacy_aux_raw (median 0.0070103019324473503); N40: 581.11904000000004 to 598.53535999999997 MPa (median 583.26944000000003).
- `Mises`: N12: 579.93798400000003 to 579.93984 MPa (median 579.93952000000002); N16: 579.94675199999995 to 579.94867199999999 MPa (median 579.94828800000005); N24: 579.94278399999996 to 579.94489599999997 MPa (median 579.94412799999998); N32: 1.9332757333333328 to 1.933567361933568 legacy_aux_raw (median 1.9332789546666664); N40: 579.92403200000001 to 579.92972799999995 MPa (median 579.92704000000003).

## Generated Outputs

- `N32_data_source_locator.csv`
- `N32_data_source_locator.md`
- `section_3_2_combined_reward_response_with_N32.csv`
- `table_3_2_teacher_metric_availability_with_N32.csv`
- `figure_3_4_reward_metric_distributions_with_N32.csv`
- `figure_3_4_reward_metric_distribution_summary_with_N32.csv`
- `figure_3_4_reward_metric_distributions_with_N32.png/.pdf/.svg`
- `figure_3_5_within_N_normalisation_demo_with_N32.png/.pdf/.svg`

## Interpretation Boundary

Section 3.2 reports reward-response metric availability and distributions only. N12/N16/N24/N40 are the native Stage 3 multi-N claim scope. N32 is auxiliary fixed-N32 / Stage 2 legacy-compatible context. These artifacts do not support a global optimum claim, arbitrary-N generalisation, or physical experiment validation. Reward hierarchy claims belong to Section 3.3, not this section.
