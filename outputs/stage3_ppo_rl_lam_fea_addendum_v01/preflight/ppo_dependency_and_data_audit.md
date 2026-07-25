# PPO Dependency And Data Audit

- Timestamp UTC: `2026-06-23T09:36:15.892530+00:00`
- Python executable: `D:\XianLab\envs\conda\torch-gpu\python.exe`
- Python version: `3.11.13 | packaged by Anaconda, Inc. | (main, Jun  5 2025, 13:03:15) [MSC v.1929 64 bit (AMD64)]`
- Platform: `Windows-10-10.0.26200-SP0`
- Verdict: `PASS_PPO_ADDENDUM_DEPENDENCIES_AND_DATA_READY`

## Dependencies

| Dependency | Available | Version | Error |
| --- | ---: | --- | --- |
| torch | True | 2.6.0+cu124 |  |
| numpy | True | 2.1.2 |  |
| pandas | True | 2.3.1 |  |
| gymnasium | True | 1.2.3 |  |
| stable_baselines3 | True | 2.8.0 |  |
| sb3_contrib | True | 2.8.0 |  |
| maskable_ppo | True | 2.8.0 |  |

## Native Combined552 Dataset

- Path: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\FROZEN_stage3_native_combined552_teacher_dataset.csv`
- Exists: `True`
- Rows: `552`
- N values: `[12, 16, 24, 40]`
- Row counts by N: `{'12': 78, '16': 78, '24': 190, '40': 206}`

## Plus-N32 Auxiliary Dataset

- Path: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_78_final_evidence_freeze_package\FROZEN_stage3_combined552_plus_N32_RL_ready_dataset.csv`
- Exists: `True`
- Rows: `884`
- N values: `[12, 16, 24, 32, 40]`
- Row counts by N: `{'12': 78, '16': 78, '24': 190, '40': 206, '32': 332}`

## Required Column Groups

| Group | Alternatives | Found | Available or Mappable |
| --- | --- | --- | ---: |
| n | `['n']` | `['n']` | True |
| strategy_identifier | `['strategy_name', 'candidate_id']` | `['strategy_name']` | True |
| scan_order | `['order_json', 'order_compact', 'scan_order']` | `['order_json', 'order_compact']` | True |
| u2_range | `['u2_range']` | `['u2_range']` | True |
| peeq_max | `['peeq_max']` | `['peeq_max']` | True |
| surface_t_proxy | `['surface_t_proxy']` | `['surface_t_proxy']` | True |
| mises_max | `['mises_max']` | `['mises_max']` | True |
| teacher_validation_status | `['teacher_validation_status']` | `['teacher_validation_status']` | True |

## Readiness Checks

- Native N12/N16/N24/N40 present: `True`
- N32 present in native dataset: `False`
- N32 present in plus-N32 auxiliary dataset: `True`
- MaskablePPO importable: `True`

## Missing Or Partial

- None.
