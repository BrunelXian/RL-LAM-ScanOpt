# Probe60 Base JNL Object-Name Audit

Verdict: `WARNING_BASE_JNL_OBJECT_MAPPING_PARTIAL`

## Summary

| N | Heat sets | Scan steps | Cool steps | Body heat loads | Mapping status |
|---:|---:|---:|---:|---:|---|
| 12 | 12 | 1 | 1 | 1 | `MANUAL_OBJECT_MAPPING_REQUIRED` |
| 16 | 16 | 1 | 1 | 1 | `MANUAL_OBJECT_MAPPING_REQUIRED` |
| 24 | 24 | 1 | 1 | 1 | `MANUAL_OBJECT_MAPPING_REQUIRED` |
| 40 | 40 | 1 | 1 | 1 | `MANUAL_OBJECT_MAPPING_REQUIRED` |

## Heat-Load Mapping Status

The journals provide N-specific heat-region set names (`set_body_heat_XX`) matching N for each base. They do not provide an N-step or N-load scan sequence in journal text. Each base journal records one scan step, one cool step, and one body heat flux load bound to `set_body_heat_00`.

Classification: `MANUAL_OBJECT_MAPPING_REQUIRED`.

Generic export is blocked by default because it would preserve the base journal's single recorded heat-load mapping rather than creating candidate-specific scan-order INPs.

## Outputs

- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_cae_probe60_generation\base_jnl_object_name_audit.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_cae_probe60_generation\base_jnl_object_name_audit.md`
