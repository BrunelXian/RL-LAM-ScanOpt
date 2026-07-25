# Figure 3.10 Scan-Order Data Export

Generated CSV data for replotting `figure_3_10_best_scan_order_patterns_across_N.png`.

Files:

- `figure_3_10_best_scan_order_patterns_across_N_data_long.csv`: native N12/N16/N24/N40 only, one row per scan position.
- `figure_3_10_best_scan_order_patterns_across_N_data_wide.csv`: native N12/N16/N24/N40 only, one row per N.
- `figure_3_10_best_scan_order_patterns_across_N_data_summary.csv`: native selected strategy metadata.
- `figure_3_10_best_scan_order_patterns_across_N_plus_N32_aux_data_long.csv`: native rows plus auxiliary N32, one row per scan position.
- `figure_3_10_best_scan_order_patterns_across_N_plus_N32_aux_data_wide.csv`: native rows plus auxiliary N32, one row per N.
- `figure_3_10_best_scan_order_patterns_across_N_plus_N32_aux_data_summary.csv`: selected strategy metadata including N32.

N32 note:

- N32 selected row: `RL20_A15_V01` from frozen plus-N32 constrained reward rank/value.
- N32 scan order source: `D:\Projects\RL-LAM-ScanOpt\LDED_2D_CAE_Framework\cae_models\32track_rl_approved20_v01\RL20_A15_V01\scan_order.json`.
- N32 claim role: `auxiliary_fixed_N32_context_not_native_multi_N_claim`.
- N32 metric semantic warning: `True`.
- N32 legacy compatibility status: `LEGACY_COMPATIBLE_WITH_WARNINGS`.

Do not treat N32 as native Stage 3 multi-N evidence. It is auxiliary fixed-N32 context only.
