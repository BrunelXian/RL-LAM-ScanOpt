# Run 06 Variable-N Probe60 Candidate Order Generation Report

## Executive Verdict

WARNING_VARIABLE_N_PROBE60_POLICY_PROXY_FALLBACK_USED

Generated 15 legal scan-order designs for each N in `[12, 16, 24, 40]`, total `60`. A01-A09 are deterministic engineering baselines. A10-A15 are deterministic graph pointer proxy candidates, not trained or teacher validated.

## Guardrails

- No Abaqus jobs.
- No datacheck.
- No ODB opened.
- No CAE/INP/JNL generated.
- No abqjobpilot execution.
- No teacher validation.

## Claim Boundary

These are candidate-generation outputs only. Variable-N is not yet a validated RL result, and no shared full-32 U2 guard is used.

## Outputs

- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_06_variable_n_probe60_candidate_order_generation\variable_N_probe60_candidate_orders.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_06_variable_n_probe60_candidate_order_generation\variable_N_probe60_legality_audit.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_06_variable_n_probe60_candidate_order_generation\variable_N_probe60_structural_summary.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_06_variable_n_probe60_candidate_order_generation\variable_N_probe60_pairwise_diversity.csv`
- `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_run_06_variable_n_probe60_candidate_order_generation\variable_N_probe60_candidate_design_manifest.json`
- `E:\Projects\RL-LAM-ScanOpt\artifacts\manifests\stage3_run_06_manifest.json`
