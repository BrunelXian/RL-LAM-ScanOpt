# N32/N64 Policy Extrapolation Report

## Purpose

Infer predicted N32 and N64 scan-order candidates from existing trained RL/PPO policy evidence, subject to architecture constraints.

## Architecture Boundary

The final Stage X MaskablePPO policy uses a `Discrete(40)` action space. It may be used for N32 with action masking, but it must not be used directly for N64. N64 requires a variable-N / graph-pointer / pointer-policy checkpoint with explicit N64 or max_N >= 64 support.

## Policy/Checkpoint Inventory

See `policy_checkpoint_inventory.csv`. Inventory verdict: WARNING_STAGEY_POLICY_CAPABILITY_PARTIAL.

## Surrogate/Model Inventory

See `surrogate_model_inventory.csv`. Existing models are used for inventory only unless compatibility is explicit.

## Existing N32/N64 Artifact Recovery

Recovered candidate tables were written for N32 and N64. Recovered artifacts are not new Stage Y policy inference unless stated.

## N32 Inference Result

Status: RAN. N32 used final MaskablePPO Discrete(40) with a custom N32 action mask. Output is prediction only.

## N64 Inference Result

Status: BLOCKED: no safe variable-N / graph-pointer / pointer-policy checkpoint with explicit N64 or max_N>=64 support found. The final Discrete(40) MaskablePPO was blocked for N64 by architecture.

## Best N32 Predicted Candidate

```json
{
  "rank": 1,
  "N": 32,
  "strategy_name": "POLICYX_N32_0696_stochastic",
  "evidence_tier": "policy_inferred_prediction_only",
  "teacher_metrics_available": false,
  "policy_source": "final MaskablePPO Discrete(40) with N32 action mask",
  "policy_checkpoint": "E:\\Projects\\RL-LAM-ScanOpt\\outputs\\stage3_ppo_rl_lam_fea_addendum_v01\\ppo_training\\checkpoints\\maskable_ppo_lam_scan_order_final.zip",
  "inference_mode": "stochastic",
  "generated_by": "StageY_N32_policy_inference",
  "u2_range": NaN,
  "peeq_max": NaN,
  "surface_t_proxy": NaN,
  "mises_max": NaN,
  "predicted_reward": 0.7920603971374354,
  "predicted_u2": NaN,
  "predicted_peeq": NaN,
  "predicted_surface_t": NaN,
  "policy_score": -24.22014211677015,
  "total_travel_proxy": 183.0,
  "mean_abs_jump": 5.903225806451613,
  "max_abs_jump": 17.0,
  "long_jump_count": 2,
  "adjacent_fraction": 0.0,
  "order_compact": "11,1,3,15,21,23,31,25,29,19,27,17,13,9,5,0,7,24,28,20,22,30,26,18,16,6,4,2,8,10,14,12",
  "order_hash": "0a9379cabb4547ef",
  "order_json_path": "E:\\Projects\\RL-LAM-ScanOpt\\outputs\\stage3_policy_extrapolation_N32_N64_predicted_optima\\scan_orders\\N32\\scan_order_POLICYX_N32_0696_stochastic.json",
  "safe_claim": "N32 policy-inferred prediction ranked by frozen v01 surrogate; not Abaqus teacher-metric-extracted",
  "caveat": "surrogate score is prediction only; no physical optimum claim"
}
```

## Best N64 Predicted Candidate

```json
{}
```

## Industrial-Efficiency Proxy Interpretation

Travel and jump descriptors are path proxies only. They are not Abaqus teacher metrics and cannot establish physical or industrial optimality.

## Whether Teacher Metrics Exist

Any recovered teacher metrics are reported as recovered existing evidence. Stage Y did not run Abaqus, open ODBs, or extract metrics.

## Claim Boundary

See `E:\Projects\RL-LAM-ScanOpt\docs\stage3_policy_extrapolation_N32_N64_predicted_optima\N32_N64_POLICY_EXTRAPOLATION_CLAIM_BOUNDARY.md`.

## Recommended Wording

"N32 candidates are policy-inferred predictions from the trained Discrete(40) MaskablePPO with N32 masking. N64 policy inference was blocked unless an explicit variable-N policy supporting N64 is identified. These outputs are predicted scan orders only and are not teacher-validated optima."

## Final Verdict

PASS_N32_COMPLETED_N64_BLOCKED_BY_ARCHITECTURE
