# PPO Surrogate Target Schema

- Schema: `ppo_surrogate_reward_targets_v01`
- Primary target: `reward_lex_u2_peeq_surfacet`
- Formula: `1.0*reward_u2_rank + 0.1*reward_peeq_rank + 0.01*reward_surfacet_rank`
- Normalization: `within-N only; no cross-N leakage`
- Reward direction: `larger_is_better`
- Mises role: `diagnostic_only`
- Strict threshold status: `NOT_FOUND`

## Target Columns

- `reward_lex_u2_peeq_surfacet`
- `reward_u2_primary`
- `reward_constrained`
- `cost_u2_norm`
- `cost_peeq_norm`
- `cost_surfacet_norm`
- `cost_mises_norm`
- `reward_strict_penalty_guard_like`

## Metric Columns

- `u2`: `u2_range`
- `peeq`: `peeq_max`
- `surfacet`: `surface_t_proxy`
- `mises`: `mises_max`
