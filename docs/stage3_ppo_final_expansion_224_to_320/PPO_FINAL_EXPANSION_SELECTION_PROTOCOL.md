# PPO Final Expansion Selection Protocol

## Purpose

Stage T creates a fixed-budget 224-case PPO-generated expansion set intended to bring the cumulative PPO teacher-validation pool from 96 cases to 320 cases after future validation.

This is not another open-ended reward-redesign loop. Stage S showed weak surrogate-to-teacher alignment for v03, so Stage T uses existing PPO checkpoints and controlled selection buckets to build a broad, auditable candidate pool.

## Allocation

| N | Selected candidates |
|---:|---:|
| 12 | 32 |
| 16 | 32 |
| 24 | 80 |
| 40 | 80 |
| **Total** | **224** |

## Batch Structure

| Batch | Allocation |
|---|---|
| final_expansion_batch01 | N12=16, N16=16 |
| final_expansion_batch02 | N12=16, N16=16 |
| final_expansion_batch03 | N24=32 |
| final_expansion_batch04 | N24=32 |
| final_expansion_batch05 | N24=16, N40=16 |
| final_expansion_batch06 | N40=32 |
| final_expansion_batch07 | N40=32 |

## Selection Buckets

| Bucket | Target share | Role |
|---|---:|---|
| quality-seeking | 35% | Prefer high available PPO/surrogate score from existing models. |
| diversity-seeking | 25% | Maximize scan-order distance from already selected candidates. |
| industrial-efficiency-seeking | 20% | Prefer smoother/shorter proxy paths using sequence descriptors. |
| novelty-seeking | 10% | Prefer candidates distant from combined552 and previous PPO pools. |
| baseline-proximity / conventional-comparison | 10% | Preserve candidates near recognizable conventional patterns for later comparison. |

## Industrial-Efficiency Proxy Descriptors

The expansion records:

- `mean_abs_jump`
- `max_abs_jump`
- `long_jump_count`
- `adjacent_fraction`
- `total_travel_proxy`
- `jump_variance`
- `local_continuity_score`
- `path_complexity_score`

These are sequence descriptors only. They are not physical teacher metrics and must not be claimed as physically validated efficiency improvements until separately justified or validated.

## Claim Boundary

Safe after Stage T: a legal PPO-generated 224-case candidate expansion set is ready for later CAE/INP handoff.

Unsafe after Stage T: physical improvement, teacher validation, industrial efficiency validation, or superiority over combined552.
