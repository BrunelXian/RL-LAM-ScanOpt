# PPO Stage J Claim Decision Memo

## Claim Level Decision

| Level | Claim | Decision | Evidence |
|---|---|---|---|
| 0 | PPO trained only | PASS | Stage C MaskablePPO checkpoint and training artifacts |
| 1 | PPO generated legal orders | PASS | Stage D PPO-only batch32 legality audit |
| 2 | PPO-generated orders were FEA teacher-validated | PASS | Stage G 32/32 teacher metrics |
| 3 | PPO showed top-k competitiveness | PASS_BOUNDED_SMALL_N | Stage H/J top-k evidence, concentrated in N12/N16 |
| 4 | PPO created new records | NOT_SUPPORTED | Stage H/J new-record audit: 0 |
| 5 | PPO dominated all N / global optimum | NOT_SUPPORTED | N24/N40 primary-metric limitations and no new records |

## What The Paper Can Now Claim

The paper can claim that a surrogate-trained MaskablePPO policy generated legal scan-order candidates, these candidates were independently Abaqus teacher-validated, and the batch showed bounded small-N/top-k competitiveness without producing new combined552 records.

## What The Paper Cannot Claim

The paper cannot claim PPO outperformed the mature active-learning reference, produced a global best, solved N40, solved arbitrary-N scan-order optimisation, or performed online Abaqus RL.

## Should PPO Remain In The Current Paper?

Yes, if framed as bounded policy-generation evidence rather than record-level optimisation. It strengthens the manuscript by adding a complete RL policy-to-teacher-validation chain.

## Are Additional PPO Experiments Recommended?

Not required for the current bounded claim. Additional PPO v02 experiments are recommended only if the user wants to pursue stronger N24/N40 or new-record claims and is willing to spend more Abaqus validation budget.
