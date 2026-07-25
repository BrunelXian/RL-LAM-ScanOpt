# PPO Final 320 One-Page Summary

## Pool Composition

- Total PPO teacher-metric-extracted cases: 320
- v01: 32
- v02K2: 32
- v03: 32
- final expansion: 224
- N12: 40
- N16: 40
- N24: 120
- N40: 120

## Stage Chain

Stage A established PPO environment foundation. Stage B trained a surrogate reward model. Stage C trained PPO v01. Stage D/E/G/H/I/J completed v01 candidate generation, handoff, teacher extraction, ranking, freeze and evidence strengthening. Stage K/K2 built targeted v02 for N24/N40. Stage P/R/S built and ranked v03. Stage T generated final 224 expansion candidates. Stage V extracted final expansion teacher metrics. Stage W ranked the cumulative 320 pool. Stage X froze final evidence.

## Main Results

- New records vs combined552: 0
- Primary top25-any count: 106
- Equal-budget bootstrap interpretation: weak
- Best primary lex N12: rank 6
- Best primary lex N16: rank 2
- Best primary lex N24: rank 114
- Best primary lex N40: rank 147

## Final Interpretation

PPO is evidence for feasible policy-gradient scan-order generation under teacher evaluation. It is not evidence that PPO surpassed the mature surrogate-assisted optimiser.
