# PPO Final Pool 320 Manuscript Integration Memo

## Recommended Final Positioning

The manuscript should present PPO as a strict policy-gradient evidence chain and large-scale teacher-validated policy-generation addendum, not as the strongest optimiser in the study.

## Recommended Abstract Sentence

"A 320-case PPO-generated scan-order pool was independently evaluated by Abaqus teacher simulations. The pool demonstrated legal executable policy generation and bounded teacher-validated competitiveness, but did not exceed the best records from the mature surrogate-assisted optimisation reference."

## Recommended Results Paragraph

"The final PPO evidence pool contained 320 teacher-metric-extracted candidates across N12, N16, N24 and N40. No PPO candidate produced a new combined552 record. The strongest PPO lexicographic ranks occurred in N12 and N16, while N24 and N40 remained limited in U2/lex performance. Although 106 PPO candidates entered at least one primary top25 region, equal-budget bootstrap comparison indicated weak global enrichment relative to the existing teacher-labelled reference distribution. These results support PPO-generated policy feasibility and bounded competitiveness, rather than dominance over the mature surrogate-assisted optimiser."

## Recommended Discussion Paragraph

"The PPO results clarify the distinction between policy generation and mature surrogate-assisted optimisation. The surrogate-assisted loop remains the stronger optimiser in the present evidence pool, whereas PPO provides a reusable policy-gradient mechanism that can generate legal, executable and teacher-evaluable scan orders. The lack of new records and the weak bootstrap enrichment indicate a practical boundary of the current surrogate-trained PPO formulation, especially for high-N U2/lex optimisation."

## What To Avoid

- Do not say PPO found the best scan orders.
- Do not say PPO outperformed surrogate-assisted optimisation.
- Do not say PPO solved high-N scan-order optimisation.
- Do not treat SurfaceT-only signals as U2/lex dominance.
- Do not claim industrial efficiency improvement without validation.
