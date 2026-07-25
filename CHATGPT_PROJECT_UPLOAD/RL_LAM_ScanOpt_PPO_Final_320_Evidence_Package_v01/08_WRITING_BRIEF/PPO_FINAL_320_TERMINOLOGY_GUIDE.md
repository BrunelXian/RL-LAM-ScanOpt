# PPO Final 320 Terminology Guide

- teacher-metric-extracted: Abaqus-derived teacher metrics were extracted for a case.
- teacher-evaluated: a case has finite-element teacher metrics available.
- teacher-validated: use cautiously; here it means teacher metrics were extracted, not that a candidate is superior.
- surrogate reward model: supervised emulator trained from teacher-labelled data and used for PPO reward.
- mature surrogate-assisted optimiser: the stronger combined552 reference optimisation evidence.
- MaskablePPO policy generation: action-masked PPO generating legal scan-order permutations.
- combined552 reference: native Stage 3 frozen teacher-labelled reference dataset.
- top-k competitiveness: entering top10/top25 regions under defined metrics/ranks.
- equal-budget bootstrap: comparison against same-size draws from existing teacher-labelled reference distribution.
- bounded no-new-records: teacher-evaluated evidence exists, but no new best records were found.
- SurfaceT signal vs U2/lex dominance: SurfaceT enrichment does not imply primary U2->PEEQ->SurfaceT lexicographic dominance.
- industrial-efficiency proxy: sequence descriptor, not physically validated efficiency.
