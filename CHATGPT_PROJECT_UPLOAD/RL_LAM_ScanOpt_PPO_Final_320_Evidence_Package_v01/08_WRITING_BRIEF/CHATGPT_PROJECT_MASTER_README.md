# ChatGPT Project Master README

## What This Package Is

This is a compact evidence package for manuscript writing and claim checking around the final 320-case PPO addendum for RL-LAM-ScanOpt.

## How To Use It

Start with:

1. `08_WRITING_BRIEF/PPO_FINAL_320_ONE_PAGE_SUMMARY.md`
2. `08_WRITING_BRIEF/PPO_FINAL_320_SAFE_MANUSCRIPT_PARAGRAPH.md`
3. `01_FINAL_STAGE_X_FREEZE/PPO_FINAL_POOL_320_EVIDENCE_FREEZE_REPORT.md`
4. `01_FINAL_STAGE_X_FREEZE/PPO_FINAL_POOL_320_FINAL_CLAIM_BOUNDARY.md`
5. `02_STAGE_W_ANALYSIS/PPO_FINAL_POOL_320_STAGEW_RANKING_AND_COMPARISON_REPORT.md`

Use CSV tables only when checking exact numeric claims.

## Final PPO Conclusion

A 320-case PPO-generated scan-order pool was teacher-metric extracted using Abaqus. PPO produced legal, executable and independently teacher-evaluated scan orders with bounded small-N competitiveness and SurfaceT-related signals, but produced 0 new records against native combined552 and did not outperform the mature surrogate-assisted optimiser.

## Safe Claims

- 320 PPO-generated scan orders were teacher-metric extracted.
- PPO generated legal and executable scan-order permutations.
- PPO showed bounded small-N competitiveness.
- PPO showed SurfaceT-related signals but not U2/lex dominance.
- Final physical claims are based on Abaqus teacher metrics, not surrogate scores.

## Unsafe Claims

- PPO beat combined552 best.
- PPO outperformed the mature surrogate-assisted optimiser.
- PPO solved N24/N40 scan-order optimisation.
- PPO demonstrated experimentally validated industrial-efficiency improvement.
- This was online Abaqus-in-the-loop PPO.

## Manuscript Placement

Use PPO as a policy-gradient evidence-chain addendum in Methods/Results/Discussion. Do not present PPO as the best optimiser in the study.

## Distinctions

- Mature surrogate-assisted optimiser: the stronger reference optimisation loop represented by combined552.
- Surrogate-trained PPO policy generation: PPO trained on surrogate rewards to generate scan-order candidates.
- Abaqus teacher metrics: independent finite-element teacher evaluation used for final physical claims.
- Experimental validation: not performed here.
