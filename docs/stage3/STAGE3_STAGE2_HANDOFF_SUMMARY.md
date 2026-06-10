# Stage 3 Stage 2 Handoff Summary

Stage 2 is frozen and should be used only as reference evidence. Stage 3 imports selected Stage 2 summary and boundary documents into `docs/stage2_reference/` so the new branch has a compact, GitHub-safe handoff record.

## Imported Reference Targets

- `STAGE2_FINAL_SUMMARY.md`
- `STAGE2_CLAIM_BOUNDARY.md`
- `STAGE2_STAGE3_HANDOFF.md`
- `STAGE2_KEY_RESULTS_TABLE.csv`

## Boundary

Stage 3 does not continue fixed-32 leaderboard tuning. It tests a Variable-N Graph Pointer RL Policy framework for `N_train = {16, 32}` and `N_test = {24, 40}` with within-N ranking and normalized improvement.
