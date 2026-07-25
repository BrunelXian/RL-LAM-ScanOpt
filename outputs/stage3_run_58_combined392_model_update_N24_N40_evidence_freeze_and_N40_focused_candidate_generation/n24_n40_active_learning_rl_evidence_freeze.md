# N24/N40 Active-Learning RL Evidence Freeze

Verdict: `RUN58_N24_N40_EVIDENCE_FREEZE_READY_FULL_VARIABLE_N_LIMITED_BY_N12_N16`

N24/N40 active-learning evidence is mature enough to freeze at 160 native teacher rows each; full variable-N RL remains limited by N12/N16 at 36 rows each.

This freeze preserves the distinction between mature N24/N40 focused evidence and the still-limited full variable-N setting. N32 rows remain legacy-compatible auxiliary data, not native Stage 3 teacher validation.

## Timeline
- combined172_baseline: N24=54, N40=54 - pre-Run36 native baseline before N24/N40 focused loops
- Run36_N32_informed_native_batch32: N24=66, N40=66 - N24 and N40 U2 records refreshed; no N32 cases in validation
- Run41_N24_N40_focused_batch60: N24=96, N40=96 - near-top density increased; N40 PEEQ record observed earlier, U2 did not extend
- Run46_constrained_batch32: N24=112, N40=112 - N24 U2/reward and N40 reward gains supported constrained selection
- Run51_stricter_constrained_batch32: N24=128, N40=128 - N24 U2 and N40 strict/reward behavior strengthened; raw penalty records still limited
- Run56_calibrated_batch64: N24=160, N40=160 - N40 U2/reward-family records advanced; N24 produced density but no new combined328 bests
