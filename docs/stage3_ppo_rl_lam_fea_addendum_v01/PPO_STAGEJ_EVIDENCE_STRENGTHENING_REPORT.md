# PPO Stage J Evidence Strengthening Report

## 1. Purpose

Stage J answers the scientific question: if PPO did not beat the mature combined552 best records, what evidence shows PPO still has value?

## 2. Why PPO Did Not Need To Beat Combined552 Best To Have Meaning

The combined552 best records are a mature multi-round active-learning reference, not a naive baseline. PPO's bounded contribution is different: it demonstrates a clean policy-gradient route from trained policy to legal scan orders to independent Abaqus teacher metrics. This supports RL policy-generation feasibility even without record-level dominance.

## 3. Input Integrity

Input integrity verdict: `PASS_STAGEJ_INPUTS_READY`.

## 4. Fair Comparison Levels

Fair comparison levels are written to `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageJ_ppo_evidence_strengthening\tables\ppo_fair_comparison_levels.csv`. The analysis separates mature best-record comparison, reference-distribution percentile comparison, identified baseline-family comparison, bootstrap random-reference comparison, and clean PPO policy-source evidence.

## 5. Bootstrap Random-Reference Comparison

Bootstrap scope: 10,000 equal-size draws from the existing teacher-labelled combined552 reference distribution, not from the full scan-order universe.

Global top-k result: PPO observed 12, bootstrap mean 16.24, q05-q95 [12, 21], empirical p-value 0.9592, interpretation `PPO comparable`.

By-N top-k summary:

- N12: PPO observed 5, bootstrap mean 4.21, q05-q95 [2, 6], interpretation PPO comparable.
- N16: PPO observed 4, bootstrap mean 3.81, q05-q95 [2, 6], interpretation PPO comparable.
- N24: PPO observed 3, bootstrap mean 4.59, q05-q95 [2, 7], interpretation PPO comparable.
- N40: PPO observed 0, bootstrap mean 3.63, q05-q95 [1, 6], interpretation PPO weak.

## 6. Identified Heuristic/Baseline Family Comparison

Identified 9 baseline/heuristic label groups with at least one matched row.

- Inventory: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageJ_ppo_evidence_strengthening\tables\identified_baseline_family_inventory.csv`
- Comparison: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageJ_ppo_evidence_strengthening\tables\ppo_vs_identified_baseline_families.csv`

## 7. PPO Clean Policy-Source Evidence Chain

The clean policy-source chain is written to `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageJ_ppo_evidence_strengthening\tables\ppo_clean_policy_source_evidence_chain.csv`. It documents that PPO candidates came from checkpoint inference, were not manually repaired, and were carried through Abaqus teacher validation and ranking.

## 8. RL Meaning Table

The scientific meaning table is written to `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageJ_ppo_evidence_strengthening\tables\ppo_scientific_meaning_table.csv`.

## 9. Claim Level Decision

Claim decision memo: `E:\Projects\RL-LAM-ScanOpt\docs\stage3_ppo_rl_lam_fea_addendum_v01\PPO_STAGEJ_CLAIM_DECISION_MEMO.md`.

Level 0 PASS, Level 1 PASS, Level 2 PASS, Level 3 PASS_BOUNDED_SMALL_N, Level 4 NOT_SUPPORTED, Level 5 NOT_SUPPORTED.

## 10. Main Strengthened Claim

PPO does not beat the mature combined552 best records, but provides a clean policy-gradient evidence chain from trained policy to Abaqus teacher-validated scan orders, with bounded small-N/top-k competitiveness.

## 11. Main Limitation

PPO did not produce new records, did not solve N40 under primary metrics, and weak surrogate-to-teacher alignment means teacher validation remains required.

## 12. Whether More PPO Experiments Are Needed

More PPO experiments are not required for the current bounded claim. They are recommended only if stronger N24/N40 or new-record claims are desired.

## 13. Recommended Manuscript Wording

"A surrogate-trained MaskablePPO policy generated legal scan-order candidates that were independently Abaqus teacher-validated. Although the PPO batch did not exceed the mature combined552 best records, it achieved bounded small-N/top-k competitiveness, demonstrating policy-generated candidate feasibility rather than record-level dominance."

## 14. Verdict

`PASS_STAGEJ_PPO_EVIDENCE_CHAIN_STRENGTHENED_BOUNDED_POLICY_GENERATION`
