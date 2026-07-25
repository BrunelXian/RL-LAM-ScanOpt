# PPO Final Evidence Freeze Report

## 1. Purpose

Freeze the completed PPO + LAM + FEA addendum evidence and provide manuscript-facing claim support.

## 2. Evidence Chain Overview

FEA teacher-labelled native combined552 data -> supervised surrogate terminal reward model -> MaskablePPO training -> PPO checkpoint inference -> PPO-only batch32 -> Abaqus CAE/INP handoff -> solver completion -> ODB teacher metric extraction -> ranking against combined552.

## 3. Frozen Inputs And Outputs

- Frozen table directory: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageI_final_ppo_evidence_freeze\frozen_tables`
- Hash table: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageI_final_ppo_evidence_freeze\hashes\FROZEN_PPO_file_hashes.csv`
- Manifest: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\stageI_final_ppo_evidence_freeze\stageI_ppo_final_evidence_freeze_manifest.json`

## 4. Surrogate Reward Model Summary

The surrogate model was trained on native combined552 only, without N32. The selected model was HistGradientBoostingRegressor for `reward_lex_u2_peeq_surfacet`, with validation Spearman 0.8786 and Pearson 0.8863.

## 5. PPO Training Summary

MaskablePPO + MlpPolicy was trained in the surrogate reward environment for 200352 timesteps. Parameter count was 72937. Checkpoint: `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_rl_lam_fea_addendum_v01\ppo_training\checkpoints\maskable_ppo_lam_scan_order_final.zip`.

## 6. PPO Candidate Generation Summary

Stage D generated 32 PPO-only candidates from checkpoint inference: 8 each for N12, N16, N24, and N40. Candidate orders were not modified in later stages.

## 7. Abaqus Teacher-Validation Execution Summary

Stage E converted all 32 PPO candidates into Abaqus CAE/INP cases. Solver execution later completed 32/32 and produced nonzero ODB files, with nonfatal warnings only.

## 8. Teacher Metric Extraction Summary

Stage G extracted teacher metrics for 32/32 PPO cases: U2, PEEQ, S/SurfaceT proxy, Mises, and NT11/temperature output metadata.

## 9. Ranking And Comparison Summary

Best PPO lexicographic candidates:

- N12: `PPOV01_N12_B08_stochastic_highreward`, reference lex rank 6
- N16: `PPOV01_N16_B02_surrogate_top`, reference lex rank 2
- N24: `PPOV01_N24_B08_stochastic_highreward`, reference lex rank 134
- N40: `PPOV01_N40_B07_novelty_tophalf`, reference lex rank 147

## 10. New-Record Audit

New-record count versus combined552: 0. No PPO candidate beat the prior combined552 best in the primary ranking evidence.

## 11. Top-K Competitiveness Audit

Top-k count: 12 total. Distribution: N12=5, N16=4, N24=3, N40=0. The result supports bounded small-N competitiveness.

## 12. Surrogate-Vs-Teacher Alignment

Overall Spearman: 0.2790. Overall Pearson: 0.2092. Alignment is weak positive, with 1 false positive and 2 true positives.

## 13. Recovery-Anchor Duplicate Audit

`PPOV01_N12_B02_surrogate_top` matched `S3R69SNR_N12_B01_n12_run66_local` by parsed_order_equality and is not a novel PPO discovery.

## 14. Manuscript-Safe Claims

Use the bounded final claim boundary in `E:\Projects\RL-LAM-ScanOpt\docs\stage3_ppo_rl_lam_fea_addendum_v01\PPO_FINAL_CLAIM_BOUNDARY.md`. The strongest concise claim is: PPO batch32 was teacher-validated and small-N/top-k competitive, but produced no new combined552 records.

## 15. Unsafe Claims

Do not claim PPO produced a global best, dominated all native N, solved arbitrary-N scan-order optimisation, was trained online in Abaqus, is experimentally validated, or is first in the world.

## 16. Limitations

The PPO policy was trained in a surrogate environment. Surrogate-to-teacher alignment was weak. N40 primary-metric competitiveness was not observed. The recovery anchor is not a novel PPO discovery.

## 17. Next Manuscript Action

Integrate the evidence-chain, performance-summary, and claim-support tables into the manuscript addendum, preserving the bounded claim language.

## 18. Verdict

`PASS_PPO_FINAL_EVIDENCE_FREEZE_TEACHER_VALIDATED_COMPETITIVE_BOUNDED`
