# PPO Final Expansion 224 Stage T Repair Report

Timestamp: 2026-06-29T08:37:23.506774

Duplicate groups found: 9
Replacement count: 21

Same-N exact scan-order duplicates were treated as real physical duplicate cases. No manual scan-order mutation was performed; replacements came from the existing Stage T rollout pool.

## Outputs

- Duplicate groups: E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_expansion_224_to_320\stageT_repair_duplicate_orders\stageT_repair_duplicate_groups.csv
- Replacements: E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_expansion_224_to_320\stageT_repair_duplicate_orders\stageT_repair_replacements.csv
- Duplicate audit: E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_expansion_224_to_320\audits\final_expansion_duplicate_audit.csv
- Legality audit: E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_expansion_224_to_320\audits\final_expansion_legality_audit.csv
- Novelty audit: E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_expansion_224_to_320\audits\final_expansion_novelty_audit.csv
- Repaired master CSV: E:\Projects\RL-LAM-ScanOpt\outputs\stage3_ppo_final_expansion_224_to_320\selected_candidates\PPO_FINAL_EXPANSION_224_SELECTED_MASTER.csv

## Final Counts

- N12: 32
- N16: 32
- N24: 80
- N40: 80
- Total: 224

## Claim Boundary

This repair only fixes handoff candidate uniqueness. It does not run solver, datacheck, enqueue, ODB extraction, PPO training, surrogate training, or teacher validation.
