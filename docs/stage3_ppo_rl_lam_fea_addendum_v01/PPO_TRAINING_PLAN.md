# PPO Training Plan

## Scope

This plan creates a future route from FEA teacher-labelled scan-order data to PPO-specific evidence. It does not train PPO, train a surrogate, generate final candidates, or run Abaqus in the foundation step.

## Stages

### Stage A: Dependency And Data Preflight

Confirm Python runtime, package availability, MaskablePPO importability, native combined552 availability, plus-N32 auxiliary availability, required columns, row counts by N, and native support for N12/N16/N24/N40.

### Stage B: Supervised Surrogate Reward Model Training

Train a supervised reward model on the native combined552 teacher-labelled dataset. The reward target should follow the final physical hierarchy: lexicographic U2 -> PEEQ -> SurfaceT. N32 remains auxiliary unless explicitly enabled later.

### Stage C: PPO Training In Surrogate Reward Environment

Train PPO in `LamScanOrderPPOEnv` using action masking and the supervised surrogate terminal reward. This is surrogate-environment PPO, not online Abaqus PPO.

### Stage D: Checkpoint And Reproducibility Artifact Freeze

Freeze PPO checkpoints, configs, logs, seeds, reward model metadata, package versions, and source hashes before any candidate-generation claims are made.

### Stage E: PPO-Only Inference Candidate Generation

Generate scan-order candidates using PPO policy inference only. The candidate source audit must show that the orders came from the PPO policy checkpoint and not from heuristic or active-learning selection.

### Stage F: Abaqus Teacher Validation Of PPO-Only Candidates

Submit PPO-only candidates to independent Abaqus teacher validation. This is the first stage where PPO-generated scan orders may acquire teacher-validated metrics.

### Stage G: Ingestion And Comparison Against Combined552

Ingest teacher-validated PPO candidates into a separate addendum evidence table. Compare them against initial heuristic baselines, combined552 final bests/top-k, and hybrid active-learning candidates.

## Required Future PPO Artifacts

- `ppo_policy_checkpoint.zip` or `ppo_policy_checkpoint.pt`
- `ppo_training_log.csv`
- `ppo_config.json`
- `ppo_env_config.json`
- `ppo_reward_model_config.json`
- `ppo_parameter_count.json`
- `ppo_seed_manifest.json`
- `ppo_inference_candidate_orders.csv`
- `ppo_candidate_source_audit.csv`

## First Validation Batch

- N12: 8
- N16: 8
- N24: 8
- N40: 8
- Total: 32

## Expansion Rule

If PPO-only candidates are competitive, expand to 48:

- N12: 8
- N16: 8
- N24: 16
- N40: 16

## Stop Conditions

Do not claim PPO policy performance until PPO training artifacts, PPO-only inference outputs, and independent Abaqus teacher validation outputs exist. Do not mix PPO-generated candidates with frozen Stage 3 final evidence unless an explicit addendum ingestion protocol records provenance.
