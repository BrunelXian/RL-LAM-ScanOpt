# PPO Scan-Order Environment Specification

## Purpose

This document specifies the Stage 3 PPO addendum environment foundation. The evidence chain is:

FEA teacher-labelled dataset -> PPO-compatible scan-order environment -> PPO policy training artifacts -> PPO-only candidate generation -> later independent Abaqus teacher validation.

This is not online Abaqus PPO. PPO will be trained in a surrogate or reduced-order reward environment derived from FEA teacher-labelled LDED scan-order data, then PPO-generated scan orders will be handed off for independent Abaqus teacher validation.

## Environment

- Environment name: `LamScanOrderPPOEnv`
- Supported native N values: `[12, 16, 24, 40]`
- Maximum action dimension: `40`
- Episode definition: one full scan-order construction episode of length `N`
- Solver dependency: none
- ODB dependency: none
- File writes during `step`: none

## Action Space

The action space is `Discrete(40)` with action masking.

Valid actions are unvisited tracks from `0` to `N - 1`. Invalid actions are:

- Tracks outside current `N`
- Already visited tracks
- Any action attempted after the episode has terminated

The environment exposes `action_masks()` as a length-40 Boolean mask compatible with `sb3_contrib` MaskablePPO conventions. Illegal actions should normally be prevented by the mask. If an illegal action reaches `step`, the environment returns a large negative reward and terminates.

## Observation

The observation uses a fixed-size max-N representation for N up to 40. The foundation implementation uses a flattened vector with:

- Current `N` normalized by 40
- Current step normalized by `N`
- Previous selected track normalized, or `-1` before any track is selected
- Visited mask length 40
- Valid-action mask length 40
- Per-track features length `40 x k`, where `k = 10`

Per-track features are:

- Normalized track index
- Parity
- Center-distance proxy
- Edge-distance proxy
- Visited flag
- Valid flag
- Previous-track flag
- Normalized distance from previous track
- Signed distance from previous track
- Remaining flag

The fixed observation size is `3 + 40 + 40 + 40 * 10 = 483`.

## Reward

For this foundation step, rewards are sparse and terminal-only for normal valid rollouts.

- Intermediate rewards: `0`
- Terminal reward: supplied by a reward interface
- Illegal action reward: large negative value, currently `-100`
- Illegal action termination: true

The skeleton includes a deterministic smoke reward for testing only. It makes no physical claim and is not a training reward.

The preferred final reward hierarchy is lexicographic U2 -> PEEQ -> SurfaceT, consistent with the final physical hierarchy chosen for the Stage 3 paper. Mises may remain available as a guarded or diagnostic metric, but it is not the primary hierarchy for PPO policy claims unless explicitly reintroduced later.

## Future Reward Implementations

### A. Rank-Based Table Reward

A rank-based table reward may map a completed scan order to the nearest known teacher-labelled order and return a table-derived score. This is acceptable only for smoke testing and plumbing checks.

It is not recommended as the final PPO training reward because it can collapse exploration toward known table entries and does not define a smooth or general reward surface for unseen scan orders.

### B. Surrogate-Predicted Terminal Reward

The preferred PPO training environment will use a supervised reward model trained on the native combined552 teacher-labelled dataset. PPO will interact with this surrogate terminal reward environment. Later, PPO-only candidates will be independently evaluated by Abaqus teacher validation.

Required separation:

- Supervised surrogate training uses teacher-labelled data.
- PPO training uses the surrogate environment.
- PPO candidate generation uses PPO policy inference only.
- Abaqus teacher validation is independent and happens after PPO generation.

## Claim Boundary

The current Stage 3 final evidence package does not support claiming that final results came from a deployed PPO or GNN-pointer RL policy. PPO claims must be based only on future PPO-specific training artifacts and PPO-generated candidates.

This environment specification supports only the foundation claim that an action-masked PPO-compatible scan-order environment has been designed and implemented.
