# Legacy Proxy-RL Track

This document records the branch that is no longer the active mainline.

## What The Old Branch Tried To Do

The old branch attempted to use a cheap thermal proxy as the main optimization target for scan-order learning.

It explored:

- stripe actions
- fixed segments
- variable-length segments
- local windows
- local primitives
- directional primitives
- selector-based actions
- selector-history state extensions
- reward-only rewrites
- selector-state coupling
- offline partial-state ranking

## What Was Learned

The repository now has enough evidence to treat that branch as closed for the time being.

Confirmed outcomes:

- PPO mainline: `NO-GO`
- patch-based action family: `NO-GO`
- reward-only redesign under the same cheap proxy: `NO-GO`
- selector-state coupling: `NO-GO`
- offline ranking under the same action abstraction and proxy semantics: `NO-GO`

## Important Interpretation

The failure was not mainly "PPO could not optimize."

The stronger interpretation is:

- the cheap evaluator was not sufficiently path-sensitive
- scan-order consequences were not stably separable enough
- learning targets built on that evaluator were therefore unreliable

## Why The Legacy Assets Are Still Kept

They remain useful for:

- traceability
- negative-result documentation
- future comparison if a new evaluator is introduced

Key legacy folders and files include:

- `assets/fea_teacher_pool/`
- `assets/models/top_10_sequences_twi_64x64.json`
- `assets/models/top_10_sequences_twi_64x64.txt`
- legacy selector and PPO diagnostics under `assets/models/` and `assets/figures/`

## What Not To Restart

Do not restart these branches as the default next move:

- more PPO smoke tests on the old proxy
- more reward micro-tuning
- more selector-preview tweaks
- direct ranking-model training on the old partial-state dataset

The active branch has already moved to line-order benchmarking and FEA-teacher preparation.
