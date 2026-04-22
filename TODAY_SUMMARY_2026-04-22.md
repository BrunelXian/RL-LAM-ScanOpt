# RL-LAM-ScanOpt — Summary for 2026-04-22

## Final verdict for today

The current **selector-based learning route is NO-GO**.

This is no longer a case of:
- PPO not trained long enough,
- reward weights not tuned enough,
- or one more observation tweak being missing.

At this point, the evidence indicates that the main problem is **the current action abstraction itself**.
The selector actions are too coarse and too semantically overlapping, so different actions do not create large or stable enough downstream consequences under the current thermo proxy.

---

## What failed today

### 1. Reward-only rescue failed
We tried to rescue the current PPO formulation by changing reward design.
The core idea was to move from strongly myopic step-level thermal shaping toward a more sequence-aware objective.

Result:
- This did **not** produce a strong enough learning-facing signal.
- The reward redesign did not clearly separate strong policies from weak ones.
- The PPO route remained structurally unconvinced.

Conclusion:
- The issue is not just reward weighting.
- Reward surgery alone is **not enough**.

---

### 2. Selector-state coupling failed
We then tested a stricter selector-preview design.
The environment was modified so that PPO could explicitly see, for each selector:
- what region it would choose now,
- and what one-step proxy consequence it would produce.

Observation was expanded from `3 x H x W` to `11 x H x W`:
- `target_mask`
- `scanned_mask`
- `heat_map`
- 4 selector preview masks
- 4 selector preview score planes

Result:
- Diagnostics failed before PPO sanity check was allowed to start.
- Key numbers:
  - `argmin(score_s) == coolest_global` agreement: `0.525`
  - preview score margin mean: `0.0292`
  - preview score margin p10: `0.0000`
  - `coolest_global vs random_selector` separability index: `0.094`
- `coolest_global` and `min_reheat_risk` remained heavily overlapped.
- Many states had near-zero best-vs-second-best margin.

Conclusion:
- Even after making selector consequences explicit, the action-conditioned signal was still too weak.
- So the failure is **not** just that PPO could not “see enough”.

---

### 3. Offline ranking on the current selector action set also failed
After that, we stopped PPO and switched to an offline ranking check.
The goal was to answer a more basic question:

> Given a partial state, do different next actions lead to stably separable terminal sequence outcomes?

A ranking dataset generator was built.
For each partial state, candidate actions were evaluated by rollout completion and a sequence-level objective `J(tau)` was computed.

Outputs were generated successfully:
- `assets/data/ranking_action_values.csv`
- `assets/data/ranking_pairwise.csv`
- ranking summaries and diagnostics

Result:
- Partial-state ranking signal exists, but is weak and unstable.
- Key numbers:
  - `best-vs-second-best margin median = 0.214`
  - `restart pairwise consistency = 0.556`
  - `small best-vs-second margin ratio (< 1.0) = 0.917`
  - `small pairwise margin ratio (< 1.0) = 0.694`
- Bucket breakdown:
  - `early median = 0.751`
  - `middle median = 0.215`
  - `late median = 0.036`

Conclusion:
- There is some signal in early stages.
- But in middle and late stages, actions mostly collapse toward weakly separable outcomes.
- This action set is **not worth training a ranking model on**.

---

## What today’s failures mean

The accumulated evidence now supports the following judgment:

### Not the main bottleneck
- PPO itself is not the primary bottleneck.
- Reward tuning is not the primary bottleneck.
- Simple state augmentation is not the primary bottleneck.
- The thermo proxy is probably **not completely blind**, because early-stage separability still exists.

### Main bottleneck
The main bottleneck is:

> **The current selector-based action abstraction compresses the decision space too aggressively.**

It removes too much downstream difference between actions.
As a result:
- PPO cannot learn a strong thermo-aware policy,
- selector preview features are not separable enough,
- and ranking labels are too weak and unstable.

In short:

> The current selector action set is not a good learning-facing representation.

---

## What should be stopped now

The following directions should now be stopped:

- more PPO tuning on the current selector action set
- more reward micro-adjustment on the current selector action set
- more selector-preview coupling tweaks
- immediate ranking model training on the current selector action set
- imitation learning on the same selector abstraction

All of these would likely learn noise, weak preference, or unstable policy behavior.

---

## What should be kept

The following assets are still useful and should be retained:

- the cheap thermo proxy environment
- the rollout framework
- the baseline framework
- the diagnostics framework
- the sequence-level evaluation objective `J(tau)`
- the generated ranking data and separability analysis
- the general experimental conclusion that action formulation matters more than PPO tuning

These are not wasted work.
They are the diagnostic foundation for the next stage.

---

## Recommended next step

The next step should be:

## **Candidate-action search baseline**

Do **not** continue with learning yet.
Instead, replace the 4 selector actions with a small set of explicit candidate scan actions at each state.

For example, at each partial state construct `K` candidate actions such as:
- coolest local regions
- low reheat-risk candidates
- boundary candidates
- long-jump candidates
- random candidates for coverage

Then evaluate these candidates using the same completion-and-terminal-objective pipeline:
1. force the candidate action,
2. complete with a fixed completion policy,
3. compute terminal `J(tau)`,
4. measure best-vs-second-best margin and restart stability.

Purpose:
- test whether the real problem still has meaningful separability once we stop compressing actions into 4 selectors.

If candidate-level action search produces significantly larger and more stable margins, then the current selector abstraction was the real failure point.
If candidate-level action search still collapses, then the next suspect becomes the thermo proxy itself.

---

## Higher-level model direction after today

Today’s results also strengthen a broader architectural conclusion:

## **Pure cheap proxy should not be the final judge for scan-order optimisation.**

If scan order is fundamentally history-sensitive, then a heavily simplified evaluator can easily wash out the ordering effect.
That does **not** mean FEA should replace the entire inner loop.
It means the longer-term direction should be:

## **Bilevel structure**
- **Lower level:** cheap proxy for fast rollout, search, pruning, and candidate generation
- **Upper level:** path-sensitive thermo-mechanical FEA as sparse teacher / judge / correction signal

So the realistic roadmap now is:
1. finish candidate-action search diagnostics,
2. verify whether finer action candidates restore separability,
3. then move toward a bilevel proxy + FEA teacher architecture.

---

## Today’s bottom-line statement

Today’s work did not show that “RL for scan-order optimisation is impossible”.
It showed something more precise:

> **The current selector-based PPO/ranking formulation is not a valid route.**

That is an important result.
It closes one branch cleanly and prevents further wasted time on a structurally weak action representation.
