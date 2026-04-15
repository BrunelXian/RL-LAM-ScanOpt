# PROJECT_PLAN

## Objective

Build a small, modular demo that shows how scan-order planning, including reinforcement learning, can improve thermal uniformity and reduce stress-related proxy risk over constrained additive manufacturing geometries.

## Phase 1: Baselines + Thermal Proxy + Visualisation

Focus on a minimal but convincing baseline system.

- Represent constrained 2D geometries as valid scan domains
- Implement raster and random scan baselines
- Add a simple greedy cooling-aware baseline for comparison
- Build a lightweight thermal accumulation proxy
- Define interpretable metrics such as temperature variance, reheating concentration, and stress-risk proxy
- Visualise scan order, per-step heating, and summary comparison plots

Success criteria:

- A user can run several planners on the same geometry
- The system produces comparable metrics and visual outputs
- The implementation stays fast enough for repeated experimentation

## Phase 2: RL Environment + PPO / Maskable PPO

Introduce a learning-based planner on top of the baseline system.

- Build a Gymnasium-compatible environment with action masking for invalid scan moves
- Add PPO and Maskable PPO training entry points
- Define rewards around thermal smoothness, coverage completion, and proxy risk reduction
- Evaluate learned policies against raster, random, and greedy baselines
- Keep experiments compact and reproducible

Success criteria:

- A trained policy can run inference on the same geometry representation as the baselines
- Results can be compared with the same evaluator stack
- Invalid-action handling remains explicit and easy to inspect

## Phase 3: Interactive Demo

Wrap the system in a simple interactive interface.

- Build a Streamlit demo for geometry selection, planner choice, and run configuration
- Show scan sequence animation or stepwise playback
- Display metric summaries and comparison figures
- Keep the demo lightweight and suitable for research communication or industrial discussion

Success criteria:

- A user can compare multiple planners without editing code
- Visual outputs clearly communicate trade-offs
- The UI remains optional and does not drive core logic design

## Extensibility Notes

### Geometry replacement

- Geometry handling should stay isolated from planners and UI
- Text-based shapes, bitmap masks, and later polygon inputs should be swappable behind a common representation

### Evaluator replacement

- Thermal proxy, variance, and stress-risk evaluators should share a common interface
- New evaluators should be addable without rewriting planners or rollout logic

### UI replacement

- Core execution should remain headless
- Streamlit should be a thin presentation layer so it can later be replaced by another UI or notebook workflow

## Non-Goals for the Demo

- No FEM
- No exact residual stress prediction
- No claim of full process fidelity
- No over-optimised production architecture at the scaffold stage
