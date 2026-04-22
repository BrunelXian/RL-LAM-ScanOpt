"""Run a small PPO smoke test using the selected calibrated Stage A reward."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.planners.cool_first import plan_cool_first
from core.planners.distance_aware_cool_first import plan_distance_aware_cool_first
from core.planners.random_planner import plan_random
from core.rollout import run_plan
from core.viz import (
    save_comparison_grid,
    save_reward_breakdown_chart,
    save_scan_path_gif,
    save_training_curves_figure,
)
from rl.env_scan import ScanPlanningEnv, build_twi_mask
from rl.train_maskable_ppo import (
    BATCH_SIZE,
    CANVAS_SIZE,
    ENT_COEF,
    GAMMA,
    GRID_SIZE,
    LEARNING_RATE,
    N_STEPS,
    SEED,
    TEXT,
    TrainingMetricsCallback,
)


SELECTION_PATH = PROJECT_ROOT / "assets" / "models" / "reward_calibration_selection.json"
MODEL_PATH = PROJECT_ROOT / "assets" / "models" / "maskable_ppo_smoke_calibrated.zip"
TRAINING_HISTORY_PATH = PROJECT_ROOT / "assets" / "models" / "maskable_ppo_smoke_calibrated_history.json"
TRAINING_CURVES_PATH = PROJECT_ROOT / "assets" / "figures" / "maskable_ppo_smoke_training_curves.png"
BREAKDOWN_CSV_PATH = PROJECT_ROOT / "assets" / "models" / "ppo_smoke_reward_breakdown.csv"
BREAKDOWN_PLOT_PATH = PROJECT_ROOT / "assets" / "figures" / "ppo_smoke_reward_breakdown.png"
HEATMAP_COMPARISON_PATH = PROJECT_ROOT / "assets" / "figures" / "ppo_smoke_heatmap_comparison.png"
SCAN_ORDER_COMPARISON_PATH = PROJECT_ROOT / "assets" / "figures" / "ppo_smoke_scan_order_comparison.png"
PPO_GIF_PATH = PROJECT_ROOT / "assets" / "figures" / "ppo_smoke_scan_path.gif"
INTERPRETATION_PATH = PROJECT_ROOT / "assets" / "models" / "ppo_smoke_interpretation.txt"
TIMESTEPS = 10_240


def _mask_fn(env: object) -> object:
    return env.unwrapped.action_masks()


def make_training_env(reward_weights: dict[str, float]) -> object:
    """Create the wrapped Maskable PPO training environment."""
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    planning_mask = build_twi_mask(grid_size=GRID_SIZE, canvas_size=CANVAS_SIZE, text=TEXT)

    def _factory() -> object:
        env = ScanPlanningEnv(
            planning_mask=planning_mask,
            grid_size=GRID_SIZE,
            text=TEXT,
            canvas_size=CANVAS_SIZE,
            reward_weights=reward_weights,
        )
        return ActionMasker(Monitor(env), _mask_fn)

    return DummyVecEnv([_factory])


def make_eval_env(reward_weights: dict[str, float]) -> object:
    """Create a wrapped evaluation environment using the same calibrated reward."""
    from sb3_contrib.common.wrappers import ActionMasker

    planning_mask = build_twi_mask(grid_size=GRID_SIZE, canvas_size=CANVAS_SIZE, text=TEXT)
    env = ScanPlanningEnv(
        planning_mask=planning_mask,
        grid_size=GRID_SIZE,
        text=TEXT,
        canvas_size=CANVAS_SIZE,
        reward_weights=reward_weights,
    )
    return ActionMasker(env, _mask_fn)


def evaluate_masked_policy_once(model: object, env: object, reward_weights: dict[str, float]) -> tuple[dict, dict]:
    """Run one deterministic masked rollout and return baseline-comparable outputs plus env stats."""
    from sb3_contrib.common.maskable.utils import get_action_masks

    obs, _ = env.reset(seed=SEED)
    terminated = False
    truncated = False
    final_info: dict = {}

    while not (terminated or truncated):
        action_masks = get_action_masks(env)
        action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
        obs, reward, terminated, truncated, final_info = env.step(int(action))

    actions = list(env.unwrapped.executed_actions)
    mask = env.unwrapped.target_mask.copy()
    rollout_result = run_plan(
        mask=mask,
        actions=actions,
        record_history=True,
        history_stride=8,
        reward_weights=reward_weights,
    )
    eval_stats = {
        "steps_taken": int(final_info.get("steps_taken", 0)),
        "invalid_action_count": int(final_info.get("invalid_action_count", 0)),
        "coverage_ratio": float(final_info.get("coverage_ratio", 0.0)),
        "peak_heat": float(final_info.get("peak_heat", 0.0)),
        "heat_variance": float(final_info.get("heat_variance", 0.0)),
    }
    return rollout_result, eval_stats


def main() -> None:
    """Train a tiny PPO policy with the selected reward variant and evaluate it against key baselines."""
    try:
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.callbacks import ConvertCallback
    except ImportError as exc:
        raise ImportError(
            "Maskable PPO dependencies are missing. Install torch and sb3-contrib first."
        ) from exc

    if not SELECTION_PATH.exists():
        raise FileNotFoundError(f"Missing reward selection file: {SELECTION_PATH}")

    selection_payload = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    selected_variant = selection_payload["selected_variant"]
    reward_weights = {key: float(value) for key, value in selection_payload["reward_weights"].items()}

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRAINING_CURVES_PATH.parent.mkdir(parents=True, exist_ok=True)

    env = make_training_env(reward_weights=reward_weights)
    metrics_callback = TrainingMetricsCallback()
    model = MaskablePPO(
        policy="CnnPolicy",
        env=env,
        verbose=1,
        seed=SEED,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        ent_coef=ENT_COEF,
        device="cuda",
    )
    model.learn(
        total_timesteps=TIMESTEPS,
        progress_bar=False,
        log_interval=1,
        callback=ConvertCallback(metrics_callback),
    )
    model.save(str(MODEL_PATH))
    TRAINING_HISTORY_PATH.write_text(json.dumps(metrics_callback.history, indent=2), encoding="utf-8")
    save_training_curves_figure(metrics_callback.history, TRAINING_CURVES_PATH)
    env.close()

    eval_env = make_eval_env(reward_weights=reward_weights)
    ppo_result, eval_stats = evaluate_masked_policy_once(model=model, env=eval_env, reward_weights=reward_weights)
    eval_env.close()

    mask = ppo_result["target_mask"]
    comparison_results = {
        "random": run_plan(mask=mask, actions=plan_random(mask, seed=42), reward_weights=reward_weights),
        "cool_first": run_plan(mask=mask, actions=plan_cool_first(mask), reward_weights=reward_weights),
        "distance_aware_cool_first": run_plan(
            mask=mask,
            actions=plan_distance_aware_cool_first(mask),
            reward_weights=reward_weights,
        ),
        "ppo_smoke": ppo_result,
    }

    reward_rows: list[dict[str, float | int | str]] = []
    for planner_name, result in comparison_results.items():
        row = {"planner": planner_name, **result["reward_breakdown"]}
        reward_rows.append(row)

    with BREAKDOWN_CSV_PATH.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "planner",
            "total_reward",
            "coverage",
            "invalid",
            "peak",
            "variance",
            "reheat",
            "jump",
            "completion_bonus",
            "support_risk",
            "steps",
            "coverage_ratio",
            "peak_heat_final",
            "heat_variance_final",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in reward_rows:
            writer.writerow({name: row[name] for name in fieldnames})

    save_reward_breakdown_chart(reward_rows, BREAKDOWN_PLOT_PATH)
    save_comparison_grid(
        comparison_results,
        HEATMAP_COMPARISON_PATH,
        field_key="final_thermal",
        title="PPO Smoke Heatmap Comparison",
        cmap="magma",
        colorbar_label="Proxy Thermal Level",
    )
    save_comparison_grid(
        comparison_results,
        SCAN_ORDER_COMPARISON_PATH,
        field_key="order_map",
        title="PPO Smoke Scan Order Comparison",
        cmap="inferno",
        colorbar_label="Scan Step",
    )
    save_scan_path_gif(
        target_mask=mask,
        scanned_history=ppo_result["scanned_history"],
        path=PPO_GIF_PATH,
        title="PPO Smoke Scan Path",
    )

    random_row = next(row for row in reward_rows if row["planner"] == "random")
    distance_aware_row = next(row for row in reward_rows if row["planner"] == "distance_aware_cool_first")
    ppo_row = next(row for row in reward_rows if row["planner"] == "ppo_smoke")
    invalid_rate = eval_stats["invalid_action_count"] / max(eval_stats["steps_taken"], 1)
    reheat_improvement = (
        (abs(float(random_row["reheat"])) - abs(float(ppo_row["reheat"]))) / max(abs(float(random_row["reheat"])), 1e-9)
    )
    peak_improvement = (
        (abs(float(random_row["peak"])) - abs(float(ppo_row["peak"]))) / max(abs(float(random_row["peak"])), 1e-9)
    )
    improvement_met = reheat_improvement >= 0.15 and peak_improvement >= 0.15

    interpretation_lines = [
        "PPO smoke interpretation",
        f"- Selected reward variant: {selected_variant}",
        f"- Timesteps used: {TIMESTEPS}",
        f"- Coverage ratio: {eval_stats['coverage_ratio']:.3f}",
        f"- Invalid action rate: {invalid_rate:.3%}",
        (
            f"- PPO vs random on reward terms: reheat improvement={reheat_improvement:.1%}, "
            f"peak improvement={peak_improvement:.1%}"
        ),
        f"- Mandatory improvement criterion met: {improvement_met}",
        (
            f"- PPO total_reward={float(ppo_row['total_reward']):.3f}, "
            f"random={float(random_row['total_reward']):.3f}, "
            f"distance_aware_cool_first={float(distance_aware_row['total_reward']):.3f}"
        ),
        (
            f"- PPO appears closer to "
            f"{'distance_aware_cool_first' if abs(float(ppo_row['total_reward']) - float(distance_aware_row['total_reward'])) < abs(float(ppo_row['total_reward']) - float(random_row['total_reward'])) else 'random'} "
            "in overall reward."
        ),
        (
            f"- PPO learned something clearly better than random: "
            f"{float(ppo_row['total_reward']) > float(random_row['total_reward']) and improvement_met}"
        ),
        (
            f"- PPO still worse than distance-aware cool-first: "
            f"{float(ppo_row['total_reward']) < float(distance_aware_row['total_reward'])}"
        ),
        (
            "- Next step: "
            + (
                "continue PPO training"
                if float(ppo_row["total_reward"]) > float(random_row["total_reward"]) and invalid_rate < 0.05
                else "inspect environment / reward logging"
            )
        ),
    ]
    INTERPRETATION_PATH.write_text("\n".join(interpretation_lines) + "\n", encoding="utf-8")

    print(f"Saved smoke model to: {MODEL_PATH}")
    print(f"Saved smoke training history to: {TRAINING_HISTORY_PATH}")
    print(f"Saved smoke training curves to: {TRAINING_CURVES_PATH}")
    print(f"Saved smoke reward breakdown CSV to: {BREAKDOWN_CSV_PATH}")
    print(f"Saved smoke interpretation to: {INTERPRETATION_PATH}")


if __name__ == "__main__":
    main()
