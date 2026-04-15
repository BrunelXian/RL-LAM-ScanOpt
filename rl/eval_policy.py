"""Evaluate a trained Maskable PPO policy on the default TWI planning task."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.planners.greedy_cool_first import plan_greedy_cool_first
from core.planners.random_planner import plan_random
from core.planners.raster import plan_raster
from core.rollout import run_plan
from core.viz import (
    save_comparison_grid,
    save_metrics_bar_chart,
    save_order_map_figure,
    save_scan_path_gif,
    save_thermal_map_figure,
)
from rl.env_scan import ScanPlanningEnv, build_twi_mask


TEXT = "TWI"
GRID_SIZE = 64
CANVAS_SIZE = 1024
MODEL_PATH = PROJECT_ROOT / "assets" / "models" / "maskable_ppo_twi.zip"
FIGURE_DIR = PROJECT_ROOT / "assets" / "figures"
TRAINING_HISTORY_PATH = PROJECT_ROOT / "assets" / "models" / "training_history_maskable_ppo_twi.json"
SUMMARY_PATH = PROJECT_ROOT / "training_results.md"


def _mask_fn(env: object) -> object:
    return env.unwrapped.action_masks()


def make_env() -> object:
    """Create the wrapped evaluation environment."""
    from sb3_contrib.common.wrappers import ActionMasker

    planning_mask = build_twi_mask(grid_size=GRID_SIZE, canvas_size=CANVAS_SIZE, text=TEXT)
    env = ScanPlanningEnv(
        planning_mask=planning_mask,
        grid_size=GRID_SIZE,
        text=TEXT,
        canvas_size=CANVAS_SIZE,
        max_steps=max(int(planning_mask.sum()) * 2, 1),
    )
    return ActionMasker(env, _mask_fn)


def main() -> None:
    """Load a saved model, run one masked evaluation episode, and save figures."""
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.utils import get_action_masks
    except ImportError as exc:
        raise ImportError(
            "Maskable PPO dependencies are missing. Install torch and sb3-contrib first."
        ) from exc

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    env = make_env()
    model = MaskablePPO.load(str(MODEL_PATH))

    obs, _ = env.reset(seed=42)
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action_masks = get_action_masks(env)
        action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
        obs, reward, terminated, truncated, info = env.step(int(action))

    actions = list(env.unwrapped.executed_actions)
    mask = env.unwrapped.target_mask.copy()
    rl_result = run_plan(mask=mask, actions=actions, record_history=True, history_stride=8)

    save_order_map_figure(
        rl_result["order_map"],
        FIGURE_DIR / "order_map_rl_maskable_ppo.png",
        title="RL Maskable PPO Order Map",
    )
    save_thermal_map_figure(
        rl_result["final_thermal"],
        FIGURE_DIR / "thermal_map_rl_maskable_ppo.png",
        title="RL Maskable PPO Final Thermal Field",
    )
    save_scan_path_gif(
        target_mask=mask,
        scanned_history=rl_result["scanned_history"],
        path=FIGURE_DIR / "scan_path_rl_maskable_ppo.gif",
        title="RL Maskable PPO Scan Path",
    )

    comparison_results = {
        "raster": run_plan(mask=mask, actions=plan_raster(mask)),
        "random": run_plan(mask=mask, actions=plan_random(mask, seed=42)),
        "greedy_cool_first": run_plan(mask=mask, actions=plan_greedy_cool_first(mask)),
        "rl_maskable_ppo": rl_result,
    }
    save_metrics_bar_chart(comparison_results, FIGURE_DIR / "metrics_comparison_with_rl.png")
    save_comparison_grid(
        comparison_results,
        FIGURE_DIR / "thermal_map_comparison_grid.png",
        field_key="final_thermal",
        title="Thermal Map Comparison",
        cmap="magma",
        colorbar_label="Proxy Thermal Level",
    )
    save_comparison_grid(
        comparison_results,
        FIGURE_DIR / "order_map_comparison_grid.png",
        field_key="order_map",
        title="Scan Order Comparison",
        cmap="inferno",
        colorbar_label="Scan Step",
    )

    metrics = rl_result["metrics"]
    print("RL Maskable PPO evaluation summary")
    print(
        f"- coverage={metrics['coverage_ratio']:.3f}, "
        f"mean={metrics['thermal_mean']:.3f}, "
        f"peak={metrics['thermal_peak']:.3f}, "
        f"variance={metrics['thermal_variance']:.3f}, "
        f"steps={metrics['steps']}"
    )

    print("Comparison snapshot")
    for planner_name, result in comparison_results.items():
        m = result["metrics"]
        print(
            f"- {planner_name}: coverage={m['coverage_ratio']:.3f}, "
            f"mean={m['thermal_mean']:.3f}, peak={m['thermal_peak']:.3f}, "
            f"variance={m['thermal_variance']:.3f}, steps={m['steps']}"
        )

    training_history = None
    if TRAINING_HISTORY_PATH.exists():
        with TRAINING_HISTORY_PATH.open("r", encoding="utf-8") as file:
            training_history = json.load(file)

    summary_lines = [
        "# Training Results",
        "",
        "## Model Performance",
        "",
        "| Planner | Coverage | Thermal Mean | Thermal Peak | Thermal Variance | Steps |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for planner_name, result in comparison_results.items():
        m = result["metrics"]
        summary_lines.append(
            f"| {planner_name} | {m['coverage_ratio']:.3f} | {m['thermal_mean']:.3f} | "
            f"{m['thermal_peak']:.3f} | {m['thermal_variance']:.3f} | {m['steps']} |"
        )

    summary_lines.extend(
        [
            "",
            "## Graphical Comparison",
            "",
            "### Scan Paths",
            "![Scan Order Comparison](assets/figures/order_map_comparison_grid.png)",
            "",
            "### Thermal Maps",
            "![Thermal Map Comparison](assets/figures/thermal_map_comparison_grid.png)",
            "",
            "### RL Scan Animation",
            "![RL Scan Path GIF](assets/figures/scan_path_rl_maskable_ppo.gif)",
            "",
            "## Training Curves",
            "",
            "![Training Curves](assets/figures/training_curves_maskable_ppo.png)",
            "",
            "## Discussion",
            "",
            "The current Maskable PPO model reaches full coverage on the `TWI` letter mask and "
            "is compared against raster, random, and greedy cool-first baselines using the "
            "same lightweight thermal proxy. The updated reward mixes coverage, global thermal "
            "variance penalties, local temperature-difference penalties, hotspot-distribution "
            "penalties, and invalid-action penalties while leaving path jumps unconstrained.",
            "",
            "Current limitations:",
            "- This is still a proxy thermal environment rather than a calibrated physical model.",
            "- Training uses one fixed geometry and one environment, so generalisation is limited.",
            "- Even with longer training, RL may still trail the strongest handcrafted baselines on this simplified task.",
        ]
    )
    if training_history:
        summary_lines.extend(
            [
                "",
                "### Training Snapshot",
                "",
                f"- Episodes recorded: {len(training_history.get('coverage_ratio', []))}",
                f"- Final recorded coverage: {training_history.get('coverage_ratio', [0.0])[-1]:.3f}",
                f"- Final recorded thermal variance: {training_history.get('thermal_variance', [0.0])[-1]:.3f}",
                f"- Final recorded thermal peak: {training_history.get('thermal_peak', [0.0])[-1]:.3f}",
            ]
        )
    SUMMARY_PATH.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"Saved summary markdown to: {SUMMARY_PATH}")

    env.close()


if __name__ == "__main__":
    main()
