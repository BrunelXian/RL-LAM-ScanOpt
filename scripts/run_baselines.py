"""Run the Phase 1 baseline scan planners on a text-shaped target mask."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.geometry import downsample_mask, generate_text_mask
from core.planners.greedy_cool_first import plan_greedy_cool_first
from core.planners.random_planner import plan_random
from core.planners.raster import plan_raster
from core.rollout import run_plan
from core.viz import (
    save_metrics_bar_chart,
    save_order_map_figure,
    save_target_mask_figure,
    save_thermal_map_figure,
)


def main() -> None:
    """Generate the target, run baselines, save figures, and print summaries."""
    output_dir = PROJECT_ROOT / "assets" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    high_res_mask = generate_text_mask(text="TWI", canvas_size=1024)
    target_mask = downsample_mask(mask=high_res_mask, grid_size=64, threshold=0.2)
    save_target_mask_figure(target_mask, output_dir / "target_mask_twi_64.png")

    planners = {
        "raster": plan_raster(target_mask),
        "random": plan_random(target_mask, seed=42),
        "greedy_cool_first": plan_greedy_cool_first(target_mask),
    }

    results: dict[str, dict] = {}
    for planner_name, actions in planners.items():
        result = run_plan(mask=target_mask, actions=actions)
        results[planner_name] = result

        save_order_map_figure(
            result["order_map"],
            output_dir / f"order_map_{planner_name}.png",
            title=f"{planner_name.replace('_', ' ').title()} Order Map",
        )
        save_thermal_map_figure(
            result["final_thermal"],
            output_dir / f"thermal_map_{planner_name}.png",
            title=f"{planner_name.replace('_', ' ').title()} Final Thermal Field",
        )

    save_metrics_bar_chart(results, output_dir / "metrics_comparison.png")

    print("RL-LAM-ScanOpt Phase 1 baseline summary")
    for planner_name, result in results.items():
        metrics = result["metrics"]
        print(
            f"- {planner_name}: coverage={metrics['coverage_ratio']:.3f}, "
            f"mean={metrics['thermal_mean']:.3f}, "
            f"peak={metrics['thermal_peak']:.3f}, "
            f"variance={metrics['thermal_variance']:.3f}, "
            f"steps={metrics['steps']}"
        )


if __name__ == "__main__":
    main()
