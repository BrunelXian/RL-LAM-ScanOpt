"""Run the Phase 1 baseline scan planners on a text-shaped target mask."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.geometry import downsample_mask, generate_text_mask
from core.planners.checkerboard import plan_checkerboard
from core.planners.cool_first import plan_cool_first
from core.planners.distance_aware_cool_first import plan_distance_aware_cool_first
from core.planners.greedy_cool_first import plan_greedy_cool_first
from core.planners.random_planner import plan_random
from core.planners.raster import plan_raster
from core.rollout import run_plan
from core.viz import (
    save_comparison_grid,
    save_metrics_bar_chart,
    save_order_map_figure,
    save_reward_breakdown_chart,
    save_target_mask_figure,
    save_thermal_map_figure,
)


def main() -> None:
    """Generate the target, run baselines, save figures, and print summaries."""
    figure_dir = PROJECT_ROOT / "assets" / "figures"
    table_dir = PROJECT_ROOT / "assets" / "models"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    high_res_mask = generate_text_mask(text="TWI", canvas_size=1024)
    target_mask = downsample_mask(mask=high_res_mask, grid_size=64, threshold=0.2)
    save_target_mask_figure(target_mask, figure_dir / "target_mask_twi_64.png")

    planners = {
        "raster": plan_raster(target_mask),
        "random": plan_random(target_mask, seed=42),
        "greedy": plan_greedy_cool_first(target_mask),
        "cool_first": plan_cool_first(target_mask),
        "checkerboard": plan_checkerboard(target_mask),
        "distance_aware_cool_first": plan_distance_aware_cool_first(target_mask),
    }

    results: dict[str, dict] = {}
    reward_rows: list[dict[str, float | int | str]] = []
    for planner_name, actions in planners.items():
        result = run_plan(mask=target_mask, actions=actions)
        results[planner_name] = result
        reward_breakdown = result["reward_breakdown"].copy()
        reward_breakdown["planner"] = planner_name
        reward_rows.append(reward_breakdown)

        save_order_map_figure(
            result["order_map"],
            figure_dir / f"order_map_{planner_name}.png",
            title=f"{planner_name.replace('_', ' ').title()} Order Map",
        )
        save_thermal_map_figure(
            result["final_thermal"],
            figure_dir / f"thermal_map_{planner_name}.png",
            title=f"{planner_name.replace('_', ' ').title()} Final Thermal Field",
        )

    save_metrics_bar_chart(results, figure_dir / "metrics_comparison.png")
    save_reward_breakdown_chart(reward_rows, figure_dir / "baseline_reward_breakdown.png")
    save_comparison_grid(
        results,
        figure_dir / "baseline_heatmap_comparison.png",
        field_key="final_thermal",
        title="Baseline Heat Accumulation Comparison",
        cmap="magma",
        colorbar_label="Proxy Thermal Level",
    )
    save_comparison_grid(
        results,
        figure_dir / "baseline_scan_order_comparison.png",
        field_key="order_map",
        title="Baseline Scan Order Comparison",
        cmap="inferno",
        colorbar_label="Scan Step",
    )

    csv_path = table_dir / "reward_breakdown_baselines.csv"
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
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in reward_rows:
            writer.writerow({name: row[name] for name in fieldnames})

    reward_rows_sorted = sorted(reward_rows, key=lambda row: float(row["total_reward"]), reverse=True)
    random_row = next(row for row in reward_rows if row["planner"] == "random")
    raster_row = next(row for row in reward_rows if row["planner"] == "raster")
    greedy_row = next(row for row in reward_rows if row["planner"] == "greedy")
    cool_first_row = next(row for row in reward_rows if row["planner"] == "cool_first")
    checkerboard_row = next(row for row in reward_rows if row["planner"] == "checkerboard")
    distance_aware_row = next(row for row in reward_rows if row["planner"] == "distance_aware_cool_first")
    sanity_lines = [
        "Stage A composite reward sanity summary",
        (
            f"- random jump penalty ({random_row['jump']:.3f}) vs raster ({raster_row['jump']:.3f}) "
            "should usually be more negative if random jumps more."
        ),
        (
            f"- raster reheat penalty ({raster_row['reheat']:.3f}) vs greedy ({greedy_row['reheat']:.3f}) "
            "helps show local hotspot clustering behavior."
        ),
        (
            f"- greedy peak penalty ({greedy_row['peak']:.3f}) and variance penalty "
            f"({greedy_row['variance']:.3f}) indicate whether cool-first reduces thermal load."
        ),
        (
            f"- cool_first total reward ({cool_first_row['total_reward']:.3f}) vs random "
            f"({random_row['total_reward']:.3f}) shows whether explicit cooling beats random dispersion."
        ),
        (
            f"- checkerboard total reward ({checkerboard_row['total_reward']:.3f}) vs raster "
            f"({raster_row['total_reward']:.3f}) shows whether structured dispersion is rewarded."
        ),
        (
            f"- distance-aware cool-first total reward ({distance_aware_row['total_reward']:.3f}) vs cool_first "
            f"({cool_first_row['total_reward']:.3f}) tests whether the current jump penalty is strong enough."
        ),
    ]
    summary_path = table_dir / "reward_breakdown_baselines_summary.txt"
    summary_path.write_text("\n".join(sanity_lines) + "\n", encoding="utf-8")

    interpretation_lines = [
        "Stronger baseline interpretation",
        (
            f"- Planner ranking by total reward: "
            + ", ".join(
                f"{row['planner']} ({float(row['total_reward']):.3f})"
                for row in reward_rows_sorted
            )
        ),
        (
            f"- cool_first vs random: cool_first {'beats' if float(cool_first_row['total_reward']) > float(random_row['total_reward']) else 'does not beat'} "
            f"random ({cool_first_row['total_reward']:.3f} vs {random_row['total_reward']:.3f})."
        ),
        (
            f"- checkerboard vs raster: checkerboard {'beats' if float(checkerboard_row['total_reward']) > float(raster_row['total_reward']) else 'does not beat'} "
            f"raster ({checkerboard_row['total_reward']:.3f} vs {raster_row['total_reward']:.3f})."
        ),
        (
            f"- distance-aware cool-first vs pure cool-first: distance-aware {'improves the trade-off' if float(distance_aware_row['total_reward']) > float(cool_first_row['total_reward']) else 'is still weaker overall'} "
            f"({distance_aware_row['total_reward']:.3f} vs {cool_first_row['total_reward']:.3f})."
        ),
        (
            f"- Jump penalty check: random jump={random_row['jump']:.3f}, cool_first jump={cool_first_row['jump']:.3f}, "
            f"distance-aware jump={distance_aware_row['jump']:.3f}. This shows whether the current jump weight materially distinguishes chaotic plans."
        ),
        (
            f"- Reheat penalty check: raster reheat={raster_row['reheat']:.3f}, checkerboard reheat={checkerboard_row['reheat']:.3f}, "
            f"cool_first reheat={cool_first_row['reheat']:.3f}. This indicates whether local reheating still dominates the ranking."
        ),
        (
            "- Interpretation: if checkerboard and cool-first both outperform raster, the reward is responding to structured thermal dispersion. "
            "If random remains very close to or better than structured heuristics, the reward likely still overvalues spatial dispersion and needs a small next-step calibration."
        ),
    ]
    interpretation_path = table_dir / "stronger_baselines_interpretation.txt"
    interpretation_path.write_text("\n".join(interpretation_lines) + "\n", encoding="utf-8")

    print("RL-LAM-ScanOpt Phase 1 baseline summary")
    for planner_name, result in results.items():
        metrics = result["metrics"]
        reward_breakdown = result["reward_breakdown"]
        print(
            f"- {planner_name}: coverage={metrics['coverage_ratio']:.3f}, "
            f"mean={metrics['thermal_mean']:.3f}, "
            f"peak={metrics['thermal_peak']:.3f}, "
            f"variance={metrics['thermal_variance']:.3f}, "
            f"steps={metrics['steps']}, "
            f"reward_total={reward_breakdown['total_reward']:.3f}, "
            f"reheat={reward_breakdown['reheat']:.3f}, "
            f"jump={reward_breakdown['jump']:.3f}"
        )
    print(f"Saved reward breakdown CSV to: {csv_path}")
    print(f"Saved reward breakdown figure to: {figure_dir / 'baseline_reward_breakdown.png'}")
    print(f"Saved reward sanity summary to: {summary_path}")
    print(f"Saved stronger baseline interpretation to: {interpretation_path}")


if __name__ == "__main__":
    main()
