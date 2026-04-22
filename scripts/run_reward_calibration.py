"""Run a minimal Stage A reward calibration over jump/reheat weights."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.geometry import downsample_mask, generate_text_mask
from core.planners.cool_first import plan_cool_first
from core.planners.distance_aware_cool_first import plan_distance_aware_cool_first
from core.planners.random_planner import plan_random
from core.planners.raster import plan_raster
from core.reward import DEFAULT_REWARD_WEIGHTS, build_reward_weights
from core.rollout import run_plan


CSV_PATH = PROJECT_ROOT / "assets" / "models" / "reward_calibration_comparison.csv"
SUMMARY_PATH = PROJECT_ROOT / "assets" / "models" / "reward_calibration_interpretation.txt"
SELECTION_PATH = PROJECT_ROOT / "assets" / "models" / "reward_calibration_selection.json"


def build_variants() -> dict[str, dict[str, float]]:
    """Build exactly three small calibration variants from the current defaults."""
    base = build_reward_weights(DEFAULT_REWARD_WEIGHTS)
    return {
        "variant_0": base,
        "variant_1": build_reward_weights(
            {
                "jump": base["jump"] * 1.5,
                "reheat": base["reheat"] * 1.2,
            }
        ),
        "variant_2": build_reward_weights(
            {
                "jump": base["jump"] * 2.0,
                "reheat": base["reheat"] * 1.25,
            }
        ),
    }


def select_variant(
    grouped_rows: dict[str, dict[str, dict[str, float | str | int]]],
    variants: dict[str, dict[str, float]],
) -> tuple[str, dict[str, float], list[str]]:
    """Select the smallest valid strengthening that widens the DA-vs-random gap."""
    base_distance_aware = float(grouped_rows["variant_0"]["distance_aware_cool_first"]["total_reward"])
    base_gap = float(grouped_rows["variant_0"]["distance_aware_cool_first"]["total_reward"]) - float(
        grouped_rows["variant_0"]["random"]["total_reward"]
    )
    notes: list[str] = []
    selected_name = "variant_0"

    for variant_name in ("variant_1", "variant_2"):
        distance_aware_reward = float(grouped_rows[variant_name]["distance_aware_cool_first"]["total_reward"])
        random_reward = float(grouped_rows[variant_name]["random"]["total_reward"])
        cool_first_reward = float(grouped_rows[variant_name]["cool_first"]["total_reward"])
        raster_reward = float(grouped_rows[variant_name]["raster"]["total_reward"])
        gap = distance_aware_reward - random_reward
        failed_red_line = distance_aware_reward < 0.85 * base_distance_aware
        notes.append(
            f"- {variant_name}: DA-random gap={gap:.3f}, cool_first={cool_first_reward:.3f}, "
            f"random={random_reward:.3f}, raster={raster_reward:.3f}, red_line_fail={failed_red_line}"
        )
        if failed_red_line:
            continue
        if gap > base_gap:
            selected_name = variant_name
            break

    return selected_name, variants[selected_name], notes


def main() -> None:
    """Run key-planner calibration and select one reward variant for PPO smoke testing."""
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    high_res_mask = generate_text_mask(text="TWI", canvas_size=1024)
    target_mask = downsample_mask(mask=high_res_mask, grid_size=64, threshold=0.2)

    planners = {
        "random": plan_random(target_mask, seed=42),
        "cool_first": plan_cool_first(target_mask),
        "distance_aware_cool_first": plan_distance_aware_cool_first(target_mask),
        "raster": plan_raster(target_mask),
    }
    variants = build_variants()

    rows: list[dict[str, float | str | int]] = []
    grouped_rows: dict[str, dict[str, dict[str, float | str | int]]] = {}
    for variant_name, reward_weights in variants.items():
        grouped_rows[variant_name] = {}
        for planner_name, actions in planners.items():
            result = run_plan(mask=target_mask, actions=actions, reward_weights=reward_weights)
            reward_breakdown = result["reward_breakdown"].copy()
            row = {
                "reward_variant": variant_name,
                "planner": planner_name,
                **reward_breakdown,
            }
            rows.append(row)
            grouped_rows[variant_name][planner_name] = row

    fieldnames = [
        "reward_variant",
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
    with CSV_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})

    selected_name, selected_weights, notes = select_variant(grouped_rows, variants)
    base_distance_aware = float(grouped_rows["variant_0"]["distance_aware_cool_first"]["total_reward"])
    interpretation_lines = [
        "Reward calibration interpretation",
        f"- Current baseline weights: jump={variants['variant_0']['jump']:.3f}, reheat={variants['variant_0']['reheat']:.3f}",
        f"- Variant 1 weights: jump={variants['variant_1']['jump']:.3f}, reheat={variants['variant_1']['reheat']:.3f}",
        f"- Variant 2 weights: jump={variants['variant_2']['jump']:.3f}, reheat={variants['variant_2']['reheat']:.3f}",
        "",
        "Per-variant gap checks:",
        *notes,
        "",
    ]

    for variant_name in ("variant_0", "variant_1", "variant_2"):
        distance_aware_reward = float(grouped_rows[variant_name]["distance_aware_cool_first"]["total_reward"])
        random_reward = float(grouped_rows[variant_name]["random"]["total_reward"])
        cool_first_reward = float(grouped_rows[variant_name]["cool_first"]["total_reward"])
        raster_reward = float(grouped_rows[variant_name]["raster"]["total_reward"])
        red_line_fail = distance_aware_reward < 0.85 * base_distance_aware
        interpretation_lines.extend(
            [
                f"{variant_name}:",
                f"- distance_aware vs random gap = {distance_aware_reward - random_reward:.3f}",
                f"- cool_first {'stays above' if cool_first_reward >= random_reward else 'falls below'} random ({cool_first_reward:.3f} vs {random_reward:.3f})",
                f"- raster remains {'clearly bad' if raster_reward < random_reward and raster_reward < cool_first_reward else 'less clearly separated'} ({raster_reward:.3f})",
                f"- distance_aware is {'best or near-best' if distance_aware_reward >= max(random_reward, cool_first_reward, raster_reward) else 'not best'} ({distance_aware_reward:.3f})",
                f"- red-line failure (>15% drop) = {red_line_fail}",
                "",
            ]
        )

    interpretation_lines.extend(
        [
            f"Selected variant: {selected_name}",
            (
                f"- Selected weights: jump={selected_weights['jump']:.3f}, "
                f"reheat={selected_weights['reheat']:.3f}"
            ),
            (
                "- Selection rule: choose the smallest valid strengthening that widens the "
                "distance-aware-cool-first vs random gap without triggering the 15% red-line."
            ),
        ]
    )
    SUMMARY_PATH.write_text("\n".join(interpretation_lines) + "\n", encoding="utf-8")

    selection_payload = {
        "selected_variant": selected_name,
        "reward_weights": selected_weights,
        "variants": variants,
    }
    SELECTION_PATH.write_text(json.dumps(selection_payload, indent=2), encoding="utf-8")

    print(f"Saved calibration CSV to: {CSV_PATH}")
    print(f"Saved calibration interpretation to: {SUMMARY_PATH}")
    print(f"Selected {selected_name} with weights: {selected_weights}")
    print(f"Saved selection payload to: {SELECTION_PATH}")


if __name__ == "__main__":
    main()
