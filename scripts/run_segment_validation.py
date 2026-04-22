"""Run the mandatory segment-environment validation round before PPO smoke testing."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.geometry import downsample_mask, generate_text_mask
from core.metrics import summarise_run
from core.planners.checkerboard import plan_checkerboard
from core.planners.cool_first import plan_cool_first
from core.planners.distance_aware_cool_first import plan_distance_aware_cool_first
from core.planners.greedy_cool_first import plan_greedy_cool_first
from core.planners.random_planner import plan_random
from core.planners.raster import plan_raster
from core.viz import (
    save_comparison_grid,
    save_order_map_figure,
    save_reward_breakdown_chart,
    save_thermal_map_figure,
)
from rl.env_scan import ScanPlanningEnv
from rl.env_scan_segment import ScanPlanningSegmentEnv


TABLE_DIR = PROJECT_ROOT / "assets" / "models"
FIGURE_DIR = PROJECT_ROOT / "assets" / "figures"
SELECTION_PATH = TABLE_DIR / "reward_calibration_selection.json"

SEGMENT_CSV = TABLE_DIR / "reward_breakdown_baselines_segment_validation.csv"
SEGMENT_SUMMARY = TABLE_DIR / "reward_breakdown_baselines_segment_validation_summary.txt"
SEGMENT_INTERPRETATION = TABLE_DIR / "reward_breakdown_baselines_segment_validation_interpretation.txt"
SEGMENT_BREAKDOWN_PLOT = FIGURE_DIR / "baseline_reward_breakdown_segment_validation.png"
SEGMENT_HEATMAP_PLOT = FIGURE_DIR / "baseline_heatmap_comparison_segment_validation.png"
SEGMENT_ORDER_PLOT = FIGURE_DIR / "baseline_scan_order_comparison_segment_validation.png"

COMPARISON_CSV = TABLE_DIR / "action_granularity_comparison_validation.csv"
COMPARISON_TXT = TABLE_DIR / "action_granularity_comparison_validation.txt"
COMPARISON_MD = TABLE_DIR / "action_granularity_comparison_validation.md"
VERDICT_TXT = TABLE_DIR / "segment_validation_verdict.txt"
VERDICT_MD = TABLE_DIR / "segment_validation_verdict.md"

EPISODES = 20
EARLY_FRACTION = 0.30
ADJACENCY_RADIUS = 2.0
ADJACENCY_HISTORY_WINDOW = 5


def load_variant1_weights() -> dict[str, float]:
    """Load the calibrated reward configuration."""
    payload = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    return {key: float(value) for key, value in payload["variants"]["variant_1"].items()}


def build_target_mask() -> np.ndarray:
    """Build the standard TWI coarse mask."""
    return downsample_mask(generate_text_mask("TWI", canvas_size=1024), grid_size=64, threshold=0.2)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def write_text_pair(txt_path: Path, md_path: Path, title: str, lines: list[str]) -> None:
    """Write matching TXT and Markdown outputs."""
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    md_path.write_text("\n".join([f"# {title}", ""] + lines) + "\n", encoding="utf-8")


def make_stripe_env(mask: np.ndarray, reward_weights: dict[str, float]) -> ScanPlanningEnv:
    """Create the original stripe environment."""
    return ScanPlanningEnv(planning_mask=mask, grid_size=mask.shape[0], reward_weights=reward_weights)


def make_segment_env(mask: np.ndarray, reward_weights: dict[str, float]) -> ScanPlanningSegmentEnv:
    """Create the fixed-segment environment."""
    return ScanPlanningSegmentEnv(planning_mask=mask, grid_size=mask.shape[0], reward_weights=reward_weights)


def build_cell_to_stripe_action(env: ScanPlanningEnv) -> dict[tuple[int, int], int]:
    """Map each target cell to its stripe action id."""
    mapping: dict[tuple[int, int], int] = {}
    for stripe_index, stripe in enumerate(env.stripes):
        for cell in env._stripe_cells(stripe):
            mapping[cell] = stripe_index
    return mapping


def compress_cell_plan(cell_plan: list[tuple[int, int]], cell_to_action: dict[tuple[int, int], int]) -> list[int]:
    """Convert a cell-order plan into a unique action sequence."""
    actions: list[int] = []
    seen: set[int] = set()
    for cell in cell_plan:
        action = cell_to_action[cell]
        if action in seen:
            continue
        actions.append(action)
        seen.add(action)
    return actions


def get_action_center(env: Any, action: int) -> tuple[int, int] | None:
    """Return a stable center for one environment action."""
    if isinstance(env, ScanPlanningSegmentEnv):
        mask = env.segment_masks[int(action)]
    else:
        mask = env.stripes[int(action)]
    remaining = np.logical_and(mask, env.target_mask & ~env.scanned_mask)
    rows, cols = np.nonzero(remaining if remaining.any() else mask)
    if len(rows) == 0:
        return None
    return int(np.rint(rows.mean())), int(np.rint(cols.mean()))


def run_env_action_plan(env_factory: Any, mask: np.ndarray, reward_weights: dict[str, float], action_plan: list[int]) -> dict[str, Any]:
    """Execute an environment action sequence and return baseline-style outputs."""
    env = env_factory(mask, reward_weights)
    _, _ = env.reset(seed=42)
    reward_totals = {
        "coverage": 0.0,
        "invalid": 0.0,
        "peak": 0.0,
        "variance": 0.0,
        "reheat": 0.0,
        "jump": 0.0,
        "completion_bonus": 0.0,
        "support_risk": 0.0,
        "total_reward": 0.0,
    }
    reward_history: list[dict[str, float]] = []
    action_centers: list[tuple[int, int]] = []

    terminated = False
    truncated = False
    for action in action_plan:
        if not env.is_valid_action(action):
            continue
        center = get_action_center(env, action)
        if center is not None:
            action_centers.append(center)
        _, _, terminated, truncated, info = env.step(action)
        reward_terms = info["reward_terms"]
        reward_history.append(reward_terms)
        for key in ("coverage", "invalid", "peak", "variance", "reheat", "jump", "completion_bonus", "support_risk"):
            reward_totals[key] += float(reward_terms[key])
        reward_totals["total_reward"] += float(reward_terms["total"])
        if terminated or truncated:
            break

    order_map = np.full(mask.shape, fill_value=-1, dtype=np.int32)
    for step_index, (row, col) in enumerate(env.executed_actions):
        order_map[row, col] = step_index
    metrics = summarise_run(
        target_mask=mask,
        scanned_mask=env.scanned_mask,
        final_thermal=env.thermal_field,
        actions=env.executed_actions,
    )
    result = {
        "target_mask": mask.copy(),
        "scanned_mask": env.scanned_mask.copy(),
        "order_map": order_map,
        "final_thermal": env.thermal_field.copy(),
        "metrics": metrics,
        "actions": list(env.executed_actions),
        "reward_history": reward_history,
        "action_centers": action_centers,
        "reward_breakdown": {
            **reward_totals,
            "steps": metrics["steps"],
            "coverage_ratio": metrics["coverage_ratio"],
            "peak_heat_final": metrics["thermal_peak"],
            "heat_variance_final": metrics["thermal_variance"],
        },
    }
    env.close()
    return result


def phase_end(length: int) -> int:
    """Return the exclusive end index for the early phase."""
    return max(1, math.ceil(length * EARLY_FRACTION))


def adjacency_ratio(points: list[tuple[int, int]]) -> float:
    """Return the fraction of actions near the recent action window."""
    if len(points) <= 1:
        return 0.0
    adjacent = 0
    considered = 0
    for index, point in enumerate(points):
        if index == 0:
            continue
        history = points[max(0, index - ADJACENCY_HISTORY_WINDOW) : index]
        considered += 1
        if any(np.hypot(point[0] - old[0], point[1] - old[1]) <= ADJACENCY_RADIUS for old in history):
            adjacent += 1
    return adjacent / max(considered, 1)


def reward_signal_share(env_factory: Any, mask: np.ndarray, reward_weights: dict[str, float]) -> dict[str, float]:
    """Estimate reward-term shares under masked-random exploratory rollouts."""
    term_names = ["coverage", "completion_bonus", "invalid", "peak", "variance", "reheat", "jump"]
    abs_totals = {term: 0.0 for term in term_names}
    total_abs = 0.0

    for episode_id in range(EPISODES):
        env = env_factory(mask, reward_weights)
        env.reset(seed=700 + episode_id)
        rng = np.random.default_rng(70_000 + episode_id)
        terminated = False
        truncated = False
        while not (terminated or truncated):
            valid_actions = np.flatnonzero(env.action_masks())
            if len(valid_actions) == 0:
                break
            action = int(rng.choice(valid_actions))
            _, _, terminated, truncated, info = env.step(action)
            for term in term_names:
                abs_value = abs(float(info["reward_terms"][term]))
                abs_totals[term] += abs_value
                total_abs += abs_value
        env.close()

    return {
        "coverage_completion_share": (abs_totals["coverage"] + abs_totals["completion_bonus"]) / max(total_abs, 1e-9),
        "thermal_share": (abs_totals["peak"] + abs_totals["variance"] + abs_totals["reheat"]) / max(total_abs, 1e-9),
        "jump_share": abs_totals["jump"] / max(total_abs, 1e-9),
    }


def baseline_clustering(result: dict[str, Any]) -> dict[str, float]:
    """Compute early adjacency/reheat/peak/variance from a baseline rollout result."""
    centers = result["action_centers"]
    reward_history = result["reward_history"]
    early_stop = phase_end(len(centers))
    early_thermal = result["final_thermal"] if not result["actions"] else result["final_thermal"]
    # For the validation decision, use rollout-derived early reheat and center-based adjacency.
    return {
        "early_adjacency_ratio": adjacency_ratio(centers[:early_stop]),
        "cumulative_early_reheat": float(sum(float(step["reheat"]) for step in reward_history[:early_stop])),
        "early_peak_heat": float(np.mean([abs(float(step["peak"])) for step in reward_history[:early_stop]])) / 0.8 if early_stop > 0 else 0.0,
        "early_heat_variance": float(np.mean([abs(float(step["variance"])) for step in reward_history[:early_stop]])) if early_stop > 0 else 0.0,
    }


def run_segment_baselines(mask: np.ndarray, reward_weights: dict[str, float]) -> list[dict[str, Any]]:
    """Run the stronger baseline suite in the segment environment and save validation outputs."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    mapping_env = make_segment_env(mask, reward_weights)
    mapping_env.reset(seed=42)
    cell_to_action = mapping_env.cell_to_segment_action.copy()
    mapping_env.close()

    cell_plans = {
        "raster": plan_raster(mask),
        "random": plan_random(mask, seed=42),
        "greedy": plan_greedy_cool_first(mask),
        "cool_first": plan_cool_first(mask),
        "checkerboard": plan_checkerboard(mask),
        "distance_aware_cool_first": plan_distance_aware_cool_first(mask),
    }
    action_plans = {
        planner_name: compress_cell_plan(cell_actions, cell_to_action)
        for planner_name, cell_actions in cell_plans.items()
    }

    results: dict[str, dict[str, Any]] = {}
    reward_rows: list[dict[str, Any]] = []
    for planner_name, action_plan in action_plans.items():
        result = run_env_action_plan(make_segment_env, mask, reward_weights, action_plan)
        results[planner_name] = result
        reward_breakdown = result["reward_breakdown"].copy()
        reward_breakdown["planner"] = planner_name
        reward_rows.append(reward_breakdown)

    write_csv(
        SEGMENT_CSV,
        [
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
        ],
        reward_rows,
    )
    save_reward_breakdown_chart(reward_rows, SEGMENT_BREAKDOWN_PLOT)
    save_comparison_grid(
        results,
        SEGMENT_HEATMAP_PLOT,
        field_key="final_thermal",
        title="Segment Validation Heatmap Comparison",
        cmap="magma",
        colorbar_label="Proxy Thermal Level",
    )
    save_comparison_grid(
        results,
        SEGMENT_ORDER_PLOT,
        field_key="order_map",
        title="Segment Validation Scan Order Comparison",
        cmap="inferno",
        colorbar_label="Scan Step",
    )

    ranking = sorted(reward_rows, key=lambda row: float(row["total_reward"]), reverse=True)
    summary_lines = [
        "Segment validation baseline summary",
        (
            "- Ranking by total reward: "
            + ", ".join(f"{row['planner']} ({float(row['total_reward']):.3f})" for row in ranking)
        ),
    ]
    SEGMENT_SUMMARY.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    top = ranking[0]["planner"]
    bottom = ranking[-1]["planner"]
    SEGMENT_INTERPRETATION.write_text(
        "\n".join(
            [
                "Segment validation interpretation",
                f"- Top planner: {top}",
                f"- Bottom planner: {bottom}",
                "- Validation is sensible if distance-aware cool-first stays near the top and raster stays near the bottom.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return reward_rows


def compare_granularity(mask: np.ndarray, reward_weights: dict[str, float]) -> dict[str, Any]:
    """Run the direct stripe-vs-segment comparison needed for the PASS/FAIL verdict."""
    stripe_env = make_stripe_env(mask, reward_weights)
    stripe_env.reset(seed=42)
    stripe_sizes = [len(stripe_env._stripe_cells(stripe)) for stripe in stripe_env.stripes]
    stripe_mapping = build_cell_to_stripe_action(stripe_env)
    stripe_env.close()

    segment_env = make_segment_env(mask, reward_weights)
    segment_env.reset(seed=42)
    segment_sizes = [len(cells) for cells in segment_env.segment_cells if cells]
    segment_mapping = segment_env.cell_to_segment_action.copy()
    segment_env.close()

    stripe_signal = reward_signal_share(make_stripe_env, mask, reward_weights)
    segment_signal = reward_signal_share(make_segment_env, mask, reward_weights)

    random_plan = plan_random(mask, seed=42)
    distance_plan = plan_distance_aware_cool_first(mask)

    stripe_random = baseline_clustering(
        run_env_action_plan(make_stripe_env, mask, reward_weights, compress_cell_plan(random_plan, stripe_mapping))
    )
    stripe_distance = baseline_clustering(
        run_env_action_plan(make_stripe_env, mask, reward_weights, compress_cell_plan(distance_plan, stripe_mapping))
    )
    segment_random = baseline_clustering(
        run_env_action_plan(make_segment_env, mask, reward_weights, compress_cell_plan(random_plan, segment_mapping))
    )
    segment_distance = baseline_clustering(
        run_env_action_plan(make_segment_env, mask, reward_weights, compress_cell_plan(distance_plan, segment_mapping))
    )

    rows = [
        {"environment": "stripe", "metric": "affected_cells_mean", "value": float(np.mean(stripe_sizes))},
        {"environment": "stripe", "metric": "affected_cells_min", "value": int(np.min(stripe_sizes))},
        {"environment": "stripe", "metric": "affected_cells_max", "value": int(np.max(stripe_sizes))},
        {"environment": "segment", "metric": "affected_cells_mean", "value": float(np.mean(segment_sizes))},
        {"environment": "segment", "metric": "affected_cells_min", "value": int(np.min(segment_sizes))},
        {"environment": "segment", "metric": "affected_cells_max", "value": int(np.max(segment_sizes))},
        {"environment": "stripe", "metric": "coverage_completion_share", "value": stripe_signal["coverage_completion_share"]},
        {"environment": "stripe", "metric": "thermal_share", "value": stripe_signal["thermal_share"]},
        {"environment": "stripe", "metric": "jump_share", "value": stripe_signal["jump_share"]},
        {"environment": "segment", "metric": "coverage_completion_share", "value": segment_signal["coverage_completion_share"]},
        {"environment": "segment", "metric": "thermal_share", "value": segment_signal["thermal_share"]},
        {"environment": "segment", "metric": "jump_share", "value": segment_signal["jump_share"]},
    ]
    for planner_name, metrics in (
        ("stripe_random", stripe_random),
        ("stripe_distance_aware_cool_first", stripe_distance),
        ("segment_random", segment_random),
        ("segment_distance_aware_cool_first", segment_distance),
    ):
        for metric_name, value in metrics.items():
            rows.append({"environment": planner_name, "metric": metric_name, "value": value})

    write_csv(COMPARISON_CSV, ["environment", "metric", "value"], rows)

    lines = [
        f"- Stripe affected cells/action: mean={np.mean(stripe_sizes):.2f}, min={min(stripe_sizes)}, max={max(stripe_sizes)}",
        f"- Segment affected cells/action: mean={np.mean(segment_sizes):.2f}, min={min(segment_sizes)}, max={max(segment_sizes)}",
        (
            f"- Stripe reward shares: coverage+completion={stripe_signal['coverage_completion_share']:.3f}, "
            f"thermal={stripe_signal['thermal_share']:.3f}, jump={stripe_signal['jump_share']:.3f}"
        ),
        (
            f"- Segment reward shares: coverage+completion={segment_signal['coverage_completion_share']:.3f}, "
            f"thermal={segment_signal['thermal_share']:.3f}, jump={segment_signal['jump_share']:.3f}"
        ),
        (
            f"- Stripe random early adjacency={stripe_random['early_adjacency_ratio']:.3f}, "
            f"early reheat={stripe_random['cumulative_early_reheat']:.3f}, "
            f"early peak={stripe_random['early_peak_heat']:.3f}"
        ),
        (
            f"- Segment random early adjacency={segment_random['early_adjacency_ratio']:.3f}, "
            f"early reheat={segment_random['cumulative_early_reheat']:.3f}, "
            f"early peak={segment_random['early_peak_heat']:.3f}"
        ),
        (
            f"- Stripe distance-aware early adjacency={stripe_distance['early_adjacency_ratio']:.3f}, "
            f"early reheat={stripe_distance['cumulative_early_reheat']:.3f}, "
            f"early peak={stripe_distance['early_peak_heat']:.3f}"
        ),
        (
            f"- Segment distance-aware early adjacency={segment_distance['early_adjacency_ratio']:.3f}, "
            f"early reheat={segment_distance['cumulative_early_reheat']:.3f}, "
            f"early peak={segment_distance['early_peak_heat']:.3f}"
        ),
    ]
    write_text_pair(COMPARISON_TXT, COMPARISON_MD, "Action Granularity Comparison Validation", lines)

    return {
        "stripe_mean": float(np.mean(stripe_sizes)),
        "segment_mean": float(np.mean(segment_sizes)),
        "stripe_signal": stripe_signal,
        "segment_signal": segment_signal,
    }


def write_verdict(granularity: dict[str, Any], reward_rows: list[dict[str, Any]]) -> bool:
    """Write the hard PASS/FAIL segment-validation verdict."""
    ranking = sorted(reward_rows, key=lambda row: float(row["total_reward"]), reverse=True)
    top_names = [row["planner"] for row in ranking[:3]]
    bottom_names = [row["planner"] for row in ranking[-2:]]
    locality_improved = granularity["segment_mean"] < granularity["stripe_mean"]
    thermal_not_weakened = granularity["segment_signal"]["thermal_share"] >= granularity["stripe_signal"]["thermal_share"]
    ranking_sensible = "distance_aware_cool_first" in top_names and "raster" in bottom_names
    thermal_discrimination_ok = (
        max(float(row["total_reward"]) for row in reward_rows if row["planner"] == "distance_aware_cool_first")
        > max(float(row["total_reward"]) for row in reward_rows if row["planner"] == "random")
    )
    passed = locality_improved and thermal_not_weakened and ranking_sensible and thermal_discrimination_ok

    verdict = "PASS: proceed to Segment-PPO smoke test" if passed else "FAIL: do not run PPO yet"
    lines = [
        (
            f"1. Action locality is {'substantially improved' if locality_improved else 'not clearly improved'}: "
            f"{granularity['stripe_mean']:.2f} -> {granularity['segment_mean']:.2f} affected cells/action."
        ),
        (
            f"2. Thermal signal is {'not weakened' if thermal_not_weakened else 'weakened'}: "
            f"stripe thermal share={granularity['stripe_signal']['thermal_share']:.3f}, "
            f"segment thermal share={granularity['segment_signal']['thermal_share']:.3f}."
        ),
        (
            f"3. Strong heuristic baselines remain {'sensible' if ranking_sensible else 'not clearly sensible'}: "
            f"top planners={', '.join(top_names)}, bottom planners={', '.join(bottom_names)}."
        ),
        (
            f"4. There is {'enough' if passed else 'not enough'} evidence that segment granularity reduces the original mismatch "
            "enough to justify PPO smoke testing."
        ),
        verdict,
    ]
    write_text_pair(VERDICT_TXT, VERDICT_MD, "Segment Validation Verdict", lines)
    return passed


def main() -> None:
    """Run the mandatory Step 0 segment validation."""
    reward_weights = load_variant1_weights()
    target_mask = build_target_mask()
    reward_rows = run_segment_baselines(target_mask, reward_weights)
    granularity = compare_granularity(target_mask, reward_weights)
    passed = write_verdict(granularity, reward_rows)
    print(f"Segment validation {'PASSED' if passed else 'FAILED'}")
    print(f"Saved validation CSV to: {SEGMENT_CSV}")
    print(f"Saved comparison CSV to: {COMPARISON_CSV}")
    print(f"Saved verdict to: {VERDICT_TXT}")


if __name__ == "__main__":
    main()
