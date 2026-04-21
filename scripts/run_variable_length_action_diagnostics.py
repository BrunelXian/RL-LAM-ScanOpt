"""Validate a variable-length segment action representation without PPO training."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

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
from core.viz import save_comparison_grid, save_reward_breakdown_chart
from rl.env_scan_segment import ScanPlanningSegmentEnv, ScanPlanningVariableSegmentEnv


FIGURE_DIR = PROJECT_ROOT / "assets" / "figures"
TABLE_DIR = PROJECT_ROOT / "assets" / "models"
SELECTION_PATH = TABLE_DIR / "reward_calibration_selection.json"

BASELINE_CSV = TABLE_DIR / "reward_breakdown_baselines_variable_segment.csv"
BASELINE_SUMMARY_TXT = TABLE_DIR / "reward_breakdown_baselines_variable_segment_summary.txt"
BASELINE_SUMMARY_MD = TABLE_DIR / "reward_breakdown_baselines_variable_segment_summary.md"
BREAKDOWN_PLOT = FIGURE_DIR / "baseline_reward_breakdown_variable_segment.png"
HEATMAP_PLOT = FIGURE_DIR / "baseline_heatmap_comparison_variable_segment.png"
ORDER_PLOT = FIGURE_DIR / "baseline_scan_order_comparison_variable_segment.png"

ACTION_SPACE_CSV = TABLE_DIR / "action_space_comparison_variable_segment.csv"
ACTION_SPACE_TXT = TABLE_DIR / "action_space_comparison_variable_segment.txt"
ACTION_SPACE_MD = TABLE_DIR / "action_space_comparison_variable_segment.md"

DIAGNOSTICS_CSV = TABLE_DIR / "variable_segment_diagnostics_comparison.csv"
DIAGNOSTICS_TXT = TABLE_DIR / "variable_segment_diagnostics_summary.txt"
DIAGNOSTICS_MD = TABLE_DIR / "variable_segment_diagnostics_summary.md"

ENV_NOTES_TXT = TABLE_DIR / "variable_segment_env_notes.txt"
ENV_NOTES_MD = TABLE_DIR / "variable_segment_env_notes.md"

EARLY_FRACTION = 0.30
ADJACENCY_RADIUS = 2.0
ADJACENCY_HISTORY_WINDOW = 5
EPISODES = 20


def load_variant1_weights() -> dict[str, float]:
    """Load the calibrated Stage A reward setting."""
    payload = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    return {key: float(value) for key, value in payload["variants"]["variant_1"].items()}


def build_target_mask() -> np.ndarray:
    """Build the standard coarse TWI mask."""
    return downsample_mask(generate_text_mask("TWI", canvas_size=1024), grid_size=64, threshold=0.2)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def write_text(path: Path, lines: list[str]) -> None:
    """Write a plain text file."""
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_text_pair(txt_path: Path, md_path: Path, title: str, lines: list[str]) -> None:
    """Write matching TXT and Markdown files."""
    write_text(txt_path, lines)
    md_path.write_text("\n".join([f"# {title}", ""] + lines) + "\n", encoding="utf-8")


def make_fixed_env(mask: np.ndarray, reward_weights: dict[str, float]) -> ScanPlanningSegmentEnv:
    """Create the historical fixed-6 segment environment."""
    return ScanPlanningSegmentEnv(
        planning_mask=mask,
        grid_size=mask.shape[0],
        reward_weights=reward_weights,
        segments_per_stripe=6,
        action_mode="fixed",
    )


def make_variable_env(mask: np.ndarray, reward_weights: dict[str, float]) -> ScanPlanningVariableSegmentEnv:
    """Create the new variable-length segment environment."""
    return ScanPlanningVariableSegmentEnv(
        planning_mask=mask,
        grid_size=mask.shape[0],
        reward_weights=reward_weights,
        min_segment_length=2,
        max_segment_length=8,
    )


def planner_cell_plans(mask: np.ndarray) -> dict[str, list[tuple[int, int]]]:
    """Return the stronger baseline cell-order plans."""
    return {
        "raster": plan_raster(mask),
        "random": plan_random(mask, seed=42),
        "greedy": plan_greedy_cool_first(mask),
        "cool_first": plan_cool_first(mask),
        "checkerboard": plan_checkerboard(mask),
        "distance_aware_cool_first": plan_distance_aware_cool_first(mask),
    }


def get_action_center(env: ScanPlanningSegmentEnv, action: int) -> tuple[int, int] | None:
    """Return the center of the currently executable cells for one action."""
    mask = env.segment_masks[int(action)]
    remaining = np.logical_and(mask, env.target_mask & ~env.scanned_mask)
    rows, cols = np.nonzero(remaining if remaining.any() else mask)
    if len(rows) == 0:
        return None
    return int(np.rint(rows.mean())), int(np.rint(cols.mean()))


def phase_end(length: int) -> int:
    """Return the exclusive end index for the early phase."""
    return max(1, math.ceil(length * EARLY_FRACTION))


def adjacency_ratio(points: list[tuple[int, int]]) -> float:
    """Measure how often action centers stay near the recent history window."""
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


def early_metrics(result: dict[str, Any]) -> dict[str, float]:
    """Compute early-phase clustering and thermal statistics."""
    centers = result["action_centers"]
    reward_history = result["reward_history"]
    step_thermal_stats = result["step_thermal_stats"]
    early_stop = phase_end(len(reward_history))
    early_peaks = [float(item["peak_heat"]) for item in step_thermal_stats[:early_stop]]
    early_variances = [float(item["heat_variance"]) for item in step_thermal_stats[:early_stop]]
    return {
        "early_adjacency_ratio": adjacency_ratio(centers[: phase_end(len(centers))]),
        "cumulative_early_reheat": float(sum(float(step["reheat"]) for step in reward_history[:early_stop])),
        "early_peak_heat": float(np.mean(early_peaks)) if early_peaks else 0.0,
        "early_heat_variance": float(np.mean(early_variances)) if early_variances else 0.0,
    }


def reward_signal_share(
    env_factory: Callable[[np.ndarray, dict[str, float]], ScanPlanningSegmentEnv],
    mask: np.ndarray,
    reward_weights: dict[str, float],
) -> dict[str, float]:
    """Estimate reward-term shares under masked-random exploratory episodes."""
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


def compress_fixed_actions(
    cell_plan: list[tuple[int, int]],
    cell_to_action: dict[tuple[int, int], int],
) -> list[int]:
    """Compress a cell-order plan to unique fixed-segment actions."""
    actions: list[int] = []
    seen: set[int] = set()
    for cell in cell_plan:
        action_index = cell_to_action[cell]
        if action_index in seen:
            continue
        actions.append(action_index)
        seen.add(action_index)
    return actions


def compress_variable_actions(
    cell_plan: list[tuple[int, int]],
    env: ScanPlanningVariableSegmentEnv,
) -> list[int]:
    """Compress a cell-order plan into variable-length stripe windows.

    We preserve planner structure by extending only along contiguous same-stripe
    runs that are also consecutive in the planner order. If no such run reaches
    length 2, we fall back to the minimum allowed window from that start cell.
    """
    plan_index = {cell: index for index, cell in enumerate(cell_plan)}
    actions: list[int] = []
    covered: set[tuple[int, int]] = set()

    for cell in cell_plan:
        if cell in covered:
            continue
        stripe_id, start_idx = env.cell_to_stripe_position[cell]
        ordered_cells = env.stripe_cell_orders[stripe_id]
        base_plan_index = plan_index[cell]
        run_length = 1
        while run_length < env.max_segment_length:
            next_idx = start_idx + run_length
            if next_idx >= len(ordered_cells):
                break
            next_cell = ordered_cells[next_idx]
            if next_cell in covered:
                break
            if plan_index.get(next_cell) != base_plan_index + run_length:
                break
            run_length += 1
        requested_length = min(max(run_length, env.min_segment_length), env.max_segment_length)
        action_index = env.action_lookup[(stripe_id, start_idx, requested_length)]
        actions.append(action_index)
        for action_cell in env.segment_cells[action_index]:
            covered.add(action_cell)
    return actions


def run_action_plan(
    env_factory: Callable[[np.ndarray, dict[str, float]], ScanPlanningSegmentEnv],
    mask: np.ndarray,
    reward_weights: dict[str, float],
    action_plan: list[int],
) -> dict[str, Any]:
    """Execute an action plan and return baseline-style outputs."""
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
    step_thermal_stats: list[dict[str, float]] = []
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
        step_thermal_stats.append(
            {
                "peak_heat": float(info["peak_heat"]),
                "heat_variance": float(info["heat_variance"]),
            }
        )
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
        "step_thermal_stats": step_thermal_stats,
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


def build_action_space_rows(
    fixed_env: ScanPlanningSegmentEnv,
    variable_env: ScanPlanningVariableSegmentEnv,
) -> list[dict[str, Any]]:
    """Summarise the old and new action spaces for side-by-side comparison."""
    fixed_sizes = [len(cells) for cells in fixed_env.segment_cells if cells]
    variable_sizes = [len(cells) for cells in variable_env.segment_cells if cells]
    return [
        {
            "env": "fixed_segment6",
            "action_format": "(stripe_id, segment_index)",
            "action_count": len(fixed_env.segment_cells),
            "mean_cells_per_action": float(np.mean(fixed_sizes)),
            "min_cells_per_action": int(np.min(fixed_sizes)),
            "max_cells_per_action": int(np.max(fixed_sizes)),
        },
        {
            "env": "variable_length",
            "action_format": "(stripe_id, start_cell, length[2..8])",
            "action_count": len(variable_env.segment_cells),
            "mean_cells_per_action": float(np.mean(variable_sizes)),
            "min_cells_per_action": int(np.min(variable_sizes)),
            "max_cells_per_action": int(np.max(variable_sizes)),
        },
    ]


def main() -> None:
    """Run baseline and diagnostics checks for the new variable-length representation."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    reward_weights = load_variant1_weights()
    target_mask = build_target_mask()

    fixed_env = make_fixed_env(target_mask, reward_weights)
    fixed_env.reset(seed=42)
    variable_env = make_variable_env(target_mask, reward_weights)
    variable_env.reset(seed=42)

    # Environment notes.
    env_lines = [
        "New variable-length segment environment",
        "- Base class: ScanPlanningSegmentEnv",
        "- Historical fixed action: (stripe_id, segment_index) with balanced fixed split into 6 parts",
        "- New variable-length action: (stripe_id, start_cell, length) where length is chosen from 2 to 8",
        "- Execution semantics: one action scans the currently unscanned cells inside that stripe window",
        "- Reward logic, thermal proxy, observation layout, and variant_1 reward weights are unchanged",
    ]
    write_text_pair(ENV_NOTES_TXT, ENV_NOTES_MD, "Variable Segment Environment Notes", env_lines)

    # Action-space comparison.
    action_space_rows = build_action_space_rows(fixed_env, variable_env)
    write_csv(
        ACTION_SPACE_CSV,
        [
            "env",
            "action_format",
            "action_count",
            "mean_cells_per_action",
            "min_cells_per_action",
            "max_cells_per_action",
        ],
        action_space_rows,
    )
    action_lines = [
        (
            f"- fixed_segment6: action_count={action_space_rows[0]['action_count']}, "
            f"cells/action={action_space_rows[0]['mean_cells_per_action']:.2f} "
            f"({action_space_rows[0]['min_cells_per_action']}..{action_space_rows[0]['max_cells_per_action']})"
        ),
        (
            f"- variable_length: action_count={action_space_rows[1]['action_count']}, "
            f"cells/action={action_space_rows[1]['mean_cells_per_action']:.2f} "
            f"({action_space_rows[1]['min_cells_per_action']}..{action_space_rows[1]['max_cells_per_action']})"
        ),
    ]
    write_text_pair(ACTION_SPACE_TXT, ACTION_SPACE_MD, "Action Space Comparison", action_lines)

    # Baseline conversion.
    cell_plans = planner_cell_plans(target_mask)
    fixed_mapping = fixed_env.cell_to_segment_action.copy()
    variable_mapping_env = make_variable_env(target_mask, reward_weights)
    variable_mapping_env.reset(seed=42)

    variable_results: dict[str, dict[str, Any]] = {}
    reward_rows: list[dict[str, Any]] = []
    for planner_name, cell_plan in cell_plans.items():
        action_plan = compress_variable_actions(cell_plan, variable_mapping_env)
        result = run_action_plan(make_variable_env, target_mask, reward_weights, action_plan)
        variable_results[planner_name] = result
        reward_rows.append({"planner": planner_name, **result["reward_breakdown"]})

    variable_mapping_env.close()
    fixed_env.close()
    variable_env.close()

    write_csv(
        BASELINE_CSV,
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
    save_reward_breakdown_chart(reward_rows, BREAKDOWN_PLOT)
    save_comparison_grid(
        variable_results,
        HEATMAP_PLOT,
        field_key="final_thermal",
        title="Variable-Length Segment Heatmap Comparison",
        cmap="magma",
        colorbar_label="Proxy Thermal Level",
    )
    save_comparison_grid(
        variable_results,
        ORDER_PLOT,
        field_key="order_map",
        title="Variable-Length Segment Scan Order Comparison",
        cmap="inferno",
        colorbar_label="Scan Step",
    )

    ranking = sorted(reward_rows, key=lambda row: float(row["total_reward"]), reverse=True)
    summary_lines = [
        "Variable-length segment baseline summary",
        "- Ranking by total reward: " + ", ".join(
            f"{row['planner']} ({float(row['total_reward']):.3f})" for row in ranking
        ),
        "- Expected sanity condition: distance_aware_cool_first should remain near the top and raster should remain near the bottom.",
    ]
    write_text_pair(BASELINE_SUMMARY_TXT, BASELINE_SUMMARY_MD, "Variable Segment Baseline Summary", summary_lines)

    # Diagnostics comparison against historical fixed segment6.
    fixed_signal = reward_signal_share(make_fixed_env, target_mask, reward_weights)
    variable_signal = reward_signal_share(make_variable_env, target_mask, reward_weights)

    fixed_random = run_action_plan(
        make_fixed_env,
        target_mask,
        reward_weights,
        compress_fixed_actions(cell_plans["random"], fixed_mapping),
    )
    fixed_distance = run_action_plan(
        make_fixed_env,
        target_mask,
        reward_weights,
        compress_fixed_actions(cell_plans["distance_aware_cool_first"], fixed_mapping),
    )
    fixed_cool = run_action_plan(
        make_fixed_env,
        target_mask,
        reward_weights,
        compress_fixed_actions(cell_plans["cool_first"], fixed_mapping),
    )

    variable_mapping_env = make_variable_env(target_mask, reward_weights)
    variable_mapping_env.reset(seed=42)
    variable_random = run_action_plan(
        make_variable_env,
        target_mask,
        reward_weights,
        compress_variable_actions(cell_plans["random"], variable_mapping_env),
    )
    variable_distance = run_action_plan(
        make_variable_env,
        target_mask,
        reward_weights,
        compress_variable_actions(cell_plans["distance_aware_cool_first"], variable_mapping_env),
    )
    variable_cool = run_action_plan(
        make_variable_env,
        target_mask,
        reward_weights,
        compress_variable_actions(cell_plans["cool_first"], variable_mapping_env),
    )
    variable_mapping_env.close()

    diagnostics_rows = [
        {
            "environment": "fixed_segment6",
            "metric": "thermal_share",
            "value": fixed_signal["thermal_share"],
        },
        {
            "environment": "fixed_segment6",
            "metric": "coverage_completion_share",
            "value": fixed_signal["coverage_completion_share"],
        },
        {
            "environment": "fixed_segment6",
            "metric": "jump_share",
            "value": fixed_signal["jump_share"],
        },
        {
            "environment": "variable_length",
            "metric": "thermal_share",
            "value": variable_signal["thermal_share"],
        },
        {
            "environment": "variable_length",
            "metric": "coverage_completion_share",
            "value": variable_signal["coverage_completion_share"],
        },
        {
            "environment": "variable_length",
            "metric": "jump_share",
            "value": variable_signal["jump_share"],
        },
    ]

    for environment_name, result in (
        ("fixed_random", fixed_random),
        ("fixed_distance_aware_cool_first", fixed_distance),
        ("fixed_cool_first", fixed_cool),
        ("variable_random", variable_random),
        ("variable_distance_aware_cool_first", variable_distance),
        ("variable_cool_first", variable_cool),
    ):
        for metric_name, value in early_metrics(result).items():
            diagnostics_rows.append({"environment": environment_name, "metric": metric_name, "value": value})

    write_csv(DIAGNOSTICS_CSV, ["environment", "metric", "value"], diagnostics_rows)

    diagnostics_lines = [
        (
            f"- Thermal share: fixed_segment6={fixed_signal['thermal_share']:.3f}, "
            f"variable_length={variable_signal['thermal_share']:.3f}"
        ),
        (
            f"- Coverage+completion share: fixed_segment6={fixed_signal['coverage_completion_share']:.3f}, "
            f"variable_length={variable_signal['coverage_completion_share']:.3f}"
        ),
        (
            f"- Random early adjacency: fixed_segment6={early_metrics(fixed_random)['early_adjacency_ratio']:.3f}, "
            f"variable_length={early_metrics(variable_random)['early_adjacency_ratio']:.3f}"
        ),
        (
            f"- Distance-aware early adjacency: fixed_segment6={early_metrics(fixed_distance)['early_adjacency_ratio']:.3f}, "
            f"variable_length={early_metrics(variable_distance)['early_adjacency_ratio']:.3f}"
        ),
        (
            f"- Distance-aware cumulative early reheat: fixed_segment6={early_metrics(fixed_distance)['cumulative_early_reheat']:.3f}, "
            f"variable_length={early_metrics(variable_distance)['cumulative_early_reheat']:.3f}"
        ),
        (
            f"- Baseline top planner in variable environment: {ranking[0]['planner']}"
        ),
        (
            "- Interpretation: the new representation is better only if thermal share does not weaken "
            "and distance-aware/cool-first remain clearly structured relative to random."
        ),
    ]
    write_text_pair(DIAGNOSTICS_TXT, DIAGNOSTICS_MD, "Variable Segment Diagnostics Summary", diagnostics_lines)

    print("Variable-length segment diagnostics complete.")
    print(f"Saved baseline CSV to: {BASELINE_CSV}")
    print(f"Saved action-space comparison to: {ACTION_SPACE_CSV}")
    print(f"Saved diagnostics summary to: {DIAGNOSTICS_TXT}")


if __name__ == "__main__":
    main()
