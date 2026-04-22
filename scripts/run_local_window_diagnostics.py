"""Validate the local-window action representation against fixed segment-6."""

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
from rl.env_scan_local_window import ScanPlanningLocalWindowEnv
from rl.env_scan_segment import ScanPlanningSegmentEnv


FIGURE_DIR = PROJECT_ROOT / "assets" / "figures"
TABLE_DIR = PROJECT_ROOT / "assets" / "models"
SELECTION_PATH = TABLE_DIR / "reward_calibration_selection.json"

BASELINE_CSV = TABLE_DIR / "reward_breakdown_baselines_local_window.csv"
BASELINE_SUMMARY_TXT = TABLE_DIR / "reward_breakdown_baselines_local_window_summary.txt"
BASELINE_SUMMARY_MD = TABLE_DIR / "reward_breakdown_baselines_local_window_summary.md"
BREAKDOWN_PLOT = FIGURE_DIR / "baseline_reward_breakdown_local_window.png"
HEATMAP_PLOT = FIGURE_DIR / "baseline_heatmap_comparison_local_window.png"
ORDER_PLOT = FIGURE_DIR / "baseline_scan_order_comparison_local_window.png"

COMPARISON_CSV = TABLE_DIR / "local_window_vs_segment6_comparison.csv"
COMPARISON_TXT = TABLE_DIR / "local_window_vs_segment6_comparison.txt"
COMPARISON_MD = TABLE_DIR / "local_window_vs_segment6_comparison.md"
VERDICT_TXT = TABLE_DIR / "local_window_action_verdict.txt"
VERDICT_MD = TABLE_DIR / "local_window_action_verdict.md"

EARLY_FRACTION = 0.30
ADJACENCY_RADIUS = 2.0
ADJACENCY_HISTORY_WINDOW = 5
EPISODES = 20


def load_variant1_weights() -> dict[str, float]:
    """Load the calibrated Stage A reward setting."""
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
    """Write matching TXT and Markdown files."""
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    md_path.write_text("\n".join([f"# {title}", ""] + lines) + "\n", encoding="utf-8")


def make_segment6_env(mask: np.ndarray, reward_weights: dict[str, float]) -> ScanPlanningSegmentEnv:
    """Create the validated fixed segment-6 environment."""
    return ScanPlanningSegmentEnv(
        planning_mask=mask,
        grid_size=mask.shape[0],
        reward_weights=reward_weights,
        segments_per_stripe=6,
        action_mode="fixed",
    )


def make_local_env(mask: np.ndarray, reward_weights: dict[str, float]) -> ScanPlanningLocalWindowEnv:
    """Create the new local-window environment."""
    return ScanPlanningLocalWindowEnv(
        planning_mask=mask,
        grid_size=mask.shape[0],
        reward_weights=reward_weights,
        window_size=3,
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


def get_action_center(env: Any, action: int) -> tuple[int, int] | None:
    """Return the center of the currently executable cells for one action."""
    if isinstance(env, ScanPlanningLocalWindowEnv):
        mask = env.action_masks_catalog[int(action)]
    else:
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
    """Measure how often action centers remain near recent history."""
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
    """Compute early clustering and thermal summaries from one rollout."""
    centers = result["action_centers"]
    reward_history = result["reward_history"]
    step_stats = result["step_stats"]
    early_stop = phase_end(len(reward_history))
    return {
        "early_adjacency_ratio": adjacency_ratio(centers[: phase_end(len(centers))]),
        "cumulative_early_reheat": float(sum(float(step["reheat"]) for step in reward_history[:early_stop])),
        "early_peak_heat": float(np.mean([float(item["peak_heat"]) for item in step_stats[:early_stop]])) if early_stop > 0 else 0.0,
        "early_heat_variance": float(np.mean([float(item["heat_variance"]) for item in step_stats[:early_stop]])) if early_stop > 0 else 0.0,
    }


def reward_signal_share(
    env_factory: Callable[[np.ndarray, dict[str, float]], Any],
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


def compress_segment6_actions(
    cell_plan: list[tuple[int, int]],
    cell_to_action: dict[tuple[int, int], int],
) -> list[int]:
    """Compress a cell-order plan into unique segment-6 actions."""
    actions: list[int] = []
    seen: set[int] = set()
    for cell in cell_plan:
        action_index = cell_to_action[cell]
        if action_index in seen:
            continue
        actions.append(action_index)
        seen.add(action_index)
    return actions


def choose_local_action(
    cell: tuple[int, int],
    plan_index: dict[tuple[int, int], int],
    lookahead_cells: list[tuple[int, int]],
    env: ScanPlanningLocalWindowEnv,
    covered: set[tuple[int, int]],
) -> int:
    """Choose the best local-window action centered at one cell."""
    row, col = cell
    best_action: int | None = None
    best_score: tuple[float, float, int] | None = None
    lookahead_set = set(lookahead_cells)
    for orientation_index in range(len(env.ORIENTATION_NAMES)):
        action_index = env.action_lookup[(int(row), int(col), int(orientation_index))]
        action_cells = [action_cell for action_cell in env.action_cells[action_index] if action_cell not in covered]
        if not action_cells:
            continue
        lookahead_hits = sum(1 for action_cell in action_cells if action_cell in lookahead_set)
        score = (float(lookahead_hits), float(len(action_cells)), -int(orientation_index))
        if best_score is None or score > best_score:
            best_score = score
            best_action = action_index
    if best_action is None:
        raise RuntimeError("No valid local-window action found for planner compression.")
    return best_action


def compress_local_actions(
    cell_plan: list[tuple[int, int]],
    env: ScanPlanningLocalWindowEnv,
) -> list[int]:
    """Compress a cell-order plan into local-window actions."""
    actions: list[int] = []
    covered: set[tuple[int, int]] = set()
    for index, cell in enumerate(cell_plan):
        if cell in covered:
            continue
        lookahead_cells = cell_plan[index : index + 3]
        action_index = choose_local_action(cell, {}, lookahead_cells, env, covered)
        actions.append(action_index)
        for action_cell in env.action_cells[action_index]:
            covered.add(action_cell)
    return actions


def run_action_plan(
    env_factory: Callable[[np.ndarray, dict[str, float]], Any],
    mask: np.ndarray,
    reward_weights: dict[str, float],
    action_plan: list[int],
) -> dict[str, Any]:
    """Execute one action plan and return baseline-style outputs."""
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
    step_stats: list[dict[str, float]] = []
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
        step_stats.append(
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
        "step_stats": step_stats,
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


def build_local_catalog_rows(env: ScanPlanningLocalWindowEnv) -> list[dict[str, Any]]:
    """Summarise local-window catalog locality."""
    sizes = [len(cells) for cells in env.action_cells if cells]
    return [
        {
            "environment": "local_window",
            "metric": "action_catalog_size",
            "value": float(len(env.action_cells)),
        },
        {
            "environment": "local_window",
            "metric": "affected_cells_mean",
            "value": float(np.mean(sizes)),
        },
        {
            "environment": "local_window",
            "metric": "affected_cells_std",
            "value": float(np.std(sizes)),
        },
        {
            "environment": "local_window",
            "metric": "affected_cells_min",
            "value": float(np.min(sizes)),
        },
        {
            "environment": "local_window",
            "metric": "affected_cells_max",
            "value": float(np.max(sizes)),
        },
    ]


def main() -> None:
    """Run baseline validation and diagnostics for the local-window environment."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    reward_weights = load_variant1_weights()
    target_mask = build_target_mask()

    segment_env = make_segment6_env(target_mask, reward_weights)
    segment_env.reset(seed=42)
    segment_mapping = segment_env.cell_to_segment_action.copy()
    segment_action_count = len(segment_env.segment_cells)
    segment_sizes = [len(cells) for cells in segment_env.segment_cells if cells]
    segment_env.close()

    local_env = make_local_env(target_mask, reward_weights)
    local_env.reset(seed=42)
    local_catalog_rows = build_local_catalog_rows(local_env)

    cell_plans = planner_cell_plans(target_mask)
    local_results: dict[str, dict[str, Any]] = {}
    reward_rows: list[dict[str, Any]] = []
    for planner_name, cell_plan in cell_plans.items():
        action_plan = compress_local_actions(cell_plan, local_env)
        result = run_action_plan(make_local_env, target_mask, reward_weights, action_plan)
        local_results[planner_name] = result
        reward_rows.append({"planner": planner_name, **result["reward_breakdown"]})

    local_env.close()

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
        local_results,
        HEATMAP_PLOT,
        field_key="final_thermal",
        title="Local-Window Baseline Heatmap Comparison",
        cmap="magma",
        colorbar_label="Proxy Thermal Level",
    )
    save_comparison_grid(
        local_results,
        ORDER_PLOT,
        field_key="order_map",
        title="Local-Window Baseline Scan Order Comparison",
        cmap="inferno",
        colorbar_label="Scan Step",
    )

    ranking = sorted(reward_rows, key=lambda row: float(row["total_reward"]), reverse=True)
    summary_lines = [
        "Local-window baseline summary",
        "- Ranking by total reward: " + ", ".join(
            f"{row['planner']} ({float(row['total_reward']):.3f})" for row in ranking
        ),
    ]
    write_text_pair(BASELINE_SUMMARY_TXT, BASELINE_SUMMARY_MD, "Local-Window Baseline Summary", summary_lines)

    segment_signal = reward_signal_share(make_segment6_env, target_mask, reward_weights)
    local_signal = reward_signal_share(make_local_env, target_mask, reward_weights)

    segment_random = run_action_plan(
        make_segment6_env,
        target_mask,
        reward_weights,
        compress_segment6_actions(cell_plans["random"], segment_mapping),
    )
    segment_distance = run_action_plan(
        make_segment6_env,
        target_mask,
        reward_weights,
        compress_segment6_actions(cell_plans["distance_aware_cool_first"], segment_mapping),
    )
    segment_cool = run_action_plan(
        make_segment6_env,
        target_mask,
        reward_weights,
        compress_segment6_actions(cell_plans["cool_first"], segment_mapping),
    )

    local_env_for_compression = make_local_env(target_mask, reward_weights)
    local_env_for_compression.reset(seed=42)
    local_random = run_action_plan(
        make_local_env,
        target_mask,
        reward_weights,
        compress_local_actions(cell_plans["random"], local_env_for_compression),
    )
    local_distance = run_action_plan(
        make_local_env,
        target_mask,
        reward_weights,
        compress_local_actions(cell_plans["distance_aware_cool_first"], local_env_for_compression),
    )
    local_cool = run_action_plan(
        make_local_env,
        target_mask,
        reward_weights,
        compress_local_actions(cell_plans["cool_first"], local_env_for_compression),
    )
    local_env_for_compression.close()

    comparison_rows = [
        {
            "environment": "segment6",
            "metric": "action_catalog_size",
            "value": float(segment_action_count),
        },
        {
            "environment": "segment6",
            "metric": "affected_cells_mean",
            "value": float(np.mean(segment_sizes)),
        },
        {
            "environment": "segment6",
            "metric": "affected_cells_std",
            "value": float(np.std(segment_sizes)),
        },
        {
            "environment": "segment6",
            "metric": "affected_cells_min",
            "value": float(np.min(segment_sizes)),
        },
        {
            "environment": "segment6",
            "metric": "affected_cells_max",
            "value": float(np.max(segment_sizes)),
        },
        *local_catalog_rows,
        {
            "environment": "segment6",
            "metric": "coverage_completion_share",
            "value": segment_signal["coverage_completion_share"],
        },
        {
            "environment": "segment6",
            "metric": "thermal_share",
            "value": segment_signal["thermal_share"],
        },
        {
            "environment": "segment6",
            "metric": "jump_share",
            "value": segment_signal["jump_share"],
        },
        {
            "environment": "local_window",
            "metric": "coverage_completion_share",
            "value": local_signal["coverage_completion_share"],
        },
        {
            "environment": "local_window",
            "metric": "thermal_share",
            "value": local_signal["thermal_share"],
        },
        {
            "environment": "local_window",
            "metric": "jump_share",
            "value": local_signal["jump_share"],
        },
    ]
    for environment_name, result in (
        ("segment6_random", segment_random),
        ("segment6_distance_aware_cool_first", segment_distance),
        ("segment6_cool_first", segment_cool),
        ("local_window_random", local_random),
        ("local_window_distance_aware_cool_first", local_distance),
        ("local_window_cool_first", local_cool),
    ):
        for metric_name, value in early_metrics(result).items():
            comparison_rows.append({"environment": environment_name, "metric": metric_name, "value": value})

    write_csv(COMPARISON_CSV, ["environment", "metric", "value"], comparison_rows)

    segment_distance_metrics = early_metrics(segment_distance)
    local_distance_metrics = early_metrics(local_distance)
    local_mean = next(row["value"] for row in local_catalog_rows if row["metric"] == "affected_cells_mean")
    local_catalog_size = next(row["value"] for row in local_catalog_rows if row["metric"] == "action_catalog_size")
    distance_top = ranking[0]["planner"]
    raster_bottom = ranking[-1]["planner"]

    comparison_lines = [
        f"- Segment6 locality mean={np.mean(segment_sizes):.2f}, local-window locality mean={float(local_mean):.2f}",
        f"- Segment6 thermal share={segment_signal['thermal_share']:.3f}, local-window thermal share={local_signal['thermal_share']:.3f}",
        f"- Segment6 action catalog size={segment_action_count}, local-window action catalog size={int(local_catalog_size)}",
        f"- Segment6 distance-aware early adjacency={segment_distance_metrics['early_adjacency_ratio']:.3f}, local-window distance-aware early adjacency={local_distance_metrics['early_adjacency_ratio']:.3f}",
        f"- Segment6 distance-aware cumulative early reheat={segment_distance_metrics['cumulative_early_reheat']:.3f}, local-window distance-aware cumulative early reheat={local_distance_metrics['cumulative_early_reheat']:.3f}",
        f"- Local-window ranking top={distance_top}, bottom={raster_bottom}",
    ]
    write_text_pair(COMPARISON_TXT, COMPARISON_MD, "Local Window vs Segment6 Comparison", comparison_lines)

    more_local = float(local_mean) < float(np.mean(segment_sizes))
    thermal_not_weaker = local_signal["thermal_share"] >= segment_signal["thermal_share"]
    sensible_ranking = distance_top == "distance_aware_cool_first" and raster_bottom == "raster"
    distance_structured = local_distance_metrics["early_adjacency_ratio"] <= 0.10
    better_candidate = more_local and thermal_not_weaker and sensible_ranking and distance_structured

    verdict_lines = [
        f"1. Is the new local-window action more local than fixed segment-6? {'YES' if more_local else 'NO'}",
        f"2. Is thermal signal share stronger or at least not weaker than fixed segment-6? {'YES' if thermal_not_weaker else 'NO'}",
        f"3. Does the new representation preserve sensible baseline ranking? {'YES' if sensible_ranking else 'NO'}",
        f"4. Does distance_aware_cool_first remain structured and low-clustering? {'YES' if distance_structured else 'NO'}",
        f"5. Does this representation look more promising for PPO than fixed segment-6? {'YES' if better_candidate else 'NO'}",
        "",
        (
            f"Most important quantitative improvement: action locality mean {np.mean(segment_sizes):.2f} -> {float(local_mean):.2f} cells/action."
        ),
        (
            f"Most important new risk: thermal share {segment_signal['thermal_share']:.3f} -> {local_signal['thermal_share']:.3f} "
            f"with distance-aware early adjacency {segment_distance_metrics['early_adjacency_ratio']:.3f} -> {local_distance_metrics['early_adjacency_ratio']:.3f}."
        ),
        "",
        (
            "GO: local-window action is a better candidate than segment-6 for the next PPO smoke test"
            if better_candidate
            else "NO-GO: local-window action does not improve the representation enough, do not use it for PPO yet"
        ),
    ]
    write_text_pair(VERDICT_TXT, VERDICT_MD, "Local Window Action Verdict", verdict_lines)

    print("Local-window diagnostics complete.")
    print(f"Saved baseline CSV to: {BASELINE_CSV}")
    print(f"Saved comparison CSV to: {COMPARISON_CSV}")
    print(f"Saved verdict to: {VERDICT_TXT}")


if __name__ == "__main__":
    main()
