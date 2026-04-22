"""Validate the directional primitive against segment-6 and local primitive."""

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
from rl.env_scan_directional_primitive import ScanPlanningDirectionalPrimitiveEnv
from rl.env_scan_local_primitive import ScanPlanningLocalPrimitiveEnv
from rl.env_scan_segment import ScanPlanningSegmentEnv


FIGURE_DIR = PROJECT_ROOT / "assets" / "figures"
TABLE_DIR = PROJECT_ROOT / "assets" / "models"
SELECTION_PATH = TABLE_DIR / "reward_calibration_selection.json"

BASELINE_CSV = TABLE_DIR / "reward_breakdown_baselines_directional_primitive.csv"
BASELINE_SUMMARY_TXT = TABLE_DIR / "reward_breakdown_baselines_directional_primitive_summary.txt"
BASELINE_SUMMARY_MD = TABLE_DIR / "reward_breakdown_baselines_directional_primitive_summary.md"
BREAKDOWN_PLOT = FIGURE_DIR / "baseline_reward_breakdown_directional_primitive.png"
HEATMAP_PLOT = FIGURE_DIR / "baseline_heatmap_comparison_directional_primitive.png"
ORDER_PLOT = FIGURE_DIR / "baseline_scan_order_comparison_directional_primitive.png"

SEGMENT_COMPARISON_CSV = TABLE_DIR / "directional_primitive_vs_segment6_comparison.csv"
SEGMENT_COMPARISON_TXT = TABLE_DIR / "directional_primitive_vs_segment6_comparison.txt"
SEGMENT_COMPARISON_MD = TABLE_DIR / "directional_primitive_vs_segment6_comparison.md"
LOCAL_COMPARISON_CSV = TABLE_DIR / "directional_primitive_vs_local_primitive_comparison.csv"
LOCAL_COMPARISON_TXT = TABLE_DIR / "directional_primitive_vs_local_primitive_comparison.txt"
LOCAL_COMPARISON_MD = TABLE_DIR / "directional_primitive_vs_local_primitive_comparison.md"
VERDICT_TXT = TABLE_DIR / "directional_primitive_action_verdict.txt"
VERDICT_MD = TABLE_DIR / "directional_primitive_action_verdict.md"

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


def make_local_primitive_env(mask: np.ndarray, reward_weights: dict[str, float]) -> ScanPlanningLocalPrimitiveEnv:
    """Create the earlier local primitive environment."""
    return ScanPlanningLocalPrimitiveEnv(
        planning_mask=mask,
        grid_size=mask.shape[0],
        reward_weights=reward_weights,
    )


def make_directional_primitive_env(
    mask: np.ndarray,
    reward_weights: dict[str, float],
) -> ScanPlanningDirectionalPrimitiveEnv:
    """Create the final directional primitive environment."""
    return ScanPlanningDirectionalPrimitiveEnv(
        planning_mask=mask,
        grid_size=mask.shape[0],
        reward_weights=reward_weights,
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
    if hasattr(env, "action_masks_catalog"):
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


def choose_directional_action(
    cell: tuple[int, int],
    lookahead_cells: list[tuple[int, int]],
    env: Any,
    covered: set[tuple[int, int]],
) -> int:
    """Choose the best 2-cell directional primitive anchored at one cell."""
    row, col = cell
    best_action: int | None = None
    best_score: tuple[float, float, int] | None = None
    lookahead_set = set(lookahead_cells)
    for direction_index in range(len(env.DIRECTION_NAMES)):
        action_index = env.action_lookup.get((int(row), int(col), int(direction_index)))
        if action_index is None:
            continue
        action_cells = [action_cell for action_cell in env.action_cells[action_index] if action_cell not in covered]
        if not action_cells:
            continue
        lookahead_hits = sum(1 for action_cell in action_cells if action_cell in lookahead_set)
        score = (float(lookahead_hits), float(len(action_cells)), -int(direction_index))
        if best_score is None or score > best_score:
            best_score = score
            best_action = action_index
    if best_action is None:
        raise RuntimeError("No valid directional primitive action found for planner compression.")
    return best_action


def compress_directional_actions(
    cell_plan: list[tuple[int, int]],
    env: Any,
) -> list[int]:
    """Compress a cell-order plan into 2-cell directional primitive actions."""
    actions: list[int] = []
    covered: set[tuple[int, int]] = set()
    for index, cell in enumerate(cell_plan):
        if cell in covered:
            continue
        lookahead_cells = cell_plan[index : index + 2]
        action_index = choose_directional_action(cell, lookahead_cells, env, covered)
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


def build_catalog_rows(environment_name: str, action_cells: list[list[tuple[int, int]]]) -> list[dict[str, Any]]:
    """Summarise action locality for one catalog."""
    sizes = [len(cells) for cells in action_cells if cells]
    return [
        {"environment": environment_name, "metric": "action_catalog_size", "value": float(len(action_cells))},
        {"environment": environment_name, "metric": "affected_cells_mean", "value": float(np.mean(sizes))},
        {"environment": environment_name, "metric": "affected_cells_std", "value": float(np.std(sizes))},
        {"environment": environment_name, "metric": "affected_cells_min", "value": float(np.min(sizes))},
        {"environment": environment_name, "metric": "affected_cells_max", "value": float(np.max(sizes))},
    ]


def main() -> None:
    """Run baseline validation and diagnostics for the directional primitive."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    reward_weights = load_variant1_weights()
    target_mask = build_target_mask()

    cell_plans = planner_cell_plans(target_mask)

    segment_env = make_segment6_env(target_mask, reward_weights)
    segment_env.reset(seed=42)
    segment_mapping = segment_env.cell_to_segment_action.copy()
    segment_action_count = len(segment_env.segment_cells)
    segment_catalog_rows = build_catalog_rows("segment6", segment_env.segment_cells)
    segment_env.close()

    local_env = make_local_primitive_env(target_mask, reward_weights)
    local_env.reset(seed=42)
    local_catalog_rows = build_catalog_rows("local_primitive", local_env.action_cells)
    local_env.close()

    directional_env = make_directional_primitive_env(target_mask, reward_weights)
    directional_env.reset(seed=42)
    directional_catalog_rows = build_catalog_rows("directional_primitive", directional_env.action_cells)

    directional_results: dict[str, dict[str, Any]] = {}
    reward_rows: list[dict[str, Any]] = []
    for planner_name, cell_plan in cell_plans.items():
        action_plan = compress_directional_actions(cell_plan, directional_env)
        result = run_action_plan(make_directional_primitive_env, target_mask, reward_weights, action_plan)
        directional_results[planner_name] = result
        reward_rows.append({"planner": planner_name, **result["reward_breakdown"]})
    directional_env.close()

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
        directional_results,
        HEATMAP_PLOT,
        field_key="final_thermal",
        title="Directional Primitive Baseline Heatmap Comparison",
        cmap="magma",
        colorbar_label="Proxy Thermal Level",
    )
    save_comparison_grid(
        directional_results,
        ORDER_PLOT,
        field_key="order_map",
        title="Directional Primitive Baseline Scan Order Comparison",
        cmap="inferno",
        colorbar_label="Scan Step",
    )

    ranking = sorted(reward_rows, key=lambda row: float(row["total_reward"]), reverse=True)
    summary_lines = [
        "Directional primitive baseline summary",
        "- Ranking by total reward: " + ", ".join(
            f"{row['planner']} ({float(row['total_reward']):.3f})" for row in ranking
        ),
    ]
    write_text_pair(
        BASELINE_SUMMARY_TXT,
        BASELINE_SUMMARY_MD,
        "Directional Primitive Baseline Summary",
        summary_lines,
    )

    segment_signal = reward_signal_share(make_segment6_env, target_mask, reward_weights)
    local_signal = reward_signal_share(make_local_primitive_env, target_mask, reward_weights)
    directional_signal = reward_signal_share(make_directional_primitive_env, target_mask, reward_weights)

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

    local_env = make_local_primitive_env(target_mask, reward_weights)
    local_env.reset(seed=42)
    local_random = run_action_plan(
        make_local_primitive_env,
        target_mask,
        reward_weights,
        compress_directional_actions(cell_plans["random"], local_env),
    )
    local_distance = run_action_plan(
        make_local_primitive_env,
        target_mask,
        reward_weights,
        compress_directional_actions(cell_plans["distance_aware_cool_first"], local_env),
    )
    local_cool = run_action_plan(
        make_local_primitive_env,
        target_mask,
        reward_weights,
        compress_directional_actions(cell_plans["cool_first"], local_env),
    )
    local_env.close()

    directional_env = make_directional_primitive_env(target_mask, reward_weights)
    directional_env.reset(seed=42)
    directional_random = run_action_plan(
        make_directional_primitive_env,
        target_mask,
        reward_weights,
        compress_directional_actions(cell_plans["random"], directional_env),
    )
    directional_distance = run_action_plan(
        make_directional_primitive_env,
        target_mask,
        reward_weights,
        compress_directional_actions(cell_plans["distance_aware_cool_first"], directional_env),
    )
    directional_cool = run_action_plan(
        make_directional_primitive_env,
        target_mask,
        reward_weights,
        compress_directional_actions(cell_plans["cool_first"], directional_env),
    )
    directional_env.close()

    segment_distance_metrics = early_metrics(segment_distance)
    local_distance_metrics = early_metrics(local_distance)
    directional_distance_metrics = early_metrics(directional_distance)

    segment_rows = [
        *segment_catalog_rows,
        *directional_catalog_rows,
        {"environment": "segment6", "metric": "coverage_completion_share", "value": segment_signal["coverage_completion_share"]},
        {"environment": "segment6", "metric": "thermal_share", "value": segment_signal["thermal_share"]},
        {"environment": "segment6", "metric": "jump_share", "value": segment_signal["jump_share"]},
        {"environment": "directional_primitive", "metric": "coverage_completion_share", "value": directional_signal["coverage_completion_share"]},
        {"environment": "directional_primitive", "metric": "thermal_share", "value": directional_signal["thermal_share"]},
        {"environment": "directional_primitive", "metric": "jump_share", "value": directional_signal["jump_share"]},
    ]
    for environment_name, result in (
        ("segment6_random", segment_random),
        ("segment6_distance_aware_cool_first", segment_distance),
        ("segment6_cool_first", segment_cool),
        ("directional_primitive_random", directional_random),
        ("directional_primitive_distance_aware_cool_first", directional_distance),
        ("directional_primitive_cool_first", directional_cool),
    ):
        for metric_name, value in early_metrics(result).items():
            segment_rows.append({"environment": environment_name, "metric": metric_name, "value": value})
    write_csv(SEGMENT_COMPARISON_CSV, ["environment", "metric", "value"], segment_rows)

    local_rows = [
        *local_catalog_rows,
        *directional_catalog_rows,
        {"environment": "local_primitive", "metric": "coverage_completion_share", "value": local_signal["coverage_completion_share"]},
        {"environment": "local_primitive", "metric": "thermal_share", "value": local_signal["thermal_share"]},
        {"environment": "local_primitive", "metric": "jump_share", "value": local_signal["jump_share"]},
        {"environment": "directional_primitive", "metric": "coverage_completion_share", "value": directional_signal["coverage_completion_share"]},
        {"environment": "directional_primitive", "metric": "thermal_share", "value": directional_signal["thermal_share"]},
        {"environment": "directional_primitive", "metric": "jump_share", "value": directional_signal["jump_share"]},
    ]
    for environment_name, result in (
        ("local_primitive_random", local_random),
        ("local_primitive_distance_aware_cool_first", local_distance),
        ("local_primitive_cool_first", local_cool),
        ("directional_primitive_random", directional_random),
        ("directional_primitive_distance_aware_cool_first", directional_distance),
        ("directional_primitive_cool_first", directional_cool),
    ):
        for metric_name, value in early_metrics(result).items():
            local_rows.append({"environment": environment_name, "metric": metric_name, "value": value})
    write_csv(LOCAL_COMPARISON_CSV, ["environment", "metric", "value"], local_rows)

    segment_mean = next(row["value"] for row in segment_catalog_rows if row["metric"] == "affected_cells_mean")
    local_mean = next(row["value"] for row in local_catalog_rows if row["metric"] == "affected_cells_mean")
    directional_mean = next(row["value"] for row in directional_catalog_rows if row["metric"] == "affected_cells_mean")
    segment_catalog_size = next(row["value"] for row in segment_catalog_rows if row["metric"] == "action_catalog_size")
    local_catalog_size = next(row["value"] for row in local_catalog_rows if row["metric"] == "action_catalog_size")
    directional_catalog_size = next(row["value"] for row in directional_catalog_rows if row["metric"] == "action_catalog_size")
    distance_top = ranking[0]["planner"]
    raster_bottom = ranking[-1]["planner"]

    segment_lines = [
        f"- Segment6 locality mean={float(segment_mean):.2f}, directional locality mean={float(directional_mean):.2f}",
        f"- Segment6 thermal share={segment_signal['thermal_share']:.3f}, directional thermal share={directional_signal['thermal_share']:.3f}",
        f"- Segment6 action catalog size={int(segment_catalog_size)}, directional action catalog size={int(directional_catalog_size)}",
        f"- Segment6 distance-aware early adjacency={segment_distance_metrics['early_adjacency_ratio']:.3f}, directional distance-aware early adjacency={directional_distance_metrics['early_adjacency_ratio']:.3f}",
        f"- Segment6 distance-aware cumulative early reheat={segment_distance_metrics['cumulative_early_reheat']:.3f}, directional distance-aware cumulative early reheat={directional_distance_metrics['cumulative_early_reheat']:.3f}",
        f"- Directional ranking top={distance_top}, bottom={raster_bottom}",
    ]
    write_text_pair(
        SEGMENT_COMPARISON_TXT,
        SEGMENT_COMPARISON_MD,
        "Directional Primitive vs Segment6 Comparison",
        segment_lines,
    )

    local_lines = [
        f"- Local-primitive locality mean={float(local_mean):.2f}, directional locality mean={float(directional_mean):.2f}",
        f"- Local-primitive thermal share={local_signal['thermal_share']:.3f}, directional thermal share={directional_signal['thermal_share']:.3f}",
        f"- Local-primitive action catalog size={int(local_catalog_size)}, directional action catalog size={int(directional_catalog_size)}",
        f"- Local-primitive distance-aware early adjacency={local_distance_metrics['early_adjacency_ratio']:.3f}, directional distance-aware early adjacency={directional_distance_metrics['early_adjacency_ratio']:.3f}",
        f"- Local-primitive distance-aware cumulative early reheat={local_distance_metrics['cumulative_early_reheat']:.3f}, directional distance-aware cumulative early reheat={directional_distance_metrics['cumulative_early_reheat']:.3f}",
    ]
    write_text_pair(
        LOCAL_COMPARISON_TXT,
        LOCAL_COMPARISON_MD,
        "Directional Primitive vs Local Primitive Comparison",
        local_lines,
    )

    more_local_than_segment = float(directional_mean) < float(segment_mean)
    thermal_not_weaker_than_segment = directional_signal["thermal_share"] >= segment_signal["thermal_share"]
    thermal_stronger_than_local = directional_signal["thermal_share"] > local_signal["thermal_share"]
    sensible_ranking = distance_top == "distance_aware_cool_first" and raster_bottom == "raster"
    distance_structured = directional_distance_metrics["early_adjacency_ratio"] <= 0.10
    better_candidate = (
        more_local_than_segment
        and thermal_not_weaker_than_segment
        and thermal_stronger_than_local
        and sensible_ranking
        and distance_structured
    )

    verdict_lines = [
        f"1. Is directional primitive more local than fixed segment-6? {'YES' if more_local_than_segment else 'NO'}",
        f"2. Is thermal signal share stronger or at least not weaker than fixed segment-6? {'YES' if thermal_not_weaker_than_segment else 'NO'}",
        f"3. Is thermal signal share stronger than the previous local-primitive? {'YES' if thermal_stronger_than_local else 'NO'}",
        f"4. Does baseline ranking remain sensible? {'YES' if sensible_ranking else 'NO'}",
        f"5. Does distance_aware_cool_first remain low-clustering and structured? {'YES' if distance_structured else 'NO'}",
        f"6. Is this genuinely a better PPO candidate than fixed segment-6? {'YES' if better_candidate else 'NO'}",
        "",
        (
            f"Most important quantitative improvement: locality mean {float(segment_mean):.2f} -> {float(directional_mean):.2f} "
            f"with max affected cells {int(next(row['value'] for row in directional_catalog_rows if row['metric'] == 'affected_cells_max'))}."
        ),
        (
            f"Most important remaining weakness: thermal share segment6/local/directional = "
            f"{segment_signal['thermal_share']:.3f} / {local_signal['thermal_share']:.3f} / {directional_signal['thermal_share']:.3f}, "
            f"while distance-aware early adjacency = {segment_distance_metrics['early_adjacency_ratio']:.3f} / "
            f"{local_distance_metrics['early_adjacency_ratio']:.3f} / {directional_distance_metrics['early_adjacency_ratio']:.3f}."
        ),
        "",
        (
            "GO: directional primitive is the final patch-based candidate worth testing with PPO"
            if better_candidate
            else "NO-GO: directional primitive does not improve the representation enough; patch-based action family should now be deprioritized"
        ),
    ]
    write_text_pair(VERDICT_TXT, VERDICT_MD, "Directional Primitive Action Verdict", verdict_lines)

    print("Directional primitive diagnostics complete.")
    print(f"Saved baseline CSV to: {BASELINE_CSV}")
    print(f"Saved segment comparison CSV to: {SEGMENT_COMPARISON_CSV}")
    print(f"Saved local comparison CSV to: {LOCAL_COMPARISON_CSV}")
    print(f"Saved verdict to: {VERDICT_TXT}")


if __name__ == "__main__":
    main()
