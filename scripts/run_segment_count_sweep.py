"""Run a minimal 4/6/8 segment-count sweep using baselines and focused diagnostics only."""

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
from core.viz import save_comparison_grid, save_reward_breakdown_chart
from rl.env_scan import ScanPlanningEnv
from rl.env_scan_segment import ScanPlanningSegmentEnv


TABLE_DIR = PROJECT_ROOT / "assets" / "models"
FIGURE_DIR = PROJECT_ROOT / "assets" / "figures"
SELECTION_PATH = TABLE_DIR / "reward_calibration_selection.json"

SEGMENT_COUNTS = (4, 6, 8)
EPISODES = 20
EARLY_FRACTION = 0.30
ADJACENCY_RADIUS = 2.0
ADJACENCY_HISTORY_WINDOW = 5

COMPARISON_CSV = TABLE_DIR / "action_granularity_sweep_comparison.csv"
COMPARISON_TXT = TABLE_DIR / "action_granularity_sweep_comparison.txt"
COMPARISON_MD = TABLE_DIR / "action_granularity_sweep_comparison.md"
VERDICT_TXT = TABLE_DIR / "segment_count_sweep_verdict.txt"
VERDICT_MD = TABLE_DIR / "segment_count_sweep_verdict.md"


def load_variant1_weights() -> dict[str, float]:
    """Load the current calibrated reward weights."""
    payload = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    return {key: float(value) for key, value in payload["variants"]["variant_1"].items()}


def build_target_mask() -> np.ndarray:
    """Build the standard TWI mask."""
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


def make_segment_env(mask: np.ndarray, reward_weights: dict[str, float], segments_per_stripe: int) -> ScanPlanningSegmentEnv:
    """Create a segment environment with the requested segment count."""
    return ScanPlanningSegmentEnv(
        planning_mask=mask,
        grid_size=mask.shape[0],
        reward_weights=reward_weights,
        segments_per_stripe=segments_per_stripe,
    )


def make_stripe_env(mask: np.ndarray, reward_weights: dict[str, float]) -> ScanPlanningEnv:
    """Create the original stripe environment."""
    return ScanPlanningEnv(planning_mask=mask, grid_size=mask.shape[0], reward_weights=reward_weights)


def build_cell_to_stripe_action(env: ScanPlanningEnv) -> dict[tuple[int, int], int]:
    """Map target cells to stripe actions."""
    mapping: dict[tuple[int, int], int] = {}
    for stripe_index, stripe in enumerate(env.stripes):
        for cell in env._stripe_cells(stripe):
            mapping[cell] = stripe_index
    return mapping


def compress_cell_plan(cell_plan: list[tuple[int, int]], cell_to_action: dict[tuple[int, int], int]) -> list[int]:
    """Compress a cell ordering to a unique environment action sequence."""
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
    """Return a stable center for one stripe/segment action."""
    if isinstance(env, ScanPlanningSegmentEnv):
        mask = env.segment_masks[int(action)]
    else:
        mask = env.stripes[int(action)]
    remaining = np.logical_and(mask, env.target_mask & ~env.scanned_mask)
    rows, cols = np.nonzero(remaining if remaining.any() else mask)
    if len(rows) == 0:
        return None
    return int(np.rint(rows.mean())), int(np.rint(cols.mean()))


def run_env_action_plan(env: Any, mask: np.ndarray, action_plan: list[int]) -> dict[str, Any]:
    """Execute one action plan inside a given environment instance."""
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
    step_stats: list[dict[str, float]] = []

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
        "action_centers": action_centers,
        "step_stats": step_stats,
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
    """Return the exclusive early-phase end index."""
    return max(1, math.ceil(length * EARLY_FRACTION))


def adjacency_ratio(points: list[tuple[int, int]]) -> float:
    """Fraction of actions that stay near the recent history window."""
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
    """Compute early clustering metrics from one rollout result."""
    centers = result["action_centers"]
    reward_history = result["reward_history"]
    step_stats = result["step_stats"]
    early_stop = phase_end(len(centers))
    return {
        "early_adjacency_ratio": adjacency_ratio(centers[:early_stop]),
        "cumulative_early_reheat": float(sum(float(step["reheat"]) for step in reward_history[:early_stop])),
        "early_peak_heat": float(np.mean([float(step["peak_heat"]) for step in step_stats[:early_stop]])) if early_stop > 0 else 0.0,
        "early_heat_variance": float(np.mean([float(step["heat_variance"]) for step in step_stats[:early_stop]])) if early_stop > 0 else 0.0,
    }


def reward_signal_share(env_factory: Any, mask: np.ndarray, reward_weights: dict[str, float], segments_per_stripe: int | None = None) -> dict[str, float]:
    """Estimate reward term shares under masked-random exploratory rollouts."""
    term_names = ["coverage", "completion_bonus", "invalid", "peak", "variance", "reheat", "jump"]
    abs_totals = {term: 0.0 for term in term_names}
    total_abs = 0.0

    for episode_id in range(EPISODES):
        if segments_per_stripe is None:
            env = env_factory(mask, reward_weights)
        else:
            env = env_factory(mask, reward_weights, segments_per_stripe)
        env.reset(seed=800 + episode_id)
        rng = np.random.default_rng(80_000 + episode_id)
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


def run_segment_baselines_for_count(mask: np.ndarray, reward_weights: dict[str, float], segment_count: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the stronger baseline suite for one segment count and save outputs."""
    mapping_env = make_segment_env(mask, reward_weights, segment_count)
    mapping_env.reset(seed=42)
    cell_to_action = mapping_env.cell_to_segment_action.copy()
    segment_sizes = [len(cells) for cells in mapping_env.segment_cells if cells]
    mapping_env.close()

    results: dict[str, dict[str, Any]] = {}
    reward_rows: list[dict[str, Any]] = []
    early_metrics_map: dict[str, dict[str, float]] = {}
    for planner_name, cell_plan in planner_cell_plans(mask).items():
        action_plan = compress_cell_plan(cell_plan, cell_to_action)
        result = run_env_action_plan(
            make_segment_env(mask, reward_weights, segment_count),
            mask,
            action_plan,
        )
        results[planner_name] = result
        early_metrics_map[planner_name] = early_metrics(result)
        reward_breakdown = result["reward_breakdown"].copy()
        reward_breakdown["planner"] = planner_name
        reward_rows.append(reward_breakdown)

    csv_path = TABLE_DIR / f"reward_breakdown_baselines_segment{segment_count}.csv"
    summary_path = TABLE_DIR / f"reward_breakdown_baselines_segment{segment_count}_summary.txt"
    breakdown_plot = FIGURE_DIR / f"baseline_reward_breakdown_segment{segment_count}.png"
    heatmap_plot = FIGURE_DIR / f"baseline_heatmap_comparison_segment{segment_count}.png"
    order_plot = FIGURE_DIR / f"baseline_scan_order_comparison_segment{segment_count}.png"

    write_csv(
        csv_path,
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
    save_reward_breakdown_chart(reward_rows, breakdown_plot)
    save_comparison_grid(
        results,
        heatmap_plot,
        field_key="final_thermal",
        title=f"Segment={segment_count} Heatmap Comparison",
        cmap="magma",
        colorbar_label="Proxy Thermal Level",
    )
    save_comparison_grid(
        results,
        order_plot,
        field_key="order_map",
        title=f"Segment={segment_count} Scan Order Comparison",
        cmap="inferno",
        colorbar_label="Scan Step",
    )

    ranking = sorted(reward_rows, key=lambda row: float(row["total_reward"]), reverse=True)
    summary_lines = [
        f"Segment {segment_count} baseline summary",
        (
            "- Ranking by total reward: "
            + ", ".join(f"{row['planner']} ({float(row['total_reward']):.3f})" for row in ranking)
        ),
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return reward_rows, {
        "segment_count": segment_count,
        "segment_sizes": segment_sizes,
        "ranking": ranking,
        "early_metrics": early_metrics_map,
        "csv_path": str(csv_path),
        "breakdown_plot": str(breakdown_plot),
        "heatmap_plot": str(heatmap_plot),
        "order_plot": str(order_plot),
        "summary_path": str(summary_path),
    }


def stripe_reference(mask: np.ndarray, reward_weights: dict[str, float]) -> dict[str, Any]:
    """Compute the stripe reference metrics for the sweep comparison."""
    env = make_stripe_env(mask, reward_weights)
    env.reset(seed=42)
    stripe_sizes = [len(env._stripe_cells(stripe)) for stripe in env.stripes]
    stripe_mapping = build_cell_to_stripe_action(env)
    env.close()

    signal = reward_signal_share(make_stripe_env, mask, reward_weights)
    plans = planner_cell_plans(mask)
    clustering = {}
    for planner_name in ("random", "distance_aware_cool_first", "cool_first"):
        result = run_env_action_plan(
            make_stripe_env(mask, reward_weights),
            mask,
            compress_cell_plan(plans[planner_name], stripe_mapping),
        )
        clustering[planner_name] = early_metrics(result)

    return {
        "segment_count": "stripe",
        "sizes": stripe_sizes,
        "signal": signal,
        "clustering": clustering,
    }


def choose_best_candidate(stripe_ref: dict[str, Any], segment_runs: list[dict[str, Any]]) -> tuple[str | None, list[str]]:
    """Choose the best segment count for the next PPO smoke test, or return no-go."""
    notes: list[str] = []
    candidates: list[tuple[float, int]] = []
    stripe_thermal = float(stripe_ref["signal"]["thermal_share"])

    six_run = next(run for run in segment_runs if run["segment_count"] == 6)
    six_distance = six_run["early_metrics"]["distance_aware_cool_first"]["early_adjacency_ratio"]

    for run in segment_runs:
        count = int(run["segment_count"])
        ranking = [row["planner"] for row in run["ranking"]]
        top3 = ranking[:3]
        bottom2 = ranking[-2:]
        sensible = "distance_aware_cool_first" in top3 and "raster" in bottom2
        mean_cells = float(np.mean(run["segment_sizes"]))
        improved_locality = mean_cells < float(np.mean(stripe_ref["sizes"]))
        thermal_not_weakened = float(run["signal"]["thermal_share"]) >= stripe_thermal
        early_not_worse_than_six = (
            run["early_metrics"]["distance_aware_cool_first"]["early_adjacency_ratio"] <= six_distance + 1e-9
        )
        gap = float(run["ranking"][0]["total_reward"]) - float(
            next(row["total_reward"] for row in run["ranking"] if row["planner"] == "random")
        )
        notes.append(
            f"- segment={count}: locality={mean_cells:.2f}, thermal_share={run['signal']['thermal_share']:.3f}, "
            f"sensible={sensible}, early_not_worse_than_6={early_not_worse_than_six}, DA-random gap={gap:.3f}"
        )
        if sensible and improved_locality and thermal_not_weakened and early_not_worse_than_six:
            # Favor a balance of thermal signal and stable structure over smallest cell count alone.
            score = float(run["signal"]["thermal_share"]) + 0.25 * gap - 0.02 * mean_cells
            candidates.append((score, count))

    if not candidates:
        return None, notes
    candidates.sort(reverse=True)
    return str(candidates[0][1]), notes


def main() -> None:
    """Run the 4/6/8 segment-count sweep and select the next PPO candidate or a no-go."""
    reward_weights = load_variant1_weights()
    target_mask = build_target_mask()
    stripe_ref = stripe_reference(target_mask, reward_weights)

    sweep_rows: list[dict[str, Any]] = [
        {"setting": "stripe", "metric": "affected_cells_mean", "value": float(np.mean(stripe_ref["sizes"]))},
        {"setting": "stripe", "metric": "affected_cells_std", "value": float(np.std(stripe_ref["sizes"]))},
        {"setting": "stripe", "metric": "affected_cells_min", "value": int(np.min(stripe_ref["sizes"]))},
        {"setting": "stripe", "metric": "affected_cells_max", "value": int(np.max(stripe_ref["sizes"]))},
        {"setting": "stripe", "metric": "coverage_completion_share", "value": stripe_ref["signal"]["coverage_completion_share"]},
        {"setting": "stripe", "metric": "thermal_share", "value": stripe_ref["signal"]["thermal_share"]},
        {"setting": "stripe", "metric": "jump_share", "value": stripe_ref["signal"]["jump_share"]},
    ]
    for planner_name, metrics in stripe_ref["clustering"].items():
        for metric_name, value in metrics.items():
            sweep_rows.append(
                {"setting": "stripe", "metric": f"{planner_name}_{metric_name}", "value": value}
            )

    segment_runs: list[dict[str, Any]] = []
    for segment_count in SEGMENT_COUNTS:
        reward_rows, metadata = run_segment_baselines_for_count(target_mask, reward_weights, segment_count)
        signal = reward_signal_share(make_segment_env, target_mask, reward_weights, segments_per_stripe=segment_count)
        metadata["signal"] = signal
        segment_runs.append(metadata)

        sweep_rows.extend(
            [
                {"setting": f"segment{segment_count}", "metric": "affected_cells_mean", "value": float(np.mean(metadata["segment_sizes"]))},
                {"setting": f"segment{segment_count}", "metric": "affected_cells_std", "value": float(np.std(metadata["segment_sizes"]))},
                {"setting": f"segment{segment_count}", "metric": "affected_cells_min", "value": int(np.min(metadata["segment_sizes"]))},
                {"setting": f"segment{segment_count}", "metric": "affected_cells_max", "value": int(np.max(metadata["segment_sizes"]))},
                {"setting": f"segment{segment_count}", "metric": "coverage_completion_share", "value": signal["coverage_completion_share"]},
                {"setting": f"segment{segment_count}", "metric": "thermal_share", "value": signal["thermal_share"]},
                {"setting": f"segment{segment_count}", "metric": "jump_share", "value": signal["jump_share"]},
            ]
        )
        for planner_name in ("random", "distance_aware_cool_first", "cool_first"):
            metrics = metadata["early_metrics"][planner_name]
            for metric_name, value in metrics.items():
                sweep_rows.append(
                    {"setting": f"segment{segment_count}", "metric": f"{planner_name}_{metric_name}", "value": value}
                )

    write_csv(COMPARISON_CSV, ["setting", "metric", "value"], sweep_rows)

    best_count, notes = choose_best_candidate(stripe_ref, segment_runs)
    summary_lines = [
        f"- Stripe action locality mean={np.mean(stripe_ref['sizes']):.2f}, thermal_share={stripe_ref['signal']['thermal_share']:.3f}",
        *notes,
    ]
    write_text_pair(COMPARISON_TXT, COMPARISON_MD, "Action Granularity Sweep Comparison", summary_lines)

    if best_count is None:
        verdict_lines = [
            "1. No single segment count clearly dominates the 4/6/8 sweep across locality, thermal signal, and baseline structure.",
            "2. Thermal signal differences across 4/6/8 are too small or the task becomes too fragmented/noisy to justify PPO next.",
            "3. Recommendation: no-go for PPO at this stage.",
            "Single next action: revisit action representation beyond segment count.",
        ]
    else:
        best_run = next(run for run in segment_runs if str(run["segment_count"]) == best_count)
        verdict_lines = [
            f"1. Best overall trade-off: segment={best_count}.",
            (
                f"2. Thermal signal comparison: stripe={stripe_ref['signal']['thermal_share']:.3f}, "
                + ", ".join(
                    f"{run['segment_count']}={run['signal']['thermal_share']:.3f}" for run in segment_runs
                )
                + "."
            ),
            (
                "3. Early clustering comparison (distance-aware): "
                + ", ".join(
                    f"{run['segment_count']}={run['early_metrics']['distance_aware_cool_first']['early_adjacency_ratio']:.3f}"
                    for run in segment_runs
                )
                + "."
            ),
            f"4. Next PPO smoke test should use segment={best_count}.",
            "Single next action: run the next PPO smoke test with this segment count.",
        ]

    write_text_pair(VERDICT_TXT, VERDICT_MD, "Segment Count Sweep Verdict", verdict_lines)
    print(f"Saved sweep comparison CSV to: {COMPARISON_CSV}")
    print(f"Saved sweep verdict to: {VERDICT_TXT}")


if __name__ == "__main__":
    main()
