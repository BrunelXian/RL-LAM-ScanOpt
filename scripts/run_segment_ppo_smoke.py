"""Run a strict PPO smoke test in the validated segment=6 environment."""

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
from core.planners.cool_first import plan_cool_first
from core.planners.distance_aware_cool_first import plan_distance_aware_cool_first
from core.planners.random_planner import plan_random
from core.viz import (
    save_comparison_grid,
    save_reward_breakdown_chart,
    save_scan_path_gif,
    save_training_curves_figure,
)
from rl.env_scan_segment import ScanPlanningSegmentEnv
from rl.train_maskable_ppo import (
    BATCH_SIZE,
    ENT_COEF,
    GAMMA,
    LEARNING_RATE,
    N_STEPS,
    SEED,
    TrainingMetricsCallback,
)


TABLE_DIR = PROJECT_ROOT / "assets" / "models"
FIGURE_DIR = PROJECT_ROOT / "assets" / "figures"
SELECTION_PATH = TABLE_DIR / "reward_calibration_selection.json"
VERDICT_PATH = TABLE_DIR / "segment_count_sweep_verdict.txt"

MODEL_PATH = TABLE_DIR / "maskable_ppo_smoke_segment6.zip"
HISTORY_PATH = TABLE_DIR / "maskable_ppo_smoke_segment6_history.json"
CURVE_PATH = FIGURE_DIR / "maskable_ppo_smoke_segment6_training_curve.png"

BREAKDOWN_CSV = TABLE_DIR / "ppo_smoke_segment6_reward_breakdown.csv"
BREAKDOWN_PLOT = FIGURE_DIR / "ppo_smoke_segment6_reward_breakdown.png"
HEATMAP_PLOT = FIGURE_DIR / "ppo_smoke_segment6_heatmap_comparison.png"
ORDER_PLOT = FIGURE_DIR / "ppo_smoke_segment6_scan_order_comparison.png"
GIF_PATH = FIGURE_DIR / "ppo_smoke_segment6_scan_path.gif"

PPO_VS_BASELINES_CSV = TABLE_DIR / "ppo_vs_baselines_segment6.csv"
PPO_VS_BASELINES_TXT = TABLE_DIR / "ppo_vs_baselines_segment6.txt"
VERDICT_TXT = TABLE_DIR / "ppo_smoke_segment6_verdict.txt"
VERDICT_MD = TABLE_DIR / "ppo_smoke_segment6_verdict.md"

TIMESTEPS = 10_240
SEGMENTS_PER_STRIPE = 6
EARLY_FRACTION = 0.30
ADJACENCY_RADIUS = 2.0
ADJACENCY_HISTORY_WINDOW = 5


def load_variant1_weights() -> dict[str, float]:
    """Load the selected calibrated reward setting."""
    payload = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    return {key: float(value) for key, value in payload["variants"]["variant_1"].items()}


def build_target_mask() -> np.ndarray:
    """Build the standard TWI mask."""
    return downsample_mask(generate_text_mask("TWI", canvas_size=1024), grid_size=64, threshold=0.2)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    """Write CSV rows to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def write_text(path: Path, lines: list[str]) -> None:
    """Write a plain-text report."""
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_text_pair(txt_path: Path, md_path: Path, title: str, lines: list[str]) -> None:
    """Write matching TXT and Markdown reports."""
    write_text(txt_path, lines)
    md_path.write_text("\n".join([f"# {title}", ""] + lines) + "\n", encoding="utf-8")


def ensure_segment6_selected() -> None:
    """Stop immediately if the last segment sweep did not endorse segment=6."""
    verdict = VERDICT_PATH.read_text(encoding="utf-8")
    if "segment=6" not in verdict:
        raise RuntimeError("Segment-count sweep did not select segment=6. Do not run this PPO smoke test.")


def _mask_fn(env: object) -> object:
    return env.unwrapped.action_masks()


def make_training_env(mask: np.ndarray, reward_weights: dict[str, float]) -> object:
    """Create a wrapped training environment for Maskable PPO."""
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    def _factory() -> object:
        env = ScanPlanningSegmentEnv(
            planning_mask=mask,
            grid_size=mask.shape[0],
            reward_weights=reward_weights,
            segments_per_stripe=SEGMENTS_PER_STRIPE,
        )
        return ActionMasker(Monitor(env), _mask_fn)

    return DummyVecEnv([_factory])


def make_eval_env(mask: np.ndarray, reward_weights: dict[str, float]) -> object:
    """Create a wrapped deterministic evaluation environment."""
    from sb3_contrib.common.wrappers import ActionMasker

    env = ScanPlanningSegmentEnv(
        planning_mask=mask,
        grid_size=mask.shape[0],
        reward_weights=reward_weights,
        segments_per_stripe=SEGMENTS_PER_STRIPE,
    )
    return ActionMasker(env, _mask_fn)


def get_action_center(env: ScanPlanningSegmentEnv, action: int) -> tuple[int, int] | None:
    """Return the center of the currently executable portion of an action."""
    mask = env.segment_masks[int(action)]
    remaining = np.logical_and(mask, env.target_mask & ~env.scanned_mask)
    rows, cols = np.nonzero(remaining if remaining.any() else mask)
    if len(rows) == 0:
        return None
    return int(np.rint(rows.mean())), int(np.rint(cols.mean()))


def compress_cell_plan(
    cell_plan: list[tuple[int, int]],
    cell_to_action: dict[tuple[int, int], int],
) -> list[int]:
    """Compress a cell-order plan into unique segment actions."""
    actions: list[int] = []
    seen: set[int] = set()
    for cell in cell_plan:
        action = cell_to_action[cell]
        if action in seen:
            continue
        actions.append(action)
        seen.add(action)
    return actions


def phase_end(length: int) -> int:
    """Return the exclusive index for the early phase."""
    return max(1, math.ceil(length * EARLY_FRACTION))


def adjacency_ratio(points: list[tuple[int, int]]) -> float:
    """Compute local clustering against the recent action history window."""
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
    """Compute early-phase clustering and thermal diagnostics."""
    centers = result["action_centers"]
    reward_history = result["reward_history"]
    thermal_history = result["thermal_history"]
    target_mask = result["target_mask"]
    early_stop = phase_end(len(centers))
    if early_stop <= 0:
        return {
            "early_adjacency_ratio": 0.0,
            "cumulative_early_reheat": 0.0,
            "early_peak_heat": 0.0,
        }
    early_peak = 0.0
    if thermal_history:
        early_index = min(max(early_stop - 1, 0), len(thermal_history) - 1)
        early_field = thermal_history[early_index]
        target_values = early_field[target_mask]
        if target_values.size:
            early_peak = float(target_values.max())
    return {
        "early_adjacency_ratio": adjacency_ratio(centers[:early_stop]),
        "cumulative_early_reheat": float(sum(float(step["reheat"]) for step in reward_history[:early_stop])),
        "early_peak_heat": early_peak,
    }


def run_segment_action_plan(
    mask: np.ndarray,
    reward_weights: dict[str, float],
    action_plan: list[int],
) -> dict[str, Any]:
    """Execute a segment-action sequence and return baseline-style outputs."""
    env = ScanPlanningSegmentEnv(
        planning_mask=mask,
        grid_size=mask.shape[0],
        reward_weights=reward_weights,
        segments_per_stripe=SEGMENTS_PER_STRIPE,
    )
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
    step_thermal_stats: list[dict[str, float]] = []
    scanned_history: list[np.ndarray] = []
    thermal_history: list[np.ndarray] = []

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
        scanned_history.append(env.scanned_mask.copy())
        thermal_history.append(env.thermal_field.copy())
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
        "scanned_history": scanned_history,
        "thermal_history": thermal_history,
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


def evaluate_segment_ppo(
    model: object,
    mask: np.ndarray,
    reward_weights: dict[str, float],
) -> tuple[dict[str, Any], dict[str, float]]:
    """Run one deterministic masked PPO rollout and replay it for exact baseline-style outputs."""
    from sb3_contrib.common.maskable.utils import get_action_masks

    env = make_eval_env(mask, reward_weights)
    obs, _ = env.reset(seed=SEED)
    terminated = False
    truncated = False
    action_centers: list[tuple[int, int]] = []
    final_info: dict[str, Any] = {}

    while not (terminated or truncated):
        action_masks = get_action_masks(env)
        action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
        center = get_action_center(env.unwrapped, int(action))
        if center is not None:
            action_centers.append(center)
        obs, _, terminated, truncated, final_info = env.step(int(action))

    replay_actions = [int(action) for action in env.unwrapped.executed_segments]
    replay_result = run_segment_action_plan(mask.copy(), reward_weights, replay_actions)
    replay_result["action_centers"] = action_centers
    eval_stats = {
        "coverage_ratio": float(final_info.get("coverage_ratio", 0.0)),
        "invalid_action_rate": float(final_info.get("invalid_action_count", 0)) / max(float(final_info.get("steps_taken", 1)), 1.0),
        "steps_taken": float(final_info.get("steps_taken", 0.0)),
    }
    env.close()
    return replay_result, eval_stats


def main() -> None:
    """Run the strict segment=6 PPO smoke-test gate."""
    ensure_segment6_selected()
    reward_weights = load_variant1_weights()
    target_mask = build_target_mask()

    try:
        import torch
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.callbacks import ConvertCallback
    except ImportError as exc:
        raise ImportError("Maskable PPO dependencies are missing.") from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = make_training_env(target_mask, reward_weights)
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
        device=device,
    )
    model.learn(
        total_timesteps=TIMESTEPS,
        progress_bar=False,
        log_interval=1,
        callback=ConvertCallback(metrics_callback),
    )
    model.save(str(MODEL_PATH))
    HISTORY_PATH.write_text(json.dumps(metrics_callback.history, indent=2), encoding="utf-8")
    save_training_curves_figure(metrics_callback.history, CURVE_PATH)
    env.close()

    ppo_result, ppo_eval_stats = evaluate_segment_ppo(model, target_mask, reward_weights)

    mapping_env = ScanPlanningSegmentEnv(
        planning_mask=target_mask,
        grid_size=target_mask.shape[0],
        reward_weights=reward_weights,
        segments_per_stripe=SEGMENTS_PER_STRIPE,
    )
    mapping_env.reset(seed=42)
    cell_to_action = mapping_env.cell_to_segment_action.copy()
    mapping_env.close()

    comparison_results = {
        "random_segment6": run_segment_action_plan(
            target_mask,
            reward_weights,
            compress_cell_plan(plan_random(target_mask, seed=42), cell_to_action),
        ),
        "cool_first_segment6": run_segment_action_plan(
            target_mask,
            reward_weights,
            compress_cell_plan(plan_cool_first(target_mask), cell_to_action),
        ),
        "distance_aware_cool_first_segment6": run_segment_action_plan(
            target_mask,
            reward_weights,
            compress_cell_plan(plan_distance_aware_cool_first(target_mask), cell_to_action),
        ),
        "ppo_segment6": ppo_result,
    }

    reward_rows: list[dict[str, Any]] = []
    for planner_name, result in comparison_results.items():
        reward_rows.append({"planner": planner_name, **result["reward_breakdown"]})

    write_csv(
        BREAKDOWN_CSV,
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
        comparison_results,
        HEATMAP_PLOT,
        field_key="final_thermal",
        title="PPO Smoke Segment6 Heatmap Comparison",
        cmap="magma",
        colorbar_label="Proxy Thermal Level",
    )
    save_comparison_grid(
        comparison_results,
        ORDER_PLOT,
        field_key="order_map",
        title="PPO Smoke Segment6 Scan Order Comparison",
        cmap="inferno",
        colorbar_label="Scan Step",
    )
    save_scan_path_gif(
        target_mask=target_mask,
        scanned_history=ppo_result["scanned_history"] if ppo_result["scanned_history"] else [ppo_result["scanned_mask"]],
        path=GIF_PATH,
        title="PPO Smoke Segment6 Scan Path",
    )

    random_result = comparison_results["random_segment6"]
    distance_result = comparison_results["distance_aware_cool_first_segment6"]
    ppo_breakdown = comparison_results["ppo_segment6"]["reward_breakdown"]

    ppo_early = early_metrics(comparison_results["ppo_segment6"])
    random_early = early_metrics(random_result)
    distance_early = early_metrics(distance_result)

    comparison_rows = [
        {
            "policy": "ppo_segment6",
            "total_reward": float(ppo_breakdown["total_reward"]),
            "coverage_ratio": float(ppo_breakdown["coverage_ratio"]),
            "invalid_action_rate": float(ppo_eval_stats["invalid_action_rate"]),
            "reheat": float(ppo_breakdown["reheat"]),
            "peak": float(ppo_breakdown["peak"]),
            "variance": float(ppo_breakdown["variance"]),
            "jump": float(ppo_breakdown["jump"]),
            **ppo_early,
        },
        {
            "policy": "random_segment6",
            "total_reward": float(random_result["reward_breakdown"]["total_reward"]),
            "coverage_ratio": float(random_result["reward_breakdown"]["coverage_ratio"]),
            "invalid_action_rate": 0.0,
            "reheat": float(random_result["reward_breakdown"]["reheat"]),
            "peak": float(random_result["reward_breakdown"]["peak"]),
            "variance": float(random_result["reward_breakdown"]["variance"]),
            "jump": float(random_result["reward_breakdown"]["jump"]),
            **random_early,
        },
        {
            "policy": "distance_aware_cool_first_segment6",
            "total_reward": float(distance_result["reward_breakdown"]["total_reward"]),
            "coverage_ratio": float(distance_result["reward_breakdown"]["coverage_ratio"]),
            "invalid_action_rate": 0.0,
            "reheat": float(distance_result["reward_breakdown"]["reheat"]),
            "peak": float(distance_result["reward_breakdown"]["peak"]),
            "variance": float(distance_result["reward_breakdown"]["variance"]),
            "jump": float(distance_result["reward_breakdown"]["jump"]),
            **distance_early,
        },
    ]
    write_csv(
        PPO_VS_BASELINES_CSV,
        [
            "policy",
            "total_reward",
            "coverage_ratio",
            "invalid_action_rate",
            "reheat",
            "peak",
            "variance",
            "jump",
            "early_adjacency_ratio",
            "cumulative_early_reheat",
            "early_peak_heat",
        ],
        comparison_rows,
    )

    random_total = float(random_result["reward_breakdown"]["total_reward"])
    distance_total = float(distance_result["reward_breakdown"]["total_reward"])
    ppo_total = float(ppo_breakdown["total_reward"])
    ppo_invalid = float(ppo_eval_stats["invalid_action_rate"])
    ppo_coverage = float(ppo_breakdown["coverage_ratio"])
    ppo_early_adj = float(ppo_early["early_adjacency_ratio"])
    random_early_adj = float(random_early["early_adjacency_ratio"])

    reheat_improvement = (abs(float(random_result["reward_breakdown"]["reheat"])) - abs(float(ppo_breakdown["reheat"]))) / max(abs(float(random_result["reward_breakdown"]["reheat"])), 1e-9)
    peak_improvement = (abs(float(random_result["reward_breakdown"]["peak"])) - abs(float(ppo_breakdown["peak"]))) / max(abs(float(random_result["reward_breakdown"]["peak"])), 1e-9)
    reward_improvement = (ppo_total - random_total) / max(abs(random_total), 1e-9)

    passed_coverage = ppo_coverage >= 0.999
    passed_invalid = ppo_invalid < 0.01
    passed_early_adj = ppo_early_adj < 0.5
    passed_reheat = reheat_improvement >= 0.20
    passed_peak = peak_improvement >= 0.20
    passed_reward = reward_improvement >= 0.15

    comparison_lines = [
        f"- PPO total reward: {ppo_total:.3f}",
        f"- Random total reward: {random_total:.3f}",
        f"- Distance-aware total reward: {distance_total:.3f}",
        f"- PPO early adjacency ratio: {ppo_early_adj:.3f} (random={random_early_adj:.3f}, distance-aware={float(distance_early['early_adjacency_ratio']):.3f})",
        f"- PPO cumulative early reheat: {float(ppo_early['cumulative_early_reheat']):.3f} (random={float(random_early['cumulative_early_reheat']):.3f}, distance-aware={float(distance_early['cumulative_early_reheat']):.3f})",
        f"- PPO early peak heat: {float(ppo_early['early_peak_heat']):.3f} (random={float(random_early['early_peak_heat']):.3f}, distance-aware={float(distance_early['early_peak_heat']):.3f})",
        f"- Reheat improvement vs random: {reheat_improvement * 100.0:.1f}%",
        f"- Peak improvement vs random: {peak_improvement * 100.0:.1f}%",
        f"- Total reward improvement vs random: {reward_improvement * 100.0:.1f}%",
    ]
    write_text(PPO_VS_BASELINES_TXT, comparison_lines)

    failed_criteria: list[str] = []
    if not passed_coverage:
        failed_criteria.append("coverage")
    if not passed_invalid:
        failed_criteria.append("invalid_action_rate")
    if not passed_early_adj:
        failed_criteria.append("early_adjacency_ratio")
    if not passed_reheat:
        failed_criteria.append("reheat_vs_random")
    if not passed_peak:
        failed_criteria.append("peak_vs_random")
    if not passed_reward:
        failed_criteria.append("total_reward_vs_random")

    structure_line = (
        "PPO scan order looks less random and partially more structured, but it is still not close to the distance-aware heuristic."
        if ppo_total > random_total and ppo_early_adj <= random_early_adj
        else "PPO behavior still looks closer to random than to the distance-aware thermo-aware pattern."
    )
    go_no_go = "GO: proceed to larger PPO training" if not failed_criteria else "NO-GO: do NOT scale PPO, revisit representation"
    verdict_lines = [
        f"1. Did PPO pass all required criteria? {'YES' if not failed_criteria else 'NO'}",
        f"2. Failed criteria: {', '.join(failed_criteria) if failed_criteria else 'none'}",
        f"3. PPO coverage ratio: {ppo_coverage:.3f}",
        f"4. PPO invalid action rate: {ppo_invalid:.3%}",
        f"5. PPO early adjacency ratio: {ppo_early_adj:.3f}",
        f"6. PPO reheat improvement vs random: {reheat_improvement * 100.0:.1f}%",
        f"7. PPO peak improvement vs random: {peak_improvement * 100.0:.1f}%",
        f"8. PPO total reward improvement vs random: {reward_improvement * 100.0:.1f}%",
        f"9. {structure_line}",
        "10. Credit assignment is now sufficient only if PPO clears the thermal and reward margins; it does not if those margins fail.",
        "",
        go_no_go,
    ]
    write_text_pair(VERDICT_TXT, VERDICT_MD, "PPO Smoke Segment6 Verdict", verdict_lines)

    print(f"Device used: {device}")
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved history to: {HISTORY_PATH}")
    print(f"Saved training curve to: {CURVE_PATH}")
    print(f"Saved reward breakdown CSV to: {BREAKDOWN_CSV}")
    print(f"Saved PPO-vs-baselines CSV to: {PPO_VS_BASELINES_CSV}")
    print(f"Saved verdict to: {VERDICT_TXT}")


if __name__ == "__main__":
    main()
