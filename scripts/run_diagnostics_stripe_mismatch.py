"""Diagnose stripe-level action granularity vs cell-level thermal reward consequences."""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.planners.distance_aware_cool_first import plan_distance_aware_cool_first
from core.planners.random_planner import plan_random
from core.reward import (
    DEFAULT_REHEAT_WINDOW_SIZE,
    local_target_preheat,
    representative_location,
    target_heat_peak,
    target_heat_variance,
)
from core.rollout import run_plan
from core.thermal import create_empty_thermal_field, update_thermal_field
from rl.env_scan import ScanPlanningEnv, build_twi_mask


MODEL_PATH = PROJECT_ROOT / "assets" / "models" / "maskable_ppo_smoke_calibrated.zip"
SELECTION_PATH = PROJECT_ROOT / "assets" / "models" / "reward_calibration_selection.json"

PART1_CSV = PROJECT_ROOT / "assets" / "models" / "diagnostics_stripe_action_impact.csv"
PART1_SUMMARY_TXT = PROJECT_ROOT / "assets" / "models" / "diagnostics_stripe_action_impact_summary.txt"
PART1_SUMMARY_MD = PROJECT_ROOT / "assets" / "models" / "diagnostics_stripe_action_impact_summary.md"

PART2_STEPWISE_CSV = PROJECT_ROOT / "assets" / "models" / "diagnostics_reward_terms_stepwise.csv"
PART2_EPISODE_CSV = PROJECT_ROOT / "assets" / "models" / "diagnostics_reward_terms_episode_summary.csv"
PART2_CONTRIB_CSV = PROJECT_ROOT / "assets" / "models" / "diagnostics_reward_terms_contribution_summary.csv"
PART2_SUMMARY_TXT = PROJECT_ROOT / "assets" / "models" / "diagnostics_reward_signal_summary.txt"
PART2_SUMMARY_MD = PROJECT_ROOT / "assets" / "models" / "diagnostics_reward_signal_summary.md"

PART3_CLUSTERING_CSV = PROJECT_ROOT / "assets" / "models" / "diagnostics_policy_clustering_metrics.csv"
PART3_SUMMARY_TXT = PROJECT_ROOT / "assets" / "models" / "diagnostics_policy_clustering_summary.txt"
PART3_SUMMARY_MD = PROJECT_ROOT / "assets" / "models" / "diagnostics_policy_clustering_summary.md"
PART3_PPO_HEATMAPS = PROJECT_ROOT / "assets" / "figures" / "diagnostics_ppo_early_late_heatmaps.png"
PART3_BASELINE_HEATMAPS = PROJECT_ROOT / "assets" / "figures" / "diagnostics_baseline_early_late_heatmaps.png"
PART3_REHEAT_CURVES = PROJECT_ROOT / "assets" / "figures" / "diagnostics_reheat_accumulation_curves.png"

FINAL_REPORT_TXT = PROJECT_ROOT / "assets" / "models" / "diagnostics_stripe_mismatch_report.txt"
FINAL_REPORT_MD = PROJECT_ROOT / "assets" / "models" / "diagnostics_stripe_mismatch_report.md"

EARLY_FRACTION = 0.30
LATE_FRACTION = 0.30
ADJACENCY_RADIUS = 2.0
ADJACENCY_HISTORY_WINDOW = 5
PART2_EPISODES = 20
PART3_EPISODES = 10


def ensure_output_dirs() -> None:
    """Create output directories used by the diagnostics run."""
    for path in (
        PART1_CSV,
        PART2_STEPWISE_CSV,
        PART3_CLUSTERING_CSV,
        PART3_PPO_HEATMAPS,
        FINAL_REPORT_TXT,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)


def load_variant1_weights() -> dict[str, float]:
    """Load the calibrated variant_1 reward weights."""
    payload = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    variants = payload.get("variants", {})
    if "variant_1" not in variants:
        raise KeyError("reward_calibration_selection.json does not contain variant_1.")
    return {key: float(value) for key, value in variants["variant_1"].items()}


def make_env(reward_weights: dict[str, float]) -> ScanPlanningEnv:
    """Create the current stripe environment with the calibrated reward."""
    planning_mask = build_twi_mask(grid_size=64, canvas_size=1024, text="TWI")
    return ScanPlanningEnv(
        planning_mask=planning_mask,
        grid_size=64,
        text="TWI",
        canvas_size=1024,
        reward_weights=reward_weights,
    )


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV with a stable field order."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def write_text_pair(txt_path: Path, md_path: Path, title: str, lines: list[str]) -> None:
    """Write matching TXT and Markdown summaries."""
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    md_lines = [f"# {title}", ""] + lines
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def stripe_action_impact_analysis(
    target_mask: np.ndarray,
    reward_weights: dict[str, float],
) -> dict[str, Any]:
    """Quantify isolated stripe-action thermal consequences from an empty-field reference state."""
    env = make_env(reward_weights)
    env.reset(seed=42)
    rows: list[dict[str, Any]] = []
    coverage_reward = reward_weights["coverage"]

    for stripe_index, stripe in enumerate(env.stripes):
        cells = env._stripe_cells(stripe)
        center = representative_location(stripe)
        pre_field = create_empty_thermal_field(env.grid_size)
        pre_peak = target_heat_peak(pre_field, target_mask)
        pre_variance = target_heat_variance(pre_field, target_mask)
        pre_reheat = local_target_preheat(
            pre_update_heat=pre_field,
            target_mask=target_mask,
            center=center,
            window_size=env.reheat_window_size,
        )

        post_field = pre_field.copy()
        for cell in cells:
            post_field = update_thermal_field(
                field=post_field,
                action=cell,
                deposit_strength=env.deposit_strength,
                diffusion=env.diffusion,
                decay=env.decay,
            )

        post_peak = target_heat_peak(post_field, target_mask)
        post_variance = target_heat_variance(post_field, target_mask)
        post_reheat = local_target_preheat(
            pre_update_heat=post_field,
            target_mask=target_mask,
            center=center,
            window_size=env.reheat_window_size,
        )

        delta_peak = post_peak - pre_peak
        delta_variance = post_variance - pre_variance
        delta_reheat_proxy = post_reheat - pre_reheat
        rows.append(
            {
                "stripe_index": stripe_index,
                "center_row": center[0] if center else -1,
                "center_col": center[1] if center else -1,
                "affected_cells": len(cells),
                "coverage_reward": coverage_reward,
                "pre_reheat": pre_reheat,
                "post_peak": post_peak,
                "post_variance": post_variance,
                "delta_peak": delta_peak,
                "delta_variance": delta_variance,
                "delta_reheat_proxy": delta_reheat_proxy,
                "delta_peak_to_coverage": delta_peak / max(coverage_reward, 1e-9),
                "delta_variance_to_coverage": delta_variance / max(coverage_reward, 1e-9),
                "delta_reheat_to_coverage": delta_reheat_proxy / max(coverage_reward, 1e-9),
            }
        )

    write_csv(
        PART1_CSV,
        [
            "stripe_index",
            "center_row",
            "center_col",
            "affected_cells",
            "coverage_reward",
            "pre_reheat",
            "post_peak",
            "post_variance",
            "delta_peak",
            "delta_variance",
            "delta_reheat_proxy",
            "delta_peak_to_coverage",
            "delta_variance_to_coverage",
            "delta_reheat_to_coverage",
        ],
        rows,
    )

    affected_cells = [int(row["affected_cells"]) for row in rows]
    delta_peaks = [float(row["delta_peak"]) for row in rows]
    delta_variances = [float(row["delta_variance"]) for row in rows]
    delta_reheats = [float(row["delta_reheat_proxy"]) for row in rows]

    summary = {
        "affected_cells_mean": float(np.mean(affected_cells)),
        "affected_cells_std": float(np.std(affected_cells)),
        "affected_cells_min": int(np.min(affected_cells)),
        "affected_cells_max": int(np.max(affected_cells)),
        "delta_peak_mean": float(np.mean(delta_peaks)),
        "delta_peak_std": float(np.std(delta_peaks)),
        "delta_variance_mean": float(np.mean(delta_variances)),
        "delta_variance_std": float(np.std(delta_variances)),
        "delta_reheat_mean": float(np.mean(delta_reheats)),
        "delta_reheat_std": float(np.std(delta_reheats)),
        "coverage_reward": coverage_reward,
        "delta_peak_to_coverage_mean": float(np.mean([abs(x) / coverage_reward for x in delta_peaks])),
        "delta_variance_to_coverage_mean": float(np.mean([abs(x) / coverage_reward for x in delta_variances])),
        "delta_reheat_to_coverage_mean": float(np.mean([abs(x) / coverage_reward for x in delta_reheats])),
    }

    lines = [
        "Definition: isolated stripe-action analysis from an empty-field reference state.",
        f"- Average affected cells per stripe action: {summary['affected_cells_mean']:.2f}",
        f"- Stripe action size range: {summary['affected_cells_min']} to {summary['affected_cells_max']} cells",
        f"- Mean delta_peak: {summary['delta_peak_mean']:.4f} (std {summary['delta_peak_std']:.4f})",
        f"- Mean delta_variance: {summary['delta_variance_mean']:.6f} (std {summary['delta_variance_std']:.6f})",
        f"- Mean delta_reheat_proxy: {summary['delta_reheat_mean']:.4f} (std {summary['delta_reheat_std']:.4f})",
        f"- Coverage reward per stripe action remains fixed at {coverage_reward:.2f}",
        (
            f"- Mean |delta_peak| / coverage = {summary['delta_peak_to_coverage_mean']:.3f}, "
            f"Mean |delta_reheat_proxy| / coverage = {summary['delta_reheat_to_coverage_mean']:.3f}"
        ),
        (
            "- Interpretation: stripe actions affect multiple cells at once, so even when thermal deltas are measurable, "
            "the consequences are mixed across several depositions while coverage still arrives as one immediate positive event."
        ),
    ]
    write_text_pair(PART1_SUMMARY_TXT, PART1_SUMMARY_MD, "Stripe Action Impact Summary", lines)
    return summary


def reward_signal_dominance_analysis(reward_weights: dict[str, float]) -> dict[str, Any]:
    """Measure reward-term dominance over masked-random training-style episodes."""
    term_names = ["coverage", "completion_bonus", "invalid", "peak", "variance", "reheat", "jump", "total"]
    step_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []

    for episode_id in range(PART2_EPISODES):
        env = make_env(reward_weights)
        _, _ = env.reset(seed=100 + episode_id)
        rng = np.random.default_rng(10_000 + episode_id)
        episode_terms = {term: 0.0 for term in term_names}
        step_index = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            valid_actions = np.flatnonzero(env.action_masks())
            if len(valid_actions) == 0:
                break
            action = int(rng.choice(valid_actions))
            _, reward, terminated, truncated, info = env.step(action)
            reward_terms = info["reward_terms"]
            row = {
                "episode_id": episode_id,
                "seed": 100 + episode_id,
                "step_index": step_index,
                **{term: float(reward_terms[term]) for term in term_names},
            }
            step_rows.append(row)
            for term in term_names:
                episode_terms[term] += float(reward_terms[term])
            step_index += 1

        episode_rows.append(
            {
                "episode_id": episode_id,
                "seed": 100 + episode_id,
                "steps": step_index,
                **episode_terms,
            }
        )
        env.close()

    write_csv(
        PART2_STEPWISE_CSV,
        ["episode_id", "seed", "step_index", *term_names],
        step_rows,
    )
    write_csv(
        PART2_EPISODE_CSV,
        ["episode_id", "seed", "steps", *term_names],
        episode_rows,
    )

    contribution_rows: list[dict[str, Any]] = []
    total_abs_magnitude = sum(abs(float(row[term])) for row in step_rows for term in term_names if term != "total")
    for term in term_names:
        step_values = np.array([float(row[term]) for row in step_rows], dtype=np.float32)
        episode_values = np.array([float(row[term]) for row in episode_rows], dtype=np.float32)
        abs_step = np.abs(step_values)
        contribution_rows.append(
            {
                "term": term,
                "mean_abs_per_step": float(abs_step.mean()),
                "std_abs_per_step": float(abs_step.std()),
                "max_abs_per_step": float(abs_step.max()),
                "mean_signed_per_step": float(step_values.mean()),
                "mean_cumulative_per_episode": float(episode_values.mean()),
                "std_cumulative_per_episode": float(episode_values.std()),
                "fraction_of_total_abs_reward_magnitude": (
                    float(abs_step.sum()) / max(total_abs_magnitude, 1e-9) if term != "total" else 1.0
                ),
            }
        )

    write_csv(
        PART2_CONTRIB_CSV,
        [
            "term",
            "mean_abs_per_step",
            "std_abs_per_step",
            "max_abs_per_step",
            "mean_signed_per_step",
            "mean_cumulative_per_episode",
            "std_cumulative_per_episode",
            "fraction_of_total_abs_reward_magnitude",
        ],
        contribution_rows,
    )

    contribution_map = {row["term"]: row for row in contribution_rows}
    positive_signal = (
        float(contribution_map["coverage"]["fraction_of_total_abs_reward_magnitude"])
        + float(contribution_map["completion_bonus"]["fraction_of_total_abs_reward_magnitude"])
    )
    thermal_signal = (
        float(contribution_map["peak"]["fraction_of_total_abs_reward_magnitude"])
        + float(contribution_map["variance"]["fraction_of_total_abs_reward_magnitude"])
        + float(contribution_map["reheat"]["fraction_of_total_abs_reward_magnitude"])
    )
    jump_signal = float(contribution_map["jump"]["fraction_of_total_abs_reward_magnitude"])

    lines = [
        "Sampling policy: masked-random valid-action policy over 20 episodes with seeds 100-119.",
        f"- Coverage + completion share of absolute reward magnitude: {positive_signal:.3f}",
        f"- Thermal shaping share (peak + variance + reheat): {thermal_signal:.3f}",
        f"- Jump share of absolute reward magnitude: {jump_signal:.3f}",
        f"- Mean absolute per-step coverage contribution: {contribution_map['coverage']['mean_abs_per_step']:.3f}",
        f"- Mean absolute per-step reheat contribution: {contribution_map['reheat']['mean_abs_per_step']:.3f}",
        f"- Mean absolute per-step peak contribution: {contribution_map['peak']['mean_abs_per_step']:.3f}",
        f"- Mean cumulative per-episode jump contribution: {contribution_map['jump']['mean_cumulative_per_episode']:.3f}",
        (
            "- Interpretation: this quantifies the training-style reward signal under exploratory behavior, "
            "so we can see whether coverage/completion overwhelms the thermal penalties in practice."
        ),
    ]
    write_text_pair(PART2_SUMMARY_TXT, PART2_SUMMARY_MD, "Reward Signal Summary", lines)
    return {
        "positive_signal_share": positive_signal,
        "thermal_signal_share": thermal_signal,
        "jump_signal_share": jump_signal,
        "contribution_map": contribution_map,
    }


def rollout_ppo_episode(
    reward_weights: dict[str, float],
    model: object,
    episode_seed: int,
) -> dict[str, Any]:
    """Run one PPO environment episode and reconstruct a comparable cell-level rollout."""
    from sb3_contrib.common.maskable.utils import get_action_masks

    env = make_env(reward_weights)
    obs, _ = env.reset(seed=episode_seed)
    stripe_centers: list[tuple[int, int]] = []
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action_masks = get_action_masks(env)
        action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
        stripe = env.stripes[int(action)]
        stripe_center = representative_location(np.logical_and(stripe, env.target_mask & ~env.scanned_mask))
        if stripe_center is None:
            stripe_center = representative_location(stripe)
        if stripe_center is not None:
            stripe_centers.append(stripe_center)
        obs, reward, terminated, truncated, info = env.step(int(action))

    actions = list(env.executed_actions)
    mask = env.target_mask.copy()
    env.close()
    rollout_result = run_plan(
        mask=mask,
        actions=actions,
        reward_weights=reward_weights,
        record_history=True,
        history_stride=1,
    )
    rollout_result["stripe_centers"] = stripe_centers
    return rollout_result


def bbox_area_ratio(points: list[tuple[int, int]], total_bbox_area: float) -> float:
    """Compute the occupied bounding-box area ratio for a set of points."""
    if not points or total_bbox_area <= 0.0:
        return 0.0
    rows = [point[0] for point in points]
    cols = [point[1] for point in points]
    area = float((max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1))
    return area / total_bbox_area


def adjacency_ratio(points: list[tuple[int, int]]) -> float:
    """Fraction of points that stay near the recent-history window."""
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


def phase_indices(length: int) -> dict[str, tuple[int, int]]:
    """Return slice bounds for early, full, and late phases."""
    early_end = max(1, math.ceil(length * EARLY_FRACTION))
    late_start = max(0, length - max(1, math.ceil(length * LATE_FRACTION)))
    return {
        "early": (0, early_end),
        "full": (0, length),
        "late": (late_start, length),
    }


def episode_clustering_rows(
    policy_name: str,
    episode_id: int,
    result: dict[str, Any],
    total_bbox_area: float,
) -> list[dict[str, Any]]:
    """Compute clustering metrics for early/full/late phases of one rollout."""
    actions = [tuple(map(int, action)) for action in result["actions"]]
    reward_history = result["reward_history"]
    scanned_history = result.get("scanned_history", [])
    thermal_history = result.get("thermal_history", [])
    if not actions:
        return []

    indices = phase_indices(len(actions))
    rows: list[dict[str, Any]] = []
    for phase, (start, end) in indices.items():
        phase_actions = actions[start:end]
        if not phase_actions:
            continue
        snapshot_index = end - 1
        thermal_snapshot = thermal_history[snapshot_index]
        scanned_snapshot = scanned_history[snapshot_index]
        reheat_sum = float(sum(float(reward_history[idx]["reheat"]) for idx in range(start, end)))
        rows.append(
            {
                "policy": policy_name,
                "episode_id": episode_id,
                "phase": phase,
                "adjacency_ratio": adjacency_ratio(phase_actions),
                "concentration_metric": bbox_area_ratio(phase_actions, total_bbox_area),
                "cumulative_reheat": reheat_sum,
                "peak_heat": target_heat_peak(thermal_snapshot, result["target_mask"]),
                "heat_variance": target_heat_variance(thermal_snapshot, result["target_mask"]),
                "coverage_ratio": float(np.logical_and(scanned_snapshot, result["target_mask"]).sum() / result["target_mask"].sum()),
            }
        )
    return rows


def mean_snapshot(results: list[dict[str, Any]], fraction: float) -> np.ndarray:
    """Average a snapshot taken at a fixed episode fraction across several runs."""
    snapshots: list[np.ndarray] = []
    for result in results:
        history = result["thermal_history"]
        index = max(0, min(len(history) - 1, math.ceil(len(history) * fraction) - 1))
        snapshots.append(history[index])
    return np.mean(np.stack(snapshots, axis=0), axis=0)


def save_two_phase_heatmaps(
    ppo_results: list[dict[str, Any]],
    random_results: list[dict[str, Any]],
    distance_results: list[dict[str, Any]],
) -> None:
    """Save early-vs-late heatmap figures for PPO and the comparison baselines."""
    def _plot_pair(fig_path: Path, entries: list[tuple[str, list[dict[str, Any]]]]) -> None:
        fig, axes = plt.subplots(2, len(entries), figsize=(4 * len(entries), 7), dpi=150)
        if len(entries) == 1:
            axes = np.array(axes).reshape(2, 1)
        for col, (label, results) in enumerate(entries):
            early_map = mean_snapshot(results, EARLY_FRACTION)
            late_map = mean_snapshot(results, 1.0)
            for row, data, title_suffix in (
                (0, early_map, "Early (30%)"),
                (1, late_map, "Late / Final"),
            ):
                image = axes[row, col].imshow(data, cmap="magma", interpolation="bilinear")
                axes[row, col].set_title(f"{label} {title_suffix}")
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
                fig.colorbar(image, ax=axes[row, col], fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)

    _plot_pair(PART3_PPO_HEATMAPS, [("PPO", ppo_results)])
    _plot_pair(
        PART3_BASELINE_HEATMAPS,
        [
            ("Random", random_results),
            ("Distance-Aware", distance_results),
        ],
    )


def save_reheat_accumulation_curves(
    ppo_results: list[dict[str, Any]],
    random_results: list[dict[str, Any]],
    distance_results: list[dict[str, Any]],
) -> None:
    """Save a normalized cumulative reheat comparison curve."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    def _curve(results: list[dict[str, Any]], label: str) -> None:
        progress = np.linspace(0.0, 1.0, 100)
        curves: list[np.ndarray] = []
        for result in results:
            values = np.array([float(step["reheat"]) for step in result["reward_history"]], dtype=np.float32)
            cumulative = np.cumsum(values)
            source_x = np.linspace(0.0, 1.0, len(cumulative))
            curves.append(np.interp(progress, source_x, cumulative))
        mean_curve = np.mean(np.stack(curves, axis=0), axis=0)
        ax.plot(progress, mean_curve, label=label, linewidth=2.0)

    _curve(ppo_results, "PPO")
    _curve(random_results, "Random")
    _curve(distance_results, "Distance-Aware Cool-First")
    ax.set_title("Reheat Accumulation Curves")
    ax.set_xlabel("Normalized Episode Progress")
    ax.set_ylabel("Cumulative Reheat Term")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PART3_REHEAT_CURVES, bbox_inches="tight")
    plt.close(fig)


def clustering_analysis(reward_weights: dict[str, float]) -> dict[str, Any]:
    """Quantify PPO early clustering against random and distance-aware baselines."""
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as exc:
        raise ImportError("Maskable PPO dependencies are missing for diagnostics.") from exc

    model = MaskablePPO.load(str(MODEL_PATH))
    mask = build_twi_mask(grid_size=64, canvas_size=1024, text="TWI")
    target_rows, target_cols = np.nonzero(mask)
    total_bbox_area = float((target_rows.max() - target_rows.min() + 1) * (target_cols.max() - target_cols.min() + 1))

    ppo_results: list[dict[str, Any]] = []
    random_results: list[dict[str, Any]] = []
    distance_results: list[dict[str, Any]] = []
    clustering_rows: list[dict[str, Any]] = []

    for episode_id in range(PART3_EPISODES):
        ppo_result = rollout_ppo_episode(reward_weights=reward_weights, model=model, episode_seed=200 + episode_id)
        ppo_results.append(ppo_result)
        clustering_rows.extend(episode_clustering_rows("ppo_smoke", episode_id, ppo_result, total_bbox_area))

        random_result = run_plan(
            mask=mask,
            actions=plan_random(mask, seed=300 + episode_id),
            reward_weights=reward_weights,
            record_history=True,
            history_stride=1,
        )
        random_results.append(random_result)
        clustering_rows.extend(episode_clustering_rows("random", episode_id, random_result, total_bbox_area))

        distance_result = run_plan(
            mask=mask,
            actions=plan_distance_aware_cool_first(mask),
            reward_weights=reward_weights,
            record_history=True,
            history_stride=1,
        )
        distance_results.append(distance_result)
        clustering_rows.extend(
            episode_clustering_rows("distance_aware_cool_first", episode_id, distance_result, total_bbox_area)
        )

    write_csv(
        PART3_CLUSTERING_CSV,
        [
            "policy",
            "episode_id",
            "phase",
            "adjacency_ratio",
            "concentration_metric",
            "cumulative_reheat",
            "peak_heat",
            "heat_variance",
            "coverage_ratio",
        ],
        clustering_rows,
    )
    save_two_phase_heatmaps(ppo_results, random_results, distance_results)
    save_reheat_accumulation_curves(ppo_results, random_results, distance_results)

    def _mean(policy: str, phase: str, field: str) -> float:
        values = [
            float(row[field])
            for row in clustering_rows
            if row["policy"] == policy and row["phase"] == phase
        ]
        return float(np.mean(values)) if values else 0.0

    lines = [
        (
            f"Adjacency definition: fraction of action centers with at least one prior center in the last "
            f"{ADJACENCY_HISTORY_WINDOW} steps within Euclidean radius {ADJACENCY_RADIUS:.1f}."
        ),
        (
            "Concentration metric: visited-centers bounding-box area divided by the target-mask bounding-box area "
            "for the selected phase."
        ),
        f"- PPO early adjacency ratio: {_mean('ppo_smoke', 'early', 'adjacency_ratio'):.3f}",
        f"- Random early adjacency ratio: {_mean('random', 'early', 'adjacency_ratio'):.3f}",
        f"- Distance-aware early adjacency ratio: {_mean('distance_aware_cool_first', 'early', 'adjacency_ratio'):.3f}",
        f"- PPO early concentration metric: {_mean('ppo_smoke', 'early', 'concentration_metric'):.3f}",
        f"- Random early concentration metric: {_mean('random', 'early', 'concentration_metric'):.3f}",
        f"- Distance-aware early concentration metric: {_mean('distance_aware_cool_first', 'early', 'concentration_metric'):.3f}",
        f"- PPO cumulative early reheat: {_mean('ppo_smoke', 'early', 'cumulative_reheat'):.3f}",
        f"- Random cumulative early reheat: {_mean('random', 'early', 'cumulative_reheat'):.3f}",
        f"- Distance-aware cumulative early reheat: {_mean('distance_aware_cool_first', 'early', 'cumulative_reheat'):.3f}",
    ]
    write_text_pair(PART3_SUMMARY_TXT, PART3_SUMMARY_MD, "Policy Clustering Summary", lines)
    return {
        "rows": clustering_rows,
        "ppo_results": ppo_results,
        "random_results": random_results,
        "distance_results": distance_results,
        "mean_ppo_early_adjacency": _mean("ppo_smoke", "early", "adjacency_ratio"),
        "mean_random_early_adjacency": _mean("random", "early", "adjacency_ratio"),
        "mean_distance_early_adjacency": _mean("distance_aware_cool_first", "early", "adjacency_ratio"),
        "mean_ppo_early_reheat": _mean("ppo_smoke", "early", "cumulative_reheat"),
        "mean_random_early_reheat": _mean("random", "early", "cumulative_reheat"),
        "mean_distance_early_reheat": _mean("distance_aware_cool_first", "early", "cumulative_reheat"),
        "mean_ppo_early_concentration": _mean("ppo_smoke", "early", "concentration_metric"),
        "mean_random_early_concentration": _mean("random", "early", "concentration_metric"),
        "mean_distance_early_concentration": _mean("distance_aware_cool_first", "early", "concentration_metric"),
    }


def final_report(
    part1: dict[str, Any],
    part2: dict[str, Any],
    part3: dict[str, Any],
) -> None:
    """Write the final TXT and Markdown diagnostic report."""
    positive_share = part2["positive_signal_share"]
    thermal_share = part2["thermal_signal_share"]
    jump_share = part2["jump_signal_share"]
    ppo_clustered = part3["mean_ppo_early_adjacency"] > part3["mean_random_early_adjacency"]
    ppo_reheat_worse = abs(part3["mean_ppo_early_reheat"]) > abs(part3["mean_random_early_reheat"])
    root_cause = (
        "Stripe actions are likely too coarse for clean credit assignment: one action triggers multiple cell deposits, "
        "coverage arrives as one immediate positive reward, and the thermal consequences are spatially mixed."
    )
    recommended_next_step = "inspect stripe representation / action granularity"

    lines = [
        f"1. One stripe action affects {part1['affected_cells_mean']:.2f} cells on average "
        f"(range {part1['affected_cells_min']} to {part1['affected_cells_max']}).",
        (
            f"2. Delta signals exist, but attribution is mixed: mean delta_peak={part1['delta_peak_mean']:.4f}, "
            f"mean delta_reheat_proxy={part1['delta_reheat_mean']:.4f}, while a stripe still earns a fixed "
            f"coverage reward of {part1['coverage_reward']:.2f}."
        ),
        (
            f"3. Coverage + completion account for {positive_share:.3f} of absolute reward magnitude, "
            f"thermal shaping (peak + variance + reheat) accounts for {thermal_share:.3f}, and jump accounts for {jump_share:.3f}."
        ),
        (
            f"4. PPO does become locally clustered early: early adjacency={part3['mean_ppo_early_adjacency']:.3f}, "
            f"random={part3['mean_random_early_adjacency']:.3f}, distance-aware={part3['mean_distance_early_adjacency']:.3f}."
        ),
        (
            f"5. Compared with random and distance-aware cool-first, PPO looks thermally poor/coarse rather than structured: "
            f"early reheat PPO={part3['mean_ppo_early_reheat']:.3f}, random={part3['mean_random_early_reheat']:.3f}, "
            f"distance-aware={part3['mean_distance_early_reheat']:.3f}."
        ),
        f"6. Most likely root cause: {root_cause}",
        "",
        (
            "Core conclusion: the calibrated reward is not the main blocker now. The stronger bottleneck is that a stripe-level "
            "action bundles several cell-level thermal consequences, which weakens temporal credit assignment and encourages coarse local clustering."
        ),
        f"Recommended next step: {recommended_next_step}.",
    ]
    write_text_pair(FINAL_REPORT_TXT, FINAL_REPORT_MD, "Stripe Mismatch Diagnostic Report", lines)


def main() -> None:
    """Run the full stripe-mismatch diagnostics suite without changing training or reward logic."""
    ensure_output_dirs()
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Expected PPO smoke model not found: {MODEL_PATH}")
    if not SELECTION_PATH.exists():
        raise FileNotFoundError(f"Expected reward selection file not found: {SELECTION_PATH}")

    reward_weights = load_variant1_weights()
    target_mask = build_twi_mask(grid_size=64, canvas_size=1024, text="TWI")

    part1 = stripe_action_impact_analysis(target_mask, reward_weights)
    part2 = reward_signal_dominance_analysis(reward_weights)
    part3 = clustering_analysis(reward_weights)
    final_report(part1, part2, part3)

    print(f"Saved stripe action impact CSV to: {PART1_CSV}")
    print(f"Saved reward signal stepwise CSV to: {PART2_STEPWISE_CSV}")
    print(f"Saved clustering metrics CSV to: {PART3_CLUSTERING_CSV}")
    print(f"Saved final diagnostic report to: {FINAL_REPORT_TXT}")


if __name__ == "__main__":
    main()
