"""Matplotlib figure helpers for target masks, plans, thermal maps, metrics, and animation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _prepare_output_path(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def save_target_mask_figure(mask: np.ndarray, path: str | Path) -> None:
    """Save a clean view of the binary target grid."""
    output_path = _prepare_output_path(path)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.imshow(mask, cmap="gray_r", interpolation="nearest")
    ax.set_title("Target Mask")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_order_map_figure(order_map: np.ndarray, path: str | Path, title: str) -> None:
    """Save a figure showing the scan order over valid cells."""
    output_path = _prepare_output_path(path)
    masked = np.ma.masked_where(order_map < 0, order_map)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    image = ax.imshow(masked, cmap="inferno", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Scan Step")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_thermal_map_figure(field: np.ndarray, path: str | Path, title: str) -> None:
    """Save a figure of the final proxy thermal field."""
    output_path = _prepare_output_path(path)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    image = ax.imshow(field, cmap="magma", interpolation="bilinear")
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Proxy Thermal Level")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_metrics_bar_chart(results_dict: dict[str, dict], path: str | Path) -> None:
    """Save a multi-panel bar chart comparing key metrics across planners."""
    output_path = _prepare_output_path(path)
    planner_names = list(results_dict.keys())
    metric_names = ["coverage_ratio", "thermal_mean", "thermal_peak", "thermal_variance"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=150)
    axes = axes.ravel()
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(planner_names), 3)))

    for idx, metric_name in enumerate(metric_names):
        values = [float(results_dict[name]["metrics"][metric_name]) for name in planner_names]
        ax = axes[idx]
        ax.bar(planner_names, values, color=colors[: len(planner_names)])
        ax.set_title(metric_name.replace("_", " ").title())
        ax.set_ylabel("Value")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle("Baseline Planner Metric Comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_reward_breakdown_chart(
    reward_rows: list[dict[str, Any]],
    path: str | Path,
) -> None:
    """Save a grouped bar chart comparing Stage A reward totals per planner."""
    output_path = _prepare_output_path(path)
    planner_names = [str(row["planner"]) for row in reward_rows]
    metric_names = ["coverage", "invalid", "peak", "variance", "reheat", "jump", "total_reward"]
    x = np.arange(len(planner_names))
    width = 0.11

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    for idx, metric_name in enumerate(metric_names):
        values = [float(row[metric_name]) for row in reward_rows]
        ax.bar(x + (idx - (len(metric_names) - 1) / 2) * width, values, width=width, label=metric_name)

    ax.set_title("Baseline Reward Breakdown")
    ax.set_xlabel("Planner")
    ax.set_ylabel("Reward Contribution")
    ax.set_xticks(x)
    ax.set_xticklabels(planner_names)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(ncols=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_training_curves_figure(training_history: dict[str, list[float]], path: str | Path) -> None:
    """Save training curves for episode reward and terminal metrics."""
    output_path = _prepare_output_path(path)
    episodes = np.arange(1, len(training_history.get("coverage_ratio", [])) + 1)
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), dpi=150)
    axes = axes.ravel()
    metric_specs = [
        ("episode_reward", "Episode Reward"),
        ("coverage_ratio", "Coverage Ratio"),
        ("thermal_mean", "Thermal Mean"),
        ("thermal_variance", "Thermal Variance"),
        ("thermal_peak", "Thermal Peak"),
    ]

    for ax, (metric_key, label) in zip(axes, metric_specs):
        values = training_history.get(metric_key, [])
        if values:
            ax.plot(episodes, values, color="#1f77b4", linewidth=1.8)
        ax.set_title(label)
        ax.set_xlabel("Episode")
        ax.set_ylabel(label)
        ax.grid(True, linestyle="--", alpha=0.35)

    axes[-1].axis("off")

    fig.suptitle("Maskable PPO Training Curves", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_comparison_grid(
    results_dict: dict[str, dict[str, Any]],
    path: str | Path,
    field_key: str,
    title: str,
    cmap: str,
    colorbar_label: str,
) -> None:
    """Save a side-by-side comparison grid for planner outputs."""
    output_path = _prepare_output_path(path)
    planner_names = list(results_dict.keys())
    n = len(planner_names)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5), dpi=150)
    if n == 1:
        axes = [axes]

    for ax, planner_name in zip(axes, planner_names):
        data = results_dict[planner_name][field_key]
        if field_key == "order_map":
            data = np.ma.masked_where(data < 0, data)
        image = ax.imshow(data, cmap=cmap, interpolation="nearest")
        ax.set_title(planner_name.replace("_", " ").title())
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(colorbar_label)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_scan_path_gif(
    target_mask: np.ndarray,
    scanned_history: list[np.ndarray],
    path: str | Path,
    title: str = "RL Scan Path",
) -> None:
    """Save a simple GIF showing scan coverage progression over time."""
    output_path = _prepare_output_path(path)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    image = ax.imshow(np.zeros((*target_mask.shape, 3), dtype=np.float32), interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

    def update(frame_idx: int) -> list[Any]:
        frame = scanned_history[frame_idx]
        overlay = np.zeros((*frame.shape, 3), dtype=np.float32)
        overlay[..., 0] = np.where(target_mask, 0.2, 0.0)
        overlay[..., 1] = np.where(target_mask, 0.2, 0.0)
        overlay[..., 2] = np.where(target_mask, 0.2, 0.0)
        overlay[..., 1] = np.where(frame, 0.85, overlay[..., 1])
        overlay[..., 0] = np.where(frame, 0.95, overlay[..., 0])
        image.set_data(overlay)
        ax.set_title(f"{title} ({frame_idx + 1}/{len(scanned_history)})")
        return [image]

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(scanned_history),
        interval=100,
        blit=False,
        repeat=False,
    )
    anim.save(output_path, writer=animation.PillowWriter(fps=8))
    plt.close(fig)


def main() -> None:
    """Regenerate training curves and evaluation visualisations from saved outputs."""
    training_history_path = PROJECT_ROOT / "assets" / "models" / "training_history_maskable_ppo_twi.json"
    training_curves_path = PROJECT_ROOT / "assets" / "figures" / "training_curves_maskable_ppo.png"

    if training_history_path.exists():
        with training_history_path.open("r", encoding="utf-8") as file:
            training_history = json.load(file)
        save_training_curves_figure(training_history, training_curves_path)
        print(f"Saved training curves to: {training_curves_path}")
    else:
        print(f"Training history not found: {training_history_path}")

    from rl.eval_policy import main as eval_main

    eval_main()


if __name__ == "__main__":
    main()
