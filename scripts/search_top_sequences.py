"""Large-scale GPU-accelerated scan-sequence search."""

from __future__ import annotations

import argparse
import heapq
import json
import multiprocessing as mp
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from core.geometry import downsample_mask, generate_text_mask
from core.rollout import run_plan
from core.viz import (
    save_order_map_figure,
    save_scan_path_gif,
    save_target_mask_figure,
    save_thermal_map_figure,
)

DEFAULT_TEXT = "TWI"
DEFAULT_GRID_SIZE = 64
DEFAULT_CANVAS_SIZE = 1024
DEFAULT_ITERATIONS = 100_000
DEFAULT_TOP_K = 10
DEFAULT_PROGRESS_EVERY = 1_000
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = max(1, mp.cpu_count() - 1)
DEFAULT_DEPOSIT_STRENGTH = 200.0
DEFAULT_DIFFUSION = 0.08
DEFAULT_DECAY = 0.96
DEFAULT_SIGMA = 1.0


@dataclass(order=True)
class RankedSequence:
    """Container used for heap-based top-k ranking."""

    score: float
    iteration: int
    metrics: dict = field(compare=False)
    actions: list[tuple[int, int]] = field(compare=False)


def build_target_mask(text: str, canvas_size: int, grid_size: int) -> np.ndarray:
    """Build the legal scan region from centered text."""
    high_res = generate_text_mask(text=text, canvas_size=canvas_size)
    return downsample_mask(high_res, grid_size=grid_size)


def _worker_generate_batch(task: tuple[int, int, int, int]) -> tuple[int, np.ndarray]:
    """Generate one batch of candidate permutations in a worker process."""
    start_iteration, batch_size, num_valid_cells, seed = task
    rng = np.random.default_rng(seed)
    batch = np.empty((batch_size, num_valid_cells), dtype=np.int32)
    for idx in range(batch_size):
        batch[idx] = rng.permutation(num_valid_cells)
    return start_iteration, batch


def _batched_tasks(
    iterations: int,
    batch_size: int,
    num_valid_cells: int,
    seed: int,
) -> list[tuple[int, int, int, int]]:
    """Split total work into deterministic generation batches."""
    tasks: list[tuple[int, int, int, int]] = []
    start_iteration = 1
    while start_iteration <= iterations:
        current_batch_size = min(batch_size, iterations - start_iteration + 1)
        task_seed = seed + start_iteration * 9973
        tasks.append((start_iteration, current_batch_size, num_valid_cells, task_seed))
        start_iteration += current_batch_size
    return tasks


def _precompute_heat_templates(
    valid_cells: np.ndarray,
    grid_size: int,
    device: torch.device,
    deposit_strength: float,
    sigma: float,
) -> torch.Tensor:
    """Precompute a Gaussian heat map for each legal cell on the target grid."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    yy, xx = torch.meshgrid(
        torch.arange(grid_size, device=device, dtype=torch.float32),
        torch.arange(grid_size, device=device, dtype=torch.float32),
        indexing="ij",
    )
    cells = torch.as_tensor(valid_cells, device=device, dtype=torch.float32)
    rows = cells[:, 0].view(-1, 1, 1)
    cols = cells[:, 1].view(-1, 1, 1)
    squared_dist = (yy.unsqueeze(0) - rows) ** 2 + (xx.unsqueeze(0) - cols) ** 2
    amplitude = float(deposit_strength) / (np.pi * sigma**2)
    heat_templates = amplitude * torch.exp(-squared_dist / (sigma**2))
    return heat_templates.to(dtype=torch.float32)


def _diffuse_and_decay_gpu(
    field: torch.Tensor,
    diffusion: float,
    decay: float,
    kernel: torch.Tensor,
) -> torch.Tensor:
    """Apply 3x3 mean diffusion followed by global cooling on GPU."""
    neighborhood_mean = F.conv2d(field, kernel, padding=1)
    diffused = (1.0 - diffusion) * field + diffusion * neighborhood_mean
    return torch.clamp(diffused * decay, min=0.0)


def score_from_metrics(metrics: dict[str, float]) -> float:
    """Return a scalar score where larger is better."""
    return float(
        metrics["coverage_ratio"] * 1000.0
        - metrics["thermal_variance"] * 100.0
        - metrics["thermal_peak"] * 10.0
        - metrics["thermal_mean"] * 10.0
    )


def compute_scan_sequence(
    permutation_batch: np.ndarray,
    valid_cells: np.ndarray,
    heat_templates: torch.Tensor,
    grid_size: int,
    device: torch.device,
    diffusion: float = DEFAULT_DIFFUSION,
    decay: float = DEFAULT_DECAY,
) -> list[tuple[list[tuple[int, int]], float, dict[str, float]]]:
    """Score a batch of candidate scan sequences on GPU."""
    if permutation_batch.ndim != 2:
        raise ValueError("permutation_batch must be a 2D array")

    batch_size, num_steps = permutation_batch.shape
    permutations = torch.as_tensor(permutation_batch, dtype=torch.long, device=device)
    field = torch.zeros((batch_size, 1, grid_size, grid_size), device=device, dtype=torch.float32)
    kernel = torch.full((1, 1, 3, 3), 1.0 / 9.0, device=device, dtype=torch.float32)

    for step_idx in range(num_steps):
        action_indices = permutations[:, step_idx]
        field = field + heat_templates[action_indices].unsqueeze(1)
        field = _diffuse_and_decay_gpu(field, diffusion=diffusion, decay=decay, kernel=kernel)

    final_fields = field.squeeze(1)
    thermal_mean = final_fields.mean(dim=(1, 2))
    thermal_peak = final_fields.amax(dim=(1, 2))
    thermal_variance = final_fields.var(dim=(1, 2), unbiased=False)

    thermal_mean_np = thermal_mean.detach().cpu().numpy()
    thermal_peak_np = thermal_peak.detach().cpu().numpy()
    thermal_variance_np = thermal_variance.detach().cpu().numpy()

    results: list[tuple[list[tuple[int, int]], float, dict[str, float]]] = []
    for batch_idx in range(batch_size):
        metrics = {
            "target_cells": int(num_steps),
            "scanned_cells": int(num_steps),
            "steps": int(num_steps),
            "coverage_ratio": 1.0,
            "thermal_mean": float(thermal_mean_np[batch_idx]),
            "thermal_peak": float(thermal_peak_np[batch_idx]),
            "thermal_variance": float(thermal_variance_np[batch_idx]),
        }
        score = score_from_metrics(metrics)
        actions = [tuple(map(int, valid_cells[index])) for index in permutation_batch[batch_idx]]
        results.append((actions, score, metrics))
    return results


def update_top_sequences(
    heap: list[RankedSequence],
    candidate: RankedSequence,
    top_k: int,
) -> None:
    """Insert a candidate into the bounded top-k heap."""
    if len(heap) < top_k:
        heapq.heappush(heap, candidate)
        return
    if candidate.score > heap[0].score:
        heapq.heapreplace(heap, candidate)


def serialise_ranked_sequences(sequences: Iterable[RankedSequence]) -> list[dict]:
    """Convert ranked results into JSON-friendly structures."""
    return [
        {
            "rank": rank,
            "score": item.score,
            "iteration": item.iteration,
            "metrics": item.metrics,
            "actions": item.actions,
        }
        for rank, item in enumerate(sorted(sequences, reverse=True), start=1)
    ]


def save_top_sequences(
    top_sequences: list[RankedSequence],
    output_dir: Path,
    text: str,
    grid_size: int,
    iterations: int,
) -> tuple[Path, Path]:
    """Write the ranked scan sequences to JSON and text files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"top_10_sequences_{text.lower()}_{grid_size}x{grid_size}.json"
    txt_path = output_dir / f"top_10_sequences_{text.lower()}_{grid_size}x{grid_size}.txt"

    payload = {
        "text": text,
        "grid_size": grid_size,
        "iterations": iterations,
        "generated_at_epoch": time.time(),
        "top_sequences": serialise_ranked_sequences(top_sequences),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines: list[str] = [
        f"Top {len(top_sequences)} scan sequences for text={text!r} on {grid_size}x{grid_size}",
        f"Iterations: {iterations}",
        "",
    ]
    for item in payload["top_sequences"]:
        lines.append(f"Rank {item['rank']} | score={item['score']:.6f} | iteration={item['iteration']}")
        lines.append(f"Metrics: {item['metrics']}")
        lines.append(f"Actions: {item['actions']}")
        lines.append("")

    txt_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, txt_path


def save_top1_visualisations(
    top_sequences: list[RankedSequence],
    target_mask: np.ndarray,
    text: str,
    grid_size: int,
) -> tuple[Path, Path, Path, Path] | None:
    """Render target, order-map, thermal-map, and GIF for the best sequence."""
    if not top_sequences:
        return None

    figures_dir = PROJECT_ROOT / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    best = max(top_sequences, key=lambda item: item.score)
    rollout_result = run_plan(mask=target_mask, actions=best.actions, record_history=True, history_stride=8)

    target_path = figures_dir / f"target_mask_{text.lower()}_{grid_size}_search.png"
    order_path = figures_dir / f"order_map_top1_search_{text.lower()}_{grid_size}x{grid_size}.png"
    thermal_path = figures_dir / f"thermal_map_top1_search_{text.lower()}_{grid_size}x{grid_size}.png"
    gif_path = figures_dir / f"scan_path_top1_search_{text.lower()}_{grid_size}x{grid_size}.gif"

    save_target_mask_figure(rollout_result["target_mask"], target_path)
    save_order_map_figure(
        rollout_result["order_map"],
        order_path,
        title=f"Top 1 Search Order Map ({text}, {grid_size}x{grid_size})",
    )
    save_thermal_map_figure(
        rollout_result["final_thermal"],
        thermal_path,
        title=f"Top 1 Search Thermal Map ({text}, {grid_size}x{grid_size})",
    )
    save_scan_path_gif(
        target_mask=rollout_result["target_mask"],
        scanned_history=rollout_result["scanned_history"],
        path=gif_path,
        title=f"Top 1 Search Scan Path ({text})",
    )
    return target_path, order_path, thermal_path, gif_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the search job."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Text shape used as the legal scan region.")
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE, help="Decision grid resolution.")
    parser.add_argument("--canvas-size", type=int, default=DEFAULT_CANVAS_SIZE, help="High-resolution text canvas.")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS, help="Number of candidate plans to test.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of best plans to keep.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible search.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="GPU scoring batch size.")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS, help="CPU workers for path generation.")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=DEFAULT_PROGRESS_EVERY,
        help="Print progress every N iterations.",
    )
    return parser.parse_args()


def main() -> None:
    """Run large-scale scan-path search and optionally shut down afterward."""
    args = parse_args()
    if args.iterations <= 0:
        raise ValueError("iterations must be positive")
    if args.top_k <= 0:
        raise ValueError("top-k must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.num_workers <= 0:
        raise ValueError("num-workers must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_mask = build_target_mask(args.text, args.canvas_size, args.grid_size)
    valid_cells = np.argwhere(np.asarray(target_mask, dtype=bool))
    if len(valid_cells) == 0:
        raise ValueError("target mask contains no valid scan cells")

    heat_templates = _precompute_heat_templates(
        valid_cells=valid_cells,
        grid_size=args.grid_size,
        device=device,
        deposit_strength=DEFAULT_DEPOSIT_STRENGTH,
        sigma=DEFAULT_SIGMA,
    )
    top_sequences: list[RankedSequence] = []
    tasks = _batched_tasks(
        iterations=args.iterations,
        batch_size=args.batch_size,
        num_valid_cells=len(valid_cells),
        seed=args.seed,
    )

    print(
        f"Searching {args.iterations} candidate scan sequences for {args.text!r} "
        f"on a {args.grid_size}x{args.grid_size} grid..."
    )
    print(f"Using device: {device.type}")
    print(f"Using CPU workers for candidate generation: {args.num_workers}")

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.num_workers) as pool:
        for start_iteration, permutation_batch in pool.imap(_worker_generate_batch, tasks):
            batch_results = compute_scan_sequence(
                permutation_batch=permutation_batch,
                valid_cells=valid_cells,
                heat_templates=heat_templates,
                grid_size=args.grid_size,
                device=device,
                diffusion=DEFAULT_DIFFUSION,
                decay=DEFAULT_DECAY,
            )
            for offset, (actions, score, metrics) in enumerate(batch_results):
                iteration = start_iteration + offset
                candidate = RankedSequence(
                    score=score,
                    iteration=iteration,
                    metrics=metrics,
                    actions=actions,
                )
                update_top_sequences(top_sequences, candidate, args.top_k)

                if iteration == 1 or iteration % args.progress_every == 0:
                    best = max(top_sequences, key=lambda item: item.score)
                    print(
                        f"[{iteration}/{args.iterations}] best_score={best.score:.6f} "
                        f"coverage={best.metrics['coverage_ratio']:.3f} "
                        f"peak={best.metrics['thermal_peak']:.3f} "
                        f"variance={best.metrics['thermal_variance']:.3f}"
                    )

    output_dir = PROJECT_ROOT / "assets" / "models"
    json_path, txt_path = save_top_sequences(
        top_sequences=top_sequences,
        output_dir=output_dir,
        text=args.text,
        grid_size=args.grid_size,
        iterations=args.iterations,
    )
    print(f"Saved Top {len(top_sequences)} results to: {json_path}")
    print(f"Saved text summary to: {txt_path}")
    visual_paths = save_top1_visualisations(
        top_sequences=top_sequences,
        target_mask=target_mask,
        text=args.text,
        grid_size=args.grid_size,
    )
    if visual_paths is not None:
        target_path, order_path, thermal_path, gif_path = visual_paths
        print(f"Saved Top 1 target figure to: {target_path}")
        print(f"Saved Top 1 order-map figure to: {order_path}")
        print(f"Saved Top 1 thermal-map figure to: {thermal_path}")
        print(f"Saved Top 1 scan GIF to: {gif_path}")

    print("Search completed.")


if __name__ == "__main__":
    main()
