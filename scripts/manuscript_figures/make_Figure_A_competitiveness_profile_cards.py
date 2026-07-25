"""Create Figure A: N-dependent PPO competitiveness profile cards.

This script uses only the Stage H aggregate competitiveness summary provided
for manuscript figure preparation. It does not read ODB files, run Abaqus,
generate CAE/INP files, or modify teacher metrics.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import pandas as pd


PROJECT_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
OUT_DIR = PROJECT_ROOT / "outputs" / "manuscript_figures"
DATA_CSV = OUT_DIR / "ppo_competitiveness_native_summary_data.csv"
BASE = OUT_DIR / "Figure_A_competitiveness_profile_cards"


DATA = [
    {"N_label": "N12", "native_pool": 78, "ppo_candidates": 8, "topk_competitive": 5, "best_rank": 6},
    {"N_label": "N16", "native_pool": 78, "ppo_candidates": 8, "topk_competitive": 4, "best_rank": 2},
    {"N_label": "N24", "native_pool": 190, "ppo_candidates": 8, "topk_competitive": 3, "best_rank": 134},
    {"N_label": "N40", "native_pool": 206, "ppo_candidates": 8, "topk_competitive": 0, "best_rank": 147},
]


def prepare_data() -> pd.DataFrame:
    df = pd.DataFrame(DATA)
    df["best_rank_percentile"] = df["best_rank"] / df["native_pool"]
    df["best_rank_percentile_pct"] = 100.0 * df["best_rank_percentile"]
    df.to_csv(DATA_CSV, index=False)
    return df


def progress_bar(ax: plt.Axes, x: float, y: float, width: float, height: float, frac: float, color: str, label: str) -> None:
    ax.add_patch(Rectangle((x, y), width, height, facecolor="#f3f3f3", edgecolor="#d0d0d0", linewidth=0.7))
    ax.add_patch(Rectangle((x, y), width * max(0.0, min(1.0, frac)), height, facecolor=color, edgecolor="none"))
    ax.text(x, y + height + 0.025, label, ha="left", va="bottom", fontsize=8.5, color="#333333")


def draw_card(ax: plt.Axes, row: pd.Series, card_color: str) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(
        FancyBboxPatch(
            (0.03, 0.04),
            0.94,
            0.90,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=0.9,
            edgecolor="#c9c9c9",
            facecolor="white",
        )
    )
    ax.add_patch(Rectangle((0.03, 0.83), 0.94, 0.11, facecolor=card_color, edgecolor="none", alpha=0.18))
    ax.text(0.08, 0.885, row["N_label"], ha="left", va="center", fontsize=16, fontweight="bold", color="#111111")

    lines = [
        f"Native pool: {int(row['native_pool'])}",
        f"PPO candidates: {int(row['ppo_candidates'])}",
        f"Top-k competitive: {int(row['topk_competitive'])}",
        f"Best rank: {int(row['best_rank'])} / {int(row['native_pool'])}",
        f"Best-rank percentile: {row['best_rank_percentile_pct']:.1f}%",
    ]
    y = 0.76
    for line in lines:
        ax.text(0.08, y, line, ha="left", va="top", fontsize=9.7, color="#202020")
        y -= 0.095

    progress_bar(
        ax,
        0.08,
        0.20,
        0.78,
        0.035,
        row["topk_competitive"] / row["ppo_candidates"],
        card_color,
        f"Top-k occupancy: {int(row['topk_competitive'])}/8",
    )
    rank_frac = row["best_rank_percentile"]
    progress_bar(
        ax,
        0.08,
        0.095,
        0.78,
        0.035,
        rank_frac,
        "#6f6f6f",
        "Best-rank percentile (lower is better)",
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = prepare_data()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    colors = ["#2f78a5", "#2f78a5", "#9a7b2f", "#9b4d4d"]
    fig, axes = plt.subplots(1, 4, figsize=(11.0, 3.4), constrained_layout=True)
    for ax, (_, row), color in zip(axes, df.iterrows(), colors):
        draw_card(ax, row, color)
    fig.suptitle("Figure A. N-dependent competitiveness profile of PPO-generated candidates", fontsize=12.5, fontweight="bold", y=1.02)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(BASE.with_suffix(f".{ext}"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    caption = (
        "Figure A. N-dependent competitiveness profile of the teacher-validated PPO cohort. "
        "Each native track-count setting was evaluated using 8 PPO-generated candidates and "
        "ranked against its own native reference pool. The profile shows strong competitiveness "
        "in N12 and N16, limited competitiveness in N24, and no top-k competitiveness in N40. "
        "Auxiliary N32 cases are excluded from this ranking boundary.\n"
    )
    BASE.with_name(BASE.name + "_caption.txt").write_text(caption, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
