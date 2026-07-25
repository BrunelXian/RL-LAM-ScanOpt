"""Create Figure C: joint PPO competitiveness bubble plot."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


PROJECT_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
OUT_DIR = PROJECT_ROOT / "outputs" / "manuscript_figures"
DATA_CSV = OUT_DIR / "ppo_competitiveness_native_summary_data.csv"
BASE = OUT_DIR / "Figure_C_bubble_competitiveness_plot"


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


def bubble_size(count: int) -> float:
    return 80.0 + count * 95.0


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = prepare_data()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, ax = plt.subplots(figsize=(5.8, 4.2), constrained_layout=True)
    x = range(len(df))
    colors = ["#397ca6", "#397ca6", "#a8873f", "#ffffff"]
    edges = ["#24536d", "#24536d", "#6c5a2f", "#9b4d4d"]
    for i, row in df.iterrows():
        if int(row["topk_competitive"]) == 0:
            ax.scatter(i, row["best_rank_percentile_pct"], s=115, color="#9b4d4d", linewidth=1.8, marker="x", zorder=4)
        else:
            ax.scatter(
                i,
                row["best_rank_percentile_pct"],
                s=bubble_size(int(row["topk_competitive"])),
                facecolor=colors[i],
                edgecolor=edges[i],
                linewidth=1.0,
                alpha=0.72,
                zorder=3,
            )
        yoff = -6.5 if row["best_rank_percentile_pct"] > 60 else 5.2
        va = "top" if yoff < 0 else "bottom"
        ax.text(i, row["best_rank_percentile_pct"] + yoff, f"rank {int(row['best_rank'])}", ha="center", va=va, fontsize=9)

    ax.plot(list(x), df["best_rank_percentile_pct"], color="#b7b7b7", linewidth=0.9, zorder=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["N_label"])
    ax.set_ylabel("Best-rank percentile (%)\nLower is better")
    ax.set_xlabel("Native track-count setting")
    ax.set_ylim(0, 82)
    ax.set_title("Figure C. Joint view of PPO competitiveness: rank percentile and top-k occupancy", fontsize=11.2, fontweight="bold")
    ax.grid(axis="y", color="#e3e3e3", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=8, markerfacecolor="#397ca6", markeredgecolor="#24536d", label="Bubble size = top-k competitive PPO candidates"),
        Line2D([0], [0], marker="x", linestyle="", markersize=8, markeredgewidth=1.6, color="#9b4d4d", label="0 top-k competitive candidates"),
        Line2D([0], [0], linestyle="-", color="#b7b7b7", label="y-axis = best-rank percentile"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", frameon=False, fontsize=8.2)

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(BASE.with_suffix(f".{ext}"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    caption = (
        "Figure C. Joint view of PPO competitiveness across native track-count settings. "
        "The vertical axis shows the best PPO rank percentile (lower is better), while "
        "marker size represents the number of top-k competitive PPO candidates within "
        "the corresponding 8-candidate PPO batch. The plot highlights bounded "
        "competitiveness in N12 and N16, limited competitiveness in N24, and the absence "
        "of top-k competitiveness in N40.\n"
    )
    BASE.with_name(BASE.name + "_caption.txt").write_text(caption, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
