"""Create Figure B: best PPO rank percentile bar chart."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
OUT_DIR = PROJECT_ROOT / "outputs" / "manuscript_figures"
DATA_CSV = OUT_DIR / "ppo_competitiveness_native_summary_data.csv"
BASE = OUT_DIR / "Figure_B_best_rank_percentile_bar"


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
    fig, ax = plt.subplots(figsize=(5.4, 3.8), constrained_layout=True)
    colors = ["#3b7ea1", "#3b7ea1", "#a8873f", "#a45d5d"]
    bars = ax.bar(df["N_label"], df["best_rank_percentile_pct"], color=colors, edgecolor="#333333", linewidth=0.7, width=0.62)
    for bar, pct in zip(bars, df["best_rank_percentile_pct"]):
        ax.text(bar.get_x() + bar.get_width() / 2, pct + 2.2, f"{pct:.1f}%", ha="center", va="bottom", fontsize=9.5)
    ax.set_ylabel("Best PPO rank percentile (%)\nLower is better")
    ax.set_xlabel("Native track-count setting")
    ax.set_ylim(0, 82)
    ax.set_title("Figure B. Best PPO rank percentile across native track-count settings", fontsize=10.5, fontweight="bold", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e3e3e3", linewidth=0.6)
    ax.set_axisbelow(True)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(BASE.with_suffix(f".{ext}"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    caption = (
        "Figure B. Best PPO rank percentile across native track-count settings. "
        "The best PPO lexicographic rank in each N-specific native reference pool is "
        "normalized by the corresponding pool size. Lower values indicate stronger "
        "competitiveness. PPO rank concentration is strongest in the smaller-N regimes "
        "and weakens substantially as N increases.\n"
    )
    BASE.with_name(BASE.name + "_caption.txt").write_text(caption, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
