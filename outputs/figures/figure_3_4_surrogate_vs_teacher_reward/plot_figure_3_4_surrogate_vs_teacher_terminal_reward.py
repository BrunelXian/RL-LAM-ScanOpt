"""Plot Figure 3.4 surrogate-vs-teacher terminal reward agreement.

This script uses the frozen Stage B surrogate reward model holdout prediction
table. It does not train models, run Abaqus, open ODB files, or generate
CAE/INP/JNL files.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
OUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "figure_3_4_surrogate_vs_teacher_reward"
INPUT_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "stage3_ppo_rl_lam_fea_addendum_v01"
    / "surrogate_reward_model"
    / "tables"
    / "surrogate_reward_model_candidate_predictions_holdout.csv"
)
SUMMARY_JSON = (
    PROJECT_ROOT
    / "outputs"
    / "stage3_ppo_rl_lam_fea_addendum_v01"
    / "surrogate_reward_model"
    / "tables"
    / "surrogate_reward_model_validation_summary.json"
)

TEACHER_COL = "reward_lex_u2_peeq_surfacet"
PRED_COL = "predicted_reward_lex_u2_peeq_surfacet"
SUPPORTED_N = [12, 16, 24, 40]

PNG = OUT_DIR / "figure_3_4_surrogate_vs_teacher_terminal_reward.png"
PDF = OUT_DIR / "figure_3_4_surrogate_vs_teacher_terminal_reward.pdf"
SVG = OUT_DIR / "figure_3_4_surrogate_vs_teacher_terminal_reward.svg"
PLOT_DATA = OUT_DIR / "figure_3_4_surrogate_vs_teacher_terminal_reward_plot_data.csv"
REPORT_MD = OUT_DIR / "figure_3_4_surrogate_vs_teacher_terminal_reward_report.md"
REPORT_JSON = OUT_DIR / "figure_3_4_surrogate_vs_teacher_terminal_reward_report.json"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV)
    required = ["n", "strategy_name", TEACHER_COL, PRED_COL]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {INPUT_CSV}: {missing}")
    df = df[df["n"].isin(SUPPORTED_N)].copy()
    df["teacher_derived_terminal_reward"] = pd.to_numeric(df[TEACHER_COL], errors="coerce")
    df["surrogate_predicted_terminal_reward"] = pd.to_numeric(df[PRED_COL], errors="coerce")
    if df[["teacher_derived_terminal_reward", "surrogate_predicted_terminal_reward"]].isna().any().any():
        raise ValueError("Found missing/non-numeric reward values in holdout prediction table.")
    return df


def make_plot(df: pd.DataFrame, spearman: float, pearson: float) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10.5,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    colors = {12: "#356f9f", 16: "#d08a24", 24: "#4f9a5a", 40: "#b94b4b"}
    markers = {12: "o", 16: "s", 24: "^", 40: "D"}

    fig, ax = plt.subplots(figsize=(4.8, 4.4), constrained_layout=True)
    for n in SUPPORTED_N:
        sub = df[df["n"] == n]
        ax.scatter(
            sub["teacher_derived_terminal_reward"],
            sub["surrogate_predicted_terminal_reward"],
            s=34,
            marker=markers[n],
            color=colors[n],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.82,
            label=f"N{n} (n={len(sub)})",
        )

    low = min(df["teacher_derived_terminal_reward"].min(), df["surrogate_predicted_terminal_reward"].min())
    high = max(df["teacher_derived_terminal_reward"].max(), df["surrogate_predicted_terminal_reward"].max())
    pad = 0.04 * (high - low)
    low -= pad
    high += pad
    ax.plot([low, high], [low, high], color="#333333", linewidth=1.0, linestyle="--", label="y = x")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Teacher-derived terminal reward")
    ax.set_ylabel("Surrogate-predicted terminal reward")
    ax.set_title("Surrogate-predicted vs teacher-derived terminal reward")
    ax.text(
        0.04,
        0.96,
        f"Spearman = {spearman:.4f}\nPearson = {pearson:.4f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#d0d0d0", "linewidth": 0.6},
    )
    ax.legend(frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.savefig(PNG, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(PDF, bbox_inches="tight", facecolor="white")
    fig.savefig(SVG, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    spearman = float(df["teacher_derived_terminal_reward"].corr(df["surrogate_predicted_terminal_reward"], method="spearman"))
    pearson = float(df["teacher_derived_terminal_reward"].corr(df["surrogate_predicted_terminal_reward"], method="pearson"))
    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    reported = summary["overall_metrics"]

    plot_cols = [
        "source_row_index",
        "n",
        "strategy_name",
        TEACHER_COL,
        PRED_COL,
        "teacher_derived_terminal_reward",
        "surrogate_predicted_terminal_reward",
    ]
    df[plot_cols].to_csv(PLOT_DATA, index=False)
    make_plot(df, reported["spearman"], reported["pearson"])

    n_counts = {f"N{int(k)}": int(v) for k, v in df["n"].value_counts().sort_index().items()}
    report = {
        "input_data_path": str(INPUT_CSV),
        "cases_used": int(len(df)),
        "N_wise_counts": n_counts,
        "computed_spearman": spearman,
        "computed_pearson": pearson,
        "reported_spearman": reported["spearman"],
        "reported_pearson": reported["pearson"],
        "native_N_only": sorted(df["n"].unique().astype(int).tolist()) == SUPPORTED_N,
        "output_png": str(PNG),
        "output_pdf": str(PDF),
        "output_svg": str(SVG),
        "plot_data_csv": str(PLOT_DATA),
        "interpretation_boundary": "Validates surrogate as PPO training environment; not a substitute for final Abaqus teacher validation.",
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    REPORT_MD.write_text(
        "\n".join(
            [
                "# Figure 3.4 Surrogate vs Teacher Terminal Reward Report",
                "",
                f"- Input data path: `{INPUT_CSV}`",
                f"- Number of validation cases used: `{len(df)}`",
                f"- N-wise counts: `{n_counts}`",
                f"- Spearman correlation: `{reported['spearman']:.4f}`",
                f"- Pearson correlation: `{reported['pearson']:.4f}`",
                f"- Computed Spearman from plot data: `{spearman:.12f}`",
                f"- Computed Pearson from plot data: `{pearson:.12f}`",
                f"- PNG: `{PNG}`",
                f"- PDF: `{PDF}`",
                f"- SVG: `{SVG}`",
                f"- Plot data CSV: `{PLOT_DATA}`",
                "",
                "Interpretation boundary: this figure validates the surrogate reward model as a PPO training environment. It is not a substitute for final Abaqus teacher validation, and final physical claims remain grounded in independent teacher simulations.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
