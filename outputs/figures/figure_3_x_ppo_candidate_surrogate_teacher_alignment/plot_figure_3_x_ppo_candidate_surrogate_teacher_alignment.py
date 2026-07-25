from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
INPUT_CSV = ROOT / "outputs" / "stage3_ppo_rl_lam_fea_addendum_v01" / "stageH_teacher_metric_ranking" / "tables" / "ppo_surrogate_vs_teacher_alignment.csv"
INPUT_SUMMARY_JSON = ROOT / "outputs" / "stage3_ppo_rl_lam_fea_addendum_v01" / "stageH_teacher_metric_ranking" / "tables" / "ppo_surrogate_vs_teacher_alignment_summary.json"
OUTPUT_DIR = ROOT / "outputs" / "figures" / "figure_3_x_ppo_candidate_surrogate_teacher_alignment"

PNG_PATH = OUTPUT_DIR / "figure_3_x_ppo_candidate_surrogate_teacher_alignment.png"
PDF_PATH = OUTPUT_DIR / "figure_3_x_ppo_candidate_surrogate_teacher_alignment.pdf"
SVG_PATH = OUTPUT_DIR / "figure_3_x_ppo_candidate_surrogate_teacher_alignment.svg"
PLOT_DATA_PATH = OUTPUT_DIR / "figure_3_x_ppo_candidate_surrogate_teacher_alignment_plot_data.csv"
REPORT_PATH = OUTPUT_DIR / "figure_3_x_ppo_candidate_surrogate_teacher_alignment_report.md"
REPORT_JSON_PATH = OUTPUT_DIR / "figure_3_x_ppo_candidate_surrogate_teacher_alignment_report.json"

X_COL = "predicted_surrogate_reward_lex"
Y_COL = "teacher_lex_reward_rank_normalized"
N_COL = "n"


def backup_existing(path: Path) -> None:
    if not path.exists():
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.stem}.bak_{stamp}{path.suffix}")
    path.replace(backup)


def load_inputs() -> tuple[pd.DataFrame, dict]:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing alignment CSV: {INPUT_CSV}")
    if not INPUT_SUMMARY_JSON.exists():
        raise FileNotFoundError(f"Missing alignment summary JSON: {INPUT_SUMMARY_JSON}")

    df = pd.read_csv(INPUT_CSV)
    with INPUT_SUMMARY_JSON.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    required = [
        N_COL,
        "strategy_name",
        "dataset_source",
        "is_ppo_candidate",
        "teacher_metrics_extracted",
        "teacher_validation_status",
        X_COL,
        Y_COL,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {INPUT_CSV}: {missing}")

    df = df.copy()
    df[N_COL] = df[N_COL].astype(int)
    if len(df) != 32:
        raise ValueError(f"Expected 32 PPO candidates, found {len(df)}")
    counts = df[N_COL].value_counts().sort_index().to_dict()
    expected_counts = {12: 8, 16: 8, 24: 8, 40: 8}
    if counts != expected_counts:
        raise ValueError(f"Expected N-wise counts {expected_counts}, found {counts}")
    if not df["is_ppo_candidate"].astype(bool).all():
        raise ValueError("Not every row is marked as a PPO candidate.")
    if not df["teacher_metrics_extracted"].astype(bool).all():
        raise ValueError("Not every row is marked as teacher-metric extracted.")
    if set(df["dataset_source"].dropna().unique()) != {"stage3_ppo_policy_only_batch32"}:
        raise ValueError("Unexpected dataset_source values; expected only stage3_ppo_policy_only_batch32.")
    if df[[X_COL, Y_COL]].isna().any().any():
        raise ValueError("Missing surrogate or teacher reward values.")

    return df, summary


def compute_stats(df: pd.DataFrame) -> dict:
    return {
        "spearman": float(df[X_COL].corr(df[Y_COL], method="spearman")),
        "pearson": float(df[X_COL].corr(df[Y_COL], method="pearson")),
        "n": int(len(df)),
        "n_counts": {f"N{int(k)}": int(v) for k, v in df[N_COL].value_counts().sort_index().items()},
    }


def make_plot(df: pd.DataFrame, stats: dict) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    colors = {12: "#3B6FB6", 16: "#D98C28", 24: "#4E9A51", 40: "#8B5FBF"}
    markers = {12: "o", 16: "s", 24: "^", 40: "D"}

    fig, ax = plt.subplots(figsize=(4.75, 3.35), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for n in [12, 16, 24, 40]:
        sub = df[df[N_COL] == n]
        ax.scatter(
            sub[X_COL],
            sub[Y_COL],
            s=42,
            marker=markers[n],
            color=colors[n],
            edgecolor="white",
            linewidth=0.55,
            alpha=0.92,
            label=f"N{n}",
            zorder=3,
        )

    x = df[X_COL].to_numpy(dtype=float)
    y = df[Y_COL].to_numpy(dtype=float)
    if len(np.unique(x)) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(
            x_line,
            slope * x_line + intercept,
            color="#5F6368",
            linewidth=1.1,
            alpha=0.65,
            label="linear trend",
            zorder=2,
        )

    ax.text(
        0.04,
        0.96,
        f"Spearman = {stats['spearman']:.4f}\nPearson = {stats['pearson']:.4f}\nn = {stats['n']}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.7,
        bbox={"facecolor": "white", "edgecolor": "#D9D9D9", "boxstyle": "round,pad=0.25", "linewidth": 0.6},
    )

    ax.set_xlabel("Surrogate-predicted terminal reward")
    ax.set_ylabel("Abaqus teacher-derived terminal reward")
    ax.set_title("Surrogate-to-teacher alignment for PPO candidates", pad=19, fontsize=10.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(width=0.8, length=3.5)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.5, alpha=0.55)
    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=5,
        borderaxespad=0.0,
        handletextpad=0.35,
        columnspacing=1.0,
    )

    for path in [PNG_PATH, PDF_PATH, SVG_PATH]:
        backup_existing(path)
    fig.savefig(PNG_PATH, dpi=600, facecolor="white")
    fig.savefig(PDF_PATH, facecolor="white")
    fig.savefig(SVG_PATH, facecolor="white")
    plt.close(fig)


def write_outputs(df: pd.DataFrame, stats: dict, summary: dict) -> None:
    keep_cols = [
        "strategy_name",
        N_COL,
        X_COL,
        Y_COL,
        "teacher_validation_status",
        "ppo_selection_tag",
        "surrogate_false_positive_flag",
        "surrogate_true_positive_flag",
    ]
    df[keep_cols].to_csv(PLOT_DATA_PATH, index=False)

    expected_spearman = summary.get("overall_spearman_predicted_vs_teacher_reward")
    expected_pearson = summary.get("overall_pearson_predicted_vs_teacher_reward")
    if expected_spearman is not None and not math.isclose(stats["spearman"], float(expected_spearman), abs_tol=1e-12):
        raise ValueError("Computed Spearman does not match Stage H summary JSON.")
    if expected_pearson is not None and not math.isclose(stats["pearson"], float(expected_pearson), abs_tol=1e-12):
        raise ValueError("Computed Pearson does not match Stage H summary JSON.")

    report = {
        "input_data_path": str(INPUT_CSV),
        "input_summary_path": str(INPUT_SUMMARY_JSON),
        "candidate_count_used": stats["n"],
        "n_wise_counts": stats["n_counts"],
        "surrogate_reward_column": X_COL,
        "teacher_performance_column": Y_COL,
        "spearman_correlation": stats["spearman"],
        "pearson_correlation": stats["pearson"],
        "generated_outputs": [str(PNG_PATH), str(PDF_PATH), str(SVG_PATH), str(PLOT_DATA_PATH), str(REPORT_PATH)],
        "interpretation": "The weak but positive surrogate-to-teacher alignment indicates that the surrogate provides a useful prioritisation signal for PPO-generated candidates, but final physical evaluation must remain grounded in independent Abaqus teacher validation.",
    }
    with REPORT_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    lines = [
        "# Figure 3.x - PPO Candidate Surrogate-to-Teacher Alignment",
        "",
        "## Input",
        f"- Alignment data: `{INPUT_CSV}`",
        f"- Stage H summary: `{INPUT_SUMMARY_JSON}`",
        "",
        "## Scope",
        f"- Candidates used: {stats['n']}",
        f"- N-wise counts: {', '.join(f'{k}={v}' for k, v in stats['n_counts'].items())}",
        "- Data scope: 32 teacher-validated PPO-generated candidates only.",
        "",
        "## Columns",
        f"- Surrogate-predicted terminal reward: `{X_COL}`",
        f"- Abaqus teacher-derived terminal reward: `{Y_COL}`",
        "",
        "## Correlations",
        f"- Spearman correlation: {stats['spearman']:.4f}",
        f"- Pearson correlation: {stats['pearson']:.4f}",
        "",
        "## Generated Outputs",
        f"- PNG: `{PNG_PATH}`",
        f"- PDF: `{PDF_PATH}`",
        f"- SVG: `{SVG_PATH}`",
        f"- Plot data: `{PLOT_DATA_PATH}`",
        "",
        "## Interpretation",
        "The weak but positive surrogate-to-teacher alignment indicates that the surrogate provides a useful prioritisation signal for PPO-generated candidates, but final physical evaluation must remain grounded in independent Abaqus teacher validation.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df, summary = load_inputs()
    stats = compute_stats(df)
    make_plot(df, stats)
    write_outputs(df, stats, summary)
    print(json.dumps({"status": "OK", **stats, "output_dir": str(OUTPUT_DIR)}, indent=2))


if __name__ == "__main__":
    main()
