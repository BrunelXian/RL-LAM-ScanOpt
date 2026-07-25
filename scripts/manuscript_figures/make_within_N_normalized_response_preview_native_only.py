from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
INPUT_CSV = ROOT / "outputs" / "manuscript_figures" / "figure_3_1_response_landscape_plot_data.csv"
OUT_DIR = ROOT / "outputs" / "manuscript_figures"
PLOT_DATA = OUT_DIR / "figure_Sx_within_N_normalized_response_plot_data_native_only.csv"
SUMMARY_CSV = OUT_DIR / "figure_Sx_within_N_normalized_response_summary_native_only.csv"
PNG = OUT_DIR / "figure_Sx_within_N_normalized_teacher_response_distributions_native_only.png"
PDF = OUT_DIR / "figure_Sx_within_N_normalized_teacher_response_distributions_native_only.pdf"
SVG = OUT_DIR / "figure_Sx_within_N_normalized_teacher_response_distributions_native_only.svg"
AUDIT = OUT_DIR / "figure_Sx_within_N_normalized_teacher_response_distributions_native_only_audit.md"

N_ORDER = ["N12", "N16", "N24", "N40"]
METRICS = [
    ("U2", "Vertical displacement range, U2\nwithin-N normalized cost"),
    ("PEEQ", "Maximum equivalent plastic strain, PEEQ\nwithin-N normalized cost"),
    ("SurfaceT", "Surface tensile-stress index, SurfaceT / MPa\nwithin-N normalized cost"),
    ("Mises", "Maximum Mises stress / MPa\nwithin-N normalized cost"),
]
COLORS = {
    "N12": "#4C78A8",
    "N16": "#E19735",
    "N24": "#59A14F",
    "N40": "#8B63B6",
}


def minmax_within_n(df: pd.DataFrame, metric: str) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for n_label, sub in df.groupby("N_label", sort=False, observed=True):
        vals = pd.to_numeric(sub[metric], errors="coerce")
        lo = vals.min()
        hi = vals.max()
        if not np.isfinite(lo) or not np.isfinite(hi):
            continue
        if hi == lo:
            out.loc[sub.index] = 0.0
        else:
            out.loc[sub.index] = (vals - lo) / (hi - lo)
    return out


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_CSV)
    df = df[df["N_label"].isin(N_ORDER)].copy()
    df["N_label"] = pd.Categorical(df["N_label"], categories=N_ORDER, ordered=True)
    df = df.sort_values(["N_label", "strategy_id"]).reset_index(drop=True)

    if len(df) != 552:
        raise ValueError(f"Expected 552 native rows, found {len(df)}")
    counts = df["N_label"].value_counts().reindex(N_ORDER).astype(int).to_dict()
    expected_counts = {"N12": 78, "N16": 78, "N24": 190, "N40": 206}
    if counts != expected_counts:
        raise ValueError(f"Unexpected N counts: {counts}")

    for metric, _ in METRICS:
        df[f"{metric}_within_N_minmax_cost"] = minmax_within_n(df, metric)

    df.to_csv(PLOT_DATA, index=False)

    summary_rows = []
    for n_label in N_ORDER:
        sub = df[df["N_label"] == n_label]
        row = {"N_label": n_label, "case_count": int(len(sub))}
        for metric, _ in METRICS:
            col = f"{metric}_within_N_minmax_cost"
            vals = sub[col].dropna()
            row[f"{metric}_norm_min"] = float(vals.min())
            row[f"{metric}_norm_q25"] = float(vals.quantile(0.25))
            row[f"{metric}_norm_median"] = float(vals.median())
            row[f"{metric}_norm_q75"] = float(vals.quantile(0.75))
            row[f"{metric}_norm_max"] = float(vals.max())
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV, index=False)

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    rng = np.random.default_rng(20260628)
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.45), sharey=True)
    axes = axes.ravel()
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    for ax, (metric, ylabel), panel_label in zip(axes, METRICS, panel_labels):
        col = f"{metric}_within_N_minmax_cost"
        data = [df.loc[df["N_label"] == n_label, col].dropna().to_numpy() for n_label in N_ORDER]
        positions = np.arange(1, len(N_ORDER) + 1)
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#222222", "linewidth": 1.2},
            boxprops={"linewidth": 0.9, "color": "#333333"},
            whiskerprops={"linewidth": 0.8, "color": "#333333"},
            capprops={"linewidth": 0.8, "color": "#333333"},
        )
        for patch, n_label in zip(bp["boxes"], N_ORDER):
            patch.set_facecolor(COLORS[n_label])
            patch.set_alpha(0.28)

        for i, n_label in enumerate(N_ORDER, start=1):
            vals = df.loc[df["N_label"] == n_label, col].dropna().to_numpy()
            if len(vals) > 0:
                sample = vals if len(vals) <= 90 else rng.choice(vals, size=90, replace=False)
                jitter = rng.normal(0.0, 0.035, size=len(sample))
                ax.scatter(
                    np.full(len(sample), i) + jitter,
                    sample,
                    s=7,
                    color=COLORS[n_label],
                    alpha=0.32,
                    linewidth=0,
                    zorder=2,
                )

        ax.set_xticks(positions)
        ax.set_xticklabels(N_ORDER)
        ax.set_ylim(-0.04, 1.04)
        ax.set_ylabel(ylabel)
        ax.text(0.015, 0.97, panel_label, transform=ax.transAxes, ha="left", va="top", fontweight="bold")
        ax.grid(axis="y", color="#E8E8E8", linewidth=0.55)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.085, top=0.975, wspace=0.18, hspace=0.30)
    fig.savefig(PNG, dpi=600, facecolor="white")
    fig.savefig(PDF, facecolor="white")
    fig.savefig(SVG, facecolor="white")
    plt.close(fig)

    audit = [
        "# Within-N Normalized Teacher-Response Preview Audit",
        "",
        f"- Input: `{INPUT_CSV}`",
        f"- Rows used: {len(df)}",
        f"- N counts: {', '.join(f'{k}={v}' for k, v in counts.items())}",
        "- N32 included: no",
        "- Normalization: min-max cost computed separately within each N and metric; 0 is best/lower response, 1 is worst/higher response.",
        "- Metrics: U2, PEEQ, SurfaceT, Mises.",
        f"- Plot data: `{PLOT_DATA}`",
        f"- Summary CSV: `{SUMMARY_CSV}`",
        f"- PNG: `{PNG}`",
        f"- PDF: `{PDF}`",
        f"- SVG: `{SVG}`",
        "- Verdict: WITHIN_N_NORMALIZED_NATIVE_ONLY_PREVIEW_READY",
    ]
    AUDIT.write_text("\n".join(audit) + "\n", encoding="utf-8")
    print(f"Wrote {PNG}")


if __name__ == "__main__":
    main()
