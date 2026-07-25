"""Build manuscript Figure 3.1 response-landscape plot data and figure.

This script only reads frozen CSV evidence and writes manuscript figure data.
It does not run Abaqus, open ODB files, generate CAE/INP/JNL files, or modify
teacher metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
FREEZE_DIR = PROJECT_ROOT / "outputs" / "stage3_run_78_final_evidence_freeze_package"
NATIVE552 = FREEZE_DIR / "FROZEN_stage3_native_combined552_teacher_dataset.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "manuscript_figures"

PLOT_DATA_CSV = OUT_DIR / "figure_3_1_response_landscape_plot_data.csv"
SUMMARY_CSV = OUT_DIR / "figure_3_1_response_landscape_summary.csv"
FIG_PNG = OUT_DIR / "figure_3_1_teacher_response_distributions.png"
FIG_PDF = OUT_DIR / "figure_3_1_teacher_response_distributions.pdf"
FIG_SVG = OUT_DIR / "figure_3_1_teacher_response_distributions.svg"
AUDIT_MD = OUT_DIR / "figure_3_1_response_landscape_audit.md"
AUDIT_JSON = OUT_DIR / "figure_3_1_response_landscape_audit.json"

NATIVE_N = [12, 16, 24, 40]
ALL_RESPONSE_N = [12, 16, 24, 40]
EXPECTED_COUNTS = {12: 78, 16: 78, 24: 190, 40: 206}


def first_existing(row: pd.Series, columns: Iterable[str]) -> object:
    for col in columns:
        if col in row.index and pd.notna(row[col]):
            return row[col]
    return np.nan


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def build_plot_data(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for i, row in df.iterrows():
        n = int(row["n"])
        is_native = n in NATIVE_N
        strategy = first_existing(row, ["strategy_name", "handoff_strategy_name", "candidate_id"])
        if pd.isna(strategy) or str(strategy).strip() == "":
            strategy = f"case_{i:04d}"

        surface_mpa = np.nan
        mises_mpa = np.nan
        stress_note = ""
        if is_native:
            surface_pa = first_existing(row, ["surface_t_proxy", "surface_t_proxy_max_tensile_pa"])
            surface_existing_mpa = first_existing(row, ["surface_t_proxy_mpa", "surface_t_proxy_max_tensile_mpa"])
            if pd.notna(surface_existing_mpa):
                surface_mpa = float(surface_existing_mpa)
            elif pd.notna(surface_pa):
                surface_mpa = float(surface_pa) / 1.0e6
            mises_pa = first_existing(row, ["mises_max"])
            if pd.notna(mises_pa):
                mises_mpa = float(mises_pa) / 1.0e6
            stress_note = "native Stage 3 MPa-scale stress definitions"

        rows.append(
            {
                "N_label": f"N{n}",
                "scope": "native_stage3_combined552",
                "strategy_id": str(strategy),
                "U2": float(row["u2_range"]) if pd.notna(row.get("u2_range")) else np.nan,
                "PEEQ": float(row["peeq_max"]) if pd.notna(row.get("peeq_max")) else np.nan,
                "SurfaceT": surface_mpa,
                "Mises": mises_mpa,
                "include_in_u2_panel": bool_text(n in ALL_RESPONSE_N),
                "include_in_peeq_panel": bool_text(n in ALL_RESPONSE_N),
                "include_in_surfacet_panel": bool_text(is_native),
                "include_in_mises_panel": bool_text(is_native),
                "stress_semantics_note": stress_note,
            }
        )
    out = pd.DataFrame(rows)
    out["N_label"] = pd.Categorical(out["N_label"], [f"N{n}" for n in ALL_RESPONSE_N], ordered=True)
    return out.sort_values(["N_label", "strategy_id"]).reset_index(drop=True)


def q(series: pd.Series, prob: float) -> float:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return np.nan
    return float(series.quantile(prob))


def summarize(plot_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label in [f"N{n}" for n in ALL_RESPONSE_N]:
        sub = plot_df[plot_df["N_label"].astype(str) == label]
        rec: dict[str, object] = {"N_label": label, "case_count": int(len(sub))}
        for metric in ["U2", "PEEQ", "SurfaceT", "Mises"]:
            values = pd.to_numeric(sub[metric], errors="coerce").dropna()
            rec[f"median_{metric}"] = float(values.median()) if len(values) else np.nan
            rec[f"q25_{metric}"] = q(values, 0.25)
            rec[f"q75_{metric}"] = q(values, 0.75)
            rec[f"min_{metric}"] = float(values.min()) if len(values) else np.nan
            rec[f"max_{metric}"] = float(values.max()) if len(values) else np.nan
            rec[f"available_count_{metric}"] = int(len(values))
        rows.append(rec)
    return pd.DataFrame(rows)


def panel_boxplot(ax: plt.Axes, plot_df: pd.DataFrame, metric: str, labels: list[str], ylabel: str, panel: str, color: str) -> None:
    rng = np.random.default_rng(20260626)
    data = [pd.to_numeric(plot_df.loc[(plot_df["N_label"].astype(str) == label), metric], errors="coerce").dropna().to_numpy() for label in labels]
    positions = np.arange(1, len(labels) + 1)
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 1.2},
        whiskerprops={"color": "#444444", "linewidth": 0.9},
        capprops={"color": "#444444", "linewidth": 0.9},
        boxprops={"edgecolor": "#333333", "linewidth": 0.9},
    )
    for box in bp["boxes"]:
        box.set_facecolor(color)
        box.set_alpha(0.58)

    for x, values in zip(positions, data):
        if len(values) == 0:
            continue
        jitter = rng.normal(0, 0.045, len(values))
        ax.scatter(np.full(len(values), x) + jitter, values, s=8, alpha=0.32, color="#242424", linewidths=0)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.text(0.02, 0.96, panel, transform=ax.transAxes, va="top", ha="left", fontsize=12, fontweight="bold")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", labelsize=9, width=0.8, length=3)
    if metric == "U2":
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useOffset=False)
    else:
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        ax.yaxis.set_major_formatter(formatter)


def make_figure(plot_df: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.8), constrained_layout=True)
    native_labels = ["N12", "N16", "N24", "N40"]
    panel_boxplot(axes[0, 0], plot_df[plot_df["include_in_u2_panel"] == "true"], "U2", native_labels, "Vertical displacement range, U2", "(a)", "#7aa6c2")
    panel_boxplot(axes[0, 1], plot_df[plot_df["include_in_peeq_panel"] == "true"], "PEEQ", native_labels, "Maximum equivalent plastic strain, PEEQ", "(b)", "#d9a441")
    panel_boxplot(axes[1, 0], plot_df[plot_df["include_in_surfacet_panel"] == "true"], "SurfaceT", native_labels, "Surface tensile-stress index, SurfaceT / MPa", "(c)", "#7fb685")
    panel_boxplot(axes[1, 1], plot_df[plot_df["include_in_mises_panel"] == "true"], "Mises", native_labels, "Maximum Mises stress / MPa", "(d)", "#c97979")
    fig.patch.set_facecolor("white")
    for ax in axes.ravel():
        ax.set_facecolor("white")
    fig.savefig(FIG_PNG, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_PDF, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_SVG, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_audit(plot_df: pd.DataFrame, summary_df: pd.DataFrame) -> dict[str, object]:
    counts = plot_df["N_label"].astype(str).value_counts().sort_index().to_dict()
    native_counts = {k: int(counts.get(f"N{k}", 0)) for k in NATIVE_N}
    missing = {}
    for metric in ["U2", "PEEQ", "SurfaceT", "Mises"]:
        missing[metric] = int(pd.to_numeric(plot_df[metric], errors="coerce").isna().sum())
    panel_counts = {
        "u2": plot_df[plot_df["include_in_u2_panel"] == "true"]["N_label"].astype(str).value_counts().sort_index().to_dict(),
        "peeq": plot_df[plot_df["include_in_peeq_panel"] == "true"]["N_label"].astype(str).value_counts().sort_index().to_dict(),
        "surfacet": plot_df[plot_df["include_in_surfacet_panel"] == "true"]["N_label"].astype(str).value_counts().sort_index().to_dict(),
        "mises": plot_df[plot_df["include_in_mises_panel"] == "true"]["N_label"].astype(str).value_counts().sort_index().to_dict(),
    }
    hard_ok = (
        len(plot_df) == 552
        and counts == {f"N{k}": v for k, v in EXPECTED_COUNTS.items()}
        and "N32" not in counts
        and "N32" not in panel_counts["u2"]
        and "N32" not in panel_counts["peeq"]
        and "N32" not in panel_counts["surfacet"]
        and "N32" not in panel_counts["mises"]
        and missing["U2"] == 0
        and missing["PEEQ"] == 0
        and missing["SurfaceT"] == 0
        and missing["Mises"] == 0
    )
    verdict = "FIGURE_3_1_RESPONSE_LANDSCAPE_READY" if hard_ok else "WARNING_FIGURE_3_1_RESPONSE_LANDSCAPE_REVIEW_REQUIRED"
    audit = {
        "verdict": verdict,
        "total_rows_full_response_landscape_pool": int(len(plot_df)),
        "native_row_counts": native_counts,
        "auxiliary_N32_row_count": 0,
        "panel_counts": panel_counts,
        "missing_values": missing,
        "dropped_rows": 0,
        "N32_removed_from_all_panels": True,
        "plot_data_csv": str(PLOT_DATA_CSV),
        "summary_csv": str(SUMMARY_CSV),
        "figure_png": str(FIG_PNG),
        "figure_pdf": str(FIG_PDF),
        "figure_svg": str(FIG_SVG),
    }
    lines = [
        "# Figure 3.1 Response Landscape Audit",
        "",
        f"- Total rows used in native response-landscape pool: {len(plot_df)}",
        f"- Native row counts: N12={native_counts[12]}, N16={native_counts[16]}, N24={native_counts[24]}, N40={native_counts[40]}",
        "- Auxiliary N32 row count: 0",
        "- Metric completeness:",
        f"  - U2 complete rows: {len(plot_df) - missing['U2']} / {len(plot_df)}",
        f"  - PEEQ complete rows: {len(plot_df) - missing['PEEQ']} / {len(plot_df)}",
        f"  - SurfaceT native MPa rows: {int(summary_df['available_count_SurfaceT'].sum())} / 552 native rows",
        f"  - Mises native MPa rows: {int(summary_df['available_count_Mises'].sum())} / 552 native rows",
        "- N32 is removed from U2 and PEEQ panels: yes",
        "- N32 is removed from SurfaceT/Mises panels: yes",
        f"- Missing values by output column: U2={missing['U2']}, PEEQ={missing['PEEQ']}, SurfaceT={missing['SurfaceT']}, Mises={missing['Mises']}",
        "- Dropped rows: 0",
        "- Stress semantics note: native Stage 3 MPa-scale stress definitions only",
        f"- Final verdict: {verdict}",
    ]
    AUDIT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    AUDIT_JSON.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(NATIVE552)
    plot_df = build_plot_data(df)
    plot_df.to_csv(PLOT_DATA_CSV, index=False)
    summary_df = summarize(plot_df)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    make_figure(plot_df)
    audit = make_audit(plot_df, summary_df)
    print(json.dumps(audit, indent=2))
    return 0 if audit["verdict"] == "FIGURE_3_1_RESPONSE_LANDSCAPE_READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
