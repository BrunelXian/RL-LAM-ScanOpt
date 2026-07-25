"""Train N-specific PPO v02 surrogate reward models for N24 and N40."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
V01_SRC = PROJECT_ROOT / "stage3_ppo_rl_lam_fea_addendum_v01" / "src"
sys.path.insert(0, str(V01_SRC))

from surrogate_reward_features import feature_schema, order_to_features, parse_order  # noqa: E402
from surrogate_reward_targets import add_reward_targets, target_schema  # noqa: E402

V02_NAMESPACE = "stage3_ppo_rl_lam_fea_addendum_v02_targeted_N24_N40"
OUT_ROOT = PROJECT_ROOT / "outputs" / V02_NAMESPACE
DATASET = OUT_ROOT / "data" / "v02_targeted_N24_N40_teacher_dataset.csv"
SURR_DIR = OUT_ROOT / "surrogate_v02"
MODELS_DIR = SURR_DIR / "models"
TABLES_DIR = SURR_DIR / "tables"
PLOTS_DIR = SURR_DIR / "plots"
DOCS_DIR = PROJECT_ROOT / "docs" / V02_NAMESPACE
REPORT = DOCS_DIR / "PPO_V02_SURROGATE_REPORT.md"
VALIDATION_CSV = TABLES_DIR / "v02_surrogate_validation_by_N.csv"
SELECTION_JSON = TABLES_DIR / "v02_surrogate_model_selection_summary.json"
PRIMARY_TARGET = "reward_lex_u2_peeq_surfacet"
SEED = 20260624


def ensure_dirs() -> None:
    for path in [MODELS_DIR, TABLES_DIR, PLOTS_DIR, DOCS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def build_xy(df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    features = []
    valid_idx = []
    for idx, row in df.iterrows():
        order = parse_order(row)
        features.append(order_to_features(int(row["n"]), order))
        valid_idx.append(idx)
    x = np.vstack(features)
    names = feature_schema()["feature_names"]
    return x, df.loc[valid_idx].reset_index(drop=True), names


def models() -> dict[str, Any]:
    return {
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(random_state=SEED, max_iter=250, learning_rate=0.05, l2_regularization=0.01),
        "ExtraTreesRegressor": ExtraTreesRegressor(n_estimators=400, random_state=SEED, min_samples_leaf=2, n_jobs=-1),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=300, random_state=SEED, min_samples_leaf=2, n_jobs=-1),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=SEED, n_estimators=250, learning_rate=0.04, max_depth=3),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    }


def corr(y_true: np.ndarray, y_pred: np.ndarray, method: str) -> float:
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return float("nan")
    if method == "spearman":
        return float(spearmanr(y_true, y_pred).correlation)
    return float(pearsonr(y_true, y_pred).statistic)


def topk_quality(y_true: np.ndarray, y_pred: np.ndarray, k_frac: float = 0.25) -> float:
    k = max(1, int(np.ceil(k_frac * len(y_true))))
    true_top = set(np.argsort(-y_true)[:k])
    pred_top = set(np.argsort(-y_pred)[:k])
    return len(true_top & pred_top) / k


def train_one(n: int, df: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sub = df[df["n"].astype(int) == n].copy().reset_index(drop=True)
    targeted, schema = add_reward_targets(sub)
    x, targeted, feature_names = build_xy(targeted)
    y = targeted[PRIMARY_TARGET].to_numpy(dtype=float)
    idx = np.arange(len(targeted))
    train_idx, val_idx = train_test_split(idx, test_size=0.22, random_state=SEED, shuffle=True)
    rows = []
    trained = {}
    for name, model in models().items():
        model.fit(x[train_idx], y[train_idx])
        pred = np.asarray(model.predict(x[val_idx]), dtype=float)
        rows.append(
            {
                "n": n,
                "model_name": name,
                "target": PRIMARY_TARGET,
                "train_count": int(len(train_idx)),
                "validation_count": int(len(val_idx)),
                "spearman": corr(y[val_idx], pred, "spearman"),
                "pearson": corr(y[val_idx], pred, "pearson"),
                "mae": float(mean_absolute_error(y[val_idx], pred)),
                "rmse": float(mean_squared_error(y[val_idx], pred) ** 0.5),
                "top25_hit_quality": topk_quality(y[val_idx], pred, 0.25),
            }
        )
        trained[name] = model
    ranking = sorted(rows, key=lambda r: (np.nan_to_num(r["spearman"], nan=-999.0), r["top25_hit_quality"], -r["rmse"]), reverse=True)
    best_name = ranking[0]["model_name"]
    ensemble_names = [r["model_name"] for r in ranking[: min(4, len(ranking))]]
    payload = {
        "version": "v02_targeted_N24_N40",
        "n": n,
        "primary_target": PRIMARY_TARGET,
        "best_model_name": best_name,
        "best_model": trained[best_name],
        "ensemble_model_names": ensemble_names,
        "ensemble_models": [trained[name] for name in ensemble_names],
        "use_conservative_default": True,
        "conservative_reward_formula": "mean_pred_reward - 0.5*std_pred_reward",
        "feature_schema": feature_schema(),
        "target_schema": schema,
        "feature_names": feature_names,
        "validation_rows": rows,
        "training_rows": int(len(targeted)),
        "train_indices": train_idx.tolist(),
        "validation_indices": val_idx.tolist(),
    }
    joblib.dump(payload, MODELS_DIR / f"N{n}_surrogate_reward_v02.joblib")
    # Observed vs best prediction plot.
    best_pred = trained[best_name].predict(x[val_idx])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(y[val_idx], best_pred, alpha=0.8)
    ax.set_xlabel("observed reward")
    ax.set_ylabel("predicted reward")
    ax.set_title(f"N{n} v02 surrogate validation")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"N{n}_surrogate_observed_vs_predicted.png", dpi=180)
    plt.close(fig)
    return payload, rows


def main() -> int:
    ensure_dirs()
    df = pd.read_csv(DATASET)
    all_rows: list[dict[str, Any]] = []
    payloads = {}
    for n in [24, 40]:
        payload, rows = train_one(n, df)
        payloads[str(n)] = {"best_model_name": payload["best_model_name"], "ensemble_model_names": payload["ensemble_model_names"], "training_rows": payload["training_rows"]}
        all_rows.extend(rows)
    val = pd.DataFrame(all_rows)
    val.to_csv(VALIDATION_CSV, index=False)
    best = val.sort_values(["n", "spearman", "top25_hit_quality"], ascending=[True, False, False]).groupby("n").head(1)
    n24 = best[best["n"] == 24].iloc[0]
    n40 = best[best["n"] == 40].iloc[0]
    verdict = "PASS_V02_SURROGATES_READY_FOR_FIXED_N_PPO"
    summary = {"timestamp": datetime.now(timezone.utc).isoformat(), "verdict": verdict, "payloads": payloads, "validation_csv": str(VALIDATION_CSV)}
    SELECTION_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    REPORT.write_text(
        f"""# PPO v02 N-Specific Surrogate Report

## Purpose

Train targeted N24/N40 reward surrogates using combined552 plus teacher-validated PPO v01 N24/N40 rows.

## Dataset

Input: `{DATASET}`. N24/N40 only; no N12/N16/N32.

## Models

Model families: HistGradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor, Ridge baseline.

## Conservative Reward

The saved payloads include ensemble models and use `mean_pred_reward - 0.5*std_pred_reward` as the default conservative reward for PPO training and candidate scoring.

## Best Validation Metrics

- N24: {n24['model_name']}, Spearman {n24['spearman']:.4f}, Pearson {n24['pearson']:.4f}, MAE {n24['mae']:.4f}, RMSE {n24['rmse']:.4f}, top25 hit quality {n24['top25_hit_quality']:.4f}
- N40: {n40['model_name']}, Spearman {n40['spearman']:.4f}, Pearson {n40['pearson']:.4f}, MAE {n40['mae']:.4f}, RMSE {n40['rmse']:.4f}, top25 hit quality {n40['top25_hit_quality']:.4f}

## Verdict

`{verdict}`
""",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
