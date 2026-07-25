"""Train PPO v03 lex-primary N-specific surrogate/ranking models."""

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
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
NS = "stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40"
V01_SRC = PROJECT_ROOT / "stage3_ppo_rl_lam_fea_addendum_v01" / "src"
if str(V01_SRC) not in sys.path:
    sys.path.insert(0, str(V01_SRC))

from surrogate_reward_features import feature_schema, order_to_features, parse_order  # noqa: E402

OUT_ROOT = PROJECT_ROOT / "outputs" / NS
DATASET = OUT_ROOT / "data" / "v03_N24_N40_teacher_dataset.csv"
SURR_DIR = OUT_ROOT / "surrogate_v03"
MODELS_DIR = SURR_DIR / "models"
TABLES_DIR = SURR_DIR / "tables"
PLOTS_DIR = SURR_DIR / "plots"
DOCS_DIR = PROJECT_ROOT / "docs" / NS
REPORT = DOCS_DIR / "PPO_V03_LEX_PRIMARY_SURROGATE_REPORT.md"
VALIDATION_CSV = TABLES_DIR / "v03_surrogate_validation_by_N_target.csv"
SELECTION_JSON = TABLES_DIR / "v03_surrogate_model_selection_summary.json"
FALSE_POSITIVE_CSV = TABLES_DIR / "v03_surrogate_false_positive_audit.csv"
TARGETS = ["reward_lex_primary_v03", "reward_u2_guarded_v03", "reward_record_seeking_v03"]
SEED = 20260627


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
    return x, df.loc[valid_idx].reset_index(drop=True), feature_schema()["feature_names"]


def regressors() -> dict[str, Any]:
    return {
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(random_state=SEED, max_iter=300, learning_rate=0.04, l2_regularization=0.02),
        "ExtraTreesRegressor": ExtraTreesRegressor(n_estimators=500, random_state=SEED, min_samples_leaf=2, n_jobs=-1),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=350, random_state=SEED, min_samples_leaf=2, n_jobs=-1),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=SEED, n_estimators=300, learning_rate=0.035, max_depth=3),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    }


def classifiers() -> dict[str, Any]:
    return {
        "RandomForestClassifier": RandomForestClassifier(n_estimators=400, random_state=SEED, min_samples_leaf=2, class_weight="balanced", n_jobs=-1),
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)),
    }


def corr(y_true: np.ndarray, y_pred: np.ndarray, method: str) -> float:
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return float("nan")
    if method == "spearman":
        return float(spearmanr(y_true, y_pred).correlation)
    return float(pearsonr(y_true, y_pred).statistic)


def top_hit_quality(y_true: np.ndarray, y_pred: np.ndarray, frac: float) -> float:
    k = max(1, int(np.ceil(frac * len(y_true))))
    true_top = set(np.argsort(-y_true)[:k])
    pred_top = set(np.argsort(-y_pred)[:k])
    return len(true_top & pred_top) / k


def train_regression_target(n: int, target: str, x: np.ndarray, df: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    y = df[target].to_numpy(dtype=float)
    rows = []
    trained = {}
    for name, model in regressors().items():
        model.fit(x[train_idx], y[train_idx])
        pred = np.asarray(model.predict(x[val_idx]), dtype=float)
        rows.append(
            {
                "n": n,
                "target": target,
                "model_name": name,
                "train_count": int(len(train_idx)),
                "validation_count": int(len(val_idx)),
                "spearman": corr(y[val_idx], pred, "spearman"),
                "pearson": corr(y[val_idx], pred, "pearson"),
                "mae": float(mean_absolute_error(y[val_idx], pred)),
                "rmse": float(mean_squared_error(y[val_idx], pred) ** 0.5),
                "top10_hit_quality": top_hit_quality(y[val_idx], pred, 0.10),
                "top25_hit_quality": top_hit_quality(y[val_idx], pred, 0.25),
            }
        )
        trained[name] = model
    ranking = sorted(rows, key=lambda r: (np.nan_to_num(r["spearman"], nan=-999.0), r["top25_hit_quality"], -r["rmse"]), reverse=True)
    best_name = ranking[0]["model_name"]
    ensemble_names = [r["model_name"] for r in ranking[: min(4, len(ranking))]]
    ensemble_preds = np.asarray([trained[name].predict(x[val_idx]) for name in ensemble_names], dtype=float)
    conservative = ensemble_preds.mean(axis=0) - 0.5 * ensemble_preds.std(axis=0)
    payload = {
        "best_model_name": best_name,
        "best_model": trained[best_name],
        "ensemble_model_names": ensemble_names,
        "ensemble_models": [trained[name] for name in ensemble_names],
        "validation_rows": rows,
    }
    return payload, rows, conservative


def train_classifier(n: int, x: np.ndarray, df: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray) -> tuple[Any | None, list[dict[str, Any]]]:
    y = df["reward_topk_classifier_v03"].to_numpy(dtype=int)
    rows = []
    if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[val_idx])) < 2:
        return None, [{"n": n, "target": "reward_topk_classifier_v03", "model_name": "NOT_TRAINED_SINGLE_CLASS", "accuracy": np.nan, "roc_auc": np.nan, "precision": np.nan, "recall": np.nan}]
    trained = {}
    for name, clf in classifiers().items():
        clf.fit(x[train_idx], y[train_idx])
        pred = np.asarray(clf.predict(x[val_idx]), dtype=int)
        if hasattr(clf, "predict_proba"):
            proba = np.asarray(clf.predict_proba(x[val_idx]))[:, -1]
        else:
            proba = pred.astype(float)
        rows.append(
            {
                "n": n,
                "target": "reward_topk_classifier_v03",
                "model_name": name,
                "train_count": int(len(train_idx)),
                "validation_count": int(len(val_idx)),
                "accuracy": float(accuracy_score(y[val_idx], pred)),
                "roc_auc": float(roc_auc_score(y[val_idx], proba)),
                "precision": float(precision_score(y[val_idx], pred, zero_division=0)),
                "recall": float(recall_score(y[val_idx], pred, zero_division=0)),
            }
        )
        trained[name] = clf
    ranking = sorted(rows, key=lambda r: (np.nan_to_num(r["roc_auc"], nan=-999.0), r["recall"], r["precision"]), reverse=True)
    return trained[ranking[0]["model_name"]], rows


def train_one_n(n: int, df: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    sub = df[df["n"].astype(int) == n].copy().reset_index(drop=True)
    x, sub, feature_names = build_xy(sub)
    idx = np.arange(len(sub))
    strat = sub["reward_topk_classifier_v03"].astype(int) if sub["reward_topk_classifier_v03"].nunique() > 1 else None
    train_idx, val_idx = train_test_split(idx, test_size=0.22, random_state=SEED, shuffle=True, stratify=strat)
    target_models: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    conservative_preds: dict[str, np.ndarray] = {}
    for target in TARGETS:
        payload, rows, conservative = train_regression_target(n, target, x, sub, train_idx, val_idx)
        target_models[target] = payload
        all_rows.extend(rows)
        conservative_preds[target] = conservative
    classifier, class_rows = train_classifier(n, x, sub, train_idx, val_idx)
    all_rows.extend(class_rows)

    fp_rows = []
    lex_pred = conservative_preds["reward_lex_primary_v03"]
    u2_pred = conservative_preds["reward_u2_guarded_v03"]
    val = sub.iloc[val_idx].copy().reset_index(drop=True)
    for i, row in val.iterrows():
        high_pred = lex_pred[i] >= np.quantile(lex_pred, 0.75)
        poor_lex = row["lex_percentile_u2_peeq_surfacet"] > 0.50
        poor_u2 = row["percentile_u2"] > 0.50
        surface_only = bool(row["surfaceT_only_false_positive_teacher"])
        if high_pred and (poor_lex or poor_u2 or surface_only):
            fp_rows.append(
                {
                    "n": n,
                    "strategy_name": row["strategy_name"],
                    "dataset_source": row["dataset_source"],
                    "predicted_lex_primary_conservative": float(lex_pred[i]),
                    "predicted_u2_guarded_conservative": float(u2_pred[i]),
                    "teacher_lex_percentile": float(row["lex_percentile_u2_peeq_surfacet"]),
                    "teacher_u2_percentile": float(row["percentile_u2"]),
                    "teacher_surfaceT_percentile": float(row["percentile_surface_t"]),
                    "surfaceT_only_false_positive_teacher": surface_only,
                }
            )

    payload = {
        "version": "v03_lex_primary_N24_N40",
        "n": n,
        "targets": TARGETS,
        "target_models": target_models,
        "top25_classifier": classifier,
        "feature_schema": feature_schema(),
        "feature_names": feature_names,
        "training_rows": int(len(sub)),
        "train_indices": train_idx.tolist(),
        "validation_indices": val_idx.tolist(),
        "final_formula": "0.7*(0.6*lex_primary_conservative + 0.3*u2_guarded_conservative + 0.1*record_conservative - surface_only_penalty) + 0.3*top25_probability",
        "false_positive_audit_rows": fp_rows,
    }
    joblib.dump(payload, MODELS_DIR / f"N{n}_v03_lex_primary_surrogate.joblib")

    fig, ax = plt.subplots(figsize=(5, 4))
    y = sub.iloc[val_idx]["reward_lex_primary_v03"].to_numpy(dtype=float)
    ax.scatter(y, conservative_preds["reward_lex_primary_v03"], alpha=0.75)
    ax.set_xlabel("teacher lex-primary reward")
    ax.set_ylabel("conservative predicted reward")
    ax.set_title(f"N{n} v03 lex-primary surrogate")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"N{n}_v03_lex_primary_observed_vs_predicted.png", dpi=180)
    plt.close(fig)
    return payload, all_rows, fp_rows


def main() -> int:
    ensure_dirs()
    df = pd.read_csv(DATASET)
    rows: list[dict[str, Any]] = []
    fp_rows: list[dict[str, Any]] = []
    payload_summary: dict[str, Any] = {}
    for n in [24, 40]:
        payload, val_rows, fp = train_one_n(n, df)
        rows.extend(val_rows)
        fp_rows.extend(fp)
        payload_summary[str(n)] = {
            "model_path": str(MODELS_DIR / f"N{n}_v03_lex_primary_surrogate.joblib"),
            "training_rows": payload["training_rows"],
            "classifier_available": payload["top25_classifier"] is not None,
            "false_positive_validation_count": len(fp),
        }
    val = pd.DataFrame(rows)
    val.to_csv(VALIDATION_CSV, index=False)
    pd.DataFrame(fp_rows).to_csv(FALSE_POSITIVE_CSV, index=False)
    best_rows = []
    for n in [24, 40]:
        for target in TARGETS:
            sub = val[(val["n"] == n) & (val["target"] == target)].copy()
            best_rows.append(sub.sort_values(["spearman", "top25_hit_quality"], ascending=[False, False]).iloc[0].to_dict())
    best = pd.DataFrame(best_rows)
    min_spearman = float(best["spearman"].min())
    verdict = "PASS_V03_LEX_PRIMARY_SURROGATES_READY_FOR_PPO" if min_spearman >= 0.55 else "WARNING_V03_LEX_PRIMARY_SURROGATES_PARTIAL_REVIEW"
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "payloads": payload_summary,
        "validation_csv": str(VALIDATION_CSV),
        "false_positive_audit_csv": str(FALSE_POSITIVE_CSV),
        "best_rows": best.to_dict("records"),
        "no_Abaqus": True,
        "no_ODB_opening": True,
        "no_solver": True,
    }
    SELECTION_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    n24_lex = best[(best["n"] == 24) & (best["target"] == "reward_lex_primary_v03")].iloc[0]
    n40_lex = best[(best["n"] == 40) & (best["target"] == "reward_lex_primary_v03")].iloc[0]
    REPORT.write_text(
        f"""# PPO v03 Lex-Primary Surrogate Report

## Purpose
Train N-specific v03 surrogate/ranking models that prioritize U2 and lexicographic U2->PEEQ->SurfaceT reward while penalizing SurfaceT-only false positives.

## Dataset
Input: `{DATASET}`. N24/N40 only; combined552 + teacher-validated PPO v01 + teacher-validated PPO v02K2.

## Targets
- `reward_lex_primary_v03`
- `reward_u2_guarded_v03`
- `reward_record_seeking_v03`
- `reward_topk_classifier_v03`

## Best Lex-Primary Validation
- N24: {n24_lex['model_name']}, Spearman {n24_lex['spearman']:.4f}, Pearson {n24_lex['pearson']:.4f}, MAE {n24_lex['mae']:.4f}, RMSE {n24_lex['rmse']:.4f}, top25 hit quality {n24_lex['top25_hit_quality']:.4f}
- N40: {n40_lex['model_name']}, Spearman {n40_lex['spearman']:.4f}, Pearson {n40_lex['pearson']:.4f}, MAE {n40_lex['mae']:.4f}, RMSE {n40_lex['rmse']:.4f}, top25 hit quality {n40_lex['top25_hit_quality']:.4f}

## SurfaceT False-Positive Audit
Validation false-positive audit: `{FALSE_POSITIVE_CSV}`.

## Verdict
`{verdict}`
""",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0 if not verdict.startswith("FAIL") else 1


if __name__ == "__main__":
    raise SystemExit(main())
