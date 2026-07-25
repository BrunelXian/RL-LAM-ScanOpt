"""Train Stage B supervised surrogate reward models for PPO terminal reward.

Allowed scope: ordinary Python supervised learning on frozen native combined552.
Forbidden scope: no Abaqus, ODB, solver, PPO training, RL training, candidate
generation, CAE/INP/JNL generation, commit, or push.
"""

from __future__ import annotations

import json
import math
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
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from surrogate_reward_features import (  # noqa: E402
    feature_schema,
    order_to_features,
    parse_order,
    schema_markdown as feature_schema_markdown,
    validate_order,
)
from surrogate_reward_targets import (  # noqa: E402
    TARGET_COLUMNS,
    add_reward_targets,
    schema_markdown as target_schema_markdown,
)


PROJECT_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
BRANCH = "stage3-variable-n-graph-pointer-init-v01"
NAMESPACE = "stage3_ppo_rl_lam_fea_addendum_v01"
DATASET_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "stage3_run_78_final_evidence_freeze_package"
    / "FROZEN_stage3_native_combined552_teacher_dataset.csv"
)
OUT_DIR = PROJECT_ROOT / "outputs" / NAMESPACE / "surrogate_reward_model"
MODELS_DIR = OUT_DIR / "models"
TABLES_DIR = OUT_DIR / "tables"
REPORTS_DIR = OUT_DIR / "reports"
PLOTS_DIR = OUT_DIR / "plots"
DOCS_DIR = PROJECT_ROOT / "docs" / NAMESPACE
RANDOM_STATE = 20260623
PRIMARY_TARGET = "reward_lex_u2_peeq_surfacet"
DIAGNOSTIC_TARGETS = [
    "reward_u2_primary",
    "cost_u2_norm",
    "cost_peeq_norm",
    "cost_surfacet_norm",
    "cost_mises_norm",
]


def ensure_dirs() -> None:
    for directory in [MODELS_DIR, TABLES_DIR, REPORTS_DIR, PLOTS_DIR, DOCS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def clean_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def corr_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        spearman = None
        pearson = None
    else:
        spearman = clean_float(spearmanr(y_true, y_pred).correlation)
        pearson = clean_float(pearsonr(y_true, y_pred)[0])
    return {
        "spearman": spearman,
        "pearson": pearson,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
    }


def make_models() -> dict[str, Any]:
    return {
        "ExtraTreesRegressor": ExtraTreesRegressor(
            n_estimators=600,
            random_state=RANDOM_STATE,
            min_samples_leaf=2,
            max_features=0.85,
            n_jobs=-1,
        ),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=500,
            random_state=RANDOM_STATE,
            min_samples_leaf=2,
            max_features=0.85,
            n_jobs=-1,
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "ElasticNet": make_pipeline(StandardScaler(), ElasticNet(alpha=0.001, l1_ratio=0.15, random_state=RANDOM_STATE, max_iter=20000)),
    }


def load_training_frame() -> tuple[pd.DataFrame, np.ndarray, list[str], dict[str, object], dict[str, object]]:
    raw = pd.read_csv(DATASET_PATH)
    rows: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    for index, row in raw.iterrows():
        n = int(row["n"])
        order = parse_order(row)
        if not validate_order(n, order):
            raise ValueError(f"Illegal order at dataset row {index}: n={n}, order={order}")
        vectors.append(order_to_features(n, order))
        record = row.to_dict()
        record["source_row_index"] = int(index)
        record["parsed_order"] = json.dumps(order, separators=(",", ":"))
        rows.append(record)

    frame = pd.DataFrame(rows)
    frame, target_schema = add_reward_targets(frame)
    required_targets = [PRIMARY_TARGET, *DIAGNOSTIC_TARGETS]
    frame = frame.dropna(subset=required_targets).reset_index(drop=True)
    x = np.vstack([vectors[int(i)] for i in frame["source_row_index"].to_numpy()])
    schema = feature_schema()
    return frame, x, list(schema["feature_names"]), schema, target_schema


def evaluate_by_n(validation: pd.DataFrame, target: str, pred_col: str) -> pd.DataFrame:
    rows = []
    for n, group in validation.groupby("n"):
        metrics = corr_metrics(group[target].to_numpy(dtype=float), group[pred_col].to_numpy(dtype=float))
        rows.append(
            {
                "n": int(n),
                "validation_rows": int(len(group)),
                "spearman": metrics["spearman"],
                "pearson": metrics["pearson"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
            }
        )
    return pd.DataFrame(rows).sort_values("n")


def topk_quality_by_n(validation: pd.DataFrame, target: str, pred_col: str) -> pd.DataFrame:
    rows = []
    for n, group in validation.groupby("n"):
        group = group.copy()
        k = min(5, len(group))
        predicted_top = group.nlargest(k, pred_col)
        observed_top = group.nlargest(k, target)
        overlap = len(set(predicted_top["source_row_index"]) & set(observed_top["source_row_index"]))
        observed_best = float(group[target].max())
        observed_mean = float(group[target].mean())
        predicted_top_teacher_mean = float(predicted_top[target].mean())
        predicted_top_teacher_best = float(predicted_top[target].max())
        denom = observed_best - observed_mean
        lift_fraction = None if abs(denom) < 1e-12 else (predicted_top_teacher_mean - observed_mean) / denom
        rows.append(
            {
                "n": int(n),
                "validation_rows": int(len(group)),
                "k": int(k),
                "topk_hit_fraction": float(overlap / k),
                "predicted_topk_teacher_reward_mean": predicted_top_teacher_mean,
                "predicted_topk_teacher_reward_best": predicted_top_teacher_best,
                "observed_validation_best_reward": observed_best,
                "observed_validation_mean_reward": observed_mean,
                "predicted_topk_lift_fraction_vs_mean_to_best": clean_float(lift_fraction),
            }
        )
    return pd.DataFrame(rows).sort_values("n")


def choose_best_model(candidate_rows: list[dict[str, Any]], by_n_lookup: dict[str, pd.DataFrame], topk_lookup: dict[str, pd.DataFrame]) -> str:
    def score(row: dict[str, Any]) -> tuple[float, float, float, float]:
        model_name = str(row["model_type"])
        spearman = row["spearman"]
        spearman_score = -999.0 if spearman is None else float(spearman)
        by_n = by_n_lookup[model_name]
        valid_spearman = by_n["spearman"].dropna()
        min_by_n = -999.0 if valid_spearman.empty else float(valid_spearman.min())
        topk = topk_lookup[model_name]
        mean_topk_lift = topk["predicted_topk_lift_fraction_vs_mean_to_best"].dropna().mean()
        if pd.isna(mean_topk_lift):
            mean_topk_lift = -999.0
        rmse = float(row["rmse"])
        return (spearman_score, float(mean_topk_lift), min_by_n, -rmse)

    return str(max(candidate_rows, key=score)["model_type"])


def plot_observed_vs_predicted(validation: pd.DataFrame, target: str, pred_col: str) -> None:
    plt.figure(figsize=(6, 5))
    for n, group in validation.groupby("n"):
        plt.scatter(group[target], group[pred_col], label=f"N{int(n)}", alpha=0.8)
    limits = [
        min(float(validation[target].min()), float(validation[pred_col].min())),
        max(float(validation[target].max()), float(validation[pred_col].max())),
    ]
    plt.plot(limits, limits, color="black", linewidth=1)
    plt.xlabel("Observed teacher-derived reward")
    plt.ylabel("Predicted surrogate reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "observed_vs_predicted_reward.png", dpi=180)
    plt.close()


def plot_rank_by_n(validation: pd.DataFrame, target: str, pred_col: str) -> None:
    plt.figure(figsize=(7, 5))
    for n, group in validation.groupby("n"):
        observed_rank = group[target].rank(ascending=False, method="average")
        predicted_rank = group[pred_col].rank(ascending=False, method="average")
        plt.scatter(observed_rank, predicted_rank, label=f"N{int(n)}", alpha=0.8)
    plt.xlabel("Observed reward rank within holdout N")
    plt.ylabel("Predicted reward rank within holdout N")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "predicted_vs_observed_rank_by_N.png", dpi=180)
    plt.close()


def plot_spearman_by_n(by_n: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 4))
    plt.bar(by_n["n"].astype(str), by_n["spearman"].fillna(0.0))
    plt.ylim(-1.0, 1.0)
    plt.xlabel("N")
    plt.ylabel("Validation Spearman")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "validation_spearman_by_N.png", dpi=180)
    plt.close()


def plot_feature_importance(model: Any, feature_names: list[str]) -> pd.DataFrame | None:
    estimator = model
    if hasattr(estimator, "named_steps"):
        estimator = list(estimator.named_steps.values())[-1]
    if not hasattr(estimator, "feature_importances_"):
        return None
    importance = np.asarray(estimator.feature_importances_, dtype=float)
    table = pd.DataFrame({"feature": feature_names, "importance": importance}).sort_values("importance", ascending=False)
    top = table.head(30).iloc[::-1]
    plt.figure(figsize=(8, 8))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "feature_importance_top30.png", dpi=180)
    plt.close()
    return table


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(str(column) for column in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    selected_model_name: str,
    validation_summary: dict[str, Any],
    by_n: pd.DataFrame,
    topk: pd.DataFrame,
    feature_schema_path: Path,
    target_schema_path: Path,
    model_path: Path,
) -> str:
    min_spearman = by_n["spearman"].dropna().min()
    overall_spearman = validation_summary["overall_metrics"]["spearman"]
    if overall_spearman is not None and overall_spearman >= 0.55 and min_spearman >= 0.25:
        verdict = "PASS_PPO_SURROGATE_REWARD_MODEL_READY_FOR_PPO_TRAINING"
    elif overall_spearman is not None and overall_spearman >= 0.25:
        verdict = "WARNING_PPO_SURROGATE_REWARD_MODEL_PARTIAL_USE_WITH_CAUTION"
    else:
        verdict = "FAIL_PPO_SURROGATE_REWARD_MODEL_NOT_READY"

    lines = [
        "# PPO Surrogate Reward Model Report",
        "",
        "## 1. Purpose",
        "",
        "Train and freeze a supervised terminal reward emulator from the FEA teacher-labelled native combined552 dataset. This surrogate is for later PPO environment rewards; it is not a PPO policy and it is not the physical teacher.",
        "",
        "## 2. Dataset",
        "",
        f"- Dataset: `{DATASET_PATH}`",
        f"- Rows used: `{validation_summary['rows_used']}`",
        f"- Row counts by N: `{validation_summary['row_counts_by_n']}`",
        "- N32 usage: not used",
        "",
        "## 3. Feature Schema",
        "",
        f"- Feature schema JSON: `{feature_schema_path}`",
        f"- Feature count: `{validation_summary['feature_count']}`",
        "",
        "## 4. Target Schema",
        "",
        f"- Target schema JSON: `{target_schema_path}`",
        f"- Primary target: `{PRIMARY_TARGET}`",
        "- Reward direction: larger is better",
        "- Mises role: diagnostic only",
        "",
        "## 5. Train/Validation Split",
        "",
        f"- Split random state: `{RANDOM_STATE}`",
        "- Split type: deterministic stratified split by N",
        f"- Train rows: `{validation_summary['train_rows']}`",
        f"- Validation rows: `{validation_summary['validation_rows']}`",
        "",
        "## 6. Model Candidates",
        "",
        "- ExtraTreesRegressor",
        "- RandomForestRegressor",
        "- GradientBoostingRegressor",
        "- HistGradientBoostingRegressor",
        "- Ridge baseline",
        "- ElasticNet baseline",
        "",
        "## 7. Selected Model",
        "",
        f"- Selected primary model: `{selected_model_name}`",
        f"- Model artifact: `{model_path}`",
        "",
        "## 8. Validation Metrics",
        "",
        f"- Spearman: `{validation_summary['overall_metrics']['spearman']}`",
        f"- Pearson: `{validation_summary['overall_metrics']['pearson']}`",
        f"- MAE: `{validation_summary['overall_metrics']['mae']}`",
        f"- RMSE: `{validation_summary['overall_metrics']['rmse']}`",
        "",
        "## 9. Per-N Stability",
        "",
        dataframe_to_markdown(by_n),
        "",
        "## 10. Top-k Quality",
        "",
        dataframe_to_markdown(topk),
        "",
        "## 11. Known Limitations",
        "",
        "- The model is trained on 552 teacher-labelled native-N examples, so PPO exploration outside the teacher-labelled distribution can be mis-scored.",
        "- The reward is teacher-derived and surrogate-predicted; it is not an Abaqus solve.",
        "- No physical feasibility threshold was invented for strict penalty guards.",
        "- PPO candidates produced later must still be independently validated by Abaqus.",
        "",
        "## 12. Whether Suitable For PPO Training",
        "",
        "The model is suitable as a first surrogate terminal reward for PPO training if used with the claim boundary and later Abaqus teacher validation. PPO training should log all surrogate versions and freeze the reward model artifact used for each policy run.",
        "",
        "## 13. Claim Boundary",
        "",
        "- This surrogate is a terminal reward emulator trained on FEA teacher-labelled scan-order data.",
        "- It is not the physical teacher.",
        "- It is not a PPO policy.",
        "- PPO candidates must still be independently validated by Abaqus.",
        "",
        "## 14. Verdict",
        "",
        f"`{verdict}`",
    ]
    (DOCS_DIR / "PPO_SURROGATE_REWARD_MODEL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return verdict


def main() -> int:
    ensure_dirs()
    timestamp = datetime.now(timezone.utc).isoformat()
    frame, x, feature_names, feature_schema_data, target_schema_data = load_training_frame()
    y = frame[PRIMARY_TARGET].to_numpy(dtype=float)

    train_idx, val_idx = train_test_split(
        np.arange(len(frame)),
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=frame["n"],
    )
    x_train, x_val = x[train_idx], x[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    train_frame = frame.iloc[train_idx].copy()
    val_frame = frame.iloc[val_idx].copy()

    candidates = make_models()
    candidate_rows: list[dict[str, Any]] = []
    by_n_lookup: dict[str, pd.DataFrame] = {}
    topk_lookup: dict[str, pd.DataFrame] = {}
    fitted_models: dict[str, Any] = {}
    for model_name, model in candidates.items():
        fitted = clone(model)
        fitted.fit(x_train, y_train)
        pred = fitted.predict(x_val)
        metrics = corr_metrics(y_val, pred)
        row = {"model_type": model_name, **metrics}
        candidate_rows.append(row)
        tmp = val_frame.copy()
        tmp["prediction"] = pred
        by_n_lookup[model_name] = evaluate_by_n(tmp, PRIMARY_TARGET, "prediction")
        topk_lookup[model_name] = topk_quality_by_n(tmp, PRIMARY_TARGET, "prediction")
        fitted_models[model_name] = fitted

    selected_name = choose_best_model(candidate_rows, by_n_lookup, topk_lookup)
    selected_model = fitted_models[selected_name]
    val_frame["predicted_reward_lex_u2_peeq_surfacet"] = selected_model.predict(x_val)

    diagnostic_models: dict[str, Any] = {}
    diagnostic_metrics: dict[str, dict[str, float | None]] = {}
    diagnostic_predictions: dict[str, np.ndarray] = {}
    for target in DIAGNOSTIC_TARGETS:
        model = clone(candidates[selected_name])
        model.fit(x_train, train_frame[target].to_numpy(dtype=float))
        pred = model.predict(x_val)
        diagnostic_models[target] = model
        diagnostic_metrics[target] = corr_metrics(val_frame[target].to_numpy(dtype=float), pred)
        diagnostic_predictions[target] = pred
        val_frame[f"predicted_{target}"] = pred

    overall_metrics = corr_metrics(y_val, val_frame["predicted_reward_lex_u2_peeq_surfacet"].to_numpy(dtype=float))
    by_n = evaluate_by_n(val_frame, PRIMARY_TARGET, "predicted_reward_lex_u2_peeq_surfacet")
    topk = topk_quality_by_n(val_frame, PRIMARY_TARGET, "predicted_reward_lex_u2_peeq_surfacet")

    row_counts_by_n = {str(int(k)): int(v) for k, v in frame["n"].value_counts().sort_index().items()}
    validation_summary = {
        "timestamp": timestamp,
        "branch": BRANCH,
        "dataset_path": str(DATASET_PATH),
        "rows_used": int(len(frame)),
        "row_counts_by_n": row_counts_by_n,
        "train_rows": int(len(train_frame)),
        "validation_rows": int(len(val_frame)),
        "feature_count": int(x.shape[1]),
        "primary_target": PRIMARY_TARGET,
        "selected_model_type": selected_name,
        "overall_metrics": overall_metrics,
        "diagnostic_metrics": diagnostic_metrics,
        "no_Abaqus": True,
        "no_ODB": True,
        "no_solver": True,
        "no_CAE_INP_JNL": True,
        "no_PPO_training": True,
        "no_candidate_generation": True,
        "no_commit_or_push": True,
    }

    feature_schema_path = MODELS_DIR / "ppo_surrogate_feature_schema.json"
    target_schema_path = MODELS_DIR / "ppo_surrogate_target_schema.json"
    feature_schema_path.write_text(json.dumps(feature_schema_data, indent=2), encoding="utf-8")
    target_schema_path.write_text(json.dumps(target_schema_data, indent=2), encoding="utf-8")
    (MODELS_DIR / "ppo_surrogate_feature_schema.md").write_text(feature_schema_markdown(feature_schema_data) + "\n", encoding="utf-8")
    (MODELS_DIR / "ppo_surrogate_target_schema.md").write_text(target_schema_markdown(target_schema_data) + "\n", encoding="utf-8")
    (DOCS_DIR / "PPO_SURROGATE_FEATURE_SCHEMA.md").write_text(feature_schema_markdown(feature_schema_data) + "\n", encoding="utf-8")
    (DOCS_DIR / "PPO_SURROGATE_TARGET_SCHEMA.md").write_text(target_schema_markdown(target_schema_data) + "\n", encoding="utf-8")

    model_config = {
        "timestamp": timestamp,
        "branch": BRANCH,
        "dataset_path": str(DATASET_PATH),
        "primary_target": PRIMARY_TARGET,
        "diagnostic_targets": DIAGNOSTIC_TARGETS,
        "selected_model_type": selected_name,
        "random_state": RANDOM_STATE,
        "split": "train_test_split(test_size=0.25, stratify=n)",
        "feature_schema_path": str(feature_schema_path),
        "target_schema_path": str(target_schema_path),
    }
    model_config_path = MODELS_DIR / "ppo_surrogate_reward_model_config.json"
    model_config_path.write_text(json.dumps(model_config, indent=2), encoding="utf-8")

    model_path = MODELS_DIR / "ppo_surrogate_reward_model_best.joblib"
    payload = {
        "model_type": selected_name,
        "primary_model": selected_model,
        "diagnostic_models": diagnostic_models,
        "feature_names": feature_names,
        "feature_schema": feature_schema_data,
        "target_schema": target_schema_data,
        "primary_target": PRIMARY_TARGET,
        "diagnostic_targets": DIAGNOSTIC_TARGETS,
        "model_config": model_config,
        "validation_summary": validation_summary,
    }
    joblib.dump(payload, model_path)

    candidate_table = pd.DataFrame(candidate_rows).sort_values(["spearman", "rmse"], ascending=[False, True])
    candidate_table.to_csv(TABLES_DIR / "surrogate_reward_model_candidate_model_metrics.csv", index=False)
    val_out_cols = [
        "source_row_index",
        "n",
        "strategy_name",
        "parsed_order",
        PRIMARY_TARGET,
        "predicted_reward_lex_u2_peeq_surfacet",
        *DIAGNOSTIC_TARGETS,
        *[f"predicted_{target}" for target in DIAGNOSTIC_TARGETS],
    ]
    val_frame[val_out_cols].sort_values(["n", "predicted_reward_lex_u2_peeq_surfacet"], ascending=[True, False]).to_csv(
        TABLES_DIR / "surrogate_reward_model_candidate_predictions_holdout.csv",
        index=False,
    )
    by_n.to_csv(TABLES_DIR / "surrogate_reward_model_validation_by_N.csv", index=False)
    topk.to_csv(TABLES_DIR / "surrogate_reward_model_topk_quality_by_N.csv", index=False)

    importance_table = plot_feature_importance(selected_model, feature_names)
    if importance_table is not None:
        importance_table.to_csv(TABLES_DIR / "surrogate_reward_model_feature_importance.csv", index=False)

    plot_observed_vs_predicted(val_frame, PRIMARY_TARGET, "predicted_reward_lex_u2_peeq_surfacet")
    plot_rank_by_n(val_frame, PRIMARY_TARGET, "predicted_reward_lex_u2_peeq_surfacet")
    plot_spearman_by_n(by_n)

    validation_summary_path = TABLES_DIR / "surrogate_reward_model_validation_summary.json"
    validation_summary_path.write_text(json.dumps(validation_summary, indent=2), encoding="utf-8")
    reports_validation_summary_path = REPORTS_DIR / "surrogate_reward_model_validation_summary.json"
    reports_validation_summary_path.write_text(json.dumps(validation_summary, indent=2), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "selected_model_type": selected_name,
                "primary_target": PRIMARY_TARGET,
                **overall_metrics,
                "train_rows": len(train_frame),
                "validation_rows": len(val_frame),
            }
        ]
    ).to_csv(TABLES_DIR / "surrogate_reward_model_validation_summary.csv", index=False)

    selection_audit = {
        "timestamp": timestamp,
        "candidate_model_metrics": candidate_rows,
        "selected_model_type": selected_name,
        "selection_rule": [
            "validation Spearman for reward_lex_u2_peeq_surfacet",
            "within-N top-k quality",
            "stability across N",
            "no catastrophic failure for any N",
        ],
    }
    (REPORTS_DIR / "surrogate_reward_model_selection_audit.json").write_text(json.dumps(selection_audit, indent=2), encoding="utf-8")

    verdict = write_report(
        selected_model_name=selected_name,
        validation_summary=validation_summary,
        by_n=by_n,
        topk=topk,
        feature_schema_path=feature_schema_path,
        target_schema_path=target_schema_path,
        model_path=model_path,
    )
    validation_summary["final_verdict"] = verdict
    validation_summary_path.write_text(json.dumps(validation_summary, indent=2), encoding="utf-8")
    reports_validation_summary_path.write_text(json.dumps(validation_summary, indent=2), encoding="utf-8")

    claim_boundary = "\n".join(
        [
            "# PPO Reward Model Claim Boundary",
            "",
            "- This surrogate is a terminal reward emulator trained on FEA teacher-labelled scan-order data.",
            "- It is not the physical teacher.",
            "- It is not a PPO policy.",
            "- PPO candidates must still be independently validated by Abaqus.",
            "- Strong PPO claims require PPO-only candidate generation plus Abaqus validation.",
            "- The current artifact supports only a supervised surrogate reward-model claim.",
            "- No online Abaqus PPO, PPO training, or PPO candidate generation was performed in Stage B.",
            "",
        ]
    )
    (DOCS_DIR / "PPO_REWARD_MODEL_CLAIM_BOUNDARY.md").write_text(claim_boundary, encoding="utf-8")

    manifest = {
        "branch": BRANCH,
        "timestamp": timestamp,
        "dataset_path": str(DATASET_PATH),
        "dataset_row_counts_by_N": row_counts_by_n,
        "feature_file_paths": {
            "feature_builder": str(SRC_DIR / "surrogate_reward_features.py"),
            "feature_schema_json": str(feature_schema_path),
            "feature_schema_md_model_copy": str(MODELS_DIR / "ppo_surrogate_feature_schema.md"),
            "feature_schema_md": str(DOCS_DIR / "PPO_SURROGATE_FEATURE_SCHEMA.md"),
        },
        "target_schema_paths": {
            "target_builder": str(SRC_DIR / "surrogate_reward_targets.py"),
            "target_schema_json": str(target_schema_path),
            "target_schema_md_model_copy": str(MODELS_DIR / "ppo_surrogate_target_schema.md"),
            "target_schema_md": str(DOCS_DIR / "PPO_SURROGATE_TARGET_SCHEMA.md"),
        },
        "model_artifact_paths": {
            "best_model": str(model_path),
            "model_config": str(model_config_path),
            "reward_model_interface": str(SRC_DIR / "ppo_surrogate_reward_model.py"),
        },
        "validation_report_paths": {
            "model_report": str(DOCS_DIR / "PPO_SURROGATE_REWARD_MODEL_REPORT.md"),
            "claim_boundary": str(DOCS_DIR / "PPO_REWARD_MODEL_CLAIM_BOUNDARY.md"),
            "validation_summary_csv": str(TABLES_DIR / "surrogate_reward_model_validation_summary.csv"),
            "validation_summary_json": str(TABLES_DIR / "surrogate_reward_model_validation_summary.json"),
            "validation_summary_json_report_copy": str(REPORTS_DIR / "surrogate_reward_model_validation_summary.json"),
            "validation_by_N_csv": str(TABLES_DIR / "surrogate_reward_model_validation_by_N.csv"),
            "topk_quality_by_N_csv": str(TABLES_DIR / "surrogate_reward_model_topk_quality_by_N.csv"),
            "holdout_predictions_csv": str(TABLES_DIR / "surrogate_reward_model_candidate_predictions_holdout.csv"),
            "selection_audit_json": str(REPORTS_DIR / "surrogate_reward_model_selection_audit.json"),
        },
        "selected_model_type": selected_name,
        "validation_metrics": overall_metrics,
        "no_Abaqus": True,
        "no_ODB": True,
        "no_solver": True,
        "no_CAE_INP_JNL": True,
        "no_PPO_training": True,
        "no_candidate_generation": True,
        "no_commit_or_push": True,
        "final_verdict": verdict,
    }
    manifest_path = OUT_DIR / "ppo_surrogate_reward_model_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "verdict": verdict,
                "selected_model_type": selected_name,
                "overall_metrics": overall_metrics,
                "model_path": str(model_path),
                "manifest_path": str(manifest_path),
            },
            indent=2,
        )
    )
    return 0 if not verdict.startswith("FAIL") else 1


if __name__ == "__main__":
    raise SystemExit(main())
