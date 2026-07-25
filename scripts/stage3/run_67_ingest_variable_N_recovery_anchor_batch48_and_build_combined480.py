from __future__ import annotations

import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
RUN_ID = "run_67_variable_N_recovery_anchor_batch48_teacher_metrics_ingestion_and_combined480_ranking"
RUN_NAME = "variable-N recovery anchor batch48 teacher metrics ingestion and combined480 ranking"
SCRIPT_PATH = ROOT / "scripts" / "stage3" / "run_67_ingest_variable_N_recovery_anchor_batch48_and_build_combined480.py"

RUN66_DIR = ROOT / "outputs" / "stage3_run_66_variable_N_recovery_anchor_batch48_odb_teacher_validation"
RUN66_METRICS = RUN66_DIR / "run66_variable_N_recovery_anchor_batch48_teacher_metrics.csv"
RUN66_EXTRACTION = RUN66_DIR / "run66_variable_N_recovery_anchor_batch48_odb_extraction_summary.csv"
RUN66_EXTRACTION_JSON = RUN66_DIR / "run66_variable_N_recovery_anchor_batch48_odb_extraction_summary.json"
RUN66_SOLVER = RUN66_DIR / "run66_variable_N_recovery_anchor_batch48_solver_completion_audit.csv"
RUN66_SOLVER_JSON = RUN66_DIR / "run66_variable_N_recovery_anchor_batch48_solver_completion_audit.json"
RUN66_SUMMARY = RUN66_DIR / "run66_variable_N_recovery_anchor_batch48_odb_teacher_validation_summary.json"
RUN66_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_66_variable_N_recovery_anchor_batch48_odb_teacher_validation" / "RUN_66_VARIABLE_N_RECOVERY_ANCHOR_BATCH48_ODB_TEACHER_VALIDATION_REPORT.md"
RUN66_MANIFEST = ROOT / "artifacts" / "manifests" / "stage3_run_66_manifest.json"

RUN64_HANDOFF = ROOT / "outputs" / "stage3_run_64_run63_variable_N_recovery_anchor_batch48_handoff_package" / "stage3_run64_variable_N_recovery_anchor_batch48_candidate_orders.csv"
RUN64_SCAN_DIR = ROOT / "outputs" / "stage3_run_64_run63_variable_N_recovery_anchor_batch48_handoff_package" / "scan_orders"
RUN63_POOL = ROOT / "outputs" / "stage3_run_63_combined432_model_update_N24_N40_evidence_freeze_and_N12_N16_recovery_candidate_generation" / "run63_candidate_pool_scored.csv"
RUN63_OPTION_A = ROOT / "outputs" / "stage3_run_63_combined432_model_update_N24_N40_evidence_freeze_and_N12_N16_recovery_candidate_generation" / "run63_variable_N_recovery_anchor_batch48_candidate_orders.csv"
RUN63_EVIDENCE_FREEZE = ROOT / "outputs" / "stage3_run_63_combined432_model_update_N24_N40_evidence_freeze_and_N12_N16_recovery_candidate_generation" / "n24_n40_final_active_learning_rl_evidence_freeze.md"
RUN63_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_63_combined432_model_update_N24_N40_evidence_freeze_and_N12_N16_recovery_candidate_generation" / "RUN_63_COMBINED432_MODEL_UPDATE_N24_N40_EVIDENCE_FREEZE_AND_N12_N16_RECOVERY_CANDIDATE_GENERATION_REPORT.md"

COMBINED432_TEACHER = ROOT / "outputs" / "stage3_run_62_custom_N40_focused_batch40_teacher_metrics_ingestion_and_combined432_ranking" / "combined432_teacher_dataset.csv"
COMBINED432_READY = ROOT / "outputs" / "stage3_run_62_custom_N40_focused_batch40_teacher_metrics_ingestion_and_combined432_ranking" / "combined432_RL_ready_dataset.csv"
COMBINED432_PLUS_N32_READY = ROOT / "outputs" / "stage3_run_62_custom_N40_focused_batch40_teacher_metrics_ingestion_and_combined432_ranking" / "combined432_plus_N32_RL_ready_dataset.csv"
N32_DEDUP = ROOT / "outputs" / "stage3_run_32a_stage2_n32_legacy_teacher_label_ingestion_for_stage3" / "n32_legacy_teacher_dataset_dedup_training_332.csv"
RUN62_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_62_custom_N40_focused_batch40_teacher_metrics_ingestion_and_combined432_ranking" / "RUN_62_CUSTOM_N40_FOCUSED_BATCH40_TEACHER_METRICS_INGESTION_AND_COMBINED432_RANKING_REPORT.md"

OUTPUT_DIR = ROOT / "outputs" / "stage3_run_67_variable_N_recovery_anchor_batch48_teacher_metrics_ingestion_and_combined480_ranking"
REPORT_DIR = ROOT / "docs" / "stage3" / "runs" / "run_67_variable_N_recovery_anchor_batch48_teacher_metrics_ingestion_and_combined480_ranking"
REPORT_PATH = REPORT_DIR / "RUN_67_VARIABLE_N_RECOVERY_ANCHOR_BATCH48_TEACHER_METRICS_INGESTION_AND_COMBINED480_RANKING_REPORT.md"
MANIFEST_PATH = ROOT / "artifacts" / "manifests" / "stage3_run_67_manifest.json"
CLAIM_BOUNDARY_MD = OUTPUT_DIR / "run67_claim_boundary.md"
CLAIM_BOUNDARY_JSON = OUTPUT_DIR / "run67_claim_boundary.json"

EXPECTED_RUN66_COUNTS = {12: 12, 16: 12, 24: 8, 40: 16}
EXPECTED_COMBINED432_COUNTS = {12: 36, 16: 36, 24: 176, 40: 184}
EXPECTED_COMBINED432_PLUS_N32_COUNTS = {12: 36, 16: 36, 24: 176, 32: 332, 40: 184}
EXPECTED_COMBINED480_COUNTS = {12: 48, 16: 48, 24: 184, 40: 200}
EXPECTED_COMBINED480_PLUS_N32_COUNTS = {12: 48, 16: 48, 24: 184, 32: 332, 40: 200}

RAW_METRICS = {
    "U2": "u2_range",
    "PEEQ": "peeq_max",
    "SurfaceT": "surface_t_proxy",
    "Mises": "mises_max",
}
REWARD_DEFS = {
    "u2_primary": {"U2": 0.65, "PEEQ": 0.20, "SurfaceT": 0.10, "Mises": 0.05},
    "constrained_u2_reward_balanced": {"U2": 0.50, "PEEQ": 0.25, "SurfaceT": 0.15, "Mises": 0.10},
    "strict_penalty_guard": {"U2": 0.40, "PEEQ": 0.30, "SurfaceT": 0.20, "Mises": 0.10},
    "penalty_repair": {"U2": 0.30, "PEEQ": 0.30, "SurfaceT": 0.25, "Mises": 0.15},
}


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, tuple):
        return [clean_json(v) for v in value]
    if isinstance(value, (pd.Series, pd.Index)):
        return [clean_json(v) for v in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return clean_json(value.to_dict(orient="records"))
    if hasattr(value, "item"):
        try:
            return clean_json(value.item())
        except Exception:
            pass
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(payload), indent=2, sort_keys=False) + "\n", encoding="utf-8")


def write_table_json(path: Path, df: pd.DataFrame) -> None:
    write_json(path, {"schema": "records", "rows": df.to_dict(orient="records")})


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def current_branch() -> str:
    try:
        return subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def counts(df: pd.DataFrame) -> dict[int, int]:
    return {int(k): int(v) for k, v in df["n"].astype(int).value_counts().sort_index().to_dict().items()}


def as_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def parse_order(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(x) for x in value]
    text = str(value).strip()
    if text.startswith("["):
        return [int(x) for x in json.loads(text)]
    return [int(x) for x in text.replace(",", "-").replace(";", "-").replace(" ", "").split("-") if x != ""]


def valid_order(value: Any, n: int) -> bool:
    try:
        order = parse_order(value)
    except Exception:
        return False
    return len(order) == n and sorted(order) == list(range(n))


def order_hash_from_order(value: Any) -> str:
    order = parse_order(value)
    return hashlib.sha256(",".join(str(x) for x in order).encode("utf-8")).hexdigest()[:16]


def normalize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "surface_t_proxy" in out.columns:
        out["surface_t_proxy"] = as_float(out["surface_t_proxy"])
    elif "surface_t_proxy_max_tensile_pa" in out.columns:
        out["surface_t_proxy"] = as_float(out["surface_t_proxy_max_tensile_pa"])
    elif "surface_t_proxy_max_tensile_mpa" in out.columns:
        out["surface_t_proxy"] = as_float(out["surface_t_proxy_max_tensile_mpa"]) * 1_000_000.0
    else:
        out["surface_t_proxy"] = math.nan
    if "surface_t_proxy_mpa" not in out.columns:
        out["surface_t_proxy_mpa"] = as_float(out["surface_t_proxy"]) / 1_000_000.0
    for col in ["u2_range", "peeq_max", "mises_max"]:
        out[col] = as_float(out[col])
    return out


def rank_score(ranks: pd.Series, count: int) -> pd.Series:
    return 1.0 - ((ranks.astype(float) - 1.0) / max(1, count - 1))


def add_rank_scores(df: pd.DataFrame, prefix: str, rank_label: str) -> pd.DataFrame:
    out = normalize_metrics(df)
    for label, col in RAW_METRICS.items():
        score_col = f"target_{label.lower() if label != 'SurfaceT' else 'surfaceT'}_score_{prefix}_rank"
        rank_col = f"{label.lower() if label != 'SurfaceT' else 'surfaceT'}_rank_{rank_label}_within_n"
        cost_col = f"{label.lower() if label != 'SurfaceT' else 'surfaceT'}_cost_minmax_{rank_label}_within_n"
        out[rank_col] = math.nan
        out[score_col] = math.nan
        out[cost_col] = math.nan
        for _, idx in out.groupby("n").groups.items():
            vals = as_float(out.loc[idx, col])
            ranks = vals.rank(method="average", ascending=True)
            out.loc[idx, rank_col] = ranks
            out.loc[idx, score_col] = rank_score(ranks, len(idx))
            mn, mx = vals.min(), vals.max()
            out.loc[idx, cost_col] = 0.0 if mx == mn else (vals - mn) / (mx - mn)
    score_cols = {
        "U2": f"target_u2_score_{prefix}_rank",
        "PEEQ": f"target_peeq_score_{prefix}_rank",
        "SurfaceT": f"target_surfaceT_score_{prefix}_rank",
        "Mises": f"target_mises_score_{prefix}_rank",
    }
    for reward_name, weights in REWARD_DEFS.items():
        target_col = f"target_reward_{prefix}_{reward_name}"
        out[target_col] = sum(weights[label] * out[score_cols[label]] for label in weights)
        reward_rank = f"{reward_name}_reward_rank_{rank_label}_within_n"
        out[reward_rank] = math.nan
        for _, idx in out.groupby("n").groups.items():
            out.loc[idx, reward_rank] = out.loc[idx, target_col].rank(method="average", ascending=False)
    return out


def add_run66_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_metrics(df)
    for label, col in RAW_METRICS.items():
        out[f"rank_{label}_run66_within_n"] = math.nan
        out[f"score_{label}_run66_within_n"] = math.nan
        for _, idx in out.groupby("n").groups.items():
            ranks = as_float(out.loc[idx, col]).rank(method="average", ascending=True)
            out.loc[idx, f"rank_{label}_run66_within_n"] = ranks
            out.loc[idx, f"score_{label}_run66_within_n"] = rank_score(ranks, len(idx))
    for reward_name, weights in REWARD_DEFS.items():
        out[f"reward_run66_{reward_name}"] = sum(weights[label] * out[f"score_{label}_run66_within_n"] for label in weights)
    return out


def load_summary_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def validate_inputs(run66: pd.DataFrame, handoff: pd.DataFrame, combined432: pd.DataFrame, plus432: pd.DataFrame) -> dict[str, Any]:
    errors: list[str] = []
    required_paths = [
        RUN66_METRICS, RUN66_EXTRACTION, RUN66_SOLVER, RUN66_SUMMARY, RUN64_HANDOFF,
        RUN63_POOL, RUN63_OPTION_A, COMBINED432_TEACHER, COMBINED432_READY,
        COMBINED432_PLUS_N32_READY, N32_DEDUP,
    ]
    for path in required_paths:
        if not path.exists():
            errors.append(f"missing {path}")
    if len(run66) != 48 or counts(run66) != EXPECTED_RUN66_COUNTS:
        errors.append(f"Run66 count mismatch rows={len(run66)} counts={counts(run66)}")
    if set(run66["n"].astype(int)) - {12, 16, 24, 40}:
        errors.append("Run66 contains N outside N12/N16/N24/N40")
    if (run66["n"].astype(int) == 32).any():
        errors.append("Run66 contains N32 rows")
    for col in ["u2_range", "peeq_max", "surface_t_proxy", "mises_max"]:
        if col not in run66.columns:
            errors.append(f"Run66 missing metric {col}")
        elif as_float(run66[col]).isna().any():
            errors.append(f"Run66 has missing values in {col}")
    if "teacher_validation_status" in run66.columns and not run66["teacher_validation_status"].astype(str).str.contains("PASS").all():
        errors.append("Run66 teacher_validation_status is not PASS for all rows")
    if "final_step_name" in run66.columns and not (run66["final_step_name"].astype(str) == "step_final_cooling").all():
        errors.append("Run66 final_step_name is not step_final_cooling for all rows")
    if "extracted_field_names" in run66.columns:
        fields_text = ";".join(run66["extracted_field_names"].astype(str).tolist())
        for field in ["U", "PEEQ", "S", "NT11"]:
            if field not in fields_text:
                errors.append(f"Run66 extracted fields missing {field}")
    if len(handoff) != 48 or counts(handoff) != EXPECTED_RUN66_COUNTS:
        errors.append("Run64 handoff count mismatch")
    for _, row in handoff.iterrows():
        order_col = "order_json" if "order_json" in handoff.columns else "scan_order"
        if not valid_order(row[order_col], int(row["n"])):
            errors.append(f"invalid scan order for {row.get('handoff_strategy_name', row.name)}")
            break
    if handoff["handoff_strategy_name"].astype(str).duplicated().any():
        errors.append("Run64 handoff has duplicate handoff_strategy_name values")
    if "order_hash" in handoff.columns and handoff.groupby("n")["order_hash"].apply(lambda s: s.astype(str).duplicated().any()).any():
        errors.append("Run64 handoff has duplicate order within same N")
    if len(combined432) != 432 or counts(combined432) != EXPECTED_COMBINED432_COUNTS:
        errors.append(f"combined432 count mismatch rows={len(combined432)} counts={counts(combined432)}")
    if (combined432["n"].astype(int) == 32).any():
        errors.append("native combined432 contains N32 rows")
    if len(plus432) != 764 or counts(plus432) != EXPECTED_COMBINED432_PLUS_N32_COUNTS:
        errors.append(f"combined432_plus_N32 count mismatch rows={len(plus432)} counts={counts(plus432)}")
    n32 = plus432[plus432["n"].astype(int) == 32]
    if "metric_semantic_warning" not in n32.columns:
        errors.append("combined432_plus_N32 N32 rows missing metric_semantic_warning")
    matched = set(run66["handoff_strategy_name"].astype(str)) <= set(handoff["handoff_strategy_name"].astype(str))
    if not matched:
        errors.append("not all Run66 rows match Run64 handoff_strategy_name")
    verdict = "PASS_RUN67_VARIABLE_N_RECOVERY_ANCHOR_BATCH48_TEACHER_METRICS_48_OF_48_READY" if not errors else "FAIL_RUN67_INPUT_VALIDATION"
    return {
        "timestamp": now_iso(),
        "verdict": verdict,
        "errors": errors,
        "run66_teacher_rows": int(len(run66)),
        "run66_per_N_counts": counts(run66),
        "combined432_rows": int(len(combined432)),
        "combined432_per_N_counts": counts(combined432),
        "combined432_plus_N32_rows": int(len(plus432)),
        "combined432_plus_N32_per_N_counts": counts(plus432),
        "run66_contains_N12": bool((run66["n"].astype(int) == 12).any()),
        "run66_contains_N16": bool((run66["n"].astype(int) == 16).any()),
        "run66_contains_N32": bool((run66["n"].astype(int) == 32).any()),
    }


def make_enriched(run66: pd.DataFrame, handoff: pd.DataFrame) -> pd.DataFrame:
    metrics = normalize_metrics(run66)
    meta = handoff.copy()
    merged = metrics.merge(meta, on=["handoff_strategy_name", "n"], how="left", suffixes=("", "_run64"))
    merged["strategy_name"] = merged["handoff_strategy_name"]
    merged["dataset_source"] = "run66_variable_N_recovery_anchor_batch48"
    merged["batch_name"] = "stage3_run64_variable_N_recovery_anchor_batch48_v01"
    merged["native_validation_N"] = True
    merged["variable_N_recovery"] = True
    merged["smallN_recovery"] = merged["n"].astype(int).isin([12, 16])
    merged["N24_N40_anchor"] = merged["n"].astype(int).isin([24, 40])
    merged["includes_N12_case"] = merged["n"].astype(int) == 12
    merged["includes_N16_case"] = merged["n"].astype(int) == 16
    merged["includes_N32_case"] = False
    merged["final_step"] = merged.get("final_step_name", "step_final_cooling")
    merged["extracted_fields"] = merged.get("extracted_field_names", "")
    merged["solver_audit_status"] = merged.get("completion_status", "")
    merged["nonfatal_warning_flag"] = merged.get("completion_status", "").astype(str).str.contains("WARNING", na=False)
    if "order_hash" not in merged.columns or merged["order_hash"].isna().any():
        merged["order_hash"] = merged["order_json"].map(order_hash_from_order)
    merged["notes"] = "Run67 ingestion of Run66 variable-N recovery anchor batch48 teacher metrics."
    return merged


def leaderboard(df: pd.DataFrame, reward_cols: list[str]) -> pd.DataFrame:
    rows = []
    for n, group in df.groupby("n"):
        for metric_name, col in RAW_METRICS.items():
            best = group.sort_values(col, ascending=True).iloc[0]
            rows.append({"n": int(n), "metric": metric_name, "strategy_name": best.get("strategy_name", best.get("handoff_strategy_name", "")), "dataset_source": best.get("dataset_source", ""), "value": float(best[col])})
        for col in reward_cols:
            if col in group.columns:
                best = group.sort_values(col, ascending=False).iloc[0]
                rows.append({"n": int(n), "metric": col, "strategy_name": best.get("strategy_name", best.get("handoff_strategy_name", "")), "dataset_source": best.get("dataset_source", ""), "value": float(best[col])})
    return pd.DataFrame(rows)


def compare_against_baseline(combined: pd.DataFrame, baseline_source_name: str, run_source_name: str, baseline_label: str) -> pd.DataFrame:
    rows = []
    run = combined[combined["dataset_source"].astype(str) == run_source_name]
    baseline = combined[combined["dataset_source"].astype(str) != run_source_name] if baseline_source_name == "combined432" else combined[combined["dataset_source"].astype(str) == baseline_source_name]
    reward_cols = {
        "u2_primary": "target_reward_combined480_u2_primary",
        "constrained_u2_reward_balanced": "target_reward_combined480_constrained_u2_reward_balanced",
        "strict_penalty_guard": "target_reward_combined480_strict_penalty_guard",
        "penalty_repair": "target_reward_combined480_penalty_repair",
    }
    for n in [12, 16, 24, 40]:
        b = baseline[baseline["n"].astype(int) == n]
        r = run[run["n"].astype(int) == n]
        if b.empty or r.empty:
            continue
        for metric, col in RAW_METRICS.items():
            bbest = b.sort_values(col).iloc[0]
            rbest = r.sort_values(col).iloc[0]
            improvement = float(bbest[col]) - float(rbest[col])
            rows.append({
                "comparison": f"Run66_vs_{baseline_label}",
                "n": n,
                "metric": metric,
                f"{baseline_label}_best_strategy": bbest.get("strategy_name", ""),
                f"{baseline_label}_best_value": float(bbest[col]),
                "run66_best_strategy": rbest.get("strategy_name", rbest.get("handoff_strategy_name", "")),
                "run66_best_value": float(rbest[col]),
                "run66_beats_baseline": bool(improvement > 0),
                "absolute_improvement": improvement,
                "relative_improvement_pct": (improvement / abs(float(bbest[col])) * 100.0) if float(bbest[col]) != 0 else None,
                "combined480_new_best_strategy": combined[combined["n"].astype(int) == n].sort_values(col).iloc[0].get("strategy_name", ""),
                "combined480_new_best_source": combined[combined["n"].astype(int) == n].sort_values(col).iloc[0].get("dataset_source", ""),
            })
        for metric, col in reward_cols.items():
            bbest = b.sort_values(col, ascending=False).iloc[0]
            rbest = r.sort_values(col, ascending=False).iloc[0]
            improvement = float(rbest[col]) - float(bbest[col])
            rows.append({
                "comparison": f"Run66_vs_{baseline_label}",
                "n": n,
                "metric": metric,
                f"{baseline_label}_best_strategy": bbest.get("strategy_name", ""),
                f"{baseline_label}_best_value": float(bbest[col]),
                "run66_best_strategy": rbest.get("strategy_name", rbest.get("handoff_strategy_name", "")),
                "run66_best_value": float(rbest[col]),
                "run66_beats_baseline": bool(improvement > 0),
                "absolute_improvement": improvement,
                "relative_improvement_pct": (improvement / abs(float(bbest[col])) * 100.0) if float(bbest[col]) != 0 else None,
                "combined480_new_best_strategy": combined[combined["n"].astype(int) == n].sort_values(col, ascending=False).iloc[0].get("strategy_name", ""),
                "combined480_new_best_source": combined[combined["n"].astype(int) == n].sort_values(col, ascending=False).iloc[0].get("dataset_source", ""),
            })
    return pd.DataFrame(rows)


def spearman(a: pd.Series, b: pd.Series) -> float | None:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 3 or x[mask].nunique() < 2 or y[mask].nunique() < 2:
        return None
    return float(x[mask].corr(y[mask], method="spearman"))


def top_overlap(real: pd.Series, pred: pd.Series, k: int) -> int | None:
    x = pd.to_numeric(real, errors="coerce")
    y = pd.to_numeric(pred, errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() == 0:
        return None
    kk = min(k, int(mask.sum()))
    real_idx = set(x[mask].sort_values(ascending=False).head(kk).index)
    pred_idx = set(y[mask].sort_values(ascending=False).head(kk).index)
    return len(real_idx & pred_idx)


def prediction_audit(run66: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    pred_map = {
        "u2_primary": ("u2_primary_prediction", "target_reward_combined480_u2_primary"),
        "constrained": ("constrained_reward_prediction", "target_reward_combined480_constrained_u2_reward_balanced"),
        "strict": ("strict_penalty_guard_prediction", "target_reward_combined480_strict_penalty_guard"),
        "penalty_repair": ("penalty_repair_prediction", "target_reward_combined480_penalty_repair"),
        "U2": ("u2_primary_prediction", "target_u2_score_combined480_rank"),
        "PEEQ": ("penalty_repair_prediction", "target_peeq_score_combined480_rank"),
        "SurfaceT": ("variable_N_recovery_prediction", "target_surfaceT_score_combined480_rank"),
        "Mises": ("strict_penalty_guard_prediction", "target_mises_score_combined480_rank"),
        "N12_recovery": ("N12_recovery_prediction", "target_reward_combined480_constrained_u2_reward_balanced"),
        "N16_recovery": ("N16_recovery_prediction", "target_reward_combined480_constrained_u2_reward_balanced"),
        "variable_N_recovery": ("variable_N_recovery_prediction", "target_reward_combined480_constrained_u2_reward_balanced"),
    }
    rows = []
    for label, (pred, real) in pred_map.items():
        if pred in run66.columns and real in run66.columns:
            rows.append({"scope": "overall", "target": label, "spearman": spearman(run66[pred], run66[real]), "top5_overlap": top_overlap(run66[real], run66[pred], 5), "top10_overlap": top_overlap(run66[real], run66[pred], 10), "top1_hit": top_overlap(run66[real], run66[pred], 1)})
            for n, group in run66.groupby("n"):
                rows.append({"scope": f"N{int(n)}", "target": label, "spearman": spearman(group[pred], group[real]), "top5_overlap": top_overlap(group[real], group[pred], 5), "top10_overlap": top_overlap(group[real], group[pred], 10), "top1_hit": top_overlap(group[real], group[pred], 1)})
    audit = pd.DataFrame(rows)
    summary = {
        "headline": "Run63/Run64 prediction calibration was evaluated on realized Run66 variable-N labels; use these diagnostics as calibration evidence, not teacher-validation claims for future candidates.",
        "overall_reward_spearman": next((r["spearman"] for r in rows if r["scope"] == "overall" and r["target"] == "u2_primary"), None),
        "overall_penalty_repair_spearman": next((r["spearman"] for r in rows if r["scope"] == "overall" and r["target"] == "penalty_repair"), None),
        "mean_top5_overlap": float(pd.to_numeric(audit[audit["scope"] == "overall"]["top5_overlap"], errors="coerce").mean()) if not audit.empty else None,
        "top1_hits": int(pd.to_numeric(audit[audit["scope"] == "overall"]["top1_hit"], errors="coerce").fillna(0).sum()) if not audit.empty else 0,
        "by_candidate_source": {},
        "by_selection_bucket": {},
        "disagreement_vs_abs_error_spearman": None,
        "uncertainty_vs_abs_error_spearman": None,
        "novelty_vs_realized_reward_spearman": spearman(run66.get("novelty_distance", pd.Series(dtype=float)), run66.get("target_reward_combined480_u2_primary", pd.Series(dtype=float))),
    }
    if "target_reward_combined480_u2_primary" in run66 and "u2_primary_prediction" in run66:
        abs_err = (pd.to_numeric(run66["target_reward_combined480_u2_primary"], errors="coerce") - pd.to_numeric(run66["u2_primary_prediction"], errors="coerce")).abs()
        summary["disagreement_vs_abs_error_spearman"] = spearman(run66.get("gnn_vs_surrogate_disagreement", pd.Series(dtype=float)), abs_err)
        summary["uncertainty_vs_abs_error_spearman"] = spearman(run66.get("uncertainty_score", pd.Series(dtype=float)), abs_err)
        for col, key in [("candidate_source", "by_candidate_source"), ("selection_bucket", "by_selection_bucket"), ("generation_method", "by_generation_method")]:
            if col in run66.columns:
                summary[key] = {
                    str(k): {
                        "count": int(len(g)),
                        "mean_abs_error": float(abs_err.loc[g.index].mean()),
                        "mean_realized_reward": float(pd.to_numeric(g["target_reward_combined480_u2_primary"], errors="coerce").mean()),
                    }
                    for k, g in run66.groupby(col)
                }
    return audit, summary


def effectiveness_audit(run66: pd.DataFrame, combined480: pd.DataFrame, comparison: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    reward_cols = [
        "target_reward_combined480_u2_primary",
        "target_reward_combined480_constrained_u2_reward_balanced",
        "target_reward_combined480_strict_penalty_guard",
        "target_reward_combined480_penalty_repair",
    ]
    for n, group in run66.groupby("n"):
        alln = combined480[combined480["n"].astype(int) == int(n)]
        for metric, col in RAW_METRICS.items():
            ranks = alln[col].rank(method="average", ascending=True)
            top5_hashes = set(alln.loc[ranks <= 5, "order_hash"].astype(str))
            top10_hashes = set(alln.loc[ranks <= 10, "order_hash"].astype(str))
            rows.append({"n": int(n), "metric": metric, "run66_top5_entries": int(group["order_hash"].astype(str).isin(top5_hashes).sum()), "run66_top10_entries": int(group["order_hash"].astype(str).isin(top10_hashes).sum()), "run66_best_rank": float(ranks.loc[group.index].min()) if set(group.index) <= set(ranks.index) else None})
        for col in reward_cols:
            ranks = alln[col].rank(method="average", ascending=False)
            top5_hashes = set(alln.loc[ranks <= 5, "order_hash"].astype(str))
            top10_hashes = set(alln.loc[ranks <= 10, "order_hash"].astype(str))
            rows.append({"n": int(n), "metric": col, "run66_top5_entries": int(group["order_hash"].astype(str).isin(top5_hashes).sum()), "run66_top10_entries": int(group["order_hash"].astype(str).isin(top10_hashes).sum()), "run66_best_rank": float(ranks.loc[group.index].min()) if set(group.index) <= set(ranks.index) else None})
    audit = pd.DataFrame(rows)
    new_best_count = int(comparison["run66_beats_baseline"].sum()) if "run66_beats_baseline" in comparison else 0
    by_cols = {}
    for col in ["candidate_source", "generation_method", "selection_bucket", "priority_role"]:
        if col in run66.columns:
            by_cols[col] = run66.groupby(col).agg(
                count=("n", "count"),
                mean_u2_primary=("target_reward_combined480_u2_primary", "mean"),
                mean_penalty_repair=("target_reward_combined480_penalty_repair", "mean"),
                mean_u2=("u2_range", "mean"),
                mean_peeq=("peeq_max", "mean"),
                mean_surfaceT=("surface_t_proxy", "mean"),
                mean_mises=("mises_max", "mean"),
            ).reset_index().to_dict(orient="records")
    headline = (
        f"Run66 produced {new_best_count} new metric/reward records versus combined432; "
        "the batch directly tests N12/N16 recovery while preserving N24/N40 frozen-anchor coverage."
    )
    summary = {
        "headline": headline,
        "new_best_count_vs_combined432": new_best_count,
        "top_density_rows": rows,
        "performance_by_group": by_cols,
        "N12_benefited": bool(comparison[(comparison["n"] == 12) & (comparison["run66_beats_baseline"] == True)].shape[0] > 0),
        "N16_benefited": bool(comparison[(comparison["n"] == 16) & (comparison["run66_beats_baseline"] == True)].shape[0] > 0),
        "N24_anchor_benefited": bool(comparison[(comparison["n"] == 24) & (comparison["run66_beats_baseline"] == True)].shape[0] > 0),
        "N40_anchor_benefited": bool(comparison[(comparison["n"] == 40) & (comparison["run66_beats_baseline"] == True)].shape[0] > 0),
        "variable_N_recovery_batch48_role": "accepted_for_analysis; evaluate future action from small-N recovery gains, anchor stability, and prediction calibration",
    }
    return audit, summary


def prior_records(combined480: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    sources = {
        "Run66": "run66_variable_N_recovery_anchor_batch48",
        "Run61": "run61_custom_N40_focused_batch40",
        "Run56": "run56_calibrated_N24_N40_batch64",
        "Run51": "run51_stricter_constrained_N24_N40_batch32",
        "Run46": "run46_constrained_N24_N40_batch32",
        "Run41": "run41_native_N24_N40_focused_batch60",
        "Run36": "run36_N32_informed_native_batch32",
        "Run27": "shortlist64_run27",
    }
    rows = []
    reward_cols = [
        "target_reward_combined480_u2_primary",
        "target_reward_combined480_constrained_u2_reward_balanced",
        "target_reward_combined480_strict_penalty_guard",
        "target_reward_combined480_penalty_repair",
    ]
    for n in [12, 16, 24, 40]:
        for label, source in sources.items():
            g = combined480[(combined480["n"].astype(int) == n) & (combined480["dataset_source"].astype(str) == source)]
            if g.empty:
                continue
            row = {"record_set": label, "dataset_source": source, "n": n, "count": int(len(g))}
            for metric, col in RAW_METRICS.items():
                best = g.sort_values(col).iloc[0]
                row[f"best_{metric}_strategy"] = best.get("strategy_name", "")
                row[f"best_{metric}_value"] = float(best[col])
            for col in reward_cols:
                best = g.sort_values(col, ascending=False).iloc[0]
                row[f"best_{col}_strategy"] = best.get("strategy_name", "")
                row[f"best_{col}_value"] = float(best[col])
            alln = combined480[combined480["n"].astype(int) == n]
            reward_rank = alln["target_reward_combined480_u2_primary"].rank(method="average", ascending=False)
            top5 = set(alln.loc[reward_rank <= 5, "order_hash"].astype(str))
            top10 = set(alln.loc[reward_rank <= 10, "order_hash"].astype(str))
            row["top5_reward_entries_in_combined480"] = int(g["order_hash"].astype(str).isin(top5).sum())
            row["top10_reward_entries_in_combined480"] = int(g["order_hash"].astype(str).isin(top10).sum())
            rows.append(row)
    df = pd.DataFrame(rows)
    summary = {
        "headline": "Run66 was compared with Run61, Run56, Run51, Run46, Run41, Run36, Run27, and recomputed combined480 best records.",
        "run66_complements_prior_n24_n40_freeze": True,
        "full_variable_n_maturity_warning": "N12/N16 improved to 48 rows each, but remain much less dense than N24/N40; avoid claiming full variable-N maturity.",
    }
    return df, summary


def maturity_audit(combined480: pd.DataFrame, comparison: pd.DataFrame, pred_summary: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], str]:
    counts_now = counts(combined480)
    rows = [
        {"n": 12, "native_teacher_rows": counts_now.get(12, 0), "status": "smallN_recovery_improved_but_still_less_dense"},
        {"n": 16, "native_teacher_rows": counts_now.get(16, 0), "status": "smallN_recovery_improved_but_still_less_dense"},
        {"n": 24, "native_teacher_rows": counts_now.get(24, 0), "status": "mature_frozen_anchor"},
        {"n": 32, "native_teacher_rows": 0, "legacy_teacher_rows": 332, "status": "legacy_compatible_auxiliary_only"},
        {"n": 40, "native_teacher_rows": counts_now.get(40, 0), "status": "mature_frozen_anchor"},
    ]
    new_best_count = int(comparison["run66_beats_baseline"].sum()) if "run66_beats_baseline" in comparison else 0
    summary = {
        "headline": "After Run66, combined480 has N12=48, N16=48, N24=184, and N40=200; full variable-N evidence is stronger, but N12/N16 remain the limiting under-sampled regimes relative to mature N24/N40 anchors.",
        "n24_teacher_rows": counts_now.get(24, 0),
        "n40_teacher_rows": counts_now.get(40, 0),
        "n32_legacy_teacher_rows": 332,
        "n12_teacher_rows": counts_now.get(12, 0),
        "n16_teacher_rows": counts_now.get(16, 0),
        "run66_new_best_count_vs_combined432": new_best_count,
        "full_variable_n_evidence_stronger_after_run66": True,
        "n24_n40_evidence_frozen_anchor_status": True,
        "gnn_pointer_auxiliary_only": True,
        "n24_n40_mature_for_bounded_active_learning_evidence": True,
        "full_variable_n_rl_maturity_limited_by_n12_n16": True,
        "safe_paper_claim": "Run66 strengthens small-N evidence while preserving mature N24/N40 anchors; full variable-N maturity should still be bounded by N12/N16 evidence density.",
        "prediction_calibration_summary": pred_summary,
    }
    md = (
        "# Full Variable-N Updated Maturity and Claim-Boundary Audit\n\n"
        f"{summary['headline']}\n\n"
        "- N12 native teacher rows: 48\n"
        "- N16 native teacher rows: 48\n"
        "- N24 native teacher rows: 184\n"
        "- N40 native teacher rows: 200\n"
        "- N32 legacy-compatible teacher rows: 332\n"
        "\n"
        "N32 remains legacy-compatible auxiliary data, not native Stage 3 teacher validation. "
        "GNN and graph-pointer diagnostics remain auxiliary unless future diagnostics materially improve.\n"
    )
    return pd.DataFrame(rows), summary, md


def write_claim_boundary() -> None:
    safe = [
        "Run67 ingests 48/48 teacher-validated Run66 variable-N recovery anchor batch48 cases.",
        "Run67 builds native combined480 with N12=48, N16=48, N24=184, N40=200.",
        "Run67 builds combined480_plus_N32 with N12=48, N16=48, N24=184, N32=332, N40=200.",
        "Run67 evaluates whether the variable-N recovery anchor batch improved small-N evidence and preserved N24/N40 anchor behavior.",
        "Run66 is teacher validation of native N12/N16/N24/N40 recovery-anchor candidates, not N32 cases.",
        "Run67 updates the full variable-N maturity and claim boundary.",
    ]
    unsafe = [
        "Do not claim N32 itself was newly teacher-validated in Run66.",
        "Do not claim N32 caused Run66 improvements.",
        "Do not claim GNN-RL superiority unless supported.",
        "Do not claim online RL.",
        "Do not claim arbitrary-N generalization.",
        "Do not claim physical optimum.",
        "Do not claim solver/ODB extraction happened in Run67.",
        "Do not claim full variable-N RL maturity while N12/N16 remain under-sampled.",
    ]
    CLAIM_BOUNDARY_MD.write_text("# Run67 Claim Boundary\n\n## Safe claims\n" + "\n".join(f"- {x}" for x in safe) + "\n\n## Unsafe claims\n" + "\n".join(f"- {x}" for x in unsafe) + "\n", encoding="utf-8")
    write_json(CLAIM_BOUNDARY_JSON, {"verdict": "RUN67_INGESTION_AND_ANALYSIS_ONLY_NO_SOLVER_OR_CANDIDATE_GENERATION", "safe_claims": safe, "unsafe_claims": unsafe})


def write_report(summary: dict[str, Any]) -> None:
    REPORT_PATH.write_text(f"""# Stage 3 Run 67 - Variable-N Recovery Anchor Batch48 Teacher Metrics Ingestion and Combined480 Ranking

## 1. Purpose
Run67 ingests Run66 teacher metrics, builds native combined480 and combined480_plus_N32 datasets, and evaluates whether the variable-N recovery anchor batch48 improved small-N evidence while preserving mature N24/N40 anchors.

## 2. Inputs
- Run66 teacher metrics: `{RUN66_METRICS}`
- Run64 handoff metadata: `{RUN64_HANDOFF}`
- Native combined432: `{COMBINED432_READY}`
- combined432_plus_N32: `{COMBINED432_PLUS_N32_READY}`

## 3. Run66 Teacher-Validation Status
Run66 completed 48/48 with N12=12, N16=12, N24=8, and N40=16. Nonfatal warnings were present, with no failed or incomplete cases.

## 4. Input Validation
Verdict: `{summary['validation']['verdict']}`.

## 5. Run66 Enriched Teacher Dataset
Output: `{OUTPUT_DIR / 'run66_variable_N_recovery_anchor_batch48_teacher_dataset_enriched.csv'}`

## 6. Run66 Within-Batch Ranking
Output: `{OUTPUT_DIR / 'run66_variable_N_recovery_anchor_batch48_ranked_within_batch.csv'}`

## 7. Native Combined480 Construction
Native combined480 counts: `{summary['combined480_counts']}`.

## 8. combined480_plus_N32 Construction
combined480_plus_N32 counts: `{summary['combined480_plus_N32_counts']}`. N32 metric semantic warnings are preserved.

## 9. Run66 vs Combined432 Best Comparison
{summary['comparison_headline']}

## 10. Run66 vs Prior Key Records
{summary['prior_summary']['headline']}

## 11. Variable-N Recovery Anchor Batch48 Effectiveness Audit
{summary['effectiveness_summary']['headline']}

## 12. Prediction Audit for Run63/Run64 Variable-N Batch48
{summary['prediction_summary']['headline']}

## 13. Small-N Recovery Versus N24/N40 Anchors
N12/N16 recovery is the primary purpose. N24/N40 are retained as mature anchors rather than fresh exploitation targets.

## 14. Updated Full Variable-N Maturity and Claim-Boundary Audit
{summary['maturity_summary']['headline']}

## 15. Metric Semantic Boundary for N32
N32 rows are legacy-compatible auxiliary data. They are not native Stage 3 teacher validation, and PEEQ/Mises mappings must retain semantic warnings.

## 16. Claim Boundary
Verdict: `RUN67_INGESTION_AND_ANALYSIS_ONLY_NO_SOLVER_OR_CANDIDATE_GENERATION`.

## 17. Output Files
- combined480 RL-ready: `{OUTPUT_DIR / 'combined480_RL_ready_dataset.csv'}`
- combined480_plus_N32 RL-ready: `{OUTPUT_DIR / 'combined480_plus_N32_RL_ready_dataset.csv'}`
- Run66 comparison: `{OUTPUT_DIR / 'run66_vs_combined432_best_comparison.csv'}`
- Updated maturity audit: `{OUTPUT_DIR / 'full_variable_N_updated_maturity_and_claim_boundary_audit.md'}`
- Manifest: `{MANIFEST_PATH}`

## 18. Recommended Run68
{summary['recommended_run68']}
""", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    run66 = normalize_metrics(read_csv(RUN66_METRICS))
    handoff = read_csv(RUN64_HANDOFF)
    combined432 = normalize_metrics(read_csv(COMBINED432_READY))
    plus432 = normalize_metrics(read_csv(COMBINED432_PLUS_N32_READY))

    validation = validate_inputs(run66, handoff, combined432, plus432)
    write_json(OUTPUT_DIR / "run67_input_validation_summary.json", validation)
    if not validation["verdict"].startswith("PASS"):
        raise SystemExit(validation["errors"])

    enriched = make_enriched(run66, handoff)
    write_csv(OUTPUT_DIR / "run66_variable_N_recovery_anchor_batch48_teacher_dataset_enriched.csv", enriched)
    write_table_json(OUTPUT_DIR / "run66_variable_N_recovery_anchor_batch48_teacher_dataset_enriched.json", enriched)

    ranked = add_run66_scores(enriched)
    write_csv(OUTPUT_DIR / "run66_variable_N_recovery_anchor_batch48_ranked_within_batch.csv", ranked)
    run66_leaderboard = leaderboard(ranked, [f"reward_run66_{x}" for x in REWARD_DEFS])
    write_csv(OUTPUT_DIR / "run66_variable_N_recovery_anchor_batch48_per_N_leaderboard.csv", run66_leaderboard)

    combined480 = pd.concat([combined432, enriched], ignore_index=True, sort=False)
    combined480 = add_rank_scores(combined480, "combined480", "combined480")
    write_csv(OUTPUT_DIR / "combined480_teacher_dataset.csv", combined480)
    write_csv(OUTPUT_DIR / "combined480_RL_ready_dataset.csv", combined480)
    combined480_leaderboard = leaderboard(combined480, [f"target_reward_combined480_{x}" for x in REWARD_DEFS])
    write_csv(OUTPUT_DIR / "combined480_per_N_leaderboard.csv", combined480_leaderboard)
    combined480_summary = {"rows": int(len(combined480)), "per_N_counts": counts(combined480), "leaderboard": combined480_leaderboard.to_dict(orient="records")}
    write_json(OUTPUT_DIR / "combined480_summary.json", combined480_summary)

    n32 = plus432[plus432["n"].astype(int) == 32].copy()
    combined480_plus = pd.concat([combined480, n32], ignore_index=True, sort=False)
    combined480_plus = add_rank_scores(combined480_plus, "combined480_plus_N32", "combined480_plus_N32")
    if "target_reward_combined480_plus_N32_strict_u2_surfaceT" not in combined480_plus.columns:
        combined480_plus["target_reward_combined480_plus_N32_strict_u2_surfaceT"] = (
            0.70 * combined480_plus["target_u2_score_combined480_plus_N32_rank"]
            + 0.30 * combined480_plus["target_surfaceT_score_combined480_plus_N32_rank"]
        )
    if "target_reward_combined480_plus_N32_mapped_u2_primary" not in combined480_plus.columns:
        combined480_plus["target_reward_combined480_plus_N32_mapped_u2_primary"] = combined480_plus["target_reward_combined480_plus_N32_u2_primary"]
    write_csv(OUTPUT_DIR / "combined480_plus_N32_teacher_dataset.csv", combined480_plus)
    write_csv(OUTPUT_DIR / "combined480_plus_N32_RL_ready_dataset.csv", combined480_plus)
    plus_leaderboard = leaderboard(combined480_plus, ["target_reward_combined480_plus_N32_mapped_u2_primary", "target_reward_combined480_plus_N32_strict_u2_surfaceT"])
    write_csv(OUTPUT_DIR / "combined480_plus_N32_per_N_leaderboard.csv", plus_leaderboard)
    plus_summary = {"rows": int(len(combined480_plus)), "per_N_counts": counts(combined480_plus), "n32_semantic_warning_preserved": "metric_semantic_warning" in n32.columns}
    write_json(OUTPUT_DIR / "combined480_plus_N32_summary.json", plus_summary)

    comparison = compare_against_baseline(combined480, "combined432", "run66_variable_N_recovery_anchor_batch48", "combined432")
    write_csv(OUTPUT_DIR / "run66_vs_combined432_best_comparison.csv", comparison)
    write_json(OUTPUT_DIR / "run66_vs_combined432_best_comparison.json", comparison.to_dict(orient="records"))

    prior_df, prior_summary = prior_records(combined480)
    write_csv(OUTPUT_DIR / "run66_vs_prior_key_records.csv", prior_df)
    write_json(OUTPUT_DIR / "run66_vs_prior_key_records_summary.json", prior_summary)

    run66_combined = combined480[combined480["dataset_source"].astype(str) == "run66_variable_N_recovery_anchor_batch48"].copy()
    eff_df, eff_summary = effectiveness_audit(run66_combined, combined480, comparison)
    write_csv(OUTPUT_DIR / "run66_variable_N_recovery_anchor_batch48_effectiveness_audit.csv", eff_df)
    write_json(OUTPUT_DIR / "run66_variable_N_recovery_anchor_batch48_effectiveness_summary.json", eff_summary)

    pred_df, pred_summary = prediction_audit(run66_combined)
    write_csv(OUTPUT_DIR / "run66_prediction_audit_for_run63_batch48.csv", pred_df)
    write_json(OUTPUT_DIR / "run66_prediction_audit_for_run63_batch48_summary.json", pred_summary)

    maturity_df, maturity_summary, maturity_md = maturity_audit(combined480, comparison, pred_summary)
    write_csv(OUTPUT_DIR / "full_variable_N_updated_maturity_and_claim_boundary_audit.csv", maturity_df)
    write_json(OUTPUT_DIR / "full_variable_N_updated_maturity_and_claim_boundary_summary.json", maturity_summary)
    (OUTPUT_DIR / "full_variable_N_updated_maturity_and_claim_boundary_audit.md").write_text(maturity_md, encoding="utf-8")

    write_claim_boundary()

    n12n16_records = int(comparison[(comparison["n"].isin([12, 16])) & (comparison["run66_beats_baseline"] == True)].shape[0])
    if n12n16_records > 0:
        recommended = "Update models with combined480 and generate the next N12/N16 recovery-focused batch, keeping N24/N40 as frozen anchors."
    elif eff_summary.get("top_density_rows"):
        recommended = "Update models with combined480 and run a small follow-up recovery diagnostic before any full variable-N maturity claim."
    else:
        recommended = "Diagnose small-N recovery calibration and keep the next loop small; do not claim full variable-N maturity."
    report_summary = {
        "validation": validation,
        "combined480_counts": counts(combined480),
        "combined480_plus_N32_counts": counts(combined480_plus),
        "comparison_headline": f"Run66 created {int(comparison['run66_beats_baseline'].sum())} new metric/reward records versus combined432.",
        "prior_summary": prior_summary,
        "effectiveness_summary": eff_summary,
        "prediction_summary": pred_summary,
        "maturity_summary": maturity_summary,
        "recommended_run68": recommended,
    }
    write_report(report_summary)

    output_files = [
        OUTPUT_DIR / "run67_input_validation_summary.json",
        OUTPUT_DIR / "run66_variable_N_recovery_anchor_batch48_teacher_dataset_enriched.csv",
        OUTPUT_DIR / "run66_variable_N_recovery_anchor_batch48_teacher_dataset_enriched.json",
        OUTPUT_DIR / "run66_variable_N_recovery_anchor_batch48_ranked_within_batch.csv",
        OUTPUT_DIR / "run66_variable_N_recovery_anchor_batch48_per_N_leaderboard.csv",
        OUTPUT_DIR / "combined480_teacher_dataset.csv",
        OUTPUT_DIR / "combined480_RL_ready_dataset.csv",
        OUTPUT_DIR / "combined480_per_N_leaderboard.csv",
        OUTPUT_DIR / "combined480_summary.json",
        OUTPUT_DIR / "combined480_plus_N32_teacher_dataset.csv",
        OUTPUT_DIR / "combined480_plus_N32_RL_ready_dataset.csv",
        OUTPUT_DIR / "combined480_plus_N32_per_N_leaderboard.csv",
        OUTPUT_DIR / "combined480_plus_N32_summary.json",
        OUTPUT_DIR / "run66_vs_combined432_best_comparison.csv",
        OUTPUT_DIR / "run66_vs_combined432_best_comparison.json",
        OUTPUT_DIR / "run66_vs_prior_key_records.csv",
        OUTPUT_DIR / "run66_vs_prior_key_records_summary.json",
        OUTPUT_DIR / "run66_variable_N_recovery_anchor_batch48_effectiveness_audit.csv",
        OUTPUT_DIR / "run66_variable_N_recovery_anchor_batch48_effectiveness_summary.json",
        OUTPUT_DIR / "run66_prediction_audit_for_run63_batch48.csv",
        OUTPUT_DIR / "run66_prediction_audit_for_run63_batch48_summary.json",
        OUTPUT_DIR / "full_variable_N_updated_maturity_and_claim_boundary_audit.csv",
        OUTPUT_DIR / "full_variable_N_updated_maturity_and_claim_boundary_summary.json",
        OUTPUT_DIR / "full_variable_N_updated_maturity_and_claim_boundary_audit.md",
        CLAIM_BOUNDARY_MD,
        CLAIM_BOUNDARY_JSON,
        REPORT_PATH,
        MANIFEST_PATH,
    ]
    manifest = {
        "run_id": RUN_ID,
        "run_name": RUN_NAME,
        "timestamp": now_iso(),
        "branch": current_branch(),
        "script_path": str(SCRIPT_PATH),
        "input_files": [str(p) for p in [RUN66_METRICS, RUN66_EXTRACTION, RUN66_SOLVER, RUN66_SUMMARY, RUN66_REPORT, RUN66_MANIFEST, RUN64_HANDOFF, RUN64_SCAN_DIR, RUN63_POOL, RUN63_OPTION_A, RUN63_EVIDENCE_FREEZE, RUN63_REPORT, COMBINED432_TEACHER, COMBINED432_READY, COMBINED432_PLUS_N32_READY, N32_DEDUP, RUN62_REPORT]],
        "output_files": [str(p) for p in output_files],
        "run66_teacher_rows": 48,
        "combined480_rows": int(len(combined480)),
        "combined480_plus_N32_rows": int(len(combined480_plus)),
        "per_N_combined480_counts": counts(combined480),
        "per_N_combined480_plus_N32_counts": counts(combined480_plus),
        "new_best_counts": {"run66_vs_combined432": int(comparison["run66_beats_baseline"].sum())},
        "prediction_audit_summary": pred_summary,
        "maturity_audit_summary": maturity_summary,
        "report_path": str(REPORT_PATH),
        "claim_boundary_path": str(CLAIM_BOUNDARY_MD),
        "no_solver_run": True,
        "no_odb_opened": True,
        "no_abqjobpilot_run": True,
        "no_cae_inp_generated": True,
        "no_teacher_validation_performed_by_run67": True,
        "no_training": True,
        "no_candidate_generation": True,
        "no_commit_or_push": True,
    }
    write_json(MANIFEST_PATH, manifest)
    print(json.dumps({
        "verdict": validation["verdict"],
        "run66_counts": counts(run66),
        "combined480_counts": counts(combined480),
        "combined480_plus_N32_counts": counts(combined480_plus),
        "new_bests_vs_combined432": int(comparison["run66_beats_baseline"].sum()),
        "report": str(REPORT_PATH),
    }, indent=2))


if __name__ == "__main__":
    main()
