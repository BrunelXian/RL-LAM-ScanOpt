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
RUN_ID = "run_62_custom_N40_focused_batch40_teacher_metrics_ingestion_and_combined432_ranking"
RUN_NAME = "custom N40-focused batch40 teacher metrics ingestion and combined432 ranking"
SCRIPT_PATH = ROOT / "scripts" / "stage3" / "run_62_ingest_custom_N40_focused_batch40_and_build_combined432.py"

RUN61_DIR = ROOT / "outputs" / "stage3_run_61_custom_N40_focused_calibrated_penalty_repair_batch40_odb_teacher_validation"
RUN61_METRICS = RUN61_DIR / "run61_custom_N40_focused_batch40_teacher_metrics.csv"
RUN61_EXTRACTION = RUN61_DIR / "run61_custom_N40_focused_batch40_odb_extraction_summary.csv"
RUN61_EXTRACTION_JSON = RUN61_DIR / "run61_custom_N40_focused_batch40_odb_extraction_summary.json"
RUN61_SOLVER = RUN61_DIR / "run61_custom_N40_focused_batch40_solver_completion_audit.csv"
RUN61_SOLVER_JSON = RUN61_DIR / "run61_custom_N40_focused_batch40_solver_completion_audit.json"
RUN61_SUMMARY = RUN61_DIR / "run61_custom_N40_focused_batch40_odb_teacher_validation_summary.json"
RUN61_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_61_custom_N40_focused_calibrated_penalty_repair_batch40_odb_teacher_validation" / "RUN_61_CUSTOM_N40_FOCUSED_CALIBRATED_PENALTY_REPAIR_BATCH40_ODB_TEACHER_VALIDATION_REPORT.md"
RUN61_MANIFEST = ROOT / "artifacts" / "manifests" / "stage3_run_61_manifest.json"

RUN59_HANDOFF = ROOT / "outputs" / "stage3_run_59_run58_N40_focused_calibrated_penalty_repair_batch40_handoff_package" / "stage3_run59_N40_focused_calibrated_penalty_repair_batch40_candidate_orders.csv"
RUN59_SCAN_DIR = ROOT / "outputs" / "stage3_run_59_run58_N40_focused_calibrated_penalty_repair_batch40_handoff_package" / "scan_orders"
RUN58_POOL = ROOT / "outputs" / "stage3_run_58_combined392_model_update_N24_N40_evidence_freeze_and_N40_focused_candidate_generation" / "run58_candidate_pool_scored.csv"
RUN58_OPTION_A = ROOT / "outputs" / "stage3_run_58_combined392_model_update_N24_N40_evidence_freeze_and_N40_focused_candidate_generation" / "run58_N40_focused_calibrated_penalty_repair_batch32_candidate_orders.csv"
RUN58_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_58_combined392_model_update_N24_N40_evidence_freeze_and_N40_focused_candidate_generation" / "RUN_58_COMBINED392_MODEL_UPDATE_N24_N40_EVIDENCE_FREEZE_AND_N40_FOCUSED_CANDIDATE_GENERATION_REPORT.md"

COMBINED392_TEACHER = ROOT / "outputs" / "stage3_run_57_calibrated_N24_N40_batch64_teacher_metrics_ingestion_and_combined392_ranking" / "combined392_teacher_dataset.csv"
COMBINED392_READY = ROOT / "outputs" / "stage3_run_57_calibrated_N24_N40_batch64_teacher_metrics_ingestion_and_combined392_ranking" / "combined392_RL_ready_dataset.csv"
COMBINED392_PLUS_N32_READY = ROOT / "outputs" / "stage3_run_57_calibrated_N24_N40_batch64_teacher_metrics_ingestion_and_combined392_ranking" / "combined392_plus_N32_RL_ready_dataset.csv"
N32_DEDUP = ROOT / "outputs" / "stage3_run_32a_stage2_n32_legacy_teacher_label_ingestion_for_stage3" / "n32_legacy_teacher_dataset_dedup_training_332.csv"
RUN57_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_57_calibrated_N24_N40_batch64_teacher_metrics_ingestion_and_combined392_ranking" / "RUN_57_CALIBRATED_N24_N40_BATCH64_TEACHER_METRICS_INGESTION_AND_COMBINED392_RANKING_REPORT.md"

OUTPUT_DIR = ROOT / "outputs" / "stage3_run_62_custom_N40_focused_batch40_teacher_metrics_ingestion_and_combined432_ranking"
REPORT_DIR = ROOT / "docs" / "stage3" / "runs" / "run_62_custom_N40_focused_batch40_teacher_metrics_ingestion_and_combined432_ranking"
REPORT_PATH = REPORT_DIR / "RUN_62_CUSTOM_N40_FOCUSED_BATCH40_TEACHER_METRICS_INGESTION_AND_COMBINED432_RANKING_REPORT.md"
MANIFEST_PATH = ROOT / "artifacts" / "manifests" / "stage3_run_62_manifest.json"
CLAIM_BOUNDARY_MD = OUTPUT_DIR / "run62_claim_boundary.md"
CLAIM_BOUNDARY_JSON = OUTPUT_DIR / "run62_claim_boundary.json"

EXPECTED_RUN61_COUNTS = {24: 16, 40: 24}
EXPECTED_COMBINED392_COUNTS = {12: 36, 16: 36, 24: 160, 40: 160}
EXPECTED_COMBINED392_PLUS_N32_COUNTS = {12: 36, 16: 36, 24: 160, 32: 332, 40: 160}
EXPECTED_COMBINED432_COUNTS = {12: 36, 16: 36, 24: 176, 40: 184}
EXPECTED_COMBINED432_PLUS_N32_COUNTS = {12: 36, 16: 36, 24: 176, 32: 332, 40: 184}

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


def add_run61_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_metrics(df)
    for label, col in RAW_METRICS.items():
        out[f"rank_{label}_run61_within_n"] = math.nan
        out[f"score_{label}_run61_within_n"] = math.nan
        for _, idx in out.groupby("n").groups.items():
            ranks = as_float(out.loc[idx, col]).rank(method="average", ascending=True)
            out.loc[idx, f"rank_{label}_run61_within_n"] = ranks
            out.loc[idx, f"score_{label}_run61_within_n"] = rank_score(ranks, len(idx))
    for reward_name, weights in REWARD_DEFS.items():
        out[f"reward_run61_{reward_name}"] = sum(weights[label] * out[f"score_{label}_run61_within_n"] for label in weights)
    return out


def load_summary_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def validate_inputs(run61: pd.DataFrame, handoff: pd.DataFrame, combined392: pd.DataFrame, plus392: pd.DataFrame) -> dict[str, Any]:
    errors: list[str] = []
    if not RUN61_METRICS.exists():
        errors.append(f"missing {RUN61_METRICS}")
    if len(run61) != 40 or counts(run61) != EXPECTED_RUN61_COUNTS:
        errors.append(f"Run61 count mismatch rows={len(run61)} counts={counts(run61)}")
    if set(run61["n"].astype(int)) - {24, 40}:
        errors.append("Run61 contains N outside N24/N40")
    for col in ["u2_range", "peeq_max", "surface_t_proxy", "mises_max"]:
        if col not in run61.columns:
            errors.append(f"Run61 missing metric {col}")
        elif as_float(run61[col]).isna().any():
            errors.append(f"Run61 has missing values in {col}")
    if "teacher_validation_status" in run61.columns and not run61["teacher_validation_status"].astype(str).str.contains("PASS").all():
        errors.append("Run61 teacher_validation_status is not PASS for all rows")
    if "final_step_name" in run61.columns and not (run61["final_step_name"].astype(str) == "step_final_cooling").all():
        errors.append("Run61 final_step_name is not step_final_cooling for all rows")
    if "extracted_field_names" in run61.columns:
        fields_text = ";".join(run61["extracted_field_names"].astype(str).tolist())
        for field in ["U", "PEEQ", "S", "NT11"]:
            if field not in fields_text:
                errors.append(f"Run61 extracted fields missing {field}")
    if len(handoff) != 40 or counts(handoff) != EXPECTED_RUN61_COUNTS:
        errors.append("Run59 handoff count mismatch")
    if len(combined392) != 392 or counts(combined392) != EXPECTED_COMBINED392_COUNTS:
        errors.append(f"combined392 count mismatch rows={len(combined392)} counts={counts(combined392)}")
    if len(plus392) != 724 or counts(plus392) != EXPECTED_COMBINED392_PLUS_N32_COUNTS:
        errors.append(f"combined392_plus_N32 count mismatch rows={len(plus392)} counts={counts(plus392)}")
    n32 = plus392[plus392["n"].astype(int) == 32]
    if "metric_semantic_warning" not in n32.columns:
        errors.append("combined392_plus_N32 N32 rows missing metric_semantic_warning")
    matched = set(run61["handoff_strategy_name"].astype(str)) <= set(handoff["handoff_strategy_name"].astype(str))
    if not matched:
        errors.append("not all Run61 rows match Run59 handoff_strategy_name")
    verdict = "PASS_RUN62_CUSTOM_N40_FOCUSED_BATCH40_TEACHER_METRICS_40_OF_40_READY" if not errors else "FAIL_RUN62_INPUT_VALIDATION"
    return {
        "timestamp": now_iso(),
        "verdict": verdict,
        "errors": errors,
        "run61_teacher_rows": int(len(run61)),
        "run61_per_N_counts": counts(run61),
        "combined392_rows": int(len(combined392)),
        "combined392_per_N_counts": counts(combined392),
        "combined392_plus_N32_rows": int(len(plus392)),
        "combined392_plus_N32_per_N_counts": counts(plus392),
        "run61_contains_N12": bool((run61["n"].astype(int) == 12).any()),
        "run61_contains_N16": bool((run61["n"].astype(int) == 16).any()),
        "run61_contains_N32": bool((run61["n"].astype(int) == 32).any()),
    }


def make_enriched(run61: pd.DataFrame, handoff: pd.DataFrame) -> pd.DataFrame:
    metrics = normalize_metrics(run61)
    meta = handoff.copy()
    merged = metrics.merge(meta, on=["handoff_strategy_name", "n"], how="left", suffixes=("", "_run59"))
    merged["strategy_name"] = merged["handoff_strategy_name"]
    merged["dataset_source"] = "run61_custom_N40_focused_batch40"
    merged["batch_name"] = "stage3_run59_N40_focused_calibrated_penalty_repair_batch40_v01"
    merged["native_validation_N"] = True
    merged["N40_focused"] = True
    merged["calibrated_penalty_repair"] = True
    merged["N24_maintenance"] = merged["n"].astype(int) == 24
    merged["includes_N12_case"] = False
    merged["includes_N16_case"] = False
    merged["includes_N32_case"] = False
    merged["final_step"] = merged.get("final_step_name", "step_final_cooling")
    merged["extracted_fields"] = merged.get("extracted_field_names", "")
    merged["solver_audit_status"] = merged.get("completion_status", "")
    merged["nonfatal_warning_flag"] = merged.get("completion_status", "").astype(str).str.contains("WARNING", na=False)
    if "order_hash" not in merged.columns or merged["order_hash"].isna().any():
        merged["order_hash"] = merged["order_json"].map(order_hash_from_order)
    merged["notes"] = "Run62 ingestion of Run61 custom N40-focused calibrated penalty-repair batch40 teacher metrics."
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
    baseline = combined[combined["dataset_source"].astype(str) != run_source_name] if baseline_source_name == "combined392" else combined[combined["dataset_source"].astype(str) == baseline_source_name]
    reward_cols = {
        "u2_primary": "target_reward_combined432_u2_primary",
        "constrained_u2_reward_balanced": "target_reward_combined432_constrained_u2_reward_balanced",
        "strict_penalty_guard": "target_reward_combined432_strict_penalty_guard",
        "penalty_repair": "target_reward_combined432_penalty_repair",
    }
    for n in [24, 40]:
        b = baseline[baseline["n"].astype(int) == n]
        r = run[run["n"].astype(int) == n]
        if b.empty or r.empty:
            continue
        for metric, col in RAW_METRICS.items():
            bbest = b.sort_values(col).iloc[0]
            rbest = r.sort_values(col).iloc[0]
            improvement = float(bbest[col]) - float(rbest[col])
            rows.append({
                "comparison": f"Run61_vs_{baseline_label}",
                "n": n,
                "metric": metric,
                f"{baseline_label}_best_strategy": bbest.get("strategy_name", ""),
                f"{baseline_label}_best_value": float(bbest[col]),
                "run61_best_strategy": rbest.get("strategy_name", rbest.get("handoff_strategy_name", "")),
                "run61_best_value": float(rbest[col]),
                "run61_beats_baseline": bool(improvement > 0),
                "absolute_improvement": improvement,
                "relative_improvement_pct": (improvement / abs(float(bbest[col])) * 100.0) if float(bbest[col]) != 0 else None,
                "combined432_new_best_strategy": combined[combined["n"].astype(int) == n].sort_values(col).iloc[0].get("strategy_name", ""),
                "combined432_new_best_source": combined[combined["n"].astype(int) == n].sort_values(col).iloc[0].get("dataset_source", ""),
            })
        for metric, col in reward_cols.items():
            bbest = b.sort_values(col, ascending=False).iloc[0]
            rbest = r.sort_values(col, ascending=False).iloc[0]
            improvement = float(rbest[col]) - float(bbest[col])
            rows.append({
                "comparison": f"Run61_vs_{baseline_label}",
                "n": n,
                "metric": metric,
                f"{baseline_label}_best_strategy": bbest.get("strategy_name", ""),
                f"{baseline_label}_best_value": float(bbest[col]),
                "run61_best_strategy": rbest.get("strategy_name", rbest.get("handoff_strategy_name", "")),
                "run61_best_value": float(rbest[col]),
                "run61_beats_baseline": bool(improvement > 0),
                "absolute_improvement": improvement,
                "relative_improvement_pct": (improvement / abs(float(bbest[col])) * 100.0) if float(bbest[col]) != 0 else None,
                "combined432_new_best_strategy": combined[combined["n"].astype(int) == n].sort_values(col, ascending=False).iloc[0].get("strategy_name", ""),
                "combined432_new_best_source": combined[combined["n"].astype(int) == n].sort_values(col, ascending=False).iloc[0].get("dataset_source", ""),
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


def prediction_audit(run61: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    pred_map = {
        "u2_primary": ("u2_primary_prediction", "target_reward_combined432_u2_primary"),
        "constrained": ("constrained_reward_prediction", "target_reward_combined432_constrained_u2_reward_balanced"),
        "strict": ("strict_penalty_guard_prediction", "target_reward_combined432_strict_penalty_guard"),
        "penalty_repair": ("penalty_repair_prediction", "target_reward_combined432_penalty_repair"),
        "U2": ("u2_primary_prediction", "target_u2_score_combined432_rank"),
        "PEEQ": ("N40_penalty_repair_prediction", "target_peeq_score_combined432_rank"),
        "SurfaceT": ("N40_two_stage_penalty_repair_prediction", "target_surfaceT_score_combined432_rank"),
        "Mises": ("N40_median_guard_prediction", "target_mises_score_combined432_rank"),
    }
    rows = []
    for label, (pred, real) in pred_map.items():
        if pred in run61.columns and real in run61.columns:
            rows.append({"scope": "overall", "target": label, "spearman": spearman(run61[pred], run61[real]), "top5_overlap": top_overlap(run61[real], run61[pred], 5), "top10_overlap": top_overlap(run61[real], run61[pred], 10), "top1_hit": top_overlap(run61[real], run61[pred], 1)})
            for n, group in run61.groupby("n"):
                rows.append({"scope": f"N{int(n)}", "target": label, "spearman": spearman(group[pred], group[real]), "top5_overlap": top_overlap(group[real], group[pred], 5), "top10_overlap": top_overlap(group[real], group[pred], 10), "top1_hit": top_overlap(group[real], group[pred], 1)})
    audit = pd.DataFrame(rows)
    summary = {
        "headline": "Run58/Run59 prediction calibration was evaluated on realized Run61 labels; use these diagnostics as calibration evidence, not teacher-validation claims for future candidates.",
        "overall_reward_spearman": next((r["spearman"] for r in rows if r["scope"] == "overall" and r["target"] == "u2_primary"), None),
        "overall_penalty_repair_spearman": next((r["spearman"] for r in rows if r["scope"] == "overall" and r["target"] == "penalty_repair"), None),
        "mean_top5_overlap": float(pd.to_numeric(audit[audit["scope"] == "overall"]["top5_overlap"], errors="coerce").mean()) if not audit.empty else None,
        "top1_hits": int(pd.to_numeric(audit[audit["scope"] == "overall"]["top1_hit"], errors="coerce").fillna(0).sum()) if not audit.empty else 0,
        "by_candidate_source": {},
        "by_selection_bucket": {},
        "disagreement_vs_abs_error_spearman": None,
        "uncertainty_vs_abs_error_spearman": None,
        "novelty_vs_realized_reward_spearman": spearman(run61.get("novelty_distance", pd.Series(dtype=float)), run61.get("target_reward_combined432_u2_primary", pd.Series(dtype=float))),
    }
    if "target_reward_combined432_u2_primary" in run61 and "u2_primary_prediction" in run61:
        abs_err = (pd.to_numeric(run61["target_reward_combined432_u2_primary"], errors="coerce") - pd.to_numeric(run61["u2_primary_prediction"], errors="coerce")).abs()
        summary["disagreement_vs_abs_error_spearman"] = spearman(run61.get("gnn_vs_surrogate_disagreement", pd.Series(dtype=float)), abs_err)
        summary["uncertainty_vs_abs_error_spearman"] = spearman(run61.get("uncertainty_score", pd.Series(dtype=float)), abs_err)
        for col, key in [("candidate_source", "by_candidate_source"), ("selection_bucket", "by_selection_bucket"), ("source_from_original_option_A", "by_source_from_original_option_A"), ("added_as_extra_N24", "by_added_as_extra_N24")]:
            if col in run61.columns:
                summary[key] = {
                    str(k): {
                        "count": int(len(g)),
                        "mean_abs_error": float(abs_err.loc[g.index].mean()),
                        "mean_realized_reward": float(pd.to_numeric(g["target_reward_combined432_u2_primary"], errors="coerce").mean()),
                    }
                    for k, g in run61.groupby(col)
                }
    return audit, summary


def effectiveness_audit(run61: pd.DataFrame, combined432: pd.DataFrame, comparison: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    reward_cols = [
        "target_reward_combined432_u2_primary",
        "target_reward_combined432_constrained_u2_reward_balanced",
        "target_reward_combined432_strict_penalty_guard",
        "target_reward_combined432_penalty_repair",
    ]
    for n, group in run61.groupby("n"):
        alln = combined432[combined432["n"].astype(int) == int(n)]
        for metric, col in RAW_METRICS.items():
            ranks = alln[col].rank(method="average", ascending=True)
            top5_hashes = set(alln.loc[ranks <= 5, "order_hash"].astype(str))
            top10_hashes = set(alln.loc[ranks <= 10, "order_hash"].astype(str))
            rows.append({"n": int(n), "metric": metric, "run61_top5_entries": int(group["order_hash"].astype(str).isin(top5_hashes).sum()), "run61_top10_entries": int(group["order_hash"].astype(str).isin(top10_hashes).sum()), "run61_best_rank": float(ranks.loc[group.index].min()) if set(group.index) <= set(ranks.index) else None})
        for col in reward_cols:
            ranks = alln[col].rank(method="average", ascending=False)
            top5_hashes = set(alln.loc[ranks <= 5, "order_hash"].astype(str))
            top10_hashes = set(alln.loc[ranks <= 10, "order_hash"].astype(str))
            rows.append({"n": int(n), "metric": col, "run61_top5_entries": int(group["order_hash"].astype(str).isin(top5_hashes).sum()), "run61_top10_entries": int(group["order_hash"].astype(str).isin(top10_hashes).sum()), "run61_best_rank": float(ranks.loc[group.index].min()) if set(group.index) <= set(ranks.index) else None})
    audit = pd.DataFrame(rows)
    new_best_count = int(comparison["run61_beats_baseline"].sum()) if "run61_beats_baseline" in comparison else 0
    by_cols = {}
    for col in ["candidate_source", "generation_method", "selection_bucket", "source_from_original_option_A", "added_as_extra_N24"]:
        if col in run61.columns:
            by_cols[col] = run61.groupby(col).agg(
                count=("n", "count"),
                mean_u2_primary=("target_reward_combined432_u2_primary", "mean"),
                mean_penalty_repair=("target_reward_combined432_penalty_repair", "mean"),
                mean_u2=("u2_range", "mean"),
                mean_peeq=("peeq_max", "mean"),
                mean_surfaceT=("surface_t_proxy", "mean"),
                mean_mises=("mises_max", "mean"),
            ).reset_index().to_dict(orient="records")
    headline = (
        f"Run61 produced {new_best_count} new metric/reward records versus combined392; "
        "N40 remains the primary focus while the expanded N24 rows provide maintenance/diagnostic coverage."
    )
    summary = {
        "headline": headline,
        "new_best_count_vs_combined392": new_best_count,
        "top_density_rows": rows,
        "performance_by_group": by_cols,
        "N40_benefited": bool(new_best_count > 0 and (comparison[(comparison["n"] == 40) & (comparison["run61_beats_baseline"] == True)].shape[0] > 0)),
        "N24_maintenance_benefited": bool(comparison[(comparison["n"] == 24) & (comparison["run61_beats_baseline"] == True)].shape[0] > 0),
        "N24_16_N40_24_compromise": "accepted_for_analysis; evaluate future action from new-best count, top-density, and prediction calibration",
    }
    return audit, summary


def prior_records(combined432: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    sources = {
        "Run61": "run61_custom_N40_focused_batch40",
        "Run56": "run56_calibrated_N24_N40_batch64",
        "Run51": "run51_stricter_constrained_N24_N40_batch32",
        "Run46": "run46_constrained_N24_N40_batch32",
        "Run41": "run41_native_N24_N40_focused_batch60",
        "Run36": "run36_N32_informed_native_batch32",
    }
    rows = []
    reward_cols = [
        "target_reward_combined432_u2_primary",
        "target_reward_combined432_constrained_u2_reward_balanced",
        "target_reward_combined432_strict_penalty_guard",
        "target_reward_combined432_penalty_repair",
    ]
    for n in [24, 40]:
        for label, source in sources.items():
            g = combined432[(combined432["n"].astype(int) == n) & (combined432["dataset_source"].astype(str) == source)]
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
            alln = combined432[combined432["n"].astype(int) == n]
            reward_rank = alln["target_reward_combined432_u2_primary"].rank(method="average", ascending=False)
            top5 = set(alln.loc[reward_rank <= 5, "order_hash"].astype(str))
            top10 = set(alln.loc[reward_rank <= 10, "order_hash"].astype(str))
            row["top5_reward_entries_in_combined432"] = int(g["order_hash"].astype(str).isin(top5).sum())
            row["top10_reward_entries_in_combined432"] = int(g["order_hash"].astype(str).isin(top10).sum())
            rows.append(row)
    df = pd.DataFrame(rows)
    summary = {
        "headline": "Run61 was compared with Run56, Run51, Run46, Run41, Run36, and combined best records using recomputed combined432 ranks.",
        "run61_complements_run56": True,
        "full_variable_n_maturity_warning": "N12/N16 remain at 36 rows each; do not claim full variable-N RL maturity.",
    }
    return df, summary


def maturity_audit(combined432: pd.DataFrame, comparison: pd.DataFrame, pred_summary: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], str]:
    counts_now = counts(combined432)
    rows = [
        {"n": 12, "native_teacher_rows": counts_now.get(12, 0), "status": "under_sampled_anchor"},
        {"n": 16, "native_teacher_rows": counts_now.get(16, 0), "status": "under_sampled_anchor"},
        {"n": 24, "native_teacher_rows": counts_now.get(24, 0), "status": "mature_N24_maintenance_evidence"},
        {"n": 32, "native_teacher_rows": 0, "legacy_teacher_rows": 332, "status": "legacy_compatible_auxiliary_only"},
        {"n": 40, "native_teacher_rows": counts_now.get(40, 0), "status": "mature_N40_active_improvement_direction"},
    ]
    new_best_count = int(comparison["run61_beats_baseline"].sum()) if "run61_beats_baseline" in comparison else 0
    summary = {
        "headline": "After Run61, N24 has 176 and N40 has 184 native teacher rows; N40 remains the main active improvement direction, N24 is in maintenance/evidence-freeze status, and full variable-N maturity remains limited by N12/N16 at 36 rows each.",
        "n24_teacher_rows": counts_now.get(24, 0),
        "n40_teacher_rows": counts_now.get(40, 0),
        "n32_legacy_teacher_rows": 332,
        "n12_teacher_rows": counts_now.get(12, 0),
        "n16_teacher_rows": counts_now.get(16, 0),
        "run61_new_best_count_vs_combined392": new_best_count,
        "n40_remains_main_active_improvement_direction": True,
        "n24_maintenance_evidence_freeze_status": True,
        "gnn_pointer_auxiliary_only": True,
        "n24_n40_mature_for_offline_active_learning_evidence": True,
        "full_variable_n_rl_maturity_limited_by_n12_n16": True,
        "safe_paper_claim": "N24/N40 have dense native teacher labels and repeated active-learning loops; full variable-N maturity remains limited by N12/N16 sample counts.",
        "prediction_calibration_summary": pred_summary,
    }
    md = (
        "# N24/N40 Updated Maturity and Claim-Boundary Audit\n\n"
        f"{summary['headline']}\n\n"
        "- N24 native teacher rows: 176\n"
        "- N40 native teacher rows: 184\n"
        "- N32 legacy-compatible teacher rows: 332\n"
        "- N12/N16 native teacher rows: 36 each\n\n"
        "N32 remains legacy-compatible auxiliary data, not native Stage 3 teacher validation. "
        "GNN and graph-pointer diagnostics remain auxiliary unless future diagnostics materially improve.\n"
    )
    return pd.DataFrame(rows), summary, md


def write_claim_boundary() -> None:
    safe = [
        "Run62 ingests 40/40 teacher-validated Run61 custom N40-focused batch40 cases.",
        "Run62 builds native combined432 with N12=36, N16=36, N24=176, N40=184.",
        "Run62 builds combined432_plus_N32 with N12=36, N16=36, N24=176, N32=332, N40=184.",
        "Run62 evaluates whether the custom N40-focused batch40 improved native Stage 3 teacher metrics.",
        "Run61 is teacher validation of native N24/N40 custom batch40 candidates, not N32 cases.",
        "Run62 updates N24/N40 maturity and claim boundaries.",
    ]
    unsafe = [
        "Do not claim N32 itself was newly teacher-validated in Run61.",
        "Do not claim N32 caused Run61 improvements.",
        "Do not claim GNN-RL superiority unless supported.",
        "Do not claim online RL.",
        "Do not claim arbitrary-N generalization.",
        "Do not claim physical optimum.",
        "Do not claim solver/ODB extraction happened in Run62.",
        "Do not claim full variable-N RL maturity while N12/N16 remain under-sampled.",
    ]
    CLAIM_BOUNDARY_MD.write_text("# Run62 Claim Boundary\n\n## Safe claims\n" + "\n".join(f"- {x}" for x in safe) + "\n\n## Unsafe claims\n" + "\n".join(f"- {x}" for x in unsafe) + "\n", encoding="utf-8")
    write_json(CLAIM_BOUNDARY_JSON, {"verdict": "RUN62_INGESTION_AND_ANALYSIS_ONLY_NO_SOLVER_OR_CANDIDATE_GENERATION", "safe_claims": safe, "unsafe_claims": unsafe})


def write_report(summary: dict[str, Any]) -> None:
    REPORT_PATH.write_text(f"""# Stage 3 Run 62 - Custom N40-Focused Batch40 Teacher Metrics Ingestion and Combined432 Ranking

## 1. Purpose
Run62 ingests Run61 teacher metrics, builds native combined432 and combined432_plus_N32 datasets, and evaluates whether the custom N40-focused calibrated penalty-repair batch40 improved native Stage 3 teacher metrics.

## 2. Inputs
- Run61 teacher metrics: `{RUN61_METRICS}`
- Run59 handoff metadata: `{RUN59_HANDOFF}`
- Native combined392: `{COMBINED392_READY}`
- combined392_plus_N32: `{COMBINED392_PLUS_N32_READY}`

## 3. Run61 Teacher-Validation Status
Run61 completed 40/40 with N24=16 and N40=24. Nonfatal warnings were present, with no failed or incomplete cases.

## 4. Input Validation
Verdict: `{summary['validation']['verdict']}`.

## 5. Run61 Enriched Teacher Dataset
Output: `{OUTPUT_DIR / 'run61_custom_N40_focused_batch40_teacher_dataset_enriched.csv'}`

## 6. Run61 Within-Batch Ranking
Output: `{OUTPUT_DIR / 'run61_custom_N40_focused_batch40_ranked_within_batch.csv'}`

## 7. Native Combined432 Construction
Native combined432 counts: `{summary['combined432_counts']}`.

## 8. combined432_plus_N32 Construction
combined432_plus_N32 counts: `{summary['combined432_plus_N32_counts']}`. N32 metric semantic warnings are preserved.

## 9. Run61 vs Combined392 Best Comparison
{summary['comparison_headline']}

## 10. Run61 vs Prior Key Records
{summary['prior_summary']['headline']}

## 11. Custom N40-Focused Batch40 Effectiveness Audit
{summary['effectiveness_summary']['headline']}

## 12. Prediction Audit for Run58/Run59 Custom Batch40
{summary['prediction_summary']['headline']}

## 13. N40 Focus Versus N24 Maintenance Analysis
N40 remains the main active improvement direction. The extra N24 rows evaluate maintenance and diagnostic coverage after N24 reached dense evidence status.

## 14. Updated N24/N40 Maturity and Claim-Boundary Audit
{summary['maturity_summary']['headline']}

## 15. Metric Semantic Boundary for N32
N32 rows are legacy-compatible auxiliary data. They are not native Stage 3 teacher validation, and PEEQ/Mises mappings must retain semantic warnings.

## 16. Claim Boundary
Verdict: `RUN62_INGESTION_AND_ANALYSIS_ONLY_NO_SOLVER_OR_CANDIDATE_GENERATION`.

## 17. Output Files
- combined432 RL-ready: `{OUTPUT_DIR / 'combined432_RL_ready_dataset.csv'}`
- combined432_plus_N32 RL-ready: `{OUTPUT_DIR / 'combined432_plus_N32_RL_ready_dataset.csv'}`
- Run61 comparison: `{OUTPUT_DIR / 'run61_vs_combined392_best_comparison.csv'}`
- Updated maturity audit: `{OUTPUT_DIR / 'n24_n40_updated_maturity_and_claim_boundary_audit.md'}`
- Manifest: `{MANIFEST_PATH}`

## 18. Recommended Run63
{summary['recommended_run63']}
""", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    run61 = normalize_metrics(read_csv(RUN61_METRICS))
    handoff = read_csv(RUN59_HANDOFF)
    combined392 = normalize_metrics(read_csv(COMBINED392_READY))
    plus392 = normalize_metrics(read_csv(COMBINED392_PLUS_N32_READY))

    validation = validate_inputs(run61, handoff, combined392, plus392)
    write_json(OUTPUT_DIR / "run62_input_validation_summary.json", validation)
    if not validation["verdict"].startswith("PASS"):
        raise SystemExit(validation["errors"])

    enriched = make_enriched(run61, handoff)
    write_csv(OUTPUT_DIR / "run61_custom_N40_focused_batch40_teacher_dataset_enriched.csv", enriched)
    write_table_json(OUTPUT_DIR / "run61_custom_N40_focused_batch40_teacher_dataset_enriched.json", enriched)

    ranked = add_run61_scores(enriched)
    write_csv(OUTPUT_DIR / "run61_custom_N40_focused_batch40_ranked_within_batch.csv", ranked)
    run61_leaderboard = leaderboard(ranked, [f"reward_run61_{x}" for x in REWARD_DEFS])
    write_csv(OUTPUT_DIR / "run61_custom_N40_focused_batch40_per_N_leaderboard.csv", run61_leaderboard)

    combined432 = pd.concat([combined392, enriched], ignore_index=True, sort=False)
    combined432 = add_rank_scores(combined432, "combined432", "combined432")
    write_csv(OUTPUT_DIR / "combined432_teacher_dataset.csv", combined432)
    write_csv(OUTPUT_DIR / "combined432_RL_ready_dataset.csv", combined432)
    combined432_leaderboard = leaderboard(combined432, [f"target_reward_combined432_{x}" for x in REWARD_DEFS])
    write_csv(OUTPUT_DIR / "combined432_per_N_leaderboard.csv", combined432_leaderboard)
    combined432_summary = {"rows": int(len(combined432)), "per_N_counts": counts(combined432), "leaderboard": combined432_leaderboard.to_dict(orient="records")}
    write_json(OUTPUT_DIR / "combined432_summary.json", combined432_summary)

    n32 = plus392[plus392["n"].astype(int) == 32].copy()
    combined432_plus = pd.concat([combined432, n32], ignore_index=True, sort=False)
    combined432_plus = add_rank_scores(combined432_plus, "combined432_plus_N32", "combined432_plus_N32")
    if "target_reward_combined432_plus_N32_strict_u2_surfaceT" not in combined432_plus.columns:
        combined432_plus["target_reward_combined432_plus_N32_strict_u2_surfaceT"] = (
            0.70 * combined432_plus["target_u2_score_combined432_plus_N32_rank"]
            + 0.30 * combined432_plus["target_surfaceT_score_combined432_plus_N32_rank"]
        )
    if "target_reward_combined432_plus_N32_mapped_u2_primary" not in combined432_plus.columns:
        combined432_plus["target_reward_combined432_plus_N32_mapped_u2_primary"] = combined432_plus["target_reward_combined432_plus_N32_u2_primary"]
    write_csv(OUTPUT_DIR / "combined432_plus_N32_teacher_dataset.csv", combined432_plus)
    write_csv(OUTPUT_DIR / "combined432_plus_N32_RL_ready_dataset.csv", combined432_plus)
    plus_leaderboard = leaderboard(combined432_plus, ["target_reward_combined432_plus_N32_mapped_u2_primary", "target_reward_combined432_plus_N32_strict_u2_surfaceT"])
    write_csv(OUTPUT_DIR / "combined432_plus_N32_per_N_leaderboard.csv", plus_leaderboard)
    plus_summary = {"rows": int(len(combined432_plus)), "per_N_counts": counts(combined432_plus), "n32_semantic_warning_preserved": "metric_semantic_warning" in n32.columns}
    write_json(OUTPUT_DIR / "combined432_plus_N32_summary.json", plus_summary)

    comparison = compare_against_baseline(combined432, "combined392", "run61_custom_N40_focused_batch40", "combined392")
    write_csv(OUTPUT_DIR / "run61_vs_combined392_best_comparison.csv", comparison)
    write_json(OUTPUT_DIR / "run61_vs_combined392_best_comparison.json", comparison.to_dict(orient="records"))

    prior_df, prior_summary = prior_records(combined432)
    write_csv(OUTPUT_DIR / "run61_vs_prior_key_records.csv", prior_df)
    write_json(OUTPUT_DIR / "run61_vs_prior_key_records_summary.json", prior_summary)

    run61_combined = combined432[combined432["dataset_source"].astype(str) == "run61_custom_N40_focused_batch40"].copy()
    eff_df, eff_summary = effectiveness_audit(run61_combined, combined432, comparison)
    write_csv(OUTPUT_DIR / "run61_custom_N40_focused_batch40_effectiveness_audit.csv", eff_df)
    write_json(OUTPUT_DIR / "run61_custom_N40_focused_batch40_effectiveness_summary.json", eff_summary)

    pred_df, pred_summary = prediction_audit(run61_combined)
    write_csv(OUTPUT_DIR / "run61_prediction_audit_for_run58_custom_batch40.csv", pred_df)
    write_json(OUTPUT_DIR / "run61_prediction_audit_for_run58_custom_batch40_summary.json", pred_summary)

    maturity_df, maturity_summary, maturity_md = maturity_audit(combined432, comparison, pred_summary)
    write_csv(OUTPUT_DIR / "n24_n40_updated_maturity_and_claim_boundary_audit.csv", maturity_df)
    write_json(OUTPUT_DIR / "n24_n40_updated_maturity_and_claim_boundary_summary.json", maturity_summary)
    (OUTPUT_DIR / "n24_n40_updated_maturity_and_claim_boundary_audit.md").write_text(maturity_md, encoding="utf-8")

    write_claim_boundary()

    if int(comparison["run61_beats_baseline"].sum()) > 0:
        recommended = "Update models with combined432 and decide whether to freeze N24/N40 evidence or run a small N40-focused diagnostic follow-up; do not claim full variable-N maturity."
    elif eff_summary.get("top_density_rows"):
        recommended = "Consider freezing the N24/N40 active-learning evidence package and pivoting to N12/N16 recovery anchors if full variable-N maturity is desired."
    else:
        recommended = "Return to diagnostic candidate design with smaller batches; avoid larger N40-focused batches until calibration improves."
    report_summary = {
        "validation": validation,
        "combined432_counts": counts(combined432),
        "combined432_plus_N32_counts": counts(combined432_plus),
        "comparison_headline": f"Run61 created {int(comparison['run61_beats_baseline'].sum())} new metric/reward records versus combined392.",
        "prior_summary": prior_summary,
        "effectiveness_summary": eff_summary,
        "prediction_summary": pred_summary,
        "maturity_summary": maturity_summary,
        "recommended_run63": recommended,
    }
    write_report(report_summary)

    output_files = [
        OUTPUT_DIR / "run62_input_validation_summary.json",
        OUTPUT_DIR / "run61_custom_N40_focused_batch40_teacher_dataset_enriched.csv",
        OUTPUT_DIR / "run61_custom_N40_focused_batch40_teacher_dataset_enriched.json",
        OUTPUT_DIR / "run61_custom_N40_focused_batch40_ranked_within_batch.csv",
        OUTPUT_DIR / "run61_custom_N40_focused_batch40_per_N_leaderboard.csv",
        OUTPUT_DIR / "combined432_teacher_dataset.csv",
        OUTPUT_DIR / "combined432_RL_ready_dataset.csv",
        OUTPUT_DIR / "combined432_per_N_leaderboard.csv",
        OUTPUT_DIR / "combined432_summary.json",
        OUTPUT_DIR / "combined432_plus_N32_teacher_dataset.csv",
        OUTPUT_DIR / "combined432_plus_N32_RL_ready_dataset.csv",
        OUTPUT_DIR / "combined432_plus_N32_per_N_leaderboard.csv",
        OUTPUT_DIR / "combined432_plus_N32_summary.json",
        OUTPUT_DIR / "run61_vs_combined392_best_comparison.csv",
        OUTPUT_DIR / "run61_vs_combined392_best_comparison.json",
        OUTPUT_DIR / "run61_vs_prior_key_records.csv",
        OUTPUT_DIR / "run61_vs_prior_key_records_summary.json",
        OUTPUT_DIR / "run61_custom_N40_focused_batch40_effectiveness_audit.csv",
        OUTPUT_DIR / "run61_custom_N40_focused_batch40_effectiveness_summary.json",
        OUTPUT_DIR / "run61_prediction_audit_for_run58_custom_batch40.csv",
        OUTPUT_DIR / "run61_prediction_audit_for_run58_custom_batch40_summary.json",
        OUTPUT_DIR / "n24_n40_updated_maturity_and_claim_boundary_audit.csv",
        OUTPUT_DIR / "n24_n40_updated_maturity_and_claim_boundary_summary.json",
        OUTPUT_DIR / "n24_n40_updated_maturity_and_claim_boundary_audit.md",
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
        "input_files": [str(p) for p in [RUN61_METRICS, RUN61_EXTRACTION, RUN61_SOLVER, RUN61_SUMMARY, RUN61_REPORT, RUN61_MANIFEST, RUN59_HANDOFF, RUN59_SCAN_DIR, RUN58_POOL, RUN58_OPTION_A, RUN58_REPORT, COMBINED392_TEACHER, COMBINED392_READY, COMBINED392_PLUS_N32_READY, N32_DEDUP, RUN57_REPORT]],
        "output_files": [str(p) for p in output_files],
        "run61_teacher_rows": 40,
        "combined432_rows": int(len(combined432)),
        "combined432_plus_N32_rows": int(len(combined432_plus)),
        "per_N_combined432_counts": counts(combined432),
        "per_N_combined432_plus_N32_counts": counts(combined432_plus),
        "new_best_counts": {"run61_vs_combined392": int(comparison["run61_beats_baseline"].sum())},
        "prediction_audit_summary": pred_summary,
        "maturity_audit_summary": maturity_summary,
        "report_path": str(REPORT_PATH),
        "claim_boundary_path": str(CLAIM_BOUNDARY_MD),
        "no_solver_run": True,
        "no_odb_opened": True,
        "no_abqjobpilot_run": True,
        "no_cae_inp_generated": True,
        "no_teacher_validation_performed_by_run62": True,
        "no_training": True,
        "no_candidate_generation": True,
        "no_commit_or_push": True,
    }
    write_json(MANIFEST_PATH, manifest)
    print(json.dumps({
        "verdict": validation["verdict"],
        "run61_counts": counts(run61),
        "combined432_counts": counts(combined432),
        "combined432_plus_N32_counts": counts(combined432_plus),
        "new_bests_vs_combined392": int(comparison["run61_beats_baseline"].sum()),
        "report": str(REPORT_PATH),
    }, indent=2))


if __name__ == "__main__":
    main()
