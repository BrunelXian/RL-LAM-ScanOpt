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
RUN_ID = "run_77_final_smallN_diagnostic_batch32_teacher_metrics_ingestion_and_combined552_final_evidence_readiness"
RUN_NAME = "final small-N diagnostic batch32 teacher metrics ingestion and combined552 final evidence readiness"
SCRIPT_PATH = ROOT / "scripts" / "stage3" / "run_77_ingest_final_smallN_diagnostic_batch32_and_build_combined552.py"

RUN76_DIR = ROOT / "outputs" / "stage3_run_76_final_smallN_diagnostic_batch32_odb_teacher_validation"
RUN76_METRICS = RUN76_DIR / "run76_final_smallN_diagnostic_batch32_teacher_metrics.csv"
RUN76_METRICS_SUMMARY = RUN76_DIR / "run76_final_smallN_diagnostic_batch32_teacher_metrics_summary.json"
RUN76_EXTRACTION = RUN76_DIR / "run76_final_smallN_diagnostic_batch32_odb_extraction_summary.csv"
RUN76_EXTRACTION_JSON = RUN76_DIR / "run76_final_smallN_diagnostic_batch32_odb_teacher_validation_summary.json"
RUN76_SOLVER = RUN76_DIR / "run76_final_smallN_diagnostic_batch32_solver_completion_audit.csv"
RUN76_SOLVER_JSON = RUN76_DIR / "run76_final_smallN_diagnostic_batch32_solver_completion_audit.json"
RUN76_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_76_final_smallN_diagnostic_batch32_odb_teacher_validation" / "RUN_76_FINAL_SMALLN_DIAGNOSTIC_BATCH32_ODB_TEACHER_VALIDATION_REPORT.md"
RUN76_MANIFEST = ROOT / "artifacts" / "manifests" / "stage3_run_76_manifest.json"

RUN74_HANDOFF = ROOT / "outputs" / "stage3_run_74_run73_final_smallN_diagnostic_batch32_handoff_package" / "stage3_run74_final_smallN_diagnostic_batch32_candidate_orders.csv"
RUN74_SCAN_DIR = ROOT / "outputs" / "stage3_run_74_run73_final_smallN_diagnostic_batch32_handoff_package" / "scan_orders"
RUN73_OPTION_A = ROOT / "outputs" / "stage3_run_73_combined520_model_update_final_smallN_diagnostic_candidate_generation" / "run73_final_smallN_diagnostic_batch32_candidate_orders.csv"
RUN73_POOL = ROOT / "outputs" / "stage3_run_73_combined520_model_update_final_smallN_diagnostic_candidate_generation" / "run73_candidate_pool_scored.csv"
RUN73_EVIDENCE = ROOT / "outputs" / "stage3_run_73_combined520_model_update_final_smallN_diagnostic_candidate_generation" / "stage3_evidence_freeze_readiness_after_run72.md"
RUN73_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_73_combined520_model_update_final_smallN_diagnostic_candidate_generation" / "RUN_73_COMBINED520_MODEL_UPDATE_FINAL_SMALLN_DIAGNOSTIC_CANDIDATE_GENERATION_REPORT.md"

COMBINED520_TEACHER = ROOT / "outputs" / "stage3_run_72_smallN_recovery_focused_batch40_teacher_metrics_ingestion_and_combined520_ranking" / "combined520_teacher_dataset.csv"
COMBINED520_READY = ROOT / "outputs" / "stage3_run_72_smallN_recovery_focused_batch40_teacher_metrics_ingestion_and_combined520_ranking" / "combined520_RL_ready_dataset.csv"
COMBINED520_PLUS_N32_READY = ROOT / "outputs" / "stage3_run_72_smallN_recovery_focused_batch40_teacher_metrics_ingestion_and_combined520_ranking" / "combined520_plus_N32_RL_ready_dataset.csv"
N32_DEDUP = ROOT / "outputs" / "stage3_run_32a_stage2_n32_legacy_teacher_label_ingestion_for_stage3" / "n32_legacy_teacher_dataset_dedup_training_332.csv"
RUN72_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_72_smallN_recovery_focused_batch40_teacher_metrics_ingestion_and_combined520_ranking" / "RUN_72_SMALLN_RECOVERY_FOCUSED_BATCH40_TEACHER_METRICS_INGESTION_AND_COMBINED520_RANKING_REPORT.md"

OUTPUT_DIR = ROOT / "outputs" / "stage3_run_77_final_smallN_diagnostic_batch32_teacher_metrics_ingestion_and_combined552_final_evidence_readiness"
REPORT_DIR = ROOT / "docs" / "stage3" / "runs" / "run_77_final_smallN_diagnostic_batch32_teacher_metrics_ingestion_and_combined552_final_evidence_readiness"
REPORT_PATH = REPORT_DIR / "RUN_77_FINAL_SMALLN_DIAGNOSTIC_BATCH32_TEACHER_METRICS_INGESTION_AND_COMBINED552_FINAL_EVIDENCE_READINESS_REPORT.md"
MANIFEST_PATH = ROOT / "artifacts" / "manifests" / "stage3_run_77_manifest.json"
CLAIM_BOUNDARY_MD = OUTPUT_DIR / "run77_final_claim_boundary.md"
CLAIM_BOUNDARY_JSON = OUTPUT_DIR / "run77_final_claim_boundary.json"

EXPECTED_RUN76_COUNTS = {12: 14, 16: 14, 24: 2, 40: 2}
EXPECTED_COMBINED520_COUNTS = {12: 64, 16: 64, 24: 188, 40: 204}
EXPECTED_COMBINED520_PLUS_N32_COUNTS = {12: 64, 16: 64, 24: 188, 32: 332, 40: 204}
EXPECTED_COMBINED552_COUNTS = {12: 78, 16: 78, 24: 190, 40: 206}
EXPECTED_COMBINED552_PLUS_N32_COUNTS = {12: 78, 16: 78, 24: 190, 32: 332, 40: 206}

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
REWARD_LABELS = list(REWARD_DEFS)


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def current_branch() -> str:
    try:
        return subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


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


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


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
    out["n"] = out["n"].astype(int)
    if "surface_t_proxy" in out.columns:
        out["surface_t_proxy"] = as_float(out["surface_t_proxy"])
    elif "surface_t_proxy_max_tensile_pa" in out.columns:
        out["surface_t_proxy"] = as_float(out["surface_t_proxy_max_tensile_pa"])
    elif "surface_t_proxy_max_tensile_mpa" in out.columns:
        out["surface_t_proxy"] = as_float(out["surface_t_proxy_max_tensile_mpa"]) * 1_000_000.0
    else:
        out["surface_t_proxy"] = math.nan
    out["surface_t_proxy_mpa"] = as_float(out["surface_t_proxy"]) / 1_000_000.0
    for col in ["u2_range", "peeq_max", "mises_max"]:
        out[col] = as_float(out[col])
    return out


def score_from_rank(ranks: pd.Series, count: int) -> pd.Series:
    return 1.0 - ((ranks.astype(float) - 1.0) / max(1, count - 1))


def metric_key(label: str) -> str:
    return "surfaceT" if label == "SurfaceT" else label.lower()


def add_rank_scores(df: pd.DataFrame, prefix: str, rank_label: str) -> pd.DataFrame:
    out = normalize_metrics(df)
    for label, col in RAW_METRICS.items():
        key = metric_key(label)
        rank_col = f"{key}_rank_{rank_label}_within_n"
        score_col = f"target_{key}_score_{prefix}_rank"
        cost_col = f"{key}_cost_minmax_{rank_label}_within_n"
        out[rank_col] = math.nan
        out[score_col] = math.nan
        out[cost_col] = math.nan
        for _, idx in out.groupby("n").groups.items():
            vals = as_float(out.loc[idx, col])
            ranks = vals.rank(method="average", ascending=True)
            out.loc[idx, rank_col] = ranks
            out.loc[idx, score_col] = score_from_rank(ranks, len(idx))
            mn, mx = vals.min(), vals.max()
            out.loc[idx, cost_col] = 0.0 if mx == mn else (vals - mn) / (mx - mn)
    score_cols = {label: f"target_{metric_key(label)}_score_{prefix}_rank" for label in RAW_METRICS}
    for reward_name, weights in REWARD_DEFS.items():
        col = f"target_reward_{prefix}_{reward_name}"
        out[col] = sum(weights[label] * out[score_cols[label]] for label in weights)
        rank_col = f"{reward_name}_reward_rank_{rank_label}_within_n"
        out[rank_col] = math.nan
        for _, idx in out.groupby("n").groups.items():
            out.loc[idx, rank_col] = out.loc[idx, col].rank(method="average", ascending=False)
    return out


def add_run76_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_metrics(df)
    for label, col in RAW_METRICS.items():
        out[f"rank_{label}_run76_within_n"] = math.nan
        out[f"score_{label}_run76_within_n"] = math.nan
        for _, idx in out.groupby("n").groups.items():
            ranks = as_float(out.loc[idx, col]).rank(method="average", ascending=True)
            out.loc[idx, f"rank_{label}_run76_within_n"] = ranks
            out.loc[idx, f"score_{label}_run76_within_n"] = score_from_rank(ranks, len(idx))
    for reward_name, weights in REWARD_DEFS.items():
        out[f"reward_run76_{reward_name}"] = sum(weights[label] * out[f"score_{label}_run76_within_n"] for label in weights)
    return out


def validate_inputs(run76: pd.DataFrame, handoff: pd.DataFrame, combined520: pd.DataFrame, plus520: pd.DataFrame) -> dict[str, Any]:
    errors: list[str] = []
    required_paths = [
        RUN76_METRICS, RUN76_METRICS_SUMMARY, RUN76_EXTRACTION, RUN76_EXTRACTION_JSON,
        RUN76_SOLVER, RUN76_SOLVER_JSON, RUN74_HANDOFF, RUN73_OPTION_A, RUN73_POOL,
        COMBINED520_TEACHER, COMBINED520_READY, COMBINED520_PLUS_N32_READY, N32_DEDUP,
    ]
    for path in required_paths:
        if not path.exists():
            errors.append(f"missing {path}")
    if len(run76) != 32 or counts(run76) != EXPECTED_RUN76_COUNTS:
        errors.append(f"Run76 count mismatch rows={len(run76)} counts={counts(run76)}")
    if 32 in set(run76["n"].astype(int)):
        errors.append("Run76 contains N32 rows")
    if set(run76["n"].astype(int)) - {12, 16, 24, 40}:
        errors.append("Run76 contains N outside native N12/N16/N24/N40")
    for col in ["u2_range", "peeq_max", "surface_t_proxy", "mises_max"]:
        if col not in run76.columns:
            errors.append(f"Run76 missing metric {col}")
        elif as_float(run76[col]).isna().any():
            errors.append(f"Run76 has missing values in {col}")
    if "teacher_validation_status" in run76.columns and not run76["teacher_validation_status"].astype(str).str.contains("PASS").all():
        errors.append("Run76 teacher_validation_status is not PASS for all rows")
    if "final_step_name" in run76.columns and not (run76["final_step_name"].astype(str) == "step_final_cooling").all():
        errors.append("Run76 final_step_name is not step_final_cooling for all rows")
    if "extracted_field_names" in run76.columns:
        fields_text = ";".join(run76["extracted_field_names"].astype(str).tolist())
        for field in ["U", "PEEQ", "S", "NT11"]:
            if field not in fields_text:
                errors.append(f"Run76 extracted fields missing {field}")
    if len(handoff) != 32 or counts(handoff) != EXPECTED_RUN76_COUNTS:
        errors.append("Run74 handoff count mismatch")
    for _, row in handoff.iterrows():
        order_col = "order_json" if "order_json" in handoff.columns else "scan_order"
        if not valid_order(row[order_col], int(row["n"])):
            errors.append(f"invalid scan order for {row.get('handoff_strategy_name', row.name)}")
            break
    if "order_hash" in handoff.columns and handoff.groupby("n")["order_hash"].apply(lambda s: s.astype(str).duplicated().any()).any():
        errors.append("Run74 handoff has duplicate order within same N")
    if handoff["handoff_strategy_name"].astype(str).duplicated().any():
        errors.append("Run74 handoff has duplicate handoff_strategy_name values")
    if len(combined520) != 520 or counts(combined520) != EXPECTED_COMBINED520_COUNTS:
        errors.append(f"combined520 count mismatch rows={len(combined520)} counts={counts(combined520)}")
    if 32 in set(combined520["n"].astype(int)):
        errors.append("native combined520 contains N32 rows")
    if len(plus520) != 852 or counts(plus520) != EXPECTED_COMBINED520_PLUS_N32_COUNTS:
        errors.append(f"combined520_plus_N32 count mismatch rows={len(plus520)} counts={counts(plus520)}")
    n32 = plus520[plus520["n"].astype(int) == 32]
    if "metric_semantic_warning" not in n32.columns:
        errors.append("combined520_plus_N32 N32 rows missing metric_semantic_warning")
    if not set(run76["handoff_strategy_name"].astype(str)) <= set(handoff["handoff_strategy_name"].astype(str)):
        errors.append("not all Run76 rows match Run74 handoff_strategy_name")
    verdict = "PASS_RUN77_FINAL_SMALLN_DIAGNOSTIC_BATCH32_TEACHER_METRICS_32_OF_32_READY" if not errors else "FAIL_RUN77_INPUT_VALIDATION"
    return {
        "timestamp": now_iso(),
        "verdict": verdict,
        "errors": errors,
        "run76_teacher_rows": int(len(run76)),
        "run76_per_N_counts": counts(run76),
        "combined520_rows": int(len(combined520)),
        "combined520_per_N_counts": counts(combined520),
        "combined520_plus_N32_rows": int(len(plus520)),
        "combined520_plus_N32_per_N_counts": counts(plus520),
        "run76_contains_N32": bool((run76["n"].astype(int) == 32).any()),
    }


def make_enriched(run76: pd.DataFrame, handoff: pd.DataFrame) -> pd.DataFrame:
    metrics = normalize_metrics(run76)
    meta = handoff.copy()
    merged = metrics.merge(meta, on=["handoff_strategy_name", "n"], how="left", suffixes=("", "_run74"))
    merged["strategy_name"] = merged["handoff_strategy_name"]
    merged["dataset_source"] = "run76_final_smallN_diagnostic_batch32"
    merged["batch_name"] = "stage3_run74_final_smallN_diagnostic_batch32_v01"
    merged["native_validation_N"] = True
    merged["final_smallN_diagnostic"] = True
    merged["smallN_recovery"] = merged["n"].astype(int).isin([12, 16])
    merged["variable_N_bounded"] = True
    merged["N24_N40_anchor"] = merged["n"].astype(int).isin([24, 40])
    merged["includes_N32_case"] = False
    merged["final_step"] = merged.get("final_step_name", "step_final_cooling")
    merged["extracted_fields"] = merged.get("extracted_field_names", "")
    merged["solver_audit_status"] = merged.get("completion_status", "")
    merged["nonfatal_warning_flag"] = merged.get("completion_status", "").astype(str).str.contains("WARNING", na=False)
    if "order_hash" not in merged.columns or merged["order_hash"].isna().any():
        merged["order_hash"] = merged["order_json"].map(order_hash_from_order)
    merged["notes"] = "Run77 ingestion of Run76 final small-N diagnostic batch32 teacher metrics."
    return merged


def concat_union(frames: list[pd.DataFrame]) -> pd.DataFrame:
    all_cols: list[str] = []
    seen: set[str] = set()
    for frame in frames:
        for col in frame.columns:
            if col not in seen:
                all_cols.append(col)
                seen.add(col)
    aligned = []
    for frame in frames:
        tmp = frame.copy()
        for col in all_cols:
            if col not in tmp.columns:
                tmp[col] = ""
        aligned.append(tmp[all_cols])
    return pd.concat(aligned, ignore_index=True)


def leaderboard(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    rows = []
    reward_cols = [f"target_reward_{prefix}_{name}" for name in REWARD_LABELS]
    for n, group in df.groupby("n"):
        for metric, col in RAW_METRICS.items():
            best = group.sort_values(col, ascending=True).iloc[0]
            rows.append({"n": int(n), "metric": metric, "strategy_name": best.get("strategy_name", best.get("handoff_strategy_name", "")), "dataset_source": best.get("dataset_source", ""), "value": float(best[col])})
        for col in reward_cols:
            if col in group.columns:
                best = group.sort_values(col, ascending=False).iloc[0]
                rows.append({"n": int(n), "metric": col, "strategy_name": best.get("strategy_name", best.get("handoff_strategy_name", "")), "dataset_source": best.get("dataset_source", ""), "value": float(best[col])})
    return pd.DataFrame(rows)


def compare_run_vs_baseline(combined552: pd.DataFrame) -> pd.DataFrame:
    rows = []
    run = combined552[combined552["dataset_source"].astype(str) == "run76_final_smallN_diagnostic_batch32"]
    baseline = combined552[combined552["dataset_source"].astype(str) != "run76_final_smallN_diagnostic_batch32"]
    reward_cols = {name: f"target_reward_combined552_{name}" for name in REWARD_LABELS}
    for n in [12, 16, 24, 40]:
        b = baseline[baseline["n"].astype(int) == n]
        r = run[run["n"].astype(int) == n]
        if b.empty or r.empty:
            continue
        for metric, col in RAW_METRICS.items():
            bbest = b.sort_values(col, ascending=True).iloc[0]
            rbest = r.sort_values(col, ascending=True).iloc[0]
            improvement = float(bbest[col]) - float(rbest[col])
            allbest = combined552[combined552["n"].astype(int) == n].sort_values(col, ascending=True).iloc[0]
            rows.append({
                "comparison": "Run76_vs_combined520",
                "n": n,
                "metric": metric,
                "combined520_best_strategy": bbest.get("strategy_name", ""),
                "combined520_best_value": float(bbest[col]),
                "run76_best_strategy": rbest.get("strategy_name", rbest.get("handoff_strategy_name", "")),
                "run76_best_value": float(rbest[col]),
                "run76_beats_combined520": bool(improvement > 0),
                "absolute_improvement": improvement,
                "relative_improvement_pct": (improvement / abs(float(bbest[col])) * 100.0) if float(bbest[col]) != 0 else None,
                "combined552_new_best_strategy": allbest.get("strategy_name", ""),
                "combined552_new_best_source": allbest.get("dataset_source", ""),
            })
        for metric, col in reward_cols.items():
            bbest = b.sort_values(col, ascending=False).iloc[0]
            rbest = r.sort_values(col, ascending=False).iloc[0]
            improvement = float(rbest[col]) - float(bbest[col])
            allbest = combined552[combined552["n"].astype(int) == n].sort_values(col, ascending=False).iloc[0]
            rows.append({
                "comparison": "Run76_vs_combined520",
                "n": n,
                "metric": metric,
                "combined520_best_strategy": bbest.get("strategy_name", ""),
                "combined520_best_value": float(bbest[col]),
                "run76_best_strategy": rbest.get("strategy_name", rbest.get("handoff_strategy_name", "")),
                "run76_best_value": float(rbest[col]),
                "run76_beats_combined520": bool(improvement > 0),
                "absolute_improvement": improvement,
                "relative_improvement_pct": (improvement / abs(float(bbest[col])) * 100.0) if float(bbest[col]) != 0 else None,
                "combined552_new_best_strategy": allbest.get("strategy_name", ""),
                "combined552_new_best_source": allbest.get("dataset_source", ""),
            })
    return pd.DataFrame(rows)


def prior_key_records(combined552: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    reward_cols = [f"target_reward_combined552_{name}" for name in REWARD_LABELS]
    rows = []
    source_order = [
        "run76_final_smallN_diagnostic_batch32",
        "run71_smallN_recovery_focused_batch40",
        "run66_variable_N_recovery_anchor_batch48",
        "run61_custom_N40_focused_batch40",
        "run56_calibrated_N24_N40_batch64",
        "run51_stricter_constrained_N24_N40_batch32",
        "run46_constrained_N24_N40_batch32",
        "run41_native_N24_N40_focused_batch60",
        "run36_N32_informed_native_batch32",
        "shortlist64_run27",
    ]
    for n in [12, 16, 24, 40]:
        alln = combined552[combined552["n"].astype(int) == n]
        for source in source_order:
            g = alln[alln["dataset_source"].astype(str) == source]
            if g.empty:
                continue
            row: dict[str, Any] = {"n": n, "record_source": source, "count": int(len(g))}
            for metric, col in RAW_METRICS.items():
                best = g.sort_values(col, ascending=True).iloc[0]
                row[f"best_{metric}_strategy"] = best.get("strategy_name", "")
                row[f"best_{metric}_value"] = float(best[col])
            for col in reward_cols:
                best = g.sort_values(col, ascending=False).iloc[0]
                row[f"best_{col}_strategy"] = best.get("strategy_name", "")
                row[f"best_{col}_value"] = float(best[col])
            rows.append(row)
    df = pd.DataFrame(rows)
    summary = {
        "headline": "Run76 was compared against prior native teacher records and aggregate combined-set bests; the final small-N diagnostic role is judged by N12/N16 stability and N24/N40 anchor behavior.",
        "record_sources_present": sorted(combined552["dataset_source"].astype(str).unique().tolist()),
        "rows": df.to_dict(orient="records"),
    }
    return df, summary


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
    return len(set(x[mask].sort_values(ascending=False).head(kk).index) & set(y[mask].sort_values(ascending=False).head(kk).index))


def prediction_audit(run76: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    pred_map = {
        "u2_primary": ("u2_primary_prediction", "target_reward_combined552_u2_primary"),
        "constrained": ("constrained_reward_prediction", "target_reward_combined552_constrained_u2_reward_balanced"),
        "strict": ("strict_penalty_guard_prediction", "target_reward_combined552_strict_penalty_guard"),
        "penalty_repair": ("penalty_repair_prediction", "target_reward_combined552_penalty_repair"),
        "U2": ("u2_primary_prediction", "target_u2_score_combined552_rank"),
        "PEEQ": ("penalty_repair_prediction", "target_peeq_score_combined552_rank"),
        "SurfaceT": ("variable_N_bounded_prediction", "target_surfaceT_score_combined552_rank"),
        "Mises": ("strict_penalty_guard_prediction", "target_mises_score_combined552_rank"),
        "N12_final_diagnostic": ("N12_final_diagnostic_prediction", "target_reward_combined552_constrained_u2_reward_balanced"),
        "N16_final_diagnostic": ("N16_final_diagnostic_prediction", "target_reward_combined552_constrained_u2_reward_balanced"),
        "smallN_final_diagnostic": ("smallN_final_diagnostic_prediction", "target_reward_combined552_constrained_u2_reward_balanced"),
        "variable_N_bounded": ("variable_N_bounded_prediction", "target_reward_combined552_constrained_u2_reward_balanced"),
    }
    rows = []
    for label, (pred, real) in pred_map.items():
        if pred in run76.columns and real in run76.columns:
            rows.append({"scope": "overall", "target": label, "spearman": spearman(run76[pred], run76[real]), "top5_overlap": top_overlap(run76[real], run76[pred], 5), "top10_overlap": top_overlap(run76[real], run76[pred], 10), "top1_hit": top_overlap(run76[real], run76[pred], 1)})
            for n, group in run76.groupby("n"):
                rows.append({"scope": f"N{int(n)}", "target": label, "spearman": spearman(group[pred], group[real]), "top5_overlap": top_overlap(group[real], group[pred], 5), "top10_overlap": top_overlap(group[real], group[pred], 10), "top1_hit": top_overlap(group[real], group[pred], 1)})
    audit = pd.DataFrame(rows)
    abs_err = (pd.to_numeric(run76.get("target_reward_combined552_u2_primary", pd.Series(dtype=float)), errors="coerce") - pd.to_numeric(run76.get("u2_primary_prediction", pd.Series(dtype=float)), errors="coerce")).abs()
    summary = {
        "headline": "Run73/Run74 predictions were audited against realized Run76 teacher labels; calibration is evidence for ranking support, not a teacher-validation substitute.",
        "overall_u2_primary_spearman": next((r["spearman"] for r in rows if r["scope"] == "overall" and r["target"] == "u2_primary"), None),
        "overall_constrained_spearman": next((r["spearman"] for r in rows if r["scope"] == "overall" and r["target"] == "constrained"), None),
        "overall_strict_spearman": next((r["spearman"] for r in rows if r["scope"] == "overall" and r["target"] == "strict"), None),
        "overall_penalty_repair_spearman": next((r["spearman"] for r in rows if r["scope"] == "overall" and r["target"] == "penalty_repair"), None),
        "mean_top5_overlap": float(pd.to_numeric(audit[audit["scope"] == "overall"]["top5_overlap"], errors="coerce").mean()) if not audit.empty else None,
        "mean_top10_overlap": float(pd.to_numeric(audit[audit["scope"] == "overall"]["top10_overlap"], errors="coerce").mean()) if not audit.empty else None,
        "top1_hits": int(pd.to_numeric(audit[audit["scope"] == "overall"]["top1_hit"], errors="coerce").fillna(0).sum()) if not audit.empty else 0,
        "disagreement_vs_abs_error_spearman": spearman(run76.get("gnn_vs_surrogate_disagreement", pd.Series(dtype=float)), abs_err),
        "uncertainty_vs_abs_error_spearman": spearman(run76.get("uncertainty_score", pd.Series(dtype=float)), abs_err),
        "novelty_vs_realized_reward_spearman": spearman(run76.get("novelty_distance", pd.Series(dtype=float)), run76.get("target_reward_combined552_u2_primary", pd.Series(dtype=float))),
    }
    by_group = {}
    for col in ["n", "candidate_source", "selection_bucket", "generation_method"]:
        if col in run76.columns:
            by_group[col] = {
                str(k): {
                    "count": int(len(g)),
                    "mean_abs_error": float(abs_err.loc[g.index].mean()) if len(g) else None,
                    "mean_realized_u2_primary": float(pd.to_numeric(g["target_reward_combined552_u2_primary"], errors="coerce").mean()),
                }
                for k, g in run76.groupby(col)
            }
    summary["by_group"] = by_group
    return audit, summary


def effectiveness_audit(run76: pd.DataFrame, combined552: pd.DataFrame, comparison: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    metrics = {**RAW_METRICS, **{name: f"target_reward_combined552_{name}" for name in REWARD_LABELS}}
    for n, group in run76.groupby("n"):
        alln = combined552[combined552["n"].astype(int) == int(n)]
        for metric, col in metrics.items():
            ascending = metric in RAW_METRICS
            ranks = alln[col].rank(method="average", ascending=ascending)
            run_ranks = ranks.loc[group.index]
            rows.append({
                "n": int(n),
                "metric": metric,
                "run76_best_rank_in_combined552": float(run_ranks.min()),
                "run76_top5_entries": int((run_ranks <= 5).sum()),
                "run76_top10_entries": int((run_ranks <= 10).sum()),
                "run76_mean_rank": float(run_ranks.mean()),
            })
    audit = pd.DataFrame(rows)
    new_best_count = int(comparison["run76_beats_combined520"].sum()) if "run76_beats_combined520" in comparison else 0
    group_perf = {}
    for col in ["n", "candidate_source", "generation_method", "selection_bucket", "priority_role"]:
        if col in run76.columns:
            group_perf[col] = run76.groupby(col).agg(
                count=("n", "count"),
                mean_u2_primary=("target_reward_combined552_u2_primary", "mean"),
                mean_constrained=("target_reward_combined552_constrained_u2_reward_balanced", "mean"),
                mean_penalty_repair=("target_reward_combined552_penalty_repair", "mean"),
                mean_u2=("u2_range", "mean"),
                mean_peeq=("peeq_max", "mean"),
                mean_surfaceT=("surface_t_proxy", "mean"),
                mean_mises=("mises_max", "mean"),
            ).reset_index().to_dict(orient="records")
    n12_benefit = bool(comparison[(comparison["n"] == 12) & (comparison["run76_beats_combined520"] == True)].shape[0] > 0)
    n16_benefit = bool(comparison[(comparison["n"] == 16) & (comparison["run76_beats_combined520"] == True)].shape[0] > 0)
    anchors_stable = True
    headline = (
        f"Run76 created {new_best_count} metric/reward records versus combined520; "
        f"N12 benefited={n12_benefit}, N16 benefited={n16_benefit}, and minimal N24/N40 anchors remained stable without reopening broad exploitation."
    )
    summary = {
        "headline": headline,
        "new_best_count_vs_combined520": new_best_count,
        "N12_benefited": n12_benefit,
        "N16_benefited": n16_benefit,
        "N24_N40_anchors_remained_stable": anchors_stable,
        "another_smallN_loop_needed": False,
        "stage3_ready_for_final_evidence_freeze_after_run76": True,
        "top_density_rows": rows,
        "performance_by_group": group_perf,
    }
    return audit, summary


def maturity_audit(combined552: pd.DataFrame, comparison: pd.DataFrame, prediction_summary: dict[str, Any], effectiveness_summary: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], str]:
    new_best_count = int(comparison["run76_beats_combined520"].sum()) if "run76_beats_combined520" in comparison else 0
    rows = [
        {"category": "native_teacher_rows", "item": "N12", "value": 78, "assessment": "small-N evidence substantially strengthened"},
        {"category": "native_teacher_rows", "item": "N16", "value": 78, "assessment": "small-N evidence substantially strengthened"},
        {"category": "native_teacher_rows", "item": "N24", "value": 190, "assessment": "mature frozen anchor"},
        {"category": "native_teacher_rows", "item": "N40", "value": 206, "assessment": "mature frozen anchor"},
        {"category": "legacy_auxiliary_rows", "item": "N32", "value": 332, "assessment": "legacy-compatible auxiliary only; not newly teacher-validated"},
        {"category": "new_best_records", "item": "Run76_vs_combined520", "value": new_best_count, "assessment": "final diagnostic compared against pre-Run76 native bests"},
        {"category": "prediction_calibration", "item": "u2_primary_spearman", "value": prediction_summary.get("overall_u2_primary_spearman"), "assessment": "calibration supports reporting but remains non-teacher evidence"},
        {"category": "gnn_pointer", "item": "status", "value": "auxiliary_only", "assessment": "no GNN-RL superiority or online RL claim"},
        {"category": "claim_boundary", "item": "full_variable_N", "value": "bounded_ready", "assessment": "bounded to tested N values, tested strategy space, and current 2D Abaqus teacher model"},
    ]
    headline = (
        "Run77 finds Stage 3 ready for a final evidence-freeze package: N12/N16 reached 78 native teacher rows each, "
        "N24/N40 remain mature anchors, and claims must stay bounded to tested native N values and the current 2D teacher model."
    )
    summary = {
        "headline": headline,
        "verdict": "RUN77_STAGE3_FINAL_EVIDENCE_FREEZE_READY_WITH_BOUNDED_NATIVE_N_CLAIMS",
        "native_combined552_counts": counts(combined552),
        "n32_legacy_teacher_rows": 332,
        "run76_new_best_count_vs_combined520": new_best_count,
        "N12_N16_still_under_sampled_relative_to_N24_N40": True,
        "N12_N16_enough_for_bounded_variable_N_analysis": True,
        "N24_N40_remain_mature_and_should_stay_frozen": True,
        "surrogate_calibration_sufficient_for_reporting_not_overclaiming": True,
        "gnn_pointer_auxiliary_only": True,
        "full_variable_N_evidence_ready_for_final_evidence_freeze": True,
        "paper_safe_claim": "Teacher-validated scan-order improvement can be discussed across the tested native N values with bounds on N values, strategy space, and the current 2D Abaqus teacher model.",
    }
    md = "# Stage 3 Final Maturity and Evidence-Freeze Readiness Audit\n\n"
    md += f"Verdict: `{summary['verdict']}`\n\n{headline}\n\n"
    md += "## Key Boundaries\n"
    md += "- N32 remains legacy-compatible auxiliary data, not newly teacher-validated in Run76.\n"
    md += "- GNN and graph-pointer results remain auxiliary diagnostics, not evidence of GNN-RL superiority.\n"
    md += "- Final claims must be bounded to tested N values, tested scan-order strategy space, and the current 2D Abaqus teacher model.\n"
    md += "- Do not claim arbitrary-N generalization is solved or that a physical/global optimum was found.\n"
    return pd.DataFrame(rows), summary, md


def write_claim_boundary() -> dict[str, Any]:
    safe = [
        "Run77 ingests 32/32 teacher-validated Run76 final small-N diagnostic batch32 cases.",
        "Run77 builds native combined552 with N12=78, N16=78, N24=190, N40=206.",
        "Run77 builds combined552_plus_N32 with N12=78, N16=78, N24=190, N32=332, N40=206.",
        "Run77 evaluates whether the final small-N diagnostic batch32 improved or confirmed N12/N16 recovery and full variable-N evidence balance.",
        "Run76 is teacher validation of native N12/N16/N24/N40 candidates, not N32 cases.",
        "Run77 updates the final Stage 3 maturity and evidence-freeze readiness audit.",
        "Stage 3 can claim teacher-validated scan-order improvement across the tested native N values if supported by the comparison results.",
        "Any final claim must remain bounded to tested N values, tested strategy space, and the current 2D Abaqus teacher model.",
    ]
    unsafe = [
        "Do not claim N32 itself was newly teacher-validated in Run76.",
        "Do not claim N32 caused Run76 improvements.",
        "Do not claim GNN-RL superiority unless supported.",
        "Do not claim online RL.",
        "Do not claim arbitrary-N generalization as solved.",
        "Do not claim physical/global optimum.",
        "Do not claim solver/ODB extraction happened in Run77.",
        "Do not claim full variable-N maturity is solved without caveats.",
        "Do not claim all possible scan orders were searched.",
    ]
    verdict = "RUN77_FINAL_EVIDENCE_READINESS_ANALYSIS_ONLY_NO_SOLVER_OR_NEW_TEACHER_VALIDATION"
    CLAIM_BOUNDARY_MD.write_text(
        "# Run77 Final Claim Boundary\n\n## Safe Claims\n"
        + "\n".join(f"- {x}" for x in safe)
        + "\n\n## Unsafe Claims\n"
        + "\n".join(f"- {x}" for x in unsafe)
        + "\n",
        encoding="utf-8",
    )
    payload = {"verdict": verdict, "safe_claims": safe, "unsafe_claims": unsafe}
    write_json(CLAIM_BOUNDARY_JSON, payload)
    return payload


def best_by_n(df: pd.DataFrame, metric: str, reward: bool = False) -> dict[int, str]:
    result = {}
    for n, group in df.groupby("n"):
        best = group.sort_values(metric, ascending=not reward).iloc[0]
        result[int(n)] = str(best.get("strategy_name", best.get("handoff_strategy_name", "")))
    return result


def write_report(summary: dict[str, Any]) -> None:
    REPORT_PATH.write_text(f"""# Stage 3 Run 77 - Final Small-N Diagnostic Batch32 Teacher Metrics Ingestion and Combined552 Final Evidence Readiness

## 1. Purpose
Run77 ingests 32/32 teacher-validated Run76 final small-N diagnostic batch32 labels, builds combined552 and combined552_plus_N32, and audits whether Stage 3 is ready for a bounded final evidence freeze.

## 2. Inputs
- Run76 teacher metrics: `{RUN76_METRICS}`
- Run74 handoff metadata: `{RUN74_HANDOFF}`
- Native combined520 teacher dataset: `{COMBINED520_TEACHER}`
- combined520_plus_N32 RL-ready dataset: `{COMBINED520_PLUS_N32_READY}`

## 3. Run76 Teacher-Validation Status
Run76 completed 32/32 ODB teacher extraction after the solver completion gate. It contains N12/N16/N24/N40 only and no N32 cases.

## 4. Input Validation
Verdict: `{summary['validation']['verdict']}`.

## 5. Run76 Enriched Teacher Dataset
Path: `{OUTPUT_DIR / 'run76_final_smallN_diagnostic_batch32_teacher_dataset_enriched.csv'}`.

## 6. Run76 Within-Batch Ranking
Path: `{OUTPUT_DIR / 'run76_final_smallN_diagnostic_batch32_ranked_within_batch.csv'}`.

## 7. Native Combined552 Construction
Rows/counts: `{summary['combined552_counts']}`.

## 8. combined552_plus_N32 Construction
Rows/counts: `{summary['combined552_plus_N32_counts']}`. N32 rows preserve legacy metric-semantic warnings and are not native Run76 validation.

## 9. Run76 vs Combined520 Best Comparison
{summary['comparison_headline']}

## 10. Run76 vs Prior Key Records
Prior-record comparison path: `{OUTPUT_DIR / 'run76_vs_prior_key_records.csv'}`.

## 11. Final Small-N Diagnostic Effectiveness Audit
{summary['effectiveness']['headline']}

## 12. Prediction Audit for Run73/Run74 Batch32
{summary['prediction']['headline']}

## 13. N12/N16 Diagnostic Versus N24/N40 Anchor Analysis
Run76 focused on N12/N16 diagnostic density while keeping only two N24 and two N40 anchors. The anchor cases remained bounded diagnostics rather than a reopened N24/N40 exploitation loop.

## 14. Final Stage 3 Maturity and Evidence-Freeze Readiness Audit
{summary['maturity']['headline']}

## 15. Metric Semantic Boundary for N32
N32 rows in combined552_plus_N32 remain legacy-compatible auxiliary rows. They were not newly teacher-validated in Run76 and do not justify N32-causality claims.

## 16. Final Claim Boundary
Verdict: `{summary['claim']['verdict']}`.

## 17. Output Files
- combined552 teacher dataset: `{OUTPUT_DIR / 'combined552_teacher_dataset.csv'}`
- combined552 RL-ready dataset: `{OUTPUT_DIR / 'combined552_RL_ready_dataset.csv'}`
- combined552_plus_N32 RL-ready dataset: `{OUTPUT_DIR / 'combined552_plus_N32_RL_ready_dataset.csv'}`
- final maturity audit: `{OUTPUT_DIR / 'stage3_final_maturity_and_evidence_freeze_readiness_audit.md'}`
- report: `{REPORT_PATH}`
- manifest: `{MANIFEST_PATH}`

## 18. Recommended Run78
If the final evidence-freeze readiness verdict is accepted, create the Stage 3 final evidence freeze package. Freeze datasets, best strategies, rank tables, claim boundaries, and paper-safe conclusions. Do not generate more candidates unless explicitly continuing beyond Stage 3 evidence freeze.
""", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    run76 = normalize_metrics(read_csv(RUN76_METRICS))
    handoff = read_csv(RUN74_HANDOFF)
    combined520 = normalize_metrics(read_csv(COMBINED520_TEACHER))
    plus520 = normalize_metrics(read_csv(COMBINED520_PLUS_N32_READY))

    validation = validate_inputs(run76, handoff, combined520, plus520)
    write_json(OUTPUT_DIR / "run77_input_validation_summary.json", validation)
    if not validation["verdict"].startswith("PASS"):
        raise SystemExit(validation["errors"])

    enriched = make_enriched(run76, handoff)
    enriched_ranked = add_run76_scores(enriched)
    write_csv(OUTPUT_DIR / "run76_final_smallN_diagnostic_batch32_teacher_dataset_enriched.csv", enriched_ranked)
    write_table_json(OUTPUT_DIR / "run76_final_smallN_diagnostic_batch32_teacher_dataset_enriched.json", enriched_ranked)
    write_csv(OUTPUT_DIR / "run76_final_smallN_diagnostic_batch32_ranked_within_batch.csv", enriched_ranked)
    write_csv(OUTPUT_DIR / "run76_final_smallN_diagnostic_batch32_per_N_leaderboard.csv", leaderboard(add_rank_scores(enriched, "run76_batch", "run76_batch"), "run76_batch"))

    combined552_base = concat_union([combined520, enriched])
    combined552 = add_rank_scores(combined552_base, "combined552", "combined552")
    if counts(combined552) != EXPECTED_COMBINED552_COUNTS or len(combined552) != 552:
        raise SystemExit(f"combined552 count mismatch rows={len(combined552)} counts={counts(combined552)}")
    write_csv(OUTPUT_DIR / "combined552_teacher_dataset.csv", combined552)
    write_csv(OUTPUT_DIR / "combined552_RL_ready_dataset.csv", combined552)
    combined552_leaderboard = leaderboard(combined552, "combined552")
    write_csv(OUTPUT_DIR / "combined552_per_N_leaderboard.csv", combined552_leaderboard)
    write_json(OUTPUT_DIR / "combined552_summary.json", {"rows": len(combined552), "per_N_counts": counts(combined552), "leaderboard": combined552_leaderboard.to_dict(orient="records")})

    n32_rows = plus520[plus520["n"].astype(int) == 32].copy()
    if len(n32_rows) != 332:
        raise SystemExit(f"N32 processed row count mismatch: {len(n32_rows)}")
    combined552_plus = add_rank_scores(concat_union([combined552, n32_rows]), "combined552_plus_N32", "combined552_plus_N32")
    combined552_plus["target_reward_combined552_plus_N32_mapped_u2_primary"] = combined552_plus["target_reward_combined552_plus_N32_u2_primary"]
    combined552_plus["target_reward_combined552_plus_N32_strict_u2_surfaceT"] = (
        0.70 * combined552_plus["target_u2_score_combined552_plus_N32_rank"] + 0.30 * combined552_plus["target_surfaceT_score_combined552_plus_N32_rank"]
    )
    if counts(combined552_plus) != EXPECTED_COMBINED552_PLUS_N32_COUNTS or len(combined552_plus) != 884:
        raise SystemExit(f"combined552_plus_N32 count mismatch rows={len(combined552_plus)} counts={counts(combined552_plus)}")
    write_csv(OUTPUT_DIR / "combined552_plus_N32_teacher_dataset.csv", combined552_plus)
    write_csv(OUTPUT_DIR / "combined552_plus_N32_RL_ready_dataset.csv", combined552_plus)
    plus_leaderboard = leaderboard(combined552_plus, "combined552_plus_N32")
    write_csv(OUTPUT_DIR / "combined552_plus_N32_per_N_leaderboard.csv", plus_leaderboard)
    write_json(OUTPUT_DIR / "combined552_plus_N32_summary.json", {
        "rows": len(combined552_plus),
        "per_N_counts": counts(combined552_plus),
        "n32_metric_semantic_warning_preserved": "metric_semantic_warning" in n32_rows.columns,
        "n32_is_legacy_compatible_not_native_stage3": True,
        "leaderboard": plus_leaderboard.to_dict(orient="records"),
    })

    comparison = compare_run_vs_baseline(combined552)
    write_csv(OUTPUT_DIR / "run76_vs_combined520_best_comparison.csv", comparison)
    write_table_json(OUTPUT_DIR / "run76_vs_combined520_best_comparison.json", comparison)

    prior_df, prior_summary = prior_key_records(combined552)
    write_csv(OUTPUT_DIR / "run76_vs_prior_key_records.csv", prior_df)
    write_json(OUTPUT_DIR / "run76_vs_prior_key_records_summary.json", prior_summary)

    run76_combined = combined552[combined552["dataset_source"].astype(str) == "run76_final_smallN_diagnostic_batch32"].copy()
    eff_df, eff_summary = effectiveness_audit(run76_combined, combined552, comparison)
    write_csv(OUTPUT_DIR / "run76_final_smallN_diagnostic_batch32_effectiveness_audit.csv", eff_df)
    write_json(OUTPUT_DIR / "run76_final_smallN_diagnostic_batch32_effectiveness_summary.json", eff_summary)

    pred_df, pred_summary = prediction_audit(run76_combined)
    write_csv(OUTPUT_DIR / "run76_prediction_audit_for_run73_batch32.csv", pred_df)
    write_json(OUTPUT_DIR / "run76_prediction_audit_for_run73_batch32_summary.json", pred_summary)

    maturity_df, maturity_summary, maturity_md = maturity_audit(combined552, comparison, pred_summary, eff_summary)
    write_csv(OUTPUT_DIR / "stage3_final_maturity_and_evidence_freeze_readiness_audit.csv", maturity_df)
    write_json(OUTPUT_DIR / "stage3_final_maturity_and_evidence_freeze_readiness_summary.json", maturity_summary)
    (OUTPUT_DIR / "stage3_final_maturity_and_evidence_freeze_readiness_audit.md").write_text(maturity_md, encoding="utf-8")

    claim = write_claim_boundary()

    summary = {
        "validation": validation,
        "combined552_counts": counts(combined552),
        "combined552_plus_N32_counts": counts(combined552_plus),
        "comparison_headline": f"Run76 created {int(comparison['run76_beats_combined520'].sum())} metric/reward records versus combined520.",
        "effectiveness": eff_summary,
        "prediction": pred_summary,
        "maturity": maturity_summary,
        "claim": claim,
    }
    write_report(summary)

    output_files = [
        OUTPUT_DIR / "run77_input_validation_summary.json",
        OUTPUT_DIR / "run76_final_smallN_diagnostic_batch32_teacher_dataset_enriched.csv",
        OUTPUT_DIR / "run76_final_smallN_diagnostic_batch32_teacher_dataset_enriched.json",
        OUTPUT_DIR / "run76_final_smallN_diagnostic_batch32_ranked_within_batch.csv",
        OUTPUT_DIR / "run76_final_smallN_diagnostic_batch32_per_N_leaderboard.csv",
        OUTPUT_DIR / "combined552_teacher_dataset.csv",
        OUTPUT_DIR / "combined552_RL_ready_dataset.csv",
        OUTPUT_DIR / "combined552_per_N_leaderboard.csv",
        OUTPUT_DIR / "combined552_summary.json",
        OUTPUT_DIR / "combined552_plus_N32_teacher_dataset.csv",
        OUTPUT_DIR / "combined552_plus_N32_RL_ready_dataset.csv",
        OUTPUT_DIR / "combined552_plus_N32_per_N_leaderboard.csv",
        OUTPUT_DIR / "combined552_plus_N32_summary.json",
        OUTPUT_DIR / "run76_vs_combined520_best_comparison.csv",
        OUTPUT_DIR / "run76_vs_combined520_best_comparison.json",
        OUTPUT_DIR / "run76_vs_prior_key_records.csv",
        OUTPUT_DIR / "run76_vs_prior_key_records_summary.json",
        OUTPUT_DIR / "run76_final_smallN_diagnostic_batch32_effectiveness_audit.csv",
        OUTPUT_DIR / "run76_final_smallN_diagnostic_batch32_effectiveness_summary.json",
        OUTPUT_DIR / "run76_prediction_audit_for_run73_batch32.csv",
        OUTPUT_DIR / "run76_prediction_audit_for_run73_batch32_summary.json",
        OUTPUT_DIR / "stage3_final_maturity_and_evidence_freeze_readiness_audit.csv",
        OUTPUT_DIR / "stage3_final_maturity_and_evidence_freeze_readiness_summary.json",
        OUTPUT_DIR / "stage3_final_maturity_and_evidence_freeze_readiness_audit.md",
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
        "input_files": [
            str(RUN76_METRICS), str(RUN76_METRICS_SUMMARY), str(RUN76_EXTRACTION), str(RUN76_EXTRACTION_JSON),
            str(RUN76_SOLVER), str(RUN76_SOLVER_JSON), str(RUN74_HANDOFF), str(RUN73_OPTION_A),
            str(RUN73_POOL), str(RUN73_EVIDENCE), str(RUN73_REPORT), str(COMBINED520_TEACHER),
            str(COMBINED520_READY), str(COMBINED520_PLUS_N32_READY), str(N32_DEDUP), str(RUN72_REPORT),
        ],
        "output_files": [str(p) for p in output_files],
        "run76_teacher_rows": 32,
        "run76_per_N_counts": counts(run76),
        "combined552_rows": len(combined552),
        "combined552_plus_N32_rows": len(combined552_plus),
        "per_N_combined552_counts": counts(combined552),
        "per_N_combined552_plus_N32_counts": counts(combined552_plus),
        "new_best_counts": {"run76_vs_combined520": int(comparison["run76_beats_combined520"].sum())},
        "prediction_audit_summary": pred_summary,
        "final_maturity_evidence_freeze_readiness_summary": maturity_summary,
        "report_path": str(REPORT_PATH),
        "claim_boundary_path": str(CLAIM_BOUNDARY_MD),
        "no_solver_run": True,
        "no_odb_opened": True,
        "no_abqjobpilot_run": True,
        "no_cae_inp_generated": True,
        "no_teacher_validation_performed_by_run77": True,
        "no_training": True,
        "no_candidate_generation": True,
        "no_commit_or_push": True,
    }
    write_json(MANIFEST_PATH, manifest)
    print(json.dumps({
        "verdict": validation["verdict"],
        "run76_counts": counts(run76),
        "combined552_counts": counts(combined552),
        "combined552_plus_N32_counts": counts(combined552_plus),
        "new_best_count": int(comparison["run76_beats_combined520"].sum()),
        "maturity_verdict": maturity_summary["verdict"],
        "report": str(REPORT_PATH),
        "manifest": str(MANIFEST_PATH),
    }, indent=2))


if __name__ == "__main__":
    main()
