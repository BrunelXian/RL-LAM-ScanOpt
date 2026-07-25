from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
RUN_ID = "run_78_stage3_final_evidence_freeze_package"
RUN_NAME = "run_78_stage3_final_evidence_freeze_package"
SCRIPT_PATH = ROOT / "scripts" / "stage3" / "run_78_create_stage3_final_evidence_freeze_package.py"

RUN77_DIR = ROOT / "outputs" / "stage3_run_77_final_smallN_diagnostic_batch32_teacher_metrics_ingestion_and_combined552_final_evidence_readiness"
RUN77_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_77_final_smallN_diagnostic_batch32_teacher_metrics_ingestion_and_combined552_final_evidence_readiness" / "RUN_77_FINAL_SMALLN_DIAGNOSTIC_BATCH32_TEACHER_METRICS_INGESTION_AND_COMBINED552_FINAL_EVIDENCE_READINESS_REPORT.md"
RUN77_MANIFEST = ROOT / "artifacts" / "manifests" / "stage3_run_77_manifest.json"

NATIVE_TEACHER = RUN77_DIR / "combined552_teacher_dataset.csv"
NATIVE_READY = RUN77_DIR / "combined552_RL_ready_dataset.csv"
PLUS_N32_TEACHER = RUN77_DIR / "combined552_plus_N32_teacher_dataset.csv"
PLUS_N32_READY = RUN77_DIR / "combined552_plus_N32_RL_ready_dataset.csv"
NATIVE_LEADERBOARD = RUN77_DIR / "combined552_per_N_leaderboard.csv"
NATIVE_SUMMARY = RUN77_DIR / "combined552_summary.json"
RUN77_CLAIM_MD = RUN77_DIR / "run77_final_claim_boundary.md"
RUN77_CLAIM_JSON = RUN77_DIR / "run77_final_claim_boundary.json"
RUN77_MATURITY_MD = RUN77_DIR / "stage3_final_maturity_and_evidence_freeze_readiness_audit.md"
RUN77_MATURITY_JSON = RUN77_DIR / "stage3_final_maturity_and_evidence_freeze_readiness_summary.json"
RUN76_VS_COMBINED520 = RUN77_DIR / "run76_vs_combined520_best_comparison.csv"
RUN76_VS_PRIOR = RUN77_DIR / "run76_vs_prior_key_records.csv"
RUN76_EFFECTIVENESS = RUN77_DIR / "run76_final_smallN_diagnostic_batch32_effectiveness_audit.csv"
RUN76_PREDICTION = RUN77_DIR / "run76_prediction_audit_for_run73_batch32.csv"

EARLIER_REPORTS = {
    67: ROOT / "docs" / "stage3" / "runs" / "run_67_variable_N_recovery_anchor_batch48_teacher_metrics_ingestion_and_combined480_ranking" / "RUN_67_VARIABLE_N_RECOVERY_ANCHOR_BATCH48_TEACHER_METRICS_INGESTION_AND_COMBINED480_RANKING_REPORT.md",
    68: ROOT / "docs" / "stage3" / "runs" / "run_68_combined480_model_update_smallN_recovery_candidate_generation" / "RUN_68_COMBINED480_MODEL_UPDATE_SMALLN_RECOVERY_CANDIDATE_GENERATION_REPORT.md",
    72: ROOT / "docs" / "stage3" / "runs" / "run_72_smallN_recovery_focused_batch40_teacher_metrics_ingestion_and_combined520_ranking" / "RUN_72_SMALLN_RECOVERY_FOCUSED_BATCH40_TEACHER_METRICS_INGESTION_AND_COMBINED520_RANKING_REPORT.md",
    73: ROOT / "docs" / "stage3" / "runs" / "run_73_combined520_model_update_final_smallN_diagnostic_candidate_generation" / "RUN_73_COMBINED520_MODEL_UPDATE_FINAL_SMALLN_DIAGNOSTIC_CANDIDATE_GENERATION_REPORT.md",
}

OUTPUT_DIR = ROOT / "outputs" / "stage3_run_78_final_evidence_freeze_package"
REPORT_DIR = ROOT / "docs" / "stage3" / "runs" / "run_78_stage3_final_evidence_freeze_package"
REPORT_PATH = REPORT_DIR / "RUN_78_STAGE3_FINAL_EVIDENCE_FREEZE_PACKAGE_REPORT.md"
MANIFEST_PATH = ROOT / "artifacts" / "manifests" / "stage3_run_78_manifest.json"

EXPECTED_NATIVE_COUNTS = {12: 78, 16: 78, 24: 190, 40: 206}
EXPECTED_PLUS_COUNTS = {12: 78, 16: 78, 24: 190, 32: 332, 40: 206}

RAW_OBJECTIVES = {
    "U2": ("u2_range", "min"),
    "PEEQ": ("peeq_max", "min"),
    "SurfaceT": ("surface_t_proxy", "min"),
    "Mises": ("mises_max", "min"),
}
REWARD_OBJECTIVES = {
    "u2_primary": ("target_reward_combined552_u2_primary", "max"),
    "constrained_reward": ("target_reward_combined552_constrained_u2_reward_balanced", "max"),
    "strict_penalty_guard": ("target_reward_combined552_strict_penalty_guard", "max"),
    "penalty_repair": ("target_reward_combined552_penalty_repair", "max"),
}
ALL_OBJECTIVES = {**RAW_OBJECTIVES, **REWARD_OBJECTIVES}


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
    path.write_text(json.dumps(clean_json(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_records_json(path: Path, df: pd.DataFrame) -> None:
    write_json(path, {"schema": "records", "rows": df.to_dict(orient="records")})


def df_to_markdown(df: pd.DataFrame, max_cell_chars: int = 160) -> str:
    if df.empty:
        return "No rows."
    work = df.copy()
    for col in work.columns:
        work[col] = work[col].map(lambda v: "" if pd.isna(v) else str(v))
        work[col] = work[col].map(lambda v: v[: max_cell_chars - 3] + "..." if len(v) > max_cell_chars else v)
    headers = [str(c) for c in work.columns]
    rows = work.values.tolist()

    def esc(text: Any) -> str:
        return str(text).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(esc(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(esc(v) for v in row) + " |")
    return "\n".join(lines)


def write_md_table(path: Path, df: pd.DataFrame, title: str, intro: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    if intro:
        lines += [intro, ""]
    lines.append(df_to_markdown(df))
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def per_n_counts(df: pd.DataFrame) -> dict[int, int]:
    return {int(k): int(v) for k, v in df["n"].value_counts().sort_index().items()}


def dataset_source_to_run(source: Any) -> str:
    s = str(source or "")
    mapping = [
        ("probe60", "Run08"),
        ("batch20", "Run14"),
        ("batch28", "Run20"),
        ("shortlist64", "Run27"),
        ("n32_legacy", "Run32A"),
        ("run36", "Run36"),
        ("run41", "Run41"),
        ("run46", "Run46"),
        ("run51", "Run51"),
        ("run56", "Run56"),
        ("run61", "Run61"),
        ("run66", "Run66"),
        ("run71", "Run71"),
        ("run76", "Run76"),
    ]
    lowered = s.lower()
    for key, run in mapping:
        if key in lowered:
            return run
    return s or "UNKNOWN"


def record_category(source: Any) -> str:
    s = str(source or "").lower()
    if "probe60" in s or "batch20" in s or "batch28" in s or "shortlist64" in s:
        return "baseline / early strategy"
    if "run36" in s:
        return "N32-informed native batch"
    if "run41" in s:
        return "focused N24/N40 batch"
    if "run46" in s or "run51" in s:
        return "constrained batch"
    if "run56" in s:
        return "calibrated batch"
    if "run61" in s:
        return "N40-focused batch"
    if "run66" in s:
        return "variable-N recovery anchor"
    if "run71" in s:
        return "small-N recovery"
    if "run76" in s:
        return "final small-N diagnostic"
    return "historical native teacher record"


def supports_evidence(n: int, source: Any) -> dict[str, bool]:
    category = record_category(source)
    return {
        "supports_mature_N24_N40_evidence": n in (24, 40),
        "supports_N12_N16_recovery_evidence": n in (12, 16),
        "supports_full_bounded_variable_N_evidence": n in (12, 16, 24, 40),
        "record_category": category,
    }


def existing_or_none(path: Path) -> str | None:
    return str(path) if path.exists() else None


def validate_inputs(native: pd.DataFrame, plus: pd.DataFrame) -> dict[str, Any]:
    errors: list[str] = []
    required_files = [
        NATIVE_TEACHER,
        NATIVE_READY,
        PLUS_N32_READY,
        RUN77_MATURITY_MD,
        RUN77_CLAIM_MD,
        RUN77_REPORT,
        RUN77_MANIFEST,
    ]
    for path in required_files:
        if not path.exists():
            errors.append(f"Missing required input: {path}")

    native_counts = per_n_counts(native)
    plus_counts = per_n_counts(plus)
    if len(native) != 552:
        errors.append(f"Native combined552 row count expected 552, found {len(native)}")
    if native_counts != EXPECTED_NATIVE_COUNTS:
        errors.append(f"Native per-N counts expected {EXPECTED_NATIVE_COUNTS}, found {native_counts}")
    if 32 in native_counts:
        errors.append("Native combined552 unexpectedly contains N32 rows")
    for col, _mode in RAW_OBJECTIVES.values():
        if col not in native.columns:
            errors.append(f"Missing native metric column: {col}")
        elif native[col].isna().any():
            errors.append(f"Native metric column has missing values: {col}")
    for col, _mode in REWARD_OBJECTIVES.values():
        if col not in native.columns:
            errors.append(f"Missing native reward column: {col}")
    if "order_json" not in native.columns and "order_compact" not in native.columns:
        errors.append("Native combined552 lacks order_json/order_compact traceability")
    if not any("teacher_validation_status" in c for c in native.columns):
        errors.append("Native combined552 lacks teacher-validation status traceability")

    if len(plus) != 884:
        errors.append(f"combined552_plus_N32 row count expected 884, found {len(plus)}")
    if plus_counts != EXPECTED_PLUS_COUNTS:
        errors.append(f"plus-N32 per-N counts expected {EXPECTED_PLUS_COUNTS}, found {plus_counts}")
    n32 = plus[plus["n"].astype(int) == 32]
    warning_cols = [c for c in n32.columns if "warning" in c.lower() or "semantic" in c.lower() or "legacy" in c.lower()]
    if len(n32) != 332:
        errors.append(f"plus-N32 N32 count expected 332, found {len(n32)}")
    if not warning_cols:
        errors.append("N32 legacy semantic warning columns are not present")

    verdict = "PASS_RUN78_STAGE3_FINAL_EVIDENCE_INPUTS_READY" if not errors else "FAIL_RUN78_STAGE3_FINAL_EVIDENCE_INPUTS_NOT_READY"
    summary = {
        "timestamp": now_iso(),
        "verdict": verdict,
        "errors": errors,
        "native_combined552_rows": int(len(native)),
        "native_combined552_per_N_counts": native_counts,
        "combined552_plus_N32_rows": int(len(plus)),
        "combined552_plus_N32_per_N_counts": plus_counts,
        "n32_legacy_warning_columns": warning_cols,
    }
    write_json(OUTPUT_DIR / "run78_input_validation_summary.json", summary)
    if errors:
        raise SystemExit(json.dumps(summary, indent=2))
    return summary


def freeze_inputs() -> tuple[list[dict[str, Any]], dict[str, str]]:
    copies = {
        NATIVE_TEACHER: "FROZEN_stage3_native_combined552_teacher_dataset.csv",
        NATIVE_READY: "FROZEN_stage3_native_combined552_RL_ready_dataset.csv",
        PLUS_N32_TEACHER: "FROZEN_stage3_combined552_plus_N32_teacher_dataset.csv",
        PLUS_N32_READY: "FROZEN_stage3_combined552_plus_N32_RL_ready_dataset.csv",
        NATIVE_LEADERBOARD: "FROZEN_stage3_native_combined552_per_N_leaderboard.csv",
        NATIVE_SUMMARY: "FROZEN_stage3_native_combined552_summary.json",
        RUN77_CLAIM_MD: "FROZEN_run77_final_claim_boundary.md",
        RUN77_CLAIM_JSON: "FROZEN_run77_final_claim_boundary.json",
        RUN77_MATURITY_MD: "FROZEN_stage3_final_maturity_and_evidence_freeze_readiness_audit.md",
        RUN77_MATURITY_JSON: "FROZEN_stage3_final_maturity_and_evidence_freeze_readiness_summary.json",
    }
    frozen_paths: dict[str, str] = {}
    hash_rows: list[dict[str, Any]] = []
    for src, name in copies.items():
        dst = OUTPUT_DIR / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        frozen_paths[name] = str(dst)
        hash_rows.append({
            "frozen_file": name,
            "source_path": str(src),
            "frozen_path": str(dst),
            "bytes": dst.stat().st_size,
            "sha256": sha256_file(dst),
        })
    hash_df = pd.DataFrame(hash_rows)
    hash_df.to_csv(OUTPUT_DIR / "FROZEN_stage3_file_hashes.csv", index=False)
    write_records_json(OUTPUT_DIR / "FROZEN_stage3_file_hashes.json", hash_df)
    return hash_rows, frozen_paths


def pick_best(df: pd.DataFrame, n: int, objective: str) -> pd.Series:
    col, mode = ALL_OBJECTIVES[objective]
    subset = df[df["n"].astype(int) == n].copy()
    values = pd.to_numeric(subset[col], errors="coerce")
    idx = values.idxmin() if mode == "min" else values.idxmax()
    return subset.loc[idx]


def row_identity(row: pd.Series) -> dict[str, Any]:
    fields = [
        "strategy_name",
        "handoff_strategy_name",
        "dataset_source",
        "order_hash",
        "order_compact",
        "order_json",
        "teacher_validation_status",
        "completion_status",
        "odb_extraction_status",
        "solver_audit_status",
        "job_name",
    ]
    out = {field: row.get(field) for field in fields if field in row.index}
    out["source_run"] = dataset_source_to_run(row.get("dataset_source"))
    return out


def build_best_table(native: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for n in sorted(native["n"].astype(int).unique()):
        record: dict[str, Any] = {"n": int(n)}
        for objective, (col, _mode) in ALL_OBJECTIVES.items():
            best = pick_best(native, int(n), objective)
            prefix = objective
            record[f"best_{prefix}_strategy"] = best.get("strategy_name") or best.get("handoff_strategy_name")
            record[f"best_{prefix}_value"] = best.get(col)
            record[f"best_{prefix}_dataset_source"] = best.get("dataset_source")
            record[f"best_{prefix}_source_run"] = dataset_source_to_run(best.get("dataset_source"))
            record[f"best_{prefix}_teacher_validation_status"] = best.get("teacher_validation_status")
            record[f"best_{prefix}_order_hash"] = best.get("order_hash")
            record[f"best_{prefix}_order_compact"] = best.get("order_compact")
            if objective in ("U2", "u2_primary", "penalty_repair"):
                record[f"best_{prefix}_order_json"] = best.get("order_json")
        rows.append(record)
    best_df = pd.DataFrame(rows)
    best_df.to_csv(OUTPUT_DIR / "stage3_final_native_best_strategy_table.csv", index=False)
    write_records_json(OUTPUT_DIR / "stage3_final_native_best_strategy_table.json", best_df)
    write_md_table(
        OUTPUT_DIR / "stage3_final_native_best_strategy_table.md",
        best_df,
        "Stage 3 Final Native Best Strategy Table",
        "Final best strategies are computed from native combined552 only: N12, N16, N24, and N40.",
    )
    return best_df


def build_topk(native: pd.DataFrame, k: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for n in sorted(native["n"].astype(int).unique()):
        subset = native[native["n"].astype(int) == n].copy()
        for objective, (col, mode) in ALL_OBJECTIVES.items():
            work = subset.copy()
            work["_value"] = pd.to_numeric(work[col], errors="coerce")
            work = work.sort_values("_value", ascending=(mode == "min")).head(k)
            for rank, (_, row) in enumerate(work.iterrows(), start=1):
                out = {
                    "n": int(n),
                    "objective": objective,
                    "rank": rank,
                    "value": row["_value"],
                    "lower_is_better": mode == "min",
                }
                out.update(row_identity(row))
                rows.append(out)
    return pd.DataFrame(rows)


def write_topk(native: pd.DataFrame) -> dict[str, str]:
    top5 = build_topk(native, 5)
    top10 = build_topk(native, 10)
    top5_path = OUTPUT_DIR / "stage3_final_native_top5_by_N_and_objective.csv"
    top10_path = OUTPUT_DIR / "stage3_final_native_top10_by_N_and_objective.csv"
    top5.to_csv(top5_path, index=False)
    top10.to_csv(top10_path, index=False)
    summary = {
        "top5_rows": int(len(top5)),
        "top10_rows": int(len(top10)),
        "objectives": list(ALL_OBJECTIVES),
        "n_values": sorted(int(v) for v in native["n"].astype(int).unique()),
        "top5_path": str(top5_path),
        "top10_path": str(top10_path),
    }
    write_json(OUTPUT_DIR / "stage3_final_native_topk_summary.json", summary)
    return {"top5": str(top5_path), "top10": str(top10_path), "summary": str(OUTPUT_DIR / "stage3_final_native_topk_summary.json")}


def build_run_ledger() -> pd.DataFrame:
    rows = [
        (27, "shortlist64 teacher ranking baseline", "native early teacher labels", "combined172-era records", 64, {12: 8, 16: 8, 24: 24, 40: 24}, False, None, True, True, "native shortlist benchmark", "early heuristic/reference coverage"),
        ("32A", "Stage 2 N32 legacy teacher label ingestion", "Stage 2 N32 labels", "n32 legacy dedup training 332", 332, {32: 332}, True, "legacy", False, False, "legacy-compatible auxiliary data", "N32 semantic separation source"),
        (36, "N32-informed native batch32 ingestion", "native candidate batch32", "native combined204 contribution", 32, {12: 4, 16: 4, 24: 12, 40: 12}, False, None, True, True, "native N32-informed validation", "introduced strong N24/N40 native records"),
        (41, "native N24/N40 focused batch60 teacher extraction", "Run39/Run40 focused candidates", "Run41 teacher metrics", 60, {24: 30, 40: 30}, False, None, True, True, "focused N24/N40 validation", "confirmed U2-near saturation and N40 PEEQ improvement"),
        (46, "constrained N24/N40 batch32 teacher validation", "Run44 handoff", "combined296 contribution", 32, {24: 16, 40: 16}, False, None, True, True, "constrained batch32 validation", "N24 U2/reward and N40 reward gains"),
        (51, "stricter constrained N24/N40 batch32 validation", "Run49 handoff", "combined328 contribution", 32, {24: 16, 40: 16}, False, None, True, True, "stricter guard validation", "N24 U2 and N40 strict/reward stability"),
        (56, "calibrated N24/N40 batch64 validation", "Run54 handoff", "combined392 contribution", 64, {24: 32, 40: 32}, False, None, True, True, "overnight calibrated validation", "matured N24/N40 evidence"),
        (57, "calibrated N24/N40 batch64 ingestion", "Run56 teacher metrics", "combined392", 64, {24: 32, 40: 32}, False, None, False, False, "ingestion/ranking", "created combined392"),
        (58, "combined392 model update and N40-focused generation", "combined392", "Run58 candidate options", 0, {}, False, None, False, False, "model/candidate generation only", "evidence-freeze logic and N40 follow-up options"),
        (59, "Run58-derived N40-focused batch40 handoff", "Run58 Option A + pool", "Run59 handoff", 0, {24: 16, 40: 24}, False, None, False, False, "handoff only", "custom N40-focused package"),
        (61, "custom N40-focused batch40 validation", "Run59/Run60 cases", "Run61 teacher metrics", 40, {24: 16, 40: 24}, False, None, True, True, "N40-focused validation", "new N40 U2/reward-family records"),
        (62, "custom N40-focused batch40 ingestion", "Run61 teacher metrics", "combined432", 40, {24: 16, 40: 24}, False, None, False, False, "ingestion/ranking", "built combined432"),
        (63, "combined432 final N24/N40 evidence freeze and N12/N16 recovery generation", "combined432", "Run63 candidate options", 0, {}, False, None, False, False, "model/candidate generation only", "made N12/N16 recovery primary"),
        (64, "Run63 variable-N recovery anchor batch48 handoff", "Run63 Option A", "Run64 handoff", 0, {12: 12, 16: 12, 24: 8, 40: 16}, False, None, False, False, "handoff only", "variable-N recovery anchor package"),
        (66, "variable-N recovery anchor batch48 validation", "Run64/Run65 cases", "Run66 teacher metrics", 48, {12: 12, 16: 12, 24: 8, 40: 16}, False, None, True, True, "teacher validation", "strengthened N12/N16 and anchor evidence"),
        (67, "variable-N recovery anchor batch48 ingestion", "Run66 teacher metrics", "combined480", 48, {12: 12, 16: 12, 24: 8, 40: 16}, False, None, False, False, "ingestion/ranking", "built combined480"),
        (68, "combined480 small-N recovery model update", "combined480", "Run68 candidate options", 0, {}, False, None, False, False, "model/candidate generation only", "selected small-N recovery as primary"),
        (69, "Run68 small-N recovery batch40 handoff", "Run68 Option A", "Run69 handoff", 0, {12: 16, 16: 16, 24: 4, 40: 4}, False, None, False, False, "handoff only", "small-N recovery-focused package"),
        (71, "small-N recovery-focused batch40 validation", "Run69/Run70 cases", "Run71 teacher metrics", 40, {12: 16, 16: 16, 24: 4, 40: 4}, False, None, True, True, "teacher validation", "raised N12/N16 to 64 each"),
        (72, "small-N recovery-focused batch40 ingestion", "Run71 teacher metrics", "combined520", 40, {12: 16, 16: 16, 24: 4, 40: 4}, False, None, False, False, "ingestion/ranking", "built combined520"),
        (73, "combined520 final small-N diagnostic generation", "combined520", "Run73 Option A", 0, {}, False, None, False, False, "model/candidate generation only", "final diagnostic selection"),
        (74, "Run73 final small-N diagnostic handoff", "Run73 Option A", "Run74 handoff", 0, {12: 14, 16: 14, 24: 2, 40: 2}, False, None, False, False, "handoff only", "final diagnostic package"),
        (76, "final small-N diagnostic batch32 validation", "Run74/Run75 cases", "Run76 teacher metrics", 32, {12: 14, 16: 14, 24: 2, 40: 2}, False, None, True, True, "teacher validation", "final small-N diagnostic labels"),
        (77, "Run76 ingestion and combined552 final readiness", "Run76 metrics + combined520", "combined552 and final readiness audit", 32, {12: 14, 16: 14, 24: 2, 40: 2}, False, None, False, False, "RUN77_STAGE3_FINAL_EVIDENCE_FREEZE_READY_WITH_BOUNDED_NATIVE_N_CLAIMS", "combined552 plus maturity audit"),
        (78, "Stage 3 final evidence freeze package", "Run77 combined552", "frozen evidence package", 0, {}, False, None, False, False, "PASS_STAGE3_FINAL_EVIDENCE_FREEZE_READY_WITH_BOUNDED_NATIVE_N_CLAIMS", "freezes data, claims, and reporting evidence"),
    ]
    records = []
    for run_id, purpose, input_dataset, output_dataset, teacher_rows, counts, n32, n32_status, abaqus, teacher_validation, verdict, contribution in rows:
        run_num = str(run_id)
        records.append({
            "run_id": f"Run{run_num}",
            "run_purpose": purpose,
            "input_dataset": input_dataset,
            "output_dataset": output_dataset,
            "teacher_validated_rows_added": teacher_rows,
            "per_N_counts": json.dumps(counts, sort_keys=True),
            "N32_involved": bool(n32),
            "N32_status": n32_status or "not involved",
            "Abaqus_or_ODB_used_in_run": bool(abaqus),
            "teacher_validation_happened": bool(teacher_validation),
            "key_verdict": verdict,
            "key_contribution": contribution,
            "claim_boundary": "bounded native tested-N claims; no arbitrary-N, no global optimum, no online RL",
            "report_path": str(EARLIER_REPORTS.get(int(run_id), "")) if str(run_id).isdigit() and int(run_id) in EARLIER_REPORTS else "",
            "manifest_path": str(ROOT / "artifacts" / "manifests" / f"stage3_run_{run_num}_manifest.json") if str(run_id).isdigit() else str(ROOT / "artifacts" / "manifests" / "stage3_run_32a_manifest.json"),
        })
    ledger = pd.DataFrame(records)
    ledger.to_csv(OUTPUT_DIR / "stage3_final_run_by_run_evidence_ledger.csv", index=False)
    write_records_json(OUTPUT_DIR / "stage3_final_run_by_run_evidence_ledger.json", ledger)
    write_md_table(OUTPUT_DIR / "stage3_final_run_by_run_evidence_ledger.md", ledger, "Stage 3 Final Run-by-Run Evidence Ledger")
    return ledger


def build_record_timeline(native: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for n in sorted(native["n"].astype(int).unique()):
        for objective, (col, _mode) in ALL_OBJECTIVES.items():
            best = pick_best(native, int(n), objective)
            support = supports_evidence(int(n), best.get("dataset_source"))
            rows.append({
                "n": int(n),
                "objective": objective,
                "final_best_strategy": best.get("strategy_name") or best.get("handoff_strategy_name"),
                "value": best.get(col),
                "source_run": dataset_source_to_run(best.get("dataset_source")),
                "dataset_source": best.get("dataset_source"),
                **support,
            })
    timeline = pd.DataFrame(rows)
    timeline.to_csv(OUTPUT_DIR / "stage3_final_metric_reward_record_timeline.csv", index=False)
    write_records_json(OUTPUT_DIR / "stage3_final_metric_reward_record_timeline.json", timeline)
    write_md_table(
        OUTPUT_DIR / "stage3_final_metric_reward_record_timeline.md",
        timeline,
        "Stage 3 Final Metric/Reward Record Timeline",
        "This table identifies the source run/category for every final native combined552 best record.",
    )
    return timeline


def build_claim_map() -> pd.DataFrame:
    claims = [
        ("C1", "Stage 3 produced teacher-validated scan-order improvements across tested native N values N12, N16, N24, and N40.", "PASS", "combined552, best strategy table, record timeline", "Runs 27/36/41/46/51/56/61/66/71/76/77", "Only tested N values and current 2D Abaqus teacher model.", "Teacher-validated improvements are observed across tested native N values.", "Stage 3 solved all N or all scan strategies."),
        ("C2", "N24 and N40 evidence is mature and should be evidence-frozen.", "PASS", "combined552 counts and N24/N40 record timeline", "Runs 36/41/46/51/56/61/62/63/77", "Maturity is empirical within tested native teacher-labelled strategy space.", "N24/N40 are mature enough for evidence freeze.", "N24/N40 are globally optimized."),
        ("C3", "N12 and N16 were initially weaker / under-sampled but were substantially strengthened by recovery loops.", "PASS", "combined480, combined520, combined552 counts and Run76 comparison", "Runs 64/66/67/68/69/71/72/73/74/76/77", "N12/N16 still have fewer rows than N24/N40 but are sufficient for bounded reporting.", "N12/N16 evidence was substantially strengthened by recovery and final diagnostic loops.", "N12/N16 are fully mature for arbitrary generalization."),
        ("C4", "The final native dataset supports bounded variable-N scan-order optimization over tested N values.", "PASS", "combined552, maturity audit, claim boundary", "Runs 67/72/77/78", "Bounded to N12/N16/N24/N40, tested strategy space, current teacher model.", "combined552 supports bounded variable-N optimization over tested native N values.", "Arbitrary-N scan-order optimization is solved."),
        ("C5", "The candidate-generation loop beat initial hand-designed / heuristic baselines within the tested teacher-labelled strategy space.", "PASS", "record timeline and prior comparison tables", "Runs 27 through 77", "Baseline definition is limited to recorded heuristic/early teacher-labelled strategy pool.", "The active-learning/offline policy loop outperformed initial heuristic baselines within the tested pool.", "It beat all possible scan strategies."),
        ("C6", "GNN and graph-pointer diagnostics were explored but remain auxiliary and are not the primary evidence for final claims.", "PASS", "Run43/48/53/58/63/68/73/77 diagnostics", "Model-update runs", "Diagnostics were not dominant enough to support superiority claims.", "GNN/pointer diagnostics are auxiliary.", "GNN-RL outperformed all baselines."),
        ("C7", "N32 legacy data is useful as auxiliary context but is not native Stage 3 teacher validation and must be reported with metric-semantic caveats.", "PASS", "combined552_plus_N32 and N32 separation memo", "Run32A plus later plus-N32 diagnostics", "PEEQ/Mises semantics are proxy-mapped for N32.", "N32 is auxiliary legacy-compatible context, separated from native claims.", "N32 was newly validated or caused native improvements."),
        ("C8", "The work supports offline policy optimization / active-learning / surrogate-guided scan-order generation, not online RL or arbitrary-N generalization.", "PASS", "claim boundary and run ledger", "All Stage 3 model/candidate-generation runs", "No online Abaqus control loop was executed.", "The work supports offline active-learning/surrogate-guided policy optimization within bounds.", "Online RL solved Abaqus control."),
    ]
    df = pd.DataFrame(claims, columns=[
        "claim_id",
        "claim",
        "status",
        "supporting_datasets",
        "supporting_runs",
        "caveats",
        "recommended_wording",
        "unsafe_wording_to_avoid",
    ])
    df.to_csv(OUTPUT_DIR / "stage3_final_claim_evidence_map.csv", index=False)
    write_records_json(OUTPUT_DIR / "stage3_final_claim_evidence_map.json", df)
    write_md_table(OUTPUT_DIR / "stage3_final_claim_evidence_map.md", df, "Stage 3 Final Claim-Evidence Map")
    return df


def write_claim_boundary_and_conclusions() -> dict[str, str]:
    safe_claims = [
        "Stage 3 results are teacher-validated over tested native N values N12, N16, N24, and N40.",
        "The final native combined552 dataset supports bounded variable-N scan-order optimization within the tested strategy space and current 2D Abaqus teacher model.",
        "N24/N40 evidence is mature enough for evidence freeze; N12/N16 evidence was substantially strengthened by recovery and diagnostic loops.",
        "N32 data is legacy-compatible auxiliary context with metric-semantic caveats and is not part of native combined552.",
        "GNN and graph-pointer diagnostics are auxiliary, not the primary evidence for claims.",
    ]
    unsafe_claims = [
        "beat all known scan strategies",
        "global optimum",
        "arbitrary N solved",
        "online RL solved Abaqus control",
        "N32 was newly validated",
        "GNN outperformed all baselines",
        "physical experimental optimum",
    ]
    boundary = {
        "verdict": "PASS_STAGE3_FINAL_CLAIM_BOUNDARY_FROZEN_WITH_BOUNDED_NATIVE_N_CLAIMS",
        "safe_claims": safe_claims,
        "unsafe_claims": unsafe_claims,
        "required_phrases": [
            "teacher-validated",
            "tested native N values",
            "N12, N16, N24, N40",
            "current 2D Abaqus teacher model",
            "tested strategy space",
            "bounded variable-N",
            "N32 legacy semantic separation",
        ],
    }
    md = [
        "# Stage 3 Final Claim Boundary",
        "",
        f"Verdict: `{boundary['verdict']}`",
        "",
        "## Safe Claims",
        *[f"- {claim}" for claim in safe_claims],
        "",
        "## Unsafe Claims",
        *[f"- {claim}" for claim in unsafe_claims],
        "",
    ]
    (OUTPUT_DIR / "stage3_final_claim_boundary.md").write_text("\n".join(md), encoding="utf-8")
    write_json(OUTPUT_DIR / "stage3_final_claim_boundary.json", boundary)

    english = (
        "The Stage 3 results demonstrate that an offline policy-optimization and active-learning loop built on "
        "teacher-labelled Abaqus simulations can generate teacher-validated scan orders that outperform the initial "
        "hand-designed / heuristic baselines across the tested native N values N12, N16, N24, and N40. Evidence is "
        "mature for N24/N40 and substantially strengthened for N12/N16 through recovery and final diagnostic loops. "
        "The final native combined552 dataset supports bounded variable-N scan-order optimization within the tested "
        "strategy space and the current 2D Abaqus teacher model, without claiming global optimality, online RL "
        "control, or arbitrary-N generalization."
    )
    chinese = (
        "Stage 3 结果表明，基于 teacher-labelled Abaqus 仿真数据的离线策略优化与主动学习闭环，"
        "已经能够在测试过的原生 N 值 N12、N16、N24 和 N40 上生成超过初始人工/启发式基线的 "
        "teacher-validated scan orders。N24/N40 已形成成熟证据，N12/N16 也通过 recovery 和 final "
        "diagnostic loops 得到显著补强。最终 native combined552 数据集支持 bounded variable-N "
        "scan-order optimization，但该结论限定于当前 2D Abaqus teacher 模型、已测试策略空间和已测试 N 值，"
        "不等同于全局最优、在线 RL 控制或任意 N 泛化已经解决。"
    )
    conclusions = "\n".join([
        "# Stage 3 Final Paper-Safe Conclusions",
        "",
        "## English",
        english,
        "",
        "## Chinese",
        chinese,
        "",
        "## Unsafe Wording",
        *[f"- {item}" for item in unsafe_claims],
        "",
    ])
    (OUTPUT_DIR / "stage3_final_paper_safe_conclusions.md").write_text(conclusions, encoding="utf-8")
    return {
        "claim_boundary_md": str(OUTPUT_DIR / "stage3_final_claim_boundary.md"),
        "claim_boundary_json": str(OUTPUT_DIR / "stage3_final_claim_boundary.json"),
        "paper_safe_conclusions_md": str(OUTPUT_DIR / "stage3_final_paper_safe_conclusions.md"),
    }


def write_n32_memo() -> dict[str, str]:
    memo = {
        "verdict": "N32_LEGACY_AUXILIARY_ONLY_NOT_NATIVE_STAGE3_TEACHER_VALIDATION",
        "n32_rows": 332,
        "source": "Stage 2 legacy teacher-labelled data",
        "not_newly_validated_in_stage3_runs": ["Run66", "Run71", "Run76"],
        "not_part_of_native_combined552": True,
        "auxiliary_dataset": "combined552_plus_N32",
        "semantic_caveats": [
            "N32 peeq_max mapped from legacy peeq_guard/proxy semantics where applicable.",
            "N32 mises_max mapped from legacy mises_P95_top_band/proxy semantics where applicable.",
            "N32 is legacy-compatible, not native Stage 3 teacher validation.",
        ],
        "native_claim_basis": ["N12", "N16", "N24", "N40"],
        "not_for_arbitrary_N_generalization_claims": True,
    }
    md = [
        "# Stage 3 Final N32 Legacy-Semantic Separation Memo",
        "",
        f"Verdict: `{memo['verdict']}`",
        "",
        "- N32 rows: 332",
        "- Source: Stage 2 legacy teacher-labelled data",
        "- N32 was not newly validated in Stage 3 Run66, Run71, or Run76.",
        "- N32 is not part of native combined552.",
        "- N32 can be used only in combined552_plus_N32 auxiliary diagnostics.",
        "- Native claims must be based on N12/N16/N24/N40 only.",
        "- N32 should not be used to claim arbitrary-N generalization.",
        "",
        "## Semantic Caveats",
        *[f"- {item}" for item in memo["semantic_caveats"]],
        "",
    ]
    (OUTPUT_DIR / "stage3_final_N32_legacy_semantic_separation_memo.md").write_text("\n".join(md), encoding="utf-8")
    write_json(OUTPUT_DIR / "stage3_final_N32_legacy_semantic_separation_memo.json", memo)
    return {
        "n32_memo_md": str(OUTPUT_DIR / "stage3_final_N32_legacy_semantic_separation_memo.md"),
        "n32_memo_json": str(OUTPUT_DIR / "stage3_final_N32_legacy_semantic_separation_memo.json"),
    }


def write_ara_index(paths: dict[str, str]) -> dict[str, str]:
    index = {
        "logic": {
            "final_claim_boundary": paths["claim_boundary_md"],
            "claim_evidence_map": str(OUTPUT_DIR / "stage3_final_claim_evidence_map.md"),
            "paper_safe_conclusions": paths["paper_safe_conclusions_md"],
        },
        "data": {
            "frozen_native_combined552": str(OUTPUT_DIR / "FROZEN_stage3_native_combined552_RL_ready_dataset.csv"),
            "frozen_plus_N32": str(OUTPUT_DIR / "FROZEN_stage3_combined552_plus_N32_RL_ready_dataset.csv"),
            "file_hashes": str(OUTPUT_DIR / "FROZEN_stage3_file_hashes.csv"),
        },
        "trace": {
            "run_by_run_ledger": str(OUTPUT_DIR / "stage3_final_run_by_run_evidence_ledger.md"),
            "record_timeline": str(OUTPUT_DIR / "stage3_final_metric_reward_record_timeline.md"),
            "manifest": str(MANIFEST_PATH),
        },
        "evidence": {
            "best_strategy_table": str(OUTPUT_DIR / "stage3_final_native_best_strategy_table.md"),
            "top5": str(OUTPUT_DIR / "stage3_final_native_top5_by_N_and_objective.csv"),
            "top10": str(OUTPUT_DIR / "stage3_final_native_top10_by_N_and_objective.csv"),
            "final_maturity_audit": str(OUTPUT_DIR / "FROZEN_stage3_final_maturity_and_evidence_freeze_readiness_audit.md"),
        },
        "caveats": {
            "N32_semantic_separation": paths["n32_memo_md"],
            "GNN_pointer_auxiliary_status": "GNN and graph-pointer diagnostics remain auxiliary only.",
            "tested_N_boundary": "Native claims are bounded to N12, N16, N24, and N40.",
            "no_global_optimum_boundary": "No global or physical optimum is claimed.",
        },
        "executable_workflow": {
            "script": str(SCRIPT_PATH),
            "no_Abaqus_no_ODB_no_training": True,
        },
    }
    md = [
        "# ARA Stage 3 Final Evidence Index",
        "",
        "## Logic",
        f"- Final claim boundary: `{index['logic']['final_claim_boundary']}`",
        f"- Claim-evidence map: `{index['logic']['claim_evidence_map']}`",
        f"- Paper-safe conclusions: `{index['logic']['paper_safe_conclusions']}`",
        "",
        "## Data",
        f"- Frozen native combined552: `{index['data']['frozen_native_combined552']}`",
        f"- Frozen plus-N32: `{index['data']['frozen_plus_N32']}`",
        f"- File hashes: `{index['data']['file_hashes']}`",
        "",
        "## Trace",
        f"- Run-by-run ledger: `{index['trace']['run_by_run_ledger']}`",
        f"- Record timeline: `{index['trace']['record_timeline']}`",
        "",
        "## Evidence",
        f"- Best strategy table: `{index['evidence']['best_strategy_table']}`",
        f"- Top5 table: `{index['evidence']['top5']}`",
        f"- Top10 table: `{index['evidence']['top10']}`",
        "",
        "## Caveats",
        f"- N32 semantic separation: `{index['caveats']['N32_semantic_separation']}`",
        "- GNN/pointer diagnostics are auxiliary.",
        "- Tested-N boundary: N12/N16/N24/N40.",
        "- No global optimum claim.",
        "",
        "## Executable Workflow",
        f"- Script: `{index['executable_workflow']['script']}`",
        "- Run78 did not run Abaqus, open ODB, train models, or generate candidates.",
        "",
    ]
    (OUTPUT_DIR / "ARA_STAGE3_FINAL_EVIDENCE_INDEX.md").write_text("\n".join(md), encoding="utf-8")
    write_json(OUTPUT_DIR / "ARA_STAGE3_FINAL_EVIDENCE_INDEX.json", index)
    return {
        "ara_index_md": str(OUTPUT_DIR / "ARA_STAGE3_FINAL_EVIDENCE_INDEX.md"),
        "ara_index_json": str(OUTPUT_DIR / "ARA_STAGE3_FINAL_EVIDENCE_INDEX.json"),
    }


def build_summary(best_df: pd.DataFrame, validation: dict[str, Any]) -> dict[str, Any]:
    best_highlights: dict[str, dict[str, str]] = {}
    for _, row in best_df.iterrows():
        n = str(int(row["n"]))
        best_highlights[n] = {
            "U2": row["best_U2_strategy"],
            "u2_primary": row["best_u2_primary_strategy"],
            "constrained_reward": row["best_constrained_reward_strategy"],
            "strict_penalty_guard": row["best_strict_penalty_guard_strategy"],
            "penalty_repair": row["best_penalty_repair_strategy"],
        }
    summary = {
        "final_verdict": "PASS_STAGE3_FINAL_EVIDENCE_FREEZE_READY_WITH_BOUNDED_NATIVE_N_CLAIMS",
        "final_native_dataset": "combined552",
        "final_plus_N32_auxiliary_dataset": "combined552_plus_N32",
        "native_per_N_counts": validation["native_combined552_per_N_counts"],
        "combined552_rows": validation["native_combined552_rows"],
        "combined552_plus_N32_rows": validation["combined552_plus_N32_rows"],
        "N32_legacy_auxiliary_count": 332,
        "final_best_strategy_highlights": best_highlights,
        "N12_N16_recovery_summary": "N12/N16 reached 78 native teacher rows each after recovery and final diagnostic loops.",
        "N24_N40_maturity_summary": "N24/N40 remain mature anchors with 190 and 206 native teacher rows respectively.",
        "GNN_pointer_summary": "GNN and graph-pointer diagnostics remain auxiliary only.",
        "final_claim_boundary": "Bounded teacher-validated claims over tested native N values N12/N16/N24/N40 only.",
        "recommended_next_step": "Prepare paper/report/ARA package; do not generate more Stage 3 candidates by default.",
    }
    write_json(OUTPUT_DIR / "STAGE3_FINAL_EVIDENCE_FREEZE_SUMMARY.json", summary)
    md = [
        "# Stage 3 Final Evidence Freeze Summary",
        "",
        f"Final verdict: `{summary['final_verdict']}`",
        "",
        "## Final Datasets",
        "- Native: combined552",
        "- Auxiliary plus-N32: combined552_plus_N32",
        f"- Native counts: {summary['native_per_N_counts']}",
        "- N32 legacy auxiliary count: 332",
        "",
        "## Final Best Strategy Highlights",
        df_to_markdown(pd.DataFrame([
            {"n": n, **vals} for n, vals in best_highlights.items()
        ])),
        "",
        "## Interpretation",
        f"- {summary['N12_N16_recovery_summary']}",
        f"- {summary['N24_N40_maturity_summary']}",
        f"- {summary['GNN_pointer_summary']}",
        f"- {summary['final_claim_boundary']}",
        "",
        "## Recommended Next Step",
        summary["recommended_next_step"],
        "",
    ]
    (OUTPUT_DIR / "STAGE3_FINAL_EVIDENCE_FREEZE_SUMMARY.md").write_text("\n".join(md), encoding="utf-8")
    return summary


def write_report(validation: dict[str, Any], hash_rows: list[dict[str, Any]], best_df: pd.DataFrame, summary: dict[str, Any], output_paths: dict[str, str]) -> None:
    report_lines = [
        "# Stage 3 Run 78 - Final Evidence Freeze Package",
        "",
        "## 1. Purpose",
        "Run78 freezes the final Stage 3 evidence package for reporting, paper writing, and ARA-style evidence archiving. It is not a new optimization run.",
        "",
        "## 2. Inputs",
        f"- Native combined552 teacher dataset: `{NATIVE_TEACHER}`",
        f"- Native combined552 RL-ready dataset: `{NATIVE_READY}`",
        f"- combined552_plus_N32 RL-ready dataset: `{PLUS_N32_READY}`",
        f"- Run77 maturity audit: `{RUN77_MATURITY_MD}`",
        f"- Run77 claim boundary: `{RUN77_CLAIM_MD}`",
        "",
        "## 3. Run77 Final Readiness Verdict",
        "`RUN77_STAGE3_FINAL_EVIDENCE_FREEZE_READY_WITH_BOUNDED_NATIVE_N_CLAIMS`",
        "",
        "## 4. Frozen Datasets and File Hashes",
        f"- File hashes: `{OUTPUT_DIR / 'FROZEN_stage3_file_hashes.csv'}`",
        f"- Frozen file count: {len(hash_rows)}",
        "",
        "## 5. Final Native combined552 Summary",
        f"- Rows: {validation['native_combined552_rows']}",
        f"- Per-N counts: {validation['native_combined552_per_N_counts']}",
        "",
        "## 6. combined552_plus_N32 Auxiliary Summary",
        f"- Rows: {validation['combined552_plus_N32_rows']}",
        f"- Per-N counts: {validation['combined552_plus_N32_per_N_counts']}",
        "- N32 is auxiliary legacy-compatible context, not native Stage 3 teacher validation.",
        "",
        "## 7. Final Best Strategy Table",
        df_to_markdown(best_df, max_cell_chars=80),
        "",
        "## 8. Final Top-k Tables",
        f"- Top5: `{OUTPUT_DIR / 'stage3_final_native_top5_by_N_and_objective.csv'}`",
        f"- Top10: `{OUTPUT_DIR / 'stage3_final_native_top10_by_N_and_objective.csv'}`",
        "",
        "## 9. Run-by-Run Evidence Ledger",
        f"`{OUTPUT_DIR / 'stage3_final_run_by_run_evidence_ledger.md'}`",
        "",
        "## 10. Metric/Reward Record Timeline",
        f"`{OUTPUT_DIR / 'stage3_final_metric_reward_record_timeline.md'}`",
        "",
        "## 11. Claim-Evidence Map",
        f"`{OUTPUT_DIR / 'stage3_final_claim_evidence_map.md'}`",
        "",
        "## 12. Final Claim Boundary",
        f"`{OUTPUT_DIR / 'stage3_final_claim_boundary.md'}`",
        "",
        "## 13. Paper-Safe Conclusions",
        f"`{OUTPUT_DIR / 'stage3_final_paper_safe_conclusions.md'}`",
        "",
        "## 14. N32 Legacy-Semantic Separation",
        f"`{OUTPUT_DIR / 'stage3_final_N32_legacy_semantic_separation_memo.md'}`",
        "",
        "## 15. ARA-Style Evidence Index",
        f"`{OUTPUT_DIR / 'ARA_STAGE3_FINAL_EVIDENCE_INDEX.md'}`",
        "",
        "## 16. Final Freeze Summary",
        f"`{OUTPUT_DIR / 'STAGE3_FINAL_EVIDENCE_FREEZE_SUMMARY.md'}`",
        "",
        "## 17. Output Files",
        *[f"- `{path}`" for path in output_paths.values()],
        "",
        "## 18. Recommended Next Action",
        "Do not generate more Stage 3 candidates by default. Use this evidence-freeze package to prepare the final Stage 3 write-up, paper methods/results sections, figures, and GitHub ARA-style evidence archive.",
        "",
        "## Run78 Safety Boundary",
        "Run78 did not run Abaqus, open ODB files, run solver/datacheck/abqjobpilot/enqueue, generate CAE/INP/JNL, train models, or generate candidates.",
        "",
    ]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    native = pd.read_csv(NATIVE_READY)
    plus = pd.read_csv(PLUS_N32_READY)
    validation = validate_inputs(native, plus)
    hash_rows, frozen_paths = freeze_inputs()
    best_df = build_best_table(native)
    topk_paths = write_topk(native)
    ledger = build_run_ledger()
    timeline = build_record_timeline(native)
    claim_map = build_claim_map()
    boundary_paths = write_claim_boundary_and_conclusions()
    n32_paths = write_n32_memo()
    ara_paths = write_ara_index({**boundary_paths, **n32_paths})
    summary = build_summary(best_df, validation)

    output_paths = {
        **frozen_paths,
        "file_hashes_csv": str(OUTPUT_DIR / "FROZEN_stage3_file_hashes.csv"),
        "file_hashes_json": str(OUTPUT_DIR / "FROZEN_stage3_file_hashes.json"),
        "best_strategy_csv": str(OUTPUT_DIR / "stage3_final_native_best_strategy_table.csv"),
        "best_strategy_json": str(OUTPUT_DIR / "stage3_final_native_best_strategy_table.json"),
        "best_strategy_md": str(OUTPUT_DIR / "stage3_final_native_best_strategy_table.md"),
        **topk_paths,
        "run_ledger_csv": str(OUTPUT_DIR / "stage3_final_run_by_run_evidence_ledger.csv"),
        "record_timeline_csv": str(OUTPUT_DIR / "stage3_final_metric_reward_record_timeline.csv"),
        "claim_evidence_map_csv": str(OUTPUT_DIR / "stage3_final_claim_evidence_map.csv"),
        **boundary_paths,
        **n32_paths,
        **ara_paths,
        "freeze_summary_md": str(OUTPUT_DIR / "STAGE3_FINAL_EVIDENCE_FREEZE_SUMMARY.md"),
        "freeze_summary_json": str(OUTPUT_DIR / "STAGE3_FINAL_EVIDENCE_FREEZE_SUMMARY.json"),
    }
    write_report(validation, hash_rows, best_df, summary, output_paths)
    output_paths["main_report"] = str(REPORT_PATH)
    output_paths["manifest"] = str(MANIFEST_PATH)

    manifest = {
        "run_id": "run_78",
        "run_name": RUN_NAME,
        "timestamp": now_iso(),
        "branch": current_branch(),
        "script_path": str(SCRIPT_PATH),
        "input_files": {
            "native_combined552_teacher": str(NATIVE_TEACHER),
            "native_combined552_RL_ready": str(NATIVE_READY),
            "combined552_plus_N32_teacher": str(PLUS_N32_TEACHER),
            "combined552_plus_N32_RL_ready": str(PLUS_N32_READY),
            "run77_maturity_audit": str(RUN77_MATURITY_MD),
            "run77_claim_boundary": str(RUN77_CLAIM_MD),
            "run77_report": str(RUN77_REPORT),
            "run77_manifest": str(RUN77_MANIFEST),
        },
        "output_files": output_paths,
        "final_verdict": summary["final_verdict"],
        "frozen_dataset_paths": {k: v for k, v in frozen_paths.items() if "dataset" in k or "leaderboard" in k or "summary" in k},
        "file_hash_paths": {
            "csv": str(OUTPUT_DIR / "FROZEN_stage3_file_hashes.csv"),
            "json": str(OUTPUT_DIR / "FROZEN_stage3_file_hashes.json"),
        },
        "native_combined552_rows": validation["native_combined552_rows"],
        "combined552_plus_N32_rows": validation["combined552_plus_N32_rows"],
        "per_N_native_counts": validation["native_combined552_per_N_counts"],
        "per_N_plus_N32_counts": validation["combined552_plus_N32_per_N_counts"],
        "final_best_table_path": str(OUTPUT_DIR / "stage3_final_native_best_strategy_table.csv"),
        "top_k_table_paths": topk_paths,
        "run_by_run_ledger_path": str(OUTPUT_DIR / "stage3_final_run_by_run_evidence_ledger.csv"),
        "record_timeline_path": str(OUTPUT_DIR / "stage3_final_metric_reward_record_timeline.csv"),
        "claim_evidence_map_path": str(OUTPUT_DIR / "stage3_final_claim_evidence_map.csv"),
        "final_claim_boundary_path": str(OUTPUT_DIR / "stage3_final_claim_boundary.md"),
        "paper_safe_conclusions_path": str(OUTPUT_DIR / "stage3_final_paper_safe_conclusions.md"),
        "N32_memo_path": str(OUTPUT_DIR / "stage3_final_N32_legacy_semantic_separation_memo.md"),
        "ARA_index_path": str(OUTPUT_DIR / "ARA_STAGE3_FINAL_EVIDENCE_INDEX.md"),
        "final_freeze_summary_path": str(OUTPUT_DIR / "STAGE3_FINAL_EVIDENCE_FREEZE_SUMMARY.md"),
        "report_path": str(REPORT_PATH),
        "claim_evidence_map_status_counts": claim_map["status"].value_counts().to_dict(),
        "record_timeline_rows": int(len(timeline)),
        "run_ledger_rows": int(len(ledger)),
        "no_solver_run": True,
        "no_odb_opened": True,
        "no_abqjobpilot_run": True,
        "no_cae_inp_generated": True,
        "no_teacher_validation": True,
        "no_training": True,
        "no_candidate_generation": True,
        "no_commit_or_push": True,
    }
    write_json(MANIFEST_PATH, manifest)

    print(json.dumps({
        "verdict": validation["verdict"],
        "final_verdict": summary["final_verdict"],
        "native_counts": validation["native_combined552_per_N_counts"],
        "plus_N32_counts": validation["combined552_plus_N32_per_N_counts"],
        "hashes": str(OUTPUT_DIR / "FROZEN_stage3_file_hashes.csv"),
        "report": str(REPORT_PATH),
        "manifest": str(MANIFEST_PATH),
    }, indent=2))


if __name__ == "__main__":
    main()
