from __future__ import annotations

import json
import math
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
RUN_ID = "run_74_run73_final_smallN_diagnostic_batch32_handoff_package"
RUN_NAME = "run73 final small-N diagnostic batch32 handoff package"
SCRIPT_PATH = ROOT / "scripts" / "stage3" / "run_74_create_run73_final_smallN_diagnostic_batch32_handoff_package.py"

RUN73_DIR = ROOT / "outputs" / "stage3_run_73_combined520_model_update_final_smallN_diagnostic_candidate_generation"
RUN73_OPTION_A = RUN73_DIR / "run73_final_smallN_diagnostic_batch32_candidate_orders.csv"
RUN73_POOL = RUN73_DIR / "run73_candidate_pool_scored.csv"
RUN73_COMPARISON = RUN73_DIR / "run73_batch_options_comparison_to_previous.csv"
RUN73_COMPARISON_SUMMARY = RUN73_DIR / "run73_batch_options_comparison_summary.json"
RUN73_EVIDENCE_UPDATE = RUN73_DIR / "stage3_evidence_freeze_readiness_after_run72.md"
RUN73_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_73_combined520_model_update_final_smallN_diagnostic_candidate_generation" / "RUN_73_COMBINED520_MODEL_UPDATE_FINAL_SMALLN_DIAGNOSTIC_CANDIDATE_GENERATION_REPORT.md"
RUN73_MANIFEST = ROOT / "artifacts" / "manifests" / "stage3_run_73_manifest.json"
COMBINED520_READY = ROOT / "outputs" / "stage3_run_72_smallN_recovery_focused_batch40_teacher_metrics_ingestion_and_combined520_ranking" / "combined520_RL_ready_dataset.csv"

RUN71_HANDOFF = ROOT / "outputs" / "stage3_run_69_run68_smallN_recovery_focused_batch40_handoff_package" / "stage3_run69_smallN_recovery_focused_batch40_candidate_orders.csv"
RUN66_HANDOFF = ROOT / "outputs" / "stage3_run_64_run63_variable_N_recovery_anchor_batch48_handoff_package" / "stage3_run64_variable_N_recovery_anchor_batch48_candidate_orders.csv"
RUN61_HANDOFF = ROOT / "outputs" / "stage3_run_59_run58_N40_focused_calibrated_penalty_repair_batch40_handoff_package" / "stage3_run59_N40_focused_calibrated_penalty_repair_batch40_candidate_orders.csv"
RUN56_HANDOFF = ROOT / "outputs" / "stage3_run_54_run53_calibrated_N24_N40_batch64_handoff_package" / "stage3_run54_calibrated_N24_N40_batch64_candidate_orders.csv"
RUN51_HANDOFF = ROOT / "outputs" / "stage3_run_49_run48_stricter_constrained_N24_N40_batch32_handoff_package" / "stage3_run49_stricter_constrained_N24_N40_batch32_candidate_orders.csv"
RUN46_HANDOFF = ROOT / "outputs" / "stage3_run_44_run43_constrained_N24_N40_batch32_handoff_package" / "stage3_run44_constrained_N24_N40_batch32_candidate_orders.csv"
RUN41_HANDOFF = ROOT / "outputs" / "stage3_run_39_run38_native_N24_N40_focused_batch60_handoff_package" / "stage3_run39_native_N24_N40_focused_batch60_candidate_orders.csv"
RUN36_HANDOFF = ROOT / "outputs" / "stage3_run_34_run33_N32_informed_native_batch32_handoff_package" / "stage3_run34_N32_informed_native_batch32_candidate_orders.csv"
RUN27_HANDOFF = ROOT / "outputs" / "stage3_run_24_run23_shortlist64_active_learning_handoff_package" / "stage3_run24_shortlist64_candidate_orders.csv"
RUN31_OLD = ROOT / "outputs" / "stage3_run_30_run29_hybrid_policy_batch32_handoff_package" / "stage3_run30_hybrid_policy_batch32_candidate_orders.csv"

OUTPUT_DIR = ROOT / "outputs" / "stage3_run_74_run73_final_smallN_diagnostic_batch32_handoff_package"
SCAN_DIR = OUTPUT_DIR / "scan_orders"
REPORT_DIR = ROOT / "docs" / "stage3" / "runs" / "run_74_run73_final_smallN_diagnostic_batch32_handoff_package"
REPORT_PATH = REPORT_DIR / "RUN_74_RUN73_FINAL_SMALLN_DIAGNOSTIC_BATCH32_HANDOFF_PACKAGE_REPORT.md"
MANIFEST_PATH = ROOT / "artifacts" / "manifests" / "stage3_run_74_manifest.json"
CLAIM_BOUNDARY_MD = OUTPUT_DIR / "run74_claim_boundary.md"
CLAIM_BOUNDARY_JSON = OUTPUT_DIR / "run74_claim_boundary.json"

BATCH_NAME = "stage3_run74_final_smallN_diagnostic_batch32_v01"
BATCH_OPTION = "final_smallN_diagnostic_batch32"
EXPECTED_COUNTS = {12: 14, 16: 14, 24: 2, 40: 2}

VALIDATION_JSON = OUTPUT_DIR / "run74_input_validation_summary.json"
HANDOFF_CSV = OUTPUT_DIR / "stage3_run74_final_smallN_diagnostic_batch32_candidate_orders.csv"
CAE_TEMPLATE = OUTPUT_DIR / "stage3_run74_final_smallN_diagnostic_batch32_future_cae_handoff_manifest_TEMPLATE.csv"
ABQ_TEMPLATE = OUTPUT_DIR / "stage3_run74_final_smallN_diagnostic_batch32_abqjobpilot_commands_TEMPLATE.txt"
REVIEW_CSV = OUTPUT_DIR / "final_smallN_diagnostic_batch32_review_summary.csv"
REVIEW_JSON = OUTPUT_DIR / "final_smallN_diagnostic_batch32_review_summary.json"
REVIEW_MD = OUTPUT_DIR / "final_smallN_diagnostic_batch32_review_summary.md"


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
    if hasattr(value, "item"):
        try:
            return clean_json(value.item())
        except Exception:
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(payload), indent=2, sort_keys=False) + "\n", encoding="utf-8")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def parse_order(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(x) for x in value]
    text = str(value).strip()
    if not text:
        raise ValueError("empty order")
    if text.startswith("["):
        return [int(x) for x in json.loads(text)]
    return [int(x) for x in text.replace(",", "-").replace(";", "-").replace(" ", "").split("-") if x != ""]


def valid_order(value: Any, n: int) -> bool:
    try:
        order = parse_order(value)
    except Exception:
        return False
    return len(order) == n and sorted(order) == list(range(n))


def order_json(value: Any) -> str:
    return json.dumps(parse_order(value), separators=(",", ":"))


def compact_order(value: Any) -> str:
    return "-".join(str(x) for x in parse_order(value))


def counts(df: pd.DataFrame) -> dict[int, int]:
    return {int(float(k)): int(v) for k, v in df["n"].astype(float).astype(int).value_counts().sort_index().to_dict().items()}


def safe_fragment(text: Any, max_len: int = 28) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", str(text).strip()).strip("_")
    return (cleaned or "candidate")[:max_len]


def short_bucket(row: pd.Series) -> str:
    source = str(row.get("candidate_source", "") or row.get("selection_bucket", "candidate"))
    mapping = {
        "N12_final_diag_surrogate_top": "n12_final_top",
        "N12_local_search_around_Run71_best": "n12_run71_local",
        "N12_final_diag_u2_safe": "n12_u2_safe",
        "N12_final_diag_penalty_aware": "n12_penalty",
        "N12_final_diag_reward_balanced": "n12_reward_bal",
        "N12_final_diag_uncertainty": "n12_uncertainty",
        "N12_final_diag_diversity": "n12_diversity",
        "N12_final_diag_sentinel_control": "n12_sentinel",
        "N16_final_diag_surrogate_top": "n16_final_top",
        "N16_local_search_around_Run71_best": "n16_run71_local",
        "N16_final_diag_u2_safe": "n16_u2_safe",
        "N16_final_diag_penalty_aware": "n16_penalty",
        "N16_final_diag_reward_balanced": "n16_reward_bal",
        "N16_final_diag_uncertainty": "n16_uncertainty",
        "N16_final_diag_diversity": "n16_diversity",
        "N16_final_diag_sentinel_control": "n16_sentinel",
        "N24_anchor_top_density_reference": "n24_anchor_ref",
        "N40_anchor_u2_reward_reference": "n40_anchor_ref",
        "N12_recovery_surrogate_top": "n12_recovery_top",
        "N12_local_search_around_Run66_best": "n12_run66_local",
        "N12_recovery_u2_safe": "n12_u2_safe",
        "N12_recovery_penalty_aware": "n12_penalty",
        "N12_recovery_reward_balanced": "n12_reward_bal",
        "N12_recovery_uncertainty": "n12_uncertainty",
        "N12_recovery_diversity": "n12_diversity",
        "N12_recovery_sentinel_control": "n12_sentinel",
        "N16_recovery_surrogate_top": "n16_recovery_top",
        "N16_local_search_around_Run66_best": "n16_run66_local",
        "N16_recovery_u2_safe": "n16_u2_safe",
        "N16_recovery_penalty_aware": "n16_penalty",
        "N16_recovery_reward_balanced": "n16_reward_bal",
        "N16_recovery_uncertainty": "n16_uncertainty",
        "N16_recovery_diversity": "n16_diversity",
        "N16_recovery_sentinel_control": "n16_sentinel",
        "N24_frozen_top_density_reference": "n24_anchor_ref",
        "N24_uncertainty_anchor": "n24_uncertainty",
        "N24_sentinel_control": "n24_sentinel",
        "N40_frozen_u2_reward_reference": "n40_anchor_ref",
        "N40_uncertainty_anchor": "n40_uncertainty",
        "N40_sentinel_control": "n40_sentinel",
    }
    return safe_fragment(mapping.get(source, source.lower()), 24)


def load_hashes(path: Path) -> set[str]:
    if not path.exists():
        return set()
    df = read_csv(path)
    if "order_hash" not in df.columns:
        return set()
    return {str(x) for x in df["order_hash"].dropna().astype(str) if str(x)}


def previous_hashes() -> dict[str, set[str]]:
    return {
        "combined520": load_hashes(COMBINED520_READY),
        "run71": load_hashes(RUN71_HANDOFF),
        "run66": load_hashes(RUN66_HANDOFF),
        "run61": load_hashes(RUN61_HANDOFF),
        "run56": load_hashes(RUN56_HANDOFF),
        "run51": load_hashes(RUN51_HANDOFF),
        "run46": load_hashes(RUN46_HANDOFF),
        "run41": load_hashes(RUN41_HANDOFF),
        "run36": load_hashes(RUN36_HANDOFF),
        "run27": load_hashes(RUN27_HANDOFF),
        "superseded_run31": load_hashes(RUN31_OLD),
    }


def validate_input(df: pd.DataFrame) -> dict[str, Any]:
    errors: list[str] = []
    if not RUN73_OPTION_A.exists():
        errors.append(f"missing selected input CSV: {RUN73_OPTION_A}")
    if len(df) != 32:
        errors.append(f"row count expected 32, got {len(df)}")
    df_n = df["n"].astype(float).astype(int) if "n" in df.columns else pd.Series([], dtype=int)
    actual_counts = counts(df) if "n" in df.columns else {}
    if actual_counts != EXPECTED_COUNTS:
        errors.append(f"per-N counts expected {EXPECTED_COUNTS}, got {actual_counts}")
    if 32 in set(df_n):
        errors.append("selected batch contains N32 rows")
    for col in ["candidate_id", "strategy_name", "n", "order_json"]:
        if col not in df.columns:
            errors.append(f"missing required column {col}")
    if "batch_option" in df.columns and not (df["batch_option"].astype(str) == BATCH_OPTION).all():
        errors.append("selected batch is not uniformly Run73 Option A final_smallN_diagnostic_batch32")

    bad_orders = []
    if "n" in df.columns:
        for _, row in df.iterrows():
            n = int(float(row["n"]))
            raw_order = row.get("order_json", row.get("scan_order", ""))
            if not valid_order(raw_order, n):
                bad_orders.append(str(row.get("candidate_id", row.get("strategy_name", "UNKNOWN"))))
    if bad_orders:
        errors.append(f"invalid scan orders: {bad_orders[:5]}")

    if "order_hash" in df.columns and "n" in df.columns:
        dup_orders = df.assign(_n=df_n).groupby("_n")["order_hash"].apply(lambda s: int(s.astype(str).duplicated().sum())).to_dict()
        if any(v for v in dup_orders.values()):
            errors.append(f"duplicate order_hash within N: {dup_orders}")
    if "candidate_id" in df.columns and df["candidate_id"].astype(str).duplicated().any():
        errors.append("duplicate candidate_id values")

    prev = previous_hashes()
    hashes = set(df.get("order_hash", pd.Series([], dtype=str)).astype(str))
    overlap = {name: int(len(hashes & phashes)) for name, phashes in prev.items()}
    for name, count in overlap.items():
        if count:
            errors.append(f"exact order overlap with {name}: {count}")

    for col in ["candidate_source", "generation_method", "selection_bucket", "priority_role"]:
        if col not in df.columns:
            errors.append(f"missing metadata column {col}")
    for col in [
        "surrogate_prediction", "N12_recovery_score", "N16_recovery_score",
        "variable_N_recovery_score", "N24_maintenance_score", "N40_followup_score",
        "gnn_reward_prediction", "graph_pointer_policy_score", "novelty_distance",
    ]:
        if col not in df.columns:
            errors.append(f"missing prediction/recovery metadata column {col}")

    summary = {
        "timestamp": now_iso(),
        "verdict": "PASS_RUN74_FINAL_SMALLN_DIAGNOSTIC_BATCH32_INPUT_READY" if not errors else "FAIL_RUN74_INPUT_VALIDATION",
        "errors": errors,
        "selected_batch": "run73_final_smallN_diagnostic_batch32",
        "row_count": int(len(df)),
        "per_N_counts": actual_counts,
        "includes_N12": bool(12 in actual_counts),
        "includes_N16": bool(16 in actual_counts),
        "includes_N24": bool(24 in actual_counts),
        "includes_N32": bool(32 in actual_counts),
        "includes_N40": bool(40 in actual_counts),
        "exact_overlap_counts": overlap,
        "input_path": str(RUN73_OPTION_A),
    }
    write_json(VALIDATION_JSON, summary)
    return summary


def rename_handoff(df: pd.DataFrame) -> pd.DataFrame:
    handoff = df.copy()
    handoff["n"] = handoff["n"].astype(float).astype(int)
    handoff = handoff.sort_values(["n", "candidate_source", "surrogate_prediction"], ascending=[True, True, False]).reset_index(drop=True)
    per_n_counter: Counter[int] = Counter()
    names = []
    for _, row in handoff.iterrows():
        n = int(row["n"])
        per_n_counter[n] += 1
        names.append(f"S3R74FSD_N{n}_B{per_n_counter[n]:02d}_{short_bucket(row)}")
    handoff["handoff_strategy_name"] = names
    handoff["run_id"] = RUN_ID
    handoff["batch_option"] = BATCH_OPTION
    handoff["batch_name"] = BATCH_NAME
    handoff["original_run73_candidate_id"] = handoff.get("candidate_id", "")
    handoff["original_run73_strategy_name"] = handoff.get("strategy_name", "")
    handoff["N12_final_diagnostic_prediction"] = handoff.get("N12_recovery_score")
    handoff["N16_final_diagnostic_prediction"] = handoff.get("N16_recovery_score")
    handoff["smallN_final_diagnostic_prediction"] = handoff.get("variable_N_recovery_score")
    handoff["variable_N_bounded_prediction"] = handoff.get("variable_N_recovery_score")
    handoff["N24_anchor_prediction"] = handoff.get("N24_maintenance_score")
    handoff["N40_anchor_prediction"] = handoff.get("N40_followup_score")
    handoff["u2_primary_prediction"] = handoff.get("predicted_u2_guarded_score")
    handoff["constrained_reward_prediction"] = handoff.get("constrained_score")
    handoff["strict_penalty_guard_prediction"] = handoff.get("strict_penalty_guard_score")
    handoff["penalty_repair_prediction"] = handoff.get("penalty_repair_score")
    handoff["native_validation_N"] = True
    handoff["final_smallN_diagnostic"] = True
    handoff["smallN_recovery"] = True
    handoff["variable_N_bounded"] = True
    handoff["anchor_case"] = handoff["n"].isin([24, 40])
    handoff["order_json"] = handoff["order_json"].map(order_json)
    handoff["order_compact"] = handoff["order_json"].map(compact_order)
    handoff["teacher_validated"] = False
    handoff["teacher_validation_status"] = "NOT_RUN"
    handoff["notes"] = "Run74 handoff only. Final small-N diagnostic batch32 candidate is not teacher-validated."

    columns = [
        "run_id", "batch_option", "batch_name", "n", "handoff_strategy_name",
        "original_run73_candidate_id", "original_run73_strategy_name",
        "candidate_source", "generation_method", "selection_bucket", "priority_role",
        "surrogate_prediction", "N12_final_diagnostic_prediction", "N16_final_diagnostic_prediction",
        "smallN_final_diagnostic_prediction", "variable_N_bounded_prediction",
        "N24_anchor_prediction", "N40_anchor_prediction",
        "u2_primary_prediction", "constrained_reward_prediction", "strict_penalty_guard_prediction",
        "penalty_repair_prediction", "gnn_reward_prediction", "graph_pointer_policy_score",
        "hybrid_score", "uncertainty_score", "gnn_vs_surrogate_disagreement", "novelty_distance",
        "nearest_existing_teacher_strategy", "native_validation_N", "final_smallN_diagnostic",
        "smallN_recovery", "variable_N_bounded", "anchor_case", "order_json", "order_compact", "order_hash",
        "teacher_validated", "teacher_validation_status", "notes",
    ]
    for col in columns:
        if col not in handoff.columns:
            handoff[col] = ""
    return handoff[columns]


def row_metadata(row: pd.Series) -> dict[str, Any]:
    return {
        "surrogate_prediction": row.get("surrogate_prediction"),
        "N12_final_diagnostic_prediction": row.get("N12_final_diagnostic_prediction"),
        "N16_final_diagnostic_prediction": row.get("N16_final_diagnostic_prediction"),
        "smallN_final_diagnostic_prediction": row.get("smallN_final_diagnostic_prediction"),
        "variable_N_bounded_prediction": row.get("variable_N_bounded_prediction"),
        "N24_anchor_prediction": row.get("N24_anchor_prediction"),
        "N40_anchor_prediction": row.get("N40_anchor_prediction"),
        "u2_primary_prediction": row.get("u2_primary_prediction"),
        "constrained_reward_prediction": row.get("constrained_reward_prediction"),
        "strict_penalty_guard_prediction": row.get("strict_penalty_guard_prediction"),
        "penalty_repair_prediction": row.get("penalty_repair_prediction"),
        "gnn_reward_prediction": row.get("gnn_reward_prediction"),
        "graph_pointer_policy_score": row.get("graph_pointer_policy_score"),
        "hybrid_score": row.get("hybrid_score"),
        "uncertainty_score": row.get("uncertainty_score"),
        "gnn_vs_surrogate_disagreement": row.get("gnn_vs_surrogate_disagreement"),
        "novelty_distance": row.get("novelty_distance"),
        "nearest_existing_teacher_strategy": row.get("nearest_existing_teacher_strategy"),
    }


def write_scan_jsons(handoff: pd.DataFrame) -> None:
    SCAN_DIR.mkdir(parents=True, exist_ok=True)
    for _, row in handoff.iterrows():
        payload = {
            "run_id": RUN_ID,
            "batch_option": BATCH_OPTION,
            "batch_name": BATCH_NAME,
            "n": int(row["n"]),
            "handoff_strategy_name": row["handoff_strategy_name"],
            "original_run73_candidate_id": row["original_run73_candidate_id"],
            "candidate_source": row["candidate_source"],
            "generation_method": row["generation_method"],
            "selection_bucket": row["selection_bucket"],
            "priority_role": row["priority_role"],
            "metadata": row_metadata(row),
            "native_validation_N": True,
            "final_smallN_diagnostic": True,
            "smallN_recovery": True,
            "variable_N_bounded": True,
            "anchor_case": bool(row["anchor_case"]),
            "scan_order": parse_order(row["order_json"]),
            "order_hash": row["order_hash"],
            "teacher_validated": False,
            "teacher_validation_status": "NOT_RUN",
            "notes": "Run74 handoff only. Final small-N diagnostic batch32 candidate is not teacher-validated.",
        }
        write_json(SCAN_DIR / f"scan_order_{row['handoff_strategy_name']}.json", payload)


def write_future_templates(handoff: pd.DataFrame) -> None:
    cae_root = ROOT / "cae_model" / BATCH_NAME
    rows = []
    commands = [
        "# TEMPLATE ONLY - not ready to run until CAE/INP generation has completed and passed checks.",
        "# Do not execute this file during Run74.",
    ]
    for _, row in handoff.iterrows():
        n = int(row["n"])
        strategy = row["handoff_strategy_name"]
        case_dir = cae_root / f"N{n}{strategy}"
        job = f"J2D_{strategy}"
        inp = case_dir / f"{job}.inp"
        rows.append({
            "run_id": RUN_ID,
            "batch_name": BATCH_NAME,
            "n": n,
            "handoff_strategy_name": strategy,
            "expected_future_case_root": str(cae_root),
            "expected_future_case_dir": str(case_dir),
            "expected_future_job_name": job,
            "expected_future_inp_path": str(inp),
            "cae_inp_generated": False,
            "ready_to_run": False,
        })
        commands.append(f'enqueue --inp "{inp}" --cpus 14 --batch {BATCH_NAME} --strategy {strategy}')
    write_csv(CAE_TEMPLATE, pd.DataFrame(rows))
    ABQ_TEMPLATE.write_text("\n".join(commands) + "\n", encoding="utf-8")


def numeric_summary(df: pd.DataFrame, column: str) -> dict[int, float | None]:
    if column not in df.columns:
        return {n: None for n in sorted(df["n"].astype(int).unique())}
    values = pd.to_numeric(df[column], errors="coerce")
    return {int(n): (None if pd.isna(v) else float(v)) for n, v in values.groupby(df["n"].astype(int)).mean().items()}


def write_review(handoff: pd.DataFrame, validation: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "timestamp": now_iso(),
        "headline": "Run74 packages the recommended final small-N diagnostic before Stage 3 evidence freeze: N12/N16 are the focus while N24/N40 are retained only as minimal mature anchors.",
        "total_count": int(len(handoff)),
        "per_N_counts": counts(handoff),
        "includes_N12": True,
        "includes_N16": True,
        "includes_N24": True,
        "includes_N32": False,
        "includes_N40": True,
        "candidate_source_composition": handoff["candidate_source"].value_counts().to_dict(),
        "selection_bucket_composition": handoff["selection_bucket"].value_counts().to_dict(),
        "mean_surrogate_prediction_per_N": numeric_summary(handoff, "surrogate_prediction"),
        "mean_N12_final_diagnostic_prediction_per_N": numeric_summary(handoff, "N12_final_diagnostic_prediction"),
        "mean_N16_final_diagnostic_prediction_per_N": numeric_summary(handoff, "N16_final_diagnostic_prediction"),
        "mean_smallN_final_diagnostic_prediction_per_N": numeric_summary(handoff, "smallN_final_diagnostic_prediction"),
        "mean_variable_N_bounded_prediction_per_N": numeric_summary(handoff, "variable_N_bounded_prediction"),
        "mean_N24_anchor_prediction_per_N": numeric_summary(handoff, "N24_anchor_prediction"),
        "mean_N40_anchor_prediction_per_N": numeric_summary(handoff, "N40_anchor_prediction"),
        "mean_u2_primary_prediction_per_N": numeric_summary(handoff, "u2_primary_prediction"),
        "mean_constrained_reward_prediction_per_N": numeric_summary(handoff, "constrained_reward_prediction"),
        "mean_strict_penalty_guard_prediction_per_N": numeric_summary(handoff, "strict_penalty_guard_prediction"),
        "mean_penalty_repair_prediction_per_N": numeric_summary(handoff, "penalty_repair_prediction"),
        "mean_gnn_reward_prediction_per_N": numeric_summary(handoff, "gnn_reward_prediction"),
        "mean_hybrid_score_per_N": numeric_summary(handoff, "hybrid_score"),
        "mean_disagreement_per_N": numeric_summary(handoff, "gnn_vs_surrogate_disagreement"),
        "mean_novelty_distance_per_N": numeric_summary(handoff, "novelty_distance"),
        "expected_abaqus_cost": {"total_jobs": 32, "N12": 14, "N16": 14, "N24": 2, "N40": 2},
        "exact_overlap_status": validation["exact_overlap_counts"],
        "not_teacher_validated_until_future_abaqus_validation": True,
        "run74_created_cae_inp_files": False,
    }
    write_json(REVIEW_JSON, summary)
    review_rows = []
    for key, value in summary.items():
        review_rows.append({"metric": key, "value": json.dumps(clean_json(value), sort_keys=True) if isinstance(value, (dict, list)) else value})
    write_csv(REVIEW_CSV, pd.DataFrame(review_rows))
    REVIEW_MD.write_text(
        "# Final Small-N Diagnostic Batch32 Review Summary\n\n"
        f"{summary['headline']}\n\n"
        f"- Total: {summary['total_count']}\n"
        f"- Per-N counts: {summary['per_N_counts']}\n"
        "- Includes N12/N16/N24/N40; excludes N32.\n"
        "- This batch is the recommended final small-N diagnostic before Stage 3 evidence freeze.\n"
        "- It focuses on N12/N16 while retaining only minimal mature N24/N40 anchors.\n"
        "- This is not teacher-validated until future Abaqus validation.\n"
        "- Run74 did not create CAE/INP files.\n\n"
        "## Exact Overlap Status\n"
        + "\n".join(f"- {k}: {v}" for k, v in summary["exact_overlap_status"].items())
        + "\n",
        encoding="utf-8",
    )
    return summary


def write_claim_boundary() -> None:
    safe = [
        "Run74 packages selected Run73 Option A final small-N diagnostic batch32 candidates for human review and future CAE generation.",
        "The selected batch contains N12=14, N16=14, N24=2, and N40=2.",
        "No N32 candidates are included.",
        "The batch is designed as a final small-N diagnostic before Stage 3 evidence freeze.",
        "Handoff files include scan orders, metadata, future CAE manifest template, and abqjobpilot command template.",
        "No CAE/INP files were generated.",
    ]
    unsafe = [
        "Do not claim candidates are teacher-validated.",
        "Do not claim physical superiority.",
        "Do not claim N32 caused improvement.",
        "Do not claim GNN-RL has beaten baselines.",
        "Do not claim online RL with Abaqus.",
        "Do not claim arbitrary-N generalization.",
        "Do not claim full variable-N maturity.",
        "Do not claim surrogate/GNN/hybrid predictions are ground truth.",
        "Do not claim abqjobpilot commands are ready to execute.",
        "Do not claim CAE/INP files exist.",
        "Do not claim batch32 will improve teacher metrics before validation.",
    ]
    CLAIM_BOUNDARY_MD.write_text(
        "# Run74 Claim Boundary\n\n## Safe claims\n"
        + "\n".join(f"- {x}" for x in safe)
        + "\n\n## Unsafe claims\n"
        + "\n".join(f"- {x}" for x in unsafe)
        + "\n",
        encoding="utf-8",
    )
    write_json(CLAIM_BOUNDARY_JSON, {"verdict": "RUN74_HANDOFF_ONLY_FINAL_SMALLN_DIAGNOSTIC_BATCH32_NO_TEACHER_VALIDATION", "safe_claims": safe, "unsafe_claims": unsafe})


def write_report(validation: dict[str, Any], review: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(f"""# Stage 3 Run 74 - Run73 Final Small-N Diagnostic Batch32 Handoff Package

## 1. Purpose
Run74 packages the selected Run73 Option A final small-N diagnostic batch32 for human review and future CAE generation. It is handoff packaging only.

## 2. Inputs
- Selected Run73 Option A: `{RUN73_OPTION_A}`
- Run73 candidate pool: `{RUN73_POOL}`
- Run73 comparison summary: `{RUN73_COMPARISON_SUMMARY}`
- Run73 evidence update: `{RUN73_EVIDENCE_UPDATE}`
- Native combined520 RL-ready dataset: `{COMBINED520_READY}`

## 3. Selected Option A
Selected batch: `run73_final_smallN_diagnostic_batch32`.

Counts: `{validation['per_N_counts']}`.

## 4. Why Option A Was Selected
Option A is the recommended final small-N diagnostic before Stage 3 evidence freeze. It tests N12/N16 stability with enough density while keeping N24/N40 as minimal anchors.

## 5. Why This Is the Final Small-N Diagnostic Before Evidence Freeze
Run73 found Stage 3 evidence close to freeze-ready, but recommended one final small-N diagnostic loop before freezing. Option C remains available later if the user elects to stop validation and freeze immediately.

## 6. Why N12/N16 Remain the Focus
N12/N16 each have 64 native teacher rows, while N24/N40 have much denser mature anchor evidence at 188 and 204 rows respectively.

## 7. Why N24/N40 Are Minimal Anchors
N24/N40 are included only as minimal anchors to preserve continuity with the mature teacher regions, not as the main exploitation target for this handoff.

## 8. Validation Status
Verdict: `{validation['verdict']}`.

No N32 rows are included, and exact overlap counts are `{validation['exact_overlap_counts']}`.

## 9. Stable Naming Convention
`S3R74FSD_N{{N}}_B{{index:02d}}_{{short_bucket_or_family}}`

## 10. Candidate-Order Handoff Package
Candidate handoff CSV: `{HANDOFF_CSV}`.

## 11. Per-Candidate Scan-Order JSON Outputs
Scan-order JSON directory: `{SCAN_DIR}`.

## 12. Future CAE Handoff Template
Future CAE manifest template: `{CAE_TEMPLATE}`.

Expected future case root: `{ROOT / 'cae_model' / BATCH_NAME}`.

## 13. Future abqjobpilot Command Template
Future command template: `{ABQ_TEMPLATE}`.

This command file is a template only. INPs do not exist yet and the file is not ready to run until CAE/INP generation has completed and passed checks.

## 14. Review Summary
{review['headline']}

## 15. Claim Boundary
Verdict: `RUN74_HANDOFF_ONLY_FINAL_SMALLN_DIAGNOSTIC_BATCH32_NO_TEACHER_VALIDATION`.

## 16. Output Files
- Validation summary: `{VALIDATION_JSON}`
- Handoff CSV: `{HANDOFF_CSV}`
- Scan orders: `{SCAN_DIR}`
- Future CAE template: `{CAE_TEMPLATE}`
- Future abqjobpilot template: `{ABQ_TEMPLATE}`
- Review summary: `{REVIEW_MD}`
- Claim boundary: `{CLAIM_BOUNDARY_MD}`
- Manifest: `{MANIFEST_PATH}`

## 17. Recommended Run75
CAE module should generate CAE/INP/JNL for selected Run74 final small-N diagnostic batch32 only. Do not run solver. Do not execute abqjobpilot. Do not generate Run73 Option B batch24. Do not perform Option C stop-and-freeze unless explicitly selected later.
""", encoding="utf-8")


def write_manifest(validation: dict[str, Any], review: dict[str, Any]) -> None:
    output_files = [
        VALIDATION_JSON,
        HANDOFF_CSV,
        SCAN_DIR,
        CAE_TEMPLATE,
        ABQ_TEMPLATE,
        REVIEW_CSV,
        REVIEW_JSON,
        REVIEW_MD,
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
            str(RUN73_OPTION_A),
            str(RUN73_POOL),
            str(RUN73_COMPARISON),
            str(RUN73_COMPARISON_SUMMARY),
            str(RUN73_EVIDENCE_UPDATE),
            str(RUN73_REPORT),
            str(RUN73_MANIFEST),
            str(COMBINED520_READY),
        ],
        "output_files": [str(p) for p in output_files],
        "selected_batch": "run73_final_smallN_diagnostic_batch32",
        "batch_name": BATCH_NAME,
        "batch32_count": 32,
        "per_N_counts": validation["per_N_counts"],
        "includes_N12": True,
        "includes_N16": True,
        "includes_N24": True,
        "includes_N32": False,
        "includes_N40": True,
        "final_smallN_diagnostic": True,
        "smallN_recovery": True,
        "variable_N_bounded": True,
        "N24_N40_anchor": True,
        "report_path": str(REPORT_PATH),
        "claim_boundary_path": str(CLAIM_BOUNDARY_MD),
        "review_headline": review["headline"],
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


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SCAN_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    selected = read_csv(RUN73_OPTION_A)
    validation = validate_input(selected)
    if not validation["verdict"].startswith("PASS"):
        raise SystemExit(validation["errors"])

    handoff = rename_handoff(selected)
    write_csv(HANDOFF_CSV, handoff)
    write_scan_jsons(handoff)
    write_future_templates(handoff)
    review = write_review(handoff, validation)
    write_claim_boundary()
    write_report(validation, review)
    write_manifest(validation, review)
    print(json.dumps({
        "verdict": validation["verdict"],
        "selected_batch": "run73_final_smallN_diagnostic_batch32",
        "counts": counts(handoff),
        "handoff_csv": str(HANDOFF_CSV),
        "scan_order_json_dir": str(SCAN_DIR),
        "report": str(REPORT_PATH),
        "manifest": str(MANIFEST_PATH),
    }, indent=2))


if __name__ == "__main__":
    main()

