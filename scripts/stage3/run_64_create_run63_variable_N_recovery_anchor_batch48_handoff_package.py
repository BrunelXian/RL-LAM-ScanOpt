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
RUN_ID = "run_64_run63_variable_N_recovery_anchor_batch48_handoff_package"
RUN_NAME = "run63 variable-N recovery anchor batch48 handoff package"
SCRIPT_PATH = ROOT / "scripts" / "stage3" / "run_64_create_run63_variable_N_recovery_anchor_batch48_handoff_package.py"

RUN63_DIR = ROOT / "outputs" / "stage3_run_63_combined432_model_update_N24_N40_evidence_freeze_and_N12_N16_recovery_candidate_generation"
RUN63_OPTION_A = RUN63_DIR / "run63_variable_N_recovery_anchor_batch48_candidate_orders.csv"
RUN63_POOL = RUN63_DIR / "run63_candidate_pool_scored.csv"
RUN63_COMPARISON = RUN63_DIR / "run63_batch_options_comparison_to_previous.csv"
RUN63_COMPARISON_SUMMARY = RUN63_DIR / "run63_batch_options_comparison_summary.json"
RUN63_EVIDENCE_FREEZE = RUN63_DIR / "n24_n40_final_active_learning_rl_evidence_freeze.md"
RUN63_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_63_combined432_model_update_N24_N40_evidence_freeze_and_N12_N16_recovery_candidate_generation" / "RUN_63_COMBINED432_MODEL_UPDATE_N24_N40_EVIDENCE_FREEZE_AND_N12_N16_RECOVERY_CANDIDATE_GENERATION_REPORT.md"
RUN63_MANIFEST = ROOT / "artifacts" / "manifests" / "stage3_run_63_manifest.json"
COMBINED432_READY = ROOT / "outputs" / "stage3_run_62_custom_N40_focused_batch40_teacher_metrics_ingestion_and_combined432_ranking" / "combined432_RL_ready_dataset.csv"

RUN61_HANDOFF = ROOT / "outputs" / "stage3_run_59_run58_N40_focused_calibrated_penalty_repair_batch40_handoff_package" / "stage3_run59_N40_focused_calibrated_penalty_repair_batch40_candidate_orders.csv"
RUN56_HANDOFF = ROOT / "outputs" / "stage3_run_54_run53_calibrated_N24_N40_batch64_handoff_package" / "stage3_run54_calibrated_N24_N40_batch64_candidate_orders.csv"
RUN51_HANDOFF = ROOT / "outputs" / "stage3_run_49_run48_stricter_constrained_N24_N40_batch32_handoff_package" / "stage3_run49_stricter_constrained_N24_N40_batch32_candidate_orders.csv"
RUN46_HANDOFF = ROOT / "outputs" / "stage3_run_44_run43_constrained_N24_N40_batch32_handoff_package" / "stage3_run44_constrained_N24_N40_batch32_candidate_orders.csv"
RUN41_HANDOFF = ROOT / "outputs" / "stage3_run_39_run38_native_N24_N40_focused_batch60_handoff_package" / "stage3_run39_native_N24_N40_focused_batch60_candidate_orders.csv"
RUN36_HANDOFF = ROOT / "outputs" / "stage3_run_34_run33_N32_informed_native_batch32_handoff_package" / "stage3_run34_N32_informed_native_batch32_candidate_orders.csv"
RUN27_HANDOFF = ROOT / "outputs" / "stage3_run_24_run23_shortlist64_active_learning_handoff_package" / "stage3_run24_shortlist64_candidate_orders.csv"
RUN31_OLD = ROOT / "outputs" / "stage3_run_30_run29_hybrid_policy_batch32_handoff_package" / "stage3_run30_hybrid_policy_batch32_candidate_orders.csv"

OUTPUT_DIR = ROOT / "outputs" / "stage3_run_64_run63_variable_N_recovery_anchor_batch48_handoff_package"
SCAN_DIR = OUTPUT_DIR / "scan_orders"
REPORT_DIR = ROOT / "docs" / "stage3" / "runs" / "run_64_run63_variable_N_recovery_anchor_batch48_handoff_package"
REPORT_PATH = REPORT_DIR / "RUN_64_RUN63_VARIABLE_N_RECOVERY_ANCHOR_BATCH48_HANDOFF_PACKAGE_REPORT.md"
MANIFEST_PATH = ROOT / "artifacts" / "manifests" / "stage3_run_64_manifest.json"
CLAIM_BOUNDARY_MD = OUTPUT_DIR / "run64_claim_boundary.md"
CLAIM_BOUNDARY_JSON = OUTPUT_DIR / "run64_claim_boundary.json"

BATCH_NAME = "stage3_run64_variable_N_recovery_anchor_batch48_v01"
BATCH_OPTION = "variable_N_recovery_anchor_batch48"
EXPECTED_COUNTS = {12: 12, 16: 12, 24: 8, 40: 16}

HANDOFF_CSV = OUTPUT_DIR / "stage3_run64_variable_N_recovery_anchor_batch48_candidate_orders.csv"
CAE_TEMPLATE = OUTPUT_DIR / "stage3_run64_variable_N_recovery_anchor_batch48_future_cae_handoff_manifest_TEMPLATE.csv"
ABQ_TEMPLATE = OUTPUT_DIR / "stage3_run64_variable_N_recovery_anchor_batch48_abqjobpilot_commands_TEMPLATE.txt"
REVIEW_CSV = OUTPUT_DIR / "variable_N_recovery_anchor_batch48_review_summary.csv"
REVIEW_JSON = OUTPUT_DIR / "variable_N_recovery_anchor_batch48_review_summary.json"
REVIEW_MD = OUTPUT_DIR / "variable_N_recovery_anchor_batch48_review_summary.md"


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
    return {int(k): int(v) for k, v in df["n"].astype(int).value_counts().sort_index().to_dict().items()}


def safe_fragment(text: Any, max_len: int = 28) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", str(text).strip()).strip("_")
    return (cleaned or "candidate")[:max_len]


def short_bucket(row: pd.Series) -> str:
    source = str(row.get("candidate_source", "") or row.get("selection_bucket", "candidate"))
    mapping = {
        "N12_recovery_anchor_top": "n12_anchor_top",
        "N12_recovery_anchor_local_search": "n12_anchor_local",
        "N12_u2_safe_exploitation": "n12_u2_safe",
        "N12_penalty_aware_repair": "n12_penalty",
        "N12_uncertainty_calibration": "n12_uncertainty",
        "N12_diversity_coverage": "n12_diversity",
        "N12_sentinel_control": "n12_sentinel",
        "N16_recovery_anchor_top": "n16_anchor_top",
        "N16_recovery_anchor_local_search": "n16_anchor_local",
        "N16_u2_safe_exploitation": "n16_u2_safe",
        "N16_penalty_aware_repair": "n16_penalty",
        "N16_uncertainty_calibration": "n16_uncertainty",
        "N16_diversity_coverage": "n16_diversity",
        "N16_sentinel_control": "n16_sentinel",
        "N24_frozen_reference_maintenance": "n24_frozen_ref",
        "N24_frozen_penalty_diagnostic": "n24_penalty_diag",
        "N24_frozen_uncertainty_calibration": "n24_uncertainty",
        "N24_frozen_diversity_coverage": "n24_diversity",
        "N24_frozen_sentinel_control": "n24_sentinel",
        "N40_followup_u2_reward_retention": "n40_u2ret_anchor",
        "N40_followup_penalty_repair": "n40_penalty_anchor",
        "N40_followup_two_stage_repair": "n40_twostage_anchor",
        "N40_followup_median_guard": "n40_median_anchor",
        "N40_followup_uncertainty_calibration": "n40_uncertainty",
        "N40_followup_diversity_coverage": "n40_diversity",
        "N40_followup_sentinel_control": "n40_sentinel",
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
        "combined432": load_hashes(COMBINED432_READY),
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
    if not RUN63_OPTION_A.exists():
        errors.append(f"missing selected input CSV: {RUN63_OPTION_A}")
    if len(df) != 48:
        errors.append(f"row count expected 48, got {len(df)}")
    actual_counts = counts(df)
    if actual_counts != EXPECTED_COUNTS:
        errors.append(f"per-N counts expected {EXPECTED_COUNTS}, got {actual_counts}")
    if 32 in set(df["n"].astype(int)):
        errors.append("selected batch contains N32 rows")
    for col in ["candidate_id", "strategy_name", "n", "order_json"]:
        if col not in df.columns:
            errors.append(f"missing required column {col}")
    if "batch_option" in df.columns and not (df["batch_option"].astype(str) == "variable_N_recovery_anchor_batch48").all():
        errors.append("selected batch is not uniformly Run63 Option A variable_N_recovery_anchor_batch48")
    bad_orders = []
    for _, row in df.iterrows():
        n = int(row["n"])
        raw_order = row.get("order_json", row.get("scan_order", ""))
        if not valid_order(raw_order, n):
            bad_orders.append(str(row.get("candidate_id", row.get("strategy_name", "UNKNOWN"))))
    if bad_orders:
        errors.append(f"invalid scan orders: {bad_orders[:5]}")
    if "order_hash" in df.columns:
        dup_orders = df.groupby("n")["order_hash"].apply(lambda s: int(s.astype(str).duplicated().sum())).to_dict()
        if any(v for v in dup_orders.values()):
            errors.append(f"duplicate order_hash within N: {dup_orders}")
    if "candidate_id" in df.columns and df["candidate_id"].astype(str).duplicated().any():
        errors.append("duplicate candidate_id values")

    prev = previous_hashes()
    overlap = {}
    hashes = set(df.get("order_hash", pd.Series([], dtype=str)).astype(str))
    for name, phashes in prev.items():
        overlap[name] = int(len(hashes & phashes))
    for name, count in overlap.items():
        if count:
            errors.append(f"exact order overlap with {name}: {count}")

    for col in ["candidate_source", "generation_method", "selection_bucket", "priority_role"]:
        if col not in df.columns:
            errors.append(f"missing metadata column {col}")
    for col in ["surrogate_prediction", "N12_recovery_score", "N16_recovery_score", "variable_N_recovery_score", "N40_followup_score", "gnn_reward_prediction", "graph_pointer_policy_score", "novelty_distance"]:
        if col not in df.columns:
            errors.append(f"missing prediction/recovery metadata column {col}")

    summary = {
        "timestamp": now_iso(),
        "verdict": "PASS_RUN64_VARIABLE_N_RECOVERY_ANCHOR_BATCH48_INPUT_READY" if not errors else "FAIL_RUN64_INPUT_VALIDATION",
        "errors": errors,
        "selected_batch": "run63_variable_N_recovery_anchor_batch48",
        "row_count": int(len(df)),
        "per_N_counts": actual_counts,
        "includes_N12": bool(12 in actual_counts),
        "includes_N16": bool(16 in actual_counts),
        "includes_N24": bool(24 in actual_counts),
        "includes_N32": bool(32 in actual_counts),
        "includes_N40": bool(40 in actual_counts),
        "exact_overlap_counts": overlap,
        "input_path": str(RUN63_OPTION_A),
    }
    write_json(OUTPUT_DIR / "run64_input_validation_summary.json", summary)
    return summary


def rename_handoff(df: pd.DataFrame) -> pd.DataFrame:
    handoff = df.copy()
    handoff["n"] = handoff["n"].astype(int)
    handoff = handoff.sort_values(["n", "candidate_source", "surrogate_prediction"], ascending=[True, True, False]).reset_index(drop=True)
    per_n_counter: Counter[int] = Counter()
    names = []
    for _, row in handoff.iterrows():
        n = int(row["n"])
        per_n_counter[n] += 1
        names.append(f"S3R64VNR_N{n}_B{per_n_counter[n]:02d}_{short_bucket(row)}")
    handoff["handoff_strategy_name"] = names
    handoff["run_id"] = RUN_ID
    handoff["batch_option"] = BATCH_OPTION
    handoff["batch_name"] = BATCH_NAME
    handoff["original_run63_candidate_id"] = handoff.get("candidate_id", "")
    handoff["original_run63_strategy_name"] = handoff.get("strategy_name", "")
    handoff["N12_recovery_prediction"] = handoff.get("N12_recovery_score")
    handoff["N16_recovery_prediction"] = handoff.get("N16_recovery_score")
    handoff["variable_N_recovery_prediction"] = handoff.get("variable_N_recovery_score")
    handoff["N24_frozen_anchor_prediction"] = handoff.get("N24_maintenance_score")
    handoff["N40_followup_prediction"] = handoff.get("N40_followup_score")
    handoff["u2_primary_prediction"] = handoff.get("predicted_u2_guarded_score")
    handoff["constrained_reward_prediction"] = handoff.get("constrained_score")
    handoff["strict_penalty_guard_prediction"] = handoff.get("strict_penalty_guard_score")
    handoff["penalty_repair_prediction"] = handoff.get("penalty_repair_score")
    handoff["native_validation_N"] = True
    handoff["variable_N_recovery"] = True
    handoff["smallN_recovery"] = handoff["n"].isin([12, 16])
    handoff["frozen_anchor"] = handoff["n"].isin([24, 40])
    handoff["order_json"] = handoff["order_json"].map(order_json)
    handoff["order_compact"] = handoff["order_json"].map(compact_order)
    handoff["teacher_validated"] = False
    handoff["teacher_validation_status"] = "NOT_RUN"
    handoff["notes"] = "Run64 handoff only. Variable-N recovery anchor batch48 candidate is not teacher-validated."

    columns = [
        "run_id", "batch_option", "batch_name", "n", "handoff_strategy_name",
        "original_run63_candidate_id", "original_run63_strategy_name",
        "candidate_source", "generation_method", "selection_bucket", "priority_role",
        "surrogate_prediction", "N12_recovery_prediction", "N16_recovery_prediction",
        "variable_N_recovery_prediction", "N24_frozen_anchor_prediction", "N40_followup_prediction",
        "u2_primary_prediction", "constrained_reward_prediction", "strict_penalty_guard_prediction",
        "penalty_repair_prediction", "gnn_reward_prediction", "graph_pointer_policy_score",
        "hybrid_score", "uncertainty_score", "gnn_vs_surrogate_disagreement", "novelty_distance",
        "nearest_existing_teacher_strategy", "native_validation_N", "variable_N_recovery",
        "smallN_recovery", "frozen_anchor", "order_json", "order_compact", "order_hash",
        "teacher_validated", "teacher_validation_status", "notes",
    ]
    for col in columns:
        if col not in handoff.columns:
            handoff[col] = ""
    return handoff[columns]


def row_metadata(row: pd.Series) -> dict[str, Any]:
    return {
        "surrogate_prediction": row.get("surrogate_prediction"),
        "N12_recovery_prediction": row.get("N12_recovery_prediction"),
        "N16_recovery_prediction": row.get("N16_recovery_prediction"),
        "variable_N_recovery_prediction": row.get("variable_N_recovery_prediction"),
        "N24_frozen_anchor_prediction": row.get("N24_frozen_anchor_prediction"),
        "N40_followup_prediction": row.get("N40_followup_prediction"),
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
            "original_run63_candidate_id": row["original_run63_candidate_id"],
            "candidate_source": row["candidate_source"],
            "generation_method": row["generation_method"],
            "selection_bucket": row["selection_bucket"],
            "priority_role": row["priority_role"],
            "metadata": row_metadata(row),
            "native_validation_N": True,
            "variable_N_recovery": True,
            "smallN_recovery": bool(row["smallN_recovery"]),
            "frozen_anchor": bool(row["frozen_anchor"]),
            "scan_order": parse_order(row["order_json"]),
            "order_hash": row["order_hash"],
            "teacher_validated": False,
            "teacher_validation_status": "NOT_RUN",
            "notes": "Run64 handoff only. Variable-N recovery anchor batch48 candidate is not teacher-validated.",
        }
        write_json(SCAN_DIR / f"scan_order_{row['handoff_strategy_name']}.json", payload)


def write_future_templates(handoff: pd.DataFrame) -> None:
    cae_root = ROOT / "cae_model" / BATCH_NAME
    rows = []
    commands = [
        "# TEMPLATE ONLY - not ready to run until CAE/INP generation has completed and passed checks.",
        "# Do not execute this file during Run64.",
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
    source_comp = handoff["candidate_source"].value_counts().to_dict()
    bucket_comp = handoff["selection_bucket"].value_counts().to_dict()
    summary = {
        "timestamp": now_iso(),
        "headline": "Run64 packages the recommended full variable-N recovery follow-up after N24/N40 evidence freeze: N12/N16 recovery is primary while N24/N40 mature anchors are preserved.",
        "total_count": int(len(handoff)),
        "per_N_counts": counts(handoff),
        "includes_N12": True,
        "includes_N16": True,
        "includes_N24": True,
        "includes_N32": False,
        "includes_N40": True,
        "candidate_source_composition": source_comp,
        "selection_bucket_composition": bucket_comp,
        "mean_surrogate_prediction_per_N": numeric_summary(handoff, "surrogate_prediction"),
        "mean_N12_recovery_prediction_per_N": numeric_summary(handoff, "N12_recovery_prediction"),
        "mean_N16_recovery_prediction_per_N": numeric_summary(handoff, "N16_recovery_prediction"),
        "mean_variable_N_recovery_prediction_per_N": numeric_summary(handoff, "variable_N_recovery_prediction"),
        "mean_N24_frozen_anchor_prediction_per_N": numeric_summary(handoff, "N24_frozen_anchor_prediction"),
        "mean_N40_followup_prediction_per_N": numeric_summary(handoff, "N40_followup_prediction"),
        "mean_u2_primary_prediction_per_N": numeric_summary(handoff, "u2_primary_prediction"),
        "mean_constrained_reward_prediction_per_N": numeric_summary(handoff, "constrained_reward_prediction"),
        "mean_strict_penalty_guard_prediction_per_N": numeric_summary(handoff, "strict_penalty_guard_prediction"),
        "mean_penalty_repair_prediction_per_N": numeric_summary(handoff, "penalty_repair_prediction"),
        "mean_gnn_reward_prediction_per_N": numeric_summary(handoff, "gnn_reward_prediction"),
        "mean_hybrid_score_per_N": numeric_summary(handoff, "hybrid_score"),
        "mean_disagreement_per_N": numeric_summary(handoff, "gnn_vs_surrogate_disagreement"),
        "mean_novelty_distance_per_N": numeric_summary(handoff, "novelty_distance"),
        "expected_abaqus_cost": {"total_jobs": 48, "N12": 12, "N16": 12, "N24": 8, "N40": 16},
        "exact_overlap_status": validation["exact_overlap_counts"],
        "not_teacher_validated_until_future_abaqus_validation": True,
        "run64_created_cae_inp_files": False,
    }
    write_json(REVIEW_JSON, summary)
    review_rows = []
    for key, value in summary.items():
        review_rows.append({"metric": key, "value": json.dumps(clean_json(value), sort_keys=True) if isinstance(value, (dict, list)) else value})
    write_csv(REVIEW_CSV, pd.DataFrame(review_rows))
    REVIEW_MD.write_text(
        "# Variable-N Recovery Anchor Batch48 Review Summary\n\n"
        f"{summary['headline']}\n\n"
        f"- Total: {summary['total_count']}\n"
        f"- Per-N counts: {summary['per_N_counts']}\n"
        "- Includes N12/N16/N24/N40; excludes N32.\n"
        "- This batch repairs N12/N16 under-sampling while preserving mature N24/N40 anchors.\n"
        "- This is not teacher-validated until future Abaqus validation.\n"
        "- Run64 did not create CAE/INP files.\n\n"
        "## Exact Overlap Status\n"
        + "\n".join(f"- {k}: {v}" for k, v in summary["exact_overlap_status"].items())
        + "\n",
        encoding="utf-8",
    )
    return summary


def write_claim_boundary() -> None:
    safe = [
        "Run64 packages selected Run63 Option A variable-N recovery anchor batch48 candidates for human review and future CAE generation.",
        "The selected batch contains N12=12, N16=12, N24=8, and N40=16.",
        "No N32 candidates are included.",
        "The batch is designed to repair N12/N16 under-sampling while preserving mature N24/N40 anchors.",
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
        "Do not claim batch48 will improve teacher metrics before validation.",
    ]
    CLAIM_BOUNDARY_MD.write_text("# Run64 Claim Boundary\n\n## Safe claims\n" + "\n".join(f"- {x}" for x in safe) + "\n\n## Unsafe claims\n" + "\n".join(f"- {x}" for x in unsafe) + "\n", encoding="utf-8")
    write_json(CLAIM_BOUNDARY_JSON, {"verdict": "RUN64_HANDOFF_ONLY_VARIABLE_N_RECOVERY_ANCHOR_BATCH48_NO_TEACHER_VALIDATION", "safe_claims": safe, "unsafe_claims": unsafe})


def write_report(validation: dict[str, Any], review: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(f"""# Stage 3 Run 64 - Run63 Variable-N Recovery Anchor Batch48 Handoff Package

## 1. Purpose
Run64 packages the selected Run63 Option A variable-N recovery anchor batch48 for human review and future CAE generation. It is handoff packaging only.

## 2. Inputs
- Selected Run63 Option A: `{RUN63_OPTION_A}`
- Run63 candidate pool: `{RUN63_POOL}`
- Run63 comparison summary: `{RUN63_COMPARISON_SUMMARY}`
- Run63 evidence freeze: `{RUN63_EVIDENCE_FREEZE}`
- Native combined432 RL-ready dataset: `{COMBINED432_READY}`

## 3. Selected Option A
Selected batch: `run63_variable_N_recovery_anchor_batch48`.

Counts: `{validation['per_N_counts']}`.

## 4. Why Option A Was Selected
Option A directly addresses the full variable-N claim boundary by adding N12/N16 recovery evidence while retaining mature N24/N40 anchors.

## 5. Why N12/N16 Recovery Is Now Primary
Run63 froze N24/N40 evidence at dense native teacher counts, while N12/N16 remain at 36 rows each. N12/N16 recovery is therefore the active bottleneck for stronger variable-N claims.

## 6. Why N24/N40 Are Anchors Rather Than Exploitation Targets
N24/N40 are included as anchors to maintain continuity with mature evidence, not as the main exploitation target for this handoff.

## 7. Validation Status
Verdict: `{validation['verdict']}`.

No N32 rows are included, and exact overlap counts are `{validation['exact_overlap_counts']}`.

## 8. Stable Naming Convention
`S3R64VNR_N{{N}}_B{{index:02d}}_{{short_bucket_or_family}}`

## 9. Candidate-Order Handoff Package
Candidate handoff CSV: `{HANDOFF_CSV}`.

## 10. Per-Candidate Scan-Order JSON Outputs
Scan-order JSON directory: `{SCAN_DIR}`.

## 11. Future CAE Handoff Template
Future CAE manifest template: `{CAE_TEMPLATE}`.

Expected future case root: `{ROOT / 'cae_model' / BATCH_NAME}`.

## 12. Future abqjobpilot Command Template
Future command template: `{ABQ_TEMPLATE}`.

This command file is a template only. INPs do not exist yet and the file is not ready to run until CAE/INP generation has completed and passed checks.

## 13. Review Summary
{review['headline']}

## 14. Claim Boundary
Verdict: `RUN64_HANDOFF_ONLY_VARIABLE_N_RECOVERY_ANCHOR_BATCH48_NO_TEACHER_VALIDATION`.

## 15. Output Files
- Validation summary: `{OUTPUT_DIR / 'run64_input_validation_summary.json'}`
- Handoff CSV: `{HANDOFF_CSV}`
- Scan orders: `{SCAN_DIR}`
- Future CAE template: `{CAE_TEMPLATE}`
- Future abqjobpilot template: `{ABQ_TEMPLATE}`
- Review summary: `{REVIEW_MD}`
- Claim boundary: `{CLAIM_BOUNDARY_MD}`
- Manifest: `{MANIFEST_PATH}`

## 16. Recommended Run65
CAE module should generate CAE/INP/JNL for selected Run64 variable-N recovery anchor batch48 only. Do not run solver. Do not execute abqjobpilot. Do not generate Run63 Option B or Option C unless explicitly selected later.
""", encoding="utf-8")


def write_manifest(validation: dict[str, Any], review: dict[str, Any]) -> None:
    output_files = [
        OUTPUT_DIR / "run64_input_validation_summary.json",
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
            str(RUN63_OPTION_A),
            str(RUN63_POOL),
            str(RUN63_COMPARISON),
            str(RUN63_COMPARISON_SUMMARY),
            str(RUN63_EVIDENCE_FREEZE),
            str(RUN63_REPORT),
            str(RUN63_MANIFEST),
            str(COMBINED432_READY),
        ],
        "output_files": [str(p) for p in output_files],
        "selected_batch": "run63_variable_N_recovery_anchor_batch48",
        "batch_name": BATCH_NAME,
        "batch48_count": 48,
        "per_N_counts": validation["per_N_counts"],
        "includes_N12": True,
        "includes_N16": True,
        "includes_N24": True,
        "includes_N32": False,
        "includes_N40": True,
        "variable_N_recovery": True,
        "smallN_recovery": True,
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

    selected = read_csv(RUN63_OPTION_A)
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
        "selected_batch": "run63_variable_N_recovery_anchor_batch48",
        "counts": counts(handoff),
        "handoff_csv": str(HANDOFF_CSV),
        "scan_order_json_dir": str(SCAN_DIR),
        "report": str(REPORT_PATH),
        "manifest": str(MANIFEST_PATH),
    }, indent=2))


if __name__ == "__main__":
    main()
