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
RUN_ID = "run_59_run58_N40_focused_calibrated_penalty_repair_batch40_handoff_package"
RUN_NAME = "run58 N40-focused calibrated penalty-repair batch40 handoff package"
SCRIPT_PATH = ROOT / "scripts" / "stage3" / "run_59_create_run58_N40_focused_calibrated_penalty_repair_batch40_handoff_package.py"

RUN58_DIR = ROOT / "outputs" / "stage3_run_58_combined392_model_update_N24_N40_evidence_freeze_and_N40_focused_candidate_generation"
RUN58_OPTION_A = RUN58_DIR / "run58_N40_focused_calibrated_penalty_repair_batch32_candidate_orders.csv"
RUN58_POOL = RUN58_DIR / "run58_candidate_pool_scored.csv"
RUN58_COMPARISON = RUN58_DIR / "run58_batch_options_comparison_to_previous.csv"
RUN58_COMPARISON_SUMMARY = RUN58_DIR / "run58_batch_options_comparison_summary.json"
RUN58_EVIDENCE_FREEZE = RUN58_DIR / "n24_n40_active_learning_rl_evidence_freeze.md"
RUN58_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_58_combined392_model_update_N24_N40_evidence_freeze_and_N40_focused_candidate_generation" / "RUN_58_COMBINED392_MODEL_UPDATE_N24_N40_EVIDENCE_FREEZE_AND_N40_FOCUSED_CANDIDATE_GENERATION_REPORT.md"
RUN58_MANIFEST = ROOT / "artifacts" / "manifests" / "stage3_run_58_manifest.json"
COMBINED392_READY = ROOT / "outputs" / "stage3_run_57_calibrated_N24_N40_batch64_teacher_metrics_ingestion_and_combined392_ranking" / "combined392_RL_ready_dataset.csv"

RUN56_HANDOFF = ROOT / "outputs" / "stage3_run_54_run53_calibrated_N24_N40_batch64_handoff_package" / "stage3_run54_calibrated_N24_N40_batch64_candidate_orders.csv"
RUN51_HANDOFF = ROOT / "outputs" / "stage3_run_49_run48_stricter_constrained_N24_N40_batch32_handoff_package" / "stage3_run49_stricter_constrained_N24_N40_batch32_candidate_orders.csv"
RUN46_HANDOFF = ROOT / "outputs" / "stage3_run_44_run43_constrained_N24_N40_batch32_handoff_package" / "stage3_run44_constrained_N24_N40_batch32_candidate_orders.csv"
RUN41_HANDOFF = ROOT / "outputs" / "stage3_run_39_run38_native_N24_N40_focused_batch60_handoff_package" / "stage3_run39_native_N24_N40_focused_batch60_candidate_orders.csv"
RUN36_HANDOFF = ROOT / "outputs" / "stage3_run_34_run33_N32_informed_native_batch32_handoff_package" / "stage3_run34_N32_informed_native_batch32_candidate_orders.csv"
RUN27_HANDOFF = ROOT / "outputs" / "stage3_run_24_run23_shortlist64_active_learning_handoff_package" / "stage3_run24_shortlist64_candidate_orders.csv"
RUN31_OLD = ROOT / "outputs" / "stage3_run_30_run29_hybrid_policy_batch32_handoff_package" / "stage3_run30_hybrid_policy_batch32_candidate_orders.csv"

OUTPUT_DIR = ROOT / "outputs" / "stage3_run_59_run58_N40_focused_calibrated_penalty_repair_batch40_handoff_package"
SCAN_DIR = OUTPUT_DIR / "scan_orders"
REPORT_DIR = ROOT / "docs" / "stage3" / "runs" / "run_59_run58_N40_focused_calibrated_penalty_repair_batch40_handoff_package"
REPORT_PATH = REPORT_DIR / "RUN_59_RUN58_N40_FOCUSED_CALIBRATED_PENALTY_REPAIR_BATCH40_HANDOFF_PACKAGE_REPORT.md"
MANIFEST_PATH = ROOT / "artifacts" / "manifests" / "stage3_run_59_manifest.json"
CLAIM_BOUNDARY_MD = OUTPUT_DIR / "run59_claim_boundary.md"
CLAIM_BOUNDARY_JSON = OUTPUT_DIR / "run59_claim_boundary.json"

BATCH_NAME = "stage3_run59_N40_focused_calibrated_penalty_repair_batch40_v01"
BATCH_OPTION = "custom_N40_focused_calibrated_penalty_repair_batch40"
EXPECTED_COUNTS = {24: 16, 40: 24}

PRE_HANDOFF = OUTPUT_DIR / "run59_custom_N40_focused_calibrated_penalty_repair_batch40_candidate_orders_PRE_HANDOFF.csv"
HANDOFF_CSV = OUTPUT_DIR / "stage3_run59_N40_focused_calibrated_penalty_repair_batch40_candidate_orders.csv"
CAE_TEMPLATE = OUTPUT_DIR / "stage3_run59_N40_focused_calibrated_penalty_repair_batch40_future_cae_handoff_manifest_TEMPLATE.csv"
ABQ_TEMPLATE = OUTPUT_DIR / "stage3_run59_N40_focused_calibrated_penalty_repair_batch40_abqjobpilot_commands_TEMPLATE.txt"
REVIEW_CSV = OUTPUT_DIR / "N40_focused_calibrated_penalty_repair_batch40_review_summary.csv"
REVIEW_JSON = OUTPUT_DIR / "N40_focused_calibrated_penalty_repair_batch40_review_summary.json"
REVIEW_MD = OUTPUT_DIR / "N40_focused_calibrated_penalty_repair_batch40_review_summary.md"


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


def safe_fragment(text: Any, max_len: int = 24) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", str(text).strip().lower()).strip("_")
    return (cleaned or "candidate")[:max_len]


def short_bucket(row: pd.Series) -> str:
    source = str(row.get("candidate_source", "") or row.get("selection_bucket", "candidate"))
    mapping = {
        "N24_u2_reward_maintenance": "n24_maint",
        "N24_penalty_repair_diagnostic": "n24_penalty",
        "N24_uncertainty_calibration": "n24_uncertainty",
        "N24_diversity_coverage": "n24_diversity",
        "N24_sentinel_control": "n24_sentinel",
        "N40_u2_reward_retention_top": "n40_u2ret_top",
        "N40_u2_reward_retention_local_repair": "n40_u2ret_repair",
        "N40_penalty_repair_top": "n40_penalty_top",
        "N40_penalty_repair_local_search": "n40_penalty_repair",
        "N40_two_stage_penalty_repair": "n40_two_stage",
        "N40_median_guard_repair": "n40_median",
        "N40_no_penalty_worse_than_median": "n40_no_median",
        "N40_PEEQ_repair_candidates": "n40_peeq_repair",
        "N40_SurfaceT_repair_candidates": "n40_surface_repair",
        "N40_Mises_repair_candidates": "n40_mises_repair",
        "N40_uncertainty_calibration": "n40_uncertainty",
        "N40_diversity_coverage": "n40_diversity",
        "N40_sentinel_control": "n40_sentinel",
    }
    return safe_fragment(mapping.get(source, source), 22)


def load_hashes(path: Path) -> set[str]:
    if not path.exists():
        return set()
    df = read_csv(path)
    if "order_hash" not in df.columns:
        return set()
    return {str(x) for x in df["order_hash"].dropna().astype(str) if str(x)}


def previous_hashes() -> dict[str, set[str]]:
    combined = load_hashes(COMBINED392_READY)
    return {
        "combined392": combined,
        "run56": load_hashes(RUN56_HANDOFF),
        "run51": load_hashes(RUN51_HANDOFF),
        "run46": load_hashes(RUN46_HANDOFF),
        "run41": load_hashes(RUN41_HANDOFF),
        "run36": load_hashes(RUN36_HANDOFF),
        "run27": load_hashes(RUN27_HANDOFF),
        "superseded_run31": load_hashes(RUN31_OLD),
    }


def build_custom_batch() -> tuple[pd.DataFrame, dict[str, Any]]:
    option_a = read_csv(RUN58_OPTION_A)
    pool = read_csv(RUN58_POOL)
    option_a["source_from_original_option_A"] = True
    option_a["added_as_extra_N24"] = False

    n40 = option_a[option_a["n"].astype(int) == 40].copy()
    n24_original = option_a[option_a["n"].astype(int) == 24].copy()
    exclude = set(option_a["order_hash"].astype(str))
    prev = previous_hashes()
    prior_exclude = set().union(*prev.values()) if prev else set()

    n24_pool = pool[pool["n"].astype(int) == 24].copy()
    n24_pool = n24_pool[~n24_pool["order_hash"].astype(str).isin(exclude | prior_exclude)].copy()
    source_priority = {
        "N24_u2_reward_maintenance": 1,
        "N24_penalty_repair_diagnostic": 2,
        "N24_uncertainty_calibration": 3,
        "N24_diversity_coverage": 4,
        "N24_sentinel_control": 5,
    }
    n24_pool["source_priority"] = n24_pool["candidate_source"].map(lambda x: source_priority.get(str(x), 99))
    n24_pool["selection_score"] = (
        pd.to_numeric(n24_pool.get("N24_maintenance_score", 0), errors="coerce").fillna(0) * 0.42
        + pd.to_numeric(n24_pool.get("penalty_repair_score", 0), errors="coerce").fillna(0) * 0.28
        + pd.to_numeric(n24_pool.get("novelty_distance", 0), errors="coerce").fillna(0) * 0.20
        + pd.to_numeric(n24_pool.get("uncertainty_score", 0), errors="coerce").fillna(0) * 0.10
    )
    selected = []
    used_hashes = set(exclude)
    quota_sources = [
        "N24_u2_reward_maintenance",
        "N24_penalty_repair_diagnostic",
        "N24_uncertainty_calibration",
        "N24_diversity_coverage",
        "N24_sentinel_control",
    ]
    for source in quota_sources:
        if len(selected) >= 8:
            break
        rows = n24_pool[(n24_pool["candidate_source"].astype(str) == source) & (~n24_pool["order_hash"].astype(str).isin(used_hashes))]
        rows = rows.sort_values(["selection_score", "novelty_distance"], ascending=[False, False])
        take_n = 2 if source in {"N24_u2_reward_maintenance", "N24_penalty_repair_diagnostic"} else 1
        for _, row in rows.head(take_n).iterrows():
            if len(selected) < 8:
                selected.append(row.to_dict())
                used_hashes.add(str(row["order_hash"]))
    if len(selected) < 8:
        rows = n24_pool[~n24_pool["order_hash"].astype(str).isin(used_hashes)].sort_values(["selection_score", "source_priority", "novelty_distance"], ascending=[False, True, False])
        for _, row in rows.head(8 - len(selected)).iterrows():
            selected.append(row.to_dict())
            used_hashes.add(str(row["order_hash"]))

    extra = pd.DataFrame(selected)
    if len(extra) != 8:
        raise RuntimeError(f"could not select 8 extra N24 candidates, selected {len(extra)}")
    extra["source_from_original_option_A"] = False
    extra["added_as_extra_N24"] = True
    custom = pd.concat([n24_original, extra, n40], ignore_index=True, sort=False)
    custom = custom.drop(columns=[c for c in ["source_priority", "selection_score"] if c in custom.columns])
    custom["custom_batch_source"] = custom["source_from_original_option_A"].map(lambda x: "run58_option_A" if bool(x) else "run58_candidate_pool_extra_N24")
    write_csv(PRE_HANDOFF, custom)
    return custom, {
        "original_option_A_rows_preserved": int(len(option_a)),
        "original_option_A_N24_preserved": int(len(n24_original)),
        "original_option_A_N40_preserved": int(len(n40)),
        "extra_N24_rows_added": int(len(extra)),
        "extra_N24_source_counts": extra["candidate_source"].value_counts().to_dict(),
    }


def validate_input(df: pd.DataFrame, build_summary: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    per_n = counts(df) if "n" in df.columns else {}
    if len(df) != 40:
        errors.append(f"row count is {len(df)}, expected 40")
    if per_n != EXPECTED_COUNTS:
        errors.append(f"per-N counts are {per_n}, expected {EXPECTED_COUNTS}")
    if set(df["n"].astype(int)) - {24, 40}:
        errors.append("selected batch contains N values outside N24/N40")
    for col in ["candidate_id", "n", "order_json", "order_hash"]:
        if col not in df.columns:
            errors.append(f"missing required column {col}")
    bad_orders = []
    for _, row in df.iterrows():
        if not valid_order(row.get("order_json", row.get("scan_order", "")), int(row["n"])):
            bad_orders.append(row.get("candidate_id", row.get("strategy_name", "UNKNOWN")))
    if bad_orders:
        errors.append(f"invalid scan orders: {bad_orders[:5]}")
    if "order_hash" in df.columns and df.duplicated(["n", "order_hash"]).any():
        errors.append("duplicate order_hash within N")
    if "candidate_id" in df.columns and df["candidate_id"].duplicated().any():
        errors.append("duplicate candidate_id")
    if build_summary.get("original_option_A_rows_preserved") != 32:
        errors.append("did not preserve all 32 original Option A rows")
    if build_summary.get("extra_N24_rows_added") != 8:
        errors.append("did not add exactly 8 extra N24 rows")

    overlap_status = {}
    hashes = set(df["order_hash"].astype(str))
    for name, phashes in previous_hashes().items():
        overlap_status[name] = int(len(hashes & phashes))
        if overlap_status[name] != 0:
            errors.append(f"overlap with {name} is nonzero: {overlap_status[name]}")

    verdict = "PASS_RUN59_N40_FOCUSED_CALIBRATED_PENALTY_REPAIR_BATCH40_INPUT_READY" if not errors else "FAIL_RUN59_INPUT_VALIDATION"
    return {
        "timestamp": now_iso(),
        "verdict": verdict,
        "errors": errors,
        "pre_handoff_path": str(PRE_HANDOFF),
        "row_count": int(len(df)),
        "per_N_counts": per_n,
        "contains_N12": bool((df["n"].astype(int) == 12).any()),
        "contains_N16": bool((df["n"].astype(int) == 16).any()),
        "contains_N32": bool((df["n"].astype(int) == 32).any()),
        "selected_batch_is_custom_run58_batch40": True,
        "build_summary": build_summary,
        "overlap_status": overlap_status,
    }


def make_handoff(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for n, group in df.sort_values(["n", "source_from_original_option_A", "candidate_source", "candidate_id"], ascending=[True, False, True, True]).groupby("n"):
        for i, (_, row) in enumerate(group.iterrows(), 1):
            handoff_name = f"S3R59N40PR40_N{int(n)}_B{i:02d}_{short_bucket(row)}"
            order_value = row.get("order_json", row.get("scan_order", ""))
            out = {
                "run_id": RUN_ID,
                "batch_option": BATCH_OPTION,
                "batch_name": BATCH_NAME,
                "n": int(n),
                "handoff_strategy_name": handoff_name,
                "original_run58_candidate_id": row.get("candidate_id", ""),
                "original_run58_strategy_name": row.get("strategy_name", ""),
                "candidate_source": row.get("candidate_source", ""),
                "generation_method": row.get("generation_method", ""),
                "selection_bucket": row.get("selection_bucket", ""),
                "priority_role": row.get("priority_role", ""),
                "source_from_original_option_A": bool(row.get("source_from_original_option_A", False)),
                "added_as_extra_N24": bool(row.get("added_as_extra_N24", False)),
                "native_validation_N": True,
                "N40_focused": True,
                "N24_maintenance": int(n) == 24,
                "calibrated_penalty_repair": True,
                "order_json": order_json(order_value),
                "order_compact": compact_order(order_value),
                "order_hash": row.get("order_hash", ""),
                "teacher_validated": False,
                "teacher_validation_status": "NOT_RUN",
                "notes": "Run59 handoff only. Custom N40-focused calibrated penalty-repair batch40 candidate is not teacher-validated.",
            }
            passthrough = {
                "surrogate_prediction": "surrogate_prediction",
                "N40_u2_reward_retention_prediction": "N40_u2_reward_retention_score",
                "N40_penalty_repair_prediction": "N40_penalty_repair_score",
                "N40_two_stage_penalty_repair_prediction": "two_stage_penalty_repair_score",
                "N40_median_guard_prediction": "no_penalty_worse_than_median_score",
                "N24_maintenance_prediction": "N24_maintenance_score",
                "u2_primary_prediction": "predicted_u2_guarded_score",
                "constrained_reward_prediction": "constrained_score",
                "strict_penalty_guard_prediction": "strict_penalty_guard_score",
                "penalty_repair_prediction": "penalty_repair_score",
                "gnn_reward_prediction": "gnn_reward_prediction",
                "graph_pointer_policy_score": "graph_pointer_policy_score",
                "hybrid_score": "hybrid_score",
                "uncertainty_score": "uncertainty_score",
                "gnn_vs_surrogate_disagreement": "gnn_vs_surrogate_disagreement",
                "novelty_distance": "novelty_distance",
                "nearest_existing_teacher_strategy": "nearest_existing_teacher_strategy",
            }
            for out_col, in_col in passthrough.items():
                out[out_col] = row.get(in_col, "")
            rows.append(out)
    columns = [
        "run_id", "batch_option", "batch_name", "n", "handoff_strategy_name",
        "original_run58_candidate_id", "original_run58_strategy_name",
        "candidate_source", "generation_method", "selection_bucket", "priority_role",
        "source_from_original_option_A", "added_as_extra_N24",
        "surrogate_prediction", "N40_u2_reward_retention_prediction", "N40_penalty_repair_prediction",
        "N40_two_stage_penalty_repair_prediction", "N40_median_guard_prediction", "N24_maintenance_prediction",
        "u2_primary_prediction", "constrained_reward_prediction", "strict_penalty_guard_prediction",
        "penalty_repair_prediction", "gnn_reward_prediction", "graph_pointer_policy_score", "hybrid_score",
        "uncertainty_score", "gnn_vs_surrogate_disagreement", "novelty_distance", "nearest_existing_teacher_strategy",
        "native_validation_N", "N40_focused", "N24_maintenance", "calibrated_penalty_repair",
        "order_json", "order_compact", "order_hash", "teacher_validated", "teacher_validation_status", "notes",
    ]
    return pd.DataFrame(rows)[columns]


def write_scan_jsons(handoff: pd.DataFrame) -> None:
    SCAN_DIR.mkdir(parents=True, exist_ok=True)
    for _, row in handoff.iterrows():
        payload = row.to_dict()
        payload["scan_order"] = parse_order(row["order_json"])
        payload["teacher_validated"] = False
        payload["teacher_validation_status"] = "NOT_RUN"
        payload["notes"] = "Run59 handoff only. Custom N40-focused calibrated penalty-repair batch40 candidate is not teacher-validated."
        write_json(SCAN_DIR / f"scan_order_{row['handoff_strategy_name']}.json", payload)


def write_future_templates(handoff: pd.DataFrame) -> None:
    root = ROOT / "cae_model" / BATCH_NAME
    manifest_rows = []
    command_lines = [
        "# TEMPLATE ONLY - not ready to run until CAE/INP generation has completed and passed checks.",
        "# Do not execute this file during Run59.",
    ]
    for _, row in handoff.iterrows():
        case_dir = root / f"N{int(row['n'])}{row['handoff_strategy_name']}"
        inp = case_dir / f"J2D_{row['handoff_strategy_name']}.inp"
        manifest_rows.append({
            "batch_name": BATCH_NAME,
            "n": int(row["n"]),
            "handoff_strategy_name": row["handoff_strategy_name"],
            "expected_future_case_root": str(root),
            "expected_future_case_dir": str(case_dir),
            "expected_future_job_name": f"J2D_{row['handoff_strategy_name']}",
            "expected_future_inp_path": str(inp),
            "cae_inp_generated": False,
            "teacher_validated": False,
        })
        command_lines.append(f'enqueue --inp "{inp}" --cpus 14 --batch {BATCH_NAME} --strategy {row["handoff_strategy_name"]}')
    write_csv(CAE_TEMPLATE, pd.DataFrame(manifest_rows))
    ABQ_TEMPLATE.write_text("\n".join(command_lines) + "\n", encoding="utf-8")


def numeric_summary(df: pd.DataFrame, column: str) -> dict[int, float | None]:
    if column not in df.columns:
        return {}
    out = {}
    for n, group in df.groupby("n"):
        values = pd.to_numeric(group[column], errors="coerce").dropna()
        out[int(n)] = float(values.mean()) if len(values) else None
    return out


def write_review(handoff: pd.DataFrame, validation: dict[str, Any]) -> dict[str, Any]:
    source_counts = handoff["candidate_source"].value_counts().to_dict()
    bucket_counts = handoff["selection_bucket"].value_counts().to_dict()
    overlap_status = validation["overlap_status"]
    headline = (
        "Custom batch40 preserves all 32 Run58 Option A candidates and adds 8 complementary N24 candidates, "
        "yielding N24=16 and N40=24 with zero overlap against combined392 and prior tracked batches."
    )
    summary = {
        "headline": headline,
        "total_count": int(len(handoff)),
        "per_N_counts": counts(handoff),
        "only_N24_N40": set(handoff["n"].astype(int)) == {24, 40},
        "contains_N12": False,
        "contains_N16": False,
        "contains_N32": False,
        "original_option_A_rows_preserved": int(handoff["source_from_original_option_A"].sum()),
        "extra_N24_rows_added": int(handoff["added_as_extra_N24"].sum()),
        "candidate_source_composition": source_counts,
        "selection_bucket_composition": bucket_counts,
        "mean_surrogate_prediction_by_N": numeric_summary(handoff, "surrogate_prediction"),
        "mean_N40_u2_reward_retention_prediction_by_N": numeric_summary(handoff, "N40_u2_reward_retention_prediction"),
        "mean_N40_penalty_repair_prediction_by_N": numeric_summary(handoff, "N40_penalty_repair_prediction"),
        "mean_N40_two_stage_repair_prediction_by_N": numeric_summary(handoff, "N40_two_stage_penalty_repair_prediction"),
        "mean_N40_median_guard_prediction_by_N": numeric_summary(handoff, "N40_median_guard_prediction"),
        "mean_N24_maintenance_prediction_by_N": numeric_summary(handoff, "N24_maintenance_prediction"),
        "mean_predicted_peeq_penalty_by_N": {},
        "mean_predicted_surfaceT_penalty_by_N": {},
        "mean_predicted_mises_penalty_by_N": {},
        "mean_gnn_reward_prediction_by_N": numeric_summary(handoff, "gnn_reward_prediction"),
        "mean_hybrid_score_by_N": numeric_summary(handoff, "hybrid_score"),
        "mean_disagreement_by_N": numeric_summary(handoff, "gnn_vs_surrogate_disagreement"),
        "mean_novelty_distance_by_N": numeric_summary(handoff, "novelty_distance"),
        "expected_abaqus_cost": "40 jobs total: 16 N24 and 24 N40",
        "exact_overlap_status": overlap_status,
        "teacher_validated": False,
        "cae_inp_generated": False,
    }
    write_csv(REVIEW_CSV, pd.DataFrame([{
        "total_count": summary["total_count"],
        "per_N_counts": json.dumps(summary["per_N_counts"]),
        "original_option_A_rows_preserved": summary["original_option_A_rows_preserved"],
        "extra_N24_rows_added": summary["extra_N24_rows_added"],
        "candidate_source_composition": json.dumps(source_counts),
        "selection_bucket_composition": json.dumps(bucket_counts),
        "expected_abaqus_cost": summary["expected_abaqus_cost"],
        "headline": headline,
    }]))
    write_json(REVIEW_JSON, summary)
    REVIEW_MD.write_text(
        "# N40-Focused Calibrated Penalty-Repair Batch40 Review Summary\n\n"
        f"{headline}\n\n"
        f"- Total: {summary['total_count']}\n"
        f"- Per-N counts: {summary['per_N_counts']}\n"
        "- Included N values: N24 and N40 only\n"
        "- Excluded N values: N12, N16, N32\n"
        f"- Original Option A rows preserved: {summary['original_option_A_rows_preserved']}\n"
        f"- Extra N24 rows added: {summary['extra_N24_rows_added']}\n"
        f"- Expected Abaqus cost: {summary['expected_abaqus_cost']}\n\n"
        "This batch is a custom N40-focused follow-up after Run56 created new N40 U2/reward-family records. "
        "It tests N40 calibrated penalty repair while expanding N24 maintenance/diagnostic coverage from 8 to 16 cases. "
        "It is not teacher-validated until future Abaqus validation, and Run59 did not create CAE/INP files.\n",
        encoding="utf-8",
    )
    return summary


def write_claim_boundary() -> None:
    safe = [
        "Run59 packages a custom Run58-derived N40-focused calibrated penalty-repair batch40 for human review and future CAE generation.",
        "The selected batch contains N24=16 and N40=24.",
        "No N12, N16, or N32 candidates are included.",
        "The batch is designed to test N40 U2/reward retention and penalty repair while increasing N24 maintenance/diagnostic coverage.",
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
        "Do not claim batch40 will improve teacher metrics before validation.",
    ]
    CLAIM_BOUNDARY_MD.write_text("# Run59 Claim Boundary\n\n## Safe claims\n" + "\n".join(f"- {x}" for x in safe) + "\n\n## Unsafe claims\n" + "\n".join(f"- {x}" for x in unsafe) + "\n", encoding="utf-8")
    write_json(CLAIM_BOUNDARY_JSON, {"verdict": "RUN59_HANDOFF_ONLY_CUSTOM_N40_FOCUSED_BATCH40_NO_TEACHER_VALIDATION", "safe_claims": safe, "unsafe_claims": unsafe})


def write_report(validation: dict[str, Any], review: dict[str, Any]) -> None:
    REPORT_PATH.write_text(f"""# Stage 3 Run 59 - Run58 N40-Focused Calibrated Penalty-Repair Batch40 Handoff Package

## 1. Purpose
Run59 creates a handoff package for a custom Run58-derived N40-focused calibrated penalty-repair batch40. It is handoff packaging only.

## 2. Inputs
- Run58 Option A: `{RUN58_OPTION_A}`
- Run58 candidate pool: `{RUN58_POOL}`
- Run58 evidence freeze: `{RUN58_EVIDENCE_FREEZE}`

## 3. Custom Selected Batch40
The custom batch contains N24=16 and N40=24, for 40 total candidates. It preserves all valid Run58 Option A rows and adds 8 extra N24 candidates from the existing Run58 candidate pool.

## 4. Why N24 Was Increased From 8 to 16
N24 has 160 native teacher rows and useful diagnostic density, but Run56 did not create new N24 combined328 bests. Increasing N24 from 8 to 16 preserves the N40-heavy plan while giving N24 enough maintenance and diagnostic coverage.

## 5. Why N40 Remains the Majority
Run56 produced the strongest recent signal in N40, including U2 and reward-family improvements. N40 remains the majority target at 24/40 cases.

## 6. Validation Status
Verdict: `{validation['verdict']}`. Counts: `{validation['per_N_counts']}`. Overlap status: `{validation['overlap_status']}`.

## 7. Stable Naming Convention
Handoff names use `S3R59N40PR40_N{{N}}_B{{index:02d}}_{{short_bucket_or_family}}`.

## 8. Candidate-Order Handoff Package
CSV: `{HANDOFF_CSV}`

## 9. Per-Candidate Scan-Order JSON Outputs
Directory: `{SCAN_DIR}`

## 10. Future CAE Handoff Template
Template: `{CAE_TEMPLATE}`. Run59 did not create CAE case directories.

## 11. Future abqjobpilot Command Template
Template: `{ABQ_TEMPLATE}`. It is not ready to execute until future CAE/INP generation has completed and passed checks.

## 12. Review Summary
{review['headline']}

## 13. Claim Boundary
Verdict: `RUN59_HANDOFF_ONLY_CUSTOM_N40_FOCUSED_BATCH40_NO_TEACHER_VALIDATION`.

## 14. Output Files
- Pre-handoff selected CSV: `{PRE_HANDOFF}`
- Handoff CSV: `{HANDOFF_CSV}`
- Review summary: `{REVIEW_MD}`
- Claim boundary: `{CLAIM_BOUNDARY_MD}`
- Manifest: `{MANIFEST_PATH}`

## 15. Recommended Run60
CAE module should generate CAE/INP/JNL for selected Run59 custom N40-focused calibrated penalty-repair batch40 only. Do not run solver. Do not execute abqjobpilot. Do not generate Run58 Option B batch64 or Option C variable-N recovery anchor batch48 unless explicitly selected later.
""", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SCAN_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    custom, build_summary = build_custom_batch()
    validation = validate_input(custom, build_summary)
    write_json(OUTPUT_DIR / "run59_input_validation_summary.json", validation)
    if not validation["verdict"].startswith("PASS"):
        raise SystemExit(validation["errors"])

    handoff = make_handoff(custom)
    write_csv(HANDOFF_CSV, handoff)
    write_scan_jsons(handoff)
    write_future_templates(handoff)
    review = write_review(handoff, validation)
    write_claim_boundary()
    write_report(validation, review)

    output_files = [
        PRE_HANDOFF,
        OUTPUT_DIR / "run59_input_validation_summary.json",
        HANDOFF_CSV,
        CAE_TEMPLATE,
        ABQ_TEMPLATE,
        REVIEW_CSV,
        REVIEW_JSON,
        REVIEW_MD,
        CLAIM_BOUNDARY_MD,
        CLAIM_BOUNDARY_JSON,
        REPORT_PATH,
        MANIFEST_PATH,
    ] + sorted(SCAN_DIR.glob("scan_order_*.json"))

    manifest = {
        "run_id": RUN_ID,
        "run_name": RUN_NAME,
        "timestamp": now_iso(),
        "branch": current_branch(),
        "script_path": str(SCRIPT_PATH),
        "input_files": [
            str(RUN58_OPTION_A),
            str(RUN58_POOL),
            str(RUN58_COMPARISON),
            str(RUN58_COMPARISON_SUMMARY),
            str(RUN58_EVIDENCE_FREEZE),
            str(RUN58_REPORT),
            str(RUN58_MANIFEST),
            str(COMBINED392_READY),
        ],
        "output_files": [str(p) for p in output_files],
        "selected_batch": "custom_run58_N40_focused_calibrated_penalty_repair_batch40",
        "batch_name": BATCH_NAME,
        "batch40_count": int(len(handoff)),
        "per_N_counts": counts(handoff),
        "includes_N12": False,
        "includes_N16": False,
        "includes_N24": True,
        "includes_N32": False,
        "includes_N40": True,
        "N40_focused": True,
        "N24_maintenance": True,
        "calibrated_penalty_repair": True,
        "original_option_A_rows_preserved": int(review["original_option_A_rows_preserved"]),
        "extra_N24_rows_added": int(review["extra_N24_rows_added"]),
        "report_path": str(REPORT_PATH),
        "claim_boundary_path": str(CLAIM_BOUNDARY_MD),
        "no_solver_run": True,
        "no_odb_opened": True,
        "no_abqjobpilot_run": True,
        "no_cae_inp_generated": True,
        "no_teacher_validation": True,
        "no_training": True,
        "no_candidate_generation_new_pool": True,
        "no_commit_or_push": True,
    }
    write_json(MANIFEST_PATH, manifest)
    print(json.dumps({
        "verdict": validation["verdict"],
        "per_N_counts": counts(handoff),
        "original_option_A_rows_preserved": review["original_option_A_rows_preserved"],
        "extra_N24_rows_added": review["extra_N24_rows_added"],
        "handoff_csv": str(HANDOFF_CSV),
        "report": str(REPORT_PATH),
    }, indent=2))


if __name__ == "__main__":
    main()
