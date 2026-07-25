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
RUN_ID = "run_54_run53_calibrated_N24_N40_batch64_handoff_package"
RUN_NAME = "run53 calibrated N24/N40 batch64 handoff package"
SCRIPT_PATH = ROOT / "scripts" / "stage3" / "run_54_create_run53_calibrated_N24_N40_batch64_handoff_package.py"

RUN53_PRIMARY = ROOT / "outputs" / "stage3_run_53_combined328_calibrated_N24_N40_batch64_candidate_generation" / "run53_calibrated_N24_N40_batch64_candidate_orders.csv"
RUN53_POOL = ROOT / "outputs" / "stage3_run_53_combined328_calibrated_N24_N40_batch64_candidate_generation" / "run53_candidate_pool_scored.csv"
RUN53_COMPARISON = ROOT / "outputs" / "stage3_run_53_combined328_calibrated_N24_N40_batch64_candidate_generation" / "run53_batch64_comparison_to_previous.csv"
RUN53_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_53_combined328_calibrated_N24_N40_batch64_candidate_generation" / "RUN_53_COMBINED328_CALIBRATED_N24_N40_BATCH64_CANDIDATE_GENERATION_REPORT.md"
RUN53_MANIFEST = ROOT / "artifacts" / "manifests" / "stage3_run_53_manifest.json"
COMBINED328_READY = ROOT / "outputs" / "stage3_run_52_stricter_constrained_N24_N40_batch32_teacher_metrics_ingestion_and_combined328_ranking" / "combined328_RL_ready_dataset.csv"

OUTPUT_DIR = ROOT / "outputs" / "stage3_run_54_run53_calibrated_N24_N40_batch64_handoff_package"
SCAN_DIR = OUTPUT_DIR / "scan_orders"
REPORT_DIR = ROOT / "docs" / "stage3" / "runs" / "run_54_run53_calibrated_N24_N40_batch64_handoff_package"
REPORT_PATH = REPORT_DIR / "RUN_54_RUN53_CALIBRATED_N24_N40_BATCH64_HANDOFF_PACKAGE_REPORT.md"
MANIFEST_PATH = ROOT / "artifacts" / "manifests" / "stage3_run_54_manifest.json"
CLAIM_BOUNDARY_MD = OUTPUT_DIR / "run54_claim_boundary.md"
CLAIM_BOUNDARY_JSON = OUTPUT_DIR / "run54_claim_boundary.json"

BATCH_NAME = "stage3_run54_calibrated_N24_N40_batch64_v01"
BATCH_OPTION = "calibrated_N24_N40_batch64"
EXPECTED_COUNTS = {24: 32, 40: 32}


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
        "strict_penalty_guard_top": "strict_guard",
        "strict_penalty_guard_local_search": "strict_guard",
        "two_stage_guarded_top": "two_stage",
        "two_stage_guarded_local_search": "two_stage",
        "no_penalty_worse_than_median": "median_guard",
        "PEEQ_repair_candidates": "peeq_repair",
        "SurfaceT_repair_candidates": "surfaceT_repair",
        "Mises_repair_candidates": "mises_repair",
        "U2_top_region_penalty_repair": "u2_penalty_repair",
        "reward_balanced_local_search": "reward_balanced",
        "constrained_surrogate_top": "constr_surrogate",
        "PEEQ_guarded_candidates": "peeq_guarded",
        "SurfaceT_guarded_candidates": "surfaceT_guarded",
        "U2_guarded_local_search": "u2_guarded",
        "hybrid_agreement": "hybrid_agree",
        "hybrid_disagreement": "hybrid_disagree",
        "uncertainty_calibration": "uncertainty",
        "diversity_coverage": "diversity",
        "graph_pointer_temperature_sample": "graph_pointer",
        "GNN_reward_local_search": "gnn_reward",
        "sentinel_control": "sentinel",
        "N24_u2_retention_top": "n24_u2ret_top",
        "N24_u2_retention_local_repair": "n24_u2ret_repair",
        "N40_strict_reward_retention_top": "n40_strict_top",
        "N40_strict_reward_local_repair": "n40_strict_repair",
        "penalty_repair_top": "penalty_repair",
        "penalty_repair_local_search": "penalty_repair",
        "two_stage_penalty_repair": "two_stage_repair",
        "median_guard_repair": "median_guard",
        "strict_guard_diverse": "strict_diverse",
    }
    return safe_fragment(mapping.get(source, source), 22)


def load_hashes(path: Path) -> set[str]:
    if not path.exists():
        return set()
    df = read_csv(path)
    if "order_hash" not in df.columns:
        return set()
    return {str(x) for x in df["order_hash"].dropna().astype(str) if str(x)}


def validate_input(df: pd.DataFrame) -> dict[str, Any]:
    errors: list[str] = []
    per_n = counts(df) if "n" in df.columns else {}
    if len(df) != 64:
        errors.append(f"row count is {len(df)}, expected 64")
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
    if "batch_option" in df.columns and not df["batch_option"].astype(str).isin({BATCH_OPTION}).all():
        errors.append("selected batch is not Run53 primary calibrated_N24_N40_batch64")

    comparison_overlaps: dict[str, int] = {}
    if RUN53_COMPARISON.exists():
        comp = read_csv(RUN53_COMPARISON)
        row = comp[comp["batch_option"].astype(str) == "primary_batch64"]
        if not row.empty:
            for col in row.columns:
                if col.startswith("overlap_"):
                    comparison_overlaps[col] = int(row.iloc[0][col])
                    if int(row.iloc[0][col]) != 0:
                        errors.append(f"{col} is nonzero: {row.iloc[0][col]}")

    verdict = "PASS_RUN54_CALIBRATED_N24_N40_BATCH64_INPUT_READY" if not errors else "FAIL_RUN54_INPUT_VALIDATION"
    return {
        "timestamp": now_iso(),
        "verdict": verdict,
        "errors": errors,
        "input_path": str(RUN53_PRIMARY),
        "row_count": int(len(df)),
        "per_N_counts": per_n,
        "contains_N12": bool((df["n"].astype(int) == 12).any()),
        "contains_N16": bool((df["n"].astype(int) == 16).any()),
        "contains_N32": bool((df["n"].astype(int) == 32).any()),
        "comparison_overlap_status": comparison_overlaps,
    }


def make_handoff(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for n, group in df.sort_values(["n", "handoff_strategy_name", "candidate_id"]).groupby("n"):
        for i, (_, row) in enumerate(group.iterrows(), 1):
            handoff_name = f"S3R54CAL64_N{int(n)}_B{i:02d}_{short_bucket(row)}"
            rows.append({
                "run_id": RUN_ID,
                "batch_option": BATCH_OPTION,
                "batch_name": BATCH_NAME,
                "n": int(n),
                "handoff_strategy_name": handoff_name,
                "original_run53_candidate_id": row.get("candidate_id", ""),
                "original_run53_strategy_name": row.get("strategy_name", ""),
                "candidate_source": row.get("candidate_source", ""),
                "generation_method": row.get("generation_method", ""),
                "selection_bucket": row.get("selection_bucket", ""),
                "priority_role": row.get("priority_role", ""),
                "surrogate_prediction": row.get("surrogate_prediction", ""),
                "calibrated_reward_prediction": row.get("constrained_score", ""),
                "penalty_repair_prediction": row.get("penalty_repair_score", ""),
                "N24_u2_retention_prediction": row.get("N24_u2_retention_score", ""),
                "N40_strict_reward_retention_prediction": row.get("N40_strict_reward_retention_score", ""),
                "two_stage_repair_prediction": row.get("two_stage_penalty_repair_score", row.get("two_stage_guarded_score", "")),
                "median_guard_prediction": row.get("no_penalty_worse_than_median_score", ""),
                "u2_primary_prediction": row.get("predicted_u2_guarded_score", ""),
                "strict_penalty_guard_prediction": row.get("strict_penalty_guard_score", ""),
                "predicted_peeq_guarded_score": row.get("predicted_peeq_guarded_score", ""),
                "predicted_surfaceT_guarded_score": row.get("predicted_surfaceT_guarded_score", ""),
                "predicted_mises_guarded_score": row.get("predicted_mises_guarded_score", ""),
                "gnn_reward_prediction": row.get("gnn_reward_prediction", ""),
                "graph_pointer_policy_score": row.get("graph_pointer_policy_score", ""),
                "hybrid_score": row.get("hybrid_score", ""),
                "uncertainty_score": row.get("uncertainty_score", ""),
                "gnn_vs_surrogate_disagreement": row.get("gnn_vs_surrogate_disagreement", ""),
                "novelty_distance": row.get("novelty_distance", ""),
                "nearest_existing_teacher_strategy": row.get("nearest_existing_teacher_strategy", ""),
                "native_validation_N": True,
                "N24_N40_focused": True,
                "calibrated_batch64": True,
                "overnight_batch64": True,
                "order_json": order_json(row.get("order_json", "")),
                "order_compact": compact_order(row.get("order_json", "")),
                "order_hash": row.get("order_hash", ""),
                "teacher_validated": False,
                "teacher_validation_status": "NOT_RUN",
                "notes": "Run54 handoff only. Calibrated N24/N40 batch64 candidate is not teacher-validated.",
            })
    return pd.DataFrame(rows)


def write_scan_jsons(handoff: pd.DataFrame) -> None:
    SCAN_DIR.mkdir(parents=True, exist_ok=True)
    for old in SCAN_DIR.glob("scan_order_*.json"):
        old.unlink()
    for _, row in handoff.iterrows():
        payload = {
            "run_id": RUN_ID,
            "batch_option": BATCH_OPTION,
            "batch_name": BATCH_NAME,
            "n": int(row["n"]),
            "handoff_strategy_name": row["handoff_strategy_name"],
            "original_run53_candidate_id": row["original_run53_candidate_id"],
            "candidate_source": row["candidate_source"],
            "generation_method": row["generation_method"],
            "selection_bucket": row["selection_bucket"],
            "priority_role": row["priority_role"],
            "surrogate_prediction": row["surrogate_prediction"],
            "calibrated_reward_prediction": row["calibrated_reward_prediction"],
            "penalty_repair_prediction": row["penalty_repair_prediction"],
            "N24_u2_retention_prediction": row["N24_u2_retention_prediction"],
            "N40_strict_reward_retention_prediction": row["N40_strict_reward_retention_prediction"],
            "two_stage_repair_prediction": row["two_stage_repair_prediction"],
            "median_guard_prediction": row["median_guard_prediction"],
            "u2_primary_prediction": row["u2_primary_prediction"],
            "strict_penalty_guard_prediction": row["strict_penalty_guard_prediction"],
            "predicted_peeq_guarded_score": row["predicted_peeq_guarded_score"],
            "predicted_surfaceT_guarded_score": row["predicted_surfaceT_guarded_score"],
            "predicted_mises_guarded_score": row["predicted_mises_guarded_score"],
            "gnn_reward_prediction": row["gnn_reward_prediction"],
            "graph_pointer_policy_score": row["graph_pointer_policy_score"],
            "hybrid_score": row["hybrid_score"],
            "uncertainty_score": row["uncertainty_score"],
            "gnn_vs_surrogate_disagreement": row["gnn_vs_surrogate_disagreement"],
            "novelty_distance": row["novelty_distance"],
            "nearest_existing_teacher_strategy": row["nearest_existing_teacher_strategy"],
            "native_validation_N": True,
            "N24_N40_focused": True,
            "calibrated_batch64": True,
            "overnight_batch64": True,
            "scan_order": parse_order(row["order_json"]),
            "order_hash": row["order_hash"],
            "teacher_validated": False,
            "teacher_validation_status": "NOT_RUN",
            "notes": "Run54 handoff only. Calibrated N24/N40 batch64 candidate is not teacher-validated.",
        }
        write_json(SCAN_DIR / f"scan_order_{row['handoff_strategy_name']}.json", payload)


def make_future_cae_template(handoff: pd.DataFrame) -> pd.DataFrame:
    root = ROOT / "cae_model" / BATCH_NAME
    rows = []
    for _, row in handoff.iterrows():
        case_dir = root / f"N{int(row['n'])}{row['handoff_strategy_name']}"
        job = f"J2D_{row['handoff_strategy_name']}"
        scan_json = SCAN_DIR / f"scan_order_{row['handoff_strategy_name']}.json"
        rows.append({
            "n": int(row["n"]),
            "handoff_strategy_name": row["handoff_strategy_name"],
            "expected_case_dir": str(case_dir),
            "expected_job_name": job,
            "scan_order_json": str(scan_json),
            "expected_cae": str(case_dir / f"{job}.cae"),
            "expected_inp": str(case_dir / f"{job}.inp"),
            "expected_jnl": str(case_dir / f"{job}.jnl"),
            "expected_odb": str(case_dir / f"{job}.odb"),
            "teacher_validated": False,
            "generation_status": "NOT_GENERATED",
            "solver_status": "NOT_SUBMITTED",
            "notes": "Template only. Run54 did not generate CAE/INP/JNL or submit solver jobs.",
        })
    return pd.DataFrame(rows)


def write_command_template(handoff: pd.DataFrame) -> Path:
    path = OUTPUT_DIR / "stage3_run54_calibrated_N24_N40_batch64_abqjobpilot_commands_TEMPLATE.txt"
    root = ROOT / "cae_model" / BATCH_NAME
    lines = [
        "# TEMPLATE ONLY - do not execute until CAE/INP generation exists and passes checks.",
        "# Run54 did not generate INP files and did not run abqjobpilot/enqueue.",
    ]
    for _, row in handoff.iterrows():
        case_dir = root / f"N{int(row['n'])}{row['handoff_strategy_name']}"
        inp = case_dir / f"J2D_{row['handoff_strategy_name']}.inp"
        lines.append(f'enqueue --inp "{inp}" --cpus 14 --batch {BATCH_NAME} --strategy {row["handoff_strategy_name"]}')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def review_summary(handoff: pd.DataFrame, validation: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], str]:
    numeric_cols = [
        "surrogate_prediction", "calibrated_reward_prediction", "penalty_repair_prediction",
        "N24_u2_retention_prediction", "N40_strict_reward_retention_prediction",
        "two_stage_repair_prediction", "median_guard_prediction", "u2_primary_prediction",
        "strict_penalty_guard_prediction",
        "predicted_peeq_guarded_score", "predicted_surfaceT_guarded_score", "predicted_mises_guarded_score",
        "gnn_reward_prediction", "hybrid_score", "uncertainty_score",
        "gnn_vs_surrogate_disagreement", "novelty_distance",
    ]
    per_n = []
    for n, group in handoff.groupby("n"):
        row = {"n": int(n), "count": int(len(group))}
        for col in numeric_cols:
            if col in group:
                row[f"mean_{col}"] = pd.to_numeric(group[col], errors="coerce").mean()
        per_n.append(row)
    source_comp = handoff["candidate_source"].value_counts().to_dict()
    bucket_comp = handoff["selection_bucket"].value_counts().to_dict()
    summary = {
        "total_count": int(len(handoff)),
        "per_N_counts": counts(handoff),
        "only_N24_N40": True,
        "contains_N12": False,
        "contains_N16": False,
        "contains_N32": False,
        "candidate_source_composition": source_comp,
        "selection_bucket_composition": bucket_comp,
        "per_N_numeric_means": per_n,
        "expected_abaqus_cost": "64 jobs total, with 32 N24 and 32 N40",
        "overlap_status": validation.get("comparison_overlap_status", {}),
        "headline": "Run54 packages the user-selected overnight calibrated N24/N40 batch64 with N24 U2-retention, N40 strict/reward-retention, penalty-repair, uncertainty, diversity, and sentinel/control coverage.",
        "teacher_validated": False,
        "cae_inp_generated": False,
    }
    md = "# Run54 Calibrated N24/N40 Batch64 Review Summary\n\n"
    md += "- Total count: 64\n- Per-N counts: N24=32, N40=32\n- Included N values: N24/N40 only; no N12, N16, or N32 candidates.\n"
    md += "- Purpose: user-selected overnight batch64 testing calibrated N24 U2 retention, N40 strict/reward retention, and penalty-repair candidates.\n"
    md += "- Expected Abaqus cost: 64 jobs total, with 32 N24 and 32 N40.\n"
    md += "- Teacher validation status: NOT_RUN. Run54 did not create CAE/INP files.\n\n"
    md += "## Candidate Source Composition\n\n" + "\n".join(f"- {k}: {v}" for k, v in source_comp.items()) + "\n\n"
    md += "## Selection Bucket Composition\n\n" + "\n".join(f"- {k}: {v}" for k, v in bucket_comp.items()) + "\n"
    return pd.DataFrame(per_n), summary, md


def write_claim_boundary() -> None:
    safe = [
        "Run54 packages selected Run53 primary calibrated N24/N40 batch64 candidates for human review and future CAE generation.",
        "The selected batch contains N24=32 and N40=32.",
        "No N12, N16, or N32 candidates are included.",
        "The batch is designed to test calibrated N24 U2 retention, N40 strict/reward retention, and penalty repair.",
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
        "Do not claim surrogate/GNN/hybrid predictions are ground truth.",
        "Do not claim abqjobpilot commands are ready to execute.",
        "Do not claim CAE/INP files exist.",
        "Do not claim batch64 will improve teacher metrics before validation.",
    ]
    CLAIM_BOUNDARY_MD.write_text("# Run54 Claim Boundary\n\n## Safe claims\n" + "\n".join(f"- {x}" for x in safe) + "\n\n## Unsafe claims\n" + "\n".join(f"- {x}" for x in unsafe) + "\n", encoding="utf-8")
    write_json(CLAIM_BOUNDARY_JSON, {"verdict": "RUN54_HANDOFF_ONLY_CALIBRATED_N24_N40_BATCH64_NO_TEACHER_VALIDATION", "safe_claims": safe, "unsafe_claims": unsafe})


def write_report(validation: dict[str, Any], review: dict[str, Any], paths: dict[str, Path]) -> None:
    REPORT_PATH.write_text(f"""# Stage 3 Run 54 - Run53 Calibrated N24/N40 Batch64 Handoff Package

## 1. Purpose
Run54 packages the selected Run53 primary calibrated N24/N40 batch64 for human review and future CAE generation.

## 2. Inputs
- Selected primary batch64 candidate orders: `{RUN53_PRIMARY}`
- Run53 candidate pool: `{RUN53_POOL}`
- Run53 comparison table: `{RUN53_COMPARISON}`
- Run53 report: `{RUN53_REPORT}`

## 3. User-Selected Primary Batch64
Selected batch: `calibrated_N24_N40_batch64`. The batch contains 64 candidates: N24=32 and N40=32.

## 4. Why Batch64 Was Selected
The user explicitly selected an overnight batch64. The batch expands Run53 calibrated candidate generation after Run51 improved N24 U2 and N40 strict/reward behavior but still did not create raw PEEQ, SurfaceT, or Mises records.

## 5. Validation Status
Verdict: `{validation['verdict']}`. Only N24/N40 are present; no N12, N16, or N32 candidates are included.

## 6. Stable Naming Convention
Stable handoff names use `S3R54CAL64_N{{N}}_B{{index:02d}}_{{short_bucket}}`.

## 7. Candidate-Order Handoff Package
Candidate order CSV: `{paths['handoff_csv']}`.

## 8. Per-Candidate Scan-Order JSON Outputs
Scan-order JSON directory: `{SCAN_DIR}`.

## 9. Future CAE Handoff Template
Future CAE manifest template: `{paths['future_cae_template']}`. Run54 did not create CAE case directories.

## 10. Future abqjobpilot Command Template
Future abqjobpilot template: `{paths['command_template']}`. It is a template only and must not be executed until INPs exist and pass checks.

## 11. Review Summary
{review['headline']}

Candidate-source composition: `{review['candidate_source_composition']}`.

Selection-bucket composition: `{review['selection_bucket_composition']}`.

## 12. Claim Boundary
Claim boundary verdict: `RUN54_HANDOFF_ONLY_CALIBRATED_N24_N40_BATCH64_NO_TEACHER_VALIDATION`.

## 13. Output Files
- Handoff CSV: `{paths['handoff_csv']}`
- Scan-order JSON directory: `{SCAN_DIR}`
- Future CAE template: `{paths['future_cae_template']}`
- Future abqjobpilot template: `{paths['command_template']}`
- Review summary: `{paths['review_md']}`
- Manifest: `{MANIFEST_PATH}`

## 14. Recommended Run55
CAE module should generate CAE/INP/JNL for selected Run54 calibrated N24/N40 batch64 only. Do not run solver. Do not execute abqjobpilot. Do not generate the Run53 reference batch32 or reference recovery batch40 unless explicitly selected later.
""", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SCAN_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    selected = read_csv(RUN53_PRIMARY)
    validation = validate_input(selected)
    validation_path = OUTPUT_DIR / "run54_input_validation_summary.json"
    write_json(validation_path, validation)
    if not validation["verdict"].startswith("PASS"):
        raise SystemExit(validation["errors"])

    handoff = make_handoff(selected)
    handoff_csv = OUTPUT_DIR / "stage3_run54_calibrated_N24_N40_batch64_candidate_orders.csv"
    write_csv(handoff_csv, handoff)
    write_scan_jsons(handoff)

    future_cae = make_future_cae_template(handoff)
    future_cae_path = OUTPUT_DIR / "stage3_run54_calibrated_N24_N40_batch64_future_cae_handoff_manifest_TEMPLATE.csv"
    write_csv(future_cae_path, future_cae)
    command_template = write_command_template(handoff)

    review_csv, review, review_md = review_summary(handoff, validation)
    review_csv_path = OUTPUT_DIR / "calibrated_N24_N40_batch64_review_summary.csv"
    review_json_path = OUTPUT_DIR / "calibrated_N24_N40_batch64_review_summary.json"
    review_md_path = OUTPUT_DIR / "calibrated_N24_N40_batch64_review_summary.md"
    write_csv(review_csv_path, review_csv)
    write_json(review_json_path, review)
    review_md_path.write_text(review_md, encoding="utf-8")

    write_claim_boundary()
    paths = {
        "handoff_csv": handoff_csv,
        "future_cae_template": future_cae_path,
        "command_template": command_template,
        "review_md": review_md_path,
    }
    write_report(validation, review, paths)

    output_files = [
        validation_path,
        handoff_csv,
        *sorted(SCAN_DIR.glob("scan_order_*.json")),
        future_cae_path,
        command_template,
        review_csv_path,
        review_json_path,
        review_md_path,
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
        "input_files": [str(RUN53_PRIMARY), str(RUN53_POOL), str(RUN53_COMPARISON), str(RUN53_REPORT), str(RUN53_MANIFEST), str(COMBINED328_READY)],
        "output_files": [str(p) for p in output_files],
        "selected_batch": "run53_calibrated_N24_N40_batch64",
        "batch_name": BATCH_NAME,
        "batch64_count": 64,
        "per_N_counts": counts(handoff),
        "includes_N12": False,
        "includes_N16": False,
        "includes_N24": True,
        "includes_N32": False,
        "includes_N40": True,
        "N24_N40_focused": True,
        "calibrated_batch64": True,
        "overnight_batch64": True,
        "report_path": str(REPORT_PATH),
        "claim_boundary_path": str(CLAIM_BOUNDARY_MD),
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
    print(json.dumps({"verdict": validation["verdict"], "counts": counts(handoff), "handoff_csv": str(handoff_csv), "report": str(REPORT_PATH), "manifest": str(MANIFEST_PATH)}, indent=2))


if __name__ == "__main__":
    main()


