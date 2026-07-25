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
RUN_ID = "run_49_run48_stricter_constrained_N24_N40_batch32_handoff_package"
RUN_NAME = "run48 stricter constrained N24/N40 batch32 handoff package"
SCRIPT_PATH = ROOT / "scripts" / "stage3" / "run_49_create_run48_stricter_constrained_N24_N40_batch32_handoff_package.py"

RUN48_OPTION_A = ROOT / "outputs" / "stage3_run_48_combined296_stricter_constrained_N24_N40_candidate_generation" / "run48_stricter_constrained_N24_N40_batch32_candidate_orders.csv"
RUN48_POOL = ROOT / "outputs" / "stage3_run_48_combined296_stricter_constrained_N24_N40_candidate_generation" / "run48_candidate_pool_scored.csv"
RUN48_COMPARISON = ROOT / "outputs" / "stage3_run_48_combined296_stricter_constrained_N24_N40_candidate_generation" / "run48_batch_options_comparison_to_previous.csv"
RUN48_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_48_combined296_stricter_constrained_N24_N40_candidate_generation" / "RUN_48_COMBINED296_STRICTER_CONSTRAINED_N24_N40_CANDIDATE_GENERATION_REPORT.md"
RUN48_MANIFEST = ROOT / "artifacts" / "manifests" / "stage3_run_48_manifest.json"
COMBINED296_READY = ROOT / "outputs" / "stage3_run_47_constrained_N24_N40_batch32_teacher_metrics_ingestion_and_combined296_ranking" / "combined296_RL_ready_dataset.csv"

OUTPUT_DIR = ROOT / "outputs" / "stage3_run_49_run48_stricter_constrained_N24_N40_batch32_handoff_package"
SCAN_DIR = OUTPUT_DIR / "scan_orders"
REPORT_DIR = ROOT / "docs" / "stage3" / "runs" / "run_49_run48_stricter_constrained_N24_N40_batch32_handoff_package"
REPORT_PATH = REPORT_DIR / "RUN_49_RUN48_STRICTER_CONSTRAINED_N24_N40_BATCH32_HANDOFF_PACKAGE_REPORT.md"
MANIFEST_PATH = ROOT / "artifacts" / "manifests" / "stage3_run_49_manifest.json"
CLAIM_BOUNDARY_MD = OUTPUT_DIR / "run49_claim_boundary.md"
CLAIM_BOUNDARY_JSON = OUTPUT_DIR / "run49_claim_boundary.json"

BATCH_NAME = "stage3_run49_stricter_constrained_N24_N40_batch32_v01"
BATCH_OPTION = "stricter_constrained_N24_N40_batch32"
EXPECTED_COUNTS = {24: 16, 40: 16}


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
    if len(df) != 32:
        errors.append(f"row count is {len(df)}, expected 32")
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
    if "batch_option" in df.columns and not df["batch_option"].astype(str).isin(
        {"stricter_constrained_N24_N40_batch32", "constrained_N24_N40_batch32"}
    ).all():
        errors.append("selected batch is not Run48 Option A stricter_constrained_N24_N40_batch32")

    comparison_overlaps: dict[str, int] = {}
    if RUN48_COMPARISON.exists():
        comp = read_csv(RUN48_COMPARISON)
        row = comp[comp["batch_option"].astype(str) == "A_batch32"]
        if not row.empty:
            for col in row.columns:
                if col.startswith("overlap_"):
                    comparison_overlaps[col] = int(row.iloc[0][col])
                    if int(row.iloc[0][col]) != 0:
                        errors.append(f"{col} is nonzero: {row.iloc[0][col]}")

    verdict = "PASS_RUN49_STRICTER_CONSTRAINED_N24_N40_BATCH32_INPUT_READY" if not errors else "FAIL_RUN49_INPUT_VALIDATION"
    return {
        "timestamp": now_iso(),
        "verdict": verdict,
        "errors": errors,
        "input_path": str(RUN48_OPTION_A),
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
            handoff_name = f"S3R49SCN_N{int(n)}_B{i:02d}_{short_bucket(row)}"
            rows.append({
                "run_id": RUN_ID,
                "batch_option": BATCH_OPTION,
                "batch_name": BATCH_NAME,
                "n": int(n),
                "handoff_strategy_name": handoff_name,
                "original_run48_candidate_id": row.get("candidate_id", ""),
                "original_run48_strategy_name": row.get("strategy_name", ""),
                "candidate_source": row.get("candidate_source", ""),
                "generation_method": row.get("generation_method", ""),
                "selection_bucket": row.get("selection_bucket", ""),
                "priority_role": row.get("priority_role", ""),
                "surrogate_prediction": row.get("surrogate_prediction", ""),
                "strict_penalty_guard_prediction": row.get("strict_penalty_guard_score", ""),
                "two_stage_guarded_prediction": row.get("two_stage_guarded_score", ""),
                "no_penalty_worse_than_median_prediction": row.get("no_penalty_worse_than_median_score", ""),
                "u2_primary_prediction": row.get("constrained_score", ""),
                "u2_guarded_prediction": row.get("predicted_u2_guarded_score", ""),
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
                "stricter_penalty_guard": True,
                "order_json": order_json(row.get("order_json", "")),
                "order_compact": compact_order(row.get("order_json", "")),
                "order_hash": row.get("order_hash", ""),
                "teacher_validated": False,
                "teacher_validation_status": "NOT_RUN",
                "notes": "Run49 handoff only. Stricter constrained N24/N40 batch32 candidate is not teacher-validated.",
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
            "original_run48_candidate_id": row["original_run48_candidate_id"],
            "candidate_source": row["candidate_source"],
            "generation_method": row["generation_method"],
            "selection_bucket": row["selection_bucket"],
            "priority_role": row["priority_role"],
            "surrogate_prediction": row["surrogate_prediction"],
            "strict_penalty_guard_prediction": row["strict_penalty_guard_prediction"],
            "two_stage_guarded_prediction": row["two_stage_guarded_prediction"],
            "no_penalty_worse_than_median_prediction": row["no_penalty_worse_than_median_prediction"],
            "u2_primary_prediction": row["u2_primary_prediction"],
            "u2_guarded_prediction": row["u2_guarded_prediction"],
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
            "stricter_penalty_guard": True,
            "scan_order": parse_order(row["order_json"]),
            "order_hash": row["order_hash"],
            "teacher_validated": False,
            "teacher_validation_status": "NOT_RUN",
            "notes": "Run49 handoff only. Stricter constrained N24/N40 batch32 candidate is not teacher-validated.",
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
            "notes": "Template only. Run49 did not generate CAE/INP/JNL or submit solver jobs.",
        })
    return pd.DataFrame(rows)


def write_command_template(handoff: pd.DataFrame) -> Path:
    path = OUTPUT_DIR / "stage3_run49_stricter_constrained_N24_N40_batch32_abqjobpilot_commands_TEMPLATE.txt"
    root = ROOT / "cae_model" / BATCH_NAME
    lines = [
        "# TEMPLATE ONLY - do not execute until CAE/INP generation exists and passes checks.",
        "# Run49 did not generate INP files and did not run abqjobpilot/enqueue.",
    ]
    for _, row in handoff.iterrows():
        case_dir = root / f"N{int(row['n'])}{row['handoff_strategy_name']}"
        inp = case_dir / f"J2D_{row['handoff_strategy_name']}.inp"
        lines.append(f'enqueue --inp "{inp}" --cpus 14 --batch {BATCH_NAME} --strategy {row["handoff_strategy_name"]}')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def review_summary(handoff: pd.DataFrame, validation: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], str]:
    numeric_cols = [
        "surrogate_prediction", "strict_penalty_guard_prediction", "two_stage_guarded_prediction",
        "no_penalty_worse_than_median_prediction", "u2_primary_prediction", "u2_guarded_prediction",
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
        "expected_abaqus_cost": "32 jobs total, with 16 N24 and 16 N40",
        "overlap_status": validation.get("comparison_overlap_status", {}),
        "headline": "Run49 packages a quick stricter penalty-guard N24/N40 validation loop after Run46 improved U2/reward but did not create raw PEEQ/SurfaceT/Mises records.",
        "teacher_validated": False,
        "cae_inp_generated": False,
    }
    md = "# Run49 Stricter Constrained N24/N40 Batch32 Review Summary\n\n"
    md += "- Total count: 32\n- Per-N counts: N24=16, N40=16\n- Included N values: N24/N40 only; no N12, N16, or N32 candidates.\n"
    md += "- Purpose: test stricter penalty guards after Run46 improved U2/reward but did not create raw PEEQ/SurfaceT/Mises records.\n"
    md += "- Expected Abaqus cost: 32 jobs total, with 16 N24 and 16 N40.\n"
    md += "- Teacher validation status: NOT_RUN. Run49 did not create CAE/INP files.\n\n"
    md += "## Candidate Source Composition\n\n" + "\n".join(f"- {k}: {v}" for k, v in source_comp.items()) + "\n\n"
    md += "## Selection Bucket Composition\n\n" + "\n".join(f"- {k}: {v}" for k, v in bucket_comp.items()) + "\n"
    return pd.DataFrame(per_n), summary, md


def write_claim_boundary() -> None:
    safe = [
        "Run49 packages selected Run48 Option A stricter constrained N24/N40 batch32 candidates for human review and future CAE generation.",
        "The selected batch contains N24=16 and N40=16.",
        "No N12, N16, or N32 candidates are included.",
        "The batch is designed to test stricter penalty guards after Run46 improved U2/reward but did not create raw PEEQ/SurfaceT/Mises records.",
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
    ]
    CLAIM_BOUNDARY_MD.write_text("# Run49 Claim Boundary\n\n## Safe claims\n" + "\n".join(f"- {x}" for x in safe) + "\n\n## Unsafe claims\n" + "\n".join(f"- {x}" for x in unsafe) + "\n", encoding="utf-8")
    write_json(CLAIM_BOUNDARY_JSON, {"verdict": "RUN49_HANDOFF_ONLY_STRICTER_CONSTRAINED_N24_N40_BATCH32_NO_TEACHER_VALIDATION", "safe_claims": safe, "unsafe_claims": unsafe})


def write_report(validation: dict[str, Any], review: dict[str, Any], paths: dict[str, Path]) -> None:
    REPORT_PATH.write_text(f"""# Stage 3 Run 49 - Run48 Stricter Constrained N24/N40 Batch32 Handoff Package

## 1. Purpose
Run49 packages the selected Run48 Option A stricter constrained N24/N40 batch32 for human review and future CAE generation.

## 2. Inputs
- Selected Option A candidate orders: `{RUN48_OPTION_A}`
- Run48 candidate pool: `{RUN48_POOL}`
- Run48 comparison table: `{RUN48_COMPARISON}`
- Run48 report: `{RUN48_REPORT}`

## 3. User-Selected Batch
Selected batch: `stricter_constrained_N24_N40_batch32`. The batch contains 32 candidates: N24=16 and N40=16.

## 4. Why Option A Was Selected
Option A is the quick validation loop for the stricter penalty-guard rule. It tests candidates that keep U2 near the top region while pushing PEEQ, SurfaceT, and Mises guards harder after Run46 improved U2/reward but did not create raw penalty-metric records.

## 5. Validation Status
Verdict: `{validation['verdict']}`. Only N24/N40 are present; no N12, N16, or N32 candidates are included.

## 6. Stable Naming Convention
Stable handoff names use `S3R49SCN_N{{N}}_B{{index:02d}}_{{short_bucket}}`.

## 7. Candidate-Order Handoff Package
Candidate order CSV: `{paths['handoff_csv']}`.

## 8. Per-Candidate Scan-Order JSON Outputs
Scan-order JSON directory: `{SCAN_DIR}`.

## 9. Future CAE Handoff Template
Future CAE manifest template: `{paths['future_cae_template']}`. Run49 did not create CAE case directories.

## 10. Future abqjobpilot Command Template
Future abqjobpilot template: `{paths['command_template']}`. It is a template only and must not be executed until INPs exist and pass checks.

## 11. Review Summary
{review['headline']}

Candidate-source composition: `{review['candidate_source_composition']}`.

Selection-bucket composition: `{review['selection_bucket_composition']}`.

## 12. Claim Boundary
Claim boundary verdict: `RUN49_HANDOFF_ONLY_STRICTER_CONSTRAINED_N24_N40_BATCH32_NO_TEACHER_VALIDATION`.

## 13. Output Files
- Handoff CSV: `{paths['handoff_csv']}`
- Scan-order JSON directory: `{SCAN_DIR}`
- Future CAE template: `{paths['future_cae_template']}`
- Future abqjobpilot template: `{paths['command_template']}`
- Review summary: `{paths['review_md']}`
- Manifest: `{MANIFEST_PATH}`

## 14. Recommended Run50
CAE module should generate CAE/INP/JNL for selected Run49 stricter constrained N24/N40 batch32 only. Do not run solver. Do not execute abqjobpilot. Do not generate Option B batch60 or Option C recovery batch40 unless explicitly selected later.
""", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SCAN_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    selected = read_csv(RUN48_OPTION_A)
    validation = validate_input(selected)
    write_json(OUTPUT_DIR / "run49_input_validation_summary.json", validation)
    if not validation["verdict"].startswith("PASS"):
        raise SystemExit(validation["errors"])

    handoff = make_handoff(selected)
    handoff_csv = OUTPUT_DIR / "stage3_run49_stricter_constrained_N24_N40_batch32_candidate_orders.csv"
    write_csv(handoff_csv, handoff)
    write_scan_jsons(handoff)

    future_cae = make_future_cae_template(handoff)
    future_cae_path = OUTPUT_DIR / "stage3_run49_stricter_constrained_N24_N40_batch32_future_cae_handoff_manifest_TEMPLATE.csv"
    write_csv(future_cae_path, future_cae)
    command_template = write_command_template(handoff)

    review_csv, review, review_md = review_summary(handoff, validation)
    review_csv_path = OUTPUT_DIR / "stricter_constrained_N24_N40_batch32_review_summary.csv"
    review_json_path = OUTPUT_DIR / "stricter_constrained_N24_N40_batch32_review_summary.json"
    review_md_path = OUTPUT_DIR / "stricter_constrained_N24_N40_batch32_review_summary.md"
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
        OUTPUT_DIR / "run49_input_validation_summary.json",
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
        "input_files": [str(RUN48_OPTION_A), str(RUN48_POOL), str(RUN48_COMPARISON), str(RUN48_REPORT), str(RUN48_MANIFEST), str(COMBINED296_READY)],
        "output_files": [str(p) for p in output_files],
        "selected_batch": "run48_stricter_constrained_N24_N40_batch32",
        "batch_name": BATCH_NAME,
        "batch32_count": 32,
        "per_N_counts": counts(handoff),
        "includes_N12": False,
        "includes_N16": False,
        "includes_N24": True,
        "includes_N32": False,
        "includes_N40": True,
        "N24_N40_focused": True,
        "stricter_penalty_guard": True,
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

