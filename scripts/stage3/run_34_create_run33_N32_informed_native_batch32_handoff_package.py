from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
RUN_ID = "run_34_run33_N32_informed_native_batch32_handoff_package"
RUN_NAME = "run33 N32-informed native batch32 handoff package"

RUN33_OPTION_A = ROOT / "outputs" / "stage3_run_33_combined172_plus_N32_balanced_surrogate_gnn_candidate_generation" / "run33_N32_informed_native_batch32_candidate_orders.csv"
RUN33_POOL = ROOT / "outputs" / "stage3_run_33_combined172_plus_N32_balanced_surrogate_gnn_candidate_generation" / "run33_candidate_pool_scored.csv"
RUN33_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_33_combined172_plus_N32_balanced_surrogate_gnn_candidate_generation" / "RUN_33_COMBINED172_PLUS_N32_BALANCED_SURROGATE_GNN_CANDIDATE_GENERATION_REPORT.md"
RUN33_MANIFEST = ROOT / "artifacts" / "manifests" / "stage3_run_33_manifest.json"
RUN32A_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_32a_stage2_n32_legacy_teacher_label_ingestion_for_stage3" / "RUN_32A_STAGE2_N32_LEGACY_TEACHER_LABEL_INGESTION_FOR_STAGE3_REPORT.md"
SUPERSEDED_RUN30_BATCH32 = ROOT / "outputs" / "stage3_run_30_run29_hybrid_policy_batch32_handoff_package" / "stage3_run30_hybrid_policy_batch32_candidate_orders.csv"
RUN31_SUPERSEDED_NOTE = ROOT / "outputs" / "stage3_run_31_hybrid_policy_batch32_cae_inp_generation" / "RUN31_SUPERSEDED_DO_NOT_ENQUEUE.md"

OUTPUT_DIR = ROOT / "outputs" / "stage3_run_34_run33_N32_informed_native_batch32_handoff_package"
SCAN_ORDER_DIR = OUTPUT_DIR / "scan_orders"
REPORT_DIR = ROOT / "docs" / "stage3" / "runs" / RUN_ID
REPORT_PATH = REPORT_DIR / "RUN_34_RUN33_N32_INFORMED_NATIVE_BATCH32_HANDOFF_PACKAGE_REPORT.md"
MANIFEST_PATH = ROOT / "artifacts" / "manifests" / "stage3_run_34_manifest.json"
RUN_INDEX_PATH = ROOT / "docs" / "stage3" / "STAGE3_RUN_INDEX.md"

EXPECTED_NS = [12, 16, 24, 40]
EXPECTED_COUNTS = {12: 4, 16: 4, 24: 12, 40: 12}
SELECTED_BATCH = "run33_N32_informed_native_batch32"
BATCH_OPTION = "N32_informed_native_batch32"
BATCH_NAME = "stage3_run34_N32_informed_native_batch32_v01"
FUTURE_CASE_ROOT = ROOT / "cae_model" / BATCH_NAME


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2) + "\n", encoding="utf-8")


def parse_int(value: Any) -> int:
    text = str(value).strip()
    if text.upper().startswith("N"):
        text = text[1:]
    return int(float(text))


def parse_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def parse_order(text: Any) -> list[int] | None:
    try:
        value = json.loads(str(text))
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(value, list):
        return None
    try:
        return [int(x) for x in value]
    except (TypeError, ValueError):
        return None


def order_hash(order: list[int]) -> str:
    return hashlib.sha1(",".join(str(x) for x in order).encode("ascii")).hexdigest()[:16]


def validate_order(order: list[int] | None, n: int) -> tuple[bool, str]:
    if order is None:
        return False, "missing or unparsable scan order"
    if len(order) != n:
        return False, f"length {len(order)} != N{n}"
    if len(set(order)) != n:
        return False, "duplicate track ids"
    expected = set(range(n))
    actual = set(order)
    if actual != expected:
        return False, f"not permutation 0..{n-1}; missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
    return True, "legal permutation"


def safe_token(text: Any, fallback: str = "candidate", max_len: int = 36) -> str:
    token = re.sub(r"[^A-Za-z0-9_]+", "_", str(text).strip())
    token = re.sub(r"_+", "_", token).strip("_").lower()
    return (token or fallback)[:max_len]


def bucket_token(row: dict[str, str]) -> str:
    mapping = {
        "graph_pointer_top": "graph_pointer",
        "surrogate_top_predicted": "surrogate_top",
        "hybrid_gnn_surrogate_agreement": "hybrid_agree",
        "hybrid_gnn_surrogate_disagreement": "hybrid_disagree",
        "uncertainty_calibration": "uncertainty",
        "diversity_coverage": "diversity",
        "surrogate_known_best_local_search": "known_best",
        "sentinel_control": "sentinel",
        "N16_new_best_neighborhood": "n16_best_near",
        "N24_calibration_neighborhood": "n24_calibration",
        "N40_new_best_neighborhood": "n40_best_near",
    }
    bucket = str(row.get("selection_bucket", "")).strip()
    source = str(row.get("candidate_source", "")).strip()
    return safe_token(mapping.get(bucket, bucket or source), "n32_informed", 30)


def old_run31_hashes() -> set[str]:
    if not SUPERSEDED_RUN30_BATCH32.exists():
        return set()
    return {row.get("order_hash", "") for row in read_csv(SUPERSEDED_RUN30_BATCH32)}


def validate_selected_batch(rows: list[dict[str, str]]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    counts: Counter[int] = Counter()
    candidate_ids: Counter[str] = Counter()
    orders_by_n: dict[int, set[str]] = defaultdict(set)
    buckets: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    old_hashes = old_run31_hashes()
    exact_overlap = 0

    if not RUN33_OPTION_A.exists():
        errors.append(f"missing selected Option A file: {RUN33_OPTION_A}")
    for row in rows:
        try:
            n = parse_int(row.get("n"))
        except Exception:
            errors.append(f"invalid n={row.get('n')}")
            continue
        counts[n] += 1
        if n == 32:
            errors.append(f"N32 row present but Option A must be native-only: {row.get('candidate_id') or row.get('strategy_name')}")
        cid = str(row.get("candidate_id", "")).strip()
        strategy = str(row.get("strategy_name", "")).strip()
        if not cid and not strategy:
            errors.append("row missing candidate_id/strategy_name")
        if cid:
            candidate_ids[cid] += 1
        order = parse_order(row.get("order_json") or row.get("scan_order"))
        legal, reason = validate_order(order, n)
        if not legal:
            errors.append(f"{cid or strategy}: {reason}")
            continue
        digest = row.get("order_hash") or order_hash(order or [])
        if digest in orders_by_n[n]:
            errors.append(f"duplicate order within N{n}: {cid or strategy}")
        orders_by_n[n].add(digest)
        if digest in old_hashes:
            exact_overlap += 1
        if not row.get("candidate_source"):
            warnings.append(f"{cid or strategy}: missing candidate_source")
        if not row.get("selection_bucket"):
            warnings.append(f"{cid or strategy}: missing selection_bucket")
        for col in ["surrogate_reward_pred", "gnn_reward_pred", "hybrid_score", "uncertainty_score", "gnn_surrogate_disagreement", "novelty_distance_to_combined172_plus_N32"]:
            if not math.isfinite(parse_float(row.get(col))):
                warnings.append(f"{cid or strategy}: missing optional numeric metadata {col}")
        buckets[row.get("selection_bucket", "")] += 1
        sources[row.get("candidate_source", "")] += 1

    if len(rows) != 32:
        errors.append(f"expected 32 rows, found {len(rows)}")
    if dict(sorted(counts.items())) != EXPECTED_COUNTS:
        errors.append(f"expected per-N {EXPECTED_COUNTS}, found {dict(sorted(counts.items()))}")
    dup_cids = [cid for cid, count in candidate_ids.items() if count > 1]
    if dup_cids:
        errors.append(f"duplicate candidate_id values: {dup_cids}")
    if exact_overlap != 0:
        errors.append(f"expected zero exact overlap with superseded Run31/Run30 batch32, found {exact_overlap}")

    payload = {
        "verdict": "PASS_RUN34_N32_INFORMED_NATIVE_BATCH32_INPUT_READY" if not errors else "FAIL_RUN34_INPUT_INVALID",
        "errors": errors,
        "warnings": warnings[:30],
        "warning_count": len(warnings),
        "selected_batch": SELECTED_BATCH,
        "selected_option": "Run33 Option A",
        "row_count": len(rows),
        "per_n_counts": dict(sorted(counts.items())),
        "contains_n32_rows": counts[32] > 0,
        "selected_batch_confirmed_not_option_b_or_option_c": True,
        "candidate_source_composition": dict(sources),
        "selection_bucket_composition": dict(buckets),
        "exact_overlap_with_superseded_run31_batch32": exact_overlap,
        "N32_informed_training_context": True,
        "native_validation_N": True,
    }
    write_json(OUTPUT_DIR / "run34_input_validation_summary.json", payload)
    return payload


def handoff_name(n: int, index: int, row: dict[str, str]) -> str:
    return f"S3R34N32INF_N{n}_B{index:02d}_{bucket_token(row)}"


def build_handoff(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_n: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_n[parse_int(row["n"])].append(row)
    handoff: list[dict[str, Any]] = []
    for n in EXPECTED_NS:
        group = sorted(by_n[n], key=lambda row: parse_float(row.get("option_a_native_batch32_rank_within_n"), 999.0))
        for idx, row in enumerate(group, start=1):
            order = parse_order(row.get("order_json")) or []
            digest = row.get("order_hash") or order_hash(order)
            name = handoff_name(n, idx, row)
            handoff.append(
                {
                    "run_id": RUN_ID,
                    "batch_option": BATCH_OPTION,
                    "batch_name": BATCH_NAME,
                    "n": n,
                    "handoff_strategy_name": name,
                    "original_run33_candidate_id": row.get("candidate_id", ""),
                    "original_run33_strategy_name": row.get("strategy_name", ""),
                    "candidate_source": row.get("candidate_source", ""),
                    "generation_method": row.get("generation_method", ""),
                    "selection_bucket": row.get("selection_bucket", ""),
                    "priority_role": row.get("priority_role", ""),
                    "surrogate_prediction": parse_float(row.get("surrogate_reward_pred")),
                    "gnn_reward_prediction": parse_float(row.get("gnn_reward_pred")),
                    "graph_pointer_policy_score": parse_float(row.get("graph_pointer_mean_logprob")),
                    "hybrid_score": parse_float(row.get("hybrid_score")),
                    "uncertainty_score": parse_float(row.get("uncertainty_score")),
                    "gnn_vs_surrogate_disagreement": parse_float(row.get("gnn_surrogate_disagreement")),
                    "novelty_distance_to_combined172_plus_N32": parse_float(row.get("novelty_distance_to_combined172_plus_N32")),
                    "nearest_existing_teacher_strategy": row.get("nearest_existing_teacher_strategy", ""),
                    "N32_informed": True,
                    "native_validation_N": True,
                    "order_json": json.dumps(order, separators=(",", ":")),
                    "order_compact": row.get("order_compact", "-".join(str(x) for x in order)),
                    "order_hash": digest,
                    "teacher_validated": False,
                    "teacher_validation_status": "NOT_RUN",
                    "notes": "Run34 handoff only. N32-informed native-N candidate is not teacher-validated.",
                }
            )
    return handoff


def write_scan_order_jsons(handoff: list[dict[str, Any]]) -> list[str]:
    SCAN_ORDER_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for row in handoff:
        order = parse_order(row["order_json"]) or []
        payload = {
            "run_id": RUN_ID,
            "batch_option": BATCH_OPTION,
            "batch_name": BATCH_NAME,
            "n": row["n"],
            "handoff_strategy_name": row["handoff_strategy_name"],
            "original_run33_candidate_id": row["original_run33_candidate_id"],
            "candidate_source": row["candidate_source"],
            "generation_method": row["generation_method"],
            "selection_bucket": row["selection_bucket"],
            "priority_role": row["priority_role"],
            "surrogate_prediction": row["surrogate_prediction"],
            "gnn_reward_prediction": row["gnn_reward_prediction"],
            "graph_pointer_policy_score": row["graph_pointer_policy_score"],
            "hybrid_score": row["hybrid_score"],
            "uncertainty_score": row["uncertainty_score"],
            "gnn_vs_surrogate_disagreement": row["gnn_vs_surrogate_disagreement"],
            "novelty_distance_to_combined172_plus_N32": row["novelty_distance_to_combined172_plus_N32"],
            "nearest_existing_teacher_strategy": row["nearest_existing_teacher_strategy"],
            "N32_informed": True,
            "native_validation_N": True,
            "scan_order": order,
            "order_hash": row["order_hash"],
            "teacher_validated": False,
            "teacher_validation_status": "NOT_RUN",
            "notes": "Run34 handoff only. N32-informed native-N candidate is not teacher-validated.",
        }
        path = SCAN_ORDER_DIR / f"scan_order_{row['handoff_strategy_name']}.json"
        write_json(path, payload)
        paths.append(str(path))
    return paths


def write_future_templates(handoff: list[dict[str, Any]]) -> tuple[Path, Path]:
    manifest_rows: list[dict[str, Any]] = []
    commands = [
        "# TEMPLATE ONLY - do not execute until CAE/INP generation has completed and passed checks.",
        "# INP files do not exist yet.",
        "# Run34 did not generate CAE/INP/JNL files or run abqjobpilot/enqueue.",
        "",
    ]
    for row in handoff:
        n = row["n"]
        name = row["handoff_strategy_name"]
        case_dir = FUTURE_CASE_ROOT / f"N{n}{name}"
        job = f"J2D_{name}"
        scan_json = SCAN_ORDER_DIR / f"scan_order_{name}.json"
        manifest_rows.append(
            {
                "n": n,
                "handoff_strategy_name": name,
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
                "notes": "Template only. Do not run until future CAE/INP generation is approved and completed.",
            }
        )
        commands.append(f'enqueue --inp "{case_dir}\\{job}.inp" --cpus 14 --batch {BATCH_NAME} --strategy {name}')
    manifest_path = OUTPUT_DIR / "stage3_run34_N32_informed_native_batch32_future_cae_handoff_manifest_TEMPLATE.csv"
    commands_path = OUTPUT_DIR / "stage3_run34_N32_informed_native_batch32_abqjobpilot_commands_TEMPLATE.txt"
    write_csv(manifest_path, manifest_rows)
    commands_path.write_text("\n".join(commands) + "\n", encoding="utf-8")
    return manifest_path, commands_path


def mean_numeric(rows: list[dict[str, Any]], col: str) -> float:
    vals = [parse_float(row.get(col)) for row in rows]
    vals = [v for v in vals if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else math.nan


def write_review_summary(handoff: list[dict[str, Any]]) -> tuple[Path, Path, Path, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n in EXPECTED_NS:
        group = [row for row in handoff if row["n"] == n]
        rows.append(
            {
                "n": n,
                "count": len(group),
                "mean_surrogate_prediction": mean_numeric(group, "surrogate_prediction"),
                "mean_gnn_reward_prediction": mean_numeric(group, "gnn_reward_prediction"),
                "mean_hybrid_score": mean_numeric(group, "hybrid_score"),
                "mean_disagreement": mean_numeric(group, "gnn_vs_surrogate_disagreement"),
                "mean_novelty_distance_to_combined172_plus_N32": mean_numeric(group, "novelty_distance_to_combined172_plus_N32"),
            }
        )
    payload = {
        "total_count": len(handoff),
        "per_n_counts": dict(Counter(row["n"] for row in handoff)),
        "includes_N32_candidates": False,
        "n24_n40_share": sum(1 for row in handoff if row["n"] in (24, 40)) / max(1, len(handoff)),
        "candidate_source_composition": dict(Counter(row["candidate_source"] for row in handoff)),
        "selection_bucket_composition": dict(Counter(row["selection_bucket"] for row in handoff)),
        "expected_abaqus_cost": "32 jobs total, with 24 jobs from N24/N40",
        "exact_overlap_with_superseded_run31_batch32": 0,
        "headline": "N32-informed native batch32 replaces superseded Run31 while keeping validation to native Stage 3 N values only.",
    }
    md_lines = [
        "# Run34 N32-Informed Native Batch32 Review Summary",
        "",
        "- Selected batch: `Run33 Option A - N32-informed native batch32`",
        "- Total candidates: `32`",
        "- Per-N counts: `N12=4`, `N16=4`, `N24=12`, `N40=12`",
        "- No N32 candidates are included.",
        "- N24/N40 share: `24/32`",
        "- Expected Abaqus cost: 32 jobs total, with 24 jobs from N24/N40.",
        "- Exact overlap with superseded Run31/Run30 batch32: `0/32`.",
        "- This batch uses N32 as training information but validates only native Stage 3 N values.",
        "- This is the clean replacement for the abandoned/superseded Run31 batch.",
        "- The batch remains unvalidated until future Abaqus teacher validation.",
        "- Run34 did not create CAE/INP files.",
        "",
        "## Candidate Source Composition",
        *[f"- `{k}`: `{v}`" for k, v in payload["candidate_source_composition"].items()],
        "",
        "## Selection Bucket Composition",
        *[f"- `{k}`: `{v}`" for k, v in payload["selection_bucket_composition"].items()],
    ]
    csv_path = OUTPUT_DIR / "N32_informed_native_batch32_review_summary.csv"
    json_path = OUTPUT_DIR / "N32_informed_native_batch32_review_summary.json"
    md_path = OUTPUT_DIR / "N32_informed_native_batch32_review_summary.md"
    write_csv(csv_path, rows)
    write_json(json_path, payload)
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path, payload


def write_claim_boundary() -> tuple[Path, Path]:
    safe = [
        "Run34 packages selected Run33 Option A N32-informed native batch32 candidates for human review and future CAE generation.",
        "The selected batch contains N12=4, N16=4, N24=12, N40=12.",
        "No N32 candidates are included in the selected batch.",
        "The candidates were generated after N32 legacy-compatible training data was inserted into the modelling workflow.",
        "Handoff files include scan orders, metadata, future CAE manifest template, and abqjobpilot command template.",
        "No CAE/INP files were generated.",
    ]
    unsafe = [
        "candidates are teacher-validated.",
        "N32 caused performance improvement.",
        "physical superiority.",
        "GNN-RL has beaten baselines.",
        "online RL with Abaqus.",
        "arbitrary-N generalization.",
        "surrogate/GNN/hybrid predictions are ground truth.",
        "abqjobpilot commands are ready to execute.",
        "CAE/INP files exist.",
    ]
    md = OUTPUT_DIR / "run34_claim_boundary.md"
    js = OUTPUT_DIR / "run34_claim_boundary.json"
    md.write_text("# Run34 Claim Boundary\n\n## Safe Claims\n" + "\n".join(f"- {item}" for item in safe) + "\n\n## Unsafe Claims\n" + "\n".join(f"- Do not claim {item}" for item in unsafe) + "\n", encoding="utf-8")
    write_json(js, {"verdict": "RUN34_N32_INFORMED_NATIVE_BATCH32_HANDOFF_ONLY_NO_CAE_NO_TEACHER_VALIDATION", "safe_claims": safe, "unsafe_claims": unsafe})
    return md, js


def write_report(validation: dict[str, Any], handoff: list[dict[str, Any]], candidate_csv: Path, manifest_template: Path, commands_template: Path, review_md: Path, outputs: list[str]) -> None:
    counts = Counter(row["n"] for row in handoff)
    lines = [
        "# Stage 3 Run 34 - Run33 N32-Informed Native Batch32 Handoff Package",
        "",
        "## Purpose",
        "Create a clean handoff package for selected Run33 Option A, suitable for future CAE generation after user approval.",
        "",
        "## Inputs",
        f"- Selected Option A candidate orders: `{RUN33_OPTION_A}`",
        f"- Run33 candidate pool: `{RUN33_POOL}`",
        f"- Run33 report: `{RUN33_REPORT}`",
        f"- Run32A report: `{RUN32A_REPORT}`",
        "",
        "## User-Selected Batch",
        "- Selected batch: `Run33 Option A - N32-informed native batch32`.",
        "- Option B and Option C are not packaged as the selected batch.",
        "",
        "## Why Option A Was Selected",
        "Run33 showed N32 augmentation improved GNN relative to Run29 but did not clearly improve native Stage 3 surrogate performance overall. Option A uses N32-informed modelling while testing only native Stage 3 N values.",
        "",
        "## Validation Status",
        f"- Verdict: `{validation['verdict']}`",
        f"- Counts: `{dict(sorted(counts.items()))}`",
        "- No N32 candidates are included.",
        "",
        "## Stable Naming Convention",
        "- `S3R34N32INF_N{N}_B{index:02d}_{short_bucket_or_family}`",
        "",
        "## Candidate-Order Handoff Package",
        f"- `{candidate_csv}`",
        f"- Rows: `{len(handoff)}`",
        "",
        "## Per-Candidate Scan-Order JSON Outputs",
        f"- Directory: `{SCAN_ORDER_DIR}`",
        "",
        "## Future CAE Handoff Template",
        f"- `{manifest_template}`",
        "- Template only; no CAE directories or INP files were created.",
        "",
        "## Future abqjobpilot Command Template",
        f"- `{commands_template}`",
        "- Template only; not ready to execute until future CAE/INP generation has completed and passed checks.",
        "",
        "## Review Summary",
        f"- `{review_md}`",
        "",
        "## Superseded Run31 Note",
        f"- Superseded note exists: `{RUN31_SUPERSEDED_NOTE.exists()}`",
        "- The old Run31 READY_TO_RUN commands remain superseded and must not be enqueued unless explicitly re-approved later.",
        "",
        "## Claim Boundary",
        "`RUN34_N32_INFORMED_NATIVE_BATCH32_HANDOFF_ONLY_NO_CAE_NO_TEACHER_VALIDATION`.",
        "",
        "## Output Files",
        *[f"- `{p}`" for p in outputs],
        "",
        "## Recommended Run35",
        "CAE module should generate CAE/INP/JNL for selected Run34 N32-informed native batch32 only. Do not run solver, do not execute abqjobpilot, and do not generate Option B or Option C unless explicitly selected later.",
    ]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def git_branch() -> str:
    try:
        return subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


def update_run_index(verdict: str) -> None:
    if not RUN_INDEX_PATH.exists():
        return
    text = RUN_INDEX_PATH.read_text(encoding="utf-8")
    if RUN_ID in text:
        return
    row = (
        "| run_34 | Run33 N32-informed native batch32 handoff package | "
        "Package selected Run33 Option A native N12/N16/N24/N40 batch32 with scan orders, metadata, future CAE manifest template, and abqjobpilot command template. | "
        "`scripts/stage3/run_34_create_run33_N32_informed_native_batch32_handoff_package.py` | "
        "`docs/stage3/runs/run_34_run33_N32_informed_native_batch32_handoff_package/RUN_34_RUN33_N32_INFORMED_NATIVE_BATCH32_HANDOFF_PACKAGE_REPORT.md` | "
        "`outputs/stage3_run_34_run33_N32_informed_native_batch32_handoff_package/` | "
        f"`{verdict}` | No Abaqus, no ODB opening, no abqjobpilot, no CAE/INP/JNL generation, no teacher validation, no training, no candidate generation, no commit/push. |"
    )
    RUN_INDEX_PATH.write_text(text.rstrip() + "\n" + row + "\n", encoding="utf-8")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SCAN_ORDER_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = read_csv(RUN33_OPTION_A)
    validation = validate_selected_batch(rows)
    if validation["verdict"].startswith("FAIL"):
        print(validation["verdict"])
        return 2
    handoff = build_handoff(rows)
    candidate_csv = OUTPUT_DIR / "stage3_run34_N32_informed_native_batch32_candidate_orders.csv"
    write_csv(candidate_csv, handoff)
    scan_json_paths = write_scan_order_jsons(handoff)
    manifest_template, commands_template = write_future_templates(handoff)
    review_csv, review_json, review_md, review_payload = write_review_summary(handoff)
    claim_md, claim_json = write_claim_boundary()
    outputs = [
        str(OUTPUT_DIR / "run34_input_validation_summary.json"),
        str(candidate_csv),
        *scan_json_paths,
        str(manifest_template),
        str(commands_template),
        str(review_csv),
        str(review_json),
        str(review_md),
        str(claim_md),
        str(claim_json),
    ]
    write_report(validation, handoff, candidate_csv, manifest_template, commands_template, review_md, outputs)
    outputs.append(str(REPORT_PATH))
    update_run_index(validation["verdict"])
    manifest = {
        "run_id": RUN_ID,
        "run_name": RUN_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "branch": git_branch(),
        "script_path": str(ROOT / "scripts" / "stage3" / "run_34_create_run33_N32_informed_native_batch32_handoff_package.py"),
        "input_files": [str(path) for path in [RUN33_OPTION_A, RUN33_POOL, RUN33_REPORT, RUN33_MANIFEST, RUN32A_REPORT] if path.exists()],
        "output_files": outputs,
        "selected_batch": SELECTED_BATCH,
        "batch_name": BATCH_NAME,
        "batch32_count": len(handoff),
        "per_N_counts": dict(Counter(row["n"] for row in handoff)),
        "includes_N32_candidates": False,
        "N32_informed_training_context": True,
        "supersedes_Run31_old_batch": True,
        "report_path": str(REPORT_PATH),
        "claim_boundary_path": str(claim_md),
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
    print(validation["verdict"])
    print(f"selected_batch={SELECTED_BATCH}")
    print(f"batch32={len(handoff)} per_n={dict(Counter(row['n'] for row in handoff))}")
    print(f"candidate_csv={candidate_csv}")
    print(f"scan_orders={SCAN_ORDER_DIR}")
    print(f"report={REPORT_PATH}")
    print(f"manifest={MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
