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
RUN_ID = "run_30_run29_hybrid_policy_batch32_handoff_package"
RUN_NAME = "run29 hybrid-policy batch32 handoff package"

RUN29_BATCH32 = ROOT / "outputs" / "stage3_run_29_combined172_surrogate_gnn_hybrid_policy_update_and_candidate_generation" / "run29_hybrid_policy_batch32_candidate_orders.csv"
RUN29_POOL = ROOT / "outputs" / "stage3_run_29_combined172_surrogate_gnn_hybrid_policy_update_and_candidate_generation" / "run29_hybrid_candidate_pool_scored.csv"
RUN29_BATCH64 = ROOT / "outputs" / "stage3_run_29_combined172_surrogate_gnn_hybrid_policy_update_and_candidate_generation" / "run29_hybrid_policy_batch64_candidate_orders.csv"
RUN29_FOCUSED48 = ROOT / "outputs" / "stage3_run_29_combined172_surrogate_gnn_hybrid_policy_update_and_candidate_generation" / "run29_hybrid_policy_N24_N40_focused_batch48_candidate_orders.csv"
RUN29_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_29_combined172_surrogate_gnn_hybrid_policy_update_and_candidate_generation" / "RUN_29_COMBINED172_SURROGATE_GNN_HYBRID_POLICY_UPDATE_AND_CANDIDATE_GENERATION_REPORT.md"
COMBINED172_READY = ROOT / "outputs" / "stage3_run_28_shortlist64_teacher_metrics_ingestion_and_combined172_ranking" / "combined172_RL_ready_dataset.csv"

OUTPUT_DIR = ROOT / "outputs" / "stage3_run_30_run29_hybrid_policy_batch32_handoff_package"
SCAN_ORDER_DIR = OUTPUT_DIR / "scan_orders"
REPORT_DIR = ROOT / "docs" / "stage3" / "runs" / RUN_ID
REPORT_PATH = REPORT_DIR / "RUN_30_RUN29_HYBRID_POLICY_BATCH32_HANDOFF_PACKAGE_REPORT.md"
MANIFEST_PATH = ROOT / "artifacts" / "manifests" / "stage3_run_30_manifest.json"
RUN_INDEX_PATH = ROOT / "docs" / "stage3" / "STAGE3_RUN_INDEX.md"

EXPECTED_N = [12, 16, 24, 40]
BATCH32_COUNTS = {12: 4, 16: 4, 24: 12, 40: 12}
SELECTED_BATCH = "run29_hybrid_policy_batch32"
BATCH_OPTION = "hybrid_policy_batch32"
BATCH_NAME = "stage3_run30_hybrid_policy_batch32_v01"
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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


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
        return False, f"length {len(order)} != N {n}"
    if len(set(order)) != n:
        return False, "duplicate track ids"
    expected = set(range(n))
    actual = set(order)
    if actual != expected:
        return False, f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
    return True, "legal permutation"


def safe_token(text: Any, fallback: str = "candidate", max_len: int = 32) -> str:
    token = re.sub(r"[^A-Za-z0-9_]+", "_", str(text).strip())
    token = re.sub(r"_+", "_", token).strip("_").lower()
    return (token or fallback)[:max_len]


def bucket_token(row: dict[str, str]) -> str:
    mapping = {
        "gnn_policy_top_candidates": "gnn_policy_top",
        "surrogate_known_best_local_search": "surrogate_local",
        "new_best_local_search": "new_best_local",
        "hybrid_gnn_surrogate_agreement": "hybrid_agreement",
        "hybrid_gnn_surrogate_disagreement": "hybrid_disagreement",
        "uncertainty_calibration": "uncertainty",
        "diversity_coverage": "diversity",
        "tradeoff_probe": "tradeoff",
        "sentinel_control": "sentinel",
        "N24_surfaceT_best_neighborhood": "n24_surfaceT",
    }
    bucket = str(row.get("selection_bucket", "")).strip()
    source = str(row.get("candidate_source", "")).strip()
    return safe_token(mapping.get(bucket, bucket or source), "hybrid", 28)


def combined172_hashes() -> dict[int, set[str]]:
    hashes: dict[int, set[str]] = defaultdict(set)
    if not COMBINED172_READY.exists():
        return hashes
    for row in read_csv(COMBINED172_READY):
        try:
            n = parse_int(row.get("n"))
        except Exception:
            continue
        order = parse_order(row.get("order_json"))
        if order is not None:
            hashes[n].add(order_hash(order))
    return hashes


def validate_batch32(rows: list[dict[str, str]]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    counts: Counter[int] = Counter()
    candidate_ids: Counter[str] = Counter()
    seen_orders: dict[int, set[str]] = defaultdict(set)
    teacher_hashes = combined172_hashes()
    buckets: Counter[str] = Counter()
    sources: Counter[str] = Counter()

    if not RUN29_BATCH32.exists():
        errors.append(f"Missing selected batch file: {RUN29_BATCH32}")

    for row in rows:
        try:
            n = parse_int(row.get("n"))
        except Exception:
            errors.append(f"invalid n={row.get('n')}")
            continue
        counts[n] += 1
        cid = str(row.get("candidate_id", "")).strip()
        strategy = str(row.get("strategy_name", "")).strip()
        if not cid and not strategy:
            errors.append("row missing candidate_id/strategy_name")
        if cid:
            candidate_ids[cid] += 1
        order = parse_order(row.get("order_json"))
        legal, reason = validate_order(order, n)
        if not legal:
            errors.append(f"{cid or strategy}: {reason}")
            continue
        digest = row.get("order_hash") or order_hash(order or [])
        if digest in seen_orders[n]:
            errors.append(f"duplicate order within N{n}: {cid or strategy}")
        seen_orders[n].add(digest)
        if digest in teacher_hashes[n]:
            errors.append(f"{cid or strategy}: duplicates combined172 teacher order")
        for col in ["surrogate_reward_pred", "gnn_reward_pred", "hybrid_score", "gnn_surrogate_disagreement", "novelty_distance_to_combined172"]:
            if not math.isfinite(parse_float(row.get(col))):
                warnings.append(f"{cid or strategy}: missing optional numeric metadata {col}")
        if not row.get("selection_bucket"):
            warnings.append(f"{cid or strategy}: missing selection_bucket")
        if not row.get("candidate_source"):
            warnings.append(f"{cid or strategy}: missing candidate_source")
        buckets[row.get("selection_bucket", "")] += 1
        sources[row.get("candidate_source", "")] += 1

    if len(rows) != 32:
        errors.append(f"Expected 32 rows, found {len(rows)}")
    for n, expected in BATCH32_COUNTS.items():
        if counts[n] != expected:
            errors.append(f"Expected N{n}={expected}, found {counts[n]}")
    if any(count > 1 for count in candidate_ids.values()):
        duplicates = [cid for cid, count in candidate_ids.items() if count > 1]
        errors.append(f"Duplicate candidate_id values: {duplicates}")
    verdict = "PASS_RUN30_HYBRID_BATCH32_INPUT_READY" if not errors else "FAIL_RUN30_HYBRID_BATCH32_INPUT_INVALID"
    payload = {
        "verdict": verdict,
        "errors": errors,
        "warnings": warnings[:25],
        "warning_count": len(warnings),
        "selected_batch": SELECTED_BATCH,
        "row_count": len(rows),
        "per_n_counts": dict(sorted(counts.items())),
        "candidate_source_composition": dict(sources),
        "selection_bucket_composition": dict(buckets),
        "selected_batch_confirmed_not_batch64_or_focused48": True,
    }
    write_json(OUTPUT_DIR / "run30_input_validation_summary.json", payload)
    return payload


def handoff_name(n: int, index: int, row: dict[str, str]) -> str:
    return f"S3R30H32_N{n}_B{index:02d}_{bucket_token(row)}"


def build_handoff(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    handoff: list[dict[str, Any]] = []
    by_n: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_n[parse_int(row["n"])].append(row)
    for n in EXPECTED_N:
        group = sorted(by_n[n], key=lambda row: parse_float(row.get("batch32_rank_within_n"), 999.0))
        for index, row in enumerate(group, start=1):
            order = parse_order(row.get("order_json"))
            digest = row.get("order_hash") or order_hash(order or [])
            name = handoff_name(n, index, row)
            handoff.append(
                {
                    "run_id": RUN_ID,
                    "batch_option": BATCH_OPTION,
                    "batch_name": BATCH_NAME,
                    "n": n,
                    "handoff_strategy_name": name,
                    "original_run29_candidate_id": row.get("candidate_id", ""),
                    "original_run29_strategy_name": row.get("strategy_name", ""),
                    "candidate_source": row.get("candidate_source", ""),
                    "generation_method": row.get("generation_method", ""),
                    "selection_bucket": row.get("selection_bucket", ""),
                    "priority_role": row.get("priority_role", ""),
                    "surrogate_prediction": parse_float(row.get("surrogate_reward_pred")),
                    "gnn_reward_prediction": parse_float(row.get("gnn_reward_pred")),
                    "graph_pointer_policy_score": parse_float(row.get("graph_pointer_mean_logprob")),
                    "hybrid_score": parse_float(row.get("hybrid_score")),
                    "uncertainty_score": parse_float(row.get("pred_uncertainty_ET_F01_std")),
                    "gnn_vs_surrogate_disagreement": parse_float(row.get("gnn_surrogate_disagreement")),
                    "novelty_distance_to_combined172": parse_float(row.get("novelty_distance_to_combined172")),
                    "nearest_existing_teacher_strategy": row.get("nearest_existing_teacher_strategy", ""),
                    "pred_u2_score": parse_float(row.get("pred_u2_score")),
                    "pred_peeq_score": parse_float(row.get("pred_peeq_score")),
                    "pred_surfaceT_score": parse_float(row.get("pred_surfaceT_score")),
                    "pred_mises_score": parse_float(row.get("pred_mises_score")),
                    "surrogate_rank_within_n": parse_float(row.get("surrogate_reward_pred_rank_within_n")),
                    "gnn_rank_within_n": parse_float(row.get("gnn_reward_pred_rank_within_n")),
                    "hybrid_rank_within_n": parse_float(row.get("hybrid_score_rank_within_n")),
                    "batch32_rank_within_n": parse_float(row.get("batch32_rank_within_n")),
                    "order_json": row.get("order_json", ""),
                    "order_compact": row.get("order_compact", ""),
                    "order_hash": digest,
                    "teacher_validated": False,
                    "teacher_validation_status": "NOT_RUN",
                    "notes": "Run30 handoff only. Hybrid-policy candidate is not teacher-validated.",
                }
            )
    return handoff


def write_scan_order_jsons(handoff: list[dict[str, Any]]) -> list[str]:
    SCAN_ORDER_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for row in handoff:
        order = parse_order(row.get("order_json"))
        payload = {
            "run_id": RUN_ID,
            "batch_option": BATCH_OPTION,
            "batch_name": BATCH_NAME,
            "n": row["n"],
            "handoff_strategy_name": row["handoff_strategy_name"],
            "original_run29_candidate_id": row["original_run29_candidate_id"],
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
            "novelty_distance_to_combined172": row["novelty_distance_to_combined172"],
            "nearest_existing_teacher_strategy": row["nearest_existing_teacher_strategy"],
            "scan_order": order,
            "order_hash": row["order_hash"],
            "teacher_validated": False,
            "teacher_validation_status": "NOT_RUN",
            "notes": "Run30 handoff only. Hybrid-policy candidate is not teacher-validated.",
        }
        path = SCAN_ORDER_DIR / f"scan_order_{row['handoff_strategy_name']}.json"
        write_json(path, payload)
        paths.append(str(path))
    return paths


def write_future_templates(handoff: list[dict[str, Any]]) -> tuple[Path, Path]:
    manifest_rows: list[dict[str, Any]] = []
    commands: list[str] = [
        "# TEMPLATE ONLY - do not execute until CAE/INP generation has completed and passed checks.",
        "# INP files referenced below do not exist as part of Run30.",
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
                "notes": "Run30 template only. Do not run until CAE/INP exists and passes checks.",
            }
        )
        commands.append(f'enqueue --inp "{case_dir}\\{job}.inp" --cpus 14 --batch {BATCH_NAME} --strategy {name}')
    manifest_path = OUTPUT_DIR / "stage3_run30_hybrid_policy_batch32_future_cae_handoff_manifest_TEMPLATE.csv"
    commands_path = OUTPUT_DIR / "stage3_run30_hybrid_policy_batch32_abqjobpilot_commands_TEMPLATE.txt"
    write_csv(manifest_path, manifest_rows)
    commands_path.write_text("\n".join(commands) + "\n", encoding="utf-8")
    return manifest_path, commands_path


def mean(values: list[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return sum(clean) / len(clean) if clean else math.nan


def review_summary(handoff: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    rows: list[dict[str, Any]] = []
    for n in EXPECTED_N:
        group = [row for row in handoff if row["n"] == n]
        rows.append(
            {
                "n": n,
                "count": len(group),
                "mean_surrogate_prediction": mean([row["surrogate_prediction"] for row in group]),
                "mean_gnn_reward_prediction": mean([row["gnn_reward_prediction"] for row in group]),
                "mean_hybrid_score": mean([row["hybrid_score"] for row in group]),
                "mean_gnn_vs_surrogate_disagreement": mean([row["gnn_vs_surrogate_disagreement"] for row in group]),
                "mean_novelty_distance_to_combined172": mean([row["novelty_distance_to_combined172"] for row in group]),
                "candidate_source_composition": dict(Counter(row["candidate_source"] for row in group)),
                "selection_bucket_composition": dict(Counter(row["selection_bucket"] for row in group)),
            }
        )
    all_sources = Counter(row["candidate_source"] for row in handoff)
    all_buckets = Counter(row["selection_bucket"] for row in handoff)
    payload = {
        "total_count": len(handoff),
        "per_n_counts": dict(Counter(row["n"] for row in handoff)),
        "n24_n40_share": sum(1 for row in handoff if row["n"] in (24, 40)) / max(1, len(handoff)),
        "candidate_source_composition": dict(all_sources),
        "selection_bucket_composition": dict(all_buckets),
        "expected_abaqus_cost": "32 jobs total, with 24 jobs from N24/N40",
        "interpretation": "hybrid batch32 is selected for a faster daytime validation loop; it is not teacher-validated until future Abaqus teacher validation.",
        "no_cae_inp_generated": True,
    }
    lines = [
        "# Run30 Hybrid Batch32 Review Summary",
        "",
        "- Selected batch: `run29_hybrid_policy_batch32`",
        "- Total count: `32`",
        "- Per-N counts: `N12=4`, `N16=4`, `N24=12`, `N40=12`",
        "- N24/N40 share: `24/32`",
        f"- Candidate-source composition: `{dict(all_sources)}`",
        f"- Selection-bucket composition: `{dict(all_buckets)}`",
        "- Expected Abaqus cost: 32 jobs total, with 24 jobs from N24/N40.",
        "- Hybrid batch32 is selected because the user wants a faster daytime validation loop.",
        "- Hybrid batch32 remains unvalidated until future Abaqus teacher validation.",
        "- Run30 did not create CAE/INP files.",
        "",
    ]
    return rows, payload, "\n".join(lines)


def write_claim_boundary() -> tuple[Path, Path]:
    md = OUTPUT_DIR / "run30_claim_boundary.md"
    js = OUTPUT_DIR / "run30_claim_boundary.json"
    safe = [
        "Run30 packages selected Run29 hybrid-policy batch32 candidates for human review and future CAE generation.",
        "The selected batch contains N12=4, N16=4, N24=12, N40=12.",
        "The candidates originate from the Run29 combined172 surrogate/GNN/hybrid policy update.",
        "Handoff files include scan orders, metadata, future CAE manifest template, and abqjobpilot command template.",
        "No CAE/INP files were generated.",
    ]
    unsafe = [
        "candidates are teacher-validated.",
        "physical superiority.",
        "GNN-RL has beaten baselines.",
        "online RL with Abaqus.",
        "arbitrary-N generalization.",
        "surrogate/GNN/hybrid predictions are ground truth.",
        "abqjobpilot commands are ready to execute.",
        "CAE/INP files exist.",
    ]
    md.write_text("# Run30 Claim Boundary\n\n## Safe Claims\n" + "\n".join(f"- {item}" for item in safe) + "\n\n## Unsafe Claims\n" + "\n".join(f"- Do not claim {item}" for item in unsafe) + "\n", encoding="utf-8")
    write_json(js, {"verdict": "RUN30_HYBRID_BATCH32_HANDOFF_ONLY_NO_CAE_NO_TEACHER_VALIDATION", "safe_claims": safe, "unsafe_claims": unsafe})
    return md, js


def write_report(validation: dict[str, Any], handoff: list[dict[str, Any]], manifest_template: Path, commands_template: Path, review_md: Path, outputs: list[str]) -> None:
    counts = Counter(row["n"] for row in handoff)
    lines = [
        "# Stage 3 Run 30 - Run29 Hybrid-Policy Batch32 Handoff Package",
        "",
        "## Purpose",
        "Create a clean handoff package for the selected Run29 hybrid-policy batch32, suitable for future CAE generation after user approval.",
        "",
        "## Inputs",
        f"- Selected Run29 batch32: `{RUN29_BATCH32}`",
        f"- Run29 candidate pool: `{RUN29_POOL}`",
        f"- Run29 report: `{RUN29_REPORT}`",
        "",
        "## User-Selected Batch",
        f"- Selected batch: `{SELECTED_BATCH}`",
        "- Batch64 and focused batch48 are reference-only and were not packaged as the selected batch.",
        "",
        "## Validation Status",
        f"- Verdict: `{validation['verdict']}`",
        f"- Per-N counts: `{dict(counts)}`",
        "",
        "## Stable Naming Convention",
        "- Format: `S3R30H32_N{N}_B{index:02d}_{short_bucket_or_family}`",
        f"- Batch name: `{BATCH_NAME}`",
        "",
        "## Hybrid-Policy Batch32 Handoff Package",
        f"- Candidate-order CSV rows: `{len(handoff)}`",
        "- Metadata preserves Run29 candidate IDs, source, generation method, selection bucket, surrogate/GNN/hybrid scores, disagreement, novelty, and nearest-teacher information where available.",
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
        "- Template only; commands are not executable until INPs exist and pass checks.",
        "",
        "## Hybrid Batch32 Review Summary",
        f"- `{review_md}`",
        "",
        "## Claim Boundary",
        "`RUN30_HYBRID_BATCH32_HANDOFF_ONLY_NO_CAE_NO_TEACHER_VALIDATION`.",
        "",
        "## Output Files",
    ]
    lines.extend(f"- `{path}`" for path in outputs)
    lines.extend(
        [
            "",
            "## Recommended Run31",
            "CAE module should generate CAE/INP/JNL for selected hybrid-policy batch32 only. Do not run solver, do not execute abqjobpilot, and do not generate hybrid batch64 or focused batch48 unless explicitly selected later.",
            "",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def update_run_index(verdict: str) -> None:
    if not RUN_INDEX_PATH.exists():
        return
    text = RUN_INDEX_PATH.read_text(encoding="utf-8")
    if "| run_30 |" in text:
        return
    row = (
        "| run_30 | Run29 hybrid-policy batch32 handoff package | "
        "Package selected Run29 hybrid-policy batch32 with scan orders, metadata, future CAE manifest template, and abqjobpilot command template. | "
        "`scripts/stage3/run_30_create_run29_hybrid_policy_batch32_handoff_package.py` | "
        "`docs/stage3/runs/run_30_run29_hybrid_policy_batch32_handoff_package/RUN_30_RUN29_HYBRID_POLICY_BATCH32_HANDOFF_PACKAGE_REPORT.md` | "
        "`outputs/stage3_run_30_run29_hybrid_policy_batch32_handoff_package/` | "
        f"`{verdict}` | No Abaqus, no ODB opening, no abqjobpilot, no CAE/INP/JNL generation, no teacher validation, no online RL, no commit/push. |"
    )
    RUN_INDEX_PATH.write_text(text.rstrip() + "\n" + row + "\n", encoding="utf-8")


def git_branch() -> str:
    try:
        return subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SCAN_ORDER_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = read_csv(RUN29_BATCH32)
    validation = validate_batch32(rows)
    if validation["verdict"].startswith("FAIL"):
        print(validation["verdict"])
        return 2

    handoff = build_handoff(rows)
    candidate_csv = OUTPUT_DIR / "stage3_run30_hybrid_policy_batch32_candidate_orders.csv"
    write_csv(candidate_csv, handoff)
    scan_json_paths = write_scan_order_jsons(handoff)
    manifest_template, commands_template = write_future_templates(handoff)
    review_rows, review_payload, review_md_text = review_summary(handoff)
    review_csv = OUTPUT_DIR / "hybrid_batch32_review_summary.csv"
    review_json = OUTPUT_DIR / "hybrid_batch32_review_summary.json"
    review_md = OUTPUT_DIR / "hybrid_batch32_review_summary.md"
    write_csv(review_csv, review_rows)
    write_json(review_json, review_payload)
    review_md.write_text(review_md_text, encoding="utf-8")
    claim_md, claim_json = write_claim_boundary()

    outputs = [
        str(OUTPUT_DIR / "run30_input_validation_summary.json"),
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
    write_report(validation, handoff, manifest_template, commands_template, review_md, outputs)
    outputs.append(str(REPORT_PATH))
    update_run_index(validation["verdict"])

    manifest = {
        "run_id": RUN_ID,
        "run_name": RUN_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "branch": git_branch(),
        "script_path": str(ROOT / "scripts" / "stage3" / "run_30_create_run29_hybrid_policy_batch32_handoff_package.py"),
        "input_files": [str(path) for path in [RUN29_BATCH32, RUN29_POOL, RUN29_BATCH64, RUN29_FOCUSED48, RUN29_REPORT, COMBINED172_READY] if path.exists()],
        "output_files": outputs,
        "selected_batch": SELECTED_BATCH,
        "batch_name": BATCH_NAME,
        "batch32_count": len(handoff),
        "per_N_counts": dict(Counter(row["n"] for row in handoff)),
        "report_path": str(REPORT_PATH),
        "claim_boundary_path": str(claim_md),
        "no_solver_run": True,
        "no_odb_opened": True,
        "no_abqjobpilot_run": True,
        "no_cae_inp_generated": True,
        "no_teacher_validation": True,
        "no_online_rl": True,
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
