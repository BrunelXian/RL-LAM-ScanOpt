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
RUN_ID = "run_13_batch20_surrogate_screened_teacher_handoff"
RUN_NAME = "controlled teacher-validation handoff package for run12 batch20 candidates"
BATCH_NAME = "stage3_run13_batch20_surrogate_screened_v01"

INPUT_BATCH20 = ROOT / "outputs" / "stage3_run_12_offline_surrogate_screened_candidate_generation" / "run12_recommended_future_teacher_batch20.csv"
INPUT_SCORED_POOL = ROOT / "outputs" / "stage3_run_12_offline_surrogate_screened_candidate_generation" / "run12_candidate_pool_scored.csv"
INPUT_RUN12_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_12_offline_surrogate_screened_candidate_generation" / "RUN_12_OFFLINE_SURROGATE_SCREENED_CANDIDATE_GENERATION_REPORT.md"
INPUT_TEACHER_LABELS = ROOT / "outputs" / "stage3_run_09_variable_n_probe60_teacher_ranking_analysis" / "probe60_teacher_ranked_canonical.csv"

OUTPUT_DIR = ROOT / "outputs" / "stage3_run_13_batch20_surrogate_screened_teacher_handoff"
SCAN_ORDER_DIR = OUTPUT_DIR / "scan_orders"
REPORT_DIR = ROOT / "docs" / "stage3" / "runs" / RUN_ID
REPORT_PATH = REPORT_DIR / "RUN_13_BATCH20_SURROGATE_SCREENED_TEACHER_HANDOFF_REPORT.md"
MANIFEST_PATH = ROOT / "artifacts" / "manifests" / "stage3_run_13_manifest.json"
RUN_INDEX_PATH = ROOT / "docs" / "stage3" / "STAGE3_RUN_INDEX.md"
FUTURE_CAE_ROOT = ROOT / "cae_model" / BATCH_NAME

EXPECTED_N = [12, 16, 24, 40]


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
    return int(text)


def parse_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def parse_order(text: str) -> list[int] | None:
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(value, list):
        return None
    order = []
    for item in value:
        if isinstance(item, bool):
            return None
        try:
            order.append(int(item))
        except (TypeError, ValueError):
            return None
    return order


def order_hash(order: list[int]) -> str:
    return hashlib.sha1(",".join(str(x) for x in order).encode("ascii")).hexdigest()[:16]


def validate_order(order: list[int] | None, n: int) -> tuple[bool, str]:
    if order is None:
        return False, "order could not be parsed"
    if len(order) != n:
        return False, f"length {len(order)} != N {n}"
    expected = set(range(n))
    actual = set(order)
    if len(actual) != len(order):
        return False, "duplicate tracks"
    if actual != expected:
        return False, f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
    return True, "legal permutation"


def safe_token(text: str, fallback: str = "candidate") -> str:
    token = re.sub(r"[^A-Za-z0-9_]+", "_", text.strip())
    token = re.sub(r"_+", "_", token).strip("_").lower()
    return token[:32] or fallback


def short_bucket(row: dict[str, str]) -> str:
    bucket = row.get("selection_bucket", "")
    family = row.get("candidate_family", "")
    if bucket == "negative_control_sentinel":
        return "control_sentinel"
    if bucket:
        return safe_token(bucket)
    return safe_token(family)


def load_teacher_best() -> tuple[dict[int, dict[str, Any]], dict[int, set[str]]]:
    rows = read_csv(INPUT_TEACHER_LABELS)
    by_n: dict[int, list[dict[str, str]]] = defaultdict(list)
    teacher_hashes: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        n = parse_int(row["n"])
        by_n[n].append(row)
        order = parse_order(row.get("order_json", "") or row.get("raw_scan_order", ""))
        if order is not None:
            teacher_hashes[n].add(order_hash(order))
    best: dict[int, dict[str, Any]] = {}
    for n, group in by_n.items():
        winner = min(group, key=lambda row: parse_float(row.get("constrained_rank_within_n"), default=999.0))
        # Run10 reward_mean_all is not in run09, so use constrained rank best and attach simple rank.
        best[n] = {
            "strategy_name": winner["strategy_name"],
            "reward_proxy": parse_float(winner.get("simple_mean_rank"), default=math.nan),
            "constrained_rank_within_n": parse_float(winner.get("constrained_rank_within_n"), default=math.nan),
        }
    # Prefer run10 reward values if present in batch20 comparisons? Not an input here; keep report wording surrogate-only.
    return best, teacher_hashes


def validate_inputs(batch_rows: list[dict[str, str]], teacher_hashes: dict[int, set[str]]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    counts = Counter()
    ids = Counter()
    order_seen: dict[int, set[str]] = defaultdict(set)
    for row in batch_rows:
        try:
            n = parse_int(row.get("n"))
        except (TypeError, ValueError):
            errors.append(f"Invalid N: {row.get('n')}")
            continue
        counts[n] += 1
        candidate_id = row.get("candidate_id", "")
        ids[candidate_id] += 1
        if not candidate_id:
            errors.append("Missing candidate_id")
        order = parse_order(row.get("order_json", ""))
        legal, reason = validate_order(order, n)
        if not legal:
            errors.append(f"{candidate_id}: {reason}")
            continue
        h = order_hash(order or [])
        if h in order_seen[n]:
            errors.append(f"Duplicate order within N{n}: {candidate_id}")
        order_seen[n].add(h)
        if h in teacher_hashes[n] and not parse_bool(row.get("is_existing_teacher_order")):
            errors.append(f"{candidate_id}: duplicates existing teacher order without existing_reference mark")
        for col in ["pred_reward_mean_all", "pred_rank_within_n"]:
            if not math.isfinite(parse_float(row.get(col))):
                errors.append(f"{candidate_id}: missing {col}")
    if len(batch_rows) != 20:
        errors.append(f"Expected 20 rows, found {len(batch_rows)}")
    if sorted(counts) != EXPECTED_N:
        errors.append(f"Expected N values {EXPECTED_N}, found {sorted(counts)}")
    for n in EXPECTED_N:
        if counts[n] != 5:
            errors.append(f"Expected 5 candidates for N{n}, found {counts[n]}")
    duplicate_ids = [candidate_id for candidate_id, count in ids.items() if count > 1]
    if duplicate_ids:
        errors.append(f"Duplicate candidate_id values: {duplicate_ids}")
    return {
        "verdict": "PASS_RUN13_BATCH20_INPUT_READY" if not errors else "FAIL_RUN13_BATCH20_INPUT_INVALID",
        "errors": errors,
        "warnings": warnings,
        "row_count": len(batch_rows),
        "per_n_counts": dict(sorted(counts.items())),
        "duplicate_candidate_ids": duplicate_ids,
    }


def build_handoff_rows(batch_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    handoff: list[dict[str, Any]] = []
    for n in EXPECTED_N:
        group = sorted([row for row in batch_rows if parse_int(row["n"]) == n], key=lambda row: parse_int(row.get("teacher_batch20_rank_within_n", "999")))
        for idx, row in enumerate(group, start=1):
            order = parse_order(row["order_json"]) or []
            bucket = short_bucket(row)
            handoff_name = f"S3B20_N{n}_B{idx:02d}_{bucket}"
            if len(handoff_name) > 64:
                handoff_name = handoff_name[:64].rstrip("_")
            h = order_hash(order)
            handoff.append(
                {
                    "run_id": RUN_ID,
                    "batch_name": BATCH_NAME,
                    "n": n,
                    "handoff_strategy_name": handoff_name,
                    "original_run12_candidate_id": row.get("candidate_id", ""),
                    "original_run12_strategy_name": row.get("strategy_name", ""),
                    "candidate_family": row.get("candidate_family", ""),
                    "candidate_source": row.get("candidate_source", ""),
                    "selection_bucket": row.get("selection_bucket", ""),
                    "predicted_reward_mean_all": row.get("pred_reward_mean_all", ""),
                    "predicted_rank_within_n": row.get("pred_rank_within_n", ""),
                    "predicted_uncertainty_std": row.get("pred_uncertainty_std", ""),
                    "novelty_distance_to_teacher": row.get("novelty_distance_to_nearest_existing", ""),
                    "nearest_existing_teacher_strategy": row.get("nearest_existing_strategy", ""),
                    "order_json": json.dumps(order, separators=(",", ":")),
                    "order_compact": ",".join(str(x) for x in order),
                    "order_hash": h,
                    "is_teacher_validated": False,
                    "teacher_validation_status": "NOT_RUN",
                    "notes": "Run13 handoff only. Not teacher-validated. Do not claim physical superiority.",
                }
            )
    return handoff


def write_scan_order_jsons(handoff_rows: list[dict[str, Any]]) -> list[str]:
    paths: list[str] = []
    SCAN_ORDER_DIR.mkdir(parents=True, exist_ok=True)
    for row in handoff_rows:
        path = SCAN_ORDER_DIR / f"scan_order_{row['handoff_strategy_name']}.json"
        payload = {
            "run_id": RUN_ID,
            "batch_name": BATCH_NAME,
            "n": row["n"],
            "handoff_strategy_name": row["handoff_strategy_name"],
            "original_run12_candidate_id": row["original_run12_candidate_id"],
            "candidate_family": row["candidate_family"],
            "selection_bucket": row["selection_bucket"],
            "predicted_reward_mean_all": parse_float(row["predicted_reward_mean_all"]),
            "predicted_rank_within_n": parse_float(row["predicted_rank_within_n"]),
            "predicted_uncertainty_std": parse_float(row["predicted_uncertainty_std"]),
            "novelty_distance_to_teacher": parse_float(row["novelty_distance_to_teacher"]),
            "scan_order": parse_order(row["order_json"]),
            "order_hash": row["order_hash"],
            "teacher_validated": False,
            "teacher_validation_status": "NOT_RUN",
            "notes": "Run13 handoff only. Not teacher-validated. Do not claim physical superiority.",
        }
        write_json(path, payload)
        paths.append(str(path))
    return paths


def build_future_cae_manifest(handoff_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in handoff_rows:
        n = row["n"]
        name = row["handoff_strategy_name"]
        case_dir = FUTURE_CAE_ROOT / f"N{n}{name}"
        scan_order_json = SCAN_ORDER_DIR / f"scan_order_{name}.json"
        job = f"J2D_{name}"
        rows.append(
            {
                "n": n,
                "handoff_strategy_name": name,
                "expected_case_dir": str(case_dir),
                "expected_job_name": job,
                "scan_order_json": str(scan_order_json),
                "expected_cae": str(case_dir / f"{job}.cae"),
                "expected_inp": str(case_dir / f"{job}.inp"),
                "expected_jnl": str(case_dir / f"{job}.jnl"),
                "expected_odb": str(case_dir / f"{job}.odb"),
                "teacher_validated": False,
                "generation_status": "NOT_GENERATED",
                "solver_status": "NOT_SUBMITTED",
                "notes": "Template path only; no CAE/INP/JNL/ODB exists from run13.",
            }
        )
    return rows


def build_abqjobpilot_commands(cae_rows: list[dict[str, Any]]) -> list[str]:
    commands = []
    for row in cae_rows:
        commands.append(
            f'enqueue --inp "{row["expected_inp"]}" --cpus 14 --batch {BATCH_NAME} --strategy {row["handoff_strategy_name"]}'
        )
    return commands


def best_existing_by_reward() -> dict[int, dict[str, Any]]:
    # Use run10 reward file if available; the user-provided input is run09, but run10 is local text and gives exact reward_mean_all.
    run10 = ROOT / "outputs" / "stage3_run_10_variable_n_normalized_reward_surrogate_dataset" / "probe60_variable_n_reward_dataset.csv"
    if run10.exists():
        rows = read_csv(run10)
        result = {}
        for n in EXPECTED_N:
            group = [row for row in rows if parse_int(row["n"]) == n]
            winner = max(group, key=lambda row: parse_float(row["reward_mean_all"]))
            result[n] = {"strategy_name": winner["strategy_name"], "reward_mean_all": parse_float(winner["reward_mean_all"])}
        return result
    best, _hashes = load_teacher_best()
    return {n: {"strategy_name": row["strategy_name"], "reward_mean_all": ""} for n, row in best.items()}


def compare_to_existing(handoff_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    existing = best_existing_by_reward()
    rows = []
    for n in EXPECTED_N:
        group = [row for row in handoff_rows if row["n"] == n]
        top = max(group, key=lambda row: parse_float(row["predicted_reward_mean_all"]))
        best_reward = existing[n]["reward_mean_all"]
        top_reward = parse_float(top["predicted_reward_mean_all"])
        rows.append(
            {
                "n": n,
                "existing_best_teacher_strategy": existing[n]["strategy_name"],
                "existing_best_teacher_reward_mean_all": best_reward,
                "top_predicted_batch20_candidate": top["handoff_strategy_name"],
                "top_predicted_batch20_reward": top_reward,
                "predicted_gap": top_reward - best_reward if isinstance(best_reward, float) and math.isfinite(best_reward) else "",
                "batch20_predicted_exceeds_existing_best": top_reward > best_reward if isinstance(best_reward, float) and math.isfinite(best_reward) else False,
                "note": "Surrogate-only prediction, not teacher validation.",
            }
        )
    return rows


def write_readme(path: Path) -> None:
    text = f"""# Stage 3 Run13 Batch20 CAE Generation Handoff

Run13 created a 20-candidate handoff package from run12 surrogate-screened candidates.

These candidates are not teacher-validated. They are active-learning/diversity probes, not guaranteed improvements.

The CAE module should generate true variable-N models using the corrected N12/N16/N24/N40 `sanity_base` models. It must preserve final cooling settings:

- `step_final_cooling` duration = `1200.0`
- `initialInc = 0.01`
- `maxInc = 60.0`

The CAE module should not run the solver until the user approves.

The abqjobpilot commands in `stage3_run13_batch20_abqjobpilot_commands_TEMPLATE.txt` are template-only. Do not run them until CAE/INP generation has completed and the generated INPs have been checked.
"""
    path.write_text(text, encoding="utf-8")


def write_claim_boundary(md: Path, js: Path) -> None:
    safe = [
        "Run13 packages 20 surrogate-screened candidates for possible future teacher validation.",
        "The batch is balanced across N12/N16/N24/N40 with 5 candidates each.",
        "The candidates are derived from run12 offline surrogate screening.",
        "The handoff includes scan orders, metadata, future CAE paths, and template abqjobpilot commands.",
        "The batch is suitable for human review and possible future CAE/teacher-validation setup.",
    ]
    unsafe = [
        "Do not claim candidates are teacher-validated.",
        "Do not claim physical superiority.",
        "Do not claim surrogate predictions are ground truth.",
        "Do not claim trained variable-N RL policy success.",
        "Do not claim arbitrary-N generalization.",
        "Do not claim future CAE/INP files already exist.",
        "Do not claim abqjobpilot commands are ready to execute.",
    ]
    text = ["# Run 13 Claim Boundary", "", "## Safe Claims", *[f"- {x}" for x in safe], "", "## Unsafe Claims", *[f"- {x}" for x in unsafe]]
    md.write_text("\n".join(text) + "\n", encoding="utf-8")
    write_json(js, {"verdict": "RUN13_HANDOFF_ONLY_NO_TEACHER_VALIDATION", "safe_claims": safe, "unsafe_claims": unsafe})


def update_run_index(verdict: str) -> None:
    if not RUN_INDEX_PATH.exists():
        return
    entry = (
        "| run_13 | Batch20 surrogate-screened teacher handoff | Package 20 run12 candidates for human review and future CAE/teacher-validation setup with scan orders, metadata, future path templates, and abqjobpilot command templates. | "
        "`scripts/stage3/run_13_create_batch20_teacher_handoff.py` | "
        "`docs/stage3/runs/run_13_batch20_surrogate_screened_teacher_handoff/RUN_13_BATCH20_SURROGATE_SCREENED_TEACHER_HANDOFF_REPORT.md` | "
        "`outputs/stage3_run_13_batch20_surrogate_screened_teacher_handoff/` | "
        f"`{verdict}` | No Abaqus, no ODB, no CAE/INP/JNL generation, no abqjobpilot, no teacher validation, no RL training, no commit/push. |"
    )
    lines = RUN_INDEX_PATH.read_text(encoding="utf-8").splitlines()
    for idx, line in enumerate(lines):
        if line.startswith("| run_13 | Batch20 surrogate-screened teacher handoff |"):
            lines[idx] = entry
            break
    else:
        lines.append(entry)
    RUN_INDEX_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(
    validation: dict[str, Any],
    handoff_rows: list[dict[str, Any]],
    output_files: list[str],
    comparison_rows: list[dict[str, Any]],
) -> None:
    bucket_counts = Counter(row["selection_bucket"] for row in handoff_rows)
    per_n_counts = Counter(row["n"] for row in handoff_rows)
    lines = [
        "# Stage 3 Run 13 - Batch20 Surrogate-Screened Teacher Handoff",
        "",
        "## Purpose",
        "Create a clean handoff package for 20 run12 surrogate-screened candidates. This is active-learning/diversity validation packaging only.",
        "",
        "## Inputs",
        f"- `{INPUT_BATCH20}`",
        f"- `{INPUT_SCORED_POOL}`",
        f"- `{INPUT_RUN12_REPORT}`",
        f"- `{INPUT_TEACHER_LABELS}`",
        "",
        "## Validation Status",
        f"- `{validation['verdict']}`",
        f"- Candidate count: {len(handoff_rows)}",
        f"- Per-N counts: {dict(sorted(per_n_counts.items()))}",
        "",
        "## Batch Composition",
        f"- Batch name: `{BATCH_NAME}`",
        f"- Selection buckets: {dict(bucket_counts)}",
        "",
        "## Naming Convention",
        "- `S3B20_N{N}_B{index:02d}_{short_family_or_bucket}`",
        "- Names preserve run12 candidate IDs and order hashes in metadata.",
        "",
        "## Candidate Order Handoff Table",
        "- See `stage3_run13_batch20_candidate_orders.csv`.",
        "",
        "## Per-Candidate Scan_Order JSON Outputs",
        f"- Directory: `{SCAN_ORDER_DIR}`",
        "",
        "## Future CAE Handoff Manifest Template",
        "- Template only. No CAE/INP/JNL/ODB files were generated.",
        "",
        "## Future Abqjobpilot Command Template",
        "- Template only. Commands must not be run until INPs exist and are checked.",
        "",
        "## Comparison To Existing Teacher Best",
    ]
    for row in comparison_rows:
        lines.append(f"- N{row['n']}: top batch20 `{row['top_predicted_batch20_candidate']}` predicted {row['top_predicted_batch20_reward']:.4f}; existing best `{row['existing_best_teacher_strategy']}` reward {row['existing_best_teacher_reward_mean_all']:.4f}; exceeds `{row['batch20_predicted_exceeds_existing_best']}`. Surrogate-only.")
    lines += [
        "",
        "## Claim Boundary",
        "- Run13 does not perform teacher validation.",
        "- Run13 does not prove physical superiority.",
        "- Future CAE/INP paths and abqjobpilot commands are templates only.",
        "",
        "## Output Files",
        *[f"- `{path}`" for path in output_files],
        "",
        "## Recommended Next Step",
        "Human review of the 20 candidates. If approved, the CAE module should create a separate Stage 3 run13 batch20 CAE-generation workflow, generate CAE/INP only, validate mesh and final cooling controls, and only then prepare executable abqjobpilot commands.",
    ]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def git_branch() -> str:
    try:
        result = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception:
        return "UNKNOWN"


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SCAN_ORDER_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    batch_rows = read_csv(INPUT_BATCH20)
    _teacher_best, teacher_hashes = load_teacher_best()
    validation = validate_inputs(batch_rows, teacher_hashes)
    validation_path = OUTPUT_DIR / "run13_input_validation_summary.json"
    write_json(validation_path, validation)
    if validation["verdict"].startswith("FAIL"):
        print(validation["verdict"])
        print(json.dumps(validation, indent=2))
        return 2

    handoff_rows = build_handoff_rows(batch_rows)
    handoff_csv = OUTPUT_DIR / "stage3_run13_batch20_candidate_orders.csv"
    write_csv(handoff_csv, handoff_rows)
    scan_json_paths = write_scan_order_jsons(handoff_rows)

    cae_rows = build_future_cae_manifest(handoff_rows)
    cae_template = OUTPUT_DIR / "stage3_run13_batch20_future_cae_handoff_manifest_TEMPLATE.csv"
    write_csv(cae_template, cae_rows)

    commands = build_abqjobpilot_commands(cae_rows)
    command_path = OUTPUT_DIR / "stage3_run13_batch20_abqjobpilot_commands_TEMPLATE.txt"
    command_path.write_text("\n".join(commands) + "\n", encoding="utf-8")

    readme_path = OUTPUT_DIR / "README_FOR_CAE_GENERATION.md"
    write_readme(readme_path)

    comparison_rows = compare_to_existing(handoff_rows)
    comparison_csv = OUTPUT_DIR / "batch20_vs_existing_teacher_best_summary.csv"
    comparison_json = OUTPUT_DIR / "batch20_vs_existing_teacher_best_summary.json"
    write_csv(comparison_csv, comparison_rows)
    write_json(comparison_json, comparison_rows)

    claim_md = OUTPUT_DIR / "run13_claim_boundary.md"
    claim_json = OUTPUT_DIR / "run13_claim_boundary.json"
    write_claim_boundary(claim_md, claim_json)

    output_files = [
        str(validation_path),
        str(handoff_csv),
        *scan_json_paths,
        str(cae_template),
        str(command_path),
        str(readme_path),
        str(comparison_csv),
        str(comparison_json),
        str(claim_md),
        str(claim_json),
        str(REPORT_PATH),
    ]
    write_report(validation, handoff_rows, output_files, comparison_rows)
    update_run_index(validation["verdict"])

    manifest = {
        "run_id": RUN_ID,
        "run_name": RUN_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "branch": git_branch(),
        "script_path": str(Path(__file__).resolve()),
        "input_files": [str(INPUT_BATCH20), str(INPUT_SCORED_POOL), str(INPUT_RUN12_REPORT), str(INPUT_TEACHER_LABELS)],
        "output_files": output_files,
        "candidate_count": len(handoff_rows),
        "per_n_counts": dict(sorted(Counter(row["n"] for row in handoff_rows).items())),
        "report_path": str(REPORT_PATH),
        "claim_boundary_path": str(claim_md),
        "no_solver_run": True,
        "no_odb_opened": True,
        "no_abqjobpilot_run": True,
        "no_cae_inp_generated": True,
        "no_teacher_validation": True,
        "no_rl_policy_training": True,
        "no_commit_or_push": True,
    }
    write_json(MANIFEST_PATH, manifest)

    top_by_n = {
        f"N{n}": max([row for row in handoff_rows if row["n"] == n], key=lambda row: parse_float(row["predicted_reward_mean_all"]))["handoff_strategy_name"]
        for n in EXPECTED_N
    }
    print(validation["verdict"])
    print(f"candidate_count={len(handoff_rows)}")
    print(f"per_n_counts={manifest['per_n_counts']}")
    print(f"batch_name={BATCH_NAME}")
    print(f"top_candidate_per_n={top_by_n}")
    print(f"report={REPORT_PATH}")
    print(f"manifest={MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
