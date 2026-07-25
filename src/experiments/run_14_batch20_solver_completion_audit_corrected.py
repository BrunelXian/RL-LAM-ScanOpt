from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
REQUESTED_CASE_ROOT = PROJECT_ROOT / "cae_model" / "stage3_run13_batch20_surrogate-screened_v01"
MANIFEST_CASE_ROOT = PROJECT_ROOT / "cae_model" / "stage3_run13_batch20_surrogate_screened_v01"
RUN14_MANIFEST_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "stage3_run_14_batch20_cae_inp_generation"
    / "stage3_run14_batch20_cae_generation_manifest.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage3_run_14_batch20_odb_teacher_validation"

AUDIT_CSV = OUTPUT_DIR / "run14_batch20_solver_completion_audit_corrected.csv"
AUDIT_JSON = OUTPUT_DIR / "run14_batch20_solver_completion_audit_corrected.json"
AUDIT_MD = OUTPUT_DIR / "run14_batch20_solver_completion_audit_corrected.md"

SUCCESS_MARKER = "THE ANALYSIS HAS COMPLETED SUCCESSFULLY"
FATAL_MARKERS = [
    "Abaqus/Standard aborted",
    "THE ANALYSIS HAS BEEN TERMINATED",
    "THE ANALYSIS HAS NOT BEEN COMPLETED",
    "Too many attempts made for this increment",
    "***ERROR",
    "exited with an error",
    "ERROR in job",
    "Abaqus Error",
]
WARNING_MARKERS = ["***WARNING", "*** WARNING"]

FIELDS = [
    "run_id",
    "batch_name",
    "n",
    "handoff_strategy_name",
    "job_name",
    "case_dir",
    "sta_path",
    "dat_path",
    "msg_path",
    "log_path",
    "odb_path",
    "lck_paths",
    "sta_exists",
    "dat_exists",
    "msg_exists",
    "log_exists",
    "odb_exists",
    "odb_size_bytes",
    "lck_present",
    "sta_success_marker",
    "sta_fatal_marker",
    "dat_fatal_marker",
    "msg_fatal_marker",
    "log_fatal_marker",
    "nonfatal_warning_marker",
    "completion_status",
    "notes",
]


def read_text(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def contains_any(text: str, markers: list[str]) -> bool:
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in markers)


def contains_warning(text: str) -> bool:
    if contains_any(text, WARNING_MARKERS):
        return True
    patterns = [
        r"\b([1-9][0-9]*)\s+WARNING\s+MESSAGES?\b",
        r"\b([1-9][0-9]*)\s+WARNINGS?\b",
    ]
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def load_manifest() -> list[dict[str, str]]:
    with RUN14_MANIFEST_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 20:
        raise RuntimeError(f"Expected 20 manifest rows, got {len(rows)}")
    return rows


def audit_case(row: dict[str, str]) -> dict[str, object]:
    case_dir = Path(row["case_dir"])
    job_name = row["job_name"]
    sta_path = case_dir / f"{job_name}.sta"
    dat_path = case_dir / f"{job_name}.dat"
    msg_path = case_dir / f"{job_name}.msg"
    log_path = case_dir / f"{job_name}.log"
    odb_path = case_dir / f"{job_name}.odb"
    lck_paths = sorted(case_dir.glob("*.lck")) if case_dir.exists() else []

    sta_text = read_text(sta_path)
    dat_text = read_text(dat_path)
    msg_text = read_text(msg_path)
    log_text = read_text(log_path)

    sta_exists = sta_path.exists()
    dat_exists = dat_path.exists()
    msg_exists = msg_path.exists()
    log_exists = log_path.exists()
    odb_exists = odb_path.exists()
    odb_size_bytes = odb_path.stat().st_size if odb_exists else 0

    sta_success_marker = SUCCESS_MARKER.lower() in sta_text.lower()
    sta_fatal_marker = contains_any(sta_text, FATAL_MARKERS)
    dat_fatal_marker = contains_any(dat_text, FATAL_MARKERS)
    msg_fatal_marker = contains_any(msg_text, FATAL_MARKERS)
    log_fatal_marker = contains_any(log_text, FATAL_MARKERS)
    fatal_present = any(
        [sta_fatal_marker, dat_fatal_marker, msg_fatal_marker, log_fatal_marker]
    )
    nonfatal_warning_marker = any(
        contains_warning(text) for text in [sta_text, dat_text, msg_text, log_text]
    )

    blockers: list[str] = []
    if not case_dir.exists():
        blockers.append("missing_case_dir")
    if not sta_exists:
        blockers.append("missing_sta")
    if not dat_exists:
        blockers.append("missing_dat")
    if not msg_exists:
        blockers.append("missing_msg")
    if not sta_success_marker:
        blockers.append("missing_sta_success_marker")
    if not odb_exists:
        blockers.append("missing_odb")
    if odb_exists and odb_size_bytes <= 0:
        blockers.append("empty_odb")
    if lck_paths:
        blockers.append("lck_present")
    if fatal_present:
        blockers.append("fatal_marker_present")

    if blockers:
        completion_status = "FAIL"
        notes = "; ".join(blockers)
    elif nonfatal_warning_marker:
        completion_status = "WARNING"
        notes = "complete_with_nonfatal_warnings"
    else:
        completion_status = "PASS"
        notes = "complete_no_lck_no_fatal_markers"

    return {
        "run_id": row["run_id"],
        "batch_name": row["batch_name"],
        "n": int(row["n"]),
        "handoff_strategy_name": row["handoff_strategy_name"],
        "job_name": job_name,
        "case_dir": str(case_dir),
        "sta_path": str(sta_path),
        "dat_path": str(dat_path),
        "msg_path": str(msg_path),
        "log_path": str(log_path),
        "odb_path": str(odb_path),
        "lck_paths": ";".join(str(path) for path in lck_paths),
        "sta_exists": sta_exists,
        "dat_exists": dat_exists,
        "msg_exists": msg_exists,
        "log_exists": log_exists,
        "odb_exists": odb_exists,
        "odb_size_bytes": odb_size_bytes,
        "lck_present": bool(lck_paths),
        "sta_success_marker": sta_success_marker,
        "sta_fatal_marker": sta_fatal_marker,
        "dat_fatal_marker": dat_fatal_marker,
        "msg_fatal_marker": msg_fatal_marker,
        "log_fatal_marker": log_fatal_marker,
        "nonfatal_warning_marker": nonfatal_warning_marker,
        "completion_status": completion_status,
        "notes": notes,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def build_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    by_n: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "expected": 0,
            "pass": 0,
            "warning": 0,
            "fail": 0,
            "complete_pass_or_warning": 0,
            "odb_present": 0,
            "lck_present": 0,
        }
    )
    for row in rows:
        group = by_n[f"N{row['n']}"]
        group["expected"] += 1
        status = str(row["completion_status"]).lower()
        group[status] += 1
        if row["completion_status"] in {"PASS", "WARNING"}:
            group["complete_pass_or_warning"] += 1
        if row["odb_exists"] and int(row["odb_size_bytes"]) > 0:
            group["odb_present"] += 1
        if row["lck_present"]:
            group["lck_present"] += 1

    fail_count = sum(1 for row in rows if row["completion_status"] == "FAIL")
    warning_count = sum(1 for row in rows if row["completion_status"] == "WARNING")
    complete_count = sum(1 for row in rows if row["completion_status"] in {"PASS", "WARNING"})
    if fail_count:
        verdict = "FAIL_RUN14_BATCH20_SOLVER_COMPLETION_INCOMPLETE"
    elif warning_count:
        verdict = "WARNING_RUN14_BATCH20_SOLVER_COMPLETION_WITH_NONFATAL_WARNINGS"
    else:
        verdict = "PASS_RUN14_BATCH20_SOLVER_COMPLETION_20_OF_20"

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "total_expected": 20,
        "total_audited": len(rows),
        "total_completed_pass_or_warning": complete_count,
        "total_pass": sum(1 for row in rows if row["completion_status"] == "PASS"),
        "total_warning": warning_count,
        "total_fail": fail_count,
        "total_odb_present_nonempty": sum(
            1 for row in rows if row["odb_exists"] and int(row["odb_size_bytes"]) > 0
        ),
        "total_lck_present": sum(1 for row in rows if row["lck_present"]),
        "requested_case_root_exists": REQUESTED_CASE_ROOT.exists(),
        "manifest_case_root_exists": MANIFEST_CASE_ROOT.exists(),
        "requested_case_root": str(REQUESTED_CASE_ROOT),
        "manifest_case_root": str(MANIFEST_CASE_ROOT),
        "by_N": dict(sorted(by_n.items())),
        "guardrails": {
            "abaqus_solver_run": False,
            "job_submitted": False,
            "datacheck_run": False,
            "abqjobpilot_run": False,
            "odb_opened": False,
            "teacher_metrics_computed": False,
            "cae_inp_odb_jnl_modified": False,
            "mixed_with_probe60_or_other_runs": False,
        },
    }


def write_markdown(summary: dict[str, object], rows: list[dict[str, object]]) -> None:
    special = next(
        row
        for row in rows
        if row["handoff_strategy_name"] == "S3B20_N40_B02_diversity_top"
    )
    failed = [row for row in rows if row["completion_status"] == "FAIL"]
    warnings = [row for row in rows if row["completion_status"] == "WARNING"]

    lines = [
        "# Run14 Batch20 Solver Completion Audit Corrected",
        "",
        "## Verdict",
        "",
        f"`{summary['verdict']}`",
        "",
        "## Path Note",
        "",
        f"- Requested case root exists: `{summary['requested_case_root_exists']}`",
        f"- Requested case root: `{summary['requested_case_root']}`",
        f"- Manifest case root exists: `{summary['manifest_case_root_exists']}`",
        f"- Manifest case root: `{summary['manifest_case_root']}`",
        "- Audit case list and case directories were taken from the handoff manifest.",
        "",
        "## Summary",
        "",
        f"- Total completed cases: `{summary['total_completed_pass_or_warning']}/{summary['total_expected']}`",
        f"- PASS: `{summary['total_pass']}`",
        f"- WARNING: `{summary['total_warning']}`",
        f"- FAIL: `{summary['total_fail']}`",
        f"- Nonempty ODB present: `{summary['total_odb_present_nonempty']}/{summary['total_expected']}`",
        f"- LCK present: `{summary['total_lck_present']}`",
        "",
        "| N | expected | complete | PASS | WARNING | FAIL | odb_present | lck_present |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n_key in ["N12", "N16", "N24", "N40"]:
        group = summary["by_N"][n_key]
        lines.append(
            f"| {n_key} | {group['expected']} | {group['complete_pass_or_warning']} | "
            f"{group['pass']} | {group['warning']} | {group['fail']} | "
            f"{group['odb_present']} | {group['lck_present']} |"
        )

    lines.extend(
        [
            "",
            "## Special Check: S3B20_N40_B02_diversity_top",
            "",
            f"- status: `{special['completion_status']}`",
            f"- STA exists: `{special['sta_exists']}`",
            f"- DAT exists: `{special['dat_exists']}`",
            f"- MSG exists: `{special['msg_exists']}`",
            f"- ODB exists and size: `{special['odb_exists']}`, `{special['odb_size_bytes']}` bytes",
            f"- LCK present: `{special['lck_present']}`",
            f"- STA success marker: `{special['sta_success_marker']}`",
            f"- fatal markers: sta=`{special['sta_fatal_marker']}`, dat=`{special['dat_fatal_marker']}`, msg=`{special['msg_fatal_marker']}`, log=`{special['log_fatal_marker']}`",
            f"- notes: `{special['notes']}`",
            "",
            "## Failed Cases",
            "",
        ]
    )
    if failed:
        for row in failed:
            lines.append(
                f"- `{row['handoff_strategy_name']}` ({row['job_name']}): "
                f"`{row['notes']}`"
            )
    else:
        lines.append("None.")

    lines.extend(["", "## Warning Cases", ""])
    if warnings:
        for row in warnings:
            lines.append(
                f"- `{row['handoff_strategy_name']}` ({row['job_name']}): `{row['notes']}`"
            )
    else:
        lines.append("None.")

    lines.extend(
        [
            "",
            "## Per-Case Status",
            "",
            "| N | handoff_strategy_name | status | ODB bytes | notes |",
            "|---|---|---|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| N{row['n']} | `{row['handoff_strategy_name']}` | "
            f"`{row['completion_status']}` | {row['odb_size_bytes']} | `{row['notes']}` |"
        )

    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "This is only a completion gate. No ODB was opened, no teacher metrics were computed, and no physical or validated improvement is claimed.",
        ]
    )
    AUDIT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [audit_case(row) for row in load_manifest()]
    rows.sort(key=lambda row: (int(row["n"]), str(row["handoff_strategy_name"])))
    summary = build_summary(rows)
    write_csv(AUDIT_CSV, rows)
    AUDIT_JSON.write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(summary, rows)
    print(json.dumps(summary, indent=2))
    print(f"CSV: {AUDIT_CSV}")
    print(f"JSON: {AUDIT_JSON}")
    print(f"MD: {AUDIT_MD}")
    return 1 if str(summary["verdict"]).startswith("FAIL") else 0


if __name__ == "__main__":
    raise SystemExit(main())
