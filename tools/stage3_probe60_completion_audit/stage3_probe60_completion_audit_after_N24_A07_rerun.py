from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path


OUTPUT_DIR = Path(r"E:\Projects\RL-LAM-ScanOpt\outputs\stage3_cae_probe60_generation")
REFINED_AUDIT_JSON = OUTPUT_DIR / "probe60_solver_completion_audit_refined.json"
LEGACY_AUDIT_CSV = OUTPUT_DIR / "probe60_solver_completion_audit.csv"

CSV_OUT = OUTPUT_DIR / "probe60_solver_completion_audit_after_N24_A07_rerun.csv"
JSON_OUT = OUTPUT_DIR / "probe60_solver_completion_audit_after_N24_A07_rerun.json"
MD_OUT = OUTPUT_DIR / "probe60_solver_completion_audit_after_N24_A07_rerun.md"

SUCCESS_MARKER = "THE ANALYSIS HAS COMPLETED SUCCESSFULLY"
FATAL_MARKERS = [
    "Abaqus/Standard aborted",
    "THE ANALYSIS HAS BEEN TERMINATED",
    "THE ANALYSIS HAS NOT BEEN COMPLETED",
    "Too many attempts made for this increment",
    "exited with an error",
    "ERROR in job",
    "Abaqus Error",
]
WARNING_MARKERS = [
    "***WARNING",
    "*** WARNING",
]

FIELDNAMES = [
    "case_id",
    "N",
    "strategy_id",
    "job_name",
    "case_dir",
    "sta_path",
    "dat_path",
    "msg_path",
    "odb_path",
    "lck_paths",
    "sta_exists",
    "dat_exists",
    "msg_exists",
    "odb_exists",
    "odb_size_bytes",
    "lck_present",
    "sta_success_marker",
    "sta_abort_marker",
    "dat_fatal_marker",
    "msg_fatal_marker",
    "log_fatal_marker",
    "final_completion_status",
    "notes",
]


def read_text_lossy(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def contains_any(text: str, markers: list[str]) -> bool:
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in markers)


def contains_nonfatal_warning(text: str) -> bool:
    if contains_any(text, WARNING_MARKERS):
        return True

    warning_count_patterns = [
        r"\b([1-9][0-9]*)\s+WARNING\s+MESSAGES?\b",
        r"\b([1-9][0-9]*)\s+WARNINGS?\b",
    ]
    return any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in warning_count_patterns
    )


def derive_strategy_id(case_id: str) -> str:
    parts = case_id.split("_", 2)
    if len(parts) >= 2:
        return parts[1]
    return ""


def derive_job_name(n_value: str, case_id: str, case_dir: Path) -> str:
    expected = f"J2D_{n_value}_{case_id}"
    known_suffixes = [".sta", ".odb", ".dat", ".msg", ".log", ".com", ".inp"]
    for suffix in known_suffixes:
        candidate = case_dir / f"{expected}{suffix}"
        if candidate.exists():
            return expected

    sta_files = sorted(case_dir.glob("*.sta"))
    if sta_files:
        return sta_files[0].stem

    odb_files = sorted(case_dir.glob("*.odb"))
    if odb_files:
        return odb_files[0].stem

    return expected


def load_roster() -> list[dict[str, str]]:
    with REFINED_AUDIT_JSON.open("r", encoding="utf-8") as f:
        refined = json.load(f)

    rows = refined.get("rows", [])
    if len(rows) == 60:
        return [
            {
                "case_id": row["case"],
                "N": row["n"],
                "case_dir": row["path"],
            }
            for row in rows
        ]

    roster: list[dict[str, str]] = []
    with LEGACY_AUDIT_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            roster.append(
                {
                    "case_id": row["case"],
                    "N": row["n"],
                    "case_dir": row["path"],
                }
            )
    return roster


def audit_case(case_spec: dict[str, str]) -> dict[str, object]:
    case_id = case_spec["case_id"]
    n_value = case_spec["N"]
    case_dir = Path(case_spec["case_dir"])
    strategy_id = derive_strategy_id(case_id)
    job_name = derive_job_name(n_value, case_id, case_dir)

    sta_path = case_dir / f"{job_name}.sta"
    dat_path = case_dir / f"{job_name}.dat"
    msg_path = case_dir / f"{job_name}.msg"
    log_path = case_dir / f"{job_name}.log"
    odb_path = case_dir / f"{job_name}.odb"
    lck_paths = sorted(case_dir.glob("*.lck"))

    sta_text = read_text_lossy(sta_path)
    dat_text = read_text_lossy(dat_path)
    msg_text = read_text_lossy(msg_path)
    log_text = read_text_lossy(log_path)

    sta_exists = sta_path.exists()
    dat_exists = dat_path.exists()
    msg_exists = msg_path.exists()
    odb_exists = odb_path.exists()
    odb_size_bytes = odb_path.stat().st_size if odb_exists else 0
    lck_present = len(lck_paths) > 0

    sta_success_marker = SUCCESS_MARKER.lower() in sta_text.lower()
    sta_abort_marker = contains_any(sta_text, FATAL_MARKERS)
    dat_fatal_marker = contains_any(dat_text, FATAL_MARKERS)
    msg_fatal_marker = contains_any(msg_text, FATAL_MARKERS)
    log_fatal_marker = contains_any(log_text, FATAL_MARKERS)
    fatal_present = any(
        [sta_abort_marker, dat_fatal_marker, msg_fatal_marker, log_fatal_marker]
    )

    warning_present = any(
        contains_nonfatal_warning(text)
        for text in [sta_text, dat_text, msg_text, log_text]
    )

    blockers: list[str] = []
    if not sta_exists:
        blockers.append("missing_sta")
    if not sta_success_marker:
        blockers.append("missing_sta_success_marker")
    if not odb_exists:
        blockers.append("missing_odb")
    if odb_exists and odb_size_bytes <= 0:
        blockers.append("empty_odb")
    if lck_present:
        blockers.append("lck_present")
    if fatal_present:
        blockers.append("fatal_marker_present")

    if blockers:
        final_status = "FAIL_INCOMPLETE_OR_ABORTED"
        notes = "; ".join(blockers)
    elif warning_present:
        final_status = "WARNING_SUCCESS_WITH_WARNINGS"
        notes = "sta_success_and_odb_present_with_nonfatal_warnings"
    else:
        final_status = "PASS_SOLVER_COMPLETE"
        notes = "sta_success_and_odb_present_no_lck_no_fatal_markers"

    return {
        "case_id": case_id,
        "N": n_value,
        "strategy_id": strategy_id,
        "job_name": job_name,
        "case_dir": str(case_dir),
        "sta_path": str(sta_path),
        "dat_path": str(dat_path),
        "msg_path": str(msg_path),
        "odb_path": str(odb_path),
        "lck_paths": ";".join(str(path) for path in lck_paths),
        "sta_exists": sta_exists,
        "dat_exists": dat_exists,
        "msg_exists": msg_exists,
        "odb_exists": odb_exists,
        "odb_size_bytes": odb_size_bytes,
        "lck_present": lck_present,
        "sta_success_marker": sta_success_marker,
        "sta_abort_marker": sta_abort_marker,
        "dat_fatal_marker": dat_fatal_marker,
        "msg_fatal_marker": msg_fatal_marker,
        "log_fatal_marker": log_fatal_marker,
        "final_completion_status": final_status,
        "notes": notes,
    }


def count_success(row: dict[str, object]) -> bool:
    return row["final_completion_status"] in {
        "PASS_SOLVER_COMPLETE",
        "WARNING_SUCCESS_WITH_WARNINGS",
    }


def build_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    by_n: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "expected": 0,
            "solver_success": 0,
            "odb_present": 0,
            "lck_present": 0,
            "failed_or_incomplete": 0,
        }
    )

    for row in rows:
        group = by_n[str(row["N"])]
        group["expected"] += 1
        group["solver_success"] += int(count_success(row))
        group["odb_present"] += int(bool(row["odb_exists"]))
        group["lck_present"] += int(bool(row["lck_present"]))
        group["failed_or_incomplete"] += int(
            row["final_completion_status"] == "FAIL_INCOMPLETE_OR_ABORTED"
        )

    total_cases_expected = 60
    total_cases_audited = len(rows)
    total_solver_success = sum(int(count_success(row)) for row in rows)
    total_odb_present = sum(int(bool(row["odb_exists"])) for row in rows)
    total_lck_present = sum(int(bool(row["lck_present"])) for row in rows)
    total_failed_or_incomplete = sum(
        int(row["final_completion_status"] == "FAIL_INCOMPLETE_OR_ABORTED")
        for row in rows
    )
    warning_count = sum(
        int(row["final_completion_status"] == "WARNING_SUCCESS_WITH_WARNINGS")
        for row in rows
    )

    if total_failed_or_incomplete:
        verdict = "FAIL_PROBE60_SOLVER_COMPLETION_INCOMPLETE"
    elif warning_count:
        verdict = "WARNING_PROBE60_COMPLETION_WITH_NONFATAL_WARNINGS"
    else:
        verdict = "PASS_PROBE60_SOLVER_COMPLETION_60_OF_60"

    return {
        "verdict": verdict,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_files": {
            "refined_audit_json": str(REFINED_AUDIT_JSON),
            "legacy_audit_csv": str(LEGACY_AUDIT_CSV),
        },
        "guardrails": {
            "abaqus_run": False,
            "solver_run": False,
            "datacheck_run": False,
            "odb_opened": False,
            "odb_access_imported": False,
            "solver_outputs_modified": False,
        },
        "total_cases_expected": total_cases_expected,
        "total_cases_audited": total_cases_audited,
        "total_solver_success": total_solver_success,
        "total_odb_present": total_odb_present,
        "total_lck_present": total_lck_present,
        "total_failed_or_incomplete": total_failed_or_incomplete,
        "N12_expected": by_n["N12"]["expected"],
        "N12_success": by_n["N12"]["solver_success"],
        "N16_expected": by_n["N16"]["expected"],
        "N16_success": by_n["N16"]["solver_success"],
        "N24_expected": by_n["N24"]["expected"],
        "N24_success": by_n["N24"]["solver_success"],
        "N40_expected": by_n["N40"]["expected"],
        "N40_success": by_n["N40"]["solver_success"],
        "by_N": dict(sorted(by_n.items())),
    }


def write_csv(rows: list[dict[str, object]]) -> None:
    with CSV_OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in FIELDNAMES})


def format_bool(value: object) -> str:
    return "true" if bool(value) else "false"


def write_markdown(summary: dict[str, object], rows: list[dict[str, object]]) -> None:
    failed = [
        row
        for row in rows
        if row["final_completion_status"] == "FAIL_INCOMPLETE_OR_ABORTED"
    ]
    warning = [
        row
        for row in rows
        if row["final_completion_status"] == "WARNING_SUCCESS_WITH_WARNINGS"
    ]
    special = next(
        row for row in rows if row["case_id"] == "N24_A07_regular_jump_coprime"
    )

    lines: list[str] = [
        "# Stage 3 Probe60 Solver Completion Audit After N24_A07 Rerun",
        "",
        "## Verdict",
        "",
        f"`{summary['verdict']}`",
        "",
        "## Summary Table",
        "",
        "| N | expected | solver_success | odb_present | lck_present | failed_or_incomplete |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for n_value in ["N12", "N16", "N24", "N40"]:
        group = summary["by_N"][n_value]
        lines.append(
            f"| {n_value} | {group['expected']} | {group['solver_success']} | "
            f"{group['odb_present']} | {group['lck_present']} | "
            f"{group['failed_or_incomplete']} |"
        )

    lines.extend(
        [
            "",
            "## Special Case Check: N24_A07_regular_jump_coprime",
            "",
            f"- STA success marker: {format_bool(special['sta_success_marker'])}",
            f"- ODB exists and size: {format_bool(special['odb_exists'])}, {special['odb_size_bytes']} bytes",
            f"- LCK absence: {format_bool(not special['lck_present'])}",
            "- fatal markers: "
            f"sta={format_bool(special['sta_abort_marker'])}, "
            f"dat={format_bool(special['dat_fatal_marker'])}, "
            f"msg={format_bool(special['msg_fatal_marker'])}, "
            f"log={format_bool(special['log_fatal_marker'])}",
            f"- final status: `{special['final_completion_status']}`",
            f"- notes: {special['notes']}",
            "",
            "## Failed / Incomplete Cases",
            "",
        ]
    )

    if failed:
        for row in failed:
            lines.append(
                f"- `{row['case_id']}` ({row['N']}): "
                f"`{row['final_completion_status']}`; {row['notes']}"
            )
    else:
        lines.append("None.")

    lines.extend(["", "## Warning Cases", ""])

    if warning:
        for row in warning:
            lines.append(
                f"- `{row['case_id']}` ({row['N']}): "
                f"`{row['final_completion_status']}`; {row['notes']}"
            )
    else:
        lines.append("None.")

    lines.extend(["", "## Next-Step Gate", ""])
    if summary["verdict"] == "PASS_PROBE60_SOLVER_COMPLETION_60_OF_60":
        lines.append(
            "`Probe60 is ready for ODB teacher validation / metric postprocessing. "
            "No ODB has been opened in this audit.`"
        )
    else:
        lines.append(
            "`Do not start ODB postprocessing. Resolve failed/incomplete cases first.`"
        )

    lines.extend(
        [
            "",
            "## Audit Guardrails",
            "",
            "- No Abaqus command was run.",
            "- No solver, datacheck, or queue command was run.",
            "- No ODB was opened; only path existence, size, and timestamp metadata were checked.",
            "- No solver output files were modified, moved, renamed, archived, or deleted.",
        ]
    )

    MD_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    roster = load_roster()
    rows = [audit_case(case_spec) for case_spec in roster]
    rows.sort(key=lambda row: (str(row["N"]), str(row["case_id"])))
    summary = build_summary(rows)

    write_csv(rows)
    JSON_OUT.write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2),
        encoding="utf-8",
    )
    write_markdown(summary, rows)

    print(json.dumps(summary, indent=2))
    print(f"CSV: {CSV_OUT}")
    print(f"JSON: {JSON_OUT}")
    print(f"MD: {MD_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
