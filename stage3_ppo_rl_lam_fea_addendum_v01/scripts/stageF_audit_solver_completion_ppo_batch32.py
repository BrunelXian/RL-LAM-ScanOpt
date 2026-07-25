"""Immediate/reusable solver completion audit for Stage F PPO-only batch32.

This script reads text solver outputs only. It never opens ODB files.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
BATCH_NAME = "stage3_ppo_policy_only_batch32_v01"
COMMAND_FILE = PROJECT_ROOT / "outputs" / "stage3_ppo_rl_lam_fea_addendum_v01" / "stageE_teacher_validation_handoff" / "commands" / "stageE_ppo_batch32_abqjobpilot_commands_READY_TO_RUN.txt"
CASE_ROOT = PROJECT_ROOT / "cae_model" / BATCH_NAME
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "stage3_ppo_rl_lam_fea_addendum_v01" / "stageF_solver_execution"
CHECKS_DIR = OUTPUT_ROOT / "checks"
CSV_PATH = CHECKS_DIR / "stageF_solver_completion_audit.csv"
SUMMARY_PATH = CHECKS_DIR / "stageF_solver_completion_audit_summary.json"

EXPECTED_COUNTS = {12: 8, 16: 8, 24: 8, 40: 8}
FAILURE_KEYWORDS = [
    "Abaqus Error",
    "THE ANALYSIS HAS NOT BEEN COMPLETED",
    "ERROR",
    "TOO MANY ATTEMPTS",
    "TIME INCREMENT",
    "numerical singularity",
    "THE ANALYSIS HAS BEEN TERMINATED",
]
SUCCESS_MARKER = "THE ANALYSIS HAS COMPLETED SUCCESSFULLY"
COMMAND_RE = re.compile(r'^enqueue --inp "([^"]+)" --cpus 14 --batch (\S+) --strategy (\S+)\s*$')


def parse_n_from_path(path: Path) -> int | None:
    for part in path.parts:
        match = re.match(r"^N(12|16|24|40)[_A-Za-z0-9-]", part)
        if match:
            return int(match.group(1))
    return None


def tail_text(path: Path, max_chars: int = 6000) -> str:
    if not path.exists():
        return ""
    data = path.read_text(encoding="utf-8", errors="replace")
    return data[-max_chars:]


def file_info(path: Path) -> tuple[bool, int, str]:
    if not path.exists():
        return False, 0, ""
    return True, path.stat().st_size, path.stat().st_mtime_ns.__str__()


def main() -> int:
    CHECKS_DIR.mkdir(parents=True, exist_ok=True)
    commands = [line.strip() for line in COMMAND_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows: list[dict[str, object]] = []
    for idx, command in enumerate(commands, 1):
        match = COMMAND_RE.match(command)
        if not match:
            rows.append({"command_index": idx, "command_text": command, "status": "INVALID_COMMAND"})
            continue
        inp_path = Path(match.group(1))
        strategy = match.group(3)
        n_value = parse_n_from_path(inp_path)
        case_dir = inp_path.parent
        job_stem = inp_path.stem
        odb = case_dir / f"{job_stem}.odb"
        sim = case_dir / f"{job_stem}.sim"
        sta = case_dir / f"{job_stem}.sta"
        dat = case_dir / f"{job_stem}.dat"
        msg = case_dir / f"{job_stem}.msg"
        lck = case_dir / f"{job_stem}.lck"
        sta_tail = tail_text(sta)
        msg_tail = tail_text(msg)
        dat_tail = tail_text(dat)
        combined_tail = "\n".join([sta_tail, msg_tail, dat_tail])
        success = SUCCESS_MARKER in sta_tail
        failure_hits = [kw for kw in FAILURE_KEYWORDS if kw.lower() in combined_tail.lower()]
        has_lck = lck.exists()
        if success and not has_lck and not failure_hits:
            status = "COMPLETED_SUCCESS"
        elif failure_hits:
            status = "FAILED_OR_ERROR_MARKERS"
        elif has_lck:
            status = "RUNNING_OR_LOCK_PRESENT"
        elif sta.exists() or odb.exists() or msg.exists() or dat.exists():
            status = "PARTIAL_OR_UNKNOWN"
        else:
            status = "NOT_STARTED_OR_QUEUED"
        row = {
            "command_index": idx,
            "n": n_value if n_value is not None else "",
            "strategy_name": strategy,
            "case_dir": str(case_dir),
            "inp_path": str(inp_path),
            "inp_exists": inp_path.exists(),
            "odb_exists": odb.exists(),
            "sim_exists": sim.exists(),
            "sta_exists": sta.exists(),
            "dat_exists": dat.exists(),
            "msg_exists": msg.exists(),
            "lck_exists": has_lck,
            "sta_success_marker": success,
            "failure_keywords": ";".join(failure_hits),
            "sta_tail": sta_tail[-1200:],
            "msg_tail": msg_tail[-1200:],
            "dat_tail": dat_tail[-1200:],
            "status": status,
        }
        rows.append(row)

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else ["status"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    status_counts = Counter(str(row.get("status", "")) for row in rows)
    completed_by_n = Counter(int(row["n"]) for row in rows if row.get("status") == "COMPLETED_SUCCESS" and row.get("n") != "")
    odb_by_n = Counter(int(row["n"]) for row in rows if row.get("odb_exists") and row.get("n") != "")
    lck_by_n = Counter(int(row["n"]) for row in rows if row.get("lck_exists") and row.get("n") != "")
    failure_rows = [row for row in rows if row.get("status") == "FAILED_OR_ERROR_MARKERS"]
    if len(rows) == 32 and sum(completed_by_n.values()) == 32 and not failure_rows:
        verdict = "PASS_STAGEF_PPO_BATCH32_SOLVER_COMPLETED_32_OF_32"
    elif failure_rows:
        verdict = "FAIL_STAGEF_PPO_BATCH32_SOLVER_FAILURES_DETECTED"
    else:
        verdict = "WARNING_STAGEF_PPO_BATCH32_SOLVER_RUNNING_OR_PARTIAL"

    summary = {
        "verdict": verdict,
        "case_root": str(CASE_ROOT),
        "command_count": len(commands),
        "status_counts": dict(status_counts),
        "completed_total": sum(completed_by_n.values()),
        "completed_by_n": {str(k): completed_by_n[k] for k in sorted(EXPECTED_COUNTS)},
        "odb_total": sum(odb_by_n.values()),
        "odb_by_n": {str(k): odb_by_n[k] for k in sorted(EXPECTED_COUNTS)},
        "active_lck_total": sum(lck_by_n.values()),
        "active_lck_by_n": {str(k): lck_by_n[k] for k in sorted(EXPECTED_COUNTS)},
        "failure_count": len(failure_rows),
        "failure_rows": failure_rows,
        "csv_path": str(CSV_PATH),
        "odb_opened": False,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(verdict)
    print(f"summary={SUMMARY_PATH}")
    return 0 if verdict != "FAIL_STAGEF_PPO_BATCH32_SOLVER_FAILURES_DETECTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
