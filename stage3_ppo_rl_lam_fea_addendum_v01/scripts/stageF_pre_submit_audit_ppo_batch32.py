"""Pre-submit audit for Stage F PPO-only batch32 enqueue.

This script performs read-only checks and never submits jobs.
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
CSV_PATH = CHECKS_DIR / "stageF_pre_submit_audit.csv"
SUMMARY_PATH = CHECKS_DIR / "stageF_pre_submit_audit_summary.json"

EXPECTED_COUNTS = {12: 8, 16: 8, 24: 8, 40: 8}
SOLVER_EXTS = {".odb", ".sim", ".sta", ".dat", ".msg", ".lck"}
BAD_SCHEMA_TOKENS = ("N12N12", "N16N16", "N24N24", "N40N40")
COMMAND_RE = re.compile(r'^enqueue --inp "([^"]+)" --cpus 14 --batch (\S+) --strategy (\S+)\s*$')


def parse_n_from_path(inp_path: Path) -> int | None:
    for part in inp_path.parts:
        match = re.match(r"^N(12|16|24|40)[_A-Za-z0-9-]", part)
        if match:
            return int(match.group(1))
    return None


def main() -> int:
    CHECKS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    errors: list[str] = []
    warnings: list[str] = []

    commands: list[str] = []
    if not COMMAND_FILE.exists():
        errors.append(f"approved command file missing: {COMMAND_FILE}")
    else:
        commands = [line.strip() for line in COMMAND_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]

    if len(commands) != 32:
        errors.append(f"command count {len(commands)} != 32")

    n_counter: Counter[int] = Counter()
    referenced_inps: list[str] = []
    for idx, command in enumerate(commands, 1):
        row: dict[str, object] = {
            "command_index": idx,
            "command_text": command,
            "starts_enqueue_inp": command.startswith("enqueue --inp"),
            "has_cpus_14": "--cpus 14" in command,
            "has_gpus": "--gpus" in command,
            "batch_ok": f"--batch {BATCH_NAME}" in command,
            "strategy_name": "",
            "inp_path": "",
            "n": "",
            "inp_exists": False,
            "inp_nonzero": False,
            "under_case_root": False,
            "bad_schema": any(token in command for token in BAD_SCHEMA_TOKENS),
            "is_n32": "N32" in command,
            "verdict": "FAIL",
            "notes": "",
        }
        match = COMMAND_RE.match(command)
        if not match:
            row["notes"] = "command does not match expected format"
            rows.append(row)
            continue
        inp_path = Path(match.group(1))
        strategy = match.group(3)
        n_value = parse_n_from_path(inp_path)
        row["strategy_name"] = strategy
        row["inp_path"] = str(inp_path)
        row["n"] = n_value if n_value is not None else ""
        row["inp_exists"] = inp_path.exists()
        row["inp_nonzero"] = inp_path.exists() and inp_path.stat().st_size > 0
        row["under_case_root"] = inp_path.is_relative_to(CASE_ROOT) if hasattr(inp_path, "is_relative_to") else str(inp_path).startswith(str(CASE_ROOT))
        if n_value is not None:
            n_counter[n_value] += 1
        referenced_inps.append(str(inp_path))
        checks = [
            row["starts_enqueue_inp"],
            row["has_cpus_14"],
            not row["has_gpus"],
            row["batch_ok"],
            row["inp_exists"],
            row["inp_nonzero"],
            row["under_case_root"],
            not row["bad_schema"],
            not row["is_n32"],
            n_value in EXPECTED_COUNTS,
        ]
        row["verdict"] = "PASS" if all(checks) else "FAIL"
        rows.append(row)

    if dict(n_counter) != EXPECTED_COUNTS:
        errors.append(f"N distribution {dict(n_counter)} != {EXPECTED_COUNTS}")
    if len(set(referenced_inps)) != len(referenced_inps):
        errors.append("duplicate INP references in command file")
    if any(row["verdict"] != "PASS" for row in rows):
        errors.append("one or more command rows failed audit")

    existing_solver_outputs: list[str] = []
    active_locks: list[str] = []
    if not CASE_ROOT.exists():
        errors.append(f"case root missing: {CASE_ROOT}")
    else:
        for path in CASE_ROOT.rglob("*"):
            if not path.is_file():
                continue
            ext = path.suffix.lower()
            if ext in SOLVER_EXTS:
                existing_solver_outputs.append(str(path))
                if ext == ".lck":
                    active_locks.append(str(path))

    if active_locks:
        errors.append(f"active .lck files exist under case root: {len(active_locks)}")
    existing_nonlock_solver = [p for p in existing_solver_outputs if not p.lower().endswith(".lck")]
    if existing_nonlock_solver:
        errors.append(f"existing solver outputs under case root before submission: {len(existing_nonlock_solver)}")

    if errors:
        verdict = "FAIL_STAGEF_PPO_BATCH32_NOT_SAFE_TO_ENQUEUE"
    elif warnings:
        verdict = "WARNING_STAGEF_PPO_BATCH32_REVIEW_BEFORE_ENQUEUE"
    else:
        verdict = "PASS_STAGEF_PPO_BATCH32_READY_TO_ENQUEUE"

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else ["verdict"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "verdict": verdict,
        "command_file": str(COMMAND_FILE),
        "case_root": str(CASE_ROOT),
        "command_count": len(commands),
        "n_distribution": {str(k): n_counter[k] for k in sorted(EXPECTED_COUNTS)},
        "expected_distribution": {str(k): v for k, v in EXPECTED_COUNTS.items()},
        "existing_solver_output_count": len(existing_solver_outputs),
        "existing_solver_outputs": existing_solver_outputs,
        "active_lck_count": len(active_locks),
        "active_lck_files": active_locks,
        "csv_path": str(CSV_PATH),
        "errors": errors,
        "warnings": warnings,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(verdict)
    print(f"summary={SUMMARY_PATH}")
    return 0 if verdict == "PASS_STAGEF_PPO_BATCH32_READY_TO_ENQUEUE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
