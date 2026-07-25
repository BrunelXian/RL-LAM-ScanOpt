"""Create and validate run40 native_N24_N40_focused_batch60 abqjobpilot command file.

The file is generated for user-controlled enqueue only. This script never
executes abqjobpilot or enqueue.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
BATCH_NAME = "stage3_run39_native_N24_N40_focused_batch60_v01"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage3_run_40_native_N24_N40_focused_batch60_cae_inp_generation"
PLAN_CSV = OUTPUT_DIR / "run40_generation_plan.csv"
COMMAND_FILE = OUTPUT_DIR / "stage3_run40_native_N24_N40_focused_batch60_abqjobpilot_commands_READY_TO_RUN.txt"
CSV_PATH = OUTPUT_DIR / "stage3_run40_native_N24_N40_focused_batch60_abqjobpilot_command_validation.csv"
SUMMARY_PATH = OUTPUT_DIR / "stage3_run40_native_N24_N40_focused_batch60_abqjobpilot_command_validation_summary.json"
EXPECTED_COUNTS = {24: 30, 40: 30}
BAD_SCHEMA_TOKENS = ("N12N12_", "N16N16_", "N24N24_", "N40N40_")


def load_rows() -> list[dict[str, str]]:
    with PLAN_CSV.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def create_commands(rows: list[dict[str, str]]) -> list[str]:
    commands = []
    for row in rows:
        commands.append(
            'enqueue --inp "{}" --cpus 14 --batch {} --strategy {}'.format(
                row["inp_path"], BATCH_NAME, row["handoff_strategy_name"]
            )
        )
    COMMAND_FILE.write_text("\n".join(commands) + "\n", encoding="utf-8")
    return commands


def validate_line(line: str, row: dict[str, str]) -> dict[str, object]:
    inp = Path(row["inp_path"])
    strategy = row["handoff_strategy_name"]
    case_folder = Path(row["case_dir"]).name
    result: dict[str, object] = {
        "n": row["n"],
        "handoff_strategy_name": strategy,
        "command": line,
        "starts_enqueue_inp": line.startswith("enqueue --inp"),
        "has_cpus_14": "--cpus 14" in line,
        "has_gpus": "--gpus" in line,
        "inp_exists": inp.exists(),
        "batch_ok": "--batch {}".format(BATCH_NAME) in line,
        "strategy_matches_case_folder": case_folder.endswith(strategy),
        "bad_schema": any(token in line for token in BAD_SCHEMA_TOKENS),
        "forbidden_batch_reference": "batch32" in line.lower() or "option" in line.lower(),
        "has_forbidden_n_command": int(row["n"]) in (12, 16, 32) or case_folder.startswith(("N12", "N16", "N32")),
        "verdict": "FAIL",
    }
    checks = [
        result["starts_enqueue_inp"], result["has_cpus_14"], not result["has_gpus"],
        result["inp_exists"], result["batch_ok"], result["strategy_matches_case_folder"],
        not result["bad_schema"], not result["forbidden_batch_reference"], not result["has_forbidden_n_command"],
    ]
    result["verdict"] = "PASS" if all(checks) else "FAIL"
    return result


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    commands = create_commands(rows)
    results = [validate_line(line, row) for line, row in zip(commands, rows)]
    counts = Counter(int(row["n"]) for row in rows)
    errors = []
    if len(commands) != 60:
        errors.append("command count {} != 60".format(len(commands)))
    if dict(counts) != EXPECTED_COUNTS:
        errors.append("per-N command counts {} != {}".format(dict(counts), EXPECTED_COUNTS))
    if any(row["verdict"] != "PASS" for row in results):
        errors.append("one or more command validation rows failed")
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()) if results else ["verdict"])
        writer.writeheader()
        writer.writerows(results)
    verdict = "PASS_RUN40_ABQJOBPILOT_COMMAND_FILE_VALID" if not errors else "FAIL_RUN40_ABQJOBPILOT_COMMAND_FILE_INVALID"
    summary = {
        "verdict": verdict,
        "command_file": str(COMMAND_FILE),
        "command_count": len(commands),
        "per_n_command_counts": {str(n): counts[n] for n in EXPECTED_COUNTS},
        "csv_path": str(CSV_PATH),
        "errors": errors,
        "abqjobpilot_executed": False,
        "enqueue_executed": False,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(verdict)
    print("command_file={}".format(COMMAND_FILE))
    print("summary={}".format(SUMMARY_PATH))
    return 0 if verdict == "PASS_RUN40_ABQJOBPILOT_COMMAND_FILE_VALID" else 1


if __name__ == "__main__":
    raise SystemExit(main())

