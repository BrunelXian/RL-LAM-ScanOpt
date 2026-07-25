from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TARGET_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
RUN_ID = "run_07_variable_n_probe60_model_generation_handoff"
BATCH_NAME = "stage3_true_variable_N_probe60_v01"
BASE_OUTPUT_ROOT = TARGET_ROOT / "cae_model" / BATCH_NAME
RUN06_DIR = TARGET_ROOT / "outputs" / "stage3_run_06_variable_n_probe60_candidate_order_generation"
RUN05_DIR = TARGET_ROOT / "outputs" / "stage3_run_05_true_variable_n_base_model_inventory"
OUTPUT_DIR = TARGET_ROOT / "outputs" / "stage3_run_07_variable_n_probe60_model_generation_handoff"
REPORT_DIR = TARGET_ROOT / "docs" / "stage3" / "runs" / RUN_ID
REPORT_PATH = REPORT_DIR / "RUN_07_VARIABLE_N_PROBE60_MODEL_GENERATION_HANDOFF_REPORT.md"
MANIFEST_PATH = TARGET_ROOT / "artifacts" / "manifests" / "stage3_run_07_manifest.json"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def strategy_dir_name(n: int, strategy_name: str) -> str:
    return f"N{n}{strategy_name}"


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    candidate_path = RUN06_DIR / "variable_N_probe60_candidate_orders.csv"
    base_path = RUN05_DIR / "base_model_inventory.csv"
    candidates = read_csv(candidate_path)
    bases = {int(row["n"]): row for row in read_csv(base_path)}
    case_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    linkage_rows: list[dict[str, Any]] = []
    commands: list[str] = []
    case_json: list[dict[str, Any]] = []
    for row in candidates:
        n = int(row["n"])
        strategy_name = row["strategy_name"]
        case_dir = BASE_OUTPUT_ROOT / strategy_dir_name(n, strategy_name)
        cae_path = case_dir / f"J2D_N{n}_{strategy_name}.cae"
        inp_path = case_dir / f"J2D_N{n}_{strategy_name}.inp"
        scan_json = case_dir / f"scan_order_{strategy_name}.json"
        metadata_json = case_dir / f"strategy_metadata_{strategy_name}.json"
        base = bases.get(n, {})
        case = {
            "n": n,
            "strategy_id": row["strategy_id"],
            "strategy_name": strategy_name,
            "family": row["family"],
            "batch_name": BATCH_NAME,
            "case_dir": str(case_dir),
            "expected_cae_path": str(cae_path),
            "expected_inp_path": str(inp_path),
            "scan_order_json_path": str(scan_json),
            "strategy_metadata_path": str(metadata_json),
            "base_cae_path": base.get("cae_path", ""),
            "base_jnl_path": base.get("jnl_path", ""),
            "teacher_validated": False,
            "generation_status": "handoff_only_not_generated",
        }
        case_rows.append(case)
        path_rows.append(
            {
                "n": n,
                "strategy_name": strategy_name,
                "case_dir": str(case_dir),
                "expected_cae_path": str(cae_path),
                "expected_inp_path": str(inp_path),
                "scan_order_json_path": str(scan_json),
                "strategy_metadata_path": str(metadata_json),
            }
        )
        linkage_rows.append(
            {
                "n": n,
                "strategy_name": strategy_name,
                "base_cae_path": base.get("cae_path", ""),
                "base_cae_exists": base.get("cae_exists", ""),
                "base_jnl_path": base.get("jnl_path", ""),
                "base_jnl_exists": base.get("jnl_exists", ""),
                "true_variable_n_route": base.get("true_variable_n_route", ""),
            }
        )
        commands.append(f'enqueue --inp "{inp_path}" --cpus 14 --batch {BATCH_NAME} --strategy {strategy_name}')
        case_json.append(
            {
                **case,
                "scan_order_payload": {
                    "n": n,
                    "strategy_name": strategy_name,
                    "scan_order": json.loads(row["order_json"]),
                },
                "strategy_metadata_payload": row,
            }
        )
    verdict = "PASS_VARIABLE_N_PROBE60_MODEL_GENERATION_HANDOFF_READY"
    if len(case_rows) != 60 or len(commands) != 60:
        verdict = "FAIL_VARIABLE_N_PROBE60_HANDOFF_INVALID"
    manifest_csv = OUTPUT_DIR / "variable_N_probe60_case_manifest.csv"
    write_csv(manifest_csv, case_rows, list(case_rows[0].keys()))
    manifest_json = OUTPUT_DIR / "variable_N_probe60_case_manifest.json"
    manifest_json.write_text(json.dumps(case_json, indent=2) + "\n", encoding="utf-8")
    paths_csv = OUTPUT_DIR / "variable_N_probe60_expected_paths.csv"
    write_csv(paths_csv, path_rows, list(path_rows[0].keys()))
    linkage_csv = OUTPUT_DIR / "variable_N_probe60_base_model_linkage.csv"
    write_csv(linkage_csv, linkage_rows, list(linkage_rows[0].keys()))
    plan_md = OUTPUT_DIR / "variable_N_probe60_model_generation_plan.md"
    plan_md.write_text(
        f"""# Variable-N Probe60 Model Generation Plan

Batch: `{BATCH_NAME}`

This is a handoff plan only. Do not run it inside this task. It expects user-reviewed generation from the manual true variable-N CAE bases and the run_06 scan-order candidates.

No CAE, INP, JNL, ODB, SIM, DAT, MSG, STA, or LCK files are generated by run_07.
""",
        encoding="utf-8",
    )
    command_path = OUTPUT_DIR / "variable_N_probe60_abqjobpilot_commands.txt"
    command_path.write_text("\n".join(commands) + "\n", encoding="utf-8")
    stop_path = OUTPUT_DIR / "variable_N_probe60_STOP_BEFORE_RUNNING.md"
    stop_path.write_text(
        "# Stop Before Running\n\nThese commands are a handoff only. Do not execute abqjobpilot or Abaqus until the user explicitly decides to run the generated INP files.\n",
        encoding="utf-8",
    )
    scripts_dir = TARGET_ROOT / "scripts" / "stage3"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    template_path = scripts_dir / "generate_variable_N_probe60_from_base_TEMPLATE.py"
    template_path.write_text(
        '"""Template only. Not executed by Codex.\n\nRequires user review and likely Abaqus Python / abaqus cae noGUI.\nDo not run this script until generation behavior is reviewed.\n"""\n\nraise SystemExit("Template only: review before running with Abaqus Python.")\n',
        encoding="utf-8",
    )
    readme_path = scripts_dir / "README_STAGE3_VARIABLE_N_GENERATION_HANDOFF.md"
    readme_path.write_text(
        "# Stage 3 Variable-N Generation Handoff\n\nThis folder contains text-only handoff templates. Codex did not run Abaqus, datacheck, abqjobpilot, or model generation.\n",
        encoding="utf-8",
    )
    outputs = [str(manifest_csv), str(manifest_json), str(paths_csv), str(linkage_csv), str(plan_md), str(command_path), str(stop_path), str(template_path), str(readme_path)]
    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "run_id": RUN_ID,
                "python_executable": sys.executable,
                "verdict": verdict,
                "batch_name": BATCH_NAME,
                "total_cases": len(case_rows),
                "abqjobpilot_command_count": len(commands),
                "outputs_written": outputs,
                "forbidden_actions_confirmed": {
                    "no_abaqus_jobs": True,
                    "no_datacheck": True,
                    "no_odb_opened": True,
                    "no_cae_generated": True,
                    "no_inp_generated": True,
                    "no_jnl_generated_except_template_text": True,
                    "no_abqjobpilot_execution": True,
                    "no_job_submission": True,
                    "no_teacher_validation": True,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        f"""# Run 07 Variable-N Probe60 Model Generation Handoff Report

## Executive Verdict

{verdict}

Prepared manifest-only handoff data for `{len(case_rows)}` cases and `{len(commands)}` abqjobpilot enqueue commands. No model generation or job submission was performed.

## Guardrails

- No Abaqus jobs.
- No datacheck.
- No ODB opened.
- No CAE generated.
- No INP generated.
- No JNL generated unless template text only.
- No abqjobpilot execution.
- No job submission.
- No teacher validation.

## Stop Point

Stop here until the user reviews the generated handoff and explicitly chooses to run generation/Abaqus externally.

## Outputs

- `{manifest_csv}`
- `{manifest_json}`
- `{paths_csv}`
- `{linkage_csv}`
- `{plan_md}`
- `{command_path}`
- `{stop_path}`
- `{template_path}`
- `{readme_path}`
- `{MANIFEST_PATH}`
""",
        encoding="utf-8",
    )
    print(verdict)
    print(f"total_cases={len(case_rows)}")
    print(f"abqjobpilot_commands={len(commands)}")
    return 1 if verdict.startswith("FAIL") else 0


if __name__ == "__main__":
    raise SystemExit(main())
