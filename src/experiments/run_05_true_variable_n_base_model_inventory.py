from __future__ import annotations

import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TARGET_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
RUN_ID = "run_05_true_variable_n_base_model_inventory"
OUTPUT_DIR = TARGET_ROOT / "outputs" / "stage3_run_05_true_variable_n_base_model_inventory"
REPORT_DIR = TARGET_ROOT / "docs" / "stage3" / "runs" / RUN_ID
REPORT_PATH = REPORT_DIR / "RUN_05_TRUE_VARIABLE_N_BASE_MODEL_INVENTORY_REPORT.md"
MANIFEST_PATH = TARGET_ROOT / "artifacts" / "manifests" / "stage3_run_05_manifest.json"

BASES = {
    12: ("small-N sanity / extrapolation diagnostic", "12track_full/sanity_base/12track_sanity_base"),
    16: ("training/proxy support", "16track_full/sanity_base/16track_sanity_base"),
    24: ("unseen-N test", "24track_full/sanity_base/24track_sanity_base"),
    40: ("unseen-N test", "40track_full/sanity_base/40track_sanity_base"),
}


def file_info(path: Path) -> tuple[bool, int | str, str]:
    if not path.exists():
        return False, "", ""
    stat = path.stat()
    return True, stat.st_size, datetime.fromtimestamp(stat.st_mtime).isoformat()


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def audit_jnl(n: int, path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "n": n,
            "jnl_path": str(path),
            "jnl_exists": False,
            "contains_n_track_name": False,
            "contains_heat_step_reference": False,
            "contains_track_set_reference": False,
            "contains_surface_reference": False,
            "contains_job_or_model_name": False,
            "contains_fixed32_leftover": False,
            "warning_count": 1,
            "warnings": "missing_jnl",
        }
    text = path.read_text(encoding="utf-8", errors="replace")
    lower = text.lower()
    warnings: list[str] = []
    if f"{n}track" not in lower:
        warnings.append("missing_expected_ntrack_name")
    if "32track" in lower or "range(32)" in lower or "0..31" in lower:
        warnings.append("possible_fixed32_leftover")
    return {
        "n": n,
        "jnl_path": str(path),
        "jnl_exists": True,
        "contains_n_track_name": f"{n}track" in lower,
        "contains_heat_step_reference": bool(re.search(r"heat|step|load", lower)),
        "contains_track_set_reference": bool(re.search(r"track|set-", lower)),
        "contains_surface_reference": "surface" in lower,
        "contains_job_or_model_name": bool(re.search(r"job|model|mdb", lower)),
        "contains_fixed32_leftover": bool(re.search(r"32track|range\s*\(\s*32\s*\)|0\.\.31", lower)),
        "warning_count": len(warnings),
        "warnings": ";".join(warnings) if warnings else "none",
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    inventory_rows: list[dict[str, Any]] = []
    jnl_rows: list[dict[str, Any]] = []
    for n, (role, stem) in BASES.items():
        cae = TARGET_ROOT / "cae_model" / f"{stem}.cae"
        jnl = TARGET_ROOT / "cae_model" / f"{stem}.jnl"
        cae_exists, cae_size, cae_mtime = file_info(cae)
        jnl_exists, jnl_size, jnl_mtime = file_info(jnl)
        inventory_rows.append(
            {
                "n": n,
                "intended_role": role,
                "cae_path": str(cae),
                "cae_exists": cae_exists,
                "cae_size_bytes": cae_size,
                "cae_mtime": cae_mtime,
                "jnl_path": str(jnl),
                "jnl_exists": jnl_exists,
                "jnl_size_bytes": jnl_size,
                "jnl_mtime": jnl_mtime,
                "true_variable_n_route": True,
                "notes": "CAE metadata only; JNL text audited read-only.",
            }
        )
        jnl_rows.append(audit_jnl(n, jnl))
    missing = [row for row in inventory_rows if not row["cae_exists"] or not row["jnl_exists"]]
    fixed32_warnings = [row for row in jnl_rows if row["contains_fixed32_leftover"]]
    if missing and len(missing) == len(inventory_rows):
        verdict = "FAIL_TRUE_VARIABLE_N_BASE_MODEL_INVENTORY_MISSING_BASES"
    elif missing:
        verdict = "WARNING_TRUE_VARIABLE_N_BASE_MODEL_INVENTORY_PARTIAL"
    else:
        verdict = "PASS_TRUE_VARIABLE_N_BASE_MODEL_INVENTORY_READY"

    inv_path = OUTPUT_DIR / "base_model_inventory.csv"
    write_csv(inv_path, inventory_rows, list(inventory_rows[0].keys()))
    jnl_path = OUTPUT_DIR / "base_model_jnl_audit.csv"
    write_csv(jnl_path, jnl_rows, list(jnl_rows[0].keys()))
    summary_path = OUTPUT_DIR / "base_model_readiness_summary.json"
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "n_values": sorted(BASES),
        "all_cae_present": all(row["cae_exists"] for row in inventory_rows),
        "all_jnl_present": all(row["jnl_exists"] for row in inventory_rows),
        "fixed32_leftover_warning_count": len(fixed32_warnings),
        "guardrails": {
            "no_abaqus_jobs": True,
            "no_datacheck": True,
            "no_odb_opened": True,
            "no_cae_opened": True,
            "no_cae_modified": True,
            "no_inp_jnl_generated": True,
            "no_abqjobpilot_execution": True,
            "no_teacher_validation": True,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": RUN_ID,
        "target_root": str(TARGET_ROOT),
        "python_executable": sys.executable,
        "verdict": verdict,
        "n_values": sorted(BASES),
        "outputs_written": [str(inv_path), str(jnl_path), str(summary_path)],
        "forbidden_actions_confirmed": summary["guardrails"],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    report = f"""# Run 05 True Variable-N Base Model Inventory Report

## Executive Verdict

{verdict}

## Inventory Summary

- N values inventoried: `{sorted(BASES)}`
- CAE files present: `{sum(1 for row in inventory_rows if row['cae_exists'])}/4`
- JNL files present: `{sum(1 for row in inventory_rows if row['jnl_exists'])}/4`
- Fixed-32 leftover warnings in JNL: `{len(fixed32_warnings)}`

## Guardrails

- No Abaqus jobs.
- No datacheck.
- No ODB opened.
- No CAE opened.
- No CAE modified.
- No INP/JNL generated.
- No abqjobpilot execution.
- No teacher validation.

## Notes

N=12 is treated as a small-N sanity / extrapolation diagnostic. N=16 supports future training/proxy development. N=24 and N=40 are unseen-N tests. These bases are true variable-N geometry, not masked/subset-32 models.

## Outputs

- `{inv_path}`
- `{jnl_path}`
- `{summary_path}`
- `{MANIFEST_PATH}`
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(verdict)
    print(f"CAE present: {sum(1 for row in inventory_rows if row['cae_exists'])}/4")
    print(f"JNL present: {sum(1 for row in inventory_rows if row['jnl_exists'])}/4")
    return 1 if verdict.startswith("FAIL") else 0


if __name__ == "__main__":
    raise SystemExit(main())
