"""Check regenerated N16/N24/N40 probe60 INPs for mesh and final cooling text.

This normal Python checker does not run Abaqus. It inspects generated INP text
for mesh sections, scan/cool sequence names, body heat flux entries, and the
final cooling increment controls required for the corrected bases.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
CASE_ROOT = PROJECT_ROOT / "cae_model" / "stage3_true_variable_N_probe60_v01"
MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "stage3_manual_probe60_handoff"
    / "variable_N_probe60_case_manifest_FIXED.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage3_cae_probe60_generation"
CSV_PATH = OUTPUT_DIR / "inp_mesh_section_check_N16_N24_N40.csv"
SUMMARY_PATH = OUTPUT_DIR / "inp_mesh_section_check_N16_N24_N40_summary.json"
TARGET_NS = {16, 24, 40}


def _read_manifest_rows() -> list[dict[str, str]]:
    with MANIFEST.open(newline="", encoding="utf-8-sig") as f:
        return [row for row in csv.DictReader(f) if int(row["n"]) in TARGET_NS]


def _count_section_entries(lines: list[str], section_keyword: str) -> int:
    count = 0
    in_section = False
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith(section_keyword.lower()):
            in_section = True
            continue
        if in_section and stripped.startswith("*"):
            break
        if in_section and stripped and not stripped.startswith("**"):
            count += 1
    return count


def _find_line_index(lines: list[str], pattern: str) -> int | None:
    regex = re.compile(pattern, re.IGNORECASE)
    for idx, line in enumerate(lines):
        if regex.search(line):
            return idx
    return None


def _final_cooling_controls_visible(lines: list[str]) -> tuple[bool, str]:
    step_idx = _find_line_index(lines, r"\*Step,\s*name=step_final_cooling\b")
    if step_idx is None:
        return False, "missing step_final_cooling step line"
    window = lines[step_idx : step_idx + 12]
    joined = "\n".join(window)
    has_initial = re.search(r"(^|[^0-9.])0\.01([^0-9.]|$)", joined) is not None
    has_duration = re.search(r"(^|[^0-9.])1200\.?([^0-9.]|$)", joined) is not None
    has_max_inc = re.search(r"(^|[^0-9.])60\.?([^0-9.]|$)", joined) is not None
    if has_initial and has_duration and has_max_inc:
        return True, "visible in INP near step_final_cooling"
    return False, "final cooling controls not fully visible near step_final_cooling"


def _check_one(row: dict[str, str]) -> dict[str, object]:
    n = int(row["n"])
    inp = Path(row["expected_inp"])
    result: dict[str, object] = {
        "n": n,
        "strategy_name": row["strategy_name"],
        "inp_path": str(inp),
        "exists": inp.exists(),
        "size_bytes": inp.stat().st_size if inp.exists() else 0,
        "node_section_exists": False,
        "element_section_exists": False,
        "node_entry_count": 0,
        "element_entry_count": 0,
        "step_final_cooling_exists": False,
        "final_cooling_controls_visible": False,
        "final_cooling_control_evidence": "",
        "scan_sequence_complete": False,
        "cool_sequence_complete": False,
        "heat_flux_entries_exist": False,
        "verdict": "FAIL_INP_MESH_SECTION_CHECK",
        "notes": "",
    }
    if not inp.exists() or inp.stat().st_size <= 0:
        result["notes"] = "missing or empty INP"
        return result

    text = inp.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    lower = text.lower()
    node_entries = _count_section_entries(lines, "*Node")
    element_entries = _count_section_entries(lines, "*Element")
    controls_ok, controls_note = _final_cooling_controls_visible(lines)
    scan_complete = all("step_scan_{:02d}".format(seq).lower() in lower for seq in range(n))
    cool_complete = all("step_cool_{:02d}".format(seq).lower() in lower for seq in range(n))
    heat_flux = "body heat flux" in lower or "*dflux" in lower or "bodyheatflux" in lower

    result.update(
        {
            "node_section_exists": "*node" in lower,
            "element_section_exists": "*element" in lower,
            "node_entry_count": node_entries,
            "element_entry_count": element_entries,
            "step_final_cooling_exists": "step_final_cooling" in lower,
            "final_cooling_controls_visible": controls_ok,
            "final_cooling_control_evidence": controls_note,
            "scan_sequence_complete": scan_complete,
            "cool_sequence_complete": cool_complete,
            "heat_flux_entries_exist": heat_flux,
        }
    )
    checks = [
        result["node_section_exists"],
        result["element_section_exists"],
        node_entries > 0,
        element_entries > 0,
        result["step_final_cooling_exists"],
        controls_ok,
        scan_complete,
        cool_complete,
        heat_flux,
    ]
    if all(checks):
        result["verdict"] = "PASS_INP_MESH_SECTION_CHECK"
    else:
        failed = [
            name
            for name, ok in [
                ("node_section", result["node_section_exists"]),
                ("element_section", result["element_section_exists"]),
                ("node_entries", node_entries > 0),
                ("element_entries", element_entries > 0),
                ("step_final_cooling", result["step_final_cooling_exists"]),
                ("final_cooling_controls", controls_ok),
                ("scan_sequence", scan_complete),
                ("cool_sequence", cool_complete),
                ("heat_flux_entries", heat_flux),
            ]
            if not ok
        ]
        result["notes"] = "failed checks: {}".format(";".join(failed))
    return result


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = _read_manifest_rows()
    results = [_check_one(row) for row in rows]
    fieldnames = [
        "n",
        "strategy_name",
        "inp_path",
        "exists",
        "size_bytes",
        "node_section_exists",
        "element_section_exists",
        "node_entry_count",
        "element_entry_count",
        "step_final_cooling_exists",
        "final_cooling_controls_visible",
        "final_cooling_control_evidence",
        "scan_sequence_complete",
        "cool_sequence_complete",
        "heat_flux_entries_exist",
        "verdict",
        "notes",
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    per_n_counts = {
        str(n): sum(1 for result in results if int(result["n"]) == n)
        for n in sorted(TARGET_NS)
    }
    overall = (
        "PASS_N16_N24_N40_INP_MESH_SECTIONS_READY"
        if len(results) == 45 and all(result["verdict"] == "PASS_INP_MESH_SECTION_CHECK" for result in results)
        else "FAIL_N16_N24_N40_INP_MESH_SECTIONS_INVALID"
    )
    summary = {
        "verdict": overall,
        "checked_count": len(results),
        "per_n_counts": per_n_counts,
        "failed": [result for result in results if result["verdict"] != "PASS_INP_MESH_SECTION_CHECK"],
        "csv_path": str(CSV_PATH),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(overall)
    print("checked_count={}".format(len(results)))
    print("summary={}".format(SUMMARY_PATH))
    return 0 if overall == "PASS_N16_N24_N40_INP_MESH_SECTIONS_READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
