"""Check Stage 3 run40 native_N24_N40_focused_batch60 generated INP files."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage3_run_65_variable_N_recovery_anchor_batch48_cae_inp_generation"
PLAN_CSV = OUTPUT_DIR / "run65_generation_plan.csv"
CSV_PATH = OUTPUT_DIR / "stage3_run65_variable_N_recovery_anchor_batch48_generated_inp_check.csv"
SUMMARY_PATH = OUTPUT_DIR / "stage3_run65_variable_N_recovery_anchor_batch48_generated_inp_check_summary.json"
CASE_ROOT = PROJECT_ROOT / "cae_model" / "stage3_run64_variable_N_recovery_anchor_batch48_v01"
EXPECTED_COUNTS = {12: 12, 16: 12, 24: 8, 40: 16}
SOLVER_EXTS = (".odb", ".sim", ".sta", ".dat", ".msg", ".lck")
NUMERIC_RE = r"[-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?"


def load_rows() -> list[dict[str, object]]:
    rows = []
    with PLAN_CSV.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            row["n"] = int(row["n"])
            row["scan_order"] = json.loads(row["scan_order"])
            rows.append(row)
    return rows


def section_entry_count(lines: list[str], keyword: str) -> int:
    count = 0
    in_section = False
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith(keyword.lower()):
            in_section = True
            continue
        if in_section and stripped.startswith("*"):
            break
        if in_section and stripped and not stripped.startswith("**"):
            count += 1
    return count


def final_cooling_visible(lines: list[str], n: int) -> tuple[bool, str, bool]:
    last_cool_index = -1
    final_index = -1
    for i, line in enumerate(lines):
        if "step_cool_{:02d}".format(n - 1).lower() in line.lower():
            last_cool_index = max(last_cool_index, i)
        if re.search(r"\*Step,\s*name=step_final_cooling\b", line, re.I):
            final_index = i
            window = "\n".join(lines[i : i + 12])
            has_initial = re.search(r"(^|[^0-9.])0\.01([^0-9.]|$)", window) is not None
            has_duration = re.search(r"(^|[^0-9.])1200\.?([^0-9.]|$)", window) is not None
            has_max = re.search(r"(^|[^0-9.])60\.?([^0-9.]|$)", window) is not None
            return has_initial and has_duration and has_max, window.replace("\r", ""), final_index > last_cool_index
    return False, "", False


def step_blocks(lines: list[str]) -> dict[str, str]:
    blocks: dict[str, list[str]] = {}
    current = None
    for line in lines:
        m = re.match(r"\*\* STEP:\s*(\S+)", line.strip(), re.I)
        if m:
            current = m.group(1)
            blocks[current] = []
            continue
        if current is not None:
            blocks[current].append(line)
    return {key: "\n".join(value) for key, value in blocks.items()}


def heat_order_matches(lines: list[str], n: int, order: list[int]) -> tuple[bool, str]:
    blocks = step_blocks(lines)
    observed = []
    for seq in range(n):
        block = blocks.get("step_scan_{:02d}".format(seq), "")
        m = re.search(r"set_body_heat_(\d+)\s*,\s*BF\s*,", block, re.I)
        if not m:
            return False, "missing BF set in step_scan_{:02d}".format(seq)
        observed.append(int(m.group(1)))
    if observed != order:
        return False, "observed {} != expected {}".format(observed, order)
    final_block = blocks.get("step_final_cooling", "")
    if re.search(r"\*Dflux", final_block, re.I):
        return False, "Dflux appears in step_final_cooling"
    return True, "observed order matches scan_order"


def n40_cool_initial_inc_visible(lines: list[str], n: int) -> tuple[bool, str]:
    if n != 40:
        return True, "not an N40 case"
    bad = []
    seen = 0
    for i, line in enumerate(lines):
        m = re.match(r"\*Step,\s*name=(step_cool_\d+)", line, re.I)
        if not m:
            continue
        seen += 1
        data_line = ""
        for candidate in lines[i : i + 8]:
            stripped = candidate.strip()
            if re.match(r"^" + NUMERIC_RE + r"\s*,", stripped):
                data_line = stripped
                break
        vals = [float(v) for v in re.findall(NUMERIC_RE, data_line)]
        if len(vals) < 2 or abs(vals[0] - 0.001) > 1e-12 or abs(vals[1] - 3.4) > 1e-9:
            bad.append("{}:{}".format(m.group(1), data_line))
    if seen != 40:
        return False, "N40 cool step count {} != 40".format(seen)
    if bad:
        return False, "; ".join(bad[:3])
    return True, "all N40 cool steps show initialInc=0.001 and timePeriod=3.4"


def check_one(row: dict[str, object]) -> dict[str, object]:
    n = int(row["n"])
    inp = Path(str(row["inp_path"]))
    case_dir = Path(str(row["case_dir"]))
    result: dict[str, object] = {
        "n": n,
        "handoff_strategy_name": row["handoff_strategy_name"],
        "inp_path": str(inp),
        "exists": inp.exists(),
        "size_bytes": inp.stat().st_size if inp.exists() else 0,
        "node_section_exists": False,
        "element_section_exists": False,
        "node_entry_count": 0,
        "element_entry_count": 0,
        "scan_sequence_complete": False,
        "cool_sequence_complete": False,
        "step_final_cooling_exists": False,
        "final_cooling_after_last_cool": False,
        "final_cooling_controls_visible": False,
        "heat_flux_entries_exist": False,
        "all_expected_heat_sets_present": False,
        "heat_order_text_verified": False,
        "heat_order_note": "",
        "n40_cool_initialInc_0p001_verified": False,
        "n40_cool_initialInc_note": "",
        "solver_output_count": 0,
        "verdict": "FAIL",
        "notes": "",
    }
    result["solver_output_count"] = sum(len(list(case_dir.glob("*{}".format(ext)))) for ext in SOLVER_EXTS)
    if not inp.exists() or inp.stat().st_size <= 0:
        result["notes"] = "missing or empty INP"
        return result
    text = inp.read_text(encoding="utf-8", errors="replace")
    lower = text.lower()
    lines = text.splitlines()
    node_count = section_entry_count(lines, "*Node")
    elem_count = section_entry_count(lines, "*Element")
    final_ok, final_evidence, final_after = final_cooling_visible(lines, n)
    order_ok, order_note = heat_order_matches(lines, n, list(row["scan_order"]))
    n40_cool_ok, n40_cool_note = n40_cool_initial_inc_visible(lines, n)
    result.update(
        {
            "node_section_exists": "*node" in lower,
            "element_section_exists": "*element" in lower,
            "node_entry_count": node_count,
            "element_entry_count": elem_count,
            "scan_sequence_complete": all("step_scan_{:02d}".format(seq).lower() in lower for seq in range(n)),
            "cool_sequence_complete": all("step_cool_{:02d}".format(seq).lower() in lower for seq in range(n)),
            "step_final_cooling_exists": "step_final_cooling" in lower,
            "final_cooling_after_last_cool": final_after,
            "final_cooling_controls_visible": final_ok,
            "heat_flux_entries_exist": "body heat flux" in lower or "*dflux" in lower,
            "all_expected_heat_sets_present": all("set_body_heat_{:02d}".format(track).lower() in lower for track in range(n)),
            "heat_order_text_verified": order_ok,
            "heat_order_note": order_note,
            "n40_cool_initialInc_0p001_verified": n40_cool_ok,
            "n40_cool_initialInc_note": n40_cool_note,
        }
    )
    checks = [
        result["node_section_exists"], result["element_section_exists"], node_count > 0, elem_count > 0,
        result["scan_sequence_complete"], result["cool_sequence_complete"], result["step_final_cooling_exists"],
        result["final_cooling_after_last_cool"], result["final_cooling_controls_visible"], result["heat_flux_entries_exist"],
        result["all_expected_heat_sets_present"], result["heat_order_text_verified"], result["solver_output_count"] == 0,
        result["n40_cool_initialInc_0p001_verified"],
    ]
    result["verdict"] = "PASS" if all(checks) else "FAIL"
    if not final_ok:
        result["notes"] = final_evidence[:400]
    return result


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    results = [check_one(row) for row in rows]
    fieldnames = list(results[0].keys()) if results else []
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    per_n = {str(n): sum(1 for row in results if int(row["n"]) == n and row["verdict"] == "PASS") for n in EXPECTED_COUNTS}
    cae_count = len(list(CASE_ROOT.rglob("*.cae"))) if CASE_ROOT.exists() else 0
    inp_count = len(list(CASE_ROOT.rglob("*.inp"))) if CASE_ROOT.exists() else 0
    forbidden_inp_count = 0
    if CASE_ROOT.exists():
        forbidden_inp_count = len([
            path for path in CASE_ROOT.rglob("*.inp")
            if any(part.startswith(prefix) for part in path.parts for prefix in ("N32",))
        ])
    solver_count = sum(len(list(CASE_ROOT.rglob("*{}".format(ext)))) for ext in SOLVER_EXTS) if CASE_ROOT.exists() else 0
    verdict = "PASS_RUN65_VARIABLE_N_RECOVERY_ANCHOR_BATCH48_48_INPS_READY_FOR_USER_REVIEW" if len(results) == 48 and all(row["verdict"] == "PASS" for row in results) and inp_count == 48 and cae_count == 48 and solver_count == 0 and forbidden_inp_count == 0 else "FAIL_RUN65_VARIABLE_N_RECOVERY_ANCHOR_BATCH48_INP_CHECK_INVALID"
    summary = {
        "verdict": verdict,
        "checked_count": len(results),
        "pass_count": sum(1 for row in results if row["verdict"] == "PASS"),
        "per_n_pass_counts": per_n,
        "total_cae_count": cae_count,
        "total_inp_count": inp_count,
        "solver_output_count": solver_count,
        "forbidden_N32_inp_count": forbidden_inp_count,
        "failed": [row for row in results if row["verdict"] != "PASS"],
        "csv_path": str(CSV_PATH),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(verdict)
    print("summary={}".format(SUMMARY_PATH))
    return 0 if verdict == "PASS_RUN65_VARIABLE_N_RECOVERY_ANCHOR_BATCH48_48_INPS_READY_FOR_USER_REVIEW" else 1


if __name__ == "__main__":
    raise SystemExit(main())




