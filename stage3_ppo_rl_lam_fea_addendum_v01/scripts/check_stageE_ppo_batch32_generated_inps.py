"""Check Stage E PPO batch32 generated CAE/INP files.

Normal Python only. This script inspects generated text/files and does not run
solver, datacheck, abqjobpilot, enqueue, or ODB extraction.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
NAMESPACE = "stage3_ppo_rl_lam_fea_addendum_v01"
BATCH = "stage3_ppo_policy_only_batch32_v01"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / NAMESPACE / "stageE_teacher_validation_handoff"
MANIFEST_CSV = OUTPUT_ROOT / "manifest" / "stageE_ppo_batch32_case_manifest.csv"
CSV_OUT = OUTPUT_ROOT / "checks" / "stageE_ppo_batch32_generated_inp_check.csv"
JSON_OUT = OUTPUT_ROOT / "checks" / "stageE_ppo_batch32_generated_inp_check_summary.json"
COMMAND_FILE = OUTPUT_ROOT / "commands" / "stageE_ppo_batch32_abqjobpilot_commands_READY_TO_RUN.txt"
CASE_ROOT = PROJECT_ROOT / "cae_model" / BATCH
EXPECTED_COUNTS = {12: 8, 16: 8, 24: 8, 40: 8}
SOLVER_EXTS = (".odb", ".sim", ".sta", ".dat", ".msg", ".lck")
NUMERIC_RE = r"[-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?"


def load_rows() -> list[dict[str, object]]:
    rows = []
    with MANIFEST_CSV.open(newline="", encoding="utf-8-sig") as f:
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


def final_cooling_visible(lines: list[str], n: int) -> tuple[bool, str, bool]:
    last_cool_index = -1
    for i, line in enumerate(lines):
        if "step_cool_{:02d}".format(n - 1).lower() in line.lower():
            last_cool_index = max(last_cool_index, i)
    for i, line in enumerate(lines):
        if re.search(r"\*Step,\s*name=step_final_cooling\b", line, re.I):
            window = "\n".join(lines[i : i + 12])
            has_initial = re.search(r"(^|[^0-9.])0\.01([^0-9.]|$)", window) is not None
            has_duration = re.search(r"(^|[^0-9.])1200\.?([^0-9.]|$)", window) is not None
            has_max = re.search(r"(^|[^0-9.])60\.?([^0-9.]|$)", window) is not None
            return has_initial and has_duration and has_max, window.replace("\r", ""), i > last_cool_index
    return False, "", False


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


def output_fields_visible(text: str) -> tuple[bool, str]:
    upper = text.upper()
    def has_field(var: str) -> bool:
        return re.search(r"(^|[,\s]){}([,\s]|$)".format(re.escape(var)), upper) is not None

    missing = []
    for var in ("U", "PEEQ", "S"):
        if not has_field(var):
            missing.append(var)
    thermal_ok = has_field("NT11") or has_field("NT")
    if not thermal_ok:
        missing.append("NT11_or_NT")
    return not missing, "missing={}".format(missing)


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
    cae = Path(str(row["cae_path"]))
    case_dir = Path(str(row["case_dir"]))
    result: dict[str, object] = {
        "n": n,
        "strategy_name": row["strategy_name"],
        "inp_path": str(inp),
        "cae_path": str(cae),
        "inp_exists": inp.exists(),
        "cae_exists": cae.exists(),
        "inp_size_bytes": inp.stat().st_size if inp.exists() else 0,
        "cae_size_bytes": cae.stat().st_size if cae.exists() else 0,
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
        "output_fields_U_PEEQ_S_NT11_visible": False,
        "n40_cool_initialInc_0p001_verified": False,
        "solver_output_count": sum(len(list(case_dir.glob("*{}".format(ext)))) for ext in SOLVER_EXTS),
        "verdict": "FAIL",
        "notes": "",
    }
    if not inp.exists() or not cae.exists() or inp.stat().st_size <= 0 or cae.stat().st_size <= 0:
        result["notes"] = "missing or empty INP/CAE"
        return result
    text = inp.read_text(encoding="utf-8", errors="replace")
    lower = text.lower()
    lines = text.splitlines()
    node_count = section_entry_count(lines, "*Node")
    elem_count = section_entry_count(lines, "*Element")
    final_ok, final_evidence, final_after = final_cooling_visible(lines, n)
    order_ok, order_note = heat_order_matches(lines, n, list(row["scan_order"]))
    n40_ok, n40_note = n40_cool_initial_inc_visible(lines, n)
    output_ok, output_note = output_fields_visible(text)
    result.update({
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
        "output_fields_U_PEEQ_S_NT11_visible": output_ok,
        "n40_cool_initialInc_0p001_verified": n40_ok,
        "notes": "; ".join([order_note, n40_note, output_note]),
    })
    checks = [
        result["node_section_exists"], result["element_section_exists"], node_count > 0, elem_count > 0,
        result["scan_sequence_complete"], result["cool_sequence_complete"], result["step_final_cooling_exists"],
        result["final_cooling_after_last_cool"], result["final_cooling_controls_visible"], result["heat_flux_entries_exist"],
        result["all_expected_heat_sets_present"], result["heat_order_text_verified"], result["output_fields_U_PEEQ_S_NT11_visible"],
        result["solver_output_count"] == 0, result["n40_cool_initialInc_0p001_verified"],
    ]
    result["verdict"] = "PASS" if all(checks) else "FAIL"
    if not final_ok:
        result["notes"] += "; final_evidence={}".format(final_evidence[:400])
    return result


def write_command_file(rows: list[dict[str, object]]) -> dict[str, object]:
    COMMAND_FILE.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for row in rows:
        inp = Path(str(row["inp_path"]))
        line = 'enqueue --inp "{}" --cpus 14 --batch {} --strategy {}'.format(inp, BATCH, row["strategy_name"])
        lines.append(line)
    COMMAND_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    checks = {
        "command_count": len(lines),
        "all_start_enqueue_inp": all(line.startswith("enqueue --inp") for line in lines),
        "all_include_cpus_14": all("--cpus 14" in line for line in lines),
        "none_include_gpus": not any("--gpus" in line for line in lines),
        "all_inps_exist": all(Path(str(row["inp_path"])).exists() for row in rows),
        "commands_executed": False,
    }
    return checks


def main() -> int:
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    results = [check_one(row) for row in rows]
    with CSV_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    command_checks = write_command_file(rows)
    per_n_pass = {str(n): sum(1 for row in results if int(row["n"]) == n and row["verdict"] == "PASS") for n in EXPECTED_COUNTS}
    per_n_cae = {str(n): sum(1 for row in rows if int(row["n"]) == n and Path(str(row["cae_path"])).exists()) for n in EXPECTED_COUNTS}
    per_n_inp = {str(n): sum(1 for row in rows if int(row["n"]) == n and Path(str(row["inp_path"])).exists()) for n in EXPECTED_COUNTS}
    solver_count = sum(len(list(CASE_ROOT.rglob("*{}".format(ext)))) for ext in SOLVER_EXTS) if CASE_ROOT.exists() else 0
    all_pass = len(results) == 32 and all(row["verdict"] == "PASS" for row in results)
    counts_ok = per_n_cae == {str(k): v for k, v in EXPECTED_COUNTS.items()} and per_n_inp == {str(k): v for k, v in EXPECTED_COUNTS.items()}
    commands_ok = (
        command_checks["command_count"] == 32
        and command_checks["all_start_enqueue_inp"] is True
        and command_checks["all_include_cpus_14"] is True
        and command_checks["none_include_gpus"] is True
        and command_checks["all_inps_exist"] is True
        and command_checks["commands_executed"] is False
    )
    verdict = "PASS_STAGEE_PPO_BATCH32_CAE_INP_READY_FOR_USER_CONTROLLED_SOLVER" if all_pass and counts_ok and solver_count == 0 and commands_ok else "FAIL_STAGEE_PPO_BATCH32_GENERATED_INP_NOT_READY"
    summary = {
        "verdict": verdict,
        "checked_count": len(results),
        "pass_count": sum(1 for row in results if row["verdict"] == "PASS"),
        "per_n_pass_counts": per_n_pass,
        "generated_cae_count_by_N": per_n_cae,
        "generated_inp_count_by_N": per_n_inp,
        "total_cae_count": sum(per_n_cae.values()),
        "total_inp_count": sum(per_n_inp.values()),
        "solver_output_count": solver_count,
        "final_cooling_controls_verified": all(bool(row["final_cooling_controls_visible"]) for row in results),
        "output_fields_U_PEEQ_S_NT11_verified": all(bool(row["output_fields_U_PEEQ_S_NT11_visible"]) for row in results),
        "command_file": str(COMMAND_FILE),
        "command_file_checks": command_checks,
        "failed": [row for row in results if row["verdict"] != "PASS"],
        "csv_path": str(CSV_OUT),
        "no_solver": True,
        "no_datacheck": True,
        "no_ODB": True,
        "no_abqjobpilot_execution": True,
        "no_enqueue_execution": True,
        "no_teacher_validation": True,
    }
    JSON_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if verdict.startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
