"""Audit generated Stage Q PPO v03 CAE/INP files and write command file."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
NAMESPACE = "stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40"
BATCH_NAME = "stage3_ppo_v03_lex_primary_N24_N40_batch32_v01"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / NAMESPACE / "stageQ_CAE_INP_handoff"
MANIFEST_CSV = OUTPUT_ROOT / "manifest" / "stageQ_v03_case_manifest.csv"
CSV_OUT = OUTPUT_ROOT / "checks" / "stageQ_generated_CAE_INP_audit.csv"
JSON_OUT = OUTPUT_ROOT / "checks" / "stageQ_generated_CAE_INP_audit_summary.json"
COMMAND_FILE = OUTPUT_ROOT / "commands" / "stageQ_v03_batch32_abqjobpilot_commands_READY_TO_RUN.txt"
CASE_ROOT = PROJECT_ROOT / "cae_model" / BATCH_NAME
EXPECTED_COUNTS = {24: 16, 40: 16}
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
        match = re.match(r"\*\* STEP:\s*(\S+)", line.strip(), re.I)
        if match:
            current = match.group(1)
            blocks[current] = []
            continue
        if current is not None:
            blocks[current].append(line)
    return {key: "\n".join(value) for key, value in blocks.items()}


def final_cooling_visible(lines: list[str], n: int) -> tuple[bool, bool]:
    last_cool_index = -1
    for i, line in enumerate(lines):
        if f"step_cool_{n - 1:02d}".lower() in line.lower():
            last_cool_index = max(last_cool_index, i)
    for i, line in enumerate(lines):
        if re.search(r"\*Step,\s*name=step_final_cooling\b", line, re.I):
            window = "\n".join(lines[i : i + 12])
            has_initial = re.search(r"(^|[^0-9.])0\.01([^0-9.]|$)", window) is not None
            has_duration = re.search(r"(^|[^0-9.])1200\.?([^0-9.]|$)", window) is not None
            has_max = re.search(r"(^|[^0-9.])60\.?([^0-9.]|$)", window) is not None
            return has_initial and has_duration and has_max, i > last_cool_index
    return False, False


def heat_order_matches(lines: list[str], n: int, order: list[int]) -> tuple[bool, str]:
    blocks = step_blocks(lines)
    observed = []
    for seq in range(n):
        block = blocks.get(f"step_scan_{seq:02d}", "")
        match = re.search(r"set_body_heat_(\d+)\s*,\s*BF\s*,", block, re.I)
        if not match:
            return False, f"missing BF set in step_scan_{seq:02d}"
        observed.append(int(match.group(1)))
    if observed != order:
        return False, f"observed {observed} != expected {order}"
    final_block = blocks.get("step_final_cooling", "")
    if re.search(r"\*Dflux", final_block, re.I):
        return False, "Dflux appears in final cooling"
    return True, "observed order matches scan_order"


def output_fields_visible(text: str) -> tuple[bool, str]:
    upper = text.upper()
    def has_field(var: str) -> bool:
        return re.search(rf"(^|[,\s]){re.escape(var)}([,\s]|$)", upper) is not None

    missing = [var for var in ("U", "PEEQ", "S") if not has_field(var)]
    if not (has_field("NT11") or has_field("NT")):
        missing.append("NT11_or_NT")
    return not missing, f"missing={missing}"


def n40_cool_initial_inc_visible(lines: list[str], n: int) -> tuple[bool, str]:
    if n != 40:
        return True, "not N40"
    seen = 0
    bad = []
    for i, line in enumerate(lines):
        match = re.match(r"\*Step,\s*name=(step_cool_\d+)", line, re.I)
        if not match:
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
            bad.append(f"{match.group(1)}:{data_line}")
    if seen != 40:
        return False, f"N40 cool step count {seen} != 40"
    if bad:
        return False, "; ".join(bad[:3])
    return True, "all N40 cool steps show initialInc=0.001 and timePeriod=3.4"


def check_one(row: dict[str, object]) -> dict[str, object]:
    n = int(row["n"])
    inp = Path(str(row["expected_inp"]))
    cae = Path(str(row["expected_cae"]))
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
        "node_entry_count": 0,
        "element_entry_count": 0,
        "scan_sequence_complete": False,
        "cool_sequence_complete": False,
        "step_final_cooling_exists": False,
        "final_cooling_controls_visible": False,
        "final_cooling_after_last_cool": False,
        "heat_flux_entries_exist": False,
        "all_expected_heat_sets_present": False,
        "heat_order_text_verified": False,
        "output_fields_U_PEEQ_S_NT11_visible": False,
        "n40_cool_initialInc_0p001_verified": False,
        "solver_output_count": sum(len(list(case_dir.glob(f"*{ext}"))) for ext in SOLVER_EXTS),
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
    final_ok, final_after = final_cooling_visible(lines, n)
    heat_ok, heat_note = heat_order_matches(lines, n, list(row["scan_order"]))
    output_ok, output_note = output_fields_visible(text)
    n40_ok, n40_note = n40_cool_initial_inc_visible(lines, n)
    result.update({
        "node_entry_count": node_count,
        "element_entry_count": elem_count,
        "scan_sequence_complete": all(f"step_scan_{seq:02d}".lower() in lower for seq in range(n)),
        "cool_sequence_complete": all(f"step_cool_{seq:02d}".lower() in lower for seq in range(n)),
        "step_final_cooling_exists": "step_final_cooling" in lower,
        "final_cooling_controls_visible": final_ok,
        "final_cooling_after_last_cool": final_after,
        "heat_flux_entries_exist": "body heat flux" in lower or "*dflux" in lower,
        "all_expected_heat_sets_present": all(f"set_body_heat_{track:02d}".lower() in lower for track in range(n)),
        "heat_order_text_verified": heat_ok,
        "output_fields_U_PEEQ_S_NT11_visible": output_ok,
        "n40_cool_initialInc_0p001_verified": n40_ok,
        "notes": "; ".join([heat_note, output_note, n40_note]),
    })
    checks = [
        node_count > 0,
        elem_count > 0,
        result["scan_sequence_complete"],
        result["cool_sequence_complete"],
        result["step_final_cooling_exists"],
        result["final_cooling_controls_visible"],
        result["final_cooling_after_last_cool"],
        result["heat_flux_entries_exist"],
        result["all_expected_heat_sets_present"],
        result["heat_order_text_verified"],
        result["output_fields_U_PEEQ_S_NT11_visible"],
        result["n40_cool_initialInc_0p001_verified"],
        result["solver_output_count"] == 0,
    ]
    result["verdict"] = "PASS" if all(checks) else "FAIL"
    return result


def write_command_file(rows: list[dict[str, object]]) -> dict[str, object]:
    COMMAND_FILE.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f'enqueue --inp "{row["expected_inp"]}" --cpus 14 --batch {BATCH_NAME} --strategy {row["strategy_name"]}'
        for row in rows
    ]
    COMMAND_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    counts = Counter(int(row["n"]) for row in rows)
    errors = []
    if len(lines) != 32:
        errors.append(f"command count {len(lines)} != 32")
    if {n: counts[n] for n in EXPECTED_COUNTS} != EXPECTED_COUNTS:
        errors.append(f"command counts {dict(counts)} != {EXPECTED_COUNTS}")
    if any("--gpus" in line for line in lines):
        errors.append("command file contains --gpus")
    if any("--cpus 14" not in line for line in lines):
        errors.append("one or more commands missing --cpus 14")
    return {"path": str(COMMAND_FILE), "command_count": len(lines), "per_n_counts": {str(k): counts[k] for k in sorted(EXPECTED_COUNTS)}, "errors": errors}


def main() -> int:
    rows = load_rows()
    results = [check_one(row) for row in rows]
    with CSV_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    counts = Counter(int(row["n"]) for row in results if row["verdict"] == "PASS")
    cae_counts = Counter(int(row["n"]) for row in rows if Path(str(row["expected_cae"])).exists())
    inp_counts = Counter(int(row["n"]) for row in rows if Path(str(row["expected_inp"])).exists())
    forbidden_dirs = [p for p in CASE_ROOT.iterdir() if p.is_dir() and re.match(r"^N(12|16|32)", p.name)]
    solver_outputs = [p for ext in SOLVER_EXTS for p in CASE_ROOT.rglob(f"*{ext}")] if CASE_ROOT.exists() else []
    command_status = write_command_file(rows)
    failed = [row for row in results if row["verdict"] != "PASS"]
    ok = (
        len(results) == 32
        and not failed
        and {n: cae_counts[n] for n in EXPECTED_COUNTS} == EXPECTED_COUNTS
        and {n: inp_counts[n] for n in EXPECTED_COUNTS} == EXPECTED_COUNTS
        and not forbidden_dirs
        and not solver_outputs
        and not command_status["errors"]
    )
    verdict = "PASS_STAGEQ_V03_CAE_INP_READY_FOR_USER_CONTROLLED_SOLVER" if ok else "FAIL_STAGEQ_V03_CAE_INP_GENERATION_FAILED"
    summary = {
        "verdict": verdict,
        "checked_count": len(results),
        "pass_count": sum(1 for row in results if row["verdict"] == "PASS"),
        "cae_counts_by_n": {str(k): cae_counts[k] for k in sorted(EXPECTED_COUNTS)},
        "inp_counts_by_n": {str(k): inp_counts[k] for k in sorted(EXPECTED_COUNTS)},
        "solver_output_count": len(solver_outputs),
        "forbidden_case_dirs": [str(p) for p in forbidden_dirs],
        "failed": failed,
        "command_file": command_status,
        "csv_path": str(CSV_OUT),
    }
    JSON_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(verdict)
    print(f"summary={JSON_OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())


