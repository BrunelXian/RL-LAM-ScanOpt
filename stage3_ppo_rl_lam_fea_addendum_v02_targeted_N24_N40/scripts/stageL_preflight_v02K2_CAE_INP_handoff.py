"""Stage L preflight for PPO v02K2 targeted N24/N40 CAE/INP handoff."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
NAMESPACE = "stage3_ppo_rl_lam_fea_addendum_v02_targeted_N24_N40"
BATCH_NAME = "stage3_ppo_v02K2_targeted_N24_N40_batch32_v01"
SELECTED_CSV = PROJECT_ROOT / "outputs" / NAMESPACE / "stageK2_n40_completion" / "selected_batch32_K2" / "v02K2_ppo_targeted_N24_N40_candidate_batch32.csv"
SCAN_JSON_DIR = PROJECT_ROOT / "outputs" / NAMESPACE / "stageK2_n40_completion" / "selected_batch32_K2" / "scan_orders"
CASE_ROOT = PROJECT_ROOT / "cae_model" / BATCH_NAME
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / NAMESPACE / "stageL_CAE_INP_handoff"
CHECKS_DIR = OUTPUT_ROOT / "checks"
CSV_OUT = CHECKS_DIR / "stageL_preflight_v02K2_CAE_INP_handoff.csv"
JSON_OUT = CHECKS_DIR / "stageL_preflight_v02K2_CAE_INP_handoff_summary.json"
BASES = {
    24: PROJECT_ROOT / "cae_model" / "24track_full" / "sanity_base" / "24track_sanity_base.cae",
    40: PROJECT_ROOT / "cae_model" / "40track_full" / "sanity_base" / "40track_sanity_base.cae",
}
EXPECTED_COUNTS = {24: 16, 40: 16}
SOLVER_EXTS = (".odb", ".sim", ".sta", ".dat", ".msg", ".lck")
GENERATED_EXTS = (".cae", ".inp", ".jnl")


def parse_order(value: str) -> list[int]:
    value = str(value).strip()
    if value.startswith("["):
        return [int(x) for x in json.loads(value)]
    return [int(x) for x in value.split(",") if x != ""]


def order_legal(n: int, order: list[int]) -> bool:
    return len(order) == n and sorted(order) == list(range(n))


def bool_false(value: object) -> bool:
    return str(value).strip().lower() in ("false", "0", "no", "")


def read_rows() -> list[dict[str, str]]:
    with SELECTED_CSV.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main() -> int:
    for path in (OUTPUT_ROOT, CHECKS_DIR, OUTPUT_ROOT / "commands", OUTPUT_ROOT / "manifest", OUTPUT_ROOT / "reports", OUTPUT_ROOT / "tables"):
        path.mkdir(parents=True, exist_ok=True)

    rows_out: list[dict[str, str]] = []
    blockers: list[str] = []
    warnings: list[str] = []
    selected_rows: list[dict[str, str]] = []

    selected_exists = SELECTED_CSV.exists()
    rows_out.append({"check": "selected_K2_batch_exists", "status": "PASS" if selected_exists else "FAIL", "details": str(SELECTED_CSV)})
    if not selected_exists:
        blockers.append("selected K2 batch CSV missing")
    else:
        selected_rows = read_rows()
        counts = Counter(int(row["n"]) for row in selected_rows)
        checks = {
            "row_count_32": len(selected_rows) == 32,
            "N24_count_16": counts[24] == 16,
            "N40_count_16": counts[40] == 16,
            "no_N12_N16_N32": all(int(row["n"]) in EXPECTED_COUNTS for row in selected_rows),
            "no_duplicate_strategy_name": len({row["strategy_name"] for row in selected_rows}) == len(selected_rows),
            "teacher_validated_false": all(bool_false(row.get("teacher_validated", "")) for row in selected_rows),
            "abaqus_validated_false": all(bool_false(row.get("abaqus_validated", "")) for row in selected_rows),
            "N24_source_retained_StageK": all(row.get("stageK2_role") == "N24_retained_from_StageK_v02" for row in selected_rows if int(row["n"]) == 24),
            "N40_source_refreshed_StageK2": all(row.get("stageK2_role") == "N40_refreshed_after_K2_training" for row in selected_rows if int(row["n"]) == 40),
        }
        for name, ok in checks.items():
            rows_out.append({"check": name, "status": "PASS" if ok else "FAIL", "details": json.dumps({str(k): counts[k] for k in sorted(counts)}) if "count" in name else ""})
            if not ok:
                blockers.append(name)

        for row in selected_rows:
            strategy = row["strategy_name"]
            n = int(row["n"])
            csv_order = parse_order(row["order_json"])
            json_path = SCAN_JSON_DIR / f"scan_order_{strategy}.json"
            json_exists = json_path.exists()
            json_match = False
            legal = order_legal(n, csv_order)
            if json_exists:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
                json_order = [int(x) for x in payload.get("order", payload.get("scan_order", []))]
                json_match = json_order == csv_order
            status = "PASS" if json_exists and json_match and legal else "FAIL"
            details = f"{json_path}; json_exists={json_exists}; json_match={json_match}; legal={legal}"
            rows_out.append({"check": f"scan_order_json_matches:{strategy}", "status": status, "details": details})
            if status != "PASS":
                blockers.append(f"scan_order_json issue: {strategy}")

    for n, base in BASES.items():
        ok = base.exists()
        rows_out.append({"check": f"base_cae_exists_N{n}", "status": "PASS" if ok else "FAIL", "details": str(base)})
        if not ok:
            blockers.append(f"missing base CAE N{n}")

    solver_files: list[Path] = []
    generated_files: list[Path] = []
    lcks: list[Path] = []
    if CASE_ROOT.exists():
        for ext in SOLVER_EXTS:
            found = list(CASE_ROOT.rglob(f"*{ext}"))
            solver_files.extend(found)
            if ext == ".lck":
                lcks.extend(found)
        for ext in GENERATED_EXTS:
            generated_files.extend(CASE_ROOT.rglob(f"*{ext}"))
    rows_out.append({"check": "case_root_no_active_lck", "status": "PASS" if not lcks else "FAIL", "details": ";".join(str(p) for p in lcks[:20])})
    rows_out.append({"check": "case_root_no_solver_outputs", "status": "PASS" if not solver_files else "FAIL", "details": ";".join(str(p) for p in solver_files[:20])})
    rows_out.append({"check": "case_root_existing_generation_files", "status": "PASS" if not generated_files else "WARNING", "details": f"existing CAE/INP/JNL count={len(generated_files)}"})
    if lcks:
        blockers.append("active .lck under target case root")
    if solver_files:
        blockers.append("solver outputs under target case root")
    if generated_files:
        warnings.append("target case root contains existing CAE/INP/JNL; generator will fail closed unless removed")

    with CSV_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["check", "status", "details"])
        writer.writeheader()
        writer.writerows(rows_out)

    verdict = "FAIL_STAGEL_V02K2_CAE_INP_PREFLIGHT_BLOCKED" if blockers else ("WARNING_STAGEL_V02K2_CAE_INP_PREFLIGHT_REVIEW" if warnings else "PASS_STAGEL_V02K2_CAE_INP_PREFLIGHT_READY")
    summary = {
        "verdict": verdict,
        "selected_batch": str(SELECTED_CSV),
        "scan_json_dir": str(SCAN_JSON_DIR),
        "case_root": str(CASE_ROOT),
        "row_count": len(selected_rows),
        "counts_by_n": {str(n): Counter(int(row["n"]) for row in selected_rows)[n] for n in sorted(EXPECTED_COUNTS)},
        "csv_path": str(CSV_OUT),
        "blockers": blockers,
        "warnings": warnings,
        "no_solver": True,
        "no_datacheck": True,
        "no_enqueue": True,
        "no_ODB": True,
    }
    JSON_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(verdict)
    print(f"summary={JSON_OUT}")
    return 0 if not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
