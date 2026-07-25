"""Check generated Stage 3 true variable-N probe60 CAE/INP/JNL artifacts."""

from __future__ import annotations

import csv
import json
from collections import Counter
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
EXPECTED_ROWS = 60
EXPECTED_NS = (12, 16, 24, 40)


def read_manifest() -> list[dict[str, str]]:
    with MANIFEST.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def size_or_zero(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_manifest() if MANIFEST.exists() else []
    results: list[dict[str, object]] = []
    inp_counts = Counter()
    cae_counts = Counter()
    jnl_or_log_counts = Counter()

    for index, row in enumerate(rows, start=1):
        n = int(row["n"])
        expected_cae = Path(row["expected_cae"])
        expected_inp = Path(row["expected_inp"])
        expected_jnl = Path(row["expected_jnl"])
        generation_log = expected_jnl.with_suffix(".generation_log.json")
        cae_exists = expected_cae.exists() and size_or_zero(expected_cae) > 0
        inp_exists = expected_inp.exists() and size_or_zero(expected_inp) > 0
        jnl_or_log_exists = (
            (expected_jnl.exists() and size_or_zero(expected_jnl) > 0)
            or (generation_log.exists() and size_or_zero(generation_log) > 0)
        )

        if cae_exists:
            cae_counts[n] += 1
        if inp_exists:
            inp_counts[n] += 1
        if jnl_or_log_exists:
            jnl_or_log_counts[n] += 1

        row_status = "OK" if cae_exists and inp_exists and jnl_or_log_exists else "MISSING"
        results.append(
            {
                "row": index,
                "n": n,
                "strategy_name": row["strategy_name"],
                "expected_cae": str(expected_cae),
                "expected_cae_exists": cae_exists,
                "expected_cae_size": size_or_zero(expected_cae),
                "expected_inp": str(expected_inp),
                "expected_inp_exists": inp_exists,
                "expected_inp_size": size_or_zero(expected_inp),
                "expected_jnl": str(expected_jnl),
                "generation_log": str(generation_log),
                "expected_jnl_or_generation_log_exists": jnl_or_log_exists,
                "status": row_status,
            }
        )

    odb_files = sorted(str(p) for p in CASE_ROOT.rglob("*.odb")) if CASE_ROOT.exists() else []
    sim_files = sorted(str(p) for p in CASE_ROOT.rglob("*.sim")) if CASE_ROOT.exists() else []
    lock_files = sorted(str(p) for p in CASE_ROOT.rglob("*.lck")) if CASE_ROOT.exists() else []
    dat_msg_sta_files = (
        sorted(
            str(p)
            for pattern in ("*.dat", "*.msg", "*.sta")
            for p in CASE_ROOT.rglob(pattern)
        )
        if CASE_ROOT.exists()
        else []
    )

    total_inps = sum(inp_counts.values())
    total_cae = sum(cae_counts.values())
    total_jnl_or_logs = sum(jnl_or_log_counts.values())

    if total_inps == EXPECTED_ROWS and total_cae == EXPECTED_ROWS and total_jnl_or_logs == EXPECTED_ROWS and not lock_files:
        verdict = "PASS_PROBE60_60_INPS_EXIST_READY_TO_ENQUEUE"
    elif total_inps > 0:
        verdict = "WARNING_PROBE60_PARTIAL_INPS_EXIST"
    else:
        verdict = "FAIL_PROBE60_NO_INPS_FOUND"

    if odb_files or sim_files or dat_msg_sta_files or lock_files:
        if verdict.startswith("PASS"):
            verdict = "WARNING_PROBE60_PARTIAL_INPS_EXIST"

    csv_path = OUTPUT_DIR / "probe60_generated_file_check.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "row",
            "n",
            "strategy_name",
            "expected_cae",
            "expected_cae_exists",
            "expected_cae_size",
            "expected_inp",
            "expected_inp_exists",
            "expected_inp_size",
            "expected_jnl",
            "generation_log",
            "expected_jnl_or_generation_log_exists",
            "status",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    summary = {
        "verdict": verdict,
        "manifest_rows": len(rows),
        "total_expected_inp_count": EXPECTED_ROWS,
        "current_inp_count": total_inps,
        "current_cae_count": total_cae,
        "current_jnl_or_generation_log_count": total_jnl_or_logs,
        "inp_count_per_n": {str(n): inp_counts[n] for n in EXPECTED_NS},
        "cae_count_per_n": {str(n): cae_counts[n] for n in EXPECTED_NS},
        "jnl_or_generation_log_count_per_n": {
            str(n): jnl_or_log_counts[n] for n in EXPECTED_NS
        },
        "odb_files": odb_files,
        "sim_files": sim_files,
        "dat_msg_sta_files": dat_msg_sta_files,
        "lock_files": lock_files,
        "csv": str(csv_path),
    }
    summary_path = OUTPUT_DIR / "probe60_generated_file_check_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(verdict)
    print("current_inp_count={}".format(total_inps))
    print("current_cae_count={}".format(total_cae))
    print("summary={}".format(summary_path))
    return 0 if verdict.startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
