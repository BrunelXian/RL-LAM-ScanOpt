"""Text check for the Stage 3 probe60 pilot INP heat-order sequence."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage3_cae_probe60_generation"
PILOT_INP = (
    PROJECT_ROOT
    / "cae_model"
    / "stage3_true_variable_N_probe60_v01"
    / "N12"
    / "N12_A01_raster_left_to_right"
    / "J2D_N12_N12_A01_raster_left_to_right.inp"
)
PILOT_SCAN_ORDER_JSON = (
    PROJECT_ROOT
    / "cae_model"
    / "stage3_true_variable_N_probe60_v01"
    / "N12"
    / "N12_A01_raster_left_to_right"
    / "scan_order_N12_A01_raster_left_to_right.json"
)


def read_scan_order() -> list[int]:
    data = json.loads(PILOT_SCAN_ORDER_JSON.read_text(encoding="utf-8"))
    return list(data["scan_order"])


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    warnings: list[str] = []
    failures: list[str] = []

    if not PILOT_INP.exists():
        failures.append("pilot INP does not exist")
        text = ""
    else:
        text = PILOT_INP.read_text(errors="replace")
        if PILOT_INP.stat().st_size <= 0:
            failures.append("pilot INP exists but has zero size")

    scan_order = read_scan_order() if PILOT_SCAN_ORDER_JSON.exists() else []
    n = len(scan_order) if scan_order else 12

    expected_scan_steps = ["step_scan_{:02d}".format(i) for i in range(n)]
    expected_cool_steps = ["step_cool_{:02d}".format(i) for i in range(n)]
    expected_sets = ["set_body_heat_{:02d}".format(i) for i in range(n)]

    checks = {
        "file_exists": PILOT_INP.exists(),
        "nonzero_size": PILOT_INP.exists() and PILOT_INP.stat().st_size > 0,
        "contains_step_final_cooling": "step_final_cooling" in text,
        "contains_heat_load_keywords": any(
            token.lower() in text.lower()
            for token in ("*Dflux", "*Cflux", "*Dsflux", "BodyHeatFlux", "Heat Flux")
        ),
        "contains_all_expected_scan_step_names": all(name in text for name in expected_scan_steps),
        "contains_all_expected_cool_step_names": all(name in text for name in expected_cool_steps),
        "contains_all_expected_heat_set_names": all(name in text for name in expected_sets),
    }

    for name, passed in checks.items():
        rows.append({"check": name, "status": "PASS" if passed else "FAIL", "details": ""})
        if not passed:
            if name in ("file_exists", "nonzero_size"):
                failures.append(name)
            else:
                warnings.append(name)

    if text:
        final_pos = text.find("step_final_cooling")
        last_scan_pos = max((text.find(name) for name in expected_scan_steps if name in text), default=-1)
        last_cool_pos = max((text.find(name) for name in expected_cool_steps if name in text), default=-1)
        final_after_sequence = final_pos > max(last_scan_pos, last_cool_pos)
        rows.append(
            {
                "check": "step_final_cooling_after_last_scan_or_cool_name",
                "status": "PASS" if final_after_sequence else "WARN",
                "details": "final_pos={}, last_scan_pos={}, last_cool_pos={}".format(
                    final_pos, last_scan_pos, last_cool_pos
                ),
            }
        )
        if not final_after_sequence:
            warnings.append("step_final_cooling order could not be verified")

        found_set_order = []
        for match in re.finditer(r"set_body_heat_(\d+)", text):
            value = int(match.group(1))
            if value not in found_set_order:
                found_set_order.append(value)
        comparable_order = found_set_order[:n]
        order_matches = bool(scan_order) and comparable_order == scan_order
        rows.append(
            {
                "check": "first_unique_heat_set_order_matches_scan_order",
                "status": "PASS" if order_matches else "WARN",
                "details": "expected={}, found_first_unique={}".format(
                    scan_order, comparable_order
                ),
            }
        )
        if not order_matches:
            warnings.append("INP text order does not preserve enough names to verify heat order")

    csv_path = OUTPUT_DIR / "pilot_inp_heat_order_check.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["check", "status", "details"])
        writer.writeheader()
        writer.writerows(rows)

    if failures:
        verdict = "FAIL_PILOT_INP_NOT_READY"
    elif warnings:
        verdict = "WARNING_PILOT_INP_ORDER_NEEDS_MANUAL_INP_REVIEW"
    else:
        verdict = "PASS_PILOT_INP_HEAT_ORDER_TEXT_CHECK"

    summary = {
        "verdict": verdict,
        "pilot_inp": str(PILOT_INP),
        "pilot_scan_order_json": str(PILOT_SCAN_ORDER_JSON),
        "n": n,
        "file_exists": PILOT_INP.exists(),
        "file_size": PILOT_INP.stat().st_size if PILOT_INP.exists() else 0,
        "warnings": warnings,
        "failures": failures,
        "csv": str(csv_path),
    }
    summary_path = OUTPUT_DIR / "pilot_inp_heat_order_check_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(verdict)
    print("pilot_inp={}".format(PILOT_INP))
    print("summary={}".format(summary_path))
    return 0 if verdict.startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
