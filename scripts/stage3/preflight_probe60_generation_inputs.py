"""Preflight checks for Stage 3 true variable-N probe60 CAE generation.

This script intentionally does not open Abaqus CAE files and does not generate
CAE, INP, or JNL artifacts.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
PYTHON_EXE = Path(r"D:\XianLab\envs\conda\torch-gpu\python.exe")
CASE_ROOT = PROJECT_ROOT / "cae_model" / "stage3_true_variable_N_probe60_v01"
MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "stage3_manual_probe60_handoff"
    / "variable_N_probe60_case_manifest_FIXED.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage3_cae_probe60_generation"

BASES = {
    12: {
        "cae": PROJECT_ROOT
        / "cae_model"
        / "12track_full"
        / "sanity_base"
        / "12track_sanity_base.cae",
        "jnl": PROJECT_ROOT
        / "cae_model"
        / "12track_full"
        / "sanity_base"
        / "12track_sanity_base.jnl",
    },
    16: {
        "cae": PROJECT_ROOT
        / "cae_model"
        / "16track_full"
        / "sanity_base"
        / "16track_sanity_base.cae",
        "jnl": PROJECT_ROOT
        / "cae_model"
        / "16track_full"
        / "sanity_base"
        / "16track_sanity_base.jnl",
    },
    24: {
        "cae": PROJECT_ROOT
        / "cae_model"
        / "24track_full"
        / "sanity_base"
        / "24track_sanity_base.cae",
        "jnl": PROJECT_ROOT
        / "cae_model"
        / "24track_full"
        / "sanity_base"
        / "24track_sanity_base.jnl",
    },
    40: {
        "cae": PROJECT_ROOT
        / "cae_model"
        / "40track_full"
        / "sanity_base"
        / "40track_sanity_base.cae",
        "jnl": PROJECT_ROOT
        / "cae_model"
        / "40track_full"
        / "sanity_base"
        / "40track_sanity_base.jnl",
    },
}

EXPECTED_NS = (12, 16, 24, 40)
EXPECTED_ROWS = 60
EXPECTED_PER_N = 15
BAD_SCHEMA_TOKENS = ("N12N12_", "N16N16_", "N24N24_", "N40N40_")


def is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def read_manifest() -> list[dict[str, str]]:
    with MANIFEST.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_scan_order(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def validate_scan_order(scan_order: object, n: int) -> str | None:
    if not isinstance(scan_order, list):
        return "scan_order is not a list"
    if len(scan_order) != n:
        return "scan_order length {} != N {}".format(len(scan_order), n)
    if not all(isinstance(v, int) for v in scan_order):
        return "scan_order contains non-integer entries"
    if sorted(scan_order) != list(range(n)):
        return "scan_order is not a permutation of 0..{}".format(n - 1)
    return None


def check_expected_schema(row: dict[str, str], n: int) -> list[str]:
    errors: list[str] = []
    n_token = "N{}".format(n)
    case_dir = Path(row["case_dir"])
    expected_paths = {
        "expected_cae": Path(row["expected_cae"]),
        "expected_inp": Path(row["expected_inp"]),
        "expected_jnl": Path(row["expected_jnl"]),
        "scan_order_json": Path(row["scan_order_json"]),
    }

    case_dir_text = str(case_dir)
    expected_fragment = "\\{}\\{}_".format(n_token, n_token)
    if expected_fragment not in case_dir_text:
        errors.append("case_dir missing schema fragment {}".format(expected_fragment))
    for token in BAD_SCHEMA_TOKENS:
        if token in case_dir_text or any(token in str(p) for p in expected_paths.values()):
            errors.append("bad concatenated path schema token found: {}".format(token))

    if case_dir.parent != CASE_ROOT / n_token:
        errors.append("case_dir is not directly under {}".format(CASE_ROOT / n_token))
    if not case_dir.name.startswith(n_token + "_"):
        errors.append("case_dir name does not start with {}_".format(n_token))

    for key, path in expected_paths.items():
        if path.parent != case_dir:
            errors.append("{} is not inside case_dir".format(key))
        if not is_relative_to(path, CASE_ROOT / n_token):
            errors.append("{} is not under correct N folder".format(key))

    job_name = row["job_name"]
    for key in ("expected_cae", "expected_inp", "expected_jnl"):
        if Path(row[key]).stem != job_name:
            errors.append("{} stem does not match job_name".format(key))

    return errors


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    warnings: list[str] = []
    rows_out: list[dict[str, object]] = []

    for n, paths in BASES.items():
        for kind, path in paths.items():
            if not path.exists():
                errors.append("missing base {} for N{}: {}".format(kind.upper(), n, path))

    if not MANIFEST.exists():
        errors.append("missing manifest: {}".format(MANIFEST))
        rows: list[dict[str, str]] = []
    else:
        rows = read_manifest()

    if len(rows) != EXPECTED_ROWS:
        errors.append("manifest row count {} != {}".format(len(rows), EXPECTED_ROWS))

    n_counts = Counter()
    scan_order_count = 0
    for index, row in enumerate(rows, start=1):
        row_errors: list[str] = []
        row_warnings: list[str] = []
        try:
            n = int(row.get("n", ""))
        except ValueError:
            n = -1
            row_errors.append("invalid n value")

        if n not in EXPECTED_NS:
            row_errors.append("unexpected N {}".format(row.get("n")))
        else:
            n_counts[n] += 1
            row_errors.extend(check_expected_schema(row, n))

        case_dir = Path(row.get("case_dir", ""))
        scan_path = Path(row.get("scan_order_json", ""))
        if not case_dir.exists():
            row_errors.append("case_dir missing")
        if not scan_path.exists():
            row_errors.append("scan_order_json missing")
        else:
            scan_order_count += 1
            try:
                scan_data = load_scan_order(scan_path)
                scan_error = validate_scan_order(scan_data.get("scan_order"), n)
                if scan_error:
                    row_errors.append(scan_error)
                if int(scan_data.get("n", -1)) != n:
                    row_errors.append("scan_order_json n mismatch")
                if scan_data.get("strategy_name") != row.get("strategy_name"):
                    row_errors.append("strategy_name mismatch in scan_order_json")
                for key in ("expected_cae", "expected_inp", "expected_jnl"):
                    if str(Path(scan_data.get(key, ""))) != str(Path(row[key])):
                        row_errors.append("{} mismatch in scan_order_json".format(key))
            except Exception as exc:  # noqa: BLE001 - preflight should report all input issues.
                row_errors.append("failed to read scan_order_json: {}".format(exc))

        rows_out.append(
            {
                "row": index,
                "n": row.get("n", ""),
                "strategy_name": row.get("strategy_name", ""),
                "case_dir": row.get("case_dir", ""),
                "scan_order_json": row.get("scan_order_json", ""),
                "expected_cae": row.get("expected_cae", ""),
                "expected_inp": row.get("expected_inp", ""),
                "expected_jnl": row.get("expected_jnl", ""),
                "status": "OK" if not row_errors else "INVALID",
                "errors": "; ".join(row_errors),
                "warnings": "; ".join(row_warnings),
            }
        )
        errors.extend("row {}: {}".format(index, msg) for msg in row_errors)
        warnings.extend("row {}: {}".format(index, msg) for msg in row_warnings)

    for n in EXPECTED_NS:
        if n_counts[n] != EXPECTED_PER_N:
            errors.append("N{} manifest count {} != {}".format(n, n_counts[n], EXPECTED_PER_N))

    current_inp_count = len(list(CASE_ROOT.rglob("*.inp"))) if CASE_ROOT.exists() else 0
    if current_inp_count not in (0, 1, EXPECTED_ROWS):
        warnings.append("current INP count is partial: {}".format(current_inp_count))

    if errors:
        verdict = "FAIL_PROBE60_GENERATION_PREFLIGHT_INVALID"
    elif warnings:
        verdict = "WARNING_PROBE60_GENERATION_PREFLIGHT_PARTIAL"
    else:
        verdict = "PASS_PROBE60_GENERATION_PREFLIGHT_READY"

    csv_path = OUTPUT_DIR / "probe60_generation_preflight.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "row",
            "n",
            "strategy_name",
            "case_dir",
            "scan_order_json",
            "expected_cae",
            "expected_inp",
            "expected_jnl",
            "status",
            "errors",
            "warnings",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    summary = {
        "verdict": verdict,
        "manifest": str(MANIFEST),
        "manifest_rows": len(rows),
        "scan_order_files_found": scan_order_count,
        "n_counts": {str(k): n_counts[k] for k in EXPECTED_NS},
        "current_inp_count_before_generation": current_inp_count,
        "base_cae_files_exist": {
            str(n): BASES[n]["cae"].exists() for n in EXPECTED_NS
        },
        "base_jnl_files_exist": {
            str(n): BASES[n]["jnl"].exists() for n in EXPECTED_NS
        },
        "errors": errors,
        "warnings": warnings,
        "outputs": {"csv": str(csv_path)},
    }
    summary_path = OUTPUT_DIR / "probe60_generation_preflight_summary.json"
    summary["outputs"]["summary_json"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(verdict)
    print("manifest_rows={}".format(len(rows)))
    print("scan_order_files_found={}".format(scan_order_count))
    print("current_inp_count_before_generation={}".format(current_inp_count))
    print("summary={}".format(summary_path))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
