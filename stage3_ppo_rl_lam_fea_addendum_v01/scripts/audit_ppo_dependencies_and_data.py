"""Audit PPO addendum dependencies and teacher-labelled dataset readiness.

This script performs only ordinary Python inspection. It does not run Abaqus,
open ODB files, generate solver inputs, train PPO, train a surrogate, or
generate candidate scan orders.
"""

from __future__ import annotations

import csv
import importlib
import json
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
NAMESPACE = "stage3_ppo_rl_lam_fea_addendum_v01"
FREEZE_DIR = PROJECT_ROOT / "outputs" / "stage3_run_78_final_evidence_freeze_package"
NATIVE_DATASET = FREEZE_DIR / "FROZEN_stage3_native_combined552_teacher_dataset.csv"
PLUS_N32_DATASET = FREEZE_DIR / "FROZEN_stage3_combined552_plus_N32_RL_ready_dataset.csv"
PREFLIGHT_DIR = PROJECT_ROOT / "outputs" / NAMESPACE / "preflight"
JSON_OUT = PREFLIGHT_DIR / "ppo_dependency_and_data_audit.json"
MD_OUT = PREFLIGHT_DIR / "ppo_dependency_and_data_audit.md"

SUPPORTED_NATIVE_N = [12, 16, 24, 40]
REQUIRED_COLUMN_GROUPS = {
    "n": ["n"],
    "strategy_identifier": ["strategy_name", "candidate_id"],
    "scan_order": ["order_json", "order_compact", "scan_order"],
    "u2_range": ["u2_range"],
    "peeq_max": ["peeq_max"],
    "surface_t_proxy": ["surface_t_proxy"],
    "mises_max": ["mises_max"],
    "teacher_validation_status": ["teacher_validation_status"],
}


def import_status(module_name: str, symbol: str | None = None) -> dict[str, Any]:
    status: dict[str, Any] = {
        "module": module_name,
        "available": False,
        "version": None,
        "error": None,
    }
    try:
        module = importlib.import_module(module_name)
        status["available"] = True
        status["version"] = getattr(module, "__version__", None)
        if symbol is not None:
            getattr(module, symbol)
            status["symbol"] = symbol
            status["symbol_available"] = True
    except Exception as exc:  # noqa: BLE001 - audit should record exact import failure.
        status["error"] = f"{type(exc).__name__}: {exc}"
        if symbol is not None:
            status["symbol"] = symbol
            status["symbol_available"] = False
    return status


def maskable_ppo_status() -> dict[str, Any]:
    status: dict[str, Any] = {
        "module": "sb3_contrib",
        "symbol": "MaskablePPO",
        "available": False,
        "version": None,
        "error": None,
    }
    try:
        module = importlib.import_module("sb3_contrib")
        status["version"] = getattr(module, "__version__", None)
        from sb3_contrib import MaskablePPO  # noqa: F401

        status["available"] = True
    except Exception as exc:  # noqa: BLE001 - audit should record exact import failure.
        status["error"] = f"{type(exc).__name__}: {exc}"
    return status


def read_csv_summary(path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "row_count": 0,
        "columns": [],
        "required_column_groups": {},
        "row_counts_by_n": {},
        "present_n_values": [],
    }
    if not path.exists():
        return summary

    n_counts: Counter[int | str] = Counter()
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        summary["columns"] = columns
        for row in reader:
            summary["row_count"] += 1
            raw_n = row.get("n", "")
            try:
                n_value: int | str = int(float(raw_n))
            except (TypeError, ValueError):
                n_value = raw_n
            n_counts[n_value] += 1

    present_n_values = sorted(n_counts, key=lambda value: (isinstance(value, str), value))
    summary["present_n_values"] = present_n_values
    summary["row_counts_by_n"] = {str(key): value for key, value in n_counts.items()}
    for group_name, alternatives in REQUIRED_COLUMN_GROUPS.items():
        found = [column for column in alternatives if column in summary["columns"]]
        summary["required_column_groups"][group_name] = {
            "alternatives": alternatives,
            "found": found,
            "available_or_mappable": bool(found),
        }
    return summary


def verdict_for(audit: dict[str, Any]) -> str:
    native = audit["datasets"]["native_combined552"]
    plus = audit["datasets"]["plus_n32_auxiliary"]
    required_groups_ok = all(
        group["available_or_mappable"]
        for group in native["required_column_groups"].values()
    )
    native_n_ok = all(n in native["present_n_values"] for n in SUPPORTED_NATIVE_N)
    core_data_ok = native["exists"] and native["row_count"] > 0 and required_groups_ok and native_n_ok
    plus_n32_ok = plus["exists"] and 32 in plus["present_n_values"] and 32 not in native["present_n_values"]

    dependency_keys = [
        "torch",
        "numpy",
        "pandas",
        "gymnasium",
        "stable_baselines3",
        "sb3_contrib",
        "maskable_ppo",
    ]
    dependency_ok = all(audit["dependencies"][key]["available"] for key in dependency_keys)

    if core_data_ok and plus_n32_ok and dependency_ok:
        return "PASS_PPO_ADDENDUM_DEPENDENCIES_AND_DATA_READY"
    if core_data_ok:
        return "WARNING_PPO_ADDENDUM_DEPENDENCIES_PARTIAL"
    return "FAIL_PPO_ADDENDUM_PREFLIGHT_NOT_READY"


def write_markdown(audit: dict[str, Any]) -> None:
    deps = audit["dependencies"]
    native = audit["datasets"]["native_combined552"]
    plus = audit["datasets"]["plus_n32_auxiliary"]
    missing = audit["missing_or_partial"]

    lines = [
        "# PPO Dependency And Data Audit",
        "",
        f"- Timestamp UTC: `{audit['timestamp_utc']}`",
        f"- Python executable: `{audit['python']['executable']}`",
        f"- Python version: `{audit['python']['version']}`",
        f"- Platform: `{audit['python']['platform']}`",
        f"- Verdict: `{audit['verdict']}`",
        "",
        "## Dependencies",
        "",
        "| Dependency | Available | Version | Error |",
        "| --- | ---: | --- | --- |",
    ]
    for key in ["torch", "numpy", "pandas", "gymnasium", "stable_baselines3", "sb3_contrib", "maskable_ppo"]:
        item = deps[key]
        lines.append(
            f"| {key} | {item['available']} | {item.get('version')} | {item.get('error') or ''} |"
        )

    lines.extend(
        [
            "",
            "## Native Combined552 Dataset",
            "",
            f"- Path: `{native['path']}`",
            f"- Exists: `{native['exists']}`",
            f"- Rows: `{native['row_count']}`",
            f"- N values: `{native['present_n_values']}`",
            f"- Row counts by N: `{native['row_counts_by_n']}`",
            "",
            "## Plus-N32 Auxiliary Dataset",
            "",
            f"- Path: `{plus['path']}`",
            f"- Exists: `{plus['exists']}`",
            f"- Rows: `{plus['row_count']}`",
            f"- N values: `{plus['present_n_values']}`",
            f"- Row counts by N: `{plus['row_counts_by_n']}`",
            "",
            "## Required Column Groups",
            "",
            "| Group | Alternatives | Found | Available or Mappable |",
            "| --- | --- | --- | ---: |",
        ]
    )
    for group_name, group in native["required_column_groups"].items():
        lines.append(
            f"| {group_name} | `{group['alternatives']}` | `{group['found']}` | {group['available_or_mappable']} |"
        )

    lines.extend(
        [
            "",
            "## Readiness Checks",
            "",
            f"- Native N12/N16/N24/N40 present: `{audit['checks']['native_supported_n_present']}`",
            f"- N32 present in native dataset: `{audit['checks']['n32_present_in_native']}`",
            f"- N32 present in plus-N32 auxiliary dataset: `{audit['checks']['n32_present_in_plus_n32']}`",
            f"- MaskablePPO importable: `{deps['maskable_ppo']['available']}`",
            "",
            "## Missing Or Partial",
            "",
        ]
    )
    if missing:
        for item in missing:
            lines.append(f"- {item}")
    else:
        lines.append("- None.")

    MD_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    PREFLIGHT_DIR.mkdir(parents=True, exist_ok=True)
    audit: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "version_info": list(sys.version_info[:5]),
            "platform": platform.platform(),
        },
        "dependencies": {
            "torch": import_status("torch"),
            "numpy": import_status("numpy"),
            "pandas": import_status("pandas"),
            "gymnasium": import_status("gymnasium"),
            "stable_baselines3": import_status("stable_baselines3"),
            "sb3_contrib": import_status("sb3_contrib"),
            "maskable_ppo": maskable_ppo_status(),
        },
        "datasets": {
            "native_combined552": read_csv_summary(NATIVE_DATASET),
            "plus_n32_auxiliary": read_csv_summary(PLUS_N32_DATASET),
        },
        "scope_guards": {
            "no_Abaqus": True,
            "no_ODB": True,
            "no_solver": True,
            "no_CAE_INP_JNL": True,
            "no_PPO_training": True,
            "no_candidate_generation": True,
        },
    }

    native_n = audit["datasets"]["native_combined552"]["present_n_values"]
    plus_n = audit["datasets"]["plus_n32_auxiliary"]["present_n_values"]
    audit["checks"] = {
        "native_supported_n_present": all(n in native_n for n in SUPPORTED_NATIVE_N),
        "n32_present_in_native": 32 in native_n,
        "n32_present_in_plus_n32": 32 in plus_n,
    }

    missing: list[str] = []
    for dep_name, dep in audit["dependencies"].items():
        if not dep["available"]:
            missing.append(f"Missing dependency: {dep_name} ({dep.get('error')})")
    native = audit["datasets"]["native_combined552"]
    if not native["exists"]:
        missing.append(f"Missing native dataset: {native['path']}")
    for group_name, group in native["required_column_groups"].items():
        if not group["available_or_mappable"]:
            missing.append(f"Missing native dataset column group: {group_name} alternatives={group['alternatives']}")
    if not audit["checks"]["native_supported_n_present"]:
        missing.append(f"Native dataset does not contain all supported N values: {SUPPORTED_NATIVE_N}")
    if not audit["checks"]["n32_present_in_plus_n32"]:
        missing.append("Plus-N32 auxiliary dataset does not contain N32.")
    if audit["checks"]["n32_present_in_native"]:
        missing.append("Native dataset contains N32; expected N32 to remain auxiliary only.")
    audit["missing_or_partial"] = missing
    audit["verdict"] = verdict_for(audit)

    JSON_OUT.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    write_markdown(audit)
    print(json.dumps({"verdict": audit["verdict"], "json": str(JSON_OUT), "markdown": str(MD_OUT)}, indent=2))
    return 0 if not audit["verdict"].startswith("FAIL") else 1


if __name__ == "__main__":
    raise SystemExit(main())
