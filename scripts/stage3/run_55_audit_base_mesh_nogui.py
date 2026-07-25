"""Abaqus noGUI base mesh audit for Stage 3 run40 native_N24_N40_focused_batch60 generation."""

from __future__ import annotations

import csv
import json
import traceback
from pathlib import Path


PROJECT_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage3_run_55_calibrated_N24_N40_batch64_cae_inp_generation"
CSV_PATH = OUTPUT_DIR / "run55_base_mesh_audit.csv"
SUMMARY_PATH = OUTPUT_DIR / "run55_base_mesh_audit_summary.json"
BASES = {
    24: PROJECT_ROOT / "cae_model" / "24track_full" / "sanity_base" / "24track_sanity_base.cae",
    40: PROJECT_ROOT / "cae_model" / "40track_full" / "sanity_base" / "40track_sanity_base.cae",
}


def _safe_len(obj) -> int:  # noqa: ANN001
    try:
        return len(obj)
    except Exception:
        return 0


def _heat_set_exists(model, name: str) -> bool:  # noqa: ANN001
    root = model.rootAssembly
    if name in root.sets:
        return True
    for instance in root.instances.values():
        if name in instance.sets:
            return True
    return False


def audit_one(n: int, cae_path: Path) -> dict[str, object]:
    from abaqus import mdb, openMdb  # type: ignore

    row: dict[str, object] = {
        "n": n,
        "base_cae": str(cae_path),
        "base_exists": cae_path.exists(),
        "model_name": "",
        "part_names": "",
        "part_count": 0,
        "per_part_node_counts": "",
        "per_part_element_counts": "",
        "assembly_instance_count": 0,
        "total_node_count": 0,
        "total_element_count": 0,
        "heat_set_count": 0,
        "missing_heat_sets": "",
        "step_scan_00_exists": False,
        "step_cool_00_exists": False,
        "load_body_hflux_00_exists": False,
        "load_body_hflux_00_type": "",
        "final_cooling_template_family": "",
        "verdict": "FAIL_BASE_MESH_MISSING",
        "error": "",
    }
    if not cae_path.exists():
        row["error"] = "missing base CAE"
        return row
    try:
        openMdb(pathName=str(cae_path))
        if not mdb.models:
            row["error"] = "no models in CAE"
            return row
        model_name = "Model-1" if "Model-1" in mdb.models else sorted(mdb.models.keys())[0]
        model = mdb.models[model_name]
        row["model_name"] = model_name
        part_node_counts = {}
        part_element_counts = {}
        for part_name, part in model.parts.items():
            part_node_counts[part_name] = _safe_len(part.nodes)
            part_element_counts[part_name] = _safe_len(part.elements)
        row["part_names"] = ";".join(sorted(model.parts.keys()))
        row["part_count"] = len(model.parts)
        row["per_part_node_counts"] = json.dumps(part_node_counts, sort_keys=True)
        row["per_part_element_counts"] = json.dumps(part_element_counts, sort_keys=True)
        row["assembly_instance_count"] = len(model.rootAssembly.instances)
        row["total_node_count"] = sum(list(part_node_counts.values()))
        row["total_element_count"] = sum(list(part_element_counts.values()))
        missing = []
        for track in range(n):
            name = "set_body_heat_{:02d}".format(track)
            if _heat_set_exists(model, name):
                row["heat_set_count"] = int(row["heat_set_count"]) + 1
            else:
                missing.append(name)
        row["missing_heat_sets"] = ";".join(missing)
        row["step_scan_00_exists"] = "step_scan_00" in model.steps
        row["step_cool_00_exists"] = "step_cool_00" in model.steps
        row["load_body_hflux_00_exists"] = "load_body_hflux_00" in model.loads
        if row["load_body_hflux_00_exists"]:
            row["load_body_hflux_00_type"] = model.loads["load_body_hflux_00"].__class__.__name__
        if row["step_cool_00_exists"]:
            row["final_cooling_template_family"] = model.steps["step_cool_00"].__class__.__name__
        ok = (
            int(row["total_node_count"]) > 0
            and int(row["total_element_count"]) > 0
            and int(row["assembly_instance_count"]) > 0
            and int(row["heat_set_count"]) == n
            and bool(row["step_scan_00_exists"])
            and bool(row["step_cool_00_exists"])
            and bool(row["load_body_hflux_00_exists"])
            and row["load_body_hflux_00_type"] == "BodyHeatFlux"
        )
        row["verdict"] = "PASS_BASE_MESH_READY" if ok else "FAIL_BASE_MESH_MISSING"
    except Exception:
        row["verdict"] = "WARNING_BASE_MESH_AUDIT_PARTIAL"
        row["error"] = traceback.format_exc()
    return row


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [audit_one(n, path) for n, path in BASES.items()]
    fieldnames = [
        "n", "base_cae", "base_exists", "model_name", "part_names", "part_count",
        "per_part_node_counts", "per_part_element_counts", "assembly_instance_count",
        "total_node_count", "total_element_count", "heat_set_count", "missing_heat_sets",
        "step_scan_00_exists", "step_cool_00_exists", "load_body_hflux_00_exists",
        "load_body_hflux_00_type", "final_cooling_template_family", "verdict", "error",
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    overall = "PASS_RUN55_BASE_MESH_AUDIT_READY" if all(row["verdict"] == "PASS_BASE_MESH_READY" for row in rows) else "FAIL_RUN55_BASE_MESH_AUDIT_INVALID"
    summary = {"verdict": overall, "rows": rows, "csv_path": str(CSV_PATH), "summary_path": str(SUMMARY_PATH)}
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(overall)
    print("summary={}".format(SUMMARY_PATH))
    return 0 if overall == "PASS_RUN55_BASE_MESH_AUDIT_READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())




