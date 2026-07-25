from __future__ import annotations

import csv
import json
import math
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from odbAccess import openOdb


PROJECT_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
SOLVER_AUDIT_JSON = (
    PROJECT_ROOT
    / "outputs"
    / "stage3_cae_probe60_generation"
    / "probe60_solver_completion_audit_after_N24_A07_rerun.json"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage3_run_08_probe60_odb_teacher_validation"
REPORT_DIR = PROJECT_ROOT / "docs" / "stage3" / "runs" / "run_08_variable_n_probe60_odb_teacher_validation"
REPORT_PATH = REPORT_DIR / "RUN_08_VARIABLE_N_PROBE60_ODB_TEACHER_VALIDATION_REPORT.md"
MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "manifests" / "stage3_run_08_manifest.json"

LABELS_CSV = OUTPUT_DIR / "probe60_odb_teacher_labels.csv"
LABELS_JSON = OUTPUT_DIR / "probe60_odb_teacher_labels.json"
SUMMARY_JSON = OUTPUT_DIR / "probe60_odb_teacher_validation_summary.json"

PEEQ_THRESHOLD = 0.002
FINAL_STEP_NAME = "step_final_cooling"

STRATEGY_FAMILY_BY_CODE = {
    "A01": "raster",
    "A02": "odd_even",
    "A03": "maximin",
    "A04": "method_c",
    "A05": "center_out",
    "A06": "edge_in",
    "A07": "regular_jump",
    "A08": "block_interleaved",
    "A09": "center_edge",
    "A10": "graph_pointer_proxy",
    "A11": "graph_pointer_proxy",
    "A12": "graph_pointer_proxy",
    "A13": "graph_pointer_proxy",
    "A14": "graph_pointer_proxy",
    "A15": "graph_pointer_proxy",
}

FIELDNAMES = [
    "case_id",
    "N",
    "strategy_id",
    "strategy_code",
    "strategy_family",
    "job_name",
    "case_dir",
    "odb_path",
    "odb_size_bytes",
    "scan_order_json",
    "scan_order",
    "odb_extraction_status",
    "notes",
    "final_step_name",
    "final_frame_value",
    "final_frame_description",
    "u_node_count",
    "u2_min",
    "u2_max",
    "u2_range",
    "u2_abs_max",
    "u2_mean",
    "u2_mean_abs",
    "u2_rms",
    "u_magnitude_max",
    "peeq_value_count",
    "peeq_max",
    "peeq_mean",
    "peeq_fraction_gt_0p002",
    "peeq_count_gt_0p002",
    "s_value_count",
    "mises_max",
    "mises_mean",
    "max_principal_stress_max",
    "max_principal_stress_mean",
    "surface_t_proxy_max_tensile_pa",
    "surface_t_proxy_max_tensile_mpa",
    "nt11_node_count",
    "nt11_min",
    "nt11_max",
    "nt11_mean",
    "rank_within_N_u2_range",
    "rank_within_N_peeq_max",
    "rank_within_N_surface_t_proxy",
    "rank_within_N_mises_max",
]


def load_solver_rows() -> list[dict[str, object]]:
    with SOLVER_AUDIT_JSON.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("rows", [])
    if len(rows) != 60:
        raise RuntimeError(f"Expected 60 solver audit rows, got {len(rows)}")
    bad = [
        row
        for row in rows
        if row.get("final_completion_status")
        not in {"PASS_SOLVER_COMPLETE", "WARNING_SUCCESS_WITH_WARNINGS"}
    ]
    if bad:
        bad_ids = ", ".join(str(row.get("case_id")) for row in bad)
        raise RuntimeError(f"Solver completion gate is not clean: {bad_ids}")
    return rows


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def rms(values: list[float]) -> float | None:
    return math.sqrt(sum(value * value for value in values) / len(values)) if values else None


def safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def scalar_values(field) -> list[float]:
    out: list[float] = []
    for value in field.values:
        data = value.data
        if hasattr(data, "__iter__") and not isinstance(data, str):
            if data:
                out.append(float(data[0]))
        else:
            out.append(float(data))
    return out


def vector_component_values(field, component_index: int) -> list[float]:
    out: list[float] = []
    for value in field.values:
        data = value.data
        if hasattr(data, "__iter__") and len(data) > component_index:
            out.append(float(data[component_index]))
    return out


def vector_magnitude_values(field) -> list[float]:
    out: list[float] = []
    for value in field.values:
        data = value.data
        if hasattr(data, "__iter__") and not isinstance(data, str):
            out.append(math.sqrt(sum(float(x) * float(x) for x in data)))
    return out


def stress_metrics(field) -> dict[str, float | int | None]:
    mises: list[float] = []
    max_principal: list[float] = []
    tensile_proxy: list[float] = []
    for value in field.values:
        mises_value = safe_float(getattr(value, "mises", None))
        max_principal_value = safe_float(getattr(value, "maxPrincipal", None))
        data = value.data
        component_max = None
        if hasattr(data, "__iter__") and not isinstance(data, str) and len(data) > 0:
            component_max = max(float(x) for x in data)
        candidates = [
            candidate
            for candidate in [max_principal_value, component_max]
            if candidate is not None
        ]
        if mises_value is not None:
            mises.append(mises_value)
        if max_principal_value is not None:
            max_principal.append(max_principal_value)
        if candidates:
            tensile_proxy.append(max(0.0, max(candidates)))
    surface_t_proxy = max(tensile_proxy) if tensile_proxy else None
    return {
        "s_value_count": len(field.values),
        "mises_max": max(mises) if mises else None,
        "mises_mean": mean(mises),
        "max_principal_stress_max": max(max_principal) if max_principal else None,
        "max_principal_stress_mean": mean(max_principal),
        "surface_t_proxy_max_tensile_pa": surface_t_proxy,
        "surface_t_proxy_max_tensile_mpa": surface_t_proxy / 1.0e6
        if surface_t_proxy is not None
        else None,
    }


def load_scan_order(case_dir: Path, case_id: str) -> tuple[str, str]:
    path = case_dir / f"scan_order_{case_id}.json"
    if not path.exists():
        return "", ""
    data = json.loads(path.read_text(encoding="utf-8"))
    return str(path), json.dumps(data.get("scan_order", []), separators=(",", ":"))


def parse_strategy(case_id: str) -> tuple[str, str, str]:
    parts = case_id.split("_", 2)
    code = parts[1] if len(parts) > 1 else ""
    strategy_id = "_".join(parts[:2]) if len(parts) > 1 else case_id
    return strategy_id, code, STRATEGY_FAMILY_BY_CODE.get(code, "unknown")


def extract_case(row: dict[str, object]) -> dict[str, object]:
    case_id = str(row["case_id"])
    n_value = str(row["N"])
    strategy_id, strategy_code, strategy_family = parse_strategy(case_id)
    case_dir = Path(str(row["case_dir"]))
    odb_path = Path(str(row["odb_path"]))
    scan_order_json, scan_order = load_scan_order(case_dir, case_id)

    result: dict[str, object] = {
        "case_id": case_id,
        "N": n_value,
        "strategy_id": strategy_id,
        "strategy_code": strategy_code,
        "strategy_family": strategy_family,
        "job_name": str(row["job_name"]),
        "case_dir": str(case_dir),
        "odb_path": str(odb_path),
        "odb_size_bytes": int(row["odb_size_bytes"]),
        "scan_order_json": scan_order_json,
        "scan_order": scan_order,
        "odb_extraction_status": "PENDING",
        "notes": "",
    }

    odb = openOdb(path=str(odb_path), readOnly=True)
    try:
        step = odb.steps[FINAL_STEP_NAME] if FINAL_STEP_NAME in odb.steps else list(odb.steps.values())[-1]
        frame = step.frames[-1]
        field_outputs = frame.fieldOutputs
        result.update(
            {
                "final_step_name": step.name,
                "final_frame_value": float(frame.frameValue),
                "final_frame_description": frame.description,
            }
        )

        if "U" in field_outputs:
            u_field = field_outputs["U"]
            u2_values = vector_component_values(u_field, 1)
            u_magnitudes = vector_magnitude_values(u_field)
            result.update(
                {
                    "u_node_count": len(u2_values),
                    "u2_min": min(u2_values) if u2_values else None,
                    "u2_max": max(u2_values) if u2_values else None,
                    "u2_range": (max(u2_values) - min(u2_values)) if u2_values else None,
                    "u2_abs_max": max(abs(value) for value in u2_values) if u2_values else None,
                    "u2_mean": mean(u2_values),
                    "u2_mean_abs": mean([abs(value) for value in u2_values]),
                    "u2_rms": rms(u2_values),
                    "u_magnitude_max": max(u_magnitudes) if u_magnitudes else None,
                }
            )

        if "PEEQ" in field_outputs:
            peeq_values = scalar_values(field_outputs["PEEQ"])
            gt_count = sum(1 for value in peeq_values if value > PEEQ_THRESHOLD)
            result.update(
                {
                    "peeq_value_count": len(peeq_values),
                    "peeq_max": max(peeq_values) if peeq_values else None,
                    "peeq_mean": mean(peeq_values),
                    "peeq_fraction_gt_0p002": gt_count / len(peeq_values)
                    if peeq_values
                    else None,
                    "peeq_count_gt_0p002": gt_count,
                }
            )

        if "S" in field_outputs:
            result.update(stress_metrics(field_outputs["S"]))

        if "NT11" in field_outputs:
            nt11_values = scalar_values(field_outputs["NT11"])
            result.update(
                {
                    "nt11_node_count": len(nt11_values),
                    "nt11_min": min(nt11_values) if nt11_values else None,
                    "nt11_max": max(nt11_values) if nt11_values else None,
                    "nt11_mean": mean(nt11_values),
                }
            )

        required_metrics = ["u2_range", "peeq_max", "mises_max", "surface_t_proxy_max_tensile_pa"]
        missing = [key for key in required_metrics if result.get(key) is None]
        if missing:
            result["odb_extraction_status"] = "WARNING_EXTRACTED_WITH_MISSING_METRICS"
            result["notes"] = "missing_metrics=" + ",".join(missing)
        else:
            result["odb_extraction_status"] = "PASS_ODB_EXTRACTED"
            result["notes"] = "final_frame_metrics_extracted"
        return result
    finally:
        odb.close()


def assign_ranks(rows: list[dict[str, object]]) -> None:
    metrics = [
        ("u2_range", "rank_within_N_u2_range"),
        ("peeq_max", "rank_within_N_peeq_max"),
        ("surface_t_proxy_max_tensile_pa", "rank_within_N_surface_t_proxy"),
        ("mises_max", "rank_within_N_mises_max"),
    ]
    for n_value in sorted({str(row["N"]) for row in rows}):
        group = [row for row in rows if str(row["N"]) == n_value]
        for metric, rank_field in metrics:
            ranked = sorted(
                [row for row in group if row.get(metric) is not None],
                key=lambda row: float(row[metric]),
            )
            for idx, row in enumerate(ranked, start=1):
                row[rank_field] = idx


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDNAMES})


def best_case(rows: list[dict[str, object]], n_value: str, metric: str) -> dict[str, object] | None:
    candidates = [
        row
        for row in rows
        if str(row["N"]) == n_value and row.get(metric) is not None
    ]
    return min(candidates, key=lambda row: float(row[metric])) if candidates else None


def build_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    by_n: dict[str, dict[str, object]] = {}
    for n_value in ["N12", "N16", "N24", "N40"]:
        group = [row for row in rows if str(row["N"]) == n_value]
        by_n[n_value] = {
            "case_count": len(group),
            "extracted_count": sum(
                1 for row in group if row["odb_extraction_status"] == "PASS_ODB_EXTRACTED"
            ),
            "warning_count": sum(
                1
                for row in group
                if row["odb_extraction_status"]
                == "WARNING_EXTRACTED_WITH_MISSING_METRICS"
            ),
            "failed_count": sum(
                1 for row in group if str(row["odb_extraction_status"]).startswith("FAIL")
            ),
            "best_u2_range_case": best_case(rows, n_value, "u2_range"),
            "best_peeq_max_case": best_case(rows, n_value, "peeq_max"),
            "best_surface_t_proxy_case": best_case(
                rows, n_value, "surface_t_proxy_max_tensile_pa"
            ),
            "best_mises_max_case": best_case(rows, n_value, "mises_max"),
        }
    total_failed = sum(1 for row in rows if str(row["odb_extraction_status"]).startswith("FAIL"))
    total_warnings = sum(
        1
        for row in rows
        if row["odb_extraction_status"] == "WARNING_EXTRACTED_WITH_MISSING_METRICS"
    )
    verdict = "PASS_STAGE3_RUN08_PROBE60_ODB_TEACHER_VALIDATION_60_OF_60"
    if total_failed:
        verdict = "FAIL_STAGE3_RUN08_PROBE60_ODB_TEACHER_VALIDATION"
    elif total_warnings:
        verdict = "WARNING_STAGE3_RUN08_PROBE60_ODB_TEACHER_VALIDATION_PARTIAL_METRICS"
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "total_cases_expected": 60,
        "total_cases_extracted": len(rows),
        "total_pass_extracted": sum(
            1 for row in rows if row["odb_extraction_status"] == "PASS_ODB_EXTRACTED"
        ),
        "total_warning_extracted": total_warnings,
        "total_failed": total_failed,
        "peeq_threshold": PEEQ_THRESHOLD,
        "final_step_name": FINAL_STEP_NAME,
        "by_N": by_n,
        "guardrails": {
            "read_only_odb_open": True,
            "abaqus_solver_run": False,
            "datacheck_run": False,
            "job_submission": False,
            "solver_outputs_modified": False,
            "shared_full32_u2_guard_used": False,
        },
    }


def compact_best(row: dict[str, object] | None, metric: str) -> str:
    if not row:
        return "n/a"
    value = row.get(metric)
    if isinstance(value, float):
        value_text = f"{value:.6g}"
    else:
        value_text = str(value)
    return f"`{row['case_id']}` ({value_text})"


def write_report(summary: dict[str, object], rows: list[dict[str, object]]) -> None:
    lines = [
        "# Run 08 Variable-N Probe60 ODB Teacher Validation Report",
        "",
        "## Verdict",
        "",
        f"`{summary['verdict']}`",
        "",
        "## Scope",
        "",
        "Read-only ODB teacher-metric extraction for the Stage 3 true variable-N Probe60 batch after external Abaqus completion.",
        "",
        "## Summary Table",
        "",
        "| N | cases | extracted | warnings | failed | best U2 range | best PEEQ max | best SurfaceT proxy |",
        "|---|---:|---:|---:|---:|---|---|---|",
    ]
    for n_value in ["N12", "N16", "N24", "N40"]:
        group = summary["by_N"][n_value]
        lines.append(
            f"| {n_value} | {group['case_count']} | {group['extracted_count']} | "
            f"{group['warning_count']} | {group['failed_count']} | "
            f"{compact_best(group['best_u2_range_case'], 'u2_range')} | "
            f"{compact_best(group['best_peeq_max_case'], 'peeq_max')} | "
            f"{compact_best(group['best_surface_t_proxy_case'], 'surface_t_proxy_max_tensile_pa')} |"
        )
    lines.extend(
        [
            "",
            "## Metric Contract",
            "",
            "- Final frame: last frame of `step_final_cooling`.",
            "- U2: nodal `U` component `U2`; lower `u2_range` is better for within-N warpage ranking.",
            f"- PEEQ: final-frame integration-point `PEEQ`; threshold fraction uses `{PEEQ_THRESHOLD}`.",
            "- SurfaceT proxy: final-frame maximum positive principal/component tensile stress over available integration points.",
            "- Stress diagnostic: final-frame `S` Mises maximum.",
            "- Temperature diagnostic: final-frame nodal `NT11` summary.",
            "- No shared full-32 U2 guard is applied; rankings are within each N.",
            "",
            "## Failed / Partial Cases",
            "",
        ]
    )
    partial = [row for row in rows if row["odb_extraction_status"] != "PASS_ODB_EXTRACTED"]
    if partial:
        for row in partial:
            lines.append(f"- `{row['case_id']}`: `{row['odb_extraction_status']}`; {row['notes']}")
    else:
        lines.append("None.")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `{LABELS_CSV}`",
            f"- `{LABELS_JSON}`",
            f"- `{SUMMARY_JSON}`",
            f"- `{REPORT_PATH}`",
            f"- `{MANIFEST_PATH}`",
            "",
            "## Guardrails",
            "",
            "- ODB files were opened read-only.",
            "- No Abaqus solver job was run.",
            "- No datacheck or job submission was run.",
            "- No solver output files were modified.",
            "- No physical superiority claim is made by this extraction report.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    solver_rows = load_solver_rows()
    extracted_rows: list[dict[str, object]] = []
    for index, row in enumerate(solver_rows, start=1):
        case_id = str(row["case_id"])
        print(f"[{index:02d}/60] extracting {case_id}")
        try:
            extracted_rows.append(extract_case(row))
        except Exception as exc:
            failed_row = {
                "case_id": case_id,
                "N": str(row.get("N", "")),
                "strategy_id": str(row.get("strategy_id", "")),
                "job_name": str(row.get("job_name", "")),
                "case_dir": str(row.get("case_dir", "")),
                "odb_path": str(row.get("odb_path", "")),
                "odb_size_bytes": int(row.get("odb_size_bytes", 0) or 0),
                "odb_extraction_status": "FAIL_ODB_EXTRACTION_ERROR",
                "notes": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            extracted_rows.append(failed_row)

    assign_ranks(extracted_rows)
    extracted_rows.sort(key=lambda row: (str(row["N"]), str(row["case_id"])))
    summary = build_summary(extracted_rows)

    write_csv(LABELS_CSV, extracted_rows)
    LABELS_JSON.write_text(json.dumps(extracted_rows, indent=2, default=json_safe) + "\n", encoding="utf-8")
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, default=json_safe) + "\n", encoding="utf-8")
    write_report(summary, extracted_rows)
    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "run_id": "run_08_variable_n_probe60_odb_teacher_validation",
                "python_executable": sys.executable,
                "verdict": summary["verdict"],
                "inputs": {"solver_completion_audit": str(SOLVER_AUDIT_JSON)},
                "outputs_written": [
                    str(LABELS_CSV),
                    str(LABELS_JSON),
                    str(SUMMARY_JSON),
                    str(REPORT_PATH),
                    str(MANIFEST_PATH),
                ],
                "guardrails": summary["guardrails"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, default=json_safe))
    return 1 if str(summary["verdict"]).startswith("FAIL") else 0


if __name__ == "__main__":
    raise SystemExit(main())
