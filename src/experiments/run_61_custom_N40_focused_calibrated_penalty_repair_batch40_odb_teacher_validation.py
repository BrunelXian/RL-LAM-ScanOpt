from __future__ import annotations

import csv
import json
import math
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from odbAccess import openOdb


PROJECT_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
RUN_ID = "run_61_custom_N40_focused_calibrated_penalty_repair_batch40_odb_teacher_validation"
BATCH_NAME = "stage3_run59_N40_focused_calibrated_penalty_repair_batch40_v01"
CASE_ROOT = PROJECT_ROOT / "cae_model" / BATCH_NAME
RUN60_MANIFEST_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "stage3_run_60_custom_N40_focused_calibrated_penalty_repair_batch40_cae_inp_generation"
    / "stage3_run60_custom_N40_focused_batch40_cae_generation_manifest.csv"
)
RUN60_REPORT_MD = (
    PROJECT_ROOT
    / "docs"
    / "stage3"
    / "runs"
    / "run_60_custom_N40_focused_calibrated_penalty_repair_batch40_cae_inp_generation"
    / "RUN_60_CUSTOM_N40_FOCUSED_CALIBRATED_PENALTY_REPAIR_BATCH40_CAE_INP_GENERATION_REPORT.md"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage3_run_61_custom_N40_focused_calibrated_penalty_repair_batch40_odb_teacher_validation"
REPORT_DIR = PROJECT_ROOT / "docs" / "stage3" / "runs" / "run_61_custom_N40_focused_calibrated_penalty_repair_batch40_odb_teacher_validation"
MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "manifests" / "stage3_run_61_manifest.json"

AUDIT_CSV = OUTPUT_DIR / "run61_custom_N40_focused_batch40_solver_completion_audit.csv"
AUDIT_JSON = OUTPUT_DIR / "run61_custom_N40_focused_batch40_solver_completion_audit.json"
AUDIT_MD = OUTPUT_DIR / "run61_custom_N40_focused_batch40_solver_completion_audit.md"
EXTRACTION_SUMMARY_CSV = OUTPUT_DIR / "run61_custom_N40_focused_batch40_odb_extraction_summary.csv"
EXTRACTION_SUMMARY_JSON = OUTPUT_DIR / "run61_custom_N40_focused_batch40_odb_extraction_summary.json"
TEACHER_METRICS_CSV = OUTPUT_DIR / "run61_custom_N40_focused_batch40_teacher_metrics.csv"
TEACHER_METRICS_JSON = OUTPUT_DIR / "run61_custom_N40_focused_batch40_teacher_metrics_summary.json"
SUMMARY_JSON = OUTPUT_DIR / "run61_custom_N40_focused_batch40_odb_teacher_validation_summary.json"
REPORT_MD = REPORT_DIR / "RUN_61_CUSTOM_N40_FOCUSED_CALIBRATED_PENALTY_REPAIR_BATCH40_ODB_TEACHER_VALIDATION_REPORT.md"

FINAL_STEP_NAME = "step_final_cooling"
PEEQ_THRESHOLD = 0.002
SUCCESS_MARKER = "THE ANALYSIS HAS COMPLETED SUCCESSFULLY"
FATAL_MARKERS = [
    "Abaqus/Standard aborted",
    "THE ANALYSIS HAS BEEN TERMINATED",
    "THE ANALYSIS HAS NOT BEEN COMPLETED",
    "Too many attempts made for this increment",
    "***ERROR",
    "exited with an error",
    "ERROR in job",
    "Abaqus Error",
]
WARNING_MARKERS = ["***WARNING", "*** WARNING"]
REQUIRED_FIELDS = ["U", "PEEQ", "S", "NT11"]
EXPECTED_BY_N = {24: 16, 40: 24}
TOTAL_EXPECTED = 40
SPECIAL_GENERATION_NOTES = {
    "includes_N12": False,
    "includes_N16": False,
    "includes_N24": True,
    "includes_N32_cases": False,
    "includes_N40": True,
    "generated_CAE_INP_JNL_only_in_run60": True,
    "final_step_required_for_metrics": "step_final_cooling",
    "n40_step_cool_initialInc": 0.001,
    "step_final_cooling_initialInc": 0.01,
    "step_final_cooling_timePeriod": 1200.0,
}
PATCHED_STRATEGIES: set[str] = set()

AUDIT_FIELDS = [
    "run_id",
    "batch_name",
    "n",
    "handoff_strategy_name",
    "patched_after_cool_initialInc_failure",
    "expected_job_name",
    "job_name_matches_expected",
    "job_name",
    "case_dir",
    "inp_path",
    "sta_path",
    "dat_path",
    "msg_path",
    "odb_path",
    "lck_paths",
    "sta_exists",
    "dat_exists",
    "msg_exists",
    "odb_exists",
    "odb_size_bytes",
    "lck_present",
    "sta_success_marker",
    "sta_fatal_marker",
    "dat_fatal_marker",
    "msg_fatal_marker",
    "nonfatal_warning_marker",
    "completion_status",
    "notes",
    "last_sta_lines",
    "last_msg_lines",
    "manual_rerun_command",
]

EXTRACTION_FIELDS = [
    "run_id",
    "batch_name",
    "n",
    "handoff_strategy_name",
    "patched_after_cool_initialInc_failure",
    "job_name",
    "odb_path",
    "completion_status",
    "odb_extraction_status",
    "teacher_validation_status",
    "final_step_name",
    "final_frame_time",
    "extracted_field_names",
    "missing_required_fields",
    "notes",
]

METRIC_FIELDS = EXTRACTION_FIELDS + [
    "u_node_count",
    "u2_min",
    "u2_max",
    "u2_range",
    "u2_abs_max",
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
    "norm_cost_within_N_u2_range",
    "norm_cost_within_N_peeq_max",
    "norm_cost_within_N_surface_t_proxy",
    "norm_cost_within_N_mises_max",
]


def read_text(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def tail_lines(text: str, count: int = 12) -> str:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    return " | ".join(lines[-count:])


def contains_any(text: str, markers: list[str]) -> bool:
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in markers)


def contains_warning(text: str) -> bool:
    if contains_any(text, WARNING_MARKERS):
        return True
    patterns = [
        r"\b([1-9][0-9]*)\s+WARNING\s+MESSAGES?\b",
        r"\b([1-9][0-9]*)\s+WARNINGS?\b",
    ]
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def load_manifest_rows() -> list[dict[str, str]]:
    with RUN60_MANIFEST_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != TOTAL_EXPECTED:
        raise RuntimeError(f"Expected {TOTAL_EXPECTED} run60 custom N40-focused batch40 manifest rows, got {len(rows)}")
    if any(row.get("batch_name") != BATCH_NAME for row in rows):
        raise RuntimeError("Manifest contains rows outside run60 custom N40-focused batch40 batch name")
    return rows


def audit_case(row: dict[str, str]) -> dict[str, object]:
    case_dir = Path(row["case_dir"])
    strategy = row["handoff_strategy_name"]
    expected_job_name = f"J2D_{strategy}"
    job_name = row["job_name"]
    job_name_matches_expected = job_name == expected_job_name
    inp_path = Path(row["inp_path"])
    sta_path = case_dir / f"{job_name}.sta"
    dat_path = case_dir / f"{job_name}.dat"
    msg_path = case_dir / f"{job_name}.msg"
    odb_path = case_dir / f"{job_name}.odb"
    lck_paths = sorted(case_dir.glob("*.lck")) if case_dir.exists() else []

    sta_text = read_text(sta_path)
    dat_text = read_text(dat_path)
    msg_text = read_text(msg_path)

    sta_exists = sta_path.exists()
    dat_exists = dat_path.exists()
    msg_exists = msg_path.exists()
    odb_exists = odb_path.exists()
    odb_size_bytes = odb_path.stat().st_size if odb_exists else 0
    sta_success_marker = SUCCESS_MARKER.lower() in sta_text.lower()
    sta_fatal_marker = contains_any(sta_text, FATAL_MARKERS)
    dat_fatal_marker = contains_any(dat_text, FATAL_MARKERS)
    msg_fatal_marker = contains_any(msg_text, FATAL_MARKERS)
    warning_marker = any(contains_warning(text) for text in [sta_text, dat_text, msg_text])

    blockers: list[str] = []
    if not case_dir.exists():
        blockers.append("missing_case_dir")
    if not job_name_matches_expected:
        blockers.append("job_name_mismatch")
    if not sta_exists:
        blockers.append("missing_sta")
    if not dat_exists:
        blockers.append("missing_dat")
    if not msg_exists:
        blockers.append("missing_msg")
    if not sta_success_marker:
        blockers.append("missing_sta_success_marker")
    if not odb_exists:
        blockers.append("missing_odb")
    if odb_exists and odb_size_bytes <= 0:
        blockers.append("empty_odb")
    if lck_paths:
        blockers.append("lck_present")
    if sta_fatal_marker or dat_fatal_marker or msg_fatal_marker:
        blockers.append("fatal_marker_present")

    if blockers:
        completion_status = "FAIL_INCOMPLETE_OR_ABORTED"
        notes = "; ".join(blockers)
    elif warning_marker:
        completion_status = "WARNING_SUCCESS_WITH_WARNINGS"
        notes = "complete_with_nonfatal_warnings"
    else:
        completion_status = "PASS_SOLVER_COMPLETE"
        notes = "complete_no_lck_no_fatal_markers"

    return {
        "run_id": RUN_ID,
        "batch_name": row["batch_name"],
        "n": int(row["n"]),
        "handoff_strategy_name": strategy,
        "patched_after_cool_initialInc_failure": strategy in PATCHED_STRATEGIES,
        "expected_job_name": expected_job_name,
        "job_name_matches_expected": job_name_matches_expected,
        "job_name": job_name,
        "case_dir": str(case_dir),
        "inp_path": str(inp_path),
        "sta_path": str(sta_path),
        "dat_path": str(dat_path),
        "msg_path": str(msg_path),
        "odb_path": str(odb_path),
        "lck_paths": ";".join(str(path) for path in lck_paths),
        "sta_exists": sta_exists,
        "dat_exists": dat_exists,
        "msg_exists": msg_exists,
        "odb_exists": odb_exists,
        "odb_size_bytes": odb_size_bytes,
        "lck_present": bool(lck_paths),
        "sta_success_marker": sta_success_marker,
        "sta_fatal_marker": sta_fatal_marker,
        "dat_fatal_marker": dat_fatal_marker,
        "msg_fatal_marker": msg_fatal_marker,
        "nonfatal_warning_marker": warning_marker,
        "completion_status": completion_status,
        "notes": notes,
        "last_sta_lines": tail_lines(sta_text),
        "last_msg_lines": tail_lines(msg_text),
        "manual_rerun_command": f'enqueue --inp "{inp_path}" --cpus 14 --batch {BATCH_NAME} --strategy {strategy}',
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


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
            if len(data) > 0:
                out.append(float(data[0]))
        else:
            out.append(float(data))
    return out


def vector_component_values(field, component_index: int) -> list[float]:
    out: list[float] = []
    for value in field.values:
        data = value.data
        if hasattr(data, "__iter__") and not isinstance(data, str) and len(data) > component_index:
            out.append(float(data[component_index]))
    return out


def vector_magnitude_values(field) -> list[float]:
    out: list[float] = []
    for value in field.values:
        data = value.data
        if hasattr(data, "__iter__") and not isinstance(data, str):
            out.append(math.sqrt(sum(float(x) * float(x) for x in data)))
    return out


def stress_metrics(field) -> dict[str, object]:
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
        if mises_value is not None:
            mises.append(mises_value)
        if max_principal_value is not None:
            max_principal.append(max_principal_value)
        candidates = [candidate for candidate in [max_principal_value, component_max] if candidate is not None]
        if candidates:
            tensile_proxy.append(max(0.0, max(candidates)))
    surface_t = max(tensile_proxy) if tensile_proxy else None
    return {
        "s_value_count": len(field.values),
        "mises_max": max(mises) if mises else None,
        "mises_mean": mean(mises),
        "max_principal_stress_max": max(max_principal) if max_principal else None,
        "max_principal_stress_mean": mean(max_principal),
        "surface_t_proxy_max_tensile_pa": surface_t,
        "surface_t_proxy_max_tensile_mpa": surface_t / 1.0e6 if surface_t is not None else None,
    }


def extract_case(audit_row: dict[str, object]) -> dict[str, object]:
    base = {
        "run_id": RUN_ID,
        "batch_name": BATCH_NAME,
        "n": audit_row["n"],
        "handoff_strategy_name": audit_row["handoff_strategy_name"],
        "patched_after_cool_initialInc_failure": audit_row["patched_after_cool_initialInc_failure"],
        "job_name": audit_row["job_name"],
        "odb_path": audit_row["odb_path"],
        "completion_status": audit_row["completion_status"],
    }
    odb = openOdb(path=str(audit_row["odb_path"]), readOnly=True)
    try:
        if FINAL_STEP_NAME not in odb.steps:
            return {
                **base,
                "odb_extraction_status": "FAIL_FINAL_STEP_MISSING",
                "teacher_validation_status": "FAIL_TEACHER_FIELDS_NOT_EXTRACTED",
                "missing_required_fields": FINAL_STEP_NAME,
                "notes": "final step missing",
            }
        frame = odb.steps[FINAL_STEP_NAME].frames[-1]
        fields = frame.fieldOutputs
        field_names = sorted(list(fields.keys()))
        missing_fields = [field for field in REQUIRED_FIELDS if field not in fields]
        result: dict[str, object] = {
            **base,
            "odb_extraction_status": "PENDING",
            "teacher_validation_status": "PENDING",
            "final_step_name": FINAL_STEP_NAME,
            "final_frame_time": float(frame.frameValue),
            "extracted_field_names": ";".join(field_names),
            "missing_required_fields": ";".join(missing_fields),
            "notes": "",
        }
        if not missing_fields:
            u2_values = vector_component_values(fields["U"], 1)
            u_magnitudes = vector_magnitude_values(fields["U"])
            peeq_values = scalar_values(fields["PEEQ"])
            peeq_gt_count = sum(1 for value in peeq_values if value > PEEQ_THRESHOLD)
            nt11_values = scalar_values(fields["NT11"])
            result.update(
                {
                    "u_node_count": len(u2_values),
                    "u2_min": min(u2_values) if u2_values else None,
                    "u2_max": max(u2_values) if u2_values else None,
                    "u2_range": max(u2_values) - min(u2_values) if u2_values else None,
                    "u2_abs_max": max(abs(value) for value in u2_values) if u2_values else None,
                    "u2_mean_abs": mean([abs(value) for value in u2_values]),
                    "u2_rms": rms(u2_values),
                    "u_magnitude_max": max(u_magnitudes) if u_magnitudes else None,
                    "peeq_value_count": len(peeq_values),
                    "peeq_max": max(peeq_values) if peeq_values else None,
                    "peeq_mean": mean(peeq_values),
                    "peeq_fraction_gt_0p002": peeq_gt_count / len(peeq_values) if peeq_values else None,
                    "peeq_count_gt_0p002": peeq_gt_count,
                    "nt11_node_count": len(nt11_values),
                    "nt11_min": min(nt11_values) if nt11_values else None,
                    "nt11_max": max(nt11_values) if nt11_values else None,
                    "nt11_mean": mean(nt11_values),
                }
            )
            result.update(stress_metrics(fields["S"]))

        metric_keys = ["u2_range", "peeq_max", "mises_max", "surface_t_proxy_max_tensile_pa"]
        missing_metrics = [key for key in metric_keys if result.get(key) is None]
        if missing_fields or missing_metrics:
            result["odb_extraction_status"] = "FAIL_REQUIRED_FIELDS_OR_METRICS_MISSING"
            result["teacher_validation_status"] = "FAIL_TEACHER_FIELDS_NOT_EXTRACTED"
            details = []
            if missing_fields:
                details.append("missing_fields=" + ",".join(missing_fields))
            if missing_metrics:
                details.append("missing_metrics=" + ",".join(missing_metrics))
            result["notes"] = "; ".join(details)
        else:
            result["odb_extraction_status"] = "PASS_ODB_FINAL_FRAME_EXTRACTED"
            result["teacher_validation_status"] = "PASS_TEACHER_FIELDS_EXTRACTED"
            result["notes"] = "final_frame_required_fields_and_metrics_extracted"
        return result
    finally:
        odb.close()


def add_within_n_ranks_and_norms(rows: list[dict[str, object]]) -> None:
    metrics = [
        ("u2_range", "rank_within_N_u2_range", "norm_cost_within_N_u2_range"),
        ("peeq_max", "rank_within_N_peeq_max", "norm_cost_within_N_peeq_max"),
        ("surface_t_proxy_max_tensile_pa", "rank_within_N_surface_t_proxy", "norm_cost_within_N_surface_t_proxy"),
        ("mises_max", "rank_within_N_mises_max", "norm_cost_within_N_mises_max"),
    ]
    for n in sorted({int(row["n"]) for row in rows}):
        group = [row for row in rows if int(row["n"]) == n]
        for metric, rank_field, norm_field in metrics:
            valid = [row for row in group if row.get(metric) not in (None, "")]
            ranked = sorted(valid, key=lambda row: float(row[metric]))
            for rank, row in enumerate(ranked, start=1):
                row[rank_field] = rank
            values = [float(row[metric]) for row in valid]
            if not values:
                continue
            lo = min(values)
            hi = max(values)
            denom = hi - lo
            for row in valid:
                row[norm_field] = 0.0 if denom == 0.0 else (float(row[metric]) - lo) / denom


def build_audit_summary(audit_rows: list[dict[str, object]]) -> dict[str, object]:
    by_n: dict[str, dict[str, int]] = {}
    for n, expected in EXPECTED_BY_N.items():
        group = [row for row in audit_rows if int(row["n"]) == n]
        by_n[f"N{n}"] = {
            "expected": expected,
            "audited": len(group),
            "complete": sum(1 for row in group if row["completion_status"] in {"PASS_SOLVER_COMPLETE", "WARNING_SUCCESS_WITH_WARNINGS"}),
            "warning": sum(1 for row in group if row["completion_status"] == "WARNING_SUCCESS_WITH_WARNINGS"),
            "failed_or_incomplete": sum(1 for row in group if row["completion_status"] == "FAIL_INCOMPLETE_OR_ABORTED"),
        }
    failed = [row for row in audit_rows if row["completion_status"] == "FAIL_INCOMPLETE_OR_ABORTED"]
    warnings = [row for row in audit_rows if row["completion_status"] == "WARNING_SUCCESS_WITH_WARNINGS"]
    expected_ok = all(by_n[f"N{n}"]["audited"] == expected for n, expected in EXPECTED_BY_N.items())
    if failed or not expected_ok:
        verdict = "FAIL_RUN61_CUSTOM_N40_FOCUSED_BATCH40_SOLVER_COMPLETION_INCOMPLETE"
    elif warnings:
        verdict = "WARNING_RUN61_CUSTOM_N40_FOCUSED_BATCH40_SOLVER_COMPLETION_WITH_NONFATAL_WARNINGS"
    else:
        verdict = "PASS_RUN61_CUSTOM_N40_FOCUSED_BATCH40_SOLVER_COMPLETION_40_OF_40"
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "total_expected": TOTAL_EXPECTED,
        "total_audited": len(audit_rows),
        "total_complete": sum(1 for row in audit_rows if row["completion_status"] in {"PASS_SOLVER_COMPLETE", "WARNING_SUCCESS_WITH_WARNINGS"}),
        "total_warning": len(warnings),
        "total_failed_or_incomplete": len(failed),
        "total_lck_present": sum(1 for row in audit_rows if row["lck_present"]),
        "patched_cases": sorted(PATCHED_STRATEGIES),
        "generation_details": SPECIAL_GENERATION_NOTES,
        "run60_report": str(RUN60_REPORT_MD),
        "by_N": by_n,
    }


def build_extraction_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    failed = [row for row in rows if row["teacher_validation_status"] != "PASS_TEACHER_FIELDS_EXTRACTED"]
    by_n: dict[str, dict[str, int]] = {}
    for n, expected in EXPECTED_BY_N.items():
        group = [row for row in rows if int(row["n"]) == n]
        by_n[f"N{n}"] = {
            "expected": expected,
            "extracted": sum(1 for row in group if row["teacher_validation_status"] == "PASS_TEACHER_FIELDS_EXTRACTED"),
            "failed": sum(1 for row in group if row["teacher_validation_status"] != "PASS_TEACHER_FIELDS_EXTRACTED"),
        }
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "verdict": "PASS_RUN61_CUSTOM_N40_FOCUSED_BATCH40_ODB_TEACHER_VALIDATION_40_OF_40" if not failed and len(rows) == TOTAL_EXPECTED else "FAIL_RUN61_CUSTOM_N40_FOCUSED_BATCH40_ODB_TEACHER_VALIDATION",
        "total_expected": TOTAL_EXPECTED,
        "total_extracted": len(rows),
        "total_pass": sum(1 for row in rows if row["teacher_validation_status"] == "PASS_TEACHER_FIELDS_EXTRACTED"),
        "total_failed": len(failed),
        "by_N": by_n,
        "required_fields": REQUIRED_FIELDS,
        "final_step_name": FINAL_STEP_NAME,
        "peeq_threshold": PEEQ_THRESHOLD,
    }


def build_teacher_metrics_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": RUN_ID,
        "batch_name": BATCH_NAME,
        "total_rows": len(rows),
        "fields": METRIC_FIELDS,
        "by_N": {},
    }
    numeric_fields = [
        "u2_range",
        "u2_abs_max",
        "peeq_max",
        "peeq_mean",
        "peeq_fraction_gt_0p002",
        "mises_max",
        "mises_mean",
        "max_principal_stress_max",
        "surface_t_proxy_max_tensile_pa",
        "surface_t_proxy_max_tensile_mpa",
        "nt11_min",
        "nt11_max",
        "nt11_mean",
    ]
    for n in EXPECTED_BY_N:
        group = [row for row in rows if int(row["n"]) == n]
        stats: dict[str, object] = {"count": len(group)}
        for field in numeric_fields:
            values = [float(row[field]) for row in group if row.get(field) not in (None, "")]
            if values:
                stats[field] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values),
                }
        summary["by_N"][f"N{n}"] = stats
    return summary


def write_audit_markdown(audit_summary: dict[str, object], audit_rows: list[dict[str, object]]) -> None:
    lines = [
        "# Run61 Custom N40-Focused Calibrated Penalty-Repair Batch40 Solver Completion Audit",
        "",
        f"Verdict: `{audit_summary['verdict']}`",
        "",
        f"- total: `{audit_summary['total_complete']}/{audit_summary['total_expected']}`",
        f"- warning: `{audit_summary['total_warning']}`",
        f"- lck present: `{audit_summary['total_lck_present']}`",
        "",
        "| N | expected | complete | warning | failed_or_incomplete |",
        "|---|---:|---:|---:|---:|",
    ]
    for n in EXPECTED_BY_N:
        group = audit_summary["by_N"][f"N{n}"]
        lines.append(f"| N{n} | {group['expected']} | {group['complete']} | {group['warning']} | {group['failed_or_incomplete']} |")
    incomplete = [row for row in audit_rows if row["completion_status"] == "FAIL_INCOMPLETE_OR_ABORTED"]
    lines.extend(["", "## Failed / Incomplete Cases", ""])
    if incomplete:
        for row in incomplete:
            lines.append(f"- `{row['handoff_strategy_name']}`: `{row['notes']}`")
            lines.append(f"  - STA tail: `{row['last_sta_lines']}`")
            lines.append(f"  - MSG tail: `{row['last_msg_lines']}`")
            lines.append(f"  - Manual rerun command: `{row['manual_rerun_command']}`")
    else:
        lines.append("None.")
    lines.extend(
        [
            "",
            "## Gate",
            "",
            "ODB extraction may proceed only when failed_or_incomplete is 0 and all expected cases are complete.",
        ]
    )
    AUDIT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(audit_summary: dict[str, object], extraction_summary: dict[str, object] | None, audit_rows: list[dict[str, object]], extraction_rows: list[dict[str, object]]) -> None:
    verdict = extraction_summary["verdict"] if extraction_summary is not None else audit_summary["verdict"]
    lines = [
        "# Stage 3 Run61 Custom N40-Focused Calibrated Penalty-Repair Batch40 ODB Teacher Validation Report",
        "",
        "## Verdict",
        "",
        f"`{verdict}`",
        "",
        "## Solver Completion Audit",
        "",
        f"- total: `{audit_summary['total_complete']}/{audit_summary['total_expected']}`",
        f"- warning: `{audit_summary['total_warning']}`",
        f"- lck present: `{audit_summary['total_lck_present']}`",
        "",
        "| N | expected | complete | warning | failed_or_incomplete |",
        "|---|---:|---:|---:|---:|",
    ]
    for n in EXPECTED_BY_N:
        group = audit_summary["by_N"][f"N{n}"]
        lines.append(f"| N{n} | {group['expected']} | {group['complete']} | {group['warning']} | {group['failed_or_incomplete']} |")
    lines.extend(
        [
            "",
            "## Run60 Generation Details",
            "",
            "- No N12, N16, or N32 cases were generated.",
            "- N24 and N40 only: N24=16, N40=24.",
            "- Metrics are extracted from the final frame of `step_final_cooling`.",
            "- N40 cases were generated with `step_cool_XX` initialInc = `0.001`.",
        ]
    )
    patched_rows = [row for row in audit_rows if row["patched_after_cool_initialInc_failure"]]
    lines.extend(["", "## Patched/Rerun Cases", ""])
    for row in patched_rows:
        lines.append(f"- `{row['handoff_strategy_name']}`: `{row['completion_status']}`; {row['notes']}")
    if not patched_rows:
        lines.append("None.")
    incomplete = [row for row in audit_rows if row["completion_status"] == "FAIL_INCOMPLETE_OR_ABORTED"]
    lines.extend(["", "## Incomplete Cases", ""])
    if incomplete:
        for row in incomplete:
            lines.append(f"- `{row['handoff_strategy_name']}`: `{row['completion_status']}`; {row['notes']}")
            lines.append(f"  - STA tail: `{row['last_sta_lines']}`")
            lines.append(f"  - MSG tail: `{row['last_msg_lines']}`")
            lines.append(f"  - Manual rerun command: `{row['manual_rerun_command']}`")
        lines.append("")
        lines.append("ODB extraction was not run because the solver completion gate failed.")
    else:
        lines.append("None.")
    if extraction_summary is not None:
        lines.extend(
            [
                "",
                "## ODB Extraction Summary",
                "",
                f"- total: `{extraction_summary['total_pass']}/{extraction_summary['total_expected']}`",
                f"- final step: `{FINAL_STEP_NAME}`",
                f"- required fields: `{';'.join(REQUIRED_FIELDS)}`",
                "",
                "| N | expected | extracted | failed |",
                "|---|---:|---:|---:|",
            ]
        )
        for n in EXPECTED_BY_N:
            group = extraction_summary["by_N"][f"N{n}"]
            lines.append(f"| N{n} | {group['expected']} | {group['extracted']} | {group['failed']} |")
        failed_extracts = [row for row in extraction_rows if row["teacher_validation_status"] != "PASS_TEACHER_FIELDS_EXTRACTED"]
        lines.extend(["", "## Failed Extractions", ""])
        if failed_extracts:
            for row in failed_extracts:
                lines.append(f"- `{row['handoff_strategy_name']}`: `{row['teacher_validation_status']}`; {row['notes']}")
        else:
            lines.append("None.")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `{AUDIT_CSV}`",
            f"- `{AUDIT_JSON}`",
            f"- `{AUDIT_MD}`",
            f"- `{EXTRACTION_SUMMARY_CSV}`",
            f"- `{EXTRACTION_SUMMARY_JSON}`",
            f"- `{TEACHER_METRICS_CSV}`",
            f"- `{TEACHER_METRICS_JSON}`",
            f"- `{SUMMARY_JSON}`",
            f"- `{REPORT_MD}`",
            f"- `{MANIFEST_PATH}`",
            "",
            "## Scientific Boundary",
            "",
            "This report records computed teacher metrics for Run60 custom N40-focused calibrated penalty-repair batch40. It does not claim physical superiority beyond computed teacher metrics, RL/GNN success, or arbitrary-N generalisation.",
            "",
            "## Guardrails",
            "",
            "- ODB files were opened read-only only after 40/40 solver completion.",
            "- No Abaqus solver job was run.",
            "- No datacheck was run.",
            "- No abqjobpilot/enqueue command was run.",
            "- No CAE/INP/JNL or base sanity files were modified.",
            "- No commit or push was made.",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    manifest_rows = load_manifest_rows()
    audit_rows = [audit_case(row) for row in manifest_rows]
    audit_rows.sort(key=lambda row: (int(row["n"]), str(row["handoff_strategy_name"])))
    audit_summary = build_audit_summary(audit_rows)
    write_csv(AUDIT_CSV, audit_rows, AUDIT_FIELDS)
    AUDIT_JSON.write_text(json.dumps({"summary": audit_summary, "rows": audit_rows}, indent=2, default=json_safe) + "\n", encoding="utf-8")
    write_audit_markdown(audit_summary, audit_rows)

    extraction_rows: list[dict[str, object]] = []
    extraction_summary = None
    teacher_metrics_summary = None
    if audit_summary["total_failed_or_incomplete"] == 0 and audit_summary["total_complete"] == TOTAL_EXPECTED:
        for index, row in enumerate(audit_rows, start=1):
            print(f"[{index:02d}/{TOTAL_EXPECTED}] extracting {row['handoff_strategy_name']}")
            try:
                extraction_rows.append(extract_case(row))
            except Exception as exc:
                extraction_rows.append(
                    {
                        "run_id": RUN_ID,
                        "batch_name": BATCH_NAME,
                        "n": row["n"],
                        "handoff_strategy_name": row["handoff_strategy_name"],
                        "patched_after_cool_initialInc_failure": row["patched_after_cool_initialInc_failure"],
                        "job_name": row["job_name"],
                        "odb_path": row["odb_path"],
                        "completion_status": row["completion_status"],
                        "odb_extraction_status": "FAIL_ODB_EXTRACTION_ERROR",
                        "teacher_validation_status": "FAIL_TEACHER_FIELDS_NOT_EXTRACTED",
                        "notes": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )
        extraction_rows.sort(key=lambda row: (int(row["n"]), str(row["handoff_strategy_name"])))
        add_within_n_ranks_and_norms(extraction_rows)
        extraction_summary = build_extraction_summary(extraction_rows)
        teacher_metrics_summary = build_teacher_metrics_summary(extraction_rows)
        write_csv(EXTRACTION_SUMMARY_CSV, extraction_rows, EXTRACTION_FIELDS)
        EXTRACTION_SUMMARY_JSON.write_text(
            json.dumps({"summary": extraction_summary, "rows": extraction_rows}, indent=2, default=json_safe) + "\n",
            encoding="utf-8",
        )
        write_csv(TEACHER_METRICS_CSV, extraction_rows, METRIC_FIELDS)
        TEACHER_METRICS_JSON.write_text(
            json.dumps({"summary": teacher_metrics_summary, "rows": extraction_rows}, indent=2, default=json_safe) + "\n",
            encoding="utf-8",
        )
    else:
        write_csv(EXTRACTION_SUMMARY_CSV, [], EXTRACTION_FIELDS)
        EXTRACTION_SUMMARY_JSON.write_text(json.dumps({"summary": None, "rows": []}, indent=2) + "\n", encoding="utf-8")
        write_csv(TEACHER_METRICS_CSV, [], METRIC_FIELDS)
        TEACHER_METRICS_JSON.write_text(json.dumps({"summary": None, "rows": []}, indent=2) + "\n", encoding="utf-8")

    summary_payload = {
        "audit_summary": audit_summary,
        "extraction_summary": extraction_summary,
        "teacher_metrics_summary": teacher_metrics_summary,
        "guardrails": {
            "solver_run": False,
            "datacheck_run": False,
            "abqjobpilot_or_enqueue_run": False,
            "cae_inp_jnl_base_modified": False,
            "commit_or_push": False,
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary_payload, indent=2, default=json_safe) + "\n", encoding="utf-8")
    write_report(audit_summary, extraction_summary, audit_rows, extraction_rows)
    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "run_id": RUN_ID,
                "batch_name": BATCH_NAME,
                "case_root": str(CASE_ROOT),
                "python_executable": sys.executable,
                "verdict": extraction_summary["verdict"] if extraction_summary else audit_summary["verdict"],
                "inputs": {
                    "run60_manifest_csv": str(RUN60_MANIFEST_CSV),
                    "run60_report_md": str(RUN60_REPORT_MD),
                },
                "generation_details": SPECIAL_GENERATION_NOTES,
                "outputs_written": [
                    str(AUDIT_CSV),
                    str(AUDIT_JSON),
                    str(AUDIT_MD),
                    str(EXTRACTION_SUMMARY_CSV),
                    str(EXTRACTION_SUMMARY_JSON),
                    str(TEACHER_METRICS_CSV),
                    str(TEACHER_METRICS_JSON),
                    str(SUMMARY_JSON),
                    str(REPORT_MD),
                    str(MANIFEST_PATH),
                ],
                "guardrails": summary_payload["guardrails"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    final_verdict = extraction_summary["verdict"] if extraction_summary else audit_summary["verdict"]
    print(json.dumps({"verdict": final_verdict, **summary_payload}, indent=2, default=json_safe))
    return 1 if str(final_verdict).startswith("FAIL") else 0


if __name__ == "__main__":
    raise SystemExit(main())
