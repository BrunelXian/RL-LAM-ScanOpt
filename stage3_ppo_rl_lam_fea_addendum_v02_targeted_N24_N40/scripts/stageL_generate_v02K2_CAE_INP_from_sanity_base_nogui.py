"""Generate Stage L PPO v02K2 targeted N24/N40 CAE/INP files.

Run with:
abaqus cae noGUI="E:\Projects\RL-LAM-ScanOpt\stage3_ppo_rl_lam_fea_addendum_v02_targeted_N24_N40\scripts\stageL_generate_v02K2_CAE_INP_from_sanity_base_nogui.py"

Generation only: no solver, no datacheck, no abqjobpilot/enqueue, no ODB.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import traceback
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "stage3"))
import run_75_generate_final_smallN_diagnostic_batch32_from_sanity_base_nogui as helpers  # noqa: E402


NAMESPACE = "stage3_ppo_rl_lam_fea_addendum_v02_targeted_N24_N40"
RUN_ID = "stageL_v02K2_CAE_INP_handoff"
BATCH_NAME = "stage3_ppo_v02K2_targeted_N24_N40_batch32_v01"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / NAMESPACE / "stageL_CAE_INP_handoff"
MANIFEST_CSV = OUTPUT_ROOT / "manifest" / "stageL_v02K2_case_manifest.csv"
GENERATION_SUMMARY = OUTPUT_ROOT / "manifest" / "stageL_v02K2_generation_summary.json"
GENERATION_MANIFEST_CSV = OUTPUT_ROOT / "manifest" / "stageL_v02K2_CAE_generation_manifest.csv"
GENERATION_MANIFEST_JSON = OUTPUT_ROOT / "manifest" / "stageL_v02K2_CAE_generation_manifest.json"
FAILURE_LOG = OUTPUT_ROOT / "reports" / "stageL_v02K2_generator_failure.json"
CASE_ROOT = PROJECT_ROOT / "cae_model" / BATCH_NAME
EXPECTED_COUNTS = {24: 16, 40: 16}
BASES = {
    24: PROJECT_ROOT / "cae_model" / "24track_full" / "sanity_base" / "24track_sanity_base.cae",
    40: PROJECT_ROOT / "cae_model" / "40track_full" / "sanity_base" / "40track_sanity_base.cae",
}
OUTPUT_VARIABLES = ("U", "PEEQ", "S", "NT11")


class StageLGenerationError(RuntimeError):
    pass


def load_rows() -> list[dict[str, object]]:
    if not MANIFEST_CSV.exists():
        raise StageLGenerationError(f"missing case manifest: {MANIFEST_CSV}")
    rows: list[dict[str, object]] = []
    with MANIFEST_CSV.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            n = int(row["n"])
            if n not in EXPECTED_COUNTS:
                raise StageLGenerationError(f"unexpected N{n}")
            scan_order = json.loads(row["scan_order"])
            helpers.validate_scan_order(n, scan_order)
            if str(row.get("teacher_validated")).lower() != "false" or str(row.get("abaqus_validated")).lower() != "false":
                raise StageLGenerationError(f"candidate already validated: {row['strategy_name']}")
            rows.append(dict(row, n=n, scan_order=scan_order))
    counts = Counter(int(row["n"]) for row in rows)
    if len(rows) != 32 or {n: counts[n] for n in EXPECTED_COUNTS} != EXPECTED_COUNTS:
        raise StageLGenerationError(f"invalid manifest counts: total={len(rows)} counts={dict(counts)}")
    return rows


def validate_paths(rows: list[dict[str, object]]) -> None:
    for row in rows:
        case_dir = Path(str(row["case_dir"]))
        if case_dir.parent != CASE_ROOT:
            raise StageLGenerationError(f"case_dir not under Stage L root: {case_dir}")
        for key in ("order_json", "expected_cae", "expected_inp", "expected_jnl", "generation_log"):
            if Path(str(row[key])).parent != case_dir:
                raise StageLGenerationError(f"{key} is not inside case_dir for {row['strategy_name']}")
        for key in ("expected_cae", "expected_inp", "expected_jnl"):
            if Path(str(row[key])).exists():
                raise StageLGenerationError(f"generated file already exists: {row[key]}")
        helpers.validate_no_solver_outputs(case_dir)


def ensure_output_fields(model) -> dict[str, object]:  # noqa: ANN001
    status = {"requested_variables": list(OUTPUT_VARIABLES), "updated_requests": [], "created_request": False, "warnings": []}
    if getattr(model, "fieldOutputRequests", None):
        for name, request in model.fieldOutputRequests.items():
            try:
                request.setValues(variables=OUTPUT_VARIABLES)
                status["updated_requests"].append(str(name))
            except Exception as exc:
                status["warnings"].append(f"could not update {name}: {exc}")
    if not status["updated_requests"]:
        creator = getattr(model, "FieldOutputRequest", None)
        if creator is None:
            status["warnings"].append("FieldOutputRequest creator unavailable; relying on base output requests")
        else:
            try:
                creator(name="F-Output-PPO-v02K2-TeacherMetrics", createStepName="step_scan_00", variables=OUTPUT_VARIABLES)
                status["created_request"] = True
            except Exception as exc:
                raise StageLGenerationError(f"could not request output fields {OUTPUT_VARIABLES}: {exc}")
    return status


def write_case_files(row: dict[str, object]) -> None:
    from abaqus import mdb  # type: ignore
    from abaqusConstants import OFF  # type: ignore

    case_dir = Path(str(row["case_dir"]))
    case_dir.mkdir(parents=True, exist_ok=True)
    helpers.validate_no_solver_outputs(case_dir)
    os.chdir(str(case_dir))
    job_name = str(row["job_name"])
    if job_name in mdb.jobs:
        del mdb.jobs[job_name]
    mdb.Job(name=job_name, model="Model-1" if "Model-1" in mdb.models else list(mdb.models.keys())[0])
    mdb.saveAs(pathName=str(row["expected_cae"]))
    mdb.jobs[job_name].writeInput(consistencyChecking=OFF)
    if not Path(str(row["expected_inp"])).exists() or Path(str(row["expected_inp"])).stat().st_size <= 0:
        raise StageLGenerationError(f"INP was not written: {row['expected_inp']}")
    Path(str(row["expected_jnl"])).write_text(
        f"Stage L PPO v02K2 generation placeholder for {row['strategy_name']}.\nCAE/INP generated only; no solver/datacheck/enqueue/ODB.\n",
        encoding="utf-8",
    )


def write_generation_log(row: dict[str, object], records: list[dict[str, object]], deactivations: list[dict[str, str]], final_record: dict[str, object], output_status: dict[str, object]) -> None:
    payload = {
        "status": "GENERATED",
        "run_id": RUN_ID,
        "batch_name": BATCH_NAME,
        "case": {
            "n": row["n"],
            "strategy_name": row["strategy_name"],
            "job_name": row["job_name"],
            "cae_path": row["expected_cae"],
            "inp_path": row["expected_inp"],
        },
        "sequence_records": records,
        "deactivation_records": deactivations,
        "final_cooling": final_record,
        "output_field_request": output_status,
        "solver_submitted": False,
        "datacheck_run": False,
        "abqjobpilot_run": False,
        "enqueue_run": False,
        "odb_opened": False,
        "teacher_validation_run": False,
    }
    Path(str(row["generation_log"])).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_generation_manifest(rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "run_id", "batch_name", "n", "strategy_name", "job_name", "case_dir",
        "scan_order_json", "cae_path", "inp_path", "jnl_path", "generation_status",
        "inp_check_status", "teacher_validated", "solver_status", "notes",
    ]
    manifest_rows = []
    for row in rows:
        manifest_rows.append({
            "run_id": RUN_ID,
            "batch_name": BATCH_NAME,
            "n": row["n"],
            "strategy_name": row["strategy_name"],
            "job_name": row["job_name"],
            "case_dir": row["case_dir"],
            "scan_order_json": row["order_json"],
            "cae_path": row["expected_cae"],
            "inp_path": row["expected_inp"],
            "jnl_path": row["expected_jnl"],
            "generation_status": "GENERATED",
            "inp_check_status": "PENDING",
            "teacher_validated": "False",
            "solver_status": "NOT_SUBMITTED",
            "notes": "Stage L PPO v02K2 CAE/INP generation only; no teacher metrics yet.",
        })
    with GENERATION_MANIFEST_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    GENERATION_MANIFEST_JSON.write_text(json.dumps({"rows": manifest_rows, "row_count": len(manifest_rows)}, indent=2), encoding="utf-8")


def generate_all(rows: list[dict[str, object]]) -> None:
    from abaqus import mdb, openMdb  # type: ignore

    generated = []
    for row in rows:
        n = int(row["n"])
        openMdb(pathName=str(BASES[n]))
        model = mdb.models["Model-1"] if "Model-1" in mdb.models else mdb.models[list(mdb.models.keys())[0]]
        helpers.infer_templates(model)
        heat_regions = helpers.validate_heat_sets_exist_in_model(model, n)
        scan_order = list(row["scan_order"])
        records = helpers.create_sequence(model, n, scan_order)
        for record in records:
            helpers.create_heat_load(model, int(record["seq"]), int(record["track"]), heat_regions)
        deactivations = helpers.deactivate_loads(model, records)
        final_record = helpers.append_final_cooling(model, records)
        output_status = ensure_output_fields(model)
        helpers.validate_model_before_write(model, n, scan_order, records, deactivations, final_record)
        write_case_files(row)
        write_generation_log(row, records, deactivations, final_record, output_status)
        generated.append({"n": n, "strategy_name": row["strategy_name"], "cae_path": row["expected_cae"], "inp_path": row["expected_inp"]})
    write_generation_manifest(rows)
    summary = {
        "verdict": "PASS_STAGEL_V02K2_GENERATION_COMPLETE",
        "generated_count": len(generated),
        "per_n_counts": {str(n): len([row for row in generated if int(row["n"]) == n]) for n in EXPECTED_COUNTS},
        "generated": generated,
        "manifest_csv": str(GENERATION_MANIFEST_CSV),
        "manifest_json": str(GENERATION_MANIFEST_JSON),
        "no_solver_run": True,
        "no_datacheck_run": True,
        "no_abqjobpilot_run": True,
        "no_enqueue_run": True,
        "no_odb_opened": True,
        "no_teacher_validation": True,
    }
    GENERATION_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> int:
    try:
        (OUTPUT_ROOT / "reports").mkdir(parents=True, exist_ok=True)
        rows = load_rows()
        validate_paths(rows)
        generate_all(rows)
        print("PASS_STAGEL_V02K2_GENERATION_COMPLETE")
        print(f"summary={GENERATION_SUMMARY}")
        return 0
    except Exception as exc:
        FAILURE_LOG.parent.mkdir(parents=True, exist_ok=True)
        FAILURE_LOG.write_text(json.dumps({"error": str(exc), "traceback": traceback.format_exc()}, indent=2), encoding="utf-8")
        print("FAIL_STAGEL_V02K2_GENERATION")
        print(str(exc))
        print(f"failure_log={FAILURE_LOG}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
