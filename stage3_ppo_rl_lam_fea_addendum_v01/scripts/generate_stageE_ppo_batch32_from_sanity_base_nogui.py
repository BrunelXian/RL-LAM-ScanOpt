"""Generate Stage E PPO batch32 CAE/INP artifacts from sanity bases.

Run with:
abaqus cae noGUI="E:\Projects\RL-LAM-ScanOpt\stage3_ppo_rl_lam_fea_addendum_v01\scripts\generate_stageE_ppo_batch32_from_sanity_base_nogui.py"

This script writes CAE/INP/JNL/generation logs only. It never submits jobs,
runs datacheck, executes abqjobpilot/enqueue, opens ODB, or performs teacher
validation.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import traceback
from pathlib import Path


PROJECT_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
STAGE3_SCRIPTS = PROJECT_ROOT / "scripts" / "stage3"
sys.path.insert(0, str(STAGE3_SCRIPTS))

import run_75_generate_final_smallN_diagnostic_batch32_from_sanity_base_nogui as helpers  # noqa: E402


RUN_ID = "stageE_ppo_batch32_cae_inp_generation"
BATCH_NAME = "stage3_ppo_policy_only_batch32_v01"
NAMESPACE = "stage3_ppo_rl_lam_fea_addendum_v01"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / NAMESPACE / "stageE_teacher_validation_handoff"
MANIFEST_CSV = OUTPUT_ROOT / "manifest" / "stageE_ppo_batch32_case_manifest.csv"
GENERATION_SUMMARY = OUTPUT_ROOT / "manifest" / "stageE_ppo_batch32_generation_summary.json"
GENERATION_MANIFEST_CSV = OUTPUT_ROOT / "manifest" / "stageE_ppo_batch32_cae_generation_manifest.csv"
GENERATION_MANIFEST_JSON = OUTPUT_ROOT / "manifest" / "stageE_ppo_batch32_cae_generation_manifest.json"
FAILURE_LOG = OUTPUT_ROOT / "reports" / "stageE_ppo_batch32_generator_failure.json"
CASE_ROOT = PROJECT_ROOT / "cae_model" / BATCH_NAME
EXPECTED_COUNTS = {12: 8, 16: 8, 24: 8, 40: 8}
BASES = {
    12: PROJECT_ROOT / "cae_model" / "12track_full" / "sanity_base" / "12track_sanity_base.cae",
    16: PROJECT_ROOT / "cae_model" / "16track_full" / "sanity_base" / "16track_sanity_base.cae",
    24: PROJECT_ROOT / "cae_model" / "24track_full" / "sanity_base" / "24track_sanity_base.cae",
    40: PROJECT_ROOT / "cae_model" / "40track_full" / "sanity_base" / "40track_sanity_base.cae",
}
SOLVER_EXTS = (".odb", ".sim", ".sta", ".dat", ".msg", ".lck")
OUTPUT_VARIABLES = ("U", "PEEQ", "S", "NT11")


class StageEGenerationError(RuntimeError):
    pass


def load_rows() -> list[dict[str, object]]:
    if not MANIFEST_CSV.exists():
        raise StageEGenerationError("missing Stage E case manifest: {}".format(MANIFEST_CSV))
    rows: list[dict[str, object]] = []
    with MANIFEST_CSV.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            n = int(row["n"])
            if n not in EXPECTED_COUNTS:
                raise StageEGenerationError("unexpected or forbidden N{}".format(n))
            scan_order = json.loads(row["scan_order"])
            helpers.validate_scan_order(n, scan_order)
            if row.get("candidate_source") != "PPO_checkpoint_inference":
                raise StageEGenerationError("non-PPO candidate source: {}".format(row.get("strategy_name")))
            if str(row.get("teacher_validated")).lower() != "false" or str(row.get("abaqus_validated")).lower() != "false":
                raise StageEGenerationError("candidate already marked validated: {}".format(row.get("strategy_name")))
            rows.append(dict(row, n=n, scan_order=scan_order))
    counts = {n: len([row for row in rows if int(row["n"]) == n]) for n in EXPECTED_COUNTS}
    if len(rows) != 32 or counts != EXPECTED_COUNTS:
        raise StageEGenerationError("row counts invalid: total={} counts={}".format(len(rows), counts))
    return rows


def validate_paths(rows: list[dict[str, object]]) -> None:
    for row in rows:
        case_dir = Path(str(row["case_dir"]))
        if case_dir.parent != CASE_ROOT:
            raise StageEGenerationError("case_dir not under Stage E case root: {}".format(case_dir))
        for key in ("scan_order_json", "cae_path", "inp_path", "jnl_path", "generation_log_path"):
            if Path(str(row[key])).parent != case_dir:
                raise StageEGenerationError("{} is not inside case_dir for {}".format(key, row["strategy_name"]))
        for key in ("cae_path", "inp_path", "jnl_path"):
            if Path(str(row[key])).exists():
                raise StageEGenerationError("generated file already exists: {}".format(row[key]))
        if not Path(str(row["scan_order_json"])).exists():
            raise StageEGenerationError("missing scan_order_json: {}".format(row["scan_order_json"]))
        helpers.validate_no_solver_outputs(case_dir)


def ensure_output_fields(model) -> dict[str, object]:  # noqa: ANN001
    status = {"requested_variables": list(OUTPUT_VARIABLES), "updated_requests": [], "create_attempted": False, "created_request": False, "warnings": []}
    if getattr(model, "fieldOutputRequests", None):
        for name, request in model.fieldOutputRequests.items():
            try:
                request.setValues(variables=OUTPUT_VARIABLES)
                status["updated_requests"].append(str(name))
            except Exception:
                status["warnings"].append("could not update field output request {}".format(name))
    if not status["updated_requests"]:
        creator = getattr(model, "FieldOutputRequest", None)
        if creator is not None:
            status["create_attempted"] = True
            try:
                creator(name="F-Output-PPO-TeacherMetrics", createStepName="step_scan_00", variables=OUTPUT_VARIABLES)
                status["created_request"] = True
            except Exception as exc:
                raise StageEGenerationError("could not request output fields {}: {}".format(OUTPUT_VARIABLES, exc))
        else:
            # Some imported sanity bases expose existing output-request objects but
            # not the creation helper. Do not block CAE/INP generation here; the
            # post-generation INP audit verifies whether required output variables
            # are visible and will fail closed if they are absent.
            status["warnings"].append("model has no FieldOutputRequest creator; relying on existing base output requests")
    return status


def write_case_files(model, row: dict[str, object]) -> None:  # noqa: ANN001
    from abaqus import mdb  # type: ignore
    from abaqusConstants import OFF  # type: ignore

    case_dir = Path(str(row["case_dir"]))
    case_dir.mkdir(parents=True, exist_ok=True)
    helpers.validate_no_solver_outputs(case_dir)
    os.chdir(str(case_dir))
    job_name = str(row["job_name"])
    if job_name in mdb.jobs:
        del mdb.jobs[job_name]
    mdb.Job(name=job_name, model=model.name)
    mdb.saveAs(pathName=str(row["cae_path"]))
    mdb.jobs[job_name].writeInput(consistencyChecking=OFF)
    if not Path(str(row["inp_path"])).exists() or Path(str(row["inp_path"])).stat().st_size <= 0:
        raise StageEGenerationError("INP was not written: {}".format(row["inp_path"]))
    Path(str(row["jnl_path"])).write_text(
        "Stage E PPO batch32 generation journal placeholder for {}.\nCAE/INP generated only; no solver, datacheck, abqjobpilot, enqueue, ODB, or teacher validation run.\n".format(row["strategy_name"]),
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
            "cae_path": row["cae_path"],
            "inp_path": row["inp_path"],
        },
        "candidate_source": row["candidate_source"],
        "teacher_validated": False,
        "abaqus_validated": False,
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
    Path(str(row["generation_log_path"])).write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
            "scan_order_json": row["scan_order_json"],
            "cae_path": row["cae_path"],
            "inp_path": row["inp_path"],
            "jnl_path": row["jnl_path"],
            "generation_status": "GENERATED",
            "inp_check_status": "PENDING",
            "teacher_validated": "False",
            "solver_status": "NOT_SUBMITTED",
            "notes": "Stage E PPO batch32 CAE/INP generation only; not teacher-validated.",
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
        write_case_files(model, row)
        write_generation_log(row, records, deactivations, final_record, output_status)
        generated.append({"n": n, "strategy_name": row["strategy_name"], "cae_path": row["cae_path"], "inp_path": row["inp_path"]})
    write_generation_manifest(rows)
    summary = {
        "verdict": "PASS_STAGEE_PPO_BATCH32_GENERATION_COMPLETE",
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
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FAILURE_LOG.parent.mkdir(parents=True, exist_ok=True)
    try:
        rows = load_rows()
        validate_paths(rows)
        generate_all(rows)
        print("PASS_STAGEE_PPO_BATCH32_GENERATION_COMPLETE")
        print("summary={}".format(GENERATION_SUMMARY))
        return 0
    except Exception as exc:
        FAILURE_LOG.write_text(json.dumps({"error": str(exc), "traceback": traceback.format_exc()}, indent=2), encoding="utf-8")
        print("FAIL_STAGEE_PPO_BATCH32_GENERATION")
        print(str(exc))
        print("failure_log={}".format(FAILURE_LOG))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
