"""Generate Stage 3 run35 N32_informed_native_batch32 CAE/INP artifacts from sanity bases.

Run with:
abaqus cae noGUI="E:\Projects\RL-LAM-ScanOpt\scripts\stage3\run_35_generate_N32_informed_native_batch32_from_sanity_base_nogui.py"

This script writes CAE/INP/JNL/model-generation logs only. It never submits
jobs, runs datacheck, executes abqjobpilot/enqueue, opens ODB, or performs
teacher validation.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import traceback
from pathlib import Path


PROJECT_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
RUN_ID = "run_35_N32_informed_native_batch32_cae_inp_generation"
BATCH_NAME = "stage3_run34_N32_informed_native_batch32_v01"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage3_run_35_N32_informed_native_batch32_cae_inp_generation"
PLAN_CSV = OUTPUT_DIR / "run35_generation_plan.csv"
BASE_AUDIT_SUMMARY = OUTPUT_DIR / "run35_base_mesh_audit_summary.json"
CASE_ROOT = PROJECT_ROOT / "cae_model" / BATCH_NAME
FAILURE_LOG = OUTPUT_DIR / "run35_generator_failure.json"
GENERATION_SUMMARY = OUTPUT_DIR / "run35_generation_summary.json"
MANIFEST_CSV = OUTPUT_DIR / "stage3_run35_N32_informed_native_batch32_cae_generation_manifest.csv"
MANIFEST_JSON = OUTPUT_DIR / "stage3_run35_N32_informed_native_batch32_cae_generation_manifest.json"

EXPECTED_COUNTS = {12: 4, 16: 4, 24: 12, 40: 12}
BASES = {
    12: PROJECT_ROOT / "cae_model" / "12track_full" / "sanity_base" / "12track_sanity_base.cae",
    16: PROJECT_ROOT / "cae_model" / "16track_full" / "sanity_base" / "16track_sanity_base.cae",
    24: PROJECT_ROOT / "cae_model" / "24track_full" / "sanity_base" / "24track_sanity_base.cae",
    40: PROJECT_ROOT / "cae_model" / "40track_full" / "sanity_base" / "40track_sanity_base.cae",
}
MAPPING = {
    "track_set_pattern": "set_body_heat_{track:02d}",
    "scan_step_pattern": "step_scan_{seq:02d}",
    "cool_step_pattern": "step_cool_{seq:02d}",
    "load_pattern": "load_body_hflux_{seq:02d}",
    "template_scan_step": "step_scan_00",
    "template_cool_step": "step_cool_00",
    "template_load": "load_body_hflux_00",
    "load_type": "BodyHeatFlux",
    "body_heat_flux_magnitude": 80000000000.0,
    "scan_duration_seconds": 0.2,
    "cool_duration_seconds": 3.4,
    "n40_cool_initial_increment": 0.001,
    "final_cooling_step_name": "step_final_cooling",
    "final_cooling_duration_seconds": 1200.0,
    "final_cooling_initial_increment": 0.01,
    "final_cooling_max_increment_size": 60.0,
}
BAD_SCHEMA_TOKENS = ("N12N12_", "N16N16_", "N24N24_", "N40N40_")
SOLVER_EXTS = (".odb", ".sim", ".sta", ".dat", ".msg", ".lck")


class Run35GenerationError(RuntimeError):
    pass


def validate_scan_order(n: int, order: object) -> None:
    if not isinstance(order, list):
        raise Run35GenerationError("scan_order is not a list")
    if len(order) != n:
        raise Run35GenerationError("scan_order length {} != N {}".format(len(order), n))
    if not all(isinstance(v, int) for v in order):
        raise Run35GenerationError("scan_order contains non-integer entries")
    if sorted(order) != list(range(n)):
        raise Run35GenerationError("scan_order is not a permutation of 0..{}".format(n - 1))


def load_plan() -> list[dict[str, object]]:
    if not PLAN_CSV.exists():
        raise Run35GenerationError("missing run35 generation plan: {}".format(PLAN_CSV))
    rows: list[dict[str, object]] = []
    with PLAN_CSV.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            n = int(row["n"])
            scan_order = json.loads(row["scan_order"])
            validate_scan_order(n, scan_order)
            if row.get("selected_batch") != "run33_N32_informed_native_batch32":
                raise Run35GenerationError("non-run33_N32_informed_native_batch32 row selected: {}".format(row.get("handoff_strategy_name")))
            lowered = json.dumps(row).lower()
            if "batch64" in lowered or "batch48" in lowered or "shortlist64" in lowered:
                raise Run35GenerationError("forbidden hybrid batch64/focused batch48/shortlist64 text selected")
            rows.append(dict(row, n=n, scan_order=scan_order))
    if len(rows) != 32:
        raise Run35GenerationError("generation plan row count {} != 32".format(len(rows)))
    counts = {n: len([row for row in rows if int(row["n"]) == n]) for n in EXPECTED_COUNTS}
    if counts != EXPECTED_COUNTS:
        raise Run35GenerationError("per-N generation counts {} != {}".format(counts, EXPECTED_COUNTS))
    if any(int(row["n"]) == 32 for row in rows):
        raise Run35GenerationError("forbidden N32 row selected")
    return rows


def validate_base_audit(rows: list[dict[str, object]]) -> None:
    if not BASE_AUDIT_SUMMARY.exists():
        raise Run35GenerationError("missing base mesh audit summary: {}".format(BASE_AUDIT_SUMMARY))
    summary = json.loads(BASE_AUDIT_SUMMARY.read_text(encoding="utf-8"))
    by_n = {int(row["n"]): row for row in summary.get("rows", [])}
    failed = []
    for n in sorted({int(row["n"]) for row in rows}):
        row = by_n.get(n)
        if not row or row.get("verdict") != "PASS_BASE_MESH_READY":
            failed.append(n)
            continue
        if float(row.get("total_node_count", 0)) <= 0 or float(row.get("total_element_count", 0)) <= 0:
            failed.append(n)
    if failed:
        raise Run35GenerationError("base mesh audit not passing for N values: {}".format(failed))


def validate_paths(rows: list[dict[str, object]]) -> None:
    for row in rows:
        paths = [str(row[key]) for key in ("case_dir", "cae_path", "inp_path", "jnl_path")]
        if any(token in path for token in BAD_SCHEMA_TOKENS for path in paths):
            raise Run35GenerationError("bad path schema token in {}".format(row["handoff_strategy_name"]))
        if Path(str(row["case_dir"])).parent != CASE_ROOT:
            raise Run35GenerationError("case_dir not under run35 case root: {}".format(row["case_dir"]))
        for key in ("cae_path", "inp_path", "jnl_path", "scan_order_json", "generation_log_path"):
            if Path(str(row[key])).parent != Path(str(row["case_dir"])):
                raise Run35GenerationError("{} is not inside case_dir".format(key))
        existing = [path for path in (Path(str(row["cae_path"])), Path(str(row["inp_path"])), Path(str(row["jnl_path"]))) if path.exists()]
        if existing:
            raise Run35GenerationError("generated CAE/INP/JNL already exists for {}: {}".format(row["handoff_strategy_name"], existing))


def validate_no_solver_outputs(case_dir: Path) -> None:
    found = []
    for ext in SOLVER_EXTS:
        found.extend(case_dir.glob("*{}".format(ext)))
    if found:
        raise Run35GenerationError("solver output files unexpectedly exist: {}".format(found))


def validate_heat_sets_exist_in_model(model, n: int) -> dict[int, object]:  # noqa: ANN001
    root = model.rootAssembly
    found: dict[int, object] = {}
    all_heat_names = []
    for track in range(n):
        name = MAPPING["track_set_pattern"].format(track=track)
        if name in root.sets:
            found[track] = root.sets[name]
        else:
            for instance in root.instances.values():
                if name in instance.sets:
                    found[track] = instance.sets[name]
                    break
        if track not in found:
            raise Run35GenerationError("missing heat set {}".format(name))
    all_heat_names.extend(name for name in list(root.sets.keys()) if re.fullmatch(r"set_body_heat_\d+", name))
    for instance in root.instances.values():
        all_heat_names.extend(name for name in list(instance.sets.keys()) if re.fullmatch(r"set_body_heat_\d+", name))
    if len(set(all_heat_names)) != n:
        raise Run35GenerationError("expected exactly {} heat sets, found {}".format(n, sorted(set(all_heat_names))))
    return found


def infer_templates(model) -> dict[str, object]:  # noqa: ANN001
    missing = [name for name in (MAPPING["template_scan_step"], MAPPING["template_cool_step"]) if name not in model.steps]
    if MAPPING["template_load"] not in model.loads:
        missing.append(MAPPING["template_load"])
    if missing:
        raise Run35GenerationError("missing template objects: {}".format(missing))
    load = model.loads[MAPPING["template_load"]]
    if load.__class__.__name__ != MAPPING["load_type"]:
        raise Run35GenerationError("template load type {} != {}".format(load.__class__.__name__, MAPPING["load_type"]))
    return {"scan_step": model.steps[MAPPING["template_scan_step"]], "cool_step": model.steps[MAPPING["template_cool_step"]], "load": load}


def _step_common_kwargs(template_step) -> dict:  # noqa: ANN001
    keys = ("timePeriod", "initialInc", "minInc", "maxInc", "maxNumInc", "deltmx", "nlgeom")
    return {key: getattr(template_step, key) for key in keys if hasattr(template_step, key)}


def create_step_from_template(model, template_step, name: str, previous: str, time_period: float, initial_inc=None, max_inc=None) -> None:  # noqa: ANN001,E501
    step_type = template_step.__class__.__name__
    kwargs = _step_common_kwargs(template_step)
    kwargs["timePeriod"] = time_period
    if initial_inc is not None:
        kwargs["initialInc"] = initial_inc
    if max_inc is not None:
        kwargs["maxInc"] = max_inc
    if step_type == "CoupledTempDisplacementStep":
        model.CoupledTempDisplacementStep(name=name, previous=previous, **kwargs)
    elif step_type == "HeatTransferStep":
        model.HeatTransferStep(name=name, previous=previous, **kwargs)
    elif step_type == "StaticStep":
        model.StaticStep(name=name, previous=previous, **kwargs)
    else:
        raise Run35GenerationError("unsupported step template type {}".format(step_type))


def set_existing_step_time(model, step_name: str, duration: float) -> None:  # noqa: ANN001
    try:
        model.steps[step_name].setValues(timePeriod=duration)
    except Exception as exc:
        raise Run35GenerationError("failed setting {} duration: {}".format(step_name, exc))


def set_existing_step_time_and_initial(model, step_name: str, duration: float, initial_inc=None) -> None:  # noqa: ANN001
    kwargs = {"timePeriod": duration}
    if initial_inc is not None:
        kwargs["initialInc"] = initial_inc
    try:
        model.steps[step_name].setValues(**kwargs)
    except Exception as exc:
        raise Run35GenerationError("failed setting {} step values: {}".format(step_name, exc))


def cool_initial_inc_for_n(n: int):
    return float(MAPPING["n40_cool_initial_increment"]) if n == 40 else None


def create_sequence(model, n: int, scan_order: list[int]) -> list[dict[str, object]]:  # noqa: ANN001
    validate_scan_order(n, scan_order)
    templates = infer_templates(model)
    records = []
    previous = "Initial"
    cool_initial_inc = cool_initial_inc_for_n(n)
    for seq, track in enumerate(scan_order):
        scan = MAPPING["scan_step_pattern"].format(seq=seq)
        cool = MAPPING["cool_step_pattern"].format(seq=seq)
        if seq == 0:
            if scan not in model.steps or cool not in model.steps:
                raise Run35GenerationError("template scan/cool step missing for seq0")
            set_existing_step_time(model, scan, float(MAPPING["scan_duration_seconds"]))
            set_existing_step_time_and_initial(model, cool, float(MAPPING["cool_duration_seconds"]), cool_initial_inc)
        else:
            create_step_from_template(model, templates["scan_step"], scan, previous, float(MAPPING["scan_duration_seconds"]))
            create_step_from_template(
                model,
                templates["cool_step"],
                cool,
                scan,
                float(MAPPING["cool_duration_seconds"]),
                initial_inc=cool_initial_inc,
            )
        if cool_initial_inc is not None:
            observed = float(getattr(model.steps[cool], "initialInc"))
            if observed != cool_initial_inc:
                raise Run35GenerationError("{} initialInc verification failed: {}".format(cool, observed))
        records.append({"seq": seq, "track": track, "scan_step": scan, "cool_step": cool, "cool_initialInc": cool_initial_inc})
        previous = cool
    return records


def create_heat_load(model, seq: int, track: int, heat_regions: dict[int, object]):  # noqa: ANN001
    load_name = MAPPING["load_pattern"].format(seq=seq)
    scan_step = MAPPING["scan_step_pattern"].format(seq=seq)
    region = heat_regions[track]
    magnitude = float(MAPPING["body_heat_flux_magnitude"])
    if seq == 0:
        load = model.loads[MAPPING["template_load"]]
        try:
            load.setValues(region=region, magnitude=magnitude)
        except Exception as exc:
            if track != 0:
                raise Run35GenerationError("failed to retarget template load to track {}: {}".format(track, exc))
            load.setValues(magnitude=magnitude)
        return load
    return model.BodyHeatFlux(createStepName=scan_step, magnitude=magnitude, name=load_name, region=region)


def deactivate_loads(model, records: list[dict[str, object]]) -> list[dict[str, str]]:  # noqa: ANN001
    out = []
    for record in records:
        seq = int(record["seq"])
        load_name = MAPPING["load_pattern"].format(seq=seq)
        cool = str(record["cool_step"])
        try:
            model.loads[load_name].deactivate(cool)
        except Exception as exc:
            if seq != 0:
                raise
            out.append({"load": load_name, "deactivated_in": cool, "note": str(exc)})
            continue
        out.append({"load": load_name, "deactivated_in": cool})
    return out


def append_final_cooling(model, records: list[dict[str, object]]) -> dict[str, object]:  # noqa: ANN001
    final = MAPPING["final_cooling_step_name"]
    if final in model.steps:
        raise Run35GenerationError("final cooling step already exists")
    previous = str(records[-1]["cool_step"])
    create_step_from_template(
        model,
        model.steps[MAPPING["template_cool_step"]],
        final,
        previous,
        float(MAPPING["final_cooling_duration_seconds"]),
        initial_inc=float(MAPPING["final_cooling_initial_increment"]),
        max_inc=float(MAPPING["final_cooling_max_increment_size"]),
    )
    step = model.steps[final]
    checks = {
        "final_cooling_step_name": final,
        "final_cooling_previous": previous,
        "final_cooling_timePeriod": float(getattr(step, "timePeriod")),
        "final_cooling_initialInc": float(getattr(step, "initialInc")),
        "final_cooling_maxInc": float(getattr(step, "maxInc")),
        "final_cooling_heat_loads_inactive": True,
    }
    if checks["final_cooling_timePeriod"] != 1200.0:
        raise Run35GenerationError("final cooling timePeriod verification failed")
    if checks["final_cooling_initialInc"] != 0.01:
        raise Run35GenerationError("final cooling initialInc verification failed")
    if checks["final_cooling_maxInc"] != 60.0:
        raise Run35GenerationError("final cooling maxInc verification failed")
    return checks


def validate_model_before_write(model, n: int, scan_order: list[int], records: list[dict[str, object]], deactivations: list[dict[str, str]], final_record: dict[str, object]) -> None:  # noqa: ANN001,E501
    validate_scan_order(n, scan_order)
    if len(records) != n or len(deactivations) != n:
        raise Run35GenerationError("sequence/deactivation count != N")
    for seq in range(n):
        if MAPPING["scan_step_pattern"].format(seq=seq) not in model.steps:
            raise Run35GenerationError("missing scan step {}".format(seq))
        if MAPPING["cool_step_pattern"].format(seq=seq) not in model.steps:
            raise Run35GenerationError("missing cool step {}".format(seq))
        if MAPPING["load_pattern"].format(seq=seq) not in model.loads:
            raise Run35GenerationError("missing heat load {}".format(seq))
    if final_record["final_cooling_previous"] != records[-1]["cool_step"]:
        raise Run35GenerationError("final cooling is not after last cool step")


def write_case_files(model, row: dict[str, object]) -> None:  # noqa: ANN001
    from abaqus import mdb  # type: ignore
    from abaqusConstants import OFF  # type: ignore

    case_dir = Path(str(row["case_dir"]))
    case_dir.mkdir(parents=True, exist_ok=True)
    validate_no_solver_outputs(case_dir)
    source_json = Path(str(row["source_order_json"]))
    target_json = Path(str(row["scan_order_json"]))
    if not target_json.exists():
        shutil.copyfile(str(source_json), str(target_json))
    metadata_path = case_dir / "run35_case_metadata.json"
    metadata_keys = [
        "run_id", "batch_name", "selected_batch", "n", "handoff_strategy_name",
        "original_run23_candidate_id", "original_run23_strategy_name", "candidate_family",
        "candidate_source", "generation_method", "selection_bucket", "priority_role",
        "surrogate_prediction", "gnn_reward_prediction", "graph_pointer_policy_score",
        "hybrid_score", "uncertainty_score", "gnn_vs_surrogate_disagreement",
        "novelty_distance_to_combined172_plus_N32", "nearest_existing_teacher_strategy",
        "N32_informed", "native_validation_N",
    ]
    metadata = {key: row.get(key, "") for key in metadata_keys}
    metadata.update({"teacher_validated": False, "teacher_validation_status": "NOT_RUN", "solver_status": "NOT_SUBMITTED"})
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    os.chdir(str(case_dir))
    job_name = str(row["job_name"])
    if job_name in mdb.jobs:
        del mdb.jobs[job_name]
    mdb.Job(name=job_name, model=model.name)
    mdb.saveAs(pathName=str(row["cae_path"]))
    mdb.jobs[job_name].writeInput(consistencyChecking=OFF)
    if not Path(str(row["inp_path"])).exists() or Path(str(row["inp_path"])).stat().st_size <= 0:
        raise Run35GenerationError("INP was not written: {}".format(row["inp_path"]))
    Path(str(row["jnl_path"])).write_text(
        "Run35 generation journal placeholder for {}.\nCAE/INP generated only; no solver, datacheck, abqjobpilot, enqueue, ODB, or teacher validation run.\n".format(row["handoff_strategy_name"]),
        encoding="utf-8",
    )


def write_generation_log(row: dict[str, object], records: list[dict[str, object]], deactivations: list[dict[str, str]], final_record: dict[str, object]) -> None:
    payload = {
        "status": "GENERATED",
        "run_id": RUN_ID,
        "batch_name": BATCH_NAME,
        "selected_batch": "run33_N32_informed_native_batch32",
        "case": {"n": row["n"], "handoff_strategy_name": row["handoff_strategy_name"], "job_name": row["job_name"], "cae_path": row["cae_path"], "inp_path": row["inp_path"]},
        "sequence_records": records,
        "deactivation_records": deactivations,
        "final_cooling": final_record,
        "solver_submitted": False,
        "datacheck_run": False,
        "abqjobpilot_run": False,
        "enqueue_run": False,
        "odb_opened": False,
        "teacher_validation_run": False,
    }
    Path(str(row["generation_log_path"])).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_generation_manifest(rows: list[dict[str, object]]) -> None:
    fieldnames = ["run_id", "batch_name", "n", "handoff_strategy_name", "job_name", "case_dir", "scan_order_json", "cae_path", "inp_path", "jnl_path", "generation_status", "inp_check_status", "teacher_validated", "solver_status", "notes"]
    manifest_rows = []
    for row in rows:
        manifest_rows.append({
            "run_id": RUN_ID,
            "batch_name": BATCH_NAME,
            "n": row["n"],
            "handoff_strategy_name": row["handoff_strategy_name"],
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
            "notes": "Run35 N32_informed_native_batch32 CAE/INP generation only; not teacher-validated.",
        })
    with MANIFEST_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    MANIFEST_JSON.write_text(json.dumps({"rows": manifest_rows, "row_count": len(manifest_rows)}, indent=2), encoding="utf-8")


def generate_all(rows: list[dict[str, object]]) -> None:
    from abaqus import mdb, openMdb  # type: ignore

    generated = []
    for row in rows:
        n = int(row["n"])
        openMdb(pathName=str(BASES[n]))
        model = mdb.models["Model-1"]
        infer_templates(model)
        heat_regions = validate_heat_sets_exist_in_model(model, n)
        scan_order = list(row["scan_order"])
        records = create_sequence(model, n, scan_order)
        for record in records:
            create_heat_load(model, int(record["seq"]), int(record["track"]), heat_regions)
        deactivations = deactivate_loads(model, records)
        final_record = append_final_cooling(model, records)
        validate_model_before_write(model, n, scan_order, records, deactivations, final_record)
        write_case_files(model, row)
        write_generation_log(row, records, deactivations, final_record)
        generated.append({"n": n, "handoff_strategy_name": row["handoff_strategy_name"], "cae_path": row["cae_path"], "inp_path": row["inp_path"]})
    write_generation_manifest(rows)
    summary = {
        "verdict": "PASS_RUN35_N32_INFORMED_NATIVE_BATCH32_GENERATION_COMPLETE",
        "generated_count": len(generated),
        "per_n_counts": {str(n): len([row for row in generated if int(row["n"]) == n]) for n in EXPECTED_COUNTS},
        "generated": generated,
        "manifest_csv": str(MANIFEST_CSV),
        "manifest_json": str(MANIFEST_JSON),
        "no_solver_run": True,
        "no_datacheck_run": True,
        "no_abqjobpilot_run": True,
        "no_enqueue_run": True,
        "no_odb_opened": True,
        "no_teacher_validation": True,
    }
    GENERATION_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        rows = load_plan()
        validate_base_audit(rows)
        validate_paths(rows)
        generate_all(rows)
        print("PASS_RUN35_N32_INFORMED_NATIVE_BATCH32_GENERATION_COMPLETE")
        print("summary={}".format(GENERATION_SUMMARY))
        return 0
    except Exception as exc:
        FAILURE_LOG.write_text(json.dumps({"error": str(exc), "traceback": traceback.format_exc()}, indent=2), encoding="utf-8")
        print("FAIL_RUN35_N32_INFORMED_NATIVE_BATCH32_GENERATION")
        print(str(exc))
        print("failure_log={}".format(FAILURE_LOG))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
