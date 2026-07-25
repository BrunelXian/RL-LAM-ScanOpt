"""Patch run25 shortlist64 N40 B02-B05 cooling step initial increments.

This modifies exactly four generated run25 shortlist64 cases:

- S3R24L64_N40_B02_top_region
- S3R24L64_N40_B03_top_region
- S3R24L64_N40_B04_top_region
- S3R24L64_N40_B05_top_region

It sets every step_cool_XX initialInc to 0.01, saves each existing CAE, and
rewrites each existing INP. It does not submit jobs, run datacheck, execute
abqjobpilot/enqueue, open ODB, or touch any other case.
"""

from __future__ import annotations

import json
import os
import re
import traceback
from pathlib import Path


PROJECT_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage3_run_25_shortlist64_active_learning_cae_inp_generation"
CASE_ROOT = PROJECT_ROOT / "cae_model" / "stage3_run24_shortlist64_active_learning_calibration_v01"
SUMMARY_PATH = OUTPUT_DIR / "run25_N40_B02_B03_B04_B05_cool_initialInc_patch_summary.json"
FAILURE_PATH = OUTPUT_DIR / "run25_N40_B02_B03_B04_B05_cool_initialInc_patch_failure.json"

TARGET_INITIAL_INC = 0.01
EXPECTED_COOL_STEP_COUNT = 40
TARGETS = [
    "S3R24L64_N40_B02_top_region",
    "S3R24L64_N40_B03_top_region",
    "S3R24L64_N40_B04_top_region",
    "S3R24L64_N40_B05_top_region",
]
SOLVER_EXTS = (".odb", ".sim", ".sta", ".dat", ".msg", ".lck")


class PatchError(RuntimeError):
    pass


def fail_if_solver_outputs(case_dir: Path) -> None:
    found = []
    for ext in SOLVER_EXTS:
        found.extend(case_dir.glob("*{}".format(ext)))
    if found:
        raise PatchError("target case still has solver outputs: {}".format(found))


def patch_one(strategy: str) -> dict[str, object]:
    from abaqus import mdb, openMdb  # type: ignore
    from abaqusConstants import OFF  # type: ignore

    case_dir = CASE_ROOT / ("N40{}".format(strategy))
    job_name = "J2D_{}".format(strategy)
    cae_path = case_dir / "{}.cae".format(job_name)
    inp_path = case_dir / "{}.inp".format(job_name)
    jnl_path = case_dir / "{}.jnl".format(job_name)
    gen_log_path = case_dir / "{}.generation_log.json".format(job_name)

    if not case_dir.exists():
        raise PatchError("missing target case dir: {}".format(case_dir))
    if not cae_path.exists():
        raise PatchError("missing target CAE: {}".format(cae_path))
    if not inp_path.exists():
        raise PatchError("missing target INP: {}".format(inp_path))
    fail_if_solver_outputs(case_dir)

    openMdb(pathName=str(cae_path))
    if "Model-1" not in mdb.models:
        raise PatchError("Model-1 not found in target CAE: {}".format(cae_path))
    model = mdb.models["Model-1"]
    cool_names = sorted(
        name for name in model.steps.keys() if re.fullmatch(r"step_cool_\d+", name)
    )
    if len(cool_names) != EXPECTED_COOL_STEP_COUNT:
        raise PatchError(
            "{} cool step count {} != {}".format(strategy, len(cool_names), EXPECTED_COOL_STEP_COUNT)
        )

    records = []
    for name in cool_names:
        step = model.steps[name]
        before = getattr(step, "initialInc", None)
        step.setValues(initialInc=TARGET_INITIAL_INC)
        after = getattr(step, "initialInc", None)
        if float(after) != TARGET_INITIAL_INC:
            raise PatchError("{} {} initialInc verification failed: {}".format(strategy, name, after))
        records.append({"step": name, "initialInc_before": before, "initialInc_after": after})

    final_step = model.steps["step_final_cooling"] if "step_final_cooling" in model.steps else None
    if final_step is None:
        raise PatchError("{} missing step_final_cooling".format(strategy))
    final_record = {
        "timePeriod": getattr(final_step, "timePeriod", None),
        "initialInc": getattr(final_step, "initialInc", None),
        "maxInc": getattr(final_step, "maxInc", None),
    }
    if float(final_record["timePeriod"]) != 1200.0:
        raise PatchError("{} final cooling timePeriod changed unexpectedly".format(strategy))
    if float(final_record["initialInc"]) != 0.01:
        raise PatchError("{} final cooling initialInc is not 0.01".format(strategy))
    if float(final_record["maxInc"]) != 60.0:
        raise PatchError("{} final cooling maxInc is not 60.0".format(strategy))

    os.chdir(str(case_dir))
    if job_name in mdb.jobs:
        del mdb.jobs[job_name]
    mdb.Job(name=job_name, model=model.name)
    mdb.saveAs(pathName=str(cae_path))
    mdb.jobs[job_name].writeInput(consistencyChecking=OFF)
    if not inp_path.exists() or inp_path.stat().st_size <= 0:
        raise PatchError("{} INP was not rewritten".format(strategy))

    jnl_path.write_text(
        "Run25 patch journal placeholder for {}.\n"
        "All step_cool_XX initialInc values set to 0.01.\n"
        "CAE/INP rewritten only; no solver, datacheck, abqjobpilot, enqueue, ODB, or teacher validation run.\n".format(
            strategy
        ),
        encoding="utf-8",
    )

    generation_log = {}
    if gen_log_path.exists():
        try:
            generation_log = json.loads(gen_log_path.read_text(encoding="utf-8"))
        except Exception:
            generation_log = {}
    generation_log["cool_initialInc_patch"] = {
        "status": "PATCHED",
        "target_initialInc": TARGET_INITIAL_INC,
        "patched_cool_step_count": len(records),
        "patched_steps": records,
        "final_cooling_unchanged_observed_values": final_record,
        "solver_submitted": False,
        "datacheck_run": False,
        "abqjobpilot_run": False,
        "enqueue_run": False,
        "odb_opened": False,
    }
    gen_log_path.write_text(json.dumps(generation_log, indent=2), encoding="utf-8")

    return {
        "strategy": strategy,
        "case_dir": str(case_dir),
        "job_name": job_name,
        "cae_path": str(cae_path),
        "inp_path": str(inp_path),
        "jnl_path": str(jnl_path),
        "generation_log_path": str(gen_log_path),
        "target_initialInc": TARGET_INITIAL_INC,
        "patched_cool_step_count": len(records),
        "records": records,
        "final_cooling_unchanged_observed_values": final_record,
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        patched = [patch_one(strategy) for strategy in TARGETS]
        summary = {
            "verdict": "PASS_RUN25_N40_B02_B03_B04_B05_COOL_INITIAL_INC_PATCHED",
            "target_initialInc": TARGET_INITIAL_INC,
            "target_count": len(TARGETS),
            "patched": patched,
            "solver_submitted": False,
            "datacheck_run": False,
            "abqjobpilot_run": False,
            "enqueue_run": False,
            "odb_opened": False,
        }
        SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("PASS_RUN25_N40_B02_B03_B04_B05_COOL_INITIAL_INC_PATCHED")
        print("summary={}".format(SUMMARY_PATH))
        return 0
    except Exception as exc:
        FAILURE_PATH.write_text(
            json.dumps({"error": str(exc), "traceback": traceback.format_exc()}, indent=2),
            encoding="utf-8",
        )
        print("FAIL_RUN25_N40_B02_B03_B04_B05_COOL_INITIAL_INC_PATCH")
        print(str(exc))
        print("failure={}".format(FAILURE_PATH))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
