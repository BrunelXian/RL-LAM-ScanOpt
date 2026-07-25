"""Patch run20 N40_B01 cooling step initial increments to 0.001.

This modifies exactly one generated run20 batch28 case:
S3R19B28_N40_B01_surrogate_top.

It sets every step_cool_XX initialInc to 0.001, saves the existing CAE, and
rewrites the existing INP. It does not submit jobs, run datacheck, execute
abqjobpilot/enqueue, open ODB, or touch any other case.
"""

from __future__ import annotations

import json
import os
import re
import traceback
from pathlib import Path


PROJECT_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage3_run_20_batch28_cae_inp_generation"
CASE_DIR = (
    PROJECT_ROOT
    / "cae_model"
    / "stage3_run19_batch28_combined80_surrogate_screened_v01"
    / "N40S3R19B28_N40_B01_surrogate_top"
)
JOB_NAME = "J2D_S3R19B28_N40_B01_surrogate_top"
CAE_PATH = CASE_DIR / "{}.cae".format(JOB_NAME)
INP_PATH = CASE_DIR / "{}.inp".format(JOB_NAME)
SUMMARY_PATH = OUTPUT_DIR / "N40_B01_cooling_initial_inc_patch_summary.json"
FAILURE_PATH = OUTPUT_DIR / "N40_B01_cooling_initial_inc_patch_failure.json"

TARGET_INITIAL_INC = 0.001
EXPECTED_COOL_STEP_COUNT = 40


class PatchError(RuntimeError):
    pass


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        if not CAE_PATH.exists():
            raise PatchError("missing target CAE: {}".format(CAE_PATH))
        if not INP_PATH.exists():
            raise PatchError("missing target INP: {}".format(INP_PATH))
        if list(CASE_DIR.glob("*.lck")):
            raise PatchError("target case has .lck; refusing to patch")

        from abaqus import mdb, openMdb  # type: ignore
        from abaqusConstants import OFF  # type: ignore

        openMdb(pathName=str(CAE_PATH))
        if "Model-1" not in mdb.models:
            raise PatchError("Model-1 not found in target CAE")
        model = mdb.models["Model-1"]
        cool_names = sorted(
            name for name in model.steps.keys() if re.fullmatch(r"step_cool_\d+", name)
        )
        if len(cool_names) != EXPECTED_COOL_STEP_COUNT:
            raise PatchError(
                "cool step count {} != {}".format(len(cool_names), EXPECTED_COOL_STEP_COUNT)
            )

        records = []
        for name in cool_names:
            step = model.steps[name]
            before = getattr(step, "initialInc", None)
            step.setValues(initialInc=TARGET_INITIAL_INC)
            after = getattr(step, "initialInc", None)
            if float(after) != TARGET_INITIAL_INC:
                raise PatchError("{} initialInc verification failed: {}".format(name, after))
            records.append({"step": name, "initialInc_before": before, "initialInc_after": after})

        final_step = model.steps["step_final_cooling"] if "step_final_cooling" in model.steps else None
        final_record = {}
        if final_step is not None:
            final_record = {
                "timePeriod": getattr(final_step, "timePeriod", None),
                "initialInc": getattr(final_step, "initialInc", None),
                "maxInc": getattr(final_step, "maxInc", None),
            }

        os.chdir(str(CASE_DIR))
        if JOB_NAME in mdb.jobs:
            del mdb.jobs[JOB_NAME]
        mdb.Job(name=JOB_NAME, model=model.name)
        mdb.saveAs(pathName=str(CAE_PATH))
        mdb.jobs[JOB_NAME].writeInput(consistencyChecking=OFF)
        if not INP_PATH.exists() or INP_PATH.stat().st_size <= 0:
            raise PatchError("INP was not rewritten")

        summary = {
            "status": "PATCHED",
            "case_dir": str(CASE_DIR),
            "job_name": JOB_NAME,
            "cae_path": str(CAE_PATH),
            "inp_path": str(INP_PATH),
            "target_initialInc": TARGET_INITIAL_INC,
            "patched_cool_step_count": len(records),
            "records": records,
            "final_cooling_unchanged_observed_values": final_record,
            "solver_submitted": False,
            "datacheck_run": False,
            "abqjobpilot_run": False,
            "enqueue_run": False,
            "odb_opened": False,
        }
        SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("PATCHED_RUN20_N40_B01_COOLING_INITIAL_INC")
        print("summary={}".format(SUMMARY_PATH))
        return 0
    except Exception as exc:
        FAILURE_PATH.write_text(
            json.dumps({"error": str(exc), "traceback": traceback.format_exc()}, indent=2),
            encoding="utf-8",
        )
        print("FAIL_RUN20_N40_B01_COOLING_INITIAL_INC_PATCH")
        print(str(exc))
        print("failure={}".format(FAILURE_PATH))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
