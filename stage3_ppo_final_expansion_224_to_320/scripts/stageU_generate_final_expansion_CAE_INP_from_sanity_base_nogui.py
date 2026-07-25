"""Generate Stage U final expansion 224 CAE/INP files.
Run with Abaqus CAE noGUI only. No solver, datacheck, enqueue, or ODB.
"""
from __future__ import annotations
import csv, json, os, sys, traceback
from collections import Counter
from pathlib import Path

PROJECT_ROOT=Path(r"E:\Projects\RL-LAM-ScanOpt")
sys.path.insert(0,str(PROJECT_ROOT/"scripts"/"stage3"))
import run_75_generate_final_smallN_diagnostic_batch32_from_sanity_base_nogui as helpers  # noqa: E402

NS="stage3_ppo_final_expansion_224_to_320"
RUN_ID="stageU_final_expansion_CAE_INP_handoff"
BATCH_NAME="stage3_ppo_final_expansion_224_to_320_batch224_v01"
OUT=PROJECT_ROOT/"outputs"/NS/"stageU_CAE_INP_handoff"
MANIFEST_CSV=OUT/"manifest"/"stageU_final_expansion_case_manifest.csv"
GEN_SUMMARY=OUT/"manifest"/"stageU_final_expansion_generation_summary.json"
GEN_CSV=OUT/"manifest"/"stageU_final_expansion_CAE_generation_manifest.csv"
GEN_JSON=OUT/"manifest"/"stageU_final_expansion_CAE_generation_manifest.json"
FAILURE_LOG=OUT/"reports"/"stageU_final_expansion_generator_failure.json"
CASE_ROOT=PROJECT_ROOT/"cae_model"/BATCH_NAME
BASES={12:PROJECT_ROOT/"cae_model"/"12track_full"/"sanity_base"/"12track_sanity_base.cae",16:PROJECT_ROOT/"cae_model"/"16track_full"/"sanity_base"/"16track_sanity_base.cae",24:PROJECT_ROOT/"cae_model"/"24track_full"/"sanity_base"/"24track_sanity_base.cae",40:PROJECT_ROOT/"cae_model"/"40track_full"/"sanity_base"/"40track_sanity_base.cae"}
EXPECTED_N={12:32,16:32,24:80,40:80}
EXPECTED_BATCH={"final_expansion_batch01":32,"final_expansion_batch02":32,"final_expansion_batch03":32,"final_expansion_batch04":32,"final_expansion_batch05":32,"final_expansion_batch06":32,"final_expansion_batch07":32}
OUTPUT_VARIABLES=("U","PEEQ","S","NT11")
class StageUGenerationError(RuntimeError): pass

def load_rows():
    if not MANIFEST_CSV.exists(): raise StageUGenerationError(f"missing manifest {MANIFEST_CSV}")
    rows=[]
    with MANIFEST_CSV.open(newline="",encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            n=int(r["n"]); order=json.loads(r["scan_order"]); helpers.validate_scan_order(n,order)
            if str(r.get("teacher_validated")).lower()!="false" or str(r.get("abaqus_validated")).lower()!="false": raise StageUGenerationError(f"already validated {r['strategy_name']}")
            rows.append(dict(r,n=n,scan_order=order))
    cn=Counter(int(r["n"]) for r in rows); cb=Counter(str(r["final_expansion_batch"]) for r in rows)
    if len(rows)!=224 or {n:cn[n] for n in EXPECTED_N}!=EXPECTED_N: raise StageUGenerationError(f"bad N counts total={len(rows)} counts={dict(cn)}")
    if {b:cb[b] for b in EXPECTED_BATCH}!=EXPECTED_BATCH: raise StageUGenerationError(f"bad batch counts {dict(cb)}")
    return rows

def validate_paths(rows):
    for r in rows:
        case_dir=Path(str(r["case_dir"])); batch=str(r["final_expansion_batch"]); n=int(r["n"])
        if CASE_ROOT not in case_dir.parents: raise StageUGenerationError(f"case outside root {case_dir}")
        if case_dir.parent != CASE_ROOT/batch/f"N{n}": raise StageUGenerationError(f"bad case schema {case_dir}")
        for key in ("order_json","expected_cae","expected_inp","expected_jnl","generation_log"):
            if Path(str(r[key])).parent!=case_dir: raise StageUGenerationError(f"{key} outside case dir for {r['strategy_name']}")
        for key in ("expected_cae","expected_inp","expected_jnl"):
            if Path(str(r[key])).exists(): raise StageUGenerationError(f"generated file already exists: {r[key]}")
        helpers.validate_no_solver_outputs(case_dir)

def ensure_output_fields(model):
    status={"requested_variables":list(OUTPUT_VARIABLES),"updated_requests":[],"created_request":False,"warnings":[]}
    if getattr(model,"fieldOutputRequests",None):
        for name,req in model.fieldOutputRequests.items():
            try: req.setValues(variables=OUTPUT_VARIABLES); status["updated_requests"].append(str(name))
            except Exception as exc: status["warnings"].append(f"could not update {name}: {exc}")
    if not status["updated_requests"]:
        creator=getattr(model,"FieldOutputRequest",None)
        if creator is None: status["warnings"].append("FieldOutputRequest creator unavailable; relying on base output requests")
        else:
            try: creator(name="F-Output-PPO-FinalExpansion-TeacherMetrics",createStepName="step_scan_00",variables=OUTPUT_VARIABLES); status["created_request"]=True
            except Exception as exc: raise StageUGenerationError(f"could not request output fields {OUTPUT_VARIABLES}: {exc}")
    return status

def write_case_files(row):
    from abaqus import mdb  # type: ignore
    from abaqusConstants import OFF  # type: ignore
    case_dir=Path(str(row["case_dir"])); case_dir.mkdir(parents=True,exist_ok=True); helpers.validate_no_solver_outputs(case_dir); os.chdir(str(case_dir))
    job=str(row["job_name"])
    if job in mdb.jobs: del mdb.jobs[job]
    model_name="Model-1" if "Model-1" in mdb.models else list(mdb.models.keys())[0]
    mdb.Job(name=job,model=model_name)
    mdb.saveAs(pathName=str(row["expected_cae"]))
    mdb.jobs[job].writeInput(consistencyChecking=OFF)
    if not Path(str(row["expected_inp"])).exists() or Path(str(row["expected_inp"])).stat().st_size<=0: raise StageUGenerationError(f"INP not written {row['expected_inp']}")
    Path(str(row["expected_jnl"])).write_text(f"Stage U final expansion generation placeholder for {row['strategy_name']}.\nCAE/INP generated only; no solver/datacheck/enqueue/ODB.\n",encoding="utf-8")

def write_log(row,records,deactivations,final_record,output_status):
    payload={"status":"GENERATED","run_id":RUN_ID,"batch_name":BATCH_NAME,"final_expansion_batch":row["final_expansion_batch"],"case":{"n":row["n"],"strategy_name":row["strategy_name"],"job_name":row["job_name"],"cae_path":row["expected_cae"],"inp_path":row["expected_inp"]},"sequence_records":records,"deactivation_records":deactivations,"final_cooling":final_record,"output_field_request":output_status,"solver_submitted":False,"datacheck_run":False,"abqjobpilot_run":False,"enqueue_run":False,"odb_opened":False,"teacher_validation_run":False}
    Path(str(row["generation_log"])).write_text(json.dumps(payload,indent=2),encoding="utf-8")

def write_generation_manifest(rows):
    fields=["run_id","batch_name","final_expansion_batch","n","strategy_name","job_name","case_dir","scan_order_json","cae_path","inp_path","jnl_path","generation_status","inp_check_status","teacher_validated","solver_status","notes"]
    out=[]
    for r in rows:
        out.append({"run_id":RUN_ID,"batch_name":BATCH_NAME,"final_expansion_batch":r["final_expansion_batch"],"n":r["n"],"strategy_name":r["strategy_name"],"job_name":r["job_name"],"case_dir":r["case_dir"],"scan_order_json":r["order_json"],"cae_path":r["expected_cae"],"inp_path":r["expected_inp"],"jnl_path":r["expected_jnl"],"generation_status":"GENERATED","inp_check_status":"PENDING","teacher_validated":"False","solver_status":"NOT_SUBMITTED","notes":"Stage U PPO final expansion CAE/INP generation only; no teacher metrics yet."})
    with GEN_CSV.open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(out)
    GEN_JSON.write_text(json.dumps({"rows":out,"row_count":len(out)},indent=2),encoding="utf-8")

def generate_all(rows):
    from abaqus import mdb, openMdb  # type: ignore
    generated=[]
    for i,row in enumerate(rows,1):
        n=int(row["n"]); openMdb(pathName=str(BASES[n])); model=mdb.models["Model-1"] if "Model-1" in mdb.models else mdb.models[list(mdb.models.keys())[0]]
        helpers.infer_templates(model); heat_regions=helpers.validate_heat_sets_exist_in_model(model,n); order=list(row["scan_order"])
        records=helpers.create_sequence(model,n,order)
        for rec in records: helpers.create_heat_load(model,int(rec["seq"]),int(rec["track"]),heat_regions)
        deact=helpers.deactivate_loads(model,records); final=helpers.append_final_cooling(model,records); output_status=ensure_output_fields(model)
        helpers.validate_model_before_write(model,n,order,records,deact,final); write_case_files(row); write_log(row,records,deact,final,output_status)
        generated.append({"final_expansion_batch":row["final_expansion_batch"],"n":n,"strategy_name":row["strategy_name"],"cae_path":row["expected_cae"],"inp_path":row["expected_inp"]})
        if i%16==0: print(f"generated {i}/224")
    write_generation_manifest(rows)
    cn=Counter(g["n"] for g in generated); cb=Counter(g["final_expansion_batch"] for g in generated)
    summary={"verdict":"PASS_STAGEU_FINAL_EXPANSION_GENERATION_COMPLETE","generated_count":len(generated),"per_n_counts":{str(n):cn[n] for n in sorted(EXPECTED_N)},"per_batch_counts":{b:cb[b] for b in sorted(EXPECTED_BATCH)},"manifest_csv":str(GEN_CSV),"manifest_json":str(GEN_JSON),"no_solver_run":True,"no_datacheck_run":True,"no_abqjobpilot_run":True,"no_enqueue_run":True,"no_odb_opened":True,"no_teacher_validation":True}
    GEN_SUMMARY.write_text(json.dumps(summary,indent=2),encoding="utf-8")

def main():
    try:
        (OUT/"reports").mkdir(parents=True,exist_ok=True); rows=load_rows(); validate_paths(rows); generate_all(rows); print("PASS_STAGEU_FINAL_EXPANSION_GENERATION_COMPLETE"); print(f"summary={GEN_SUMMARY}"); return 0
    except Exception as exc:
        FAILURE_LOG.parent.mkdir(parents=True,exist_ok=True); FAILURE_LOG.write_text(json.dumps({"error":str(exc),"traceback":traceback.format_exc()},indent=2),encoding="utf-8"); print("FAIL_STAGEU_FINAL_EXPANSION_GENERATION"); print(str(exc)); print(f"failure_log={FAILURE_LOG}"); return 1
if __name__=="__main__": raise SystemExit(main())
