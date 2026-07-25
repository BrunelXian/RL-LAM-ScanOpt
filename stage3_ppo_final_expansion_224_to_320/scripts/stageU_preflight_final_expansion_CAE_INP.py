"""Stage U preflight for PPO final expansion 224 CAE/INP handoff."""
from __future__ import annotations
import csv, json
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT=Path(r"E:\Projects\RL-LAM-ScanOpt")
NS="stage3_ppo_final_expansion_224_to_320"
BATCH_NAME="stage3_ppo_final_expansion_224_to_320_batch224_v01"
SELECTED=PROJECT_ROOT/"outputs"/NS/"selected_candidates"/"PPO_FINAL_EXPANSION_224_SELECTED_MASTER.csv"
BATCH_DIR=PROJECT_ROOT/"outputs"/NS/"selected_candidates"/"batches"
SCAN_DIR=PROJECT_ROOT/"outputs"/NS/"selected_candidates"/"scan_orders"
CASE_ROOT=PROJECT_ROOT/"cae_model"/BATCH_NAME
OUT=PROJECT_ROOT/"outputs"/NS/"stageU_CAE_INP_handoff"
CHECKS=OUT/"checks"
CSV_OUT=CHECKS/"stageU_preflight_CAE_INP.csv"
JSON_OUT=CHECKS/"stageU_preflight_CAE_INP_summary.json"
BASES={12:PROJECT_ROOT/"cae_model"/"12track_full"/"sanity_base"/"12track_sanity_base.cae",16:PROJECT_ROOT/"cae_model"/"16track_full"/"sanity_base"/"16track_sanity_base.cae",24:PROJECT_ROOT/"cae_model"/"24track_full"/"sanity_base"/"24track_sanity_base.cae",40:PROJECT_ROOT/"cae_model"/"40track_full"/"sanity_base"/"40track_sanity_base.cae"}
EXPECTED_N={12:32,16:32,24:80,40:80}
EXPECTED_BATCH={"final_expansion_batch01":{12:16,16:16},"final_expansion_batch02":{12:16,16:16},"final_expansion_batch03":{24:32},"final_expansion_batch04":{24:32},"final_expansion_batch05":{24:16,40:16},"final_expansion_batch06":{40:32},"final_expansion_batch07":{40:32}}
SOLVER_EXTS=(".odb",".sim",".sta",".dat",".msg",".lck")
GEN_EXTS=(".cae",".inp",".jnl")

def parse_order(v):
    v=str(v).strip()
    return [int(x) for x in (json.loads(v) if v.startswith("[") else [p for p in v.split(",") if p!=""])]

def falseish(v): return str(v).strip().lower() in ("false","0","no","")
def ok_order(n,o): return len(o)==n and sorted(o)==list(range(n))
def add(rows,name,ok,details=""):
    rows.append({"check":name,"status":"PASS" if ok else "FAIL","details":details})
    return ok

def read_csv(p):
    with p.open(newline="",encoding="utf-8-sig") as f: return list(csv.DictReader(f))

def main():
    for p in (OUT,CHECKS,OUT/"commands",OUT/"manifest",OUT/"reports",OUT/"tables"): p.mkdir(parents=True,exist_ok=True)
    rows_out=[]; blockers=[]; warnings=[]; rows=[]
    if not add(rows_out,"selected_master_exists",SELECTED.exists(),str(SELECTED)): blockers.append("selected master missing")
    else:
        rows=read_csv(SELECTED)
        counts=Counter(int(r["n"]) for r in rows)
        checks={"row_count_224":len(rows)==224,"N_counts_match":{n:counts[n] for n in EXPECTED_N}==EXPECTED_N,"no_unexpected_N":all(int(r["n"]) in EXPECTED_N for r in rows),"strategy_unique":len({r["strategy_name"] for r in rows})==len(rows),"candidate_source_final_expansion":all(r.get("candidate_source") in ("PPO_final_expansion_checkpoint_inference","PPO_final_expansion_checkpoint_inference_equivalent") for r in rows),"teacher_validated_false":all(falseish(r.get("teacher_validated","")) for r in rows),"abaqus_validated_false":all(falseish(r.get("abaqus_validated","")) for r in rows)}
        by_order=set(); dup_order=False
        for r in rows:
            key=(int(r["n"]),tuple(parse_order(r["order_json"])))
            if key in by_order: dup_order=True
            by_order.add(key)
        checks["no_duplicate_order_within_same_N"]=not dup_order
        for name,ok in checks.items():
            if not add(rows_out,name,ok,json.dumps({str(k):counts[k] for k in sorted(counts)}) if "count" in name else ""): blockers.append(name)
        for b,alloc in EXPECTED_BATCH.items():
            p=BATCH_DIR/f"PPO_FINAL_EXPANSION_{b.replace('final_expansion_','')}.csv"
            brow=[]
            exists=p.exists()
            if exists: brow=read_csv(p)
            bc=Counter(int(r["n"]) for r in brow)
            ok=exists and len(brow)==32 and {n:bc[n] for n in sorted(alloc)}==alloc and all(r.get("final_expansion_batch")==b for r in brow)
            if not add(rows_out,f"batch_file_{b}",ok,f"{p}; rows={len(brow)}; counts={dict(bc)}"): blockers.append(f"batch allocation issue {b}")
        for r in rows:
            strategy=r["strategy_name"]; n=int(r["n"]); order=parse_order(r["order_json"]); jp=SCAN_DIR/f"scan_order_{strategy}.json"
            exists=jp.exists(); match=False; legal=ok_order(n,order)
            if exists:
                payload=json.loads(jp.read_text(encoding="utf-8")); jo=[int(x) for x in payload.get("order",payload.get("scan_order",[]))]; match=jo==order
            if not add(rows_out,f"scan_json_{strategy}",exists and match and legal,f"exists={exists}; match={match}; legal={legal}; {jp}"): blockers.append(f"scan_order issue {strategy}")
    for n,p in BASES.items():
        if not add(rows_out,f"base_cae_N{n}",p.exists(),str(p)): blockers.append(f"missing base N{n}")
    solver=[]; gen=[]; lck=[]
    if CASE_ROOT.exists():
        for ext in SOLVER_EXTS:
            found=list(CASE_ROOT.rglob(f"*{ext}")); solver.extend(found)
            if ext==".lck": lck.extend(found)
        for ext in GEN_EXTS: gen.extend(CASE_ROOT.rglob(f"*{ext}"))
    add(rows_out,"case_root_no_lck",not lck,";".join(map(str,lck[:10])))
    add(rows_out,"case_root_no_solver_outputs",not solver,";".join(map(str,solver[:10])))
    rows_out.append({"check":"case_root_existing_generation_files","status":"PASS" if not gen else "WARNING","details":f"existing CAE/INP/JNL count={len(gen)}"})
    if lck: blockers.append(".lck under case root")
    if solver: blockers.append("solver outputs under case root")
    if gen: warnings.append("existing CAE/INP/JNL under case root")
    with CSV_OUT.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["check","status","details"]); w.writeheader(); w.writerows(rows_out)
    verdict="FAIL_STAGEU_FINAL_EXPANSION_CAE_INP_PREFLIGHT_BLOCKED" if blockers else ("WARNING_STAGEU_FINAL_EXPANSION_CAE_INP_PREFLIGHT_REVIEW" if warnings else "PASS_STAGEU_FINAL_EXPANSION_CAE_INP_PREFLIGHT_READY")
    summary={"verdict":verdict,"selected_master":str(SELECTED),"batch_dir":str(BATCH_DIR),"scan_json_dir":str(SCAN_DIR),"case_root":str(CASE_ROOT),"row_count":len(rows),"counts_by_n":{str(n):Counter(int(r["n"]) for r in rows)[n] for n in sorted(EXPECTED_N)},"expected_batch_allocation":{b:{str(k):v for k,v in alloc.items()} for b,alloc in EXPECTED_BATCH.items()},"csv_path":str(CSV_OUT),"blockers":blockers,"warnings":warnings,"no_solver":True,"no_datacheck":True,"no_enqueue":True,"no_ODB":True}
    JSON_OUT.write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(verdict); print(f"summary={JSON_OUT}")
    return 0 if not blockers else 1
if __name__=="__main__": raise SystemExit(main())
