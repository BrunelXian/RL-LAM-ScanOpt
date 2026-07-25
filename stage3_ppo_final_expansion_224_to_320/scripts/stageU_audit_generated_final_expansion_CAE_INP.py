"""Audit generated Stage U final expansion CAE/INP files and write command files."""
from __future__ import annotations
import csv, json, re
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT=Path(r"E:\Projects\RL-LAM-ScanOpt")
NS="stage3_ppo_final_expansion_224_to_320"
BATCH_NAME="stage3_ppo_final_expansion_224_to_320_batch224_v01"
OUT=PROJECT_ROOT/"outputs"/NS/"stageU_CAE_INP_handoff"
MANIFEST_CSV=OUT/"manifest"/"stageU_final_expansion_case_manifest.csv"
CSV_OUT=OUT/"checks"/"stageU_generated_CAE_INP_audit.csv"
JSON_OUT=OUT/"checks"/"stageU_generated_CAE_INP_audit_summary.json"
COMMAND_DIR=OUT/"commands"
ALL_COMMAND=COMMAND_DIR/"stageU_final_expansion_all224_abqjobpilot_commands_READY_TO_RUN.txt"
CASE_ROOT=PROJECT_ROOT/"cae_model"/BATCH_NAME
EXPECTED_N={12:32,16:32,24:80,40:80}
EXPECTED_BATCH={"final_expansion_batch01":32,"final_expansion_batch02":32,"final_expansion_batch03":32,"final_expansion_batch04":32,"final_expansion_batch05":32,"final_expansion_batch06":32,"final_expansion_batch07":32}
SOLVER_EXTS=(".odb",".sim",".sta",".dat",".msg",".lck")
NUMERIC_RE=r"[-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?"

def load_rows():
    rows=[]
    with MANIFEST_CSV.open(newline="",encoding="utf-8-sig") as f:
        for r in csv.DictReader(f): r["n"]=int(r["n"]); r["scan_order"]=json.loads(r["scan_order"]); rows.append(r)
    return rows

def section_entry_count(lines,keyword):
    count=0; in_sec=False
    for line in lines:
        s=line.strip()
        if s.lower().startswith(keyword.lower()): in_sec=True; continue
        if in_sec and s.startswith("*"): break
        if in_sec and s and not s.startswith("**"): count+=1
    return count

def step_blocks(lines):
    blocks={}; cur=None
    for line in lines:
        m=re.match(r"\*\* STEP:\s*(\S+)",line.strip(),re.I)
        if m: cur=m.group(1); blocks[cur]=[]; continue
        if cur is not None: blocks[cur].append(line)
    return {k:"\n".join(v) for k,v in blocks.items()}

def final_cooling_visible(lines,n):
    last=-1
    for i,line in enumerate(lines):
        if f"step_cool_{n-1:02d}".lower() in line.lower(): last=max(last,i)
    for i,line in enumerate(lines):
        if re.search(r"\*Step,\s*name=step_final_cooling\b",line,re.I):
            win="\n".join(lines[i:i+12]); has_i=re.search(r"(^|[^0-9.])0\.01([^0-9.]|$)",win) is not None; has_t=re.search(r"(^|[^0-9.])1200\.?([^0-9.]|$)",win) is not None; has_m=re.search(r"(^|[^0-9.])60\.?([^0-9.]|$)",win) is not None
            return has_i and has_t and has_m, i>last
    return False,False

def heat_order_matches(lines,n,order):
    blocks=step_blocks(lines); obs=[]
    for seq in range(n):
        block=blocks.get(f"step_scan_{seq:02d}",""); m=re.search(r"set_body_heat_(\d+)\s*,\s*BF\s*,",block,re.I)
        if not m: return False,f"missing BF set in step_scan_{seq:02d}"
        obs.append(int(m.group(1)))
    if obs!=order: return False,"observed order mismatch"
    if re.search(r"\*Dflux",blocks.get("step_final_cooling",""),re.I): return False,"Dflux appears in final cooling"
    return True,"observed order matches scan_order"

def output_fields_visible(text):
    upper=text.upper()
    def has(v): return re.search(rf"(^|[,\s]){re.escape(v)}([,\s]|$)",upper) is not None
    missing=[v for v in ("U","PEEQ","S") if not has(v)]
    if not (has("NT11") or has("NT")): missing.append("NT11_or_NT")
    return not missing,f"missing={missing}"

def n40_cool_initial_inc_visible(lines,n):
    if n!=40: return True,"not N40"
    seen=0; bad=[]
    for i,line in enumerate(lines):
        m=re.match(r"\*Step,\s*name=(step_cool_\d+)",line,re.I)
        if not m: continue
        seen+=1; data=""
        for cand in lines[i:i+8]:
            s=cand.strip()
            if re.match(r"^"+NUMERIC_RE+r"\s*,",s): data=s; break
        vals=[float(v) for v in re.findall(NUMERIC_RE,data)]
        if len(vals)<2 or abs(vals[0]-0.001)>1e-12 or abs(vals[1]-3.4)>1e-9: bad.append(f"{m.group(1)}:{data}")
    if seen!=40: return False,f"N40 cool step count {seen} != 40"
    if bad: return False,"; ".join(bad[:3])
    return True,"all N40 cool steps show initialInc=0.001 and timePeriod=3.4"

def check_one(r):
    n=int(r["n"]); inp=Path(str(r["expected_inp"])); cae=Path(str(r["expected_cae"])); case=Path(str(r["case_dir"]))
    res={"final_expansion_batch":r["final_expansion_batch"],"n":n,"strategy_name":r["strategy_name"],"inp_path":str(inp),"cae_path":str(cae),"inp_exists":inp.exists(),"cae_exists":cae.exists(),"inp_size_bytes":inp.stat().st_size if inp.exists() else 0,"cae_size_bytes":cae.stat().st_size if cae.exists() else 0,"node_entry_count":0,"element_entry_count":0,"scan_sequence_complete":False,"cool_sequence_complete":False,"step_final_cooling_exists":False,"final_cooling_controls_visible":False,"final_cooling_after_last_cool":False,"heat_flux_entries_exist":False,"all_expected_heat_sets_present":False,"heat_order_text_verified":False,"output_fields_U_PEEQ_S_NT11_visible":False,"n40_cool_initialInc_0p001_verified":False,"solver_output_count":sum(len(list(case.glob(f"*{ext}"))) for ext in SOLVER_EXTS),"verdict":"FAIL","notes":""}
    if not inp.exists() or not cae.exists() or inp.stat().st_size<=0 or cae.stat().st_size<=0: res["notes"]="missing/empty CAE or INP"; return res
    text=inp.read_text(encoding="utf-8",errors="replace"); low=text.lower(); lines=text.splitlines(); nodes=section_entry_count(lines,"*Node"); elems=section_entry_count(lines,"*Element"); final_ok,final_after=final_cooling_visible(lines,n); heat_ok,heat_note=heat_order_matches(lines,n,list(r["scan_order"])); out_ok,out_note=output_fields_visible(text); n40_ok,n40_note=n40_cool_initial_inc_visible(lines,n)
    res.update({"node_entry_count":nodes,"element_entry_count":elems,"scan_sequence_complete":all(f"step_scan_{s:02d}".lower() in low for s in range(n)),"cool_sequence_complete":all(f"step_cool_{s:02d}".lower() in low for s in range(n)),"step_final_cooling_exists":"step_final_cooling" in low,"final_cooling_controls_visible":final_ok,"final_cooling_after_last_cool":final_after,"heat_flux_entries_exist":"body heat flux" in low or "*dflux" in low,"all_expected_heat_sets_present":all(f"set_body_heat_{t:02d}".lower() in low for t in range(n)),"heat_order_text_verified":heat_ok,"output_fields_U_PEEQ_S_NT11_visible":out_ok,"n40_cool_initialInc_0p001_verified":n40_ok,"notes":"; ".join([heat_note,out_note,n40_note])})
    checks=[nodes>0,elems>0,res["scan_sequence_complete"],res["cool_sequence_complete"],res["step_final_cooling_exists"],res["final_cooling_controls_visible"],res["final_cooling_after_last_cool"],res["heat_flux_entries_exist"],res["all_expected_heat_sets_present"],res["heat_order_text_verified"],res["output_fields_U_PEEQ_S_NT11_visible"],res["n40_cool_initialInc_0p001_verified"],res["solver_output_count"]==0]
    res["verdict"]="PASS" if all(checks) else "FAIL"; return res

def write_command_files(rows):
    COMMAND_DIR.mkdir(parents=True,exist_ok=True); by_batch=defaultdict(list); all_lines=[]
    for r in rows:
        line=f'enqueue --inp "{r["expected_inp"]}" --cpus 14 --batch {BATCH_NAME} --strategy {r["strategy_name"]}'
        by_batch[r["final_expansion_batch"]].append(line); all_lines.append(line)
    paths={}; errors=[]
    for batch in sorted(EXPECTED_BATCH):
        path=COMMAND_DIR/f"stageU_{batch}_abqjobpilot_commands_READY_TO_RUN.txt"; lines=by_batch.get(batch,[]); path.write_text("\n".join(lines)+"\n",encoding="utf-8"); paths[batch]=str(path)
        if len(lines)!=32: errors.append(f"{batch} command count {len(lines)} != 32")
    ALL_COMMAND.write_text("\n".join(all_lines)+"\n",encoding="utf-8")
    if len(all_lines)!=224: errors.append(f"all command count {len(all_lines)} != 224")
    if any("--gpus" in l for l in all_lines): errors.append("contains --gpus")
    if any("--cpus 14" not in l for l in all_lines): errors.append("missing --cpus 14")
    if any(f"--batch {BATCH_NAME}" not in l for l in all_lines): errors.append("bad batch name")
    return {"per_batch_paths":paths,"all_batch_path":str(ALL_COMMAND),"all_command_count":len(all_lines),"errors":errors}

def main():
    rows=load_rows(); results=[check_one(r) for r in rows]
    with CSV_OUT.open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=list(results[0].keys())); w.writeheader(); w.writerows(results)
    cae_counts=Counter(int(r["n"]) for r in rows if Path(str(r["expected_cae"])).exists()); inp_counts=Counter(int(r["n"]) for r in rows if Path(str(r["expected_inp"])).exists()); batch_counts=Counter(str(r["final_expansion_batch"]) for r in rows if Path(str(r["expected_inp"])).exists())
    solver=[p for ext in SOLVER_EXTS for p in CASE_ROOT.rglob(f"*{ext}")] if CASE_ROOT.exists() else []
    command_status=write_command_files(rows); failed=[r for r in results if r["verdict"]!="PASS"]
    ok=len(results)==224 and not failed and {n:cae_counts[n] for n in EXPECTED_N}==EXPECTED_N and {n:inp_counts[n] for n in EXPECTED_N}==EXPECTED_N and {b:batch_counts[b] for b in EXPECTED_BATCH}==EXPECTED_BATCH and not solver and not command_status["errors"]
    verdict="PASS_STAGEU_FINAL_EXPANSION_CAE_INP_READY_FOR_USER_CONTROLLED_SOLVER" if ok else "FAIL_STAGEU_FINAL_EXPANSION_CAE_INP_GENERATION_FAILED"
    summary={"verdict":verdict,"checked_count":len(results),"pass_count":sum(1 for r in results if r["verdict"]=="PASS"),"cae_counts_by_n":{str(k):cae_counts[k] for k in sorted(EXPECTED_N)},"inp_counts_by_n":{str(k):inp_counts[k] for k in sorted(EXPECTED_N)},"inp_counts_by_batch":{b:batch_counts[b] for b in sorted(EXPECTED_BATCH)},"solver_output_count":len(solver),"failed":failed,"command_files":command_status,"csv_path":str(CSV_OUT)}
    JSON_OUT.write_text(json.dumps(summary,indent=2),encoding="utf-8"); print(verdict); print(f"summary={JSON_OUT}"); return 0 if ok else 1
if __name__=="__main__": raise SystemExit(main())
