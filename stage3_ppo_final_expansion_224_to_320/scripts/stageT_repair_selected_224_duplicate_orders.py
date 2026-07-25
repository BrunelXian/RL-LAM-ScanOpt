"""Repair Stage T selected 224 by replacing same-N exact scan-order duplicates.

Uses only existing Stage T rollout pool candidates. Does not generate new PPO
candidates, does not mutate scan orders manually, and does not run Abaqus.
"""
from __future__ import annotations
import csv, hashlib, json, shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT=Path(r"E:\Projects\RL-LAM-ScanOpt")
NS="stage3_ppo_final_expansion_224_to_320"
OUT=ROOT/"outputs"/NS
DOCS=ROOT/"docs"/NS
SELECTED=OUT/"selected_candidates"/"PPO_FINAL_EXPANSION_224_SELECTED_MASTER.csv"
BATCH_DIR=OUT/"selected_candidates"/"batches"
SCAN_DIR=OUT/"selected_candidates"/"scan_orders"
POOL=OUT/"rollout_pools"/"ppo_final_expansion_rollout_pool.csv"
AUDIT_DIR=OUT/"audits"
REPAIR_DIR=OUT/"stageT_repair_duplicate_orders"
MANIFEST=OUT/"stageT_repair_duplicate_orders_manifest.json"
REPORT=DOCS/"PPO_FINAL_EXPANSION_224_STAGE_T_REPAIR_REPORT.md"
COMBINED552=ROOT/"outputs"/"stage3_run_78_final_evidence_freeze_package"/"FROZEN_stage3_native_combined552_teacher_dataset.csv"
V01=ROOT/"outputs"/"stage3_ppo_rl_lam_fea_addendum_v01"/"stageI_final_ppo_evidence_freeze"/"frozen_tables"/"FROZEN_PPO_batch32_teacher_metrics.csv"
V02=ROOT/"outputs"/"stage3_ppo_rl_lam_fea_addendum_v02_targeted_N24_N40"/"stageM_ODB_teacher_metric_extraction"/"stageM_v02K2_teacher_metrics.csv"
V03=ROOT/"outputs"/"stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40"/"stageR_ODB_teacher_metric_extraction"/"stageR_v03_teacher_metrics.csv"
EXPECTED_N={12:32,16:32,24:80,40:80}
EXPECTED_BATCH={"final_expansion_batch01":32,"final_expansion_batch02":32,"final_expansion_batch03":32,"final_expansion_batch04":32,"final_expansion_batch05":32,"final_expansion_batch06":32,"final_expansion_batch07":32}
BATCH_ALLOCATION={"final_expansion_batch01":{12:16,16:16},"final_expansion_batch02":{12:16,16:16},"final_expansion_batch03":{24:32},"final_expansion_batch04":{24:32},"final_expansion_batch05":{24:16,40:16},"final_expansion_batch06":{40:32},"final_expansion_batch07":{40:32}}
BUCKET_PRIORITY=["quality","diversity","industrial_efficiency","efficiency","novelty","baseline_proximity"]
CANON_COLS=("order_json","order_compact","scan_order","order")

def read_csv(path:Path):
    with path.open(newline="",encoding="utf-8-sig") as f: return list(csv.DictReader(f))

def write_csv(path:Path, rows:list[dict], fieldnames:list[str]):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fieldnames,extrasaction="ignore"); w.writeheader(); w.writerows(rows)

def parse_order(v):
    if v is None: return None
    s=str(v).strip()
    if not s or s.lower()=="nan": return None
    try:
        if s.startswith("["): return [int(x) for x in json.loads(s)]
        return [int(x) for x in s.split(",") if x!=""]
    except Exception:
        return None

def order_from_row(row):
    for c in CANON_COLS:
        if c in row:
            order=parse_order(row.get(c))
            if order is not None: return order
    return None

def canonical_hash(order): return hashlib.sha256(",".join(str(int(v)) for v in order).encode("utf-8")).hexdigest()[:16]
def compact(order): return ",".join(str(int(v)) for v in order)
def legal(n, order): return order is not None and len(order)==n and sorted(order)==list(range(n))
def falseish(v): return str(v).strip().lower() in ("false","0","no","")
def truthy(v): return str(v).strip().lower() in ("true","1","yes")

def ref_keys(path:Path):
    refs=defaultdict(set)
    if not path.exists(): return refs
    for r in read_csv(path):
        try: n=int(float(r.get("n","")))
        except Exception: continue
        o=order_from_row(r)
        if legal(n,o): refs[n].add(tuple(o))
    return refs

def merge_refs():
    merged=defaultdict(set)
    for label,path in (("combined552",COMBINED552),("ppo_v01",V01),("ppo_v02K2",V02),("ppo_v03",V03)):
        refs=ref_keys(path)
        for n,vals in refs.items(): merged[n].update(vals)
    return merged

def duplicate_groups(rows):
    groups=defaultdict(list)
    for i,r in enumerate(rows):
        n=int(r["n"]); o=order_from_row(r); groups[(n,tuple(o))].append(i)
    return {k:v for k,v in groups.items() if len(v)>1}

def as_float(v, default=0.0):
    try:
        if v is None or str(v).strip()=="": return default
        return float(v)
    except Exception:
        return default


def bucket_sort_key(row, bucket):
    quality=-as_float(row.get("predicted_quality_score"),0.0)
    novelty=-as_float(row.get("novelty_distance_score"),0.0)
    travel=as_float(row.get("total_travel_proxy"),999999.0)
    complexity=as_float(row.get("path_complexity_score"),999999.0)
    adjacent=-as_float(row.get("adjacent_fraction"),0.0)
    baseline=as_float(row.get("min_hamming_to_conventional_baseline"),999999.0)
    if bucket=="quality": return (quality, novelty, travel)
    if bucket=="diversity": return (novelty, quality, travel)
    if bucket in ("industrial_efficiency","efficiency"): return (complexity, travel, adjacent, quality)
    if bucket=="novelty": return (novelty, quality, travel)
    if bucket=="baseline_proximity": return (baseline, quality, travel)
    return (quality, novelty, travel)


def eligible_pool_rows(pool_rows, n, used, refs):
    rows=[]
    for pr in pool_rows:
        pn=int(pr["n"])
        if pn!=n: continue
        o=order_from_row(pr)
        if not legal(n,o): continue
        key=(n,tuple(o))
        if key in used or tuple(o) in refs[n]: continue
        if truthy(pr.get("duplicate_vs_combined552")) or truthy(pr.get("duplicate_vs_ppo_v01")) or truthy(pr.get("duplicate_vs_ppo_v02K2")) or truthy(pr.get("duplicate_vs_ppo_v03")): continue
        if not falseish(pr.get("teacher_validated","False")) or not falseish(pr.get("abaqus_validated","False")): continue
        if pr.get("candidate_source") not in ("PPO_final_expansion_checkpoint_inference_pool","PPO_final_expansion_checkpoint_inference"): continue
        rows.append(pr)
    return rows


def pool_candidates(pool_rows, n, bucket_order, used, refs):
    eligible=eligible_pool_rows(pool_rows,n,used,refs)
    if not eligible: return None, ""
    for bucket in bucket_order:
        ranked=sorted(eligible,key=lambda r: bucket_sort_key(r,bucket))
        if ranked: return ranked[0], bucket
    return None, ""

def make_json(row, order):
    payload={
        "strategy_name":row["strategy_name"],"final_expansion_batch":row["final_expansion_batch"],"n":int(row["n"]),"order":order,"order_json":json.dumps(order),"order_compact":compact(order),"order_hash":canonical_hash(order),"ppo_checkpoint":row.get("ppo_checkpoint",""),"ppo_version_source":row.get("ppo_version_source",""),"ppo_generation_mode":row.get("ppo_generation_mode",""),"selected_by_bucket":row.get("selected_by_bucket",""),"candidate_source":row.get("candidate_source",""),"teacher_validated":False,"abaqus_validated":False,"notes":row.get("notes","")}
    for k in ("predicted_quality_score","predicted_reward_available","mean_abs_jump","max_abs_jump","long_jump_count","adjacent_fraction","total_travel_proxy","jump_variance","local_continuity_score","path_complexity_score","novelty_distance_score","min_hamming_to_combined552_sameN","nearest_conventional_baseline","min_hamming_to_conventional_baseline"):
        if k in row: payload[k]=row[k]
    return payload

def update_from_pool(slot, pr, replacement_bucket):
    old_strategy=slot["strategy_name"]; old_batch=slot["final_expansion_batch"]; old_global=slot["global_candidate_index"]
    old_n=slot["n"]
    preserve={"strategy_name":old_strategy,"final_expansion_batch":old_batch,"global_candidate_index":old_global,"n":old_n}
    for k,v in pr.items():
        if k in slot: slot[k]=v
    slot.update(preserve)
    o=order_from_row(pr)
    slot["order_json"]=json.dumps(o)
    slot["order_compact"]=compact(o)
    slot["order_hash"]=canonical_hash(o)
    slot["candidate_source"]="PPO_final_expansion_checkpoint_inference"
    slot["teacher_validated"]="False"; slot["abaqus_validated"]="False"; slot["selected_already"]="False"
    slot["selected_by_bucket"]=replacement_bucket if replacement_bucket else slot.get("selected_by_bucket","")
    slot["notes"]="Stage T repair replacement from rollout pool; original duplicate physical order removed; not physically validated yet."
    return slot

def main():
    REPAIR_DIR.mkdir(parents=True,exist_ok=True); AUDIT_DIR.mkdir(parents=True,exist_ok=True); DOCS.mkdir(parents=True,exist_ok=True)
    selected=read_csv(SELECTED); pool=read_csv(POOL); refs=merge_refs()
    original_fields=list(selected[0].keys())
    groups=duplicate_groups(selected)
    dup_rows=[]; replacements=[]
    repaired=[dict(r) for r in selected]
    # Any exact duplicate within selected or exact duplicate versus reference pools
    # must be replaced. Keep the first non-reference member of each internal group
    # where possible; otherwise all reference-duplicate members are replaced.
    replace_indices=set()
    for (n,order),idxs in sorted(groups.items(), key=lambda kv:(kv[0][0], min(kv[1]))):
        non_ref=[idx for idx in idxs if tuple(order) not in refs[n]]
        keep=non_ref[0] if non_ref else None
        for idx in idxs:
            r=selected[idx]
            action="KEEP" if idx==keep else "REPLACE"
            if tuple(order) in refs[n]: action="REPLACE_REFERENCE_DUPLICATE"
            dup_rows.append({"n":n,"canonical_order_hash":canonical_hash(order),"order_hash":r.get("order_hash",""),"order_compact":compact(order),"strategy_name":r["strategy_name"],"final_expansion_batch":r["final_expansion_batch"],"selected_by_bucket":r["selected_by_bucket"],"duplicate_group_size":len(idxs),"duplicate_action":action,"real_exact_order_duplicate":True})
            if action!="KEEP": replace_indices.add(idx)
    for idx,r in enumerate(selected):
        n=int(r["n"]); o=order_from_row(r)
        if tuple(o) in refs[n]:
            replace_indices.add(idx)
            if not any(d.get("strategy_name")==r["strategy_name"] for d in dup_rows):
                dup_rows.append({"n":n,"canonical_order_hash":canonical_hash(o),"order_hash":r.get("order_hash",""),"order_compact":compact(o),"strategy_name":r["strategy_name"],"final_expansion_batch":r["final_expansion_batch"],"selected_by_bucket":r["selected_by_bucket"],"duplicate_group_size":1,"duplicate_action":"REPLACE_REFERENCE_DUPLICATE","real_exact_order_duplicate":True})
    used=set()
    for idx,r in enumerate(repaired):
        if idx in replace_indices: continue
        used.add((int(r["n"]),tuple(order_from_row(r))))
    for idx in sorted(replace_indices):
        slot=repaired[idx]
        n=int(slot["n"])
        old_order=order_from_row(slot)
        preferred=slot.get("selected_by_bucket","")
        bucket_order=[preferred]+[b for b in BUCKET_PRIORITY if b!=preferred]
        pr,bucket=pool_candidates(pool,n,bucket_order,used,refs)
        if pr is None: raise RuntimeError(f"No replacement candidate found for N{n} {slot['strategy_name']} bucket={preferred}")
        new_order=order_from_row(pr); new_key=(n,tuple(new_order))
        used.add(new_key)
        old_strategy=slot["strategy_name"]
        repaired[idx]=update_from_pool(slot,pr,bucket)
        replacements.append({"old_strategy_name":old_strategy,"old_n":n,"old_order_hash":canonical_hash(old_order),"old_order_compact":compact(old_order),"old_bucket":preferred,"replacement_source_row_index":pr.get("source_row_index",""),"replacement_ppo_version_source":pr.get("ppo_version_source",""),"replacement_bucket":bucket,"replacement_order_hash":canonical_hash(new_order),"replacement_order_compact":compact(new_order),"replacement_candidate_source":pr.get("candidate_source","")})    # Recompute all canonical hashes and assert legality/novelty.
    final_keys=set(); legality=[]; novelty=[]; blockers=[]
    for r in repaired:
        n=int(r["n"]); o=order_from_row(r); key=(n,tuple(o))
        if not legal(n,o): blockers.append(f"illegal {r['strategy_name']}")
        if key in final_keys: blockers.append(f"duplicate selected {r['strategy_name']}")
        final_keys.add(key); r["order_json"]=json.dumps(o); r["order_compact"]=compact(o); r["order_hash"]=canonical_hash(o)
        dup_ref=tuple(o) in refs[n]
        if dup_ref: blockers.append(f"duplicate reference {r['strategy_name']}")
        if r.get("candidate_source") not in ("PPO_final_expansion_checkpoint_inference",): blockers.append(f"bad source {r['strategy_name']} {r.get('candidate_source')}")
        if not falseish(r.get("teacher_validated")) or not falseish(r.get("abaqus_validated")): blockers.append(f"validated flag {r['strategy_name']}")
        legality.append({"strategy_name":r["strategy_name"],"n":n,"order_hash":r["order_hash"],"legal":legal(n,o),"candidate_source_ok":r.get("candidate_source")=="PPO_final_expansion_checkpoint_inference","teacher_validated":not falseish(r.get("teacher_validated")),"abaqus_validated":not falseish(r.get("abaqus_validated"))})
        novelty.append({"strategy_name":r["strategy_name"],"n":n,"order_hash":r["order_hash"],"duplicate_within_selected":False,"duplicate_vs_combined552_v01_v02K2_v03":dup_ref,"order_compact":compact(o)})
    counts=Counter(int(r["n"]) for r in repaired); batch_counts=Counter(r["final_expansion_batch"] for r in repaired)
    if {n:counts[n] for n in EXPECTED_N}!=EXPECTED_N: blockers.append(f"bad N counts {dict(counts)}")
    if {b:batch_counts[b] for b in EXPECTED_BATCH}!=EXPECTED_BATCH: blockers.append(f"bad batch counts {dict(batch_counts)}")
    for b,alloc in BATCH_ALLOCATION.items():
        bc=Counter(int(r["n"]) for r in repaired if r["final_expansion_batch"]==b)
        if {n:bc[n] for n in alloc}!={n:alloc[n] for n in alloc}: blockers.append(f"bad batch allocation {b} {dict(bc)}")
    if blockers: raise RuntimeError("; ".join(blockers[:20]))
    # Write repaired selected master, per-batch CSVs, scan-order JSONs.
    write_csv(SELECTED,repaired,original_fields)
    for b in sorted(EXPECTED_BATCH):
        rows=[r for r in repaired if r["final_expansion_batch"]==b]
        write_csv(BATCH_DIR/f"PPO_FINAL_EXPANSION_{b.replace('final_expansion_','')}.csv",rows,original_fields)
    # Replace scan_order JSONs for all selected cases from repaired row metadata.
    if SCAN_DIR.exists():
        archive=REPAIR_DIR/"scan_orders_before_repair"
        if not archive.exists(): shutil.copytree(SCAN_DIR,archive)
    for r in repaired:
        o=order_from_row(r); path=SCAN_DIR/f"scan_order_{r['strategy_name']}.json"; path.write_text(json.dumps(make_json(r,o),indent=2),encoding="utf-8")
    # Audits.
    write_csv(REPAIR_DIR/"stageT_repair_duplicate_groups.csv",dup_rows,["n","canonical_order_hash","order_hash","order_compact","strategy_name","final_expansion_batch","selected_by_bucket","duplicate_group_size","duplicate_action","real_exact_order_duplicate"])
    write_csv(REPAIR_DIR/"stageT_repair_replacements.csv",replacements,["old_strategy_name","old_n","old_order_hash","old_order_compact","old_bucket","replacement_source_row_index","replacement_ppo_version_source","replacement_bucket","replacement_order_hash","replacement_order_compact","replacement_candidate_source"])
    write_csv(AUDIT_DIR/"final_expansion_duplicate_audit.csv",novelty,["strategy_name","n","order_hash","duplicate_within_selected","duplicate_vs_combined552_v01_v02K2_v03","order_compact"])
    write_csv(AUDIT_DIR/"final_expansion_legality_audit.csv",legality,["strategy_name","n","order_hash","legal","candidate_source_ok","teacher_validated","abaqus_validated"])
    write_csv(AUDIT_DIR/"final_expansion_novelty_audit.csv",novelty,["strategy_name","n","order_hash","duplicate_within_selected","duplicate_vs_combined552_v01_v02K2_v03","order_compact"])
    batch_audit=[]
    for b in sorted(EXPECTED_BATCH):
        rows=[r for r in repaired if r["final_expansion_batch"]==b]; bc=Counter(int(r["n"]) for r in rows)
        batch_audit.append({"final_expansion_batch":b,"total_count":len(rows),"N12":bc[12],"N16":bc[16],"N24":bc[24],"N40":bc[40],"pass":len(rows)==32})
    write_csv(AUDIT_DIR/"final_expansion_batch_count_audit.csv",batch_audit,["final_expansion_batch","total_count","N12","N16","N24","N40","pass"])
    # Handoff preview update.
    preview=[]
    for r in repaired:
        case=f"E:\\Projects\\RL-LAM-ScanOpt\\cae_model\\stage3_ppo_final_expansion_224_to_320_batch224_v01\\{r['final_expansion_batch']}\\N{r['n']}\\{r['strategy_name']}"
        job=f"J2D_{r['strategy_name']}"
        preview.append({"final_expansion_batch":r["final_expansion_batch"],"strategy_name":r["strategy_name"],"n":r["n"],"expected_case_dir":case,"expected_job_name":job,"expected_INP_path_placeholder":f"{case}\\{job}.inp","order_hash":r["order_hash"],"selected_by_bucket":r["selected_by_bucket"],"candidate_source":r["candidate_source"],"teacher_validated":False,"abaqus_validated":False})
    write_csv(OUT/"handoff_preview"/"PPO_FINAL_EXPANSION_224_CAE_INP_HANDOFF_PREVIEW.csv",preview,list(preview[0].keys()))
    report_lines=["# PPO Final Expansion 224 Stage T Repair Report","",f"Timestamp: {datetime.now().isoformat()}","",f"Duplicate groups found: {len(groups)}",f"Replacement count: {len(replacements)}","","Same-N exact scan-order duplicates were treated as real physical duplicate cases. No manual scan-order mutation was performed; replacements came from the existing Stage T rollout pool.","","## Outputs","",f"- Duplicate groups: {REPAIR_DIR/'stageT_repair_duplicate_groups.csv'}",f"- Replacements: {REPAIR_DIR/'stageT_repair_replacements.csv'}",f"- Duplicate audit: {AUDIT_DIR/'final_expansion_duplicate_audit.csv'}",f"- Legality audit: {AUDIT_DIR/'final_expansion_legality_audit.csv'}",f"- Novelty audit: {AUDIT_DIR/'final_expansion_novelty_audit.csv'}",f"- Repaired master CSV: {SELECTED}","","## Final Counts","",f"- N12: {counts[12]}",f"- N16: {counts[16]}",f"- N24: {counts[24]}",f"- N40: {counts[40]}","- Total: 224","","## Claim Boundary","","This repair only fixes handoff candidate uniqueness. It does not run solver, datacheck, enqueue, ODB extraction, PPO training, surrogate training, or teacher validation."]
    REPORT.write_text("\n".join(report_lines)+"\n",encoding="utf-8")
    manifest={"timestamp":datetime.now().isoformat(),"selected_master":str(SELECTED),"rollout_pool":str(POOL),"duplicate_group_count":len(groups),"replacement_count":len(replacements),"counts_by_n":{str(n):counts[n] for n in sorted(EXPECTED_N)},"batch_counts":{b:batch_counts[b] for b in sorted(EXPECTED_BATCH)},"duplicate_groups_csv":str(REPAIR_DIR/"stageT_repair_duplicate_groups.csv"),"replacements_csv":str(REPAIR_DIR/"stageT_repair_replacements.csv"),"legality_audit":str(AUDIT_DIR/"final_expansion_legality_audit.csv"),"novelty_audit":str(AUDIT_DIR/"final_expansion_novelty_audit.csv"),"duplicate_audit":str(AUDIT_DIR/"final_expansion_duplicate_audit.csv"),"report":str(REPORT),"no_solver":True,"no_datacheck":True,"no_enqueue":True,"no_ODB":True,"no_training":True,"no_candidate_generation":True,"final_verdict":"PASS_STAGE_T_REPAIR_DUPLICATES_REMOVED"}
    MANIFEST.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print("PASS_STAGE_T_REPAIR_DUPLICATES_REMOVED")
    print(f"duplicate_groups={len(groups)} replacements={len(replacements)} manifest={MANIFEST}")
    return 0
if __name__=="__main__": raise SystemExit(main())


