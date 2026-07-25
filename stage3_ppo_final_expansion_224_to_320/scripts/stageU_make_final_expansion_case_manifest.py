"""Create Stage U final expansion case manifest and case directories."""
from __future__ import annotations
import csv, json, shutil
from pathlib import Path

PROJECT_ROOT=Path(r"E:\Projects\RL-LAM-ScanOpt")
NS="stage3_ppo_final_expansion_224_to_320"
BATCH_NAME="stage3_ppo_final_expansion_224_to_320_batch224_v01"
SELECTED=PROJECT_ROOT/"outputs"/NS/"selected_candidates"/"PPO_FINAL_EXPANSION_224_SELECTED_MASTER.csv"
SCAN_DIR=PROJECT_ROOT/"outputs"/NS/"selected_candidates"/"scan_orders"
CASE_ROOT=PROJECT_ROOT/"cae_model"/BATCH_NAME
MANIFEST_DIR=PROJECT_ROOT/"outputs"/NS/"stageU_CAE_INP_handoff"/"manifest"
CSV_OUT=MANIFEST_DIR/"stageU_final_expansion_case_manifest.csv"
JSON_OUT=MANIFEST_DIR/"stageU_final_expansion_case_manifest.json"
BASES={12:PROJECT_ROOT/"cae_model"/"12track_full"/"sanity_base"/"12track_sanity_base.cae",16:PROJECT_ROOT/"cae_model"/"16track_full"/"sanity_base"/"16track_sanity_base.cae",24:PROJECT_ROOT/"cae_model"/"24track_full"/"sanity_base"/"24track_sanity_base.cae",40:PROJECT_ROOT/"cae_model"/"40track_full"/"sanity_base"/"40track_sanity_base.cae"}
EXTRA_FIELDS=("predicted_quality_score","mean_abs_jump","max_abs_jump","long_jump_count","adjacent_fraction","total_travel_proxy","jump_variance","local_continuity_score","path_complexity_score","novelty_distance_score","min_hamming_to_combined552_sameN","min_hamming_to_conventional_baseline","nearest_conventional_baseline")

def parse_order(v):
    v=str(v).strip()
    return [int(x) for x in (json.loads(v) if v.startswith("[") else [p for p in v.split(",") if p!=""])]

def main():
    MANIFEST_DIR.mkdir(parents=True,exist_ok=True); CASE_ROOT.mkdir(parents=True,exist_ok=True)
    with SELECTED.open(newline="",encoding="utf-8-sig") as f: input_rows=list(csv.DictReader(f))
    rows=[]
    for r in sorted(input_rows,key=lambda x:(x["final_expansion_batch"],int(x["n"]),int(x["global_candidate_index"]))):
        n=int(r["n"]); strategy=r["strategy_name"]; batch=r["final_expansion_batch"]; order=parse_order(r["order_json"])
        job=f"J2D_{strategy}"; case_dir=CASE_ROOT/batch/f"N{n}"/strategy; case_dir.mkdir(parents=True,exist_ok=True)
        src=SCAN_DIR/f"scan_order_{strategy}.json"; dst=case_dir/f"scan_order_{strategy}.json"
        if not dst.exists(): shutil.copyfile(src,dst)
        row={
            "final_expansion_batch":batch,"strategy_name":strategy,"n":n,"order_json":str(dst),"source_order_json":str(src),"scan_order":json.dumps(order,separators=(",",":")),"order_compact":r["order_compact"],"order_hash":r["order_hash"],"candidate_source":r["candidate_source"],"ppo_checkpoint":r["ppo_checkpoint"],"ppo_version_source":r["ppo_version_source"],"ppo_generation_mode":r["ppo_generation_mode"],"selected_by_bucket":r["selected_by_bucket"],"global_candidate_index":r["global_candidate_index"],"case_dir":str(case_dir),"job_name":job,"expected_cae":str(case_dir/f"{job}.cae"),"expected_inp":str(case_dir/f"{job}.inp"),"expected_jnl":str(case_dir/f"{job}.jnl"),"generation_log":str(case_dir/f"{job}_stageU_generation_log.json"),"base_cae":str(BASES[n]),"batch_name":BATCH_NAME,"teacher_validated":r["teacher_validated"],"abaqus_validated":r["abaqus_validated"],"notes":r.get("notes","")}
        for field in EXTRA_FIELDS: row[field]=r.get(field,"")
        rows.append(row)
    with CSV_OUT.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    JSON_OUT.write_text(json.dumps({"batch_name":BATCH_NAME,"case_root":str(CASE_ROOT),"row_count":len(rows),"rows":rows},indent=2),encoding="utf-8")
    print(json.dumps({"manifest_csv":str(CSV_OUT),"manifest_json":str(JSON_OUT),"row_count":len(rows)},indent=2))
    return 0
if __name__=="__main__": raise SystemExit(main())
