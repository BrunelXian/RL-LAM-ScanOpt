"""Aggregate completed ablation evidence; never launches or modifies a model."""
import argparse,json,pathlib,datetime,re

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);args=ap.parse_args()
    plan=json.loads(pathlib.Path(args.plan).read_text());rep=pathlib.Path(plan['report_dir'])
    baseline=json.loads((rep/'baseline_inspection.json').read_text())['models']['Model-1']
    planned={name:float(re.search(r"'timePeriod': ([\d.eE+-]+)",str(data)).group(1)) for name,data in baseline['steps'].items() if name!='Initial'}
    verdicts={'A':'BASELINE_FAILURE_CONFIRMED','B':'ALE_GEOMETRY_IMPROVED_BUT_NOT_SOLVED','C':'STABILIZATION_NO_MEANINGFUL_IMPROVEMENT','D':'ALE_STAB_NO_MEANINGFUL_IMPROVEMENT'}
    # These are the reviewed run_005 outcomes, not a generic automatic classifier.
    assert plan['run_id']=='run_005','Other runs require their own evidence-based verdict review'
    rows={}
    for case in 'ABCD':
        logs=json.loads((rep/(case+'_logs.json')).read_text())
        odb=json.loads((rep/(case+'_odb_diagnostic.json')).read_text())
        build=json.loads((rep/(case+'_build.json')).read_text())
        audit=json.loads((rep/(case+'_input_diff_audit.json')).read_text())
        assert audit['status']=='passed' and logs['solver_status']=='FAILED'
        assert odb['available'] and 'error' not in odb
        expected={'A':40,'B':42,'C':6,'D':6}
        assert logs['fatal_step']==expected[case]
        logs.pop('evidence',None);logs.pop('per_step',None)
        last=odb['last_saved_geometry'];s=next(s for s in odb['steps'] if s['name']==last['step'])
        # Failed-step ODB timePeriod is truncated to elapsed time; use CAE duration.
        completed=[s for s in odb['steps'] if s['frames'] and s['last_frame_time'] is not None and abs(s['last_frame_time']-planned[s['name']])<=max(1e-12,abs(planned[s['name']])*1e-6)]
        rows[case]={'configuration':{'A':'baseline','B':'ALE','C':'stabilization','D':'ALE + stabilization'}[case],'verdict':verdicts[case],'logs':logs,'energy':odb['energy_audit'],'last_completed_step':max(s['number'] for s in completed),'last_saved_step_time':last['frame_time'],'last_saved_total_time':odb['energy_audit']['last']['time'],'cae_path':build['cae_path'],'job_dir':build['job_dir'],'input_diff_audit':audit['status'],'physical_snapshot_unchanged':build['physical_snapshot_unchanged'],'ale':build['ale'],'stabilization':build['stabilization']}
    assert rows['A']['logs']['area_error'] and not rows['B']['logs']['area_error']
    safety=json.loads((rep/'safety_after.json').read_text())
    assert safety['protected_hashes_unchanged'] and safety['stage3_inventory_unchanged'] and not safety['active_matrix_job_locks']
    common={}
    for case in 'ABCD':
        logs=json.loads((rep/(case+'_logs.json')).read_text())
        common[case]={str(end):{k:sum(v[k] for n,v in logs['per_step'].items() if int(n)<=end) for k in ['attempt_retries','negative_eigenvalue_warnings','equilibrium_checks']} for end in ([5,39] if case in 'AB' else [5])}
    result={'run_id':plan['run_id'],'cases':rows,'common_completed_prefix_comparison':common,'diagnostic_verdict':'NO_COMPLETE_NUMERICAL_RESCUE_IN_TESTED_MATRIX','formal_model_fixed':False,'validation_scope':'diagnostic evidence and ablation isolation, not solver/model success','case_ids':[],'parameter_sweeps':False,'warnings':['B attempt001 was invalidated because ALE was reset after the first step; terminated and preserved. B valid attempt002 repairs only ALE lifecycle with unchanged region/default settings.','A direct interactive launcher did not create native .log; native .msg/.dat/.sta and transcribed launcher console are preserved.','ALE CAE API values frequency=10/meshSweeps=1 are omitted in emitted keyword; actual Standard .msg reports five sweeps. Actual behavior, not API value, is authoritative.','C/D failed at Step6; their lower total warning/cutback counts do not demonstrate improvement, and they did not test Step40.','C/D energy ratios cover converged history only, including only the first converged stabilized increment.','Stage3 unchanged check is path/size/mtime_ns inventory, not full content hashing.']}
    (rep/'matrix_results.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    header='| CASE | Configuration | Solver | Last completed / fatal step | Area error | Distortion warnings | Cutbacks | Neg. eigen | Stabilization energy | Verdict |\n|---|---|---|---|---|---:|---:|---:|---|---|\n'
    table=header
    for c,row in rows.items():
        x=row['logs'];energy='N/A' if c in 'AB' else 'ALLSD/ALLIE = %.4f%%'%(100*row['energy']['last']['SD_IE_ratio'])
        table+='| %s | %s | %s | %s / %s | %s | %s | %s | %s | %s | %s |\n'%(c,row['configuration'],x['solver_status'],row['last_completed_step'],x['fatal_step'],'Yes: '+','.join(map(str,x['area_error_labels'])) if x['area_error'] else 'Not observed',x['excessive_distortion_warnings'],x['cutbacks'],x['negative_eigenvalue_warnings'],energy,row['verdict'])
    (rep/'comparison_table.md').write_text(table,encoding='utf8')
    print(table)

if __name__=='__main__':main()
