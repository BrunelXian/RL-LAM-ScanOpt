"""Read Standard diagnostic logs without modifying solver outputs."""
import argparse,json,re,pathlib,collections,os

def parse(folder,job,step_order):
    p=pathlib.Path(folder)
    texts={ext:(p/(job+ext)).read_text(errors='replace') if (p/(job+ext)).exists() else '' for ext in ['.msg','.dat','.sta','.log']}
    msg=texts['.msg'];combined=msg+'\n'+texts['.dat'];steps={};cur=0;inc=None
    for line in msg.splitlines():
        match=re.search(r'(?:S T E P|^\s*STEP)\s+(\d+)',line)
        if match:cur=int(match.group(1));steps.setdefault(cur,{'negative_eigenvalue_warnings':0,'excessive_distortion_warnings':0,'material_failures':0,'divergence':0,'attempt_retries':0,'equilibrium_checks':0,'area_errors':0})
        header=re.search(r'^\s*STEP\s+\d+\s+INCREMENT\s+(\d+)',line)
        if header:inc=int(header.group(1))
        if not cur:continue
        d=steps[cur];u=line.upper()
        m=re.search(r'INCREMENT\s+(\d+) STARTS\. ATTEMPT NUMBER\s+(\d+)',line)
        if m:
            inc=int(m.group(1));d['attempt_retries']+=int(m.group(2))>1
        d['negative_eigenvalue_warnings']+=bool(re.search(r'HAS\s+\d+\s+NEGATIVE EIGENVALUES',u))
        d['excessive_distortion_warnings']+=('EXCESSIVE DISTORTION' in u)
        d['material_failures']+=('PLASTICITY' in u and ('FAIL' in u or 'DID NOT' in u or 'CALCULATION AT' in u)) or ('MATERIAL' in u and 'FAILED' in u)
        d['divergence']+=('THE SOLUTION APPEARS TO BE DIVERGING' in u)
        d['equilibrium_checks']+=('CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION' in u)
        d['area_errors']+=('ERRELEMAREASMALLNEGZERO' in u or 'VERY SMALL' in u and 'AREA' in u or 'NEGATIVE AREA' in u)
    rows=[]
    for line in texts['.sta'].splitlines():
        tok=line.split()
        if len(tok)>=9 and tok[0].isdigit() and tok[1].isdigit() and re.fullmatch(r'\d+U?',tok[2]):
            try:rows.append({'step':int(tok[0]),'increment':int(tok[1]),'attempt':tok[2],'severe_discontinuity_iterations':int(tok[3]),'equilibrium_iterations':int(tok[4]),'total_iterations':int(tok[5]),'total_time':float(tok[6]),'step_time':float(tok[7]),'time_increment':float(tok[8])})
            except ValueError:pass
    accepted=[r for r in rows if not r['attempt'].endswith('U')]
    completed='THE ANALYSIS HAS BEEN COMPLETED' in msg or 'THE ANALYSIS HAS COMPLETED SUCCESSFULLY' in texts['.sta']
    fatal=bool(re.search(r'\*\*\*ERROR|ANALYSIS HAS BEEN TERMINATED|ANALYSIS HAS NOT BEEN COMPLETED',combined,re.I))
    last=accepted[-1] if accepted else None
    def total(pattern):
        m=re.search(pattern,msg,re.I);return int(m.group(1)) if m else None
    final_step=cur or (last['step'] if last else None)
    errors=[]
    for ext,txt in texts.items():
        lines=txt.splitlines()
        for i,line in enumerate(lines):
            if re.search(r'\*\*\*ERROR|ErrElemAreaSmallNegZero|WarnElemDistorted|SMALL.*AREA|NEGATIVE.*AREA|ISOPARAMETRIC|JACOBIAN',line,re.I):
                errors.append({'file':job+ext,'line':i+1,'text':'\n'.join(lines[max(0,i-1):min(len(lines),i+7)])})
    negative=sum(v['negative_eigenvalue_warnings'] for v in steps.values())
    result={'job_name':job,'solver_status':'COMPLETED' if completed else 'FAILED' if fatal else 'RUNNING_OR_UNKNOWN','last_accepted':last,'last_observed_step':final_step,'last_observed_step_name':step_order[final_step] if final_step is not None and final_step<len(step_order) else None,'fatal_step':None if completed or not fatal else final_step,'fatal_increment':None if completed or not fatal else inc,'total_increments':total(r'TOTAL OF\s+(\d+)\s+INCREMENTS'),'cutbacks':total(r'(\d+)\s+CUTBACKS IN AUTOMATIC INCREMENTATION'),'total_iterations':total(r'(\d+)\s+ITERATIONS INCLUDING CONTACT'),'negative_eigenvalue_warnings':negative,'excessive_distortion_warnings':sum(v['excessive_distortion_warnings'] for v in steps.values()),'area_error':'ERRELEMAREASMALLNEGZERO' in combined.upper() or bool(re.search('VERY SMALL.*AREA|NEGATIVE AREA',combined,re.I)),'minimum_time_increment_reached':bool(re.search(r'TIME INCREMENT REQUIRED IS LESS THAN THE MINIMUM|TIME INCREMENT.*BELOW THE MINIMUM',combined,re.I)),'wallclock_seconds':total(r'WALLCLOCK TIME \(SEC\)\s*=\s*(\d+)'),'per_step':steps,'sta_accepted_increments':len(accepted),'sta_unsuccessful_attempts':len(rows)-len(accepted),'evidence':errors,'log_files':{ext:{'exists':(p/(job+ext)).exists(),'bytes':(p/(job+ext)).stat().st_size if (p/(job+ext)).exists() else 0} for ext in texts}}
    result['fatal_step_equilibrium_checks']=steps.get(final_step,{}).get('equilibrium_checks') if not completed else None
    result['area_error_labels']=sorted(set(int(x) for x in re.findall(r'THE AREA OF ELEMENT\s+(\d+)\s+INSTANCE',combined,re.I)))
    result['distorted_element_labels']=sorted(set(int(x) for x in re.findall(r'ELEMENT\s+(?:INST_PLATE\.)?(\d+)(?:\s+INSTANCE\s+INST_PLATE)?\s+IS (?:DISTORTING|DISTORTED)',combined,re.I)))
    result['isoparametric_angle_warnings']=[{'element':int(e),'reported_min_or_max_angle_degrees':float(angle)} for e,angle in re.findall(r'ELEMENT\s+(\d+)\s+INSTANCE\s+INST_PLATE IS DISTORTED\.[\s\S]{0,230}?MIN/MAX ANGLE IS\s+([\d.Ee+-]+)',combined,re.I)]
    result['material_calculation_warnings']=sum(v['material_failures'] for v in steps.values())
    result['divergence_warnings']=sum(v['divergence'] for v in steps.values())
    result['equilibrium_checks']=sum(v['equilibrium_checks'] for v in steps.values())
    result['sta_equilibrium_iterations']=sum(r['equilibrium_iterations'] for r in rows)
    result['sta_total_iterations']=sum(r['total_iterations'] for r in rows)
    result['iteration_count_note']='Native total_iterations includes contact; sta_equilibrium_iterations sums equilibrium column over all accepted/unsuccessful attempts. equilibrium_checks counts printed convergence-check headings.'
    result['excessive_element_distortion_lines']=len(re.findall('IS DISTORTING EXCESSIVELY',msg))
    result['adaptive_mesh']={'smoothing_calls':msg.count('COMPUTING NEW MESH IN ADAPTIVE MESH DOMAIN'),'reported_mesh_sweeps':sorted(set(int(x) for x in re.findall(r'LOAD INCREMENTS AND\s+(\d+) MESH SWEEPS',msg))),'warning_lines':[x.strip() for x in msg.splitlines() if ('WARNING' in x or 'ERROR' in x) and re.search('ADAPTIVE|SMOOTH|MESH',x,re.I)]}
    completed_steps=[r['step'] for r in accepted if r['step']<final_step] if final_step else []
    last_reacts=[n for n in sorted(set(completed_steps)) if step_order[n].endswith('-react')]
    result['last_successful_reactivation']={'step':last_reacts[-1],'name':step_order[last_reacts[-1]]} if last_reacts else None
    result['failure_phase']='reactivation_geometry_before_equilibrium' if fatal and result['area_error'] and result['fatal_step_equilibrium_checks']==0 and result['last_observed_step_name'].endswith('-react') else ('analysis_failure' if fatal else None)
    result['too_many_attempts']=bool(re.search('TOO MANY ATTEMPTS MADE FOR THIS INCREMENT',msg))
    result['automatic_starting_damping_factors']=[float(x) for x in re.findall(r'ITS STARTING VALUE OF\s+([\d.Ee+-]+)',msg)]
    result['failure_start_step_time']='0 (before first equilibrium iteration)' if result['failure_phase']=='reactivation_geometry_before_equilibrium' else (last['step_time'] if last else None)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--case',required=True);a=p.parse_args();plan=json.load(open(a.plan));rep=pathlib.Path(plan['report_dir']);b=json.load(open(rep/(a.case+'_build.json')));ins=json.load(open(rep/'baseline_inspection.json'))
    result=parse(b['job_dir'],b['job_name'],ins['models']['Model-1']['step_order'])
    out=rep/(a.case+'_logs.json');out.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['per_step','evidence']},indent=2))
if __name__=='__main__':main()



