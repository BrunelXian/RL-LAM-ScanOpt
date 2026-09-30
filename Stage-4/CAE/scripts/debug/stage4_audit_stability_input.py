"""Compare emitted input decks, permitting only ALE/stabilization keywords."""
import sys,pathlib,json,re,difflib

def normalize(path):
    lines=path.read_text().splitlines();out=[];skip=False;changes=[]
    for line in lines:
        s=line.strip()
        if not s or s.startswith('**'):continue
        if s.startswith('*'):
            skip=False
            lo=s.lower()
            if lo.startswith('*elset') and 'diag_ale_pool_band' in lo:
                skip=True;changes.append(s);continue
            if lo.startswith('*adaptive mesh'):
                skip=True;changes.append(s);continue
            if lo.startswith('*coupled temperature-displacement'):
                parts=[p.strip() for p in s.split(',')]
                kept=[p for p in parts if not p.lower().startswith(('stabilize','allsdtol','continue'))]
                if kept!=parts:changes.append(s)
                s=','.join(kept)
        if not skip:out.append(s.lower().replace(' ',''))
    return out,changes
plan=json.loads(pathlib.Path(sys.argv[1]).read_text());case=sys.argv[2];rep=pathlib.Path(plan['report_dir']);a=json.loads((rep/'A_build.json').read_text());b=json.loads((rep/(case+'_build.json')).read_text());pa=pathlib.Path(a['job_dir'])/(a['job_name']+'.inp');pb=pathlib.Path(b['job_dir'])/(b['job_name']+'.inp');la,_=normalize(pa);lb,changes=normalize(pb)
diff=list(difflib.unified_diff(la,lb,fromfile='A',tofile=case));report={'case':case,'status':'passed' if not diff else 'failed','allowed_numerical_changes':changes,'unexpected_diff':diff,'meaning':'All input lines identical after removal of job comments and explicitly allowed ALE/stabilization definitions; original mesh/material/BC/load/interaction/duration/output data unchanged.'};(rep/(case+'_input_diff_audit.json')).write_text(json.dumps(report,indent=2))
print(report['status'],'allowed lines',len(changes),'unexpected',len(diff));assert not diff,diff[:20]

# Independently inspect the emitted per-step controls, including explicit resets.
blocks=re.split(r'(?im)^\*step\s*,',pb.read_text())[1:]
ale_steps=[];stabilized=[];current_domain=None;failures=[]
for block in blocks:
    name=re.search(r'(?i)name=([^,\r\n]+)',block).group(1).strip()
    for line in block.splitlines():
        lo=line.lower().strip()
        if lo.startswith('*adaptive mesh,'):
            match=re.search(r'elset=([^,]+)',lo)
            current_domain=match.group(1).strip() if match else None
        if lo.startswith('*coupled temperature-displacement') and 'stabilize' in lo:
            stabilized.append(name)
            if '0.0002' not in lo and '2e-04' not in lo and '2.e-04' not in lo:
                failures.append('Unexpected stabilization magnitude: '+line)
    if current_domain:ale_steps.append(name)
expected_ale=b['ale']['steps'] if b['ale'] else []
expected_stab=b['stabilization']['steps'] if b['stabilization'] else []
if ale_steps!=expected_ale:failures.append('ALE step lifecycle differs from build intent')
if stabilized!=expected_stab:failures.append('Stabilization step scope differs from build intent')
report.update(ale_effective_steps=ale_steps,stabilized_steps=stabilized,numerical_control_failures=failures)
report['status']='passed' if not diff and not failures else 'failed'
(rep/(case+'_input_diff_audit.json')).write_text(json.dumps(report,indent=2))
print('Control lifecycle',report['status'],'ALE steps',len(ale_steps),'stabilized steps',len(stabilized));assert not failures,failures
