"""Stage4 numerical stability ablation, one case per invocation, build only.
A preserves baseline model settings. B/C/D each independently copy the baseline.
No solver submission in this script; caller runs jobs sequentially after build audit.
"""
from abaqus import *
from abaqusConstants import *
from part import *
from material import *
from section import *
from assembly import *
from step import *
from interaction import *
from load import *
from mesh import *
from job import *
from builtins import sum,all,any
import argparse,os,sys,json,hashlib,shutil,re

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def write(p,d):
    with open(p+'.tmp','w') as f:json.dump(d,f,indent=2)
    os.replace(p+'.tmp',p)
def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--case',choices=['A','B','C','D'],required=True)
    a=p.parse_args(sys.argv[sys.argv.index('--plan'):]);plan=json.load(open(a.plan));c=a.case
    names={'A':'A_baseline','B':'B_ALE','C':'C_STAB','D':'D_ALE_STAB'}
    cae_names={'A':'A_baseline','B':'B_ALE','C':'C_stabilization','D':'D_ALE_stabilization'}
    attempt=plan.get('attempts',{}).get(c,1)
    suffix=('_%02d'%attempt) if attempt>1 else ''
    out=os.path.join(plan['model_dir'],'stage4_N32_diag_'+cae_names[c]+suffix+'.cae')
    base_hash=sha(plan['baseline'])
    if os.path.exists(out):
        if c!='A' or sha(out)!=base_hash:raise RuntimeError('Refuse existing output')
    else:
        with open(plan['baseline'],'rb') as src,open(out,'xb') as dst:shutil.copyfileobj(src,dst)
    db=openMdb(pathName=out);m=db.models['Model-1']
    folder=os.path.join(plan['job_root'],names[c]+suffix);os.makedirs(folder,exist_ok=True);os.chdir(folder)
    jobname='stage4_diag_'+names[c]+suffix+'_'+plan['run_id']
    if os.path.exists(jobname+'.inp'):raise RuntimeError('Refuse existing job input')
    sys.path.insert(0,os.path.join(os.path.dirname(os.path.dirname(sys.argv[sys.argv.index('-noGUI')+1])),'audit'))
    from stage4_audit_n32_deposition_model import snapshot
    before=snapshot(m)
    changes={'case':c,'baseline':plan['baseline'],'baseline_sha256':base_hash,'solver':'Abaqus/Standard','job_name':jobname,'cae_path':out,'job_dir':folder,'ale':None,'stabilization':None}
    if c in ('B','D'):
        inst=m.rootAssembly.instances['inst_plate']
        labels=sorted(set(e.label for i in range(1,33) for e in inst.sets['set_pool_%d'%i].elements))
        removed=set(e.label for e in inst.sets['set_addi_all'].elements)
        assert not removed.intersection(labels)
        assert all(str(inst.elements.sequenceFromLabels(labels=(i,))[0].type)=='CPE4T' for i in labels)
        region=m.rootAssembly.Set(name='diag_ALE_pool_band',elements=inst.elements.sequenceFromLabels(labels=tuple(labels)))
        ale_steps=list(m.steps.keys())[list(m.steps.keys()).index('Step-01-preheat'):]
        for step_name in ale_steps:
            m.steps[step_name].AdaptiveMeshDomain(region=region,frequency=10,meshSweeps=1)
        changes['ale']={'region':'diag_ALE_pool_band','source_sets':['set_pool_%d'%i for i in range(1,33)],'labels':labels,'start_step':'Step-01-preheat','steps':ale_steps,'frequency':10,'meshSweeps':1,'controls':'default','initialMeshSweeps':'API default 5 (Explicit-only keyword parameter; Standard ignores initial-sweep control)','removed_element_overlap':0,'purpose':'always-active substrate pool strip next to reactivation interface; no deposit removal/addition in ALE domain'}
    if c in ('C','D'):
        selected=json.load(open(os.path.join(plan['report_dir'],'stabilization_selection.json')))['steps']
        for name in selected:
            m.steps[name].setValues(stabilizationMethod=DISSIPATED_ENERGY_FRACTION,stabilizationMagnitude=2e-4,adaptiveDampingRatio=0.05,continueDampingFactors=False)
        changes['stabilization']={'method':'DISSIPATED_ENERGY_FRACTION','magnitude':2e-4,'adaptiveDampingRatio':0.05,'continueDampingFactors':False,'steps':selected}
    after=snapshot(m)
    # Snapshot excludes stabilization/ALE controls; all original physical settings
    # and lifecycle data represented here must remain exactly unchanged.
    assert before==after,'Physical snapshot changed outside allowed controls'
    changes['physical_snapshot_unchanged']=True
    changes['step_controls']={name:{'stabilizationMethod':str(s.stabilizationMethod),'stabilizationMagnitude':s.stabilizationMagnitude,'adaptiveDampingRatio':s.adaptiveDampingRatio,'adaptiveMeshDomains':{k:str(v) for k,v in s.adaptiveMeshDomains.items()}} for name,s in m.steps.items() if name!='Initial'}
    j=db.Job(name=jobname,model='Model-1',type=ANALYSIS,numCpus=14,numDomains=14,numGPUs=1,memory=90,memoryUnits=PERCENTAGE)
    j.writeInput(consistencyChecking=ON)
    if c!='A':db.save()
    db.close()
    changes['cae_sha256']=sha(out);changes['inp_sha256']=sha(os.path.join(folder,jobname+'.inp'));changes['status']='built'
    assert sha(plan['baseline'])==base_hash
    write(os.path.join(plan['report_dir'],c+'_build.json'),changes)
    print('BUILT',c,jobname)
if __name__=='__main__':main()
