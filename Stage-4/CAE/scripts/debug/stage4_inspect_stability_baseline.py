"""Read-only capability inspection on an exclusive byte copy; no solver."""
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
from builtins import sum
import sys,os,json,shutil,collections
plan=json.load(open(sys.argv[sys.argv.index('--plan')+1]))
out=os.path.join(plan['model_dir'],'stage4_N32_diag_A_baseline.cae')
if not os.path.exists(out):
    with open(plan['baseline'],'rb') as src,open(out,'xb') as dst:shutil.copyfileobj(src,dst)
db=openMdb(pathName=out)
d={'version':str(db.version),'models':{},'jobs':{k:str(v) for k,v in db.jobs.items()}}
for name,m in db.models.items():
 inst=m.rootAssembly.instances['inst_plate']
 types=collections.Counter(str(e.type) for e in inst.elements)
 sets={k:{'count':len(v.elements),'types':dict(collections.Counter(str(e.type) for e in v.elements)), 'labels':[e.label for e in v.elements]} for k,v in inst.sets.items()}
 d['models'][name]={'element_types':dict(types),'sets':sets,'steps':{k:str(v) for k,v in m.steps.items()},'step_order':list(m.steps.keys()),'loads':{k:str(v) for k,v in m.loads.items()},'interactions':{k:str(v) for k,v in m.interactions.items()},'history_requests':{k:str(v) for k,v in m.historyOutputRequests.items()},'materials':{k:str(v) for k,v in m.materials.items()}}
with open(os.path.join(plan['report_dir'],'baseline_inspection.json'),'w') as f:json.dump(d,f,indent=2)
db.close()

