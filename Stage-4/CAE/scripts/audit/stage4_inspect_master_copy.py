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
import os,json,shutil
root=r'E:\Projects\Stage-4'
report=os.path.join(root,'CAE','Reports','N32','diagnostic','run_003')
copy=os.path.join(report,'master_inspection_copy.cae')
shutil.copy2(os.path.join(root,'CAE','Models','N32','Sanity','N32_sanity.cae'),copy)
openMdb(pathName=copy)
out={}
for name,m in mdb.models.items():
 d={}
 for repo in ('steps','interactions','loads','boundaryConditions','predefinedFields','fieldOutputRequests','historyOutputRequests'):
  d[repo]={k:str(v) for k,v in getattr(m,repo).items()}
 d['states']={s.name:{repo:{k:str(v) for k,v in getattr(s,repo).items()} for repo in ('interactionStates','loadStates')} for s in m.steps.values()}
 d['sets']=list(m.rootAssembly.sets.keys())
 d['instances']={k:{'sets':list(v.sets.keys()),'surfaces':list(v.surfaces.keys())} for k,v in m.rootAssembly.instances.items()}
 out[name]=d
with open(os.path.join(report,'master_inspection.json'),'w') as f:json.dump(out,f,indent=2)
mdb.close()

