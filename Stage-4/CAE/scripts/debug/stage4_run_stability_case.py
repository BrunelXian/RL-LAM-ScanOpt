import json,pathlib,subprocess,sys,datetime,os
p=pathlib.Path(sys.argv[1]);case=sys.argv[2];plan=json.loads(p.read_text());rep=pathlib.Path(plan['report_dir']);build=json.loads((rep/(case+'_build.json')).read_text());folder=pathlib.Path(build['job_dir']);name=build['job_name']
# The caller must finish and record the previous matrix case before invoking this.
for lock in pathlib.Path(plan['job_root']).rglob('*.lck'):raise RuntimeError('Another matrix job or ODB session is active: '+str(lock))
record={'case':case,'command':['D:\\ABAQUS2024\\commands\\abaqus.bat','job='+name,'input='+name+'.inp','cpus=14','gpus=1','interactive'],'started_at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()}
log=folder/(name+'.log')
with log.open('x',encoding='utf8') as f:
 result=subprocess.run(record['command'],cwd=folder,stdout=f,stderr=subprocess.STDOUT)
record.update(returncode=result.returncode,completed_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat())
(rep/(case+'_execution.json')).write_text(json.dumps(record,indent=2),encoding='utf8')
print(json.dumps(record))
