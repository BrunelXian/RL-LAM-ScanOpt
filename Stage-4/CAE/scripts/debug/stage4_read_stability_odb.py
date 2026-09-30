"""Read-only diagnostic ODB history and geometry audit (not teacher metrics)."""
import sys,os,json,math
from odbAccess import openOdb
from builtins import sum,min,max,abs,all,any

def main():
    plan=json.load(open(sys.argv[1]));case=sys.argv[2];rep=plan['report_dir'];b=json.load(open(os.path.join(rep,case+'_build.json')));path=os.path.join(b['job_dir'],b['job_name']+'.odb')
    result={'case':case,'odb_path':path,'read_only':True,'available':False}
    try:
        odb=openOdb(path=path,readOnly=True);result['available']=True
        result['steps']=[{'name':k,'number':v.number,'timePeriod':v.timePeriod,'frames':len(v.frames),'last_frame_time':v.frames[-1].frameValue if v.frames else None} for k,v in odb.steps.items()]
        curves=[];t0=0.0;latest=None
        for name,s in odb.steps.items():
            region={k:v for k,v in s.historyRegions.items() if k.upper().startswith('ASSEMBLY')}
            histories={}
            for rn,reg in region.items():
                for vn,out in reg.historyOutputs.items():
                    if vn in ['ALLSD','ALLIE','ALLSE','ALLPD','ALLAE','ALLKE','ALLWK','ETOTAL']:histories[vn]=list(out.data or [])
            curves.append({'step':name,'time_offset':t0,'histories':histories})
            if len(s.frames)>0:latest=(name,s.frames[-1])
            t0+=s.timePeriod
        result['history_curves']=curves
        points=[]
        for c in curves:
            h=c['histories'];maps={k:dict(v) for k,v in h.items()}
            for t,sd in h.get('ALLSD',[]):
                ie=maps.get('ALLIE',{}).get(t);se=maps.get('ALLSE',{}).get(t);pd=maps.get('ALLPD',{}).get(t)
                points.append({'step':c['step'],'time':t+c['time_offset'],'ALLSD':sd,'ALLIE':ie,'ALLSE':se,'ALLPD':pd})
        energy={'available_variables':sorted(set(k for c in curves for k in c['histories'])),'samples':len(points),'ratio_definition':'abs(ALLSD)/abs(ALLIE); companion abs(ALLSD)/(abs(ALLSE)+abs(ALLPD))','threshold':0.05}
        peak=max([abs(p['ALLIE']) for p in points if p['ALLIE'] is not None]+[0.0]);valid=[p for p in points if p['ALLIE'] is not None and abs(p['ALLIE'])>max(1e-30,peak*0.01)]
        for p in valid:
            p['SD_IE_ratio']=abs(p['ALLSD'])/abs(p['ALLIE'])
            denom=abs(p['ALLSE'] or 0)+abs(p['ALLPD'] or 0)
            p['SD_SE_PD_ratio']=abs(p['ALLSD'])/denom if denom>1e-30 else None
        if valid:
            energy.update(last=valid[-1],max_ratio=max(p['SD_IE_ratio'] for p in valid),fraction_samples_above_5pct=sum(p['SD_IE_ratio']>0.05 for p in valid)/float(len(valid)),denominator_screen='abs(ALLIE) > 1% of peak abs(ALLIE); avoids zero-energy start-up singularity')
            energy['dominated_or_suspect']=valid[-1]['SD_IE_ratio']>0.05 or energy['fraction_samples_above_5pct']>0.1
        else:energy['missing_reason']='No aligned nonzero ALLSD/ALLIE history'
        result['energy_audit']=energy
        result['error_element_sets']={}
        for k,reg in odb.rootAssembly.elementSets.items():
            if 'ERR' in k.upper() or 'DISTORT' in k.upper():
                result['error_element_sets'][k]=[[e.label for e in arr] for arr in reg.elements]
        if latest:
            name,frame=latest;inst=odb.rootAssembly.instances['INST_PLATE'];coords={n.label:list(n.coordinates) for n in inst.nodes}
            if 'U' in frame.fieldOutputs:
                for v in frame.fieldOutputs['U'].getSubset(region=inst).values:
                    coords[v.nodeLabel]=[coords[v.nodeLabel][j]+v.data[j] if j<len(v.data) else coords[v.nodeLabel][j] for j in range(len(coords[v.nodeLabel]))]
            geom=[]
            for elem in inst.elements:
                if elem.label not in [721,723,725]:continue
                xy=[coords[n][:2] for n in elem.connectivity];angles=[];jac=[]
                if len(xy)==4:
                    for xi,eta in [(-1,-1),(1,-1),(1,1),(-1,1),(0,0)]:
                        dx=[-(1-eta)/4,(1-eta)/4,(1+eta)/4,-(1+eta)/4];de=[-(1-xi)/4,-(1+xi)/4,(1+xi)/4,(1-xi)/4]
                        u=[sum(dx[j]*xy[j][q] for j in range(4)) for q in range(2)];v=[sum(de[j]*xy[j][q] for j in range(4)) for q in range(2)]
                        jac.append(u[0]*v[1]-u[1]*v[0]);norm=math.hypot(*u)*math.hypot(*v)
                        if norm>0:angles.append(math.degrees(math.acos(max(-1,min(1,sum(u[q]*v[q] for q in range(2))/norm)))))
                    area=0.5*sum(xy[j][0]*xy[(j+1)%4][1]-xy[(j+1)%4][0]*xy[j][1] for j in range(4))
                    geom.append({'element':elem.label,'signed_area':area,'minimum_sampled_jacobian':min(jac),'minimum_isoparametric_line_angle_deg':min(min(z,180-z) for z in angles),'maximum_sampled_angle_deg':max(max(z,180-z) for z in angles)})
            result['last_saved_geometry']={'step':name,'frame_time':frame.frameValue,'elements':geom,'method':'Bilinear Q4 Jacobian at corners and center using initial coordinates + saved U. This is a diagnostic reconstruction, not the internal solver reactivation check.'}
        odb.close()
    except Exception:
        import traceback;result['error']=traceback.format_exc()
    with open(os.path.join(rep,case+'_odb_diagnostic.json'),'w') as f:json.dump(result,f,indent=2)
    print('ODB_DIAGNOSTIC',case,result.get('energy_audit',{}))
if __name__=='__main__':main()

