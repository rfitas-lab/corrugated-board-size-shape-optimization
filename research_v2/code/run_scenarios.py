#!/usr/bin/env python3
"""Finite perturbation qualification and explicitly charged targeted refinement."""
from __future__ import annotations
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import argparse,json,time,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import replace
import numpy as np,pandas as pd
from scipy.stats import qmc
from geometry_v2 import Curve,PerturbedCurve,ROOT,metrics
from compression_v2 import solve_curve_path,CompressionProtocol
from cbopt.optimizers import _nondominated_mask
OUT=ROOT/'research_v2/results'

def shortlist():
 source=OUT/'fixed_resource_calls.csv';out=OUT/'shortlist.csv'
 if out.exists():return pd.read_csv(out)
 d=pd.read_csv(source);d=d[d.path_success.astype(bool)].copy();selected=[]
 def add(row,role):
  key=(row.family,row.curve_parameters)
  for old in selected:
   if (old['family'],old['curve_parameters'])==key:
    old['selection_reason']+='; '+role;return
  v=row.to_dict();v['selection_reason']=role;v['shortlist_id']=f'V{len(selected):02d}';selected.append(v)
 add(d[d.family=='sine'].iloc[0],'sinusoidal reference')
 for fam in ['nurbs','fourier']:
  f=d[d.family==fam].copy();v=f[['potential_per_footprint_N_per_mm','stress_pnorm_utilization']].to_numpy()*[-1,1];f=f[_nondominated_mask(v)]
  add(f.loc[f.potential_per_footprint_N_per_mm.idxmax()],fam+' nominal energy extreme')
  add(f.loc[f.stress_pnorm_utilization.idxmin()],fam+' nominal stress extreme')
  v=f[['potential_per_footprint_N_per_mm','stress_pnorm_utilization']].to_numpy()*[-1,1];z=(v-v.min(0))/np.maximum(np.ptp(v,axis=0),1e-12)
  add(f.iloc[int(np.argmin(np.linalg.norm(z,axis=1)))],fam+' nominal normalized knee')
 f=d[d.family=='nurbs'];add(f.loc[f.radius_min_mm.idxmax()],'largest nominal NURBS radius')
 v=d[['potential_per_footprint_N_per_mm','stress_pnorm_utilization']].to_numpy()*[-1,1];f=d[_nondominated_mask(v)].sort_values('potential_per_footprint_N_per_mm')
 for i in np.unique(np.rint(np.linspace(0,len(f)-1,4)).astype(int)):add(f.iloc[i],'spread pooled nominal front')
 s=pd.DataFrame(selected);s.to_csv(out,index=False);(OUT/'shortlist_source_sha256.txt').write_text(hashlib.sha256(source.read_bytes()).hexdigest()+'\n');return s

def scenarios():
 rows=[{'scenario_id':'S00','stage':'axial','perturbation':np.zeros(5)}]
 for i in range(5):
  for sign in [-1,1]:
   z=np.zeros(5);z[i]=sign;rows.append({'scenario_id':f'S{len(rows):02d}','stage':'axial','perturbation':z})
 for z in 2*qmc.Sobol(5,scramble=True,seed=4151).random_base2(4)-1:rows.append({'scenario_id':f'S{len(rows):02d}','stage':'joint','perturbation':z})
 return rows

def evaluate(job):
 design,scenario,mesh,refinepath=job[:4];maxiter=int(job[4]) if len(job)>4 else 1800
 ident=f"{design['shortlist_id']}_{scenario['scenario_id']}_m{mesh}"+('_steps9' if refinepath else '')
 od=OUT/('refinement_retries' if maxiter!=1800 else 'refinement' if mesh!=24 or refinepath else 'scenarios');od.mkdir(exist_ok=True)
 out=od/(ident+'.json')
 if out.exists():return json.loads(out.read_text())
 z=np.asarray(scenario['perturbation']);base=Curve(design['family'],np.array(json.loads(design['curve_parameters'])),float(design['pitch_mm']),float(design['height_mm'])*(1+.03*z[0]));c=PerturbedCurve(base,.02*z[4]);t=float(design['thickness_mm'])*(1+.05*z[1]);protocol=CompressionProtocol(elastic_modulus_MPa=2899*(1+.1*z[2]),yield_stress_MPa=60*(1+.1*z[3]))
 if refinepath:protocol=replace(protocol,strains=tuple(np.linspace(0,.2,9)))
 row={'shortlist_id':design['shortlist_id'],'design_id':design['design_id'],'family':design['family'],'scenario_id':scenario['scenario_id'],'scenario_stage':scenario['stage'],'perturbation':z.tolist(),'mesh':mesh,'steps':len(protocol.strains),'solver_max_iterations':maxiter,'E_MPa':protocol.elastic_modulus_MPa,'yield_MPa':protocol.yield_stress_MPa,'shape_ripple_fraction':.02*z[4],**metrics(c,t)}
 start=time.perf_counter()
 try:
  states,m=solve_curve_path(c,t,elements_per_wavelength=mesh,protocol=protocol,solver_max_iterations=maxiter);row.update(m);nodes,_=c.nodes(mesh)
  np.savez_compressed(od/(ident+'.npz'),nodes=nodes,displacements=np.stack([s['displacement'] for s in states]),stress=np.stack([s['element_stress_MPa'] for s in states]),strain=[s['strain'] for s in states],reaction=[s['reaction_N'] for s in states],energy=[s['potential_energy_Nmm'] for s in states])
  row['discrete_medium_volume_per_area_mm']=t*np.linalg.norm(np.diff(nodes,axis=0),axis=1).sum()/c.pitch
 except Exception as e:row.update({'path_success':False,'failure':repr(e)})
 row['wall_time_s']=time.perf_counter()-start;row['potential_per_footprint_N_per_mm']=row.get('target_potential_energy_Nmm',np.nan)/(25*c.pitch)
 row['scenario_feasible']=bool(row.get('path_success',False) and row['geometry_feasible'])
 out.write_text(json.dumps(row,indent=2));return row

def aggregate():
 d=pd.DataFrame([json.loads(p.read_text()) for p in (OUT/'scenarios').glob('V*.json')]);d.to_csv(OUT/'scenario_results.csv',index=False)
 rows=[]
 for (ident,group) in d.groupby('shortlist_id'):
  for stage in ['axial','enriched']:
   g=group if stage=='enriched' else group[group.scenario_stage=='axial'];nom=g[g.scenario_id=='S00'].iloc[0]
   rows.append({'shortlist_id':ident,'family':g.family.iloc[0],'stage':stage,'scenario_count':len(g),'accepted_count':int(g.path_success.sum()),'geometry_feasible_count':int(g.geometry_feasible.sum()),'robust_feasible':bool(g.scenario_feasible.all()),'worst_radius_mm':float(g.radius_min_mm.min()),'worst_potential_per_footprint_N_per_mm':float(g.potential_per_footprint_N_per_mm.min()),'worst_stress_utilization':float(g.stress_pnorm_utilization.max()),'min_medium_volume_per_area_mm':float(g.medium_volume_per_area_mm.min()),'max_medium_volume_per_area_mm':float(g.medium_volume_per_area_mm.max()),'nominal_U_per_footprint':float(nom.potential_per_footprint_N_per_mm),'nominal_Omega':float(nom.stress_pnorm_utilization),'min_U_scenario':g.loc[g.potential_per_footprint_N_per_mm.idxmin(),'scenario_id'],'max_Omega_scenario':g.loc[g.stress_pnorm_utilization.idxmax(),'scenario_id'],'min_R_scenario':g.loc[g.radius_min_mm.idxmin(),'scenario_id']})
 r=pd.DataFrame(rows);r['robust_pareto']=False
 for stage in ['axial','enriched']:
  mask=(r.stage==stage)&r.robust_feasible;r.loc[mask,'robust_pareto']=_nondominated_mask(r.loc[mask,['worst_potential_per_footprint_N_per_mm','worst_stress_utilization']].to_numpy()*[-1,1])
 r.to_csv(OUT/'scenario_qualification.csv',index=False);return d,r

def main():
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=2);p.add_argument('--refine',action='store_true');a=p.parse_args();s=shortlist();sc=scenarios();(OUT/'scenarios_frozen.json').write_text(json.dumps([{**r,'perturbation':r['perturbation'].tolist()} for r in sc],indent=2))
 if not a.refine:jobs=[(d,r,24,False) for d in s.to_dict('records') for r in sc]
 else:
  d,r=aggregate();r=r[r.stage=='enriched'];chosen=set(['V00'])
  for fam in ['nurbs','fourier']:
   g=r[(r.family==fam)&r.robust_feasible]
   if len(g):
    v=g[['worst_potential_per_footprint_N_per_mm','worst_stress_utilization']].to_numpy()*[-1,1];vv=(v-v.min(0))/np.maximum(np.ptp(v,axis=0),1e-12);chosen.add(g.iloc[int(np.argmin(np.linalg.norm(vv,axis=1)))].shortlist_id)
  # Nominal energy and stress extremes are included by selection reason.
  chosen.update(s[s.selection_reason.str.contains('nominal energy extreme|nominal stress extreme')].shortlist_id)
  jobs=[]
  for ident in sorted(chosen):
   de=s[s.shortlist_id==ident].iloc[0].to_dict();rr=r[r.shortlist_id==ident].iloc[0]
   ids={'S00',rr.min_U_scenario,rr.max_Omega_scenario}
   for ss in sc:
    if ss['scenario_id'] in ids:jobs.append((de,ss,32,False))
   jobs.append((de,sc[0],24,True))
  (OUT/'refinement_plan.json').write_text(json.dumps([{'shortlist_id':de['shortlist_id'],'scenario_id':ss['scenario_id'],'mesh':me,'refinepath':rp} for de,ss,me,rp in jobs],indent=2))
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  for i,f in enumerate(as_completed([pool.submit(evaluate,j) for j in jobs]),1):
   rr=f.result()
   if i%10==0:print(f'{i}/{len(jobs)} calls; {rr["shortlist_id"]}/{rr["scenario_id"]} accepted={rr.get("path_success")}',flush=True)
 if not a.refine:aggregate()
 else:pd.DataFrame([json.loads(p.read_text()) for p in (OUT/'refinement').glob('V*.json')]).to_csv(OUT/'refinement_results.csv',index=False)
 print('Complete',len(jobs),'calls',flush=True)
if __name__=='__main__':main()
