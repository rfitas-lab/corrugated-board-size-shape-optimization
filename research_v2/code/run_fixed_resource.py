#!/usr/bin/env python3
"""Frozen-budget direct FEM comparison, with atomic per-call evidence."""
from __future__ import annotations
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import argparse,json,time,hashlib,platform
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np,pandas as pd
from scipy.stats import qmc
from geometry_v2 import from_unit,metrics,ROOT
from compression_v2 import solve_curve_path,CompressionProtocol
from cbopt.optimizers import _nondominated_mask,_fast_fronts,_crowding,_select_nsga,_sbx,_polynomial_mutation,_hypervolume_2d
OUT=ROOT/'research_v2/results'; SEEDS=[4101,4102,4103]; BUDGET=64; POP=16; AM=.27;MESH=16
U0=2.530711193024368/(25*7.9)
PROTOCOL={'version':'2026-09-04-v2-fixed-resource-1','pitch_mm':7.9,'height_mm':3.,'medium_volume_per_initial_footprint_mm':AM,'thickness_bounds_mm':[.15,.25],'radius_limit_mm':.9,'strain_path':[0,.05,.1,.15,.2],'material':{'E_MPa':2899.,'G_over_E':1/55,'yield_MPa':60.,'hardening':.02,'width_mm':25},'search_mesh':MESH,'nurbs_seeds':SEEDS,'calls_per_method_seed':BUDGET,'population':POP,'algorithms':['Sobol','NSGA-II'],'paired_initial_feasible_designs':POP,'nurbs_parameters':'x1..x4=exp(4(z-.5)), x5=1; weight parameters=z5,z6','fourier_parameters':'a=.24(z1-.5),b=.16(z2-.5)','fourier_seed':4110,'fourier_calls':64,'sine_calls':1,'objectives':['maximize U/(bP)','minimize Omega8'],'hypervolume_coordinates':['U_sine_mesh16/U','Omega8'],'hypervolume_reference':[2.,1.],'scenario_axes':['height +/-3%','thickness +/-5%','E and G +/-10%','yield +/-10%','shape ripple +/-0.02 H'],'scenario_training':'nominal plus ten signed axial perturbations','scenario_enrichment':'16 scrambled Sobol joint perturbations, seed 4151','shortlist_rule':'sine; U extreme, stress extreme, normalized knee in each pooled family; max radius among NURBS; 4 evenly spread pooled nominal front members, deduplicated by family+parameters','qualification_mesh':24,'refinement_mesh':32,'scenario_interpretation':'finite deterministic sensitivity set, not a probability model','note':'Recorded after timing/geometry pilot, before comparative FEM solves. No external preregistration.'}

def jsonable(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,(np.integer,np.floating,np.bool_)):return x.item()
 raise TypeError(type(x).__name__)

def admissible(z,family):
 c=from_unit(z,family);L=c.arc_length();t=AM*c.pitch/L;m=metrics(c,t)
 ok=bool(m['geometry_feasible'] and .15<=t<=.25)
 return ok,c,t,m

class Evaluator:
 def __init__(self,method,seed,family,out):
  self.method=method;self.seed=seed;self.family=family;self.out=out;out.mkdir(parents=True,exist_ok=True);self.records=[];self.attempts=0;self.seen=set();self.start=time.perf_counter()
 def precheck(self,z):
  self.attempts+=1; key=tuple(np.round(z,12))
  if key in self.seen:return None
  ok,c,t,m=admissible(z,self.family)
  if not ok:return None
  self.seen.add(key);return c,t,m
 def run(self,z,prepared):
  c,t,g=prepared;idx=len(self.records);start=time.perf_counter()
  row={'method':self.method,'seed':self.seed,'family':self.family,'call':idx+1,'design_id':f'{self.family}_{self.method}_{self.seed}_{idx+1:03d}','unit_parameters':json.dumps(np.asarray(z).tolist()),'curve_parameters':json.dumps(c.parameters.tolist()),**g}
  try:
   states,m=solve_curve_path(c,t,elements_per_wavelength=MESH);row.update(m)
   nodes,_=c.nodes(MESH)
   np.savez_compressed(self.out/f'state_{idx+1:03d}.npz',nodes=nodes,displacements=np.stack([s['displacement'] for s in states]),stress=np.stack([s['element_stress_MPa'] for s in states]),strain=[s['strain'] for s in states],reaction=[s['reaction_N'] for s in states],energy=[s['potential_energy_Nmm'] for s in states])
   row['discrete_medium_volume_per_area_mm']=t*np.linalg.norm(np.diff(nodes,axis=0),axis=1).sum()/c.pitch
  except Exception as e:row.update({'path_success':False,'failure':repr(e)})
  row['wall_time_s']=time.perf_counter()-start;row['geometry_attempts_cumulative']=self.attempts
  row['potential_per_footprint_N_per_mm']=row.get('target_potential_energy_Nmm',np.nan)/(25*c.pitch)
  ok=bool(row['path_success']);U=row['potential_per_footprint_N_per_mm'];omega=row.get('stress_pnorm_utilization',np.nan)
  f=np.array([U0/U,omega]) if ok and U>0 and np.isfinite(omega) else np.array([1e6,1e6])
  row['objective_1']=float(f[0]);row['objective_2']=float(f[1]);self.records.append(row)
  (self.out/f'record_{idx+1:03d}.json').write_text(json.dumps(row,default=jsonable,indent=2))
  if (idx+1)%16==0:print(f'{self.family} {self.method} seed{self.seed}: {idx+1}/{BUDGET}, {self.attempts} geometry attempts',flush=True)
  return f,ok

def tournament(rng,F):
 fronts=_fast_fronts(F,np.ones(len(F),bool));rank=np.zeros(len(F),int);crowd=np.zeros(len(F))
 for i,fr in enumerate(fronts):rank[fr]=i;crowd[fr]=_crowding(F,fr)
 def choose():
  a,b=rng.integers(len(F),size=2)
  return a if (rank[a],-crowd[a])<(rank[b],-crowd[b]) else b
 return choose

def run_one(job):
 method,seed,family=job;out=OUT/'fixed_resource'/f'{family}_{method}_{seed}'
 completed=out/'summary.json'
 if completed.exists():return json.loads(completed.read_text())
 e=Evaluator(method,seed,family,out);d=6 if family=='nurbs' else 2;sampler=qmc.Sobol(d,scramble=True,seed=seed);pool=sampler.random_base2(15);poolidx=0;rng=np.random.default_rng(seed+10000)
 def next_sobol():
  nonlocal poolidx
  while poolidx<len(pool):
   z=pool[poolidx];poolidx+=1;prep=e.precheck(z)
   if prep is not None:return z,prep
  raise RuntimeError('Feasible Sobol pool exhausted')
 X=[];F=[];success=[]
 for _ in range(POP):
  z,prep=next_sobol();f,ok=e.run(z,prep);X.append(z);F.append(f);success.append(ok)
 X=np.array(X);F=np.array(F);history=[]
 history.append({'calls':len(e.records),'hv':_hypervolume_2d(F,np.array(success),np.array([2.,1.]))})
 for generation in range(1,BUDGET//POP):
  CX=[];CF=[];choose=tournament(rng,F);tries=0
  while len(CX)<POP:
   if method=='Sobol':z,prep=next_sobol()
   else:
    a,b=choose(),choose();z,_=_sbx(rng,X[a],X[b]);z=_polynomial_mutation(rng,z,np.zeros(d),np.ones(d));prep=e.precheck(z);tries+=1
    if prep is None:
     if tries<10000:continue
     z,prep=next_sobol()
   f,ok=e.run(z,prep);CX.append(z);CF.append(f)
  if method=='NSGA-II':
   xx=np.vstack([X,CX]);ff=np.vstack([F,CF]);chosen=_select_nsga(ff,np.all(ff<1e6,axis=1),POP);X=xx[chosen];F=ff[chosen]
  allf=np.array([[r['objective_1'],r['objective_2']] for r in e.records]);feas=np.array([r['path_success'] for r in e.records]);history.append({'calls':len(e.records),'hv':_hypervolume_2d(allf,feas,np.array([2.,1.]))})
 df=pd.DataFrame(e.records);df.to_csv(out/'calls.csv',index=False);pd.DataFrame(history).to_csv(out/'history.csv',index=False)
 summary={'method':method,'seed':seed,'family':family,'expensive_calls':len(df),'successful_paths':int(df.path_success.sum()),'geometry_attempts':e.attempts,'hv':history[-1]['hv'],'wall_time_s':time.perf_counter()-e.start}
 completed.write_text(json.dumps(summary,indent=2));return summary

def sine():
 out=OUT/'fixed_resource'/'sine_reference';out.mkdir(parents=True,exist_ok=True)
 if (out/'calls.csv').exists():return
 c=from_unit([],family='sine');e=Evaluator('reference',0,'sine',out);e.run(np.zeros(0),(c,AM*c.pitch/c.arc_length(),metrics(c,AM*c.pitch/c.arc_length())));pd.DataFrame(e.records).to_csv(out/'calls.csv',index=False)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=2);parser.add_argument('--freeze-only',action='store_true');a=parser.parse_args()
 txt=json.dumps(PROTOCOL,sort_keys=True,indent=2);p=OUT/'protocol_frozen.json'
 if p.exists() and p.read_text()!=txt:raise RuntimeError('Frozen protocol differs')
 p.write_text(txt);(OUT/'protocol_sha256.txt').write_text(hashlib.sha256(txt.encode()).hexdigest()+'\n')
 if a.freeze_only:print('Frozen',hashlib.sha256(txt.encode()).hexdigest());return
 jobs=[(m,s,'nurbs') for s in SEEDS for m in ['Sobol','NSGA-II']]+[('Sobol',4110,'fourier')]
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  for f in as_completed([pool.submit(run_one,j) for j in jobs]):print(json.dumps(f.result()),flush=True)
 sine();allcalls=pd.concat([pd.read_csv(p) for p in (OUT/'fixed_resource').glob('*/calls.csv')],ignore_index=True);allcalls.to_csv(OUT/'fixed_resource_calls.csv',index=False)
 print('Completed',len(allcalls),'calls',flush=True)
if __name__=='__main__':main()
