#!/usr/bin/env python3
"""Retry six recorded mesh64 iteration-limit failures with a larger cap.

The constitutive law, geometric mesh, load states, tolerances, and acceptance
predicate are unchanged. Failed original records remain in refinement/.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json,pandas as pd
from concurrent.futures import ProcessPoolExecutor,as_completed
from run_scenarios import OUT,shortlist,scenarios,evaluate

def main():
 s=shortlist().set_index('shortlist_id');sc=scenarios();f=pd.read_csv(OUT/'refinement_results.csv');failed=f[(f.mesh==64)&~f.path_success.astype(bool)];jobs=[]
 for _,r in failed.iterrows():
  de=s.loc[r.shortlist_id].to_dict();de['shortlist_id']=r.shortlist_id;ss=next(z for z in sc if z['scenario_id']==r.scenario_id);jobs.append((de,ss,64,False,6000))
 (OUT/'fine_retry_plan.json').write_text(json.dumps({'reason':'All six mesh64 paths exhausted inherited iteration limits. Increase primary limit1800→6000 without relaxing optimizer success or normalized projected-gradient threshold.','calls':len(jobs),'max_iterations':6000},indent=2))
 with ProcessPoolExecutor(max_workers=2) as p:
  for f in as_completed([p.submit(evaluate,j) for j in jobs]):
   r=f.result();print(r['shortlist_id'],r['scenario_id'],r['path_success'],r['runtime_s'],flush=True)
 pd.DataFrame([json.loads(p.read_text()) for p in (OUT/'refinement_retries').glob('V*.json')]).to_csv(OUT/'refinement_retry_results.csv',index=False)
if __name__=='__main__':main()
