#!/usr/bin/env python3
"""Resolve the observed mesh40 energy sensitivity of qualified V01 at mesh64."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json,pandas as pd
from concurrent.futures import ProcessPoolExecutor,as_completed
from run_scenarios import OUT,shortlist,scenarios,evaluate

def main():
 s=shortlist().set_index('shortlist_id');sc=scenarios();jobs=[]
 for ident in ['V00','V01']:
  for sid in ['S00','S11','S24']:
   de=s.loc[ident].to_dict();de['shortlist_id']=ident;ss=next(r for r in sc if r['scenario_id']==sid);jobs.append((de,ss,64,False))
 (OUT/'final_mesh_check_plan.json').write_text(json.dumps({'reason':'V01 mesh32-to40 stored potential change reached2.37%; evaluate the interpreted scenario states atmesh64 along with matched sine.','jobs':[{'shortlist_id':d['shortlist_id'],'scenario_id':ss['scenario_id'],'mesh':m} for d,ss,m,_ in jobs]},indent=2))
 with ProcessPoolExecutor(max_workers=2) as p:
  for f in as_completed([p.submit(evaluate,j) for j in jobs]):
   r=f.result();print(r['shortlist_id'],r['scenario_id'],r['mesh'],r['path_success'],flush=True)
 pd.DataFrame([json.loads(p.read_text()) for p in (OUT/'refinement').glob('V*.json')]).to_csv(OUT/'refinement_results.csv',index=False)
if __name__=='__main__':main()
