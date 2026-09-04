#!/usr/bin/env python3
"""Targeted follow-up to the observed V01 reaction sensitivity at mesh32.

This extension was declared after the initial refinement, to resolve a concrete
remaining numerical risk. It is not part of the frozen algorithm comparison.
The qualified stress extreme V08 is also checked at mesh32.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json,pandas as pd
from concurrent.futures import ProcessPoolExecutor,as_completed
from run_scenarios import OUT,shortlist,scenarios,evaluate
def main():
 s=shortlist().set_index('shortlist_id');sc=scenarios();jobs=[]
 for ident,mesh in [('V00',40),('V01',40),('V08',32)]:
  for sid in ['S00','S11','S24']:
   de=s.loc[ident].to_dict();de['shortlist_id']=ident;ss=next(r for r in sc if r['scenario_id']==sid);jobs.append((de,ss,mesh,False))
 (OUT/'additional_refinement_plan.json').write_text(json.dumps({'reason':'V01 mesh24-to32 reaction change up to6.35%; check mesh40 on V01 and matched sine. Also verify newly qualified stress extreme V08.','jobs':[{'shortlist_id':d['shortlist_id'],'scenario_id':ss['scenario_id'],'mesh':m} for d,ss,m,_ in jobs]},indent=2))
 with ProcessPoolExecutor(max_workers=2) as p:
  for f in as_completed([p.submit(evaluate,j) for j in jobs]):
   r=f.result();print(r['shortlist_id'],r['scenario_id'],r['mesh'],r['path_success'],flush=True)
 pd.DataFrame([json.loads(p.read_text()) for p in (OUT/'refinement').glob('V*.json')]).to_csv(OUT/'refinement_results.csv',index=False)

if __name__=='__main__':main()
