#!/usr/bin/env python3
"""Reclassify archived evidence without altering archived data or FEM states."""
from pathlib import Path
import json,hashlib,sys
import numpy as np,pandas as pd
from geometry_v2 import Curve
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'research_v2'/'results';SRC=ROOT/'project/results/paper_a/physics_optimization'
VARS=['d1','d2','d3','d4','d5','w1','w2']

def nondom(v):
 v=np.asarray(v);return np.array([not np.any(np.all(v<=r,axis=1)&np.any(v<r,axis=1)) for r in v])

def augment(d,audit=False):
 d=d.copy();d['medium_volume_per_area_mm']=d.thickness_mm*d.arc_length_mm/d.pitch_mm
 d['equal_t_board_volume_per_area_mm']=d.height_mm*d.board_material_fraction
 d['potential_per_footprint_N_per_mm']=d.target_potential_energy_Nmm/(25*d.pitch_mm)
 d['force_per_footprint_MPa']=d.target_reaction_N/(25*d.pitch_mm)
 if audit:
  exact=[];dense=[];loc=[];arcs=[]
  for i,r in d.iterrows():
   c=Curve('nurbs',r[VARS].to_numpy(float),r.pitch_mm,r.height_mm);R,u=c.radius();exact.append(R);loc.append(u);arcs.append(c.arc_length());dense.append(c.dense_radius(1025))
  d['analytic_radius_mm']=exact;d['analytic_radius_u']=loc;d['dense_analytic_radius_mm']=dense;d['analytic_arc_length_mm']=arcs
 return d

def diagnostic_case_reclassification():
 d=augment(pd.read_csv(SRC/'fem_verified_terminal_fast.csv.gz'));summary={}
 for case,g in d.groupby('case'):
  eligible=g[g.fem_constraint_violation<=1e-12].copy()
  cols=['medium_volume_per_area_mm','potential_per_footprint_N_per_mm'] if case=='C037' else ['medium_volume_per_area_mm','stress_pnorm_utilization'] if case=='C038' else ['potential_per_footprint_N_per_mm','stress_pnorm_utilization']
  signs=[1,-1] if case=='C037' else [1,1] if case=='C038' else [-1,1]
  mask=nondom(eligible[cols].to_numpy()*signs)
  summary[case]={'eligible_under_legacy_constraints':len(eligible),'old_front':int(g.fem_pareto.sum()),'corrected_objective_front':int(mask.sum()),'common_members':int((eligible.fem_pareto&mask).sum())}
  eligible['diagnostic_corrected_front']=mask;eligible.to_csv(OUT/f'{case}_diagnostic_reclassification.csv',index=False)
 (OUT/'diagnostic_reclassification_summary.json').write_text(json.dumps(summary,indent=2))

def main():
 d=augment(pd.read_csv(SRC/'fem_unique_geometries_mesh24.csv.gz'),True);s=augment(pd.read_csv(SRC/'selected_optimized_geometries.csv'),True)
 old=pd.read_csv(SRC/'fem_verified_terminal_fast.csv.gz')
 # Archived classifications use their historical constraints unchanged.
 d['new_radius_feasible']=d.analytic_radius_mm>=.9
 d['old_radius_feasible']=d.radius_min_mm>=.9
 for name,cols,signs in [('fraction_potential',['board_material_fraction','stored_potential_number'],[1,-1]),('footprint_medium_potential',['medium_volume_per_area_mm','potential_per_footprint_N_per_mm'],[1,-1]),('footprint_medium_stress',['medium_volume_per_area_mm','stress_pnorm_utilization'],[1,1]),('height_medium_potential_stress',['height_mm','medium_volume_per_area_mm','potential_per_footprint_N_per_mm','stress_pnorm_utilization'],[1,1,-1,1])]:
  for suffix,mask in [('historical_radius',d.path_success.astype(bool)&d.old_radius_feasible),('analytic_radius',d.path_success.astype(bool)&d.new_radius_feasible)]:
   key=name+'_'+suffix;d[key]=False;d.loc[mask,key]=nondom(d.loc[mask,cols].to_numpy()*signs)
 d.to_csv(OUT/'archived_572_corrected.csv',index=False);s.to_csv(OUT/'archived_representatives_corrected.csv',index=False)
 x=s.set_index('selection_id');changes={}
 for a,b in [('S4','S6'),('S9','S7')]:
  changes[a+'_versus_'+b]={k:float(100*(x.loc[a,k]/x.loc[b,k]-1)) for k in ['board_material_fraction','height_mm','medium_volume_per_area_mm','equal_t_board_volume_per_area_mm','potential_per_footprint_N_per_mm','stored_potential_number','stress_pnorm_utilization']}
 summary={'archive_paths':len(d),'old_case_rows':len(old),'old_pareto_by_case':old[old.fem_pareto.astype(bool)].groupby('case').size().to_dict(),'successful_paths':int(d.path_success.sum()),'old_radius_feasible':int(d.old_radius_feasible.sum()),'analytic_radius_feasible':int(d.new_radius_feasible.sum()),'radius_false_feasible':int((d.old_radius_feasible&~d.new_radius_feasible).sum()),'radius_false_infeasible':int((~d.old_radius_feasible&d.new_radius_feasible).sum()),'radius_overestimation_max_percent':float(100*(d.radius_min_mm/d.analytic_radius_mm-1).max()),'radius_overestimation_median_percent':float(100*(d.radius_min_mm/d.analytic_radius_mm-1).median()),'analytic_radius_vs_dense_max_relative':float(np.max(np.abs(d.analytic_radius_mm/d.dense_analytic_radius_mm-1))),'analytic_arc_vs_archive_max_relative':float(np.max(np.abs(d.analytic_arc_length_mm/d.arc_length_mm-1))),'pair_changes_percent':changes,'front_counts':{k:int(d[k].sum()) for k in d if k.endswith('_radius')},'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [SRC/'fem_unique_geometries_mesh24.csv.gz',SRC/'fem_verified_terminal_fast.csv.gz',SRC/'selected_optimized_geometries.csv']}}
 (OUT/'archive_audit_summary.json').write_text(json.dumps(summary,indent=2));diagnostic_case_reclassification();print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
