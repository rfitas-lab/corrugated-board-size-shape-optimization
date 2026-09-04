#!/usr/bin/env python3
"""Independent dense derivative/envelope checks of all saved scenario geometries."""
from pathlib import Path
import json,numpy as np,pandas as pd
from geometry_v2 import ROOT,Curve,PerturbedCurve
OUT=ROOT/'research_v2/results'
s=pd.read_csv(OUT/'shortlist.csv').set_index('shortlist_id');d=pd.read_csv(OUT/'scenario_results.csv');rows=[]
for _,r in d.iterrows():
 de=s.loc[r.shortlist_id];z=np.array(json.loads(r.perturbation));base=Curve(de.family,np.array(json.loads(de.curve_parameters)),de.pitch_mm,de.height_mm*(1+.03*z[0]));c=PerturbedCurve(base,.02*z[4]);R,u=c.radius();dense=c.dense_radius(8193)
 uu=np.linspace(c.breaks[0],c.breaks[-1],32769);xy=c.evaluate(uu);v=c.evaluate(uu,1)[:,1];half=len(uu)//2;monotone=bool(np.all(v[:half+1]<=1e-9) and np.all(v[half:]>=-1e-9));radiuspass=R>=.9
 rows.append({'shortlist_id':r.shortlist_id,'scenario_id':r.scenario_id,'analytic_refined_radius_mm':R,'dense_radius_mm':dense,'dense_radius_relative_gap':dense/R-1,'radius_screen_pass':bool(radiuspass),'monotone_halves_dense':monotone,'upper_envelope_overshoot_mm':max(float(xy[:,1].max()-c.height),0.),'lower_envelope_overshoot_mm':max(float(-xy[:,1].min()),0.),'classification_matches':bool((radiuspass and monotone)==r.geometry_feasible),'radius_ledger_relative_difference':abs(R/r.radius_min_mm-1)})
f=pd.DataFrame(rows);f.to_csv(OUT/'scenario_geometry_verification.csv',index=False)
assert f.classification_matches.all()
assert f.radius_ledger_relative_difference.max()<1e-10
assert f.dense_radius_relative_gap.min()>-1e-7
assert f.dense_radius_relative_gap.max()<1e-6
summary={'scenario_geometries_checked':len(f),'classification_matches':int(f.classification_matches.sum()),'radius_screen_failures':int((~f.radius_screen_pass).sum()),'half_period_monotonicity_failures':int((~f.monotone_halves_dense).sum()),'either_geometry_failure':int((~f.radius_screen_pass|~f.monotone_halves_dense).sum()),'dense_radius_max_relative_gap':float(f.dense_radius_relative_gap.max()),'radius_reconstruction_max_relative_difference':float(f.radius_ledger_relative_difference.max()),'max_envelope_overshoot_mm':float(f[['upper_envelope_overshoot_mm','lower_envelope_overshoot_mm']].max().max()),'status':'passed'}
(OUT/'scenario_geometry_verification_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
