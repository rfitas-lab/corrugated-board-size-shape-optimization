#!/usr/bin/env python3
"""Numerical checks of geometry identities, one-sided extrema and resource map."""
from pathlib import Path
import json,numpy as np,pandas as pd
from geometry_v2 import Curve,from_unit,PerturbedCurve,ROOT
from cbopt.evaluator import nurbs_profile
OUT=ROOT/'research_v2/results'
rng=np.random.default_rng(4199);archive=pd.read_csv(OUT/'archived_572_corrected.csv');idx=np.unique(np.r_[rng.choice(len(archive),24,replace=False),np.where(~archive.new_radius_feasible)[0]])
rows=[]
for i in idx:
 r=archive.iloc[i];p=r[['d1','d2','d3','d4','d5','w1','w2']].to_numpy(float);c=Curve('nurbs',p,r.pitch_mm,r.height_mm);R,loc=c.radius();d=c.dense_radius(8193)
 u=np.array([.29,.41,.57,.70]);h=1e-6;fd=(c.evaluate(u+h)-c.evaluate(u-h))/(2*h);ad=c.evaluate(u,1)
 pp=p.copy();pp[:5]*=1.738;cc=Curve('nurbs',pp,r.pitch_mm,r.height_mm)
 x,y,(a,b,k)=nurbs_profile(p,sample_size=801,wavelength_mm=r.pitch_mm,amplitude_mm=r.height_mm);xy=np.column_stack([x[a:k+1]-x[a],y[a:k+1]-y[b]]);xx=c.evaluate(np.linspace(.25,.75,len(xy)))
 rows.append({'archive_index':int(i),'analytic_radius':R,'dense_radius':d,'dense_gap_relative':d/R-1,'derivative_relative_error':float(np.max(np.abs(fd-ad))/np.max(np.abs(ad))),'rescaling_max_mm':float(np.max(np.abs(c.evaluate(u)-cc.evaluate(u)))),'legacy_coordinate_max_mm':float(np.max(np.abs(xy-xx))),'periodic_tangent_gap':float(np.max(np.abs(c.evaluate(.25,1)-c.evaluate(.75,1))))})
df=pd.DataFrame(rows);df.to_csv(OUT/'geometry_verification.csv',index=False)
sine=from_unit([],family='sine');R=sine.pitch**2/(2*np.pi**2*sine.height)
assert abs(sine.radius()[0]/R-1)<1e-12
assert df.dense_gap_relative.min()>-1e-7
assert df.dense_gap_relative.max()<1e-6
assert df.derivative_relative_error.max()<1e-7
assert df.legacy_coordinate_max_mm.max()<1e-10
assert df.rescaling_max_mm.max()<1e-10
assert df.periodic_tangent_gap.max()<1e-8
s={'checked_curves':len(df),'sine_radius_formula_mm':R,'max_dense_gap_relative':float(df.dense_gap_relative.max()),'max_derivative_relative_error':float(df.derivative_relative_error.max()),'max_scale_invariance_mm':float(df.rescaling_max_mm.max()),'max_legacy_coordinate_difference_mm':float(df.legacy_coordinate_max_mm.max()),'max_periodic_tangent_difference':float(df.periodic_tangent_gap.max()),'status':'passed'}
(OUT/'geometry_verification_summary.json').write_text(json.dumps(s,indent=2));print(json.dumps(s,indent=2))
