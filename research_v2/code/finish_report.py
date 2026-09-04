#!/usr/bin/env python3
"""Finalize numerical-resolution reporting without rerunning mechanics."""
from pathlib import Path
import json,platform,hashlib
import numpy as np,pandas as pd
from geometry_v2 import ROOT
from build_outputs import OUT,MAN,FIG,BLUE,GRAY,plt,save,axesclean,write_table

def main():
 s=json.loads((OUT/'research_summary.json').read_text());fine=pd.read_csv(OUT/'refinement_results.csv');retry=pd.read_csv(OUT/'refinement_retry_results.csv');scenario=pd.read_csv(OUT/'scenario_results.csv')
 fields=['target_potential_energy_Nmm','stress_pnorm_utilization','target_reaction_N']
 m40=fine[(fine.mesh==40)&fine.path_success].merge(fine[(fine.mesh==32)&fine.path_success],on=['shortlist_id','scenario_id'],suffixes=('_40','_32'))
 m64=retry[retry.path_success].merge(fine[(fine.mesh==40)&fine.path_success],on=['shortlist_id','scenario_id'],suffixes=('_64','_40'))
 for field in fields:
  m40[field+'_relative_change']=m40[field+'_40']/m40[field+'_32']-1;m64[field+'_relative_change']=m64[field+'_64']/m64[field+'_40']-1
 m40.to_csv(OUT/'mesh40_comparison.csv',index=False);m64.to_csv(OUT/'mesh64_comparison.csv',index=False)
 s['mesh40_calls']=int((fine.mesh==40).sum());s['mesh64_default_calls']=int((fine.mesh==64).sum());s['mesh64_default_failures']=int(((fine.mesh==64)&~fine.path_success).sum());s['mesh64_retry_calls']=len(retry);s['mesh64_retry_accepted']=int(retry.path_success.sum());s['total_comparative_and_qualification_attempts']=int(s['nominal_calls']+s['scenario_calls']+len(fine)+len(retry));s['total_rejected_solver_paths']=int((~fine.path_success).sum()+(~retry.path_success).sum())
 s['mesh32_to40_max_relative_change']={f:float(np.abs(m40[f+'_relative_change']).max()) for f in fields};s['mesh40_to64_max_relative_change']={f:float(np.abs(m64[f+'_relative_change']).max()) for f in fields}
 u=retry.pivot(index='scenario_id',columns='shortlist_id',values='potential_per_footprint_N_per_mm');o=retry.pivot(index='scenario_id',columns='shortlist_id',values='stress_pnorm_utilization');s['mesh64_v01_versus_sine_potential_percent']=(100*(u.V01/u.V00-1)).to_dict();s['mesh64_v01_versus_sine_stress_percent']=(100*(o.V01/o.V00-1)).to_dict()
 (OUT/'research_summary.json').write_text(json.dumps(s,indent=2))
 # Only accepted high-resolution states enter a response comparison.
 high=pd.concat([scenario[(scenario.shortlist_id.isin(['V00','V01']))&(scenario.scenario_id.isin(['S00','S11','S24']))],fine[(fine.shortlist_id.isin(['V00','V01']))&fine.path_success&(fine.steps==5)],retry[retry.path_success]],ignore_index=True)
 high.to_csv(OUT/'accepted_resolution_states.csv',index=False)
 fig,axs=plt.subplots(1,3,figsize=(7.2,2.6))
 for ax,field,title in zip(axs,fields,['Stored potential','Stress utilization','Reaction']):
  for ident,color in [('V00',GRAY),('V01',BLUE)]:
   g=high[(high.shortlist_id==ident)&(high.scenario_id=='S00')].sort_values('mesh');base=g[g.mesh==24][field].iloc[0];ax.plot(g.mesh,100*(g[field]/base-1),'o-',ms=3,color=color,label=ident)
  ax.set(xlabel='Elements per period',ylabel='Change from mesh 24 [%]',title=title,xticks=[24,32,40,64]);axesclean(ax)
 axs[0].legend();fig.tight_layout(w_pad=1.4);save(fig,'resolution_sensitivity')
 tb=retry[['shortlist_id','scenario_id','potential_per_footprint_N_per_mm','stress_pnorm_utilization','target_reaction_N','maximum_normalized_projected_gradient']].sort_values(['shortlist_id','scenario_id']).copy();tb['potential_per_footprint_N_per_mm']*=1000;tb.columns=['ID','Scenario','$u_A$ [J m$^{-2}$]','$\\Omega_8$','$F$ [N]','Normalized PG'];write_table(tb,MAN/'table_mesh64.tex','%.6f')
 res=r'''The nominal search, scenario qualification, targeted refinement and fine-mesh retries together contain 764 expensive path attempts, in addition to three disclosed pilot calls. All 449 nominal and 270 scenario solver paths satisfy the inherited algorithmic checks. The initial targeted refinement and its first extension supply 21 mesh-32 paths, six mesh-40 paths and six nine-point load paths. Maximum 24--32 changes are 1.20\% in potential, 0.334\% in stress utilization and 6.35\% in reaction. The reaction sensitivity of V01 motivated the additional fine checks.

All six default mesh-64 paths reached the inherited iteration limits without satisfying the full success predicate. They remain recorded as rejected. Six retries increased the primary iteration cap from 1800 to 6000, leaving the mechanics, load protocol, tolerances and success predicate unchanged; all six then passed. No rejected high-resolution response enters the comparisons.

The sine changes by at most 0.171\% in potential between meshes 40 and 64. For V01, potential changes by 1.83--1.85\%, stress utilization by 0.65--0.89\%, and reaction by 0.58--0.77\% in absolute value over this interval. At the three checked mesh-64 scenarios, its potential advantage over the matched sine remains positive, 13.11--14.33\%, with 4.87--6.07\% greater stress utilization. These are checks at the original nominal and extremal scenarios, not a repeated 27-scenario scan at mesh 64. The sign of the interpreted trade-off persists, while its magnitude remains resolution-dependent. The 7.91\% and 3.38\% results therefore specifically describe the mesh-24 finite-scenario calculation, rather than converged continuum values.

Increasing the nominal load path from five to nine points changes terminal potential by at most 0.000098\%, stress utilization by 0.0253\% and reaction by 0.290\% in the six checks. Continuous resource matching is exact by construction; the largest polygonal-mesh resource deficit is 0.758\% at mesh 16 and 0.335\% at mesh 24. Thus small differences near the nominal front must be read alongside mesh sensitivity. The finer evidence, failed attempts and state arrays are retained in the supplement and numerical package.
'''
 (MAN/'resolution_results.tex').write_text(res)
 supp=r'''The solver ledger contains 764 comparative/qualification attempts: 449 nominal paths, 270 scenario paths, 39 default-limit refinement paths and six high-resolution retries. Six of the 39 default refinement paths were rejected at mesh 64. All six retries passed with the primary iteration cap increased to 6000; tolerances and the acceptance predicate were unchanged. Three timing/geometry pilot calls are additional to this total.

Figure~\ref{fig:resolution} shows nominal response sensitivity for the two profiles used in the interpreted energy comparison. The inspected fine results retain the energy advantage and stress cost of V01, but the magnitude is mesh-dependent. Since only selected scenarios were refined, the mesh-64 values are not new worst-case coordinates over the full 27-scenario set.
\begin{figure}[ht]
\centering\includegraphics[width=\textwidth]{resolution_sensitivity.pdf}
\caption{Accepted nominal-state results relative to mesh 24. Mesh 64 uses the separately recorded successful retries. Failed default-limit paths are excluded.}\label{fig:resolution}
\end{figure}
\begin{table}[ht]\centering\small
\caption{Accepted mesh-64 responses at the nominal and two originally extremal scenarios. S11 minimizes potential and S24 maximizes stress in the original mesh-24 qualification of both listed designs.}
\resizebox{\textwidth}{!}{\input{table_mesh64}}
\end{table}

At mesh 24, the maximum normalized projected gradient in the 270 scenario paths is $1.500\times10^{-5}$, below the prescribed $5\times10^{-4}$ acceptance threshold. The largest upper--lower reaction imbalance is 0.01988 N, or $6.095\times10^{-5}$ of the yield-force scale. These residual checks assess numerical equilibrium, not an experimentally observed configuration or global minimizer.

Independent dense geometry verification reproduces all 270 scenario classifications. It finds 32 radius violations and seven half-period monotonicity violations, with no overlap. Maximum relative radius discrepancy is $1.678\times10^{-7}$ against 8193 points per smooth span. The seven monotonicity failures concern V05; its largest envelope overshoot is 0.00133 mm. The nominal resource deficit caused by polygonal discretization is at most 0.758\% for the 449 mesh-16 candidates and 0.335\% for scenario mesh-24 curves. These values and the larger local force sensitivity prevent a blanket mesh-convergence claim for all candidates.

The initially fixed refinement rule was extended only after V01's observed mesh-32 reaction sensitivity. The extensions and their reasons are stored in \texttt{additional\_refinement\_plan.json}, \texttt{final\_mesh\_check\_plan.json} and \texttt{fine\_retry\_plan.json}. The nominal selection, scenario vectors and geometric admissibility rules were unchanged. The campaign ended after the six documented retries.
'''
 (MAN/'supplement_results.tex').write_text(supp)
 print(json.dumps(s,indent=2))
if __name__=='__main__':main()
