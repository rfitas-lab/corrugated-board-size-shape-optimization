#!/usr/bin/env python3
"""Generate publication figures and exact tables exclusively from saved ledgers."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from geometry_v2 import ROOT,Curve
from cbopt.optimizers import _nondominated_mask
OUT=ROOT/'research_v2/results';MAN=ROOT/'research_v2/manuscript';FIG=MAN/'figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':9,'axes.titlesize':9,'legend.fontsize':7.2,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight','figure.dpi':120})
BLUE='#1f5676';GOLD='#c08022';GREEN='#268065';RED='#b74042';GRAY='#9ba6aa'

def save(fig,name):
 fig.savefig(FIG/(name+'.pdf'));fig.savefig(FIG/(name+'.png'),dpi=180);plt.close(fig)
def axesclean(ax):ax.grid(color='#e6e9e9',lw=.6,alpha=.8);ax.set_axisbelow(True)

def write_table(frame,path,fmt='%.5f'):
 lines=[r'\begin{tabular}{'+('l'*len(frame.columns))+'}',r'\toprule',' & '.join(frame.columns)+r' \\',r'\midrule']
 for row in frame.itertuples(index=False,name=None):
  cells=[fmt%v if isinstance(v,(float,np.floating)) else str(v) for v in row]
  lines.append(' & '.join(cells)+r' \\')
 lines.extend([r'\bottomrule',r'\end{tabular}']);path.write_text('\n'.join(lines)+'\n')

def archive_figures():
 d=pd.read_csv(OUT/'C038_diagnostic_reclassification.csv');s=pd.read_csv(OUT/'archived_representatives_corrected.csv').set_index('selection_id');fig,axs=plt.subplots(1,2,figsize=(7.2,2.65))
 for ax,col,xlab in zip(axs,['board_material_fraction','medium_volume_per_area_mm'],[r'Envelope material fraction $\phi_b$',r'Medium volume / initial area $a_m$ [mm]']):
  ax.scatter(d[col],d.stress_pnorm_utilization,s=12,color=GRAY,alpha=.55,label='Historically eligible')
  f=d[d.fem_pareto.astype(bool)].sort_values(col);ax.plot(f[col],f.stress_pnorm_utilization,'o-',ms=3,lw=.6,color=GOLD,label='Original front')
  for ident,mark,color in [('S4','D',RED),('S6','*',BLUE)]:
   r=s.loc[ident];ax.scatter(r[col],r.stress_pnorm_utilization,s=80 if mark=='*' else 32,marker=mark,color=color,zorder=4);ax.annotate(ident,(r[col],r.stress_pnorm_utilization),xytext=(5,7),textcoords='offset points',fontsize=8,color=color)
  ax.set(xlabel=xlab,ylabel=r'Stress utilization $\Omega_8$');axesclean(ax)
 axs[0].set_title('Original coordinates: 31 nondominated members');axs[1].set_title('Correct medium resource: one member (S6)');axs[0].legend(loc='upper left');fig.tight_layout(w_pad=2);save(fig,'archive_resource_audit')
 d=pd.read_csv(OUT/'archived_572_corrected.csv');fig,axs=plt.subplots(1,2,figsize=(7.2,2.6));ax=axs[0];false=~d.new_radius_feasible
 ax.scatter(d.analytic_radius_mm,d.radius_min_mm,s=9,color=GRAY,alpha=.6);ax.scatter(d.loc[false,'analytic_radius_mm'],d.loc[false,'radius_min_mm'],s=21,color=RED,label='16 changed classifications',zorder=3)
 lo=min(d.analytic_radius_mm.min(),.87);hi=max(d.radius_min_mm.max(),1.1);ax.plot([lo,hi],[lo,hi],color=BLUE,lw=.7);ax.axvline(.9,color=RED,ls='--',lw=.7);ax.axhline(.9,color=RED,ls='--',lw=.7);ax.set(xlabel='Analytic minimum radius [mm]',ylabel='Historical smoothed radius [mm]');ax.legend(loc='upper left');axesclean(ax)
 ax=axs[1];ss=s.reset_index();x=np.arange(len(ss));ax.bar(x-.17,ss.radius_min_mm,width=.34,color=GRAY,label='Historical');ax.bar(x+.17,ss.analytic_radius_mm,width=.34,color=BLUE,label='Analytic');ax.axhline(.9,color=RED,ls='--',lw=.8);ax.set(xticks=x,xticklabels=ss.selection_id,ylabel='Minimum radius [mm]',ylim=(.88,1.04));ax.legend(loc='upper left',ncol=2);axesclean(ax);fig.tight_layout(w_pad=2);save(fig,'curvature_audit')


def nominal_figures():
 d=pd.read_csv(OUT/'fixed_resource_calls.csv');d=d[d.path_success.astype(bool)];fig,axs=plt.subplots(1,2,figsize=(7.2,2.75));ax=axs[0]
 for fam,color in [('nurbs',BLUE),('fourier',GOLD)]:
  g=d[d.family==fam];ax.scatter(1000*g.potential_per_footprint_N_per_mm,g.stress_pnorm_utilization,s=10,color=color,alpha=.23)
  f=g[_nondominated_mask(g[['potential_per_footprint_N_per_mm','stress_pnorm_utilization']].to_numpy()*[-1,1])].sort_values('potential_per_footprint_N_per_mm');ax.plot(1000*f.potential_per_footprint_N_per_mm,f.stress_pnorm_utilization,'o-',ms=2.8,lw=.85,color=color,label=fam.upper() if fam=='nurbs' else 'Fourier')
 g=d[d.family=='sine'];ax.scatter(1000*g.potential_per_footprint_N_per_mm,g.stress_pnorm_utilization,marker='*',s=90,color='black',label='Sinusoid',zorder=4);ax.set(xlabel=r'Stored potential / initial area $u_A$ [J m$^{-2}$]',ylabel=r'Stress utilization $\Omega_8$',title='449 direct nominal paths, matched resources');ax.legend(loc='lower right');axesclean(ax)
 ax=axs[1]
 for method,color in [('Sobol',GOLD),('NSGA-II',BLUE)]:
  hs=[pd.read_csv(p) for p in (OUT/'fixed_resource').glob(f'nurbs_{method}_*/history.csv')];v=np.array([g.hv for g in hs]);calls=hs[0].calls.to_numpy();ax.fill_between(calls,v.min(0),v.max(0),color=color,alpha=.13);ax.plot(calls,np.median(v,axis=0),'o-',ms=3,color=color,label=method)
 ax.set(xlabel='Expensive FEM path calls',ylabel='Pooled-candidate hypervolume',xticks=[16,32,48,64],title='Three paired seeds: median and range');ax.legend(loc='lower right');axesclean(ax);fig.tight_layout(w_pad=2);save(fig,'fixed_resource_fronts')
 summaries=[json.loads(p.read_text()) for p in (OUT/'fixed_resource').glob('*/summary.json')];pd.DataFrame(summaries).to_csv(OUT/'nominal_run_summary.csv',index=False)


def scenario_figures():
 r=pd.read_csv(OUT/'scenario_qualification.csv');s=pd.read_csv(OUT/'shortlist.csv');a=r[r.stage=='axial'].set_index('shortlist_id');e=r[r.stage=='enriched'].set_index('shortlist_id');ids=sorted(e.index)
 fig,axs=plt.subplots(1,2,figsize=(7.2,3.55));ax=axs[0]
 for ident in ids:
  row=e.loc[ident];color=GREEN if row.robust_feasible else RED;mark='*' if ident=='V00' else 'o';size=90 if ident=='V00' else 30
  ax.plot([1000*row.nominal_U_per_footprint,1000*row.worst_potential_per_footprint_N_per_mm],[row.nominal_Omega,row.worst_stress_utilization],color=color,alpha=.5,lw=.7)
  ax.scatter(1000*row.nominal_U_per_footprint,row.nominal_Omega,s=16,facecolor='white',edgecolor=color,lw=.7)
  ax.scatter(1000*row.worst_potential_per_footprint_N_per_mm,row.worst_stress_utilization,s=size,color=color,marker=mark,zorder=6 if ident=='V00' else 4)
  positions={'V00':(9.55,.610),'V03':(8.90,.591),'V04':(10.76,.596),'V06':(9.55,.552),'V09':(10.78,.567),'V05':(10.45,.550),'V07':(8.88,.580),'V08':(8.37,.558),'V02':(8.02,.531),'V01':(11.32,.607)};ax.annotate(ident,(1000*row.worst_potential_per_footprint_N_per_mm,row.worst_stress_utilization),xytext=positions[ident],textcoords='data',fontsize=7,arrowprops={'arrowstyle':'-','lw':.4,'color':'#737373'})
 ax.scatter([],[],s=20,color=GREEN,label='Passes all 27 scenarios');ax.scatter([],[],s=20,color=RED,label='Fails a geometry screen');ax.set(xlabel=r'Stored potential / initial area [J m$^{-2}$]',ylabel=r'Stress utilization $\Omega_8$',title='Nominal (open) to sampled extrema (filled)');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.24),frameon=False);axesclean(ax)
 ax=axs[1];x=np.arange(len(ids));ax.bar(x-.17,[a.loc[i].worst_radius_mm for i in ids],width=.34,color=GRAY,label='11 axial scenarios');ax.bar(x+.17,[e.loc[i].worst_radius_mm for i in ids],width=.34,color=BLUE,label='27 enriched scenarios');ax.axhline(.9,color=RED,ls='--',lw=.8);ax.set(xticks=x,xticklabels=[i+('*' if i=='V05' else '') for i in ids],ylabel='Smallest scenario radius [mm]',ylim=(.7,1.65),title='Minimum radius after enrichment');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.24),frameon=False);ax.text(.5,-.47,'* V05 also fails half-period monotonicity',ha='center',transform=ax.transAxes,fontsize=7);axesclean(ax);fig.tight_layout(w_pad=1.8);save(fig,'scenario_qualification')
 # Three clearly keyed exemplars: sine, nominal NURBS energy extreme, robust NURBS knee.
 g=e[(e.family=='nurbs')&e.robust_feasible];v=g[['worst_potential_per_footprint_N_per_mm','worst_stress_utilization']].to_numpy()*[-1,1];z=(v-v.min(0))/np.maximum(np.ptp(v,axis=0),1e-12);robust=g.iloc[int(np.argmin(np.linalg.norm(z,axis=1)))].name if len(g) else 'V07';chosen=list(dict.fromkeys(['V00','V01',robust]));fig,axs=plt.subplots(len(chosen),2,figsize=(7.2,1.75*len(chosen)),squeeze=False)
 for axes,ident in zip(axs,chosen):
  st=np.load(OUT/'scenarios'/f'{ident}_S00_m24.npz');nodes=st['nodes'];cur=nodes+st['displacements'][-1].reshape(-1,3)[:,:2];seg=np.stack([cur[:-1],cur[1:]],axis=1);lc=LineCollection(seg,cmap='viridis',linewidth=2,norm=plt.Normalize(0,45));lc.set_array(st['stress'][-1]);axes[0].add_collection(lc);axes[0].plot(nodes[:,0],nodes[:,1],color=GRAY,lw=.8,ls='--');axes[0].axhline(0,color=GRAY,lw=.6);axes[0].axhline(2.4,color=GRAY,lw=.6);axes[0].set(xlim=(-.2,9.5),ylim=(-.12,3.2),aspect='equal',xlabel='$x$ [mm]',ylabel='$y$ [mm]',title=f'{ident}: {s.set_index("shortlist_id").loc[ident,"family"]} at 20% strain');fig.colorbar(lc,ax=axes[0],label='Stress [MPa]',fraction=.028,pad=.02)
  paths=[np.load(p) for p in (OUT/'scenarios').glob(f'{ident}_S*_m24.npz')];F=np.stack([v['reaction'] for v in paths]);strain=st['strain'];axes[1].fill_between(strain,F.min(0),F.max(0),color=BLUE,alpha=.15,label='27-scenario range');axes[1].plot(strain,st['reaction'],'o-',ms=3,color=BLUE,label='Nominal');axes[1].set(xlabel='Engineering compression strain',ylabel='Reaction [N]');axes[1].legend(loc='upper left');axesclean(axes[1])
 fig.tight_layout(h_pad=1.8,w_pad=2);save(fig,'qualified_profiles')


def tables_and_summary():
 d=pd.read_csv(OUT/'fixed_resource_calls.csv');s=pd.read_csv(OUT/'shortlist.csv');q=pd.read_csv(OUT/'scenario_qualification.csv');runs=pd.read_csv(OUT/'nominal_run_summary.csv');sc=pd.read_csv(OUT/'scenario_results.csv');fine=pd.read_csv(OUT/'refinement_results.csv') if (OUT/'refinement_results.csv').exists() else pd.DataFrame()
 e=q[q.stage=='enriched'].copy();ax=q[q.stage=='axial'].copy();sine=d[d.family=='sine'].iloc[0]
 scalar={'nominal_calls':len(d),'nominal_successful':int(d.path_success.sum()),'shortlist_designs':len(s),'scenario_calls':len(sc),'scenario_successful':int(sc.path_success.sum()),'axial_qualified':ax[ax.robust_feasible].shortlist_id.tolist(),'enriched_qualified':e[e.robust_feasible].shortlist_id.tolist(),'axial_robust_front':ax[ax.robust_pareto].shortlist_id.tolist(),'enriched_robust_front':e[e.robust_pareto].shortlist_id.tolist(),'scenario_radius_failures':int((sc.radius_min_mm<.9).sum()),'scenario_geometry_failures':int((~sc.geometry_feasible.astype(bool)).sum()),'scenario_monotonicity_failures':int((~sc.geometry_feasible.astype(bool)&(sc.radius_min_mm>=.9)).sum()),'scenario_solver_failures':int((~sc.path_success.astype(bool)).sum()),'nominal_max_relative_discrete_resource_deficit':float(1-d.discrete_medium_volume_per_area_mm.min()/.27),'scenario_max_relative_discrete_resource_deficit':float((1-sc.discrete_medium_volume_per_area_mm/sc.medium_volume_per_area_mm).max()),'nominal_thickness_interval_mm':[float(d.thickness_mm.min()),float(d.thickness_mm.max())],'scenario_medium_resource_interval_mm':[float(sc.medium_volume_per_area_mm.min()),float(sc.medium_volume_per_area_mm.max())]}
 n=runs[runs.family=='nurbs'].pivot(index='seed',columns='method',values='hv');scalar['paired_hv_relative_difference_nsga_over_sobol_percent']=(100*(n['NSGA-II']/n['Sobol']-1)).to_dict();scalar['nurbs_mean_hv']=runs[runs.family=='nurbs'].groupby('method').hv.mean().to_dict()
 ss=s.set_index('shortlist_id');scalar['nominal_extremes_versus_sine_percent']={i:{'U':float(100*(ss.loc[i,'potential_per_footprint_N_per_mm']/sine.potential_per_footprint_N_per_mm-1)),'Omega':float(100*(ss.loc[i,'stress_pnorm_utilization']/sine.stress_pnorm_utilization-1))} for i in ['V01','V02','V03','V04','V05','V06']}
 scalar['nominal_solver_gradient_max']=float(d.maximum_normalized_projected_gradient.max());scalar['scenario_solver_gradient_max']=float(sc.maximum_normalized_projected_gradient.max());scalar['scenario_reaction_imbalance_force_fraction_max']=float(sc.maximum_reaction_imbalance_force_fraction.max());scalar['scenario_reaction_imbalance_N_max']=float(sc.maximum_absolute_reaction_imbalance_N.max())
 if len(fine):
  matched=fine[fine.mesh==32].merge(sc,on=['shortlist_id','scenario_id'],suffixes=('_32','_24'));steps=fine[fine.steps==9].merge(sc[sc.scenario_id=='S00'],on=['shortlist_id','scenario_id'],suffixes=('_9','_5'))
  for col in ['target_potential_energy_Nmm','stress_pnorm_utilization','target_reaction_N']:
   matched[col+'_relative_change']=np.abs(matched[col+'_32']/matched[col+'_24']-1);steps[col+'_relative_change']=np.abs(steps[col+'_9']/steps[col+'_5']-1)
  matched.to_csv(OUT/'mesh_comparison.csv',index=False);steps.to_csv(OUT/'step_comparison.csv',index=False)
  scalar.update({'refinement_calls':len(fine),'refinement_successful':int(fine.path_success.sum()),'mesh32_calls':len(matched),'load9_calls':len(steps),'mesh24_to32_max_relative_change':{c:float(matched[c+'_relative_change'].max()) for c in ['target_potential_energy_Nmm','stress_pnorm_utilization','target_reaction_N']},'load5_to9_max_relative_change':{c:float(steps[c+'_relative_change'].max()) for c in ['target_potential_energy_Nmm','stress_pnorm_utilization','target_reaction_N']}})
 (OUT/'research_summary.json').write_text(json.dumps(scalar,indent=2))
 # Compact publication tables.
 tb=runs[runs.family=='nurbs'].sort_values(['seed','method'])[['seed','method','expensive_calls','geometry_attempts','hv','wall_time_s']].copy();tb.columns=['Seed','Method','FEM calls','Geometry proposals','HV','Elapsed [s]'];write_table(tb,MAN/'table_budget.tex','%.4f')
 tb=s[['shortlist_id','family','thickness_mm','radius_min_mm','potential_per_footprint_N_per_mm','stress_pnorm_utilization']].copy();tb['potential_per_footprint_N_per_mm']*=1000;tb.columns=['ID','Family','$t$ [mm]','$R_{min}$ [mm]','$u_A$ [J m$^{-2}$]','$\\Omega_8$'];write_table(tb,MAN/'table_shortlist.tex')
 tb=e[['shortlist_id','worst_radius_mm','worst_potential_per_footprint_N_per_mm','worst_stress_utilization','robust_feasible','robust_pareto']].copy();tb['worst_potential_per_footprint_N_per_mm']*=1000;tb['robust_feasible']=tb.robust_feasible.map({True:'yes',False:'no'});tb['robust_pareto']=tb.robust_pareto.map({True:'yes',False:'no'});tb.columns=['ID','$\\min R$ [mm]','$\\min u_A$ [J m$^{-2}$]','$\\max\\Omega_8$','Qualified','Front'];write_table(tb,MAN/'table_qualification.tex')
 return scalar

def main():
 FIG.mkdir(exist_ok=True);archive_figures();nominal_figures()
 if (OUT/'scenario_qualification.csv').exists():scenario_figures();print(json.dumps(tables_and_summary(),indent=2))
if __name__=='__main__':main()
