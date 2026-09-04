#!/usr/bin/env python3
"""Verify released counts, resources, state provenance, and failed-state exclusion.

This reads saved evidence; it does not call the mechanics solver or change any
candidate, acceptance predicate, scenario, or response value.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'research_v2/results'
CHECKS = []


def check(name, condition, **details):
    if not bool(condition):
        raise AssertionError(name + ': ' + str(details))
    CHECKS.append({'check': name, 'passed': True, **details})


def close(a, b, atol=1e-12):
    return np.allclose(a, b, rtol=1e-11, atol=atol)


def main():
    protocol_hash = hashlib.sha256((OUT / 'protocol_frozen.json').read_bytes()).hexdigest()
    check('frozen protocol SHA-256', protocol_hash == (OUT / 'protocol_sha256.txt').read_text().strip()
          == '2753e8015cdd8aeb09c7e92e3b9a03066c171504b86739c520ab4139c1f5fbdf')
    original = json.loads((ROOT / 'research_v2/v1_manifest.json').read_text())
    check('supplied v1 files unchanged', all(hashlib.sha256((ROOT / 'research_v2/v1_supplied' / n).read_bytes()).hexdigest() == h for n, h in original.items()), files=len(original))
    archive_summary = json.loads((OUT / 'archive_audit_summary.json').read_text())
    src = ROOT / 'project/results/paper_a/physics_optimization'
    check('archived FEM input hashes unchanged', all(hashlib.sha256((src / n).read_bytes()).hexdigest() == h for n, h in archive_summary['source_sha256'].items()))
    a = pd.read_csv(OUT / 'archived_572_corrected.csv')
    check('572 archived paths and 16 analytic radius violations', len(a) == 572 and a.path_success.all() and ((a.radius_min_mm >= .9) & (a.analytic_radius_mm < .9)).sum() == 16)
    c = pd.read_csv(OUT / 'C038_diagnostic_reclassification.csv')
    check('conditional C038 reclassification 31 to 1', len(c) == 129 and c.fem_pareto.sum() == 31 and c.diagnostic_corrected_front.sum() == 1)
    selected = pd.read_csv(OUT / 'archived_representatives_corrected.csv').set_index('selection_id')
    front = c[c.diagnostic_corrected_front].iloc[0]
    check('sole corrected C038 member equals S6', close(front[['pitch_mm', 'height_mm', 'thickness_mm', 'medium_volume_per_area_mm', 'stress_pnorm_utilization']].to_numpy(float), selected.loc['S6', ['pitch_mm', 'height_mm', 'thickness_mm', 'medium_volume_per_area_mm', 'stress_pnorm_utilization']].to_numpy(float)))
    n = pd.read_csv(OUT / 'fixed_resource_calls.csv')
    check('449 accepted nominal paths', len(n) == 449 and n.path_success.all() and n.geometry_feasible.all())
    check('nominal equal continuous resources and height/pitch', close(n.medium_volume_per_area_mm, .27) and close(n.pitch_mm, 7.9) and close(n.height_mm, 3.0) and n.thickness_mm.between(.15, .25).all())
    check('areal medium resource identity', close(n.medium_volume_per_area_mm, n.thickness_mm * n.arc_length_mm / n.pitch_mm))
    for seed in [4101, 4102, 4103]:
        pair = [n[(n.family == 'nurbs') & (n.seed == seed) & (n.method == method)].sort_values('call') for method in ['NSGA-II', 'Sobol']]
        check(f'paired seed {seed}: budgets and 16 identical initial designs', all(len(d) == 64 for d in pair) and pair[0].iloc[:16].curve_parameters.tolist() == pair[1].iloc[:16].curve_parameters.tolist())
        check(f'paired seed {seed}: initial responses agree', close(pair[0].iloc[:16].target_potential_energy_Nmm, pair[1].iloc[:16].target_potential_energy_Nmm) and close(pair[0].iloc[:16].stress_pnorm_utilization, pair[1].iloc[:16].stress_pnorm_utilization))
    check('shortlist source hash unchanged', hashlib.sha256((OUT / 'fixed_resource_calls.csv').read_bytes()).hexdigest() == (OUT / 'shortlist_source_sha256.txt').read_text().strip())
    sc = pd.read_csv(OUT / 'scenario_results.csv')
    gv = pd.read_csv(OUT / 'scenario_geometry_verification.csv')
    check('complete 10 by 27 scenario grid', len(sc) == 270 and sc.groupby('shortlist_id').scenario_id.nunique().eq(27).all() and sc.path_success.all())
    check('dense geometry checks preserve every classification', len(gv) == 270 and gv.classification_matches.all() and (~gv.radius_screen_pass).sum() == 32 and (~gv.monotone_halves_dense).sum() == 7 and (~sc.geometry_feasible).sum() == 39)
    q = pd.read_csv(OUT / 'scenario_qualification.csv')
    for stage, count, ids in [('axial', 11, ['V00', 'V01', 'V07', 'V08', 'V09']), ('enriched', 27, ['V00', 'V01', 'V07', 'V08'])]:
        v = q[q.stage == stage]
        check(f'{stage}: qualification count', v.scenario_count.eq(count).all() and sorted(v[v.robust_feasible].shortlist_id) == ids)
        for _, row in v.iterrows():
            d = sc[sc.shortlist_id == row.shortlist_id]
            if stage == 'axial':
                d = d[d.scenario_stage == 'axial']
            check(f'{stage}/{row.shortlist_id}: all-scenario extrema and acceptance', bool(row.robust_feasible) == bool(d.path_success.all() and d.geometry_feasible.all()) and close(row.worst_potential_per_footprint_N_per_mm, d.potential_per_footprint_N_per_mm.min()) and close(row.worst_stress_utilization, d.stress_pnorm_utilization.max()) and close(row.worst_radius_mm, d.radius_min_mm.min()))
    ref = pd.read_csv(OUT / 'refinement_results.csv')
    retry = pd.read_csv(OUT / 'refinement_retry_results.csv')
    accepted = pd.read_csv(OUT / 'accepted_resolution_states.csv')
    check('refinement ledger retains six failed default mesh64 paths', len(ref) == 39 and ref.path_success.sum() == 33 and len(ref[ref.mesh == 64]) == 6 and not ref[ref.mesh == 64].path_success.any())
    check('six distinct 6000-iteration retries accepted', len(retry) == 6 and retry.mesh.eq(64).all() and retry.path_success.all() and retry.solver_max_iterations.eq(6000).all())
    check('resolution comparison contains only accepted paths', len(accepted) == 24 and accepted.path_success.all() and accepted[accepted.mesh == 64].solver_max_iterations.eq(6000).all())
    fine = accepted[accepted.mesh == 64].set_index(['shortlist_id', 'scenario_id'])
    for _, row in retry.iterrows():
        check(f'{row.shortlist_id}/{row.scenario_id}: resolution uses accepted retry values', close(fine.loc[(row.shortlist_id, row.scenario_id), 'target_potential_energy_Nmm'], row.target_potential_energy_Nmm))
    records = [(p, p.with_name(p.name.replace('record_', 'state_')).with_suffix('.npz')) for p in (OUT / 'fixed_resource').glob('*/record_*.json')]
    for folder in ['scenarios', 'refinement', 'refinement_retries']:
        records.extend((p, p.with_suffix('.npz')) for p in (OUT / folder).glob('V*.json'))
    rejected = []
    for p, state in records:
        r = json.loads(p.read_text())
        if not state.exists():
            raise AssertionError(f'Missing saved state: {state}')
        with np.load(state, allow_pickle=False) as z:
            if not close(z['energy'][-1], r['target_potential_energy_Nmm']) or not close(z['reaction'][-1], r['target_reaction_N']):
                raise AssertionError(f'JSON/NPZ response mismatch: {p}')
            if r['path_success']:
                if not all(np.isfinite(z[k]).all() for k in z.files):
                    raise AssertionError(f'Nonfinite accepted state: {p}')
                if r['maximum_normalized_projected_gradient'] > 5e-4:
                    raise AssertionError(f'Accepted path outside gradient gate: {p}')
                if not all(s['accepted'] for s in json.loads(r['solver_diagnostics_json'])):
                    raise AssertionError(f'Accepted path contains rejected increment: {p}')
            else:
                rejected.append(str(p.relative_to(ROOT)))
    check('764 attempt records each have matching saved state evidence', len(records) == 764, accepted=758, rejected=len(rejected))
    check('all six rejected paths preserved in default refinement directory', len(rejected) == 6 and all('/refinement/' in p and '_m64.json' in p for p in rejected), paths=rejected)
    summary = {'status': 'passed', 'checks': len(CHECKS), 'verification': CHECKS, 'mechanics_calls_performed_by_this_check': 0}
    (OUT / 'evidence_verification.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'{len(CHECKS)} evidence checks passed; 764 saved attempt/state pairs verified; no new mechanics calls.')


if __name__ == '__main__':
    main()
