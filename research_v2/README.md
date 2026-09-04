# Paper A, version 2

**Resource-consistent shape optimization of formed corrugated media: analytic curvature and finite-scenario qualification**

This revision preserves the latest supplied manuscript as v1. It corrects the physical material coordinate, computes exact rational-curve curvature candidates, compares fixed-resource NURBS/Fourier/sinusoidal designs, and qualifies a finite scenario set. No new physical experiments were performed.

## Reproduce and inspect

Run from the repository root. Install the dependencies in `project/requirements.txt`; the baseline mechanics and vendored automatic differentiation remain in `project/src/cbfem` and `project/vendor/autograd`. Set `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `MKL_NUM_THREADS=1`.

To audit the supplied saved evidence without a new mechanics campaign:

```bash
python research_v2/code/verify_evidence.py
```

The completed check passes 51 controls and verifies 764 JSON/state pairs. The 764 attempts comprise 449 nominal paths, 270 scenario paths, 39 default-limit refinement attempts (including six rejected mesh-64 paths), and six accepted fine-mesh retries at the unchanged tolerance and an increased iteration cap. Three pilot calls are additional.

The complete replay sequence is in `manuscript/supplement.tex`, section Executable reproduction. Existing completed results are reused; a fresh campaign must use a separate output copy and preserve the frozen protocol and input records. Wall times and near-boundary classifications can vary with numerical libraries. `results/protocol_frozen.json` records the nominal search and scenario decisions; later refinement decisions and reasons remain separate JSON records.

## Main evidence

- `code/`: thirteen new scientific and verification scripts.
- `results/archive_audit_summary.json`: archived 572-geometry audit and unchanged input hashes.
- `results/fixed_resource_calls.csv`: all 449 nominal attempts and geometry-screen accounting.
- `results/scenario_results.csv` and `scenario_qualification.csv`: all 270 paths and finite-set qualification.
- `results/refinement_results.csv`, `refinement_retry_results.csv`: rejected and accepted refinement attempts.
- `results/research_summary.json`, `evidence_verification.json`: final numerical summaries and evidence checks.
- `results/fixed_resource`, `scenarios`, `refinement`, `refinement_retries`: complete state arrays and individual records.
- `manuscript/`: self-contained main and supplementary LaTeX sources, figures and compiled PDFs.
- `v1_supplied/`, `v1_manifest.json`: unchanged supplied manuscript baseline.

## Interpretation

At mesh 24 and within the 27 prescribed scenarios, the qualified NURBS energy extreme has 7.91% greater minimum stored potential per footprint than the sinusoid, with 3.38% greater maximum stress utilization. Three selected mesh-64 states preserve the sign of that trade-off, but do not establish convergence of its magnitude or of the full scenario extrema. Stored deformation potential is not measured dissipation. The three paired 64-call comparisons show no consistent NSGA-II advantage over Sobol; geometric admissibility and resource comparability are the supported contributions. Companion B supplies distinct experimental evidence; C evaluates numerical state/computation; C2 certifies finite elastica states under different mechanics.

## Build manuscripts

```bash
cd research_v2/manuscript
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex
```

The released main manuscript has nine pages and the supplement seven. The figures come from deterministic plotting of the recorded data. The authors must review the new scientific analysis and interpretation before submission.
