# Corrugated-Board Size and Shape Optimization

Reproducible NURBS-based multi-objective optimization of periodic corrugated-board profiles. This repository separates archived reconstruction from new computation and reports size-only, shape-only, control-point-dimensionality, radius-constrained, and coupled size--shape cases.

The accompanying manuscript is **“Multi-Objective Size and Shape Optimization of Corrugated-Board Profiles Using NURBS and MO-ETPSO.”** Both a neutral two-column PDF and an Elsevier-layout PDF are included under `project/manuscripts/paper_a/`.

## What is included

- A seven-variable rational quadratic B-spline evaluator with explicit units and curvature screening.
- Corrected MO-ETPSO-R and NSGA-II implementations with equal-budget, paired-seed comparison.
- Reconstructed public cases C001--C033 and newly executed coupled cases C034--C036.
- Twelve Pareto/design plates, with three keyed solutions and numerical design-vector tables per plate.
- Hyperparameter and convergence data, source/version audit tables, and publication-ready LaTeX figures.

## Scientific status

The public archive is retained as historical evidence. Its original radius branches returned `True` in both cases, so the stored front is not retroactively described as radius constrained. The corrected runs use continuous radius violation, standard Pareto dominance, and a frozen evaluator manifest. The paired ten-seed benchmark does not establish a universal optimizer winner; it establishes equivalence for the declared case, scale, and evaluation budget.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r project/requirements.txt
export PYTHONPATH="$PWD/project"
export MPLCONFIGDIR=/tmp/corrugated-board-mpl
```

Run a short optimizer demonstration:

```bash
python project/run_paper_a.py --seeds 2 --population 20 --generations 20
```

Rebuild the archived atlas and new coupled cases:

```bash
python project/make_paper_a_gallery.py
python project/run_coupled_size_shape.py
```

Reproduce the manuscript benchmark and hyperparameter study:

```bash
python project/run_paper_a.py --seeds 10 --population 50 --generations 80
python project/revision_studies.py paper_a
```

Outputs are written to `project/results/paper_a/` and `project/figures/paper_a/`.

## Compile the paper

```bash
cd project/manuscripts/paper_a
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

The numeric-citation Elsevier wrapper is `elsevier.tex`; the required class and bibliography style are bundled under `project/tex/elsevier/`.

## Repository map

- `project/src/cbopt/`: NURBS evaluator and optimizers.
- `project/data/paper_a/`: archived and new final fronts.
- `project/results/paper_a/`: audit, histories, seeds, metrics, and manifests.
- `project/figures/paper_a/`: vector and raster publication figures.
- `project/manuscripts/`: LaTeX source, bibliography, and compiled PDFs.
- `tests/`: fast evaluator smoke test.

## Test

```bash
python -m unittest discover -s tests -v
```

## Citation and licensing

Use `CITATION.cff` for software citation and cite the manuscript when using its scientific results. Original code in this package is MIT licensed. Archived public data and third-party LaTeX assets retain their source terms; see `DATA_AND_THIRD_PARTY_NOTICE.md`.
