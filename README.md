# Corrugated-board size–shape optimization

This package contains the arXiv manuscript, geometric benchmarks, nonlinear finite-element model, dimensionless energy-consistent surrogate, and multi-objective optimization results for periodic corrugated-board profiles.

## Main study

The physics-based campaign couples a contact-conforming corotational assumed-strain beam FEM model to a dimensionless neural energy potential. Three constrained optimization cases balance material use, strain-energy storage, and a smooth high-order stress measure. Terminal particles are merged with all FEM-design geometries before direct verification. The reported pooled-candidate nondominated sets are reconstructed from the complete 572-geometry mesh-24 ledger. Three representatives per case are checked at mesh 32, and only the compromise from each case is refined at mesh 40.

The generated evidence includes:

- 80 feasible training and validation geometries and 400 nonlinear load states;
- five independent energy-network fits and a separately validated stress model;
- 540 terminal particles plus all 80 FEM designs, yielding 732 case classifications and 572 unique broad-verification geometries;
- direct mesh-24 front reconstruction, nine targeted mesh-32 checks, and three mesh-40 compromise refinements;
- independent dimensionless inputs and response groups for material, force, energy, stress, work, localization, and forming severity;
- solver, cache-fingerprint, representative-convergence, and front-membership audits.

## Reproduce

Create a Python environment and install the declared dependencies. The numerical ledgers needed to reproduce the reported nondominated sets are included, so the default verification is intentionally short:

```bash
python -m pip install -r project/requirements-paper-a.txt
python project/prepare_paper_a_data.py
python project/run_paper_a_fast_verification.py
python -m unittest discover -s tests -v
```

The preparation command verifies and, when needed, losslessly extracts the two large ledgers distributed as deterministic `.csv.gz` files. The fast command reclassifies all 732 case rows through float-safe geometry keys, reconstructs the three direct-FEM fronts, checks nine representatives at mesh 32, and refines three compromises at mesh 40. Cached accepted paths and the serialized mesh-40 states are reused, so a repeated run does not repeat FEM. `project/run_paper_a_physics_optimization.py` retains the full mechanics/optimization implementation and the superseded exhaustive campaign for audit purposes; it is not the default reproduction command and requires `--allow-exhaustive-sweep` before the discarded broad fine-mesh plan can run.

Build the manuscript with:

```bash
cd project/manuscripts/paper_a
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## Contents

- `project/src/cbfem/`: nonlinear periodic beam solver;
- `project/src/cbenergy/`: differentiable, energy-consistent dimensionless neural potential;
- `project/src/cbopt/`: geometry, mechanics, surrogate, and Pareto utilities;
- `project/run_paper_a_fast_verification.py`: short, reported verification protocol;
- `project/prepare_paper_a_data.py`: verified extraction of compressed evidence ledgers;
- `project/run_paper_a_physics_optimization.py`: mechanics/optimization engine and archived exhaustive protocol;
- `project/results/paper_a/physics_optimization/`: numerical evidence;
- `project/figures/paper_a/`: publication figures;
- `project/manuscripts/paper_a/`: arXiv source and compiled paper.

The original software is released under the MIT License. Use `CITATION.cff` when citing the software and cite the manuscript when using its scientific results.
