# Paper A, version 2

Resource-consistent shape optimization of formed corrugated media: analytic curvature and finite-scenario qualification

This research revision treats the latest supplied manuscript as v1. It contains new analyses and numerical experiments, with no new physical tests. The main branch and earlier manuscript versions remain available.

## Complete evidence and reproduction

Download [the v2 research overlay](releases/Paper_A_v2_Research_Overlay.zip) and extract it at the root of this repository. It retains the relative paths of all 1674 new or updated research files, including computed states, ledgers, figures, manuscripts and audit protocols. Readable source and protocol files are also available directly in this branch. Files shared with the base repository are supplied by this checkout; this overlay is not a stand-alone copy of the entire repository.

```bash
python -m zipfile -e releases/Paper_A_v2_Research_Overlay.zip .
```

Begin with [research_v2/README.md](research_v2/README.md) after extraction. Python and LaTeX dependencies and exact replay commands are declared in the research documentation. Measured wall times and numerical solver paths can vary with hardware and software. The frozen protocols, unsuccessful attempts and limitations are part of the evidence.

## Provenance

Remote base commit: `9144fee32a0d98be2fc6229b5a94f3658807d0d6`.

Local completed research snapshot: `dfc89f58d25a4b4f8da95b75ada3ac418e861ff3`. For archive-provisioned repositories this local snapshot is not the original remote ancestry. The new branch commit uses the actual remote base as its parent.

Overlay SHA-256: `a7ea587a04dd85df84f1a0a8e4b7f33b8af6e3f867b1df05a3d01e73f0fd7abb`.

Every overlay member is listed in `V2_A_FILE_MANIFEST.json`, with byte counts and hashes. The four companion papers distinguish experimental observations, numerical design, continuous-state computation and local-state certification; their models and validation claims are not interchangeable.
