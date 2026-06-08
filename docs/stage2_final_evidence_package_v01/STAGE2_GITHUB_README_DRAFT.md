# Stage 2 GitHub README Draft

This repository contains the Stage 2 evidence package for RL-LAM-ScanOpt.

## What To Include
- Scripts under `rl-training/v01/src/experiments`.
- Documentation under `docs/stage2`.
- Small CSV summaries and reports.
- Evidence manifest and key result tables.

## What To Exclude
- ODB, CAE, SIM, PRT, STA, MSG, DAT, LCK, and other Abaqus outputs.
- Large raw extraction folders.
- Cache folders and temporary Abaqus files.

## .gitignore Recommendation
Verify entries for:

```gitignore
*.odb
*.cae
*.sim
*.prt
*.sta
*.msg
*.dat
*.lck
__pycache__/
.pytest_cache/
outputs/**/large_raw/
cae_models/**/large_raw/
```

## Scientific Positioning
Stage 2 is a teacher-guided evidence package. It supports U2-first + SurfaceT-secondary constrained search and GNN/RL policy-learning feasibility. It does not claim global optimum or autonomous closed-loop physical optimisation.
