# Local Archive and Workspace Policy

## Canonical Git Workspace

The canonical Git working tree is:

`E:\Projects\RL-LAM-ScanOpt`

All future source-code, documentation, configuration, manifest, and evidence-freeze Git operations should be performed from this workspace.

## Historical Stage 1 and Stage 2 Data

Stage 1 and Stage 2 were historically developed on the D drive.

The D-drive directories are treated as local historical and large-data archives unless their reproducibility-critical files have been intentionally copied into the canonical E-drive repository.

Exact archive path: TO_BE_VERIFIED_LOCALLY

The exact D-drive archive locations must be verified locally before being written here. Do not invent paths.

## What Belongs in Git

- source code
- tests
- documentation
- configuration
- small CSV and JSON datasets
- manifests
- checksums
- evidence summaries
- claim-boundary documents
- small reproducibility examples

## What Normally Stays Outside Git

- ODB files
- CAE working databases
- Abaqus solver outputs
- temporary job directories
- large generated case pools
- Python environments
- caches
- machine-specific scratch data
- duplicate archives

## Single Source of Truth

Do not modify independent D-drive and E-drive copies of the same source files in parallel.

The E-drive Git repository is the source of truth for all future development.

Historical D-drive files should be treated as read-only archives until explicitly consolidated.
