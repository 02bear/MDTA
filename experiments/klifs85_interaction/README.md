# KLIFS-Interact-P13D

This experiment keeps the validated P13D global predictor protected and adds a
Davis-native KLIFS-85 atom-residue interaction branch.  External KLIFS data are
used only to standardize kinase pocket positions; no PDBbind contact labels are
used in version 1.

## Stage 1: KLIFS-85 preprocessing

`prepare_klifs85.py` maps each of the 85 aligned KLIFS positions through a
selected experimental PDB chain onto the exact Davis target sequence, including
its mutation state, then retrieves the corresponding AlphaFold/GVP coordinates.
The output is resumable and includes a per-protein audit CSV.

For kinases with no KLIFS experimental structure, the 85-residue reference
sequence is aligned to the full Davis sequence with a constrained gapped
alignment.  Its gap parameters were selected on 295 structure-mapped targets:
exact residue-index accuracy 99.167%, accuracy within one residue 99.307%.
Targets with truncated/non-kinase sequences or poor agreement are masked, so
their local correction is exactly zero and the frozen P13D path remains active.

```bash
/data1/ztx/.conda/envs/mdta/bin/python \
  experiments/klifs85_interaction/prepare_klifs85.py \
  --project-root /data1/ztx/MyModel-MDTA \
  --output-dir /data1/ztx/MyModel-MDTA/experiments/klifs85_interaction/data/klifs85
```

## Stage 2: zero-training similarity audit

`build_similarity_and_audit.py` computes ECFP4 Tanimoto and aligned KLIFS-85
pocket identity, then audits their product on training and validation only.  It
does not select or score the held-out test indices.

Fold-1 audit results:

- 423/442 proteins accepted; 430/442 have at least 80 mapped positions before
  the final quality gate.
- Validation joint-kernel RMSE 0.6970 versus drug-only 0.7647 and global mean
  0.8718.
- Overall edgewise monotonic smoothness is weak (Spearman +0.0305), so K2 uses
  only a weak, continuous high-similarity soft target rather than a hard rule.

## Stage 3: protected interaction experiments

- K1: frozen P13D prediction + KLIFS atom-residue local residual; affinity only.
- K2: K1 + weak joint-similarity soft contrast (`lambda=0.02`).
- The atom encoder is a two-layer GINE network over rich 97-D atom features.
- The residue branch uses both per-residue ProtT5 and AFDB/GVP geometry at the
  standardized 85 positions, followed by bilinear atom-residue attention.
- Epoch zero is the exact frozen baseline. A seed is enabled only if validation
  MSE improves; otherwise its checkpoint records a disabled local branch.

Final fold-1 decision: K1 enabled only 1/3 seeds and its paired drug-bootstrap
95% CI crossed zero; K2 enabled 0/3 seeds. K3/K4 were therefore not launched
under the predefined go/no-go rule. See `EXPERIMENT_REPORT.md`.
