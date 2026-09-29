# Rank16 + bidirectional atom-residue cross-attention + coverage PCIM + R1

The new interaction block is inserted after the frozen Rank16 local encoders
and their 32-dimensional projections, but before coverage-aware Top-64
atom-residue pair construction.

## Feature provenance

- Atom states: 128D `node_feat` from the frozen Rank16
  `drug_3d_encoder(..., return_node=True)`, projected to 32D.
- Residue states: 128D `node_feat` from the frozen Rank16
  `protein_3d_encoder(..., return_node=True)`, concatenated with an 8D residue
  type embedding and projected to 32D.
- Rank16 global drug/protein states are excluded from the new block. They stay
  only in the already validated PCIM condition gate and pooling layers.

Atoms query residues to obtain protein-conditioned atom states; residues query
atoms to obtain drug-conditioned residue states. Each update is residual and
bounded by `0.25*tanh(gamma)`, with `gamma=0` initially, so all cross-attention
variants are exactly the baseline before learning.

## Fresh comparisons

- `bidirectional`: both updates enabled.
- `atom_conditioned`: only atoms query residues.
- `residue_conditioned`: only residues query atoms.
- `baseline`: no cross-attention.

Each variant uses adapter seeds 42, 2026, and 3407 on five drug-cold folds: 60
fresh runs. Frozen backbone, coverage Top-64 selection, pair graph, optimizer,
validation-selected lambda, and rebuilt R1 are otherwise unchanged.

## Remote output

`/data1/ztx/MyModel-MDTA/outputs/Refine_experiment/davis/cold_start/drug_cold_similarity_affinity_balanced_5fold_seed42_v2_final/rank16_pcim_bidir_xattn_20260915`
