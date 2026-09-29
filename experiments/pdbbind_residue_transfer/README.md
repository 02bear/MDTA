# PDBbind residue-to-Davis transfer pilot

This experiment tests whether a pose-free, ligand-conditioned residue-contact
predictor pretrained with PDBbind/PLIP labels contains pair-specific information
that transfers to the Davis drug-cold task.

The experiment is intentionally staged:

1. Audit existing DTBind PLIP labels and graph caches.
2. Validate a small ligand-conditioned residue predictor against protein-only
   and mismatched-ligand controls.
3. Only after passing local-interaction checks, freeze the predictor and test a
   capacity-matched residual head on Davis fold 1.

No existing MyModel-MDTA files or DTBind data are modified.

