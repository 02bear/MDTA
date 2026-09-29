# Progressive Rank16 unfreezing with standard pair-RNC

This experiment warm-starts the standard pair-RNC adapter (`alpha=0.01`) from
the locked seed-42 stage-1 runs and replaces cached backbone features with a
raw-input differentiable Rank16 forward pass.

Two variants run concurrently:

- `unfreeze_last` on physical GPU 0: epochs 1-2 adapter only, then the Rank16
  low-rank head, fusion modules, 1D encoders, 3D output projections, and the
  third drug/protein EGNN layers.
- `unfreeze_full` on physical GPU 1: the same schedule through epoch 5, then
  every Rank16 parameter is trainable from epoch 6 onward.

Validation selection always includes the original frozen Rank16+R1 and the
locked frozen pair-RNC+R1, in addition to fine-tuned predictions at residual
scales 0, 0.25, 0.5, and 1. Test labels are accessed only after that selection
is persisted.

Remote output:

`/data1/ztx/MyModel-MDTA/outputs/Refine_experiment/davis/cold_start/drug_cold_similarity_affinity_balanced_5fold_seed42_v2_final/rank16_pcim_pair_rnc_unfreeze_20260916`
