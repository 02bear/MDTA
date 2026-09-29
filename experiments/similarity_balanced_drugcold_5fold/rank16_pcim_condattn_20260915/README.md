# Rank16 + coverage PCIM + Conditional Attention

This is a 45-run, five-fold drug-cold experiment on the locked Rank16 feature
cache. The PCIM residual remains before R1, and R1 is rebuilt from the combined
Rank16 + scaled residual training predictions.

## Controlled groups

- `baseline`: fresh rerun of the validated coverage-selected pair graph.
- `dynamic`: adds the sample-specific PCIM condition-to-key attention term.
- `shared`: identical conditional layer and parameter count, but uses one
  fold-specific shared condition derived without validation or test labels.

Every group uses adapter seeds 42, 2026, and 3407 over folds 1-5. Epoch,
lambda, optimizer, coverage selection, frozen backbone, and R1 are unchanged.

The new score is

`(q_u @ k_v + b(c) @ k_v) / sqrt(16) + relation_bias`.

`b(c)` is a zero-initialized `Linear(16, 32)`, adding 544 parameters. Thus the
conditional models are exactly equal to baseline at initialization. Post-run
diagnostics record condition-score strength, attention change, condition-off
and condition-shuffle effects, and rebuilt-R1 validation behavior.

## Remote locations

- Code: `/data1/ztx/MyModel-MDTA/experiments/similarity_balanced_drugcold_5fold/rank16_pcim_condattn_20260915`
- Results: `/data1/ztx/MyModel-MDTA/outputs/Refine_experiment/davis/cold_start/drug_cold_similarity_affinity_balanced_5fold_seed42_v2_final/rank16_pcim_condattn_20260915`
