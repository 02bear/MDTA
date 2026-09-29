# Coverage-aware atom-residue selection before R1

New experiment: rank16_pcim_coverage_20260915.

Select at most 32 per-atom best residue pairs across distinct valid atoms, then
fill the remaining budget to 64 from the highest-scoring unselected pairs.
Padded pairs are excluded. No new trainable parameters are introduced.

Prediction path: frozen Rank16 + lambda * bounded PCIM delta, then R1 with
training-reference residuals rebuilt for the combined prediction.
The loss, lambda grid, early stopping, learning rate, backbone checkpoints,
five-fold split and adapter seeds are inherited from rank16_pcim_20260914.

60 fresh runs: coverage/global selection x pair_graph/pair_pool x seeds
42,2026,3407 x folds 1-5. Historical artifacts are hash verified but used only
as descriptive references. Legacy pair_pool first-epoch validation replay
differed by 0.0002466, above the predeclared 0.0001 tolerance; consequently
all 30 global-top64 controls are rerun contemporaneously. No training runs
were launched before this protocol revision.

Record validation atom coverage, maximum per-atom pair count fraction,
distinct residues, maximum per-atom attention mass, residual correlations and
pre/post R1 validation errors. Diagnostics do not change selection.

Results server directory:
 /data1/ztx/MyModel-MDTA/outputs/Refine_experiment/davis/cold_start/drug_cold_similarity_affinity_balanced_5fold_seed42_v2_final/rank16_pcim_coverage_20260915

Code server directory:
 /data1/ztx/MyModel-MDTA/experiments/similarity_balanced_drugcold_5fold/rank16_pcim_coverage_20260915

status.json: queue progress; summary.json: new results and historical controls.
runs/{variant}/seed_{seed}/fold_{fold}: checkpoints, history, predictions, results.
logs/: preflight and per-run logs.
Safety fallback is exact at lambda=0; validation selection cannot guarantee
non-degradation on test data. These are development results on the existing split.
