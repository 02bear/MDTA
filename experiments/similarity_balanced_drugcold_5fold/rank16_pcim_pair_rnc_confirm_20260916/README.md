# Two-seed confirmation of pair-level RNC

This follow-up confirms the stage-1 MSE winner, standard pair-level RNC at
weight 0.01, on adapter/RNC seeds 2026 and 3407 over all five locked Davis
drug-cold folds. The frozen Rank16 backbone remains seed 42, matching stage 1.

RNC is applied to a 16D normalized projection of the 32D drug-protein pair
vector pooled after coverage Top-64 atom-residue construction and PairGraph.
The regression head still predicts the raw residual from frozen Rank16, and R1
is rebuilt from training residuals for every checkpoint and PCIM scale.

No failed stage-1 weight or high-affinity-weighted variant is repeated. The
seed-42 winner is kept as a locked historical control and is not rerun.

Remote output:

`/data1/ztx/MyModel-MDTA/outputs/Refine_experiment/davis/cold_start/drug_cold_similarity_affinity_balanced_5fold_seed42_v2_final/rank16_pcim_pair_rnc_confirm_20260916`
