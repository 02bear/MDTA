# Pair-level RNC on Rank16 + bidirectional coverage PCIM + R1

Stage 1 compares five fresh configurations on the locked five-fold Davis
drug-cold split with adapter seed 42: no RNC, standard RNC at weights 0.01 and
0.03, and high-affinity-weighted RNC at the same two weights.

RNC is applied to a 16D normalized projection of the 32D drug-protein pair
vector pooled after coverage Top-64 atom-residue construction and PairGraph.
The regression head still predicts the raw residual from frozen Rank16, and R1
is rebuilt from training residuals for every checkpoint and PCIM scale.

The high-affinity variant preserves the original continuous-label RNC ordering
and changes only anchor importance with `1 + 2*sigmoid((pKd-7)/0.5)`. It does
not collapse every high-affinity observation into one class.

Remote output:

`/data1/ztx/MyModel-MDTA/outputs/Refine_experiment/davis/cold_start/drug_cold_similarity_affinity_balanced_5fold_seed42_v2_final/rank16_pcim_pair_rnc_20260915`
