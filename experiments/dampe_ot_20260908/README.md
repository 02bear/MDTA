# DAMPE-style OT: Fold 1 frozen-representation mechanism pilot

Authorized 2026-09-08 after reviewing the proposed experiment. No baseline file is modified.

## Locked scope

- Latest Davis similarity/affinity balanced `v2_final` split, development Fold 1 only.
- Common original baseline seed42 checkpoint, epoch41 (hash in cache metadata).
- Four frozen encoders in evaluation mode, cached unique train/validation entities.
- New original `ConcatFusion` x2 and original `Decoder`, identically initialized per seed.
- F0 identity; F1 normalized drug OT; F2 normalized shuffled-correspondence drug OT;
  F3 raw drug OT; F4 drug 3D / 128. Protein mapping is identity in all five.
- Main maps use A = 128T; raw control uses T. Uniform probability marginals.
- Raw dimension-wise RMSE, epsilon 0.001, float64 log-Sinkhorn, max50k iterations,
  relative marginal tolerance 1e-7. No cost multiplication or embedding standardization.
- F2 target permutation seed20260908 is fixed across downstream seeds.
- Train-drug paired bootstrap:20 replicates, seed20260909; failures are recorded.
- Formal downstream seeds42,43,44; batch16, Adam3e-4, weight_decay1e-5, MSE,
  dropout0.1, hidden128, max500 epochs, original val-RMSE stopping60/min_delta1e-4.
- Original baseline training/evaluation functions and metrics are reused.
- Cached head training may run on CPU (one thread); this is a paired frozen protocol,
  not a claim to exactly reproduce the historical end-to-end GPU baseline.
- No test evaluation, no Fold2-5, and no automatic end-to-end OT expansion.

## Commands (run from project root)

```bash
PY=/data1/ztx/.conda/envs/mdta/bin/python
EXP=experiments/dampe_ot_20260908
CUDA_VISIBLE_DEVICES=0 $PY -B $EXP/prepare_cache.py --fold 1 --device cuda:0
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 $PY -B $EXP/test_ot_alignment.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 $PY -B $EXP/prepare_ot.py --fold 1
$PY -B $EXP/train_frozen_ot.py --fold 1 --mode F0 --seed 42 --device cpu --smoke --epochs 2
$PY -B $EXP/run_suite.py --fold 1 --device cpu
$PY -B $EXP/summarize.py --fold 1
```

Existing outputs are never overwritten by an experiment command. Inspect partial failures
before rerunning. `suite_status.json`, per-run `progress.json`, logs and summaries track status.
Best checkpoint replay is verified and saves its exact mapping buffers.

## Interpretation

F1-F0 estimates the effect of normalized OT within the frozen protocol. F1-F2 checks
whether paired entities matter. F3-F4 separates raw coupling from pure downscaling.
Validation is reused to select the original encoder and the downstream head: results are
development evidence, not independent confirmation. Three seeds vary downstream training
only. Test metrics and further protocol selection are deferred until the pilot is reviewed.

The frozen global cache replaces repeated encoder evaluation, not feature definitions.
Dataset row identity, pair/drug isolation, checkpoint split, source hashes, feature hashes,
cache identity and whole-model prediction equivalence are checked. OT only receives rows
whose IDs exactly equal the unique training-drug ID set.
