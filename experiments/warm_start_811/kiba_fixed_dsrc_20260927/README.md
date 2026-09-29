# KIBA full 8:1:1 warm start, seed 42

Server code/work directory: `/data1/ztx/MyModel-MDTA/experiments/warm_start_811/kiba_fixed_dsrc_20260927`.

Results: `/data1/ztx/MyModel-MDTA/outputs/Refine_experiment/kiba/warm_start/random_pair_811_seed42/p13d_pcim_rnc_dsrc_20260927/seed_42`.

Input split is the existing `data/splits/kiba_fixed_split_811_full.json`: 94603 train / 11825 validation / 11826 test, 2111 drugs / 229 proteins. No rows removed, no split regeneration, no affinity imputation. Existing four required feature families are reused without modification. The optional drug-2D encoder is disabled, as in Davis.

Model: original-MLP P13D (no Rank16) trained from scratch on KIBA; freeze its encoders; train bidirectional PCIM with standard RNC weight 0.01; apply fixed Davis-selected DSRC. Base: Adam lr 3e-4, weight decay 1e-5, batch 16, hidden 128, dropout 0.1, maximum 500 epochs, validation RMSE patience 60 / min delta 1e-4. Adapter: AdamW lr 3e-4, batch 16, maximum 60 epochs, validation MSE patience 10 / min delta 1e-5. Adapter architecture and lambda screening match the Davis multi-seed protocol. Final adapter lambda 0.85 and all DSRC numeric hyperparameters remain fixed.

KIBA-specific adaptation: raw label scale retained; RNC mid/high sampling thresholds are training-only 50th/75th percentiles (11.520216403 / 11.953871964). These are relative sampling strata, not a biological activity cutoff. RNC batch 32, quotas 8 high / 8 middle, remaining distinct training pairs, temperature 2, interval 2, one-epoch warmup. Task MSE + 0.001 delta-squared regularizer; active RNC step weight 0.01 * 2 * warmup. Logging includes the actual scheduled RNC weight.

DSRC uses training residuals only. Missing KIBA matrix entries are NaN, masked out of all reference sets. Drug similarity is Morgan radius2/2048-bit Tanimoto; protein similarity is accepted KLIFS85 pocket masked identity / 85, preserving the Davis definition. Exact-sequence Davis pockets are reused; others use the same Davis-calibrated KLIFS sequence fallback, with >=80 positions and >=0.75 pocket identity acceptance. Unavailable/low-quality mappings have zero protein-branch support; drug correction remains available. External pocket data use no affinity labels. Validation/test label-poisoning invariance is tested before scientific result output.

New files: setup/data-preparation/smoke/orchestration/training scripts; immutable `source/` snapshot and `source_manifest.json`; `data_audit.json`, `feature_manifest.json`; `data/klifs85/by_protein/*.pt`, `data/klifs85/raw_api/*.json`, mapping audit; `entity_similarities.npz`, `similarity_audit.json`; smoke-only artifacts under `smoke_output/`; real training checkpoints, logs, protocols, predictions and final metrics under the results directory. Existing Davis experiments and KIBA raw/base feature files are not overwritten.

`smoke_output/` predictions are software tests using an untrained model and must never be reported as experiment performance. `smoke_result.json` must exist before real training starts. Both base and adapter stages support epoch-boundary resume; `run_seed.py` runs all real stages sequentially on GPU 1. Early stopping/checkpoint selection uses validation only; tests are evaluated after their relevant checkpoints are fixed and never guide hyperparameters.

Launch from project root using `/opt/anaconda3/bin/python`, environment `CUDA_VISIBLE_DEVICES=1 WARM_GPU=1 WARM_SEED=42 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4`. Run `setup_experiment.py` once, then `prepare_data.py`, `smoke_test.py`, then `run_seed.py`. Pipeline final completion is signaled by result-directory `status.json` with state `complete` and `pcim_dsrc_fixed/result.json` with complete true.
