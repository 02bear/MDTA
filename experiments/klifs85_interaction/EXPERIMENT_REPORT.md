# KLIFS-Interact-P13D experiment report

## Scope and guardrails

- Main task: Davis drug-cold fold 1, using the existing P13D early-stop
  concatenation checkpoint as an immutable skip path.
- Validation selects whether the local branch is enabled. Epoch 0 is the exact
  frozen P13D prediction.
- Test indices and test metrics are not materialized or evaluated.
- PDBbind contact labels are not used in this version.

## New preprocessing

### KLIFS-85 mapping

Server location:

`/data1/ztx/MyModel-MDTA/experiments/klifs85_interaction/data/klifs85/`

Pipeline:

1. Map Davis target names to UniProt IDs; explicitly resolve 17 legacy aliases.
2. Query the official KLIFS API for the standardized 85-position pocket.
3. When an experimental structure exists, select by missing pocket residues,
   missing atoms, KLIFS quality score, and resolution; map
   `KLIFS position -> PDB residue -> exact Davis sequence index`.
4. For kinases without an experimental structure, use a constrained gapped
   sequence mapping calibrated on 295 structure-mapped Davis targets.
5. Retrieve the mapped AFDB/GVP coordinates and mask truncated, non-kinase, or
   low-agreement targets instead of fabricating residues.

Audit:

- Calibrated sequence fallback exact position accuracy: 99.167%.
- Accuracy within one residue: 99.307%.
- Accepted high-quality targets: 423/442.
- Targets with at least 80 mapped sites before final quality gating: 430/442.

### Compact local features

Server file:

`/data1/ztx/MyModel-MDTA/experiments/klifs85_interaction/data/klifs85_features.pt`

Each target stores fixed-size tensors at the 85 standardized sites:

- ProtT5 per-residue sequence representation: `85 x 1024` (float16).
- GVP scalar/vector features plus centered AFDB coordinates: `85 x 18`.
- Valid-position mask; all-zero mask for rejected mappings.

### Similarity matrices and zero-training audit

Server location:

`/data1/ztx/MyModel-MDTA/experiments/klifs85_interaction/data/similarity_audit_fold1/`

- Drug similarity: ECFP4 Tanimoto.
- Target similarity: exact identity over common aligned KLIFS-85 sites,
  penalized by missing-site coverage.
- Joint similarity: product of drug and target similarities.

Validation kernel audit (no test):

| Predictor | RMSE | Pearson | Spearman |
|---|---:|---:|---:|
| Global mean | 0.8718 | 0.0000 | undefined |
| Drug-only kernel | 0.7647 | 0.4856 | 0.4330 |
| Joint drug-pocket kernel | 0.6970 | 0.6137 | 0.5033 |

The joint kernel is useful, but edgewise similarity versus absolute affinity
difference is not globally monotonic (Spearman +0.0305). Therefore K2 uses a
weak continuous soft target concentrated above joint similarity 0.60, not a
hard equality constraint.

## Model conditions

- **K1**: protected P13D + two-layer rich-atom GINE + dual-view KLIFS residue
  encoders + bilinear atom-residue attention; affinity loss only.
- **K2**: K1 + joint-similarity soft contrast, weight 0.02.
- Low-quality/missing KLIFS targets have local reliability zero, hence exactly
  use the frozen P13D prediction.

## Fold-1 results

Frozen P13D validation baseline: MSE 0.489465, RMSE 0.699618, CI 0.788562,
Rm2 0.341028, Spearman 0.531947.

| Condition | Seed | Enabled | Best epoch | MSE | Relative MSE gain | CI |
|---|---:|:---:|---:|---:|---:|---:|
| K1 | 42 | no | 0 | 0.489465 | 0.000% | 0.788562 |
| K1 | 43 | yes | 7 | 0.488662 | 0.164% | 0.793698 |
| K1 | 44 | no | 0 | 0.489465 | 0.000% | 0.788562 |
| K2 | 42 | no | 0 | 0.489465 | 0.000% | 0.788562 |
| K2 | 43 | no | 0 | 0.489465 | 0.000% | 0.788562 |
| K2 | 44 | no | 0 | 0.489465 | 0.000% | 0.788562 |

K1 mean relative MSE gain is 0.055%, with only 1/3 seeds enabled. K2 has
0/3 enabled seeds. During K2 training the soft contrastive loss decreased and
train MSE improved strongly, while validation MSE worsened and the residual
standard deviation grew. Thus the failure is not an inability to optimize the
similarity objective; the learned relation does not produce a transferable
affinity correction.

### K1 seed43 paired drug bootstrap

Server location:

`/data1/ztx/MyModel-MDTA/experiments/klifs85_interaction/outputs/k1_seed43_analysis/`

- Validation drugs improved/worsened: 3/4.
- Drug-macro MSE improvement: 0.000803.
- Paired drug-bootstrap 95% CI: `[-0.01053, 0.01383]`.
- Bootstrap probability that the improvement is positive: 53.2%.

The confidence interval crosses zero broadly, so the isolated seed43 gain is
not treated as reproducible evidence.

## Go/no-go decision

Do not run K3/K4 in this version:

1. K1 fails the minimum repeatability requirement (only 1/3 seeds).
2. K2, the required similarity-aligned precursor, fails in all three seeds.
3. The only enabled K1 seed is not supported by drug-level paired bootstrap.
4. Adding view/attention consistency after these failures would test a more
   complex model without a validated affinity-improving local signal.

The preprocessing and similarity audit remain useful reusable outputs. The next
iteration should first test a directly calibrated joint-kernel correction or a
ranking/neighbor-prediction auxiliary head with the frozen baseline, before
returning to atom-residue attention-map consistency.

## Residual-kernel follow-up (fold 1)

Server location:

`/data1/ztx/MyModel-MDTA/experiments/klifs85_interaction/outputs/residual_kernel_fold1_v2/`

This experiment keeps frozen P13D as `R0` and learns no neural branch. All
hyperparameters are selected by five inner drug-cold folds over the 47 outer
training drugs. The seven outer validation drugs are evaluated once after
selection; outer test indices and metrics remain inaccessible.

- `R1`: predict P13D residuals using ECFP4-similar training drugs for the same
  target, with support/variance shrinkage and clipping.
- `R2`: supplement the same-target estimate using KLIFS-85-similar targets.
- `R3`: repeat the active KLIFS configuration with 20 shuffled protein
  identities as a negative control.

The corrected R2 search explicitly contains `same_weight=1.0`, which is the
no-KLIFS boundary. It also reports the best genuinely KLIFS-active candidate
separately, preventing model selection from being forced to accept KLIFS.

| Condition | MSE | Relative MSE gain vs R0 | CI | Rm2 |
|---|---:|---:|---:|---:|
| R0 frozen P13D | 0.489465 | 0.000% | 0.788567 | 0.341029 |
| R1 drug residual kernel | 0.461358 | 5.742% | 0.797643 | 0.379291 |
| R2 selected | 0.461358 | 5.742% | 0.797643 | 0.379291 |
| R2 active (5% KLIFS) | 0.461946 | 5.622% | 0.797816 | 0.378335 |

Inner CV selected R1 parameters `gamma=2`, `k_drug=8`, `tau=0.05`, no
variance penalty, and unit clipping/scale. For R2 it selected the no-KLIFS
boundary. The best active candidate used 95% same-target signal and 5% KLIFS
cross-target signal, but was already worse than R1 in inner CV by 0.0000295
MSE and remained worse on outer validation by 0.000589 MSE.

For R1, five of seven validation drugs improved. Drug-level paired bootstrap
gave a macro MSE improvement of 0.028108, 95% CI `[0.007135, 0.050914]`, and
99.85% probability of a positive improvement. Thus the residual-correction
signal passes the fold-1 pilot gate.

The active real-KLIFS condition beat all 20 shuffled-identity controls
(real MSE 0.461946; shuffled mean 0.462175), showing that KLIFS is more
meaningful than random protein smoothing. However, it still failed the stricter
incremental-value test because R1 alone was better. Therefore KLIFS should not
be added to this residual estimator in its current cross-target averaging form;
the next confirmatory experiment is five-fold evaluation of R1, not expansion
of the KLIFS neural interaction branch.

## Five-fold confirmation

The fold-1 pilot gate was passed, so the same leakage-safe procedure was run on
folds 2--5. Each fold used its own frozen P13D checkpoint and regenerated its
own 30,056-row prediction cache. All cache audits report
`test_metrics_computed=false`; factorized versus direct-forward maximum error
was at most `4.77e-7`.

| Fold | R0 MSE | R1 MSE | Relative gain | R0 CI | R1 CI | Active KLIFS minus R1 MSE |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.489465 | 0.461358 | 5.743% | 0.788567 | 0.797643 | +0.000589 |
| 2 | 0.677205 | 0.613616 | 9.390% | 0.812850 | 0.819464 | -0.002751 |
| 3 | 0.772518 | 0.739921 | 4.220% | 0.720205 | 0.730943 | +0.000034 |
| 4 | 0.474335 | 0.456672 | 3.724% | 0.795554 | 0.797334 | +0.000710 |
| 5 | 0.733410 | 0.669230 | 8.751% | 0.557700 | 0.637836 | +0.000852 |

R1 improved MSE and CI in all five folds. Pooled over 35 validation-drug
instances, MSE changed from 0.629387 to 0.588159, a 6.550% relative gain.
Twenty-seven drug instances improved, seven worsened, and one was unchanged.
Because four validation drugs recur across folds, the primary aggregate
uncertainty calculation clusters repeat occurrences into 31 unique drugs. Its
macro MSE improvement is 0.038821 with 95% bootstrap CI
`[0.016584, 0.065165]` and 99.996% probability of positive improvement.

KLIFS cross-target smoothing was selected by inner CV in only folds 2 and 5,
and the genuinely active KLIFS estimate beat R1 on outer validation only in
fold 2 (1/5 folds). Therefore the five-fold evidence supports the
same-target drug-residual mechanism, while rejecting the current KLIFS
cross-target averaging mechanism as a reliable incremental component.

Final five-fold summary:

`/data1/ztx/MyModel-MDTA/experiments/klifs85_interaction/outputs/residual_kernel_5fold_summary.json`
