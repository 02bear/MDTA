# PDBbind atom–residue supervision experiment report (2026-08-31)

## Scope

This report covers the diagnostic and typed-supervision experiments performed in
`experiments/pdbbind_residue_transfer`.  Existing DTBind graphs, PLIP labels,
checkpoints, and Davis/E2 source files were not overwritten.

## 1. Binary pair-checkpoint diagnostics

| Split | Test complexes | Correct micro AP | Label permutation AP | Atom-feature shuffle AP | Residue-feature shuffle AP | Correct macro AP |
|---|---:|---:|---:|---:|---:|---:|
| random | 297 | 0.09455 | 0.00190 | 0.02485 | 0.00258 | 0.16329 |
| scaffold-disjoint | 328 | 0.07467 | 0.00150 | 0.01957 | 0.00203 | 0.12479 |
| exact-protein-disjoint | 328 | 0.07686 | 0.00151 | 0.01818 | 0.00207 | 0.12981 |

Atom-feature shuffling reduced micro AP by 73.7%, 73.8%, and 76.3%,
respectively, and reduced per-complex AP in 93.6%, 94.5%, and 91.8% of test
complexes.  The model therefore uses atom identity, although residual AP under
atom ablation shows that protein and graph priors remain.

Top-K performance remained modest.  At K equal to the true contact count,
precision/recall was 16.1% (random), 12.6% (scaffold), and 13.6%
(exact-protein).  This checkpoint is a useful ranker but not yet a high-precision
contact predictor.

## 2. Input-feature audit and rich ligand cache

The legacy PDBbind graph builder called `Chem.Kekulize(...,
clearAromaticFlags=True)`.  Consequently all cached aromatic node flags were
zero, making a valid pi-stacking atom mask impossible.  The 82-dimensional
features also lacked formal charge, donor/acceptor, hybridization, and ring
membership.

An independent 97-dimensional rich ligand cache was built from the original
PDBbind SDFs.  It preserves aromaticity and adds formal charge, hybridization,
donor, acceptor, and ring features.  All 3,281 previously aligned samples passed
an exact per-atom element-order check.  The same 122 atom-count-mismatched
samples excluded by the earlier alignment audit failed and remain excluded.

The final chemistry-mask positive-label coverage is 100% for hydrogen bonds,
hydrophobic contacts, water bridges, pi-stacking, and halogen bonds; 99.946% for
salt bridges; and 99.930% for pi-cation contacts.  Observed positives are always
unioned into training candidates, so no training label is deleted by a mask.

## 3. Typed rich-ligand experiment

The scaffold-disjoint model was warm-started from the matching binary checkpoint
and trained for 15 epochs.  Validation selected epoch 13 by masked typed-union
macro AP.  The shared base head is retained for general contact localization;
typed heads are auxiliary semantic predictions.

| Scaffold test score | Micro AP | Macro AP |
|---|---:|---:|
| legacy binary pair checkpoint | 0.07467 | 0.12480 |
| rich model, base head | **0.10486** | **0.16785** |
| rich model, raw typed union | 0.09713 | 0.16154 |
| rich model, chemistry-masked typed union | 0.08711 | 0.15327 |

The rich base head improved micro AP by 40.4% and macro AP by 34.5% over the
legacy checkpoint.  The typed union improved over the legacy checkpoint but did
not outperform the base head, indicating that typed-head calibration/gating is
still needed before using a union score.

Typed masked micro AP on scaffold test was 0.0721 (hbond), 0.0438
(hydrophobic), 0.0222 (waterbridge), 0.2352 (saltbridge), 0.0831
(pi-stacking), 0.0434 (pi-cation), and 0.0797 (halogen bond; only 23 positives).
The last value is highly uncertain because of its very small sample count.

Post-hoc diagnostics confirmed that the gain is atom-conditioned.  Base-head AP
fell from 0.10486 to 0.02264 after atom-feature permutation (-78.4%) and to
0.00200 after residue-feature permutation (-98.1%).  Precision@5 improved from
12.9% to 20.5%, Precision@10 from 12.0% to 17.9%, and precision/recall at K
equal to the true contact count from 12.6% to 16.5%.

## 4. Current interpretation

The experiment passes the local-information test: the improved model depends on
the correct atom features and provides better early retrieval.  It is still not
a high-precision mechanistic contact predictor.  For transfer, use the shared
rich encoder and base contact head as the primary local representation; expose
typed heads as auxiliary channels or gated refinements rather than replacing the
base map with a max-over-types union.

The exact-protein-disjoint replication selected epoch 11 and showed the same
head ordering:

| Exact-protein test score | Micro AP | Macro AP |
|---|---:|---:|
| legacy binary pair checkpoint | 0.07686 | 0.12980 |
| rich model, base head | **0.12823** | **0.19800** |
| rich model, raw typed union | 0.11613 | 0.18584 |
| rich model, chemistry-masked typed union | 0.10716 | 0.17637 |

The rich base head improved micro AP by 66.8% and macro AP by 52.5%.  This split
only removes exact duplicate sequences.  A homology-clustered protein split is
not yet available because no mmseqs/cd-hit executable was found in the existing
server environments; exact sequence disjointness must not be reported as
homology disjointness.

## 5. Scaffold fine-tuning stability

Seeds 42, 43, and 44 used the same scaffold split and the same legacy binary
warm-start checkpoint.  Only typed-stage stochastic negative sampling and
optimization varied.  Validation selected epochs 13, 10, and 11, respectively.

| Seed | Base micro AP | Base macro AP | Raw typed micro AP | Raw typed macro AP | Masked typed micro AP | Masked typed macro AP |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.10486 | 0.16785 | 0.09713 | 0.16154 | 0.08711 | 0.15327 |
| 43 | 0.10631 | 0.16972 | 0.10082 | 0.16547 | 0.09228 | 0.15643 |
| 44 | 0.10726 | 0.17090 | 0.10047 | 0.16485 | 0.09093 | 0.15796 |
| Mean +/- sample SD | **0.10614 +/- 0.00121** | **0.16949 +/- 0.00154** | 0.09947 +/- 0.00203 | 0.16395 +/- 0.00211 | 0.09011 +/- 0.00268 | 0.15588 +/- 0.00239 |

Relative to the fixed legacy scaffold checkpoint (micro AP 0.07467, macro AP
0.12480), the mean base-head gain is 42.1% in micro AP and 35.8% in macro AP.
Every seed preserves the ordering base head > raw typed union > masked typed
union.  The improvement is therefore stable to typed-stage randomness, but this
is not a full multi-seed pretraining estimate because the warm-start checkpoint
and data split were held fixed.

## 6. Homology-plus-scaffold dual-cold experiment

MMseqs2 version `eec9c354be4276d2373996af2e50808b1390d527` was installed only
under this experiment's `tools/` directory.  An all-vs-all search used minimum
sequence identity 0.30, bidirectional coverage 0.80 (`cov-mode 0`), sensitivity
7.5, and `max-seqs 10000`.  Every reported qualifying hit was joined into a
protein connected component.  Samples sharing either a protein component or an
identical Bemis-Murcko scaffold were then joined into a dual connected
component before an 80/10/10 grouped split.

The resulting split contains 2,624/328/328 train/validation/test samples.  It
has zero protein-component overlap, zero scaffold overlap, and zero direct
qualifying MMseqs hits across every pair of splits.  MMseqs found 147,916
directed non-self hits and 846 protein components.  The protein/scaffold
bipartite relation is highly connected: one dual component contains 2,572
samples and necessarily dominates the training split.  Validation and test
therefore represent unusually isolated protein/scaffold components; this is a
useful stress test but not an IID estimate.  The no-overlap claim is operational
with respect to the stated MMseqs search and thresholds, not a proof that no
remote structural homology exists.

The old scaffold checkpoint was not reused because it had seen samples that
belong to the new validation/test partition.  The 82-dimensional first stage
was retrained from scratch on the dual-cold split, after which the 97-dimensional
rich/typed stage was warm-started only from that clean checkpoint.

| Dual-cold test score | Micro AP | Macro AP | Enrichment over positive rate |
|---|---:|---:|---:|
| clean 82-d first-stage base checkpoint | 0.02424 | 0.05786 | 22.6x |
| rich model before typed-stage fine-tuning | 0.02478 | 0.05941 | 23.1x |
| rich model, final base head | **0.02854** | **0.06323** | **26.6x** |
| rich model, final raw typed union | 0.02750 | 0.06103 | 25.7x |
| rich model, final masked typed union | 0.02510 | 0.05967 | 23.4x |

The rich base head improved micro AP by 15.2% and macro AP by 6.4% over its
pre-fine-tuning value, but absolute performance fell sharply relative to the
ordinary scaffold split.  The ordering base > raw typed union > masked typed
union remains unchanged.

Strict-test perturbations show that the pair map remains genuinely
atom-conditioned.  Base micro AP fell from 0.02854 to 0.01194 after within-
ligand atom-feature permutation (-58.1%), to 0.00827 after atom features were
zeroed (-71.0%), and to 0.00157 after residue-feature permutation (-94.5%).
Precision@5 is 8.60%, Precision@10 is 7.38%, and precision at K equal to the
true contact count is 7.58%; all are substantially below the ordinary scaffold
results.

At residue level, atom-feature permutation barely changed AP (0.17462 to
0.17590), while zeroing atoms reduced it to 0.10676, using a mismatched ligand
reduced it to 0.14756, and residue permutation reduced it to 0.01884.  Thus the
fine-grained pair map uses atom identity, but the aggregated binding-residue
prediction is still dominated by protein-side priors.  Any Davis transfer must
therefore expose the base pair map explicitly and include controls for a random
or mismatched ligand; transferring only the residue aggregation would mostly
transfer protein binding-site propensity rather than ligand-specific local
interaction.
