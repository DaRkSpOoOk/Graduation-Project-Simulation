# TASK-009B — Core-28 LSTM Baseline: Results and Ablation Analysis

## Objective

Measure whether the frozen TASK-006 virtual-glove representation carries
signer-independent Arabic letter identity, using a deliberately modest LSTM
baseline over the frozen `task009a_sequence_input_v1` input contract, and
quantify what each sensor family and each design decision contributes.

Every number below was recomputed from the persisted artifacts in the external
run root. Nothing was trained, retrained or modified in this analysis.

## Dataset

Frozen TASK-008 Core-28 virtual-glove dataset: 4,222 sequences, 83,659 frames,
28 Arabic letters (SignIDs 0032–0059), 3 signers, sequence lengths 9–70 frames.
Leave-one-signer-out with the frozen TASK-008B splits.

| Fold | Held-out | train | validation | test |
| --- | --- | ---: | ---: | ---: |
| S01 | signer 01 | 2,372 | 448 | 1,402 |
| S02 | signer 02 | 2,373 | 448 | 1,401 |
| S03 | signer 03 | 2,355 | 448 | 1,419 |

Validation uses the two non-held-out signers; the held-out signer appears only in
test.

## Frozen TASK-009A contract

`task009a_sequence_input_v1`, unchanged. Hand order LEFT, RIGHT; per-hand channel
order bend (finger-major × chain-minor), spread (adjacent pairs), palm quaternion
WXYZ. Masks all use one polarity, True = real/valid/present. Original sequence
lengths preserved — no cropping, no fixed window, no resampling. No fitted
normalization, so nothing can leak across a LOSO boundary.

## Model and training protocol

```
values ‖ feature_valid  [B,T,2D] → LSTM(192, 2 layers) → masked mean over real frames
                                 → dropout 0.3 → Linear(192→28)
```

521,500 parameters. Input 92/76/60 channels for `full`/`bend_spread`/`bend_only`.
AdamW lr 1e-3, weight decay 1e-4, gradient clip 5.0, batch 32, max 60 epochs,
early stopping patience 12, unweighted cross-entropy (class imbalance ratio
1.074), seed 1337.

**Checkpoint selection was validation macro F1 only.** The held-out test split was
not loaded until after the checkpoint had been chosen. Every `result.json`
records this policy.

## Completeness audit

`reports/recognition/TASK-009B-completeness-audit.json`

| Check | Result |
| --- | --- |
| Expected experiments | 15 (5 configurations × 3 folds, seed 1337) |
| **Complete** | **15 / 15** |
| Problems found | **0** |
| `status.json` = COMPLETE | 15 / 15 |
| `result.json`, `history.json`, `predictions.json` present and non-empty | 15 / 15 |
| `best.pt`, `last.pt` present | 15 / 15 |
| `contract_version` = `task009a_sequence_input_v1` | 15 / 15 |
| `input_policy` = `values_and_feature_valid` | 15 / 15 |
| feature_set / pooling / fold / seed match the directory | 15 / 15 |
| Duration Control A present | yes |

### Metadata note — `quaternion_policy` on quaternion-free feature sets

Confirmed as predicted: the `bend_only` and `bend_spread` experiments sit under
`q-na/` in the directory tree, but their `experiment.quaternion_policy` field
records the CLI default `"absolute"`.

This is a **non-applicable default metadata field, not a defect**. Those feature
sets contain no quaternion channels at all, so the field has no effect on
tensorization, on the model input dimension (60 and 76, quaternion-free), or on
the checkpoint. The checkpoint loader deliberately skips the quaternion check
when `feature_set != "full"`, and the audit therefore records the value without
treating it as a mismatch. It causes no checkpoint or result inconsistency and is
not analysed as a quaternion experiment anywhere in this report.

### Control A provenance

The persisted sweep contained no `duration_controls/` directory, so Control A was
executed against this run root during the analysis. It is a **classical
CPU-only scikit-learn logistic regression, not a neural training run**, takes
seconds, and is deterministic. Its values reproduce the infrastructure-stage
figures exactly. No neural experiment was launched.

## Main LOSO results

`reports/recognition/TASK-009B-results-by-fold.csv` ·
`figures/task009b-headline.png`

Held-out-signer test performance, seed 1337:

| Configuration | S01 acc | S01 F1 | S02 acc | S02 F1 | S03 acc | S03 F1 | **mean acc** | **SD acc** | **mean F1** | **SD F1** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `bend_only` | 49.43% | 0.4576 | 72.66% | 0.7180 | 66.17% | 0.6425 | 62.76% | 11.99 | 0.6060 | 0.1340 |
| `bend_spread` | 60.06% | 0.5707 | 70.24% | 0.7064 | 54.33% | 0.5224 | 61.54% | 8.05 | 0.5998 | 0.0954 |
| **`full_absolute_masked_mean`** | 56.78% | 0.5441 | 74.16% | 0.7307 | 71.95% | 0.7073 | **67.63%** | 9.46 | **0.6607** | 0.1017 |
| `full_relative_masked_mean` | 58.20% | 0.5650 | 70.74% | 0.7052 | 59.20% | 0.5713 | 62.71% | 6.97 | 0.6139 | 0.0792 |
| `full_absolute_final_hidden` | 63.27% | 0.5976 | 55.75% | 0.5465 | 76.96% | 0.7546 | 65.32% | 10.75 | 0.6329 | 0.1085 |

28-way chance is 3.57%. **The pre-registered primary configuration
(`full` + `absolute` + `masked_mean` + `values_and_feature_valid`) has the best
mean accuracy and the best mean macro F1 of the five**, at 67.63% / 0.6607 — 19×
chance. It was designated primary before any test score existed and has not been
redefined.

`full_absolute_final_hidden` has a higher single-fold peak (76.96% on S03) but a
lower mean and a catastrophic S02, discussed under Control B.

## Duration Control A — length alone

`duration_controls/control_a_length_only.json`

Multinomial logistic regression on sequence length alone, fitted on train only:

| Fold | Test accuracy | Test macro F1 |
| --- | ---: | ---: |
| S01 | 3.71% | 0.0095 |
| S02 | 3.57% | 0.0038 |
| S03 | 4.23% | 0.0088 |
| **chance** | **3.57%** | — |

**Sequence duration alone does not explain the model's cross-signer letter
recognition.** The primary model reaches 67.63% mean accuracy against a
duration-only baseline of 3.8% mean — indistinguishable from chance.

This does **not** mean duration is irrelevant. Length remains a strong *signer*
cue (TASK-009A: 78.1% oracle vs 33.6% chance), and a recurrent model can exploit
duration through its dynamics even when a static length feature cannot. That is
precisely what Control B tests.

## Control B — pooling under duration shift

`reports/recognition/TASK-009B-duration-pooling-analysis.csv` ·
`figures/task009b-pooling-control.png`

`full` + `absolute`, everything held fixed except pooling:

| Fold | test/train length ratio | `masked_mean` acc | `final_hidden` acc | **Δ acc** | **Δ macro F1** |
| --- | ---: | ---: | ---: | ---: | ---: |
| S01 | 1.12 | 56.78% | 63.27% | **+6.49 pp** | +5.36 |
| S02 | **0.60** | 74.16% | 55.75% | **−18.42 pp** | **−18.42** |
| S03 | 1.39 | 71.95% | 76.96% | **+5.00 pp** | +4.73 |
| mean | — | 67.63% | 65.32% | −2.31 pp | −2.78 |

The pattern is stark and it lines up exactly with the duration shift. On the two
folds whose held-out durations sit near the training distribution, `final_hidden`
**wins** by 5–6.5 pp. On S02 — the one fold where only ~5.2% of test lengths fall
inside the training p5–p95 band — it **loses 18.4 pp**, more than three times the
magnitude of either gain and in the opposite direction.

A second, independent signal points the same way: `final_hidden` on S02 has the
**lowest best-validation macro F1 of all 15 runs** (0.9364, against 0.98–0.99
everywhere else) and the **earliest best epoch** (18). Its difficulty on that fold
is visible before the test set is ever touched, which makes it less likely to be
a test-set fluke.

**Interpretation, stated with its limits.** This is consistent with the
hypothesis that final-hidden pooling is substantially more vulnerable to duration
and domain shift than a masked mean, which is why the masked mean was
pre-registered as primary. But it is **one severely shifted fold, one seed, and
an observational ablation**: n=1 in the condition that matters. It is evidence,
not proof of causality. A dedicated study — several seeds, and synthetic duration
perturbation on a fixed signer — would be needed to establish the mechanism.

## Sensor ablation — spread

`reports/recognition/TASK-009B-sensor-ablation.csv` ·
`figures/task009b-class-ablation.png`

`bend_spread` − `bend_only`:

| Fold | Δ accuracy | Δ macro F1 |
| --- | ---: | ---: |
| S01 | **+10.63 pp** | +11.31 |
| S02 | −2.43 pp | −1.16 |
| S03 | **−11.84 pp** | −12.01 |
| **mean** | **−1.21 pp** | **−0.62** |

**Adding the four spread channels did not reliably improve recognition.** It
helped on exactly 1 of 3 folds, improved 13 of 28 classes, and the mean effect is
slightly negative. The statement "spread sensors improve recognition" is **not
supported** by this evidence.

The effect is strongly signer-dependent (+10.6 pp on S01, −11.8 pp on S03) and
strongly class-dependent:

| Gains | Δ F1 | | Losses | Δ F1 |
| --- | ---: | --- | --- | ---: |
| ك | +0.166 | | ظ | −0.305 |
| س | +0.165 | | ث | −0.257 |
| ا | +0.160 | | ع | −0.151 |
| ق | +0.149 | | ت | −0.148 |
| ه | +0.114 | | ج | −0.093 |
| ح | +0.086 | | ص | −0.081 |
| ي | +0.082 | | | |

Plausible reading: the four spread channels add real geometric information for
letters distinguished by finger separation (ك، س، ا، ق) while adding four
partly-invalid, partly-noisy dimensions that dilute letters distinguished by
other cues. Notably, several of the classes spread *hurts* (ظ، ث، ع) are the same
classes palm orientation later *rescues* — see the next section — which suggests
the spread channels alone are an incomplete addition rather than a harmful one.

## Sensor ablation — palm orientation

`full_absolute_masked_mean` − `bend_spread`:

| Fold | Δ accuracy | Δ macro F1 |
| --- | ---: | ---: |
| S01 | −3.28 pp | −2.66 |
| S02 | +3.93 pp | +2.43 |
| S03 | **+17.62 pp** | +18.49 |
| **mean** | **+6.09 pp** | **+6.08** |

Improved 2 of 3 folds and 19 of 28 classes. Largest per-class gains:

| ع | ج | د | ز | ظ | ه | ث | ط |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| +0.379 | +0.269 | +0.198 | +0.184 | +0.176 | +0.166 | +0.163 | +0.116 |

Palm orientation is valuable on average and large where it helps, but it is
**signer-dependent**, costing 3.3 pp on S01.

## Full glove vs bend only — the core project result

`full_absolute_masked_mean` − `bend_only`:

| Fold | Δ accuracy | Δ macro F1 |
| --- | ---: | ---: |
| S01 | +7.35 pp | +8.65 |
| S02 | +1.50 pp | +1.27 |
| S03 | +5.78 pp | +6.48 |
| **mean** | **+4.87 pp** | **+5.47** |

**This is the only contrast that improves on every fold: 3 / 3, and 20 of 28
classes.** Largest per-class gains: ه +0.280, ز +0.244, س +0.228, ع +0.228,
ج +0.176.

The complete simulated glove representation therefore contains additional useful
**signer-independent** discriminative information beyond bend alone. Its
consistency across all three held-out signers is what distinguishes this result
from the spread and orientation contrasts taken separately, each of which is
positive only on average.

## Quaternion ablation — absolute vs relative

`full_relative_masked_mean` − `full_absolute_masked_mean`:

| Fold | Δ accuracy | Δ macro F1 |
| --- | ---: | ---: |
| S01 | +1.43 pp | +2.10 |
| S02 | −3.43 pp | −2.55 |
| S03 | **−12.76 pp** | −13.60 |
| **mean** | **−4.92 pp** | **−4.68** |

Absolute camera-frame palm orientation **outperforms** the sequence-relative
representation, improving 2 of 3 folds, and beating relative on 17 of 28 classes.

`relative_first_valid` re-expresses each hand's orientation against its own first
valid frame, which discards the absolute palm attitude and keeps only the change
from the sequence's own start. The result suggests that absolute attitude — how
the palm is oriented in the camera frame — is itself class-discriminative for
Arabic letters, and that removing it costs more than the signer-invariance it
might buy. Camera-frame information should not be assumed undesirable by default.

The caveat: with three signers all recorded frontally under similar conditions,
absolute camera-frame orientation is close to absolute *anatomical* orientation.
A corpus with varied camera placement could reverse this, and the relative policy
remains implemented and tested for exactly that reason.

## Validation vs held-out test domain gap

`figures/task009b-domain-gap.png`

Best validation macro F1 minus held-out test macro F1, all 15 runs:

| Configuration | S01 | S02 | S03 | mean |
| --- | ---: | ---: | ---: | ---: |
| `bend_only` | 0.520 | 0.275 | 0.346 | 0.380 |
| `bend_spread` | 0.416 | 0.280 | 0.469 | 0.388 |
| **`full_absolute_masked_mean`** | 0.440 | 0.251 | 0.284 | **0.325** |
| `full_relative_masked_mean` | 0.417 | 0.286 | 0.422 | 0.375 |
| `full_absolute_final_hidden` | 0.395 | 0.390 | 0.239 | 0.341 |
| **mean by fold** | **0.438** | **0.297** | **0.352** | 0.362 |

Validation macro F1 is 0.98–0.99 in 14 of 15 runs while held-out test macro F1 is
0.46–0.75. **This gap is cross-signer domain generalization, not ordinary
overfitting.** Validation is drawn from the same two signers as training, so a
model that has memorized those signers' articulation scores ~0.99 there and still
loses 25–52 points on an unseen signer. In-domain validation of 98–99% must not
be reported as if it were recognition performance.

The primary configuration has the **lowest mean gap** (0.325) of the five, and
S01 is the hardest signer to generalize to (mean gap 0.438) under every
configuration.

## Per-class analysis

`reports/recognition/TASK-009B-results-by-class.csv` (full per-class precision,
recall, F1 and support for all 15 runs) and
`reports/recognition/TASK-009B-class-ablation.csv` (per-class deltas for all five
contrasts).

### Low-spread-validity classes

TASK-008C measured that ض، ي، ص have only 55–59% spread channel validity, the
lowest of the 28 classes. Mean test F1 across folds:

| Class | spread validity | `bend_only` | `bend_spread` | `full_absolute` |
| --- | ---: | ---: | ---: | ---: |
| ض | 55.4% | 0.488 | **0.512** | 0.475 |
| ي | 56.6% | 0.760 | 0.842 | **0.875** |
| ص | 59.1% | **0.736** | 0.655 | 0.706 |

**Low spread validity must not be read as spread being useless.** ي has among the
lowest spread validity in the corpus and is one of the classes spread helps most
(+0.082), reaching 0.875 with the full glove. ض also improves when spread is
added. ص moves the other way. The relationship between a channel's validity rate
and its usefulness is not monotonic, and TASK-008C's decision to mask invalid
spread rather than impute it is what makes these classes usable at all.

## Confusion analysis

`reports/recognition/TASK-009B-top-confusions.csv` ·
`figures/task009b-primary-confusion.png`

Top confusions for the primary model, from the saved confusion matrices:

| S01 | S02 | S03 |
| --- | --- | --- |
| ض→ي 40 | ظ→ط 35 | ت→ز 25 |
| ك→س 32 | ز→ت 30 | ص→ض 22 |
| ب→ل 26 | ا→خ 25 | و→ا 20 |
| ث→ه 26 | ن→د 22 | ث→ت 20 |
| د→ظ 24 | ض→ص 21 | ل→ب 18 |
| ق→ف 23 | ف→و 20 | ل→ن 18 |

**Quantified rather than described anecdotally.** Across the three folds' top-10
confusion pairs there are **27 distinct (true → predicted) pairs**:

| Appears in | Pairs | Share |
| --- | ---: | ---: |
| 1 fold only | 24 | **88.9%** |
| 2 folds | 3 | 11.1% |
| **all 3 folds** | **0** | **0%** |

**Not one top confusion pair is shared by all three held-out signers.** The
dominant errors are overwhelmingly signer-specific, not universal properties of
the letter set. This matters for interpretation: the model is not failing on a
fixed set of intrinsically ambiguous letters, it is failing on whichever letters
a *particular* unseen signer articulates differently from the training signers.
It also argues that adding signers would help more than redesigning the label set.

There is a hint of near-symmetry worth noting without overclaiming: ض→ي on S01
and ض→ص on S02 and ص→ض on S03 all involve the low-spread-validity cluster, which
is at least consistent with those letters being genuinely harder — but they never
confuse in the same direction on all three folds.

## Training dynamics

`figures/task009b-training-curves.png`

| Configuration | mean best epoch | mean epochs run | mean \|Δ val macro F1\| per epoch |
| --- | ---: | ---: | ---: |
| `bend_only` | 46.7 | 55.7 | 0.061 |
| `bend_spread` | 39.7 | 49.0 | 0.050 |
| `full_absolute_masked_mean` | **25.0** | **37.0** | 0.052 |
| `full_relative_masked_mean` | 49.3 | 56.3 | 0.048 |
| `full_absolute_final_hidden` | 36.0 | 47.7 | 0.057 |

All 15 runs converged. 10 of 15 stopped early on the patience-12 rule; the
other 5 ran the full 60 epochs, and of those only `full_relative_masked_mean` on
S03 had its best epoch at 60 — that one run may have been still improving when
the budget ended. Final training loss ranged 0.047–0.202, and best validation
macro F1 was ≥ 0.98 in 13 of 15 runs. Epoch-to-epoch oscillation in validation
macro F1 was modest (~0.05) and comparable across configurations — no
configuration-specific instability.

The two runs below 0.98 validation macro F1 are `bend_only` S01 (0.9773) and
`full_absolute_final_hidden` S02 (0.9364). The first is the weakest feature set
on the hardest signer; the second is the Control B anomaly discussed above.

The primary configuration converged **fastest** (mean best epoch 25.0 against
36–49 for the others), which is a mild independent signal that the full
representation makes the task easier to fit, not just easier to score.

The single anomaly is `final_hidden` on S02: best validation macro F1 0.9364, the
only run below 0.98, with the earliest best epoch (18).

## Runtime

| Quantity | Value |
| --- | --- |
| Total sweep wall time | **3,453 s ≈ 57.6 minutes** (15 experiments) |
| Mean seconds per epoch | 4.68 s |
| Peak GPU memory | **87.9 MiB** of 4,096 MiB |
| Parameters | 521,500 |
| Persisted outputs | 186 MB (checkpoints excluded from git) |

For context, producing the dataset took 8 h 35 m of GPU time in TASK-008C; the
entire recognition sweep took under an hour on the same 4 GB laptop GPU.

## Physical-glove interpretation

The ablations speak to the ideal 19-Hall + 1-IMU per-hand design frozen in
TASK-006:

* **15 bend packages per hand already carry substantial discriminative
  information** — 62.76% mean accuracy alone, 17.6× chance.
* **The 4 spread packages are not uniformly beneficial in isolation.** Mean effect
  −1.21 pp, helping 1 of 3 signers and 13 of 28 classes. Their value is
  class-specific and interacts with the rest of the representation.
* **The palm IMU orientation package is valuable on average** (+6.09 pp over
  bend+spread, +18 pp on one fold), and **absolute** attitude beats
  sequence-relative by 4.92 pp.
* **The complete glove is the strongest mean predefined representation** and the
  only contrast that improves on all three held-out signers (+4.87 pp over bend
  alone).

**This is not yet a sensor-reduction result.** These are single-seed,
three-signer, one-factor ablations, and the spread channels' negative mean effect
in isolation does not license removing them — they help specific letters, and
they may interact with orientation in ways a 2-way contrast cannot separate. What
this evidence supports is a **dedicated sensor-reduction study** (19 → 15 → 10 →
5 Hall channels, multiple seeds, with and without the IMU) as a distinct future
task, exactly as TASK-006A anticipated. It also gives a concrete argument for
including a palm IMU in any physical prototype, without yet establishing hardware
necessity.

## Statistical reporting

There are **3 LOSO folds and one fixed seed (1337)**.

**The reported standard deviation is across held-out signers.** It measures
signer-to-signer variation, *not* initialization or training variance. No
statement in this report should be read as a seed-stability claim.

No significance test is reported. With n=3 folds, no seed replication and
observational one-factor contrasts, a p-value would convey false precision. Effect
sizes and **fold-wise consistency** are reported instead — which is why "improves
on 3 / 3 folds" (full vs bend-only) is treated as a stronger result than a larger
mean that flips sign across folds (orientation, spread).

## Limitations

1. **One seed.** Every configuration was run once at seed 1337. Differences of a
   few points could be initialization noise; only the fold-consistent effects
   should be relied on.
2. **Three signers.** LOSO over three folds is a high-variance generalization
   estimate. Per-fold swings of 20+ points are common.
3. **One severely shifted fold.** The Control B conclusion rests on S02 being the
   only fold with major duration shift — n=1 in the condition of interest.
4. **Observational ablations.** Configurations differ in one factor at a time, but
   nothing was randomized and interactions were not tested factorially.
5. **Large domain gap.** Validation 0.98–0.99 vs test 0.46–0.75. In-domain numbers
   are not recognition performance.
6. **Signer-specific confusions.** 89% of top confusion pairs occur on one fold
   only, so error analysis does not transfer between signers.
7. **Simulated glove.** These are WiLoR-derived virtual sensors, not physical
   hardware readings; sensor noise, drift and calibration are absent.
8. **No hyperparameter search**, by design. The architecture was fixed in advance
   and is not claimed to be optimal.

## Deployment recommendation

For a future **all-signers deployment/demo model** (not yet trained):

```
feature_set        = full
quaternion_policy  = absolute
pooling            = masked_mean
input_policy       = values_and_feature_valid
```

Basis: best mean accuracy (67.63%) and macro F1 (0.6607) of the five; the only
representation that improves on all three held-out signers over bend-only; the
lowest mean validation→test domain gap (0.325); the fastest convergence; and the
pooling that degraded least under severe duration shift.

**This is a post-evaluation design choice, not a new unbiased scientific test.**
The configuration was selected after seeing held-out results, so its 67.63% must
not be quoted as the expected accuracy of the deployed model. A deployment model
trained on all three signers has no held-out signer at all, and its performance on
a genuinely new signer remains unmeasured.

## Reproducibility

```bash
cd /home/hatim/Graduation-Project-Simulation-task009b

# Regenerate every table, JSON summary and figure from the persisted sweep.
PYTHONPATH=. python scripts/run_task009b_analysis.py \
  --run-root /home/hatim/graduation-project-runs/task009b-core28-lstm

# Duration Control A (classical, CPU, seconds).
PYTHONPATH=. python scripts/run_task009b_duration_control.py \
  --run-root /home/hatim/graduation-project-runs/task009b-core28-lstm

# Tests.
PYTHONPATH=. python -m unittest discover -s tests -p "test_*.py"
python -m compileall -q recognition scripts tests
```

Machine-readable outputs: `TASK-009B-results-summary.json`,
`TASK-009B-completeness-audit.json`, `TASK-009B-results-by-fold.csv`,
`TASK-009B-results-by-class.csv`, `TASK-009B-sensor-ablation.csv`,
`TASK-009B-class-ablation.csv`, `TASK-009B-duration-pooling-analysis.csv`,
`TASK-009B-top-confusions.csv`, and six figures under `figures/`.

## Verdict

| Item | Result |
| --- | --- |
| Completeness | **15 / 15 experiments COMPLETE, 0 problems** |
| Primary configuration | `full` + `absolute` + `masked_mean` (pre-registered) |
| Primary mean accuracy | **67.63%** (SD across signers 9.46 pp) |
| Primary mean macro F1 | **0.6607** (SD 0.1017) |
| vs 28-way chance | 19× |
| Full glove vs bend-only | **+4.87 pp, 3/3 folds, 20/28 classes** |
| Spread alone | −1.21 pp, 1/3 folds — **not consistently beneficial** |
| Palm orientation | +6.09 pp, 2/3 folds |
| Absolute vs relative quaternion | absolute wins by 4.92 pp |
| Control A (length only) | 3.6–4.2%, at chance |
| Control B (final_hidden) | +6.5/+5.0 pp on aligned folds, **−18.4 pp on shifted S02** |
| Tests | 613 passed, 0 failed |
| compileall | 0 errors |

**TASK-009B CORE-28 LSTM BASELINE COMPLETE — READY FOR RECOGNITION ANALYSIS / NEXT MILESTONE**
