# TASK-009B — Core-28 LSTM Baseline: Training Infrastructure

> **Status: pre-training.** This report documents the infrastructure, the frozen
> protocol and the smoke evidence that it runs. **It contains no accuracy
> results**, because the real training has not been performed. The user runs the
> experiment matrix; a follow-up task analyses the outputs.

## Task

Implement the complete TASK-009B model / training / evaluation infrastructure on
top of the frozen `task009a_sequence_input_v1` contract, validate it with small
smoke runs only, and hand over exact deterministic, resumable commands.

## Architecture

```
TASK-009A batch (values + feature_valid + frame_valid + lengths)
  → concatenate values with feature_valid            [B, T, 2D]
  → (optional linear projection — off by default)
  → LSTM, 2 layers, hidden 192, unidirectional, packed
  → temporal pooling: masked mean over real frames   [B, 192]
  → dropout 0.3
  → Linear(192 → 28)                                 [B, 28] logits
```

Deliberately modest and deliberately boring: no transformer, no convolutional
front end, no attention, no CNN-LSTM hybrid, no architecture search. The
scientific question is whether the TASK-006 virtual-glove representation carries
letter identity, and a baseline that is easy to reason about answers that better
than a tuned one.

| Property | Value |
| --- | --- |
| Parameters | **521,500** (full feature set) |
| Input dimension | 92 (`full`), 76 (`bend_spread`), 60 (`bend_only`) |
| Hidden size / layers | 192 / 2 |
| Dropout | 0.3 (between LSTM layers and before the classifier) |
| Direction | unidirectional |

**Why no input projection by default.** A `Linear` before an LSTM is redundant
with the LSTM's own input-to-hidden matrix `W_ih`. `--input-projection` exists
because the brief asks for the option, not because it is expected to help.

## Input and mask policy

**Model input = `values` ‖ `feature_valid`**, via the frozen TASK-009A helper
`concat_features_and_masks`. A numeric zero cannot express "no measurement", so
validity must be a real input channel rather than an implicit convention.

| Feature set | D | Model input |
| --- | ---: | ---: |
| `bend_only` | 30 | **60** |
| `bend_spread` | 38 | **76** |
| `full` | 46 | **92** |

**`hand_present` is deliberately not fed.** Per-channel `feature_valid` already
carries strictly more information about what is missing, and TASK-009A measured
that in this corpus a hand's presence is *exactly* equal to its channels being
valid (0 disagreements in 2,532 sampled hand instances). Feeding it would add two
channels perfectly correlated with 46 existing ones. It remains in the batch for
diagnostics and would matter if a future corpus broke that equality.

**`tracking_state_code` remains diagnostics only** and never enters the model.

`--input-policy values_only` is available as an ablation to measure what the mask
channels are actually worth.

## Temporal pooling

**Primary: masked mean.** LSTM outputs are averaged over `frame_valid` steps
only; padding gets exactly zero weight and is excluded from the denominator.

The reason is the TASK-009A duration confound. Sequence length identifies the
*signer* at 78.1% (chance 33.6%) but the *letter* at only 7.6% (chance 3.8%), and
signer 02 averages 13.7 frames against signer 03's 24.3. A final hidden state is
implicitly a function of how many steps the recurrence consumed, so it is the
pooling most exposed to that confound. A mean weights each observed frame equally
and does not grow or shrink with sequence length.

**Secondary: `final_hidden`**, retained as duration Control B, taken from the
packed LSTM's `h_n` so it is each sequence's own last **real** step and is never
contaminated by padding. A test asserts both poolings give bit-comparable results
whether a sequence is batched alone or beside a much longer one.

## Training protocol

Frozen before any test evaluation:

| Setting | Value |
| --- | --- |
| Loss | cross-entropy, unweighted |
| Optimizer | AdamW, lr 1e-3, weight decay 1e-4 |
| Gradient clipping | global norm 5.0 |
| Batch size | 32 |
| Max epochs | 60 |
| Early stopping | patience 12 on validation macro F1 |
| Checkpoint selection | **validation macro F1 only** |
| Seed | explicit, `--seed`, default 1337 |

**Class balance:** counts range 149–160 per class (ratio **1.074**). That is
negligible, so ordinary cross-entropy is used. A weighted loss would be an extra
unjustified degree of freedom.

**Test-signer isolation.** The held-out test split is not loaded until the final
evaluation, after the checkpoint has already been chosen. It never influences
early stopping, checkpoint selection, normalization or any architecture decision.
Every `result.json` records this policy verbatim.

**Normalization:** none, inherited from TASK-009A. Values are already physically
normalized (`deg / 180`), quaternions are unit-norm. Nothing is fitted, so nothing
can leak across a LOSO boundary.

Recorded per epoch: epoch, train loss, train accuracy, validation loss,
validation accuracy, validation macro F1, best-so-far, epoch seconds.

## Determinism

`seed_everything(seed)` seeds Python `random`, `PYTHONHASHSEED`, NumPy, torch and
all CUDA devices, sets `cudnn.deterministic=True`, `cudnn.benchmark=False` and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, and **returns what it set** so the settings are
recorded in the checkpoint and the result JSON rather than assumed. DataLoader
workers get distinct reproducible seeds via `worker_init_fn`. RNG state is
captured in `last.pt` so an interrupted run resumes the same stream.

No seed is cherry-picked: the default is a single fixed value and `--seed` is a
first-class CLI argument.

## Libraries

Standard problems use standard libraries; project-specific logic is ours.

| Concern | Implementation | Why |
| --- | --- | --- |
| Model, optimizer, packing | PyTorch | standard |
| accuracy, precision, recall, macro/weighted F1, confusion matrix, classification report, log loss | **scikit-learn** | standard, battle-tested |
| Length-only control | **`sklearn.linear_model.LogisticRegression`** in a `Pipeline` with `StandardScaler` | classical baseline; the pipeline also enforces train-only scaler fitting |
| Masking, LOSO enforcement, sensor-contract checks, pooling semantics, checkpoint compatibility | ours | project-specific, and the whole point of TASK-009A |

`scikit-learn>=1.5` is declared in `pyproject.toml` under the `recognition`
optional-dependency group.

Two deliberate exceptions where custom code is retained, with the technical
reason:

* **Fixed 28-class label space.** `confusion_matrix` and the macro averages wrap
  scikit-learn but always pass `labels=range(28)`. Without it scikit-learn sizes
  its output to the classes that happen to occur, so matrices from different
  folds would not be comparable and a never-predicted class would silently vanish
  from a macro average.
* **`oracle_accuracy_from_length`.** Not a fitted model but a combinatorial
  ceiling — assign every distinct length its own majority target. scikit-learn has
  no equivalent, and it must match the statistic TASK-009A already reported.

## Experiment matrix

| # | Feature set | Quaternion | Pooling | Purpose |
| --- | --- | --- | --- | --- |
| 1 | `bend_only` | n/a | `masked_mean` | sensor ablation: 15 bend channels per hand |
| 2 | `bend_spread` | n/a | `masked_mean` | + 4 spread channels per hand |
| 3 | `full` | `absolute` | `masked_mean` | **primary**: + palm orientation |
| 4 | `full` | `relative_first_valid` | `masked_mean` | quaternion policy comparison |
| 5 | `full` | `absolute` | `final_hidden` | **duration Control B**: pooling comparison |

× 3 LOSO folds = **15 experiments**. All five are CLI options on one program;
there is no duplicated per-ablation script.

## Duration controls

**Control A — length-only classifier.** Multinomial logistic regression on the
single scalar `sequence_length`, fitted on **train only**. Already run (it costs
seconds and needs no GPU):

| Fold | Test accuracy | Test macro F1 | In-sample oracle |
| --- | ---: | ---: | ---: |
| S01 | 3.71% | 0.0095 | 10.27% |
| S02 | 3.57% | 0.0038 | 8.85% |
| S03 | 4.23% | 0.0088 | 11.35% |
| chance | 3.57% | — | — |

**This is a useful result already.** Across signers, duration alone predicts the
letter at essentially chance. So while length is a strong *signer* cue (78.1%
oracle), it is not a usable *letter* cue, and any cross-signer accuracy the LSTM
achieves above ~4% cannot be explained by duration alone. It does not rule out
the model using duration as a shortcut *within* a signer — that is what Control B
is for.

**Control B — pooling comparison.** Experiment 5 (`final_hidden`) against
experiment 3 (`masked_mean`), same feature set, same model, same training
settings, same seed. If `final_hidden` does markedly better on the folds with
matched durations and worse on S02 (the near-disjoint fold), that is evidence of
duration dependence rather than letter dynamics.

## Metrics

Per evaluation: accuracy, macro/weighted precision, recall and F1, cross-entropy,
per-class precision/recall/F1/support, a 28×28 confusion matrix, ranked top
confusion pairs with Arabic labels, and a per-sample prediction record
(`sample_id`, signer, true, predicted, confidence, sequence length, correctness)
written to `predictions.json` for validation and test.

## Checkpoint format

`task009b_checkpoint_v1`, written atomically (temp file then `replace`) so an
interrupt cannot leave a truncated file.

Stores: TASK-009A contract version, feature set, quaternion policy, pooling,
input policy, fold, seed, input dimension, hidden size, layers, dropout, class
count, full training configuration, seeding report, epoch, best epoch, best
validation metric and its name, early-stopping counter, history, optimizer state
and RNG state.

`load_checkpoint(..., expect=spec)` **rejects** a mismatch in contract version,
checkpoint schema, feature set, quaternion policy, pooling or input policy. Fold
and seed identify a run but are not correctness constraints, so a checkpoint may
be deliberately scored against another fold. A `bend_only` checkpoint loaded into
a `full` pipeline raises instead of producing confident nonsense.

## CLI

`scripts/run_task009b_lstm_baseline.py` — one experiment.

`--run-root --data-run-root --index --splits-dir --labels --fold --feature-set
--quaternion-policy --pooling --input-policy --seed --device --epochs
--batch-size --learning-rate --weight-decay --grad-clip-norm
--early-stopping-patience --hidden-size --num-layers --dropout
--input-projection --num-workers --resume --status --evaluate-only --checkpoint
--force --preload --limit-train --limit-eval`

`--limit-train` / `--limit-eval` are smoke-only and take an **evenly strided**
subset, not a prefix: fold rows are sorted by `sample_id`, so a prefix would be a
handful of adjacent classes and would produce meaningless smoke metrics. A run
using them prints a SMOKE MODE warning.

`scripts/run_task009b_duration_control.py` — Control A, all folds, no GPU.
`scripts/run_task009b_required_sweep.py` — the 15-experiment matrix, sequential,
skips COMPLETE experiments, stops on first failure, `--dry-run` supported.

### Output layout

```
<run-root>/<feature_set>/q-<policy>/<pooling>/fold<NN>/seed<NNNN>/
    best.pt  last.pt  history.json  result.json  predictions.json  status.json
<run-root>/duration_controls/control_a_length_only.json
```

Deterministic: the same experiment always maps to the same directory.

## Resume semantics

`last.pt` is written every epoch with model, optimizer, epoch, best metric, best
epoch, early-stopping counter, history and RNG state. The **same command plus
`--resume`** continues from the next epoch.

A directory whose `status.json` says `COMPLETE` is **skipped**, not overwritten;
`--force` re-runs it and `--evaluate-only` re-scores it. Verified: an interrupted
2-epoch run resumed at epoch 3 with `rng_restored=True` and carried its best
metric and history forward.

## Progress output

One line per epoch, flushed immediately:

```
[epoch   2/60] train_loss 2.6531 | val_loss 2.1500 | val_acc  27.90% | val_macroF1 0.2026 |
               best 0.2026@2 * | 4.2s | elapsed 00:00:09 | ETA 00:00:00 | GPU 85 MiB
```

with a header naming the experiment, fold, feature set, quaternion policy,
pooling, seed, device, parameter count and input dimension. On completion:
`BEST CHECKPOINT:`, `RESULT JSON:`, `STATUS:` / `COMPLETE`.

## Smoke-test evidence

**Not scientific results — 2 epochs, unconverged.** Recorded in
`reports/recognition/TASK-009B-smoke-evidence.json`.

Full fold S01 (2,372 train / 448 validation / 1,402 test), `full` +
`absolute` + `masked_mean`, seed 1337, batch 32, 2 epochs, CUDA:

| Epoch | train loss | val loss | val acc | val macro F1 | seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.2404 | 2.9785 | 9.82% | 0.0228 | 4.9 |
| 2 | 2.6531 | 2.1500 | 27.90% | 0.2026 | 4.2 |

Verified end to end: CUDA forward/backward runs; loss is finite and falls;
validation improves; `best.pt`/`last.pt` written; **resume** continues from the
stored epoch with RNG restored; **evaluate-only** re-scores an existing
checkpoint; **status** reports experiment state; a COMPLETE experiment is skipped
rather than silently overwritten; the **inference API** classifies a
variable-length sequence and returns the authoritative Arabic label.

## Runtime estimate

Approximate, from the smoke run — not a promise.

| Quantity | Measured / estimated |
| --- | --- |
| Seconds per epoch (fold S01, batch 32, `full`) | **~4.2–4.9 s** |
| Peak GPU memory | **~85 MiB** of 4,096 MiB |
| One experiment, 60 epochs | **~5 minutes** |
| Full 15-experiment matrix | **~60–90 minutes** |
| Control A (all folds) | seconds, CPU only |
| Per experiment on disk | ~13 MB (`best.pt` 6.3 MB + `last.pt` 6.3 MB + JSON) |
| Full matrix on disk | **~200 MB** |

Early stopping (patience 12) will usually end runs before 60 epochs, so the true
total is likely lower. The 4 GB GPU is nowhere near a constraint — this model is
tiny compared with the WiLoR extraction that produced the data.

## Tests

`tests/test_task009b_lstm_baseline.py` — **44 tests**, synthetic, CPU-only.
Covering: model input dimensions for D=30/38/46 and 60/76/92; mask concatenation
order; masked-mean and final-hidden pooling; padding invariance for both;
9-frame and 70-frame sequences; forward shape `[B, 28]`; finite loss; one
optimizer step changing weights; wrong-dimension batch rejected; deterministic
evaluation; seeding report; metric values against hand-computed references;
never-predicted class penalised; ranked confusions; checkpoint save/load
round-trip; contract, feature-set and pooling mismatch rejection; corrupt and
missing checkpoints; resume state round-trip; deterministic experiment slugs;
frozen fold sizes and no signer leakage; the length-only control; single-sequence
inference through a checkpoint to an authoritative Arabic label; and the label
table.

**Full suite: 592 passed, 0 failed** (TASK-009A baseline 548 + 44 new; no
existing test weakened or modified). `compileall`: 0 errors.

## What is NOT in this report

No accuracy tables, no confusion matrices, no per-class results, no ablation
comparison, no fold averages, no conclusions about representation quality. The
long experiments have not run. Fabricating any of that would be worse than
leaving it blank.

## Exact user-run commands

See the handoff printed at the end of the TASK-009B session and the
"Reproducibility" section of the follow-up report. In short:

```bash
cd /home/hatim/Graduation-Project-Simulation-task009b

# A. primary first run (verify real training behaviour before the sweep)
PYTHONPATH=. python scripts/run_task009b_lstm_baseline.py \
  --run-root /home/hatim/graduation-project-runs/task009b-core28-lstm \
  --data-run-root /home/hatim/graduation-project-runs/task008-core28-full \
  --fold 01 --feature-set full --quaternion-policy absolute \
  --pooling masked_mean --seed 1337 --device cuda --epochs 60 --batch-size 32

# B. status      C. resume (same command + --resume)      D. evaluate-only
PYTHONPATH=. python scripts/run_task009b_lstm_baseline.py \
  --run-root /home/hatim/graduation-project-runs/task009b-core28-lstm --status

# E. the remaining required matrix (15 experiments, ~60-90 min)
PYTHONPATH=. python scripts/run_task009b_required_sweep.py \
  --run-root /home/hatim/graduation-project-runs/task009b-core28-lstm \
  --data-run-root /home/hatim/graduation-project-runs/task008-core28-full \
  --device cuda --epochs 60 --batch-size 32 --resume

# duration Control A (cheap, CPU)
PYTHONPATH=. python scripts/run_task009b_duration_control.py \
  --run-root /home/hatim/graduation-project-runs/task009b-core28-lstm
```

## Verdict

**TASK-009B TRAINING INFRASTRUCTURE READY — USER MUST RUN TRAINING**
