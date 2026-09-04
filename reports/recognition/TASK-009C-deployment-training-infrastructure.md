# TASK-009C — All-Signers Core-28 Deployment Model: Training Infrastructure

> **Status: superseded by the completed run.** This report documents the
> infrastructure and the epoch-policy rationale, written before training. The
> training has since been performed and verified — see
> **[`TASK-009C-deployment-model.md`](TASK-009C-deployment-model.md)** for the
> final artifact, its SHA-256 and the verification results.
>
> This report contains **no deployment accuracy**, and by construction never will:
> the deployment model has no held-out data.

## Purpose

TASK-009B answered a scientific question: *how well does the virtual-glove
representation generalize to an unseen signer?* That answer is frozen.

TASK-009C answers a different, engineering question: *given that the
configuration has already been selected, how do we fit one final model on all
available Core-28 data for the application and demo?*

## Scientific vs deployment training — the distinction that matters

| | TASK-009B (scientific) | TASK-009C (deployment) |
| --- | --- | --- |
| Question | does it generalize to a new signer? | what model do we ship? |
| Data | LOSO: one signer held out per fold | **all 4,222 sequences** |
| Held-out data | yes, an entire unseen signer | **none** |
| Stopping rule | early stopping on validation macro F1 | **fixed budget frozen in advance** |
| Checkpoint | `best.pt`, selected by a metric | `deployment.pt`, the final epoch |
| Reportable performance | **67.63% / 0.6607 held-out** | **none — nothing was held out** |

**The recognition result for this project remains the TASK-009B LOSO evidence.**
The deployment model's training accuracy is an in-sample number on data it was
fitted to. It is not a performance estimate, it must never be quoted as one, and
this checkpoint must never be described as independently tested. That statement is
carried in the plan, in the checkpoint metadata, in the training summary, and is
printed by the CLI at the end of every run.

### Why all 4,222 samples may now be used

The held-out-signer evaluation is already complete and frozen. The LOSO
train/validation/test roles existed to answer the scientific question; that
question has been answered and its answer is not revised by anything here. For
deployment fitting there is no reason to withhold scarce data from a model whose
generalization has already been characterized separately.

## Selected configuration — frozen, no new model selection

Taken from TASK-009B evidence. TASK-009C compares nothing, tunes nothing and
searches nothing.

| Setting | Value |
| --- | --- |
| `feature_set` | `full` |
| `quaternion_policy` | `absolute` |
| `pooling` | `masked_mean` |
| `input_policy` | `values_and_feature_valid` |
| `hidden_size` / `num_layers` | 192 / 2 |
| `dropout` | 0.3 |
| `bidirectional` / `input_projection` | false / none |
| Model input dimension | **92** (46 values ‖ 46 validity flags) |
| Parameters | 521,500 |
| Optimizer | AdamW, lr 1e-3, weight decay 1e-4 |
| Gradient clipping | 5.0 |
| Batch size | 32 |
| Loss | unweighted cross-entropy |
| Seed | 1337 |

These were **re-read from the three persisted TASK-009B `result.json` files and
checked for agreement**, not retyped from memory. The plan builder refuses to
proceed if any fold disagrees with the frozen expectation or with the other folds.

## LOSO performance reference

Verified from the persisted TASK-009B artifacts:

| | S01 | S02 | S03 | **mean** |
| --- | ---: | ---: | ---: | ---: |
| Accuracy | 56.78% | 74.16% | 71.95% | **67.63%** |
| Macro F1 | 0.5441 | 0.7307 | 0.7073 | **0.6607** |

This is the only generalization evidence that exists for this configuration, and
it is copied into the deployment plan and into the deployment checkpoint's
metadata so the two can never be separated.

## The epoch budget — the only new decision

An all-data model has no validation split, so early stopping is impossible
without either withholding scarce data or choosing a stopping point by watching
training loss. Both are bad: the first wastes data, the second selects on the
data being fitted.

Instead the budget comes from evidence already spent. Each TASK-009B primary fold
chose its best epoch on legitimate validation data **before any test set was
touched**:

| Fold | best epoch | best validation macro F1 | stopped early |
| --- | ---: | ---: | --- |
| S01 | **28** | 0.9841 | yes (40 epochs run) |
| S02 | **20** | 0.9820 | yes (32 epochs run) |
| S03 | **27** | 0.9910 | yes (39 epochs run) |

```
sorted     : [20, 27, 28]
median     : 27.0
FROZEN     : 27 epochs      (policy: median_primary_loso_best_epoch)
```

**Deterministic rounding rule**, stated so the derivation is reproducible rather
than dependent on a library's tie-breaking: an odd count yields the exact middle
value; an even count yields the mean of the two middle values rounded half-up
(`floor(x + 0.5)`). With three folds no rounding is needed.

The median rather than the mean: the mean of (28, 20, 27) is 25, and the median is
the more robust summary of three values where one (S02, the severe duration-shift
fold) stopped notably earlier than the others.

### A disclosed caveat about epochs vs steps

The LOSO best epochs were measured on **2,372** training sequences per fold.
Deployment trains on **4,222** — **1.78× more data** — so one epoch here is 1.78×
more gradient steps than one epoch there. A step-matched budget would be roughly
15 epochs rather than 27.

The frozen rule is a median of **epochs**, as specified, and it is applied exactly
as specified. This consequence is recorded in the plan under
`epoch_budget_caveat` and stated here rather than silently adjusted, because
changing the rule after seeing its implication would be exactly the kind of
post-hoc choice the rule exists to prevent.

## Training scope and data audit

Run at `--prepare` time, loading **every** sequence through the frozen TASK-009A
path — not assumed:

| Check | Result |
| --- | --- |
| Indexed sequences | **4,222 / 4,222** |
| Loaded successfully | **4,222** |
| **Rejected** | **0** |
| Signers | 01: 1,402 · 02: 1,401 · 03: 1,419 |
| Classes | 28 / 28 |
| Duplicate sample IDs | 0 |
| Contract | `task009a_sequence_input_v1` |
| Feature dimension observed | 46 (uniform across all 4,222) |
| Model input dimension | 92 |

No sample is silently skipped: a rejection anywhere fails plan generation.

## Checkpoint compatibility — zero code change

**This was the critical requirement, and zero-change compatibility was achieved
without misleading metadata.**

The existing `task009b_checkpoint_v1` schema already represents a deployment model
honestly, because its loader treats `fold` as *run identity* and explicitly not as
a compatibility constraint. The deployment checkpoint therefore carries:

```
experiment.fold          = "all"        (not a fold number — there is no held-out signer)
best_metric_name         = "not_applicable_fixed_epoch_budget"
extra.training_role      = "deployment_all_signers"
extra.training_scope     = "all_core28_sequences"
extra.training_samples   = 4222
extra.signers            = ["01", "02", "03"]
extra.epoch_policy       = "median_primary_loso_best_epoch"
extra.deployment_epochs  = 27
extra.source_primary_best_epochs = {"01": 28, "02": 20, "03": 27}
extra.held_out_data      = "none -- fitted on every available sequence"
extra.loso_reference     = { the TASK-009B numbers }
extra.scientific_status  = "POST-EVALUATION DEPLOYMENT TRAINING ..."
```

`best_metric_name` is deliberately **not** left at the research default
`"validation_macro_f1"`, which would imply a selection that never happened.

**Proven in the smoke run, both directions:**

* the TASK-009C `deployment.pt` loads through `SequenceRecognizer.from_checkpoint`
  and classifies a variable-length sequence to an authoritative Arabic label;
* a real TASK-009B research `best.pt` still loads through the same recognizer,
  unchanged.

**Required integration change for the visualizer runtime: none.** The recognizer
reads `feature_set` and `quaternion_policy` from the checkpoint and is indifferent
to `fold`. A runtime that wishes to *display* provenance can optionally read
`extra.training_role` to show whether it is serving a research fold model or the
deployment model — an additive, non-breaking read.

> Note: no TASK-007D visualizer package exists on this branch. The only
> `SequenceRecognizer` consumer here is `recognition/models/inference.py`, and
> compatibility was proven against that.

## Deployment plan

Written to `<run-root>/deployment_plan.json`; a copy is committed at
`reports/recognition/TASK-009C-deployment-plan.json` as the record of the frozen
decision, and a test asserts the committed copy still matches the persisted
TASK-009B evidence.

The plan is **immutable**: `--prepare` over an existing plan raises rather than
overwriting, and `--force-plan` is required to discard one deliberately. Resume
additionally refuses to continue if the stored run's epoch budget differs from the
plan's, so the budget cannot drift mid-run.

## CLI

`scripts/run_task009c_deployment_training.py`

`--run-root --data-run-root --loso-run-root --index --labels --deployment-plan
--prepare --force-plan --status --resume --force --device --seed --batch-size
--num-workers --limit-samples --smoke-epochs`

The epoch count is **read from the frozen plan**; the user never guesses it.
`--limit-samples` and `--smoke-epochs` are smoke-only, print a SMOKE MODE warning,
and mark the resulting `training_summary.json` with `smoke_mode: true`.

### Output layout

```
<run-root>/
    deployment_plan.json     frozen recipe (immutable)
    last.pt                  resume checkpoint, written every epoch
    deployment.pt            THE deployment model (final epoch)
    history.json             per-epoch training loss and accuracy
    training_summary.json    run summary + LOSO reference + in-sample warning
    status.json              COMPLETE marker
```

There is no `best.pt`, deliberately: nothing selected a "best", the frozen budget
determined the endpoint.

## Resume semantics

`last.pt` is written every epoch with model, optimizer, epoch, history and RNG
state. The same command plus `--resume` continues from the next epoch. A run whose
`status.json` says COMPLETE is skipped unless `--force` is given.

**Verified by genuine interruption**, not simulation: a 6-epoch fit was killed
after epoch 3, then resumed — it continued at epoch 4 with `rng_restored=True`,
carried its history forward, and the loss continued falling (2.83 → 2.72 → 2.50 →
2.33).

## Smoke evidence

**Not a deployment model and not a result** — tiny strided subsets, few epochs.

| Check | Result |
| --- | --- |
| Plan derivation from real TASK-009B results | 28 / 20 / 27 → median **27** |
| Full data audit | 4,222 loaded, **0 rejected** |
| Plan immutability | second `--prepare` raised, as designed |
| Data loading + tensorization | frozen TASK-009A path, 92-channel input |
| CUDA forward/backward | ran; loss finite and decreasing |
| Optimizer step | weights change |
| Checkpoint write | `deployment.pt` + `last.pt`, atomic |
| Genuine mid-run resume | killed at epoch 3, resumed at 4, RNG restored |
| Budget-change guard | resume with a different budget correctly refused |
| `--status` | reports plan, frozen epochs, state, checkpoint |
| Skip-if-COMPLETE | second run skipped, `--force` required |
| Deployment checkpoint schema | all deployment metadata present and correct |
| `SequenceRecognizer` on `deployment.pt` | loads, predicts, resolves Arabic label |
| `SequenceRecognizer` on research `best.pt` | still loads (backwards compatible) |

## Runtime estimate

Approximate, from the smoke run and the TASK-009B measurement — not a promise.

| Quantity | Value |
| --- | --- |
| Parameters | 521,500 |
| Peak GPU memory (smoke) | **86.8 MiB** of 4,096 MiB |
| Seconds per epoch (smoke, 900 sequences) | 1.45 s |
| TASK-009B reference | 4.68 s/epoch on 2,372 sequences |
| **Estimated seconds per epoch on 4,222** | **~7–8 s** |
| **Estimated total for 27 epochs** | **~3–5 minutes** |
| `deployment.pt` | ~6.3 MB (`last.pt` another ~6.3 MB) |
| Whole run root | ~13 MB |

GPU memory matches TASK-009B (85–88 MiB) as expected — same architecture, same
batch size. Nothing warrants investigation.

## Tests

`tests/test_task009c_deployment.py` — **31 tests**. Covering: median derivation
including order-independence and the even-count rounding rule; the three primary
best epochs being found; rejection of a missing fold, a foreign configuration, a
foreign contract version (top-level *and* experiment-level), and a missing best
epoch; plan immutability and `--force-plan`; malformed and foreign plans;
all-signer dataset scope (4,222 / 28 classes / 3 signers / no duplicates / no LOSO
filtering); deployment checkpoint round-trip; deployment metadata correctness;
`fold="all"` not being a fold number; research-checkpoint backwards compatibility;
both checkpoint kinds serving inference through one recognizer; authoritative
Arabic label resolution; the fixed-budget training loop writing `deployment.pt`
and no `best.pt`; the in-sample warning; genuine partial-run resume; and the
budget-change guard.

**Full suite: 644 passed, 0 failed** (TASK-009B baseline 613 + 31 new; no existing
test weakened). `compileall`: 0 errors.

Two things the tests found and that were fixed: the plan reader checked only the
top-level `contract_version` and not the experiment block's copy, and an early
version of the resume test had itself changed the budget — the guard was right and
the test was wrong.

## Exact user commands

```bash
cd /home/hatim/Graduation-Project-Simulation-task009c

# The plan is ALREADY FROZEN at:
#   /home/hatim/graduation-project-runs/task009c-core28-deployment/deployment_plan.json
# Re-run --prepare only with --force-plan, and only to deliberately discard it.

# FINAL TRAINING (~3-5 minutes)
PYTHONPATH=. python scripts/run_task009c_deployment_training.py \
  --run-root /home/hatim/graduation-project-runs/task009c-core28-deployment \
  --data-run-root /home/hatim/graduation-project-runs/task008-core28-full \
  --device cuda --batch-size 32

# RESUME (identical command plus --resume)
# STATUS
PYTHONPATH=. python scripts/run_task009c_deployment_training.py \
  --run-root /home/hatim/graduation-project-runs/task009c-core28-deployment --status
```

## Verdict

**TASK-009C DEPLOYMENT TRAINING INFRASTRUCTURE READY — USER MUST RUN FINAL TRAINING**
