# Core-28 v1.0 milestone freeze

## Task

Consolidate the finalized Core-28 dataset, recognition, deployment, and visualizer work before TASK-010A and TASK-011A.

## Scope

This integration contains no TASK-010A, TASK-011A, NLP, TTS, retraining, or new model work. It preserves the validated TASK-008 production dataset and external TASK-009 checkpoints. The TASK-008 disaster-recovery backup was already uploaded manually by the user and was not recreated or reuploaded.

## Source tips and ancestry audit

Remote `main` was `8cafff48c53522bd9e458f56b47507d3a025930d`.

- Recognition/deployment: `origin/opus/task-009c-core28-deployment-model` at `5a41215fad33860c50bff440c8d17c0b0471b15e`.
- Visualizer/application: `origin/luna/task-007e-deployment-checkpoint-handoff` at `fb42a3da6503753874b29324bbc53c5bce6fb33b`.

Their merge base was `19972cc77ed7345b599c5881d010386c8191bea0` (frozen TASK-009A). The tips were independent: recognition was four commits ahead of that base and visualizer was six commits ahead. No older stacked PR was merged separately.

## Integration branch and merge

Branch: `integration/core28-v1.0-complete`.

The recognition tip was merged first, followed by the visualizer tip, both with `git merge --no-ff`. The second merge had one content conflict in `pyproject.toml`; it was resolved by preserving the recognition optional dependency/package discovery and adding the visualizer optional dependency/`visualizer*` discovery. No subsystem was discarded.

Integration merge commit before this report: `a22fd8c`. The final branch commit containing this report is recorded by Git and PR #34. The final `main` merge commit is recorded after PR merge and can be resolved with `git rev-parse core28-v1.0`.

Integration PR: #34 — Core-28 v1.0: consolidate virtual glove, recognition, deployment, and visualizer.

## Combined validation

```text
PYTHONPATH=. python -m unittest discover -s tests -p 'test_*.py'
Ran 732 tests in 10.987s — OK

python -m compileall -q evaluation pose tracking kinematics virtual_glove recognition visualizer scripts tests
0 errors
```

The merged tree contains TASK-008A/B/C/D, TASK-009A/B/C, and TASK-007A/B/C/D/E source, tests, and reports. No forbidden checkpoint, dataset, run, video, NPZ, or backup ZIP is tracked.

## Deployment verification

External checkpoint:

```text
/home/hatim/graduation-project-runs/task009c-core28-deployment/deployment.pt
SHA-256: e3df0f007c542d15a6ff4a7ad090d6a8af58b583357ca1905c4cdcb20c82ad1e
```

The external run contains `deployment_plan.json`, `deployment.pt`, `last.pt`, `history.json`, `training_summary.json`, and `status.json`. Its status is `COMPLETE`, with 27/27 epochs and `smoke_mode=false`.

Configuration: `full`, `absolute`, `masked_mean`, `values_and_feature_valid`, 92 effective input channels, 28 classes, hidden size 192, two layers, seed 1337, epoch 27, and 521,500 parameters. The model was not committed to Git.

The real application smoke completed:

```text
م → ح → م → د
```

It identified `training_role=deployment_all_signers`, `fold=all`, and displayed TASK-009B LOSO `67.63%` mean accuracy / `0.6607` macro F1 as a scientific reference only. It displayed no held-out signer for deployment and did not report demo predictions as accuracy. The no-checkpoint visualization-only smoke also passed.

## Scientific recognition record

TASK-009B remains the generalization evidence: held-out-signer LOSO mean accuracy `67.63%` and mean macro F1 `0.6607`. TASK-009C's `97.70%` training accuracy is in-sample and is not deployment accuracy.

## Backup status

### TASK-008

Already uploaded manually by the user and untouched here. Historical archive: `KArSL-Core28-TASK008-VirtualGlove-4222-20260904.zip`; known SHA-256: `6d1fa3cdcba11ac72f1416600b08a7f3c84681d7be38666ff939121bfd8be7e6`.

### TASK-009B

```text
ZIP: /home/hatim/graduation-project-backups/TASK009B-Core28-LOSO-15runs-20260904.zip
bytes: 194299917
SHA-256: 05e3dece100766dc6f511ec1bb87406af6dda323db9d3e6daaa6c32bccc164a2
checksum: /home/hatim/graduation-project-backups/TASK009B-Core28-LOSO-15runs-20260904.zip.sha256
```

`unzip -t` passed. The archive has 91 entries: five requested experiment families × folds 01/02/03, with `result.json`, `history.json`, `predictions.json`, `status.json`, `best.pt`, and `last.pt` in each of the 15 neural runs, plus duration-control output. No cache or Python bytecode was included. A temporary extraction of deterministic beginning/middle/end results passed.

### TASK-009C

```text
ZIP: /home/hatim/graduation-project-backups/TASK009C-Core28-Deployment-20260904.zip
bytes: 12580154
SHA-256: 21d26de18e0125b0af1c4e01de7a8014b2bfbef2b956800c34410365211ca498
checksum: /home/hatim/graduation-project-backups/TASK009C-Core28-Deployment-20260904.zip.sha256
```

`unzip -t` passed. The archive includes the six deployment files and an internal `files.sha256`. The four authoritative source-file hashes were verified before and after archiving. Temporary extraction passed `COMPLETE` status and deployment checkpoint hash checks.

The user must upload these four files manually to Google Drive; no cloud tool, API, credential, or upload mechanism was used:

1. `/home/hatim/graduation-project-backups/TASK009B-Core28-LOSO-15runs-20260904.zip`
2. `/home/hatim/graduation-project-backups/TASK009B-Core28-LOSO-15runs-20260904.zip.sha256`
3. `/home/hatim/graduation-project-backups/TASK009C-Core28-Deployment-20260904.zip`
4. `/home/hatim/graduation-project-backups/TASK009C-Core28-Deployment-20260904.zip.sha256`

## PR disposition

PR #34 is the only consolidation PR. After it is merged and tagged, older stacked PRs #22–#33 should be closed as superseded/incorporated without deleting their branches, retaining history for provenance.

## Artifact policy and home-PC readiness

Checkpoints, datasets, production runs, and backup archives remain outside Git. The visualizer accepts the deployment checkpoint explicitly through `--checkpoint`; its library has no machine-specific deployment default. Historical training/verification scripts and reports may retain provenance paths. No application behavior depends on the old RTX 3050.

## Remaining limitations

- The deployment checkpoint has no held-out evaluation set; scientific claims must continue to use TASK-009B LOSO.
- The visualizer is an application/demo surface, not a new recognition benchmark.
- The optional derived gyro remains diagnostic and the accelerometer remains deferred.
- TASK-010A and TASK-011A are intentionally not started.

## Next tasks

Start TASK-010A and TASK-011A independently from the clean, tagged `main` after consolidation and manual backup upload.
