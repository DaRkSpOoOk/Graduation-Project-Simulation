# Task

TASK-007E — Deployment checkpoint handoff to the Core-28 visualizer.

# Branch

`luna/task-007e-deployment-checkpoint-handoff`, based on frozen TASK-007D
commit `84f9e4e11ddb264e0ddad1f428bea0bab28a7a7c`.

# Scope

This change makes the existing TASK-007D application understand the
metadata of the TASK-009C all-signers checkpoint. It does not retrain a model,
change the TASK-009A tensor contract, change the model architecture, alter
TASK-008 artifacts, or change queue semantics. The checkpoint remains an
explicit CLI input; no machine-specific path is embedded in library code.

The application remains a demonstration/debugging surface. A deployment model
trained on all available sequences has no held-out evaluation set, so its
training behavior is not reported as deployment accuracy.

# Approach

`RecognizerAdapter.from_checkpoint()` continues to validate the existing
TASK-009B checkpoint schema and primary configuration, then reads the
TASK-009C provenance from the checkpoint `extra` mapping. The adapter
classifies provenance as deployment only when the explicit
`training_role=deployment_all_signers` metadata is present. A checkpoint with
`fold=all` but no deployment role is therefore shown as unknown rather than
being inferred from its filename.

The GUI now presents model role, training scope, scientific reference, and
checkpoint filename separately. The exact stored virtual-glove sequence still
goes through the existing TASK-009A tensorization and TASK-009B
`SequenceRecognizer` API once per queued sign.

# Evidence / Sources

- Frozen TASK-007D integration commit:
  `84f9e4e11ddb264e0ddad1f428bea0bab28a7a7c`.
- TASK-009A input contract: `task009a_sequence_input_v1`.
- TASK-009B primary configuration: `full` / `absolute` / `masked_mean` /
  `values_and_feature_valid`, 92 effective input channels and 28 classes.
- Deployment checkpoint:
  `/home/hatim/graduation-project-runs/task009c-core28-deployment/deployment.pt`.
- Deployment checkpoint SHA-256:
  `e3df0f007c542d15a6ff4a7ad090d6a8af58b583357ca1905c4cdcb20c82ad1e`.
- The checkpoint payload records `fold=all`, `seed=1337`,
  `training_role=deployment_all_signers`, `training_scope=all_core28_sequences`,
  `training_samples=4222`, and signers `01`, `02`, `03`.
- Its embedded scientific reference is TASK-009B held-out-signer LOSO mean
  accuracy `67.63%` and macro F1 `0.6607`. This is reference evidence only;
  it is not a metric measured on `deployment.pt`.

# Files Changed

- `visualizer/recognition/adapter.py` — provenance parsing, strict deployment
  metadata validation, and display-safe metadata fields.
- `visualizer/app/main_window.py` — deployment/research provenance fields in
  the recognition panel.
- `tests/test_task007e_deployment_handoff.py` — metadata, compatibility,
  visualizer-only, and real deployment smoke coverage.
- `reports/visualizer/TASK-007E-deployment-checkpoint-handoff.md` — this
  handoff record.

No checkpoint, generated prediction file, TASK-008 artifact, or model weight
is committed.

# How to Run

From the repository root, select the desired checkpoint explicitly:

```bash
PYTHONPATH=. python scripts/run_task007d_visualizer_recognizer.py \
  --run-root /home/hatim/graduation-project-runs/task008-core28-full \
  --checkpoint /home/hatim/graduation-project-runs/task009c-core28-deployment/deployment.pt \
  --text "محمد"
```

For a terminal-only engineering smoke:

```bash
PYTHONPATH=. python scripts/run_task007d_visualizer_recognizer.py \
  --run-root /home/hatim/graduation-project-runs/task008-core28-full \
  --checkpoint /home/hatim/graduation-project-runs/task009c-core28-deployment/deployment.pt \
  --text "محمد" --headless
```

Without `--checkpoint`, the existing visualization-only path remains usable:

```bash
PYTHONPATH=. python scripts/run_task007d_visualizer_recognizer.py \
  --run-root /home/hatim/graduation-project-runs/task008-core28-full \
  --text "م" --headless
```

# Evaluation

The deployment checkpoint was loaded through the unchanged
`SequenceRecognizer.from_checkpoint` path and tested on exact stored
TASK-008 virtual-glove sequences. `محمد` was resolved in logical order and
each sign item received one sequence-level prediction. The repeated `م`
remained two queue events; it reused the adapter's cached result for its same
canonical sample, as designed.

The GUI was also started with the deployment adapter and inspected through
its live Tk variables. It displayed `DEPLOYMENT MODEL`, the all-signers
training scope, the LOSO-reference disclaimer, and `deployment.pt`.

# Results

Deployment metadata loaded successfully:

| Field | Value |
| --- | --- |
| feature set | `full` |
| quaternion policy | `absolute` |
| pooling | `masked_mean` |
| input policy | `values_and_feature_valid` |
| effective input | `92` |
| classes | `28` |
| fold/run | `all` |
| seed | `1337` |
| training role | `deployment_all_signers` |
| training scope | all Core-28 signers / 4,222 sequences |
| best epoch | `27` |

The deterministic deployment smoke produced the following application
outputs. These are demonstration predictions, not accuracy measurements:

| Queue item | Expected | SignID | Sample | Predicted | Max softmax probability |
| ---: | --- | --- | --- | --- | ---: |
| 0 | م | 0055 | `karsl_core28_s01_sign0055_train_rep010` | م | 99.9523% |
| 1 | ح | 0037 | `karsl_core28_s01_sign0037_train_rep026` | ح | 99.0362% |
| 2 | م | 0055 | `karsl_core28_s01_sign0055_train_rep010` | م | 99.9523% |
| 3 | د | 0039 | `karsl_core28_s03_sign0039_test_rep001` | د | 77.2458% |

All four outputs had finite 28-class probability vectors and completed the
queue in indices `0 → 1 → 2 → 3`. The application did not calculate or label
this as `محمد` accuracy.

# Failures / Limitations

- `deployment.pt` has no held-out signer. Its displayed 67.63% / 0.6607
  values are explicitly labeled as the TASK-009B LOSO scientific reference.
- The UI still identifies the application surface as demo-only because
  canonical stored sequences may have been part of model fitting.
- The deployment checkpoint is not part of Git and must be supplied with
  `--checkpoint` on each machine.
- Unknown or malformed deployment provenance is rejected or displayed as
  unknown; the application never silently falls back to a research checkpoint.
- Prediction confidence is the model's maximum softmax probability. It is not
  a calibrated confidence estimate.

# Performance

The checkpoint was loaded once per process on CPU for the engineering smoke;
each sign sequence was inferred once and then cached for repeated queue
items. No timing or accuracy claim is made for the deployment model.

# Comparison

Research checkpoints remain supported. The existing fold-01 checkpoint
continues to display `DEMO / RESEARCH CHECKPOINT` and `held-out signer S01`.
The two roles are metadata-driven and are not selected by filename. The
scientific comparison remains the TASK-009B held-out-signer evaluation, not
the `محمد` application demo.

# Recommendation

KEEP — the deployment handoff is compatible with the frozen recognition and
visualizer contracts, while preserving explicit scientific provenance.

# Reproducibility

The branch was based on the exact TASK-007D commit above. Tests used the
repository's existing suite plus the TASK-007E test module. The real smoke
used the recorded deployment checkpoint SHA-256, the frozen TASK-008 run
root, the repository Core-28 manifest/labels, CPU inference, and the command
shown above. No training, extraction, or artifact regeneration was performed.

Validation commands and results:

```text
python -m unittest discover -s tests -p 'test_*.py'
Ran 648 tests in 9.694s — OK

python -m compileall -q evaluation tracking kinematics visualizer recognition scripts tests
0 errors
```

# Next Steps

Use the deployment checkpoint as the intended all-signers application model
by passing it explicitly. TASK-009C or a later deployment task may provide a
new compatible checkpoint; the adapter should validate its metadata rather
than assuming a filename. Scientific deployment/generalization claims remain
the responsibility of a future held-out evaluation, while the next model
integration point can replace the checkpoint without changing the visualizer
architecture.
