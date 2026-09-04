# TASK-007D — Recognizer / Visualizer Integration

## Task

Integrate the frozen TASK-009B inference API with the TASK-007C Core-28
keyboard, exemplar resolver, playback queue, and 3D virtual-glove visualizer.
This is an engineering demonstration surface, not a new recognition
experiment.

## Branch

`luna/task-007d-recognizer-visualizer-integration`

The branch starts from the exact TASK-007C full commit
`cbc6b1f48c7fd03b7466814416fdb84edd227700`, itself based on the TASK-009A
starting point `19972cc77ed7345b599c5881d010386c8191bea0`.

## Scope

Included:

* one explicit, optional TASK-009B checkpoint selection;
* checkpoint contract validation and display metadata;
* sequence-level recognition for the exact queued TASK-008 virtual-glove
  artifact;
* cached predictions attached to queue sign items;
* GUI and headless presentation of expected label, predicted label, and maximum
  softmax probability;
* graceful visualization-only and per-sequence recognition failure behavior.

Not included:

* training, fine-tuning, checkpoint selection, or preprocessing changes;
* a new benchmark or accuracy claim;
* changes to TASK-004, TASK-005, TASK-006, TASK-008, or TASK-009B;
* LSTM code, keyboard semantics, exemplar ranking, or model architecture
  changes.

## Approach

The integrated path is:

```text
PlaybackQueueItem
  → SequenceDescriptor.sample_id + virtual_glove_relative_path
  → exact configured run-root/virtual_glove/*.npz
  → TASK-009A load_sequence_arrays/build_feature_tensor/collate_sequences
  → TASK-009B SequenceRecognizer.predict_sequence
  → RecognitionResult
  → GUI/headless display
```

The renderer never derives features from screen coordinates.  The recognizer
does not receive RGB, neutral gaps, synthetic frames, or rendered geometry.
One sign is classified once when it becomes active; a small adapter cache keeps
restarts and repeated canonical exemplars from recomputing the same stored
sample unnecessarily.

## Evidence / Sources

The visualizer input is the frozen TASK-007C implementation and production
artifact contract.  The integrated recognition runtime is the exact
TASK-009B infrastructure commit:

`6ea653e59a0bd824bc9ee1c42a53346f59150ddd`

This commit supplies `SequenceRecognizer`, TASK-009A tensorization use,
checkpoint loading, and model reconstruction.  The separate TASK-009B
analysis/final-results commit is recorded but intentionally not copied because
its post-training analysis is not a runtime dependency:

`0ddf0e6fed31a08753b0d7a20ece7b5a5cdc953b`

No recognition experiment result or checkpoint was modified.

## Files Changed

* `visualizer/recognition/adapter.py` — checkpoint validation, exact artifact
  loading, sequence-level adapter, result schema, top-3 mapping.
* `visualizer/recognition/controller.py` — queue-facing cache and graceful
  unavailable state.
* `visualizer/recognition/__init__.py` — optional integration exports.
* `visualizer/app/integration.py` — headless recognizer queue traversal while
  keeping recognition imports lazy for TASK-007C visualization-only use.
* `visualizer/app/main_window.py` — optional recognition panel and per-item
  presentation state.
* `visualizer/app/__init__.py` — integration surface exports.
* `scripts/run_task007d_visualizer_recognizer.py` — GUI/headless entry point.
* `tests/test_task007d_recognizer_integration.py` — adapter, cache, failure,
  visualization-only, checkpoint, and real smoke tests.
* this report.

## How to Run

From the repository root, visualization-only headless playback remains
available without a checkpoint:

```bash
PYTHONPATH=. python scripts/run_task007d_visualizer_recognizer.py \
  --run-root /home/hatim/graduation-project-runs/task008-core28-full \
  --text "م" --headless
```

For the engineering smoke checkpoint, checkpoint selection is explicit:

```bash
PYTHONPATH=. python scripts/run_task007d_visualizer_recognizer.py \
  --run-root /home/hatim/graduation-project-runs/task008-core28-full \
  --checkpoint \
    /home/hatim/graduation-project-runs/task009b-core28-lstm/full/q-absolute/masked_mean/fold01/seed1337/best.pt \
  --text "محمد"
```

Use `--headless` on the same command for terminal output.  The GUI also
accepts `--device cuda` when a compatible CUDA environment is intentionally
selected; the default is CPU for portability.  `--labels`, `--manifest`, and
`--catalog` remain configurable.

## Evaluation

### Checkpoint validation

The adapter rejects a selected checkpoint unless the existing TASK-009B
loader accepts its schema and input contract, and these additional runtime
conditions hold:

| Field | Required value / check |
| --- | --- |
| checkpoint schema | `task009b_checkpoint_v1` |
| TASK-009A contract | `task009a_sequence_input_v1` |
| feature set | `full` |
| quaternion policy | `absolute` |
| pooling | `masked_mean` |
| input policy | `values_and_feature_valid` |
| raw TASK-009A dimension | 46 |
| model effective input | 92 |
| output classes | 28 |
| label table | contiguous indices 0–27 and compatible with queue labels |

The checked engineering checkpoint records fold `01`, seed `1337`, best epoch
`28`, and the required primary configuration.  Its UI label is explicitly
`DEMO / RESEARCH CHECKPOINT — held-out signer S01`; it is not presented as a
deployment model.

### Queue semantics

Core-28 logical character order remains authoritative.  `محمد` creates four
items, including two distinct `م` items.  Each sign item receives one sequence
prediction; a whitespace/punctuation gap returns no recognition result and is
never sent to the LSTM.  Queue completion still advances automatically through
all four items.

On checkpoint load failure, the application keeps the TASK-007C visualizer
usable and displays recognition as unavailable.  On an individual model or
artifact error, the adapter/controller records an explicit unavailable result;
it never replaces a missing prediction with the expected character and never
marks the queue item failed solely because recognition failed.

## Results

### Real engineering smoke

Source run:
`/home/hatim/graduation-project-runs/task008-core28-full`

Checkpoint:
`/home/hatim/graduation-project-runs/task009b-core28-lstm/full/q-absolute/masked_mean/fold01/seed1337/best.pt`

The deterministic `محمد` smoke traversed the following stored sequences:

| Queue index | Expected | SignID | Sample | Frames | Observed predicted label | Max softmax probability |
| ---: | :---: | :---: | --- | ---: | :---: | ---: |
| 0 | م | 0055 | `karsl_core28_s01_sign0055_train_rep010` | 20 | م | 99.7890% |
| 1 | ح | 0037 | `karsl_core28_s01_sign0037_train_rep026` | 20 | ح | 99.6466% |
| 2 | م | 0055 | `karsl_core28_s01_sign0055_train_rep010` | 20 | م | 99.7890% |
| 3 | د | 0039 | `karsl_core28_s03_sign0039_test_rep001` | 21 | د | 99.2787% |

These are observed outputs from a known stored demonstration sequence.  They
are not a new accuracy result: no visualizer accuracy, keyboard accuracy,
`محمد` accuracy, or canonical-exemplar accuracy is claimed.

The headless result confirmed completion indices `0, 1, 2, 3`, exact sample
and frame associations, and no generated ML frames.  The repeated `م` reused
the adapter's sample cache while remaining two separate queue/playback events.

### Visualization-only smoke

Without `--checkpoint`, the same entry point loaded the real `م` sequence and
completed the queue with no recognition field or model invocation.  A desktop
Tk/Matplotlib smoke with the checkpoint also started, attached the panel, and
closed cleanly after the real sequence was activated.

## Failures / Limitations

* The selected fold-01 checkpoint is a research/demo checkpoint.  It is not a
  universal deployment checkpoint; TASK-009C must supply that later.
* The checkpoint is selected through the explicit `--checkpoint` CLI option;
  the current UI displays validated metadata but does not implement a file
  picker.  No checkpoint path is hardcoded as the permanent model.
* Confidence is labeled as maximum softmax probability, exactly as returned by
  TASK-009B.  No calibration claim is made.
* Recognition is sequence-level, so predictions are intentionally not updated
  at every rendered frame.
* GUI playback requires a local desktop/Tk/Matplotlib display.  Headless mode
  is provided for CI and integration checks.
* A failed checkpoint or individual inference is visible as unavailable; this
  integration does not silently retry, fabricate sensor values, or alter
  queue expectations.

## Performance

The adapter loads the selected model once at application startup and calls the
model once per unique queued sample.  It does not preload the 4,222-sequence
production run.  The four-item CPU `محمد` headless smoke completed in about
3.9 seconds wall time in this development environment, including checkpoint
load and four sequence predictions.  This is an engineering observation, not a
scientific throughput benchmark.

## Comparison

The visualizer is not a comparison protocol.  The scientific TASK-009B LOSO
results remain those in the frozen TASK-009B analysis history.  Demonstration
predictions may come from a signer/split represented in the selected checkpoint
and must not be converted into a new metric.

## Recommendation

NEEDS MORE EVALUATION

The integration is suitable as a deployment-model handoff surface, but a
deployment checkpoint and a scientifically valid held-out evaluation remain
separate work.

## Reproducibility

* base visualizer commit: `cbc6b1f48c7fd03b7466814416fdb84edd227700`;
* integrated TASK-009B infrastructure: `6ea653e59a0bd824bc9ee1c42a53346f59150ddd`;
* frozen TASK-009B analysis reference: `0ddf0e6fed31a08753b0d7a20ece7b5a5cdc953b`;
* source run: `/home/hatim/graduation-project-runs/task008-core28-full`;
* checkpoint configuration: full / absolute / masked_mean / fold01 / seed1337;
* test command:
  `python -m unittest discover -s tests -p 'test_*.py'`;
* compile command:
  `python -m compileall -q evaluation tracking kinematics visualizer recognition scripts tests`.

All production input files remain outside Git and were read only.  No model
weights, raw NPZs, generated videos, or synthetic frames were added.

## Next Steps

TASK-009C can provide an explicitly documented deployment checkpoint that is
swappable through the same adapter without an architectural change.  A later
task may add a scientifically valid recognition evaluation surface; this
visualizer must remain a demonstration/debugging consumer of the frozen stored
sequences.
