# TASK-007B — Core-28 keyboard, exemplar catalog, and playback queue

## Task

Build the non-rendering application/control layer that resolves Arabic Core-28
characters to authoritative SignIDs, deterministic TASK-008 visualizer
exemplars, and renderer-neutral isolated-sequence queue items.

## Objective

This branch provides a reusable Arabic keyboard model, mapping API, quality-
ranked exemplar catalog, and ordered playback queue. It does not implement 3D
rendering, geometry, sensor mathematics, a recognizer, or an LSTM.

## Branch

`luna/task-007b-keyboard-exemplar-queue`, based exactly on
`19972cc77ed7345b599c5881d010386c8191bea0` (TASK-009A). No parallel branch was
merged, TASK-005 was not modified, and the external TASK-008 artifacts were
read only.

## Scope

Owned paths are `visualizer/catalog/`, `visualizer/mapping/`,
`visualizer/keyboard/`, `visualizer/queue/`, the two TASK-007B scripts, this
report, and the TASK-007B tests. `pyproject.toml` only adds `visualizer*` to
package discovery so the new control-layer package is installable.

## Contract

The authoritative label source is
`datasets/manifests/karsl_core28_labels.csv`, validated through the existing
`evaluation.dataset.core28` contract. It supplies exactly 28 classes, in
label-index order 0–27, with SignIDs 0032–0059. The virtual-glove source is
`datasets/manifests/karsl_core28_virtual_glove.csv`; it supplies sample IDs,
signer, partition, repetition, source provenance, sequence lengths, and
relative TASK-008 paths. No directory or filename ordering is used for label
resolution.

The supported mapping is:

| label index | character | SignID |
| ---: | :---: | :---: |
| 0 | ا | 0032 |
| 1 | ب | 0033 |
| 2 | ت | 0034 |
| 3 | ث | 0035 |
| 4 | ج | 0036 |
| 5 | ح | 0037 |
| 6 | خ | 0038 |
| 7 | د | 0039 |
| 8 | ذ | 0040 |
| 9 | ر | 0041 |
| 10 | ز | 0042 |
| 11 | س | 0043 |
| 12 | ش | 0044 |
| 13 | ص | 0045 |
| 14 | ض | 0046 |
| 15 | ط | 0047 |
| 16 | ظ | 0048 |
| 17 | ع | 0049 |
| 18 | غ | 0050 |
| 19 | ف | 0051 |
| 20 | ق | 0052 |
| 21 | ك | 0053 |
| 22 | ل | 0054 |
| 23 | م | 0055 |
| 24 | ن | 0056 |
| 25 | ه | 0057 |
| 26 | و | 0058 |
| 27 | ي | 0059 |

`ة`, `أ`, `ؤ`, `ئ`, `ئـ`, `ء`, `إ`, `آ`, `ى`, `لا`, and `ال` are not silently
mapped. A resolver raises `UnsupportedCharacterError` with the original
character. The multi-codepoint tokens `ئـ`, `لا`, and `ال` are rejected as
tokens during text enqueue rather than decomposed into other signs. Text
enqueue is atomic by default: every unsupported position is reported by
`UnsupportedTextError` and the queue remains unchanged. The explicit
`unsupported_policy="report"` mode enqueues supported letters and separator
gaps while returning/storing all unsupported positions; nothing is dropped
without a report.

Spaces, newlines, tabs, and Unicode punctuation are explicit queue items with
`item_type="gap"`, `transition_policy="neutral_gap"`, and a default 250 ms
duration. They are playback boundaries, never fake signs or sensor frames.

## Architecture

The control path is:

`Core28Mapping` → `Core28ExemplarCatalog` → `Core28Resolver` →
`PlaybackQueue`.

`SequenceDescriptor` is a frozen dataclass containing the sample ID, run root,
pose/tracking/kinematics/virtual-glove relative paths, source provenance, and
sequence length. `PlaybackQueueItem` contains the character, SignID,
label-index, sample ID, signer, descriptor, state, and optional gap hint. No
TASK-007A renderer or GUI type is imported.

The catalog contains one `entries` record for each canonical class, a
`signer_exemplars` record for each available signer, and a compact candidate
index used by seeded random selection. The canonical and signer selectors are
quality ranked. `random` is also deterministic when an explicit integer
`rng_seed` is supplied; omitting the seed is an error.

## Alignment

Catalog construction audits all 4,222 virtual-glove manifest rows and checks
the row's class/label-index mapping, sample ID, virtual-glove path, sequence
length against `frame_index`, strict frame/timestamp order, and available
TASK-008 sidecar provenance. Descriptor paths are derived from the sample ID
and manifest relative path, not filesystem enumeration.

The queue preserves input order exactly. For `محمد`, canonical resolution is
four sign items:

`م → 0055`, `ح → 0037`, `م → 0055`, `د → 0039`.

Repeated characters therefore remain repeated queue items and share the same
canonical sample selection.

## Sensor Layout QA

The catalog builder requires a machine-readable layout with exactly 15 bend
Hall/magnetic entries, 4 adjacent-spread Hall/magnetic entries, and 1 palm
IMU entry. It checks unique IDs, complete five-finger/three-joint bend
coverage, complete four adjacent-spread pair coverage, valid sensor types,
logical locations, descriptions, and no duplicate logical assignment. Every
Hall entry must have `display_marker="H"`; the palm IMU must have
`display_marker="IMU"`. A candidate with a layout violation is not eligible
for selection.

The generated production catalog audited 20 sensors per hand, with all 4,222
candidate layouts passing. The historical standalone-surface-file probe was
too narrow: TASK-008 does not emit separate OBJ/PLY/GLB surface files, but its
raw pose artifacts do contain embedded MANO vertex clouds. The corrected
catalog terminology now keeps these facts separate:

* `embedded_mano_vertices_available` — stored in
  `pose/raw/<sample_id>/wilor_raw.npz`;
* `tracked_landmarks_3d_available` — stored in the tracked pose artifact;
* `surface_triangle_topology_available` — false for this TASK-008 run because
  triangle faces are not serialized.

The prior `mesh_available=false` wording therefore must not be read as
geometry-unavailable; it described only the absence of a separate surface
file.

## Normalization QA

For every valid bend/spread channel, the builder checks
`normalized ≈ angle_deg / 180.0` with a `1e-5` absolute tolerance and checks
the physical normalized range `[0, 1]`. It never repairs, clamps, or rewrites a
value. Invalid angle/normalized channels must remain NaN. Metadata is checked
for the fixed physical contract and no fitted or run-specific min/max terms
are used.

## Model-B Validity

Validity is channel independent. `bend_valid`, `spread_valid`, and
`palm_imu_valid` are never collapsed into an all-or-nothing hand flag. A
finite valid bend channel, a NaN invalid spread pair, and a valid palm IMU is a
legal partial sequence. The synthetic test covers this case and confirms the
catalog remains selectable. The queue carries the original descriptor; it does
not impute, interpolate, or create transition frames.

## Rotation QA

For valid orientation entries the builder reports and ranks using matrix
orthogonality, determinant, quaternion norm, and matrix/quaternion consistency
checks. No orientation is normalized or repaired by TASK-007B. Across the
4,222 production candidates, the worst observed values were:

| diagnostic | worst observed |
| --- | ---: |
| matrix orthogonality max error | `9.892103847164435e-08` |
| quaternion norm max error | `4.715822465861663e-08` |
| matrix/quaternion max absolute error | `9.696114355861596e-08` |
| minimum determinant | `0.9999998989619531` |

All valid orientation entries passed the builder tolerance of `1e-4`.

## ADC QA

ADC fields remain optional. When present, valid channels are checked against
`0..4095`, invalid channels use `-1`, and half-up transfer agreement is
checked. The production run contains all optional ADC arrays. There were 50
one-count differences when recomputing from serialized float32 normalized
values (48 bend and 2 spread); all agree with the source angle at the exact
half-up boundary and are recorded as float32 representation effects. A true
range, sentinel, or transfer disagreement rejects that candidate. No ADC value
is repaired.

## Temporal Diagnostics

TASK-007B does not smooth or resample sequences. The descriptor preserves the
original sequence length, frame index, timestamp, and source paths for the
future renderer/recognizer. Queue transitions are explicit neutral gaps only;
no synthetic geometry frames are generated and no queue item is suitable as
ML training data. Exemplar ranking uses class-median duration proximity and a
deterministic IQR-scaled outlier penalty.

## Canonical Exemplars

All 28 classes have exactly one canonical exemplar in
`visualizer/catalog/core28_exemplars.json` and the compact CSV companion.
`hand` is the pose-bearing tracking fraction used by the class-aware score;
`geometry` reports TASK-008 pose/tracking/kinematics availability. The
corrected catalog separately reports embedded MANO vertices, tracked
21-landmarks, and surface topology.

| character | SignID | sample_id | signer | length | hand | bend | spread | IMU | geometry | score | reason |
| :---: | :---: | --- | :---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | --- |
| ا | 0032 | `karsl_core28_s03_sign0032_train_rep019` | 03 | 25 | 1.0 | 1.0 | 1.0 | 1.0 | true | 1.0 | quality rank #1 |
| ب | 0033 | `karsl_core28_s01_sign0033_train_rep022` | 01 | 26 | 1.0 | 1.0 | 1.0 | 1.0 | true | 0.9772727273 | quality rank #1 |
| ت | 0034 | `karsl_core28_s01_sign0034_train_rep010` | 01 | 19 | 1.0 | 1.0 | 1.0 | 1.0 | true | 1.0 | quality rank #1 |
| ث | 0035 | `karsl_core28_s01_sign0035_train_rep019` | 01 | 19 | 1.0 | 1.0 | 1.0 | 1.0 | true | 1.0 | quality rank #1 |
| ج | 0036 | `karsl_core28_s01_sign0036_test_rep007` | 01 | 20 | 1.0 | 1.0 | 1.0 | 1.0 | true | 1.0 | quality rank #1 |
| ح | 0037 | `karsl_core28_s01_sign0037_train_rep026` | 01 | 20 | 1.0 | 1.0 | 1.0 | 1.0 | true | 1.0 | quality rank #1 |
| خ | 0038 | `karsl_core28_s03_sign0038_train_rep042` | 03 | 22 | 1.0 | 1.0 | 0.9659090909 | 1.0 | true | 0.9837752525 | quality rank #1 |
| د | 0039 | `karsl_core28_s03_sign0039_test_rep001` | 03 | 21 | 1.0 | 1.0 | 0.75 | 1.0 | true | 0.9569444444 | quality rank #1 |
| ذ | 0040 | `karsl_core28_s03_sign0040_train_rep004` | 03 | 19 | 1.0 | 1.0 | 0.9934210526 | 1.0 | true | 0.9990131579 | quality rank #1 |
| ر | 0041 | `karsl_core28_s01_sign0041_test_rep006` | 01 | 19 | 1.0 | 1.0 | 0.75 | 1.0 | true | 0.9625 | quality rank #1 |
| ز | 0042 | `karsl_core28_s01_sign0042_test_rep003` | 01 | 21 | 1.0 | 1.0 | 1.0 | 1.0 | true | 1.0 | quality rank #1 |
| س | 0043 | `karsl_core28_s01_sign0043_test_rep003` | 01 | 20 | 1.0 | 1.0 | 1.0 | 1.0 | true | 1.0 | quality rank #1 |
| ش | 0044 | `karsl_core28_s01_sign0044_train_rep007` | 01 | 19 | 1.0 | 1.0 | 1.0 | 1.0 | true | 1.0 | quality rank #1 |
| ص | 0045 | `karsl_core28_s03_sign0045_train_rep014` | 03 | 23 | 1.0 | 1.0 | 1.0 | 1.0 | true | 0.9828571429 | quality rank #1 |
| ض | 0046 | `karsl_core28_s01_sign0046_train_rep041` | 01 | 20 | 1.0 | 1.0 | 0.75 | 1.0 | true | 0.9625 | quality rank #1 |
| ط | 0047 | `karsl_core28_s03_sign0047_train_rep016` | 03 | 19 | 1.0 | 1.0 | 1.0 | 1.0 | true | 0.995 | quality rank #1 |
| ظ | 0048 | `karsl_core28_s01_sign0048_test_rep004` | 01 | 20 | 1.0 | 1.0 | 0.75 | 1.0 | true | 0.9625 | quality rank #1 |
| ع | 0049 | `karsl_core28_s01_sign0049_train_rep013` | 01 | 20 | 1.0 | 1.0 | 1.0 | 1.0 | true | 1.0 | quality rank #1 |
| غ | 0050 | `karsl_core28_s01_sign0050_train_rep003` | 01 | 20 | 1.0 | 1.0 | 0.75 | 1.0 | true | 0.9625 | quality rank #1 |
| ف | 0051 | `karsl_core28_s03_sign0051_test_rep006` | 03 | 24 | 1.0 | 1.0 | 1.0 | 1.0 | true | 0.9833333333 | quality rank #1 |
| ق | 0052 | `karsl_core28_s03_sign0052_train_rep002` | 03 | 19 | 1.0 | 1.0 | 0.8618421053 | 1.0 | true | 0.9741481107 | quality rank #1 |
| ك | 0053 | `karsl_core28_s01_sign0053_test_rep001` | 01 | 21 | 1.0 | 1.0 | 1.0 | 1.0 | true | 1.0 | quality rank #1 |
| ل | 0054 | `karsl_core28_s01_sign0054_train_rep005` | 01 | 20 | 1.0 | 1.0 | 1.0 | 1.0 | true | 1.0 | quality rank #1 |
| م | 0055 | `karsl_core28_s01_sign0055_train_rep010` | 01 | 20 | 1.0 | 1.0 | 1.0 | 1.0 | true | 1.0 | quality rank #1 |
| ن | 0056 | `karsl_core28_s01_sign0056_train_rep026` | 01 | 19 | 1.0 | 1.0 | 0.9605263158 | 1.0 | true | 0.9890789474 | quality rank #1 |
| ه | 0057 | `karsl_core28_s01_sign0057_train_rep003` | 01 | 20 | 1.0 | 1.0 | 1.0 | 1.0 | true | 1.0 | quality rank #1 |
| و | 0058 | `karsl_core28_s01_sign0058_test_rep002` | 01 | 20 | 1.0 | 1.0 | 1.0 | 1.0 | true | 1.0 | quality rank #1 |
| ي | 0059 | `karsl_core28_s01_sign0059_test_rep008` | 01 | 21 | 1.0 | 1.0 | 0.75 | 1.0 | true | 0.9625 | quality rank #1 |

## Queue Semantics

`PlaybackQueue` supports `enqueue_character`, `enqueue_text`, `peek`,
`start`, `advance`, `pop`, `clear`, `reset`, `current`, `remaining`,
`completed`, `completed_items`, and `remaining_items`. Item state transitions
are `PENDING → PLAYING → COMPLETED`; `FAILED` and `UNAVAILABLE` are explicit
terminal states for future artifact/load failures. A failed item is never
substituted with a different valid SignID.

The queue models isolated KArSL signs letter by letter. It does not claim to
produce natural continuous Arabic fingerspelling or co-articulated motion.

## Transition Semantics

Only neutral gap instructions are emitted for separators. TASK-007B never
interpolates sensor values or inserts geometry frames. TASK-007A may choose how
to display the gap. These presentation instructions must not be sent to ML
training or recognition tensorization.

## Renderer Integration Contract

`PlaybackQueueItem.sequence_descriptor` is the integration boundary. It
contains:

```text
sample_id
run_root
pose_relative_path
tracking_relative_path
kinematics_relative_path
virtual_glove_relative_path
sequence_length
source_relative_path
source_sha256
manifest_sha256
signer_id / official_partition / repetition_id
```

TASK-007A can load the exact sequence from those plain data fields without
label resolution or importing this task's queue internals.

## Future LSTM Integration Contract

No model is created here. The exact `sample_id` and virtual-glove descriptor
remain attached to each sign item so a future recognizer can pass the same
isolated sequence through TASK-009A tensorization/TASK-009B inference and
display expected character, predicted character, and confidence. Queue gaps
and renderer-only transition instructions are not recognition samples.

## CLI

Build the catalog from the immutable external run:

```bash
python scripts/build_task007b_exemplar_catalog.py \
  --run-root /home/hatim/graduation-project-runs/task008-core28-full
```

This writes the deterministic JSON and compact canonical CSV catalog. Demo
the required repeated-character input:

```bash
python scripts/demo_task007b_queue.py --text "محمد"
```

Use `--mode signer02`/`signer03` for deterministic signer-specific choices or
`--mode random --seed 17` for seeded selection. `--no-contract-demo` limits
output to the requested queue.

## Tests

Synthetic tests cover complete and malformed catalog inputs, missing/duplicate
sensors, H/IMU marker violations, normalization/range faults, partial spread
NaN, illegal NaN, provenance/frame mismatch, rotation/quaternion faults,
optional ADC mismatch, deterministic JSON/CSV, mapping/index integrity,
repeated `محمد` queue behavior, spaces, unsupported Unicode, signer modes,
seeded random mode, descriptor fields, and production sample existence.

Executed:

```bash
python -W error -m unittest -q tests.test_task007b_keyboard_exemplar_queue
python -m unittest discover -s tests -p 'test_*.py'
python -m compileall -q visualizer scripts tests evaluation recognition pose tracking video_io hand_kinematics
```

Results: 13 TASK-007B tests passed; 561 repository tests passed; scoped
compileall passed. A broad `compileall .` also traversed the local `.venv` and
reported pre-existing Python 3.14 syntax incompatibilities in third-party
`OpenGL`, `chumpy`, and a MediaPipe test file; those files are not repository
changes. `pytest` is not installed in the environment, so the repository's
documented unittest runner was used.

## Files Changed

- `visualizer/catalog/descriptor.py`, `catalog.py`, `builder.py`
- `visualizer/catalog/core28_exemplars.json` and `.csv`
- `visualizer/mapping/core28.py`
- `visualizer/keyboard/core28.py`
- `visualizer/queue/playback_queue.py`
- `scripts/build_task007b_exemplar_catalog.py`
- `scripts/demo_task007b_queue.py`
- `tests/test_task007b_keyboard_exemplar_queue.py`
- `pyproject.toml`
- this report

No production arrays, videos, model weights, TASK-005 files, renderer files,
or recognition model files were changed.

## Evidence / Sources

- `datasets/manifests/karsl_core28_labels.csv` — SHA-256
  `491122a0967943a783b5805e6469e501fdfa1f9f9d2c057070c5a842281a0d45`.
- `datasets/manifests/karsl_core28_virtual_glove.csv` — SHA-256
  `418a78fdaf189f26b0b76d76707c312c018321e45067bfd11a1cbce1c7cf4ef7`.
- `visualizer/catalog/core28_exemplars.json` — SHA-256
  `46b890a51ad8682bc9a62036efc12cfb8cb9f7c5e7d366bb7a56a5185772b770`.
- `visualizer/catalog/core28_exemplars.csv` — SHA-256
  `4d8ff3aa729c6693c85babd32a6e7401b665b61bbef6955ad49ae53dab5a91b1`.
- `evaluation/dataset/core28.py` — existing authoritative label validation.
- `recognition/data/contract.py` — existing TASK-009A frozen feature/order
  contract, read only.
- `/home/hatim/graduation-project-runs/task008-core28-full` — read-only
  production TASK-008 run; the catalog records its absolute run root and
  relative artifact paths.

## Evaluation

The builder audited 4,222 rows, accepted 4,222 candidates, and generated 28
canonical exemplars plus 28 entries for each available signer (`01`, `02`,
`03`). Every canonical sample ID exists in the authoritative manifest and its
virtual-glove, kinematics, tracking, and pose artifacts exist in the external
run. The generated catalog is deterministic for identical inputs.

## Results

The required demo produces four queue entries for `محمد` and preserves both
occurrences of `م`. The canonical mode is stable across repeated resolutions;
signer modes select only the requested signer; random mode requires and uses an
explicit seed. All 28 classes have one and only one canonical entry.

## Failures / Limitations

TASK-008 optional ADC data has 50 one-count float32-boundary differences when
recomputed from the serialized normalized array; the catalog records this
transparent audit detail and accepts the source-angle-consistent values. The
run has embedded MANO vertex clouds and tracked landmarks, but no
standalone surface mesh file or triangle topology; the descriptor therefore
exposes the pose/tracking/kinematics paths and the renderer loads the embedded
vertices from the pose artifact. Queue playback remains isolated-sign
playback, not continuous co-articulation. GUI rendering, recognizer
inference, and confidence display remain future work.

## Performance

Catalog generation audits 4,222 compressed virtual-glove files in approximately
20 seconds in the development environment and writes a roughly 3.5 MB JSON
metadata catalog plus an 11 KB canonical CSV. Queue resolution is in-memory
metadata lookup; no array is loaded by the queue.

## Comparison

The implementation consumes the existing authoritative manifests and TASK-009A
orders rather than defining a competing label map or renderer data model.
Canonical selection is quality-ranked, signer-aware, and deterministic; queue
items preserve repeated text rather than deduplicating characters.

## Reproducibility

Run root: `/home/hatim/graduation-project-runs/task008-core28-full`.
Catalog build inputs, SHA-256 values, selector policy, and class profiles are
stored in the generated catalog. Canonical/tie behavior is seed-free and uses
descending score followed by ascending `sample_id`. Random behavior derives a
per-SignID SHA-256 seed from the supplied integer seed. Development runtime was
Python 3.14.4 and NumPy 2.5.2 on the existing local environment.

## Recommendation

KEEP — ready for TASK-007A renderer integration and future TASK-009B inference
integration, subject to the ADC boundary note above.

## Next Steps

TASK-007A can consume `PlaybackQueueItem.sequence_descriptor` and implement its
own presentation of neutral gaps. A future recognizer integration can consume
the exact sample IDs through TASK-009A without treating gap items as data.
