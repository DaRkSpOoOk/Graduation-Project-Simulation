# TASK-007C — Core-28 visualizer integration

## Task

TASK-007C — integrate the Core-28 Arabic keyboard/exemplar queue with the
TASK-007A 3D virtual smart-glove playback engine.

## Branch

`luna/task-007c-visualizer-integration` (unmerged; based on the exact TASK-009A
base listed below).

## Scope

This branch owns only the queue-to-renderer integration, desktop presentation,
and the required embedded-vertex terminology correction. It does not modify
recognition/model/training code, TASK-008 production artifacts, or either
underlying task's mathematical extraction behavior.

## Objective

Integrate the independent TASK-007A 3D playback foundation and TASK-007B
Core-28 keyboard/catalog/queue into one local application. The application
accepts Arabic Core-28 text or keyboard clicks, resolves each logical
character to a deterministic TASK-008 exemplar, loads one stored sequence at a
time, and advances automatically through the queue. It does not run a
recognizer, train a model, or create new ML frames.

## Approach

Combine the two frozen task heads in an isolated worktree, resolve only their
package/export overlap, then validate the complete path with real production
artifacts and a renderer-independent headless simulation.

## Branches integrated

The integration branch is `luna/task-007c-visualizer-integration`, based on
`main`'s TASK-009A base
`19972cc77ed7345b599c5881d010386c8191bea0`.

| input | source head | integrated commit | result |
| --- | --- | --- | --- |
| TASK-007A | `58c548e21e47cbb3bb2e444b40560b425d709f64` | `432f2f55f1f62c231f687ea4b591f5a10c879cd0` | renderer, loader, playback, geometry |
| TASK-007B | `16a00b233102b38f533ce90d31631dce7cff9540` | `c457179bb06a160c62e550af982ee0feb8e8e3d6` | mapping, catalog, keyboard, queue |

Both source heads were verified against the remote branches before
cherry-picking. Because the source branches were parallel, the cherry-picked
commit IDs differ from the source commit IDs while preserving their complete
patches. The only integration conflicts were package discovery in
`pyproject.toml` and exports in `visualizer/__init__.py`; both sides were
retained. PR #28 and PR #29 remain unmerged.

## Integration strategy

The queue remains responsible for Unicode mapping, exemplar selection, item
state, and neutral gaps. The renderer remains responsible for loading and
presenting one `PlaybackSequence`. `visualizer.app.integration` is the small
adapter between them:

```text
Core28Mapping / Resolver
          ↓
PlaybackQueueItem
          ↓
load_sequence_for_item(run_root, descriptor)
          ↓
PlaybackSequence
          ↓
MatplotlibGloveViewer
```

The adapter uses the caller's configurable run root and the descriptor's
sample ID, then checks descriptor length, label/SignID metadata, and geometry
presence. It never creates a sample for a gap and never changes the stored
arrays.

## Architecture

The desktop entry point is
`scripts/run_task007c_visualizer_app.py`. `visualizer.app.integration` is
headless and renderer-neutral; `visualizer.app.main_window` contains only Tk
presentation and controls. Existing TASK-007A loader, playback controller,
geometry marker placement, and Matplotlib viewer are reused. Existing TASK-007B
mapping, catalog, keyboard, `SequenceDescriptor`, and `PlaybackQueue` are
reused rather than reimplemented.

The application loads the requested sequence on demand. It does not preload
the 4,222 production samples or copy production artifacts.

## GUI layout

The Tk/Matplotlib desktop window contains:

* Arabic text input with a `Queue typed text` action and `Clear queue`;
* an RTL-presented Core-28 keyboard whose callbacks retain the authoritative
  Unicode character;
* canonical, signer-01/02/03, and explicitly seeded-random exemplar modes;
* an embedded 3D viewport on the left and queue/state panel on the right;
* Start, Pause, Next, Restart current, and Restart queue controls;
* speed controls for `0.5×`, `1×`, and `2×`;
* current sample, stored frame, timestamp, label, SignID, and sensor dashboard.

The Matplotlib viewport shows stored 778-point MANO vertex clouds, tracked
21-joint skeletons, and landmark-derived sensor markers. It does not claim to
show a shaded MANO surface.

## Core-28 keyboard

The keyboard is generated from the existing `Core28Keyboard.rtl_rows` and the
authoritative 28-row label mapping. A key press appends exactly one queue item;
repeated presses are not deduplicated. The canonical catalog contains exactly
28 entries, one per Arabic letter, and canonical selection is stable.

## RTL handling

Arabic input is displayed right-justified for native UI readability, but the
internal queue uses the original Unicode/code-point sequence. No reversal is
performed for playback. Thus `محمد` is internally and temporally ordered as
`م → ح → م → د`.

Only the frozen Core-28 characters are accepted. Unsupported characters and
multi-codepoint tokens remain explicit errors under TASK-007B's atomic default
policy. Spaces and Unicode punctuation remain neutral gap items.

## Queue behavior

Sign items transition `PENDING → PLAYING → COMPLETED`. When the current
sequence reaches its last stored timestamp, the viewer callback marks that
item completed, loads the next sign or presents the next neutral gap, and
continues playback. A gap is a timed presentation pause only; it has no
sample ID, geometry, sensor reading, or fabricated frame. Failed loads are
shown as `UNAVAILABLE` and are not silently substituted with another exemplar.

## `محمد` demonstration

The headless integration run used the real TASK-008 production run and
produced:

| queue index | character | SignID | sample ID | stored frames | geometry |
| ---: | :---: | :---: | --- | ---: | :---: |
| 0 | م | 0055 | `karsl_core28_s01_sign0055_train_rep010` | 20 | yes |
| 1 | ح | 0037 | `karsl_core28_s01_sign0037_train_rep026` | 20 | yes |
| 2 | م | 0055 | `karsl_core28_s01_sign0055_train_rep010` | 20 | yes |
| 3 | د | 0039 | `karsl_core28_s03_sign0039_test_rep001` | 21 | yes |

Completion indices were `(0, 1, 2, 3)` and the queue ended in the complete
state. The two `م` requests remain two distinct queue entries even though
canonical mode resolves both to the same deterministic sample.

## Canonical exemplar behavior

All 28 canonical entries resolve through the adapter and TASK-007A loader:

* 28/28 labels and descriptors load successfully;
* every loaded sequence has the descriptor's exact frame count;
* every sequence has valid timestamp/frame alignment and the expected 20-entry
  sensor layout;
* canonical selections are unchanged from the TASK-007B v1 catalog;
* the corrected catalog is versioned `task007b_core28_exemplars_v2`.

Signer-specific modes select a deterministic exemplar from the requested
signer. Random mode requires an explicit integer seed and is deterministic for
the same character/seed. Changing mode affects future queue entries; existing
items are not silently re-resolved.

## Geometry terminology reconciliation

TASK-007B's previous `mesh_available=false` field described the absence of a
standalone surface file, not the absence of 3D geometry. The integration
catalog and loader now expose three independent facts:

| field | production result | meaning |
| --- | ---: | --- |
| `embedded_mano_vertices_available` | 28/28 canonical; 4,222/4,222 run | stored MANO vertex cloud is present in `wilor_raw.npz` |
| `tracked_landmarks_3d_available` | 28/28 canonical | tracked 21-joint geometry is present |
| `surface_triangle_topology_available` | 0/28 canonical | no trusted triangle/index topology is serialized |

Canonical sample IDs and ranking order did not change after the terminology
correction. No triangle connectivity is fabricated.

## Evidence / Sources

Evidence comes from the committed TASK-008 manifests and the read-only
production run, plus the exact source heads identified in the integration
table. The integration tests use the same configured paths and do not generate
or modify production data.

## Embedded MANO vertex evidence

The read-only production run is
`/home/hatim/graduation-project-runs/task008-core28-full`, with committed
manifest `datasets/manifests/karsl_core28.csv`. The authoritative TASK-007A
geometry audit found:

* 4,222/4,222 samples contain both `vertices` and `vertices_keys`;
* each stored vertex row has shape `(778, 3)` and all 164,562 mesh rows are
  finite;
* 83,659 unique source-frame rows are represented;
* frame-prefix and raw-detection-suffix association mismatches are both zero.

The integration spot checks covered beginning, middle, end, multiple signers,
and a missing-hand sample. The loader associates a vertex row using
`frame_index:raw_detection_index`, then keeps physical `LEFT` and `RIGHT`
track identity from tracking metadata.

## Sensor rendering

Each physical track displays the frozen TASK-006 layout: 15 bend Hall-type
packages, 4 adjacent-spread Hall-type packages, and one palm IMU. The viewer
uses `H` for valid Hall/magnetic packages and `IMU` for a valid palm package.
It uses the stored `bend_valid`, `spread_valid`, and `palm_imu_valid` masks;
numeric zero remains a valid displayed value, while an unavailable channel is
shown as `H?`/`IMU?` and `INVALID (mask=0)` in the dashboard. No ADC or gyro
value is promoted to a new primary feature.

## Missing-hand semantics

The loader retains the fixed order `LEFT, RIGHT` and never reorders by screen
position, confidence, or which hand is visible. For
`karsl_core28_s01_sign0037_test_rep007`, the first frame has `LEFT=MISSING`
and a visible `RIGHT`; the UI leaves the left geometry absent and keeps the
right geometry/sensor readings. It does not mirror or copy the other hand.

## Performance

Only the active sequence is loaded. The production samples are short (the
headless `محمد` demonstration contains 20, 20, 20, and 21 stored frames), so
queue transitions do not require bulk memory. A real Tk smoke test created the
embedded Matplotlib/Tk viewer for a production sample, ran the event loop, and
closed cleanly without rendering exceptions. A headless Agg smoke render also
produced a PNG from a real mesh-cloud sample; the PNG is temporary and is not
tracked.

## Files Changed

* `visualizer/app/integration.py` — queue-to-loader adapter and headless
  playback session.
* `visualizer/app/main_window.py` — Tk/Matplotlib Core-28 application.
* `visualizer/catalog/` — v2 geometry terminology and regenerated compact
  catalog metadata; canonical sample selections are unchanged.
* `visualizer/loader.py` — explicit embedded-vertex, tracked-landmark, and
  surface-topology metadata.
* `visualizer/rendering/matplotlib_viewer.py` — embedded-canvas callbacks and
  queue-completion hooks.
* `scripts/run_task007c_visualizer_app.py` — GUI/headless entry point.
* `tests/test_task007c_visualizer_integration.py` — production integration and
  headless queue tests.
* `reports/visualizer/TASK-007B-keyboard-exemplar-queue.md` — corrected
  geometry wording.
* this report.

No production arrays, videos, checkpoints, screenshots, or generated ML
frames are included.

## Evaluation

Validation covers both control paths:

```text
Core-28 character → resolver → queue item → descriptor → loader → PlaybackSequence
PlaybackSequence → Matplotlib/Tk viewer → exact stored frame/timestamp
```

The production integration test loads all 28 canonical entries. The headless
test exercises repeated characters, neutral gaps, unsupported input, exemplar
modes, masks, and automatic completion. A real Tk event-loop smoke test and a
real-data headless Agg render were also executed.

## Tests

The added integration tests cover:

* all 28 canonical queue-to-loader resolutions;
* exact `محمد` ordering, repeated `م`, and automatic completion;
* neutral gaps without fabricated frames;
* unsupported-character reporting;
* canonical, signer, and seeded-random modes;
* precise embedded-geometry terminology;
* missing-hand masks and physical track preservation;
* descriptor/frame/timestamp compatibility and malformed items.

The full repository command passed 586 tests:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

The scoped compile command completed with zero errors:

```bash
python -m compileall -q evaluation tracking kinematics visualizer scripts tests
```

A real Tk event-loop smoke test and a headless Agg render both completed
successfully.

## Failures / Limitations

* The production run stores MANO vertex clouds, not triangle faces, so the
  visual is a dense 3D cloud plus tracked skeleton rather than a shaded
  surface.
* Sensor positions are landmark-derived visualization approximations; the
  generic thumb proximal/middle/distal names are not claims of exact clinical
  anatomy or CAD placement.
* Interactive playback requires a working Tk/Matplotlib display. The
  headless queue and PNG paths remain available for CI, SSH, and WSL checks.
* Playback remains isolated-sign playback with neutral gaps; it does not model
  continuous Arabic co-articulation.

## Comparison

This task makes no pose-model or recognizer comparison. It preserves the
frozen TASK-004 through TASK-006 data contracts and integrates the two
independent TASK-007 branches without selecting a new model or changing
stored results.

## Recommendation

KEEP — the integrated queue-to-renderer foundation is ready for later
recognizer integration, subject to the documented vertex-cloud and GUI
limitations.

## Future TASK-007B integration point

The planned TASK-007B boundary is now implemented: a resolver returns a
`PlaybackQueueItem.sequence_descriptor`, and the TASK-007C adapter passes that
descriptor's `sample_id` to `load_sequence`. The queue/catalog packages remain
independent of rendering, so future catalog policy changes can be reviewed
without moving queue semantics into the renderer.

## Future TASK-009B recognizer integration point

No recognizer or confidence UI is included. A future recognizer may consume
the same loaded `PlaybackSequence` or its exact TASK-009A-compatible source
sequence through a separate adapter and may report predictions alongside the
expected queue character. It must not change queue ordering, physical hand
identity, masks, timestamps, or stored artifacts. Neutral gaps must remain
outside recognition samples.

## How to Run

Headless end-to-end queue validation:

```bash
PYTHONPATH=. python scripts/run_task007c_visualizer_app.py \
  --run-root /home/hatim/graduation-project-runs/task008-core28-full \
  --manifest datasets/manifests/karsl_core28.csv \
  --text "محمد" \
  --headless
```

Interactive desktop application:

```bash
PYTHONPATH=. python scripts/run_task007c_visualizer_app.py \
  --run-root /home/hatim/graduation-project-runs/task008-core28-full \
  --manifest datasets/manifests/karsl_core28.csv \
  --text "محمد"
```

Omit `--text` to open an empty application and use the keyboard. Use
`--mode signer01`, `--mode signer02`, or `--mode signer03` for signer-specific
selection. Use `--mode random --seed <integer>` for reproducible random
selection. No `--mode` option means canonical mode.

## Reproducibility

The exact production input is external and read-only. The integration adds
`visualizer/app/`, `scripts/run_task007c_visualizer_app.py`, and
`tests/test_task007c_visualizer_integration.py`, plus the versioned catalog
terminology correction in `visualizer/catalog/`, the loader/UI metadata
correction, and this report. No KArSL videos, raw NPZs, model files,
generated frames, or GUI screenshots are committed.

## Results

The 28 canonical exemplars load through the complete catalog → descriptor →
TASK-007A loader path, and the real `محمد` headless queue completed all four
items in logical order. Embedded MANO vertices are available and rendered as a
cloud; tracked landmarks, sensor masks, and fixed physical LEFT/RIGHT
identity remain visible. The integration is ready for a later recognizer
adapter without involving recognition code now.

## Next Steps

Keep the integration PR unmerged for review, then allow the future TASK-009B
recognizer work to consume the same resolved sequence descriptor or loaded
`PlaybackSequence`. Recognition should be added through a separate adapter and
must not change queue ordering, geometry identity, masks, or stored artifacts.
