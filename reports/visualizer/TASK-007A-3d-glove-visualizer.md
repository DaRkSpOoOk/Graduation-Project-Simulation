# Task

TASK-007A — 3D MANO / virtual smart-glove visualizer foundation.

# Branch

`luna/task-007a-3d-glove-visualizer`, based on `19972cc77ed7345b599c5881d010386c8191bea0`.

# Scope

This branch adds a renderer-facing loader, exact-frame playback controller, 3D geometry/sensor-marker helpers, and an optional local viewer for one finalized TASK-008 sequence at a time. It does not modify WiLoR, tracking, kinematics, virtual-glove extraction, recognition, keyboard/catalog/queue code, or production artifacts.

The viewer is deliberately a playback/debug surface. It is not a classifier, recognizer, data generator, or replacement for the TASK-006 sensor contract.

# Approach

The implementation is split into four small layers:

1. `visualizer.loader.load_sequence()` validates and loads the four stored stages for one `sample_id`.
2. `visualizer.contract` exposes immutable renderer-facing `PlaybackSequence`, `FrameData`, `HandGeometry`, and explicit sensor-validity objects.
3. `visualizer.geometry` maps the frozen sensor layout to 3D landmark-derived marker locations.
4. `visualizer.playback.PlaybackController` and `visualizer.rendering.MatplotlibGloveViewer` provide timestamp-aware playback and a local technical dashboard.

The renderer consumes physical track identity from TASK-004/TASK-008 `raw_detection_index` and tracked `LEFT`/`RIGHT` arrays. It never infers identity from screen position.

# Evidence / Sources

The production audit used the external finalized run at:

`/home/hatim/graduation-project-runs/task008-core28-full`

and the committed manifest:

`datasets/manifests/karsl_core28.csv`

The inspected artifacts were `pose/raw/<sample>/wilor_raw.npz`, `tracking/<sample>/wilor_tracked.npz` and metadata, `kinematics/<sample>/hand_kinematics.npz`, and `virtual_glove/<sample>/virtual_glove.npz`, metadata, and `sensor_layout.json`.

# Production geometry audit

The manifest contains 4,222 samples. A full raw-pose scan found:

| Check | Result |
|---|---:|
| Samples with both `vertices` and `vertices_keys` | 4,222 / 4,222 |
| Mesh row shape | `(778, 3)` for all 4,222 samples |
| Mesh rows | 164,562 |
| Finite mesh rows | 164,562 / 164,562 |
| Unique decoded frame rows represented by raw mesh keys | 83,659 |
| Key frame-prefix mismatches | 0 |
| Per-frame raw-index suffix mismatches | 0 |

`vertices_keys` uses the deterministic format `frame_index:raw_detection_index`. Within each frame, the raw detection suffixes were exactly `0..N-1`; this is the association used by the loader to connect a stored MANO vertex cloud to a tracked hand. The full raw scan had 2,775 one-detection frames, 80,865 two-detection frames, and 19 three-detection frames. The extra-detection cases remain keyed rather than being silently truncated.

Deterministic samples inspected from the beginning, middle, end, and a missing-hand case were:

| Sample | Signer / label | Frames | Raw mesh rows | Result |
|---|---|---:|---:|---|
| `karsl_core28_s01_sign0032_test_rep001` | S01 / `ا` | 29 | 58 | mesh and two raw detections per frame |
| `karsl_core28_s02_sign0043_train_rep023` | S02 / `س` | 13 | 26 | mesh and two raw detections per frame |
| `karsl_core28_s03_sign0059_train_rep052` | S03 / `ي` | 24 | 48 | mesh and two raw detections per frame |
| `karsl_core28_s01_sign0037_test_rep007` | S01 / `ح` | 18 | 18 | one visible hand; the other is missing in the first frame |

Raw MANO metadata was present in these samples. `mano_params_json` included `betas`, `global_orient_rotmat`, `hand_pose_rotmat`, and representation metadata; `mano_references_json` included camera/box references, image size, focal length, and `vertices_ref`. Raw landmarks had shape `(N, 21, 3)`, while tracked landmarks had shape `(F, 2, 21, 3)`. Tracking, kinematics, and virtual-glove `frame_index`/timestamp arrays were aligned for the inspected samples and are checked by the loader.

The run has 4,222 sample directories in each of `pose/raw`, `tracking`, `kinematics`, and `virtual_glove`. The raw mesh row count is larger than the sample count because it is a detection-level table rather than one row per video frame.

# Chosen rendering stack

The viewer uses Matplotlib 3.11.1 with the Python/Tk interactive stack. Matplotlib is imported lazily, so the non-GUI loader and tests do not need a display server. Tk 8.6.17 was available in the development environment. A headless `--save-png --no-show` path is provided for CI, SSH, and WSL validation. The optional package dependency is declared as `.[visualizer]`.

This keeps the foundation local and Python-native without introducing a web runtime or a large GUI framework. Matplotlib is suitable for these short sequences; each requested sample is loaded on demand rather than loading the 4,222-sequence run.

# Architecture

The renderer-facing flow is:

```text
run_root + sample_id
        |
        v
load_sequence()
        |
        v
PlaybackSequence
   |       |
   |       +--> PlaybackController (play/pause/restart/scrub/speed)
   +----------> MatplotlibGloveViewer (3D geometry + sensor dashboard)
```

`PlaybackSequence` has no Arabic keyboard or recognizer dependency. A future queue can resolve a character to a `sample_id` and call `load_sequence()`; a future recognizer can consume the same loaded sequence through a separate integration layer.

# Mesh / joint source

When available, the viewer renders the stored 778-point MANO vertex cloud associated by `vertices_keys` and the tracked 21-joint skeleton. The production run stores vertices but does not store MANO triangle topology/faces, so the viewer intentionally renders a translucent dense point cloud rather than fabricating a surface mesh. The tracked 21-joint chains are drawn on top for readable finger structure and sensor placement.

If a future valid sequence has no stored mesh arrays, the loader falls back to `tracked_landmarks_3d` and exposes `geometry_source=tracked_landmarks_3d`. It never reconstructs or invents a mesh during playback.

# Sensor placement

`sensor_layout.json` is validated against the frozen `ideal_virtual_glove_v1` contract: 15 bend entries, 4 spread entries, one palm IMU, canonical array indices, canonical IDs/order, and markers `H`/`IMU`.

Placement is derived from 21-joint geometry:

* Bend markers use the corresponding proximal/middle/distal chain joint. These names preserve the TASK-006 generic contract; thumb locations are not claimed to be exact CMC/MCP/IP anatomy.
* Spread markers use 70% of the midpoint between adjacent finger-base landmarks and 30% of the palm centre as an interdigital-web approximation.
* The palm IMU uses the mean of wrist, index-MCP, middle-MCP, and pinky-MCP landmarks as a dorsal-palm-centre approximation.

Every logical layout entry is returned by the placement helper. The viewer displays `H` for valid Hall/magnetic packages and `IMU` for the palm orientation package. Invalid channels use a grey `H?`/`IMU?` marker when geometry exists and `INVALID (mask=0)` in the dashboard.

# Playback contract

Playback uses the stored `frame_index` and `timestamp_seconds` arrays. The loader rejects empty, non-finite, non-increasing frame/timestamp sequences and checks raw, tracking, kinematics, and virtual-glove alignment. Scrubbing selects an exact stored frame; no resampling, interpolation, forward filling, or smoothing is performed.

The controller supports play, pause, restart, exact frame seeking, and 0.5×/1×/2× speed. A timer advances according to stored timestamp differences and stops at the final stored frame. A valid numeric zero is retained as `value=0.0`; an invalid channel is represented separately as `valid=False` and `value=None`.

# Validity semantics

Hand geometry is rendered only when the tracked row has a usable source detection index and finite 21-joint coordinates. `MISSING`, `LIKELY_OCCLUDED`, and `REJECTED_QUALITY` states produce no invented geometry. The other hand remains independently renderable. `AMBIGUOUS` state is retained and shown as metadata when a source geometry is available.

Sensor readings use the stored TASK-006 masks, not numeric zero tests. Invalid hand/sensor values are never copied from the other hand or replaced with zeros. The dashboard reports physical track, state, geometry presence, sensor identity, value, and mask status.

# Performance

The headless smoke render loaded only the 13-frame sample `karsl_core28_s02_sign0043_train_rep023`, including stored mesh clouds, and saved a PNG in 1.44 seconds on the development environment (maximum resident set size approximately 119 MB for that process). This is a startup/render measurement, not an inference benchmark. The visualizer does not load all production sequences into memory.

# Evaluation

Automated tests cover:

* exact layout counts and marker counts;
* canonical mesh-key/frame/raw-detection association;
* LEFT/RIGHT preservation independent of screen position;
* valid zero versus invalid-mask behavior;
* marker coverage and landmark-derived placement;
* no geometry for a missing hand;
* frame/timestamp synchronization;
* finite sequence bounds;
* timestamp-aware scrub, restart, and speed behavior;
* malformed layout and missing-artifact rejection;
* landmark-only fallback;
* real production artifact loading.

The test suite completed with 562 tests passing. `compileall` completed with zero errors. Two headless real-data smoke renders were manually inspected: a dual-hand mesh sample and a sample whose first frame has a missing LEFT hand. The latter showed the surviving RIGHT geometry and explicitly marked the LEFT hand as missing without drawing replacement geometry.

# Results

The foundation successfully loads real TASK-008 output, associates stored MANO vertices to physical tracked hands, renders both hands when present, exposes all 20 logical sensor packages per hand, and synchronizes the dashboard with exact source frames/timestamps. The smoke sample rendered `geometry=stored_mano_vertices+tracked_landmarks_3d` and label `س`.

# Files Changed

* `visualizer/contract.py` — renderer contract, layout validation, and masked readings.
* `visualizer/geometry.py` — 21-joint skeleton and landmark-derived marker placement.
* `visualizer/loader.py` — single-sequence artifact/provenance validation and mesh association.
* `visualizer/playback.py` — timestamp-aware playback state machine.
* `visualizer/rendering/matplotlib_viewer.py` — optional 3D/dashboard viewer.
* `visualizer/__init__.py`, `visualizer/rendering/__init__.py` — package exports.
* `scripts/run_task007a_visualizer.py` — command-line entry point.
* `tests/test_task007a_visualizer.py` — synthetic and real-artifact tests.
* `pyproject.toml` — package discovery and optional `visualizer` dependency.
* `reports/visualizer/TASK-007A-3d-glove-visualizer.md` — this report.

No production run, source video, model, mesh file, or generated image is committed.

# How to Run

Install the optional renderer dependency in the project environment:

```bash
pip install -e '.[visualizer]'
```

Headless validation and PNG export:

```bash
PYTHONPATH=. python scripts/run_task007a_visualizer.py \
  --run-root /home/hatim/graduation-project-runs/task008-core28-full \
  --manifest datasets/manifests/karsl_core28.csv \
  --sample-id karsl_core28_s02_sign0043_train_rep023 \
  --save-png /tmp/task007a-smoke.png \
  --no-show
```

Interactive local playback uses the same command without `--no-show` (and may omit `--save-png`). An exact stored frame can be selected with, for example, `--frame-index 5`; the CLI rejects a frame index not present in the sequence.

Run the tests and compile check with:

```bash
python -m unittest discover -s tests -p 'test_*.py'
python -m compileall -q visualizer evaluation tracking kinematics scripts tests
```

# Comparison

This task performs no pose-model comparison and makes no recognition decision. It preserves the finalized WiLoR/TASK-004/TASK-005/TASK-006 outputs as the source of truth. MediaPipe and any future alternative remain outside this renderer foundation.

# Failures / Limitations

* The production serializer provides MANO vertices but no triangle topology, so the current 3D representation is a dense vertex cloud plus a 21-joint skeleton rather than a shaded surface mesh.
* The sensor locations are practical landmark-derived approximations, especially for interdigital web locations and the thumb; they are not claims of physical CAD placement or clinical anatomy.
* Interactive display requires a working Matplotlib GUI backend and desktop/display server. The headless PNG route remains available for remote environments.
* Mesh rendering is limited to the currently loaded sample; there is no bulk sequence browser or cache policy in this foundation.
* This branch does not provide the Arabic keyboard/catalog/queue, exemplar selection, recognition inference, or any new extraction.

# Performance / dependency record

Development smoke environment:

* Python `3.14.4`
* NumPy `2.5.2`
* Matplotlib `3.11.1`
* Tk `8.6.17`

The new GUI dependency is optional; core artifact loading and tests remain independent of Matplotlib imports.

# Reproducibility

Use the exact branch commit reported with this PR and the external run root above. The sample IDs and manifest row are deterministic. The loader records the run root, sample ID, manifest record, track order, geometry source, and mesh key format in `PlaybackSequence.metadata`. Production artifacts are read-only inputs and remain outside Git.

Future TASK-007B integration should resolve an Arabic character to an existing `sample_id`, then call `load_sequence(run_root, sample_id, manifest_path=...)`. Future TASK-009B integration can consume the same sequence through an explicit adapter; no recognizer or keyboard dependency is present here.

# Recommendation

KEEP

The foundation is ready for queue integration, subject to the documented GUI/display and vertex-topology limitations.

# Next Steps

TASK-007B may connect its catalog/queue/exemplar resolver to the `sample_id` playback contract. A later visualization enhancement may add MANO faces if a trusted topology asset is explicitly provided. Recognition integration remains a separate task and must not alter this renderer’s raw/masked playback semantics.
