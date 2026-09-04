# Task

TASK-007F — CORE-28 Application V2: persistent dual-hand GPU 3D UI.

# Branch

`luna/task-007f-application-v2`, based exactly on `main` at
`a5dfda4b994227570fb2a53decfbdf12140a752b`, tagged `core28-v1.0`.

# Scope

This change replaces the primary TASK-007 Tk/Matplotlib presentation surface
with a native PySide6/QML/Qt Quick 3D application. It is presentation and
application integration work only. TASK-004 tracking, TASK-005 kinematics,
TASK-006 virtual-glove semantics, TASK-008 production artifacts, the
TASK-009A tensor contract, TASK-009B evaluation, and TASK-009C model files are
read-only inputs.

The old `run_task007d_visualizer_recognizer.py` entry point remains available
as a legacy/debug viewer. The new application does not import Tkinter or
Matplotlib and does not depend on Tk classes.

# Approach

The application owns one `PersistentRenderScene` for its entire lifetime. It
creates one LEFT and one RIGHT `HandMeshState`, one Qt Quick 3D geometry object
per track, one marker model per track, one QML `View3D`, one camera, and the
scene lights at startup. Queue changes and frame ticks upload new vertex and
normal bytes to those existing geometry objects. The static face/index buffer
is configured once.

QML provides the dark dashboard shell, two large hand presentations, the
recognition card, queue/text status, Arabic keyboard, and playback controls.
Qt Quick `FrameAnimation` drives the optional presentation-FPS diagnostic;
Python's timer advances the timestamp-authoritative source playback. Sequence
loading, checkpoint loading, and inference use `QRunnable` jobs and only
publish results back to the GUI thread.

The renderer has two explicit modes:

- Surface mode uses stored TASK-008 778-vertex rows, authoritative MANO faces,
  area-weighted vertex normals, depth testing, MSAA, directional/point lights,
  and a rough Principled/PBR glove material.
- If no local MANO topology is supplied, both persistent geometry objects stay
  visible as the existing 778-point presentation fallback and the UI shows
  `SURFACE TOPOLOGY UNAVAILABLE — POINT-CLOUD FALLBACK`.

Idle hands are presentation-only neutral templates. A missing/invalid stored
observation keeps the corresponding hand visible as a dimmed last-visible pose
or dimmed neutral pose. Smooth rendering interpolates only at display time;
the recognizer receives the original queue item through the existing
TASK-009A/TASK-009B adapter path and never receives a `PresentationFrame`.

# Evidence / Sources

- Frozen base: `core28-v1.0` / `a5dfda4b994227570fb2a53decfbdf12140a752b`.
- Existing renderer contract and sequence loader:
  `visualizer/contract.py`, `visualizer/loader.py`, and
  `visualizer/app/integration.py`.
- Existing mapping/queue contract: `visualizer/mapping/`,
  `visualizer/keyboard/`, and `visualizer/queue/`.
- Existing left/right MANO convention and local-only asset boundary:
  `reports/pose/wilor/TASK-002-wilor-karsl-pilot.md`.
- Qt custom geometry API: [QQuick3DGeometry](https://doc.qt.io/qt-6/qquick3dgeometry.html)
  and the [PySide6 binding](https://doc.qt.io/qtforpython-6/PySide6/QtQuick3D/QQuick3DGeometry.html).
- Qt scene quality controls: [SceneEnvironment](https://doc.qt.io/qt-6/qml-qtquick3d-sceneenvironment.html),
  [PrincipledMaterial](https://doc.qt.io/qt-6/qml-qtquick3d-principledmaterial.html),
  and [Material culling](https://doc.qt.io/qt-6/qml-qtquick3d-material.html).
- Render-synchronized diagnostics: [Qt Quick FrameAnimation](https://doc.qt.io/qt-6/qml-qtquick-frameanimation.html).
- MANO acquisition and license boundary: [official MANO project](https://mano.is.tue.mpg.de/).
  The established loader path is [official vchoutas/smplx](https://github.com/vchoutas/smplx).

# Files Changed

- `.gitignore` — ignores `/assets-local/` for user-provided licensed MANO
  files.
- `pyproject.toml` — adds the optional `gui` extra and QML package data.
- `README.md` — documents native PowerShell installation and launch paths.
- `smart_glove_app/` — primary Python controller, worker, playback,
  recognition bridge, MANO topology, presentation mesh state, Qt geometry,
  marker model, and QML application/components.
- `scripts/run_core28_application.py` — script wrapper for the same primary
  entry point.
- `tests/test_task007f_application_v2.py` — topology, surface/fallback,
  persistent-scene, interpolation isolation, keyboard, and Qt-buffer tests.
- `reports/visualizer/TASK-007F-application-v2.md` — this report.

No MANO pickle, checkpoint, dataset, video, or generated runtime output is
tracked.

# How to Run

From the repository root in native PowerShell, visualization-only mode needs
only the GUI extra:

```powershell
python -m pip install -e ".[gui]"
python -m smart_glove_app `
  --run-root "..\graduation-project-runs\task008-core28-full"
```

For the deployment checkpoint, install the existing recognition extra as well
(`torch` remains an external CUDA/runtime dependency in this repository):

```powershell
python -m pip install -e ".[gui,recognition]"
python -m smart_glove_app `
  --run-root "..\graduation-project-runs\task008-core28-full" `
  --checkpoint "..\graduation-project-runs\task009c-core28-deployment\deployment.pt" `
  --mano-model ".\assets-local\mano\MANO_RIGHT.pkl"
```

`--mano-model` is optional. Omit it to use the explicit point-cloud warning
mode. If a licensed asset is available, place it at
`assets-local/mano/MANO_RIGHT.pkl` (or pass any explicit local path). The
official MANO site requires the user to obtain and accept the asset terms;
the application does not download, redistribute, or reserialize it.

For a terminal queue smoke without opening Qt:

```powershell
python -m smart_glove_app --headless `
  --run-root "..\graduation-project-runs\task008-core28-full" `
  --text "محمد"
```

For native diagnostics and an automatic clean exit:

```powershell
python -m smart_glove_app `
  --run-root "..\graduation-project-runs\task008-core28-full" `
  --text "محمد" `
  --smoke-seconds 4 `
  --print-metrics
```

# Evaluation

The following were run in the provided native Windows Python 3.12.4
environment with PySide6 6.11.2. The real external TASK-008 run root and
TASK-009C checkpoint were used where noted. A licensed MANO file was not
present in the Windows workspace, so real-asset topology loading was tested
with a temporary synthetic 778/1538 pickle having the same file fields; that
temporary file was deleted and was never committed.

Focused TASK-007F tests:

```text
python -m unittest tests.test_task007f_application_v2 -v
Ran 13 tests — OK
```

Compilation:

```text
python -m compileall -q smart_glove_app scripts/run_core28_application.py
0 errors
```

Native GUI/QML smoke commands completed with exit code 0:

- point fallback, idle, Direct3D11, 3 seconds;
- point fallback with real TASK-008 playback of `محمد`, 4 seconds;
- surface-mode temporary 778/1538 MANO-shaped asset, 2 seconds;
- point fallback plus external deployment checkpoint, `محمد`, 6 seconds.

The offscreen QML smoke also loaded the QML tree, but Qt correctly reports
that Qt Quick 3D is not functional under a non-RHI offscreen renderer. It is
not used as evidence for 3D output; the native Direct3D11 smoke is the GUI
runtime evidence.

The existing repository discovery found 745 tests after adding the 13 new
TASK-007F tests (the pre-existing baseline is 732). The new tests all pass.
The full native-Windows discovery remains affected by pre-existing platform
assumptions outside this change: openpyxl/NPZ handles remain locked during
temporary-directory cleanup on Windows, and one legacy production test
compares the catalog's recorded POSIX `/home/hatim` run-root string against a
Windows-normalized path. Those failures were reproduced before/independently
of the new application and no frozen scientific module was changed to hide
them.

# Results

## Persistent scene and surface

- `PersistentRenderScene.scene_creation_count == 1` and
  `view3d_creation_count == 1`.
- LEFT and RIGHT `QtHandGeometry` objects each report
  `geometry_creation_count == 1` across frame uploads.
- Qt buffer tests confirm stable object identity and stable index-buffer size;
  only the interleaved position/normal vertex data is updated.
- Surface mode reports `MANO SURFACE · 778 VERTICES`, configures 1538 indexed
  triangles, computes normals, and renders through two persistent Qt Quick 3D
  `Model` items.
- No-topology mode reports the exact required warning and retains two visible
  778-point idle hands.

## MANO topology and winding

`mano_topology.py` accepts an explicit `MANO_RIGHT.pkl` or `MANO_LEFT.pkl`,
tries `smplx.create(..., model_type="mano")` when the optional package is
available, and otherwise extracts only known `f`/`faces`/`face_indices` and
`v_template` fields from the local pickle. It rejects a non-triangle array,
wrong 1538-face count, non-integral indices, degenerate triangles, out-of-range
indices, and incomplete 0..777 vertex coverage.

The canonical source is right-handed. A left-source asset is canonicalized by
reversing each triangle after its x-reflection; the displayed LEFT track then
uses x-reflected vertices and reversed winding again. Thus both displayed
tracks preserve front-face orientation under back-face culling. The source
file SHA-256 and canonical topology SHA-256 are available from
`ManoTopology.summary()`; no licensed source hash can be reported for this
workspace because no MANO file was supplied.

## Persistence, idle, missing, and smooth rendering

Both tracks are visible from startup. A missing observation never removes a
viewport: it uses a dimmed last-visible pose or neutral presentation template,
while the scientific frame/mask remains untouched. Display interpolation is
kept in `PresentationFrame` and only blends adjacent valid 778-vertex rows;
source arrays, masks, timestamps, and recognizer inputs remain unchanged.

## Keyboard, queue, and text

The UI exposes three large RTL rows backed by the existing `Core28Mapping` and
`Core28Keyboard`; no second label mapping was created. Each click calls
`enqueueCharacter()` once. Repeated letters remain distinct queue items.
The logical string `محمد` is preserved in Unicode order and resolves to four
events:

| queue index | character | SignID |
| ---: | --- | ---: |
| 0 | م | 0055 |
| 1 | ح | 0037 |
| 2 | م | 0055 |
| 3 | د | 0039 |

Spaces are explicit neutral-gap queue items. No NLP, autocorrection, or
manual Unicode reversal was added.

## Recognition

Without `--checkpoint`, the UI remains fully usable and reports
`Recognition disabled`. With the supplied deployment checkpoint
(`e3df0f007c542d15a6ff4a7ad090d6a8af58b583357ca1905c4cdcb20c82ad1e`), the
headless and native paths loaded the existing recognition adapter and reported
`DEPLOYMENT MODEL`, `All Core-28 signers / 4,222 training sequences`, and
`LOSO reference only (not deployment accuracy): 67.63% accuracy / 0.6607 macro F1`.
The native GUI smoke reached `Prediction ready`; no `97.70%` deployment
accuracy label is used.

# Failures / Limitations

- The Windows workspace does not contain a user-supplied licensed MANO asset.
  The application is ready for the asset, and surface mode was verified with
  the validated temporary topology path, but the exact licensed MANO surface
  was not visually inspected on this machine.
- The default no-asset presentation is the required 778-point fallback, not a
  shaded mesh. This is clearly labeled in the header/diagnostics.
- The optional diagnostics do not measure process CPU, GPU utilization, or
  VRAM; those require a profiler or vendor tool and were intentionally not
  added to the application.
- Qt Quick 3D requires a native RHI/3D backend. Offscreen software smoke can
  validate QML construction but cannot display the 3D scene.
- The legacy Tk/Matplotlib viewer remains unchanged and is explicitly debug /
  historical infrastructure; its known Tcl lifecycle issue is not used by
  the new app.
- TASK-010A NLP and TASK-011A TTS are intentionally not started.

# Performance

The native window reported `Direct3D11` through Qt Quick RHI and a measured
Qt Quick `FrameAnimation` presentation rate of approximately 180 FPS on the
provided desktop. This was observed both in point-fallback and temporary
surface-mode smoke runs. Real TASK-008 source sequences are timestamped at
approximately 30 source frames per second; the display clock preserves those
timestamps and optionally interpolates between them.

The 4-second native `محمد` playback smoke produced 162 LEFT and 162 RIGHT
geometry uploads with one creation per track. The 2-second surface smoke
produced 31 uploads per track with one creation per track. These are engineering
observations, not a benchmark claim; CPU utilization, GPU utilization, and
peak VRAM were not measured.

# Comparison

The legacy Tk/Matplotlib viewer repeatedly created and destroyed its visual
surface for queue items and is therefore retained only as a reproducibility
surface. The V2 application has one long-lived QML `View3D`, persistent GPU
geometry/index resources, a real indexed surface when topology is supplied,
and modern controls around the hand hero scene. The scientific/data/model
pipeline is reused rather than reimplemented.

# Recommendation

KEEP — the V2 application foundation satisfies the persistent-scene,
dual-track, queue, fallback, topology-validation, native Windows, and
recognition-provenance requirements. Supply the locally licensed MANO asset on
the graduation machine before presenting the final shaded surface mode.

# Reproducibility

Environment:

- OS/runtime target: native Windows, Python 3.12.4, CPython 64-bit.
- PySide6 / PySide6-Essentials / PySide6-Addons: 6.11.2.
- GPU/API observed: NVIDIA RTX 2060 Super target, Qt Quick RHI Direct3D11.
- Scientific run root: external TASK-008 `task008-core28-full`.
- Deployment checkpoint: external `deployment.pt`, SHA-256
  `e3df0f007c542d15a6ff4a7ad090d6a8af58b583357ca1905c4cdcb20c82ad1e`.
- MANO asset: intentionally not present or committed; synthetic topology
  smoke used a deterministic NumPy seed and a temporary pickle only.
- Queue text: logical Unicode `محمد`; no random exemplar mode was used.

Commands used are listed above. No task extraction, retraining, tensor
contract change, checkpoint modification, or raw-data overwrite was performed.

# Next Steps

1. On the presentation machine, obtain the licensed MANO package from the
   official source, place `MANO_RIGHT.pkl` under ignored `assets-local/mano/`,
   run the surface-mode command, and record the returned source/topology hashes.
2. Keep the persistent scene as the integration point for future
   TASK-010A text/NLP queue producers.
3. Add TASK-011A TTS as a separate consumer of recognized/queued logical text;
   do not route either future feature through the legacy viewer.
