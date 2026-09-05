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

The normal user-facing presentation loads the locally supplied, rigged
`application_hands.glb` through one persistent Qt Quick 3D `RuntimeLoader`.
The profile indexes the exported skeleton by structural runtime paths because
Qt 6.11 does not expose the glTF node names through `RuntimeLoader`. Existing
bone nodes receive local quaternion deltas; the loader, armatures, meshes, and
materials remain alive for the application lifetime.

The renderer has three explicit presentation paths:

- Rigged-GLB mode is the normal application path: skinned triangle hand meshes,
  persistent left/right skeletons, depth testing, MSAA, directional/point
  lights, and the graphite Principled material.
- MANO surface diagnostics use stored TASK-008 778-vertex rows,
  authoritative MANO faces, area-weighted vertex normals, and persistent
  custom geometry when a locally licensed MANO topology is supplied.
- The old 778-point representation is available only with the explicit
  `--debug-mano-points` diagnostics flag. It is not the normal fallback.

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
- Rigged asset loading: [Qt Quick 3D RuntimeLoader](https://doc.qt.io/qt-6/qml-qtquick3d-assetutils-runtimeloader.html)
  and the [Qt Quick 3D skinning example](https://doc.qt.io/qt-6/qtquick3d-skinning-example.html).
- MANO acquisition and license boundary: [official MANO project](https://mano.is.tue.mpg.de/).
  The established loader path is [official vchoutas/smplx](https://github.com/vchoutas/smplx).
- Source hand asset: [BlendSwap Hands Rigged](https://www.blendswap.com/blend/22269),
  creator SparrowHawk, marked CC0 on the source page.

# Files Changed

- `.gitignore` — ignores `/assets-local/` for user-provided licensed MANO
  files.
- `pyproject.toml` — adds the optional `gui` extra and QML package data.
- `README.md` — documents native PowerShell installation and launch paths.
- `smart_glove_app/` — primary Python controller, worker, playback,
  recognition bridge, MANO topology, presentation mesh state, Qt geometry,
  marker model, rig profile/retargeter, and QML application/components.
- `smart_glove_app/assets/rig_profiles/blendswap_hands_v1.json` — project-owned
  calibrated channel map and Qt RuntimeLoader node paths.
- `scripts/run_core28_application.py` — script wrapper for the same primary
  entry point.
- `tests/test_task007f_application_v2.py` — topology, surface/fallback,
  persistent-scene, interpolation isolation, keyboard, and Qt-buffer tests.
- `reports/visualizer/TASK-007F-application-v2.md` — this report.

No MANO pickle, checkpoint, dataset, video, Blender working copy, or generated
runtime output is tracked. The local working asset is intentionally under the
ignored `assets-local/` boundary.

# How to Run

From the repository root in native PowerShell, visualization-only mode needs
only the GUI extra:

```powershell
python -m pip install -e ".[gui]"
python -m smart_glove_app `
  --run-root "..\graduation-project-runs\task008-core28-full"
```

With the prepared rigged asset (the normal application path):

```powershell
python -m smart_glove_app `
  --run-root "..\graduation-project-runs\task008-core28-full" `
  --rig-asset ".\assets-local\blendswap_hands_v1\application_hands.glb" `
  --rig-profile ".\smart_glove_app\assets\rig_profiles\blendswap_hands_v1.json" `
  --text "محمد"
```

For the deployment checkpoint, install the existing recognition extra as well
(`torch` remains an external CUDA/runtime dependency in this repository):

```powershell
python -m pip install -e ".[gui,recognition]"
python -m smart_glove_app `
  --run-root "..\graduation-project-runs\task008-core28-full" `
  --checkpoint "..\graduation-project-runs\task009c-core28-deployment\deployment.pt" `
  --rig-asset ".\assets-local\blendswap_hands_v1\application_hands.glb" `
  --rig-profile ".\smart_glove_app\assets\rig_profiles\blendswap_hands_v1.json" `
  --mano-model ".\assets-local\mano\MANO_RIGHT.pkl"
```

`--mano-model` is optional and affects only the MANO surface-diagnostics path.
The rigged GLB is the normal user-facing path. If a licensed MANO asset is
available, place it at
`assets-local/mano/MANO_RIGHT.pkl` (or pass any explicit local path). The
official MANO site requires the user to obtain and accept the asset terms;
the application does not download, redistribute, or reserialize it.

To intentionally inspect the old point representation:

```powershell
python -m smart_glove_app `
  --run-root "..\graduation-project-runs\task008-core28-full" `
  --debug-mano-points
```

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
environment with PySide6 6.11.2 and Blender 4.4.0. The real external TASK-008
run root and TASK-009C checkpoint were used where noted. A licensed MANO file
was not present in the Windows workspace, so MANO topology loading was tested
with a temporary synthetic 778/1538 pickle having the same file fields; that
temporary file was deleted and was never committed. The prepared local
BlendSwap working copy and exported GLB were tested directly.

Focused TASK-007F tests:

```text
python -m unittest tests.test_task007f_application_v2 -v
Ran 17 tests — OK
```

Compilation:

```text
python -m compileall -q smart_glove_app scripts/run_core28_application.py
0 errors
```

Native GUI/QML smoke commands completed with exit code 0:

- rigged GLB idle/startup, Direct3D11, persistent skeleton loaded;
- rigged GLB with real TASK-008 playback of `محمد`, 2 seconds;
- explicit point diagnostics with a missing GLB, 1 second;
- MANO surface-mode temporary 778/1538 topology, 2 seconds;
- rigged GLB plus the external deployment checkpoint, 8 seconds;
- headless real-sequence deployment inference for `م`.

The offscreen QML smoke also loaded the QML tree, but Qt correctly reports
that Qt Quick 3D is not functional under a non-RHI offscreen renderer. It is
not used as evidence for 3D output; the native Direct3D11 smoke is the GUI
runtime evidence.

The existing repository discovery found 749 tests after adding the 17 new
TASK-007F tests (the pre-existing baseline is 732). The new tests all pass.
The full native-Windows discovery completed with 1 failure, 18 errors, and 29
skips. The failures are outside this change: one legacy CLI test needs UTF-8
console configuration when launched under the default CP1252 console, one
legacy renderer test expects an unavailable external production sample, and
the official workbook/NPZ tests leave Windows file handles open during
temporary-directory cleanup. Running with `PYTHONUTF8=1` removes the console
encoding failure but leaves the external-sample failure and handle-cleanup
errors. No frozen scientific module was changed to hide these baseline issues.

# Results

## Persistent scene and surface

- `PersistentRenderScene.scene_creation_count == 1` and
  `view3d_creation_count == 1`.
- LEFT and RIGHT `QtHandGeometry` objects each report
  `geometry_creation_count == 1` across frame uploads.
- Qt buffer tests confirm stable object identity and stable index-buffer size;
  only the interleaved position/normal vertex data is updated.
- The prepared GLB contains two skinned triangle meshes, two 22-joint skins,
  the two presentation roots, and one graphite material. Each source mesh has
  1102 base vertices; the GLB exporter triangulates the 1090 source polygons
  to 2178 triangles. It contains no camera, light, or animation.
- MANO surface diagnostics report `MANO SURFACE · 778 VERTICES`, configure
  1538 indexed triangles, compute normals, and render through two persistent
  Qt Quick 3D `Model` items when topology is supplied.
- With no rigged asset, the point representation is hidden by default. It is
  shown only when `--debug-mano-points` is explicitly selected, with the clear
  `SURFACE TOPOLOGY UNAVAILABLE — POINT-CLOUD FALLBACK` diagnostics message.

## Blender working asset and direct retargeting

The original downloaded BlendSwap file was not modified. The protected working
copy is:

`assets-local/blendswap_hands_v1/blendswap_hands_v1_working.blend`

The application export is:

`assets-local/blendswap_hands_v1/application_hands.glb`

The source page identifies the asset as *Hands Rigged*, creator SparrowHawk,
and marks it CC0: [BlendSwap source page](https://www.blendswap.com/blend/22269).
The working copy and export remain local and ignored; the project commits only
the project-owned profile. The profile records this attribution boundary and
the Qt runtime node paths.

Both meshes use the shared `TASK007F_Graphite_Glove` Principled material:
dark graphite base `(0.018, 0.028, 0.045)`, roughness `0.72`, metallic `0.04`,
IOR `1.45`, specular IOR level `0.32`, and a restrained clear-coat value
`0.08`. The material is embedded once in the GLB and survives the Blender
round-trip import check.

The scene contains `LEFT_PRESENTATION_ROOT` and `RIGHT_PRESENTATION_ROOT`.
Their saved display transforms are respectively `(-3.07, -0.09, 0)` and
`(3.95, -0.50, 0)`, identity rotation, and uniform scale `1.8`. The armatures
remain children of those roots, so root transforms control placement while
`Bone.016` receives recorded palm orientation. The final world bounds have a
horizontal gap of approximately `2.0446` Blender units; the hands do not touch
and are not staged as a praying pose.

The source armatures remain editable. Each has 22 deform bones, its original
12 `COPY_ROTATION` and 5 `IK` constraints, and all 17 are muted for direct
runtime mode rather than deleted. The original mesh vertex groups and
armature modifiers remain intact. Blender controlled probes were restored to
neutral after each test. Rest-pose sweeps over `0..120°` for every phalange and
`0..60°` for every spread base established the signs below: every one of the
15 phalange channels and all 4 base spread drivers produced a finite,
monotonic deformation on both hands.

The final calibrated project mapping is:

| PROJECT_CHANNEL | LEFT_BONE | RIGHT_BONE | ROTATION_AXIS / SIGN | NOTES |
| --- | --- | --- | --- | --- |
| `thumb[0]` | `Bone.017` | `Bone.017` | `Z +` / `Z -` | thumb proximal |
| `thumb[1]` | `Bone.018` | `Bone.018` | `X -` / `X -` | thumb middle |
| `thumb[2]` | `Bone.019` | `Bone.019` | `X -` / `X -` | thumb distal |
| `index[0]` | `Bone` | `Bone` | `Z +` / `Z -` | index proximal |
| `index[1]` | `Bone.001` | `Bone.001` | `X -` / `X -` | index middle |
| `index[2]` | `Bone.002` | `Bone.002` | `Z +` / `Z -` | index distal |
| `middle[0]` | `Bone.003` | `Bone.003` | `X -` / `X -` | middle proximal |
| `middle[1]` | `Bone.004` | `Bone.004` | `X -` / `X -` | middle middle |
| `middle[2]` | `Bone.005` | `Bone.005` | `X -` / `X -` | middle distal |
| `ring[0]` | `Bone.006` | `Bone.006` | `Z -` / `Z +` | ring proximal |
| `ring[1]` | `Bone.007` | `Bone.007` | `X -` / `X -` | ring middle |
| `ring[2]` | `Bone.008` | `Bone.008` | `X -` / `X -` | ring distal |
| `pinky[0]` | `Bone.009` | `Bone.009` | `Z -` / `Z +` | pinky proximal |
| `pinky[1]` | `Bone.010` | `Bone.010` | `X -` / `X -` | pinky middle |
| `pinky[2]` | `Bone.011` | `Bone.011` | `Z -` / `Z +` | pinky distal |
| `thumb-index` | `Bone.021` (ref `Bone.012`) | `Bone.021` (ref `Bone.012`) | `Z -` / `Z +` | thumb base spread |
| `index-middle` | `Bone.013` (ref `Bone.012`) | `Bone.013` (ref `Bone.012`) | `Z +` / `Z -` | index base spread |
| `middle-ring` | `Bone.014` (ref `Bone.013`) | `Bone.014` (ref `Bone.013`) | `Z +` / `Z -` | middle base spread |
| `ring-pinky` | `Bone.015` (ref `Bone.014`) | `Bone.015` (ref `Bone.014`) | `Z +` / `Z -` | ring base spread |

All rotations are local pose-space deltas after rest, with neutral offset `0°`,
bend clamp `0..120°`, and spread clamp `0..60°`. `Bone.020` is the deform root
and `Bone.016` is the palm/wrist bridge. The palm WXYZ input is converted to a
delta relative to the first valid sequence frame, then applied to `Bone.016`.
The direct-deform strategy was selected because the four authored controller
bones cannot provide three independent bends per finger. Presentation roots
are intentionally separate from this recorded palm orientation.

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
For rigged playback, `HandRigRetargeter` applies the same policy to local bone
quaternions: invalid bends/spreads hold their last presentation transform and
invalid palm data holds its last valid palm delta. Adjacent poses use
quaternion SLERP, never Euler-angle interpolation. The retargeter output is a
separate `HandRigPose` and is never passed to TASK-009A or the recognizer.

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
The native GLB smoke reached `Recognition ready`; the real headless TASK-008
sequence for `م` predicted `م` with confidence `0.9995227`. No `97.70%`
deployment accuracy label is used.

# Failures / Limitations

- The Windows workspace does not contain a user-supplied licensed MANO asset.
  The rigged GLB path is fully exercised; MANO surface diagnostics were
  verified with the validated temporary topology path, but the exact licensed
  MANO surface was not visually inspected on this machine.
- The local GLB is ignored because the source asset is kept outside the Git
  artifact boundary. A clean checkout must receive the working GLB and use the
  supplied rig profile; without it the UI clearly reports the missing asset.
  The 778-point representation is diagnostics-only (`--debug-mano-points`).
- The exported hand mesh is the CC0 donor rig surface, not a regenerated MANO
  surface. TASK-008 vertices remain the authoritative source for MANO surface
  diagnostics; the retargeted GLB is a presentation asset driven by frozen
  TASK-005 channels.
- Qt glTF export is limited to four normalized influences per vertex because
  Qt Quick 3D skinning accepts the top four influences. The exporter was
  configured accordingly and the Blender source retains its original groups.
- The RuntimeLoader profile stores structural child paths for this exact GLB;
  if the export hierarchy changes, the profile must be recalibrated rather
  than relying on guessed node names.
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
Qt Quick `FrameAnimation` presentation rate of approximately `180..181 FPS` on
the provided desktop. The real TASK-008 `محمد` rigged-GLB smoke reported
`active_sequence_fps = 30.0`, 113 LEFT and 113 RIGHT persistent geometry
uploads, 108 persistent rig-pose updates, and one geometry creation per track. The
deployment GLB smoke also reported one creation per track and approximately
180 FPS. These are engineering observations, not a benchmark claim; process
CPU utilization, GPU utilization, and peak VRAM were not measured.

# Comparison

The legacy Tk/Matplotlib viewer repeatedly created and destroyed its visual
surface for queue items and is therefore retained only as a reproducibility
surface. The V2 application has one long-lived QML `View3D`, persistent GPU
geometry/index resources, a real indexed surface when topology is supplied,
and modern controls around the hand hero scene. The scientific/data/model
pipeline is reused rather than reimplemented.

# Recommendation

KEEP — the V2 application foundation satisfies the persistent-scene,
dual-track, rigged surface, queue, topology-validation, native Windows, and
recognition-provenance requirements. Supply the locally licensed MANO asset
only when the MANO diagnostic surface is needed.

# Reproducibility

Environment:

- OS/runtime target: native Windows, Python 3.12.4, CPython 64-bit.
- PySide6 / PySide6-Essentials / PySide6-Addons: 6.11.2.
- GPU/API observed: NVIDIA RTX 2060 Super target, Qt Quick RHI Direct3D11.
- Blender asset preparation: Blender 4.4.0; working-copy and GLB paths are
  under ignored `assets-local/blendswap_hands_v1/`.
- Current local working-copy SHA-256:
  `9e6a4c191054dec1b59c0a153bf7ac3680f7ad03802c09ee355298c82c00ac97`.
- Current local application-GLB SHA-256:
  `5f8fc6815d710d549804b0b0082d15b9e6062feb852e7a56068d9af356a2f4e2`.
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
