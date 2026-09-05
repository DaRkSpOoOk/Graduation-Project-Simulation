# TASK-007J — Virtual-glove sensor visualization

## Task

Add a presentation-only visualization of the frozen Core-28 virtual-glove
contract to the native Qt Quick 3D application.  Each rendered hand exposes 19
Hall packages (15 bend and 4 spread) plus one palm IMU package.  The UI shows
persistent H/IMU anchors on the displayed rig and a collapsible right-side
SENSORS drawer with the current source-frame values.

## Branch

`luna/task-007j-sensor-visualization`

Exact base:

`1c2df04dcf57c7b147368735daf630c74feb2515`

The branch was created from the TASK-007I tip.  `main` was not modified or
merged.

## Scope

In scope:

- a project-owned, validated presentation map for all 20 packages per hand;
- persistent marker nodes attached to the already rendered TASK-007G/I rig;
- readable overlay and depth-tested physical marker modes;
- synchronized LEFT/RIGHT source-value models and the SENSORS drawer;
- source-frame validity, normalized values, derived bend/spread angles and
  WXYZ palm quaternion display;
- transition provenance, marker-selection feedback and CLI documentation;
- tests and native Windows smoke validation.

Out of scope:

- changes to TASK-004 through TASK-006, TASK-008, TASK-009A, TASK-009B or
  TASK-009C;
- changes to the TASK-007G/I GLB assets, camera, material, pose solver or
  transition controller;
- fake ADC, accelerometer, gyroscope or magnetometer channels;
- hardware integration, word recognition, TASK-010A or TASK-011A;
- the legacy Tk/Matplotlib debug viewer.

## Approach

The sensor overlay is a separate presentation layer over the trusted
TASK-007G/I renderer.  The main QML scene still owns one persistent View3D,
one RuntimeLoader per hand and two long-lived RiggedHand nodes.  Each
RiggedHand declares 20 SensorBadge3D instances once, giving 40 marker nodes
for the two hands.  Queue changes, source frames and transitions update
properties on those nodes; they do not create or destroy marker, hand or scene
objects.

The numeric path is deliberately different from the marker-position path:

```text
real PlaybackSequence source frame
    -> sensor_readings(position, LEFT/RIGHT)
    -> persistent SensorValueModel rows

final displayed rig pose
    -> bone/spread/palm anchor transforms
    -> persistent H/IMU marker positions
```

The marker position follows the final displayed skeleton.  The value model
reads the frozen TASK-008 source observation for the active source anchor.  It
never reads a presentation interpolation or raw landmark position.

During a TASK-007H transition the markers move with the displayed SLERP pose,
but the panel explicitly reports:

`TRANSITION · Presentation-only motion · Holding last source sensor frame`

The previous real source values remain visible until the first real source
frame of the next sign is active.  Queue completion holds the last real pose
and values with an `IDLE` status.  A missing/cleared source frame keeps the
persistent rows and marks them `NO`; it does not fabricate readings.

## Evidence / Sources

- Frozen sensor contract: `virtual_glove.layout.layout_document()` and
  `visualizer.contract.validate_sensor_layout()`.
- TASK-008 external run root:
  `..\graduation-project-runs\task008-core28-full`.
- TASK-007G/I presentation profile and the existing one-GLB-per-hand asset
  under ignored `assets-local\blendswap_hands_v1`.
- TASK-007G visual acceptance and TASK-007H motion-quality reports.
- Read-only Blender MCP inspection of the trusted rig.  The inspected palm
  and phalange bone frames showed local `+Z` pointing toward the dorsal hand
  surface in the canonical exported orientation.  The source Blender scene
  was not edited, saved or re-exported for this task.
- Qt Quick 3D's existing `View3D.mapFrom3DScene()` projection API is used for
  the optional readable overlay; see the [Qt Quick 3D View3D
  documentation](https://doc.qt.io/qt-6/qml-qtquick3d-view3d.html).

The visual evidence was generated outside Git so the repository does not gain
large screenshot artifacts.  Representative files are:

- `C:\Users\hatem\Desktop\GP\task007j-startup-sensors.png` — clean startup
  with both hands, overlay markers and panel;
- `C:\Users\hatem\Desktop\GP\task007j-overlay-dorsal.png` — PALM view with
  projected H/IMU badges;
- `C:\Users\hatem\Desktop\GP\task007j-physical-playback.png` — BACK view
  with depth-tested dorsal markers during `م` playback;
- `C:\Users\hatem\Desktop\GP\task007j-muhammad_00.png` through
  `task007j-muhammad_04.png` — multi-letter queue and transition series;
- `C:\Users\hatem\Desktop\GP\task007j-deployment.png` — deployment
  checkpoint smoke with the sensor panel enabled.

## Files Changed

- `smart_glove_app/assets/sensor_layouts/core28_virtual_glove_v1.json` — the
  authoritative project-owned display anchor map.
- `smart_glove_app/rendering/sensor_layout.py` — strict loader cross-checking
  IDs, order, types, array names and indices against the frozen contract.
- `smart_glove_app/rendering/sensor_markers.py` — persistent Qt value model;
  the old raw marker model remains only for compatibility with existing
  diagnostics/tests.
- `smart_glove_app/qml/components/SensorBadge3D.qml` — persistent rig-local
  physical marker.
- `smart_glove_app/qml/components/SensorBadgeOverlay.qml` — optional projected
  H/IMU label and marker selection interaction.
- `smart_glove_app/qml/components/SensorPanel.qml` — collapsible LEFT/RIGHT
  source-value drawer.
- `smart_glove_app/qml/components/HandStage.qml` — persistent marker instances,
  final-rig anchor resolution and overlay projection.
- `smart_glove_app/qml/Main.qml` — Sensors toolbar control and right drawer.
- `smart_glove_app/app/application_controller.py` — source-frame value
  synchronization, transition provenance and QML controls.
- `smart_glove_app/app/main.py` — `--sensors`, `--sensor-panel` and
  `--sensor-visibility` CLI options; sensor metrics.
- `pyproject.toml` — package the sensor layout JSON.
- `README.md` — native application launch documentation.
- `tests/test_task007j_sensor_visualization.py` — contract, formatting,
  persistence and QML structure tests.
- this report.

No Blender asset, GLB, scientific artifact, recognition checkpoint, camera,
lighting, skin material, pose solver or transition-controller file was
modified.

## Sensor model and complete layout

The same layout applies independently to LEFT and RIGHT.  Thus the map below
represents 20 packages on each hand and 40 packages in the application.  H1–H15
use `bend_angle_deg[ finger, joint ]`; H16–H19 use
`spread_angle_deg[ pair ]`; IMU uses `imu_quaternion_wxyz`.

| ID | Logical package | Frozen value channel | Final-rig anchor |
|---|---|---|---|
| H1 | Thumb bend 0 / base | `bend[0,0]` | `thumb_1` local dorsal offset |
| H2 | Thumb bend 1 / middle | `bend[0,1]` | `thumb_2` local dorsal offset |
| H3 | Thumb bend 2 / distal | `bend[0,2]` | `thumb_3` local dorsal offset |
| H4 | Index bend 0 / MCP | `bend[1,0]` | `index_1` local dorsal offset |
| H5 | Index bend 1 / PIP | `bend[1,1]` | `index_2` local dorsal offset |
| H6 | Index bend 2 / DIP | `bend[1,2]` | `index_3` local dorsal offset |
| H7 | Middle bend 0 / MCP | `bend[2,0]` | `middle_1` local dorsal offset |
| H8 | Middle bend 1 / PIP | `bend[2,1]` | `middle_2` local dorsal offset |
| H9 | Middle bend 2 / DIP | `bend[2,2]` | `middle_3` local dorsal offset |
| H10 | Ring bend 0 / MCP | `bend[3,0]` | `ring_1` local dorsal offset |
| H11 | Ring bend 1 / PIP | `bend[3,1]` | `ring_2` local dorsal offset |
| H12 | Ring bend 2 / DIP | `bend[3,2]` | `ring_3` local dorsal offset |
| H13 | Pinky bend 0 / MCP | `bend[4,0]` | `pinky_1` local dorsal offset |
| H14 | Pinky bend 1 / PIP | `bend[4,1]` | `pinky_2` local dorsal offset |
| H15 | Pinky bend 2 / DIP | `bend[4,2]` | `pinky_3` local dorsal offset |
| H16 | Thumb–Index spread | `spread[0]` | midpoint(`thumb_meta`, `index_meta`) + palm blend |
| H17 | Index–Middle spread | `spread[1]` | midpoint(`index_meta`, `middle_meta`) + palm blend |
| H18 | Middle–Ring spread | `spread[2]` | midpoint(`middle_meta`, `ring_meta`) + palm blend |
| H19 | Ring–Pinky spread | `spread[3]` | midpoint(`ring_meta`, `pinky_meta`) + palm blend |
| IMU | Palm IMU | `imu_quaternion_wxyz` | `palm` local dorsal offset |

The exact offsets, display labels, schema and source metadata are kept in
`core28_virtual_glove_v1.json`, not duplicated in QML or controller code.
Spread anchors recompute their current midpoint from the two displayed
metacarpal scene positions on each render revision, then blend 20% toward the
displayed palm anchor and add the map-owned dorsal offset.  The IMU is attached
only to `palm`, so finger articulation cannot make it drift.

## Marker presentation

`OVERLAY` is the readable default when sensors are enabled.  It projects a
small circular H badge (or IMU badge) from the final rig-local marker position
through the persistent `View3D`; it does not move the physical anchor.  A
click selects the sensor and switches the panel to that hand.

`PHYSICAL` keeps the markers as small Qt Quick 3D cylinders at the dorsal
anchors with normal depth testing.  A BACK view makes their physical placement
easy to inspect; a PALM view may occlude them, which is intentional.  The
panel's toggle switches between the two modes.  `Sensors OFF` leaves the hero
hand scene clean while retaining all persistent objects for the next toggle.

Invalid source channels dim their marker and show `NO`; they do not change the
displayed hand pose or alter the scientific mask.  Startup with no source
frame shows both hands and a clean idle panel with no fabricated readings.

## Value semantics

For valid Hall readings the drawer shows the exact normalized TASK-008 value
to three decimal places and a clearly labeled derived angle:

```text
Normalized    0.422
Derived angle 75.96°
Valid         YES
```

The angle is display-only and is calculated as `normalized × 180°`.  The IMU
shows the actual WXYZ tuple to four decimal places and its validity.  No ADC
counts or unmeasured inertial channels are displayed.

The unit tests compare exact source reading formatting on both sides,
including `0.422 -> 75.96°`, `0.125 -> 22.50°`, RIGHT `0.314 -> 56.52°`, and
WXYZ values.  Model row objects and row count remain stable across updates;
there is no per-frame model reset.

## Evaluation

### Required application behavior

| Check | Result | Evidence |
|---|---|---|
| Normal startup with no `--text` | PASS | Native Windows exit 0; both hands load with no queue text |
| Overlay/PALM markers | PASS | Native smoke and `task007j-overlay-dorsal.png` |
| Physical/BACK markers | PASS | Native smoke and `task007j-physical-playback.png` |
| Sensors ON/OFF and collapsible panel | PASS | CLI startup plus panel controls |
| LEFT/RIGHT panel tabs | PASS | Two persistent value models and native panel smoke |
| `م` source values | PASS | H/IMU panel values follow real source frame 0–19 |
| `محمد` queue | PASS | Four queue items; all boundary traces complete |
| Required sequence `ا → ب → م → د → ي → ح → ا` | PASS | Native run; all seven items reached terminal source anchors |
| Repeated `مم` and `اا` | PASS | Native smoke exit 0 for both; queue events remain separate |
| Transition with panel open | PASS | Panel reports presentation-only hold and retains last real values |
| Deployment checkpoint mode | PASS | `Recognition ready`, `DEPLOYMENT MODEL`, LOSO provenance retained |
| Clean shutdown | PASS | All native smoke processes exited 0 |

### Marker motion validation

The 15 bend markers are attached to the corresponding displayed phalange
bones, not to an independent landmark renderer.  Thumb, index, middle, ring
and pinky markers therefore inherit the same absolute rest-pose transforms as
the visible skinned mesh.  The four spread markers recompute pairwise base
midpoints from the current displayed metacarpal nodes.  The palm IMU is tied to
the palm bone.

The native playback tests showed the H/IMU markers moving with the hands in
both overlay and physical modes.  No marker remained at a previous sign pose,
lagged by a source frame, crossed to the other hand, or followed a wrong
finger.  The QML structure test additionally verifies the final-rig mapping
calls (`mapPositionToScene` / `mapPositionFromScene`) and rejects a raw
`landmarks_3d` marker path.

### Source-frame and transition isolation

The final native `م` metrics trace was:

```text
source sequence:       0..19
displayed source:      0,1,2,...,19
LAST FRAME PRESENTED:  YES
queue advanced after:  frame 19
early queue advance:   false
sensor source frame:   19
sensor status:          IDLE · Holding last source sensor frame
```

The `محمد` and `ا → ب → م → د → ي → ح → ا` runs showed the same complete
terminal-anchor behavior for every item.  TASK-007H's direct final-pose to
first-pose transition, boundary hold and quaternion SLERP remain unchanged.
During transition, the sensor panel does not publish interpolated values to
the value model or to recognition.  Recognition continues to receive the
original queue item's scientific sequence through the existing bridge.

### Test results

The focused application regression suite passed:

```text
python -m unittest -q \
  tests.test_task007e_deployment_handoff \
  tests.test_task007f_application_v2 \
  tests.test_task007g_presentation \
  tests.test_task007h_motion_quality \
  tests.test_task007i_sign_pose_fidelity \
  tests.test_task007j_sensor_visualization

Ran 66 tests in 12.116s
OK (skipped=4)
```

The new sensor test module alone passed 6 tests.  `python -m compileall -q
smart_glove_app tests` passed.

## Results

TASK-007G/I visual behavior is preserved.  The application now has an
optional, readable sensor instrumentation layer with two hand-specific sets
of 20 persistent packages.  The project-owned map is contract-checked, both
hand models update in place, and panel values are tied to real TASK-008 source
frames.  The deployment checkpoint smoke retained its existing role and LOSO
reference semantics.

The native Windows RHI reported:

```text
graphics API: Direct3D11
render FPS:   180.0
hand scene creation count: LEFT 1, RIGHT 1
```

This confirms that sensor instrumentation did not recreate the hand scene or
replace the Qt Quick 3D renderer.  The metrics field
`SURFACE TOPOLOGY UNAVAILABLE — POINT-CLOUD FALLBACK` refers to the optional
MANO diagnostic geometry path when no local MANO topology is supplied; it does
not describe the normal hero hands, which remain the TASK-007G/I skinned GLB
surface asset.

## Failures / Limitations

- No real hardware ADC, accelerometer, gyroscope or magnetometer values exist
  in the frozen TASK-008 runtime artifact, so none are displayed.
- Physical dorsal badges can be hidden by the hand in PALM view.  OVERLAY is
  the intended readable mode; PHYSICAL is available for inspecting believable
  placement and depth behavior.
- Marker selection is implemented for the projected overlay; physical
  cylinders are not individually pickable.
- CPU/GPU utilization, VRAM and process memory were not instrumented.  FPS and
  Qt RHI/API were measured.
- The full repository unittest discovery was also attempted.  It reported 791
  tests with 2 unrelated failures and 18 unrelated errors: a legacy 007B CLI
  test writes Arabic text to a CP1252 subprocess stream, the environment lacks
  an external production virtual-glove artifact expected by another 007B
  test, and several existing 008B workbook tests hit Windows file-lock cleanup
  errors.  The task-focused 007E–007J suite is clean; no files in those
  failing legacy areas were changed here.
- `pytest` is not installed in this Python environment; the repository's
  unittest suite was used.

## Performance

On the native Windows 11 development machine (Python 3.12, PySide6 6.11.2,
RTX 2060 Super), idle and playback smoke runs reported approximately 180 FPS
with Direct3D11.  The sensor overlay uses 40 persistent 3D marker nodes and
two persistent Qt value models.  A representative playback run updated each
hand's source model 20 times for the 20 real source frames while keeping hand
scene creation at one per side.  No pathological frame drop was observed.

## Comparison

Before TASK-007J, the application had the virtual-glove data contract but no
user-facing physical sensor locations or synchronized sensor-value panel.  The
new implementation adds:

| Concern | Before | TASK-007J |
|---|---|---|
| Hall/IMU location | Not visible in the main app | 19 H + 1 IMU marker per hand |
| Sensor motion | No final-rig overlay | Bone-local, spread-midpoint and palm anchors |
| Numeric values | Not exposed in the main app | Exact source-frame normalized/WXYZ values |
| Invalid values | No panel state | Dim/`NO`, with scientific masks unchanged |
| Queue transition values | Not applicable | Explicit transition hold; no generated reading claim |
| Rendering scene | TASK-007G/I persistent scene | Same scene; marker nodes are persistent |

## Recommendation (KEEP / REVISE / REJECT / NEEDS MORE EVALUATION)

**KEEP**.  The sensor layer is isolated, contract-checked and ready for user
review without regressing the trusted hand renderer or recognition boundary.

## Reproducibility

Repository and runtime details:

- Branch: `luna/task-007j-sensor-visualization`.
- Base: `1c2df04dcf57c7b147368735daf630c74feb2515`.
- Windows 11 native Python 3.12 virtual environment.
- PySide6 6.11.2; Qt Quick RHI Direct3D11; RTX 2060 Super.
- External run root: `..\graduation-project-runs\task008-core28-full`.
- Optional checkpoint: `..\graduation-project-runs\task009c-core28-deployment\deployment.pt`.
- Local ignored rig asset: `assets-local\blendswap_hands_v1`.
- Sensor map: `smart_glove_app/assets/sensor_layouts/core28_virtual_glove_v1.json`.

Default PowerShell launch (visualizer-only, sensors off):

```powershell
$runRoot = Resolve-Path "..\graduation-project-runs\task008-core28-full"
$rigAsset = Resolve-Path ".\assets-local\blendswap_hands_v1"

.venv\Scripts\python.exe -m smart_glove_app.app.main `
  --run-root $runRoot `
  --rig-asset $rigAsset
```

Sensor demonstration launch:

```powershell
.venv\Scripts\python.exe -m smart_glove_app.app.main `
  --run-root $runRoot `
  --rig-asset $rigAsset `
  --sensors `
  --sensor-panel `
  --sensor-visibility overlay `
  --text "محمد"
```

Physical BACK-view demonstration:

```powershell
.venv\Scripts\python.exe -m smart_glove_app.app.main `
  --run-root $runRoot `
  --rig-asset $rigAsset `
  --sensors `
  --sensor-panel `
  --sensor-visibility physical `
  --view back `
  --text "م"
```

The application has no hard-coded user home or run-root path.  The ignored
Blender/GLB assets are required for the full hand presentation but are not
part of the Git change.

## Next Steps

- User review of the overlay/physical placement and panel density.
- If desired, add a small visual calibration tool for per-asset offsets without
  moving scientific values.
- Keep future TASK-010A text integration and TASK-011A TTS integration behind
  this stable sensor-instrumented application boundary.
