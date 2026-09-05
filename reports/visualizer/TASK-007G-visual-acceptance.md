# TASK-007G — Visual acceptance pass on the Core-28 application

**Branch:** `opus/task-007g-visual-overhaul`
**Base:** `422ed699b80f118f9b2b3db8c13b3256dbbb4ff1` (Luna TASK-007F tip, PR #35, open and unmerged)
**Scope:** presentation/application layer only. No scientific contract was modified.

---

## 1. Initial visual failures

The starting point was captured before any change, by adding a window-grab flag
to the existing entry point and launching it with no arguments beyond
`--run-root`. The rendered result was worse than the written rejection
suggested: **the two hands occupied roughly 40 × 30 pixels of a 1210 × 450
viewport** — about 0.2 % of the visualisation area — as an almost-black blob on
a dark background.

Measuring the shipped asset and the running scene gave the causes:

| # | Observed failure | Measured cause |
|---|---|---|
| 1 | Hands unrecognisable as hands | The exported GLB contained **1,102 vertices / 2,178 triangles per hand** — the multires *base cage*. The source mesh evaluates to **69,825 vertices / 69,728 faces** at multires level 3. TASK-007F exported with `export_apply=False` and no baked multires, so all sculpted detail was silently dropped. |
| 2 | Hands "look weird", not natural | Same as (1), plus the two hands were different meshes: `Hand.R` carried an object scale of **0.7048** and a non-identity rotation relative to its armature, so the right hand was ~70 % the size of the left. |
| 3 | Dark graphite material | Material `TASK007F_Graphite_Glove`, `baseColorFactor = (0.018, 0.028, 0.045)` — near-black blue. |
| 4 | Camera too far / hands not dominant | Fixed `cameraDistance: 8.0` with a default 60° vertical FOV, giving ~25 scene units of visible width for a 4.4-unit-wide subject. |
| 5 | Hands "detached", edge-on | **The dominant cause.** The exported rest pose had the fingers pointing *at the camera*. Measured skinned bounds per hand were `X 1.12 × Y 0.65 × Z 2.07` — the hands were 2.07 units *deep* and only 0.65 units *tall*. Their projected screen area was therefore tiny regardless of camera distance. The armature rotations were the original artist's arbitrary posed values (Euler `(-185.9, 18.8, -164.1)` and `(-184.0, -22.5, 159.9)`). |
| 6 | Left/right inconsistent | `Hand.L` carried `COLOR_0` and `COLOR_1` vertex-colour attributes; `Hand.R` carried none. Root bone offsets were asymmetric (`[0,0,0]` vs `[0.372, 0.076, -0.04]`), as were the presentation roots (`x = -3.07` vs `x = +3.95`). |
| 7 | Strange orbit after pressing a letter | See §7 — a genuine, reproducible defect, and not the one the symptom suggested. |

---

## 2. Technology decision — Qt Quick 3D was **kept**

The renderer was evaluated against the observed evidence rather than a
preference. Every failure above traces to the **asset export, the material, the
camera model or the transform maths** — none of them to the renderer. The
Blender viewport rendering the same geometry looked correct throughout, which
made "the renderer is inadequate" untenable as a diagnosis.

Keeping PySide6 + Qt Quick 3D also preserves the single most valuable property
of the current architecture: the Python scientific backend (TASK-008 loading,
TASK-009 recognition, Core-28 mapping, queue semantics) runs **in-process**.
Replacing the frontend with Godot/Unity/Unreal/three.js would have introduced an
IPC bridge, a second toolchain and a packaging story, to fix problems that live
in the asset pipeline.

The renderer did, however, force two concrete engine-level findings that shaped
the design (§4, §6). Both are worked around explicitly rather than papered over.

**Verdict:** renderer kept; asset, material, camera, rig profile, retargeter,
node-resolution strategy and UI all replaced.

---

## 3. Asset

**Kept the CC0 source, rebuilt the derived asset.** The BlendSwap mesh has real
sculpted detail once the multires is actually applied, so replacing the asset
would have discarded a good, correctly-licensed model to fix a build problem.

| Field | Value |
|---|---|
| Source | <https://www.blendswap.com/blend/22269> — "Hands Rigged" |
| Creator | SparrowHawk |
| Licence | **CC0** |
| Preserved original | `assets-local/blendswap_hands_v1/blendswap_hands_v1_ORIGINAL_PRESERVED.blend` (untouched snapshot, created before any edit) |
| Derived working file | `assets-local/blendswap_hands_v1/task007g_presentation_hands.blend` |
| Shipped runtime assets | `task007g_hand_left.glb` (5.31 MB), `task007g_hand_right.glb` (5.31 MB) |
| Tracked profile | `smart_glove_app/assets/rig_profiles/task007g_hands.json` (project-authored names, calibration and layout only) |

All `.blend` and `.glb` files stay under the git-ignored `assets-local/` path.
The tracked profile contains no third-party geometry or texture data.

---

## 4. Blender work performed

Executed through the Blender MCP bridge against the live working copy, after
snapshotting the untouched original.

1. **Preserved the original** as `..._ORIGINAL_PRESERVED.blend` before any edit.
2. **Chose one canonical hand.** The source `Hand.L` was taken as the single
   reference because it alone had identity rotation, unit scale and a UV map.
   Measuring its anatomy showed it is geometrically a **right** hand (the thumb
   sits opposite `cross(finger_direction, palm_normal)`), and it was renamed
   accordingly — the source naming was wrong.
3. **Baked the sculpt.** Applied the multires modifier: **1,102 → 69,825
   vertices**, smooth-shaded. This is the single change that makes the hands
   read as hands.
4. **Cleaned the rig.** Removed the four non-deform control bones
   (`index control`, `Major control`, `Ring control`, `Pinky control`) and all
   **17 pose constraints** (12 × `COPY_ROTATION`, 5 × `IK`). glTF cannot carry
   constraints, and they fight direct posing.
5. **Renamed all 22 deform bones anatomically** (`Bone.016` → `palm`,
   `Bone.003` → `middle_1`, …), renaming the matching vertex groups in the same
   pass. The TASK-007F profile's `Bone.0NN` identifiers made the channel table
   impossible to review.
6. **Established a canonical presentation frame.** The anatomical basis
   (finger direction, palm normal, palm lateral) was measured from bone
   positions, and a single corrective rotation `R = T · Sᵀ` was solved and
   applied so that, in glTF Y-up space, **fingers point +Y and the palm normal
   points +Z (at the camera)**. Each hand was then centred on its own origin.
   Resulting bounds: `0.44 × 1.188 × 0.344` — 3.5× taller than deep, versus the
   old asset's 3.2× deeper than tall.
7. **Mirrored the counterpart exactly.** The opposite hand was produced by
   mirroring every bone matrix as `M · matrix · M` with `M = diag(-1,1,1)`, and
   mirroring the mesh vertices with reversed winding. A first attempt using
   Blender's usual `roll = -roll` convention produced *inconsistent* flexion
   axes across fingers and was discarded on measurement. Verified: **max bone
   mirror error 0.0 across all 22 bones**, identical vertex counts, identical
   bounding boxes, and left flexion displacement an exact X-mirror of right.
8. **Authored a natural skin material** (§8) and removed the asymmetric vertex
   colour attributes so both hands shade identically.
9. **Exported one GLB per hand** (§6).

---

## 5. Retargeting approach

The frozen **15 bend + 4 spread + palm orientation** channels were kept —
option (A). The 21-joint route was not needed: the measured problem was never
the channel data, and the direct mapping produces correct, distinctive shapes
once applied to a correctly-oriented rig with the right axes.

What was replaced is **how** those channels reach the rig
(`smart_glove_app/app/hand_pose_solver.py`, replacing `hand_rig_retargeter.py`):

- **One uniform convention instead of a per-bone sign table.** Because the left
  rig is an exact matrix mirror of the right, a negative rotation about a bone's
  **local X** flexes it toward the palm on *both* hands, and local Z abducts it.
  The TASK-007F profile instead carried a hand-calibrated axis and sign for each
  of the 38 channel/side pairs, mixing X and Z within a single finger chain
  (index proximal about Z, index middle about X, index distal about Z) — which
  is anatomically impossible and is why finger motion did not read as flexion.
- **Anatomical joint limits.** MCP ≤ 90°, PIP ≤ 110°, DIP ≤ 70°, thumb
  55/60/75°, replacing a flat 0–120° clamp on every joint.
- **Spread measured outward from the middle finger** and applied on the
  metacarpals, which carry no bend channel. In TASK-007F a bone could receive
  both a bend and a spread and the later write silently won; a regression test
  now asserts the two channel families never share a bone.
- **Spread neutral and clamp**: neutral per pair (thumb-index 45°, others
  10–12°), applied delta clamped to `[-14°, +18°]`, so spread cannot splay the
  hand implausibly. The rig's own measured rest spread (31.6 / 2.1 / 1.6 / 4.2°)
  is recorded in the profile.

---

## 6. Camera model and presentation/motion separation

### Camera

The camera is **solved from the layout**, not stored as state:

```
distance = max( contentHalfHeight / tan(fov/2),
                contentHalfWidth  / (tan(fov/2) · aspect) ) / zoom
```

with `fov = 35°` (a longer lens than the previous 60°, less perspective
distortion on a hero subject), `contentHalfHeight = 0.68`,
`contentHalfWidth = 0.90` against a 1.188-unit-tall hand. The hands therefore
occupy ~80 % of the viewport height and stay **identically framed at every
window size**. Verified numerically: the left hand's measured screen offset
(−225 px) matches the predicted −214 px for its −0.55 scene position.

There is **no free orbit in normal operation**. The orbit pivot exists but sits
at identity unless *Inspect* is explicitly enabled from the diagnostics drawer,
and `Reset` returns everything to the default.

### PALM / BACK

Each hand hangs under a QML-owned presentation `Node`:

```
PresentationNode (position, view-mode rotation, opacity)   ← application only
  └─ RuntimeLoader → armature → palm → metacarpals → phalanges   ← recorded data
```

`PALM` is `eulerRotation = (0,0,0)` — the asset's canonical rest already faces
the camera. `BACK` is `(0,180,0)` applied to **each hand about its own axis**,
so the backs face the viewer while **LEFT stays left and RIGHT stays right**.
The switch is a declarative binding on a value read from the profile: it is
deterministic, cannot accumulate, and animates over 260 ms.

### Separation

Recorded motion can reach exactly two places: the finger/metacarpal bones, and
the wrist joint. It can never reach the presentation node, the camera or the
layout. The recorded wrist rotation is additionally taken **relative to each
sequence's own first valid frame** and **hard-clamped to 25°**, so even a
pathological input cannot rotate a hand out of frame.

Measured over the six acceptance letters, applying the solver to the real GLB
hierarchy offline: **maximum wrist rotation 4.01°**, **maximum centroid drift
0.12 units** on a 1.188-unit hand, left-hand X-span always within
`[-0.82, -0.40]` and right-hand within `[+0.38, +0.81]` — **zero overlapping
frames**.

---

## 7. The "orbit" bug — root cause

The reported symptom was investigated rather than guessed at, and the first two
hypotheses were both **wrong**:

- *"The recorded palm quaternion swings the hand."* Measured across six signs:
  the relative wrist rotation never exceeds **4.0°** over an entire sequence,
  with a median frame-to-frame step under 1°. Far too small to see.
- *"The wrist reference leaks between letters."* Real, but secondary — the
  existing code did already reset between queue items.

The actual defect is in the QML bone application, and it existed in TASK-007F's
`HandViewport.qml` in exactly the same form:

```qml
bases[side][boneName] = boneNode.rotation          // intended: snapshot the rest pose
...
node.rotation = baseRotation.times(delta)          // rest ∘ solved delta
```

**QML value types are handed back by reference.** `boneNode.rotation` does not
return a copy; the cached "rest" rotation stays bound to the live property. So
after the first posed frame the cache reads back the *posed* value, and every
subsequent frame composes onto the previous frame's result:

```
frame 1:  rest · δ₁
frame 2:  (rest · δ₁) · δ₂
frame 3:  ((rest · δ₁) · δ₂) · δ₃      → unbounded accumulation
```

With ~20 frames per sign at up to 100° per joint this diverges within a few
frames, which is what produced the crumpled, tumbling hands the user read as a
"strange orbit", and why the distortion **persisted after playback finished**
(the corrupted rest snapshot never recovered).

Isolating it took a dedicated single-bone QML harness, because the same
composition applied in Blender — using the identical solved quaternions — always
produced a clean, plausible pose.

**Fix:** snapshot the rest rotation as plain numbers and rebuild the pose
absolutely from it every frame.

```qml
var r = bone.rotation
rest[name] = [r.scalar, r.x, r.y, r.z]     // real snapshot, not a live reference
...
bone.rotation = Qt.quaternion(rest[0], rest[1], rest[2], rest[3])
                  .times(Qt.quaternion(q[0], q[1], q[2], q[3]))
```

**Reproducible test:** `tools/visual_acceptance/rig_probe.py` rebuilds the glTF
hierarchy and performs the identical composition offline, and
`tests/test_task007g_presentation.py::RealSequenceFramingTests` drives it over
real TASK-008 sequences, asserting per frame that the wrist stays inside its
clamp, that neither hand crosses the centre line, and that the two never
overlap.

### Two further engine-level defects found the same way

1. **Qt Quick 3D mis-binds a glTF containing two skins.** With both hands in one
   file, the second skinned mesh is deformed by the *first* skeleton: the right
   hand rendered mangled on top of the left, and appeared simply "missing".
   Proven by hiding each `Model` in turn. **Fix:** ship one GLB per hand, each
   in its own `RuntimeLoader`.
2. **`ExtendedSceneEnvironment` post-processing composites the scene to a
   near-black silhouette** on this D3D11/RHI path — reproducibly, with tonemap,
   SSAO or vignette enabled in any combination, and *with an unlit material*,
   which rules out lighting. **Fix:** a plain `SceneEnvironment`, with all
   shaping done by the light rig.

---

## 8. Material approach

Default is **natural light skin**, not graphite and not chalk-white.

| Property | Value |
|---|---|
| Base colour | sRGB `(0.855, 0.663, 0.553)` / linear `(0.692, 0.393, 0.263)` |
| Roughness | 0.55 |
| Metalness | 0.0 |
| Specular | 0.40, with a 0.05 clearcoat for a subtle skin sheen |
| Culling | none (the source mesh is double-sided) |

Lighting is a four-light rig, calibrated between two measured exposure points
(a blown-out reference and an under-lit one): warm key at 1.35, cool fill 0.60,
cool rim 0.70 to separate the hands from the background, and a 0.45 frontal
bounce so palm modelling stays readable in PALM view.

Optional appearance modes — **dark glove** and **unlit wireframe** — are
reachable from the diagnostics drawer and via `--appearance`. The MANO point
cloud remains behind the explicit `--debug-mano-points` flag and never appears
in normal playback.

---

## 9. Node resolution (why the profile no longer stores structural paths)

Qt's `RuntimeLoader` **discards glTF node names** — every node in the loaded
tree has an empty `objectName`, so a lookup by name is impossible. TASK-007F
worked around this with structural child-index paths hand-written into the rig
profile, which go stale silently on any re-export or engine change.

TASK-007G derives those paths from the GLB that is actually about to be loaded
(`smart_glove_app/rendering/glb_index.py`), and the QML side *searches for and
validates* the subtree they are relative to, accepting a candidate only if all
22 bones resolve to distinct nodes. Stale calibration is no longer possible.

---

## 10. UI

Redesigned around the hands as the hero element:

- The 3D stage takes the entire upper area (~55 % of a 1500 × 980 window).
- Arabic Core-28 keyboard, laid out right-to-left, one click = one sign event.
- Recognised/queued text with queue progress on a single quiet strip.
- Playback controls in one row; status reduced to a single muted line.
- **All technical diagnostics moved behind a drawer** (closed by default):
  render FPS, source FPS, frame, graphics API, sample id, hand-asset state,
  expected/predicted/confidence, recognition status, the LOSO reference string,
  and the appearance/speed/smoothing/inspect controls.
- No debug warning banner in normal startup; the asset notice appears only on
  an actual failure.

`smart_glove_app/qml/components/PlaybackControls.qml` and `RecognitionCard.qml`
were removed; their functionality is in the new layout and the drawer.

---

## 11. Manual visual acceptance

Every check below was performed against **rendered screenshots of the real
application**, not from automated tests alone. Captures were produced with the
new `--screenshot` / `--screenshot-series` flags.

| Test | Result |
|---|---|
| Normal startup, no `--text` | Two large natural-skin hands, both palms to viewer, empty queue, keyboard ready, no warning banner |
| `ا` (alif) | Both hands curled; distinct, plausible |
| `ب` (baa) | Right index extended vertically, left curled — immediately recognisable |
| `ح` (haa) | Both hands curled, distinct from alif |
| `م` (meem) | Right closed fist, left with pinky/index extended — matches the Blender ground-truth render of the same solved pose |
| `د` (dal) | Distinct curl shape |
| `ي` (yaa) | Right hand with two fingers extended |
| `محمد` | All four signs play in order; the repeated `م` is a separate event; framing, size and separation identical throughout |
| PALM → BACK | Knuckles and tendons face the viewer, thumbs swing outward, LEFT stays left, RIGHT stays right, framing unchanged |
| Camera stability | No orbit, drift or reframing at any point during playback |
| Hand overlap | None, in any frame of any tested sign |

Recognition, on the TASK-009C deployment checkpoint:

| Sign | Sample | Expected | Predicted | Confidence |
|---|---|---|---|---|
| م | `karsl_core28_s01_sign0055_train_rep010` | م | م | 0.9995 |
| ح | `karsl_core28_s01_sign0037_train_rep026` | ح | ح | 0.9904 |
| م | `karsl_core28_s01_sign0055_train_rep010` | م | م | 0.9995 |
| د | `karsl_core28_s03_sign0039_test_rep001` | د | د | 0.7725 |

The application continues to report the reference verbatim: *"LOSO reference
only (not deployment accuracy): 67.63% accuracy / 0.6607 macro F1"*.

---

## 12. Acceptance criteria

| # | Criterion | Status |
|---|---|---|
| 1 | Both hands immediately recognisable as human hands | PASS |
| 2 | Default appearance natural/light skin | PASS |
| 3 | Both palms face the viewer by default | PASS |
| 4 | Hands fill most of the visualisation area | PASS (~80 % of viewport height) |
| 5 | Camera close and stable | PASS (fit-solved, no stored orbit) |
| 6 | Pressing a letter never causes weird global orbit | PASS (root cause fixed, §7) |
| 7 | Recorded motion does not destroy framing | PASS (≤4.01° measured, 25° clamp) |
| 8 | LEFT remains left | PASS (X-span always negative) |
| 9 | RIGHT remains right | PASS (X-span always positive) |
| 10 | PALM/BACK toggle predictable | PASS |
| 11 | Finger bends anatomically plausible | PASS |
| 12 | Thumb movement plausible | PASS |
| 13 | Hands do not overlap | PASS (0 overlapping frames) |
| 14 | No praying pose | PASS |
| 15 | No giant-ball fallback | PASS (removed) |
| 16 | No point cloud in normal mode | PASS (explicit flag only) |
| 17 | Keyboard works | PASS |
| 18 | Recognition still works | PASS |
| 19 | Scientific contracts untouched | PASS (no file under `recognition/`, `kinematics/`, `tracking/`, `pose/`, `virtual_glove/`, `evaluation/`, `datasets/`, `configs/` changed) |
| 20 | Windows native application works | PASS (Direct3D11 RHI) |

---

## 13. Performance

- **180 FPS** measured (`FrameAnimation`-synced), Qt Quick RHI · **Direct3D11**,
  on the RTX 2060 SUPER — three times the 60 FPS requirement.
- ~140 k triangles per hand, 280 k total, two skinned meshes, four directional
  lights, MSAA high.
- The geometry providers are still created once and updated in place
  (`geometry_creation_count == 1`).

---

## 14. Tests

- New: `tests/test_task007g_presentation.py` — **25 tests**, all passing.
  Profile consistency, the no-shared-bone invariant, uniform flexion sign, joint
  and spread clamping, wrist clamping, the reset-between-signs regression,
  invalid-channel hold, exported-asset structure (mirror equality, vertex count,
  facing-the-camera check, skin-tone check), and the real-sequence framing
  property.
- New: `tools/visual_acceptance/rig_probe.py` — offline reproduction of the QML
  transform maths.
- Updated: `tests/test_task007f_application_v2.py` — the four direct-bone
  retargeting tests were removed as superseded; MANO topology, persistent-scene,
  keyboard, geometry-buffer and marker coverage retained, and the QML contract
  test repointed at `HandStage.qml`.
- **Full suite: 770 tests. The failure set is byte-identical to the base commit**
  (20 pre-existing failures caused by missing local workbook/run artifacts, a
  Windows temp-file cleanup issue and a console-encoding assertion). **No new
  failures were introduced.**

---

## 15. Remaining limitations

1. **SSAO and tonemapping are off.** Qt's `ExtendedSceneEnvironment` post chain
   renders this scene near-black on this D3D11 path (§7). Contact shadows
   between fingers would add depth; the light rig compensates but does not fully
   replace them. Worth revisiting on a newer Qt or the Vulkan backend.
2. **Wrist motion is presentation-normalised, not physically transformed.** The
   recorded palm quaternion is expressed in the tracking camera frame; an exact
   basis change into the presentation frame is not derivable from the current
   contract. Because the measured motion is ≤4°, the clamped delta is applied
   directly. This is a deliberate, documented approximation, and the clamp makes
   it safe.
3. **Spread neutrals are anatomical estimates**, not derived from the dataset
   distribution. The clamp bounds the consequence.
4. **Signs hold their final pose** after the queue empties rather than returning
   to neutral. This is intentional but is a presentation choice, not a data one.
5. **Sensor markers are not rendered on the new stage.** The TASK-006 marker
   overlay was tied to the old MANO-scale geometry; the model is still
   maintained but not drawn. The legacy `visualizer` entry point still shows it.
6. **The forearm is long.** The source mesh includes a substantial forearm
   stub. It reads naturally as hands rising into frame and was kept, but it does
   consume vertical space.
7. **Skin has no texture map.** A single PBR colour with no albedo/normal detail;
   it reads as clean and natural rather than photoreal.

---

## 16. Reproduction

```powershell
python -m pip install -e ".[gui,recognition]"
python -m smart_glove_app `
  --run-root "..\graduation-project-runs\task008-core28-full" `
  --checkpoint "..\graduation-project-runs\task009c-core28-deployment\deployment.pt"
```

Visual acceptance captures:

```powershell
python -m smart_glove_app --run-root "..\graduation-project-runs\task008-core28-full" `
  --text "محمد" --speed 0.5 `
  --screenshot ".\shots\mhmd.png" --screenshot-series 16 --screenshot-interval 0.55 `
  --screenshot-delay 4.0 --smoke-seconds 16
```

Tests:

```powershell
python -m unittest tests.test_task007g_presentation -v
python -m unittest discover -s tests -p "test_*.py"
```
