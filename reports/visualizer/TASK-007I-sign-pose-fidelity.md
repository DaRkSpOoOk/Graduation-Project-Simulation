# TASK-007I — Core-28 sign-pose fidelity

## Task

Improve the visual fidelity of all 28 Core-28 keyboard signs after TASK-007H,
without changing the frozen scientific pipeline.  The task specifically
audits source exemplars, stored TASK-008 geometry, TASK-005 kinematics,
presentation clamps, retargeting, and rig limitations.  Sensor markers and a
live sensor panel were intentionally postponed.

## Branch

`luna/task-007i-sign-pose-fidelity`

Exact base:

`32ab2f2ead0ab8a2e6da93b961a3bec40a9a252d`

The branch was created from the TASK-007H tip, not from `main`.

## Scope

In scope:

- identify whether source frames were cut at queue boundaries;
- audit one deterministic canonical exemplar for every Core-28 label;
- compare the TASK-007H channel-only presentation against stored TASK-008
  hand landmarks;
- correct the presentation retargeter where the frozen channels do not carry
  enough signed geometric information;
- preserve the TASK-007G renderer, asset, camera, material, PALM/BACK behavior,
  persistent scene, and absolute rest-pose application;
- keep presentation-only geometry guidance isolated from recognition.

Out of scope:

- TASK-004 through TASK-006 scientific artifacts;
- TASK-008 production artifacts;
- TASK-009A tensorization, TASK-009B evaluation, and TASK-009C weights;
- KArSL dataset modification or training-time exemplar selection;
- sensor markers, H circles, IMU markers, or a live sensor sidebar;
- TASK-010A or TASK-011A.

## Approach

The investigation followed the requested order.

1. The canonical keyboard resolver and TASK-008 loader were used for all 28
   labels.  The transition boundary controller was run independently at 60 Hz
   and in native Qt playback.
2. Source frame arrays, masks, timestamps, hand presence, and landmark
   geometry were snapshotted before presentation solving and compared after it.
3. The old channel-only pose was compared with the direction of every stored
   metacarpal/phalange segment in both hands.
4. The trusted TASK-007G Blender rig was inspected through Blender MCP.  A
   temporary one-bone-at-a-time deformation test was restored in a `finally`
   block and the working `.blend` was not saved.
5. A presentation-only landmark-guided solve was implemented.  TASK-005
   values and masks remain frozen inputs; valid TASK-008 landmark directions
   supply the signed spatial information missing from unsigned pairwise
   spread.  Each bone is solved from immutable GLB rest rotations and the
   shortest world-space swing to the current target direction.
6. A full landmark-frame fit was deliberately rejected after a controlled
   test produced large axial roll and visible skin twisting.  The final solve
   preserves the authored TASK-007G axial roll.
7. The native Windows application was run for all-28 screenshots, the required
   transition matrix, repeated letters, no-smoothing mode, PALM/BACK, and
   deployment-checkpoint mode.

## Evidence / Sources

The evidence sources were:

- `datasets/manifests/karsl_core28.csv` and the authoritative Core-28 mapping;
- the external TASK-008 run root:
  `..\graduation-project-runs\task008-core28-full`;
- the external TASK-009C deployment checkpoint, used only for recognition
  smoke testing;
- the shipped TASK-007G one-hand GLBs under the ignored `assets-local` path;
- `reports/visualizer/TASK-007G-visual-acceptance.md`;
- `reports/visualizer/TASK-007H-motion-quality.md`;
- native Windows Qt Quick RHI diagnostics;
- Blender MCP inspection of the currently open trusted working asset.

The original KArSL RGB videos were not available locally.  The manifest
contains relative source paths and hashes, but matching files were absent from
the repository, the external run root, the project Desktop tree, Downloads,
and Documents.  The unrelated videos found elsewhere on the Desktop did not
match KArSL manifest paths.  Consequently, the visual audit uses the strongest
available ground truth: stored TASK-008 21-point 3D geometry, frozen TASK-005
validity/kinematics, the exported skinned rig, and native rendered frames.
No RGB frame is claimed or synthesized.

The reproducible external audit outputs are kept outside Git:

- rendered all-28 contact sheet:
  `C:\Users\hatem\AppData\Local\Temp\task007i-all28-swing\contact_sheet.png`;
- source-landmark diagnostic contact sheet:
  `C:\Users\hatem\AppData\Local\Temp\task007i-sourceaudit\contact_sheet.png`.

## Files Changed

- `smart_glove_app/app/application_controller.py` — passes the indexed,
  per-hand GLB paths to the presentation solver after asset discovery.
- `smart_glove_app/app/hand_pose_solver.py` — adds optional immutable GLB rest
  calibration and validity-gated TASK-008 landmark-direction guidance with
  shortest-arc swings; the TASK-007H channel solver remains the fallback.
- `smart_glove_app/rendering/rig_pose_calibration.py` — read-only GLB node,
  parent, local-rotation, and world-rest-rotation calibration.
- `smart_glove_app/assets/rig_profiles/task007g_hands.json` — records the
  project-owned source/presentation basis and presentation-only isolation
  policy.
- `scripts/audit_task007i_sign_pose_fidelity.py` — reproducible read-only
  all-28 boundary, source-array, exemplar, clamp, and direction audit.
- `tests/test_task007i_sign_pose_fidelity.py` — rotation, swing, GLB
  calibration, direction, and scientific-array isolation tests.
- this report.

No QML renderer, GLB, camera, lighting, material, PALM/BACK, or sensor UI file
was changed.

## How to Run

PowerShell visualizer-only mode:

```powershell
$runRoot = Resolve-Path "..\graduation-project-runs\task008-core28-full"
$rigAsset = Resolve-Path ".\assets-local\blendswap_hands_v1"

.venv\Scripts\python.exe scripts\run_core28_application.py `
  --run-root $runRoot `
  --rig-asset $rigAsset `
  --text "محمد" `
  --boundary-hold-ms 80 `
  --transition-min-ms 150 `
  --transition-max-ms 350
```

Exact all-28 audit command:

```powershell
.venv\Scripts\python.exe scripts\audit_task007i_sign_pose_fidelity.py `
  --run-root $runRoot `
  --rig-asset $rigAsset `
  > "$env:TEMP\task007i-audit.json"
```

If a user has the licensed KArSL RGB tree, its root may be supplied explicitly
for matching manifest paths:

```powershell
.venv\Scripts\python.exe scripts\audit_task007i_sign_pose_fidelity.py `
  --run-root $runRoot `
  --rig-asset $rigAsset `
  --rgb-root "D:\datasets\KArSL"
```

Deployment checkpoint mode:

```powershell
$checkpoint = Resolve-Path "..\graduation-project-runs\task009c-core28-deployment\deployment.pt"

.venv\Scripts\python.exe scripts\run_core28_application.py `
  --run-root $runRoot `
  --rig-asset $rigAsset `
  --checkpoint $checkpoint `
  --text "م"
```

## Evaluation

### Transition and boundary audit

The source playback path was not cutting frames.  The all-28 audit found:

- 28/28 canonical sequences loaded successfully;
- source frame indices were contiguous `0..N-1` for every sequence;
- every canonical sequence had both hands present on every stored frame;
- every sequence had complete boundary traces;
- every terminal source frame was presented before queue advance;
- `early_queue_advance` was false for every sequence;
- source bend, spread, and validity arrays were unchanged after solving.

The native matrix `ا → ب → م → د → ي → ح → ا` also visited every source anchor
and terminal frame for all seven items.  No recognition callback altered the
playback trace, no camera state changed, and no scene or geometry object was
recreated.

The existing TASK-007H transition controller remains in use: an 80 ms
configurable final-pose hold, direct final-pose-to-first-pose blending,
smoothstep easing, quaternion SLERP, and a pose-distance adaptive duration
clamped to 150–350 ms.  The new pose solve changes the endpoints; it does not
change this transition policy.

### All-28 visual/source audit

`Old error` is the mean angle, in degrees, between the TASK-007H channel-only
presentation and the stored TASK-008 segment direction, measured over valid
presentation segments.  `Guided max` is the largest corresponding residual
after the TASK-007I shortest-swing solve.  The source caveat column records
validity gaps or channel-fallback clamp observations; it is not a fabricated
scientific value.

All rows are `PASS` against the available stored-geometry ground truth.  Direct
RGB comparison was unavailable, as described above.

| SignID | Character | Canonical sample / signer | Frames / duration | Both | Bend / spread valid | Old error | Guided max | Source caveat | Status |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| 0032 | ا | `karsl_core28_s03_sign0032_train_rep019` / 03 | 25 / 0.800 s | 1.000 | 1.000 / 1.000 | 59.6° | 0.000001° | ring distal clamp ×19 in fallback | PASS |
| 0033 | ب | `karsl_core28_s01_sign0033_train_rep022` / 01 | 26 / 0.833 s | 1.000 | 1.000 / 1.000 | 53.2° | 0.000001° | - | PASS |
| 0034 | ت | `karsl_core28_s01_sign0034_train_rep010` / 01 | 19 / 0.600 s | 1.000 | 1.000 / 1.000 | 50.7° | 0.000002° | - | PASS |
| 0035 | ث | `karsl_core28_s01_sign0035_train_rep019` / 01 | 19 / 0.600 s | 1.000 | 1.000 / 1.000 | 49.7° | 0.000001° | - | PASS |
| 0036 | ج | `karsl_core28_s01_sign0036_test_rep007` / 01 | 20 / 0.633 s | 1.000 | 1.000 / 1.000 | 69.1° | 0.000001° | - | PASS |
| 0037 | ح | `karsl_core28_s01_sign0037_train_rep026` / 01 | 20 / 0.633 s | 1.000 | 1.000 / 1.000 | 66.3° | 0.000001° | - | PASS |
| 0038 | خ | `karsl_core28_s03_sign0038_train_rep042` / 03 | 22 / 0.700 s | 1.000 | 1.000 / 0.966 | 88.6° | 0.000001° | partial `index-middle`, `middle-ring` spread | PASS |
| 0039 | د | `karsl_core28_s03_sign0039_test_rep001` / 03 | 21 / 0.667 s | 1.000 | 1.000 / 0.750 | 79.4° | 0.000001° | partial `middle-ring`, `ring-pinky`; ring/pinky MCP clamps ×21 | PASS |
| 0040 | ذ | `karsl_core28_s03_sign0040_train_rep004` / 03 | 19 / 0.600 s | 1.000 | 1.000 / 0.993 | 74.2° | 0.000001° | partial `ring-pinky` spread | PASS |
| 0041 | ر | `karsl_core28_s01_sign0041_test_rep006` / 01 | 19 / 0.600 s | 1.000 | 1.000 / 0.750 | 65.4° | 0.000001° | partial `middle-ring`, `ring-pinky`; pinky MCP clamp ×19 | PASS |
| 0042 | ز | `karsl_core28_s01_sign0042_test_rep003` / 01 | 21 / 0.667 s | 1.000 | 1.000 / 1.000 | 68.7° | 0.000001° | pinky MCP clamp ×15 in fallback | PASS |
| 0043 | س | `karsl_core28_s01_sign0043_test_rep003` / 01 | 20 / 0.633 s | 1.000 | 1.000 / 1.000 | 48.7° | 0.000001° | - | PASS |
| 0044 | ش | `karsl_core28_s01_sign0044_train_rep007` / 01 | 19 / 0.600 s | 1.000 | 1.000 / 1.000 | 47.7° | 0.000001° | - | PASS |
| 0045 | ص | `karsl_core28_s03_sign0045_train_rep014` / 03 | 23 / 0.733 s | 1.000 | 1.000 / 1.000 | 63.4° | 0.000001° | - | PASS |
| 0046 | ض | `karsl_core28_s01_sign0046_train_rep041` / 01 | 20 / 0.633 s | 1.000 | 1.000 / 0.750 | 60.6° | 0.000000° | partial `middle-ring`, `ring-pinky`; several fallback clamps | PASS |
| 0047 | ط | `karsl_core28_s03_sign0047_train_rep016` / 03 | 19 / 0.600 s | 1.000 | 1.000 / 1.000 | 65.6° | 0.000001° | wrist reached 23.0° of 25° cap | PASS |
| 0048 | ظ | `karsl_core28_s01_sign0048_test_rep004` / 01 | 20 / 0.633 s | 1.000 | 1.000 / 0.750 | 64.8° | 0.000001° | partial `index-middle`, `middle-ring` spread | PASS |
| 0049 | ع | `karsl_core28_s01_sign0049_train_rep013` / 01 | 20 / 0.633 s | 1.000 | 1.000 / 1.000 | 61.8° | 0.000001° | pinky MCP clamp ×11 in fallback | PASS |
| 0050 | غ | `karsl_core28_s01_sign0050_train_rep003` / 01 | 20 / 0.633 s | 1.000 | 1.000 / 0.750 | 71.2° | 0.000001° | partial `middle-ring`, `ring-pinky`; ring/pinky MCP fallback clamps | PASS |
| 0051 | ف | `karsl_core28_s03_sign0051_test_rep006` / 03 | 24 / 0.767 s | 1.000 | 1.000 / 1.000 | 73.6° | 0.000001° | - | PASS |
| 0052 | ق | `karsl_core28_s03_sign0052_train_rep002` / 03 | 19 / 0.600 s | 1.000 | 1.000 / 0.862 | 55.8° | 0.000001° | partial `middle-ring`, `ring-pinky` spread | PASS |
| 0053 | ك | `karsl_core28_s01_sign0053_test_rep001` / 01 | 21 / 0.667 s | 1.000 | 1.000 / 1.000 | 55.9° | 0.000001° | - | PASS |
| 0054 | ل | `karsl_core28_s01_sign0054_train_rep005` / 01 | 20 / 0.633 s | 1.000 | 1.000 / 1.000 | 54.8° | 0.000001° | - | PASS |
| 0055 | م | `karsl_core28_s01_sign0055_train_rep010` / 01 | 20 / 0.633 s | 1.000 | 1.000 / 1.000 | 54.4° | 0.000002° | - | PASS |
| 0056 | ن | `karsl_core28_s01_sign0056_train_rep026` / 01 | 19 / 0.600 s | 1.000 | 1.000 / 0.961 | 63.4° | 0.000001° | partial `ring-pinky` spread | PASS |
| 0057 | ه | `karsl_core28_s01_sign0057_train_rep003` / 01 | 20 / 0.633 s | 1.000 | 1.000 / 1.000 | 83.6° | 0.000001° | thumb proximal clamp ×20 in fallback | PASS |
| 0058 | و | `karsl_core28_s01_sign0058_test_rep002` / 01 | 20 / 0.633 s | 1.000 | 1.000 / 1.000 | 62.8° | 0.000001° | pinky distal clamp ×20 in fallback | PASS |
| 0059 | ي | `karsl_core28_s01_sign0059_test_rep008` / 01 | 21 / 0.667 s | 1.000 | 1.000 / 0.750 | 65.0° | 0.000001° | partial `middle-ring`, `ring-pinky`; middle MCP fallback clamp ×21 | PASS |

## Results

### 1. Transition root cause and regression result

The reported “cut in the middle” was not a source-frame drop in the current
TASK-007H playback boundary.  Every source frame was visited and the terminal
frame was presented before the queue advanced.  The transition matrix, `محمد`,
`مم`, and `اا` all retained this result.

The task therefore did not hide a playback bug with smoothing.  It retained
the TASK-007H direct-pose transition.  The visible improvement comes from
correcting the pose endpoints, not from inserting frames into the scientific
sequence.

### 2. Actual sign-pose root cause

The old presentation solve interpreted TASK-005 spread as signed metacarpal
rotations around guessed neutral values.  TASK-005 intentionally stores
unsigned pairwise angles, so the same measurement cannot identify whether a
finger lies toward the thumb or toward the pinky.  This was especially damaging
to thumb opposition and to signs whose fingers were separated in different
directions.

The evidence was decisive:

- valid bend channels already produced the expected per-segment turn in both
  hands; no global flexion-axis or sign rewrite was justified;
- old channel-only direction error ranged from 47.7° to 88.6° across the
  canonical signs;
- using the stored TASK-008 landmark chain directions produced a numerical
  guided residual no greater than 0.000002° on the valid guided segments;
- the all-28 native filmstrip showed readable, distinct silhouettes without
  dark skin deformation, hand overlap, camera movement, or left/right swap.

Therefore the primary defect was presentation retargeting of unsigned spread,
not the queue, labels, or the scientific data.

### 3. Source exemplar diagnosis

No canonical label was silently replaced.  The canonical samples are complete
in the available run: both hands are present on every frame, bend validity is
1.000 for every label, and sequence lengths are 19–26 frames.  The partial
spread masks and fallback clamp observations are reported explicitly in the
table.

Candidate pools were inspected for the flagged signs.  The top alternatives
for `د`, `ر`, `ض`, `ظ`, `غ`, and `ي` retained the same 0.750 spread-validity
limitation; the best `خ`, `ذ`, `ق`, and `ن` alternatives did not provide direct
RGB evidence of a better sign; and the tied complete candidates for `ه` and
`و` had no objective visual basis for choosing one over the canonical entry.
The current resolver path was therefore preserved.

No `core28_presentation_exemplars.json` file was added.  Exemplar selection is
about visual quality, not recognition confidence, and direct RGB inspection was
unavailable.  Making an arbitrary alternate choice would have been less
reproducible than keeping the existing deterministic catalog.

### 4. TASK-008 / TASK-005 comparison

The presentation solver reads, but does not write, the stored 21-point
landmarks, bend values, spread values, palm quaternion, and validity masks.
The source chain is the authoritative `kinematics.layout.FINGER_CHAINS` chain:

```text
wrist -> MCP/meta -> proximal -> middle -> distal/tip
```

The stored bend mask gates each phalange direction.  The relevant spread masks
gate each metacarpal direction.  If the required presentation input is invalid,
the solver keeps the last displayed bone transform or uses the original
channel fallback.  It does not invent a scientific value.

The audit snapshots all four scientific presentation inputs per frame and
confirms `source_arrays_unchanged: true` for all 28 sequences.

### 5. Clamp diagnosis

The conservative TASK-007G fallback limits were not globally loosened.  The
audit identified valid source values that would hit those limits, including
ring distal in `ا`, thumb proximal in `ه`, pinky distal in `و`, and several MCP
or DIP channels in `ض`.

The evidence-based correction is that a valid landmark-guided segment uses its
real stored direction rather than the old guessed spread/clamped fallback.  The
numeric profile limits remain available for missing/invalid landmark frames
and channel-only operation.  This avoids faking a sign to satisfy an
anatomical clamp while keeping a safe fallback for incomplete data.

### 6. Retargeting and additional degrees of freedom

The project-owned profile now declares the proper-rotation mapping from the
TASK-005/TASK-008 palm frame `[lateral, palmar_normal, distal]` to the canonical
GLB frame `[-lateral, distal, palmar_normal]`.

For each valid source segment, the solver:

1. computes the source segment direction from the stored landmark chain;
2. maps it into the canonical presentation frame;
3. applies the recorded wrist delta to the target shape;
4. computes a shortest world-space swing from the current rest-relative bone
   direction to the target direction;
5. solves the child-local delta from immutable GLB rest rotations;
6. applies `rest × delta` through the existing QML RuntimeLoader path.

This adds only presentation-side signed segment direction and thumb-opposition
information.  It does not add a scientific channel.  The 15 bends, 4 spreads,
and palm quaternion remain the frozen scientific inputs and recognition still
uses the original TASK-009A tensor path.

A full landmark-derived orientation frame was tested but rejected.  Its axial
roll was not observable reliably from a tracked segment direction and produced
visible skin twisting.  The final shortest swing preserves the authored
TASK-007G axial roll; no accumulated or incremental transform is used.

### 7. Blender MCP validation

Blender MCP inspected the currently open file:

```text
C:\Users\hatem\Desktop\GP\Graduation-Project-Simulation\assets-local\blendswap_hands_v1\blendswap_hands_v1_working.blend
```

The read-only inspection confirmed the trusted separate presentation roots,
`RIG_LEFT` / `RIG_RIGHT`, `HAND_LEFT` / `HAND_RIGHT`, skinning, and the
one-palm/22-bone derived hierarchy.  The derived rigs have no authored pose
constraints.  The temporary deformation experiment set one local-X bone at a
time from neutral and measured independent weighted mesh motion:

| Bone | Moved vertices | Maximum local displacement |
|---|---:|---:|
| `index_1` | 11,977 | 0.055447396 |
| `index_2` | 7,137 | 0.033459669 |
| `index_3` | 4,505 | 0.016018916 |
| `thumb_1` | 12,924 | 0.057941718 |
| `thumb_2` | 9,476 | 0.041975221 |
| `thumb_3` | 5,401 | 0.016927923 |

The pose snapshot was restored in `finally` to Blender float precision and the
scene was never saved.  No Blender asset redesign or export occurred in this
task.

### 8. `محمد` result

The native Windows run played four distinct queue events in logical Unicode
order: `م`, `ح`, `م`, `د`.  Each source sequence visited every anchor and its
last frame.  The measured target-boundary plans were:

| Boundary | Pose distance | Transition duration | Hold |
|---|---:|---:|---:|
| `م → ح` | 15.734° | 184.96 ms | 80 ms |
| `ح → م` | 16.787° | 187.30 ms | 80 ms |
| `م → د` | 23.854° | 203.01 ms | 80 ms |

The application transitioned directly between signs; it did not reset through
an open neutral pose.

### 9. Required transition matrix and repeated signs

Native Windows `ا → ب → م → د → ي → ح → ا` completed all seven items.  The
target-boundary durations were approximately 187.1, 188.6, 203.0, 213.2,
199.5, and 185.1 ms respectively, all inside the requested adaptive range.

`مم` and `اا` each remained two queue events.  Their direct same-sign
transitions were 1.748° / 153.88 ms and 4.752° / 160.56 ms, respectively, with
the same 80 ms hold.  Neither sequence visibly snapped back to neutral.

### 10. Recognition isolation

Recognition was not fed any interpolation, transition frame, landmark-guided
presentation pose, or QML bone state.  The headless deployment smoke loaded:

```text
deployment.pt
role: DEPLOYMENT MODEL
scope: All Core-28 signers / 4,222 training sequences
reference: LOSO reference only (not deployment accuracy):
           67.63% accuracy / 0.6607 macro F1
```

The queued `م` sample was predicted as `م` with confidence `0.9995226860` in
the existing recognition bridge.  The native GUI also reached
`Recognition ready` with the same deployment provenance.  The frozen 67.63%
LOSO result was not relabeled as deployment accuracy.

## Performance

Measured on native Windows 11 with Python 3.12.4, Qt Quick RHI Direct3D11,
and the RTX 2060 Super desktop:

| Scenario | Result |
|---|---:|
| no-text startup / idle smoke | 179 FPS |
| steady short two-sign playback smoke | 180 FPS |
| longer seven-sign transition matrix diagnostic interval | 68.9 FPS |
| Python presentation-solver audit throughput | approximately 199–271 frames/s for both hands, per sequence |
| persistent LEFT geometry creation | 1 per process |
| persistent RIGHT geometry creation | 1 per process |
| native graphics API | Qt Quick RHI · Direct3D11 |

The longer matrix metric is an interval diagnostic affected by process startup,
queue completion, and native smoke timing; the representative steady playback
run remained near 180 FPS.  CPU utilization, GPU utilization, and VRAM were
not instrumented.  The `surface_mode` diagnostic remains false when no local
MANO topology is supplied, but the normal user-facing path is the existing
high-detail skinned GLB surface, not the debug point representation.

## Tests

Passed:

- targeted TASK-007I, TASK-007G, and TASK-007H unittest suite: **40 tests**;
- `python -m compileall -q smart_glove_app scripts tests`;
- `git diff --check`;
- all-28 read-only source/direction/boundary audit: **28 PASS**, zero
  QUESTIONABLE, zero FAIL;
- native GUI startup, PALM view, BACK view, no-smoothing mode, all-28
  filmstrip, transition matrix, `محمد`, `مم`, and `اا`;
- headless and native GUI deployment-checkpoint smoke.

The full repository unittest discovery was also run.  It discovered 785 tests
and reported 736 passing, 29 skipped, 2 failures, and 18 errors.  The failures
and errors are outside the TASK-007I files and reproduce existing Windows/test
environment problems: locked temporary `.xlsx`/`.npz` cleanup files (`WinError
32`), the legacy TASK-007B CLI printing Arabic through a cp1252 console, and a
test requiring an unavailable external production virtual-glove artifact.  The
new and affected application suites pass independently; no unrelated baseline
test was changed to hide those failures.

## Failures / Limitations

- Original KArSL RGB videos were unavailable locally, so the report cannot
  claim direct RGB-to-render comparisons.  Supply `--rgb-root` on a machine
  containing the licensed source tree to extend the audit.
- Partial TASK-005 spread masks remain partial in the source data.  The
  presentation policy holds or falls back for those bases; it does not
  fabricate spread.
- The selected landmark direction provides reliable signed swing but not
  observable axial roll.  The authored GLB axial roll is intentionally kept.
- `--mano-model` is still optional for the separate MANO topology diagnostic;
  the normal application presentation remains the TASK-007G skinned GLB.
- Existing full-suite Windows cleanup/encoding/external-artifact issues remain
  outside this task.
- User review against the real KArSL RGB videos is still the correct final
  visual authority when those files can be supplied.

## Comparison

| Layer | TASK-007H / before | TASK-007I / after |
|---|---|---|
| Queue boundaries | Correct source-anchor traversal, but investigated again | Unchanged; all 28 terminal traces complete |
| Finger bend mapping | Correct shared flexion convention | Preserved; no global axis/sign rewrite |
| Spread mapping | Unsigned pairwise angles converted to guessed signed meta rotations | Stored landmark segment directions recover signed finger placement and thumb opposition in presentation only |
| Clamp behavior | Conservative fallback limits could compress valid source poses | Numeric limits unchanged; valid source-guided directions avoid false clipping, invalid data still falls back |
| Bone application | Absolute immutable rest × per-frame delta | Same; shortest swing preserves authored axial roll |
| Transition controller | 80 ms hold, direct adaptive SLERP transition | Unchanged and regression-tested |
| Renderer/asset | TASK-007G one-GLB-per-hand skin/camera/material | Unchanged |
| Scientific input | Frozen TASK-005/TASK-008 → TASK-009A path | Unchanged; source-array audit confirms no mutation |

## Recommendation

**KEEP** the landmark-guided presentation retargeting as the default when both
TASK-007G GLBs and valid TASK-008 landmarks are available.  Keep the original
channel solver as the documented fallback for missing assets or invalid source
geometry.

## Reproducibility

- Branch base: `32ab2f2ead0ab8a2e6da93b961a3bec40a9a252d`.
- Python: 3.12.4 in `.venv`.
- Qt runtime: PySide6 / Qt Quick RHI Direct3D11.
- GPU: NVIDIA RTX 2060 Super.
- Run root: `..\graduation-project-runs\task008-core28-full`.
- Checkpoint: external `task009c-core28-deployment\deployment.pt`.
- GUI rig asset: ignored `assets-local\blendswap_hands_v1`.
- Canonical resolver: existing `visualizer.catalog.core28_exemplars.json`.
- Audit script: `scripts/audit_task007i_sign_pose_fidelity.py`.
- Deterministic text tests: `محمد`, `ا → ب → م → د → ي → ح → ا`, `مم`, `اا`.
- No random exemplar mode was used for the all-28 audit.

## Next Steps

1. Repeat the all-28 comparison with the licensed KArSL RGB tree supplied via
   `--rgb-root` and review any remaining signer-specific disagreement.
2. If a future source-grounded review finds a better real exemplar, add an
   explicit presentation-only override file with its sample ID and evidence.
3. Keep sensor markers/panel work postponed until this sign-fidelity result is
   accepted.
4. Proceed to TASK-010A text integration only after user review.
