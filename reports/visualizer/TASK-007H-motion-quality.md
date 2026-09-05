# TASK-007H — Motion quality and sign-transition audit

## Task

Improve the motion quality of the Core-28 application after TASK-007G,
diagnose the reported mid-sign transition cut, verify every stored source
frame, and add presentation-only transitions without changing the scientific
pipeline.

**Branch:** `luna/task-007h-motion-quality`
**Base:** `3ce6cb9a205007abe82ea835932f577c2cdf34fa`
**Scope:** application playback, presentation motion, diagnostics and tests
only. TASK-004 through TASK-009C contracts and artifacts were not modified.

## Scope

The TASK-007G Qt Quick 3D renderer, one-GLB-per-hand asset arrangement,
canonical palm-facing frame, PALM/BACK nodes, camera/framing, natural-skin
material and absolute rest-pose bone application remain the reference
implementation. This task adds a boundary controller around that renderer; it
does not replace the renderer or rebuild the rig.

## Approach

### 1. Transition root cause

The source playback clock was not dropping frames. The cut was caused by the
queue boundary in the old application controller:

1. the terminal stored frame was published;
2. `_finish_current_if_needed()` advanced the queue immediately;
3. `_begin_current()` called `_reset_render_state()`;
4. that reset published the neutral QML pose before the next sequence had
   loaded.

The visible result was effectively:

```text
last pose A → neutral/reset → first pose B
```

This looked like a cut in the middle of a sign. It was not caused by a missing
TASK-008 frame, a recognizer callback, an Euler interpolation discontinuity, or
a camera movement.

The fix preserves the final absolute hand-pose snapshot while the next stored
sequence loads. Once the next first frame is solved, the application runs a
short direct A→B presentation transition. No neutral pose is inserted between
ordinary signs.

### 2. Source-frame boundary instrumentation

`PlaybackBoundaryTrace` records, per sign:

- source sequence length and source frame indices;
- unique source anchor positions published in order;
- first and final source frame displayed;
- whether every source position was visited;
- whether the terminal frame was published before queue advance;
- adaptive transition distance/duration for the following sign.

If a GUI timer callback lands late, the controller publishes crossed source
anchors in order before the current interpolated display state. This makes
source visitation observable without altering the timestamp clock or the
scientific arrays.

The trace is presentation diagnostics only. It is not passed to
TASK-009A/TASK-009B.

### 3. Presentation-only transition algorithm

`PresentationTransition` receives two immutable solved `HandPose` maps:

```text
final pose A → boundary hold → smoothstep(SLERP(A, B)) → first pose B
```

- quaternion rotations use shortest-arc SLERP;
- the scalar blend parameter uses cubic `smoothstep` easing;
- endpoint poses are snapshotted and never mutated;
- every transition sample is evaluated from those endpoints, so no displayed
  pose accumulates into the next pose;
- a slow worker load cannot consume the whole transition: if the target loads
  after the original hold window, the full blend begins when the target is
  available.

The default presentation policy is configurable from the CLI:

| Setting | Default |
|---|---:|
| Boundary final-pose hold | 80 ms |
| Minimum transition | 150 ms |
| Maximum transition | 350 ms |

The duration is adaptive. The metric is a weighted average of shortest
quaternion angular differences over both hands: wrist weight `2.0`, spread
metacarpals weight `1.25`, and the 15 phalange joints weight `1.0`. Layout/root
transforms are excluded. A 0° distance gives the minimum duration; 90° or more
reaches the maximum, with linear timing-policy interpolation between those
limits. The metric is used only for display timing.

### 4. Motion and recognition separation

Stored sequences continue to enter the existing `PersistentPlaybackController`
and TASK-009 recognition bridge unchanged. Rendering interpolation and
boundary poses operate only on solved `HandPose` objects. No transition sample
is written into TASK-008, sent to TASK-009A, or passed to the LSTM.

Invalid bend/spread values continue to follow the existing presentation policy
(hold the last valid joint value or use the declared neutral presentation
value). The scientific validity mask and raw value remain untouched.

## Evidence / Sources

- TASK-007G acceptance report:
  `reports/visualizer/TASK-007G-visual-acceptance.md`.
- Current TASK-007G presentation profile:
  `smart_glove_app/assets/rig_profiles/task007g_hands.json`.
- Existing canonical resolver/catalog and TASK-008 loader.
- Read-only Blender MCP inspection of the open hand scene. No Blender scene
  changes were made for TASK-007H. The inspection confirmed the authored
  left/right rig and mesh structure used by the TASK-007G asset; Blender was
  used as a diagnostic reference, not as a runtime dependency.
- Source asset attribution retained from TASK-007G: BlendSwap “Hands Rigged”,
  SparrowHawk, BlendSwap asset 22269, CC0. The source `.blend` remains under
  `assets-local/` and is not re-licensed or copied into tracked application
  code.

## Files Changed

- `smart_glove_app/app/motion_quality.py` — immutable endpoint snapshots,
  weighted distance, adaptive policy, smoothstep and quaternion transition
  sampler.
- `smart_glove_app/app/playback_controller.py` — source-frame/queue-boundary
  trace.
- `smart_glove_app/app/application_controller.py` — final-pose preservation,
  transition lifecycle, source-anchor catch-up and diagnostics.
- `smart_glove_app/app/main.py` — transition CLI settings and JSON metrics.
- `scripts/audit_task007h_motion_quality.py` — read-only all-28 source and
  presentation audit.
- `tests/test_task007h_motion_quality.py` — transition, boundary, isolation,
  repeated-character, 15-bend and 4-spread tests.
- `reports/visualizer/TASK-007H-motion-quality.md` — this report.

No QML scene, GLB, camera, material, rig profile or scientific artifact was
changed in this task.

## How to Run

PowerShell, visualizer-only mode:

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

Deployment checkpoint mode:

```powershell
$checkpoint = Resolve-Path "..\graduation-project-runs\task009c-core28-deployment\deployment.pt"

.venv\Scripts\python.exe scripts\run_core28_application.py `
  --run-root $runRoot `
  --rig-asset $rigAsset `
  --checkpoint $checkpoint `
  --text "م"
```

All-28 read-only audit:

```powershell
.venv\Scripts\python.exe scripts\audit_task007h_motion_quality.py `
  --run-root $runRoot
```

The optional `--no-smooth-rendering` flag still displays exact stored source
anchors; it does not disable the separate sign-boundary policy.

## Evaluation

### Boundary trace result

The all-28 audit was run with the same timestamp-aware playback controller at
60 Hz:

```text
canonical_entry_count: 28
all_boundary_traces_complete: true
source lengths: 19–26 frames
source durations: 0.600–0.833 seconds
source indices: contiguous 0..N-1 for every sign
```

The native Windows transition matrix `ا → ب → م → د → ي → ح → ا` also
reported every anchor for every completed item, `LAST FRAME PRESENTED: YES`,
and `early_queue_advance: false`. No worker callback or recognition result
changed the source trace.

### All-28 canonical audit

Every row uses the existing deterministic canonical resolver path. `Both` is
the number of frames with both stored hands present. Bend/spread validity is
the fraction of stored validity-mask entries, not a fabricated presentation
value. The final `Trace` column means the read-only controller visited every
source index and presented the terminal frame before queue advance.

| SignID | Character | Canonical sample / signer | Frames | Duration | Both | Bend valid | Spread valid | Max wrist | Bend clamp observations | Trace |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 0032 | ا | `karsl_core28_s03_sign0032_train_rep019` / 03 | 25 | 0.800 s | 25 | 1.000 | 1.000 | 2.04° | ring distal ×19 | PASS |
| 0033 | ب | `karsl_core28_s01_sign0033_train_rep022` / 01 | 26 | 0.833 s | 26 | 1.000 | 1.000 | 4.01° | — | PASS |
| 0034 | ت | `karsl_core28_s01_sign0034_train_rep010` / 01 | 19 | 0.600 s | 19 | 1.000 | 1.000 | 3.71° | — | PASS |
| 0035 | ث | `karsl_core28_s01_sign0035_train_rep019` / 01 | 19 | 0.600 s | 19 | 1.000 | 1.000 | 2.77° | — | PASS |
| 0036 | ج | `karsl_core28_s01_sign0036_test_rep007` / 01 | 20 | 0.633 s | 20 | 1.000 | 1.000 | 6.14° | — | PASS |
| 0037 | ح | `karsl_core28_s01_sign0037_train_rep026` / 01 | 20 | 0.633 s | 20 | 1.000 | 1.000 | 3.59° | — | PASS |
| 0038 | خ | `karsl_core28_s03_sign0038_train_rep042` / 03 | 22 | 0.700 s | 22 | 1.000 | 0.966 | 10.59° | — | PASS |
| 0039 | د | `karsl_core28_s03_sign0039_test_rep001` / 03 | 21 | 0.667 s | 21 | 1.000 | 0.750 | 1.87° | ring MCP ×21; pinky MCP ×21 | PASS |
| 0040 | ذ | `karsl_core28_s03_sign0040_train_rep004` / 03 | 19 | 0.600 s | 19 | 1.000 | 0.993 | 5.59° | — | PASS |
| 0041 | ر | `karsl_core28_s01_sign0041_test_rep006` / 01 | 19 | 0.600 s | 19 | 1.000 | 0.750 | 1.49° | pinky MCP ×19 | PASS |
| 0042 | ز | `karsl_core28_s01_sign0042_test_rep003` / 01 | 21 | 0.667 s | 21 | 1.000 | 1.000 | 1.94° | pinky MCP ×15 | PASS |
| 0043 | س | `karsl_core28_s01_sign0043_test_rep003` / 01 | 20 | 0.633 s | 20 | 1.000 | 1.000 | 3.03° | — | PASS |
| 0044 | ش | `karsl_core28_s01_sign0044_train_rep007` / 01 | 19 | 0.600 s | 19 | 1.000 | 1.000 | 2.66° | — | PASS |
| 0045 | ص | `karsl_core28_s03_sign0045_train_rep014` / 03 | 23 | 0.733 s | 23 | 1.000 | 1.000 | 10.06° | — | PASS |
| 0046 | ض | `karsl_core28_s01_sign0046_train_rep041` / 01 | 20 | 0.633 s | 20 | 1.000 | 0.750 | 3.92° | index distal ×2; middle MCP ×20; middle distal ×20; ring MCP ×20; pinky MCP ×14 | PASS |
| 0047 | ط | `karsl_core28_s03_sign0047_train_rep016` / 03 | 19 | 0.600 s | 19 | 1.000 | 1.000 | 23.00° | —; wrist near 25° cap | PASS |
| 0048 | ظ | `karsl_core28_s01_sign0048_test_rep004` / 01 | 20 | 0.633 s | 20 | 1.000 | 0.750 | 2.24° | — | PASS |
| 0049 | ع | `karsl_core28_s01_sign0049_train_rep013` / 01 | 20 | 0.633 s | 20 | 1.000 | 1.000 | 3.02° | pinky MCP ×11 | PASS |
| 0050 | غ | `karsl_core28_s01_sign0050_train_rep003` / 01 | 20 | 0.633 s | 20 | 1.000 | 0.750 | 2.90° | ring MCP ×2; pinky MCP ×20 | PASS |
| 0051 | ف | `karsl_core28_s03_sign0051_test_rep006` / 03 | 24 | 0.767 s | 24 | 1.000 | 1.000 | 5.95° | — | PASS |
| 0052 | ق | `karsl_core28_s03_sign0052_train_rep002` / 03 | 19 | 0.600 s | 19 | 1.000 | 0.862 | 2.52° | — | PASS |
| 0053 | ك | `karsl_core28_s01_sign0053_test_rep001` / 01 | 21 | 0.667 s | 21 | 1.000 | 1.000 | 2.41° | — | PASS |
| 0054 | ل | `karsl_core28_s01_sign0054_train_rep005` / 01 | 20 | 0.633 s | 20 | 1.000 | 1.000 | 1.44° | — | PASS |
| 0055 | م | `karsl_core28_s01_sign0055_train_rep010` / 01 | 20 | 0.633 s | 20 | 1.000 | 1.000 | 2.96° | — | PASS |
| 0056 | ن | `karsl_core28_s01_sign0056_train_rep026` / 01 | 19 | 0.600 s | 19 | 1.000 | 0.961 | 4.54° | — | PASS |
| 0057 | ه | `karsl_core28_s01_sign0057_train_rep003` / 01 | 20 | 0.633 s | 20 | 1.000 | 1.000 | 3.70° | thumb proximal ×20 | PASS |
| 0058 | و | `karsl_core28_s01_sign0058_test_rep002` / 01 | 20 | 0.633 s | 20 | 1.000 | 1.000 | 2.02° | pinky distal ×20 | PASS |
| 0059 | ي | `karsl_core28_s01_sign0059_test_rep008` / 01 | 21 | 0.667 s | 21 | 1.000 | 0.750 | 3.81° | middle MCP ×21 | PASS |

## Results

### Suspicious letters and layer diagnosis

The all-28 screenshot sequence was inspected as a visual audit. No letter
showed evidence of a swapped LEFT/RIGHT hand, a cross-wired finger, a
mirrored-culling defect, camera movement, hand overlap or rest-pose
accumulation. The following letters were flagged for review because the
source values or validity masks create presentation risk. They are not
silently “corrected” by changing the rig.

| Letters | Observed risk | Diagnosed layer and treatment |
|---|---|---|
| ا | Ring distal bend is above the 70° authored DIP limit in all 19 frames. | TASK-005-derived valid value reaches the presentation clamp; the channel is still the ring distal channel. No mapping change. |
| خ, ذ, ق, ن | Spread validity is 96.6%, 99.3%, 86.2% and 96.1%. | Source validity masks are partial. Presentation holds/uses the declared presentation state; it does not fabricate or rewrite scientific spreads. |
| د | Ring/pinky MCP values exceed 90° in every frame; spread validity is 75%. | Source kinematics plus presentation anatomical limit and source mask. This is not a missing frame or a wrong finger mapping. |
| ر | Pinky MCP exceeds 90° in every frame; spread validity is 75%. | Same source-value/clamp and validity-mask diagnosis. |
| ز | Pinky MCP exceeds 90° in 15 frames. | Valid source channel saturates the authored presentation limit. |
| ض | 76 total bend clamp observations across index distal, middle MCP/distal, ring MCP and pinky MCP; spread validity is 75%. | Source pose is outside the conservative visual joint limits. The rendered clipping is a presentation limitation, not evidence to fake a different sign. |
| ط | Wrist reaches 22.995°, close to the 25° wrist cap. | Source palm quaternion is unusually large but remains inside the explicit presentation bound; no clamp or retargeting change was justified. |
| ظ, غ, ي | Spread validity is 75%. | Source validity-mask limitation; no sign-specific rig edit. |
| ع | Pinky MCP exceeds 90° in 11 frames. | Valid source bend saturates the presentation limit. |
| ه | Thumb proximal bend exceeds the 55° thumb limit in all 20 frames. | Source value exceeds the conservative thumb presentation range. |
| و | Pinky distal bend exceeds the 70° DIP limit in all 20 frames. | Source value exceeds the conservative presentation range. |
| م | No clamp or validity issue; used for transition/recognition checks. | No defect found in source, kinematics or rig layer. |

This separates “the pose may look compressed because a valid source value is
outside a visual limit” from “the renderer applied the wrong joint”. The
automated bend-isolation tests exercise each of the 15 independent channels,
and the spread-isolation tests exercise each of the four spread inputs.

### Exemplar selection

No `core28_presentation_exemplars.json` override was added. The existing
canonical/default exemplar path remains authoritative for the GUI demo.

Candidate sequences for the most suspicious labels (`ر`, `ظ`, `ض`, `ط`, `ي`)
were compared using the available catalog score, duration and validity. The
short `ر` candidate with a better spread fraction was only 11 frames / 0.333 s
and scored below the canonical 19-frame sample; `ظ` and `ي` candidates shared
the same 75% spread-validity limitation; the canonical `ض` candidate was the
highest-ranked option; and `ط` alternatives retained the same source wrist
behavior. None was materially better without making a presentation-only
choice that would be harder to reproduce. Therefore no silent substitution
was made.

### Rig/retargeter changes

There were no TASK-007G rig or retargeting changes. The existing absolute
solver remains:

```text
immutable TASK-007G rest pose × current solved presentation delta
```

The new code only snapshots those solved endpoint maps and blends them during
the queue boundary. It does not mutate bone names, GLB geometry, camera,
PALM/BACK nodes, material or presentation placement.

### Required transition matrix

Native Windows screenshots were captured from the real Qt Quick 3D application
for `ا → ب → م → د → ي → ح → ا`. The visible hands remained separated and
the view stayed fixed. The completed source traces were all full. Observed
adaptive plans were:

| Boundary | Distance | Duration | Hold |
|---|---:|---:|---:|
| ا → ب | 14.513° | 182.25 ms | 80 ms |
| ب → م | 15.614° | 184.70 ms | 80 ms |
| م → د | 20.217° | 194.93 ms | 80 ms |
| د → ي | 24.093° | 203.54 ms | 80 ms |
| ي → ح | 20.598° | 195.77 ms | 80 ms |
| ح → ا | 15.508° | 184.46 ms | 80 ms |

### `محمد`

The native queue test preserved Unicode logical order and played four separate
events: `م`, `ح`, `م`, `د`. All four stored sequences completed with full
source traces. The three boundary plans were:

| Boundary | Distance | Duration |
|---|---:|---:|
| م → ح | 15.005° | 183.34 ms |
| ح → م | 16.440° | 186.53 ms |
| م → د | 20.217° | 194.93 ms |

The application did not return to a fully open neutral hand between these
events.

### Repeated letters

`مم` and `اا` each produced two queue items with the same deterministic sample
ID, not one coalesced item. Their measured transitions were respectively
1.510° / 153.36 ms and 4.408° / 159.80 ms, with the same 80 ms boundary hold.
The small non-zero distance is the direct final-pose-to-first-pose motion of
the same stored sign; it is not a neutral reset.

### Recognition isolation and checkpoint

The native GUI loaded the deployment checkpoint with the expected provenance:

```text
role: DEPLOYMENT MODEL
scope: All Core-28 signers / 4,222 training sequences
scientific reference: LOSO reference only — 67.63% accuracy / 0.6607 macro F1
graphics API: Qt Quick RHI · Direct3D11
```

The headless recognition bridge then predicted the exact queued `م` sample as
`م` with 99.952% confidence. Recognition receives the queue item and its
original stored arrays; it has no parameter or call path accepting a
`PresentationTransition`. The GUI motion trace remained complete in the
checkpoint run, and recognition callbacks are token-checked so stale results
cannot alter playback.

## Performance

Measured on native Windows 11 with the project Python 3.12 virtual
environment and the current RTX 2060 Super desktop:

| Scenario | Observed diagnostic |
|---|---:|
| Idle application | 180 FPS Qt `FrameAnimation` window |
| Continuous all-28 playback | 173 FPS window metric in the 7.5 s matrix run |
| Short startup playback sample | 56 FPS while the 1.4 s process included startup/asset load |
| Stored sequence rate | 30 FPS for the representative TASK-008 sequences |
| Qt graphics API | Direct3D11 via Qt Quick RHI |
| LEFT geometry creation | 1 per process |
| RIGHT geometry creation | 1 per process |

The 173–180 FPS figure is an application diagnostic, not a claim that the
monitor refreshes at that rate. CPU utilization, GPU utilization and memory
were not collected. The runtime normal path uses the sculpted skinned GLB
surface from TASK-007G. `surface_mode=false` in the printed metrics only means
that no user-supplied MANO topology `.pkl` was present for the optional debug
MANO-point path; it does not replace the normal GLB hands with spheres or
points.

## Tests

Passed:

- `python -m unittest tests.test_task007h_motion_quality` — 10 tests.
- `python -m unittest tests.test_task007f_application_v2 tests.test_task007g_presentation tests.test_task007h_motion_quality` — 48 tests.
- `python -m compileall -q smart_glove_app scripts tests`.
- Native Windows GUI smoke: idle, `اب`, all 28 canonical characters,
  `ا → ب → م → د → ي → ح → ا`, `محمد`, `مم`, `اا`, and deployment-checkpoint
  loading.
- All-28 read-only boundary audit — 28/28 complete traces.
- Added tests cover immutable transitions, source terminal-frame ordering,
  repeated Unicode events, recognition separation, all 15 independent bends
  and all 4 spread inputs.

The repository-wide unittest discovery run executed 780 tests, with 29
skipped and 20 failures/errors already outside this task: two existing
environment-dependent failures and 18 native-Windows temporary-file lock
teardown errors in older TASK-008/TASK-009 tests. None is in the TASK-007H
tests or in the TASK-007F/TASK-007G regression subset above. `pytest` is not
installed in this venv, so unittest was used for the repository baseline.

## Failures / Limitations

- No locally licensed MANO topology file was supplied during the Windows
  smoke. The normal application still renders the TASK-007G skinned GLB
  surface; the optional topology/point diagnostics remain governed by the
  existing fallback warning.
- The all-28 audit identified source poses that saturate conservative visual
  joint limits and source spread masks that are not complete. Those are
  explicitly reported rather than hidden. A future product decision can
  review the visual limits or choose a presentation exemplar, but changing
  them globally would risk falsifying signs.
- Screenshot review was visual and deterministic, not a pixel-level automated
  sign-quality classifier.
- CPU/GPU utilization and memory were not instrumented.
- NLP/TASK-010A and TTS/TASK-011A were not started.

## Comparison

| Behavior | Before TASK-007H | After TASK-007H |
|---|---|---|
| Ordinary sign boundary | Reset to neutral while next sequence loaded | Final pose held, then direct A→B transition |
| Source frame evidence | No boundary trace | Every source position and terminal frame traced |
| Inter-sign rotation | Potential visual cut | Quaternion SLERP with eased adaptive duration |
| Repeated characters | Queue semantics existed but boundary could reset | Two events retained, direct small transition |
| Scientific arrays | Existing | Unchanged |
| TASK-007G renderer | Reference | Preserved |

## Recommendation

**KEEP** — the motion-quality implementation is ready for user review. The
remaining source-pose clamp/mask observations are documented data/presentation
limitations, not evidence for a renderer rewrite or a scientific-contract
change.

## Reproducibility

- Base commit: `3ce6cb9a205007abe82ea835932f577c2cdf34fa`.
- Branch: `luna/task-007h-motion-quality`.
- Platform: native Windows 11, Python 3.12.4 venv, RTX 2060 Super;
  PySide6 6.11.2, NumPy 2.5.2, PyTorch 2.14.0+cu132, CUDA available.
- Canonical source: existing `karsl_core28` labels/catalog and external
  `task008-core28-full` run root.
- Checkpoint: existing external `deployment.pt`; it was loaded but not
  modified.
- Transition defaults: hold 80 ms, range 150–350 ms, distance saturation 90°.
- No random exemplar mode was used for the audit; all GUI letter tests used
  the deterministic canonical/default selection path.
- Temporary screenshot sequences were written outside the repository under
  `C:\Users\hatem\AppData\Local\Temp\task007h-*` and are not committed.

## Next Steps

- Obtain and configure a locally licensed MANO model only if the optional
  topology/debug surface is needed.
- Review the explicitly flagged clamp/mask letters with the project owner
  before changing presentation limits or adding a presentation-only exemplar
  override.
- Integrate the queued logical text field with TASK-010A later.
- Integrate TTS with TASK-011A later.

TASK-007H does not start either downstream task.
