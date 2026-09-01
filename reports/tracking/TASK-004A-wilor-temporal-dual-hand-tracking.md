# Objective

Convert raw, independent-per-frame WiLoR detections into a deterministic
temporal representation of the **physical LEFT hand** and **physical RIGHT
hand** across a video, stable through hand approach, crossing, overlap,
short disappearance, duplicate detections, WiLoR's occasional third-hand
false positive, degenerate reconstructions and detector handedness
disagreement.

This stage establishes **temporal identity plus quality/state metadata
only**. It deliberately computes no MCP/PIP/DIP flexion, no abduction, no
simulated Hall or IMU values and no LSTM features — those belong to
TASK-005 and later.

Raw WiLoR output is immutable. Tracking writes a **new derived stage**
(`tracked/`) beside the raw stage (`raw/`) and never rewrites, smooths,
interpolates, fabricates or deletes raw data. Immutability is asserted in
the test suite and re-verified by SHA-256 on the pilot run below.

Run date: 2026-09-02.

# Inputs

| Input | Value |
|---|---|
| Branch | `opus/task-004a-temporal-hand-tracking`, worktree `../Graduation-Project-Simulation-task004a`, created from `origin/main` (`b35f616`) |
| Raw WiLoR run | `/home/hatim/graduation-project-runs/wilor_karsl_pilot_full` (explicit path; `runs/wilor*` is never globbed) |
| Manifest | `datasets/manifests/karsl_milestone1_pilot.csv`, SHA-256 `4a8e0d24f587d9c54ad87ebd896c6df0290b3ebc80885774100d71b29dbc3c0c` |
| Pilot | 18 videos, 894 frames, 1,779 reconstructed rows, signs 0171-0176, signers 01-03 |
| Tracked output | `/home/hatim/graduation-project-runs/wilor_karsl_pilot_tracked` (ignored, not committed) |
| Debug overlays | `/home/hatim/graduation-project-runs/wilor_karsl_pilot_tracked/debug` (ignored) |
| Tracker config | `configs/tracking/wilor_tracker.json` |

The loader hard-refuses any run whose `summary.json` or per-row metadata is
not exact `mode="full"`, so the historical detector-only Phase-A output can
never be tracked by accident. WiLoR was not re-run; no MediaPipe data is
used anywhere in this task.

# Tracker Design

Two canonical tracks exist for the whole video: `LEFT` and `RIGHT`, meaning
**subject-anatomical** identity. Per frame the tracker runs a fixed,
deterministic pipeline:

1. **Quality gate** every raw detection — mark, never fabricate.
2. **Same-label duplicate suppression** — collapse near-duplicate boxes that
   share a detector label.
3. **Exact 2 x N assignment** of the two tracks to the surviving candidates.
4. **Bind** assignments; classify every unbound track.
5. **Record** extra/rejected detections, ambiguity, label disagreement and
   reassociation evidence.

Detector handedness is treated as **evidence with a cost, never as an
override**. TASK-003B found the two extractors disagreeing on handedness
during occlusion and MediaPipe flickering its label on a stationary hand, so
a label alone is not trusted to define identity.

Every threshold was calibrated against the 1,779 real reconstructed rows
rather than guessed; the measured distribution behind each is quoted in
`tracking/wilor/config.py` and repeated below. All positions are
image-normalized (pixel / frame dimension), so thresholds are
resolution-independent.

# Association Cost / State Machine

## Cost function

For track `T` (with a remembered last centre, last box scale and staleness)
and candidate detection `D`:

```text
cost(T, D) = 0.55 * position + 0.25 * label + 0.10 * scale + 0.10 * confidence
```

| Term | Definition | Range |
|---|---|---|
| `position` | normalized bbox-centre distance / gate radius (0.5 for an uninitialized track) | [0, 1] |
| `label` | 1.0 if the detector label contradicts `T`; 0.5 if the label is unusable; else 0 | {0, 0.5, 1} |
| `scale` | `min(1, abs(log(size_D / size_T)) / log 2)` | [0, 1] |
| `confidence` | `1 - detector_confidence` | [0, 1] |

Weights sum to 1.0 (validated at construction), so the total cost is
directly interpretable.

## Gating

A pair is **infeasible** when the normalized centre distance exceeds the
track's gate:

```text
gate(T) = min(0.30, 0.08 * (1 + 0.5 * frames_since_observed))
```

The 0.08 base comes from the measured frame-to-frame normalized bbox-centre
displacement of a same-label hand: p50 0.0015, p95 0.0153, p99 0.0262,
p99.9 0.0434, **max 0.0535** — so the gate is roughly 1.5x the largest real
motion observed in the pilot. The gate widens for a stale track (a hand may
travel while unobserved) but is capped so a long-absent track can never
capture an arbitrarily distant detection.

## Assignment

With exactly two tracks the problem is 2 x N. All feasible injections are
enumerated (including the single-track and empty cases), which is exact —
identical to Hungarian at this size — deterministic and free of any solver
dependency. Preference order: bind more tracks first, then lower total cost;
ties resolve lexicographically so repeated runs are byte-identical.

## Ambiguity

Two independent triggers, either of which flags the bound tracks
`AMBIGUOUS`:

* **cost margin** — the best and second-best assignments of the same
  cardinality differ by less than `0.10`;
* **spatial proximity** — the two bound candidates are closer than `0.02`
  normalized units, i.e. physically on top of each other and unresolvable
  regardless of cost separation.

## State machine

```text
                 +-------------------- reassociation ------------------+
                 v                                                     |
   [start] -> OBSERVED <-> AMBIGUOUS                                   |
                 |  \                                                  |
                 |   +--> (unbound, gated-out detection nearby) --> REJECTED_QUALITY
                 |   +--> (unbound, other hand observed & near) --> LIKELY_OCCLUDED --+
                 +-------> (unbound, no other evidence) ------------> MISSING --------+
```

Precedence for an unbound track is fixed: `REJECTED_QUALITY` (a detection
was here but was not trusted) > `LIKELY_OCCLUDED` (heuristic) > `MISSING`
(no evidence either way).

## Seeding

The first usable frame seeds both tracks from distinct detector labels. If
both detections carry the *same* label, seeding falls back to the image
convention TASK-003A verified empirically (subject RIGHT hand at the
smaller image x in ~95-97% of two-hand frames for this facing,
non-mirrored capture), and records a `seed_by_position` event with its
basis. The fallback is used for seeding only and is never presented as
ground truth. It did not trigger on this pilot.

# Track Schema

Output per video, in a **new** directory beside the raw stage:

```text
<out>/<sample_id>/wilor_tracked.npz
<out>/<sample_id>/wilor_tracked_meta.json
```

The NPZ contains plain numeric arrays indexed `[frame, track]` with track
order `("left", "right")` — no pickled Python objects. Only inherently
variable-length fields are JSON strings, matching the convention already
used by `pose/wilor/npz_io.py`.

| Array | Shape | Notes |
|---|---|---|
| `frame_index`, `timestamp_seconds` | `[F]` | carried from raw, unmodified |
| `state_code` | `[F,2]` int8 | 0 MISSING, 1 OBSERVED, 2 AMBIGUOUS, 3 REJECTED_QUALITY, 4 LIKELY_OCCLUDED |
| `raw_detection_index` | `[F,2]` int16 | **provenance** back to the raw row; `-1` when none |
| `detector_label_code` | `[F,2]` int8 | original detector handedness (0 left, 1 right, -1 none) — never destroyed |
| `detector_confidence` | `[F,2]` float32 | raw detector confidence |
| `assignment_cost` | `[F,2]` float32 | this binding's cost |
| `assignment_margin` | `[F]` float32 | best vs second-best total cost |
| `landmarks_3d` | `[F,2,21,3]` float32 | NaN when no pose was observed |
| `hand_pose_rotmat` | `[F,2,15,3,3]` float32 | MANO local joint rotations |
| `global_orient_rotmat` | `[F,2,3,3]` float32 | wrist/root orientation |
| `betas` | `[F,2,10]` float32 | MANO shape |
| `camera_translation` | `[F,2,3]` float32 | camera-space placement |
| `box_center_xy`, `box_size` | `[F,2,2]`, `[F,2]` float32 | detector box |
| `quality_flags_json` | `[F,2]` U256 | raw flags + gate flags for that hand |
| `number_of_raw_detections` | `[F]` int16 | frame-level |
| `extra_detection_count` | `[F]` int16 | candidates that no track could take |
| `rejected_detections_json` | `[F]` U512 | indices + a reason per rejected raw detection |
| `tracking_flags_json` | `[F]` U256 | frame-level tracking flags |

`wilor_tracked_meta.json` carries the schema version, track order, state and
label code tables, the exact tracker config, the source raw path **with its
SHA-256**, the event log (reassociation / handedness disagreement / seeding)
and the per-video metrics.

Crucially, an `OBSERVED` entry keeps `raw_detection_index` **and** the
original `detector_label`, so the exact raw row is always recoverable and
the tracker's identity decision can always be audited against what the
detector originally said.

# Missing / Occlusion Policy

When a hand disappears the tracker emits state + evidence and **no pose**:

* `landmarks_3d`, MANO fields and camera translation are `NaN`;
* `raw_detection_index` is `-1`;
* nothing is interpolated, copied from a neighbouring frame, or synthesized.

`LIKELY_OCCLUDED` is an explicitly heuristic refinement of `MISSING`,
applied only when **all** of the following hold: the other track is observed
this frame, this track has been observed before, its last known centre lies
within `0.10` normalized units of the other hand's current centre, and it
has been absent for at most 15 frames. The measured distance is stored in
the hand's `quality_flags` (e.g. `heuristic_occlusion_distance=0.0107`). It
is a suggestion, never a claim of ground truth, and not every missing hand
is labelled occluded.

Downstream stages may later choose masking, padding, interpolation or
confidence gating. TASK-004A deliberately preserves the truth that **no
pose was observed**, so that choice remains open and auditable.

# Quality Gate

The gate is intentionally conservative: it **marks** and explains, and never
fabricates a replacement hand. The raw row always remains recoverable.

| Check | Threshold | Measured pilot range | Action |
|---|---|---|---|
| Non-finite joints / MANO / camera | any | none observed | reject |
| Detector confidence | `>= 0.30` | min 0.302, p1 0.588, p50 0.844 | reject |
| Joint-span / palm-length ratio | `[1.0, 3.0]` | min 1.28, p1 1.37, p99 2.28, max 2.34 | reject |
| Projected-centroid vs bbox-centre offset | `<= 0.35` box sizes | max 0.151 | reject |
| Global-orientation geodesic jump | `> 60 deg` | p50 1.8, p95 16.5, p99 38.1, max 102.2 | **flag only** |

The detector's own operating point (confidence 0.3) is fixed by TASK-002 and
TASK-003B and was **not** retuned here; the confidence floor is defensive
only.

**Honest finding, and the most important design lesson of this task.**
TASK-003B warned that MANO's parametric shape model constrains bone lengths,
so bone consistency alone cannot detect a bad pose. The pilot data confirms
this far more strongly than expected:

* palm length across all 1,779 rows spans only **0.0941-0.0970**;
* the projected joints fall inside the detector box for **100%** of rows;
* the documented spurious third detection (frame 39 of
  `karsl_test_s02_sign0176_repfirst`) has **entirely normal** intrinsic
  geometry — palm 0.0948, span/palm 1.94, projection offset 0.110.

So intrinsic geometry checks fire essentially never on WiLoR output, and
they did not remove the known bad detection. They are retained as a genuine
safety net for gross collapse (and unit tests prove they fire on collapsed
geometry), but **the mechanism that actually resolves WiLoR's real
failure mode is duplicate suppression plus the two-track capacity limit**,
not the geometry gate. Reporting otherwise would overstate the gate.

Consequently, on this pilot the gate rejected **0** detections. That is a
result, not an omission: no detection in the validated full-mode run was
independently implausible.

# Extra-Detection Handling

Three complementary mechanisms, in order:

1. **Same-label duplicate suppression.** Two detections that share a
   detector label and overlap at IoU >= 0.5 collapse to the higher-confidence
   one. Suppression is restricted to *same-label* pairs deliberately: the two
   genuinely different hands in this pilot reach a bbox IoU of **p90 0.688 /
   max 0.924**, so a label-agnostic NMS would delete real hands. Among
   same-label pairs, exactly one exceeded the threshold in the whole pilot —
   the known frame-39 duplicate at IoU 0.747.
2. **Two-track capacity.** Only two tracks exist, so a third candidate can
   never become a permanent identity even if it survives step 1; it is left
   unassigned and recorded as `extra_detection_count`.
3. **Full recording.** Every removed candidate is listed in
   `rejected_detections_json` with an explicit reason
   (`duplicate_same_label_iou=0.747>=(0.5)_of_detection_1`, or
   `unassigned_extra_detection`, or `quality:...`), the frame is flagged
   `EXTRA_DETECTIONS`, and `number_of_raw_detections` preserves the original
   count. **Nothing is deleted from the raw data.**

# Pilot Results

18 videos, 894 frames, run with `configs/tracking/wilor_tracker.json`.

| Metric | Value |
|---|---:|
| Videos / total frames | 18 / 894 |
| Observed LEFT frames | 894 (100.000%) |
| Observed RIGHT frames | 884 (98.881%) |
| Missing LEFT frames | 0 |
| Missing RIGHT frames | 10 (all `LIKELY_OCCLUDED`) |
| Frames with both tracks | 884 (98.881%) |
| Frames with no track | 0 (0.000%) |
| Ambiguous frames | 2 (0.224%) |
| Quality-rejected detections | 0 |
| Frames with rejected detections | 1 |
| Duplicate-suppressed detections | 1 |
| Frames with > 2 raw detections | 1 |
| Extra detections left unassigned | 0 |
| Handedness-disagreement events | 0 |
| Both-labels-swapped frames | 0 |
| Reassociation events | 1 |
| Suspected identity-switch events | 0 |
| Longest LEFT missing run (max over videos) | 0 |
| Longest RIGHT missing run (max over videos) | 10 |

Per-video, 17 of 18 clips produced both tracks on every frame. The single
exception is `karsl_test_s01_sign0174_repfirst` (34 frames: LEFT 34, RIGHT
24). Tracked dual-hand coverage therefore equals the raw dual-hand coverage
(884/894) — the tracker neither lost a usable hand nor invented one.

**Assignment margins.** Only 297 of 894 frames had any competing feasible
assignment at all; in the other 597 the swapped assignment was outside the
gate entirely (the strongest form of non-ambiguity). Across those 297, the
margin ranged 0.2815-1.5791 with p50 1.3346, and **no frame** fell below the
0.10 cost-margin threshold. Both ambiguous frames were therefore raised by
the *spatial-proximity* rule, not the cost margin — a useful signal that on
this data the cost function separates the two hands cleanly except when they
are physically coincident.

**Raw immutability verified.** The SHA-256 of every source
`wilor_raw.npz` recorded before tracking matches the file after the full
run: **18/18 unchanged**.

# Challenge Clip Analysis

## `karsl_test_s01_sign0174_repfirst` — occlusion / crossing (frames 11-33)

The known hard region behaved exactly as the state contract intends:

| Frames | LEFT | RIGHT | Evidence |
|---|---|---|---|
| 8-15 | OBSERVED | OBSERVED | both hands converging; margin falls 1.337 -> 0.673 as the swap becomes feasible |
| 16 | AMBIGUOUS | AMBIGUOUS | hands 0.01797 normalized apart — below the 0.02 proximity threshold |
| 17-26 | OBSERVED | **LIKELY_OCCLUDED** | only 1 raw detection exists; RIGHT gets no pose, no interpolation |
| 27 | OBSERVED | OBSERVED (**reassociated**) | absent 10 frames, rebound at cost 0.071, detector label agreed |
| 28-32 | OBSERVED | OBSERVED | stable |
| 33 | AMBIGUOUS | AMBIGUOUS | hands 0.01070 apart at the end of the gesture |

This is the clip where TASK-003B recorded a handedness disagreement between
the two extractors and a MediaPipe label flicker. The WiLoR-only tracker
kept a single consistent identity through the whole occluded span, emitted
`LIKELY_OCCLUDED` with the measured proximity as evidence rather than
asserting occlusion, and reattached the RIGHT hand on its first frame back
with a low cost and detector agreement. Debug overlay frame 20 shows
`LEFT [OBSERVED] raw#0 det=left` / `RIGHT [LIKELY_OCCLUDED] raw#-1 det=-`
with a single skeleton drawn on the one visible hand.

**Caveat carried forward:** the tracker maintains identity *consistently*,
but there is no ground truth for which physical hand remains visible during
the occlusion. TASK-003B showed the two extractors labelling that surviving
hand oppositely. Consistency is demonstrated; correctness is not, and cannot
be from this data.

## `karsl_test_s02_sign0176_repfirst` — extra hand / degenerate crop (frame 39)

Raw frame 39 contains three complete reconstructions: `right` (conf 0.845),
`left` (conf 0.640) and a second `left` (conf 0.565) whose box overlaps the
first at IoU 0.747.

The tracker suppressed the lower-confidence duplicate, bound LEFT to raw
detection 1 and RIGHT to raw detection 0, flagged the frame
`EXTRA_DETECTIONS`, recorded `number_of_raw_detections = 3` and stored the
reason `duplicate_same_label_iou=0.747>=(0.5)_of_detection_1`. The spurious
hand never became an identity in any frame, and no raw data was deleted.
The debug overlay renders **two** skeletons at frame 39, where the TASK-003B
visualization of the same frame rendered three overlapping ones.

Note the honest detail from the Quality Gate section: the spurious detection
was geometrically normal, so it was removed by *duplicate suppression*, not
by the plausibility gate.

## Other difficult clips

`karsl_test_s02_sign0172_repfirst`, `karsl_test_s02_sign0175_repfirst`,
`karsl_test_s03_sign0173_repfirst` and `karsl_test_s03_sign0174_repfirst`
each produced **both tracks on 100% of frames**, with zero ambiguous frames,
zero rejections, zero reassociations and zero handedness disagreements.
These are precisely the clips where MediaPipe struggled in TASK-003B; on
WiLoR raw output they are unremarkable for the tracker.

# Failure Cases

| Case | Observed | Handling | Residual risk |
|---|---|---|---|
| Hands physically coincident | 2 frames (`s01_0174` f16, f33) | flagged `AMBIGUOUS`, still bound | identity could be wrong in exactly these frames; flag lets downstream skip or down-weight them |
| Genuine occlusion | 10 frames (`s01_0174` f17-26) | `LIKELY_OCCLUDED`, no pose emitted | no monocular method can recover the hidden hand; the label is heuristic |
| Duplicate same-label detection | 1 frame (`s02_0176` f39) | suppressed, recorded with IoU | a duplicate below IoU 0.5 would survive to become an unassigned extra |
| Degenerate reconstruction on a bad crop | 0 caught | geometry gate did not fire | **known blind spot**: MANO makes bad crops look geometrically normal; confidence and duplicate structure are currently the only usable signals |
| Detector handedness disagreement | 0 events on this pilot | cost term, not an override | untested at scale here; TASK-003B showed such cases exist |
| Long absence beyond gate growth | not observed | gate caps at 0.30 normalized | a hand absent very long and reappearing far away would start a fresh binding rather than reassociate |
| More than 3 detections | not observed | capacity + suppression | untested beyond N=3 |

The most important open weakness is the fourth row: the quality gate cannot
currently detect a *plausible-looking but wrong* WiLoR reconstruction. This
is stated plainly rather than hidden behind a gate that never fires.

# Tests

```bash
python -m unittest discover -s tests -p 'test_*.py'
python -m compileall -q evaluation scripts tests tracking
```

**87 tests pass** (30 new in this task: 23 in `tests/test_tracking_wilor.py`,
7 in `tests/test_tracking_source.py`). None require the WiLoR checkpoint,
MANO assets or KArSL videos — all inputs are synthetic or temporary files.

Coverage of the required scenarios:

| # | Scenario | Test |
|---|---|---|
| 1 | stable two-hand sequence | `TestStableTwoHandSequence` |
| 2 | hands moving toward each other | `TestConvergingHands` |
| 3 | hands crossing | `TestCrossingHands` |
| 4 | detector labels flipping for one frame | `TestLabelFlip` |
| 5 | one hand temporarily disappearing | `TestTemporaryDisappearance.test_missing_then_reassociated` |
| 6 | correct reassociation after disappearance | same, plus the emitted `reassociation` event |
| 7 | duplicate same-label detections | `TestDuplicateDetections` (incl. a negative test that overlapping *different*-label boxes are kept) |
| 8 | three detections in one frame | `TestThreeDetections` |
| 9 | low-quality false candidate | `TestQualityGate` |
| 10 | no detections | `TestNoDetections` |
| 11 | deterministic repeated execution | `TestDeterminism` (state sequences, events and serialized arrays) |
| 12 | raw input remains unchanged | `TestRawImmutability` (SHA-256 before/after) + detector-only rejection |

Additional guards: tracked-NPZ round trip preserves states and provenance;
missing hands are never interpolated; aggregate metrics sum correctly;
config weights must sum to 1.0; unknown config keys are rejected; and
`tests/test_tracking_source.py` asserts the tracking-side completeness
predicate agrees with `evaluation.comparison.common_contract` for WiLoR so
the two cannot silently diverge.

The test suite found two real defects during development, both fixed: an
unbound track was reported as a bare `MISSING` when a gated-out detection
plausibly belonged to it (now `REJECTED_QUALITY` with provenance), and a
metric counted a duplicate suppression as a quality rejection.

# Limitations

- **No identity ground truth.** Every "suspected identity switch",
  "ambiguous" and "likely occluded" figure is the tracker's own evidence,
  not a validated accuracy measurement. Luna is preparing an independent
  annotation/benchmark set; this report deliberately does not pre-empt it.
- **The occlusion label is heuristic** and proximity-based. It cannot
  distinguish a hand hidden behind the other from a detector failure that
  happens to coincide with hand proximity.
- **The quality gate has a known blind spot** for plausible-but-wrong MANO
  reconstructions (see Failure Cases); on this pilot it rejected nothing.
- **Thresholds are calibrated on one capture setup** — 18 clips, 1920x1080,
  30 fps, single seated signer, static camera. The normalized units should
  transfer, but the gate radius and proximity thresholds should be
  re-checked on materially different footage.
- **Only one clip exercised occlusion and one exercised a duplicate.** The
  pilot contains a single reassociation event and a single >2-detection
  frame, so those code paths have far more synthetic than real coverage.
- **Constant-position prediction.** The gate assumes a hand does not move
  more than the gate radius between frames; measured p99.9 motion is 0.0434
  against a 0.08 gate, so this holds comfortably here, but faster capture or
  faster motion would need a velocity model.
- The tracker is **WiLoR-only** by design; no MediaPipe data is read, and
  no cross-extractor comparison is performed.

# Recommendation

The derived stage is complete, deterministic, fully provenance-linked and
raw-preserving, and it behaves correctly on both documented challenge cases.
Its remaining uncertainties are exactly the ones that require external
annotation to resolve — which is the purpose of the independent validation
set being prepared separately.

Proceed to independent tracking validation. Do **not** treat the pilot
numbers above as accuracy results.

# Reproducibility

```bash
# from the task worktree, with the repository .venv active
python -m unittest discover -s tests -p 'test_*.py'
python -m compileall -q evaluation scripts tests tracking

python scripts/run_task004a_tracking.py \
  --manifest datasets/manifests/karsl_milestone1_pilot.csv \
  --wilor-run /home/hatim/graduation-project-runs/wilor_karsl_pilot_full \
  --out-dir /home/hatim/graduation-project-runs/wilor_karsl_pilot_tracked \
  --config configs/tracking/wilor_tracker.json \
  --strict-counts

python scripts/render_task004a_tracking_debug.py \
  --sample-ids karsl_test_s01_sign0174_repfirst karsl_test_s02_sign0176_repfirst \
  --manifest datasets/manifests/karsl_milestone1_pilot.csv \
  --video-root /home/hatim/Graduation-Project-Simulation \
  --tracked-run /home/hatim/graduation-project-runs/wilor_karsl_pilot_tracked \
  --out-dir /home/hatim/graduation-project-runs/wilor_karsl_pilot_tracked/debug
```

- Base commit: `origin/main` `b35f616`; frozen upstream WiLoR experiment
  `20e83afd7a54493523389fe02ca7077b1afc5866`.
- Tracking is deterministic: no randomness, no learned component, no
  adaptive threshold. `--strict-counts` asserts the exact 18-video /
  894-frame pilot.
- Each `wilor_tracked_meta.json` records the source raw path and its
  SHA-256, so any tracked output can be tied back to the exact raw bytes it
  was derived from.
- Generated artifacts (tracked NPZs, metadata, summary, debug MP4s) live
  outside the repository and are not committed. Committed here: tracking
  source, config, scripts, tests and this report.

# DECISION

**READY FOR INDEPENDENT TRACKING VALIDATION**
