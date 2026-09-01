# Objective

Determine whether the frozen TASK-004A tracker actually maintains the
correct **physical subject LEFT hand** and **physical subject RIGHT hand**
through time, measured against the independently authored TASK-004B human
benchmark.

Internal continuity is not the question. A tracker can be perfectly smooth
while carrying `LEFT = physical RIGHT`. This task answers identity
*correctness* against human reference, not tracker self-consistency.

Both inputs are frozen. Although the same session authored TASK-004A,
nothing in the tracker — logic, cost function, thresholds, config, quality
gate, duplicate suppression, ambiguity policy, state machine or output —
was changed after seeing the annotations. Luna's annotations were likewise
not edited to agree with the tracker. Where this validation found a tracker
defect it is reported here and deferred to `TASK-004D`; it is not fixed.

Run date: 2026-09-02.

# Blind Evaluation Design

The evaluation was constructed so the tracker could not be flattered:

* **Detector handedness is never used as identity evidence.** Physical
  identity is decided only from Luna's independent reference points.
* **Alignment is exact** on `(sample_id, frame_index)`. No timestamp
  matching, no nearest-frame fallback. Duplicate annotation keys, missing
  annotated clips, missing tracker frames and duplicated tracker frames all
  hard-fail.
* **Human uncertainty is protected.** Frames the annotator marked
  `AMBIGUOUS`/`IDENTITY_AMBIGUOUS`, or recorded at `LOW` confidence, are
  excluded from strict identity scoring and reported separately. They are
  never converted into tracker failures.
* **The tracker is not asked to hallucinate.** A hand annotated
  `FULLY_OCCLUDED` or `OUT_OF_FRAME` is never counted as a recall miss.
* **Identity is a relative test.** For each frame the assignment
  (annotated LEFT -> tracker LEFT, annotated RIGHT -> tracker RIGHT) is
  compared against the swapped assignment. Only the *ordering* matters, so a
  systematic offset between an approximate human wrist point and the
  tracker's hand centre cannot by itself create a wrong verdict.
* **Acceptance criteria A-F were fixed before results were computed** and
  were not altered afterwards.

The validation code lives in `evaluation/tracking/`, separate from the
tracker, and only ever reads the frozen artifacts.

# Frozen Inputs

| Input | Value |
|---|---|
| Tracker | `opus/task-004a-temporal-hand-tracking` @ `00ec1d7de21837012fa3eb8faecbf635ac2503d6` (PR #6, open) |
| Benchmark | `luna/task-004b-tracking-benchmark` @ `012d58a989a079dbeca6e5cb49b26c384dd80c21` (PR #7, open) |
| Validation branch | `evaluation/task-004c-tracking-validation`, worktree `../Graduation-Project-Simulation-task004c`, from `origin/main` `b35f616` |
| Annotations | `evaluation/annotations/task004_hand_identity_visibility.csv`, SHA-256 `bae2c771ee5b8e4396e2e8d662b980318af708bbf0c58a946f24489091c0a261` |
| Raw WiLoR run | `/home/hatim/graduation-project-runs/wilor_karsl_pilot_full` |
| Tracked run | `/home/hatim/graduation-project-runs/wilor_karsl_pilot_tracked` (reused; see Tracker Integrity) |

Both frozen commits were cherry-picked into the validation worktree without
conflicts. Every file in both cherry-picks was then verified to have a git
blob hash identical to its source commit — 17/17 tracker files and 7/7
benchmark files are byte-identical. No integration edit was required.

# Annotation Integrity

The benchmark contract was **recomputed from the CSV** before any validation
ran. The runner hard-fails and refuses to continue on any mismatch.

| Statistic | Expected | Recomputed | Result |
|---|---:|---:|---|
| Videos | 8 | 8 | OK |
| Frames | 399 | 399 | OK |
| LEFT `VISIBLE` | 371 | 371 | OK |
| LEFT `PARTIALLY_OCCLUDED` | 28 | 28 | OK |
| RIGHT `VISIBLE` | 328 | 328 | OK |
| RIGHT `PARTIALLY_OCCLUDED` | 43 | 43 | OK |
| RIGHT `FULLY_OCCLUDED` | 18 | 18 | OK |
| RIGHT `AMBIGUOUS` | 10 | 10 | OK |
| Total occluded labels | 89 | 89 | OK |
| Ambiguous frames | 10 | 10 | OK |
| `HAND_CROSSING` frames | 124 | 124 | OK |
| `MOTION_BLUR` frames | 26 | 26 | OK |

Benchmark composition: 6 challenge clips + 2 control clips. The 10
human-ambiguous frames are `s01_0174` frames 11-15 and `s03_0174` frames
8-12.

One process note for transparency: a first pass of this validation split
`scene_flags` on `|` and produced 69/16 for crossing/blur. That was a defect
in the validator's own throwaway parsing, not in the annotations — the field
separator is `;`. Using Luna's own parser reproduces every expected figure
exactly, as shown above.

# Tracker Integrity

The existing tracked run was reused rather than regenerated, after
verifying it is exactly the frozen tracker's output:

* every one of the 18 `wilor_tracked_meta.json` files records a source raw
  NPZ whose **SHA-256 still matches** (18/18);
* the config embedded in the tracked run is **identical** to the frozen
  `configs/tracking/wilor_tracker.json` (no differing keys);
* WiLoR was not re-run, and no tracker parameter was changed.

# Evaluation Protocol

* **Frame alignment**: exact `(sample_id, frame_index)` join. 399/399
  annotation rows aligned to a tracker frame with zero alignment errors.
* **Tracker "has a pose"**: state `OBSERVED` or `AMBIGUOUS` *and* a finite
  reconstruction present.
* **Coordinate conversion**: Luna's protocol defines
  `x = pixel_x / (width - 1)`, `y = pixel_y / (height - 1)`; the validator
  inverts exactly that.
* **Tracker position**: the detector box centre is used as the primary
  comparison point, with the projected MANO wrist (joint 0) as an
  independent cross-check. Over the 223 frames carrying both reference
  points, the identity assignment beat the swapped assignment in 222 under
  *both* position sources (mean identity distance 139 px box / 150 px wrist
  versus 388 px / 401 px swapped), so the conclusion does not depend on that
  choice.
* **Decision margin**: 25 px mean-per-hand. Luna's protocol states the
  points are approximate wrist/hand-centre references and explicitly "not
  pixel-perfect keypoint ground truth", so a smaller separation is reported
  as *indeterminate* rather than scored either way. The margin is applied on
  the **absolute** difference, so a confidently wrong frame is as decisive
  as a confidently right one.
* **Switch persistence**: a coordinate-derived identity flip must hold for
  at least 2 consecutive decisive frames before it is called confirmed, so
  one noisy reference point cannot manufacture a switch.

# Visibility Recall

Denominator: every hand instance the human annotated `VISIBLE` or
`PARTIALLY_OCCLUDED`.

| Stratum | Posed / Expected | Recall |
|---|---:|---:|
| LEFT | 399 / 399 | **100.00%** |
| RIGHT | 371 / 371 | **100.00%** |
| Overall | 770 / 770 | **100.00%** |
| Fully visible | 699 / 699 | **100.00%** |
| Partially occluded | 71 / 71 | **100.00%** |

Zero recall misses. Whenever the human said a hand was there, the tracker
had a pose on the corresponding canonical track.

# False Presence

Denominator: every hand instance annotated `FULLY_OCCLUDED` or
`OUT_OF_FRAME` (18 instances, all RIGHT hand, all in `s01_0174`).

| Tracker state on those 18 instances | Count |
|---|---:|
| `LIKELY_OCCLUDED` (no pose — correct) | 10 |
| `OBSERVED` (pose exposed — false presence) | 6 |
| `AMBIGUOUS` (pose exposed — false presence) | 2 |

**False presence: 8 / 18 = 44.44%.** This is the one substantive failure
found by this validation and is analysed in detail under Failures.

An important precision: the tracker did **not** synthesize or interpolate a
pose. It never fabricates geometry. In all 8 frames it bound a real WiLoR
reconstruction that existed in the raw output. The failure is that it
*accepted* that reconstruction as the physical RIGHT hand when the human
reference — corroborated by independent physical evidence below — says the
right hand was not observable.

# Physical Identity Accuracy

| Metric | Value |
|---|---:|
| Frames with both reference points | 223 |
| Strict identity-evaluable frames | **221** |
| Correct identity frames | **220** |
| Incorrect identity frames | **1** |
| Identity accuracy | **99.55%** |
| Decisive frames (margin >= 25 px) | 190 |
| Decisive correct | **190** |
| Decisive accuracy | **100.00%** |
| Indeterminate frames (margin < 25 px) | 31 |

Exclusions from strict scoring: 148 frames without sufficient reference
points (both control clips carry no coordinates by design), 20 low-confidence
frames, 10 human-ambiguous frames.

The single non-matching frame is `s01_0174` frame 8: identity distance
95.6 px versus swapped 90.8 px, a margin of **−4.8 px** on a ~95 px scale,
in a frame flagged `MOTION_BLUR`. That difference is an order of magnitude
smaller than the annotation's own approximation error, so it is classified
indeterminate, not a tracking error. The wrist-based cross-check gives the
identical 220/221.

**On every frame where the coordinate test is actually decisive, the
tracker's LEFT/RIGHT assignment is correct.**

# Identity Switches

| Metric | Value |
|---|---:|
| Confirmed persistent identity switches | **0** |
| Suspected / unresolved switches | **0** |

No sequence of two or more consecutive decisive frames ever showed the
canonical tracks carrying the opposite physical hands, in any clip.

# Reacquisition

| Metric | Value |
|---|---:|
| Annotation-supported disappearance/reappearance events | **0** |
| Correct reacquisitions | 0 |
| Incorrect reacquisitions | 0 |
| Accuracy | not applicable (empty denominator) |

**The benchmark contains no annotated reappearance.** In `s01_0174` the
human reference marks the RIGHT hand `FULLY_OCCLUDED` from frame 16 through
frame 33, which is the last frame of the clip — the hand never returns
within the annotated material. Reacquisition is therefore **untested** by
this benchmark, and a "100%" figure here would be vacuous.

Because an empty denominator would hide a tracker that *claims* a
reacquisition the reference does not support, the validator also checks the
opposite direction:

| Tracker-claimed reacquisitions | 1 |
|---|---:|
| Supported by the reference | **0** |
| Contradicted by the reference | **1** |

The tracker recorded a reacquisition of the RIGHT hand at `s01_0174`
frame 27 after 10 absent frames. The human reference states that hand was
`FULLY_OCCLUDED` at frame 27. That claimed reacquisition is therefore not
supported, and it is the same event as the false-presence failure.

# Occlusion Evaluation

Cross-tabulation of the tracker's `LIKELY_OCCLUDED` heuristic against the
human visibility label, for all 10 hand instances where the tracker raised
it:

| Tracker `LIKELY_OCCLUDED` + human state | Count |
|---|---:|
| `FULLY_OCCLUDED` | **10** |
| `PARTIALLY_OCCLUDED` | 0 |
| `VISIBLE` | 0 |
| `AMBIGUOUS` | 0 |

Every frame on which the tracker raised `LIKELY_OCCLUDED` was a frame the
human independently annotated as fully occluded. The heuristic produced no
false alarms in this benchmark.

This is a coincidence count, not a claim of true occlusion classification —
the flag is a proximity heuristic, and its recall is the weak side: it
covered 10 of the 18 human `FULLY_OCCLUDED` instances, with the other 8
being the false-presence frames where the tracker had a pose instead.

# Ambiguity Calibration

| Metric | Value |
|---|---:|
| Tracker `AMBIGUOUS` frames | 2 |
| Human ambiguous frames | 10 |
| Overlap | **0** |
| Crossing frames in benchmark | 124 |
| Tracker-ambiguous frames that are crossing frames | 2 / 2 |
| Human-ambiguous but tracker-confident | 10 |

The two concepts genuinely differ, and the zero overlap is explained rather
than being a simple failure:

* The tracker's two ambiguous frames are `s01_0174` f16 and f33. Both were
  raised by its **spatial-proximity** rule (the two detections were 0.0180
  and 0.0107 normalized units apart, inside its 0.02 threshold).
* Luna's 10 ambiguous frames are `s01_0174` f11-15 and `s03_0174` f8-12 —
  frames where a *human* could not tell which physical hand was which.
* At `s01_0174` f11-15 the two detections were 0.027-0.033 normalized apart,
  above the tracker's proximity threshold, and the cost margin was large, so
  the tracker was confident.

Crucially, **being confident where the human was unsure was not wrong
here**: for all 10 human-ambiguous frames the tracker's assignment agrees
with the spatial ordering of Luna's own reference points, and none of them
produced a confirmed switch. So the tracker's ambiguity signal is
*narrower* than human ambiguity — it flags geometric coincidence, not
perceptual uncertainty — rather than being simply mis-calibrated. It is,
however, clearly **under-sensitive** as a proxy for "a human would struggle
here": it caught 0 of 10 such frames, and 2 of 124 crossing frames.

# Extra Detection Handling

`karsl_test_s02_sign0176_repfirst` frame 39: **PASS**

Independently verified:

| Check | Result |
|---|---|
| Human reference describes two physical hands | yes (`VISIBLE` / `VISIBLE`) |
| Raw WiLoR reconstruction count at frame 39 | **3** |
| Tracker canonical identities exposed | exactly 2 (`left`, `right`) |
| Tracker recorded that 3 raw detections existed | yes (`number_of_raw_detections = 3`, flag `EXTRA_DETECTIONS`) |
| Provenance for both identities present and distinct | `left <- raw#1`, `right <- raw#0` |
| Identity correct at frames 38, 39, 40 | correct at all three (margins 80.5 / 81.1 / 65.0 px, all decisive) |

The rejected duplicate did not replace either physical hand, identity was
stable across the event, and the raw evidence remains recorded.

# Quality Gate Evaluation

Measured over the annotated clips only:

| Metric | Value |
|---|---:|
| Quality-rejected detections | **0** |
| Duplicate-suppressed detections (useful) | **1** |
| Unassigned extra detections | 0 |
| `LOW_QUALITY_*` advisory flags on kept hands | 4 |
| Advisory flags on hands the human calls visible | 4 |
| Clearly-bad-pose events the gate missed | **8** |

Findings:

* **The one true extra-detection event was solved by duplicate
  suppression**, not by the geometry gate — exactly as TASK-004A predicted.
* **The geometry gate rejected nothing**, confirming TASK-004A's own honest
  warning that MANO makes a bad crop look geometrically plausible.
* **It missed 8 clearly bad acceptances** (the false-presence frames). These
  poses were geometrically normal but physically wrong.
* The 4 advisory flags were all `LOW_QUALITY_POSE_JUMP` on hands the human
  says are visible. Because that flag only annotates and never rejects,
  these are not false rejects — there were **0 false quality rejections** —
  but they are also not useful signal here.

Net: the quality gate as currently configured is **inert** on this
benchmark. It neither helped nor harmed.

# Stratified Results

| Stratum | Frames | Identity correct/evaluable | Identity accuracy | Recall |
|---|---:|---|---:|---:|
| control | 148 | 0 / 0 (no reference points by design) | — | 100.00% |
| crossing | 124 | 25 / 25 | 100.00% | 100.00% |
| motion blur | 26 | 20 / 21 | 95.24% | 100.00% |
| partially occluded | 53 | 41 / 41 | 100.00% | 100.00% |
| fully occluded | 18 | 0 / 0 (excluded by protocol) | — | 100.00% |
| human ambiguous | 10 | 0 / 0 (excluded by protocol) | — | 100.00% |
| other | 159 | 159 / 159 | 100.00% | 100.00% |

The difficult temporal cases hold up: identity is 100% correct on all 25
identity-evaluable crossing frames and on all 41 partially-occluded frames.
The only sub-100% stratum is motion blur, driven entirely by the single
indeterminate −4.8 px frame discussed above.

Control clips carry no reference coordinates by design, so they validate
visibility and continuity (100% recall, 0 switches, 0 false presence) rather
than coordinate identity.

# Challenge Clips

| Clip | Role | Frames | Recall | Identity | Switches | Reacq | Tracker amb. | Human amb. | False presence | >2 raw det |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| `s01_0171` | control | 81 | 100% | n/a | 0 | 0 | 0 | 0 | 0 | 0 |
| `s02_0171` | control | 67 | 100% | n/a | 0 | 0 | 0 | 0 | 0 | 0 |
| `s01_0174` | challenge | 34 | 100% | 8/9 | 0 | 0 | 2 | 5 | **8** | 0 |
| `s02_0172` | challenge | 36 | 100% | 36/36 | 0 | 0 | 0 | 0 | 0 | 0 |
| `s02_0175` | challenge | 48 | 100% | 48/48 | 0 | 0 | 0 | 0 | 0 | 0 |
| `s02_0176` | challenge | 57 | 100% | 57/57 | 0 | 0 | 0 | 0 | 0 | 1 |
| `s03_0173` | challenge | 38 | 100% | 38/38 | 0 | 0 | 0 | 0 | 0 | 0 |
| `s03_0174` | challenge | 38 | 100% | 33/33 | 0 | 0 | 0 | 5 | 0 | 0 |

Seven of eight clips are clean on every metric. All failures are confined to
`s01_0174`.

## Primary challenge: `karsl_test_s01_sign0174_repfirst`, frames 11-33

The seven questions this task set, answered from Luna's annotation plus
independent evidence:

**1. Which physical hand is visible during frames 17-26?**
Luna: LEFT is `VISIBLE`, RIGHT is `FULLY_OCCLUDED`, for every frame 16-33.

**2. Is the tracker correct to keep the visible hand on track LEFT?**
Yes. The tracker holds LEFT `OBSERVED` throughout, and this matches the
human reference.

**3. Is physical RIGHT actually the occluded hand?**
Yes, per the reference. Raw WiLoR independently corroborates this for frames
17-26, where it produced only a **single** detection.

**4. Frames excluded as human-ambiguous.**
Frames 11-15 are `IDENTITY_AMBIGUOUS` (`LOW` confidence) and are excluded
from strict identity scoring, per protocol.

**5. Is the frame-27 reacquisition physically correct?**
**No.** This is the defect. From frame 27 raw WiLoR emits two detections
again, and the tracker binds the second one as the returning RIGHT hand.
The human reference says RIGHT is still fully occluded at frames 27-33.
Independent physical evidence supports the reference:

| Frame | detections | separation (norm.) | bbox IoU | confidences | tracker |
|---:|---|---:|---:|---|---|
| 16 | left,right | 0.0180 | 0.880 | 0.79 / 0.30 | AMBIGUOUS (posed) |
| 17-26 | left | — | — | 0.77-0.78 | LIKELY_OCCLUDED (no pose) |
| 27 | left,right | 0.0228 | 0.835 | 0.77 / 0.34 | OBSERVED (posed) |
| 28-32 | left,right | 0.0218-0.0233 | 0.833-0.856 | 0.77-0.78 / 0.34-0.39 | OBSERVED (posed) |
| 33 | left,right | 0.0107 | 0.924 | 0.77 / 0.37 | AMBIGUOUS (posed) |

The two boxes overlap at **IoU 0.83-0.86** and sit ~24 px apart, while the
second detection's confidence (0.34-0.39) is barely above WiLoR's own 0.3
threshold and less than half the primary's. Rendering the tracked output
confirms it visually: at frames 27, 30 and 33 the LEFT and RIGHT skeletons
are drawn **superimposed on the same visible hand**, with no distinct second
hand present. The second detection is a low-confidence duplicate of the
visible hand, not the returning right hand.

Why both tracker guards missed it:

* **Same-label duplicate suppression** did not fire because the two
  detections carry *different* detector labels (`left`, `right`), and the
  suppression is restricted to same-label pairs. TASK-004A restricted it
  deliberately and with good evidence (genuinely different hands reach
  IoU 0.92 in this pilot, so label-agnostic NMS would delete real hands) —
  but that leaves precisely this cross-label-duplicate hole.
* **Proximity ambiguity** did not fire because the separation was
  0.0218-0.0233, just above its 0.02 threshold. At frames 16 and 33
  (0.0180 and 0.0107) it *did* fire, which is why those two frames are
  `AMBIGUOUS` — the tracker was closest to catching the problem exactly
  where the geometry was tightest.

**6. Are frames 16 and 33 sensible ambiguity points?**
Yes. Both are frames where the two detections are nearly coincident. Marking
them `AMBIGUOUS` is the correct behaviour, and it is the only reason those
two frames are less severe than 27-32.

**7. Does the tracker ever fabricate the hidden hand?**
No. It never interpolates, never copies a neighbouring frame, and never
synthesizes geometry. Every pose it exposed traces to a real raw WiLoR row
via `raw_detection_index`. The defect is *acceptance of a spurious
detection*, not fabrication.

# Failures

**F1 — False presence during full occlusion (`s01_0174`, 8 hand instances).**
Severity: **blocking for acceptance criterion C**, and it also invalidates
the tracker's only claimed reacquisition (criterion B).

* Frames: 16, 27, 28, 29, 30, 31, 32, 33 — RIGHT track.
* Expected behaviour: the RIGHT track should have remained `MISSING` /
  `LIKELY_OCCLUDED` with no pose for frames 16 and 27-33, i.e. the same
  behaviour it correctly produced for frames 17-26.
* Root cause (evidence above): a low-confidence WiLoR detection that
  overlaps the visible hand at IoU 0.83-0.92 but carries the opposite
  detector label, so it passes both same-label duplicate suppression and the
  0.02 proximity-ambiguity test.
* Caveat stated plainly: the annotator recorded `LOW` confidence on these
  frames. The finding does not rest on the annotation alone — the IoU,
  confidence ratio and rendered overlay independently support it — but a
  future benchmark pass at higher confidence would strengthen it.

**F2 — Ambiguity signal is under-sensitive.**
The tracker flagged 2 of 124 crossing frames and 0 of 10 human-ambiguous
frames. Not a correctness failure (identity was right on all of them), but
it means downstream stages cannot rely on `AMBIGUOUS` to find the frames a
human would find hard.

**F3 — Quality gate inert.**
0 rejections, 0 useful flags, 8 missed bad acceptances. Confirms TASK-004A's
own disclosure rather than contradicting it.

**No fixes were applied.** Remediation belongs to `TASK-004D`, which must
disclose that it was changed after validation feedback.

# Limitations

* **Reacquisition is untested.** The benchmark contains no annotated
  reappearance, so criterion B rests only on the contradicted tracker-claimed
  event. A benchmark clip where a hand genuinely leaves and returns is needed.
* **Identity coverage is partial.** 221 of 399 frames were strictly
  identity-evaluable; control clips (148 frames) carry no reference points by
  design, so identity there is unverified.
* **The reference is approximate.** Luna's protocol states the points are not
  pixel-perfect. This is why identity uses a relative assignment test and why
  31 frames are reported indeterminate rather than scored.
* **LOW annotator confidence on the decisive failure region.** All 18
  `FULLY_OCCLUDED` labels are `LOW` confidence.
* **Single annotator, single pass**, no inter-annotator agreement measure.
* **Small benchmark**: 8 clips, 399 frames, one signer setup. `FULLY_OCCLUDED`
  occurs in exactly one clip.
* **No absolute 3D validation.** This task validates identity and presence,
  not the metric accuracy of any reconstruction.

# Acceptance Criteria

Criteria were fixed before results and are reported unchanged.

| # | Requirement | Observed | Result |
|---|---|---|---|
| A | 0 confirmed persistent LEFT/RIGHT identity switches | 0 | **PASS** |
| B | 100% correct reacquisition on unambiguous reappearances | 0 annotated events; 1 tracker-claimed reacquisition contradicted by the reference | **FAIL** |
| C | 0 fabricated poses for human `FULLY_OCCLUDED` instances | 8 / 18 | **FAIL** |
| D | `s02_0176` frame 39 extra-detection handling | PASS | **PASS** |
| E | 18/18 source WiLoR raw NPZ unchanged | 18/18 | **PASS** |
| F | >= 98% recall for clearly visible hands (L/R separately) | 100.00% / 100.00% | **PASS** |

4 of 6 pass. B and C fail, both caused by the same single defect F1.

# Final Verdict

**TASK-004 TRACKING NEEDS REVISION**

The tracker's core identity behaviour is strong: 100% visibility recall on
770 hand instances, 220/221 identity accuracy with 190/190 on decisive
frames, zero confirmed identity switches across all 8 clips, correct
handling of the known three-detection frame, and full raw-data integrity.
Seven of eight clips are clean on every metric.

It fails on one specific, well-localised defect: in `s01_0174` frames 16 and
27-33 it accepts a low-confidence, heavily overlapping, opposite-labelled
duplicate detection as the physical RIGHT hand while that hand is occluded,
producing 8 false-presence instances and one unsupported reacquisition.

### Required for TASK-004D (do not implement here)

| Frames | Current behaviour | Expected behaviour |
|---|---|---|
| `s01_0174` 16, 33 | RIGHT `AMBIGUOUS` with a pose | no pose exposed for RIGHT |
| `s01_0174` 27-32 | RIGHT `OBSERVED` with a pose | RIGHT `MISSING`/`LIKELY_OCCLUDED`, no pose |

Candidate directions for TASK-004D to consider (none evaluated here, and the
choice must be made on its own evidence, not tuned to this benchmark):
extend duplicate suppression to *cross-label* pairs above a high IoU;
require a minimum confidence or confidence ratio before reacquiring a track
after an absence; or make the proximity-ambiguity test scale-relative
(e.g. relative to box size) rather than a fixed normalized distance. Any
such change must disclose that it was made after validation feedback and be
re-validated against this frozen benchmark.

# Reproducibility

```bash
git worktree add -b evaluation/task-004c-tracking-validation \
  ../Graduation-Project-Simulation-task004c origin/main
git cherry-pick 00ec1d7de21837012fa3eb8faecbf635ac2503d6   # TASK-004A tracker
git cherry-pick 012d58a989a079dbeca6e5cb49b26c384dd80c21   # TASK-004B benchmark

python -m unittest discover -s tests -p 'test_*.py'
python -m compileall -q evaluation tracking scripts tests

python scripts/run_task004c_validation.py \
  --annotations evaluation/annotations/task004_hand_identity_visibility.csv \
  --tracked-run /home/hatim/graduation-project-runs/wilor_karsl_pilot_tracked \
  --raw-run     /home/hatim/graduation-project-runs/wilor_karsl_pilot_full \
  --output      reports/tracking/TASK-004C-validation-results.json
```

* The runner exits non-zero when acceptance fails, and exits early with a
  hard error if the recomputed annotation statistics do not match the locked
  contract.
* Validation is deterministic: no randomness, no learned component, no
  threshold fitted to the results.
* Machine-readable output: `reports/tracking/TASK-004C-validation-results.json`.
* Tests: **127 pass** (31 new in `tests/test_task004c_validation.py`), none
  requiring the WiLoR checkpoint, MANO assets, KArSL videos or a generated
  run directory.
* Nothing outside the validation layer, its tests and these reports was
  modified. No raw NPZ, tracked NPZ, video, overlay, checkpoint or MANO file
  is committed.
