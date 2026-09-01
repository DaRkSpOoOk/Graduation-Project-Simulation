# TASK-004D — Tracking False-Presence Remediation and Revalidation

**Status:** TASK-004 REMEDIATED — READY FOR TASK-005
**Branch:** `opus/task-004d-tracking-remediation`
**Base commit:** `c6bdfd04c47a9977da674b0c34b030c21bc03cb9` (TASK-004C validation)
**Remediation commit:** `a609842b79e850641f3f8c6d7e19c1e6a3be7440`
**Date:** 2026-09-02

---

## 1. Objective

Remove the single failure mode that caused TASK-004C to return
`TASK-004 TRACKING NEEDS REVISION`: the tracker reported a posed RIGHT hand on
8 frames that the independent human benchmark marks `FULLY_OCCLUDED`.

Scope is deliberately narrow. The tracker is not redesigned, the TASK-004B
annotations are not modified, and the failed TASK-004C result is not erased.
This task fixes the general defect that produced the failure, re-runs the full
pilot into a new output directory, and revalidates with the TASK-004C
methodology held constant.

Explicitly out of scope: kinematics, recognition, any change to the raw WiLoR
extraction stage, and any retuning of the TASK-004A association or quality
thresholds.

---

## 2. Post-Validation Disclosure — This Is Not Blind Development

**TASK-004D was designed and implemented AFTER the TASK-004C validation results
were known.**

This is `validation-informed remediation`. The frozen TASK-004B annotations were
used both to diagnose the failure and to confirm the outcome. The benchmark that
scores TASK-004D is therefore no longer blind with respect to it.

Concretely, this means:

* TASK-004D is **not** independently blind-tested on this benchmark, and no
  claim in this report should be read as if it were.
* The post-remediation numbers in §12 measure whether the specific defect was
  removed **without collateral damage on the same data**. They are not evidence
  of generalisation to unseen signers, signs, or datasets.
* The mitigation against benchmark overfitting is procedural rather than
  statistical: thresholds were chosen from the distribution of the whole pilot
  rather than from the failing frames (§7), the rule is written in terms of
  physical signals rather than frame or sample identifiers, and the regression
  tests in §10 are synthetic and contain no pilot data.
* A genuinely blind re-test requires annotations that do not yet exist. That is
  recorded as a limitation in §15, not claimed as done.

The TASK-004C report and `TASK-004C-validation-results.json` are untouched and
remain the record of the failure.

---

## 3. Frozen Inputs

| Input | Value |
|---|---|
| Raw WiLoR run (read-only) | `/home/hatim/graduation-project-runs/wilor_karsl_pilot_full` |
| Human annotations | `evaluation/annotations/task004_hand_identity_visibility.csv` |
| Annotation SHA-256 | `bae2c771ee5b8e4396e2e8d662b980318af708bbf0c58a946f24489091c0a261` |
| Annotation commit | `012d58a989a079dbeca6e5cb49b26c384dd80c21` |
| Annotation rows / clips | 399 rows, 8 clips |
| Validator | `evaluation/tracking/validation.py` — **byte-identical to TASK-004C** |
| Pre-remediation tracked run | `/home/hatim/graduation-project-runs/wilor_karsl_pilot_tracked` (TASK-004A, retained) |
| Post-remediation tracked run | `/home/hatim/graduation-project-runs/wilor_karsl_pilot_tracked_task004d` (new) |
| Pilot scale | 18 videos, 894 frames, 1 779 reconstructed detection rows |

Nothing in `evaluation/` or in the annotation CSV was modified by this task;
`git status` on those paths is clean at the remediation commit.

---

## 4. The TASK-004C Failure

TASK-004C failed two acceptance criteria:

* **C — full occlusion:** 8 fabricated-presence events out of 18 annotated
  fully-occluded/out-of-frame hand instances (44.4 %).
* **B — reacquisition:** 1 tracker-claimed reacquisition, contradicted by the
  reference, which says the hand was still fully occluded.

All 8 events fall in one clip, `karsl_test_s01_sign0174_repfirst`, at frames
16 and 27–33. Tracker states on those frames were `AMBIGUOUS` (f16, f33) and
`OBSERVED` (f27–32), i.e. the tracker exposed a RIGHT-hand pose while the
annotator recorded the right hand as fully occluded behind the left.

Everything else in TASK-004C already passed: 0 confirmed identity switches,
100 % visibility recall on both hands, 190/190 decisive identity frames correct,
the s02_0176 frame 39 extra-detection case PASS, and 18/18 raw NPZ unchanged.

---

## 5. Root Cause

The detector emitted a **second, weak detection sitting on top of the visible
left hand and carrying the opposite handedness label.**

TASK-004A suppressed duplicates only when both detections shared a handedness
label:

```python
if candidate.detector_label != other.detector_label:
    continue                      # <- the ghost escapes here
```

That restriction was a deliberate TASK-004A decision, made to protect genuinely
interlocked two-hand configurations where the two boxes overlap heavily. It has
a blind spot: a duplicate wearing the *wrong* label is invisible to it.

The consequence chains through the rest of the frame pipeline:

1. The ghost survives duplicate suppression as a legitimate candidate.
2. The intrinsic quality gate accepts it — the MANO output is geometrically
   plausible, merely wrong about which hand it describes (§15).
3. The RIGHT track is absent and unmatched, and the ghost lies inside its
   gate, so 2×N assignment binds them: from the tracker's point of view a
   plausible detection appeared exactly where the missing hand should be.
4. At frame 27 the track had been absent for 10 frames, so binding is recorded
   as a **reacquisition** — the tracker asserts the physical hand came back.

So the defect is not a threshold being slightly wrong. It is a missing evidence
class: nothing in the pipeline could express "this detection is a redraw of a
hand we have already accounted for".

A secondary observation, recorded because it corrects a TASK-004A claim: that
report justified label-agnostic overlap tolerance by noting that genuine hands
reach IoU 0.92. That measurement was **contaminated by the ghost pairs
themselves**. Excluding them, genuine two-hand pairs in this pilot top out at
IoU 0.778 (§7).

---

## 6. Remediation Design

Two changes, both deterministic, both config-driven. No model was trained and no
learned component was introduced.

### 6.1 Cross-label duplicate suppression

`tracking/wilor/association.py` gains `suppress_cross_label_ghosts()`, which
drops a candidate when **all four** of the following hold against an
already-kept, more-confident detection:

| # | Condition | Evidence type |
|---|---|---|
| 1 | Opposite, known handedness labels | detection |
| 2 | `bbox_iou >= 0.80` | detection (spatial) |
| 3 | centre distance / mean box size `<= 0.07` | detection (scale-relative) |
| 4 | weak confidence / strong confidence `<= 0.55` | detection (trust) |

Condition 3 is expressed relative to hand size on purpose: a fixed pixel or
normalised-image distance means very different things for a hand near the camera
and one far from it.

Detections are examined strongest-first, so the survivor of a ghost pair is
always the confident one and input order cannot decide the outcome.

The naive rule the brief rules out — `if IoU high: suppress` — is not what is
implemented. IoU alone is the single worst discriminator available here, because
interlocked hands in sign language legitimately produce high overlap; §7 shows
genuine pairs in this pilot reaching 0.778.

### 6.2 Reacquisition guard

`tracking/wilor/tracker.py` gains a guard, active when
`require_distinct_candidate_for_reacquisition` is set: a track that has been
absent may only be bound to a candidate that is **physically distinct** from the
other track's detection in the same frame. If the returning candidate is
ghost-suspect against the other track's detection, the binding is rejected, the
track stays `MISSING`/`LIKELY_OCCLUDED`, and the reason
`unsupported_reacquisition_cross_label_duplicate` is recorded.

This is where the **temporal** half of the evidence lives. §6.1 asks "is this
detection a redraw of a hand already accounted for?"; §6.2 asks "has this track
been absent, and if so is the thing bringing it back actually a separate
object?". The two conditions are combined: a candidate must survive the
detection test to remain, and a returning track must additionally survive the
temporal test to be re-bound.

The guard is deliberately redundant with §6.1 on the pilot — a ghost is normally
already gone by the time assignment runs, and it fired 0 times in the full run
(§11). It is retained as an independent second line of defence so that an
unsupported reacquisition cannot be reported even if the duplicate rule is
loosened or bypassed in future work.

### 6.3 What was deliberately not done

* **No temporal exemption for recently-tracked ghosts.** An earlier design
  protected a candidate that matched where the track was on the previous frame.
  It was rejected: at frame 16 the previous frame was already near-merged, so
  such an exemption would have permanently protected a track that had been
  tracked *into* the ghost.
* **No reacquisition-only fix.** A guard alone handles frames 27–33 but not
  frame 16, where RIGHT was observed on frame 15 and so was never in an "absent"
  context. This is why the primary rule sits at candidate level.
* **No pose fabrication, copying, or interpolation.** A suppressed ghost yields
  no pose. LEFT's pose is never copied into RIGHT, and no gap is interpolated.
* **No change to the raw stage.** Suppression happens in the derived tracking
  stage only.

---

## 7. Threshold / Distribution Evidence

Thresholds were chosen from the whole pilot, not from the failing frames.
Population: **all 886 same-frame detection pairs across 894 frames and 1 779
reconstructed rows** — 885 cross-label pairs (the only population the rule can
act on) and 1 same-label pair.

Ranges over the 885 cross-label pairs:

| Signal | 8 known-bad pairs | 877 other pairs | Gap between them | Threshold |
|---|---|---|---|---|
| bbox IoU | 0.8329 – 0.9240 | 0.0000 – **0.7782** | (0.7782, 0.8329) | **0.80** |
| centre sep / mean box | 0.0275 – **0.0613** | 0.0831 – 1.7350 | (0.0613, 0.0831) | **0.07** |
| confidence ratio | 0.3813 – **0.5107** | 0.6383 – 1.0000 | (0.5107, 0.6383) | **0.55** |
| weaker confidence | 0.3019 – 0.3936 | 0.4776 – 0.9088 | (0.394, 0.478) | *not used as a rule* |

Each threshold sits **inside** the gap between the two populations, not adjacent
to the failing observation. The brief's anti-overfitting example — "0.023 failed,
therefore threshold = 0.024" — would correspond to setting IoU to 0.833 or the
confidence ratio to 0.511. Neither was done.

**Sensitivity.** Sweeping each threshold with the other two held at default,
counting suppressed pairs across the full pilot and how many of them are genuine
hands:

| IoU | suppressed | genuine lost | | sep/box | suppressed | genuine lost | | conf ratio | suppressed | genuine lost |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.70 | 8 | 0 | | 0.030 | 1 | 0 | | 0.35 | 0 | 0 |
| 0.75 | 8 | 0 | | 0.050 | 2 | 0 | | 0.45 | 3 | 0 |
| 0.79 | 8 | 0 | | 0.062 | 8 | 0 | | 0.52 | 8 | 0 |
| **0.80** | **8** | **0** | | **0.070** | **8** | **0** | | **0.55** | **8** | **0** |
| 0.82 | 8 | 0 | | 0.083 | 8 | 0 | | 0.63 | 8 | 0 |
| 0.85 | 3 | 0 | | 0.150 | 8 | 0 | | 0.80 | 8 | 0 |
| 0.93 | 0 | 0 | | 0.400 | 8 | 0 | | 1.00 | 8 | 0 |

The IoU threshold sits on a plateau spanning at least 0.70–0.82, so 0.80 is not
a knife edge. The other two sweeps are conditional on the remaining two
conditions and so are weaker evidence on their own; they are reported for
completeness rather than as independent validation.

**The nearest genuine pair to the rule meets 0 of the 3 numeric conditions.**
It is `s01_0174` frame 15 (IoU 0.778, sep/box 0.0831, ratio 0.638) — the frame
immediately before the merge. The rule is not operating near its margin.

### 7.1 An honest limitation of this evidence

On this pilot the three signals **co-vary**: they all measure facets of the same
physical event, and consequently *any one of them alone* would have isolated the
same 8 pairs with 0 genuine hands lost. This benchmark therefore does **not**
demonstrate that the conjunction is necessary.

The conjunction is justified *a priori*, for robustness on data this pilot does
not contain, because each signal is known to fail alone:

* **High IoU alone** is met by genuinely interlocked hands; this pilot already
  reaches 0.778 and a larger corpus will exceed 0.80.
* **Near-coincident centres alone** occur during any true hand crossing.
* **Low confidence ratio alone** occurs whenever one hand is genuinely harder to
  see than the other — a faint but real hand.

Requiring all three means a real hand is protected if *any* single signal says
it is real. That is the intended failure direction: the rule should fail towards
keeping hands. §7 does not prove this is required here; it is a design choice
stated as such.

### 7.2 The same-label outlier

One pair in the pilot has centre separation 0.0419, inside the ghost range. It
is the **same-label** pair at `s02_0176` frame 39 — the extra-detection test
case. The cross-label rule never examines it (condition 1 excludes same-label
pairs), and the pre-existing same-label rule continues to handle it unchanged.
It is noted here because a naive separation-only rule computed over all pairs
would have looked contaminated.

---

## 8. Cross-Label Duplicate Handling and Provenance

When a cross-label duplicate is rejected:

* **The raw WiLoR detection is untouched.** Suppression happens only in the
  derived tracking stage. Raw integrity is verified at 18/18 (§12).
* **The detection remains in the tracked record.** Its index appears in
  `rejected_detections_json.indices`, and `number_of_raw_detections` still
  reports the true count (2, not 1), so the record never implies the detector
  produced fewer detections than it did.
* **A full reason string is stored**, carrying the reason code and all three
  measured values, for example:

  ```
  cross_label_duplicate_suspected_iou=0.880_sep_over_box=0.0481_conf_ratio=0.381_of_detection_0
  ```

* **The frame is flagged** `CROSS_LABEL_DUPLICATE_SUPPRESSED`.
* **No pose is exposed for the affected track.** LEFT's pose is not copied,
  nothing is interpolated, and the track's state is decided by the existing
  TASK-004A occlusion policy, unchanged.

Every suppression is therefore auditable and reversible from the stored record.

---

## 9. Reacquisition Policy

A track that has been absent is re-bound only when all of the following hold.

**Accepted when:**

1. A candidate falls within the track's gate, which widens with absence
   (`min(0.30, 0.08·(1 + 0.5·frames_missing))`) — unchanged from TASK-004A.
2. The candidate survived quality gating and both duplicate rules.
3. The candidate is physically distinct from the other track's detection in the
   same frame, i.e. it is not ghost-suspect against it (§6.2).
4. 2×N assignment selects the binding as the global minimum-cost solution.

**Rejected when:**

1. The only available candidate is a cross-label duplicate of the other track's
   detection — reason `unsupported_reacquisition_cross_label_duplicate`.
2. The candidate is a suppressed duplicate (it is no longer a candidate).
3. No candidate falls within the gate.

On rejection the track keeps its absent state (`LIKELY_OCCLUDED` while a
plausible occluder is nearby and the track is young enough, otherwise `MISSING`)
and no pose is emitted. The tracker records **no reacquisition event**, so
downstream consumers are never told a hand returned when the evidence does not
support it.

### 9.1 Real-benchmark status

**REAL-BENCHMARK REACQUISITION: UNTESTED / NO GROUND-TRUTH EVENT.**

The frozen TASK-004B annotations contain **0 annotated unambiguous
disappear-then-reappear events**. There is consequently nothing in this
benchmark against which real-video reacquisition accuracy can be measured.

No accuracy figure is claimed. In particular this report does **not** claim
100 % reacquisition accuracy; the correct statement is that the capability is
untested on real video. Reacquisition behaviour is exercised only by the
synthetic tests in §10, which are retained for that reason.

**UNSUPPORTED REACQUISITION CLAIMS: 0.** Post-remediation the tracker asserts
zero reacquisitions on the pilot, down from one that the reference contradicted.

---

## 10. Regression Protection

All tests are synthetic, built in-memory or in a temp directory, and contain no
pilot frames, sample IDs, or annotation data. `TestCrossLabelGhostSuppression`
adds 11 tests:

| Test | Guards against |
|---|---|
| `test_weak_opposite_label_duplicate_is_suppressed` | the defect itself |
| `test_the_confident_detection_of_the_pair_survives` | input order deciding the survivor |
| `test_two_genuine_hands_at_high_overlap_are_kept` | regressing to naive IoU suppression |
| `test_near_coincident_but_equally_confident_pair_is_kept` | suppressing interlocked real hands |
| `test_weak_but_well_separated_pair_is_kept` | suppressing a faint but distant real hand |
| `test_same_label_pairs_are_left_to_the_same_label_rule` | the two rules overlapping |
| `test_unlabelled_detections_are_never_ghost_suppressed` | acting without label evidence |
| `test_confidence_ratio_sweep_is_monotone` | a non-monotone or drifting decision boundary |
| `test_ghost_does_not_produce_a_second_tracked_hand` | end-to-end: absent track stays absent, provenance recorded, raw count preserved |
| `test_genuine_return_after_absence_is_still_reacquired` | the guard blocking a real hand |
| `test_suppression_is_deterministic_across_repeated_runs` | non-determinism |

The 12 original TASK-004A synthetic tests, the TASK-004C validator tests, and
`TestFrozenConfigProtection` (which asserts the TASK-004A thresholds are
unaltered) all still run and pass.

---

## 11. Full Pilot Results

Re-run: 18 videos, 894 frames, into
`/home/hatim/graduation-project-runs/wilor_karsl_pilot_tracked_task004d`.
The TASK-004A output directory was not overwritten.

| Quantity | Value |
|---|---|
| Videos / frames | 18 / 894 |
| Frames with both tracks | 876 (was 894) |
| Frames with no track | 0 |
| **Cross-label suppression events, full pilot** | **8** |
| Unsupported-reacquisition rejections | 0 |
| Same-label suppressions (pre-existing rule) | 1 |
| Clips affected by any suppression | 1 (`s01_0174`) |
| Clips byte-identical to TASK-004A | 17 of 18 |

### Required negative test

Every one of the 8 suppressions was cross-referenced against the frozen human
benchmark:

| | |
|---|---|
| Suppressions on annotated frames | 8 / 8 (100 % coverage — none unverified) |
| Human state of the suppressed hand | `FULLY_OCCLUDED` on all 8 |
| **Suppressions where the suppressed hand is human-visible** | **0** |
| **Suppressions where both hands are human-visible** | **0** |

**0 known human-visible true hands were suppressed**, which is the required
target. Visibility recall stayed at 100 % for both hands (§12), independently
confirming that no real hand was lost.

---

## 12. Frozen Benchmark Revalidation

Run with `evaluation/tracking/validation.py` **unchanged** — same 25 px identity
decision margin, same 2-frame switch persistence, same HIGH/MEDIUM confidence
exclusions, same visibility definitions, same acceptance thresholds. The
validator's annotation-integrity gate passed against the frozen SHA-256.

| Metric | TASK-004C (before) | TASK-004D (after) | Required |
|---|---|---|---|
| **False presence** | **8 / 18 (44.4 %)** | **0 / 18 (0 %)** | 0 / 18 ✅ |
| **Unsupported reacquisitions** | **1** | **0** | 0 ✅ |
| **Confirmed identity switches** | 0 | **0** | 0 ✅ |
| **Decisive identity accuracy** | 190/190 = 100 % | **190/190 = 100 %** | 100 % ✅ |
| **LEFT visibility recall** | 100 % | **100 %** | ≥ 98 % ✅ |
| **RIGHT visibility recall** | 100 % | **100 %** | ≥ 98 % ✅ |
| **s02_0176 frame 39** | PASS | **PASS** | PASS ✅ |
| **Raw integrity** | 18/18 | **18/18** | 18/18 ✅ |
| **Benchmark hardcoding** | — | **none** | none ✅ |
| Overall identity accuracy | 99.55 % | 99.55 % | — |
| Aligned frames | 399 | 399 | — |
| Tracker-flagged ambiguous frames | 2 | 0 | — |
| Acceptance A–F | 2 failed | **all passed** | — |
| Validator verdict | NEEDS REVISION | **VALIDATED** | — |

Occluded-hand state distribution over the 18 annotated fully-occluded/out-of-frame
hand instances:

| Tracker state | Before | After |
|---|---|---|
| `OBSERVED` (fabricated pose) | 6 | **0** |
| `AMBIGUOUS` (fabricated pose) | 2 | **0** |
| `LIKELY_OCCLUDED` | 10 | 15 |
| `MISSING` | 0 | 3 |

Overall identity accuracy of 99.55 % (220/221) is **unchanged** by the
remediation. The single non-matching frame is `s01_0174` frame 8, whose margin is
−4.8 px, far inside the 25 px decision margin; it is indeterminate rather than a
decisive error, which is why decisive accuracy remains 190/190. That frame is
flagged `MOTION_BLUR` by the annotator.

---

## 13. `karsl_test_s01_sign0174_repfirst` — The Failing Clip

Human reference vs tracker, before and after. `raw` is the detector's own
detection count, which the tracked record preserves.

| Frame | Human L / R | Before: LEFT / RIGHT | After: LEFT / RIGHT | raw | Suppressed |
|---|---|---|---|---|---|
| 8 | VISIBLE / VISIBLE | OBSERVED / OBSERVED | OBSERVED / OBSERVED | 2 | — |
| 9–10 | PARTIAL / PARTIAL | OBSERVED / OBSERVED | OBSERVED / OBSERVED | 2 | — |
| 11–15 | PARTIAL / AMBIGUOUS | OBSERVED / OBSERVED | OBSERVED / OBSERVED | 2 | — |
| **16** | **VISIBLE / FULLY_OCCLUDED** | **AMBIGUOUS / AMBIGUOUS** | **OBSERVED / LIKELY_OCCLUDED** | 2 | **ghost** |
| 17–26 | VISIBLE / FULLY_OCCLUDED | OBSERVED / LIKELY_OCCLUDED | OBSERVED / LIKELY_OCCLUDED | 1 | — |
| **27–30** | **VISIBLE / FULLY_OCCLUDED** | **OBSERVED / OBSERVED** | **OBSERVED / LIKELY_OCCLUDED** | 2 | **ghost** |
| **31–33** | **VISIBLE / FULLY_OCCLUDED** | **OBSERVED / OBSERVED** (f33 AMBIGUOUS) | **OBSERVED / MISSING** | 2 | **ghost** |

Post-remediation the RIGHT track is absent for exactly the 18 frames the
annotator marks fully occluded, and the state transitions `LIKELY_OCCLUDED` →
`MISSING` at frame 31 — 15 frames after the hand disappeared, which is the
pre-existing `occlusion_max_age_frames = 15` policy behaving as specified rather
than anything added here.

Frames 8–15 are untouched. That matters: the annotator marks the right hand
`AMBIGUOUS`, not occluded, so the hand is present and must keep being tracked.
The rule leaves them alone by a wide margin (§7).

The measured signals on the 8 suppressed frames:

| Frame | IoU | sep/box | conf ratio |
|---|---|---|---|
| 16 | 0.880 | 0.0481 | 0.381 |
| 27 | 0.835 | 0.0597 | 0.434 |
| 28 | 0.833 | 0.0613 | 0.485 |
| 29 | 0.841 | 0.0597 | 0.450 |
| 30 | 0.856 | 0.0569 | 0.436 |
| 31 | 0.849 | 0.0578 | 0.486 |
| 32 | 0.849 | 0.0575 | 0.511 |
| 33 | 0.924 | 0.0275 | 0.476 |

---

## 14. `karsl_test_s02_sign0176_repfirst` Frame 39 — Extra Detection

Unchanged and still PASS. Frames 37–41 are byte-identical before and after.

Frame 39 carries 3 raw detections. The third is a **same-label** duplicate
(IoU 0.747, separation 0.0419, confidence ratio 0.883) and is handled by the
pre-existing TASK-004A same-label rule with reason `duplicate_same_label`.
Both hands are annotated VISIBLE and both remain `OBSERVED`.

This clip is the direct check that the new rule did not disturb correct
behaviour: the cross-label rule never examines a same-label pair, so the
extra-detection path is untouched, and the annotated visible hands are still
tracked.

---

## 15. Remaining Limitations

1. **Not blind-tested.** §2. TASK-004D is validation-informed. A blind re-test
   needs annotations that do not exist yet.
2. **The intrinsic geometry quality gate remains weak against plausible-but-wrong
   MANO output.** It rejected 0 of the 8 ghost detections, before and after,
   because each ghost is a geometrically self-consistent hand that is simply
   wrong about which hand it is. This defect is **not fixed**. TASK-004D solves
   the failure through association, duplicate, and reacquisition logic instead.
   These are separate concerns and are deliberately not conflated: a future
   ghost that does *not* overlap an existing detection would still pass the
   quality gate.
3. **The evidence does not show the conjunction is necessary.** §7.1. On this
   pilot any single condition would have sufficed; the three-way rule is an
   a priori robustness choice.
4. **Single-clip evidence base.** All 8 known-bad pairs come from one clip and
   one signer. The threshold gaps are wide, but they rest on 8 positive
   examples.
5. **Real-video reacquisition is untested.** §9.1. Zero ground-truth events
   exist in the benchmark.
6. **The tracker under-signals ambiguity.** It now raises `AMBIGUOUS` on 0
   frames in the whole pilot (was 2, both on the ghost pair), while the
   annotator marks 10 frames as identity-ambiguous — all 10 tracked
   confidently. This was already true in TASK-004C
   (`human_ambiguous_but_tracker_confident` is 10 before and after) and is not
   caused by this change, but the remediation removed the tracker's only two
   ambiguity signals, so the gap is now total.
7. **Annotator confidence on the failing frames is LOW.** All 8 suppression
   frames carry `LOW` annotator confidence — unsurprising, since the hand is
   fully occluded and hard to judge. The false-presence metric does not exclude
   LOW-confidence rows (only the identity metric applies confidence
   exclusions), so this is disclosed rather than adjusted for. Occlusion
   direction is consistent across all 18 frames, which is what the metric uses.
8. **Pilot scale.** 18 videos, 894 frames, 8 annotated clips, one dataset
   (KArSL), one camera setup.

---

## 16. Tests

```
python -m unittest discover -s tests -p 'test_*.py'
Ran 138 tests in 0.418s
OK
```

**138 tests, 138 passed, 0 failed, 0 errors, 0 skipped.**
(127 before TASK-004D at `c6bdfd0`, plus the 11 new cross-label tests.)

```
python -m compileall -q evaluation tracking scripts tests
```

**0 errors.** Python 3.14.4.

---

## 17. Recommendation and Reproducibility

### Verdict

**TASK-004 REMEDIATED — READY FOR TASK-005**

All required outcomes are met: false presence 0/18, unsupported reacquisitions
0, confirmed identity switches 0, decisive identity accuracy 100 %, LEFT and
RIGHT visibility recall 100 % (≥ 98 % required), `s02_0176` frame 39 PASS, raw
integrity 18/18, and no benchmark identifier hardcoded anywhere in the tracker.

This verdict is qualified by §2: it certifies that the TASK-004C defect is
removed without collateral damage **on the same benchmark that diagnosed it**.
It is not a claim of blind-tested generalisation. Recommended before relying on
this in production: an independently annotated clip set containing (a) genuine
disappear/reappear events and (b) heavy two-hand interlocking, to test the
reacquisition policy and the keep-side of the duplicate rule respectively.

### Reproducibility

```bash
git checkout a609842b79e850641f3f8c6d7e19c1e6a3be7440

# 1. re-run the full pilot tracking stage (CPU, no checkpoint needed)
python scripts/run_task004a_tracking.py \
  --wilor-run /home/hatim/graduation-project-runs/wilor_karsl_pilot_full \
  --out-dir   /home/hatim/graduation-project-runs/wilor_karsl_pilot_tracked_task004d \
  --config    configs/tracking/wilor_tracker_task004d.json \
  --strict-counts

# 2. revalidate against the frozen human benchmark
python scripts/run_task004c_validation.py \
  --tracked-run /home/hatim/graduation-project-runs/wilor_karsl_pilot_tracked_task004d \
  --output      reports/tracking/TASK-004D-validation-results.json \
  --tracker-commit a609842b79e850641f3f8c6d7e19c1e6a3be7440

# 3. tests
python -m unittest discover -s tests -p 'test_*.py'
python -m compileall -q evaluation tracking scripts tests
```

| Artefact | Path |
|---|---|
| Remediated tracker | `tracking/wilor/{config,association,tracker}.py` |
| TASK-004D config | `configs/tracking/wilor_tracker_task004d.json` |
| TASK-004A config (frozen, unchanged) | `configs/tracking/wilor_tracker.json` |
| Results JSON | `reports/tracking/TASK-004D-validation-results.json` |
| TASK-004C report + JSON (failure record, retained) | `reports/tracking/TASK-004C-*` |
| Tracked output (git-ignored, 1.7 MB) | `/home/hatim/graduation-project-runs/wilor_karsl_pilot_tracked_task004d` |

The TASK-004D config differs from the frozen TASK-004A config by exactly four
added fields and **zero changed values** — no TASK-004A association, gating,
ambiguity, or quality threshold was retuned:

```
+ cross_label_duplicate_iou                    0.80
+ cross_label_duplicate_separation_ratio       0.07
+ cross_label_duplicate_confidence_ratio       0.55
+ require_distinct_candidate_for_reacquisition true
```
