# Task

**TASK-005E2 — Coincident-MCP invalid-geometry validation fix**

| | |
|---|---|
| Branch | `opus/task-005e2-coincident-mcp-validation` |
| Base commit | `b41bc1808d09b1987ebdcf417e1bdadc42962f6d` (TASK-005D validation) |
| Implementation commit | `3404a7af5d80110777ac61bc3076ef7ebada5dd6` |
| PR | #15 |
| Production code under fix | `564167420c7f5b4f12197fe36e7d2b59ae08ace0` (TASK-005A) |
| Date | 2026-09-02 |

This report documents work that was already completed and has since been
independently validated by TASK-005F. It adds no code, tests, thresholds or
pilot runs.

---

# Objective

Close the single production invalid-geometry defect identified by TASK-005D:
the benchmark fixture

```
degenerate_coincident_mcp
```

which the independent benchmark constructs specifically to be invalid, was
accepted by the production kinematics stage as **fully valid on both tracks
with no flag**.

TASK-005D recorded the expected behaviour as:

```
invalid geometry
explicit flag
valid_palm_frame  = false
valid_kinematics  = false
derived floats    = NaN
```

This was the only core production defect in scope. Nothing else was to change.

---

# Root Cause

Palm-frame validation previously checked only the **derived axes**, never the
landmarks those axes were derived from.

The fixture makes the index MCP and the middle MCP the same point
(`evaluation/kinematics/synthetic_hand.py`, `bases[2] = bases[1]`). Both are
palm-frame-defining landmarks, so the palm has collapsed: a defining triangle
with two identical vertices. The construction nevertheless survives every check
it performed, because each derived quantity remains individually well formed:

| Derived quantity | With `index_MCP == middle_MCP` |
|---|---|
| `f = normalize(middle_MCP - wrist)` | non-zero — the wrist is still distinct |
| `u_raw = index_MCP - pinky_MCP` | non-zero — the pinky is still distinct |
| `f x u_raw` | non-zero — the two are not parallel |

All three guards (`PALM_AXIS_ZERO_LENGTH`, `MIN_BONE_LENGTH`,
`MIN_FRAME_CROSS_NORM`) therefore pass, and a **finite, exactly orthonormal
frame with determinant +1** is produced from geometry that cannot be a hand.
Every downstream channel then computes normally, so the frame, quaternion,
flexion and spread all come out finite and the hand is reported valid.

The defect was one of *coverage*, not of arithmetic: no check existed that
could observe the coincidence, because the coincidence is invisible once the
points have been reduced to axes.

---

# Fix

A general **pairwise palm-landmark distinctness rule**, applied to the four
landmarks that define the frame:

```
wrist
index MCP
middle MCP
pinky MCP
```

Criterion, for **all six** landmark pairs `(a, b)`:

```
||a - b|| / ||middle_MCP - wrist||  >=  1e-3
```

The check is performed on the landmark positions directly, rather than inferred
from the axes they produce, which is precisely the gap the root cause
identifies. It runs after the existing palm-length guard, so the denominator is
already known to be non-degenerate and the pre-existing
`PALM_AXIS_ZERO_LENGTH` behaviour is preserved unchanged.

Location: `kinematics/hand_frame.py::build_palm_frame`, with the constant
`MIN_PALM_LANDMARK_SEPARATION_RATIO` in `kinematics/geometry.py`.

## Emitted flag

On rejection the frame construction returns no frame and emits:

```
PALM_LANDMARKS_COINCIDENT_<a>_<b>_sep_over_palm=<value>
```

naming the offending pair and recording the measured normalized separation, for
example:

```
PALM_LANDMARKS_COINCIDENT_index_MCP_middle_MCP_sep_over_palm=0.000e+00
```

## Resulting semantics

```
valid_palm_frame  = false
valid_kinematics  = false
all derived floats = NaN
```

`flexion_deg`, `adjacent_spread_deg`, `palm_rotation_matrix` and
`palm_quaternion_wxyz` are all NaN for that hand-instance, matching the
behaviour of every other whole-hand rejection already in the stage. No value is
clipped, substituted or interpolated.

---

# Threshold Rationale

The bound is a **numerical-degeneracy** limit, not an anatomical one.

**Scale-relative.** The criterion normalizes by the palm length
`||middle_MCP - wrist||`, so the same decision is reached under any uniform
scale or choice of units, and under any translation, by construction.

**Numerical floor.** Joints are stored as `float32`, whose relative precision is

```
float32 eps                                  ~= 1.19e-07
rounding carried by a coordinate difference  ~= 2.4e-07
```

A separation at or below that magnitude is storage rounding rather than
geometry. The threshold sits well clear of it:

```
1e-3  is approximately 4194x above the float32 numerical floor
```

so any separation the rule accepts is real signal.

**Clearance from real data.** Measured across all 1 770 pose-bearing
hand-instances of the pilot, the tightest normalized separation between any two
of the four palm landmarks is

```
0.2648
```

giving the threshold

```
approximately 265x clearance below the pilot minimum
```

The real distribution is extremely tight — the per-instance minimum ranges only
0.2648 to 0.2667 — because the MANO palm is near-rigid. Per-pair minima over the
whole pilot:

| Landmark pair | Minimum separation / palm length |
|---|---|
| index_MCP - middle_MCP | 0.2648 |
| middle_MCP - pinky_MCP | 0.5075 |
| index_MCP - pinky_MCP | 0.7037 |
| wrist - pinky_MCP | 0.8629 |
| wrist - index_MCP | 0.9548 |
| wrist - middle_MCP | 1.0000 |

**No tuning to the fixture.** The fixture's separation is **exactly 0.0**, so
every positive threshold rejects it with infinite margin; the synthetic case
exerts no pressure on the value at all. The threshold was set from float32
precision, and the **pilot was used only to confirm clearance**, never to tune.
No fixture identifier, case name or coordinate appears anywhere in the rule.

---

# Scope Protection

TASK-005E2 explicitly did **not** modify:

* **flexion mathematics** — `kinematics/extractor.py` has a zero-line diff;
* **spread mathematics** — unchanged, including the projection formulation;
* **palm-orientation convention** — the distal/normal/lateral definitions, the
  column order `[lateral | palmar normal | distal]`, the `w >= 0` quaternion
  sign convention and the LEFT/RIGHT canonicalization identity
  `R_left = M R_right M` are all untouched;
* **15 degree spread conditioning** — `MIN_SPREAD_PROJECTION_ANGLE_DEG = 15.0`
  is unchanged;
* **tracking** — no file under `tracking/` was touched, and TASK-004 outputs
  were neither regenerated nor modified;
* **the TASK-005 final contract** — array names, shapes, dtypes, track/finger/
  chain order and metadata fields are all unchanged.

The complete change against the base commit is **additions only**:

```
 kinematics/geometry.py   |  18 ++++++
 kinematics/hand_frame.py |  26 ++++++++
 tests/test_kinematics.py | 149 +++++++++++++++++++++
 3 files changed, 193 insertions(+)
```

with **zero deleted or modified lines** in `kinematics/`.

---

# Synthetic Validation

**Coincident-MCP fixture rejected on both tracks**, LEFT and RIGHT, each
carrying the explicit `PALM_LANDMARKS_COINCIDENT_index_MCP_middle_MCP` flag and
all-NaN derived floats.

**Invalid catalog improved from 5/6 to 6/6:**

| Fixture | Before | After |
|---|---|---|
| `degenerate_coincident_mcp` | **valid, no flag** | **INVALID** — `PALM_LANDMARKS_COINCIDENT_index_MCP_middle_MCP` |
| `degenerate_zero_length_bone` | INVALID | INVALID |
| `degenerate_collinear_palm` | INVALID | INVALID |
| `degenerate_nan_joint` | INVALID | INVALID |
| `degenerate_inf_joint` | INVALID | INVALID |
| `adversarial_tiny_nonzero_bone` | INVALID | INVALID |

**80 / 80 expected-valid fixtures did not regress** — zero palm-frame
rejections were introduced anywhere in the valid catalog, across every frame and
both tracks.

**Near-valid geometry remains accepted.** A palm whose index and middle MCP sit
at 10x the threshold (0.01 of palm length) is accepted with no flag and a valid
palm frame, confirming that a genuine palm is not rejected merely for having
close MCPs.

**Boundary behaviour is monotonic.** Separations of 0.5, 0.1, 0.01 and 0.002 are
accepted; 1e-4, 1e-5 and 0.0 are rejected. The decision moves once, at the
configured bound, and does not move back.

**LEFT mirror, scale and translation preserve the validity decision.** The
mirrored LEFT hand yields an identical verdict *and an identical flag string*;
rejection and acceptance both hold unchanged across scales 0.001x to 1000x and
across three independent translations.

A guard test additionally asserts that the collapsed palm **would** otherwise
have produced a finite frame — that the distal axis, lateral axis and their
cross product are each non-zero — so the reason this check is necessary cannot
be silently lost by a future edit.

---

# Pilot Regression

Regenerated into a new output directory; the TASK-005A run was preserved and not
overwritten.

| | |
|---|---|
| Videos | 18 |
| Frames | 894 |
| Hand-instances | 1 788 |

| Metric | TASK-005A | TASK-005E2 | Delta |
|---|---|---|---|
| **valid palm frames** | **1 770 / 1 788** | **1 770 / 1 788** | **0** |
| valid_kinematics | 1 555 / 1 788 | 1 555 / 1 788 | 0 |
| flexion NaN | 270 | 270 | **0** |
| spread NaN | 546 | 546 | **0** |

**Newly rejected real frames: 0.** No justification is therefore required, and
the expected palm-valid count is retained exactly.

**Final pilot arrays remain numerically unchanged from TASK-005A for real
data.** All **198 arrays across the 18 videos compare bit-identical**, and **no
new flag appears anywhere in the pilot**. The new rule does not fire once on
real data, which is the expected outcome given the 265x clearance documented
above.

Re-running the TASK-005D validation against this build changed exactly one
result field, `acceptance_criteria.I_invalid_geometry`, from `FAIL` to `PASS`.
Criteria A, B and E remained `FAIL`; those are the flexion-definition,
spread-definition and palm-orientation-convention disagreements that TASK-005E2
was explicitly forbidden to touch. TASK-005D's frozen commit pin in
`evaluation/kinematics/pilot_contract.py` was left untouched in the repository
and was overridden only at call time for that re-run, since re-pinning a freeze
guard is the validation owner's decision.

---

# Tests

```
264 passed
0 failed
0 errors
compileall: 0 errors
```

Up from 251, **including the +13 E2 tests**. The run covers all 81 TASK-004
tests and every pre-existing TASK-005A test, all still passing.

The 13 added tests cover: exact coincidence rejected; all six landmark pairs
individually; the "would otherwise have looked valid" guard; near-but-distinct
geometry accepted; monotonic boundary behaviour; the threshold matching its
documented numerical bound; translation invariance; scale invariance for both
verdicts; LEFT mirror equivalence for both verdicts; normal synthetic hands
still valid on both tracks; propagation to both validity flags and NaN; and the
recorded clearance from real pilot geometry.

---

# Limitations

This is a **numerical and geometric degeneracy guard, not an anatomical
palm-shape validator.** Specifically:

1. It answers only whether the four landmarks are distinct enough to define a
   frame at all. It does not assess whether the resulting palm is a plausible
   human palm — proportions, MCP ordering, or shape are not examined.
2. It covers the four **frame-defining** landmarks. A coincidence involving the
   ring MCP, which the palm frame does not use, is not detected here; that would
   be an anatomical rather than a numerical criterion, and the task called for
   the smallest mathematically justified rule.
3. It is a distinctness test, not a planarity or convexity test. Near-degenerate
   but distinct configurations remain the responsibility of the pre-existing
   `MIN_FRAME_CROSS_NORM` collinearity guard.
4. The 265x clearance figure is measured on this pilot only: 18 videos, 894
   frames, one dataset, one camera setup, with a near-rigid MANO palm. A
   reconstruction backend with a deformable palm could sit closer to the bound,
   though still far above it.

---

# Recommendation

**KEEP.**

The rule closes the defect exactly, is scale- and mirror-invariant by
construction, is justified numerically rather than fitted to either the fixture
or the dataset, carries clearance of roughly 4194x from the float32 noise floor
and 265x from the tightest real geometry, and produces **zero** change to real
pilot output.

---

# Reproducibility

| | |
|---|---|
| Branch | `opus/task-005e2-coincident-mcp-validation` |
| Implementation commit | `3404a7af5d80110777ac61bc3076ef7ebada5dd6` |
| PR | #15 (open, not merged) |
| Base | `b41bc1808d09b1987ebdcf417e1bdadc42962f6d` |

```bash
git checkout 3404a7af5d80110777ac61bc3076ef7ebada5dd6

# full suite (264 tests) and compile check
python -m unittest discover -s tests -p 'test_*.py'
python -m compileall -q evaluation tracking kinematics scripts tests

# the E2 tests alone
python -m unittest tests.test_kinematics.TestCoincidentPalmLandmarks

# pilot regeneration used for the regression comparison
python scripts/run_task005a_kinematics.py \
  --tracked-run /home/hatim/graduation-project-runs/wilor_karsl_pilot_tracked_task004d \
  --out-dir     /home/hatim/graduation-project-runs/wilor_karsl_pilot_kinematics_task005e2 \
  --strict-counts
```

| Artefact | Path |
|---|---|
| Threshold constant | `kinematics/geometry.py::MIN_PALM_LANDMARK_SEPARATION_RATIO` |
| Rule | `kinematics/hand_frame.py::build_palm_frame` |
| Tests | `tests/test_kinematics.py::TestCoincidentPalmLandmarks` |
| E2 pilot output (git-ignored) | `/home/hatim/graduation-project-runs/wilor_karsl_pilot_kinematics_task005e2` |
| TASK-005A pilot output, preserved | `/home/hatim/graduation-project-runs/wilor_karsl_pilot_kinematics_task005a` |

---

# Verdict

`COINCIDENT-MCP VALIDATION FIXED — READY FOR TASK-005F`
