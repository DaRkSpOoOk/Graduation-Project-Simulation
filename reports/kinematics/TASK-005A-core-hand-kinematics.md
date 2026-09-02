# TASK-005A — Core Hand Kinematics

**Status:** KINEMATICS IMPLEMENTED — READY FOR INDEPENDENT VALIDATION
**Branch:** `opus/task-005a-hand-kinematics`
**Base commit:** `ba6389f334ea5277b303c3f7795c919def4bf08e`
**Implementation commit:** `564167420c7f5b4f12197fe36e7d2b59ae08ace0`
**Date:** 2026-09-02

---

# Objective

Convert the TASK-004D tracked LEFT/RIGHT hand poses into physically
interpretable, sensor-ready hand kinematics: per-finger joint flexion,
adjacent-finger spread, and a canonical palm frame with a global orientation.

The task ends there. No virtual Hall or IMU signals, no normalization, no
recognition features, no temporal filtering. TASK-006 will consume these
quantities; TASK-005A must not anticipate that conversion.

---

# Inputs

| Input | Value |
|---|---|
| Tracked run (read-only) | `/home/hatim/graduation-project-runs/wilor_karsl_pilot_tracked_task004d` |
| Raw WiLoR run | `/home/hatim/graduation-project-runs/wilor_karsl_pilot_full` (not re-run) |
| Manifest | `datasets/manifests/karsl_milestone1_pilot.csv` |
| Scale | 18 videos, 894 frames, 1 788 hand-instances |
| Output | `/home/hatim/graduation-project-runs/wilor_karsl_pilot_kinematics_task005a` (outside git) |

The tracked NPZ of every sample is hashed before and after processing and the
two digests compared; a mismatch aborts the run. Tracking was neither modified
nor re-executed, and WiLoR was not re-run.

## Joint layout, established from the source

WiLoR exports 21 joints in **OpenPose hand order** via `mano_to_openpose`
(`pose/wilor/npz_io.py`): wrist 0, thumb 1-4, index 5-8, middle 9-12, ring
13-16, pinky 17-20. Confirmed empirically on the pilot before any code was
written -- metacarpals measure 0.082-0.095 m with the middle longest and the
pinky shortest, phalanges shorten distally (0.033 -> 0.022 -> 0.026 m for the
index), and the wrist plus four MCPs are near-coplanar (third singular value
0.0056 against 0.077 and 0.050).

---

# Kinematic Representation

Fixed contract, preserving the TASK-004 track identities exactly.

```
track order   0 = LEFT, 1 = RIGHT          (inherited from TASK-004, never reordered)
finger order  0 thumb, 1 index, 2 middle, 3 ring, 4 pinky
chain order   0 proximal, 1 middle, 2 distal
```

For the four non-thumb fingers the chain joints correspond approximately to
MCP / PIP / DIP. The thumb chain is rooted at the wrist, so its three angles
sit approximately at CMC / MCP / IP -- the generic names are kept because that
mapping is not asserted as anatomically exact, and no clinical accuracy is
claimed for any channel.

## Why joints and not MANO rotations

`pose/wilor/frame_extraction.py` line 195:

```python
joints[:, 0] = (2 * is_right_n - 1) * joints[:, 0]
```

WiLoR reconstructs **every** hand with the canonical MANO *right*-hand model
and then mirrors the joint x axis for a left hand. Two consequences:

* the exported 3D joints are **true left-hand geometry** in camera space, so
  geometry derived from them is physically meaningful for both hands;
* `hand_pose_rotmat` and `global_orient_rotmat` are **not** mirrored and remain
  in the un-mirrored canonical-right convention.

Reading anatomical flexion off those rotation matrices would require inverting
an undocumented mirroring, exactly the axis-convention trap this task warns
about. **No output channel is derived from MANO local rotations.** Everything
below comes from joint positions, whose physical meaning is demonstrable and
is demonstrated in `# Synthetic Validation`.

---

# Flexion Mathematics

For a finger chain of five points `j0 -> j1 -> j2 -> j3 -> j4` (root, three
chain joints, tip), the angle at chain joint `k` is the turn angle of the
polyline there:

```
b_in  = normalize( j[k+1] - j[k]   )        incoming bone direction
b_out = normalize( j[k+2] - j[k+1] )        outgoing bone direction
flexion_k = atan2( |b_in x b_out| , b_in . b_out )       in degrees
```

Chain roots: thumb `(0,1,2,3,4)`, index `(0,5,6,7,8)`, middle `(0,9,10,11,12)`,
ring `(0,13,14,15,16)`, pinky `(0,17,18,19,20)`.

**Convention.** A straight (collinear, same-direction) chain gives exactly
0 degrees; the value increases monotonically to 180 as the chain folds. It is
unsigned by construction, so "straight is 0, more bend is more positive" holds
identically on both hands with no handedness correction anywhere.

**Numerics.** `atan2(|a x b|, a . b)` is used rather than `arccos(a . b)`. The
arccos form is ill-conditioned exactly where hand kinematics spends most of its
time: its derivative is unbounded as the dot product approaches +-1, so a
perfectly straight finger reads a spurious ~1e-6 degrees. The atan2 form is well
conditioned across the whole range and returns exact zero for exactly-collinear
input. This was found by a synthetic test failing at 8.5e-7 degrees and fixed at
the source rather than by loosening the tolerance.

**Invariance.** Every term is a dot or cross product of *differences* of joint
positions, then normalized. Translation cancels in the differences, uniform
scale cancels in the normalization, and any orthogonal transform preserves both
products. Flexion is therefore invariant to translation, uniform scale, global
rotation, **and reflection**.

## The proximal channel's reference bone — stated, not hidden

The proximal angle is measured against the `wrist -> MCP` vector. With a
21-joint skeleton the wrist is a **single point** at the carpus, so that vector
fans outward from the wrist rather than following the metacarpal shaft. An
anatomically flat, fully extended hand therefore reads a finger-specific
**non-zero** proximal angle.

This is visible in the pilot: the index proximal channel never falls below
12.9 degrees across 894 frames, and its 5th percentile is 24.7, while the
middle and distal channels of the same finger reach 0.1 and 0.2 degrees.

The offset is **not** removed. Subtracting a per-finger neutral would be
calibration rather than geometry, and would bake an assumed rest posture into
the representation. The channel remains strictly monotone in true MCP bend,
which is the property a downstream sensor model needs (a real Hall sensor also
carries an arbitrary offset that is calibrated out). `TestProximalReferenceBone`
asserts the offset exists, that it is zero for a finger whose metacarpal happens
to lie on the palm axis, and that the channel stays monotone.

**Known coupling.** The proximal angle is a *total* inter-bone bend and so
mixes flexion with abduction. Spread is reported separately in the same frame
so a consumer can account for it. This is recorded under `# Limitations`.

---

# Spread Mathematics

```
d_f      = normalize( proximal-phalanx direction of finger f )
d_f_perp = normalize( d_f - (d_f . n) n )        n = palmar normal
spread(a,b) = atan2( |d_a_perp x d_b_perp| , d_a_perp . d_b_perp )
```

Segments: thumb `1->2`, index `5->6`, middle `9->10`, ring `13->14`,
pinky `17->18`. Pairs, in output order: thumb-index, index-middle, middle-ring,
ring-pinky.

**Output is an unsigned magnitude** in [0, 180] degrees. No signed
adduction/abduction convention is claimed: a robust signed convention needs a
per-finger neutral axis this representation does not establish, and the task
prefers unsigned unless a signed one is demonstrated.

**Palm-relative by construction.** Projecting into the palm plane before
measuring is what makes the quantity rigid-rotation invariant: a global rotation
turns the finger directions and the normal together, leaving every in-plane
angle unchanged. Mirroring maps `n -> M n` and `d -> M d`, and reflections
preserve angles, so mirrored LEFT/RIGHT poses give identical spread.

## Conditioning limit — an a priori choice, and its cost

The projected norm equals `sin(theta)`, where theta is the angle between the
finger's proximal phalanx and the palm normal. A finger pointing along the
normal has **no direction at all** within the palm plane, and near that pole the
projected direction amplifies any input angular error by `1/sin(theta)`.

The first full-pilot run made the consequence obvious: spread reached
**176.7 degrees** between the ring and pinky, with frame-to-frame changes up to
**127 degrees**. Every implausible value traced to a poorly conditioned
projection:

| min projected norm of the pair | pairs | max spread | fraction > 120 deg |
|---|---|---|---|
| 0.00 - 0.10 | 85 | 176.7 | 24.71 % |
| 0.10 - 0.20 | 233 | 158.1 | 3.00 % |
| 0.20 - 0.30 | 312 | 133.1 | 0.64 % |
| 0.30 - 0.50 | 478 | 86.6 | 0.00 % |
| 0.50 - 1.00 | 5 972 | 88.4 | 0.00 % |

A finger within a few degrees of the palm normal is genuinely degenerate for
this quantity, which the task requires be detected and flagged. The limit is set
at **theta >= 15 degrees** (projected norm >= 0.2588), chosen a priori by
capping the error amplification at ~3.9x.

**It is not a discovered boundary.** Unlike the TASK-004D cross-label rule,
there is **no natural gap** in this distribution -- it is a smooth continuum
(histogram bins of width 0.05 run 13, 72, 122, 111, 143, 169, 172, ... with no
trough). The threshold is a stated conditioning choice and its cost is
material, so the sensitivity is reported in full:

| threshold | sin | pairs suppressed | % of pairs | max spread kept | pairs > 120 deg kept |
|---|---|---|---|---|---|
| 0 deg | 0.000 | 0 | 0.00 % | 176.7 | 30 |
| 5 deg | 0.087 | 55 | 0.78 % | 159.5 | 13 |
| 10 deg | 0.174 | 242 | 3.42 % | 133.1 | 2 |
| **15 deg** | **0.259** | **474** | **6.69 %** | **97.6** | **0** |
| 20 deg | 0.342 | 784 | 11.07 % | 88.4 | 0 |
| 30 deg | 0.500 | 1 108 | 15.65 % | 88.4 | 0 |

Nothing is clipped: a rejected pair becomes NaN with the flag
`SPREAD_DIRECTION_DEGENERATE_<finger>`, never a capped number.

---

# Palm Frame

Built from four stable landmarks -- wrist, index MCP, middle MCP, pinky MCP.

```
f (distal)  = normalize( middle_MCP - wrist )
u_raw       = index_MCP - pinky_MCP      on the RIGHT hand
            = pinky_MCP - index_MCP      on the LEFT hand
n (normal)  = normalize( f x u_raw )                     palmar / volar
u (lateral) = n x f
R           = [ u | n | f ]   as columns
```

`u` is re-derived as `n x f` rather than used raw, so the axes are exactly
orthonormal even though the four landmarks are only approximately coplanar.
`det(R) = +1` always, by construction.

**Direction of `n`, pinned by an explicit pose.** For a right hand held with the
palm facing the viewer and fingers pointing up, `n` points at the viewer -- so
`n` is the **palmar (volar) normal** and the dorsal normal is `-n`. Asserted in
`TestPalmFrameConvention` against a hand-built canonical pose, not read back
from the implementation.

**Stability.** The frame uses only differences of joint positions and
normalizes, so it is invariant to translation and uniform scale, and rotates
correctly with the hand: `R(rotated) = Rot @ R`. Both are asserted directly.

**Output.** `palm_rotation_matrix [F,2,3,3]` and `palm_quaternion_wxyz [F,2,4]`.
The quaternion is `[w, x, y, z]`, normalized, with the sign fixed so `w >= 0`
(q and -q are the same rotation, so an unconstrained conversion could return
either and the run would not be deterministic). Conversion uses Shepperd's
method, choosing the branch with the largest denominator so no square root is
taken of a near-zero quantity. **No Euler angles are emitted** -- the matrix and
quaternion are the authoritative orientation representation.

---

# LEFT / RIGHT Canonicalization

**Handedness enters in exactly one line of the codebase**: the direction of
`u_raw` in `kinematics/hand_frame.py::build_palm_frame`.

**Flexion and spread need no correction at all.** Both are built from dot and
cross products of joint differences, which every orthogonal transform preserves
-- reflections included. A mirrored pair therefore produces *identical* values
automatically. Nothing is negated anywhere, and no value was flipped to make a
test pass.

**The palm frame does need it.** For `M = diag(-1, 1, 1)` and `p_left = M
p_right`, a reflection gives `(M a) x (M b) = -M (a x b)`. Applying the same
formula to both hands would yield `n_left = -M n_right`: the normal would point
out of the palm on one hand and into it on the other -- silently. Reversing
`u_raw` on the left hand cancels that sign exactly:

```
f_left = M f_right
u_raw_left = -M u_raw_right
n_left = normalize(M f x -M u) = M n_right
u_left = (M n) x (M f) = -M u_right
```

giving the identity, asserted directly in `TestMirroredHands`:

```
R_left = diag(-1,1,1) @ R_right @ diag(-1,1,1)      det = (-1)(+1)(-1) = +1
```

**The honest consequence.** The distal axis `f` and the palmar normal `n` carry
the same anatomical meaning on both hands. The lateral axis `u` does not -- it
points radially (toward the thumb) on the right hand and ulnarly on the left.
This is unavoidable: two mirror-image hands cannot both have three anatomically
identical axes *and* both be right-handed coordinate frames. One axis must carry
the handedness, and the lateral axis was chosen because the downstream
virtual-IMU work cares about where the palm faces and where the fingers point.

A dedicated test (`test_applying_the_right_rule_to_a_left_hand_would_flip_the_normal`)
asserts that omitting the canonicalization inverts the normal, so the guard
cannot be silently removed.

---

# Tracking-State Policy

TASK-004's judgement is authoritative. Nothing is invented.

| Tracker state | Kinematics |
|---|---|
| `OBSERVED` | computed |
| `AMBIGUOUS` | computed, flagged `TRACK_STATE_AMBIGUOUS` |
| `MISSING` | all channels NaN, `valid_kinematics = false` |
| `LIKELY_OCCLUDED` | all channels NaN, `valid_kinematics = false` |
| `REJECTED_QUALITY` | all channels NaN, `valid_kinematics = false` |

`AMBIGUOUS` carries a real reconstructed pose (it mirrors TASK-004's
`POSE_STATES`); what was uncertain is the identity assignment, not the geometry.
Its kinematics are computed, the caveat travels with the frame as a flag, and
the original tracking state is preserved verbatim in `tracking_state_code` so a
consumer can exclude those frames. `kinematics` restates `POSE_STATES` rather
than importing the tracker at runtime, and a test asserts the two definitions
cannot diverge.

**Explicitly not done, anywhere:** interpolation, forward filling, copying the
other hand, smoothing, synthetic hidden hands, velocity estimation, EMA, Kalman,
Savitzky-Golay. TASK-005A is strictly per-frame, and two tests enforce it: one
perturbs a single frame and asserts no other frame's output moves; the other
asserts a gap between two usable frames is not filled.

## Two validity flags

`valid_kinematics` is **strict**: true only when every float channel for that
hand-instance is finite.

`valid_palm_frame` is true when the pose was usable and the palm frame and
quaternion are well formed -- the orientation channels are trustworthy even if
one per-channel quantity, such as a single spread pair, is undefined.

Channels that are geometrically sound are **always written**, even when the
strict flag is false. A consumer wanting only flexion should mask with
`np.isfinite(flexion_deg)` rather than relying on `valid_kinematics`, and the
metadata says so. This matters here: 215 hand-instances have a usable palm frame
and complete flexion but at least one undefined spread pair.

---

# Degenerate Geometry

All detected, none fatal to a video.

| Condition | Flag | Effect |
|---|---|---|
| No usable pose | `NO_POSE_STATE_<STATE>` | all channels NaN |
| Non-finite joint | `JOINTS_NON_FINITE` | all channels NaN |
| Wrong joint array shape | `JOINTS_WRONG_SHAPE` | all channels NaN |
| Palm landmarks non-finite | `PALM_POINTS_NON_FINITE` | all channels NaN |
| Zero-length palm axis | `PALM_AXIS_ZERO_LENGTH` | all channels NaN |
| Collinear palm landmarks | `PALM_POINTS_COLLINEAR` | all channels NaN |
| Quaternion not normalizable | `QUATERNION_NOT_NORMALIZABLE` | all channels NaN |
| Zero-length finger bone | `ZERO_LENGTH_BONE_<finger>_<chain>` | that flexion channel NaN |
| Finger within 15 deg of palm normal | `SPREAD_DIRECTION_DEGENERATE_<finger>` | that finger's spread pairs NaN |

Bone-length floor is 1e-6 m against real bones of 0.019-0.095 m -- four orders
of magnitude of clearance. Flags are stored per hand-instance as a JSON string
array in `kinematic_flags_json`.

---

# Output Schema

`hand_kinematics.npz` per video, saved and loaded with `allow_pickle=False`.
**No pickled Python objects**; a test asserts no array has dtype `object`.

| Array | Shape | dtype |
|---|---|---|
| `frame_index` | `[F]` | int32 |
| `timestamp_seconds` | `[F]` | float64 |
| `tracking_state_code` | `[F,2]` | int32 |
| `source_raw_detection_index` | `[F,2]` | int32 |
| `valid_kinematics` | `[F,2]` | bool |
| `valid_palm_frame` | `[F,2]` | bool |
| `flexion_deg` | `[F,2,5,3]` | float32 |
| `adjacent_spread_deg` | `[F,2,4]` | float32 |
| `palm_rotation_matrix` | `[F,2,3,3]` | float32 |
| `palm_quaternion_wxyz` | `[F,2,4]` | float32 |
| `kinematic_flags_json` | `[F,2]` | str |

`valid_palm_frame` is an addition to the required set (the contract says "at
least"); every required array is present with the required shape.

`hand_kinematics_meta.json` records schema version, track/finger/chain order,
the full flexion, spread and palm-frame definitions, quaternion order and sign
convention, the left/right canonicalization method and its identity, the
invalid-state policy, the spread conditioning limit and its rationale, the
source tracked path and its SHA-256, and the implementation commit.

---

# Synthetic Validation

74 tests, no MANO assets, no checkpoint, no pilot data. Fixtures are built from
explicit geometry so every expected value is known by construction.

| # | Requirement | Result |
|---|---|---|
| 1 | straight chain -> 0 | exactly 0.0 on all 15 channels |
| 2 | 30 deg bend | 30.000000 |
| 3 | 60 deg bend | 60.000000 |
| 4 | 90 deg bend | 90.000000 |
| 5 | isolated proximal bend | 55 at proximal, 0.000000 elsewhere |
| 6 | isolated middle bend | 55 at middle, 0.000000 elsewhere |
| 7 | isolated distal bend | 55 at distal, 0.000000 elsewhere |
| 8 | multiple fingers independent | index 25 / ring 65 / pinky 100, others 0 |
| 9 | translation invariance | equal to 1e-8 |
| 10 | uniform-scale invariance | equal to 1e-8 at 7.25x and 0.013x |
| 11 | global-rotation invariance of flexion | equal to 1e-8 over 3 axes |
| 12 | global-rotation invariance of spread | equal to 1e-8 |
| 13 | rotation changes palm orientation | `R' = Rot @ R` to 1e-8, and differs by > 0.1 |
| 14 | mirrored LEFT/RIGHT | flexion and spread identical to 1e-8; +70 reads +70 on both |
| 15 | quaternion normalization | \|q\| = 1 to 1e-12 |
| 16 | rotation orthonormality | max \|R'R - I\| < 1e-12 |
| 17 | determinant +1 | 1.0 to 1e-12 |
| 18 | missing hand -> NaN | all channels NaN, flagged |
| 19 | occluded hand -> NaN | all channels NaN, flagged |
| 20 | zero-length bone rejected | affected channel NaN, others intact |
| 21 | collinear palm points rejected | frame refused, flagged |
| 22 | deterministic repeated execution | bit-identical |
| 23 | source tracked input unchanged | arrays byte-equal after extraction |
| 24 | NPZ round-trip | every array and dtype preserved |
| 25 | schema / order preservation | array set equals the contract; order preserved |

Additional coverage beyond the required list: bend monotonicity; unsigned
convention under either rotation direction; the proximal reference-bone offset;
the spread conditioning limit either side of 15 degrees; the two validity flags;
`POSE_STATES` parity with the tracker; the mirror guard; quaternion sign
determinism for a negative-trace rotation; and per-frame independence.

Determinism was also confirmed at pilot scale: an independent full re-run
produced **18/18 byte-identical** NPZ files.

---

# Full Pilot Results

18 videos, 894 frames, 1 788 hand-instances.

| Quantity | Value |
|---|---|
| Strictly valid hand-instances | **1 555 / 1 788 (86.97 %)** |
| Palm-frame valid hand-instances | **1 770 / 1 788 (98.99 %)** |
| Strictly invalid but palm-frame usable | 215 |
| Invalid from tracking state | 18 |

## Tracking state to validity — exact, with no leakage

| Tracker state | valid | invalid |
|---|---|---|
| `OBSERVED` | 1 770 | 215 (spread conditioning only) |
| `LIKELY_OCCLUDED` | 0 | 15 |
| `MISSING` | 0 | 3 |

Every state without a usable pose produced zero valid kinematics, and every
`OBSERVED` instance produced a valid palm frame. The 18 masked instances are
exactly the frames TASK-004D established as occluded in
`karsl_test_s01_sign0174_repfirst`.

## NaN accounting

| Channel | NaN | total | note |
|---|---|---|---|
| `flexion_deg` | 270 | 26 820 | 18 masked instances x 15 channels, exactly |
| `adjacent_spread_deg` | 546 | 7 152 | 72 from masking + 474 from conditioning |

No unexplained NaN anywhere.

## Flags

| Flag | Count |
|---|---|
| `SPREAD_DIRECTION_DEGENERATE_ring` | 190 |
| `SPREAD_DIRECTION_DEGENERATE_pinky` | 135 |
| `SPREAD_DIRECTION_DEGENERATE_middle` | 69 |
| `NO_POSE_STATE_LIKELY_OCCLUDED` | 15 |
| `NO_POSE_STATE_MISSING` | 3 |

Never the thumb or index. That is anatomically coherent: the ring and pinky curl
most in signing handshapes, pointing their proximal phalanges out of the palm
plane. No zero-length bone, non-finite joint, collinear palm frame or
unnormalizable quaternion occurred anywhere in the pilot.

## Per-channel ranges

`flexion_deg`: min 0.14, median 21.00, max 113.65 degrees (26 550 finite).
`adjacent_spread_deg`: min 0.00, median 12.67, max 97.60 degrees (6 606 finite).

| Finger | proximal (min/med/max) | middle | distal |
|---|---|---|---|
| thumb | 5.6 / 23.8 / 58.6 | 5.3 / 23.9 / 55.1 | 1.2 / 21.8 / 84.3 |
| index | 12.9 / 42.1 / 76.6 | 0.1 / 11.0 / 113.7 | 0.2 / 13.1 / 69.9 |
| middle | 5.7 / 38.0 / 104.3 | 0.3 / 9.0 / 98.3 | 2.5 / 9.7 / 72.4 |
| ring | 8.3 / 28.6 / 101.5 | 7.3 / 15.8 / 96.4 | 3.5 / 14.2 / 80.5 |
| pinky | 2.7 / 34.3 / 104.3 | 4.3 / 13.3 / 93.3 | 3.4 / 13.5 / 85.2 |

| Spread pair | min | median | max | n |
|---|---|---|---|---|
| thumb-index | 0.0 | 45.9 | 88.4 | 1 770 |
| index-middle | 0.0 | 4.2 | 74.1 | 1 701 |
| middle-ring | 0.1 | 13.2 | 97.6 | 1 571 |
| ring-pinky | 0.0 | 7.8 | 71.7 | 1 564 |

The thumb-index median of 45.9 degrees against 4.2 for index-middle matches the
thumb's naturally abducted rest posture. Proximal medians (24-42 degrees) exceed
middle and distal medians (9-24), consistent with the reference-bone offset
documented above.

## LEFT / RIGHT distributions

| Track | flexion min/med/max | spread min/med/max |
|---|---|---|
| LEFT | 0.14 / 24.49 / 108.30 | 0.01 / 14.91 / 97.60 |
| RIGHT | 0.31 / 18.00 / 113.65 | 0.00 / 10.20 / 71.28 |

The LEFT hand shows moderately more flexion and spread. This is reported as an
observation, not a defect: the pilot is 6 signs from 3 signers, handedness and
per-sign role are uncontrolled, and the kinematics themselves are provably
mirror-symmetric (test 14). Distinguishing genuine signing asymmetry from a
reconstruction asymmetry needs an independent reference and is out of scope here.

## Frame-to-frame change

Consecutive valid pairs only; gaps are skipped, never bridged.

| Channel | n | p50 | p95 | p99 | max |
|---|---|---|---|---|---|
| `flexion_deg` | 26 010 | 0.83 | 8.89 | 22.09 | 74.96 |
| `adjacent_spread_deg` | 6 411 | 0.73 | 7.40 | 17.19 | 57.29 |
| palm orientation | 1 734 | 1.76 | 16.53 | 38.12 | 102.16 |

Anomaly rates:

| Channel | > 30 deg | > 45 deg | > 60 deg | > 90 deg |
|---|---|---|---|---|
| flexion | 122 (0.469 %) | 45 (0.173 %) | 9 (0.035 %) | -- |
| spread | 9 (0.140 %) | 2 (0.031 %) | 0 | -- |
| palm orientation | -- | 15 (0.865 %) | 7 (0.404 %) | 3 (0.173 %) |

## Numerical quality

| Diagnostic | Max over 1 770 instances |
|---|---|
| Quaternion norm error | 4.11e-08 |
| Rotation orthogonality error, max \|R'R - I\| | 8.76e-08 |
| Rotation determinant error, \|det - 1\| | 9.24e-08 |

All three sit at the float32 storage round-trip floor (~1e-7); in float64,
before storage, the tests measure < 1e-12.

---

# Failure Cases

Nothing was clipped or altered. The anomalies below are reported as found.

**1. Largest flexion jump — 74.96 deg, `s03_0173` LEFT ring middle, frames 22->23.**
The channel reads 70.1, 16.1, 14.5, 15.8, **90.7**, 78.1, 83.4 over frames
19-25, and the ring distal channel moves with it (13.6 -> 59.5). A bent finger
appearing to straighten for three frames (~100 ms at 30 fps) and then re-bend is
more consistent with reconstruction instability on a partly hidden ring finger
than with real motion, but this pilot has no per-finger ground truth and the
claim is not made either way. Both frames are `OBSERVED` with a valid palm frame.

**2. Largest palm-orientation jump — 102.16 deg, `s01_0175` LEFT, frames 26->27.**
Not isolated: the surrounding per-frame changes run 9.6, 28.3, 95.6, **102.2**,
35.2, 19.1, 6.9. A sustained multi-frame rapid rotation, all frames `OBSERVED`.
Either a genuinely fast wrist rotation or a reconstruction flip; distinguishing
them requires video review that is out of scope for this task.

**3. Sign-correlated spread loss — the clearest structural finding.**
Strict validity by clip is not random; it tracks the *sign*:

| Clip | strict | palm-frame |
|---|---|---|
| `s03_sign0172` | 32 / 100 (32.0 %) | 100 % |
| `s02_sign0172` | 31 / 72 (43.1 %) | 100 % |
| `s01_sign0172` | 65 / 104 (62.5 %) | 100 % |
| `s02_sign0173` | 64 / 88 (72.7 %) | 100 % |
| `s01_sign0174` | 50 / 68 (73.5 %) | 50 / 68 (73.5 %) |
| `s01_sign0173` | 84 / 102 (82.4 %) | 100 % |
| 9 other clips | 100 % | 100 % |

Sign 0172 is degraded across **all three signers** and sign 0173 across two.
That consistency indicates a handshape whose ring and pinky fingers curl toward
the palm normal, making their spread genuinely undefined -- not per-clip noise.
`s01_sign0174` is the only clip whose palm-frame validity is below 100 %, and
that is the TASK-004D occlusion masking, working as intended.

**4. Pre-conditioning spread anomaly, recorded for the record.** Before the
15-degree limit, spread reached 176.7 degrees with 127-degree frame-to-frame
flips, and 30 of 7 080 pairs exceeded 120 degrees. Those values were noise in an
ill-conditioned projection, not anatomy; they are now NaN with a flag rather
than capped.

---

# Limitations

1. **The proximal channel is not isolated MCP flexion.** It is a total
   inter-bone bend against the `wrist -> MCP` vector, so it couples flexion with
   abduction and carries a finger-specific non-zero neutral (index floor 12.9
   deg). Monotone in true bend; not a clinical goniometer reading.
2. **The spread conditioning limit is a stated choice, not a discovered
   boundary.** No natural gap exists in the distribution. At 15 degrees it
   suppresses 6.69 % of spread pairs; the full sensitivity table is above.
3. **Spread is unsigned.** Adduction and abduction of equal magnitude are
   indistinguishable. A signed convention needs a per-finger neutral axis this
   representation does not establish.
4. **The lateral palm axis is not anatomically consistent across hands** --
   radial on the RIGHT, ulnar on the LEFT. Mathematically unavoidable; the two
   axes that matter downstream (distal, palmar normal) are consistent.
5. **Thumb chain naming is generic.** Rooted at the wrist, its angles sit
   approximately at CMC/MCP/IP; no exact anatomical mapping is asserted.
6. **Absolute scale is MANO's, not metric truth.** Bone lengths come from the
   reconstructed betas. All kinematic outputs here are scale-invariant, so this
   does not affect them, but it would affect any future length-based quantity.
7. **No ground truth for the kinematics themselves.** Correctness is
   established analytically (synthetic tests with known angles and invariances),
   not against measured hand angles. The two large-jump cases in
   `# Failure Cases` cannot be adjudicated with the data available.
8. **Per-frame only, by design.** Single-frame reconstruction errors pass
   straight through. This is deliberate so that raw behaviour stays observable;
   whether any temporal treatment is warranted is a later decision.
9. **Pilot scale.** 18 videos, 894 frames, 6 signs, 3 signers, one dataset,
   one camera setup.

---

# Tests

```
python -m unittest discover -s tests -p 'test_*.py'
Ran 212 tests in 0.574s
OK
```

**212 tests, 212 passed, 0 failed, 0 errors, 0 skipped.**
138 pre-existing (all still passing) + 74 new kinematics tests.

```
python -m compileall -q evaluation tracking kinematics scripts tests
```

**0 errors.** Python 3.14.4.

---

# Reproducibility

```bash
git checkout 564167420c7f5b4f12197fe36e7d2b59ae08ace0

python scripts/run_task005a_kinematics.py \
  --tracked-run /home/hatim/graduation-project-runs/wilor_karsl_pilot_tracked_task004d \
  --out-dir     /home/hatim/graduation-project-runs/wilor_karsl_pilot_kinematics_task005a \
  --strict-counts

python -m unittest discover -s tests -p 'test_*.py'
python -m compileall -q evaluation tracking kinematics scripts tests
```

| Artefact | Path |
|---|---|
| Package | `kinematics/{layout,geometry,hand_frame,extractor,io}.py` |
| Runner | `scripts/run_task005a_kinematics.py` |
| Tests | `tests/test_kinematics.py` |
| Output (git-ignored) | `/home/hatim/graduation-project-runs/wilor_karsl_pilot_kinematics_task005a` |
| Summary | `.../kinematics_summary.json` |

CPU only; no checkpoint, MANO asset or video is required to reproduce this
stage. Determinism verified: an independent re-run produced 18/18 byte-identical
NPZ files. Source immutability verified: the tracked run's 18 NPZ digests are
unchanged, and each sample is re-hashed inside the runner with a mismatch
aborting the run.

---

# Decision

**KINEMATICS IMPLEMENTED — READY FOR INDEPENDENT VALIDATION**

The fixed output contract is produced for all 18 videos and 894 frames with
correct track identities. Flexion and spread are provably invariant to
translation, uniform scale, global rotation and reflection; a mirrored
LEFT/RIGHT pair yields identical values with no sign correction anywhere. The
palm frame is orthonormal with determinant +1 to 1e-12, its quaternion is
normalized and sign-deterministic, and handedness is confined to one documented
line with an asserted algebraic identity. Tracking states without a usable pose
produce NaN and never a guess, and no temporal filtering of any kind is applied.

Two substantive findings are carried forward rather than smoothed away: the
proximal channel's non-zero neutral, which is a property of the 21-joint
skeleton and is left uncalibrated on purpose; and the spread conditioning limit,
which is an a priori choice with no natural boundary in the data and a
measurable 6.69 % cost. Both are the kind of decision an independent validator
should re-examine, along with the two large-jump cases in `# Failure Cases`.

TASK-006 is not started.
