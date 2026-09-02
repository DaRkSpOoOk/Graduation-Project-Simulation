# TASK-006A — Core Ideal Virtual Smart-Glove Sensor Model

| | |
|---|---|
| Branch | `opus/task-006a-core-virtual-glove` |
| Base | `bf3d678f734cdf3fb62c6acdaf1cd774083df159` (main, TASK-005 final) |
| Implementation commit | `8646ec03cd0a220a42f8e06db7d7b1f4d52cdea9` |
| Input contract | `TASK-005-final-v2` |
| Layout version | `ideal_virtual_glove_v1` |
| Output schema | `virtual_glove_v1` |
| Date | 2026-09-02 |

---

# Objective

Build the authoritative virtual smart-glove sensor layer on top of the frozen
TASK-005 kinematics contract: convert validated geometric hand kinematics into
an **ideal simulated glove**.

The retired physical five-Hall prototype is explicitly **not** an architectural
constraint. The ideal glove is designed first; later work may ablate it
downwards to discover what physical hardware actually needs.

Out of scope and not done here: any change to TASK-004 tracking or TASK-005
mathematics, the 15 degree spread-conditioning rule, the LSTM, the final
training dataset, the sensor-ablation study, and the glove visualizer.

---

# Frozen Input

Consumed read-only from the validated TASK-005F run:

```
/home/hatim/graduation-project-runs/wilor_karsl_pilot_kinematics_task005f
```

| Field | Shape |
|---|---|
| `flexion_deg` | `[F,2,5,3]` |
| `adjacent_spread_deg` | `[F,2,4]` |
| `palm_rotation_matrix` | `[F,2,3,3]` |
| `palm_quaternion_wxyz` | `[F,2,4]` |
| `valid_kinematics`, `valid_palm_frame` | `[F,2]` |
| `frame_index`, `timestamp_seconds` | `[F]` |
| `tracking_state_code`, `source_raw_detection_index` | `[F,2]` |

Contract conformance is enforced before any conversion: track order, finger
order, chain order, quaternion order and every array shape are checked, and a
mismatch raises `GloveInputError` rather than being coerced. Track order
`LEFT, RIGHT` and all source provenance are preserved verbatim.

Each sample's NPZ is hashed before and after processing and the digests
compared. Verified after the pilot run: **all 18 kinematics NPZ unchanged**.

---

# Ideal Glove Architecture

Per hand:

```
15 bend Hall/magnetic angular sensors
 4 spread Hall/magnetic angular sensors
 1 palm IMU package
```

```
19 Hall-type sensing channels
20 logical sensing packages (including the IMU)
```

The 15 bend channels are `5 fingers x 3 chain joints`, kept **independent**.
Three joints of one finger are never aggregated into a single finger value —
asserted by a test that all 15 map to distinct array slots.

This count is a contract. `EXPECTED_HALL_SENSORS = 19` and
`EXPECTED_SENSOR_PACKAGES = 20` are asserted directly, so the architecture
cannot be changed silently.

---

# Sensor Layout

Written per sample as `sensor_layout.json` and once at the run root. Every
entry carries `sensor_id`, `sensor_type`, `finger`/`pair`, `joint`/`role`,
`logical_location`, `display_marker`, `description`, plus the exact
`array`/`array_index` slot it maps to — so a consumer resolves `sensor_id` to a
position in the output arrays without guessing.

| Sensor ID | Type | Logical location | Marker |
|---|---|---|---|
| `H_THUMB_PROXIMAL` | bend | `dorsal_thumb_carpometacarpal` | `H` |
| `H_THUMB_MIDDLE` | bend | `dorsal_thumb_metacarpophalangeal` | `H` |
| `H_THUMB_DISTAL` | bend | `dorsal_thumb_interphalangeal` | `H` |
| `H_INDEX_PROXIMAL` / `_MIDDLE` / `_DISTAL` | bend | `dorsal_index_{metacarpophalangeal, proximal_interphalangeal, distal_interphalangeal}` | `H` |
| `H_MIDDLE_*`, `H_RING_*`, `H_PINKY_*` | bend | same pattern per finger | `H` |
| `H_SPREAD_THUMB_INDEX` | spread | `interdigital_web_thumb_index` | `H` |
| `H_SPREAD_INDEX_MIDDLE` | spread | `interdigital_web_index_middle` | `H` |
| `H_SPREAD_MIDDLE_RING` | spread | `interdigital_web_middle_ring` | `H` |
| `H_SPREAD_RING_PINKY` | spread | `interdigital_web_ring_pinky` | `H` |
| `IMU_PALM` | IMU | `dorsal_palm_centre` | `IMU` |

All 19 Hall entries carry `display_marker = "H"` and the single IMU carries
`display_marker = "IMU"`; both are asserted, as is the absence of any crossover.
Logical locations are unique.

This metadata exists now specifically so a later visualization task can show
where every Hall sensor sits. **No visualizer is built in TASK-006A.**

Thumb chain names stay generic: TASK-005 roots the thumb chain at the wrist, so
its three angles sit approximately at CMC/MCP/IP, and that mapping is recorded
as approximate rather than asserted as anatomically exact.

---

# 15 Bend Sensors

Each bend channel is the frozen TASK-005 unsigned inter-bone angle for one
finger chain joint, in the fixed order thumb, index, middle, ring, pinky and
within each finger proximal, middle, distal — so `bend sensor #1` is
`H_THUMB_PROXIMAL` and `#15` is `H_PINKY_DISTAL`.

Nothing is recomputed. `bend_angle_deg` is the frozen `flexion_deg` copied
verbatim.

---

# 4 Spread Sensors

Four independent adjacent-finger channels mapping directly onto the frozen
`adjacent_spread_deg [F,2,4]`: thumb-index, index-middle, middle-ring,
ring-pinky. They are first-class sensors of the ideal glove, not a derived
convenience.

The TASK-005 15 degree spread-conditioning rule is untouched. Where it left a
channel undefined, that sensor is simply invalid here — see below.

---

# Palm IMU

The authoritative IMU output is the already-validated TASK-005 palm
orientation, **copied verbatim**:

```
imu_rotation_matrix  [F,2,3,3]
imu_quaternion_wxyz  [F,2,4]      order w, x, y, z
```

No re-normalization, no sign change, no basis mapping. In particular the
LEFT/RIGHT comparison bases in `evaluation.kinematics.final_contract`
(`FIXTURE_BASIS_LEFT`, `FROZEN_SYNTHETIC_PRODUCTION_BASIS`) are an
**evaluation-only** integration convention and are deliberately not applied to
stored production orientation. A test asserts both tracks pass through an
identity rotation unchanged.

Verified over the pilot: **1 770 / 1 770 valid instances are bitwise-identical
copies** of the frozen source arrays.

---

# Normalization

The authoritative ML-facing signal is not ADC counts:

```
bend_normalized   = bend_deg   / 180.0
spread_normalized = spread_deg / 180.0
```

TASK-005 quantities are unsigned and contractually bounded by 0..180 degrees,
so valid normalized channels lie in `[0.0, 1.0]`.

The divisor **is** that contract. It is fixed a priori, identical for every
channel, hand, subject and dataset, and has no fitted parameter:

* **no pilot min/max normalization** — a test feeds the pilot's own bend
  min/median/max and asserts the maximum does not map to 1.0. The measured
  pilot maximum is `0.6314`, not `1.0`, which is direct evidence at run scale
  that no range fitting occurred;
* **no fitting to KArSL** — the same angle maps identically whether presented
  alone, with small values, or with large ones;
* **no per-finger neutral-offset subtraction** — the proximal channel's known
  non-zero neutral is preserved, not calibrated away. `12.9` degrees maps to
  `12.9/180`, asserted directly.

The original degree values remain available in `bend_angle_deg` and
`spread_angle_deg` alongside the normalized view.

**Out-of-contract values are never silently repaired.** A finite angle outside
`[0, 180]` raises `SensorContractViolation` by default; an optional `"flag"`
mode records the violation and invalidates that channel while leaving the value
**unclamped** and the original degrees intact. NaN is treated as absence, not
as a violation. On the pilot: **0 violations**.

---

# Optional ADC Encoding

A hardware-looking transfer layer, explicitly **not authoritative for ML**:

```
adc = normalized * 4095          12-bit, full scale 0..4095
```

Rounding, when integer output is selected, is half-up (`floor(x + 0.5)`):
deterministic, monotone, and exact at both rails — `0.0 -> 0`, `1.0 -> 4095`.

Invalid channels stay explicitly invalid, carrying the sentinel `-1`, which is
outside `0..4095` and cannot be mistaken for a reading; they remain masked by
the same validity arrays as every other view.

This is deliberately **not** the retired prototype's ~850–1700 count window,
which described that specific hardware and is not the ideal simulated-glove
contract. A test asserts full scale is 4095.

The three representations are kept distinct throughout and are separately named
in the metadata: **geometric angle** (degrees), **normalized ideal sensor value**
(authoritative), **optional ADC-like encoding** (non-authoritative).

---

# Validity / Missing Channels

TASK-005 Model-B semantics are inherited **per channel**. A hand is never
discarded wholesale because one channel is absent.

| Mask | Shape | Source |
|---|---|---|
| `bend_valid` | `[F,2,5,3]` | `isfinite(frozen flexion_deg)` for that exact channel |
| `spread_valid` | `[F,2,4]` | `isfinite(frozen adjacent_spread_deg)` for that exact channel |
| `palm_imu_valid` | `[F,2]` | frozen `valid_palm_frame` |

Masks derive from the frozen channel state, never from a guessed replacement.
The strict `valid_kinematics` flag is carried through as
`source_valid_kinematics` for provenance and is **deliberately never used to
mask a sensor**.

The scenario the task calls out is asserted directly and measured at run scale:
a hand with a valid palm, all flexion finite and one NaN spread channel yields
**15 usable bend sensors, a usable IMU, 3 valid spread sensors and 1 invalid
spread sensor** — not a whole invalid hand.

**Measured payoff on the pilot: 3 826 sensor channels are retained on
hand-instances whose strict `valid_kinematics` flag is `False`** (3 611
bend/spread channels plus 215 IMU instances). A whole-hand policy would have
discarded every one of them.

No interpolation, no forward filling, no cross-hand copying, no invented values.
An invalid channel stays NaN in both the degree and normalized views.

---

# IMU Derived-Signal Feasibility

## Angular velocity — IMPLEMENTED

Body-frame angular rate between consecutive valid orientations:

```
R_rel = R[k]^T @ R[k+1]
omega = axis_angle(R_rel) / (t[k+1] - t[k])        rad/s
```

This is what a gyroscope rigidly mounted on the palm measures. It is
scale-free in position, so the uncalibrated translational scale discussed below
does not affect it; orientation is already independently validated and the
timestamps are real. The axis-angle uses the well-conditioned `atan2` form
rather than `arccos((tr-1)/2)`, whose derivative is unbounded near zero rotation
— and a nearly still hand is the common case.

A sample is emitted **only** when all of the following hold, so nothing is ever
bridged: both orientations valid, `frame_index` genuinely adjacent, and a finite
strictly positive timestamp delta. Frame 0 of every track is always invalid.
**No smoothing of any kind is applied.**

## Accelerometer — DEFER ACCELEROMETER

No accelerometer channel is emitted. Fabricating one to complete the analogy
with physical hardware would be dishonest. The evidence:

1. **The frozen TASK-005 output contains no position channel at all** — only
   orientation. Any position would have to be pulled back from the TASK-004
   tracked stage.
2. **That position is not metric.** The tracked `camera_translation` is a
   monocular weak-perspective estimate tied to WiLoR's assumed focal length.
   Measured on this pilot, `|t_z|` is **397x the reconstructed palm length**: a
   hand 38 units from the camera while measuring 0.095 across would be far
   below one pixel if those units were metres. The absolute scale is arbitrary,
   and acceleration is not scale-free.
3. **Acceleration is a second derivative.** TASK-005A records palm-orientation
   changes up to 102 degrees between adjacent frames; differentiating a noisier
   position signal twice at 30 fps amplifies error by roughly 900x.
4. **No gravity reference.** A real accelerometer measures specific force
   including gravity. With no gravity direction and no metric scale, the output
   would not be comparable to any physical device.

This judgement is recorded in code (`accelerometer_feasibility()`), in the
per-sample metadata, and in the run summary, and is asserted by tests including
one that no array name contains "accel".

---

# Output Schema

`virtual_glove.npz` per sample, `allow_pickle=False`, no pickled objects.

| Array | Shape | dtype |
|---|---|---|
| `frame_index` | `[F]` | int32 |
| `timestamp_seconds` | `[F]` | float64 |
| `bend_angle_deg` | `[F,2,5,3]` | float32 |
| `bend_normalized` | `[F,2,5,3]` | float32 |
| `bend_valid` | `[F,2,5,3]` | bool |
| `spread_angle_deg` | `[F,2,4]` | float32 |
| `spread_normalized` | `[F,2,4]` | float32 |
| `spread_valid` | `[F,2,4]` | bool |
| `imu_rotation_matrix` | `[F,2,3,3]` | float32 |
| `imu_quaternion_wxyz` | `[F,2,4]` | float32 |
| `palm_imu_valid` | `[F,2]` | bool |
| `tracking_state_code` | `[F,2]` | int32 |
| `source_raw_detection_index` | `[F,2]` | int32 |
| `bend_adc_12bit` *(optional)* | `[F,2,5,3]` | int32 |
| `spread_adc_12bit` *(optional)* | `[F,2,4]` | int32 |
| `imu_angular_velocity_rad_s` *(optional)* | `[F,2,3]` | float32 |
| `imu_angular_velocity_valid` *(optional)* | `[F,2]` | bool |
| `source_valid_kinematics` *(provenance)* | `[F,2]` | bool |
| `source_valid_palm_frame` *(provenance)* | `[F,2]` | bool |

Plus `virtual_glove_meta.json` (schema version, layout version, orders, all
three representations, contract and validity policy, IMU policy including the
accelerometer deferral, ML contract, source path and SHA-256, implementation
commit) and `sensor_layout.json`.

---

# Pilot Results

18 samples, 894 frames — **exact alignment with the TASK-005F pilot confirmed**
under `--strict-counts`.

| Quantity | Value |
|---|---|
| Bend sensors valid | **26 550 / 26 820 (98.99 %)** |
| Spread sensors valid | **6 606 / 7 152 (92.37 %)** |
| Palm IMU valid | **1 770 / 1 788 (98.99 %)** |
| Channels retained where strict flag false | **3 826** |
| Contract violations | **0** |

Every count matches the frozen TASK-005 channel state exactly, which is the
intended result: this stage re-expresses validated kinematics, it does not
re-derive them.

## Validity by tracking state

| Tracking state | Bend | Spread | IMU |
|---|---|---|---|
| `OBSERVED` (1) | 26 550 / 26 550 | 6 606 / 7 080 | 1 770 / 1 770 |
| `MISSING` (0) | 0 / 45 | 0 / 12 | 0 / 3 |
| `LIKELY_OCCLUDED` (4) | 0 / 225 | 0 / 60 | 0 / 15 |

Every state without a usable pose yields zero valid sensors; every `OBSERVED`
instance yields 15 valid bend sensors and a valid IMU. The only shortfall inside
`OBSERVED` is spread, from the frozen 15 degree conditioning rule.

## Ranges

| Channel | min | median | max |
|---|---|---|---|
| `bend_angle_deg` | 0.139 | 21.005 | 113.650 |
| `bend_normalized` | 0.00077 | 0.1167 | **0.6314** |
| `spread_angle_deg` | 0.005 | 12.669 | 97.604 |
| `spread_normalized` | 0.00003 | 0.0704 | **0.5422** |

All normalized values lie within `[0, 1]`: confirmed. The maxima being well
below 1.0 is the run-scale evidence that no min/max fitting was applied — a
pilot-fitted normalizer would place the observed maximum exactly at 1.0.

## Per-sensor extremes (degrees, both hands pooled)

| Sensor | min | median | max |
|---|---|---|---|
| `H_INDEX_PROXIMAL` | 12.9 | 42.1 | 76.6 |
| `H_INDEX_MIDDLE` | 0.1 | 11.0 | 113.7 |
| `H_MIDDLE_PROXIMAL` | 5.7 | 38.0 | 104.3 |
| `H_PINKY_PROXIMAL` | 2.7 | 34.3 | 104.3 |
| `H_SPREAD_THUMB_INDEX` | 0.0 | 45.9 | 88.4 |
| `H_SPREAD_INDEX_MIDDLE` | 0.0 | 4.2 | 74.1 |
| `H_SPREAD_MIDDLE_RING` | 0.1 | 13.2 | 97.6 |
| `H_SPREAD_RING_PINKY` | 0.0 | 7.8 | 71.7 |

The full 19-sensor table is in `virtual_glove_summary.json`. The
`H_INDEX_PROXIMAL` floor of 12.9 degrees is the documented TASK-005 proximal
reference-bone neutral, carried through uncalibrated as intended.

## Per track

| Track | bend min/median/max | spread min/median/max |
|---|---|---|
| LEFT | 0.139 / 24.487 / 108.302 | 0.013 / 14.911 / 97.604 |
| RIGHT | 0.307 / 18.001 / 113.650 | 0.005 / 10.202 / 71.284 |

## Optional ADC

| Channel | valid count | min | max |
|---|---|---|---|
| `bend_adc_12bit` | 26 550 | 3 | 2 586 |
| `spread_adc_12bit` | 6 606 | 0 | 2 220 |

816 channels carry the invalid sentinel `-1`. The observed span is far wider
than the retired prototype's ~850–1700 window, as intended for a full-scale
12-bit ideal encoding.

## IMU orientation integrity

| Diagnostic | Value |
|---|---|
| Verbatim copies confirmed | **1 770 / 1 770** |
| Quaternion norm error, max | 4.11e-08 |
| Rotation orthogonality error, max | 8.76e-08 |
| Rotation determinant error, max | 9.24e-08 |

The three error figures are identical to TASK-005A's, as they must be for a
verbatim copy; they sit at the float32 storage round-trip floor.

---

# Temporal Diagnostics

Derived angular velocity, unsmoothed:

| | |
|---|---|
| Valid samples | **1 734 / 1 788 possible** |
| Unavailable at frame 0 of each track | 36 |
| Unavailable from invalid palm orientation | 18 |
| Accounting | 36 + 18 + 1 734 = 1 788 ✓ |

The 18 invalid-palm frames (`s01_sign0174` RIGHT) run contiguously to the end of
that clip, so the gyro gap equals the palm gap exactly with no additional
"frame after the gap" loss. No pair was ever formed across the gap.

| Magnitude | rad/s | deg/s |
|---|---|---|
| p50 | 0.922 | 53 |
| p95 | 8.653 | 496 |
| p99 | 19.957 | 1 143 |
| max | **53.489** | **3 065** |

| Threshold | Samples above |
|---|---|
| > 10 rad/s (573 deg/s) | 70 (4.04 %) |
| > 20 rad/s (1 146 deg/s) | 18 (1.04 %) |
| > 30 rad/s (1 719 deg/s) | 8 (0.46 %) |

**The known TASK-005 orientation jumps propagate directly into the derived gyro
and are not hidden.** The single largest sample is `s01_sign0175` LEFT frame 27
at 53.489 rad/s, which is exactly **102.2 degrees between adjacent frames** —
the same event TASK-005A documented as its largest palm-orientation change. The
p99 of 1 143 deg/s is already at the edge of plausible human wrist rate, and the
maximum of 3 065 deg/s is well beyond it. These samples are reported, not
filtered: any consumer of the gyro channel must treat its upper tail as carrying
reconstruction artefacts, and deciding what to do about that is a later design
decision, not one to make silently here.

---

# Tests

```
python -m unittest discover -s tests -p 'test_*.py'
Ran 344 tests
OK

python -m compileall -q evaluation tracking kinematics virtual_glove scripts tests
0 errors
```

**344 passed, 0 failed, 0 errors** — 275 pre-existing (all still passing) plus
**69 new virtual-glove tests**. All synthetic; none reads the pilot, the WiLoR
checkpoint or MANO assets.

Coverage of the required list:

| # | Requirement | Result |
|---|---|---|
| 1 | 0 deg bend -> normalized 0 | exact |
| 2 | 90 deg -> 0.5 | exact to 1e-12 |
| 3 | 180 deg -> 1 | exact |
| 4 | equivalent spread tests | exact; bend and spread proven to share one transfer |
| 5 | normalization monotonicity | strictly increasing across 8 angles |
| 6 | no pilot-derived normalization | batch-independence, pilot-range non-rescaling, no neutral offset |
| 7 | NaN bend -> invalid sensor | exactly 1 of 15 invalidated |
| 8 | NaN spread affects only that sensor | exactly 1 of 4; bend and IMU untouched |
| 9 | flexion preserved when strict flag false | 15 bends + IMU + 3 spreads retained |
| 10 | LEFT/RIGHT ordering preserved | values distinguishable per track; reorder refused |
| 11 | quaternion/matrix copied unmutated | bitwise equality; no evaluation basis applied |
| 12 | 12-bit boundaries | 0/2048/4095; sentinel -1; half-up rounding; monotone |
| 13 | exactly 19 Hall IDs | set equality with the specified IDs |
| 14 | exactly one palm IMU ID | `IMU_PALM` |
| 15 | every Hall entry `display_marker="H"` | 19/19 |
| 16 | IMU `display_marker="IMU"` | 1/1, no crossover |
| 17 | deterministic output | identical across repeated extraction |
| 18 | provenance alignment | frame index, timestamps, state codes, raw indices verbatim |
| 19 | no bridging for derived temporal signals | invalid orientation, non-adjacent frames and non-positive dt all refused |

Additional coverage: contract-violation raise and flag modes, source arrays not
mutated, missing required array refused, non-`wxyz` quaternion refused, output
is a copy not a view, stationary hand reads zero rate, actual timestamp delta
used, large jump reported rather than smoothed, NPZ round-trip with no pickled
objects, the 23-channel ML shape, and the absence of any stacked training tensor.

Verified at pilot scale as well: an independent re-run produced **18/18
byte-identical** `virtual_glove.npz`, and the TASK-005F kinematics inputs were
**unchanged (18/18)**.

---

# Limitations

1. **This stage adds no new measurement.** Every angle is a frozen TASK-005
   value re-expressed. Sensor accuracy is exactly kinematics accuracy, and every
   TASK-005 limitation — the proximal reference-bone neutral, unsigned spread,
   the 15 degree conditioning rule, the lack of ground truth — is inherited
   unchanged.
2. **The glove is ideal, not physical.** There is no magnetic hysteresis, no
   temperature drift, no noise floor, no cross-talk, no sampling jitter, no
   mounting slip, no calibration error. Nothing here predicts what a real Hall
   sensor at these sites would output.
3. **`logical_location` is nominal.** It names an anatomical site for future
   visualization; no 3D coordinate, mounting geometry or sensor footprint is
   specified.
4. **Spread availability is handshape-dependent.** 7.63 % of spread channels are
   invalid, concentrated by the frozen conditioning rule on the ring, pinky and
   middle fingers. TASK-005A showed this tracks the sign rather than the clip.
   Any downstream model must handle a systematically absent channel rather than
   assume uniform availability.
5. **The derived gyro carries reconstruction artefacts in its upper tail**, up
   to 3 065 deg/s. It is unsmoothed by design and is marked DERIVED.
6. **No accelerometer**, for the reasons above. The ideal glove's IMU is
   currently orientation plus angular rate only.
7. **Pilot scale.** 18 videos, 894 frames, 6 signs, 3 signers, one dataset, one
   camera setup, uniform 30 fps.

---

# Future Physical Glove Implications

The 19-Hall layout is **intentionally information-rich**. It is a research
instrument for deciding what hardware to build, not a claim that 19 sensors are
necessary.

Later experiments should ablate it — `19 -> 15 -> 10 -> 5` Hall channels — to
find how many a physical glove actually needs, and which. **No ablation is
performed in TASK-006A**, and the layout metadata is structured to make one
straightforward: every channel has a stable ID and a resolvable array slot, so
a subset can be selected by ID without touching this stage.

Three observations that a future hardware decision may want, stated as
observations only:

* The `H_INDEX_PROXIMAL` floor of 12.9 degrees means a physical MCP sensor would
  need offset calibration; the ideal model deliberately does not pre-remove it.
* Spread channels are the least reliably available (92.4 % versus 99.0 % for
  bend), so a glove betting on inter-finger sensing should expect dropouts on
  curled handshapes.
* The distal channels have the narrowest medians, which is where an ablation
  study might most usefully start.

---

# Reproducibility

```bash
git checkout 8646ec03cd0a220a42f8e06db7d7b1f4d52cdea9

python scripts/run_task006a_virtual_glove.py \
  --kinematics-run /home/hatim/graduation-project-runs/wilor_karsl_pilot_kinematics_task005f \
  --out-dir        /home/hatim/graduation-project-runs/virtual_glove_task006a \
  --strict-counts

python -m unittest discover -s tests -p 'test_*.py'
python -m compileall -q evaluation tracking kinematics virtual_glove scripts tests
```

| Artefact | Path |
|---|---|
| Package | `virtual_glove/{layout,signals,imu,extractor,io}.py` |
| Runner | `scripts/run_task006a_virtual_glove.py` |
| Tests | `tests/test_virtual_glove.py` |
| Output (git-ignored) | `/home/hatim/graduation-project-runs/virtual_glove_task006a` |
| Summary | `.../virtual_glove_summary.json` |
| Layout | `.../sensor_layout.json` and per sample |

CPU only; no checkpoint, MANO asset or video is needed. TASK-005 artifacts are
not overwritten: the glove stage writes to its own directory and opens the
kinematics run read-only.

---

# Verdict

**IDEAL VIRTUAL GLOVE IMPLEMENTED — READY FOR INDEPENDENT VALIDATION**

The 19-Hall + 1-IMU architecture is implemented against the frozen
`TASK-005-final-v2` contract for all 18 samples and 894 frames, with per-channel
Model-B validity that retains 3 826 sensor channels a whole-hand policy would
have discarded, an a-priori 0..180 normalizer with no dataset fitting, verbatim
IMU orientation, an honest accelerometer deferral, and zero contract violations.

Two things an independent validator should scrutinize first: whether the derived
gyro's upper tail (up to 3 065 deg/s, propagated unfiltered from known TASK-005
orientation jumps) is acceptable to carry forward, and whether the accelerometer
deferral argument holds. Neither is resolvable from within this task.

TASK-006D, TASK-008 and any LSTM work are not started.
