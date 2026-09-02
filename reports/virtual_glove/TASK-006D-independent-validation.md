# Task

TASK-006D — independent virtual-glove integration and final validation.

# Branch

`evaluation/task-006d-virtual-glove-validation`

Base: `main @ bf3d678f734cdf3fb62c6acdaf1cd774083df159`.

# Scope

This validation integrates the frozen TASK-006A production converter, the
independent TASK-006B sensor oracle, and TASK-006C extractor-independent QA.
It does not change TASK-004 or TASK-005, rerun WiLoR/tracking/kinematics, build
an LSTM tensor, train a model, or start TASK-008.

# Approach

The 167 expected-valid TASK-006B fixtures were sent through the actual
TASK-006A `extract_glove_sequence` entry point. Expected values came only from
the independent TASK-006B oracle. Production arrays are float32, so comparison
was performed against an explicitly float32-cast oracle; no observed error was
used to fit a tolerance.

The frozen TASK-005F pilot was converted once into the new external run
directory `/home/hatim/graduation-project-runs/virtual_glove_task006d`. The
TASK-006C CLI then checked the output against the read-only TASK-005F source.

# Evidence / Sources

- `reports/virtual_glove/TASK-006A-core-virtual-glove.md`;
- `reports/virtual_glove/TASK-006B-independent-sensor-benchmark.md`;
- `reports/virtual_glove/TASK-006C-virtual-glove-qa-tooling.md`;
- frozen `TASK-005-final-v2` output and the TASK-006 task contract;
- TASK-006A/B/C source commits recorded below.

# Files Changed

- `evaluation/virtual_glove_integration/adapter.py` — neutral production
  adapter, synthetic comparison, layout reconciliation, pilot accounting and
  gyro-frame proof;
- `evaluation/virtual_glove_integration/__init__.py` — adapter exports;
- `evaluation/virtual_glove_qa/contract.py` — read-only companion-layout
  loading, strict production metadata compatibility, and physical-template
  expansion to runtime identities;
- `evaluation/virtual_glove_qa/validator.py` — strict support for TASK-006A's
  nested normalization/ADC metadata and explicit `kinematics_*` provenance;
- `evaluation/virtual_glove/orientation.py` and `reference.py` — versioned
  body-frame benchmark helper; the old world-frame helper remains available as
  a labelled historical helper;
- `evaluation/virtual_glove/benchmark.py` — body/world gyro distinction in
  the independent self-check;
- `scripts/run_task006d_validation.py` — reproducible validation runner;
- `tests/test_task006d_virtual_glove_validation.py` — integration tests;
- this report and `TASK-006D-validation-results.json`.

No production sensor-transfer formula, TASK-004 code, TASK-005 code, dataset,
model, checkpoint, or generated run artifact was committed.

# How to Run

From this branch with the existing environment:

```bash
python scripts/run_task006a_virtual_glove.py \
  --kinematics-run /home/hatim/graduation-project-runs/wilor_karsl_pilot_kinematics_task005f \
  --out-dir /home/hatim/graduation-project-runs/virtual_glove_task006d \
  --strict-counts

python scripts/validate_task006_virtual_glove.py \
  --kinematics-run /home/hatim/graduation-project-runs/wilor_karsl_pilot_kinematics_task005f \
  --virtual-glove-run /home/hatim/graduation-project-runs/virtual_glove_task006d \
  --output-json /home/hatim/graduation-project-runs/task006d_qa/final-qa.json \
  --output-csv /home/hatim/graduation-project-runs/task006d_qa/final-distributions.csv

python scripts/run_task006d_validation.py \
  --kinematics-run /home/hatim/graduation-project-runs/wilor_karsl_pilot_kinematics_task005f \
  --virtual-glove-run /home/hatim/graduation-project-runs/virtual_glove_task006d
```

The first command reads TASK-005F and writes only to the ignored external
directory. It does not rerun WiLoR, tracking, or TASK-005.

# Evaluation

## Frozen Inputs

| Input | PR | Original commit(s) | Branch |
|---|---:|---|---|
| TASK-006A production | #18 | `c438fab3f81402b899e38d4990bb5f86973ae59d`, `8646ec03cd0a220a42f8e06db7d7b1f4d52cdea9`, `cf6fcb1e9568b215fe6c872d83b133bc27464eb6` | `opus/task-006a-core-virtual-glove` |
| TASK-006B benchmark | #19 | `f0e1cb215f4881a84c528f885a3f5ada6f6c1026` | `luna/task-006b-virtual-glove-benchmark` |
| TASK-006C QA | #20 | `f91361096e7e670fce6819599cc3be4ec3745f92` | `opus/task-006c-virtual-glove-qa-tooling` |

The exact integrated cherry-pick commits were, respectively, `a4edc62e4c65fd1303984ec20f5d432bb0f37cbb`, `ab6b6b016aecf27c5f401a43b387027c47e60bca`, `7918a72eb08e04291b7339de161fba462e8ce8a1`, `0716b403af9c46078aa8213470e991a399d4fc91`, and `e93057fa9e73d8693a4038eaabbfbb33cd03abf1`.

## Final Sensor Contract

The final version is `TASK-006-ideal-virtual-glove-v1`:

```text
per hand: 15 independent bend Hall channels
          4 independent adjacent-spread Hall channels
          1 palm IMU
```

That is 19 Hall channels and 20 logical sensing packages per hand. The
authoritative ML-facing representation is 15 bend-normalized values, 4
spread-normalized values, and 4 WXYZ quaternion components: 23 numerical
channels per hand and timestep, plus masks. ADC and gyro are optional
diagnostic/compatibility outputs; accelerometer is deferred.

## Layout Reconciliation

TASK-006A's `sensor_layout.json` is a reusable 20-definition physical template
per hand. The neutral QA compatibility layer reads that companion file and
expands it deterministically for the canonical tracks:

```text
20 physical placement definitions / hand template
40 runtime identities: LEFT.<template_id> and RIGHT.<template_id>
38 runtime Hall identities + 2 runtime IMU identities
```

The expansion is accepted only for an exactly 20-entry hand-independent
template; malformed or partially hand-labelled layouts still fail. Each
runtime identity retains its template source, array slot, hand, and sensor
type. The observed mapping was bijective with 40 unique identities and no
slot collision.

## Visualization Contract

Every Hall/magnetic entry retains `display_marker = "H"`; each palm IMU
retains `display_marker = "IMU"`. The layout carries the 15 finger/joint
locations, four adjacent web locations, and palm location needed by a later
visualizer. No visualizer is implemented here.

## Production-to-Benchmark Adapter

`evaluation/virtual_glove_integration/adapter.py` maps the independent
`KinematicsInput` fields to TASK-006A's existing input names and invokes
`virtual_glove.extract_glove_sequence`. It then exposes the production bend
and spread slots in the oracle's `[F,2,19]` Hall shape. It does not copy the
production normalization formula into the adapter and does not compute truth
from production output.

## Bend Validation

All 120 single-bend fixtures passed. This covers every one of 15 bend slots at
all eight values `0, 1, 30, 45, 90, 135, 179, 180` degrees for both tracks,
or 240 channel instances. The two multi-channel fixtures and the remaining
valid categories also passed. There were no valid-fixture failures.

## Spread Validation

All 32 single-spread fixtures passed. This covers all four adjacent pairs at
the same eight values for both tracks, or 64 channel instances. Partial
spread validity was retained channel-by-channel; no undefined spread value
was filled.

## Normalization

The QA validator found zero normalization violations. Production output obeys:

```text
bend_normalized = bend_angle_deg / 180
spread_normalized = spread_angle_deg / 180
```

The pilot maxima were approximately `0.631391` for bend and `0.542246` for
spread, demonstrating that pilot extrema were not rescaled to 1. No min/max
fitting, neutral subtraction, clipping, or per-signer calibration was found.

## Model-B Validity

The QA validator found zero mask/NaN propagation violations and 215 partial
channel examples. Strict `valid_kinematics = false` did not discard finite
bends or valid IMU orientation. Undefined spread channels remained NaN and
invalid; no interpolation or fabricated value was introduced.

## IMU Orientation

All valid synthetic orientation cases passed direct matrix/quaternion
passthrough. Production applies no TASK-005 comparison basis mapping. The
pilot contained 1,770 valid orientation copies. QA found:

| Check | Maximum |
|---|---:|
| Matrix orthogonality error | `8.761691e-08` |
| `abs(det(R)-1)` | `9.239450e-08` |
| Quaternion norm error | `4.114060e-08` |
| Matrix/quaternion element error | `8.165514e-08` |
| Matrix/quaternion angular disagreement | `0.0160933°` |
| Non-positive determinants | `0` |

All were within the TASK-006C QA limits of `1e-5` where applicable.

## Gyroscope Convention Analysis

The final optional gyro contract is body-frame angular velocity in the earlier
palm/body axes:

```text
R_delta_body = R_previous.T @ R_current
omega_body = log(R_delta_body) / delta_t
```

The original TASK-006B quaternion helper used
`q_current * conjugate(q_previous)`, which represents the world/camera-frame
delta `R_current @ R_previous.T`. This was proven algebraically and with
identity→single-axis, already-rotated local-axis, non-commuting composed,
LEFT/RIGHT, and varying-`dt` cases. The helper remains as a labelled historical
world-frame function; the benchmark sequence oracle now uses the new fixed
body-frame helper.

Production and independent body-frame results passed all four convention
cases with maximum absolute error `2.220446e-16 rad/s`. The largest observed
world/body difference was `3.212199 rad/s`, so the frame distinction is
material and is not hidden by single-axis identity tests.

## Accelerometer Decision

`DEFER ACCELEROMETER`. TASK-005 supplies no metric translation, WiLoR camera
translation is uncalibrated weak-perspective scale, second differentiation
would amplify reconstruction noise, and no gravity/specific-force convention
is frozen. Fabricating a six-axis signal would be less reproducible than
retaining a one-palm orientation IMU. No accelerometer values are emitted.

## ADC

ADC remains `OPTIONAL / NON-AUTHORITATIVE`. The 12-bit contract is full-scale
`0..4095` with round-half-up: `0 → 0`, `0.25 → 1024`, `0.5 → 2048`,
`0.75 → 3071`, `1 → 4095`; invalid channels use `-1`. The QA validator found
zero ADC violations. The retired prototype's 850–1700 range is not used.

## Invalid Inputs

The independent boundary rejected all 12 corruption fixtures with their
expected reasons. Directly invoking the frozen production converter without
that neutral boundary accepted six temporal/provenance corruptions, which is
expected because TASK-006A assumes a validated TASK-005 source. The final
integration path validates before invoking production; no invalid fixture can
silently enter conversion.

## TASK-006C QA

The exact requested CLI produced:

```text
schema passed       : True
alignment passed    : True
provenance passed   : True
layout passed       : True
normalization issues: 0
validity issues     : 0
rotation issues     : 0
ADC present         : True
verdict             : VIRTUAL-GLOVE QA TOOLING READY
```

The compatibility changes are limited to reading TASK-006A's existing
companion layout and metadata locations. They do not relax coverage, markers,
mask, finite-value, rotation, alignment, or provenance checks.

## Pilot Results

The frozen TASK-005F source and new TASK-006D output agree on all required
alignment/provenance fields:

| Quantity | Result |
|---|---:|
| Samples | `18 / 18` |
| Frames | `894 / 894` |
| Hand instances | `1,788` |
| Bend valid | `26,550 / 26,820` |
| Spread valid | `6,606 / 7,152` |
| IMU valid | `1,770 / 1,788` |
| Partial strict-false instances | `215` |
| Channels retained on strict-false instances | `3,826` |
| Alignment mismatches | `0` |
| Provenance failures | `0` |
| Source SHA-256 records | `18` |

The 3,826 retained count includes finite bend/spread channels and valid IMU
channels on strict-false instances, consistent with Model-B channel retention.

## Temporal Diagnostics

The derived gyro had 1,734 valid magnitudes. Its pilot distribution was:

| Quantity | rad/s | deg/s |
|---|---:|---:|
| p99 | `19.957385` | `1143.474` |
| maximum | `53.489330` | `3064.713` |

The maximum is the previously documented approximately 102.2° adjacent-frame
TASK-005 palm-orientation jump. It is a mathematically valid unfiltered
derived rate under the frozen timestamps and body-frame operation, not a
reason to clip, smooth, or delete the sample. Its usefulness as an ML feature
remains unproven, so gyro is optional and not part of the authoritative 23
channel primary ML vector.

## Numeric Precision / Dtype

TASK-006B's float64 oracle self-check remains locked at `1e-10` algebraic
tolerances. TASK-006A serializes the primary arrays as float32. On the 167
valid fixture comparison:

| Measurement | Maximum |
|---|---:|
| Raw float64 oracle vs production view | `2.946878e-08` |
| Float32 quantization effect | `2.946878e-08` |
| Serialized production vs explicitly cast oracle | `0.0` |

The serialized criterion is exact equality after the declared float32 cast,
including NaN masks and direct orientation passthrough.

## Acceptance Criteria

| Criterion | Result |
|---|---|
| A — 15 bend + 4 spread + 1 IMU / hand | PASS |
| B — independent channel mapping | PASS |
| C — layout and visualization contract | PASS |
| D — bend normalization | PASS |
| E — spread normalization and masks | PASS |
| F — Model-B validity | PASS |
| G — orientation passthrough | PASS |
| H — LEFT/RIGHT consistency | PASS |
| I — ADC compatibility | PASS |
| J — 12 invalid inputs | PASS |
| K — 167 valid fixtures | PASS |
| L — exact TASK-005 alignment | PASS |
| M — rotation/quaternion QA | PASS |
| N — body-frame gyro convention | PASS |
| O — accelerometer decision | PASS (explicitly deferred) |
| P — pilot regression/accounting | PASS |
| Q — tests | PASS |

## Results

Synthetic validation passed `167 / 167` valid fixtures and `12 / 12` invalid
boundary cases. The independent TASK-006B self-check remained green with
`179` total fixtures. The frozen pilot passed all TASK-006C checks.

## Failures / Limitations

No blocking validation failure remains. The six direct-production invalid
acceptances described above reinforce the need for the neutral input boundary;
they are not hidden. The derived gyro has large diagnostic outliers inherited
from TASK-005 and is not yet justified as an ML feature. Accelerometer remains
deferred. Hall/IMU values are ideal representations, not calibrated hardware
measurements.

## Performance

This task does not claim a model throughput benchmark. Conversion was run over
18 frozen clips without rerunning upstream inference. The reported gyro and
sensor counts are data-contract diagnostics; no new timing comparison was
introduced.

## Comparison

This is an integration validation, not a comparison against another pose
model. Production values were checked against an independent mathematical
oracle and an extractor-independent serialized-output QA tool.

## Recommendation

KEEP

The ideal virtual-glove representation satisfies the frozen contract and is
ready for the next planned stage, with gyro optional and accelerometer
explicitly deferred.

## Reproducibility

Use the commands above with the external paths:

```text
TASK-005 input: /home/hatim/graduation-project-runs/wilor_karsl_pilot_kinematics_task005f
TASK-006D run:  /home/hatim/graduation-project-runs/virtual_glove_task006d
QA JSON:       /home/hatim/graduation-project-runs/task006d_qa/final-qa.json
QA CSV:        /home/hatim/graduation-project-runs/task006d_qa/final-distributions.csv
```

The source SHA-256 is checked before provenance comparison. The output is
directory-per-sample NPZ/JSON data and remains outside Git. Python was
`3.14.4`; NumPy was `2.5.2`.

## Next Steps

Proceed only to the already planned next milestone after this PR is reviewed.
Do not treat optional gyro as an authoritative ML feature and do not add an
accelerometer without a new translation/gravity contract.

# Final Verdict

TASK-006 VALIDATED — READY FOR TASK-008
