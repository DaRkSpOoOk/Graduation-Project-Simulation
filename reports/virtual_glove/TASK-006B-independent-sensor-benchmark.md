# Task

TASK-006B — Independent ideal virtual-glove sensor benchmark.

# Branch

`luna/task-006b-virtual-glove-benchmark`, based on `main` at
`bf3d678f734cdf3fb62c6acdaf1cd774083df159`.

# Scope

This task defines and self-validates an independent mathematical benchmark
for the ideal TASK-006 virtual-glove representation. It does not implement
production sensor conversion, physical sensor placement, magnetic simulation,
tracking, kinematics, recognition, or TASK-006A. No TASK-006A branch, code,
tests, or generated outputs were inspected.

# Approach

The benchmark uses deterministic in-memory TASK-005-like arrays and a small
reference oracle. The oracle exposes the expected contract for a future
production adapter without importing `virtual_sensors/` or any model
implementation. It validates the logical sensor catalog, exact normalization,
channel-level masks, direct IMU orientation passthrough, optional ADC transfer,
and optional analytic gyro behavior.

# Evidence / Sources

- `README.md` and `AGENTS.md` repository scope and artifact rules;
- `reports/README.md` reporting requirements;
- `reports/kinematics/TASK-005F-final-validation.md` final TASK-005 contract;
- `pose/common/schema.py` common pose field conventions;
- the TASK-006B frozen contract in the task specification.

The benchmark was finalized without inspecting TASK-006A. The mathematical
truth is generated from literal fixture parameters and independent rotation,
normalization, and quaternion helpers.

# Files Changed

- `evaluation/virtual_glove/contract.py` — versioned input contract, sensor
  definitions, catalog and hard validation;
- `evaluation/virtual_glove/synthetic.py` — deterministic valid and invalid
  fixtures;
- `evaluation/virtual_glove/orientation.py` — independent matrix/quaternion
  and gyro helpers;
- `evaluation/virtual_glove/reference.py` — independent ideal sensor oracle,
  comparison adapter, masks and optional ADC output;
- `evaluation/virtual_glove/benchmark.py` — self-check runner;
- `evaluation/virtual_glove/__init__.py` — public benchmark exports;
- `scripts/run_task006b_benchmark.py` — reproducible CLI;
- `tests/test_task006b_virtual_glove_benchmark.py` — benchmark tests;
- `reports/virtual_glove/TASK-006B-independent-sensor-benchmark.md` — this
  report;
- `reports/virtual_glove/TASK-006B-benchmark-results.json` — compact result.

No production package, common schema, dataset, model, or generated large
artifact was changed.

# How to Run

From the repository root with the existing environment:

```bash
python scripts/run_task006b_benchmark.py
python -m unittest tests.test_task006b_virtual_glove_benchmark
python -m unittest discover -s tests -p 'test_*.py'
python -m compileall -q evaluation tracking kinematics scripts tests
```

The CLI writes only the compact result JSON named above. All fixture arrays
are generated in memory.

# Evaluation

The required per-hand layout is 15 bend Hall channels, 4 adjacent-spread Hall
channels, and 1 palm IMU: 19 Hall channels plus 1 IMU per hand. The normalized
Hall signal is authoritative. Optional ADC counts and gyro values are
compatibility helpers and do not replace the normalized representation.

# Objective

Provide TASK-006D with an independent, deterministic oracle and test suite for
sensor count, channel mapping, normalization, masks, orientation, corruption
handling, and optional rotational velocity.

# Independence

The code under `evaluation/virtual_glove/` imports only NumPy and Python
standard-library functionality. It does not import a production virtual-glove
module, TASK-006A code, WiLoR, MediaPipe, or generated model output. The
expected sensor values are computed from fixture input parameters and direct
contract equations, not from a competing implementation.

# Ideal Sensor Layout

The schema version is `TASK-006-ideal-virtual-glove-v1`. Track order is
`LEFT, RIGHT`; finger order is `thumb, index, middle, ring, pinky`; bend order
is `proximal, middle, distal`; spread order is
`thumb-index, index-middle, middle-ring, ring-pinky`.

For each hand the stable IDs are:

```text
{HAND}.bend.{thumb|index|middle|ring|pinky}.{proximal|middle|distal}
{HAND}.spread.{thumb-index|index-middle|middle-ring|ring-pinky}
{HAND}.palm_imu
```

Each bend/spread definition has `kind=HALL` and `display_marker="H"`. The
palm definition has `kind=IMU` and `display_marker="IMU"`. The catalog contains
40 definitions total: 38 Hall and 2 IMU. The validator rejects duplicate or
missing IDs, unexpected/wrong source assignments, incorrect counts, and
incorrect display markers.

# Synthetic Inputs

The fixture catalog contains 179 deterministic cases:

| Category | Count |
|---|---:|
| neutral | 1 |
| single bend | 120 = 15 channels × 8 values |
| single spread | 32 = 4 pairs × 8 values |
| multi-channel | 2 |
| LEFT/RIGHT mirror-equivalent | 1 |
| orientation | 8 |
| validity | 3 |
| intentional invalid input | 12 |
| total | 179 |

The eight known values for every bend channel and every spread pair are
`0, 1, 30, 45, 90, 135, 179, 180` degrees. Multi-channel fixtures combine
distinct values across all 15 bend and 4 spread channels. A separate fixture
uses equal local LEFT/RIGHT values with distinct proper palm orientations.
The multi-frame fixture uses timestamps `0.0, 0.5, 1.0` and frame indices
`0, 1, 2`.

# Bend Truth

For each finite in-contract bend value the independent expected signal is:

```text
normalized_bend = bend_deg / 180.0
```

The 15 logical channels retain their exact finger/joint source assignment.
There is no clinical reinterpretation, neutral offset, pilot min/max scaling,
or clipping.

# Spread Truth

Each of the four adjacent spread values uses the same direct equation:

```text
normalized_spread = spread_deg / 180.0
```

Spread channels are independent of bend-channel masks. A missing or undefined
spread value remains invalid and is not replaced by a neighboring channel or
frame.

# Normalization Truth

The locked mapping is:

| Angle | Normalized |
|---:|---:|
| 0° | 0.0 |
| 1° | 0.005555555555555556 |
| 30° | 0.16666666666666666 |
| 45° | 0.25 |
| 90° | 0.5 |
| 135° | 0.75 |
| 179° | 0.9944444444444445 |
| 180° | 1.0 |

Finite inputs outside `[0, 180]` degrees and non-finite inputs are hard
validation errors. The reference does not clip them. Monotonicity and
dataset-statistics independence are self-checked.

# Optional ADC Truth

The optional 12-bit compatibility transfer uses:

```text
adc = floor(normalized * 4095 + 0.5)
```

This is deterministic round-half-up for non-negative values. Endpoints are
`0 → 0` and `1 → 4095`; invalid channels use the explicit sentinel `-1` and
cannot contain a finite normalized value. The normalized float remains the
authoritative signal. The self-check verifies `[0, 1024, 2048, 3071, 4095]`
for `[0, .25, .5, .75, 1]` and verifies monotonicity.

# Validity Cases

The reference maintains independent Hall masks and one palm-IMU mask:

| Case | Expected result |
|---|---|
| fully valid | all 19 Hall channels and the IMU are valid and finite |
| partial spread | all 15 bend channels and IMU remain valid; only the affected spread channel is invalid/NaN |
| whole palm invalid | finite Hall channels are preserved; orientation matrix/quaternion are missing and IMU is invalid |
| missing tracking pose | all channels for that hand are invalid/NaN; no value is fabricated |

The whole-palm case intentionally demonstrates channel-level preservation of
usable Hall data while invalidating the dependent orientation channel. The
missing-pose case requires the source arrays to carry no finite derived values.

# Orientation / IMU

The reference accepts only finite proper rotation matrices with locked
float64 tolerances. It supports identity, 90-degree X/Y/Z, 180-degree X/Y/Z,
and the composed `Rz @ Ry @ Rx` rotation `(25°, -40°, 70°)`. Quaternions are
WXYZ, normalized, and checked against their matrices. q and -q are treated as
the same rotation for comparison.

The ideal IMU output is direct passthrough of the valid TASK-005 matrix and
WXYZ quaternion. No TASK-005 synthetic benchmark convention mapping is applied
to stored sensor orientation. The self-check verifies direct passthrough and
matrix/quaternion equivalence.

# Optional Gyroscope Truth

The optional helper computes a rotation-vector angular velocity from

```text
q_delta = q_current * conjugate(q_previous)
omega = axis(q_delta) * angle(q_delta) / delta_seconds
```

It returns radians per second and uses strictly increasing timestamps. The
first row is NaN because no predecessor exists, and invalid pairs remain NaN;
gaps are never bridged. Independent checks include:

- identity → 90° about Z in 1 second: `[0, 0, π/2]` rad/s;
- identity → 180° about X in 2 seconds: `[π/2, 0, 0]` rad/s.

Gyro is optional and is not required for the benchmark readiness verdict.

# Invalid Inputs

The 12 corruption fixtures all hard-fail with deterministic error codes:

| Fixture | Rejection reason |
|---|---|
| bend below zero | `flexion_deg_outside_0_180` |
| bend above 180 | `flexion_deg_outside_0_180` |
| spread below zero | `adjacent_spread_deg_outside_0_180` |
| spread above 180 | `adjacent_spread_deg_outside_0_180` |
| malformed shape | `adjacent_spread_deg_shape` |
| non-monotonic timestamps | `timestamps_non_monotonic` |
| duplicate frame index | `frame_index_duplicate` |
| non-finite valid orientation | `invalid_palm_rotation` |
| invalid quaternion norm | `invalid_palm_quaternion` |
| wrong track order | `wrong_track_order` |
| missing provenance | `missing_or_malformed_provenance` |
| finite values on missing pose | `missing_pose_has_finite_derived` |

No invalid input is silently clipped, normalized, filled, or accepted as a
plausible sensor frame.

# Tolerances

These thresholds were locked before inspecting TASK-006A:

| Check | Tolerance |
|---|---:|
| angle algebra | `1e-10` degrees |
| orientation angular comparison | `1e-10` degrees |
| matrix orthogonality | `1e-10` |
| matrix determinant error | `1e-10` |
| quaternion norm error | `1e-10` |
| matrix/quaternion element error | `1e-10` |
| gyro analytic comparison | `1e-10` rad/s |
| ADC count comparison | exact (`0` counts) |

They are strict float64 algebra tolerances, not empirical production pass/fail
thresholds.

# Self-Check Results

The independent CLI self-check passed all checks:

| Result | Value |
|---|---:|
| total fixtures | 179 |
| valid fixtures | 167 |
| invalid fixtures | 12 |
| Hall sensors | 38 total, 19 per hand |
| IMU sensors | 2 total, 1 per hand |
| bend coverage | 15/15 channels × 8 values |
| spread coverage | 4/4 pairs × 8 values |
| invalid-input rejection | 12/12 |
| validity/mask checks | PASS |
| orientation/quaternion checks | PASS |
| optional ADC checks | PASS |
| optional gyro checks | PASS |
| deterministic output checks | PASS |

The compact machine result is
`reports/virtual_glove/TASK-006B-benchmark-results.json`.

# Results

The benchmark is self-consistent and ready to serve as the independent
TASK-006D comparison oracle. Its adapter contract accepts a future production
result only when sensor catalog, masks, normalized values, direct orientation,
and optional compatibility fields agree with this reference.

# Failures / Limitations

No benchmark self-check failed. This is an ideal mathematical benchmark; it
does not model magnetic-field cross-talk, sensor noise, Hall magnet placement,
ADC quantization error beyond the optional deterministic mapping, glove fit,
occlusion, or real hardware calibration. It does not establish clinical
goniometry.

No accelerometer benchmark is included. A defensible accelerometer oracle would
first require a translational coordinate convention, gravity convention,
sampling assumptions, and a decision about differentiating position versus
using an independently specified inertial trajectory. Those assumptions are
not frozen by TASK-005.

The optional gyro helper covers rotational velocity only. It does not smooth
or bridge missing frames and is not a required production contract yet.

# Performance

All fixtures are small and generated in memory. The benchmark is CPU-only and
makes no GPU, model-throughput, or dataset-processing claim. No KArSL video,
checkpoint, raw pose array, or generated binary artifact is produced.

# Comparison

This branch intentionally does not compare against TASK-006A and does not
select a production sensor implementation. TASK-006D can adapt a production
output to `IdealSensorOutput` and call `compare_sensor_outputs()` without
changing the frozen oracle.

# Reproducibility

- Base commit: `bf3d678f734cdf3fb62c6acdaf1cd774083df159`.
- Branch: `luna/task-006b-virtual-glove-benchmark`.
- Python: `3.14.4`.
- NumPy: `2.5.2`.
- No random seed or external dataset is required.
- Fixture parameters and known rotations are literal deterministic values.
- Commands are listed under **How to Run**.
- The JSON result records the schema version, fixture counts, coverage,
  locked tolerances, checks, and all invalid-fixture rejection outcomes.

# Recommendation

KEEP the independent benchmark contract for TASK-006D validation. Do not
promote its reference oracle into production sensor conversion without a later
review of the physical sensor assumptions.

# Next Steps

TASK-006D may connect this benchmark to the independently developed production
sensor adapter. TASK-006A remains outside this branch and was not inspected.

# Final Verdict

VIRTUAL-GLOVE BENCHMARK READY
