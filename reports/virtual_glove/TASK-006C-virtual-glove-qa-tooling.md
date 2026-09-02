# TASK

TASK-006C — virtual-glove QA and data-contract tooling.

# Branch

`opus/task-006c-virtual-glove-qa-tooling`

Base: `main @ bf3d678f734cdf3fb62c6acdaf1cd774083df159`.

# Scope

This task adds extractor-independent validation for ideal TASK-006 virtual-
glove artifacts. It does not implement production sensor mathematics, Hall
transfer modeling, IMU simulation, smoothing, or repair. TASK-005 files are
opened read-only and no TASK-005 source was modified.

# Objective

Validate serialized TASK-006 outputs, their exact TASK-005 alignment and
provenance, channel-level validity/NaN behavior, the two-hand sensor layout,
normalization, orientation quality, optional ADC/gyro channels, temporal
diagnostics, and deterministic JSON/CSV summaries.

# Contract

The canonical run layout follows the repository's directory-per-sample
convention:

```text
<TASK006_RUN>/<sample_id>/virtual_glove.npz
<TASK006_RUN>/<sample_id>/virtual_glove_meta.json
```

The required arrays are:

| Field | Shape |
|---|---|
| `bend_angle_deg` | `[F,2,5,3]` |
| `bend_normalized` | `[F,2,5,3]` |
| `bend_valid` | `[F,2,5,3]`, boolean |
| `spread_angle_deg` | `[F,2,4]` |
| `spread_normalized` | `[F,2,4]` |
| `spread_valid` | `[F,2,4]`, boolean |
| `imu_rotation_matrix` | `[F,2,3,3]` |
| `imu_quaternion_wxyz` | `[F,2,4]` |
| `palm_imu_valid` | `[F,2]`, boolean |
| `frame_index` | `[F]` |
| `timestamp_seconds` | `[F]` |
| `tracking_state_code` | `[F,2]` |
| `source_raw_detection_index` | `[F,2]` |

Optional arrays are `bend_adc_12bit [F,2,5,3]`,
`spread_adc_12bit [F,2,4]`, `imu_angular_velocity_rad_s [F,2,3]`, and its
boolean `imu_angular_velocity_valid [F,2]` mask. Optional gyro data and its
mask must be supplied together.

The metadata must identify TASK-006, the sample, frame count, and canonical
`LEFT, RIGHT` track order. A fixed normalization contract must declare a
180-degree divisor and `[0,1]` range. A source provenance object must identify
the TASK-005 sample and SHA-256 of its `hand_kinematics.npz` input.

The canonical sensor layout contains 40 entries for the two exported tracks:
per hand, exactly 15 bend Hall/magnetic sensors, 4 spread Hall/magnetic
sensors, and 1 palm IMU. This is 19 Hall channels and 20 total sensor entries
per hand. Each entry has a unique `sensor_id`, valid sensor type, non-empty
logical location, description, and the required `display_marker` (`H` for
Hall/magnetic entries and `IMU` for palm IMUs).

# Architecture

- `evaluation/virtual_glove_qa/contract.py` defines the serialized schema,
  layout parser, metadata checks, deterministic sample loading, and read-only
  TASK-005 source checks.
- `evaluation/virtual_glove_qa/validator.py` orchestrates run-level checks,
  alignment, provenance, validity/NaN checks, normalization, rotation QA,
  ADC QA, distributions, temporal diagnostics, and summary writers.
- `scripts/validate_task006_virtual_glove.py` is the command-line entry point.
- `tests/test_task006c_virtual_glove_qa.py` creates small deterministic NPZ /
  JSON runs and mutates one contract property per negative test.

The implementation reuses the existing TASK-005 QA rotation helper for
matrix/quaternion comparison. It does not import or call production
kinematics extraction or sensor-generation mathematics.

# Alignment

For every common sample, the validator requires exact equality of:

- sample ID set and frame count;
- `frame_index`;
- `timestamp_seconds`;
- `track_order`;
- `tracking_state_code` versus TASK-005 `tracking_state_code`;
- `source_raw_detection_index` versus TASK-005 `source_raw_detection_index`.

Mismatches are listed with sample, field, frame position, frame index, track,
and both observed values where applicable. Artifact-only sample directories
are also surfaced when a companion NPZ or metadata file is missing.

# Sensor Layout QA

The layout validator checks exact two-hand counts, unique IDs, duplicate
logical assignments, complete five-finger/three-joint bend coverage, all four
adjacent spread pairs, one palm IMU per track, sensor type, logical location,
description, and display markers. Layout failures contribute directly to the
overall failed verdict; they are hard contract failures for later
visualization.

# Normalization QA

For every `bend_valid` or `spread_valid` channel, the validator checks finite
angle/normalized values, `0 <= normalized <= 1`, angle range `[0,180]`, and
`normalized ≈ angle_deg / 180` with absolute and relative tolerance `1e-6`.
It never clamps, rewrites, or fits values. Violation records contain the
sample ID, frame index, track, logical channel, observed values, expected
value, and error.

Metadata is recursively inspected for run-/sample-specific or fitted
min/max normalization declarations. Such evidence is reported and fails the
normalization contract. No normalization parameters are estimated from the
data.

# Model-B Validity

Validity is checked independently for bend channels, spread channels, palm
orientation, and optional gyro data:

- a valid channel must contain finite values;
- an invalid angle channel must contain NaN angle and NaN normalized values;
- an invalid palm orientation must contain all-NaN matrix and quaternion;
- an invalid optional gyro channel must contain all-NaN angular velocity;
- finite TASK-005 source channels must retain valid output masks, while
  source NaNs must remain invalid in TASK-006.

The validator deliberately does not define hand validity as the conjunction
of all channels. A strict-false TASK-005 hand with finite bend values, valid
palm orientation, and only a partial spread NaN state is accepted when each
channel's own mask and NaN state are correct. Such partial instances are
listed as diagnostics.

# Rotation QA

For each valid `palm_imu_valid` orientation, the summary reports
orthogonality error (`R.T @ R - I`), determinant and absolute determinant
error, non-positive determinant records, quaternion norm error,
matrix/quaternion element error, angular disagreement, and worst
sample/frame/track references. The hard tolerances are:

| Check | Tolerance |
|---|---:|
| Matrix orthogonality | `1e-5` |
| `abs(det(R)-1)` | `1e-5` |
| Quaternion norm error | `1e-5` |
| Matrix/quaternion element consistency | `1e-5` |

Quaternion sign ambiguity is handled by comparing the represented rotation,
not its raw sign. Invalid orientations are not silently converted into valid
ones.

# ADC QA

ADC fields remain optional. When present, the metadata must declare a 12-bit
linear full-scale transfer with code range `0..4095` and a code tolerance.
Valid channels are checked for range and agreement with their normalized
value. Invalid channels must be NaN (or an explicitly declared invalid
sentinel for integer arrays). ADC output is never synthesized or repaired.

# Temporal Diagnostics

Adjacent-frame absolute deltas are reported independently for all 15 bend
channels and 4 spread channels on each track, palm orientation, and each
optional gyro component. Each summary contains `count`, `mean`, `p95`, `p99`,
and a maximum event with sample/frame/track/channel/sensor reference.

Only physically adjacent rows with both endpoint masks valid are compared.
Missing values are never bridged, interpolated, smoothed, forward-filled, or
used to fail a run solely because a large descriptive jump exists.

# Distributions

The JSON summary and compact CSV provide `count`, `min`, `p1`, `p50`, `p95`,
`p99`, `max`, and missing count for bend angle/normalized channels by
`LEFT`/`RIGHT`, finger, joint, and sensor ID; spread channels by track,
adjacent pair, and sensor ID; and optional gyro components by track and IMU
sensor ID. No full-frame CSV is emitted by default.

# CLI

```bash
python scripts/validate_task006_virtual_glove.py \
  --kinematics-run <TASK005_RUN> \
  --virtual-glove-run <TASK006_RUN> \
  --output-json <SUMMARY_JSON> \
  --output-csv <SUMMARY_CSV>
```

Exit code `0` means `VIRTUAL-GLOVE QA TOOLING READY`; exit code `1` means the
run was read and failed one or more QA checks; exit code `2` means an input
run could not be read at all. JSON is strict, sorted-key, indented output.
CSV uses a fixed header, stable row order, and Unix line endings.

# Tests

The synthetic suite contains 20 tests covering:

- valid complete and valid partial-spread runs;
- malformed shape, missing sensor, duplicate sensor ID, missing `H` marker,
  missing `IMU` marker;
- wrong and out-of-range normalization, run-specific min/max metadata;
- valid partial spread NaN, illegal NaN;
- provenance and frame mismatch;
- rotation and quaternion faults;
- optional ADC mismatch and out-of-range code;
- deterministic JSON, deterministic CSV, and CLI execution.

Repository-wide verification passed:

```text
python -m unittest discover -s tests -p 'test_*.py'   # 295 tests, OK
python -m compileall -q evaluation tracking kinematics scripts tests       # OK
```

# Files Changed

- `evaluation/virtual_glove_qa/__init__.py`
- `evaluation/virtual_glove_qa/contract.py`
- `evaluation/virtual_glove_qa/validator.py`
- `scripts/validate_task006_virtual_glove.py`
- `tests/test_task006c_virtual_glove_qa.py`
- `reports/virtual_glove/TASK-006C-virtual-glove-qa-tooling.md`

# How to Run

Run the CLI above against a TASK-005 directory and a TASK-006 directory. For
the synthetic/unit checks, run the test and compile commands in the Tests
section. No dataset, video, checkpoint, or generated run artifact is needed
or committed.

# Evaluation

The contract and diagnostics were evaluated with deterministic in-memory
arrays serialized to temporary directory-style runs. The fixture uses four
frames, two tracks, all 15 bend channels, all four spread channels, two palm
IMUs, optional ADC, and optional gyro. Every negative fixture failed the
intended section with a concrete sample/frame/channel reference.

# Results

All 20 TASK-006C synthetic tests and all 295 repository tests passed. The
validator produced stable JSON and CSV bytes for repeated validation of the
same input. The current repository has no TASK-006 production run to score;
the tooling is ready to run once TASK-006 outputs exist.

# Failures / Limitations

- This is QA and contract tooling only; it does not validate the physical
  correctness of an angle-to-Hall transfer model or derive sensor values.
- Run-specific normalization detection is metadata/contract evidence based;
  the validator does not statistically fit or infer hidden parameters.
- The canonical layout records one explicitly identified sensor entry per
  track, so the two-hand export has 40 layout entries. A future shared
  per-hand layout representation would need an explicit adapter rather than
  being silently guessed.
- Temporal jump values are descriptive and do not establish an anatomical
  acceptance threshold.

# Performance

Validation is linear in serialized samples, frames, and the fixed channel
count. It loads each NPZ with `allow_pickle=False` and stores only compact
diagnostic lists/statistics; it does not materialize a full-frame CSV.

# Comparison

This implementation is extractor-independent QA infrastructure. It consumes
TASK-005 fields only for alignment, provenance, finite/NaN propagation, and
orientation source state. It does not compete with or alter TASK-005A
production kinematics, TASK-005B benchmarking, or TASK-005 QA thresholds.

# Recommendation (KEEP / REVISE / REJECT / NEEDS MORE EVALUATION)

KEEP. The tooling is ready for TASK-006 output review; run-level readiness
still requires executing it against the eventual production virtual-glove
run.

# Reproducibility

- Branch: `opus/task-006c-virtual-glove-qa-tooling`
- Base: `bf3d678f734cdf3fb62c6acdaf1cd774083df159`
- Python: `3.14.4`
- NumPy: `2.5.2`
- OS/runtime: Linux x86_64 under WSL2, 16 logical CPUs; CPU-only synthetic
  validation.
- Dataset/sample IDs: synthetic `sample_a`; no external dataset or video.
- Seed: none; fixtures use deterministic analytic values and no RNG.
- Commands:
  - `python -m unittest tests/test_task006c_virtual_glove_qa.py`
  - `python -m unittest discover -s tests -p 'test_*.py'`
  - `python -m compileall -q evaluation tracking kinematics scripts tests`

# Next Steps

Run the CLI against the first TASK-006 production artifact, preserve the raw
TASK-005 and virtual-glove runs, review any reported channel-level failures,
and attach the resulting compact JSON/CSV to the TASK-006 integration PR.

# Final Verdict

VIRTUAL-GLOVE QA TOOLING READY
