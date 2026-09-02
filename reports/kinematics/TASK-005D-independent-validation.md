# Task

TASK-005D — Independent integration and validation of hand kinematics.

# Branch

`evaluation/task-005d-kinematics-validation`

# Scope

This task integrates the frozen TASK-005A production implementation, the
independent TASK-005B synthetic benchmark, and the TASK-005C QA tooling in a
temporary validation branch. It connects the benchmark to the production
math, validates the existing TASK-005A pilot artifacts, and records contract
disagreements for follow-up.

No WiLoR inference, TASK-004 tracking, threshold tuning, smoothing,
interpolation, benchmark-truth changes, or production kinematics changes were
performed. TASK-006 was not started.

# Approach

The exact TASK-005A, TASK-005B, and TASK-005C commits were cherry-picked onto
the frozen `main` base without conflicts. The adapter calls
`kinematics.compute_hand_kinematics` for each synthetic 21-joint frame and
only maps its field names to the frozen TASK-005B result contract. It contains
no duplicated production formula.

The pilot is validated from the existing read-only external runs. Synthetic
fixtures are scored against TASK-005B's analytic truth, not against any
production output. Orientation convention matrices are evaluated as a
diagnostic and are not applied to or written over raw production results.

# Evidence / Sources

- `reports/kinematics/TASK-005A-core-hand-kinematics.md`
- `reports/kinematics/TASK-005B-independent-kinematics-benchmark.md`
- `reports/kinematics/TASK-005C-kinematics-qa-tooling.md`
- `kinematics/extractor.py`, `kinematics/hand_frame.py`, and `kinematics/io.py`
- `evaluation/kinematics/synthetic_hand.py`, `benchmark_contract.py`, and
  `metrics.py`
- `datasets/manifests/karsl_milestone1_pilot.csv`

The results below are mathematical and structural validation evidence. They
do not select a pose model.

# Files Changed

TASK-005D additions/changes:

- `evaluation/kinematics/production_adapter.py` — extractor-neutral adapter
  from TASK-005B fixtures to frozen TASK-005A output.
- `evaluation/kinematics/pilot_contract.py` — hard exact-manifest, sample,
  frame, metadata, and implementation-commit input validation.
- `evaluation/kinematics/__init__.py` — exports for the adapter and pilot
  contract.
- `evaluation/kinematics_qa/contract.py` — accepts the frozen production
  metadata vocabulary and requires `valid_palm_frame`.
- `evaluation/kinematics_qa/validator.py` — implements the selected
  channel-level validity contract, includes valid palm frames in rotation QA,
  and computes finite-channel distributions/temporal diagnostics.
- `tests/test_task005c_kinematics_qa.py` — QA contract regression and partial
  channel tests.
- `tests/test_task005d_kinematics_validation.py` — adapter and Phase-A input
  selection safety tests.
- `scripts/run_task005d_validation.py` — reproducible synthetic/pilot
  validation harness.
- `reports/kinematics/TASK-005D-validation-results.json` — compact numerical
  result record.
- `reports/kinematics/TASK-005D-independent-validation.md` — this report.

The frozen TASK-005A/B/C files are present through the cherry-picked history;
their mathematics and benchmark truth were not edited.

# Frozen Inputs

| Input | Commit / reference |
|---|---|
| Main base | `ba6389f334ea5277b303c3f7795c919def4bf08e` |
| TASK-005A implementation | `564167420c7f5b4f12197fe36e7d2b59ae08ace0` |
| TASK-005A report/finalization | `60480a6` |
| TASK-005B benchmark | `5f981d9f8c44408488f02b74f73a378197422830` |
| TASK-005C Copilot PR | #11 |
| TASK-005C head commit | `59d49f49321834160943cf94d216a1d26e4979e8` |

TASK-005A/B and the Copilot PR remain unmerged. This branch is a validation
integration only.

# Integration

The exact commits were integrated once, in this order:

1. TASK-005A implementation `5641674`;
2. TASK-005A report/finalization `60480a6`;
3. TASK-005B `5f981d9`;
4. TASK-005C `59d49f4`.

No merge conflict occurred. The existing production run was not regenerated.

# Evaluation

# Validation Protocol

The TASK-005B catalog was checked before scoring and contains 86 one-frame
paired LEFT/RIGHT fixtures: 80 mathematically valid and 6 intentionally
invalid. The locked comparison limits were retained:

| Quantity | Limit |
|---|---:|
| Flexion absolute error | 1 degree |
| Spread absolute error | 1 degree |
| Known orientation angular error | 1 degree |
| Rotation orthogonality element error | 1e-5 |
| `abs(det(R)-1)` | 1e-5 |
| Quaternion norm error | 1e-5 |
| Matrix/quaternion element consistency | 1e-5 |

Finite production channels are reported separately from strict full-result
contract validity. A production NaN is never converted to a numeric value.

# Production-to-Benchmark Adapter

`extract_production_sequence()` passes every fixture joint array with track
order `LEFT, RIGHT` to the frozen production function using state
`OBSERVED`. It returns:

```text
flexion_deg              [F,2,5,3]
adjacent_spread_deg      [F,2,4]
palm_rotation_matrix    [F,2,3,3]
palm_quaternion_wxyz    [F,2,4]
```

It also retains production `valid_kinematics`, `valid_palm_frame`, and flags.
This is important for the six benchmark-valid fixtures that production marks
strictly invalid because a spread projection is mathematically undefined.

# Flexion Validation

All 80 valid fixtures were passed through production. All 2,400 flexion
values were finite. The full frozen contract comparison passed 1,790/2,400
values; the maximum absolute error was **105.000000 degrees** in
`adversarial_finger_crossing`, LEFT middle proximal.

The disagreement is localized to the first/proximal channel:

| Channel group | Values within 1 degree | Finite values | Maximum error |
|---|---:|---:|---:|
| Proximal | 190 | 800 | 105.000000 degrees |
| Middle + distal | 1,600 | 1,600 | 3.25e-12 degrees |

The 74.96-degree pilot jump is not being smoothed or suppressed; it is
reported later under temporal diagnostics.

## Synthetic category results

The category counts and exact per-fixture outcomes are also machine-readable
under `synthetic_validation.catalog.category_summary` and
`synthetic_validation.aggregate_errors.failing_fixture_ids` in the result
JSON. Every expected-valid fixture fails the complete four-channel contract
because the proximal and orientation conventions are not jointly compatible;
the component-level counts are:

| Category | Valid / invalid | Flexion pass | Spread pass | Orientation pass | Quaternion pass |
|---|---:|---:|---:|---:|---:|
| neutral | 1 / 0 | 0 | 1 | 0 | 0 |
| single bend | 45 / 0 | 0 | 40 | 0 | 0 |
| multi-joint curl | 3 / 0 | 0 | 2 | 0 | 0 |
| independent fingers | 2 / 0 | 0 | 2 | 0 | 0 |
| spread | 4 / 0 | 0 | 4 | 0 | 0 |
| mirror | 3 / 0 | 0 | 3 | 0 | 0 |
| translation | 3 / 0 | 0 | 3 | 0 | 0 |
| scale | 4 / 0 | 0 | 4 | 0 | 0 |
| quaternion/orientation | 8 / 0 | 0 | 8 | 0 | 0 |
| adversarial | 7 / 1 | 0 | 6 | 0 | 0 |

The exact failing fixture lists are retained in the JSON rather than copied
into prose: flexion, orientation, quaternion, and complete-case lists each
contain all 80 expected-valid IDs; the spread list contains the seven IDs
described above.

## Signed convention interpretation

TASK-005A computes an unsigned polyline turn angle with range 0–180 degrees.
TASK-005B stores signed generation turns, but its locked public contract and
current catalog expect positive flexion magnitudes. The catalog has no
negative flexion case. Accordingly, a future signed negative control should be
compared by absolute magnitude unless the shared contract is explicitly
revised to require signed output. No sign correction was applied in this
validation.

# Proximal-Channel Analysis

The synthetic generator defines the first bend relative to the first
finger-segment ray. TASK-005A defines the first non-thumb bend using
`wrist -> MCP` as the incoming segment, and the thumb analogously uses
`wrist -> CMC`. The 21-joint layout has no metacarpal-shaft point that could
make those definitions identical.

For the neutral synthetic hand, production returns the following proximal
values while TASK-005B truth is zero:

| Finger | Production neutral proximal angle |
|---|---:|
| thumb | 15.217593 degrees |
| index | 9.440035 degrees |
| middle | 0 degrees |
| ring | 9.440035 degrees |
| pinky | 15.217593 degrees |

This confirms a genuine contract mismatch, not a tolerance issue. A rigid
transform or coordinate-basis adapter cannot change the wrist/MCP geometry
without changing the fixture itself or the analytic truth. The production
definition is internally documented and monotone, but it is not the same
quantity as TASK-005B's first generation turn. This requires an explicit
TASK-005 contract decision before TASK-006.

# Spread Validation

The production spread implementation and the benchmark agree for the normal
base-ray fixtures. Across the valid catalog, 622 spread values were finite and
618/622 were within 1 degree. The maximum finite error was **160 degrees** in
`adversarial_near_180`, where the first index segment reverses after the
179.9-degree bend. TASK-005B truth intentionally remains the original base-ray
gap, while TASK-005A measures the projected proximal-phalanx direction. This
is a second explicit geometry-definition mismatch, not a hidden correction.

Six valid benchmark cases produce the expected production conditioning
behaviour at a 90-degree first bend:

```text
single_thumb_joint0_90deg
single_index_joint0_90deg
single_middle_joint0_90deg
single_ring_joint0_90deg
single_pinky_joint0_90deg
multi_curl_pinky
```

The relevant proximal segment projects to zero (or below the production
minimum), so production emits NaN spread channels and keeps the palm frame.
They are not silently relabeled as benchmark-invalid. The complete per-case
spread pass count is 73/80; the six conditioning cases and the near-180 case
are the seven non-passing cases.

## Spread-conditioning review

The frozen 15-degree minimum projected-angle rule is mathematically
defensible: projection amplification is bounded to approximately 3.9x, and a
direction parallel to the palm normal has no meaningful palm-plane angle. The
pilot cost is material but measured: 474 additional spread-pair NaNs, or
6.69% of all spread pairs, with a smooth distribution and no natural gap.

Recommendation for the frozen run: **KEEP** the conditioning guard and the
per-channel NaN output. TASK-006 must consume the channel mask explicitly. A
future threshold change needs a separately justified sensitivity experiment;
this task did not change 15 degrees.

# Palm-Frame Convention Analysis

The benchmark's canonical bases are:

```text
B_RIGHT = I
B_LEFT  = diag(-1, 1, -1)
```

For the benchmark's wrist/index/middle/pinky geometry, the production
`[lateral, normal, distal]` construction yields the fixed local basis:

```text
P = [[-1, 0, 0],
     [ 0, 0, 1],
     [ 0, 1, 0]]
```

Thus `R_production = R_global @ P`, while
`R_benchmark = R_global @ B_side`. The required right-multiplication matrices
are therefore:

```text
C_RIGHT = P.T @ B_RIGHT
         = [[-1, 0, 0], [0, 0, 1], [0, 1, 0]]

C_LEFT  = P.T @ B_LEFT
         = [[ 1, 0, 0], [0, 0,-1], [0, 1, 0]]
```

They are different. Applying `C_RIGHT` to both tracks gives 0 degrees on
RIGHT and 180 degrees on LEFT; applying `C_LEFT` to both gives 0 degrees on
LEFT and 180 degrees on RIGHT. Applying the two documented side-specific
matrices separately gives 0 degrees maximum on both sides, but that is not
the single common fixed mapping required by this task. No orientation
correction was applied to production output.

Result: the orientation convention cannot pass the one-common-mapping
criterion without a shared contract decision.

# LEFT / RIGHT Mirror Validation

Four catalog cases are mirror-equivalent (the three mirror cases plus
`adversarial_mirrored_composed_rotation`). Production local outputs remain
equivalent without ad-hoc sign changes:

| Quantity | Maximum LEFT/RIGHT difference |
|---|---:|
| Flexion | 5.84e-14 degrees |
| Spread | 8.89e-15 degrees |

Mirror local consistency passes the locked ideal-geometry expectation.

# Rigid-Transform Validation

Production local values were compared against the corresponding untransformed
fixture. Results are far below the benchmark's 1e-8 internal invariance
check:

| Transform set | Max flexion delta | Max spread delta | Result |
|---|---:|---:|---|
| Translation | 3.24e-12 degrees | 2.38e-12 degrees | PASS |
| Uniform scale | 1.91e-14 degrees | 1.78e-15 degrees | PASS |
| Global rotations | 2.98e-14 degrees | 1.60e-14 degrees | PASS |

# Invalid Geometry

The six intentionally invalid benchmark cases were executed through the
production entrypoint without crashing the catalog run.

| Fixture | Expected reason | Production outcome | Result |
|---|---|---|---|
| `degenerate_zero_length_bone` | zero/tiny bone | invalid on both tracks; zero-length flags | PASS |
| `degenerate_coincident_mcp` | collinear/coincident palm | valid on both tracks; no flag | **FAIL** |
| `degenerate_collinear_palm` | collinear palm | invalid; `PALM_AXIS_ZERO_LENGTH` | PASS |
| `degenerate_nan_joint` | non-finite joint | invalid; `JOINTS_NON_FINITE` | PASS |
| `degenerate_inf_joint` | non-finite joint | invalid; `JOINTS_NON_FINITE` | PASS |
| `adversarial_tiny_nonzero_bone` | below validity floor | invalid; zero-length/degenerate flags | PASS |

The coincident-MCP fixture exposes a real validation gap: the benchmark
geometry predicate rejects it, while the production frame uses a different
three-point cross product and returns a fully finite valid result. This was
not changed in the frozen production implementation.

# Copilot QA Results

## Initial unchanged validator

The first run used the Copilot validator without changes. It failed before
meaningful pilot metric scoring because the frozen production metadata uses
lowercase `['left', 'right']` and stores quaternion order in the explicit
`quaternion_order: 'wxyz'` field while its `quaternion_convention` field is
descriptive prose. The validator reported 36 metadata field failures across
18 samples and consequently 18 unavailable alignment samples.

The unchanged output is retained outside Git at:

```text
/home/hatim/graduation-project-runs/task005d_qa/initial-qa.json
/home/hatim/graduation-project-runs/task005d_qa/initial-distributions.csv
```

## Neutral QA contract correction

The validator was corrected in this branch, without altering production NPZs:

- track order is compared case-insensitively while order remains fixed;
- quaternion order accepts the explicit `quaternion_order` field;
- `valid_palm_frame` is required;
- strict `valid_kinematics=false` with `valid_palm_frame=true` permits finite
  channels and NaN only in undefined per-channel geometry;
- rotation/quaternion QA includes all valid palm frames, not only strict
  all-channel rows;
- distributions and temporal deltas use finite channel values.

This is the selected Model B channel-level contract. It preserves the strict
flag while making downstream masks machine-checkable.

The corrected Copilot command was:

```bash
python scripts/validate_task005_kinematics.py \
  --tracked-run /home/hatim/graduation-project-runs/wilor_karsl_pilot_tracked_task004d \
  --kinematics-run /home/hatim/graduation-project-runs/wilor_karsl_pilot_kinematics_task005a \
  --output-json /home/hatim/graduation-project-runs/task005d_qa/remediated-qa.json \
  --output-csv /home/hatim/graduation-project-runs/task005d_qa/remediated-distributions.csv
```

It exited 0 with `PASS`.

## Structural and rotation QA

| Check | Result |
|---|---:|
| Exact samples | 18/18 |
| Exact decoded frames | 894 |
| TASK-004 frame/state/source alignment | PASS; 0 mismatches |
| Valid palm frames checked | 1,770 |
| Max orthogonality error | 8.762e-08 |
| Max `abs(det(R)-1)` | 9.239e-08 |
| Max quaternion norm error | 4.114e-08 |
| Max matrix/quaternion element error | 8.166e-08 |
| Max matrix/quaternion angular disagreement | 0.0161 degrees |
| Non-positive determinants | 0 |
| Suspicious flexion values | 0 |

All mathematical rotation checks pass their locked QA limits.

## Flexion distributions

Counts are finite channel values. Values are `p50 / p95 / max` in degrees.

| Track | Finger | Joint | Count | p50 / p95 / max |
|---|---|---|---:|---:|
| LEFT | thumb | proximal | 894 | 25.09 / 48.12 / 58.60 |
| LEFT | thumb | middle | 894 | 22.84 / 35.99 / 55.14 |
| LEFT | thumb | distal | 894 | 23.07 / 51.53 / 84.31 |
| LEFT | index | proximal | 894 | 45.26 / 65.68 / 76.62 |
| LEFT | index | middle | 894 | 13.53 / 101.84 / 108.30 |
| LEFT | index | distal | 894 | 12.54 / 43.38 / 69.89 |
| LEFT | middle | proximal | 894 | 46.34 / 86.71 / 104.28 |
| LEFT | middle | middle | 894 | 10.52 / 88.44 / 98.30 |
| LEFT | middle | distal | 894 | 11.06 / 45.00 / 72.44 |
| LEFT | ring | proximal | 894 | 37.46 / 87.24 / 101.52 |
| LEFT | ring | middle | 894 | 20.29 / 88.23 / 96.40 |
| LEFT | ring | distal | 894 | 16.74 / 62.76 / 80.54 |
| LEFT | pinky | proximal | 894 | 42.39 / 90.25 / 104.29 |
| LEFT | pinky | middle | 894 | 18.74 / 81.55 / 93.26 |
| LEFT | pinky | distal | 894 | 15.80 / 60.41 / 85.20 |
| RIGHT | thumb | proximal | 876 | 20.62 / 39.42 / 45.09 |
| RIGHT | thumb | middle | 876 | 25.25 / 39.75 / 46.21 |
| RIGHT | thumb | distal | 876 | 19.67 / 49.65 / 67.19 |
| RIGHT | index | proximal | 876 | 40.65 / 58.37 / 73.44 |
| RIGHT | index | middle | 876 | 8.04 / 105.93 / 113.65 |
| RIGHT | index | distal | 876 | 13.65 / 38.92 / 58.07 |
| RIGHT | middle | proximal | 876 | 30.09 / 79.20 / 97.99 |
| RIGHT | middle | middle | 876 | 6.63 / 87.25 / 97.73 |
| RIGHT | middle | distal | 876 | 8.38 / 44.03 / 60.39 |
| RIGHT | ring | proximal | 876 | 21.95 / 77.46 / 94.73 |
| RIGHT | ring | middle | 876 | 13.13 / 82.43 / 96.41 |
| RIGHT | ring | distal | 876 | 11.79 / 48.99 / 65.47 |
| RIGHT | pinky | proximal | 876 | 30.83 / 79.44 / 97.08 |
| RIGHT | pinky | middle | 876 | 11.30 / 74.94 / 92.49 |
| RIGHT | pinky | distal | 876 | 11.16 / 43.27 / 73.45 |

## Spread distributions

| Track | Pair | Count | p50 / p95 / max |
|---|---|---:|---:|
| LEFT | thumb-index | 894 | 45.04 / 64.20 / 88.42 |
| LEFT | index-middle | 860 | 4.94 / 37.16 / 74.12 |
| LEFT | middle-ring | 763 | 15.58 / 35.46 / 97.60 |
| LEFT | ring-pinky | 759 | 8.28 / 28.61 / 71.67 |
| RIGHT | thumb-index | 876 | 46.28 / 61.70 / 71.28 |
| RIGHT | index-middle | 841 | 3.88 / 20.09 / 54.35 |
| RIGHT | middle-ring | 808 | 10.30 / 21.35 / 45.79 |
| RIGHT | ring-pinky | 805 | 7.44 / 19.31 / 49.25 |

## Temporal diagnostics

Pooled finite adjacent-frame deltas are:

| Quantity | Count | p95 | p99 | Maximum |
|---|---:|---:|---:|---:|
| Flexion absolute delta | 26,010 | 8.893 degrees | 22.087 degrees | 74.957 degrees |
| Spread absolute delta | 6,411 | 7.396 degrees | 17.185 degrees | 57.287 degrees |
| Palm orientation delta | 1,734 | 16.526 degrees | 38.116 degrees | 102.157 degrees |

The largest flexion event is `karsl_test_s03_sign0173_repfirst`, LEFT ring
PIP/middle channel, frame 22 to 23, 74.9566 degrees. The largest palm event
is `karsl_test_s01_sign0175_repfirst`, LEFT, frame 26 to 27, 102.1571
degrees. They are finite, mathematically valid per-frame outputs; they are
reconstruction/temporal anomalies, not rotation-schema failures. No temporal
filter was applied.

# Validity / NaN Contract

The frozen production implementation uses two flags:

```text
valid_kinematics = true only if every float output channel is finite
valid_palm_frame = true if the pose and palm orientation are valid
```

The validation branch selects **Model B — channel-level validity**:

- strict false + palm true is allowed;
- rotation/quaternion must be finite when `valid_palm_frame=true`;
- individual flexion/spread channels may be finite or NaN according to their
  geometry;
- when `valid_palm_frame=false`, all derived float fields must be NaN;
- strict true requires every field to be finite.

This matches TASK-005A's metadata and the downstream need to use flexion even
when one spread pair is undefined. It does not reinterpret
`valid_kinematics` or change any numeric output.

Pilot accounting:

| Quantity | Result |
|---|---:|
| Hand-instances | 1,788 |
| Strictly valid | 1,555 |
| Valid palm frames | 1,770 |
| Invalid from tracking/no pose | 18 (15 likely occluded, 3 missing) |
| Partial spread-conditioned instances | 215 |
| Flexion NaNs | 270 = 18 × 15; no unexplained NaNs |
| Spread NaNs | 546 total |
| Spread NaNs from no-pose masking | 72 = 18 × 4 |
| Additional conditioning NaNs | 474 |

# Full Pilot Diagnostics

The exact pilot input validator requires the committed 18-row manifest, exact
sample-ID set, exact frozen implementation commit, exact TASK-004 frame/source
provenance, and 894 total frames. It passed before metric evaluation. The
kinematics QA then reported `PASS` under the channel-level contract.

Per-video strict/palm/spread-finite hand/channel counts:

| Sample | Frames | Strict hands | Palm hands | Finite spread values |
|---|---:|---:|---:|---:|
| `karsl_test_s01_sign0171_repfirst` | 81 | 162 | 162 | 648 |
| `karsl_test_s01_sign0172_repfirst` | 52 | 65 | 104 | 333 |
| `karsl_test_s01_sign0173_repfirst` | 51 | 84 | 102 | 373 |
| `karsl_test_s01_sign0174_repfirst` | 34 | 50 | 50 | 200 |
| `karsl_test_s01_sign0175_repfirst` | 41 | 71 | 82 | 306 |
| `karsl_test_s01_sign0176_repfirst` | 38 | 76 | 76 | 304 |
| `karsl_test_s02_sign0171_repfirst` | 67 | 134 | 134 | 536 |
| `karsl_test_s02_sign0172_repfirst` | 36 | 31 | 72 | 181 |
| `karsl_test_s02_sign0173_repfirst` | 44 | 64 | 88 | 304 |
| `karsl_test_s02_sign0174_repfirst` | 36 | 72 | 72 | 288 |
| `karsl_test_s02_sign0175_repfirst` | 48 | 88 | 96 | 362 |
| `karsl_test_s02_sign0176_repfirst` | 57 | 108 | 114 | 444 |
| `karsl_test_s03_sign0171_repfirst` | 68 | 136 | 136 | 544 |
| `karsl_test_s03_sign0172_repfirst` | 50 | 32 | 100 | 255 |
| `karsl_test_s03_sign0173_repfirst` | 38 | 76 | 76 | 304 |
| `karsl_test_s03_sign0174_repfirst` | 38 | 76 | 76 | 304 |
| `karsl_test_s03_sign0175_repfirst` | 56 | 112 | 112 | 448 |
| `karsl_test_s03_sign0176_repfirst` | 59 | 118 | 118 | 472 |

The compact machine-readable complete QA and validation results are in
`reports/kinematics/TASK-005D-validation-results.json`. The full QA JSON and
CSV are also retained outside Git at:

```text
/home/hatim/graduation-project-runs/task005d_qa/remediated-qa.json
/home/hatim/graduation-project-runs/task005d_qa/remediated-distributions.csv
```

# Results

The frozen pilot is structurally aligned and mathematically valid at the
rotation/schema level under the corrected channel-level QA contract. The
independent synthetic compatibility result is not a pass: the production and
benchmark definitions require a contract decision before downstream use.

# Failures

The strict TASK-005 acceptance is not met for this integration:

1. TASK-005B's proximal flexion truth is not the same geometric quantity as
   TASK-005A's documented wrist-to-MCP proximal channel.
2. TASK-005B's base-ray spread truth diverges from TASK-005A's proximal-
   phalanx spread for the near-180 adversarial case.
3. Palm orientation requires different fixed LEFT/RIGHT basis transforms;
   no single common transform satisfies both tracks.
4. The coincident-MCP invalid fixture is accepted as fully valid by production
   without a warning.

These are not tolerance or threshold failures. The frozen production math,
benchmark truth, and pilot results were left unchanged.

# Limitations

- Synthetic fixtures are intentionally simple and do not establish anatomical
  clinical accuracy.
- A single common palm basis mapping cannot be inferred for the two frozen
  side conventions; a side-specific mapping is mathematically exact but does
  not satisfy the requested single-mapping criterion.
- No new production implementation was authorized in this validation task, so
  the proximal, spread, and coincident-MCP issues remain open.
- Full-pilot temporal statistics describe noisy per-frame reconstruction and
  are not a pass/fail smoothness test.
- The corrected QA tool's historical TASK-005C report still describes its
  original all-or-nothing assumption; this report records the neutral
  TASK-005D contract correction and rationale.

# Performance

No model was rerun, so this task introduces no new WiLoR runtime claim. The
synthetic validation and read-only QA are CPU-only, small, and complete in
under one second in the local environment. Pilot paths are external and
ignored by Git.

# Comparison

This is not a WiLoR-versus-MediaPipe or model-selection comparison. It is an
independent compatibility check between TASK-005A's frozen output semantics
and TASK-005B's frozen analytic contract. The result is deliberately not used
to select a winner.

# Acceptance Criteria

| Criterion | Status | Evidence |
|---|---|---|
| A — analytic flexion | FAIL | 1,600 middle/distal values pass; proximal definition mismatch; 1,790/2,400 full values pass |
| B — analytic spread | FAIL | 618/622 finite values pass; six conditioning cases and near-180 contract mismatch remain |
| C — mirror consistency | PASS | max local difference 5.84e-14 degrees |
| D — rigid invariance | PASS | translation, scale, and rotation deltas below 1e-8 |
| E — palm orientation | FAIL | side-specific mappings exact, no single common mapping |
| F — rotation quality | PASS | all 1,770 valid-palm frames pass matrix/quaternion QA |
| G — tracking alignment | PASS | exact 18 samples / 894 frames / state and source alignment |
| H — missing/occlusion | PASS | all 18 no-pose instances remain invalid and all-NaN |
| I — invalid geometry | FAIL | coincident-MCP fixture returns fully valid without flag |
| J — QA validity contract | PASS | Model B channel-level semantics are explicit, tested, and pilot-accounted |

# Recommendation

REVISE

# Reproducibility

Run from the repository root with the activated existing `.venv`:

```bash
python scripts/run_task005d_validation.py \
  --tracked-run /home/hatim/graduation-project-runs/wilor_karsl_pilot_tracked_task004d \
  --kinematics-run /home/hatim/graduation-project-runs/wilor_karsl_pilot_kinematics_task005a \
  --qa-json /home/hatim/graduation-project-runs/task005d_qa/remediated-qa.json \
  --output-json reports/kinematics/TASK-005D-validation-results.json

python scripts/validate_task005_kinematics.py \
  --tracked-run /home/hatim/graduation-project-runs/wilor_karsl_pilot_tracked_task004d \
  --kinematics-run /home/hatim/graduation-project-runs/wilor_karsl_pilot_kinematics_task005a \
  --output-json /home/hatim/graduation-project-runs/task005d_qa/remediated-qa.json \
  --output-csv /home/hatim/graduation-project-runs/task005d_qa/remediated-distributions.csv

python -m unittest discover -s tests -p 'test_*.py'
python -m compileall -q evaluation tracking kinematics scripts tests
```

There is no random seed. The benchmark is generated from literal parameters.
The local environment used Python 3.14.4 and NumPy 2.5.2. The stable external
artifacts are read-only inputs; no KArSL video, NPZ, checkpoint, MANO asset, or
model output is committed.

# Next Steps

Before TASK-006, create a focused follow-up for:

- `INTEGRATION/CONTRACT FIX`: choose and version the proximal flexion and
  spread definitions, and decide whether the shared orientation contract
  permits side-specific basis mapping.
- `OPUS CORE FIX`: reject or explicitly flag coincident MCP geometry if the
  production validity policy requires the broader benchmark degeneracy rule.

After those decisions, rerun this frozen 86-case adapter and the full-pilot QA.
Do not start TASK-006 from the current ambiguous contract.

# Final Verdict

TASK-005 NEEDS REVISION
