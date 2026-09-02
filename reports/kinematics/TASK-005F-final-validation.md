# Task

TASK-005F — Final kinematics revalidation and integration.

# Branch

`evaluation/task-005f-final-validation`

# Scope

This validation integrates the frozen TASK-005E1 final contract with the
TASK-005E2 production palm-landmark validity fix. It reruns the independent
86-case synthetic contract and validates a new kinematics output directory
against the existing TASK-004D tracked run. It does not rerun WiLoR or
tracking, change production kinematics mathematics, alter thresholds, rewrite
historical TASK-005D/E1 results, or start TASK-006.

# Approach

The branch was created from the exact TASK-005E1 commit and received the exact
TASK-005E2 source fix once. The final production implementation is pinned by
metadata to `3404a7af5d80110777ac61bc3076ef7ebada5dd6`. The final harness uses
the independent `TASK-005-final-v2` truth layer and the corrected Model-B QA
validator. It writes only compact results to Git; raw arrays and run outputs
remain in external ignored directories.

# Evidence / Sources

- TASK-005D historical validation: `reports/kinematics/TASK-005D-independent-validation.md`
- TASK-005E1 contract: `reports/kinematics/TASK-005E1-final-kinematics-contract.md`
- TASK-005E1 machine result: `reports/kinematics/TASK-005E1-final-contract-results.json`
- TASK-005A production report: `reports/kinematics/TASK-005A-core-hand-kinematics.md`
- TASK-005B independent benchmark: `reports/kinematics/TASK-005B-independent-kinematics-benchmark.md`
- TASK-005C QA tooling: `reports/kinematics/TASK-005C-kinematics-qa-tooling.md`
- final source: `kinematics/geometry.py`, `kinematics/hand_frame.py`
- final benchmark truth: `evaluation/kinematics/final_contract.py`
- final QA: `evaluation/kinematics_qa/validator.py`

# Files Changed

- `evaluation/kinematics/pilot_contract.py` — pins the exact E2 production
  commit before pilot metrics are accepted.
- `scripts/run_task005e1_contract.py` — adds compact defined/undefined spread
  accounting for the final validation diagnostic; the historical E1 result is
  not rewritten.
- `scripts/run_task005f_validation.py` — final synthetic/pilot harness,
  regression comparison, explicit coincident-MCP all-NaN check, and A–L
  acceptance calculation.
- `reports/kinematics/TASK-005F-final-validation.md` — this report.
- `reports/kinematics/TASK-005F-validation-results.json` — compact machine
  result.

No source video, NPZ run, model, checkpoint, or generated visualization is
tracked by this branch.

# How to Run

From the repository root, with the existing virtual environment:

```bash
python scripts/run_task005f_validation.py
python scripts/validate_task005_kinematics.py \
  --tracked-run /home/hatim/graduation-project-runs/wilor_karsl_pilot_tracked_task004d \
  --kinematics-run /home/hatim/graduation-project-runs/wilor_karsl_pilot_kinematics_task005f \
  --output-json /home/hatim/graduation-project-runs/task005f_qa/final-qa.json \
  --output-csv /home/hatim/graduation-project-runs/task005f_qa/final-distributions.csv
python -m unittest discover -s tests -p 'test_*.py'
python -m compileall -q evaluation tracking kinematics scripts tests
```

The final pilot run was produced once, without WiLoR or tracking inference,
using the existing TASK-005A kinematics entry point and the TASK-004D tracked
artifacts:

```bash
python scripts/run_task005a_kinematics.py \
  --tracked-run /home/hatim/graduation-project-runs/wilor_karsl_pilot_tracked_task004d \
  --out-dir /home/hatim/graduation-project-runs/wilor_karsl_pilot_kinematics_task005f \
  --strict-counts
```

# Evaluation

The final validation uses the locked TASK-005E1 tolerances:

| Quantity | Limit |
|---|---:|
| flexion absolute error | 1 degree |
| defined spread absolute error | 1 degree |
| orientation angular error | 1 degree |
| rotation orthogonality error | 1e-5 |
| `abs(det(R)-1)` | 1e-5 |
| quaternion norm error | 1e-5 |
| matrix/quaternion element consistency | 1e-5 |

The six invalid catalog cases are not scored as ordinary valid poses. A
conditioning-invalid spread channel must be NaN with its explicit flag. An
invalid palm frame must have both validity flags false and all derived float
channels NaN. Pilot metrics use frame index and the existing tracked source
provenance; no missing pose is bridged or numerically invented.

# Objective

Confirm that the final E1 contract and the E2 production validity correction
pass together end-to-end, while preserving the original production numbers
and all historical validation evidence.

# Final Inputs

| Input | Exact reference |
|---|---|
| frozen main base | `ba6389f334ea5277b303c3f7795c919def4bf08e` |
| TASK-005D validation | `b41bc1808d09b1987ebdcf417e1bdadc42962f6d` |
| TASK-005E1 contract | `e6029e35e49516389356fdca159f6dddc1bcfda2` |
| TASK-005E2 production source | `3404a7af5d80110777ac61bc3076ef7ebada5dd6` |
| integrated E2 cherry-pick | `248dcf5660fc2aa9237adf081a4b45ff5a021b5b` |
| TASK-005A frozen implementation | `564167420c7f5b4f12197fe36e7d2b59ae08ace0` |
| TASK-005B benchmark ancestor | `5f981d9f8c44408488f02b74f73a378197422830` |
| TASK-005C integrated ancestor | `a9e286e324b79ea89a702105c49d421c9e25f3ad` |

# Final Contract

The authoritative version is `TASK-005-final-v2`:

- track order is `LEFT, RIGHT`;
- finger order is `thumb, index, middle, ring, pinky`;
- flexion is unsigned geometric bend magnitude in `[0, 180]` degrees;
- channels are generic `proximal, middle, distal` rather than clinical labels;
- proximal is the `wrist -> base -> next joint` geometric bend proxy, not
  isolated clinical MCP flexion;
- spread is the unsigned angle between actual proximal-phalanx directions
  projected into the palm plane;
- the 15-degree projection-conditioning rule emits NaN and an explicit
  `SPREAD_DIRECTION_DEGENERATE_*` flag;
- validity follows Model B channel-level semantics;
- palm orientation is a proper matrix with normalized WXYZ quaternion;
- synthetic orientation scoring uses one fixed proper mapping per hand side.

# Integration

The branch history is:

```text
TASK-005E1 final contract
        ↓
TASK-005E2 coincident-palm validity fix (one cherry-pick)
        ↓
TASK-005F final validation
```

There were no integration conflicts and no production formula or threshold
changes. The E2 fix is the general scale-relative mutual-separation check for
the four palm-defining landmarks; it has no fixture-specific coordinate or
case-id rule. The previous TASK-005A output was retained for exact array
regression comparison.

# Synthetic Benchmark

The independent final harness evaluated the unchanged 86-case catalog:

| Set | Cases | Result |
|---|---:|---:|
| expected-valid | 80 | 80/80 complete compatibility |
| expected-invalid | 6 | 6/6 rejected or channel-invalid |
| total | 86 | pass |

The benchmark self-check also passed its catalog count, geometry-derived
truth, requested middle/distal turns, explicit conditioning mask, fixed proper
side mappings, mirror equivalence, and translation/scale/global-rotation
invariance checks.

Category coverage for valid fixtures was:

| Category | Valid cases | Flexion | Spread | Orientation | Quaternion |
|---|---:|---:|---:|---:|---:|
| neutral | 1 | 1/1 | 1/1 | 1/1 | 1/1 |
| single bend | 45 | 45/45 | 45/45 | 45/45 | 45/45 |
| multi-joint curl | 3 | 3/3 | 3/3 | 3/3 | 3/3 |
| independent fingers | 2 | 2/2 | 2/2 | 2/2 | 2/2 |
| spread | 4 | 4/4 | 4/4 | 4/4 | 4/4 |
| mirror | 3 | 3/3 | 3/3 | 3/3 | 3/3 |
| translation | 3 | 3/3 | 3/3 | 3/3 | 3/3 |
| scale | 4 | 4/4 | 4/4 | 4/4 | 4/4 |
| quaternion/orientation | 8 | 8/8 | 8/8 | 8/8 | 8/8 |
| adversarial valid | 7 | 7/7 | 7/7 | 7/7 | 7/7 |

# Flexion

All 2,400 applicable values from the 80 valid fixtures passed the 1-degree
absolute-error limit. The maximum error was 0 degrees. The truth is computed
independently from the generated output geometry, including the proximal
`wrist -> base -> next joint` turn.

# Spread

All 80 valid fixture cases passed the final spread mask and numeric comparison.
There were 622 defined numeric values, all 622 passed, with maximum finite
error `1.4210854715202004e-14` degrees. There were 18 expected conditioning
invalid values, zero unexpected NaNs, and zero unexpected finite values. The
six conditioning cases are explicit in the benchmark catalog; conditioning is
channel-level and does not invalidate the otherwise usable palm frame.

# Palm Orientation

All 80 valid fixture cases passed orientation comparison with the frozen
side-specific mappings. The worst orientation error was 0 degrees. The fixed
mappings are:

```text
C_RIGHT = [[-1, 0, 0],
           [ 0, 0, 1],
           [ 0, 1, 0]]

C_LEFT  = [[ 1, 0, 0],
           [ 0, 0,-1],
           [ 0, 1, 0]]
```

They depend only on hand side, are proper rotations, and are applied only for
synthetic comparison. Raw production matrices and quaternions are unchanged.

# LEFT / RIGHT

Mirrored fixture pairs passed local flexion and spread equivalence. No
side-specific sign correction was added to those local quantities. LEFT and
RIGHT orientation mappings are intentionally separate because the lateral
axis carries the hand-side convention while distal and palmar-normal axes
retain their physical meaning.

# Rigid Invariance

The independent self-check passed translation, uniform scale, and composed
global rotation invariance for local flexion and spread. Orientation changes
under global rotation as expected and is compared using the fixed analytic
side mappings. The checks use no sample-fitted or fixture-specific correction.

# Invalid Geometry

All six expected-invalid cases passed the final invalid handling check:

| Fixture | Expected outcome | Result |
|---|---|---|
| `degenerate_zero_length_bone` | invalid channel/flag | PASS |
| `degenerate_coincident_mcp` | invalid palm frame and NaN outputs | PASS |
| `degenerate_collinear_palm` | invalid palm frame/flag | PASS |
| `degenerate_nan_joint` | non-finite input rejected | PASS |
| `degenerate_inf_joint` | non-finite input rejected | PASS |
| `adversarial_tiny_nonzero_bone` | invalid channel/flag | PASS |

For `degenerate_coincident_mcp`, the final production result was:

```text
valid_palm_frame = [[False, False]]
valid_kinematics = [[False, False]]
all derived float channels = NaN
flag = PALM_LANDMARKS_COINCIDENT_index_MCP_middle_MCP_sep_over_palm=0.000e+00
```

The explicit harness check passed without changing the invalid fixture or
weakening its expectation.

# Pilot QA

The new output was written to:

`/home/hatim/graduation-project-runs/wilor_karsl_pilot_kinematics_task005f`

It was validated against:

`/home/hatim/graduation-project-runs/wilor_karsl_pilot_tracked_task004d`

The structural result was 18/18 samples and 894/894 frames, with zero frame,
tracking-state, source-provenance, or timestamp alignment mismatches. The
final run metadata is pinned to the E2 production source commit.

# Model-B Validity

The corrected QA validator reported zero invalid-mask violations and zero
non-finite violations. It found 215 strict-false/palm-true partial instances;
these retain finite flexion and orientation while only undefined spread
channels are NaN. The 18 invalid/no-pose instances have all derived floats
masked as NaN. No kinematic values were generated for a missing pose.

# Rotation / Quaternion QA

QA checked all 1,770 valid palm-frame hand instances:

| Diagnostic | Maximum / count | Locked limit | Result |
|---|---:|---:|---|
| orthogonality error | `8.761691461245391e-08` | `1e-5` | PASS |
| `abs(det(R)-1)` | `9.239450315945419e-08` | `1e-5` | PASS |
| non-positive determinants | 0 | 0 | PASS |
| quaternion norm error | `4.114059892756927e-08` | `1e-5` | PASS |
| matrix/quaternion element error | `8.165514298053012e-08` | `1e-5` | PASS |
| matrix/quaternion angular disagreement | `0.016093278183459556` degrees | diagnostic | PASS |

# Temporal Diagnostics

Temporal values are diagnostics only. Consecutive valid pairs are used;
missing values are skipped and never bridged. The pooled final-run summaries
were:

| Channel | Count | p95 | p99 | Maximum |
|---|---:|---:|---:|---:|
| flexion absolute delta | 26,010 | 8.8933 degrees | 22.0874 degrees | 74.9566 degrees |
| spread absolute delta | 6,411 | 7.3956 degrees | 17.1854 degrees | 57.2867 degrees |
| palm orientation delta | 1,734 | 16.5262 degrees | 38.1158 degrees | 102.1571 degrees |

The previously reported largest events remain:

- flexion: `karsl_test_s03_sign0173_repfirst`, LEFT ring middle/PIP channel,
  frames `22 -> 23`, `74.95662307739258` degrees;
- palm orientation: `karsl_test_s01_sign0175_repfirst`, LEFT, frames
  `26 -> 27`, `102.15709581399929` degrees.

These are finite reconstruction changes in a per-frame representation, not
schema or rotation-quality failures. They were not smoothed or suppressed.
The QA also recorded per-LEFT/RIGHT, per-finger/joint flexion distributions
and per-adjacent-pair spread distributions in the external CSV
`/home/hatim/graduation-project-runs/task005f_qa/final-distributions.csv`.

# Acceptance Criteria

| Criterion | Result | Evidence |
|---|---|---|
| A — final analytic flexion | PASS | 2,400/2,400 values; 80/80 cases |
| B — final analytic spread | PASS | 622/622 numeric values; 18 expected NaNs; no mask drift |
| C — mirror consistency | PASS | independent mirror self-check |
| D — rigid invariance | PASS | translation, scale, rotation self-check |
| E — palm orientation | PASS | 80/80 cases with fixed side mappings |
| F — rotation quality | PASS | all locked matrix/quaternion limits |
| G — TASK-004 alignment | PASS | 18/18, 894/894, zero mismatches |
| H — missing/occlusion policy | PASS | no-palm rows all derived NaN |
| I — invalid geometry | PASS | 6/6; coincident MCP explicitly verified |
| J — Model-B validity | PASS | zero QA contract violations |
| K — pilot regression | PASS | 1770 palm-valid; zero new real rejections; arrays exact |
| L — tests | PASS | 275 tests; compileall clean |

# Results

The final pilot counts are:

| Quantity | Result |
|---|---:|
| samples | 18 |
| decoded frames | 894 |
| hand instances | 1,788 |
| valid palm frames | 1,770 |
| invalid/no-pose instances | 18 |
| strict valid kinematics | 1,555 |
| partial spread instances | 215 |
| flexion NaNs | 270 = 18 × 15 |
| spread NaNs | 546 = 72 no-palm + 474 conditioning |
| conditioning spread NaNs | 474 |

The final NPZ arrays are exactly equal to the previous TASK-005A output for
all sample files. E2 therefore changes only degenerate-palm validity behavior;
it does not alter any real pilot numeric output or introduce a new real invalid
frame.

# Failures / Limitations

No acceptance criterion failed. The following remain documented non-blocking
properties of the frozen representation:

- the proximal channel is a geometric bend proxy, not clinical MCP
  goniometry;
- flexion and spread are unsigned geometric quantities;
- spread can be undefined near the palm normal and must remain channel-level
  NaN with its flag;
- side-specific orientation mappings are a documented comparison convention;
- large temporal jumps remain diagnostics of per-frame reconstruction and are
  not corrected in this stage;
- no smoothing, interpolation, or clinical ground-truth claim is introduced.

# Performance

This final task did not rerun WiLoR inference or tracking and therefore makes
no new model-throughput claim. The kinematics post-processing, synthetic
validation, and QA run are CPU-side validation operations. External output
generation and QA paths are recorded above; no generated output is committed.

# Comparison

This is final contract and production-geometry validation, not a model
comparison. It does not rank pose models and does not change the selected
WiLoR architecture.

# Recommendation

KEEP the frozen TASK-005 contract and corrected production validity rule, and
proceed to TASK-006 only as a separate task.

# Reproducibility

- Branch base: `e6029e35e49516389356fdca159f6dddc1bcfda2`.
- Final production source: `3404a7af5d80110777ac61bc3076ef7ebada5dd6`.
- Integrated E2 commit: `248dcf5660fc2aa9237adf081a4b45ff5a021b5b`.
- Final contract source: `e6029e35e49516389356fdca159f6dddc1bcfda2`.
- Tracked input: `/home/hatim/graduation-project-runs/wilor_karsl_pilot_tracked_task004d`.
- Final kinematics output: `/home/hatim/graduation-project-runs/wilor_karsl_pilot_kinematics_task005f`.
- Previous kinematics output used for exact regression:
  `/home/hatim/graduation-project-runs/wilor_karsl_pilot_kinematics_task005a`.
- QA JSON: `/home/hatim/graduation-project-runs/task005f_qa/final-qa.json`.
- QA distributions: `/home/hatim/graduation-project-runs/task005f_qa/final-distributions.csv`.
- No random seed or calibration was introduced.
- The final machine result is
  `reports/kinematics/TASK-005F-validation-results.json`.

# Next Steps

TASK-005 is validated for the project’s next stage. TASK-006 is not started by
this branch.

# Final Verdict

TASK-005 VALIDATED — READY FOR TASK-006
