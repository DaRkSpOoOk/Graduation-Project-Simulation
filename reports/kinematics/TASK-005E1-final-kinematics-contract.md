# Task

TASK-005E1 — Freeze the final TASK-005 kinematics contract and remediate the
benchmark/QA contract after TASK-005D integration.

# Branch

`evaluation/task-005e1-final-kinematics-contract`, based on frozen TASK-005D
commit `b41bc1808d09b1987ebdcf417e1bdadc42962f6d`.

# Scope

This task resolves the definition disagreements exposed by TASK-005D without
changing TASK-005A production mathematics, production thresholds, production
numeric outputs, TASK-005B history, or TASK-005D results. It creates the
versioned `TASK-005-final-v2` benchmark truth and freezes the channel-level
QA contract for the future virtual-glove representation.

TASK-006 was not started. The remaining coincident-MCP production defect is
intentionally reserved for Opus TASK-005E2.

# Approach

The original 86-case TASK-005B catalog is reused without editing its source
or report. A new neutral module derives final truth from each generated
21-joint output geometry with local analytic helpers. The helpers do not
import the production `kinematics` package and do not call production
functions. A separate harness sends the revised fixtures through the frozen
TASK-005A adapter only as a compatibility diagnostic.

Orientation is compared through one fixed, documented right-multiplication
matrix per hand side. The mapping is a convention adapter for scoring and
never overwrites raw production matrices or quaternions.

# Evidence / Sources

- `reports/kinematics/TASK-005D-independent-validation.md`
- `reports/kinematics/TASK-005D-validation-results.json`
- `reports/kinematics/TASK-005A-core-hand-kinematics.md`
- `reports/kinematics/TASK-005B-independent-kinematics-benchmark.md`
- `reports/kinematics/TASK-005C-kinematics-qa-tooling.md`
- `evaluation/kinematics/synthetic_hand.py`
- `kinematics/geometry.py`, `kinematics/hand_frame.py`, and
  `kinematics/extractor.py`

The frozen reports and source are the authoritative evidence for the
contract decisions. No model output was used to fit benchmark truth or
tolerances.

# Files Changed

- `evaluation/kinematics/final_contract.py` — versioned final contract,
  geometry-derived truth, conditioning masks, and fixed side mappings.
- `evaluation/kinematics/__init__.py` — exports for the final contract.
- `evaluation/kinematics_qa/contract.py` — explicit Model-B validity-contract
  version/name.
- `evaluation/kinematics_qa/validator.py` — reports the selected validity
  contract in QA output; existing channel-level checks remain in force.
- `scripts/run_task005e1_contract.py` — deterministic benchmark self-check and
  frozen-production compatibility diagnostic.
- `tests/test_task005e1_final_contract.py` — final truth, mapping, masking,
  invalid-case, immutability, and tolerance tests.
- `reports/kinematics/TASK-005E1-final-contract-results.json` — compact
  machine-readable result record.
- `reports/kinematics/TASK-005E1-final-kinematics-contract.md` — this report.

The TASK-005B and TASK-005D reports/results were not modified.

# How to Run

From the repository root with the existing environment:

```bash
python scripts/run_task005e1_contract.py
python -m unittest discover -s tests -p 'test_*.py'
python -m compileall -q evaluation tracking kinematics scripts tests
```

The script writes only the compact result JSON named above. It does not
create KArSL, model, or generated run artifacts.

# Objective

Freeze a reproducible geometric contract that reflects what the observed
21-joint skeleton can actually provide, while retaining the independent
benchmark's ability to detect incorrect production mathematics.

# Why TASK-005D Failed

TASK-005D correctly compared two independently written contracts and found
that they did not describe the same quantities:

1. TASK-005B treated the first bend as the requested base-ray turn, while
   TASK-005A uses the available `wrist -> base -> next point` polyline turn.
   In the neutral fixture this produces production proximal values of
   approximately 15.217593, 9.440035, 0, 9.440035, and 15.217593 degrees
   instead of five zeros; the original validation had a maximum 105-degree
   proximal disagreement.
2. TASK-005B used the original base-ray heading for spread truth. TASK-005A
   uses the actual proximal-phalanx direction after bending. The original
   `adversarial_near_180` case therefore differed by 160 degrees.
3. The fixture and production palm frames use different axis bases. The
   exact relationship requires a distinct constant mapping for LEFT and
   RIGHT; one common mapping cannot satisfy both.
4. Six valid fixture cases intentionally reach the production spread
   conditioning pole. Those channels are undefined, not complete-case
   failures.
5. The `degenerate_coincident_mcp` fixture remains accepted by frozen
   production without a warning. This is a real production validity defect,
   not a benchmark-contract disagreement.

The original TASK-005B and TASK-005D records remain unchanged as evidence of
these findings.

# Final Flexion Contract

Flexion is an unsigned geometric bend magnitude in degrees:

```text
0 degrees = straight geometric continuation
larger magnitude = greater turn
range = 0..180 degrees
```

The contract makes no signed flexion/extension or clinical goniometry claim.
The track order is `LEFT, RIGHT`, finger order is
`thumb,index,middle,ring,pinky`, and the three generic chain channels are
`proximal,middle,distal`.

# Proximal Bend Definition

The proximal channel is the independent analytic value:

```text
angle(wrist -> finger base, finger base -> next joint)
```

This is the available-skeleton geometric proximal bend proxy. It is not
isolated clinical MCP flexion because the 21-joint representation contains no
metacarpal-shaft point. No neutral offset is subtracted and no calibration is
applied. Middle and distal channels use the corresponding adjacent
polyline-turn geometry. The benchmark computes all three values from output
joint coordinates rather than importing the production formula.

# Final Spread Contract

Each finger's actual proximal-phalanx direction is taken from its output
joint pair:

```text
thumb  1 -> 2
index  5 -> 6
middle 9 -> 10
ring   13 -> 14
pinky  17 -> 18
```

The directions are projected into the palm plane defined by the generated
output geometry (wrist, index MCP, and middle MCP), normalized, and compared
as adjacent unsigned angles in this order:

```text
thumb-index, index-middle, middle-ring, ring-pinky
```

This deliberately uses the bent output direction, not the pre-bend base-ray
heading. No signed abduction/adduction or clinical spread claim is made.

# Spread Conditioning

The production 15-degree guard is frozen. For a unit proximal direction, the
projection norm must be at least:

```text
sin(15 degrees) = 0.2588190451...
```

If a direction is too close to the palm normal, its direction in the palm
plane is ill-conditioned. The corresponding spread pair is `NaN` and the
explicit `SPREAD_DIRECTION_DEGENERATE_<finger>` condition is retained. No
clamping or fabricated angle is permitted.

The revised catalog identifies six expected channel-conditioned cases:

```text
single_thumb_joint0_90deg
single_index_joint0_90deg
single_middle_joint0_90deg
single_ring_joint0_90deg
single_pinky_joint0_90deg
multi_curl_pinky
```

They contain 18 expected NaN spread values across both tracks and remain
valid benchmark geometries.

# Validity / NaN Contract

The final contract freezes Model B, channel-level validity:

| State | `valid_palm_frame` | `valid_kinematics` | Derived floats |
|---|---:|---:|---|
| No usable pose or invalid palm frame | false | false | all `NaN` |
| Usable palm frame, undefined spread channel(s) | true | false | finite flexion/orientation, only undefined spread channel(s) `NaN` |
| Fully usable pose | true | true | all required channels finite |

`valid_kinematics` is a strict convenience flag, not permission to discard
finite channels. `valid_palm_frame` gates orientation use, and each floating
channel must still be masked with its own finite state. The QA validator now
identifies this as `TASK-005-final-v2-model-B` and does not apply the old
all-or-nothing interpretation to strict-false/palm-true rows.

# Palm Orientation Contract

The frozen production frame remains the proper orthonormal
`[lateral | palmar normal | distal]` frame. Its raw output is retained. A
quaternion is normalized and stored in `[w,x,y,z]` order. Rotation matrices
must have orthogonality error and determinant error within the original
locked tolerances, with determinant `+1`.

For synthetic comparison only, the fixture's independently declared analytic
basis is compared after a fixed convention mapping. This does not change the
production orientation semantics or stored output. The raw production
matrix/quaternion consistency check is performed before comparison; the
cross-convention quaternion score is obtained by converting the mapped
comparison matrix to independent WXYZ form rather than pretending that the
raw quaternion was overwritten.

# LEFT / RIGHT Convention

For the frozen synthetic geometry, the production-to-fixture relationship is

```text
R_fixture = R_production @ C_side
```

The production frame basis derived from the literal wrist/MCP geometry is:

```text
P = [[-1, 0, 0],
     [ 0, 0, 1],
     [ 0, 1, 0]]
```

The fixture bases are `B_RIGHT = I` and
`B_LEFT = diag(-1, 1, -1)`, so `C_side = P.T @ B_side` gives:

```text
C_RIGHT = [[-1, 0, 0],
            [ 0, 0, 1],
            [ 0, 1, 0]]

C_LEFT  = [[ 1, 0, 0],
            [ 0, 0,-1],
            [ 0, 1, 0]]
```

Both are fixed proper rotations and are used for every fixture. They depend
only on hand side, are not fitted per case, and are not applied to raw data.
Local flexion and spread remain mirror-equivalent without ad-hoc sign
changes.

# Benchmark Revision

`evaluation/kinematics/final_contract.py` is the versioned TASK-005B2/final
truth layer (`TASK-005-final-v2`). It returns the same deterministic 86-case
catalog: 80 expected-valid and 6 intentionally invalid cases. It preserves
the original generator and computes truth independently from each generated
joint array:

- proximal/middle/distal truth is calculated from unsigned polyline turns;
- spread truth is calculated from actual output proximal directions and the
  geometry-derived palm plane;
- conditioning produces expected per-channel NaN masks;
- orientation truth keeps the fixture's analytic basis and uses fixed
  LEFT/RIGHT comparison mappings.

The original TASK-005B tolerances remain unchanged:

| Quantity | Locked limit |
|---|---:|
| Flexion absolute error | 1 degree |
| Spread absolute error | 1 degree |
| Known orientation angular error | 1 degree |
| Matrix orthogonality error | 1e-5 |
| `abs(det(R)-1)` | 1e-5 |
| Quaternion norm error | 1e-5 |
| Matrix/quaternion consistency | 1e-5 |

# QA Contract

The TASK-005C report and original behavior are retained historically. The
TASK-005D validator correction is now explicitly versioned in
`evaluation/kinematics_qa/contract.py` and included in every new QA summary.
The selected rule is:

```text
valid_palm_frame gates orientation;
valid flexion/spread channels are checked independently;
undefined channels are NaN;
strict valid_kinematics is true only when every required float is finite.
```

No production numerical output is changed by this QA update.

# Remaining Production Defect

The revised benchmark continues to require
`degenerate_coincident_mcp` to be rejected or explicitly flagged. Frozen
TASK-005A currently returns `valid_kinematics=true`, `valid_palm_frame=true`,
and no flags for both tracks. The E1 diagnostic therefore passes 5/6 invalid
fixtures and fails only this case. This remains an Opus-owned production-core
fix for TASK-005E2.

# Evaluation

The deterministic E1 harness ran every catalog case through the frozen
production adapter after separately self-checking the revised truth. It
compared finite values and expected NaN masks without invoking the historical
all-finite TASK-005B evaluator for conditioned cases.

# Results

The compact record is
`reports/kinematics/TASK-005E1-final-contract-results.json`.

| Check | Result |
|---|---:|
| Catalog | 86 cases |
| Expected-valid fixtures | 80 |
| Expected-invalid fixtures | 6 |
| Final flexion compatibility | 80/80 cases; 2,400/2,400 values; worst error 0 degrees |
| Final spread compatibility | 80/80 cases; expected masks match; worst finite error 1.42e-14 degrees |
| Side-mapped orientation compatibility | 80/80 cases; worst error 0 degrees |
| Side-mapped quaternion compatibility | 80/80 cases; worst error 0 degrees |
| Complete valid-case compatibility | 80/80 |
| Invalid-geometry handling | 5/6; coincident-MCP remains failing |

The independent self-check passed all catalog-count, geometry-derived truth,
conditioning, invalid-expectation, and fixed-mapping checks. These results
freeze the definitions; they do not constitute final TASK-005 production
readiness because the required coincident-MCP rejection is still outstanding.

The self-check also confirmed the 86-case rigid-transform invariants,
LEFT/RIGHT local equivalence, requested middle/distal bends, and exact
per-pair relationship between the degenerate-direction mask and spread NaNs.

# Failures / Limitations

- `degenerate_coincident_mcp` is intentionally not weakened or reclassified.
- The benchmark is mathematical and does not establish clinical anatomical
  accuracy.
- The orientation mapping is a documented comparison convention; raw
  production orientation remains in its own side-aware frame.
- The six conditioned cases demonstrate expected per-channel missingness;
  downstream code must preserve and consume the spread mask.
- No KArSL/WiLoR pilot rerun was necessary for this contract-only task.
- The original TASK-005B/D reports remain incompatible historical snapshots by
  design; the new versioned layer is the only contract to use after E1.

# Performance

The synthetic self-check and frozen adapter diagnostic are CPU-only and
complete in under one second locally. No model, checkpoint, KArSL video, or
large run artifact was generated. This task makes no model-throughput claim.

# Comparison

This is not a MediaPipe-versus-WiLoR comparison and selects no pose model. It
is a contract reconciliation between the frozen TASK-005A representation and
the independent TASK-005B fixtures. The historical TASK-005D disagreement
remains available for auditability.

# Tests

Executed during E1:

```text
python scripts/run_task005e1_contract.py
python -m unittest tests.test_task005e1_final_contract \
    tests.test_task005d_kinematics_validation \
    tests.test_task005c_kinematics_qa
Ran 35 tests ... OK
```

The targeted suite passed 35 tests, and the contract self-check passed. The
full repository validation also passed:

```text
python -m unittest discover -s tests -p 'test_*.py'
Ran 262 tests ... OK

python -m compileall -q evaluation tracking kinematics scripts tests
0 errors
```

The existing full-pilot QA was rerun read-only with the final Model-B
contract and returned `PASS`: 18/18 samples, 894 frames, 0 invalid-mask
issues, 0 non-finite issues, 0 non-positive determinants, and 215
strict-false/palm-true partial-channel instances.

# Recommendation

REVISE

Freeze this final contract and wait for the required Opus core correction
before declaring TASK-005 production-validated or beginning TASK-006.

# Reproducibility

- Base: `b41bc1808d09b1987ebdcf417e1bdadc42962f6d`.
- Frozen TASK-005A implementation:
  `564167420c7f5b4f12197fe36e7d2b59ae08ace0`.
- Frozen TASK-005B benchmark:
  `5f981d9f8c44408488f02b74f73a378197422830`.
- Frozen TASK-005D validation: `b41bc1808d09b1987ebdcf417e1bdadc42962f6d`.
- No random seed; all fixture parameters are literal and deterministic.
- The result JSON records the contract version, tolerances, case counts,
  side mappings, compatibility results, and remaining defect.
- Final QA was read-only against
  `/home/hatim/graduation-project-runs/wilor_karsl_pilot_tracked_task004d`
  and
  `/home/hatim/graduation-project-runs/wilor_karsl_pilot_kinematics_task005a`;
  its external result is
  `/home/hatim/graduation-project-runs/task005e1_qa/final-qa.json`.
- Original benchmark and production result files were not overwritten.

# Next Steps

1. Opus addresses `degenerate_coincident_mcp` in TASK-005E2 without changing
   the frozen contract or thresholds.
2. Rerun the E1 harness and QA checks against the exact corrected production
   commit.
3. Only after the invalid-geometry criterion passes should TASK-005 proceed
   to its final validation/merge decision.
4. TASK-006 remains out of scope.

# Final Contract

The repository's post-E1 TASK-005 contract is `TASK-005-final-v2`:

- unsigned geometric `flexion_deg` in generic `proximal,middle,distal`
  channels;
- proximal is the wrist-to-base geometric bend proxy, not clinical MCP;
- unsigned adjacent spread uses actual projected proximal-phalanx directions;
- 15-degree spread conditioning emits per-channel NaN plus explicit flag;
- Model-B channel-level validity with separate palm and strict flags;
- proper side-aware palm frames and normalized WXYZ quaternions;
- one fixed documented orientation mapping per hand side for synthetic scoring;
- original numeric thresholds remain locked;
- `degenerate_coincident_mcp` remains a required invalid case.

# Final Verdict

FINAL KINEMATICS CONTRACT FROZEN — WAITING FOR TASK-005E2
