# Task

TASK-005B — Independent Kinematics Benchmark.

# Branch

luna/task-005b-kinematics-benchmark, based on ba6389f334ea5277b303c3f7795c919def4bf08e.

# Scope

This task creates an extractor-independent mathematical benchmark for the future TASK-005A kinematics implementation. It generates compact synthetic 21-joint hand skeletons, records analytic ground truth from the generation parameters, validates the required result contract, and defines strict comparison tolerances.

This task does not implement the production kinematics extractor. It does not inspect, import, copy, or score opus/task-005a-hand-kinematics, and it does not use KArSL videos, WiLoR, MediaPipe, smoothing, interpolation, kinematics from the production package, or sensor simulation.

# Objective

Provide cases that can expose:

- incorrect flexion sign or zero-angle convention;
- wrong LEFT mirroring;
- global-frame contamination by translation, scale, or rigid rotation;
- incorrect palm-frame construction;
- scale- or translation-dependent angles;
- wrong quaternion order or orientation;
- failure to reject invalid geometry.

The future adapter can pass a production result with shapes [F,2,5,3], [F,2,4], [F,2,3,3], and [F,2,4] to evaluate_sequence and compare it with the analytic fixture truth.

# Approach

The benchmark uses no random numbers and no production implementation. Every case is generated from explicit parameters. The complete catalog is built by build_benchmark_catalog() and contains 86 one-frame paired LEFT/RIGHT cases. A compact JSON file contains representative serialized parameter descriptors for round-trip testing; bulky joint arrays are generated on demand.

Track order is fixed to LEFT, RIGHT. Finger order is fixed to thumb, index, middle, ring, pinky. The synthetic joint indices are:

    0 wrist
    1-4 thumb CMC/MCP/IP/TIP
    5-8 index MCP/PIP/DIP/TIP
    9-12 middle MCP/PIP/DIP/TIP
    13-16 ring MCP/PIP/DIP/TIP
    17-20 pinky MCP/PIP/DIP/TIP

# Fixture Geometry

The fixture is intentionally simple rather than photorealistic:

- The palm lies in a canonical XY plane, with the wrist at (0, -0.85, 0) and five base joints across the x direction at [-0.60, -0.30, 0, 0.30, 0.60].
- Each finger has three segments of lengths (0.36, 0.30, 0.24) by default.
- Four adjacent base-ray gaps are independently controllable. For requested signed gaps s0..s3, the base headings are h0 = -0.5 * sum(s) and hi = h(i-1) + s(i-1). The benchmark spread truth is the unsigned gap abs(hi - h(i-1)).
- A bend control [b0,b1,b2] applies a signed turn to the first, second, and third outgoing finger segments. b0 is measured relative to the straight base ray; each later value is the relative turn from the preceding segment. Positive bends turn toward the canonical +Z palm-normal direction. Thus the analytic local flexion truth is exactly the requested [b0,b1,b2].
- The thumb uses the same three-control construction at its CMC/MCP/IP chain. This is a mathematical fixture convention, not an anatomical claim.

For a local canonical point p, the generated world point is:

    p_world = R_global * (scale * B_side * p) + translation

where R_global is a known proper rotation, scale is positive, and B_RIGHT = I. The locked left-side basis is:

    B_LEFT = diag(-1, 1, -1)

It mirrors the across-palm coordinate and reverses the palm normal while retaining a proper right-handed local frame. The mirror_points helper is deterministic and involutive. LEFT and RIGHT flexion/spread truth is identical even though their world coordinates differ.

# Expected Angle Conventions

Flexion is signed local segment turning with:

    straight = approximately 0 degrees
    positive bend = larger positive angle

The single-bend catalog independently exercises every finger, every one of its three chain joints, and 30°, 60°, and 90°. Multi-joint curls include [30,45,60], [60,75,90], and [90,90,90]. Independent-finger cases cover index-only curl and ring-plus-pinky curl.

Spread is an unsigned angle between adjacent base rays, so mirrored equivalents have the same expected values even when the signed heading direction changes. Dedicated spread cases use 5°, 10°, 20°, and 30° for each adjacent gap.

# Rigid-Transform Invariants

The expected flexion and spread arrays are unchanged when the same synthetic pose is translated by large arbitrary vectors or uniformly scaled by 0.5, 1, 2, or 5. Eight orientation cases apply identity, 90° X/Y/Z, 180° X/Y/Z, and a composed rotation. The expected palm orientation changes as R_global * B_side, while local flexion and spread remain unchanged.

The composed rotation convention is column-vector Rz(z) @ Ry(y) @ Rx(x). This is explicit in rotation_matrix_xyz and is tested independently against the transformed point cloud.

# Quaternion / Orientation Tests

palm_rotation_matrix maps the side-local palm basis into world coordinates. palm_quaternion_wxyz is generated from that known matrix in [w,x,y,z] order. The contract checks matrix orthogonality, determinant, quaternion norm, and matrix/quaternion equivalence. Quaternion angular comparison is sign-invariant because q and -q represent the same rotation.

The known orientation catalog includes identity, 90° X/Y/Z, 180° X/Y/Z, and an arbitrary composed rotation. The 90° axis cases have analytic expected quaternions such as X: [sqrt(1/2), sqrt(1/2), 0, 0], Y: [sqrt(1/2), 0, sqrt(1/2), 0], and Z: [sqrt(1/2), 0, 0, sqrt(1/2)] on the RIGHT track.

# Adversarial Cases

The catalog includes:

- 0.1° almost-straight bend;
- 179.9° near-180° geometry;
- a tiny but nonzero bone below the numerical validity floor;
- a mirrored pose after arbitrary composed rotation;
- palm facing away through a 180° X rotation;
- palm upside down through a 180° Y rotation;
- signed spread that creates a crossing-like finger arrangement;
- thumb opposition-like spread and curl.

Five intentional degenerate cases cover zero-length bone, coincident MCP points, collinear palm-frame landmarks, NaN joint, and Inf joint. The tiny-bone adversarial case is also intentionally invalid, giving six invalid catalog cases total. geometry_validity returns a reason, and evaluate_sequence hard-fails rather than scoring an invalid fixture as plausible numeric output.

# Benchmark Contract

The later production adapter must provide:

    flexion_deg          [F,2,5,3]
    adjacent_spread_deg  [F,2,4]
    palm_rotation_matrix [F,2,3,3]
    palm_quaternion_wxyz [F,2,4]

KinematicsResult, coerce_result, and validate_result form the adapter boundary. Validation hard-fails on missing fields, wrong rank/shape, non-finite values, non-orthogonal matrices, determinants not near +1, invalid quaternion norms, or matrix/quaternion disagreement. evaluate_sequence then compares a valid adapter result to synthetic truth without recalculating truth from the production joints.

# Tolerances

These thresholds are locked in evaluation/kinematics/benchmark_contract.py before TASK-005A results are viewed:

| quantity | locked tolerance |
|---|---:|
| known flexion absolute error | ≤ 1.0° |
| known spread absolute error | ≤ 1.0° |
| rotation-matrix orthogonality element error | ≤ 1e-5 |
| abs(det(R) - 1) | ≤ 1e-5 |
| quaternion norm error | ≤ 1e-5 |
| matrix/quaternion element consistency | ≤ 1e-5 |
| known palm orientation angular error | ≤ 1.0° |

No tolerance is tuned from a production result. The synthetic generator itself uses tighter internal construction checks (1e-8) for accepting supplied global rotation matrices and uses 1e-8 geometry floors for intentionally invalid bone/palm configurations.

# Evidence / Sources

The ground truth is mathematical fixture truth derived directly from explicit generator parameters, not a dataset annotation or model output. The required output shapes and conventions are the TASK-005 contract supplied for this benchmark. Repository reporting requirements are defined in reports/README.md.

No external model, checkpoint, RGB video, or Opus branch was consulted. No production kinematics package is imported by the generator or benchmark self-check.

# Files Changed

- evaluation/kinematics/__init__.py — public benchmark exports.
- evaluation/kinematics/synthetic_hand.py — deterministic 21-joint generator, transforms, mirror convention, analytic truth, validity checks, and 86-case catalog.
- evaluation/kinematics/benchmark_contract.py — later-adapter result schema and hard validation.
- evaluation/kinematics/metrics.py — flexion/spread/orientation/quaternion comparison helpers.
- scripts/run_task005b_benchmark.py — analytic catalog self-check; it does not run a production extractor.
- tests/fixtures/kinematics/task005b_representative_cases.json — compact serialized representative parameter descriptors.
- tests/test_task005b_kinematics_benchmark.py — generator, transform, mirror, fixture, degeneracy, contract, and quaternion tests.
- reports/kinematics/TASK-005B-independent-kinematics-benchmark.md — this benchmark specification.

# How to Run

From the repository root:

    python scripts/run_task005b_benchmark.py

This prints the catalog count, category counts, six invalid cases, and locked tolerances. A later adapter can use:

    from evaluation.kinematics import evaluate_sequence
    score = evaluate_sequence(production_result, synthetic_case.generate())

where production_result is a mapping or KinematicsResult with the four required fields.

# Evaluation

The benchmark self-check validates that its own generated geometry matches its declared analytic parameters. It does not claim that a production implementation passes until a later task supplies production results. Invalid cases are expected to be rejected before numeric comparison.

# Results

The catalog contains 86 deterministic cases:

| category | count |
|---|---:|
| neutral | 1 |
| single bend | 45 |
| multi-joint curl | 3 |
| independent fingers | 2 |
| spread | 4 |
| mirror | 3 |
| translation | 3 |
| uniform scale | 4 |
| quaternion/orientation | 8 |
| degenerate | 5 |
| adversarial | 8 |
| **total** | **86** |

The analytic self-check passed all 80 valid cases and identified all six intentionally invalid cases. The compact representative fixture contains five serialized parameter cases and round-trips successfully in the unit tests.

# Tests

- python scripts/run_task005b_benchmark.py — passed; 80 valid cases self-checked and 6 invalid cases recognized.
- python -m unittest tests.test_task005b_kinematics_benchmark — passed, 15 tests.
- python -m unittest discover -s tests -p 'test_*.py' — passed, 153 tests.
- python -m compileall -q evaluation scripts tests — passed.

# Failures / Limitations

- The skeleton is mathematical and not intended to model full human anatomy, tendon coupling, camera projection, or soft-tissue behavior.
- The thumb uses the same three-turn abstraction as the other chains; the contract is tested, but anatomical thumb interpretation remains a future design decision.
- Spread is defined from synthetic base-ray headings rather than a clinical abduction/adduction convention.
- The LEFT basis is a documented coordinate convention. A future production implementation must adapt its own palm-frame convention to this contract rather than silently changing the benchmark.
- A one-frame synthetic case cannot test temporal filtering, continuity, or noise robustness. TASK-005D can repeat cases over F>1 and add controlled perturbations without changing these ideal truth cases.
- A single-frame contract result cannot demonstrate invalid-input behavior by itself; the invalid fixture metadata and geometry-validity predicate define the required rejection path.
- A single annotator or KArSL source is not involved; this is deliberately a synthetic benchmark.

# Performance

This is a lightweight CPU-only synthetic benchmark. It has no model loading, disk-heavy data, GPU requirement, or production throughput claim. The catalog self-check generates 86 small cases and is expected to run in well under one second in the repository environment.

# Comparison

No TASK-005A implementation, commit, test, output, or result was inspected or scored. This benchmark is intentionally frozen as the independent reference for a later comparison task.

# Recommendation

KEEP

Keep the benchmark and its tolerances unchanged for the later TASK-005D adapter/evaluation. Any future fixture revision should be source-justified and versioned rather than changed in response to a production disagreement.

# Reproducibility

- Base commit: ba6389f334ea5277b303c3f7795c919def4bf08e.
- Branch: luna/task-005b-kinematics-benchmark.
- Python: 3.14.4 in the working environment.
- NumPy: 2.5.2, matching the repository dependency.
- Random seed: none; generation is deterministic from literal parameters.
- Track order: LEFT, RIGHT; finger order: thumb,index,middle,ring,pinky.
- Ground truth: direct bend/spread/rotation parameters and analytic rigid transforms; no production implementation dependency.
- Fixture catalog: 86 cases; serialized representative descriptors at tests/fixtures/kinematics/task005b_representative_cases.json.
- Locked tolerance values are stored in CONTRACT_TOLERANCES and reproduced in this report.

# Next Steps

TASK-005D can connect TASK-005A output to coerce_result, validate_result, and evaluate_sequence, run every valid fixture, and verify explicit rejection for degenerate fixtures. It should report per-case flexion/spread/orientation/quaternion errors and retain the distinction between a mathematically valid result and a production result obtained from noisy RGB data. TASK-005B does not start TASK-005A or TASK-005D.
