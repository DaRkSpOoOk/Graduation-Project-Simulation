# Task
TASK-005C — Kinematics QA and data-contract tooling

# Branch
copilot/task-005c-kinematics-qa-tooling

# Scope
Add repository-level validation and diagnostics tooling for fixed TASK-005 kinematics outputs, including CLI, machine-readable summaries, and synthetic tests. No production kinematics mathematics were implemented or modified.

# Approach
Implemented a standalone evaluator under `evaluation/kinematics_qa/` that validates structure, alignment to TASK-004 tracking outputs, NaN/valid-state consistency, rotation/quaternion quality, distribution summaries, temporal diagnostics, and LEFT/RIGHT split diagnostics. Added a CLI to produce deterministic JSON and compact CSV summaries.

# Evidence / Sources
- Fixed TASK-005 contract from task statement.
- Existing repository patterns:
  - `evaluation/tracking/validation.py`
  - `scripts/run_task004c_validation.py`
  - `tracking/wilor/npz_io.py`
  - `reports/README.md`

# Files Changed
- `evaluation/kinematics_qa/__init__.py`
- `evaluation/kinematics_qa/contract.py`
- `evaluation/kinematics_qa/rotation_checks.py`
- `evaluation/kinematics_qa/statistics.py`
- `evaluation/kinematics_qa/validator.py`
- `scripts/validate_task005_kinematics.py`
- `tests/test_task005c_kinematics_qa.py`
- `reports/kinematics/TASK-005C-kinematics-qa-tooling.md`

# Objective
Build reusable QA and data-contract tooling that validates TASK-005 outputs without depending on production kinematics formulas.

# Fixed Contract
Validator enforces required `hand_kinematics.npz` arrays, `hand_kinematics_meta.json`, canonical metadata (`LEFT/RIGHT`, finger order, quaternion convention), dimensional checks, and frame/timestamp monotonicity/integrity.

# Architecture
- `contract.py`: fixed schema constants, sample loading, structural validation.
- `rotation_checks.py`: matrix orthogonality/determinant and quaternion conversion helpers.
- `statistics.py`: percentile/statistics helpers.
- `validator.py`: run-level orchestration, tracking alignment checks, mask/finite checks, diagnostics aggregation, JSON/CSV writers.
- CLI script: argument parsing and output generation.

# Structural Validation
Checks required files and arrays, exact dimensions, `valid_kinematics` boolean dtype, frame count consistency, frame uniqueness, frame monotonic order, finite/monotonic timestamps, and metadata contract fields.

# TASK-004 Alignment
Per sample checks:
- sample ID set parity with tracked run
- frame count equality
- exact `frame_index` equality
- exact `timestamp_seconds` equality
- exact `tracking_state_code` vs tracked `state_code`
- exact `source_raw_detection_index` vs tracked `raw_detection_index`

Mismatches are reported with sample/frame/track references.

# State / NaN Validation
For each frame/track:
- if `valid_kinematics=false`: all derived floating fields must be NaN
- if `valid_kinematics=true`: all derived floating fields must be finite

Violations are listed explicitly.

# Rotation Matrix QA
For valid entries reports:
- orthogonality errors (`R^T R`)
- determinant absolute error from 1
- counts of non-finite matrices
- determinant non-positive violations
- worst offending sample/frame/track

# Quaternion QA
For valid entries reports:
- non-finite quaternion count
- norm absolute error statistics
- matrix/quaternion angular disagreement statistics
- worst disagreement location

# Distribution Metrics
Computes channel-level stats (`count/min/p1/p50/p95/p99/max`) for:
- flexion by `hand/finger/joint`
- spread by `hand/adjacent pair`

Flags suspicious flexion values (e.g., strongly negative or >180°) without clamping.

# Temporal Diagnostics
For consecutive valid frames computes absolute deltas for:
- flexion channels
- spread channels
- palm orientation angular change

Reports mean/p95/p99 and max-event references.

# LEFT / RIGHT Diagnostics
Produces independent LEFT and RIGHT aggregate distributions for flexion and spread to surface convention inconsistencies without auto-failing on distribution differences.

# CLI
`python scripts/validate_task005_kinematics.py --tracked-run <TASK004_TRACKED_RUN> --kinematics-run <TASK005_KINEMATICS_RUN> --output-json <summary.json> --output-csv <summary.csv>`

# Tests
Added synthetic fixture tests in `tests/test_task005c_kinematics_qa.py` covering valid path, structural corruption, alignment corruption, NaN/valid violations, rotation/quaternion faults, provenance mismatch, deterministic JSON, and CLI output generation.

# How to Run
- `python -m unittest discover -s tests -p 'test_*.py'`
- `python -m compileall -q evaluation scripts tests`

# Evaluation
Validation outputs are machine-readable and deterministic. Contract/alignment errors produce explicit failure details with sample/frame/channel context.

# Results
Tooling and tests implemented. Full test suite and compile checks executed successfully.

# Failures / Limitations
- Validator checks structural/data consistency and mathematical validity of represented rotations, not anatomical correctness of angle derivation formulas.
- CSV output is compact channel summary only; full-frame dump is intentionally omitted by default.

# Performance
Operations are linear in number of samples, frames, and channels; designed for run-level QA and diagnostics.

# Comparison
This tooling is extractor-agnostic QA infrastructure and does not overlap TASK-005A production math or TASK-005B independent mathematics benchmark responsibilities.

# Recommendation (KEEP / REVISE / REJECT / NEEDS MORE EVALUATION)
KEEP

# Reproducibility
- Branch: `copilot/task-005c-kinematics-qa-tooling`
- Base commit observed at start: `ba6389f334ea5277b303c3f7795c919def4bf08e`
- Commands:
  - `python -m unittest discover -s tests -p 'test_*.py'`
  - `python -m compileall -q evaluation scripts tests`

# Next Steps
Run this validator against TASK-005 production outputs when generated and use JSON/CSV diagnostics to gate readiness and investigate anomalies.
