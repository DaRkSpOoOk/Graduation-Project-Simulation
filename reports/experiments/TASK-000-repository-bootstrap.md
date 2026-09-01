# Task
Bootstrap repository architecture, governance, instructions, and lightweight Python scaffolding for Milestone 1.

# Branch
`copilot/bootstrap-repository-architecture`

# Scope
Repository bootstrap only: structure, docs, minimal schemas/tests, and configuration scaffolding.

# Approach
Created the required directory architecture with `.gitkeep` placeholders, repository governance docs (`AGENTS.md`, Copilot instructions), area READMEs, report conventions, and a minimal extractor-agnostic pose schema scaffold. Added lightweight Python project metadata and small unit tests that do not require heavy datasets/models.

# Evidence / Sources
- Problem statement constraints in this task request.

# Files Changed
- Root docs/config: `README.md`, `AGENTS.md`, `.gitignore`, `pyproject.toml`
- Copilot guidance: `.github/copilot-instructions.md`, `.github/instructions/*.md`
- Area docs: `*/README.md` for required major directories
- Common schema scaffold: `pose/common/schema.py`, package init files
- Tests: `tests/test_pose_common_schema.py`, `tests/test_repository_structure.py`
- Reporting conventions: `reports/README.md`

# How to Run
```bash
python -m unittest discover -s tests -p "test_*.py"
python -m tomllib pyproject.toml  # parse validity (Python 3.11+)
```

# Evaluation
Bootstrap validation only:
- directory structure existence checks
- schema object construction and JSON serialization checks
- `pyproject.toml` parse check

# Results
Repository structure and scaffolding created as required. Lightweight tests pass locally.

# Failures / Limitations
No model inference, tracking algorithm, kinematics solver, or virtual sensor simulation was implemented by design.

# Performance
Not applicable for bootstrap scaffolding.

# Comparison
Not applicable yet; no competing implementation exists in this bootstrap PR.

# Recommendation
**KEEP** — structure and governance are ready for parallel Milestone 1 implementation/evaluation branches.

# Reproducibility
- Python target: 3.10+
- No heavy ML dependencies added
- Deterministic unit tests only
- Branch and file-level scope documented

# Next Steps
1. Add video I/O timestamp utility prototypes with unit tests.
2. Add extractor-agnostic evaluation schema/manifests.
3. Implement first isolated extractor branch (e.g., `pose/mediapipe/`) with report and benchmark outputs.
