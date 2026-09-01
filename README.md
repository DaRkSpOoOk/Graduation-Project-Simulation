# Graduation-Project-Simulation

Graduation-project research codebase for a software-only Arabic Sign Language digital-twin pipeline.

## Current Milestone 1

**CURRENT MILESTONE:** `RGB VIDEO → RELIABLE TEMPORAL 3D DUAL-HAND DATA`

Do not implement recognition, Arabic NLP, or TTS in this milestone.

**STOP AFTER MILESTONE 1 EVALUATION BEFORE SELECTING THE FINAL VIRTUAL SENSOR DESIGN.**

## Scope and roadmap

Conceptual pipeline:

Arabic Sign Language RGB Video → Dual-Hand 3D Motion Reconstruction → Temporal Hand Tracking → Anatomical Hand Kinematics → Virtual Smart Glove / Digital Twin → Later Recognition → Later Arabic NLP/TTS

The physical Hall-sensor glove architecture is not the active system design for this repository.

## Branch and review workflow

- Work on isolated branches (never commit directly to `main`).
- Keep parallel approaches independently reviewable.
- Open PRs for human review.
- Do **not** auto-merge.

## Reporting is mandatory

Substantial tasks must create/update a Markdown report under `reports/`.

Required report sections are documented in:
- `reports/README.md`
- `AGENTS.md`

## Raw data immutability

Preserve reproducible stage separation:

source RGB → raw pose → temporal tracking → normalized geometry → joint angles → evaluation

Never destructively overwrite raw extraction outputs.

## Dataset and artifact policy

- `datasets/` contains manifests and local organization only.
- Do not commit large videos, checkpoints, or generated arrays.
- Use `.gitignore` patterns and local storage for large artifacts.
- Milestone 1 initial benchmark should use a small curated sample (about 10–20 videos).

## Hardware constraint

Assume primary practical benchmark hardware is **NVIDIA RTX 2060 SUPER**.
Do not assume datacenter GPUs (A100/H100).

## Repository structure

- `video_io/`: decoding, timestamps, frame extraction
- `pose/`: extractor implementations + common interfaces
- `tracking/`: temporal hand identity and continuity
- `hand_kinematics/`: canonical hand frames and anatomical angles
- `visualization/`: overlays, playback, plotting
- `evaluation/`: metrics, protocols, benchmarks
- `virtual_sensors/`: reserved for later stages only
- `experiments/`: experiment manifests
- `research/`: sources, papers, rationale, evidence
- `reports/`: permanent execution/research reports

## Basic Python setup

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m unittest discover -s tests -p "test_*.py"
```
