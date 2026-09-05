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

## Primary Core-28 application

TASK-007G is the current native desktop surface under `smart_glove_app/`:
PySide6, QML and Qt Quick 3D keep one persistent LEFT and RIGHT rigged hand
alive for the whole application lifetime. The legacy Tk/Matplotlib entry point
remains available for historical/debug reproduction only.

Install the optional GUI dependency and launch it. Only `--run-root` is
required; the application starts with two natural-skin hands facing the viewer
and an empty queue, and `--checkpoint` adds recognition:

```powershell
python -m pip install -e ".[gui,recognition]"
python -m smart_glove_app `
  --run-root "..\graduation-project-runs\task008-core28-full" `
  --checkpoint "..\graduation-project-runs\task009c-core28-deployment\deployment.pt"
```

`--text` is optional and only exists for smoke/demo runs; the expected flow is
launch, then click the Arabic Core-28 keyboard.

### Presentation layer

- `--rig-asset` points at the directory holding the two per-hand GLBs and
  defaults to the ignored local path `assets-local/blendswap_hands_v1/`. One
  file per hand is deliberate: Qt Quick 3D mis-binds a glTF that carries two
  skins and deforms the second mesh with the first skeleton.
- `--rig-profile` is the tracked, project-owned presentation profile at
  `smart_glove_app/assets/rig_profiles/task007g_hands.json`.
- `--view palm|back` chooses the starting composition; the in-app button
  toggles between the two deterministically. There is no free orbit in normal
  operation - the camera is solved from the layout and the viewport aspect, so
  the framing is identical at every window size.
- `--appearance skin|glove|wireframe` chooses the starting material; the
  default is a natural light-skin PBR material.
- `--diagnostics` opens the technical drawer (FPS, frame, sample, recognition,
  appearance/speed/smoothing controls) at startup. It is closed by default.
- `--screenshot`, `--screenshot-series` and `--screenshot-interval` capture the
  rendered window, which is how the TASK-007G visual acceptance evidence was
  produced.
- `--debug-mano-points` still shows the legacy 778-point representation as an
  explicit diagnostics overlay. It is never part of normal playback.

The Blender working copy, the exported GLBs and MANO files are local assets
under the ignored `assets-local/` path. The untouched source snapshot is kept
alongside them as `blendswap_hands_v1_ORIGINAL_PRESERVED.blend`. See
`reports/visualizer/TASK-007G-visual-acceptance.md` for the asset derivation,
source license and the visual acceptance evidence.
