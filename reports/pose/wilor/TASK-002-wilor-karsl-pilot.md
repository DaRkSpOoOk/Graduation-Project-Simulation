# Task

Evaluate WiLoR (multi-hand, MANO-based 3D hand reconstruction) as the
high-fidelity experimental alternative to the MediaPipe baseline, on the
same KArSL Milestone-1 pilot definition, and give an evidence-based
GO/NO-GO recommendation for continuing this line of work.

This report covers two phases:

- **Phase A** (initial session): official WiLoR installation, adapter,
  evaluation/serialization infrastructure, and a **detector-only** pilot
  run, blocked from full 3D reconstruction because the licensed MANO model
  file was unavailable.
- **Phase B** (this update): the human researcher obtained a licensed MANO
  package (`MANO_RIGHT.pkl` + `MANO_LEFT.pkl`, local-only, not committed)
  and the **full official WiLoR + MANO pipeline was executed** on the same
  18 pilot videos, producing real 3D reconstruction, evaluation, and visual
  validation.

# Branch

`opus/wilor-karsl-pilot`

# Scope

`pose/wilor/`, `evaluation/metrics/hand_pose_metrics.py`,
`evaluation/benchmarks/wilor_karsl_pilot.py`, `tests/test_wilor_*.py`, this
report. No recognition, NLP, TTS, virtual sensor, or Hall/IMU simulation
work was done, per the Milestone 1 scope. The official WiLoR implementation
(model, MANO layer via SMPL-X, YOLO detector) is used as-is throughout;
nothing neural was reimplemented -- see "Implementation".

# Primary sources

- Paper: Potamias et al., *WiLoR: End-to-end 3D Hand Localization and
  Reconstruction in-the-wild*, CVPR 2025. arXiv:2409.12259.
- Official repository: https://github.com/rolpotamias/WiLoR, commit
  `fcb911312a38fa8badd30d9656a167485d61b8f9` (pushed 2026-04-07, cloned
  fresh for this task).
- Official project page: https://rolpotamias.github.io/WiLoR/
- License file: `license.txt` in the official repo (Creative Commons
  Attribution-NonCommercial-NoDerivatives 4.0).
- MANO: https://mano.is.tue.mpg.de (model) and
  https://mano.is.tue.mpg.de/license.html (license).
- Checkpoint hosting: https://huggingface.co/spaces/rolpotamias/WiLoR
  (`pretrained_models/detector.pt`, `pretrained_models/wilor_final.ckpt`).
- Source code read directly (not summarized from tutorials):
  `wilor/models/wilor.py`, `wilor/models/mano_wrapper.py`,
  `wilor/models/__init__.py` (`load_wilor`), `wilor/utils/renderer.py`,
  `wilor/datasets/vitdet_dataset.py`, `demo.py`, `requirements.txt`,
  `pretrained_models/model_config.yaml`.
- Phase B: the MANO pickle files themselves (`MANO_RIGHT.pkl`,
  `MANO_LEFT.pkl`), loaded and inspected locally to verify their structure
  and the left/right convention (see "Left/right MANO convention
  validation").

No blog posts or third-party tutorials were used for installation or
architecture claims; all commands below were run directly against the
official repository, its official checkpoint host, and (Phase B) the
locally-supplied licensed MANO files.

# WiLoR architecture findings

- Two-stage pipeline: (1) a YOLOv8-based detector (`ultralytics`, via
  `pretrained_models/detector.pt`) that localizes hand bounding boxes and
  classifies handedness per box (`cls == 0` -> left, `cls == 1` -> right);
  (2) a ViT-backbone + transformer-decoder refinement head
  (`wilor/models/wilor.py: WiLoR.forward_step`) that regresses MANO pose
  parameters, shape (betas), global orientation, and a weak-perspective
  camera from an image crop around each detected hand.
- MANO reconstruction is mandatory, not optional: `WiLoR.__init__`
  unconditionally does `self.mano = MANO(**mano_cfg)`
  (`wilor/models/wilor.py`), and `MANO` (`wilor/models/mano_wrapper.py`)
  subclasses `smplx.MANOLayer`, which loads a MANO model file at
  construction time. **Phase B confirms this path works end-to-end**: with
  `MANO_RIGHT.pkl` in place, `self.mano = MANO(**mano_cfg)` succeeds and
  produces real vertices/joints (see "Full-mode smoke test" below).
- Explicit multi-hand support: the detector proposes any number of boxes
  per image; each is cropped, batched, and passed independently through the
  reconstruction model (`demo.py`), so N hands (0, 1, 2, ...) per frame are
  handled natively, not just a fixed left+right pair. **Confirmed on real
  data in Phase B**: frame 39 of `karsl_test_s02_sign0176_repfirst`
  genuinely produced 3 detections (2 "left" + 1 "right") in a single frame
  -- see "Occlusion observations".
- Handedness convention: MANO is inherently a right-hand model. For
  left-hand detections, WiLoR mirrors the x-axis of vertices/joints and the
  camera's y-parameter post-hoc (`demo.py`:
  `verts[:,0] = (2*is_right-1)*verts[:,0]`) rather than using a separate
  left-hand MANO model. **Phase B empirically validates this**: see
  "Left/right MANO convention validation".
- Official config (`pretrained_models/model_config.yaml`, commit above):
  `MODEL.IMAGE_SIZE=256`, `EXTRA.FOCAL_LENGTH=5000`,
  `MANO.NUM_HAND_JOINTS=15`, detector confidence `0.3` and bbox
  `rescale_factor=2.0` (both from `demo.py` defaults).
- Fast/FP16 mode: official `--fast` flag does `model.half()` +
  `torch.compile(model.backbone)` + a backbone layer-skip flag, documented
  as up to 1.6x faster with ~0.05mm MPJPE degradation (repo README, "Update
  March 2026"). **Exercised in Phase B** on a 3-clip subset -- see "Fast
  mode comparison". Getting it working surfaced two real bugs/gaps (one in
  this adapter, one an environment gap), both documented and fixed under
  "Installation".
- `ViTDetDataset` (`wilor/datasets/vitdet_dataset.py`) takes an `fp16: bool`
  constructor argument that must match whether the model itself was cast to
  half precision -- `demo.py` passes `fp16=args.fast` through from its CLI
  flag. This adapter's `extract_frame_full()` initially did not forward
  this flag (a real bug, found and fixed in Phase B -- see "Failures").
- Batch size: WiLoR batches hands *within one image* (all boxes for a
  frame go through one `DataLoader` batch); it does not batch across
  video frames. Per Task 3's `batch_size = 1` instruction, this
  implementation processes exactly one video frame at a time
  (`pose/wilor/video_processing.py`), which is the natural unit here.

# License / MANO restrictions

- **WiLoR code/weights**: CC-BY-NC-ND 4.0 (`license.txt`, verified
  verbatim). Non-commercial use and reproduction are permitted; producing
  and *sharing* Adapted Material (e.g., redistributing a modified
  checkpoint) is not. This graduation-project research use is
  non-commercial and does not redistribute the model; only original
  adapter/glue code and small measurement outputs are committed.
- **Ultralytics (detector backbone library)**: AGPL-3.0 unless under a
  commercial Ultralytics license. Relevant if this were ever deployed as a
  network service; not a concern for local, non-commercial research
  inference.
- **MANO model files**: gated behind manual account creation and license
  acceptance at https://mano.is.tue.mpg.de -- there is no unauthenticated
  or programmatic download path. **Phase A blocker, now resolved**: the
  human researcher independently created a MANO account, accepted the
  license, and supplied `MANO_RIGHT.pkl` + `MANO_LEFT.pkl` locally at
  `~/.cache/wilor_assets/mano_data/`. Per the license and this task's
  explicit instructions, these files are **not committed, not copied into
  the repository, and not redistributed** -- see `.gitignore`/"Git safety"
  below. Only their SHA-256 fingerprints are recorded (a fingerprint is not
  the licensed content itself) for reproducibility bookkeeping.
- The detector checkpoint (`detector.pt`) and the WiLoR reconstruction
  checkpoint (`wilor_final.ckpt`) are hosted unrestricted on the official
  HuggingFace Space and were downloaded successfully (checksums below).

# Installation

Commands actually run in this environment (`.venv`, Python 3.14.4),
in order:

```bash
# 1. WiLoR source (not a published PyPI package; imported from checkout)
git clone https://github.com/rolpotamias/WiLoR.git ~/.cache/wilor_assets/WiLoR
# commit fcb911312a38fa8badd30d9656a167485d61b8f9

# 2. PyTorch (CUDA build auto-selected by pip for this machine)
pip install torch
# -> torch==2.13.0+cu130 (see "Exact environment")

# 3. WiLoR runtime dependencies (see pose/wilor/requirements.txt)
pip install numpy opencv-python scikit-image pytorch-lightning yacs timm \
    einops pandas smplx==0.1.28 ultralytics==8.1.34 pyrender dill

# 4. chumpy (upstream requirements.txt pins it via git; its setup.py does
#    `import pip`, which fails inside pip's isolated build env)
pip install --no-build-isolation "chumpy @ git+https://github.com/mattloper/chumpy"

# 5. Unrestricted checkpoints (no MANO account needed for these two)
wget -P ~/.cache/wilor_assets/pretrained_models \
  https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt
wget -P ~/.cache/wilor_assets/pretrained_models \
  https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/wilor_final.ckpt
cp ~/.cache/wilor_assets/WiLoR/pretrained_models/model_config.yaml \
   ~/.cache/wilor_assets/pretrained_models/
cp ~/.cache/wilor_assets/WiLoR/mano_data/mano_mean_params.npz \
   ~/.cache/wilor_assets/mano_data/

# 6. Phase B: human-supplied licensed MANO files, placed locally (not committed)
#    ~/.cache/wilor_assets/mano_data/MANO_RIGHT.pkl
#    ~/.cache/wilor_assets/mano_data/MANO_LEFT.pkl

# 7. Phase B, fast-mode only: torch.compile's Triton backend needs Python
#    development headers, absent by default in this environment
sudo apt-get install -y python3.14-dev
```

Three real, reproducible Phase-A dependency issues were found and resolved
(not skipped):

1. **`xtcocotools`** (listed in upstream `requirements.txt`) fails to build
   on this environment both with and without `--no-build-isolation`
   (`ModuleNotFoundError: numpy` isolated, then
   `KeyError: '__version__'` non-isolated -- a broken legacy `setup.py`).
   Resolution: confirmed by grepping the `wilor/` package that
   `xtcocotools` is not imported anywhere on the inference path (only
   relevant to training/COCO-format tooling not present in this repo
   checkout), so it was simply not installed. Documented as excluded in
   `pose/wilor/requirements.txt`.
2. **`torch.load` weights_only default** (PyTorch changed the default to
   `True` in 2.6; we are on 2.13.0): `ultralytics==8.1.34` (pinned by
   WiLoR, released before that change) calls `torch.load(file,
   map_location="cpu")` with no `weights_only` argument, so loading the
   official `detector.pt` raises
   `UnpicklingError: ... GLOBAL ultralytics.nn.tasks.PoseModel was not an
   allowed global`. Resolution: `pose/wilor/model_loader.py:
   _load_yolo_checkpoint` temporarily restores the pre-2.6 default
   (`weights_only=False`) only around the `YOLO(...)` construction call,
   scoped and documented, since `detector.pt` is a trusted, official
   checkpoint.
3. **`pyrender` coupling**: `wilor/utils/__init__.py` unconditionally does
   `from .renderer import Renderer`, which imports `pyrender` at module
   level -- so even `from wilor.utils import recursive_to` pulls in
   `pyrender`/OpenGL. `pyrender` itself installs and imports cleanly
   headless (no `OffscreenRenderer` is ever instantiated by this adapter),
   so this was not a hard blocker, but the *coupling* meant the small
   `cam_crop_to_full` math helper could not be imported without it either.
   Resolution: reproduced that ~10-line function locally in
   `pose/wilor/geometry.py` (attributed, unmodified formula) instead of
   importing it from `wilor.utils.renderer`, and constructed the model
   with `init_renderer=False` (bypassing `wilor.models.load_wilor()`,
   which does not expose that option, and re-implementing its config
   post-processing directly in `pose/wilor/model_loader.py`).

Two further real issues surfaced in **Phase B** (full-mode execution),
both found and fixed:

4. **Missing `fp16` pass-through (adapter bug)**: `pose/wilor/frame_extraction.py:
   extract_frame_full()` constructed `ViTDetDataset(...)` without its
   `fp16` argument. In normal mode this is harmless (defaults to `False`),
   but with `--fast` (`model.half()`), the *image batch* stayed float32
   while the model's conv weights were float16, causing
   `RuntimeError('Input type (float) and bias type (c10::Half) should be
   the same')` inside `torch.compile`'s Dynamo tracing -- every single
   frame failed, and the run silently produced 0 reconstructed hands (no
   crash, since each per-frame failure was caught and recorded as an
   explicit `extraction_failed` frame per the raw-immutability contract,
   but the aggregate looked like "0% detection"). Fixed by passing
   `fp16=bool(pipeline.fast_mode)` through to `ViTDetDataset`, matching
   `demo.py`'s own `fp16=args.fast`.
5. **`torch.compile` requires Python.h**: after fixing (4), `--fast` mode
   still failed on every unique input shape with
   `fatal error: Python.h: No such file or directory` from Triton's C
   extension build step (the environment's Python 3.14 lacked development
   headers). Resolution: `sudo apt-get install -y python3.14-dev`
   (confirmed this repository's venv actually runs on the system
   `/usr/bin/python3.14`, so the matching apt package was the correct fix).
   After this, `--fast` mode ran successfully -- see "Fast mode
   comparison".

**Net installation verdict**: the complete official WiLoR + MANO stack
installs, loads, and runs end-to-end on Python 3.14.4 / torch 2.13.0+cu130
/ RTX 3050 Laptop (4 GB), including `WiLoR.load_from_checkpoint(...)`
successfully loading the 2.56 GB `wilor_final.ckpt` (PyTorch Lightning
auto-upgraded it from checkpoint format v1.8.1) and `self.mano =
MANO(**mano_cfg)` successfully constructing the MANO layer from the
human-supplied `MANO_RIGHT.pkl`.

## Full-mode smoke test

Before running the full 18-video pilot, a single-frame smoke test was run
(per this task's explicit instructions) and inspected directly:

```
pipeline loaded in 37.4s
n hand frames: 2
--- hand 0 ---
hand_present: True  label: right  mode: full  quality_flags: []
n landmarks_3d: 21
mano_params keys: ['hand_pose_rotmat', 'global_orient_rotmat', 'betas', ...]
  hand_pose_rotmat shape: (15, 3, 3)
  global_orient_rotmat shape: (1, 3, 3)
  betas len: 10
mano_references: camera_translation_xyz=[-0.0748, 0.2146, 37.67], focal_length=37500.0, ...
--- hand 1 (left) --- (symmetric, non-empty)
vertices dict: {(0,0): (778,3) float32, (0,1): (778,3) float32}
```

All expected fields (21 3D joints, 15 hand-joint rotation matrices, 1
global-orientation matrix, 10 betas, camera translation, 778 mesh
vertices per hand) were non-empty and of the documented shape. `mode` was
`"full"` and `detector_only_no_mano` was absent from `quality_flags`, as
required.

# Exact environment

| Component | Value |
|---|---|
| Python | 3.14.4 (`.venv`, running on system `/usr/bin/python3.14`) |
| OS | Linux 6.18.33.1-microsoft-standard-WSL2 |
| GPU | NVIDIA GeForce RTX 3050 Laptop, 4096 MiB VRAM, compute capability 8.6 |
| NVIDIA driver | 610.62 |
| CUDA (driver-visible) | 13.3 (UMD) |
| torch | 2.13.0+cu130 (`torch.version.cuda == "13.0"`) |
| torch.cuda.is_available() | `True` (verified: matmul on GPU ran successfully) |
| ultralytics | 8.1.34 (pinned by WiLoR) |
| smplx | 0.1.28 (pinned by WiLoR) |
| pytorch-lightning | 2.6.5 |
| chumpy | 0.71 (`git+mattloper/chumpy@580566e`) |
| pyrender | 0.1.45 |
| python3.14-dev | installed Phase B (for `torch.compile`/Triton) |
| WiLoR commit | `fcb911312a38fa8badd30d9656a167485d61b8f9` |
| detector.pt sha256 | `5ef3df44e42d2db52d4ffe91f83a22ce9925e2acc9abebf453f2c5d22e380033` |
| wilor_final.ckpt sha256 | `3e97aafc7dd08d883a4cc5a027df61fdb6fda6136dbd1319405413862ada6bb2` |
| MANO_RIGHT.pkl | **present** (Phase B), sha256 `45d60aa3b27ef9107a7afd4e00808f307fd91111e1cfa35afd5c4a62de264767`, local-only, not committed |
| MANO_LEFT.pkl | **present** (Phase B, reference/validation only), sha256 `c4022f7083f2ca7c78b2b3d595abbab52debd32b09d372b16923a801f0ea6a30`, local-only, not committed |

**Hardware-target discrepancy (important, verified, not assumed):** the
repository's stated benchmark target is an NVIDIA RTX 2060 SUPER (8 GB
VRAM). The actual machine this session ran on has an **RTX 3050 Laptop
GPU with 4 GB VRAM** (`nvidia-smi` output captured directly) -- less VRAM
than the target class, not more. Phase B's full-mode peak VRAM (~2.9 GB,
see "GPU / VRAM results") is a real, direct measurement on this smaller
card, and leaves meaningfully less headroom than the target 8 GB class
would; treat it as a conservative (harder) data point, not an optimistic
one. Also note the repository's `python3.10` setup instructions in
`README.md` do not match this environment's Python 3.14.4; both PyTorch
and every WiLoR dependency nonetheless installed and worked on 3.14.4.

# Dataset manifest

`datasets/manifests/karsl_milestone1_pilot.csv` (unchanged from Phase A;
not re-selected, re-ordered, or filtered after seeing Phase B results, per
this task's explicit instruction not to tune the dataset after seeing
results):

- 6 sign classes (KArSL sign IDs 0171-0176: build, break, walk, love,
  hate, grill) x 3 signers (01, 02, 03) x 1 repetition (lexicographically
  first valid `.mp4` per sign/signer) = **18 videos**, matching the
  original "~6 x 3 x 1" target exactly. Same 18 `sample_id`s used in
  Phase A and Phase B, and shared with the MediaPipe branch for
  comparison.
- Source: official KArSL RGB `test`-split archives, acquired via bounded
  HTTP range requests against per-signer 7z archives (never a full-dataset
  download), each row's `checksum_sha256` verified.
- All 18 clips are 1920x1080 @ 30 fps, 32-82 frames (~1.1-2.7 s) each.
- Local layout:
  `datasets/raw/karsl_milestone1_pilot/<signer>/test/<sign_id>/<original_filename>.mp4`
  (git-ignored, not committed).

(See Phase A's original acquisition note, preserved: this manifest and its
acquisition tooling, `scripts/download_karsl_pilot.py`, were produced by a
concurrent MediaPipe-branch session sharing this task's working
environment and adopted here unchanged for `sample_id`-level
comparability, per this task's instructions. No videos were bulk-
downloaded by this branch at any point.)

# Implementation

New code lives entirely under `pose/wilor/` (extractor-specific) and
`evaluation/` (extractor-agnostic), per `.github/instructions/pose.instructions.md`
and `.github/instructions/evaluation.instructions.md`. **The official WiLoR
PyTorch implementation is used directly and unmodified** (imported from a
local clone, see "Installation") for every neural component: the YOLO
detector, the ViT backbone, the transformer refinement head, and the MANO
layer via `smplx.MANOLayer`. Nothing neural was reimplemented; this
repository's code is limited to configuration/asset management, calling
the official model, converting its output into the common schema,
serialization, metrics, and visualization -- exactly the intended
boundary. No changes were made to `pose/common/schema.py`: the existing
`HandPoseFrame` fields (`mano_params`, `mano_references`,
`extractor_metadata`, `landmarks_2d`, `landmarks_3d`, `quality_flags`,
etc.) were sufficient to represent WiLoR's raw output without
modification, so none were added.

- `pose/wilor/config.py` -- asset-path resolution (`WilorAssetPaths`,
  overridable via `WILOR_ASSETS_DIR`), runtime knobs
  (`WilorRuntimeConfig`: device, fast_mode, detector confidence, rescale
  factor, batch_size=1 default), and constants sourced from the official
  config (image size, focal length, NUM_HAND_JOINTS).
- `pose/wilor/errors.py` -- `WilorAssetMissingError`,
  `ManoAssetMissingError` (with the exact remediation URL/steps),
  `WilorDependencyError`.
- `pose/wilor/model_loader.py` -- `check_assets()` (fails fast, before any
  heavy import, on the first missing file); `load_pipeline()` (full
  detector+MANO pipeline, re-implementing `wilor.models.load_wilor()`'s
  ~20 lines of config post-processing to use absolute MANO asset paths and
  `init_renderer=False`, then calling the **official**
  `WiLoR.load_from_checkpoint(...)` classmethod); `load_detector_only()`
  (YOLO detector alone, no MANO/WiLoR checkpoint needed);
  `_load_yolo_checkpoint()` (the `torch.load` weights_only workaround,
  scoped and documented).
- `pose/wilor/geometry.py` -- local, attributed reimplementations of two
  short (~10-25 line) camera-math helpers from the official
  `wilor/utils/renderer.py` and `demo.py` (`cam_crop_to_full`,
  `project_points_full_img`), vendored to avoid an unnecessary `pyrender`
  import for pure math. No model/network logic here.
- `pose/wilor/frame_extraction.py` -- `extract_frame_full()` (calls the
  official detector + `model(batch)` forward pass, then converts the
  *output* into `list[HandPoseFrame]` + a separate
  `{(frame_index, hand_index): vertices}` dict, since mesh vertices are
  large and must not be embedded in every lightweight frame object per
  Task 4/repo convention) and `extract_frame_detector_only()` (detector
  only, used in Phase A when MANO was unavailable; every frame it produces
  is tagged with the `detector_only_no_mano` quality flag so it can never
  be mistaken for a full reconstruction downstream). Phase B fix: passes
  `fp16=pipeline.fast_mode` through to the official `ViTDetDataset` (see
  Installation, issue 4).
- `pose/wilor/video_processing.py` -- per-video drivers
  (`process_video_full`, `process_video_detector_only`) that decode frames
  with `cv2.VideoCapture` (timestamps derived from source FPS), time
  inference separately from any visualization, record CUDA peak
  allocated/reserved memory, and store an explicit failure record
  (`quality_flags=["extraction_failed", ...]`, `hand_present=False`) for
  any frame whose extraction raises -- never interpolated or dropped.
- `pose/wilor/npz_io.py` -- immutable per-video NPZ serialization: one
  `wilor_raw.npz` per video, long-format (one row per hand-detection, plus
  an explicit `hand_present=False` row for empty frames), MANO
  params/references/metadata stored as per-row JSON strings, mesh vertices
  stored separately keyed by `frame:hand`. Never smooths, interpolates, or
  fixes failed frames. Phase A and Phase B runs are kept in **separate**
  output directories (`runs/wilor_karsl_pilot/` detector-only vs.
  `runs/wilor_karsl_pilot_full/` full-mode) so neither overwrites the
  other's raw evidence.
- `pose/wilor/visualize.py` -- Phase A: bbox + handedness-label +
  confidence + frame-index/timestamp overlay. **Phase B addition**: when
  full-mode output is present, draws the actual reconstructed 21-joint
  skeleton (real `landmarks_3d`, reprojected to 2D via
  `project_points_full_img` using the real `camera_translation_xyz` +
  `focal_length` -- not placeholder points) with the standard OpenPose
  hand-bone topology, plus a new `export_mesh_obj()` helper (via
  `trimesh`, no OpenGL/EGL needed) to export the actual triangulated MANO
  mesh as a `.obj` file for offline 3D inspection. Full pyrender
  triangle-mesh *rendering* is still deliberately not used (see "Visual
  validation" for the documented trade-off).
- `evaluation/metrics/hand_pose_metrics.py` -- extractor-agnostic (works on
  any `list[HandPoseFrame]`, MediaPipe or WiLoR): detection stats
  (missingness, longest streak, both/left/right-only counts), reference-
  position jitter (Phase B: prioritizes `mano_references['camera_translation_xyz']`
  -- the hand's real position in camera space -- over the root-relative
  `wrist_position`, since MANO's own wrist joint sits near the local
  origin almost by construction and is not a useful *global* motion
  signal; see "Temporal stability observations"), per-bone-length
  variation and coefficient-of-variation (standard 20-bone OpenPose-style
  hand skeleton, matching WiLoR's `mano_to_openpose` joint order),
  hand-count-change events, a handedness-swap-candidate heuristic, and
  (Phase B additions) `compute_global_orientation_stability()` (geodesic
  rotation-angle distance between consecutive `global_orient_rotmat`
  values, via the standard trace formula, not naive matrix-element
  subtraction, per this task's instruction) and
  `compute_betas_stability()` (frame-to-frame L2 change in MANO shape
  coefficients). No acceptance thresholds are hard-coded, per
  `.github/instructions/evaluation.instructions.md`.
- `evaluation/benchmarks/wilor_karsl_pilot.py` -- the pilot runner: reads
  the manifest, auto-selects full vs. detector-only mode based on
  `check_assets()`, saves raw NPZ, computes metrics, writes a JSON
  summary. Phase B additions: `--fast` flag (official FP16 + backbone
  layer-skip mode), `--sample-ids` filter (used for the 3-clip fast-mode
  comparison subset), `--out-dir` now resolved to an absolute path (fixed
  a path-join bug hit when using a non-default output directory).
- `pose/wilor/requirements.txt` -- exact, verified pin set (kept out of the
  root `pyproject.toml` so the base repository install stays lightweight).

# Files changed

```
pose/wilor/__init__.py
pose/wilor/config.py
pose/wilor/errors.py
pose/wilor/model_loader.py
pose/wilor/geometry.py
pose/wilor/frame_extraction.py
pose/wilor/video_processing.py
pose/wilor/npz_io.py
pose/wilor/visualize.py
pose/wilor/requirements.txt
evaluation/metrics/hand_pose_metrics.py
evaluation/benchmarks/wilor_karsl_pilot.py
tests/test_wilor_config_and_errors.py
tests/test_wilor_frame_extraction.py
tests/test_wilor_geometry.py
tests/test_wilor_metrics.py
tests/test_wilor_npz_io.py
reports/pose/wilor/TASK-002-wilor-karsl-pilot.md
```

(Phase B modified `pose/wilor/geometry.py`, `pose/wilor/visualize.py`,
`pose/wilor/frame_extraction.py`, `evaluation/metrics/hand_pose_metrics.py`,
`evaluation/benchmarks/wilor_karsl_pilot.py`, and `tests/test_wilor_metrics.py`
in place; no new top-level files were needed beyond Phase A's set.)

No files outside `pose/wilor/`, `evaluation/`, `tests/`, and this report
were modified. `pose/common/schema.py` was read but not changed. No
changes to `main`; no changes to the MediaPipe branch's own files.

# How to run

```bash
# 1. Install (see Installation section above / pose/wilor/requirements.txt)
pip install -r pose/wilor/requirements.txt
pip install --no-build-isolation "chumpy @ git+https://github.com/mattloper/chumpy"
git clone https://github.com/rolpotamias/WiLoR.git ~/.cache/wilor_assets/WiLoR
# fetch detector.pt / wilor_final.ckpt / model_config.yaml / mano_mean_params.npz
# as shown above; place a licensed MANO_RIGHT.pkl (and, optionally,
# MANO_LEFT.pkl for reference) under ~/.cache/wilor_assets/mano_data/.
# For --fast mode: sudo apt-get install -y python3.14-dev (or your distro's
# Python dev-headers package).
export WILOR_ASSETS_DIR=~/.cache/wilor_assets

# 2. Lightweight unit tests (no GPU/checkpoint required)
python -m unittest discover -s tests -p "test_wilor*.py"

# 3. Full pilot benchmark (auto full-mode since MANO is present; normal precision)
python evaluation/benchmarks/wilor_karsl_pilot.py --out-dir runs/wilor_karsl_pilot_full

# 4. Fast-mode comparison on a subset
python evaluation/benchmarks/wilor_karsl_pilot.py --fast \
  --sample-ids karsl_test_s02_sign0171_repfirst karsl_test_s02_sign0176_repfirst \
               karsl_test_s01_sign0174_repfirst \
  --out-dir runs/wilor_karsl_pilot_fast_subset
```

# Raw output schema

Reused `pose.common.schema.HandPoseFrame` as-is (one instance per detected
hand per frame; a frame with zero hands gets one `hand_present=False`
instance). **Phase B confirms every field below is populated with real
values on real data** (not just the hypothetical shapes documented in
Phase A):

| Field | WiLoR full mode (Phase B, confirmed) | WiLoR detector_only mode (Phase A) |
|---|---|---|
| `frame_index`, `timestamp_seconds` | from source video, preserved | same |
| `hand_present` | detector found a box | same |
| `handedness_label` | `"left"`/`"right"` from detector `cls` | same |
| `detection_confidence` | YOLO box confidence (e.g. 0.81-0.98 observed) | same |
| `landmarks_2d` | empty (see `mano_references` for 2D reprojection inputs) | 4 bbox corners |
| `landmarks_3d` | 21 MANO->OpenPose joints (root-relative, confirmed len 21) | empty |
| `wrist_position` | `landmarks_3d[0]` (near-zero, root-relative -- see "Temporal stability observations") | empty |
| `mano_params` | `hand_pose_rotmat` (15x3x3, confirmed), `global_orient_rotmat` (1x3x3, confirmed), `betas` (10,, confirmed), `representation="rotation_matrix"`, `num_hand_joints=15` | `None` |
| `mano_references` | `camera_translation_xyz`, `focal_length`, `box_center_xy`, `box_size`, `img_size_wh`, `vertices_ref`, `num_mesh_vertices=778` (confirmed) | `None` |
| `extractor_metadata` | `extractor="wilor"`, `extractor_version`, `checkpoint_id`, `mode="full"` (confirmed, never `detector_only_no_mano`), `fast_mode` | `mode="detector_only"`, plus `bbox_xyxy` |
| `quality_flags` | `[]` on success (confirmed empty for all 884/894 hand-frames); `["extraction_failed", ...]` only on error | always includes `"detector_only_no_mano"` |

Mesh vertices (778x3 per hand, confirmed float32, full mode only) are
**not** embedded in `HandPoseFrame`. They are returned separately by
`extract_frame_full()` and persisted in the companion `vertices` array of
`wilor_raw.npz`, addressed by the `mano_references["vertices_ref"]` key.

# MANO output interpretation

Investigated directly from `wilor/models/mano_wrapper.py` and
`pretrained_models/model_config.yaml`, and **empirically confirmed against
real Phase B output shapes and against the actual MANO pickle files**:

- **Pose representation**: `hand_pose` is **15 per-joint 3x3 rotation
  matrices** (`MANO.NUM_HAND_JOINTS=15`: 3 joints x 5 fingers; confirmed
  `len(hand_pose_rotmat)==15`, each `3x3`), each a local rotation relative
  to its parent in the MANO kinematic tree, in a fixed template ("T-pose")
  reference -- *not* raw axis-angle Euler flexion/extension/abduction
  angles, and *not* directly comparable to a physical joint-angle sensor
  reading. `global_orient` is one additional 3x3 rotation matrix for the
  wrist/root orientation relative to the camera. `betas` is a
  10-dimensional PCA shape coefficient vector controlling hand
  size/proportions, not pose (confirmed `len(betas)==10`).
- **Joints derived from the model**: `wilor/models/mano_wrapper.py: MANO`
  extends `smplx.MANOLayer` and remaps its 16 native MANO joints (wrist +
  15) plus 5 fingertip vertices (via `vertex_ids['mano']`) into a
  **21-point OpenPose-convention hand skeleton**
  (`mano_to_openpose = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11,
  12, 19, 7, 8, 9, 20]`), confirmed as `pred_keypoints_3d`'s actual shape
  (`21, 3`) on real output. `evaluation/metrics/hand_pose_metrics.py` uses
  the standard 20-bone OpenPose hand skeleton topology to compute bone
  lengths from this ordering.
- **Root-relative joints, not world position**: `landmarks_3d`/
  `wrist_position` (joint 0) come directly from `mano_output.joints`,
  which is the MANO model's own local/object-space output -- the wrist
  joint sits within a few millimeters of that local origin almost by
  construction (confirmed: measured wrist-joint frame-to-frame deltas were
  on the order of 1e-4 to 1e-3 model units, i.e. reconstruction-noise
  scale, not real hand motion). The hand's actual position and motion
  through the camera/world frame is carried separately in
  `mano_references['camera_translation_xyz']` (from `cam_crop_to_full`).
  This is documented explicitly here because it changed how "3D wrist
  jitter" needed to be computed -- see "Temporal stability observations".
- **Coordinate convention / handedness**: MANO's mesh/pose space is
  inherently a right hand. WiLoR does **not** use a separate left-hand
  MANO model; for detections classified "left" by the YOLO detector, it
  mirrors the x-axis of the *output* vertices/joints and one camera
  parameter post-hoc (`demo.py`). This adapter stores the rotation
  matrices exactly as the model outputs them (unmirrored) plus the
  `handedness_label`, leaving that mirroring step for whichever downstream
  code (future `hand_kinematics/`) needs a physically-consistent left-hand
  frame. See "Left/right MANO convention validation" for empirical
  confirmation of why this mirroring is mathematically correct.
- **Limitations for direct anatomical-angle interpretation**: (1) MANO's
  per-joint rotation matrices are relative to MANO's own template pose and
  kinematic-tree parent frames, which do not correspond one-to-one to
  standard clinical/anatomical flexion-extension / abduction-adduction
  axes; converting requires an explicit per-joint local-frame definition
  (a kinematics/retargeting layer). (2) `betas` shape parameters are
  learned PCA components, not physical bone lengths, and (Phase B,
  measured) do show small but nonzero frame-to-frame drift even for a
  supposedly-fixed hand shape -- see "MANO shape stability" below. (3) No
  forearm/global body reference is estimated by WiLoR -- `global_orient`
  is hand-root-to-camera only. **What this pilot's raw output *can* feed
  into a future `hand_kinematics/` module**: the 21 canonical joint
  positions (`landmarks_3d`), the 15+1 MANO rotation matrices with an
  explicit, documented convention, per-hand shape coefficients, and the
  camera-space translation for real-world placement -- exactly the inputs
  a kinematic retargeting layer needs, none of which this task builds.

## Left/right MANO convention validation

Per this task's instructions, both licensed MANO files were compared
directly (loaded via `pickle`, not assumed from documentation) to validate
WiLoR's mirroring strategy, **without changing WiLoR's inference code
path**:

```
right v_template[:3] = [[ 0.0457 -0.0128  0.0227] [ 0.0563 -0.0170  0.0235] [ 0.0499 -0.0200  0.0337]]
left  v_template[:3] = [[-0.0457 -0.0128  0.0227] [-0.0563 -0.0170  0.0235] [-0.0499 -0.0200  0.0337]]
max abs diff between x-mirrored RIGHT template and LEFT template: 0.0
```

`MANO_LEFT.pkl`'s vertex template is **bit-identical** to
`MANO_RIGHT.pkl`'s vertex template with the x-axis negated (face winding
order is also flipped, as required to keep normals consistent under a
mirror transform). This directly confirms, from the official MANO assets
themselves rather than inference from documentation, that:

- The official MANO "left" model *is*, by construction, the mirrored right
  model -- there is no independent left-hand anatomy being modeled.
- WiLoR's approach (always run the right-hand `MANO` layer, then mirror
  the *output* vertices/joints/one camera parameter for left-hand
  detections) is therefore **mathematically equivalent** to loading a
  native `MANO_LEFT.pkl` and running it directly -- using `MANO_LEFT.pkl`
  in official inference would not change results.
- Confirmed empirically on real Phase B output too: for simultaneous
  left/right detections in the same frame, the reconstructed wrist x-
  coordinates are equal in magnitude and opposite in sign (e.g. frame 0 of
  `karsl_test_s02_sign0171_repfirst`: right wrist x=+0.0959, left wrist
  x=-0.0957), consistent with the mirror convention.
- `MANO_LEFT.pkl` was, as expected, **never opened or referenced** during
  actual WiLoR inference in this pilot -- confirmed by `model_cfg.MANO`
  only ever pointing `smplx.MANOLayer` at a single (right, `is_rhand=True`
  default) model file, and by the earlier fast-fail check
  (`check_assets()`) only requiring `MANO_RIGHT.pkl`.

**Implication for `hand_kinematics/` (future work, not built here)**: any
future anatomical-angle extraction must apply the *same* x-mirroring
convention to left-hand `hand_pose_rotmat`/`global_orient_rotmat` values
before interpreting per-joint rotations anatomically -- the raw rotation
matrices stored by this adapter are in MANO's canonical right-hand
convention for *both* hands (unmirrored, by design, see above), and a
kinematics layer that naively applies the same joint-angle formulas to
left- and right-hand rotation matrices without mirroring would silently
get left-hand flexion/abduction signs wrong.

# Evaluation methodology

Same conceptual contract intended for the MediaPipe baseline
(`.github/instructions/evaluation.instructions.md`, `reports/README.md`
placeholders), implemented extractor-agnostically in
`evaluation/metrics/hand_pose_metrics.py` operating purely on
`list[HandPoseFrame]`:

- Detection stats: total frames, frames with no/left-only/right-only/both
  hands, missing-frame percentage, longest missing streak.
- Reference-position jitter: frame-to-frame displacement of
  `mano_references['camera_translation_xyz']` (full mode -- the hand's
  real position in camera space) or, if absent, the root-relative
  `wrist_position`, or, if absent, the 2D bbox centroid (detector_only
  mode). Units/scale differ across these three and are never mixed in one
  report number.
- Bone-length variation and coefficient of variation: frame-to-frame
  absolute change, and whole-video std/mean %, in each of the 20
  OpenPose-hand bones (full mode only; requires `landmarks_3d`).
- Global orientation stability: frame-to-frame **geodesic rotation-angle
  distance** (degrees) between consecutive `global_orient_rotmat` values,
  via the standard trace formula `arccos((trace(R1^T R2) - 1) / 2)` -- not
  naive per-element matrix subtraction, per this task's explicit
  instruction.
- MANO shape (betas) stability: frame-to-frame L2 change in the 10-D
  `betas` vector.
- Hand-count changes: frames where the detected hand count changes from
  the previous frame.
- Handedness-swap candidates: frames where swapping the reported
  left/right labels would have produced a lower total reference-position
  displacement than the reported assignment (a data-driven flag, not an
  identity-tracking correction).
- Runtime/FPS/VRAM: measured directly in `pose/wilor/video_processing.py`,
  strictly excluding visualization (Task 3), via `time.perf_counter()` and
  `torch.cuda.max_memory_allocated()/reserved()` with
  `reset_peak_memory_stats()` per video.

No acceptance thresholds were invented; all numbers below are reported as
measured distributions/counts.

Per the **raw-data immutability** rule (AGENTS.md), nothing here smooths,
interpolates, or "fixes" a failed or missing frame -- a frame with no
valid detection is stored explicitly as `hand_present=False`, and this
Phase-B run kept its raw NPZ output in a directory separate from Phase A's
detector-only evidence (`runs/wilor_karsl_pilot_full/` vs.
`runs/wilor_karsl_pilot/`), so neither overwrites the other.

# Results

## Phase A -- detector-only (preserved, unmodified from the original run)

The full detector+MANO pipeline could not be exercised on real data
because `MANO_RIGHT.pkl` was gated and unavailable. Every Phase-A result
below reflects the YOLO hand-localization stage only.

18/18 pilot videos processed, zero frame-level extraction errors.

| Metric | Value |
|---|---|
| Videos processed | 18 / 18 |
| Total frames | 894 |
| Frames with both hands | 884 (98.9%) |
| Frames with exactly one hand | 10 (1.1%; all left-only, one video) |
| Longest missing-hand streak | 0 frames |
| Handedness-swap candidates (bbox-centroid proxy) | 0 |
| Mean effective FPS (detector stage) | 40.9 |
| Peak CUDA allocated / reserved | 185.8 MB / 203.4 MB |

## Phase B -- full WiLoR + MANO reconstruction (this update)

**`FULL WILOR + MANO ACTUALLY EXECUTED: YES`**, on all 18 pilot videos, in
normal (FP32) precision. `mode="full"` on every successfully-reconstructed
hand-frame; `detector_only_no_mano` never appears.

### Per-video results (normal/FP32 mode)

| sample_id | frames | fps | both-hand % | left-only | hand_count_chg | swap flags |
|---|---|---|---|---|---|---|
| karsl_test_s01_sign0171_repfirst | 81 | 3.21 | 100.0% | 0 | 0 | 0 |
| karsl_test_s01_sign0172_repfirst | 52 | 4.68 | 100.0% | 0 | 0 | 0 |
| karsl_test_s01_sign0173_repfirst | 51 | 4.69 | 100.0% | 0 | 0 | 0 |
| karsl_test_s01_sign0174_repfirst | 34 | 5.06 | 70.6% | 10 | 2 | 0 |
| karsl_test_s01_sign0175_repfirst | 41 | 4.53 | 100.0% | 0 | 0 | 0 |
| karsl_test_s01_sign0176_repfirst | 38 | 4.61 | 100.0% | 0 | 0 | 0 |
| karsl_test_s02_sign0171_repfirst | 67 | 4.53 | 100.0% | 0 | 0 | 0 |
| karsl_test_s02_sign0172_repfirst | 36 | 4.35 | 100.0% | 0 | 0 | 0 |
| karsl_test_s02_sign0173_repfirst | 44 | 4.40 | 100.0% | 0 | 0 | 0 |
| karsl_test_s02_sign0174_repfirst | 36 | 4.35 | 100.0% | 0 | 0 | 0 |
| karsl_test_s02_sign0175_repfirst | 48 | 3.98 | 100.0% | 0 | 0 | 0 |
| karsl_test_s02_sign0176_repfirst | 57 | 3.38 | 100.0% | 0 | 2 | 0 |
| karsl_test_s03_sign0171_repfirst | 68 | 2.98 | 100.0% | 0 | 0 | 0 |
| karsl_test_s03_sign0172_repfirst | 50 | 4.43 | 100.0% | 0 | 0 | 0 |
| karsl_test_s03_sign0173_repfirst | 38 | 3.61 | 100.0% | 0 | 0 | 0 |
| karsl_test_s03_sign0174_repfirst | 38 | 4.33 | 100.0% | 0 | 0 | 0 |
| karsl_test_s03_sign0175_repfirst | 56 | 4.00 | 100.0% | 0 | 0 | 0 |
| karsl_test_s03_sign0176_repfirst | 59 | 4.01 | 100.0% | 0 | 0 | 0 |

(Full per-video detail, including all jitter/stability distributions, in
`runs/wilor_karsl_pilot_full/summary.json`, git-ignored/regeneratable.)

### Aggregate results (normal/FP32 mode)

| Metric | Value |
|---|---|
| Videos processed | 18 / 18 |
| Total frames | 894 |
| Frames with no hand | 0 (0.00%) |
| Frames with both hands (3D reconstructed) | 884 (98.9%) |
| Frames with exactly one hand | 10 (1.1%) |
| Longest missing-hand streak | 0 frames |
| Frame-level extraction errors | 0 |
| Hand-count change events | 4 (2 in the occlusion clip, 2 in a spurious-detection clip -- see below) |
| Handedness-swap candidates (real 3D camera-space position) | 0 |
| Mean effective FPS (full detector+MANO reconstruction) | 4.18 |
| Total inference time, 894 frames | 223.5 s |
| Peak CUDA allocated | 2819.4 MB |
| Peak CUDA reserved | 2904.6 MB |
| OOM events | 0 |

### 3D temporal stability (Phase B only -- real MANO output)

All values below are frame-to-frame, aggregated per handedness label
across all 18 videos (`n_videos=18` for every metric -- every video
contributed at least one measurable pair).

| Metric | right hand | left hand |
|---|---|---|
| Camera-space reference-position jitter -- mean of per-video means | 0.335 (model units) | 0.633 |
| Camera-space reference-position jitter -- mean of per-video p95 | 1.065 | 2.409 |
| Camera-space reference-position jitter -- max observed | 7.58 | 16.60 |
| Bone-length variation (abs delta) -- mean of per-video means | 3.26e-5 | 5.73e-5 |
| Bone-length variation (abs delta) -- max observed | 0.0031 | 0.0051 |
| Bone-length coefficient of variation -- mean of per-video means | 0.28% | 0.53% |
| Bone-length coefficient of variation -- max observed | 6.49% | 5.86% |
| Global orientation stability -- mean of per-video means | 2.92 deg | 5.87 deg |
| Global orientation stability -- mean of per-video p95 | 9.85 deg | 18.59 deg |
| Global orientation stability -- max observed | 61.95 deg | 102.16 deg |
| MANO betas (shape) drift, L2 -- mean of per-video means | 0.0103 | 0.0218 |
| MANO betas (shape) drift, L2 -- max observed | 0.0867 | 0.7049 |

Reading these: bone-length CV staying under ~1% on average is a genuine,
positive stability signal (a rigid mesh's bone lengths should not change
frame-to-frame; here they mostly don't, even though `betas` is
re-estimated independently every frame with no temporal smoothing). The
global-orientation and betas-drift *maxima* are the outlier tail (single
frame-pairs, likely coinciding with the occlusion/spurious-detection
events documented below), not the typical case -- the p95 values are a
better "usual worst case" read. Camera/model units for translation and
bone length are WiLoR's internal weak-perspective units (tied to
`EXTRA.FOCAL_LENGTH=5000` and MANO's own template scale), **not
calibrated to real-world millimeters** in this pilot -- see Limitations.

### Fast mode comparison

The official `--fast` mode (FP16 + `torch.compile` backbone + layer-skip)
was benchmarked against normal mode on 3 representative clips: an easy
two-hand clip, a clip with a spurious extra detection, and the occlusion
clip.

| sample_id | mode | FPS | peak alloc (MB) | both-hand % |
|---|---|---|---|---|
| karsl_test_s02_sign0171_repfirst (easy) | normal | 4.53 | 2813.5 | 100.0% |
| karsl_test_s02_sign0171_repfirst (easy) | fast | 9.76 | 1455.3 | 100.0% |
| karsl_test_s02_sign0176_repfirst (spurious detection) | normal | 3.38 | 2818.5 | 100.0% |
| karsl_test_s02_sign0176_repfirst (spurious detection) | fast | 1.98 | 1455.3 | 100.0% |
| karsl_test_s01_sign0174_repfirst (occlusion) | normal | 5.06 | 2813.5 | 70.6% |
| karsl_test_s01_sign0174_repfirst (occlusion) | fast | 1.12 | 1473.1 | 70.6% |

Both-hand detection percentage is **identical** between normal and fast
mode on all 3 clips -- consistent with the official ~0.05mm MPJPE
degradation claim (this pilot cannot verify sub-pixel accuracy without
ground truth, but detection/coverage is unaffected). Peak VRAM drops by
**~48%** (2813-2819 MB -> 1455-1473 MB) with FP16, a real and meaningful
benefit for this 4 GB card. FPS is **inconsistent**, not uniformly better:
one clip sped up 2.2x, the other two slowed down to 0.22x-0.59x of normal
mode. This is attributable to `torch.compile` recompiling its graph
whenever the input batch shape changes (these clips have 1-3 hands
detected per frame, so the batch dimension is not fixed across frames),
and short pilot clips (34-67 frames) do not amortize that recompilation
cost the way a longer, shape-stable run would. **Conclusion**: `--fast`
mode's VRAM benefit is real and reproducible; its throughput benefit is
not guaranteed on short, variable-hand-count clips like this pilot's, and
should not be assumed without testing on the actual target workload.

## Occlusion observations

- `karsl_test_s01_sign0174_repfirst` (sign "love"): 10/34 frames (17-26)
  are left-hand-only (70.6% both-hand). **Visually confirmed** (see
  "Visual validation"): the signer crosses their arms over their chest for
  part of this gesture, genuinely tucking the right hand under the left
  forearm out of camera view. WiLoR correctly reports only the visible
  left hand during this span -- it does **not** fabricate a phantom right
  hand -- and correctly resumes reporting both hands (frame 27 onward)
  once the arms uncross. This is the same clip that showed 70.6%
  both-hand coverage in Phase A's detector-only run (identical detection
  behavior, as expected, since detection itself does not depend on MANO).
- `karsl_test_s02_sign0176_repfirst` (sign "grill"): frame 39 produced
  **3 simultaneous detections** (2 labeled "left", 1 labeled "right") --
  a genuine detector false positive, not a bug in this adapter (frames
  38 and 40 both have exactly 2 detections). Visually, the spurious
  "right" detection's reconstructed mesh is a visibly degenerate, nearly
  flat sliver shape rather than a plausible hand (see "Visual
  validation"), consistent with MANO fitting a low-confidence/garbage crop
  rather than a genuine hand. This is exactly the kind of "mesh/joint
  anomaly on detector failure" this task asked to look for, and it did
  **not** get silently smoothed away -- it is present, as-is, in the raw
  NPZ output for anyone re-examining this clip.
- No other clip in the pilot showed any occlusion-related detection gap.

## Temporal stability observations

- Handedness-swap heuristic, now using the **real 3D camera-space
  position** (not a 2D proxy): **zero** candidates across all 18
  clips/894 frames, including through the occlusion and spurious-detection
  events above. WiLoR's per-frame left/right classification was spatially
  self-consistent throughout this pilot.
- Bone-length coefficient of variation stayed under ~1% on average per
  hand (see table above) -- a positive, direct signal that per-frame,
  independently-re-estimated MANO shape/pose does not produce wildly
  unstable mesh geometry frame-to-frame in this pilot, despite no temporal
  smoothing being applied anywhere in the raw extraction stage (by
  design -- Task 4/5 raw-immutability policy).
- Global-orientation frame-to-frame rotation changes had a typical (mean)
  magnitude of ~3-6 degrees but a heavier tail for the left hand (p95
  ~18.6 deg, max ~102 deg) than the right (p95 ~9.9 deg, max ~62 deg).
  Given the pilot's signer population and gesture set were not designed to
  isolate causes, this is reported as a **descriptive observation**, not
  attributed to a specific mechanism (e.g., faster genuine rotation vs.
  reconstruction noise) without further investigation.
- MANO betas (shape) drift was small on average (L2 ~0.01-0.02) but the
  left hand's *maximum* single-frame jump (0.705) is nearly an order of
  magnitude larger than the right hand's (0.087) -- worth flagging as an
  asymmetry, though it reflects a small number of outlier frame-pairs
  (plausibly coincident with the occlusion/spurious-detection clips) and
  should not be over-generalized from 18 clips.

# Visual validation

**Real 3D skeleton overlays** (not detector bboxes) were generated for 4
representative clips, using the actual reconstructed `landmarks_3d`
reprojected through the real camera model (`project_points_full_img` with
the frame's own `camera_translation_xyz`/`focal_length`), git-ignored
under `runs/wilor_karsl_pilot_full/visual/`:

1. **Easy two-hand** (`karsl_test_s02_sign0171_repfirst`): both hands
   correctly labeled ("right"/"left") and colored distinctly; both
   skeletons show plausible finger fanning with the thumb correctly
   separated from the four fingers; reprojection is close to, but not
   pixel-perfect on, the real hand position (a small consistent offset,
   likely camera-model/crop calibration noise rather than a pose error).
2. **Occlusion** (`karsl_test_s01_sign0174_repfirst`, frame 20, arms
   crossed): only the visible left hand is drawn, with a clean fanned-
   finger skeleton well-aligned to the real hand; no phantom right-hand
   skeleton is drawn during the occluded span.
3. **Spurious extra detection** (`karsl_test_s02_sign0176_repfirst`, frame
   39): two overlapping "left" skeletons plus one visibly degenerate,
   near-flat "right" skeleton corresponding to the false-positive
   detection described above -- a genuine, visible reconstruction failure
   mode, left exactly as reconstructed (not filtered out).
4. **High motion / extended-arm sign** (`karsl_test_s03_sign0173_repfirst`,
   sign "walk"): the resting hand (green, "right") is well-reconstructed
   and well-aligned; the extended, foreshortened hand (blue, "left") still
   recovers a plausible thumb-separated shape despite the difficult
   viewing angle.

A `pose/wilor/visualize.py: export_mesh_obj()` helper (via `trimesh`, no
OpenGL/EGL context needed) was also added to export the actual
triangulated MANO mesh per hand as a `.obj` file for anyone wanting to
inspect the raw 3D surface offline, rather than only the projected
skeleton. Full pyrender triangle-mesh *rendering* (baked-in overlay video)
was still deliberately not used: the projected-skeleton + `.obj`-export
combination gives real 3D reconstruction visibility without the
OpenGL/EGL dependency, and was judged sufficient for Milestone 1's
"reliable temporal 3D dual-hand data" goal. Generated videos/meshes are
not committed (git-ignored per repository policy); their local paths are
recorded above for reproducibility.

# GPU / VRAM results

Measured directly with `torch.cuda.reset_peak_memory_stats()` /
`max_memory_allocated()` / `max_memory_reserved()`, on the actual RTX 3050
Laptop (4096 MiB):

| Stage | Peak allocated | Peak reserved | Notes |
|---|---|---|---|
| Phase A: detector only | 185.8 MB | 203.4 MB | YOLOv8 hand detector, FP32 |
| Phase B: full detector + MANO, normal/FP32 | **2819.4 MB** | **2904.6 MB** | includes the 2.56 GB checkpoint's weights + activations for up to 3 simultaneous hand crops |
| Phase B: full detector + MANO, `--fast`/FP16 (3-clip subset) | 1455.3-1473.1 MB | (not separately tracked) | ~48% lower than normal mode |

- **Full-mode normal-precision reconstruction fits in this 4 GB card**,
  with roughly 1.2 GB of headroom (4096 - 2905 MB) at peak. No OOM events
  occurred across all 18 videos.
- No input resizing was applied or required; all frames processed at
  native 1920x1080, batch_size=1 (one video frame at a time).
- `--fast`/FP16 mode reduces peak VRAM by roughly half, a genuine
  additional safety margin if this were run on a smaller card or
  alongside other GPU workloads -- see "Fast mode comparison" for the
  accompanying throughput caveat.

# Failures

- **Phase A's primary blocker is resolved.** `MANO_RIGHT.pkl` was
  obtained by the human researcher through the proper licensing channel
  and full reconstruction now runs end-to-end.
- Two real issues were found and fixed while getting `--fast` mode
  working (see Installation, issues 4-5): a missing `fp16` flag pass-
  through in this adapter (own bug, fixed), and a missing `Python.h`
  system dependency for `torch.compile`/Triton (environment gap, fixed
  with `python3.14-dev`). Both are now resolved and `--fast` mode runs
  successfully.
- Zero frame-level extraction errors occurred in the full-mode 18-video
  pilot run (894/894 frames processed without exception, in both normal
  and fast mode on the tested subset).
- No OOM events in either normal or fast mode on this 4 GB card.
- One genuine model-level anomaly was observed and preserved, not
  suppressed: a spurious third hand detection
  (`karsl_test_s02_sign0176_repfirst`, frame 39) produced a degenerate,
  visibly implausible mesh -- correctly a detector/reconstruction
  limitation to be aware of, not an adapter bug (see "Occlusion
  observations" and "Visual validation").

# Limitations

- **Scale/units are not calibrated to real-world measurements.** WiLoR's
  camera translation and mesh scale are in its own internal
  weak-perspective units (tied to the fixed `EXTRA.FOCAL_LENGTH=5000`
  assumption and MANO's template scale); this pilot did not calibrate
  against a known real-world hand size or camera intrinsics, so absolute
  millimeter-scale claims (e.g., "X mm of jitter") cannot be made from
  this data alone -- only relative/percentage comparisons (e.g.,
  bone-length CV%) are currently meaningful across this pilot's own
  frames.
- Mesh **rendering** (baked triangle overlay) was not attempted (see
  "Visual validation" for the documented, deliberate trade-off); a
  projected-skeleton overlay and raw `.obj` mesh export were used instead.
- The pilot set (18 clips, 6 signs, single repetition per signer,
  1.1-2.7s each) is small by design (Milestone 1 pilot scale), so
  temporal-stability and occlusion conclusions describe this pilot's
  specific clips, not a general claim about WiLoR's robustness across
  KArSL or other data.
- `--fast` mode's throughput behavior on short, variable-hand-count clips
  was found to be inconsistent (see "Fast mode comparison") -- this is a
  real, measured limitation of applying `torch.compile` to this pilot's
  specific workload shape, not a general claim that `--fast` mode is
  unhelpful.
- This machine's GPU (RTX 3050, 4 GB) is *not* the repository's stated
  target class (RTX 2060 SUPER, 8 GB); full-mode VRAM here (~2.9 GB peak)
  was measured on the smaller card and is a conservative data point for
  the target hardware, not an optimistic one.
- Global-orientation and betas-drift outlier maxima are reported
  descriptively; this pilot did not investigate their root cause frame-by-
  frame beyond the two documented events (occlusion, spurious detection).

# Reproducibility

- WiLoR commit: `fcb911312a38fa8badd30d9656a167485d61b8f9`.
- Exact checkpoint/asset identities: sha256 hashes recorded above (Exact
  environment table), including the licensed MANO files (fingerprint
  only, per license -- files themselves are not distributed).
- Exact dependency versions: recorded above; also see
  `pose/wilor/requirements.txt`.
- Manifest: `datasets/manifests/karsl_milestone1_pilot.csv`, each row
  checksum-verified (`checksum_sha256` column), unchanged between Phase A
  and Phase B.
- Deterministic selection: sign IDs 0171-0176 (first 6 in the acquired
  range, ascending), signers 01-03, lexicographically-first valid `.mp4`
  member per sign/signer.
- No random seeds are involved (no sampling/augmentation is performed;
  detection and forward inference are deterministic given fixed weights
  and `torch.no_grad()`).
- Full command sequence to reproduce: "Installation" + "How to run"
  sections above. A licensed MANO account/file is required for Phase-B
  results; Phase-A (detector-only) results are reproducible without one.
- `runs/wilor_karsl_pilot_full/summary.json` and per-video
  `runs/wilor_karsl_pilot_full/raw/<sample_id>/wilor_raw.npz` are
  regenerable by re-running `evaluation/benchmarks/wilor_karsl_pilot.py`
  (git-ignored, not committed, per repository artifact policy). Phase A's
  original `runs/wilor_karsl_pilot/` output is preserved separately and
  also regenerable (by deleting/renaming the local MANO file to force
  `detector_only` mode again).

# Comparison readiness

`evaluation/metrics/hand_pose_metrics.py` operates purely on
`pose.common.schema.HandPoseFrame` sequences with no WiLoR-specific
assumptions, so the same functions apply unchanged to a MediaPipe raw
output once that branch's extraction is available, keyed by the same 18
`sample_id`s used throughout this pilot. What can be **directly and
fairly** compared once MediaPipe raw output exists:

- Both-hand / one-hand / no-hand detection rates, missing-frame
  percentage, longest missing streak, hand-count-change events -- these
  depend only on detection presence, not on coordinate system.
- Runtime/FPS/peak VRAM -- comparable with the caveat that WiLoR's
  full-mode numbers here include *reconstruction*, not just detection;
  compare WiLoR's *detector-only* Phase-A numbers against MediaPipe's
  detection-only numbers if an apples-to-apples detection-speed
  comparison is wanted, and WiLoR's Phase-B full numbers against
  MediaPipe's own landmark-regression numbers for a reconstruction-speed
  comparison.
- Handedness-swap-candidate counts and bone-length CV -- both extractor-
  agnostic heuristics on 21-point OpenPose-topology joints.

**What requires care, not a direct number-to-number comparison**: WiLoR's
3D output (`landmarks_3d`, `camera_translation_xyz`) is in WiLoR's own
weak-perspective camera-space units (see Limitations), while MediaPipe
Hand Landmarker's "world landmarks" use MediaPipe's own metric-scale
convention -- these are **not the same coordinate system or scale**, and
raw jitter/displacement numbers from the two extractors must not be
plotted or ranked against each other without an explicit, documented unit
normalization. No metric in this module was tuned to favor either
extractor, and this report does not produce a misleading "WiLoR beats
MediaPipe" ranking on any 3D-geometric metric.

# Recommendation

**NEEDS MORE EVALUATION**

Reasoning, from measured evidence only. Phase B substantially strengthens
the case for WiLoR: the full official detector + MANO reconstruction
pipeline now runs end-to-end on all 18 real pilot clips on this
constrained 4 GB GPU, with zero frame errors, zero OOM events, 98.9%
both-hand 3D reconstruction coverage, zero suspected handedness swaps
(using real 3D position), sub-1%-average bone-length instability, and
genuine, correctly-preserved (not silently repaired) failure evidence on
the one clip with a spurious extra detection. `--fast` mode's VRAM benefit
(~48% reduction) is real and reproducible. This is meaningfully more
positive evidence than Phase A's detector-only result could offer.

It stops short of **KEEP** for three concrete, evidence-based reasons: (1)
this pilot's 3D output is not yet calibrated to real-world units, so
Milestone 1's "reliable temporal 3D dual-hand data" claim cannot yet be
verified at the precision a downstream `hand_kinematics/` consumer would
need; (2) `--fast` mode's throughput behavior on this pilot's short,
variable-hand-count clips was inconsistent, and normal-mode throughput
(~4.2 FPS average) is far below real-time, which matters for any use case
beyond offline batch processing; (3) this pilot (18 short clips, one
repetition per signer) is not yet large enough to characterize WiLoR's
robustness across KArSL's full sign/motion variety, particularly around
the occlusion and spurious-detection failure modes actually observed
here. None of these three is a reason to reject WiLoR -- they are
concrete, scoped follow-ups.

# Next steps

1. Calibrate WiLoR's camera-space output against a known real-world
   reference (e.g., a measured hand span or a calibrated camera) so 3D
   jitter/displacement numbers can be reported in physical units and
   compared meaningfully against any future ground truth or the
   MediaPipe branch's metric-scale world landmarks.
2. Re-run the fast-mode comparison on longer, shape-stable clips (or with
   a fixed maximum-hands padding scheme) to determine whether
   `torch.compile`'s recompilation overhead is specific to this pilot's
   short/variable-hand-count clips or a more general limitation on this
   hardware.
3. Run the same benchmark once the MediaPipe branch's raw output is
   available, using the shared `evaluation/metrics/hand_pose_metrics.py`
   module directly on both, keyed by `sample_id`, respecting the
   coordinate-system caveats under "Comparison readiness".
4. Expand the pilot (more repetitions/signers/sign variety) before drawing
   any general robustness conclusion, particularly for the occlusion and
   spurious-detection failure modes surfaced here.
5. Do not proceed to `hand_kinematics/`, tracking, recognition, NLP, TTS,
   or virtual-sensor work until (1)-(4) are addressed.
