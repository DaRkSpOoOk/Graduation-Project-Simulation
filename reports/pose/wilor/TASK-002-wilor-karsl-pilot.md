# Task

Evaluate WiLoR (multi-hand, MANO-based 3D hand reconstruction) as the
high-fidelity experimental alternative to the MediaPipe baseline, on the
same KArSL Milestone-1 pilot definition, and give an evidence-based
GO/NO-GO recommendation for continuing this line of work.

# Branch

`opus/wilor-karsl-pilot`

# Scope

`pose/wilor/`, `evaluation/metrics/hand_pose_metrics.py`,
`evaluation/benchmarks/wilor_karsl_pilot.py`, `tests/test_wilor_*.py`, this
report. No recognition, NLP, TTS, virtual sensor, or Hall/IMU simulation
work was done, per the Milestone 1 scope.

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
  `demo.py`, `requirements.txt`, `pretrained_models/model_config.yaml`.

No blog posts or third-party tutorials were used for installation or
architecture claims; all commands below were run directly against the
official repository and its official checkpoint host.

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
  construction time. There is no code path that returns MANO-based 3D
  joints/vertices without a local MANO asset file.
- Explicit multi-hand support: the detector proposes any number of boxes
  per image; each is cropped, batched, and passed independently through the
  reconstruction model (`demo.py`), so N hands (0, 1, 2, ...) per frame are
  handled natively, not just a fixed left+right pair.
- Handedness convention: MANO is inherently a right-hand model. For
  left-hand detections, WiLoR mirrors the x-axis of vertices/joints and the
  camera's y-parameter post-hoc (`demo.py`:
  `verts[:,0] = (2*is_right-1)*verts[:,0]`) rather than using a separate
  left-hand MANO model. See "MANO output interpretation" below for why this
  matters for anatomical angle interpretation.
- Official config (`pretrained_models/model_config.yaml`, commit above):
  `MODEL.IMAGE_SIZE=256`, `EXTRA.FOCAL_LENGTH=5000`,
  `MANO.NUM_HAND_JOINTS=15`, detector confidence `0.3` and bbox
  `rescale_factor=2.0` (both from `demo.py` defaults).
- Fast/FP16 mode: official `--fast` flag does `model.half()` +
  `torch.compile(model.backbone)` + a backbone layer-skip flag, documented
  as up to 1.6x faster with ~0.05mm MPJPE degradation (repo README, "Update
  March 2026"). Not exercised in this pilot run because the MANO blocker
  (below) prevented any full-model forward pass; the flag is wired into
  `pose/wilor/config.py: WilorRuntimeConfig.fast_mode` and
  `pose/wilor/model_loader.py` for when MANO becomes available.
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
  or programmatic download path. **This was the actual blocker
  encountered in this task** (see "Failures"): no MANO account credentials
  were available to this agent/environment, so `MANO_RIGHT.pkl` could not
  be obtained, and the full MANO-based reconstruction stage could not be
  exercised on real data.
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

# 6. MANO_RIGHT.pkl: NOT obtainable here -- requires a manual account at
#    https://mano.is.tue.mpg.de and license acceptance by a human.
```

Three real, reproducible dependency issues were found and resolved (not
skipped):

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

**Net installation verdict**: everything upstream of MANO installs and
imports cleanly on Python 3.14.4 / torch 2.13.0+cu130, including
`WiLoR.load_from_checkpoint(...)` successfully loading the 2.56 GB
`wilor_final.ckpt` (PyTorch Lightning auto-upgraded it from checkpoint
format v1.8.1). Confirmed by isolating the failure precisely:

```
FAILURE TYPE: builtins AssertionError
FAILURE MSG: Path /home/hatim/.cache/wilor_assets/mano_data/MANO_RIGHT.pkl does not exist!
```

raised from inside `smplx`'s MANO loader, with every import and every
other asset already in place. MANO is the *only* remaining blocker.

# Exact environment

| Component | Value |
|---|---|
| Python | 3.14.4 (`.venv`) |
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
| WiLoR commit | `fcb911312a38fa8badd30d9656a167485d61b8f9` |
| detector.pt sha256 | `5ef3df44e42d2db52d4ffe91f83a22ce9925e2acc9abebf453f2c5d22e380033` |
| wilor_final.ckpt sha256 | `3e97aafc7dd08d883a4cc5a027df61fdb6fda6136dbd1319405413862ada6bb2` |
| MANO_RIGHT.pkl | **absent** -- gated, not obtainable in this environment |

**Hardware-target discrepancy (important, verified, not assumed):** the
repository's stated benchmark target is an NVIDIA RTX 2060 SUPER (8 GB
VRAM). The actual machine this session ran on has an **RTX 3050 Laptop
GPU with 4 GB VRAM** (`nvidia-smi` output captured directly) -- less VRAM
than the target class, not more. All VRAM numbers in this report are
therefore a *lower bound on available headroom* relative to the intended
target hardware, which is a conservative (harder) test, not an easier one.
Also note the repository's `python3.10` setup instructions in `README.md`
do not match this environment's Python 3.14.4; both PyTorch and every
WiLoR dependency nonetheless installed and worked on 3.14.4, so this was
not itself a blocker, but it is a real deviation worth flagging for anyone
trying to reproduce this on a "clean" 3.10 environment as the README
describes.

# Dataset manifest

`datasets/manifests/karsl_milestone1_pilot.csv` (not committed by this
branch alone -- see below). This branch ran concurrently, in the same
working environment, alongside a MediaPipe-branch session
(`luna/mediapipe-karsl-pilot`) building the shared pilot manifest and its
acquisition tooling (`scripts/download_karsl_pilot.py`). Per the task
instructions ("later both branches can be compared by sample_id"), this
WiLoR pilot uses that exact shared manifest and dataset rather than
constructing an independently-chosen one:

- 6 sign classes (KArSL sign IDs 0171-0176: build, break, walk, love,
  hate, grill) x 3 signers (01, 02, 03) x 1 repetition (lexicographically
  first valid `.mp4` per sign/signer) = **18 videos**, matching the
  original "~6 x 3 x 1" target exactly.
- Source: official KArSL RGB `test`-split archives, acquired via bounded
  HTTP range requests against per-signer 7z archives (never a full-dataset
  download), each row's `checksum_sha256` verified.
- All 18 clips are 1920x1080 @ 30 fps, 32-82 frames (~1.1-2.7 s) each,
  confirmed openable and decodable with OpenCV before use.
- Local layout:
  `datasets/raw/karsl_milestone1_pilot/<signer>/test/<sign_id>/<original_filename>.mp4`
  (git-ignored, not committed; see `datasets/README.md` on the MediaPipe
  branch for the full acquisition-tool description).

**Process note for reviewers**: this repository's task setup runs
sibling-branch agent sessions in a *shared* working directory (uncommitted
files are visible across branches, since git branches don't isolate an
uncommitted working tree). Early in this task, before the shared manifest
existed, this branch independently built its own interim 18-clip
single-signer manifest from partial archive bytes it found in `/tmp`
(6 signs x 1 signer x 3 repetitions, since only one signer's bytes were
byte-intact at that point). That interim manifest was discarded once the
sibling branch's session published the complete, checksum-verified,
3-signer manifest and downloaded videos, which is a strictly better match
to the original pilot-set target and was adopted instead. No videos were
bulk-downloaded by this branch at any point; sign-ID selection (0171-0176)
converged independently and identically between both sessions.

# Implementation

New code lives entirely under `pose/wilor/` (extractor-specific) and
`evaluation/` (extractor-agnostic), per `.github/instructions/pose.instructions.md`
and `.github/instructions/evaluation.instructions.md`. No changes were made to
`pose/common/schema.py`: the existing `HandPoseFrame` fields (`mano_params`,
`mano_references`, `extractor_metadata`, `landmarks_2d`, `landmarks_3d`,
`quality_flags`, etc.) were sufficient to represent WiLoR's raw output
without modification, so none were added.

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
  detector+MANO pipeline, re-implementing `wilor.models.load_wilor()` to
  use absolute MANO asset paths and `init_renderer=False`);
  `load_detector_only()` (YOLO detector alone, no MANO/WiLoR checkpoint
  needed); `_load_yolo_checkpoint()` (the `torch.load` weights_only
  workaround, scoped and documented).
- `pose/wilor/geometry.py` -- local, attributed reimplementation of
  `cam_crop_to_full` to avoid the unnecessary `pyrender` import for pure
  camera math.
- `pose/wilor/frame_extraction.py` -- `extract_frame_full()` (detector +
  MANO forward pass -> `list[HandPoseFrame]` + a separate
  `{(frame_index, hand_index): vertices}` dict, since mesh vertices are
  large and must not be embedded in every lightweight frame object per
  Task 4/repo convention) and `extract_frame_detector_only()` (detector
  only, used because MANO is unavailable here; every frame it produces is
  tagged with the `detector_only_no_mano` quality flag so it can never be
  mistaken for a full reconstruction downstream).
- `pose/wilor/video_processing.py` -- per-video drivers
  (`process_video_full`, `process_video_detector_only`) that decode frames
  with `cv2.VideoCapture` (timestamps derived from source FPS), time
  inference separately from any visualization, record CUDA peak
  allocated/reserved memory, and store an explicit failure record
  (`quality_flags=["extraction_failed", ...]`, `hand_present=False`) for
  any frame whose extraction raises -- never interpolated or dropped.
  Frame decoding is intentionally local/minimal here rather than added to
  `video_io/`, since that area had no established interface at the time of
  writing and a shared contract should be a deliberate, reviewed choice,
  not an incidental side effect of this task.
- `pose/wilor/npz_io.py` -- immutable per-video NPZ serialization: one
  `wilor_raw.npz` per video, long-format (one row per hand-detection, plus
  an explicit `hand_present=False` row for empty frames), MANO
  params/references/metadata stored as per-row JSON strings, mesh vertices
  stored separately keyed by `frame:hand`. Never smooths, interpolates, or
  fixes failed frames.
- `pose/wilor/visualize.py` -- bbox + handedness-label + confidence +
  frame-index/timestamp overlay, run strictly after/separately from timed
  inference. Does **not** use `pyrender` mesh rendering (see Installation
  point 3 and "Visual validation" below for why).
- `evaluation/metrics/hand_pose_metrics.py` -- extractor-agnostic (works on
  any `list[HandPoseFrame]`, MediaPipe or WiLoR): detection-stats
  (missingness, longest streak, both/left/right-only counts), wrist/proxy
  jitter distributions, per-bone-length variation (standard 20-bone
  OpenPose-style hand skeleton, matching WiLoR's `mano_to_openpose` joint
  order), hand-count-change events, and a handedness-swap-candidate
  heuristic (flags frames where swapping left/right labels would have
  produced a lower total displacement than the reported assignment). No
  acceptance thresholds are hard-coded, per
  `.github/instructions/evaluation.instructions.md`.
- `evaluation/benchmarks/wilor_karsl_pilot.py` -- the pilot runner: reads
  the manifest, auto-selects full vs. detector-only mode based on
  `check_assets()`, saves raw NPZ, computes metrics, writes
  `runs/wilor_karsl_pilot/summary.json` (git-ignored).
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

No files outside `pose/wilor/`, `evaluation/`, `tests/`, and this report
were modified. `pose/common/schema.py` was read but not changed (see
Implementation). No changes to `main`.

# How to run

```bash
# 1. Install (see Installation section above / pose/wilor/requirements.txt)
pip install -r pose/wilor/requirements.txt
pip install --no-build-isolation "chumpy @ git+https://github.com/mattloper/chumpy"
git clone https://github.com/rolpotamias/WiLoR.git ~/.cache/wilor_assets/WiLoR
# fetch detector.pt / wilor_final.ckpt / model_config.yaml / mano_mean_params.npz
# as shown above; place a licensed MANO_RIGHT.pkl under
# ~/.cache/wilor_assets/mano_data/ if/when available.
export WILOR_ASSETS_DIR=~/.cache/wilor_assets

# 2. Lightweight unit tests (no GPU/checkpoint required)
python -m unittest discover -s tests -p "test_wilor*.py"

# 3. Pilot benchmark (auto full-mode if MANO present, else detector_only)
python evaluation/benchmarks/wilor_karsl_pilot.py
# -> runs/wilor_karsl_pilot/summary.json, runs/wilor_karsl_pilot/raw/*/wilor_raw.npz
```

# Raw output schema

Reused `pose.common.schema.HandPoseFrame` as-is (one instance per detected
hand per frame; a frame with zero hands gets one `hand_present=False`
instance):

| Field | WiLoR full mode | WiLoR detector_only mode |
|---|---|---|
| `frame_index`, `timestamp_seconds` | from source video, preserved | same |
| `hand_present` | detector found a box | same |
| `handedness_label` | `"left"`/`"right"` from detector `cls` | same |
| `detection_confidence` | YOLO box confidence | same |
| `landmarks_2d` | *(unused; full 2D reprojection deferred)* | 4 bbox corners |
| `landmarks_3d` | 21 MANO->OpenPose joints (root-relative) | empty |
| `wrist_position` | `landmarks_3d[0]` | empty |
| `mano_params` | `hand_pose_rotmat` (15x3x3), `global_orient_rotmat` (1x3x3), `betas` (10,), `representation="rotation_matrix"`, `num_hand_joints=15` | `None` |
| `mano_references` | `camera_translation_xyz`, `focal_length`, `box_center_xy`, `box_size`, `img_size_wh`, `vertices_ref` (key into the NPZ vertex store), `num_mesh_vertices` | `None` |
| `extractor_metadata` | `extractor="wilor"`, `extractor_version` (repo commit), `checkpoint_id`, `mode="full"`, `fast_mode` | `mode="detector_only"`, plus `bbox_xyxy` |
| `quality_flags` | `["extraction_failed", ...]` only on error | always includes `"detector_only_no_mano"`; `"no_hand_detected"` when empty |

Mesh vertices (778x3 per hand, full mode only) are **not** embedded in
`HandPoseFrame` -- they are large and would bloat every lightweight frame
object. They are returned separately by `extract_frame_full()` as
`{(frame_index, hand_index): float32[778,3]}` and persisted in the
companion `vertices` array of `wilor_raw.npz`, addressed by the
`mano_references["vertices_ref"]` key.

# MANO output interpretation

Investigated directly from `wilor/models/mano_wrapper.py` and
`pretrained_models/model_config.yaml` (not assumed):

- **Pose representation**: `hand_pose` is **15 per-joint 3x3 rotation
  matrices** (`MANO.NUM_HAND_JOINTS=15`: 3 joints x 5 fingers), each a
  local rotation relative to its parent in the MANO kinematic tree, in a
  fixed template ("T-pose") reference -- *not* raw axis-angle Euler
  flexion/extension/abduction angles, and *not* directly comparable to a
  physical joint-angle sensor reading. `global_orient` is one additional
  3x3 rotation matrix for the wrist/root orientation relative to the
  camera. `betas` is a 10-dimensional PCA shape coefficient vector
  controlling hand size/proportions, not pose.
- **Joints derived from the model**: `wilor/models/mano_wrapper.py: MANO`
  extends `smplx.MANOLayer` and remaps its 16 native MANO joints (wrist +
  15) plus 5 fingertip vertices (via `vertex_ids['mano']`) into a
  **21-point OpenPose-convention hand skeleton**
  (`mano_to_openpose = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11,
  12, 19, 7, 8, 9, 20]`), which is what `pred_keypoints_3d` actually
  contains. `evaluation/metrics/hand_pose_metrics.py` uses the standard
  20-bone OpenPose hand skeleton topology to compute bone lengths from
  this ordering.
- **Coordinate convention / handedness**: MANO's mesh/pose space is
  inherently a right hand. WiLoR does **not** use a separate left-hand
  MANO model; for detections classified "left" by the YOLO detector, it
  mirrors the x-axis of the *output* vertices/joints and one camera
  parameter post-hoc (`demo.py`). This means the raw `hand_pose_rotmat`
  values for a left-hand detection are still expressed in MANO's
  right-hand joint-rotation convention and must be mirrored consistently
  with the vertices/joints before any anatomical interpretation -- this
  adapter stores the rotation matrices exactly as the model outputs them
  (unmirrored) plus the `handedness_label`, and leaves that mirroring step
  for whichever downstream code (future `hand_kinematics/`) actually needs
  a physically-consistent left-hand frame.
- **Limitations for direct anatomical-angle interpretation**: (1) MANO's
  per-joint rotation matrices are relative to MANO's own template pose and
  kinematic-tree parent frames, which do not correspond one-to-one to
  standard clinical/anatomical flexion-extension / abduction-adduction
  axes; converting requires an explicit per-joint local-frame definition
  (a kinematics/retargeting layer). (2) `betas` shape parameters are
  learned PCA components, not physical bone lengths, and are not
  identifiable frame-to-frame without a consistency prior (see
  "Temporal stability observations": we do observe some
  frame-to-frame bone-length drift as a proxy for this). (3) No forearm/
  global body reference is estimated by WiLoR -- `global_orient` is
  hand-root-to-camera only. **What this pilot's raw output *can* feed into
  a future `hand_kinematics/` module**: the 21 canonical joint positions
  (`landmarks_3d`), the 15+1 MANO rotation matrices with an explicit,
  documented convention, and per-hand shape coefficients -- exactly the
  inputs a kinematic retargeting layer needs, none of which this task
  builds.

# Evaluation methodology

Same conceptual contract intended for the MediaPipe baseline
(`.github/instructions/evaluation.instructions.md`, `reports/README.md`
placeholders), implemented extractor-agnostically in
`evaluation/metrics/hand_pose_metrics.py` operating purely on
`list[HandPoseFrame]`:

- Detection stats: total frames, frames with no/left-only/right-only/both
  hands, missing-frame percentage, longest missing streak.
- Wrist jitter: frame-to-frame displacement of the wrist (full mode) or
  bbox centroid as a proxy (detector_only mode -- units and scale differ
  between the two and are never mixed in one report number).
- Bone-length variation: frame-to-frame absolute change in each of the 20
  OpenPose-hand bones (full mode only; requires `landmarks_3d`).
- Hand-count changes: frames where the detected hand count changes from
  the previous frame.
- Handedness-swap candidates: frames where swapping the reported
  left/right labels would have produced a lower total wrist/proxy
  displacement than the reported assignment (a data-driven flag, not an
  identity-tracking correction -- see Task 5 policy below).
- Runtime/FPS/VRAM: measured directly in `pose/wilor/video_processing.py`,
  strictly excluding visualization (Task 3), via `time.perf_counter()` and
  `torch.cuda.max_memory_allocated()/reserved()` with
  `reset_peak_memory_stats()` per video.

No acceptance thresholds were invented; all numbers below are reported as
measured distributions/counts, per
`.github/instructions/evaluation.instructions.md`.

Per the **raw-data immutability** rule (AGENTS.md, Task 4/5), nothing here
smooths, interpolates, or "fixes" a failed or missing frame -- a frame
with no valid detection is stored explicitly as `hand_present=False`.

# Results

**Actual mode run: `detector_only`.** The full detector+MANO pipeline
could not be exercised on real data because `MANO_RIGHT.pkl` is gated and
unavailable in this environment (see Failures). Every result below reflects
the YOLO hand-localization stage only -- multi-hand detection, handedness
classification, bbox localization, and detector confidence -- not MANO
mesh/pose reconstruction. The full pipeline was verified to load correctly
up to (and only up to) the missing MANO file (see Installation).

## Per-video results

18/18 pilot videos processed successfully, zero frame-level extraction
errors.

| sample_id | frames | fps | both-hand % | left-only frames | frame errors | swap flags |
|---|---|---|---|---|---|---|
| karsl_test_s01_sign0171_repfirst | 81 | 22.4 | 100.0% | 0 | 0 | 0 |
| karsl_test_s01_sign0172_repfirst | 52 | 43.0 | 100.0% | 0 | 0 | 0 |
| karsl_test_s01_sign0173_repfirst | 51 | 41.7 | 100.0% | 0 | 0 | 0 |
| karsl_test_s01_sign0174_repfirst | 34 | 37.0 | 70.6% | 10 | 0 | 0 |
| karsl_test_s01_sign0175_repfirst | 41 | 41.4 | 100.0% | 0 | 0 | 0 |
| karsl_test_s01_sign0176_repfirst | 38 | 41.7 | 100.0% | 0 | 0 | 0 |
| karsl_test_s02_sign0171_repfirst | 67 | 38.6 | 100.0% | 0 | 0 | 0 |
| karsl_test_s02_sign0172_repfirst | 36 | 41.9 | 100.0% | 0 | 0 | 0 |
| karsl_test_s02_sign0173_repfirst | 44 | 36.9 | 100.0% | 0 | 0 | 0 |
| karsl_test_s02_sign0174_repfirst | 36 | 36.8 | 100.0% | 0 | 0 | 0 |
| karsl_test_s02_sign0175_repfirst | 48 | 33.8 | 100.0% | 0 | 0 | 0 |
| karsl_test_s02_sign0176_repfirst | 57 | 40.5 | 100.0% | 0 | 0 | 0 |
| karsl_test_s03_sign0171_repfirst | 68 | 38.0 | 100.0% | 0 | 0 | 0 |
| karsl_test_s03_sign0172_repfirst | 50 | 44.3 | 100.0% | 0 | 0 | 0 |
| karsl_test_s03_sign0173_repfirst | 38 | 48.6 | 100.0% | 0 | 0 | 0 |
| karsl_test_s03_sign0174_repfirst | 38 | 51.9 | 100.0% | 0 | 0 | 0 |
| karsl_test_s03_sign0175_repfirst | 56 | 52.5 | 100.0% | 0 | 0 | 0 |
| karsl_test_s03_sign0176_repfirst | 59 | 45.5 | 100.0% | 0 | 0 | 0 |

(Full per-video detail, including bbox-proxy jitter distributions, in
`runs/wilor_karsl_pilot/summary.json`, git-ignored/regeneratable.)

## Aggregate results

| Metric | Value |
|---|---|
| Videos processed | 18 / 18 |
| Total frames | 894 |
| Frames with no hand | 0 (0.00%) |
| Frames with both hands | 884 (98.9%) |
| Frames with exactly one hand | 10 (1.1%; all left-only, one video) |
| Longest missing-hand streak (any video) | 0 frames |
| Frame-level extraction errors | 0 |
| Hand-count change events | 4 (all in the one-hand-heavy clip) |
| Handedness-swap candidates | 0 |
| Mean effective FPS (detector stage) | 40.9 |
| Total inference time, 894 frames | 23.3 s |
| Peak CUDA allocated | 185.8 MB |
| Peak CUDA reserved | 203.4 MB |

## Occlusion observations

`karsl_test_s01_sign0174_repfirst` (sign "love") is the only clip with
sustained single-hand detection (10/34 frames left-only, 70.6% both-hand)
and the only clip with hand-count-change events (4). This is consistent
with a sign whose performance brings the hands close together / partially
occludes one hand from the camera for part of the gesture, rather than a
detector failure mode specific to this pilot's other clips (which were
uniformly 100% both-hand). No thumb-specific or finger-crossing
observations are available in this mode, since detector_only mode has no
per-joint output -- see Limitations.

## Temporal stability observations

Only the bbox-centroid proxy signal is available in detector_only mode
(no 3D joints/bone lengths/global orientation without MANO). Within that
constraint:

- Handedness-swap heuristic (see Evaluation methodology) found **zero**
  candidates across all 18 clips/894 frames: the detector's per-frame
  left/right classification stayed spatially consistent throughout every
  clip in this pilot, including the occluded clip above.
- Bbox-centroid jitter (pixel units, not directly comparable to
  MANO-space wrist jitter) ranged from a per-video max of ~0.7px to
  ~67px frame-to-frame, with per-video means in the low single digits to
  ~5px -- consistent with normal hand motion during signing at 30fps,
  1920x1080, rather than erratic detector jumps; no video showed an
  isolated single-frame outlier orders of magnitude above its own
  distribution.
- Bone-length variation, wrist-jump-in-3D, global-orientation jumps, and
  hand-scale instability (all requiring MANO joints/pose) are
  **unmeasured** in this pilot -- this is the single largest gap in the
  results and is entirely attributable to the MANO blocker, not to a
  negative finding about WiLoR's temporal stability.

# Visual validation

Bbox + handedness-label + detector-confidence + frame-index/timestamp
overlay was implemented and run successfully on real pilot frames (e.g.
`karsl_test_s01_sign0171_repfirst` -> a 2.7 MB overlay MP4, git-ignored
under `runs/wilor_karsl_pilot/visual/`). Full triangle-mesh overlay via
`pyrender.OffscreenRenderer` was deliberately **not** attempted: (a) it
requires an OpenGL/EGL or OSMesa context in addition to everything already
installed, which is an extra fragile dependency uninvolved in the raw
MANO parameter/joint output this milestone actually needs; (b) it would
have needed the (unavailable) MANO mesh in the first place. 2D bbox +
label overlay is a genuine, real visual check of detection/handedness
behavior and was judged sufficient for Milestone 1's "reliable temporal 3D
dual-hand data" goal; full mesh rendering is deferred to whenever MANO
access and the reconstruction stage are unblocked.

# GPU / VRAM results

Measured directly with `torch.cuda.reset_peak_memory_stats()` /
`max_memory_allocated()` / `max_memory_reserved()` around each video's
detector-only inference loop, on the actual RTX 3050 Laptop (4096 MiB):

- Peak allocated across all 18 clips: **185.8 MB**.
- Peak reserved: **203.4 MB**.
- This is the detector (YOLOv8-based) stage only, at native 1920x1080
  input, batch_size=1 (one frame at a time), FP32, no `--fast` mode.
- No OOM events; no resizing was required or applied.
- **Full-mode (MANO reconstruction) VRAM is unmeasured.** The
  `wilor_final.ckpt` weights alone are 2.56 GB on disk (FP32), and the ViT
  backbone + transformer refinement head would add activation memory on
  top per hand crop. Given the detector stage leaves ~3.9 GB of the 4 GB
  budget free, and WiLoR's official `--fast` FP16 mode plus the model's
  reported real-time operation on much larger images in the paper's own
  benchmarks, full-mode inference fitting in 4 GB is plausible but
  **not verified** -- this must be treated as an open question, not a
  result, until MANO access is available.

# Failures

- **Primary, blocking failure**: `MANO_RIGHT.pkl` cannot be obtained
  automatically. `smplx.MANOLayer.__init__` raises
  `AssertionError: Path .../MANO_RIGHT.pkl does not exist!` -- reproduced
  directly (see Installation) with every other component (checkpoints,
  config, all Python dependencies, CUDA) already correctly in place. This
  blocks: full MANO pose/shape parameters, 3D mesh vertices, the
  model-computed 21-joint 3D positions, MANO-space camera translation, and
  therefore all of bone-length-variation, 3D wrist jitter,
  global-orientation-jump, and hand-scale-instability metrics.
- Three environment/dependency issues, all resolved (documented under
  Installation): `xtcocotools` build failure (worked around by exclusion,
  confirmed unused on the inference path), PyTorch 2.6+ `weights_only`
  default breaking `ultralytics==8.1.34`'s checkpoint loader (worked
  around with a scoped, documented monkeypatch), and `pyrender`'s
  unnecessary coupling into a pure-math helper (worked around by a local,
  attributed reimplementation).
- Zero frame-level extraction errors occurred in the actual pilot run
  (18/18 videos, 894/894 frames processed without exception).
- No OOM events (measured peak VRAM: 203 MB reserved, far under the 4 GB
  budget) in the mode that was actually exercised.
- FP16/`--fast` mode and detector-vs-full timing comparisons were not
  meaningfully testable, since the full model never ran end-to-end.

# Limitations

- The headline result of this pilot is a **detector-only** partial run.
  No MANO pose parameters, mesh vertices, or 3D joints were produced on
  real data. This is the single biggest limitation and the reason for the
  recommendation below.
- Full-mode VRAM/runtime/FP16 feasibility is inferred, not measured.
- Occlusion/thumb-reconstruction/finger-crossing observations required by
  the evaluation contract are not assessable without MANO joints; only a
  coarse both/one/no-hand-visible signal was available.
- The pilot set (18 clips, 6 signs, single repetition, 1.1-2.7s each) is
  small by design (Milestone 1 pilot scale) and single-repetition, so
  temporal-stability conclusions (e.g., "zero handedness swaps") describe
  this pilot's specific clips, not a general claim about WiLoR's
  robustness across KArSL or other data.
- This machine's GPU (RTX 3050, 4 GB) is *not* the repository's stated
  target class (RTX 2060 SUPER, 8 GB); results here are conservative for
  the target hardware but were not cross-validated against it.

# Reproducibility

- WiLoR commit: `fcb911312a38fa8badd30d9656a167485d61b8f9`.
- Exact checkpoint identities: sha256 hashes recorded above (Exact
  environment table).
- Exact dependency versions: recorded above; also see
  `pose/wilor/requirements.txt`.
- Manifest: `datasets/manifests/karsl_milestone1_pilot.csv`, each row
  checksum-verified (`checksum_sha256` column).
- Deterministic selection: sign IDs 0171-0176 (first 6 in the acquired
  range, ascending), signers 01-03, lexicographically-first valid `.mp4`
  member per sign/signer.
- No random seeds are involved in this pilot (no sampling/augmentation is
  performed; detection and forward inference are deterministic given fixed
  weights and `torch.no_grad()`).
- Full command sequence to reproduce: "Installation" + "How to run"
  sections above, run end-to-end from a clean `.venv`.
- `runs/wilor_karsl_pilot/summary.json` and per-video
  `runs/wilor_karsl_pilot/raw/<sample_id>/wilor_raw.npz` are regenerable
  by re-running `evaluation/benchmarks/wilor_karsl_pilot.py` (git-ignored,
  not committed, per repository artifact policy).

# Comparison readiness

`evaluation/metrics/hand_pose_metrics.py` operates purely on
`pose.common.schema.HandPoseFrame` sequences with no WiLoR-specific
assumptions, so the same functions apply unchanged to a MediaPipe raw
output once that branch's extraction is available, keyed by the same
`sample_id`s as this pilot's manifest (both branches independently
converged on the same 18 `sample_id`s / same sign-ID range). Comparable
dimensions already implemented: both-hand detection rate, missing-frame
percentage, longest missing streak, hand-count-change events, handedness-
swap candidates, wrist/proxy jitter, runtime/FPS, peak VRAM. Bone-length
variation and 3D-jump metrics are implemented and will populate as soon as
either branch supplies 3D joints (WiLoR's are currently blocked by MANO;
MediaPipe's Hand Landmarker does not have this dependency). No metric in
this module was tuned to favor either extractor.

# Recommendation

**NEEDS MORE EVALUATION**

Reasoning, from measured evidence only: everything WiLoR-specific that
*could* be verified on this exact target-adjacent hardware and this exact
pilot data was verified and looks good -- clean installation once three
documented dependency issues are worked around, the 2.56 GB checkpoint
loads correctly, native multi-hand detection at 1920x1080 with zero frame
errors across all 18 real clips, 98.9% both-hand coverage, zero suspected
handedness swaps, and a detector-stage VRAM footprint (203 MB) that is a
small fraction of even this constrained 4 GB GPU's budget. But the
decisive Milestone-1 question -- can WiLoR reliably produce *3D dual-hand
data* (not just 2D detections) on this hardware -- was not answered,
because the MANO model file is gated behind a manual license-acceptance
step this agent cannot perform. That is a real, external, non-technical
blocker, not a WiLoR quality or feasibility finding, and it should not be
read as a negative signal about WiLoR itself.

# Next steps

1. **Smallest next attempt (unblocks everything else in this report)**: a
   human with a MANO account downloads `mano_v*_*.zip` from
   https://mano.is.tue.mpg.de, extracts `MANO_RIGHT.pkl`, and places it at
   `$WILOR_ASSETS_DIR/mano_data/MANO_RIGHT.pkl` (default
   `~/.cache/wilor_assets/mano_data/MANO_RIGHT.pkl`). No code changes are
   needed: `evaluation/benchmarks/wilor_karsl_pilot.py` already
   auto-detects the asset and switches from `detector_only` to `full`
   mode.
2. Re-run `evaluation/benchmarks/wilor_karsl_pilot.py` in full mode on the
   same 18-clip manifest; this will populate 3D joints, MANO
   pose/shape/global-orientation, mesh vertices, camera translation, and
   therefore bone-length-variation, 3D wrist jitter, global-orientation
   jumps, and hand-scale-instability -- all already implemented and
   waiting on this input.
3. Measure full-mode peak VRAM on the RTX 3050 (4 GB) directly; if it does
   not fit, retry with the official `--fast` (FP16 + backbone layer-skip)
   mode already wired into `WilorRuntimeConfig.fast_mode`, before
   concluding NO-GO on this hardware class.
4. Re-run the same benchmark once the MediaPipe branch's raw output is
   available, using the shared `evaluation/metrics/hand_pose_metrics.py`
   module directly on both, keyed by `sample_id`, for a real side-by-side
   comparison.
5. Do not proceed to `hand_kinematics/`, tracking, recognition, NLP, TTS,
   or virtual-sensor work until (1)-(4) produce actual 3D reconstruction
   evidence.
