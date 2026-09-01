# TASK-001 — MediaPipe KArSL pilot

## Task

Establish a reproducible RGB-video baseline for raw temporal 3D dual-hand
pose: obtain a tiny KArSL pilot, validate its video inputs, extract MediaPipe
Hand Landmarker output in VIDEO mode, create validation overlays, calculate
baseline metrics, and record the result for comparison with the later WiLoR
branch.

## Branch

`luna/mediapipe-karsl-pilot`

## Scope

This task covers only the KArSL RGB pilot, video inspection, the current
MediaPipe Hand Landmarker Python API, raw pose NPZ files, human-validation
overlays, baseline evaluation, and reproducibility documentation. It does not
implement sign recognition, NLP, TTS, Hall sensors, virtual IMU, placement
optimization, or large-scale dataset processing. The common pose schema was
not changed.

## Approach

The pilot uses a fixed slice of the current official KArSL-502 download
structure: six consecutive sign IDs, three signers, the official `test` split,
and one deterministic repetition per pair. The downloader fetches only bounded
HTTP ranges from the required solid 7z archives and extracts only the six
selected archive members from each archive. Video inspection runs before
inference. MediaPipe is run once per source frame; the same pass writes the
unchanged detector-order result to a compressed NPZ and draws an overlay.

## Dataset acquisition

The current official KArSL page describes KArSL-502 as an isolated Arabic Sign
Language dataset with RGB, depth, and skeleton modalities, three professional
signers, and 50 repetitions per sign. Its download page provides signer/split
Google Drive archives and the label workbook. The pilot obtained exactly 18
RGB MP4 files (6 signs × 3 signers × 1 selected repetition). No complete
archive or bulk dataset was downloaded, and the videos remain under ignored
`datasets/raw/` paths.

The six selected signs and labels are:

| Sign ID | Arabic label | English label |
| --- | --- | --- |
| 0171 | يبني | build |
| 0172 | يكسر | break |
| 0173 | يمشي | walk |
| 0174 | يحب | love |
| 0175 | يكره | hate |
| 0176 | يشوي | grill |

## Exact pilot manifest

The machine-readable experimental contract is
[`datasets/manifests/karsl_milestone1_pilot.csv`](../../../datasets/manifests/karsl_milestone1_pilot.csv).
It contains all 18 sample IDs, labels, signer/split fields, source archive
IDs, exact archive member paths, local relative paths, observed 30 FPS and
1920×1080 resolution, and SHA-256 checksums. The selection rule is:

> For each signer in `01`, `02`, `03` and each sign in `0171`–`0176`, use the
> official `test` split and choose the lexicographically first valid `.mp4`
> member under that sign directory.

This rule is deterministic and was fixed before MediaPipe results were
examined. The exact source members are:

| Signer | 0171 | 0172 | 0173 | 0174 | 0175 | 0176 |
| --- | --- | --- | --- | --- | --- | --- |
| 01 | `03_01_0171_(17_04_17_18_51_29)_c.mp4` | `03_01_0172_(13_09_17_13_55_09)_c.mp4` | `03_01_0173_(13_09_17_13_58_08)_c.mp4` | `03_01_0174_(13_09_17_14_01_16)_c.mp4` | `03_01_0175_(13_09_17_14_03_33)_c.mp4` | `03_01_0176_(13_09_17_14_05_09)_c.mp4` |
| 02 | `03_02_0171_(01_05_17_15_51_46)_c.mp4` | `03_02_0172_(01_05_17_15_56_42)_c.mp4` | `03_02_0173_(01_05_17_15_59_20)_c.mp4` | `03_02_0174_(01_05_17_16_01_08)_c.mp4` | `03_02_0175_(01_05_17_16_05_47)_c.mp4` | `03_02_0176_(01_05_17_16_07_58)_c.mp4` |
| 03 | `03_03_0171_(18_04_17_17_30_34)_c.mp4` | `03_03_0172_(18_04_17_17_33_45)_c.mp4` | `03_03_0173_(18_04_17_17_36_28)_c.mp4` | `03_03_0174_(18_04_17_17_39_05)_c.mp4` | `03_03_0175_(18_04_17_17_43_19)_c.mp4` | `03_03_0176_(18_04_17_17_46_03)_c.mp4` |

The three source archives and their manifest-recorded total sizes are:

| Signer | Archive ID | Archive bytes | Bounded prefix read |
| --- | --- | ---: | ---: |
| 01 | `1Iz3j0wlhuo1_j9WeCwGsN01L64TI_JYo` | 186,368,001 | 73,400,320 |
| 02 | `14GAOAMOdIhVzQHc0slooy5ZiRj0ZxrFH` | 178,861,322 | 73,400,320 |
| 03 | `1ujZpg8aOye7UrVhPEkNpLBsCfWzW0ps9` | 235,841,474 | 73,400,320 |

The selected archive members were confirmed to be available from those bounded
ranges. The downloaded local files are not source-controlled.

## Sources / evidence

- [Official KArSL dataset page](https://hamzah-luqman.github.io/KArSL/) —
  dataset description, download links, labels, and original citation.
- [Official KArSL repository](https://github.com/Hamzah-Luqman/KArSL) —
  project source associated with the dataset page.
- [KArSL original research record](https://dl.acm.org/doi/10.1145/3423420) —
  publication record for the dataset paper.
- [Current MediaPipe Hand Landmarker Python guide](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python)
  — current VIDEO-mode construction, monotonic timestamps, options, and
  result fields.
- [MediaPipe Hand Landmarker Python source](https://github.com/google/mediapipe/blob/master/mediapipe/tasks/python/vision/hand_landmarker.py)
  — confirms `detect_for_video`, handedness, image landmarks, and world
  landmarks in the current Python result object.

SSHI access at [sshi.sa](https://sshi.sa/) is approved for this project as
stated in the task brief. SSHI scraping/data collection is deferred to a later
dedicated task; no SSHI scraper or data is part of this experiment.

## MediaPipe API selected

The implementation uses the current `mediapipe==1.0.1` Tasks API:

```python
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
result = landmarker.detect_for_video(mp_image, timestamp_ms)
```

Each result preserves up to two detector-order hands, 21 image landmarks,
21 world landmarks, the detector-provided handedness label and score, and
per-frame hand presence (`len(result.hand_landmarks) > 0`). The current Python
result exposes handedness scores but does not expose separate per-frame
detection, presence, or tracking confidence values; those fields are recorded
as unavailable rather than inferred.

## Implementation

- `video_io.reader` performs pre-inference decoding inspection and records
  decoded frame count, reported FPS, dimensions, duration, decoder status, and
  timestamp provenance. It prefers ffprobe frame timestamps when available,
  then decoder position timestamps, and only then an FPS fallback. MediaPipe
  timestamps are monotonically increasing integer milliseconds; source seconds
  remain separately recorded.
- `pose.mediapipe.extractor` uses VIDEO mode, RGB frames, `num_hands=2`, and
  the three configured confidence thresholds. It writes raw detector-order
  arrays to NPZ without smoothing, interpolation, identity correction, or
  destructive replacement.
- `visualization.hand_overlay` writes one ignored MP4 per successful clip with
  landmarks/connections, `LEFT`/`RIGHT`, handedness score, frame index, and
  timestamp.
- `evaluation.metrics.mediapipe_baseline` calculates detection/missingness,
  missing streaks, handedness changes, duplicate labels, a heuristic identity
  instability indicator, wrist jumps, temporal jitter, bone-length variation,
  coordinate completeness, and runtime/FPS. Diagnostic jump values (0.10 m
  world and 0.20 normalized image distance) are descriptive observations, not
  pass/fail thresholds.
- `scripts.run_mediapipe_pilot` runs inspection, checksum verification,
  extraction, overlays, evaluation, hardware capture, and run metadata from a
  single manifest.

## Files changed

Task-owned source/documentation files are:

- `.gitignore`, `pyproject.toml`
- `configs/pose/mediapipe_karsl_pilot.json`
- `datasets/README.md`, `datasets/manifests/karsl_milestone1_pilot.csv`
- `research/datasets/karsl_milestone1.md`
- `video_io/__init__.py`, `video_io/reader.py`, `video_io/README.md`
- `pose/mediapipe/__init__.py`, `pose/mediapipe/extractor.py`,
  `pose/mediapipe/hardware.py`, `pose/mediapipe/README.md`, `pose/README.md`
- `visualization/__init__.py`, `visualization/hand_overlay.py`
- `evaluation/__init__.py`, `evaluation/metrics/__init__.py`,
  `evaluation/metrics/mediapipe_baseline.py`, `evaluation/README.md`,
  `evaluation/benchmarks/__init__.py`
- `scripts/download_karsl_pilot.py`, `scripts/download_mediapipe_model.py`,
  `scripts/run_mediapipe_pilot.py`
- `tests/test_karsl_manifest.py`, `tests/test_mediapipe_metrics.py`,
  `tests/test_video_io.py`
- this report

`pose/common/schema.py` was inspected and left unchanged. Generated model,
video, NPZ, and overlay artifacts are ignored and are not part of the commit.

## How to run

From the repository root with the activated `.venv`:

```bash
python -m pip install -e .
python scripts/download_mediapipe_model.py
python scripts/download_karsl_pilot.py
python scripts/run_mediapipe_pilot.py \
  --repository-root . \
  --manifest datasets/manifests/karsl_milestone1_pilot.csv \
  --config configs/pose/mediapipe_karsl_pilot.json \
  --model datasets/raw/models/hand_landmarker.task \
  --output-dir runs/mediapipe_karsl_pilot
python -m unittest discover -s tests -p 'test_*.py'
```

The downloader requires `7z`/`7zz` and uses the official Google Drive archive
IDs in the manifest. It performs bounded range reads and extracts only the
manifest members. The extraction command produces ignored outputs in
`runs/mediapipe_karsl_pilot/`: `video_inspection.*`, `raw_pose/*.npz`,
`overlays/*.mp4`, evaluation JSON/CSV files, `hardware.json`, and
`run_metadata.json`.

## Evaluation

The evaluation is descriptive baseline collection only; no strict pass/fail
thresholds are applied. The detailed methodology follows.

## Evaluation methodology

The evaluation denominator for frame detection rates is decoded frames per
video. “Both hands” means both detector-provided handedness labels are present
in that frame; no identity correction is applied. Missing streaks are measured
per handedness-labelled channel. Handedness changes are reported both for the
set of labels in a frame and for detector-order slots. Potential identity
instability is a heuristic count of duplicate labels, detector-order reversals,
and left/right wrist-x crossings; it is not an identity-ground-truth score.

Wrist jumps are calculated in world metres and normalized image coordinates.
Temporal jitter is the finite-difference second difference of world wrist
trajectories, with source timestamps used for acceleration. Bone variation is
the coefficient of variation over available world-landmark connections. All
returned hands in this pilot had finite world coordinates and finite thumb
landmarks; this records numeric landmark availability, not a claim that a
missing thumb was reconstructed. Finger crossing and hand-hand occlusion cause
are not honestly identifiable from this output, so they are explicitly marked
unmeasured. The current API's unavailable detection/presence/tracking
confidence fields are also explicit `null` values in the per-video results.

## Results

### Video input validation

All 18 captures decoded successfully. Reported and decoded frame counts
matched for every file; container values were 30.0 FPS and 1920×1080. Decoded
clip lengths ranged from 34 to 81 frames (1.133 to 2.700 seconds). FFprobe was
not available in this environment, so the reader used OpenCV container
position timestamps for every clip, with zero monotonicity adjustments.

## Per-video results

The complete machine-readable results are in the ignored
`runs/mediapipe_karsl_pilot/evaluation_per_video.json` and `.csv` files. The
table below reports the final run. `L/R` in the streak column is the longest
missing streak in frames for the labelled left/right channel. Wrist values are
`p95 / max` world metres; jitter is `mean / p95` world metres per frame; bone
variation is `mean / max` coefficient of variation.

| Sample | Frames | No hands | Left | Right | Both | Missing % | Streak L/R | Label changes | Identity events | Wrist p95/max | Jitter mean/p95 | Bone CV mean/max | Runtime s | FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s01-0171 | 81 | 0 | 81 | 81 | 81 | 0.00 | 0/0 | 0 | 0 | .0172/.0445 | .0077/.0243 | .1758/.5836 | 3.853 | 21.02 |
| s01-0172 | 52 | 0 | 52 | 52 | 52 | 0.00 | 0/0 | 0 | 0 | .0179/.0509 | .0093/.0284 | .2275/.5873 | 2.095 | 24.83 |
| s01-0173 | 51 | 0 | 51 | 51 | 51 | 0.00 | 0/0 | 0 | 0 | .0135/.0254 | .0077/.0237 | .1201/.4464 | 2.046 | 24.92 |
| s01-0174 | 34 | 0 | 12 | 33 | 11 | 0.00 | 0/12 | 3 | 0 | .0229/.1010 | .0123/.0249 | .1899/.3535 | 1.547 | 21.98 |
| s01-0175 | 41 | 0 | 38 | 41 | 38 | 0.00 | 0/3 | 2 | 0 | .0244/.0340 | .0079/.0255 | .1258/.4869 | 1.748 | 23.45 |
| s01-0176 | 38 | 0 | 38 | 37 | 37 | 0.00 | 0/1 | 1 | 1 | .0206/.0329 | .0099/.0212 | .1634/.7123 | 1.623 | 23.41 |
| s02-0171 | 67 | 0 | 64 | 66 | 63 | 0.00 | 0/2 | 6 | 2 | .0228/.0706 | .0106/.0243 | .1813/.5256 | 2.693 | 24.88 |
| s02-0172 | 36 | 6 | 18 | 30 | 18 | 16.67 | 18/6 | 3 | 0 | .0134/.0245 | .0056/.0147 | .1070/.2156 | 1.748 | 20.59 |
| s02-0173 | 44 | 0 | 44 | 44 | 44 | 0.00 | 0/0 | 0 | 0 | .0110/.0149 | .0062/.0184 | .1128/.3432 | 1.886 | 23.33 |
| s02-0174 | 36 | 0 | 36 | 36 | 36 | 0.00 | 0/0 | 0 | 0 | .0146/.0215 | .0071/.0207 | .1232/.3183 | 1.666 | 21.61 |
| s02-0175 | 48 | 0 | 47 | 21 | 20 | 0.00 | 0/12 | 5 | 2 | .0291/.0579 | .0119/.0291 | .1565/.4493 | 2.118 | 22.66 |
| s02-0176 | 57 | 0 | 39 | 57 | 39 | 0.00 | 0/12 | 6 | 3 | .0238/.0816 | .0078/.0208 | .1056/.2397 | 2.810 | 20.28 |
| s03-0171 | 68 | 0 | 65 | 64 | 61 | 0.00 | 0/2 | 8 | 2 | .0133/.1465 | .0077/.0186 | .1446/.3646 | 3.162 | 21.50 |
| s03-0172 | 50 | 0 | 50 | 50 | 50 | 0.00 | 0/0 | 0 | 0 | .0147/.0287 | .0078/.0185 | .1780/.3957 | 2.639 | 18.95 |
| s03-0173 | 38 | 0 | 23 | 38 | 23 | 0.00 | 0/14 | 3 | 15 | .0156/.0225 | .0113/.0508 | .1653/.5408 | 1.936 | 19.63 |
| s03-0174 | 38 | 0 | 38 | 21 | 21 | 0.00 | 0/17 | 2 | 5 | .0187/.1312 | .0077/.0185 | .1132/.2765 | 1.913 | 19.86 |
| s03-0175 | 56 | 0 | 52 | 56 | 52 | 0.00 | 0/2 | 6 | 3 | .0292/.0829 | .0111/.0417 | .1460/.5393 | 2.617 | 21.40 |
| s03-0176 | 59 | 0 | 55 | 58 | 54 | 0.00 | 0/3 | 6 | 1 | .0304/.0615 | .0103/.0293 | .1213/.3503 | 2.785 | 21.18 |

## Aggregate results

Across 18 videos and 894 decoded frames:

| Metric | Result |
| --- | ---: |
| Videos observed / failed | 18 / 0 |
| Total frames | 894 |
| Frames with no hands | 6 (0.671%) |
| Frames with at least one hand | 888 (99.329%) |
| Frames with left hand | 803 (89.821%) |
| Frames with right hand | 836 (93.512%) |
| Frames with both hands | 751 (84.004%) |
| Longest missing streak, left / right | 18 / 27 frames |
| Handedness label-set changes | 51 |
| Detector-order handedness changes | 25 |
| Potential identity-instability events | 34 |
| Labelled wrist jumps > 0.10 m | 2 / 1,581 observations |
| Raw detector-order wrist jumps > 0.10 m | 4 / 1,621 observations |
| Maximum labelled/raw world jump | 0.1407 / 0.1465 m |
| Mean per-video bone CV | 0.1476 |
| Maximum per-video bone CV | 0.4294 |
| Mean per-video jitter / p95 | 0.00889 / 0.02519 m/frame |
| Mean handedness score | 0.9562 |

## Visual observations

All 18 overlays were created. Inspection of representative frames showed that
the overlays visibly provide 21-point hand skeletons, handedness labels/scores,
frame index, and timestamp over the 1920×1080 Kinect RGB frames. A good
two-hand example was `s01-0171`. In `s02-0172`, several frames show motion
blur or hands outside a recoverable configuration; this clip contains all six
no-hand frames and has only 18/36 both-hand frames. In `s03-0173`, a reviewed
frame visibly labels both returned hands `RIGHT`, matching the duplicate-label
and identity-instability signal in the metrics.

## Failures / limitations

- Decoder inspection and extraction succeeded for all 18 videos; there were no
  processing failures or overlay-write failures.
- The pilot nevertheless did not provide reliable dual-hand coverage in every
  clip: aggregate both-hand detection was 84.004%, with the weakest observed
  clips `s02-0172` (50.0%), `s02-0175` (41.7%), `s03-0173` (60.5%), and
  `s03-0174` (55.3%).
- `s02-0172` had six no-hand frames, an 18-frame left missing streak, and only
  18 left-hand frames. `s02-0175` had only 21/48 right-hand frames and a
  27-frame right missing streak. `s03-0173` produced 15 heuristic identity
  events and duplicate `RIGHT` labels in a reviewed overlay frame.
- MediaPipe's current Python result object does not expose separate detector,
  presence, or tracker confidence values. Only handedness scores are reported.
- No identity ground truth, anatomical finger-crossing classifier, or
  hand-hand occlusion-cause label is available, so those observations cannot be
  converted into accuracy claims. No pass/fail thresholds were invented.
- The run did not have ffprobe available; all 18 files used OpenCV decoder
  position timestamps, with zero timestamp adjustments. A future VFR/container
  validation run should provide ffprobe or another independent PTS source.
- MediaPipe emitted its normal CPU XNNPACK/feedback-manager notices and a
  landmark projection `NORM_RECT` warning. These did not stop processing but
  should be revisited if camera calibration becomes important.

## Performance

The run used the CPU delegate. Hardware observation recorded:

- CPU: AMD Ryzen 7 6800HS with Radeon Graphics
- GPU present: NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MB, driver 610.62
- MediaPipe execution device: CPU (the GPU was not assumed to be used)
- Peak GPU VRAM: not measured; MediaPipe was configured for CPU execution.
- RAM: 7,933,165,568 bytes reported by the environment
- Python: 3.14.4; platform: WSL2 Linux

Measured over all 894 frames: 40.887 s total extraction runtime, 18.290 s
MediaPipe inference time, 21.865 effective end-to-end processing FPS, and
48.880 inference FPS. Mean per-video runtime was 2.271 s and median effective
per-video FPS was 21.555. The full command wall time, including inspection and
file output, was 47.070 s.

## Comparison

No WiLoR implementation or result is included in this branch. The unchanged
18-row manifest, source archive member paths, local paths, and checksums are
the comparison contract for the separate WiLoR branch. The pilot set was not
altered after observing MediaPipe results.

## Recommendation

**NEEDS MORE EVALUATION**

The baseline is sufficiently reproducible to serve as the first comparison
implementation, but the measured 84.004% both-hand rate, 34 heuristic identity
events, long per-hand missing streaks, and lack of ground-truth identity labels
do not establish reliable temporal 3D dual-hand data. Keep the raw outputs and
manifest for comparison; do not treat this pilot as a final model selection.

## Reproducibility

- Date: 2026-09-01, Asia/Riyadh.
- Branch: `luna/mediapipe-karsl-pilot`.
- Dependency pins: `mediapipe==1.0.1`, `numpy==2.5.2`,
  `opencv-contrib-python==5.0.0.93`; `pip check` passed.
- Model: official [Hand Landmarker task bundle](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task),
  SHA-256 `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`.
- Manifest SHA-256 at the final run: `4a8e0d24f587d9c54ad87ebd896c6df0290b3ebc80885774100d71b29dbc3c0c`.
- Configuration: `configs/pose/mediapipe_karsl_pilot.json`; CPU delegate,
  VIDEO mode, two hands, all three confidence thresholds 0.5.
- Run command: the command in [How to Run](#how-to-run), with output directory
  `runs/mediapipe_karsl_pilot`.
- Raw stage: compressed NPZ arrays retain frame indices, source seconds,
  MediaPipe timestamps, detector-order image/world landmarks, presence,
  handedness labels/scores, and metadata. No generated raw pose, overlay,
  model, or video artifact is committed.
- Verification: 18/18 manifest checksums matched; all inspected videos reported
  30.0 FPS and 1920×1080; decoded and reported frame counts matched.

## Next steps

1. Run the separate WiLoR branch against this exact manifest and compare raw
   coverage, temporal continuity, and runtime using the same report fields.
2. Add a small manually reviewed identity/occlusion ground-truth subset before
   interpreting identity-instability or jitter as acceptance criteria.
3. Investigate temporal identity association and missing-hand handling as a
   separate tracked/cleaned stage; preserve this raw MediaPipe output.
4. Expand signs/repetitions only after the pilot comparison is stable, and
   investigate an independent PTS/ffprobe path for variable-frame-rate inputs.
5. Investigate the approved SSHI source in a later dedicated acquisition task.
