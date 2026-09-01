# Task

Resolve the pre-comparison fairness blockers identified by
`TASK-003A`, without changing either frozen pose experiment and without
selecting a model. This remediation restores the lost WiLoR full-mode
artifacts, validates both inputs against the shared pilot contract, and adds
an extractor-neutral metric and timing layer for `TASK-003B`.

Run date: 2026-09-02.

# Branch

`luna/task-003a-fairness-remediation`

The remediation was performed in the separate worktree:

`/home/hatim/Graduation-Project-Simulation-remediation`

The MediaPipe and WiLoR source worktrees were detached/read-only for this
task. Neither experimental branch was modified, merged, or retuned.

# Scope

In scope:

- regenerate and hard-validate the WiLoR normal/FP32 full-mode run;
- regenerate four real WiLoR 3D skeleton visualizations;
- validate the exact 18-video, 894-frame shared KArSL pilot contract;
- compute common reconstruction-validity, coverage, missing-streak, bone,
  duplicate/extra-hand, swap-heuristic, normalized temporal, and FPS metrics;
- measure both frozen implementations under a matched software timing
  boundary;
- preserve all generated artifacts outside the repository in ignored/stable
  run storage.

Out of scope:

- model comparison or winner selection;
- changes to either extractor, thresholds, model/checkpoint, MANO behavior,
  manifest, or raw arrays;
- sign recognition, NLP, TTS, Hall/IMU simulation, sensor optimization, or
  large-scale dataset processing.

# Approach

The neutral layer reads the two frozen NPZ formats through explicit paths. It
does not glob `runs/wilor*`, rewrite raw arrays, smooth, interpolate, or
relabel detector output. It rejects a WiLoR input before metrics if the run
is not exact `mode="full"`, and then applies a stricter per-row complete-pose
predicate. All coverage counts use decoded frames as the denominator and are
frame-weighted in aggregate.

The neutral metric layer uses one representative per valid `left` or
`right` label in a frame. Within a system, the representative is the highest
available native confidence; ties or missing confidence use the lowest raw
source index. Native confidence values are not compared between systems.
Duplicate and extra-hand events remain separately visible.

# Source Audit

The authoritative input was:

`reports/evaluation/TASK-003A-precomparison-fairness-audit.md`

at audit commit `66192c015a6fcee20ad015727a5891810a028c10`. The audit listed
one blocking issue (lost WiLoR full artifacts), five important metric issues,
and six minor issues. Before implementation, the remediation source was
checked against the frozen loader, extraction, metric, runner, and report
files. The audit findings were confirmed rather than treated as an
unverified checklist.

Supporting evidence includes the frozen experiment reports and source at
the commits below, the byte-identical pilot manifest, the decoded video
inspection outputs, and the regenerated raw NPZs. The upstream references
used by the frozen experiments remain:

- [official KArSL dataset page](https://hamzah-luqman.github.io/KArSL/);
- [official KArSL repository](https://github.com/Hamzah-Luqman/KArSL);
- [original KArSL research record](https://dl.acm.org/doi/10.1145/3423420);
- [MediaPipe Hand Landmarker Python guide](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python);
- [official WiLoR repository](https://github.com/rolpotamias/WiLoR);
- [MANO project and license pages](https://mano.is.tue.mpg.de/).

# Evidence / Sources

The primary evidence for this remediation is machine-readable and retained
at the stable artifact paths in the reproducibility section: the regenerated
WiLoR `summary.json` and 18 raw NPZs, the harmonized metrics JSON, the
matched timing JSON, and the four full-mode visualization MP4s. The code
locations below are the neutral implementations that validate and derive
those artifacts; the frozen source locations are identified by their exact
commits above.

SSHI access at `https://sshi.sa/` has been approved for this project. SSHI
data collection/scraping is intentionally deferred to a later dedicated
task and was not used here.

# Frozen Experimental Commits

| Experiment | Branch | Commit | Status during remediation |
|---|---|---|---|
| MediaPipe | `luna/mediapipe-karsl-pilot` | `ed25d9f2814493f02e16848d23c3466b54f06d6e` | frozen/read-only |
| WiLoR + MANO | `opus/wilor-karsl-pilot` | `20e83afd7a54493523389fe02ca7077b1afc5866` | frozen/read-only |
| Fairness audit | `opus/task-003a-fairness-audit` | `66192c015a6fcee20ad015727a5891810a028c10` | report retained as base |

Detached worktrees used:

- `/home/hatim/Graduation-Project-Simulation-luna` at the MediaPipe commit;
- `/home/hatim/Graduation-Project-Simulation-opus` at the WiLoR commit;
- `/home/hatim/Graduation-Project-Simulation-audit` at the audit commit;
- `/home/hatim/Graduation-Project-Simulation-remediation` for this branch.

# Dataset and Input Contract

The shared manifest is:

`datasets/manifests/karsl_milestone1_pilot.csv`

Its SHA-256 is
`4a8e0d24f587d9c54ad87ebd896c6df0290b3ebc80885774100d71b29dbc3c0c`.
It contains exactly 18 deterministic samples: signs `0171`–`0176`, signers
`01`–`03`, test split, and the lexicographically first valid MP4 for each
sign/signer pair. The local source files were checksum-verified against the
manifest before neutral evaluation. OpenCV/NPZ inspection agrees at exactly
894 decoded frames, with per-video counts preserved in the two frozen runs.

No source video is tracked by this branch.

# F-1 Artifact Regeneration

## Exact command and environment

The WiLoR worktree was recreated detached at the exact frozen commit and
the existing normal/FP32 benchmark command was run without `--fast`:

```text
cd /home/hatim/Graduation-Project-Simulation-opus
WILOR_ASSETS_DIR=/home/hatim/.cache/wilor_assets \
WILOR_SOURCE_DIR=/home/hatim/.cache/wilor_assets/WiLoR \
python evaluation/benchmarks/wilor_karsl_pilot.py \
  --out-dir runs/wilor_karsl_pilot_full
```

The explicit `WILOR_SOURCE_DIR` points to the official WiLoR checkout used
by the frozen adapter. `WILOR_ASSETS_DIR` resolves to the existing external
asset directory; no model files were copied into the repository.
The model/checkpoint, MANO behavior, confidence `0.3`, rescale factor `2.0`,
precision mode, and pilot manifest were not changed; only the ignored output
destination was recreated.

Available local MANO assets were confirmed:

```text
/home/hatim/.cache/wilor_assets/mano_data/MANO_RIGHT.pkl  3.7M
/home/hatim/.cache/wilor_assets/mano_data/MANO_LEFT.pkl   3.7M
```

The required `MANO_RIGHT.pkl`, checkpoint, detector, configuration, and
supporting files passed the frozen `check_assets()` checks. `MANO_LEFT.pkl`
was also present for the previously documented convention validation; it was
not copied or committed.

## Hard validation

The regenerated summary and every raw NPZ were checked before neutral metric
calculation:

| Check | Result |
|---|---:|
| `summary.json["mode"]` | `full` |
| Videos requested / processed / failed | 18 / 18 / 0 |
| Decoded frames | 894 |
| Full-mode raw NPZ files | 18 |
| Reconstructed rows | 1,779 |
| Rows with `hand_present=True` but invalid reconstruction | 0 |
| Rows with `detector_only_no_mano` | 0 |
| Rows with finite 21x3 `landmarks_3d` | 1,779 / 1,779 |
| Rows with non-empty `hand_pose_rotmat`, `global_orient_rotmat`, and `betas` | 1,779 / 1,779 |
| Mesh rows with finite 778x3 vertices | 1,779 / 1,779 |

Every per-video summary row also reported `mode="full"`, zero frame errors,
and its frame count matched the manifest/video inspection. This prevents the
surviving detector-only Phase-A directory, `runs/wilor_karsl_pilot/`, from
being accepted accidentally.

## Stable artifact locations

The original generated run was retained in the detached worktree while it
was validated:

`/home/hatim/Graduation-Project-Simulation-opus/runs/wilor_karsl_pilot_full`

It was then copied, without deleting the original, to this stable location
outside disposable worktrees:

`/home/hatim/graduation-project-runs/wilor_karsl_pilot_full`

The stable copy contains `summary.json`, all 18 full raw NPZs, and the four
visual MP4s listed below. The source video and model/checkpoint files remain
external and ignored.

## Regenerated visual evidence

The following videos contain actual projected full WiLoR 21-joint/MANO
reconstruction, not detector-only boxes:

| Observation | Stable visualization |
|---|---|
| easy dual-hand clip | `/home/hatim/graduation-project-runs/wilor_karsl_pilot_full/visual/karsl_test_s02_sign0171_repfirst_full3d.mp4` |
| occlusion clip | `/home/hatim/graduation-project-runs/wilor_karsl_pilot_full/visual/karsl_test_s01_sign0174_repfirst_full3d.mp4` |
| spurious-extra-detection clip | `/home/hatim/graduation-project-runs/wilor_karsl_pilot_full/visual/karsl_test_s02_sign0176_repfirst_full3d.mp4` |
| high-motion/foreshortening clip | `/home/hatim/graduation-project-runs/wilor_karsl_pilot_full/visual/karsl_test_s03_sign0173_repfirst_full3d.mp4` |

Representative frames were inspected manually. The easy clip showed two
projected skeletons; the occlusion clip showed the expected reduced visible
hand evidence; the spurious-extra clip contained the documented three-valid-
reconstruction frame at frame 39; and the high-motion clip showed projected
reconstruction over the moving hands. These are qualitative validation
observations, not ground-truth accuracy claims.

No MP4, OBJ, NPZ, checkpoint, or MANO file is committed.

# Neutral Metric Remediation

The neutral implementation is under `evaluation/comparison/`.

The complete-pose predicate is:

```text
MediaPipe:
  hand_present
  AND image_landmarks.shape == (21, 3) and finite
  AND landmarks_3d.shape == (21, 3) and finite

WiLoR:
  hand_present
  AND mode == "full"
  AND landmarks_3d.shape == (21, 3) and finite
  AND hand_pose_rotmat, global_orient_rotmat, and betas are non-empty/finite
```

MANO is required only for WiLoR because it is part of WiLoR's expected full
representation; MediaPipe uses its image plus world landmark representation.
The raw `hand_present` meaning in WiLoR was not changed.

| Finding | Original severity | Resolution | Code/report location | Status |
|---|---|---|---|---|
| F-1 lost WiLoR full artifacts | BLOCKING | Regenerated the exact full/FP32 run at `20e83af`, hard-checked 18 videos, 894 frames, finite 21-joint/MANO/mesh output, and restored four real 3D visualizations in stable ignored storage. | This report, `evaluation/comparison/loaders.py` | RESOLVED |
| F-2 inclusive vs exclusive per-hand coverage | IMPORTANT | Both systems now emit inclusive left/right, both, left-only, right-only, no-hand, and at-least-one counts over decoded frames. | `harmonized_metrics.py` | RESOLVED |
| F-3 missing-streak definitions | IMPORTANT | Both systems now emit no-hand, left-missing, and right-missing consecutive-frame streaks. Gaps in frame indices break a streak. | `harmonized_metrics.py` | RESOLVED |
| F-4 different bone edge sets | IMPORTANT | Added `COMMON_HAND_BONES_18`, asserted as the exact set intersection of the frozen 21-edge and 20-edge lists, and recomputed CV only on those 18 edges. | `common_contract.py`, `harmonized_metrics.py` | RESOLVED |
| F-5 timing scopes differ | IMPORTANT | Added matched wrappers that load each model once, then time decode, required preprocessing, inference/reconstruction, and in-memory pose conversion only. | `performance_benchmark.py` | RESOLVED |
| F-6 `hand_present` is not reconstruction validity | IMPORTANT | Added the strict system-specific `reconstructed_hand` predicate and hard failure for any WiLoR run not in exact `full` mode. | `common_contract.py`, `loaders.py` | RESOLVED |
| F-7 FPS aggregation differs | MINOR | Aggregate FPS is always `total_frames / total_elapsed_seconds`; per-video values remain separately recorded. | `harmonized_metrics.py`, `performance_benchmark.py` | RESOLVED |
| F-8 duplicate handedness labels | MINOR | One deterministic representative per label is used for temporal metrics; duplicate-left/right events remain counted. | `harmonized_metrics.py` | RESOLVED |
| F-9 two-hand cap asymmetry | MINOR | Native coverage is retained; temporal metrics use one left/one right representative; `frames_with_more_than_2_hands` and `extra_hand_events` preserve WiLoR's third-hand event. The MediaPipe cap remains disclosed. | `harmonized_metrics.py`, this report | RESOLVED |
| F-10 timestamp sources differ | MINOR | No correction was applied. MediaPipe decoder-position timestamps and WiLoR `frame_index / FPS` are disclosed; the audit measured a maximum difference of approximately `4.4e-16 s` on this CFR pilot. Frame index is the TASK-003B alignment key. | This report | DISCLOSED |
| F-11 confidence thresholds differ | MINOR | No normalization or rerun was performed. MediaPipe remains 0.5 for detection/presence/tracking; WiLoR remains detector confidence 0.3. The difference is disclosed next to coverage. | Frozen metadata, this report | DISCLOSED |
| F-12 incompatible original jitter metrics | MINOR | Added one common option-A metric: root-center each valid 21-joint 3D pose, divide by `norm(joint[9]-joint[0])`, then use the same second difference and mean per-joint L2. Original metre/model-unit values are not compared. | `harmonized_metrics.py`, this report | RESOLVED |

The common swap heuristic is also new and must not be confused with either
historical count. For consecutive frames with one selected left and one
selected right hand, it compares reported-label displacement with the
left/right-swapped displacement in normalized image-space hand-centre proxy
coordinates. A suspected swap is counted only when the swapped total is
strictly lower. It is not identity ground truth.

# Implementation

## Coverage, streaks, duplicates, and extra hands

For each decoded frame, valid reconstructed rows are grouped by frame. The
inclusive left/right flags use membership in valid rows and include
both-hand frames. `left_only` and `right_only` are the exclusive subsets.
`no_hand` means zero valid reconstructed rows, while `at_least_one` is its
complement. All of these are frame-count metrics; aggregate percentages use
894 total frames.

Duplicate events count frames containing more than one valid row with the
same recognized label. `frames_with_more_than_2_hands` counts frames with
more than two valid reconstructed rows; it is not silently truncated for
WiLoR. The normalized temporal sequence can still select one representative
per label.

## Bones and normalized temporal metric

The common topology is exactly these 18 OpenPose-index edges:

```text
(0,1), (1,2), (2,3), (3,4),
(0,5), (5,6), (6,7), (7,8),
(9,10), (10,11), (11,12),
(13,14), (14,15), (15,16),
(0,17), (17,18), (18,19), (19,20)
```

The neutral bone metric computes per-edge lengths from the selected valid
left/right sequence, then per-edge temporal coefficient of variation and a
distribution across the 18 edges. It never alters a raw joint array.

The common temporal metric is dimensionless per frame:

```text
q_t = (landmarks_3d_t - landmarks_3d_t[0]) /
      norm(landmarks_3d_t[9] - landmarks_3d_t[0])
d_t = mean_joint_norm(q[t+1] - 2*q[t] + q[t-1])
```

Only contiguous valid frames are used; no interpolation is performed. Root
translation and per-frame scale are removed, and the magnitude operator is
invariant to a fixed orthogonal coordinate change such as a fixed reflection.
This is appropriate as a common normalized temporal observation for this
pilot, but it is not a physical accuracy metric and is not used here to
select a model.

# Matched Timing

The matched timing boundary is:

```text
video decode
+ required model preprocessing
+ model inference/reconstruction
+ conversion into an in-memory pose representation
```

Excluded: model/checkpoint construction/loading, disk serialization, overlay
generation, video encoding, and report generation. Each model was loaded
once before timing. No separate warm-up pass was used; the first measured
inference therefore includes any lazy runtime initialization. Each worker
processed all 18 manifest videos and asserted 894 decoded frames.

| System | Execution hardware | Frames | Total seconds | Frame-weighted FPS |
|---|---|---:|---:|---:|
| MediaPipe | AMD Ryzen 7 6800HS CPU, MediaPipe CPU delegate | 894 | 25.337060155 | 35.284282965 |
| WiLoR | NVIDIA GeForce RTX 3050 Laptop GPU, CUDA `cuda`, normal FP32 | 894 | 269.138149174 | 3.321714156 |

These are practical throughputs under a matched software boundary using each
model's validated operating configuration. They are not hardware-normalized
and are not a performance winner declaration. The GPU was not used by
MediaPipe. The machine reported 7,747,232 kB RAM and NVIDIA driver 610.62;
WiLoR's regenerated full run reported a peak of approximately 2,819 MiB
allocated / 2,905 MiB reserved CUDA memory.

The exact machine-readable result is retained outside the repository at:

`/home/hatim/graduation-project-runs/task003a2_matched_timing.json`

## Per-video matched timing

| Sample | Frames | MediaPipe s | MediaPipe FPS | WiLoR s | WiLoR FPS |
|---|---:|---:|---:|---:|---:|
| `karsl_test_s01_sign0171_repfirst` | 81 | 2.281 | 35.518 | 32.602 | 2.485 |
| `karsl_test_s01_sign0172_repfirst` | 52 | 1.410 | 36.878 | 18.666 | 2.786 |
| `karsl_test_s01_sign0173_repfirst` | 51 | 1.502 | 33.955 | 12.365 | 4.124 |
| `karsl_test_s01_sign0174_repfirst` | 34 | 0.972 | 34.971 | 7.908 | 4.299 |
| `karsl_test_s01_sign0175_repfirst` | 41 | 1.143 | 35.877 | 10.189 | 4.024 |
| `karsl_test_s01_sign0176_repfirst` | 38 | 1.042 | 36.465 | 9.506 | 3.997 |
| `karsl_test_s02_sign0171_repfirst` | 67 | 1.791 | 37.411 | 17.679 | 3.790 |
| `karsl_test_s02_sign0172_repfirst` | 36 | 1.179 | 30.531 | 9.133 | 3.942 |
| `karsl_test_s02_sign0173_repfirst` | 44 | 1.191 | 36.938 | 11.529 | 3.816 |
| `karsl_test_s02_sign0174_repfirst` | 36 | 1.001 | 35.957 | 9.405 | 3.828 |
| `karsl_test_s02_sign0175_repfirst` | 48 | 1.423 | 33.725 | 13.603 | 3.529 |
| `karsl_test_s02_sign0176_repfirst` | 57 | 1.696 | 33.618 | 19.980 | 2.853 |
| `karsl_test_s03_sign0171_repfirst` | 68 | 1.814 | 37.495 | 26.308 | 2.585 |
| `karsl_test_s03_sign0172_repfirst` | 50 | 1.415 | 35.336 | 13.514 | 3.700 |
| `karsl_test_s03_sign0173_repfirst` | 38 | 1.060 | 35.853 | 12.130 | 3.133 |
| `karsl_test_s03_sign0174_repfirst` | 38 | 1.082 | 35.134 | 10.825 | 3.510 |
| `karsl_test_s03_sign0175_repfirst` | 56 | 1.548 | 36.187 | 16.281 | 3.440 |
| `karsl_test_s03_sign0176_repfirst` | 59 | 1.789 | 32.987 | 17.513 | 3.369 |

# Files Changed

- `evaluation/comparison/__init__.py`
- `evaluation/comparison/common_contract.py`
- `evaluation/comparison/loaders.py`
- `evaluation/comparison/harmonized_metrics.py`
- `evaluation/comparison/performance_benchmark.py`
- `scripts/run_task003a2_remediation.py`
- `tests/test_comparison_fairness.py`
- `pyproject.toml` (adds the already-used NumPy dependency and discovers the
  `evaluation` package)
- this report.

No generated run directory, raw pose file, video, visualization, mesh,
checkpoint, MANO asset, or cache is part of the change set.

# How to Run

From the remediation worktree:

```bash
python -m unittest discover -s tests -p 'test_*.py'
python -m compileall -q evaluation scripts tests
```

Generate strict harmonized metrics using the stable ignored artifacts:

```bash
python scripts/run_task003a2_remediation.py \
  --manifest /home/hatim/Graduation-Project-Simulation-luna/datasets/manifests/karsl_milestone1_pilot.csv \
  --video-root /home/hatim/Graduation-Project-Simulation \
  --mediapipe-run /home/hatim/graduation-project-runs/mediapipe_karsl_pilot \
  --wilor-run /home/hatim/graduation-project-runs/wilor_karsl_pilot_full \
  --output /home/hatim/graduation-project-runs/task003a2_harmonized_metrics.json
```

Run the matched timer only when the two detached frozen worktrees and
external assets are available:

```bash
python -m evaluation.comparison.performance_benchmark \
  --mediapipe-root /home/hatim/Graduation-Project-Simulation-luna \
  --wilor-root /home/hatim/Graduation-Project-Simulation-opus \
  --manifest /home/hatim/Graduation-Project-Simulation-luna/datasets/manifests/karsl_milestone1_pilot.csv \
  --video-root /home/hatim/Graduation-Project-Simulation \
  --model /home/hatim/Graduation-Project-Simulation/datasets/raw/models/hand_landmarker.task \
  --wilor-assets-dir /home/hatim/.cache/wilor_assets \
  --wilor-source-dir /home/hatim/.cache/wilor_assets/WiLoR \
  --output /home/hatim/graduation-project-runs/task003a2_matched_timing.json
```

The timer fails if CUDA is unavailable for the validated WiLoR configuration;
it does not silently substitute detector-only or CPU behavior.

# Evaluation

Evaluation was run only after strict input validation. The evaluator checks:

- exact manifest SHA and all 18 sample IDs;
- source-video existence and manifest checksums;
- 18 successful MediaPipe raw NPZs with frozen VIDEO/2-hand/CPU metadata and
  0.5 thresholds;
- 18 successful WiLoR full-mode NPZs with summary mode, source identity,
  finite complete reconstruction, MANO fields, and finite 778x3 meshes;
- exactly 894 frames in both validated runs.

The historical model-specific reports remain unchanged. Their old
per-channel, bone, jitter, timing, and swap figures are not relabeled as
harmonized results.

# Results

The machine-readable neutral output is:

`/home/hatim/graduation-project-runs/task003a2_harmonized_metrics.json`

## Aggregate harmonized results

Counts below are frame counts over 894 decoded frames. `left_inclusive` and
`right_inclusive` include both-hand frames. Bone CV is a unitless ratio over
the common 18 edges; the normalized temporal statistic is the dimensionless
second-difference metric defined above. These values are reported for the
next comparison and do not select a winner here.

| Metric | MediaPipe | WiLoR |
|---|---:|---:|
| Videos / total frames | 18 / 894 | 18 / 894 |
| Left inclusive | 803 (89.821%) | 894 (100.000%) |
| Right inclusive | 836 (93.512%) | 884 (98.881%) |
| Both hands | 751 (84.004%) | 884 (98.881%) |
| Left only | 52 (5.817%) | 10 (1.119%) |
| Right only | 85 (9.508%) | 0 (0.000%) |
| At least one reconstructed hand | 888 (99.329%) | 894 (100.000%) |
| No reconstructed hand | 6 (0.671%) | 0 (0.000%) |
| Longest no-hand streak | 6 | 0 |
| Longest left-missing streak | 18 | 0 |
| Longest right-missing streak | 27 | 10 |
| Duplicate-left / duplicate-right events | 9 / 23 | 1 / 0 |
| Frames with >2 valid hands | 0 | 1 |
| Extra-hand events | 0 | 1 |
| Common suspected-swap events | 0 | 0 |
| Mean per-video common-edge CV | 0.1871856973 | 0.0058696355 |
| Mean per-video normalized temporal metric | 0.1092670726 | 0.0759610925 |
| Mean per-video normalized temporal p95 | 0.2929107770 | 0.2593134440 |

The WiLoR three-hand event is retained in the counts rather than hidden by a
two-hand truncation. The MediaPipe historical heuristic count of 34 and any
historical WiLoR swap candidates are not compared with the new common
heuristic; the common event definition above is the one reserved for
`TASK-003B`.

## Per-video results

| Sample | Frames | MP both | WiLoR both | MP no-hand | WiLoR no-hand | MP L/R inclusive | WiLoR L/R inclusive | MP missing L/R/no | WiLoR missing L/R/no | MP dup L/R | WiLoR dup L/R | WiLoR >2 | MP common swap | WiLoR common swap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `karsl_test_s01_sign0171_repfirst` | 81 | 81 | 81 | 0 | 0 | 81/81 | 81/81 | 0/0/0 | 0/0/0 | 0/0 | 0/0 | 0 | 0 | 0 |
| `karsl_test_s01_sign0172_repfirst` | 52 | 52 | 52 | 0 | 0 | 52/52 | 52/52 | 0/0/0 | 0/0/0 | 0/0 | 0/0 | 0 | 0 | 0 |
| `karsl_test_s01_sign0173_repfirst` | 51 | 51 | 51 | 0 | 0 | 51/51 | 51/51 | 0/0/0 | 0/0/0 | 0/0 | 0/0 | 0 | 0 | 0 |
| `karsl_test_s01_sign0174_repfirst` | 34 | 11 | 24 | 0 | 0 | 12/33 | 34/24 | 12/1/0 | 0/10/0 | 0/0 | 0/0 | 0 | 0 | 0 |
| `karsl_test_s01_sign0175_repfirst` | 41 | 38 | 41 | 0 | 0 | 38/41 | 41/41 | 3/0/0 | 0/0/0 | 0/0 | 0/0 | 0 | 0 | 0 |
| `karsl_test_s01_sign0176_repfirst` | 38 | 37 | 38 | 0 | 0 | 38/37 | 38/38 | 0/1/0 | 0/0/0 | 1/0 | 0/0 | 0 | 0 | 0 |
| `karsl_test_s02_sign0171_repfirst` | 67 | 63 | 67 | 0 | 0 | 64/66 | 67/67 | 2/1/0 | 0/0/0 | 0/2 | 0/0 | 0 | 0 | 0 |
| `karsl_test_s02_sign0172_repfirst` | 36 | 18 | 36 | 6 | 0 | 18/30 | 36/36 | 18/6/6 | 0/0/0 | 0/0 | 0/0 | 0 | 0 | 0 |
| `karsl_test_s02_sign0173_repfirst` | 44 | 44 | 44 | 0 | 0 | 44/44 | 44/44 | 0/0/0 | 0/0/0 | 0/0 | 0/0 | 0 | 0 | 0 |
| `karsl_test_s02_sign0174_repfirst` | 36 | 36 | 36 | 0 | 0 | 36/36 | 36/36 | 0/0/0 | 0/0/0 | 0/0 | 0/0 | 0 | 0 | 0 |
| `karsl_test_s02_sign0175_repfirst` | 48 | 20 | 48 | 0 | 0 | 47/21 | 48/48 | 1/27/0 | 0/0/0 | 1/1 | 0/0 | 0 | 0 | 0 |
| `karsl_test_s02_sign0176_repfirst` | 57 | 39 | 57 | 0 | 0 | 39/57 | 57/57 | 12/0/0 | 0/0/0 | 0/1 | 1/0 | 1 | 0 | 0 |
| `karsl_test_s03_sign0171_repfirst` | 68 | 61 | 68 | 0 | 0 | 65/64 | 68/68 | 2/4/0 | 0/0/0 | 1/1 | 0/0 | 0 | 0 | 0 |
| `karsl_test_s03_sign0172_repfirst` | 50 | 50 | 50 | 0 | 0 | 50/50 | 50/50 | 0/0/0 | 0/0/0 | 0/0 | 0/0 | 0 | 0 | 0 |
| `karsl_test_s03_sign0173_repfirst` | 38 | 23 | 38 | 0 | 0 | 23/38 | 38/38 | 14/0/0 | 0/0/0 | 0/15 | 0/0 | 0 | 0 | 0 |
| `karsl_test_s03_sign0174_repfirst` | 38 | 21 | 38 | 0 | 0 | 38/21 | 38/38 | 0/17/0 | 0/0/0 | 5/0 | 0/0 | 0 | 0 | 0 |
| `karsl_test_s03_sign0175_repfirst` | 56 | 52 | 56 | 0 | 0 | 52/56 | 56/56 | 2/0/0 | 0/0/0 | 0/3 | 0/0 | 0 | 0 | 0 |
| `karsl_test_s03_sign0176_repfirst` | 59 | 54 | 59 | 0 | 0 | 55/58 | 59/59 | 3/1/0 | 0/0/0 | 1/0 | 0/0 | 0 | 0 | 0 |

The triplets in the missing-streak columns are `left/right/no-hand`; the
duplicate columns are `left/right`.

# Visual Observations

The regenerated WiLoR visualizations are real full-mode skeleton overlays
and were checked against the documented TASK-002 cases. The easy dual-hand
clip shows two projected reconstructions. The occlusion case loses visible
hand evidence in the expected frames. The spurious-extra case keeps the
third valid reconstructed row visible to metrics at frame 39. The
high-motion/foreshortening case shows projected skeletons through the rapid
motion. These observations establish that 3D visual evidence exists again;
they do not establish anatomical accuracy or ground-truth identity.

# Failures / Limitations

- The regenerated full WiLoR run is from the same frozen implementation,
  weights, normal FP32 mode, detector confidence `0.3`, and rescale factor
  `2.0`; its measured wall time is not interchangeable with the historical
  report because it includes the original runner's own scope.
- MediaPipe uses decoder-position timestamps and WiLoR uses
  `frame_index / FPS`. They are effectively equal for this CFR pilot, but
  frame index remains the alignment key.
- MediaPipe and WiLoR retain their different confidence thresholds (`0.5`
  and `0.3`). Coverage is therefore configuration-specific, not threshold
  normalized.
- Absolute 3D positions, wrist displacement in metres/model units, MANO
  translation, and other cross-coordinate-system values are not ranked.
- The common temporal metric removes root translation and scale but has no
  ground-truth motion target; it is a descriptive normalized continuity
  statistic only.
- The swap count is a 2D displacement heuristic, not identity ground truth.
- There is no annotated occlusion, false-positive, or thumb-correctness
  ground truth in this pilot. Visual observations remain qualitative.
- Matched timing uses CPU for MediaPipe and CUDA for WiLoR. It is not a
  hardware-normalized speed comparison.

# Performance

The matched timing results are the performance figures intended for any
future comparison:

- MediaPipe CPU delegate: 894 frames in 25.337060155 seconds,
  35.284282965 frame-weighted FPS.
- WiLoR normal FP32 CUDA: 894 frames in 269.138149174 seconds,
  3.321714156 frame-weighted FPS.

The historical frozen values (MediaPipe 21.865 end-to-end FPS / 48.880
inference FPS; WiLoR 4.18 mean-of-video full-mode FPS) remain in their
original reports and are not overwritten. They are not directly compared in
this remediation.

# Comparison

No MediaPipe-versus-WiLoR model comparison or winner selection was performed.
This task only makes the shared input, validity, metric, and timing contract
safe for `TASK-003B`. The neutral outputs deliberately show both systems
without converting different coordinate systems or confidence thresholds
into a synthetic common scale.

# Recommendation

**NEEDS MORE EVALUATION** — the fairness layer is ready for the planned
`TASK-003B` comparison, but no pose model should be selected from this
remediation report.

# Reproducibility

## Dependencies and hardware

The activated repository environment was used. Versions observed while
running the neutral layer were:

| Package | Version |
|---|---|
| Python | repository `.venv` interpreter |
| NumPy | 2.5.2 |
| OpenCV | 5.0.0.93 |
| MediaPipe | 1.0.1 |
| PyTorch | 2.13.0+cu130 (`torch.version.cuda == 13.0`) |
| Ultralytics | 8.1.34 |
| SMPL-X | 0.1.28 |
| Pillow | 12.3.0 |

Hardware was AMD Ryzen 7 6800HS with Radeon Graphics, NVIDIA GeForce RTX
3050 Laptop GPU (4096 MiB), driver 610.62, and approximately 7.75 GB visible
RAM. MediaPipe explicitly used CPU; WiLoR reported `torch.cuda.is_available()
== True` and used CUDA.

## Stable generated artifacts

- MediaPipe ignored run copy:
  `/home/hatim/graduation-project-runs/mediapipe_karsl_pilot`
- WiLoR ignored full run copy:
  `/home/hatim/graduation-project-runs/wilor_karsl_pilot_full`
- Harmonized metrics:
  `/home/hatim/graduation-project-runs/task003a2_harmonized_metrics.json`
- Matched timing:
  `/home/hatim/graduation-project-runs/task003a2_matched_timing.json`

The MediaPipe run copy was retained so both systems' raw inputs are available
outside temporary worktrees. All four generated directories/files are
outside Git and are reproducible from the explicit commands in this report.

# Tests

The neutral fairness suite uses synthetic records and temporary directories;
it does not require the 2.56 GB WiLoR checkpoint or KArSL videos.

```text
python -m unittest discover -s tests -p 'test_*.py'
.................
----------------------------------------------------------------------
Ran 17 tests in 0.014s

OK
```

The completed test set covers detector-only/full-mode rejection, finite
reconstruction predicates for both systems, inclusive/exclusive coverage,
all three streak definitions, exact 18-edge intersection, duplicate
normalization, frame-weighted FPS, the common swap heuristic, the normalized
temporal metric, and path safety/no-glob input handling. Bytecode compilation
also passed:

```text
python -m compileall -q evaluation scripts tests
```

The strict artifact evaluator completed with:

```text
validation: PASS
videos: 18
frames: 894
```

# Next Steps

1. Keep the stable ignored WiLoR full artifacts available to the
   `TASK-003B` worktree.
2. Have `TASK-003B` ingest the explicit manifest and both validated run
   paths; do not glob historical WiLoR directories.
3. Use harmonized coverage, streak, common-bone, duplicate/extra-hand,
   common-swap, and normalized temporal fields as defined here.
4. Keep absolute 3D and threshold-sensitive figures descriptive, with the
   hardware and threshold disclosures retained.
5. Perform model comparison only in `TASK-003B`, using the frozen raw outputs
   and this contract.

# FAIRNESS READINESS VERDICT

**READY FOR TASK-003B**

All required remediation conditions are satisfied: WiLoR full artifacts and
real 3D visual evidence were restored; 18 videos and 894 frames were hard
validated; complete MANO reconstruction was verified; common coverage,
streak, bone, duplicate, extra-hand, FPS, swap, and normalized temporal
rules exist; timing is matched and explicitly hardware-qualified; thresholds
and timestamp differences are disclosed; and the unit tests pass. No model
was selected and no frozen experimental result was altered.
