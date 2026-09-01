# Task

Final controlled comparison of the two frozen Milestone-1 pose experiments
(TASK-001 MediaPipe Hand Landmarker, TASK-002 WiLoR + MANO) using the
extractor-neutral fairness layer delivered by TASK-003A2, in order to select
the primary pose pipeline for the next project stage.

Neither model was re-implemented, re-tuned, re-thresholded, or re-run for
scoring. No frozen experiment was modified and no experimental PR was
merged. All harmonized values quoted here were re-verified from the
machine-readable artifacts rather than copied from prior reports.

Run date: 2026-09-02.

# Project Objective

This project is **not** primarily a computer-vision sign recognizer. The
intended system is:

```text
Arabic Sign Language video
  -> 3D hand reconstruction
  -> hand kinematics
  -> SIMULATED SMART GLOVE
  -> virtual sensor time-series
  -> LSTM
  -> recognized Arabic sign / character
  -> Arabic NLP
  -> Arabic TTS
```

Computer vision is being used to build a *software simulation of a smart
glove*. The physical glove is not required for the primary implementation.

Therefore the decisive question for this task is **not** "which extractor is
faster" but **which extractor yields reliable dual-hand temporal 3D data from
which virtual smart-glove sensor channels can be derived**. Real-time RGB
inference is not a requirement of the current stage; offline processing is
acceptable if the resulting 3D representation is materially better for
virtual-glove generation. Throughput is therefore weighted, but is not
permitted to dominate.

# Inputs

Read in full before analysis:

- `reports/evaluation/TASK-003A-precomparison-fairness-audit.md`
- `reports/evaluation/TASK-003A2-precomparison-remediation.md`
- `reports/pose/mediapipe/TASK-001-mediapipe-karsl-pilot.md`
- `reports/pose/wilor/TASK-002-wilor-karsl-pilot.md`

Fairness layer inspected and used unchanged:

- `evaluation/comparison/common_contract.py`
- `evaluation/comparison/loaders.py`
- `evaluation/comparison/harmonized_metrics.py`
- `evaluation/comparison/performance_benchmark.py`

Stable artifacts, referenced by explicit path (no `runs/wilor*` globbing):

| Artifact | Path |
|---|---|
| MediaPipe raw run | `/home/hatim/graduation-project-runs/mediapipe_karsl_pilot` |
| WiLoR full-MANO raw run | `/home/hatim/graduation-project-runs/wilor_karsl_pilot_full` |
| Harmonized metrics | `/home/hatim/graduation-project-runs/task003a2_harmonized_metrics.json` |
| Matched timing | `/home/hatim/graduation-project-runs/task003a2_matched_timing.json` |
| WiLoR 3D visualizations | `/home/hatim/graduation-project-runs/wilor_karsl_pilot_full/visual` |
| MediaPipe overlays | `/home/hatim/graduation-project-runs/mediapipe_karsl_pilot/overlays` |

# Frozen Experiments

| Experiment | Branch | Commit | Treatment |
|---|---|---|---|
| MediaPipe | `luna/mediapipe-karsl-pilot` | `ed25d9f2814493f02e16848d23c3466b54f06d6e` | frozen, read-only |
| WiLoR + MANO | `opus/wilor-karsl-pilot` | `20e83afd7a54493523389fe02ca7077b1afc5866` | frozen, read-only |
| Fairness audit | `opus/task-003a-fairness-audit` | `66192c015a6fcee20ad015727a5891810a028c10` | input |
| Fairness remediation | `luna/task-003a-fairness-remediation` | `63c7e683eeab19624a00480b9e0525e25ca07c44` | **base of this branch** |

This comparison branch, `evaluation/mediapipe-vs-wilor`, was created directly
from `63c7e683` in the isolated worktree
`../Graduation-Project-Simulation-comparison`.

## Independent artifact re-validation

The WiLoR full-mode artifacts were re-validated here, not taken on trust:

| Check | Result |
|---|---:|
| `summary.json["mode"]` | `full` |
| Per-video modes | `{full}` only |
| Videos processed / failed | 18 / 0 |
| Frame errors | 0 |
| Distinct decoded frames | 894 |
| Reconstructed rows | 1,779 |
| Rows with non-`full` mode or `detector_only_no_mano` flag | **0** |
| Rows with non-finite or non-21×3 `landmarks_3d` | **0** |
| Rows missing `hand_pose_rotmat` / `global_orient_rotmat` / `betas` | **0** |
| Videos with non-finite or non-778×3 mesh | **0** |

The harmonized metrics JSON was also re-read directly and every headline
value quoted in this report was reproduced from it. The MediaPipe common-18
bone CV was additionally recomputed from raw NPZ by an independent script
here (0.18719) and matches the neutral layer's 0.1871857 to five decimals,
which validates the fairness layer itself.

# Fairness Contract

The TASK-003A2 neutral layer is used unchanged. It resolves F-2 (coverage
definitions), F-3 (missing streaks), F-4 (bone topology), F-5 (timing
scope), F-6 (complete-reconstruction validity), F-7 (FPS aggregation), F-8
(duplicates), F-9 (extra-hand behavior) and F-12 (normalized temporal
metric), and discloses F-10 (timestamp source) and F-11 (thresholds). No
historical metric with a divergent definition is placed beside a harmonized
one anywhere in this report.

Two disclosures are carried into every conclusion below:

- **Thresholds are not normalized.** MediaPipe runs at 0.5
  detection/presence/tracking confidence; WiLoR at 0.3 detector confidence.
  Both are each model's validated intended default. They were deliberately
  **not** tuned. Coverage results are therefore configuration-specific.
- **Compute is not normalized.** MediaPipe ran on the **CPU** delegate;
  WiLoR ran on a **CUDA GPU**. Throughput figures are practical results for
  the validated configurations, not hardware-normalized efficiency.

**Absolute 3D quantities are not compared or ranked** anywhere in this
report: MediaPipe world landmarks are in metres, WiLoR/MANO joints and
camera translation are in the model's internal weak-perspective units. Only
the dimensionless common-bone CV and the dimensionless normalized temporal
metric are used for cross-system geometric comparison.

# Dataset

Unchanged shared pilot, verified again here:

- manifest `datasets/manifests/karsl_milestone1_pilot.csv`, SHA-256
  `4a8e0d24f587d9c54ad87ebd896c6df0290b3ebc80885774100d71b29dbc3c0c`;
- 18 videos: signs `0171`–`0176` × signers `01`–`03`, official `test` split,
  lexicographically first valid MP4 per pair;
- 894 decoded frames total, identical per video in both runs;
- 1920×1080 @ 30 fps.

No clip was added, removed, substituted, or reordered.

# Harmonized Coverage

All counts are frames, over the same 894-frame denominator. `left/right
inclusive` include both-hand frames.

| Metric | MediaPipe | WiLoR |
|---|---:|---:|
| Videos / frames | 18 / 894 | 18 / 894 |
| Both hands | **751 (84.004%)** | **884 (98.881%)** |
| Left inclusive | 803 (89.821%) | 894 (100.000%) |
| Right inclusive | 836 (93.512%) | 884 (98.881%) |
| Left only | 52 (5.817%) | 10 (1.119%) |
| Right only | 85 (9.508%) | 0 (0.000%) |
| At least one hand | 888 (99.329%) | 894 (100.000%) |
| No hand | 6 (0.671%) | 0 (0.000%) |

**Magnitude of the difference:**

- absolute improvement in dual-hand coverage: **+14.877 percentage points**;
- frames lacking both hands: **143 → 10**, a **93.0% relative reduction**;
- clips at 100% dual-hand coverage: MediaPipe **8/18**, WiLoR **17/18**.

## Per-video distribution

| Sample | Frames | MP both | MP % | WiLoR both | WiLoR % | MP no-hand | WiLoR no-hand |
|---|---:|---:|---:|---:|---:|---:|---:|
| s01_0171 | 81 | 81 | 100.0 | 81 | 100.0 | 0 | 0 |
| s01_0172 | 52 | 52 | 100.0 | 52 | 100.0 | 0 | 0 |
| s01_0173 | 51 | 51 | 100.0 | 51 | 100.0 | 0 | 0 |
| **s01_0174** | 34 | 11 | **32.4** | 24 | **70.6** | 0 | 0 |
| s01_0175 | 41 | 38 | 92.7 | 41 | 100.0 | 0 | 0 |
| s01_0176 | 38 | 37 | 97.4 | 38 | 100.0 | 0 | 0 |
| s02_0171 | 67 | 63 | 94.0 | 67 | 100.0 | 0 | 0 |
| **s02_0172** | 36 | 18 | **50.0** | 36 | 100.0 | 6 | 0 |
| s02_0173 | 44 | 44 | 100.0 | 44 | 100.0 | 0 | 0 |
| s02_0174 | 36 | 36 | 100.0 | 36 | 100.0 | 0 | 0 |
| **s02_0175** | 48 | 20 | **41.7** | 48 | 100.0 | 0 | 0 |
| **s02_0176** | 57 | 39 | **68.4** | 57 | 100.0 | 0 | 0 |
| s03_0171 | 68 | 61 | 89.7 | 68 | 100.0 | 0 | 0 |
| s03_0172 | 50 | 50 | 100.0 | 50 | 100.0 | 0 | 0 |
| **s03_0173** | 38 | 23 | **60.5** | 38 | 100.0 | 0 | 0 |
| **s03_0174** | 38 | 21 | **55.3** | 38 | 100.0 | 0 | 0 |
| s03_0175 | 56 | 52 | 92.9 | 56 | 100.0 | 0 | 0 |
| s03_0176 | 59 | 54 | 91.5 | 59 | 100.0 | 0 | 0 |

MediaPipe's missingness is concentrated in six clips (`s01_0174`,
`s02_0172`, `s02_0175`, `s02_0176`, `s03_0173`, `s03_0174`), which account
for 132 of its 143 frames lacking both hands (92%). WiLoR's entire
missingness is 10 frames in a single clip, `s01_0174`.

## Model miss versus true source occlusion

This distinction was checked frame-by-frame, not assumed.

**WiLoR's only loss (`s01_0174`, frames 17–26)** occurs in a crossed-arms
"love" gesture. Tracing both systems' detections through the crossing:

| Frames | MediaPipe | WiLoR |
|---|---|---|
| 0–10 | both hands, `right` at x≈931–968 px, `left` at x≈1260→1176 px | both hands, same ordering and convention (`right` x≈970–1004, `left` x≈1248→1096) |
| 11–13 | **loses one hand**, keeps only `Right` | still tracks both as they converge (`right` x≈1012, `left` x≈1058→1028) |
| 14–16 | still one hand | both hands, now nearly coincident (x≈1018–1024) |
| 17–26 | one hand, labelled `Right` (with a **label flicker to `Left` at frame 21** on a stationary hand, x 1010→1009→1012) | one hand, labelled `left`, stable across the whole segment |
| 27–33 | one hand | both hands recovered |

The two hands physically converge to the same image position and one becomes
occluded behind the other. Both systems consequently report a single hand —
this is a **true source-video occlusion**, not a WiLoR failure. WiLoR
retained both hands roughly six frames deeper into the convergence and
recovered them again afterwards, so its dual-hand tracking through the
approach is measurably better on the same physical event.

An important honesty note: in that occluded segment the two systems assign
**opposite handedness labels to the same surviving hand** (MediaPipe
`Right`, WiLoR `left`). At least one is wrong, and there is **no
ground-truth visibility or identity annotation**, so this report does not
claim which. The TASK-003A finding of 98.5% handedness agreement was
measured only on frames where both systems returned one left *and* one
right, which structurally excluded exactly these ambiguous single-hand
frames.

**MediaPipe's missingness is, in the inspected cases, not occlusion.**
Direct visual evidence (see Visual Review): in `s02_0175` frame 24 the
second hand rests plainly visible on the lap and MediaPipe returns nothing
for it while WiLoR reconstructs it; in `s02_0172` frame 18 both hands are
visible (one motion-blurred) and MediaPipe returns only one. These are model
misses of visible hands.

**Threshold caveat.** WiLoR's detector threshold (0.3) is lower than
MediaPipe's (0.5), and a lower threshold mechanically admits more
detections. Some of the coverage gap is therefore configuration-driven.
However, the misses shown above are of large, unambiguous, well-lit hands,
not marginal borderline detections, so the gap is not purely a threshold
artifact. Both settings are each model's validated default and were
deliberately left untouched.

# Missingness and Temporal Completeness

| Metric | MediaPipe | WiLoR |
|---|---:|---:|
| Longest no-hand streak | 6 | **0** |
| Longest left-missing streak | 18 | **0** |
| Longest right-missing streak | 27 | **10** |
| Duplicate-left events | 9 | 1 |
| Duplicate-right events | 23 | 0 |
| Frames with > 2 valid hands | 0 | 1 |
| Extra-hand events | 0 | 1 |
| Common suspected-swap events | 0 | 0 |

At 30 fps, MediaPipe's worst per-channel gap of 27 frames is **0.90 s of
continuously absent right-hand data**, and its 18-frame left gap is 0.60 s.
Its 6-frame no-hand streak is 0.20 s with no hand data at all. WiLoR's worst
gap is 10 frames (0.33 s) on one channel of one clip, and it never produces
a frame with zero hands.

**Consequence for the virtual sensor stream.** The downstream LSTM needs a
per-timestep sensor vector resembling `t0, t1, t2, …` without long
unexplained holes. With MediaPipe, 143 of 894 frames (16.0%) would arrive
with one or both glove channels absent, including sub-second contiguous
blocks; producing a usable stream would force interpolation, masking,
padding, or confidence gating over up to 27 consecutive timesteps —
i.e. fabricating ~0.9 s of glove state, which is long enough to span a
meaningful part of a 1–3 s sign. With WiLoR, 10 of 894 frames (1.1%) would
need such handling, all in one clip and all coincident with a genuine
physical occlusion where *no* extractor can recover the hidden hand.

Neither interpolation, masking, padding, nor gating is implemented in this
task; this is an evaluation of how much repair each option would later
demand.

MediaPipe's 32 duplicate-label events (9 left + 23 right) and the observed
label flicker on a stationary hand are an additional continuity hazard: a
duplicated or flipped label swaps which virtual glove a sample belongs to,
which is worse for an LSTM than an honest gap.

# Geometric Stability

Harmonized common 18-edge bone-length coefficient of variation (unitless,
scale-free):

| Metric | MediaPipe | WiLoR | Ratio |
|---|---:|---:|---:|
| Mean per-video common-edge CV | **0.1871856973** | **0.0058696355** | 31.9× |
| Per-video min / median / max | 0.1158 / 0.1805 / 0.2610 | 0.0041 / 0.0057 / 0.0102 | — |

The task required this large gap to be sanity-checked rather than accepted.
Four checks were run:

1. **Is it a metric artifact?** No. An independent recomputation from raw
   NPZ here reproduced the neutral layer's MediaPipe value (0.18719 vs
   0.1871857).
2. **Is it caused by pooling left and right hands of different sizes into
   one per-edge list?** Partly, but not mainly. Recomputing *per hand
   separately* gives MediaPipe 0.13473 and WiLoR 0.00434 — still a **31.0×**
   gap. Pooling explains 0.052 of MediaPipe's 0.187 (28%), and removing it
   from both sides leaves the conclusion unchanged.
3. **Is it a few bad clips or edges?** No. MediaPipe's CV is high in *every*
   one of the 18 videos (0.116–0.261) and across all edge groups; its worst
   edges are the proximal phalanges (`13-14` 0.305, `9-10` 0.293, `5-6`
   0.269) and even its most stable edge (`0-17`) is 0.112. WiLoR's worst
   edges are the distal fingertips (`19-20` 0.041, `3-4` 0.018).
4. **Is WiLoR's low CV tautological — i.e. are its bone lengths frozen by
   construction?** Not frozen, but structurally constrained. WiLoR's `betas`
   are re-estimated every frame and do vary (mean per-dimension std 0.0091,
   mean range 0.041), so bone lengths are free to move; they simply move
   very little because MANO's bone lengths are a smooth function of a
   10-dimensional shape space.

**Honest attribution:** the difference is a **representational** one. WiLoR
inherits an explicit parametric hand model that enforces a consistent
skeleton, whereas MediaPipe regresses 21 world landmarks per frame with no
skeletal constraint. This is *not* evidence that WiLoR's absolute anatomy is
more accurate — no ground truth exists here. It is evidence that WiLoR's
inferred skeleton is far more self-consistent over time.

For this project that distinction still matters directly: joint angles, and
therefore simulated sensor channels, must not fluctuate because the
estimated skeleton is changing its own bone proportions. A hand whose
own bone lengths vary by ~13–19% CV frame-to-frame injects that variation
into every derived flexion angle. Thumb edges specifically show the same
pattern (MediaPipe 0.176 vs WiLoR 0.0057).

# Normalized Temporal Continuity

Dimensionless second-difference metric (root-centred at joint 0, divided by
`‖joint9 − joint0‖`, operator `q[t+1] − 2q[t] + q[t−1]`):

| Metric | MediaPipe | WiLoR |
|---|---:|---:|
| Mean of per-video means | 0.1092670726 | 0.0759610925 |
| Mean of per-video p95 | 0.2929107770 | 0.2593134440 |
| Median of per-video means | 0.1017 | 0.0662 |
| Videos where WiLoR is smoother | — | **15 / 18** |

The advantage is broad, not concentrated in a few clips. WiLoR is smoother
in 15 of 18 videos, including large margins on `s03_0173` (0.1410 → 0.0494),
`s01_0176` (0.1432 → 0.0498) and `s03_0176` (0.1431 → 0.0629).

WiLoR is **worse** in exactly three clips: `s01_0175` (0.0927 → 0.1147),
`s02_0176` (0.1087 → 0.1099, effectively tied) and `s02_0172` (0.0649 →
0.2415, p95 1.109).

**These three clips are precisely the clips where MediaPipe's coverage is
lowest** (`s02_0172` 50.0% both-hand with 6 no-hand frames, `s02_0176`
68.4%, `s01_0175` 92.7%). The metric is computed only over each system's own
contiguous valid frames, so a system that returns nothing on the hard,
motion-blurred frames is scored only on the easy ones. This is a **selection
effect that biases the metric in favour of the lower-coverage system**, and
it is visually corroborated: at `s02_0172` frame 18, MediaPipe returns one
hand and skips the blurred one, while WiLoR reconstructs both including the
blurred hand — and pays for it in this metric.

This is a normalized temporal continuity observation only. It is **not** an
anatomical accuracy measurement, and it is not used as a primary decision
driver here.

# Representation Comparison

| Field | MediaPipe | WiLoR |
|---|---|---|
| 2D landmarks | 21 image landmarks (normalized) | detector bbox; 21 joints reprojectable via camera model |
| 3D joints | 21 world landmarks (metres, hand-relative) | 21 MANO→OpenPose joints (model units, root-relative) |
| Articulated rotations | **none** | **15 per-joint 3×3 local rotation matrices** |
| Global wrist/root orientation | **none** | **1 global orientation matrix** |
| Shape parameters | **none** | **10 MANO betas** (identity/shape separated from pose) |
| Global placement | **none** | camera translation (`camera_translation_xyz`) + focal length |
| Surface | none | **778-vertex hand mesh**, verified finite on all 1,779 rows |
| Handedness | label + score | label + detector confidence |
| Skeleton topology | implicit in landmark ordering | explicit MANO kinematic tree |

Both use the same 21-point index ordering (0 wrist, 1–4 thumb, 5–8 index,
9–12 middle, 13–16 ring, 17–20 pinky), so joint indices are interchangeable.

**Explicit correction of a common misconception:** WiLoR's MANO rotation
matrices are **not** smart-glove angles. They are local rotations relative to
MANO's template pose and kinematic-tree parents, not clinical
flexion/extension or abduction/adduction angles. A dedicated kinematics
layer (TASK-005) is still required regardless of which extractor is chosen.

The question is which representation gives that future layer stronger
footing. WiLoR provides (a) an explicit articulated kinematic tree with a
fixed parent/child structure, (b) pose separated from subject shape, (c) an
absolute wrist/root orientation, and (d) a dense surface for contact or
sensor-placement reasoning. MediaPipe provides positions only: any skeleton,
any parent frame, any bone-length model, and any wrist orientation must be
constructed from scratch, on top of landmarks whose implied bone lengths
drift ~13–19%.

# Virtual-Glove Suitability

Assessed against the concrete downstream requirement
(`3D → kinematics → virtual sensor channels → LSTM`).

| Requirement | MediaPipe | WiLoR |
|---|---|---|
| **Finger articulation** (MCP/PIP/DIP flexion, spread) | derivable from 21 positions, but every angle inherits the unstable bone geometry; no rotational prior | per-joint rotations already exist for 15 joints on a fixed tree; positions additionally available; stable bone lengths |
| **Thumb** | positions available; thumb-edge CV 0.176 | positions + thumb-chain rotations; thumb-edge CV 0.0057 |
| **Wrist / hand orientation** | **not provided** — must be inferred from a landmark triad, and is sensitive to landmark noise | provided directly as a global orientation matrix |
| **Coarse global movement** | hand-relative world landmarks only; no camera-space placement | camera translation available (in model units; not metrically calibrated) |
| **Dual-hand support** | 2-hand cap; 84.0% of frames carry both gloves | uncapped; 98.9% of frames carry both gloves |
| **Temporal consistency** | gaps up to 0.90 s per channel; 32 duplicate-label events | worst gap 0.33 s in one clip; 1 duplicate event |
| **Anatomical consistency** | bone CV 0.187 (0.135 per-hand) | bone CV 0.0059 (0.0043 per-hand) |

For simultaneous LEFT and RIGHT virtual gloves, WiLoR supplies both channels
on 884/894 frames versus MediaPipe's 751/894, and supplies the wrist
orientation channel that MediaPipe simply does not produce. This category is
where the two representations differ most, and it is directly aligned with
the project's stated purpose.

# Visual Review

**MANUAL QUALITATIVE REVIEW.** No ground truth exists; these are human
judgements on rendered artifacts, deliberately not converted into numeric
precision. Side-by-side frames were rendered from the validated artifacts
(MediaPipe overlay MP4s; WiLoR skeletons projected from the validated raw
NPZ with the frozen pinhole model) into
`/home/hatim/graduation-project-runs/task003b_visual/` — generated files,
not committed.

| Clip / frame | Aspect | MediaPipe | WiLoR |
|---|---|---|---|
| `s02_0171` f30 — easy dual hand | placement, fingers, thumb, L/R | **GOOD** — both hands, tight alignment, clear finger fan and thumb | **GOOD** — both hands, plausible fan and thumb |
| `s01_0174` f20 — occlusion | occlusion handling, L/R stability | **ACCEPTABLE** — one hand kept; loses second hand ~6 frames earlier; label flickers `Right→Left→Right` on a stationary hand at f21 | **ACCEPTABLE** — one hand kept through the overlap; label stable across the whole segment |
| `s02_0176` f39 — spurious/fast motion | false positives, motion | **POOR** — fast upper hand missed entirely; lower hand good | **ACCEPTABLE** — both hands found, but a duplicate third detection with visibly degenerate splayed geometry on the blurred hand |
| `s03_0173` f20 — high motion / foreshortening | foreshortening, placement | **GOOD** — both hands, foreshortened hand plausible | **GOOD** — both hands, resting hand notably clean |
| `s02_0172` f18 — MediaPipe-difficult | dual-hand under blur | **POOR** — misses the blurred second hand; the detected hand renders small/compressed | **GOOD** — both hands with plausible fanning |
| `s02_0175` f24 — MediaPipe-difficult | dual-hand, frame edge | **POOR** — the resting hand is plainly visible and returned nothing | **ACCEPTABLE** — finds both, but the low hand sits at the frame edge and its geometry looks splayed/uncertain |
| `s03_0174` f19 — MediaPipe-difficult | hand crossing | **ACCEPTABLE** — both present in this frame but heavily overlapping | **GOOD** — both present and cleanly separated |

Summary of qualitative impressions: on easy, well-separated two-hand frames
the two are comparable and both good. The divergence appears under motion
blur, low/edge hand positions and hand-hand proximity, where MediaPipe tends
to fail by **silent omission** and WiLoR tends to fail by **duplication with
degenerate geometry**. WiLoR's reconstruction quality is not uniformly
better — its worst observed geometry (blurred hand, frame-edge hand) is
visibly poor — but it produces *something* to inspect and gate, where
MediaPipe produces nothing.

# Failure Modes

| Failure | System | Observed frequency (894 frames) | Severity for simulated-glove use |
|---|---|---|---|
| Silent missing hand (visible hand not returned) | MediaPipe | 143 frames lack both hands; 6 lack any | **High** — creates gaps in the sensor stream requiring fabricated data of up to 0.90 s |
| Long single-channel dropout | MediaPipe | left streak 18, right streak 27 | **High** — a whole glove channel goes dark mid-sign |
| Duplicate handedness label | MediaPipe | 32 events (9 L + 23 R) | **High** — misassigns a sample to the wrong virtual glove; worse than an honest gap |
| Handedness flicker on a static hand | MediaPipe | observed at `s01_0174` f21 | **Medium** — identity churn for the tracking stage |
| Frame-to-frame bone-length drift | MediaPipe | CV 0.187 (0.135 per-hand), all 18 clips | **High** — propagates into every derived joint angle |
| Detector false positive / third hand | WiLoR | 1 frame | **Low** — trivially detectable (`>2 valid hands`), gateable later |
| Degenerate reconstruction on a poor crop | WiLoR | observed at `s02_0176` f39; edge hand at `s02_0175` f24 | **Medium** — plausible-looking but wrong geometry is harder to detect than a gap; needs a plausibility check |
| True hand-hand occlusion | both | `s01_0174`, 10 frames (WiLoR) / 12+ (MediaPipe) | **Low/unavoidable** — no monocular method can recover a fully hidden hand |
| Handedness disagreement under occlusion | both | `s01_0174` frames 17–26 | **Medium** — unresolved without ground truth; affects identity, not presence |
| Throughput cost | WiLoR | 3.32 FPS vs 35.28 FPS | **Low for this stage** — offline generation is acceptable |

The asymmetry that matters: MediaPipe's dominant failures are **omissions and
mislabels**, which are either invisible or actively misleading downstream;
WiLoR's dominant failures are **over-detections and locally poor geometry**,
which remain visible in the data and can be quality-gated later. That said,
WiLoR's degenerate-geometry mode is *not* free — it is plausible-looking bad
data, and a future QA gate is required. Gating is not implemented here.

# Performance

Matched software boundary (decode + preprocessing + inference/reconstruction
+ in-memory pose conversion; excludes model loading, serialization, overlay
generation, video encoding, reporting):

| System | Hardware | Frames | Seconds | Frame-weighted FPS |
|---|---|---:|---:|---:|
| MediaPipe | AMD Ryzen 7 6800HS, **CPU delegate** | 894 | 25.337 | **35.284** |
| WiLoR | RTX 3050 Laptop 4 GB, **CUDA**, normal FP32 | 894 | 269.138 | **3.322** |

MediaPipe achieves approximately **10.6× higher throughput under the
validated configurations and hardware used**. This is explicitly *not* a
hardware-normalized efficiency claim: MediaPipe = CPU, WiLoR = GPU.

## Concrete offline cost

| Workload (30 fps source) | Frames | MediaPipe (CPU) | WiLoR (GPU) |
|---|---:|---:|---:|
| This 18-clip pilot | 894 | 25.3 s | 4.5 min |
| 1 minute of video | 1,800 | 51.0 s | 9.0 min |
| 10 minutes | 18,000 | 8.5 min | 1.51 h |
| 1 hour | 108,000 | 51.0 min | 9.03 h |

For the immediate objective — **offline virtual-glove dataset generation** —
these numbers are workable. A dataset on the order of a few thousand KArSL
clips (roughly 1–2 hours of video) is a single overnight WiLoR run on this
existing 4 GB laptop GPU. Camera-rate operation is not a requirement of this
stage.

## 4 GB GPU viability and headroom

WiLoR's regenerated full run peaked at ~2,819 MiB allocated / ~2,905 MiB
reserved on a 4,096 MiB card — **70.9% reserved, ~1,191 MiB headroom**, with
zero OOM events across all 18 videos. It fits, but without much room for a
larger batch or a second concurrent job. TASK-002 measured the official
`--fast` FP16 path at ~1,455–1,473 MiB (≈48% lower) with identical both-hand
coverage on its 3-clip subset, so a future optimization path exists; its
throughput benefit was inconsistent on short clips due to `torch.compile`
recompilation and is not assumed here.

MediaPipe requires no GPU at all, which is a genuine operational advantage
for a preview/debug role.

# Weighted Scorecard

Weights are fixed by the project's stated purpose (virtual smart-glove data
generation) and were **set before scoring and not altered afterwards**.
Scores are 0–10.

| # | Category | Weight | MediaPipe | WiLoR | Justification |
|---|---|---:|---:|---:|---|
| 1 | Dual-hand reconstruction completeness | 25% | **6.0** | **9.5** | 84.004% vs 98.881% both-hand; 143 vs 10 frames lacking both hands; 8/18 vs 17/18 clips at 100%. WiLoR not 10 because its threshold is lower (0.3 vs 0.5) and its residual loss is unverified against ground truth. |
| 2 | Geometric / anatomical stability | 20% | **3.0** | **9.0** | Common 18-edge bone CV 0.1872 vs 0.0059 (31.9×), surviving per-hand separation (0.1347 vs 0.0043) and present in all 18 clips. WiLoR not 10 because the advantage is structural (MANO shape space), not proven absolute accuracy. |
| 3 | Downstream virtual-glove / kinematics suitability | 20% | **5.0** | **9.0** | WiLoR adds 15 joint rotations, global wrist orientation, 10 shape betas, camera translation and a 778-vertex mesh on an explicit kinematic tree; MediaPipe supplies positions only and no wrist orientation. Neither provides glove angles directly. |
| 4 | Temporal continuity | 15% | **4.0** | **8.5** | Worst channel gap 27 frames (0.90 s) vs 10 frames (0.33 s); no-hand streak 6 vs 0; normalized temporal metric 0.1093 vs 0.0760, WiLoR smoother in 15/18 clips. WiLoR not 9+ because it loses 3 clips, and the metric is coverage-biased in MediaPipe's favour. |
| 5 | Failure robustness | 10% | **5.0** | **8.0** | MediaPipe: 32 duplicate labels, label flicker, silent omissions. WiLoR: 1 duplicate, 1 extra-hand frame, 0 no-hand frames, 0 suspected swaps — but it does emit plausible-looking degenerate geometry on poor crops, which needs a future gate. |
| 6 | Performance / compute cost | 7% | **9.0** | **4.0** | 35.28 FPS CPU vs 3.32 FPS GPU (≈10.6× under the validated configs); 1 h of video = 51 min vs 9.03 h. WiLoR not lower because offline batch generation is the actual requirement and it fits in 4 GB. |
| 7 | Implementation / reproducibility burden | 3% | **9.5** | **4.5** | MediaPipe: one pip package + 7.5 MB model, CPU-only, no licence gate. WiLoR: 2.56 GB checkpoint, CUDA stack, three documented dependency workarounds, and gated non-commercial MANO/CC-BY-NC-ND licensing. |

| | MediaPipe | WiLoR |
|---|---:|---:|
| **Weighted total** | **5.115 / 10** | **8.465 / 10** |

# Sensitivity Analysis

| Scenario | Weighting | MediaPipe | WiLoR | Winner |
|---|---|---:|---:|---|
| Baseline | 25/20/20/15/10/7/3 | 5.115 | **8.465** | WiLoR (+3.350) |
| **A — performance-sensitive** | perf 30%, impl 10%, completeness 20%, stability 12%, kinematics 12%, temporal 10%, robustness 6% | 6.510 | **7.040** | WiLoR (+0.530) |
| **B — reconstruction-quality-sensitive** | completeness 30%, stability 25%, kinematics 25%, temporal 12%, robustness 5%, perf 2%, impl 1% | 4.805 | **8.895** | WiLoR (+4.090) |
| C — equal weights (1/7 each) | uniform | 5.929 | **7.500** | WiLoR (+1.571) |

**Break-even:** holding the relative proportions of the five quality
categories fixed, MediaPipe only overtakes WiLoR when the performance
category alone is weighted at **≥ 44% of the entire decision** — i.e. only
if raw throughput matters more than dual-hand completeness, geometric
stability, kinematics suitability, temporal continuity and failure
robustness *combined*. That is not a defensible weighting for an offline
virtual-glove dataset-generation stage.

**The winner is therefore robust across every reasonable weighting tested,
including one deliberately biased towards performance.** WiLoR does not win
under only one carefully chosen weighting.

# Hybrid Assessment

Evaluated, not implemented.

Keeping MediaPipe as a **secondary utility** adds meaningful value at close
to zero cost:

- it is already implemented, frozen, tested and reproducible;
- it needs **no GPU** and runs ~10.6× faster, making it suitable for rapid
  preview, smoke-testing new clips, and an interactive/real-time visualizer
  where a coarse skeleton is sufficient;
- it is an **independent cross-check signal**. The `s01_0174` analysis showed
  the two systems disagreeing on handedness for the same surviving hand;
  a cheap second opinion is a genuinely useful flag for the future tracking
  stage (TASK-004), even though neither is ground truth;
- it is a fallback if the CUDA environment or the gated MANO asset becomes
  unavailable — a real risk given MANO's manual licence gate.

What it should **not** be: part of the virtual-glove data path. Mixing
extractors inside one training dataset would mix coordinate systems, bone
stability characteristics and coverage profiles, which would inject exactly
the inconsistency the sensor simulation must avoid.

So the recommended arrangement is a single-primary pipeline with a retained
diagnostic tool, not a co-primary hybrid.

# Limitations

- **No ground truth.** There are no visibility, identity, handedness, thumb
  correctness or joint-angle annotations in this pilot. Every qualitative
  claim is a human judgement and every "miss vs occlusion" attribution is an
  inference from trajectory and imagery.
- **Thresholds are not normalized** (0.5 vs 0.3). Part of the coverage gap is
  configuration-driven and was intentionally not tuned away.
- **Compute is not normalized** (CPU vs GPU). Throughput figures are
  practical, not efficiency-normalized.
- **The bone-CV advantage is structural.** It measures self-consistency of
  the inferred skeleton, not anatomical accuracy; MANO enforces it by
  construction.
- **The normalized temporal metric is coverage-biased**, favouring whichever
  system returns fewer hard frames — which flatters MediaPipe here.
- **Small pilot**: 18 clips, 894 frames, 6 signs, 3 signers, one repetition
  each, single recording setup. Conclusions may not transfer to other
  capture conditions.
- **Absolute 3D quantities were never compared** across systems, so nothing
  here speaks to metric accuracy of either reconstruction.
- **The handedness disagreement under occlusion is unresolved** and is a real
  open risk for the tracking stage.
- WiLoR carries **non-commercial licensing** (CC-BY-NC-ND model, gated MANO,
  AGPL Ultralytics). Acceptable for this academic project; it would need
  revisiting for any commercial deployment.

# Final Decision

```text
PRIMARY POSE PIPELINE: WILOR
```

```text
PRIMARY OFFLINE VIRTUAL-GLOVE DATA EXTRACTOR:
  WiLoR + MANO (official implementation, normal/FP32, frozen commit 20e83af)

SECONDARY FAST/DEBUG EXTRACTOR:
  MediaPipe Hand Landmarker (frozen commit ed25d9f) — retained for fast
  CPU-only preview, smoke tests, interactive visualization and as an
  independent cross-check signal. NOT part of the virtual-glove data path.
```

The project owner's stated preference for WiLoR was not treated as evidence.
The decision follows from the harmonized metrics, the raw visual evidence,
the representation analysis and the weighting robustness check; had the
evidence favoured MediaPipe, the scorecard would have said so, and under a
sufficiently performance-dominated weighting (≥44%) it does.

# Recommended Architecture

```text
KArSL RGB video
  -> WiLoR (official YOLO detector + ViT + transformer refinement + MANO/SMPL-X)
       normal FP32, detector confidence 0.3, batch = 1 frame, CUDA
  -> raw immutable per-video NPZ
       21 joints + 15 joint rotations + global orientation + 10 betas
       + camera translation + 778-vertex mesh + handedness + confidence
  -> [TASK-004] temporal dual-hand tracking / stable identity
  -> [TASK-005] hand kinematics
  -> [TASK-006] virtual smart-glove sensor representation
  -> ... LSTM / NLP / TTS

MediaPipe Hand Landmarker (CPU, secondary)
  -> fast preview, smoke tests, visualizer, independent cross-check
  -> never mixed into the virtual-glove training data
```

Carried-forward obligations for the next stage: a quality gate for WiLoR's
extra-hand and degenerate-crop failure modes; an explicit policy for the 10
genuinely occluded frames; and resolution of the occlusion handedness
ambiguity in tracking rather than in extraction.

# Next Milestone

```text
TASK-004 — Temporal Dual-Hand Tracking / Stable Identity
```

Not started in this task. Subsequent conceptual stages remain TASK-005 hand
kinematics, TASK-006 virtual smart-glove sensor representation, TASK-007
visualizer, TASK-008 virtual sensor dataset generation, TASK-009 LSTM Arabic
sign recognition, TASK-010 Arabic NLP, TASK-011 Arabic TTS. Numbering may be
revised.

# Reproducibility

- Comparison branch `evaluation/mediapipe-vs-wilor`, created from
  `63c7e683eeab19624a00480b9e0525e25ca07c44`, worked in
  `../Graduation-Project-Simulation-comparison`.
- Frozen inputs: MediaPipe `ed25d9f2814493f02e16848d23c3466b54f06d6e`,
  WiLoR `20e83afd7a54493523389fe02ca7077b1afc5866`, audit `66192c0`,
  remediation `63c7e683`.
- Manifest SHA-256
  `4a8e0d24f587d9c54ad87ebd896c6df0290b3ebc80885774100d71b29dbc3c0c`,
  18 samples, 894 frames.
- No model was re-run for scoring; the validated frozen artifacts were used.
  The WiLoR run was independently re-validated (mode, rows, finiteness,
  MANO fields, mesh) before use.
- Machine-readable outcome: `reports/evaluation/TASK-003B-scorecard.json`.
- Side-by-side review frames were regenerated with
  `scripts/render_task003b_side_by_side.py` into
  `/home/hatim/graduation-project-runs/task003b_visual/` (ignored, not
  committed). That script performs visualization only and computes no
  comparison metric.
- Verification commands:

```bash
python -m unittest discover -s tests -p 'test_*.py'
python -m compileall -q evaluation scripts tests
```

- No MP4, NPZ, OBJ, mesh, checkpoint, MANO asset, KArSL video or cache file
  is tracked by this branch.
