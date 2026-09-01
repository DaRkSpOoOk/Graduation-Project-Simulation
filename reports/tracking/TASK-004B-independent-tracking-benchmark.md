# Task

TASK-004B — Independent Hand Identity / Visibility Benchmark.

# Objective

Create a small, human-authored reference set for evaluating the later TASK-004A temporal hand tracker. The reference records physical subject-anatomical LEFT/RIGHT identity, visibility, ambiguity, and selected image-space points from the original KArSL RGB videos. It is an evaluation protocol and annotation set only; it is not a tracker and does not produce pose, kinematics, or sensor data.

# Branch

`luna/task-004b-tracking-benchmark`, based on the newly integrated `main` at the start of this task.

# Scope

This benchmark covers eight clips and 399 decoded frames from the existing deterministic KArSL Milestone-1 pilot. Six required challenge clips and two controls are annotated frame by frame. No model output, WiLoR label, MediaPipe label, temporal tracker, smoothing, interpolation, or pose prediction was used to create the reference file.

The source videos remain outside Git under the ignored dataset directory. Only the compact annotation, validation code, source-only review renderer, tests, and this report are committed.

# Why Independent Annotation Is Needed

TASK-003B established that the RGB pilot does not contain ground truth for physical hand identity, visibility, occlusion, handedness correctness, or difficult crossing regions. A tracker cannot be evaluated fairly against its own predictions. The selected frames were reviewed from the original RGB videos, and the reference annotations were locked before any TASK-004A tracker outputs were inspected. The annotation file contains no model-prediction fields.

# Approach

The benchmark was created in an isolated worktree from `origin/main`. The two control clips were selected before viewing any TASK-004A output. The original source video was decoded in frame order and every decoded frame received one row. Frame indices are zero-based. Physical identity means the signer’s anatomical left or right hand, not the image-side label and not a detector-provided handedness label.

Approximate normalized image-space points were recorded for challenge clips when a wrist or hand centre was sufficiently visible. They are references for later identity/continuity checks, not pixel-perfect keypoint ground truth. Controls intentionally omit points because their primary purpose is bilateral visibility and continuity control; blank coordinates are valid under the protocol.

# Selected Clips

The selected IDs are a fixed subset of `datasets/manifests/karsl_milestone1_pilot.csv`. The six required challenge clips were included directly. The controls were selected from the same pilot before viewing TASK-004A output: `s02/sign0171` and `s01/sign0171`.

| sample_id | role | frames | signer | sign / label | selection rationale |
|---|---:|---:|---:|---|---|
| `karsl_test_s01_sign0174_repfirst` | challenge | 34 | 01 | 0174 / love | Required convergence and sustained hand-over-hand occlusion. |
| `karsl_test_s02_sign0172_repfirst` | challenge | 36 | 02 | 0172 / break | Required close-contact movement with motion blur during convergence. |
| `karsl_test_s02_sign0175_repfirst` | challenge | 48 | 02 | 0175 / hate | Required asymmetric visibility and image-edge proximity. |
| `karsl_test_s02_sign0176_repfirst` | challenge | 57 | 02 | 0176 / grill | Required extra-detection stress case; frame 39 was explicitly reviewed for physical hand count. |
| `karsl_test_s03_sign0173_repfirst` | challenge | 38 | 03 | 0173 / walk | Required extended-hand and frame-edge case. |
| `karsl_test_s03_sign0174_repfirst` | challenge | 38 | 03 | 0174 / love | Required prolonged central crossing and hand-hand occlusion. |
| `karsl_test_s02_sign0171_repfirst` | control | 67 | 02 | 0171 / build | Preselected clean bilateral control with both physical hands visible. |
| `karsl_test_s01_sign0171_repfirst` | control | 81 | 01 | 0171 / build | Preselected bilateral control with later stable hand contact. |

The selected set totals eight videos and 399 frames. The shared manifest supplies the exact archive member, local relative path, expected 30 FPS / 1920×1080 metadata, and source checksum for each clip.

# Annotation Protocol

Each CSV row represents exactly one `(sample_id, frame_index)` pair. For each physical side, the annotator recorded one visibility state. A point is present only as an approximate wrist centre, or a hand-centre substitute when the wrist is not visible. Points use normalized coordinates: `x = pixel_x / (width - 1)` and `y = pixel_y / (height - 1)`, constrained to `[0, 1]`. Coordinates are blank when the hand is fully hidden, ambiguous, or not useful as a stable reference.

Scene flags are frame-level observations and may include `HAND_CROSSING`, `HAND_HAND_OCCLUSION`, `MOTION_BLUR`, `FRAME_EDGE`, `SELF_OCCLUSION`, and `IDENTITY_AMBIGUOUS`. Notes explain the source-frame judgment, especially in challenge windows. `annotator_confidence` describes confidence in the human reference, not model confidence.

# Visibility States

The allowed state vocabulary is:

- `VISIBLE`: the physical hand is visibly observable enough for the side judgment.
- `PARTIALLY_OCCLUDED`: some physical hand evidence is visible, but overlap, self-occlusion, blur, or an object/arm prevents a fully clear view.
- `FULLY_OCCLUDED`: the side remains physically inferred from continuity/context, but no usable hand pixels are visible in the frame.
- `OUT_OF_FRAME`: the hand is judged to be outside the image bounds.
- `AMBIGUOUS`: the RGB evidence does not support a sufficiently reliable side or visibility judgment.

A tracker is not expected to reconstruct a fully occluded hand. For later scoring, an explicit missing or likely-occluded output may be preferable to an invented pose.

# Identity Convention

LEFT and RIGHT are anatomical identities of the signer. They are not image-left/image-right and are not copied from MediaPipe or WiLoR. During the key s01/sign0174 overlap, the foreground hand attached to the viewer-right forearm remains identifiable as the subject-anatomical LEFT hand. The subject RIGHT hand is covered behind it from frames 16–33; the annotation does not treat a detector label as evidence that the hidden hand is visible.

# Ambiguity Policy

Ambiguity is retained rather than forced into a binary answer. Strict identity scoring should use only HIGH/MEDIUM-confidence frames and should exclude frames with `AMBIGUOUS` identity. LOW-confidence and ambiguous rows remain in the file for qualitative analysis and for testing whether a quality gate expresses uncertainty.

An annotation revision must cite the source-frame evidence and include an explicit revision note. Disagreement with a future tracker is not, by itself, a reason to revise the reference.

# Challenge Windows

- `karsl_test_s01_sign0174_repfirst`, frames 11–33: hands converge and overlap. Frames 11–15 retain LEFT as partially visible while RIGHT is `AMBIGUOUS`; frames 16–33 identify the viewer-right-forearm hand as LEFT and mark RIGHT `FULLY_OCCLUDED`.
- `karsl_test_s02_sign0172_repfirst`, frames 0–23: approach/contact interval; frames 0–7 also carry `MOTION_BLUR`.
- `karsl_test_s02_sign0175_repfirst`, frames 0–47: the low subject RIGHT hand remains present while subject LEFT moves toward the image edge; frames 8–12 carry `MOTION_BLUR`.
- `karsl_test_s02_sign0176_repfirst`, frames 30–45, especially frame 39: exactly two physical hands are visible. The upper subject LEFT hand is motion-blurred around the reviewed interval; no third physical hand exists. The annotation does not encode a model-generated extra detection.
- `karsl_test_s03_sign0173_repfirst`, frames 0–37: extended subject LEFT hand is close to the right image edge; frames 11–14 carry `MOTION_BLUR` and frames 18–37 carry `FRAME_EDGE`.
- `karsl_test_s03_sign0174_repfirst`, frames 8–37: central crossing/hand-hand occlusion. Frames 8–12 mark the subject RIGHT side ambiguous; later frames retain a partial RIGHT reference where the RGB evidence supports it.
- The two controls are retained over their full frame ranges and establish bilateral visibility without selecting them in response to tracker output.

# Benchmark Metrics

The annotation is a contract for a later extractor-neutral TASK-004A evaluation. The following metrics are defined before tracker evaluation:

1. **Visibility accuracy.** For a reference side in `VISIBLE` or `PARTIALLY_OCCLUDED`, measure whether the tracker returns a corresponding side/track. Report separately for visible and partially occluded reference frames. Fully occluded and out-of-frame references do not require a returned pose.
2. **False-presence rate.** On frames where a physical side is not visibly observable (`FULLY_OCCLUDED`, `OUT_OF_FRAME`, or an explicitly absent physical hand), measure whether the tracker emits/accepts a physical identity for that side. A missing or `LIKELY_OCCLUDED` state is not a false positive.
3. **Identity continuity.** On HIGH/MEDIUM-confidence, non-ambiguous consecutive frames, check whether the returned identity remains anatomical LEFT/RIGHT. Use the normalized reference representation; do not compare old model-specific handedness-instability counts.
4. **Reacquisition accuracy.** After a temporary disappearance or occlusion, evaluate the first sufficiently confident return against the human side reference. Fully hidden intervals are not scored as reconstruction failures.
5. **Identity-switch count.** Count only switches supported by HIGH/MEDIUM-confidence human identity references. A suspected switch is not ground truth when the reference is ambiguous.
6. **Quality-gate usefulness.** On known blur, crossing, frame-edge, partial/full occlusion, ambiguous, and extra-detection frames, measure whether the tracker rejects or marks uncertainty instead of silently accepting bad geometry or an extra physical identity.

The protocol does not require a tracker to hallucinate a fully occluded hand. Metrics should report denominators and confidence exclusions explicitly. The `s02/sign0176` frame-39 annotation is a two-physical-hand/no-third-hand reference for detecting extra accepted identities.

# Evidence / Sources

- Original KArSL RGB source clips selected through the repository’s shared manifest: [`datasets/manifests/karsl_milestone1_pilot.csv`](../../datasets/manifests/karsl_milestone1_pilot.csv).
- [Official KArSL dataset page](https://hamzah-luqman.github.io/KArSL/).
- [Official KArSL repository](https://github.com/Hamzah-Luqman/KArSL).
- [Original KArSL research record](https://dl.acm.org/doi/10.1145/3423420).
- The manifest records the bounded Google Drive archive identifiers and exact archive members used for the pilot. At annotation time the local source root was `/home/hatim/Graduation-Project-Simulation/datasets/raw/karsl_milestone1_pilot`; the videos are ignored and are not part of this branch.
- The project has permission to obtain data from SSHI (`https://sshi.sa/`). No SSHI scraper or SSHI data was used in this benchmark; access can be investigated in a later dedicated task.

# Annotation Statistics

The committed CSV contains 399 frame rows across eight videos. The state counts below count hand labels, while flag counts count frames carrying each flag.

| quantity | count |
|---|---:|
| annotated videos | 8 |
| annotated frames | 399 |
| visible LEFT labels | 371 |
| partially occluded LEFT labels | 28 |
| visible RIGHT labels | 328 |
| partially occluded RIGHT labels | 43 |
| fully occluded RIGHT labels | 18 |
| ambiguous RIGHT labels | 10 |
| partially occluded hand labels, both sides | 71 |
| fully occluded hand labels, both sides | 18 |
| frames with `HAND_CROSSING` | 124 |
| frames with `HAND_HAND_OCCLUSION` | 71 |
| frames with `FRAME_EDGE` | 68 |
| frames with `MOTION_BLUR` | 26 |
| frames with `IDENTITY_AMBIGUOUS` | 10 |
| HIGH / MEDIUM / LOW confidence rows | 215 / 154 / 30 |

There were no `OUT_OF_FRAME` labels in this focused set. A `SELF_OCCLUSION` flag was not needed; the difficult hidden-hand evidence was represented with `HAND_HAND_OCCLUSION` and the explicit visibility states.

## Per-video annotation counts

| sample_id | frames | LEFT states | RIGHT states | key flags |
|---|---:|---|---|---|
| `karsl_test_s01_sign0174_repfirst` | 34 | 27 VISIBLE, 7 PARTIALLY_OCCLUDED | 9 VISIBLE, 2 PARTIALLY_OCCLUDED, 18 FULLY_OCCLUDED, 5 AMBIGUOUS | crossing/hand-hand occlusion; LOW-confidence overlap |
| `karsl_test_s02_sign0172_repfirst` | 36 | 20 VISIBLE, 16 PARTIALLY_OCCLUDED | 20 VISIBLE, 16 PARTIALLY_OCCLUDED | contact; motion blur |
| `karsl_test_s02_sign0175_repfirst` | 48 | 48 VISIBLE | 48 VISIBLE | frame edge; motion blur |
| `karsl_test_s02_sign0176_repfirst` | 57 | 57 VISIBLE | 57 VISIBLE | frame 39 two physical hands, no third |
| `karsl_test_s03_sign0173_repfirst` | 38 | 38 VISIBLE | 38 VISIBLE | frame edge; motion blur |
| `karsl_test_s03_sign0174_repfirst` | 38 | 33 VISIBLE, 5 PARTIALLY_OCCLUDED | 8 VISIBLE, 25 PARTIALLY_OCCLUDED, 5 AMBIGUOUS | crossing/hand-hand occlusion |
| `karsl_test_s02_sign0171_repfirst` | 67 | 67 VISIBLE | 67 VISIBLE | control |
| `karsl_test_s01_sign0171_repfirst` | 81 | 81 VISIBLE | 81 VISIBLE | control hand contact |

# Evaluation

No tracker was evaluated in TASK-004B. This report defines the reference data and future scoring rules only. TASK-004A outputs were intentionally not inspected or scored.

# Results

The independent benchmark was created and schema-validated. Its result is a locked eight-video, 399-frame human reference set, not a pose-model result.

# Tests

- `python scripts/validate_task004b_annotations.py` — passed; validated 399 rows and printed the expected eight-video statistics.
- `python -m unittest tests.test_task004b_annotations` — passed, 9 tests.
- `python -m unittest discover -s tests -p 'test_*.py'` — passed, 66 tests.
- `python -m compileall -q evaluation scripts tests` — passed.
- The source-only review renderer decoded all 57 frames of `karsl_test_s02_sign0176_repfirst` and wrote 57 temporary JPEG review frames outside Git.

# Files Changed

- `evaluation/annotations/task004_hand_identity_visibility.csv` — frame-level human reference annotations.
- `evaluation/annotations/task004b.py` — extractor-neutral schema, locked clip contract, validation, manifest checks, and statistics.
- `evaluation/annotations/__init__.py` — package exports.
- `scripts/validate_task004b_annotations.py` — command-line schema/manifest validator and statistics printer.
- `scripts/render_task004b_review.py` — optional source-only annotation review renderer; it does not import or run a pose model or tracker.
- `tests/test_task004b_annotations.py` — synthetic and committed-file validation tests.
- `reports/tracking/TASK-004B-independent-tracking-benchmark.md` — this protocol and report.

# How to Run

From the repository root:

```bash
python scripts/validate_task004b_annotations.py
```

The validator checks the locked selected IDs, complete frame coverage, visibility/confidence/flag enums, duplicate frame keys, coordinate pairs/ranges, and membership in the shared KArSL manifest.

To render human markers from the original RGB videos, without any model execution:

```bash
python scripts/render_task004b_review.py \
  --source-root /home/hatim/Graduation-Project-Simulation \
  --output-dir /tmp/task004b_review \
  --sample-id karsl_test_s02_sign0176_repfirst
```

The output directory is for review only and must remain outside Git. Omit `--sample-id` to render all eight clips.

# Failures / Limitations

- Monocular RGB does not provide perfect physical identity evidence during complete overlap. The s01/sign0174 frames 11–15 and s03/sign0174 frames 8–12 are explicitly ambiguous on the hidden/transitioning side rather than guessed.
- Approximate points are not pixel-level keypoint ground truth and are absent from the controls by design.
- There is one human annotator in this initial pilot; inter-rater agreement has not been measured.
- The benchmark is deliberately small and focused. It does not represent the full KArSL distribution.
- No quantitative tracker score, model comparison, or quality-gate ranking is claimed here.
- The archive source remains an external dependency. The manifest’s checksums and archive-member identifiers make the exact intended clips reproducible when access is available.

# Performance

This task performs annotation-file validation and optional JPEG review rendering only. No model runtime, inference FPS, or tracking performance was measured. The review renderer is not part of a benchmark timing scope.

# Comparison

No comparison with MediaPipe, WiLoR, or TASK-004A was performed. In particular, model labels were not used to create the annotation file, and no tracker implementation was inspected.

# Recommendation

NEEDS MORE EVALUATION

The independent reference set is ready for later tracker scoring, but tracker quality itself is intentionally unmeasured here. A future evaluation should report confidence-filtered denominators, ambiguity exclusions, and quality-gate outcomes rather than forcing hidden-hand frames into accuracy scores.

# Reproducibility

- Branch: `luna/task-004b-tracking-benchmark`.
- Selection contract: the six required challenge IDs plus the two controls listed above, all drawn from the shared Milestone-1 manifest.
- Source split: KArSL-502 official `test` RGB pilot entries; exact archive members, local paths, expected metadata, and SHA-256 values are in the manifest.
- Frame convention: zero-based decoded frame index, one CSV row per source frame; the locked frame counts are 34, 36, 48, 57, 38, 38, 67, and 81, totaling 399.
- Annotation convention: subject-anatomical LEFT/RIGHT, normalized approximate wrist/hand-centre points, explicit visibility and ambiguity states, and human confidence.
- Independence statement: annotation was created independently of TASK-004A tracker predictions and was locked before any such outputs were used. Any future correction must document source-frame evidence and a revision reason.
- Validation command: `python scripts/validate_task004b_annotations.py`.

# Next Steps

TASK-004A may later evaluate temporal identity, visibility, reacquisition, identity switches, extra-hand rejection, and uncertainty behavior against this file. A second annotator and adjudication pass would strengthen the benchmark, especially for the low-confidence overlap windows. TASK-004B does not start TASK-004A and does not implement any downstream pose or sensor stage.
