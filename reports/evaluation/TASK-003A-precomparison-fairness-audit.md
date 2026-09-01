# Task

Neutral pre-comparison fairness audit of the two completed Milestone-1 pose
experiments (TASK-001 MediaPipe, TASK-002 WiLoR) before any formal
comparison is run.

The question answered here is **not** "which model wins" — no winner is
selected, no model is tuned, and no experimental branch is modified. The
question is: **are the two experiments measuring equivalent conditions with
equivalent metrics?**

# Branches Reviewed

| Branch | Commit | Worktree used (read-only) |
|---|---|---|
| `luna/mediapipe-karsl-pilot` | `ed25d9f` | `../Graduation-Project-Simulation-luna` (detached) |
| `opus/wilor-karsl-pilot` | `20e83af` | `../Graduation-Project-Simulation-opus` (detached) |
| `opus/task-003a-fairness-audit` (this audit) | from `main` `96dabf7` | `../Graduation-Project-Simulation-audit` |

Separate worktrees were used; neither experimental branch was checked out,
modified, merged, or re-run. The shared repository working directory was
left on its existing branch and untouched.

Artifacts inspected: both reports, `pose/common/schema.py`, both extractors
(`pose/mediapipe/extractor.py`, `pose/wilor/frame_extraction.py`,
`pose/wilor/video_processing.py`), both metric modules
(`evaluation/metrics/mediapipe_baseline.py`,
`evaluation/metrics/hand_pose_metrics.py`), both runners
(`scripts/run_mediapipe_pilot.py`, `evaluation/benchmarks/wilor_karsl_pilot.py`),
`video_io/reader.py`, `visualization/hand_overlay.py`,
`pose/wilor/visualize.py`, `configs/pose/mediapipe_karsl_pilot.json`,
both NPZ schemas, both test suites, and the surviving run artifacts under
`runs/`.

Both test suites pass independently: MediaPipe branch 7/7, WiLoR branch
29/29.

The two branches touch disjoint file sets except
`datasets/manifests/karsl_milestone1_pilot.csv`, which is byte-identical in
both (same git blob) — so a neutral combined evaluation tree is possible
without merging either branch.

# Dataset Equivalence

**Verdict: equivalent. Verified, not assumed.**

| Check | Result |
|---|---|
| Manifest path | `datasets/manifests/karsl_milestone1_pilot.csv` (present in both branches) |
| Manifest SHA-256 | `4a8e0d24f587d9c54ad87ebd896c6df0290b3ebc80885774100d71b29dbc3c0c` (identical in both) |
| Manifest git blob | `eda038471ab936ce538f11583dbe4a2d2661f71a` (identical in both) |
| Byte diff between branches | identical |
| Rows | 18 |
| Sign IDs | 0171, 0172, 0173, 0174, 0175, 0176 (6) |
| Signers | 01, 02, 03 (3) |
| Split | `test` (all rows) |
| Repetition rule | `lexicographically_first_valid_mp4` (all rows) |
| Declared FPS / resolution | 30.0 / 1920×1080 (all rows) |
| Local video checksum verification | **18/18 SHA-256 matched** the manifest against the files actually on disk |

The MediaPipe run metadata records the same manifest SHA-256
(`4a8e0d24…`), confirming the MediaPipe run consumed this exact contract.
The WiLoR report records adopting the identical manifest unchanged.

No clips were selected, replaced, or reordered by this audit.

# Frame Equivalence

**Verdict: equivalent. 894 = 894 = 894, verified per video.**

Decoded frame counts were compared three ways: OpenCV container count, the
MediaPipe raw NPZ (`frame_indices`), and the WiLoR raw NPZ (distinct
`frame_index` values).

| Sample | OpenCV | MediaPipe NPZ | WiLoR NPZ | Match |
|---|---:|---:|---:|:--:|
| karsl_test_s01_sign0171_repfirst | 81 | 81 | 81 | ✓ |
| karsl_test_s01_sign0172_repfirst | 52 | 52 | 52 | ✓ |
| karsl_test_s01_sign0173_repfirst | 51 | 51 | 51 | ✓ |
| karsl_test_s01_sign0174_repfirst | 34 | 34 | 34 | ✓ |
| karsl_test_s01_sign0175_repfirst | 41 | 41 | 41 | ✓ |
| karsl_test_s01_sign0176_repfirst | 38 | 38 | 38 | ✓ |
| karsl_test_s02_sign0171_repfirst | 67 | 67 | 67 | ✓ |
| karsl_test_s02_sign0172_repfirst | 36 | 36 | 36 | ✓ |
| karsl_test_s02_sign0173_repfirst | 44 | 44 | 44 | ✓ |
| karsl_test_s02_sign0174_repfirst | 36 | 36 | 36 | ✓ |
| karsl_test_s02_sign0175_repfirst | 48 | 48 | 48 | ✓ |
| karsl_test_s02_sign0176_repfirst | 57 | 57 | 57 | ✓ |
| karsl_test_s03_sign0171_repfirst | 68 | 68 | 68 | ✓ |
| karsl_test_s03_sign0172_repfirst | 50 | 50 | 50 | ✓ |
| karsl_test_s03_sign0173_repfirst | 38 | 38 | 38 | ✓ |
| karsl_test_s03_sign0174_repfirst | 38 | 38 | 38 | ✓ |
| karsl_test_s03_sign0175_repfirst | 56 | 56 | 56 | ✓ |
| karsl_test_s03_sign0176_repfirst | 59 | 59 | 59 | ✓ |
| **Total** | **894** | **894** | **894** | ✓ |

Zero mismatches. Neither implementation silently skips or duplicates
frames. Both reports independently list the same per-video frame counts.

(The WiLoR column above is from the surviving Phase-A NPZ set; the Phase-B
full-mode run used the identical `_iter_frames` decode path and its report
lists the identical per-video counts. See BLOCKING issue **F-1**.)

Timestamp provenance differs but is numerically equivalent on this CFR
material — see **F-10**.

# Raw-Output Equivalence

**Verdict: both evaluate genuinely raw model output. No blocking issue.**

MediaPipe (`pose/mediapipe/extractor.py`):
- Writes detector-order arrays straight from the result object; NPZ
  metadata carries an explicit
  `raw_preservation: ["detector_order", "detector_handedness", "no_smoothing", "no_interpolation", "no_identity_correction"]`.
- The LEFT/RIGHT canonical view is built **only inside the metric layer**
  (`_canonical_hands`) and is never written back over the raw arrays.
- Per-frame failures would keep already-extracted frames and mark the video
  `failed`; all 18 videos reported `success`.

WiLoR (`pose/wilor/frame_extraction.py`, `video_processing.py`,
`npz_io.py`):
- Long-format rows straight from the model forward pass; a frame whose
  extraction raises is stored explicitly as `hand_present=False` with
  `quality_flags=["extraction_failed", …]` rather than dropped or imputed.
- No smoothing, interpolation, or identity repair anywhere in the raw path.
- Phase-A and Phase-B outputs were written to **separate directories** so
  neither overwrote the other.

Neither metric module mutates the raw arrays it reads. No tracked/cleaned
representation exists in either branch, so there is nothing to exclude.

# Detection Definition Audit

**Verdict: "both-hand" agrees; per-hand coverage and missing-streak
definitions do NOT. Normalization required.**

What each implementation actually counts:

| Concept | MediaPipe (`mediapipe_baseline.py`) | WiLoR (`hand_pose_metrics.py`) |
|---|---|---|
| detected hand | one entry in `result.hand_landmarks` (capped at `num_hands=2`); `hand_present=True` per returned hand | one detector box that reached the reconstruction model; `hand_present=True` (uncapped count) |
| left hand | canonical slot 0 filled — a hand labelled `left`; on duplicate labels the higher handedness score wins the slot | any hand in the frame labelled `left` (set membership) |
| right hand | canonical slot 1 filled, same rule | any hand labelled `right` |
| both hands | `canonical_present.all(axis=1)` — a left **and** a right present | `labels == {"left","right"}` |
| no hands | no detector entry at all (`~present.any(axis=1)`) | no `hand_present` row for that frame |
| denominator | decoded frames per video | distinct `frame_index` values = decoded frames |

Two definitional divergences, both material:

1. **Inclusive vs exclusive per-hand counts.** MediaPipe reports
   `frames_with_left_hand` **including** both-hand frames (803 = 89.8%).
   WiLoR reports `frames_left_only`, which **excludes** both-hand frames
   (10 = 1.1%). Applying WiLoR's exclusive rule to MediaPipe's *own* raw
   data quantifies the gap:

   | Rule applied to identical MediaPipe raw data | left | right | both |
   |---|---:|---:|---:|
   | MediaPipe-report (inclusive) | 803 (89.821%) | 836 (93.512%) | 751 (84.004%) |
   | WiLoR-report (exclusive "only") | 52 (5.817%) | 85 (9.508%) | 751 (84.004%) |

   The same data yields an ~84-percentage-point swing in "left-hand %"
   purely from the definition. **"Both-hand %" is identical under both
   rules (751), so both-hand coverage is genuinely the same concept in the
   two reports.**

2. **Max simultaneous hands.** MediaPipe is hard-capped at 2
   (`num_hands=2`, plus `_frame_arrays` truncation) and structurally cannot
   report a third hand. WiLoR is uncapped and did emit 3 detections in one
   frame. See **F-9**.

**Required common definition for TASK-003B** (computable from both raw
formats):

- `reconstructed_hand` := `hand_present` **and** run `mode == "full"`
  **and** 21 finite 3D joints **and** non-empty MANO/world coordinates.
  A detector bbox alone is never counted (see **F-6**).
- `frames_no_hand` := frames with zero reconstructed hands.
- `frames_left_incl` := frames with ≥1 reconstructed hand labelled `left`
  (**including** both-hand frames); same for right.
- `frames_both` := frames with ≥1 `left` **and** ≥1 `right`.
- `frames_left_only` / `frames_right_only` reported additionally, so both
  conventions are visible and neither report is silently reinterpreted.
- Denominator: decoded frames (894 total), frame-weighted for aggregates.

# Handedness Convention Audit

**Verdict: conventions agree. No relabeling required.** This was verified
empirically, not assumed from documentation.

| Check | Result |
|---|---|
| Label source (MediaPipe) | `result.handedness[i][0].category_name` (`Left`/`Right`), score recorded |
| Label source (WiLoR) | YOLO detector class (`cls` 0 → left, 1 → right); MANO reconstruction itself is always right-handed and mirrored post-hoc |
| Label agreement on spatially matched hands | **740 / 751 (98.5 %)** of frames where both systems returned exactly one left + one right |
| Disagreements | 11, **all in `karsl_test_s02_sign0176_repfirst`**, at frames where the two hands overlap to within ~1–3 px in image-x — the spatial matching used for this check is degenerate there, so these are matching-heuristic artefacts, not evidence of a convention conflict |
| "right" positioned at smaller image-x (viewer-left) | MediaPipe 728/751 (96.9 %); WiLoR 841/884 (95.1 %) |

For a signer **facing a non-mirrored camera**, the anatomical right hand
appears at smaller image-x. Both systems satisfy this in ~95–97 % of
two-hand frames (the remainder being genuine mid-line hand crossings, which
are expected in sign language). Both therefore appear to label
**subject-anatomical** left/right on this non-mirrored Kinect material, and
they agree with each other.

Convention differences that exist but do **not** require normalization
here:
- WiLoR's MANO mirroring: MANO is a right-hand model; left detections are
  produced by negating output x. The TASK-002 report verified empirically
  that `MANO_LEFT.pkl`'s vertex template is bit-identical to the x-mirrored
  `MANO_RIGHT.pkl` template, so the mirror is exact. Stored
  `hand_pose_rotmat` values remain in the canonical right-hand convention
  for **both** hands — a caveat for future `hand_kinematics/`, not for this
  comparison.
- MediaPipe's documented convention assumes a mirrored (selfie) input
  image. On this dataset its labels nevertheless agree with WiLoR and with
  facing-subject anatomy, so no swap is applied. **Re-verify if capture
  geometry ever changes.**
- Duplicate-label handling differs (see **F-8**).

No model was altered to make labels match; none was needed.

# 3D Coordinate-System Audit

**Verdict: the two 3D outputs are NOT in the same coordinate system. Absolute
3D quantities must not be ranked.**

| Property | MediaPipe | WiLoR |
|---|---|---|
| 3D array | `hand_world_landmarks` (21×3) | `landmarks_3d` = `pred_keypoints_3d` (21×3) |
| Origin | hand-local, approximately wrist/hand-centre | MANO model-local; joint 0 sits within ~1e-3 model units of the local origin **by construction** |
| Units | metres (MediaPipe's metric-scale world convention) | WiLoR/MANO internal weak-perspective units tied to `EXTRA.FOCAL_LENGTH=5000` and the MANO template scale — **not calibrated to millimetres in this pilot** |
| Global placement | not provided | separate `mano_references.camera_translation_xyz` (camera-space; tz ≈ 37 model units in observed frames) |
| Consequence | world landmarks are hand-relative; MediaPipe has no camera-space translation | WiLoR's global motion lives in a *different array* from its joints |

Because MediaPipe's "wrist displacement" is computed on hand-relative world
landmarks while WiLoR's is computed on camera-space translation, the two
"wrist jitter" numbers describe **different physical quantities in different
units**, in addition to being different mathematical operators
(second difference vs first difference — **F-12**).

**Safe to compare directly** (unit-free or frame-counting):
frame coverage, missingness, hand counts, missing streaks,
handedness/identity instability indicators, normalized bone-length
coefficient of variation, runtime/FPS (with the scope + hardware caveats
below).

**Must NOT be directly ranked** (no conversion is invented here):
absolute 3D wrist displacement, absolute jitter distance, absolute depth,
camera-space translation, mesh/hand scale, any metric in metres vs model
units.

# Joint-Topology Audit

**Verdict: joint *indexing* is equivalent; bone *edge sets* are not.**

Both produce a 21-point skeleton in the same index order — 0 wrist,
1–4 thumb, 5–8 index, 9–12 middle, 13–16 ring, 17–20 pinky. WiLoR reaches
this ordering through `wilor/models/mano_wrapper.py`'s
`mano_to_openpose = [0,13,14,15,16,1,2,3,17,4,5,6,18,10,11,12,19,7,8,9,20]`
remap; MediaPipe uses it natively. **No index remapping is required.**

The bone sets used for bone-length CV differ:

| | Edges | Detail |
|---|---:|---|
| MediaPipe (`HAND_CONNECTIONS`) | 21 | includes the inter-MCP palm chain `(5,9),(9,13),(13,17)` and `(0,17)` |
| WiLoR (`HAND_BONES`) | 20 | includes wrist-radiating `(0,9),(0,13)` instead of the palm chain |
| Shared | **18** | all phalanx chains, `(0,1)`, `(0,5)`, `(0,17)` |
| MediaPipe-only | 3 | `(5,9)`, `(9,13)`, `(13,17)` |
| WiLoR-only | 2 | `(0,9)`, `(0,13)` |

Minimum neutral mapping needed: **recompute bone-length CV for both systems
over the 18-edge intersection**, in the neutral evaluation layer only. Raw
joint values must not be modified. Because CV is scale-free (std/mean), it
remains valid across the two coordinate systems once the edge set matches.

# Runtime/Hardware Audit

**Verdict: no matched timing pair exists today. Runtime is comparable only
with explicit scope labels, or after a matched re-measurement.**

| Timed component | MediaPipe `runtime_seconds` | MediaPipe `inference_seconds` | WiLoR `inference_seconds` |
|---|:--:|:--:|:--:|
| model/landmarker construction | included (per video) | — | excluded (loaded once per run) |
| video decode | included | — | **included** |
| colour conversion / preprocessing | included | — | included |
| model inference | included | **only this** | included |
| output → array conversion | included | — | included |
| overlay render + video write | **included** | — | **excluded** |
| NPZ serialization | excluded | — | excluded |

Published numbers and their aggregation:

| Number | Value | Aggregation | Hardware |
|---|---:|---|---|
| MediaPipe effective end-to-end FPS | 21.865 | frame-weighted (894 / 40.887 s) | CPU delegate |
| MediaPipe inference-only FPS | 48.880 | frame-weighted (894 / 18.290 s) | CPU delegate |
| WiLoR effective FPS (report) | 4.18 | **mean of per-video FPS** | CUDA GPU |
| WiLoR effective FPS (frame-weighted) | 4.00 | 894 / 223.51 s (derived here) | CUDA GPU |

Two problems: (a) MediaPipe's end-to-end figure is *depressed* by
1920×1080 overlay rendering that WiLoR's figure excludes, and MediaPipe's
inference-only figure is a *narrower* scope than anything WiLoR published —
so neither pairing is apples-to-apples; (b) aggregation differs (**F-7**).

Hardware, recorded and non-identical by design:

| | MediaPipe | WiLoR |
|---|---|---|
| Execution device | **CPU delegate** (AMD Ryzen 7 6800HS; `hardware.json` explicitly notes GPU presence ≠ GPU execution) | **CUDA GPU** (RTX 3050 Laptop, 4 GB) |
| Peak VRAM | not measured (CPU execution) | 2819.4 MB allocated / 2904.6 MB reserved (full FP32 mode) |
| Python / OS | 3.14.4 / WSL2 | 3.14.4 / WSL2 |

**This is inherently not an identical compute-resource comparison and must
be stated as such.** The two performance questions must be reported
separately:

1. **Practical wall-clock throughput of the configuration actually used** —
   CPU MediaPipe vs GPU WiLoR, useful for deployment planning, never
   presented as a model-quality result.
2. **Model capability/quality independent of compute cost** — coverage,
   stability and plausibility metrics, which carry no timing component.

Forcing both models onto one device is **not** recommended: MediaPipe's
GPU delegate and WiLoR's CPU path are not the officially exercised
configurations here, and switching them would trade a documented default
for an unvalidated one.

# Configuration Audit

Recorded as-found. Neither configuration was changed, and neither was tuned
with knowledge of the other's results (MediaPipe's selection rule and
config were fixed before results were examined; WiLoR adopted the manifest
unchanged and used upstream demo defaults).

| MediaPipe (`configs/pose/mediapipe_karsl_pilot.json`, echoed in run metadata) | Value |
|---|---|
| running mode | VIDEO |
| num_hands | 2 |
| min_hand_detection_confidence | 0.5 |
| min_hand_presence_confidence | 0.5 |
| min_tracking_confidence | 0.5 |
| delegate | CPU |
| model | official `hand_landmarker.task` float16, SHA-256 `fbc2a300…` |

| WiLoR (`pose/wilor/config.py`, report) | Value |
|---|---|
| detector confidence | 0.3 (upstream `demo.py` default) |
| bbox rescale factor | 2.0 (upstream default) |
| precision mode used for the 18-video benchmark | **normal / FP32** (`mode="full"`; `--fast` used only on a separate 3-clip subset) |
| model input crop | 256 px (`MODEL.IMAGE_SIZE`), BBOX_SHAPE [192,256] |
| MANO | `MANO_RIGHT.pkl`, 15 hand joints, 10 betas, `init_renderer=False` |
| frames per forward pass | 1 video frame at a time (hands within a frame batched together) |
| WiLoR commit / checkpoints | `fcb9113…`; `detector.pt` `5ef3df44…`, `wilor_final.ckpt` `3e97aafc…` |

Notable default divergence: **detector confidence 0.5 (MediaPipe) vs 0.3
(WiLoR)**. Both are their own upstream defaults; a lower threshold
systematically admits more detections and therefore inflates coverage
relative to a higher one. This must be disclosed alongside any coverage
comparison. **Do not change either value** — doing so would replace a
documented intended operating point with a tuned one (**F-11**).

No evidence was found of fast/FP16 mode leaking into the normal WiLoR
benchmark: the runner sets `mode = "full_fast" if args.fast else "full"`,
the documented 18-video command passes no `--fast`, and every per-video row
in the TASK-002 results table reports `mode=full`.

# Visual-Validation Audit

| | MediaPipe | WiLoR |
|---|---|---|
| Artefact type | 21-landmark skeleton overlay MP4s with LEFT/RIGHT, handedness score, frame index, timestamp | Phase B: real reconstructed 21-joint MANO skeleton reprojected via `project_points_full_img` using the frame's own `camera_translation_xyz`/`focal_length`; plus `export_mesh_obj()` for the actual triangulated MANO mesh |
| Genuine model output? | Yes — drawn from the stored image landmarks; overlay scaling verified (`x*(width-1)`) | Yes — drawn from stored `landmarks_3d`, not placeholders |
| Present on disk now | **18/18 overlays** (`runs/mediapipe_karsl_pilot/overlays/`, ~30 MB) | **1 file only**, and it is the Phase-A **bounding-box** overlay; the 4 Phase-B 3D-skeleton overlays are **gone** |

MediaPipe's visual validation is sufficient to inspect failures today.
WiLoR's Phase-B visual evidence exists only as description in the TASK-002
report — the artefacts themselves were lost with the temporary worktree
(**F-1**) and must be regenerated.

# Implementation Issues Found

Scope was deliberately limited to issues that could materially change the
comparison. Each is documented, **not fixed** — no experimental branch was
modified.

---

### F-1 — WiLoR full-mode raw outputs and 3D visuals no longer exist on disk
- **Branch / area:** `opus/wilor-karsl-pilot` (run artefacts, not source code)
- **Severity: BLOCKING**
- **Issue:** the Phase-B (`mode="full"`) outputs — `runs/wilor_karsl_pilot_full/raw/*/wilor_raw.npz`, `summary.json`, and the 4 skeleton-overlay MP4s — were produced inside a temporary git worktree that was subsequently removed. A filesystem-wide search finds no surviving copy. The **only** WiLoR raw NPZ set present is `runs/wilor_karsl_pilot/`, which is the Phase-A **detector-only** run: verified to have `landmarks_3d` **all-NaN**, `mano_params_json` **all empty**, no `vertices` array, and `hand_present=True` on all 162 rows of the sample inspected.
- **Effect on comparison:** every normalization this audit requires (**F-2**, **F-3**, **F-4**, **F-6**) needs WiLoR's per-frame 3D data; none of them can be derived from the published report tables alone. Worse, a naive `runs/wilor*` glob silently picks up the detector-only set and would compare **WiLoR bounding boxes against MediaPipe landmarks** — precisely the failure mode this audit was asked to prevent. MediaPipe's raw NPZs, by contrast, are all present.
- **Recommended fix:** re-run the already-documented deterministic command from the WiLoR branch — no parameter change, no tuning:
  `python evaluation/benchmarks/wilor_karsl_pilot.py --out-dir runs/wilor_karsl_pilot_full`
  then regenerate the skeleton overlays, and assert `summary.json["mode"] == "full"` plus non-empty `mano_params_json` before use. Re-runs are deterministic (fixed weights, `torch.no_grad()`, no sampling) and took ≈ 224 s previously.

---

### F-2 — Per-hand coverage uses inclusive (MediaPipe) vs exclusive (WiLoR) definitions
- **Branch / area:** both metric layers (`mediapipe_baseline.py`, `hand_pose_metrics.py`)
- **Severity: IMPORTANT**
- **Issue:** `frames_with_left_hand` (MediaPipe) includes both-hand frames; `frames_left_only` (WiLoR) excludes them. Demonstrated on MediaPipe's own data: 803 (89.8 %) vs 52 (5.8 %) for the identical frames.
- **Effect:** tabulating the two reports' "left-hand %" side by side (89.8 % vs 1.1 %) would be off by ~84 percentage points for definitional reasons alone. Both-hand % is unaffected.
- **Recommended fix:** neutral layer computes `left_incl = left_only + both` and `right_incl = right_only + both` for both systems and reports both conventions explicitly.

---

### F-3 — "Longest missing streak" measures different things
- **Branch / area:** both metric layers
- **Severity: IMPORTANT**
- **Issue:** MediaPipe reports the longest run with a given **labelled channel** missing (left 18 / right 27 frames). WiLoR's `DetectionStats.longest_missing_streak` counts the longest run of frames with **no hand at all** (0).
- **Effect:** "0 vs 18/27" is not a comparison of the same quantity.
- **Recommended fix:** compute both quantities (per-channel streak *and* no-hand streak) for both systems in the neutral layer. Requires WiLoR raw data (depends on **F-1**).

---

### F-4 — Bone-length CV uses different bone edge sets
- **Branch / area:** `mediapipe_baseline.HAND_CONNECTIONS` (21 edges) vs `hand_pose_metrics.HAND_BONES` (20 edges)
- **Severity: IMPORTANT**
- **Issue:** only 18 edges are shared; MediaPipe adds the inter-MCP palm chain `(5,9),(9,13),(13,17)`, WiLoR adds wrist-radiating `(0,9),(0,13)`. Each report's CV is averaged over its own edge set.
- **Effect:** the headline "bone CV" values are means over different (and differently sized) edge populations, so the current numbers are not strictly comparable even though CV itself is scale-free.
- **Recommended fix:** recompute CV for both over the documented 18-edge intersection in the neutral layer. Joint indices already align, so no landmark remapping is needed and no raw value is altered.

---

### F-5 — Runtime scopes are not matched (and hardware differs by design)
- **Branch / area:** `pose/mediapipe/extractor.py` timing vs `pose/wilor/video_processing.py` timing
- **Severity: IMPORTANT**
- **Issue:** MediaPipe's `runtime_seconds` **includes** 1920×1080 overlay rendering and per-video landmarker construction; WiLoR's `inference_seconds` **excludes** all visualization and model loading but **includes** video decode. MediaPipe's `inference_seconds` covers only `detect_for_video`. No pair of published numbers shares a scope.
- **Effect:** MediaPipe's end-to-end FPS is penalised by work WiLoR never did; MediaPipe's inference-only FPS is measured over a narrower scope than WiLoR's only figure. Compounded by CPU vs GPU execution.
- **Recommended fix:** either (a) re-measure both with a matched scope (decode + inference only, overlays disabled, model construction outside the timer), or (b) report each number with an explicit scope label and **do not rank them**. Always state CPU vs GPU alongside.

---

### F-6 — `hand_present` does not assert that 3D reconstruction succeeded
- **Branch / area:** `opus/wilor-karsl-pilot`, `pose/wilor/frame_extraction.py`
- **Severity: IMPORTANT**
- **Issue:** `extract_frame_full` sets `hand_present=True` for every detector box that reaches the model, with no check that the 21 joints / MANO parameters exist and are finite. In detector-only mode the same field is `True` with **no 3D output at all**.
- **Effect:** taken at face value, `hand_present` conflates "detector found a box" with "3D hand was reconstructed" — the exact conflation this task prohibits.
- **Mitigation already present:** outputs are self-describing — `run_metadata_json["mode"]`, per-row `extractor_metadata["mode"]`, and `quality_flags` containing `detector_only_no_mano`.
- **Recommended fix:** the neutral layer must define
  `reconstructed := hand_present ∧ mode == "full" ∧ 21 finite landmarks_3d ∧ non-empty mano_params`,
  and hard-assert `summary.json["mode"] == "full"` before ingesting any WiLoR run. Do not modify the branch.

---

### F-7 — FPS aggregation differs (frame-weighted vs mean-of-per-video)
- **Severity: MINOR** — MediaPipe reports 21.865 FPS frame-weighted; WiLoR reports 4.18 FPS as a mean of per-video FPS (frame-weighted equivalent: 4.00). Both underlying totals are published, so this is directly derivable. Fix: standardize on `total_frames / total_seconds`.

---

### F-8 — Duplicate-label handling differs
- **Severity: MINOR** — MediaPipe's canonicalization keeps the higher-scoring hand per label slot (32/894 frames affected). WiLoR groups strictly by label with no de-duplication, so a duplicated label places two hands **from the same frame** into one temporal sequence, creating a spurious "frame-to-frame" delta in its jitter and bone-variation metrics (1/894 frames affected — `karsl_test_s02_sign0176_repfirst`, the frame with 3 simultaneous detections). Fix: apply one de-duplication policy in the neutral layer.

---

### F-9 — Maximum simultaneous hands differs structurally
- **Severity: MINOR** — MediaPipe is hard-capped at 2 hands and structurally cannot emit a third; WiLoR is uncapped (1 frame with 3 detections observed). This asymmetrically affects hand-count-change and false-positive metrics. Not a bug in either. Fix: document, and either compare hand-count changes with this caveat stated or restrict WiLoR to its top-2 detections for that metric only.

---

### F-10 — Timestamp sources differ
- **Severity: MINOR** — MediaPipe uses decoder position timestamps (ffprobe unavailable, 0 monotonicity adjustments); WiLoR uses `frame_index / fps`. Verified numerically equivalent on this CFR material: max |Δ| = 4.4 × 10⁻¹⁶ s. Frame-indexed metrics are unaffected; only MediaPipe's acceleration metric consumes dt. No action required beyond disclosure.

---

### F-11 — Detector confidence thresholds differ (0.5 vs 0.3)
- **Severity: MINOR** — each is its model's own upstream default, so this is a legitimate intended-configuration difference, but a lower threshold systematically admits more detections and inflates coverage. **Do not change either value.** Fix: disclose prominently next to every coverage number.

---

### F-12 — "Jitter" is a different mathematical quantity in each report
- **Severity: MINOR** — MediaPipe computes the **second difference** of world-space wrist position (m/frame); WiLoR computes the **first difference** of camera-space translation (model units). Different operator *and* different space/units. Fix: mark as not directly comparable; if a temporal-smoothness comparison is wanted, define one operator and recompute both (requires **F-1**).

---

### Verified clean (no issue found)

- **RGB/BGR handling** — MediaPipe converts `COLOR_BGR2RGB` before constructing an `mp.Image(SRGB, …)`; WiLoR passes BGR to the Ultralytics detector, which is the format its API expects (matching upstream `demo.py`). Both correct.
- **Frame skipping / duplication** — none in either pipeline (894 = 894 = 894, per video).
- **Silently discarded output after exception** — neither pipeline swallows results: MediaPipe preserves partial frames and marks the video `failed`; WiLoR records an explicit `extraction_failed` frame. Both reported 0 failures on this pilot.
- **Fast/FP16 mode leaking into the normal benchmark** — no evidence; the runner's mode flag and every per-video row say `full`.
- **MediaPipe result indexing** — `hand_landmarks[i]`, `hand_world_landmarks[i]`, `handedness[i][0]` are consistently paired by detector index.
- **WiLoR MANO mirroring** — applied consistently to vertices, joints and one camera parameter, matching upstream; the mirror's exactness was independently verified against `MANO_LEFT.pkl`.
- **Denominators** — both use decoded frames per video and frame-weight their coverage aggregates.
- **Overlay coordinate scaling** — MediaPipe scales normalized landmarks by `(width-1)/(height-1)`; correct.

# Common Metric Contract

`✔` = available, `✖` = not available, `~` = derivable from raw but not
currently reported.

| Metric | MediaPipe available? | WiLoR available? | Directly comparable? | Normalization required? | Reason |
|---|:--:|:--:|:--:|---|---|
| Total frames | ✔ | ✔ | **Yes** | None | 894 = 894, verified per video |
| No-hand frames | ✔ | ✔ | **Yes** | Gate WiLoR on `mode=="full"` + 3D validity (**F-6**) | Same concept, same denominator |
| At-least-one-hand frames | ✔ | ~ | **Yes** | `total − no_hand` for WiLoR | Trivially derivable |
| Both-hand % | ✔ | ✔ | **Yes** | Gate on 3D validity (**F-6**) | Verified identical concept under both rules (751 either way on MediaPipe data) |
| Left-hand % | ✔ (inclusive) | ✔ (exclusive) | **No, as published** | **Yes — recompute `left_incl = left_only + both`** (**F-2**) | ~84 pp definitional gap |
| Right-hand % | ✔ (inclusive) | ✔ (exclusive) | **No, as published** | **Yes — same fix** (**F-2**) | Same as above |
| Longest missing streak | ✔ (per channel L/R) | ✔ (no-hand only) | **No, as published** | **Yes — compute both variants for both** (**F-3**) | Different quantities |
| Hand-count changes | ~ | ✔ | Yes, with caveat | State the 2-hand cap (**F-9**); optionally top-2 WiLoR | MediaPipe structurally cannot exceed 2 |
| Handedness instability | ✔ (label-set changes, detector-order changes, duplicates) | ~ (only swap candidates) | Partially | Recompute a common indicator set from raw both sides | WiLoR reports fewer indicator types |
| Suspected swap events | ✔ (heuristic: duplicates + order reversals + x-crossings) | ✔ (heuristic: displacement-swap test) | **No** | **Yes — one shared heuristic, recomputed on both** | Different heuristics; neither is ground truth |
| Normalized bone-length CV | ✔ (21 edges) | ✔ (20 edges) | **No, as published** | **Yes — recompute on the 18-edge intersection** (**F-4**) | Scale-free once edge sets match |
| Temporal continuity / jitter | ✔ (2nd difference, world metres) | ✔ (1st difference, camera-space model units) | **No** | Not resolvable by scaling — pick one operator and recompute, or drop | Different operator *and* different units (**F-12**) |
| Absolute 3D wrist displacement / depth / translation | ✔ (metres) | ✔ (model units) | **No — must not be ranked** | No conversion invented | Different coordinate systems |
| Runtime (seconds) | ✔ | ✔ | **No, as published** | **Yes — matched scope or explicit scope labels** (**F-5**) | MediaPipe includes overlay rendering; WiLoR does not |
| Effective FPS | ✔ (frame-weighted, incl. overlay, CPU) | ✔ (mean-of-video, excl. overlay, GPU) | **No, as published** | **Yes — F-5 + F-7** | Scope, aggregation, and hardware all differ |
| Inference-only FPS | ✔ (48.880) | ✖ | **No** | Requires a WiLoR inference-only timer | WiLoR never isolated inference from decode |
| Peak VRAM | ✖ (CPU execution) | ✔ (2819 MB alloc / 2905 MB reserved) | **No** | None possible | Different execution devices by design |
| Visual plausibility | ✔ (18 overlays on disk) | ✖ **currently** (Phase-B visuals lost, **F-1**) | **MANUAL REVIEW** | Regenerate WiLoR visuals | Qualitative; no numeric precision to invent |
| Thumb articulation | ✔ (numeric availability only) | ✖ currently | **MANUAL REVIEW** | Regenerate WiLoR 3D output | MediaPipe reports thumb landmark *availability*, not correctness; no ground truth exists |
| Occlusion behaviour | ✔ (explicitly marked unmeasured) | ~ (narrative observations) | **MANUAL REVIEW** | None | Neither exposes an occlusion cause; no labels |
| False-positive behaviour | ~ | ~ (1 spurious 3-hand frame documented) | **MANUAL REVIEW** | None | No ground truth; **F-9** cap asymmetry applies |

# Blocking Issues

**1 blocking issue: F-1.**

WiLoR's full-mode raw NPZ outputs, `summary.json`, and 3D visual artefacts
no longer exist on disk; the only WiLoR raw data present is the Phase-A
**detector-only** set, whose `hand_present=True` rows contain no 3D
reconstruction whatsoever. Until the full-mode run is regenerated,
TASK-003B cannot compute a single harmonized WiLoR metric from raw data,
and risks silently comparing bounding boxes against landmarks.

No other finding prevents TASK-003B from starting, provided the
normalizations below are implemented in the neutral evaluation layer.

# Recommended Fixes Before Comparison

Ordered. Items 1 is mandatory; 2–6 are required for a fair metric contract
but are neutral-layer work only.

1. **(BLOCKING, F-1)** Regenerate the WiLoR full-mode run on the WiLoR
   branch with the already-documented command and unchanged parameters:
   `python evaluation/benchmarks/wilor_karsl_pilot.py --out-dir runs/wilor_karsl_pilot_full`,
   then regenerate the 3D skeleton overlays. Verify
   `summary.json["mode"] == "full"`, 894 frames, and non-empty
   `mano_params_json` before ingesting. **No parameter may be changed.**
2. **(F-6)** Implement an explicit `reconstructed_hand` predicate in the
   neutral layer (`hand_present ∧ mode=="full" ∧ 21 finite joints ∧
   non-empty MANO params`) and hard-fail on any run whose mode is not
   `full`.
3. **(F-2)** Recompute inclusive left/right coverage for both systems
   (`left_incl = left_only + both`); publish both inclusive and exclusive
   forms so neither report is silently reinterpreted.
4. **(F-3)** Compute both per-channel missing streaks and no-hand streaks
   for both systems.
5. **(F-4)** Recompute bone-length CV for both over the 18-edge
   intersection, documented explicitly.
6. **(F-5, F-7)** Either re-measure runtime for both with a matched scope
   (decode + inference, overlays off, model construction excluded) or label
   every timing number with its scope and decline to rank; standardize FPS
   as `total_frames / total_seconds`; always print CPU vs GPU next to any
   timing figure.

Additionally disclose, without changing anything: the 0.5 vs 0.3 detector
confidence defaults (**F-11**), the 2-hand cap asymmetry (**F-9**), the
duplicate-label policy difference (**F-8**), and the fact that all
absolute 3D quantities live in different coordinate systems.

**Explicitly not recommended:** changing either model's thresholds, forcing
both onto the same compute device, altering raw pose values, relabelling
handedness, or modifying either experimental branch to make numbers line
up.

# Fairness Verdict

**NOT READY FOR COMPARISON**

Dataset, frame decoding, raw-output integrity, handedness convention, and
joint indexing are all verified equivalent — the experimental *conditions*
are sound. The comparison is blocked on one recoverable artefact problem
(**F-1**: WiLoR's full-mode raw output no longer exists, and the only
WiLoR data on disk is detector-only), plus five metric-definition
mismatches that must be normalized in the neutral evaluation layer before
any number is placed side by side.

Once **F-1** is regenerated and fixes 2–6 are implemented in the neutral
layer, this pilot becomes suitable for a fair TASK-003B comparison. No
winner is declared here, and no model was tuned, altered, or preferred.
