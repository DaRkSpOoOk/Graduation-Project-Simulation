# TASK-008C — Full KArSL Core-28 Virtual-Glove Extraction and Final Dataset QA

## Task

Finalize TASK-008C: independently validate the completed 4,222-sequence Core-28
production run, generate the final dataset metadata/index/report artifacts, and
decide, on evidence, whether the dataset is ready for TASK-009.

The expensive `RGB → WiLoR → tracking → kinematics → virtual glove` pass was run
to completion by the user overnight. **Nothing in this task re-ran WiLoR, re-extracted
a sample, or modified a production artifact.** Every number below was re-derived from
the run root, the frozen manifest and the frozen label/split tables by tooling that
never imports the extractor.

## Scope

| In scope | Out of scope |
| --- | --- |
| Coverage accounting against the durable worker state and the disk | Any re-extraction |
| Per-sample QA over all 4,222 samples (provenance, alignment, contract) | Changing TASK-004/005/006 mathematics |
| Dataset-level aggregation (validity, tracking, availability, lengths) | LSTM design or training |
| Sensor-contract, temporal, value and quaternion verification | Extended letters, numbers |
| Label and LOSO integrity | Merging the stacked PRs |
| Final index `karsl_core28_virtual_glove.csv` | Committing bulk NPZ output |
| TASK-009 readiness analysis (documentation only) | Choosing the TASK-009 tensor layout |

## Production configuration

Read back from `provenance/shard-00.json` in the run root — not from memory.

| Item | Value |
| --- | --- |
| Run root | `/home/hatim/graduation-project-runs/task008-core28-full` (outside git) |
| Dataset root | `/home/hatim/datasets/KArSL-502` |
| Manifest | `datasets/manifests/karsl_core28.csv`, SHA-256 `365961b3…9c3eff1` |
| Git commit of the worker | `5c47de98771154d88ffa1896b96ae8933431ff42` |
| Branch | `opus/task-008c-core28-full-extraction` |
| WiLoR | upstream `fcb91131…`, detector confidence 0.30, rescale 2.0 |
| Precision | **full FP32; fast/FP16 disabled** (`mode: "full FP32; fast/FP16 disabled"`) |
| TF32 | matmul TF32 **off**; cuDNN TF32 left at the framework default |
| GPU | NVIDIA GeForce RTX 3050 Laptop, 4096 MiB, driver 610.62 |
| Torch / CUDA / NumPy / Python | 2.13.0+cu130 / 13.0 / 2.5.2 / 3.14.4 |
| Shards / workers | 1 shard, index 0 — one sequential GPU worker |
| Contracts | `TASK-005-final-v2`, `TASK-006-ideal-virtual-glove-v1` |

The manifest SHA-256 recorded in the durable state matches the manifest committed on
this branch, so the run and the repository describe the same 4,222 videos.

## Dataset

Frozen Core-28 Arabic alphabet, KArSL-502 RGB, SignIDs 0032–0059.

| Property | Value |
| --- | --- |
| Classes | 28 (`ا ب ت ث ج ح خ د ذ ر ز س ش ص ض ط ظ ع غ ف ق ك ل م ن ه و ي`) |
| Sequences | 4,222 |
| Signers | 01 (1,402), 02 (1,401), 03 (1,419) |
| Official partition | train 3,550 / test 672 |
| Frames | 83,659 |
| Source video bytes | 1,794,280,889 (1.67 GiB) |
| Samples per class | 149–160 (mode 150) |

## Extraction completion

| Metric | Value |
| --- | --- |
| Requested | 4,222 |
| `VIRTUAL_GLOVE_DONE` in durable state | 4,222 |
| `FAILED` | 0 |
| Incomplete / other status | 0 |
| Unaccounted (in manifest, absent from state) | 0 |
| Extra in state, not requested | 0 |
| Frames processed (state sum) | 83,659 — equals the manifest frame total exactly |
| Samples with `attempts > 1` | 25 (the 24 preparation samples plus the one in-flight sample, re-entered by `--resume`) |
| `model_loads` | 1 |

## Coverage

Three independent sources agree at 4,222:

| Source | pose | tracking | kinematics | virtual_glove |
| --- | ---: | ---: | ---: | ---: |
| Directories on disk | 4,222 | 4,222 | 4,222 | 4,222 |
| Stage artifacts revalidated (hash + schema) | 4,222 | 4,222 | 4,222 | 4,222 |
| Per-sample QA pass | 4,222 | 4,222 | 4,222 | 4,222 |

* Extra sample IDs on disk not in the manifest: **0** (all four stages)
* Manifest samples missing from disk: **0** (all four stages)
* Duplicate sample IDs: **0** — the manifest has 4,222 unique IDs and the filesystem cannot hold a duplicate directory name
* Per-sample QA failures: **0 / 4,222**

The per-sample pass re-hashes every source video with SHA-256 and rejects a sample whose
stage sidecar records a different source hash, a different manifest hash, or a malformed
array. All 4,222 sources hashed to their declared value.

## Frame integrity

Verified per sample across the full chain
`source video → distinct pose frame indices → tracking → kinematics → virtual glove`.

| Check | Result |
| --- | --- |
| Output sequence length == source `frame_count` (ffprobe, frozen manifest) | 4,222 / 4,222 |
| Distinct pose `frame_index` == kinematics `frame_index` | 4,222 / 4,222 |
| Tracking / glove `frame_index` == kinematics `frame_index` | 4,222 / 4,222 |
| Per-frame `timestamp_seconds` identical across all four stages | 4,222 / 4,222 |
| Length mismatches | 0 |

**The multi-row pose trap.** The raw WiLoR NPZ stores *one row per detected hand*, so a
two-hand video carries roughly twice as many rows as frames. Comparing that row count
against the video frame count is wrong, and it is a defect this project has now hit
three times (the TASK-008B resume validator, and again in `evaluation/dataset/qa.py`,
fixed in commit `5c47de9` before this run finished). The alignment check therefore
compares **distinct pose frame indices**, and separately asserts that all rows sharing a
frame index agree on that frame's timestamp. Row-vs-frame counts are never equated.

## Temporal integrity

Checked on all 4,222 glove outputs by `verify_dataset_contract`:

| Property | Result |
| --- | --- |
| `frame_index` strictly increasing | 4,222 / 4,222 |
| `frame_index` starts at 0 | 4,222 / 4,222 |
| `frame_index` contiguous (step 1, no gaps) | 4,222 / 4,222 |
| `timestamp_seconds` strictly increasing and finite | 4,222 / 4,222 |
| Padding performed | No |
| Truncation performed | No |
| Resampling performed | No |
| Interpolation performed | No |
| Fabricated hand measurements | **0** — every invalid channel is NaN |

The padding/truncation/resampling claims are not assertions of intent; they follow from
the measured equality `output length == source frame count` for every sample, together
with contiguous frame indices starting at 0. The "no fabrication" claim is measured
directly: across all 4,222 samples, the number of channels marked invalid that
nevertheless carry a finite value is **0**. An imputed zero would read to a downstream
model as a perfectly straight finger, so the dataset must keep NaN.

## Tracking distributions

167,318 hand instances (83,659 frames × 2 tracks).

| State | Count | Share |
| --- | ---: | ---: |
| OBSERVED | 164,501 | 98.316% |
| MISSING | 2,817 | 1.684% |
| AMBIGUOUS | 0 | 0% |
| LIKELY_OCCLUDED | 0 | 0% |
| REJECTED_QUALITY | 0 | 0% |

Per track:

| Track | OBSERVED | MISSING |
| --- | ---: | ---: |
| LEFT | 81,574 | 2,085 (2.49%) |
| RIGHT | 82,927 | 732 (0.87%) |

By signer (hand availability equals the OBSERVED rate, see below):

| Signer | Videos | Frames | Hand availability |
| --- | ---: | ---: | ---: |
| 01 | 1,402 | 29,947 | 99.65% |
| 02 | 1,401 | 19,245 | 99.93% |
| 03 | 1,419 | 34,467 | 96.26% |

By official partition:

| Partition | Videos | Frames | Hand availability |
| --- | ---: | ---: | ---: |
| train | 3,550 | 70,232 | 98.42% |
| test | 672 | 13,427 | 97.77% |

**Reading of the state distribution.** The tracker never entered AMBIGUOUS,
LIKELY_OCCLUDED or REJECTED_QUALITY on this corpus. That is consistent with the
recording conditions — a single seated signer, frontal camera, plain background, both
hands almost always separable — rather than evidence that those code paths are dead;
TASK-004C/004D exercised them on adversarial fixtures. It does mean TASK-009 will see a
two-valued state channel in practice, and should not assume the other three states will
never appear in future data.

A non-OBSERVED state is **not** an extraction failure. It records that no hand was
reconstructable in that frame, which is information TASK-009 needs, not a defect.

## Hand availability

Availability is measured from the frozen tracking semantics: a hand counts as available
only in a pose-bearing state (OBSERVED or AMBIGUOUS). The mere existence of an array
slot never counts.

| Case | Frames | Share of 83,659 |
| --- | ---: | ---: |
| Both hands | 80,842 | 96.63% |
| Left available (either case) | 81,574 | 97.51% |
| Right available (either case) | 82,927 | 99.13% |
| Left only | 732 | 0.87% |
| Right only | 2,085 | 2.49% |
| Neither hand | **0** | 0% |

Every one of the 83,659 frames carries at least one reconstructed hand. There is no
empty frame in the dataset.

## Sensor validity

| Channel group | Valid | Total | Overall | Conditional on a pose-bearing hand |
| --- | ---: | ---: | ---: | ---: |
| Bend (15/hand) | 2,467,515 | 2,509,770 | **98.316%** | **100.000%** |
| Spread (4/hand) | 520,974 | 669,272 | **77.842%** | **79.175%** |
| Palm IMU (1/hand) | 164,501 | 167,318 | **98.316%** | **100.000%** |

This is the single most informative result in the QA. Bend and IMU validity are
*exactly* the hand-availability rate, to the last hand instance: whenever a hand exists,
all 15 bend channels and the palm IMU are valid. Their 1.68% invalidity is entirely the
2,817 MISSING hands and contains no separate failure mode.

Spread is different, and legitimately so. The extra 20.8% of invalid spread channels
comes from the TASK-005 15° palm-plane conditioning rule: when a finger's distal
direction projects onto the palm plane with norm below `sin 15° = 0.2588`, the projected
azimuth is numerically ill-conditioned and the spread angle is emitted as NaN rather than
as a confident wrong number. This is the a-priori geometric limit fixed in TASK-005A,
not a dataset-fitted threshold, and it fires most on letters whose fingers point toward
or away from the camera.

By partition:

| Partition | Bend | Spread | IMU |
| --- | ---: | ---: | ---: |
| train | 98.42% | 78.01% | 98.42% |
| test | 97.77% | 76.96% | 97.77% |

By signer:

| Signer | Bend | Spread | IMU |
| --- | ---: | ---: | ---: |
| 01 | 99.65% | 80.24% | 99.65% |
| 02 | 99.93% | 80.23% | 99.93% |
| 03 | 96.26% | 74.43% | 96.26% |

## Per-signer findings

* **Signer 03 is the weakest**, on both hand availability (96.26% vs 99.65/99.93%) and
  spread validity (74.43% vs ~80.2%). It also has the longest sequences (mean 24.3
  frames). Signer 03 is the held-out test signer of the S03 LOSO fold, so that fold is
  the hardest of the three and its result should not be averaged naively with the others
  without noting the input-quality difference.
* **Signer 02 has by far the shortest sequences** (mean 13.7 frames, min 10, max 45)
  against signer 01 (mean 21.4) and 03 (mean 24.3) — a ~1.8× difference in signing tempo
  between signers. Any TASK-009 batching or fixed-window design must handle this; a
  window tuned to signer 02 would truncate most of signers 01 and 03.
* No signer shows a systematic sensor defect. The differences are recording differences,
  not extraction differences.

## Per-class findings

| sign_id | letter | videos | frames | hand availability | spread validity |
| --- | --- | ---: | ---: | ---: | ---: |
| 0032 | ا | 150 | 3381 | 99.47% | 72.17% |
| 0033 | ب | 152 | 3112 | 99.95% | 64.08% |
| 0034 | ت | 150 | 2858 | 100.00% | 99.43% |
| 0035 | ث | 150 | 2831 | 99.95% | 96.18% |
| 0036 | ج | 158 | 3197 | 99.80% | 99.09% |
| 0037 | ح | 149 | 2971 | 97.00% | 88.50% |
| 0038 | خ | 150 | 2952 | 99.42% | 67.60% |
| 0039 | د | 150 | 2910 | 100.00% | 64.25% |
| 0040 | ذ | 150 | 2815 | 97.42% | 86.84% |
| 0041 | ر | 150 | 2896 | 99.52% | 66.25% |
| 0042 | ز | 150 | 3037 | 100.00% | 92.71% |
| 0043 | س | 150 | 2904 | 100.00% | 100.00% |
| 0044 | ش | 150 | 2883 | 100.00% | 99.94% |
| 0045 | ص | 150 | 2974 | 99.18% | 59.13% |
| 0046 | ض | 150 | 2885 | 99.36% | 55.42% |
| 0047 | ط | 151 | 2986 | 100.00% | 71.13% |
| 0048 | ظ | 150 | 2872 | 96.03% | 61.78% |
| 0049 | ع | 150 | 2908 | 91.97% | 79.49% |
| 0050 | غ | 150 | 2942 | 100.00% | 65.70% |
| 0051 | ف | 150 | 3024 | 92.74% | 60.35% |
| 0052 | ق | 150 | 2995 | 100.00% | 76.05% |
| 0053 | ك | 150 | 3064 | 99.92% | 99.13% |
| 0054 | ل | 151 | 3063 | 99.82% | 70.89% |
| 0055 | م | 151 | 3066 | 91.29% | 84.90% |
| 0056 | ن | 150 | 3018 | 99.98% | 68.05% |
| 0057 | ه | 150 | 3029 | 100.00% | 99.83% |
| 0058 | و | 150 | 2924 | 89.71% | 75.75% |
| 0059 | ي | 160 | 3162 | 99.94% | 56.57% |

**Worst five by spread validity:** ض 55.4%, ي 56.6%, ص 59.1%, ف 60.4%, ظ 61.8%.
**Worst five by hand availability (= bend/IMU validity):** و 89.7%, م 91.3%, ع 92.0%,
ف 92.7%, ظ 96.0%.

**Are these suspicious?** They are consistent with hand geometry, not with a bug:

* The low-spread letters (ض ي ص ف ظ) are all produced with fingers pointing broadly
  toward or away from the camera, which is exactly the configuration that collapses the
  palm-plane projection and triggers the 15° conditioning rule. The bend channels of the
  same letters stay high (ض 99.4%, ي 99.9%, ص 99.2%), so the fingers were reconstructed
  fine — only the *azimuthal* spread is ill-posed.
* The low-availability letters (و م ع ف) are the ones where one hand leaves the frame or
  is occluded by the other. `ف` appears in both lists, which is expected: a hand that is
  absent contributes both a missing bend channel and a missing spread channel.
* No class falls below 89.7% hand availability, and no class has a spread validity
  anomaly that is not accompanied by a plausible geometric explanation. Nothing here
  warrants re-extraction.

TASK-009 should treat spread as a channel with a class-dependent missingness rate
between 55% and 100% and model it with an explicit validity mask, not impute it.

## Sensor contract

Verified on all 4,222 samples against the frozen TASK-006 layout.

| Property | Expected | Measured |
| --- | --- | --- |
| Distinct `sensor_layout.json` channel orderings across the dataset | 1 | **1** |
| Bend Hall channels / hand | 15 | 15 |
| Spread Hall channels / hand | 4 | 4 |
| Hall channels / hand | 19 | 19 |
| Palm IMU packages / hand | 1 | 1 |
| Logical sensing packages / hand | 20 | 20 |
| Hall channels, two hands | 38 | 38 |
| IMU packages, two hands | 2 | 2 |
| Finger order | thumb, index, middle, ring, pinky | matches |
| Bend chain order | proximal, middle, distal | matches |
| Bend `array_index` ordering | finger-major, chain-minor | matches |
| Spread pairs | thumb–index, index–middle, middle–ring, ring–pinky | matches |
| Spread `array_index` | 0…3 in that pair order | matches |
| `display_marker` | `H` on all 19 Hall, `IMU` on the palm IMU | matches |
| `layout_version` | `ideal_virtual_glove_v1` | matches |

A single distinct layout fingerprint across 4,222 independently written files is the
evidence that no channel was reordered between samples. Array shapes were separately
asserted per sample: `bend_* [F,2,5,3]`, `spread_* [F,2,4]`, `imu_rotation_matrix
[F,2,3,3]`, `imu_quaternion_wxyz [F,2,4]`, `palm_imu_valid [F,2]`.

## Value integrity

| Check | Result |
| --- | --- |
| `bend_angle_deg` on valid channels | [0.0159°, 118.070°] ⊂ [0°, 180°] |
| `spread_angle_deg` on valid channels | [0.00016°, 137.750°] ⊂ [0°, 180°] |
| `bend_normalized == bend_angle_deg / 180.0` | holds on every valid channel (atol 2e-7) |
| `spread_normalized == spread_angle_deg / 180.0` | holds on every valid channel (atol 2e-7) |
| Normalized values within [0, 1] | 4,222 / 4,222 |
| Validity mask == `isfinite(angle)` | 4,222 / 4,222 |
| Quaternion ordering | WXYZ |
| Quaternion unit norm | max deviation **4.72e-08** |
| Quaternion sign convention `w ≥ 0` | **0** violations out of 164,501 valid IMU instances |
| IMU rotation matrix orthonormal, det > 0 | 4,222 / 4,222 |
| 12-bit ADC on valid channels (sampled) | within 0…4095; observed 1…2607; `-1` sentinel on invalid |

The divisor is the fixed constant 180.0. **No dataset min/max normalization was applied
and none should be**: the observed ceiling of 118°/138° is a property of this corpus, and
fitting the scale to it would silently change the meaning of a channel the moment new
data arrives. The 12-bit ADC range does not reuse the old prototype glove's ~850–1700
band.

## Sequence length statistics

| Scope | count | min | p5 | p25 | median | p75 | p95 | max | mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall | 4,222 | 9 | 12.0 | 15.0 | 20.0 | 24.0 | 29.0 | 70 | 19.82 |
| Signer 01 | 1,402 | 14 | 17.0 | 19.0 | 21.0 | 23.0 | 29.0 | 70 | 21.36 |
| Signer 02 | 1,401 | 10 | 11.0 | 13.0 | 13.0 | 15.0 | 17.0 | 45 | 13.74 |
| Signer 03 | 1,419 | 9 | 18.0 | 22.0 | 24.0 | 26.0 | 30.0 | 67 | 24.29 |

Total 83,659 frames. The 7.8× spread between the shortest (9) and longest (70) sequence,
and the 1.8× difference in per-signer means, are the dominant constraints on TASK-009's
batching design.

## Label / class integrity

| Check | Result |
| --- | --- |
| Distinct classes in production | 28 |
| Frozen label table entries | 28 |
| SignID range | 0032 … 0059 |
| `label_index` contiguous 0…27 | yes |
| `label_ar` matches the frozen table for every sample | 4,222 / 4,222 |
| `label_index` matches the frozen table for every sample | 4,222 / 4,222 |
| `signer_id` ∈ {01, 02, 03} | 4,222 / 4,222 |
| `official_partition` ∈ {train, test} | 4,222 / 4,222 |
| `source_relative_path` and `source_sha256` present | 4,222 / 4,222 |
| Label problems | **0** |

## LOSO split integrity

Re-validated against the frozen split files, not against the TASK-008B report's numbers.

| Fold | Held-out | train | validation | test | total | Leakage | Classes per role | Covers manifest |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| S01 | signer 01 | 2,372 | 448 | 1,402 | 4,222 | none | 28 / 28 / 28 | exactly |
| S02 | signer 02 | 2,373 | 448 | 1,401 | 4,222 | none | 28 / 28 / 28 | exactly |
| S03 | signer 03 | 2,355 | 448 | 1,419 | 4,222 | none | 28 / 28 / 28 | exactly |

Every fold passes the frozen `validate_split_rows` validator, every test role contains
exactly the held-out signer and nothing else, no held-out sample appears in train or
validation, and each fold's sample IDs are exactly the 4,222 manifest IDs. The measured
counts match the TASK-008B expectations exactly.

## Runtime

| Metric | Value |
| --- | --- |
| Worker wall time (reported) | 30,896.464 s = **8 h 34 m 56 s** |
| Worker wall time (independently derived) | 30,896.439 s, from provenance `2026-09-03T09:36:13.684Z` → state `2026-09-03T18:11:10.123Z` |
| Agreement | 0.025 s |
| Throughput | 2.708 frames/s; 7.318 s per video |
| Peak VRAM | 2,813,499,392 B = 2,683 MiB of 4,096 MiB |
| Model loads | 1 |
| GPU workers | 1 (sequential) |

The two wall-clock figures come from different sources — the worker's own timer and two
independently written JSON timestamps — and agree to 25 ms, so the runtime is confirmed
rather than merely restated. Measured throughput (2.71 FPS) is below the 3.68 FPS seen
on the 24-sample preparation batch; the preparation batch was short, ran on a cold and
cool GPU, and is not a fair estimator of an 8.5-hour thermally-limited laptop run.

## Storage

| Item | Bytes | Size |
| --- | ---: | --- |
| Run root total | 2,255,316,539 | **2.10 GiB** |
| `pose/` (raw WiLoR, incl. MANO vertices) | 1,879,676,450 | 1.75 GiB (83.3%) |
| `tracking/` | 175,724,711 | 167.6 MiB |
| `virtual_glove/` | 140,210,394 | 133.7 MiB |
| `kinematics/` | 58,765,164 | 56.0 MiB |
| `state/` | 936,893 | 915 KiB |
| `provenance/` | 1,892 | 1.8 KiB |
| Source videos (input, unchanged) | 1,794,280,889 | 1.67 GiB |

The dataset TASK-009 actually consumes is the 133.7 MiB `virtual_glove/` tree. The 1.75
GiB `pose/` tree is the raw reconstruction kept for reproducibility and for any future
task that needs landmarks or vertices; it is not needed for recognition. None of this is
committed to git — only the 1.9 MB index CSV is.

## Known limitations

1. **Spread has a large, class-dependent missingness rate** (55%–100% by class, 77.8%
   overall). This is an honest consequence of the TASK-005 15° conditioning rule, not a
   fixable extraction bug, and it cannot be removed without emitting confidently wrong
   azimuths.
2. **Only two tracking states occur** (OBSERVED, MISSING). AMBIGUOUS, LIKELY_OCCLUDED and
   REJECTED_QUALITY are never produced on this corpus, so the dataset does not exercise
   them and TASK-009 cannot learn how to handle them from this data.
3. **Signer imbalance in sequence length** — signer 02's mean is 13.7 frames against
   signer 03's 24.3. LOSO folds therefore differ in difficulty as well as in signer.
4. **Only three signers.** LOSO with three folds gives a high-variance generalization
   estimate; a per-fold result is not a confidence interval.
5. **The IMU is orientation-only.** No accelerometer channel exists, and per TASK-006A
   none was fabricated: WiLoR's camera translation is uncalibrated in scale
   (|t_z|/palm_length ≈ 397), so absolute linear acceleration is not recoverable.
   `imu_angular_velocity_rad_s` exists and is a finite-difference derivative, valid on
   roughly 93.6% of hand instances (sampled) — lower than IMU orientation validity
   because it needs two consecutive pose-bearing frames.
6. **Absolute external run path.** The production tree deliberately lives outside git at
   `/home/hatim/graduation-project-runs/task008-core28-full`. The committed index stores
   only run-root-relative paths; the run root itself must be supplied by the reader. This
   is stated explicitly rather than papered over.
7. **Extended letters (0060–0070) and numbers (0001–0031) are not extracted.** Out of
   scope for Core-28.
8. **Validation split is signer-mixed by design.** Each fold's 448 validation samples are
   drawn from the two non-held-out signers, so validation measures in-signer fit while
   test measures cross-signer generalization. That asymmetry is intentional but must be
   remembered when reading learning curves.

## Readiness for TASK-009

**No LSTM is implemented here.** This section documents what TASK-009 can consume and
what it must decide. It deliberately does not choose.

### Available per hand, per frame

| Block | Channels | Notes |
| --- | ---: | --- |
| `bend_normalized` | 15 | 5 fingers × 3 chain joints, `deg/180`, valid iff hand present |
| `spread_normalized` | 4 | adjacent pairs, 79.2% valid given a present hand |
| `imu_quaternion_wxyz` | 4 | unit norm, `w ≥ 0`, valid iff hand present |
| **Primary ML channels** | **23** | 46 for two hands |
| `bend_valid` / `spread_valid` / `palm_imu_valid` | 15 / 4 / 1 | explicit masks |
| `tracking_state_code` | 1 | frozen state codes, two values observed |
| `imu_rotation_matrix` | 9 | redundant with the quaternion |
| `imu_angular_velocity_rad_s` + validity | 3 + 1 | derived, needs two consecutive frames |
| `bend_adc_12bit` / `spread_adc_12bit` | 15 / 4 | hardware-shaped view, `-1` when invalid |
| `source_valid_kinematics` / `source_valid_palm_frame` | 2 | upstream Model-B validity |

### Constraints TASK-009 inherits

* Sequence lengths are 9–70 frames; the per-signer mean varies 1.8×.
* 1.68% of hand instances have no pose at all; 3.37% of frames are single-handed; **0**
  frames are empty.
* 20.8% of spread channels on present hands are NaN, concentrated in specific classes.
* NaN means *unknown*, never *zero*. Any silent `nan_to_num` would tell the model a
  missing hand is a flat hand.
* Left/right are separate tracks with different missingness (LEFT 2.49% vs RIGHT 0.87%).
* Quaternions are absolute palm orientation in the camera frame, sign-canonicalized to
  `w ≥ 0`. That canonicalization introduces a discontinuity when `w` crosses 0.
* Only three signers exist; any statistical transform must be fitted on train only, per
  fold, or it leaks the held-out signer.

### Decisions TASK-009 must make explicitly (not made here)

Left/right representation; missing-hand representation; whether masks are input features;
whether the tracking state is a feature; invalid-spread handling; variable-length batching
and padding (with a padding mask distinct from the validity mask); whether palm orientation
is absolute, relative to the first frame, augmented or ablated; whether temporal
derivatives are added; the normalization policy; and which statistics, if any, are fitted
train-only.

**The dataset is deliberately not flattened into a fixed `[T,46]` tensor.** Flattening now
would hard-code answers to every question above, and would in particular force a fill value
for NaN before anyone has decided what missingness should mean to the model.

### Verdict on readiness

Every blocking property holds: complete coverage, zero failures, exact frame alignment,
intact sensor contract, intact labels, intact LOSO folds, no fabricated values, and a
documented, geometrically-explained missingness structure with explicit masks. TASK-009
can begin.

## Reproducibility commands

Nothing below re-runs WiLoR. All of it reads the finished run.

```bash
cd /home/hatim/Graduation-Project-Simulation-task008c

# 1. Rebuild the committed lightweight index from the run artifacts.
PYTHONPATH=. python scripts/build_karsl_core28_index.py \
  --manifest datasets/manifests/karsl_core28.csv \
  --run-root /home/hatim/graduation-project-runs/task008-core28-full \
  --output datasets/manifests/karsl_core28_virtual_glove.csv

# 2. Full final QA: coverage, contract, temporal, labels, LOSO, storage,
#    plus the per-sample pass that re-hashes every source video (~105 s).
PYTHONPATH=. python scripts/run_task008c_final_qa.py \
  --manifest datasets/manifests/karsl_core28.csv \
  --data-root /home/hatim/datasets/KArSL-502 \
  --run-root /home/hatim/graduation-project-runs/task008-core28-full \
  --output reports/dataset/TASK-008C-final-qa.json

# 3. Worker status snapshot from the durable state.
PYTHONPATH=. python scripts/run_task008a_karsl_core28.py \
  --manifest datasets/manifests/karsl_core28.csv \
  --run-root /home/hatim/graduation-project-runs/task008-core28-full --status

# 4. Tests.
PYTHONPATH=. python -m unittest discover -s tests -p "test_*.py"
python -m compileall -q evaluation scripts tests
```

Machine-readable statistics: `reports/dataset/TASK-008C-final-qa.json`
(full per-class/per-signer/per-partition breakdowns) and
`reports/dataset/TASK-008C-results.json` (summary).

## Final verdict

| Item | Result |
| --- | --- |
| Requested | 4,222 |
| Successful | 4,222 |
| Failed | 0 |
| Frame integrity | PASS — 4,222/4,222, output length == source frame count, 0 mismatches |
| Overall bend validity | 98.316% (100.000% conditional on a present hand) |
| Overall spread validity | 77.842% (79.175% conditional on a present hand) |
| Overall IMU validity | 98.316% (100.000% conditional on a present hand) |
| LOSO integrity | PASS — 3 folds, no leakage, 28 classes per role, exact coverage |
| Final production size | 2.10 GiB (virtual glove tree 133.7 MiB) |
| Production runtime | 8 h 34 m 56 s (30,896.46 s), peak VRAM 2,683 MiB, 1 model load |
| Tests | 495 passed, 0 failed; compileall 0 errors |

**TASK-008 CORE-28 SENSOR DATASET COMPLETE — READY FOR TASK-009**
