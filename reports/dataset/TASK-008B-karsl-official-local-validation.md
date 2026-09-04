# TASK-008B — Official KArSL Local-Source Verification, Core-28 Freeze, and RTX 3050 Readiness

| | |
|---|---|
| Branch | `opus/task-008b-karsl-official-local-validation` |
| Base | `a7d41b1e0c5ebef6a690c0a17e6ce8a991c52e70` (TASK-008A, **unmerged**) |
| Base dependency | **PR #22 / commit `a7d41b1e...`** — this branch is stacked on it |
| Mapping version | `karsl-core28-v2-official` |
| Date | 2026-09-03 |

---

# Objective

Convert TASK-008A from a candidate, schema-only state into an official,
populated, locally verified KArSL Core-28 dataset contract, and determine from
measurement whether the user's RTX 3050 Laptop 4 GB can realistically run the
full Core-28 extraction.

The full extraction is **not** performed here, no LSTM is trained, and no
TASK-004/005/006 mathematics is touched.

---

# Official Dataset Root

```
/home/hatim/datasets/KArSL-502
```

11 GB on disk, containing signers `01/02/03`, each with `train/` and `test/`,
the per-partition `0001-0070.7z` archives, their extracted `videos/` trees, and
the label workbook. Nothing was downloaded, renamed, reorganized, deduplicated
or copied into the repository.

---

# Official Workbook

`KARSL-502_Labels.xlsx` — 27,141 bytes, sheet `Sheet1`,
503 rows (1 header + 502 data), columns
`['SignID', 'Sign-Arabic', 'Sign-English']`, SignIDs 1–502 contiguous, 0 parse issues.

It is the authority for labels: mirrors, the repository's candidate CSV and
inferred ranges were checked against it, never preferred over it. Values are
preserved verbatim; NFC normalization is used for comparison only.

Full detail, including every label, is in
`reports/dataset/TASK-008B-official-label-verification.md`.

# Workbook SHA-256

```
c13717c549b8cb8cfa465237a3f7dfed73f84149ca5c448b2b253fd96321b14e
```

---

# Official Core-28 Mapping

**Verification: PASS.** 28 classes, SignIDs **0032–0059**, zero label
mismatches against the repository's candidate mapping on both the Arabic and
English fields. TASK-008A's candidate was already correct and required no
correction — it is now *proven* rather than assumed, and frozen as
`karsl-core28-v2-official`.

| SignID | Arabic | English | class index |
|---|---|---|---|
| 0032 | ا | alif | 0 |
| 0033 | ب | baa | 1 |
| … | … | … | … |
| 0058 | و | waaw | 26 |
| 0059 | ي | yaa | 27 |

The complete 28-row table is in the label-verification report.

---

# Extended Letter Mapping

**Verification: PASS.** 11 classes, SignIDs **0060–0070**, all confirmed to be
extended Arabic letter forms: ة, أ, ؤ, ئ, ئـ, ء, إ, آ, ى, لا, ال.

Letters therefore total **39 classes** (28 core + 11 extended, SignIDs 0032–0070).

---

# Number/Digit Mapping

**31 classes**, SignIDs **0001–0031** — not the ~30 the literature suggests. The
workbook is authoritative and the count was not forced to match the literature.

Values: 0–9 (10 classes), 10–90 by tens (9), 100–900 by hundreds (9), then
1000, 1000000, 10000000 (3). The magnitudes jump from 1000 to 1000000, so
10000 and 100000 have no class. All are stored as spreadsheet integers, with
`Sign-Arabic` identical to `Sign-English`.

---

# SignID 0031 Resolution

**SignID 0031 is the numeric magnitude `10000000` (ten million).** An integer
cell, category number — not a letter, not an extended form. It is precisely the
class that makes the number chapter 31 rather than 30.

---

# Extracted Video Layout

Discovered, not assumed:

```
<signer>/<train|test>/videos/<SignID>/<chapter>_<signer>_<SignID>_(DD_MM_YY_HH_MM_SS)_c.mp4
```

Example: `01/train/videos/0041/02_01_0041_(17_11_16_17_32_38)_c.mp4`

| Field | Meaning |
|---|---|
| `<chapter>` | `01` for SignIDs 1–31, `02` for 32–70 — corroborates the workbook's own split |
| `<signer>` | signer id; matches the directory in **all 10,584** files |
| `<SignID>` | class id; matches the directory in 10,551 of 10,584 (see integrity) |
| `(…)` | recording timestamp `DD_MM_YY_HH_MM_SS`, the de-facto repetition identity |
| `_c` | colour modality — the only modality present |

The archive inserts a `videos/` directory between the partition and the class,
which TASK-008A's layout parser could not express. That was fixed (see
*Local-Source Workflow*).

---

# Source Integrity

All 10,584 videos were probed with `ffprobe` and hashed.

| Check | Result |
|---|---|
| Probe failures | **0** |
| Resolution | 1920×1080 for all 10,584 |
| Frame rate | 30.0 fps for all 10,584 |
| Codec / container | h264 / mp4 for all 10,584 |
| Missing files | 0 |
| Corrupt files | 0 |

Metadata reliability was validated rather than trusted: a decode-count check on
a deterministic sample found **0 mismatches** between container frame counts and
fully decoded frame counts, so container probing was used for the bulk audit
instead of decoding 247,766 frames.

## Two upstream anomalies — both outside Core-28

**1. Filename/directory SignID disagreement — 33 files.** All in
`01/*/videos/0007/`, all named `..._0001_...`, all from one contiguous recording
session. The directory is treated as authoritative (that is how the official
archive organises classes) and the disagreement is recorded per row as
`path_sign_id_matches_filename=false` rather than silently resolved.

**2. Byte-identical duplicate videos — 50 groups, 100 files.** Every group pairs
SignID **0027** with **0028**, signer 01 only, same content hash. This also
explains SignID 0027 carrying 200 videos where the modal class has 150.

**Neither anomaly touches Core-28 (0032–0059): 0 affected groups, 0 affected
files.** Both are in the number chapter and are reported rather than repaired —
no source file was renamed, moved or deleted.

For completeness, 20 of the 70 SignIDs deviate from the modal 150 videos, the
largest being 0027 (200), 0005 (183), 0006 (117, signer 01 contributing only 17
of an expected 50) and 0059 (160, in Core-28).

---

# Core-28 Exact Size

| Quantity | Value |
|---|---|
| Classes | **28** |
| Videos | **4,222** |
| Bytes | 1,794,280,889 (1.671 GiB) |
| Frames | **83,659** |
| Duration | 0.77 h |

## By signer

| Signer | Videos | Frames | GiB | Median frames |
|---|---|---|---|---|
| S01 | 1,402 | 29,947 | 0.662 | 21 |
| S02 | 1,401 | 19,245 | 0.189 | 13 |
| S03 | 1,419 | 34,467 | 0.819 | 24 |

Signer 02's clips are markedly shorter and smaller (median 13 frames, 0.189 GiB)
than S01 (21, 0.662) and S03 (24, 0.819). Any per-signer throughput or LOSO
result should be read with that in mind.

## By partition

| Partition | Videos | Frames | GiB |
|---|---|---|---|
| train | 3,550 | 70,232 | 1.401 |
| test | 672 | 13,427 | 0.270 |

Per-class video counts range 149–160 across the 28 classes (median 150). Full
per-class figures are in `TASK-008B-results.json`.

---

# Full Letter Exact Size

39 classes (0032–0070): **5,876 videos**,
123,607 frames, 2.448 GiB,
1.14 h.

Of which the 11 extended forms are 1,654 videos,
39,948 frames, 0.777 GiB.

---

# Number/Digit Exact Size

31 classes (0001–0031): **4,708 videos**,
124,159 frames, 2.609 GiB,
1.15 h.

The number chapter carries **more frames than all 39 letter classes combined**
(124,159 vs 123,607) from fewer videos, because its clips are longer
(mean 26.4 vs 21.0 frames).

---

# Full 0001–0070 Size

| Quantity | Value |
|---|---|
| Classes | 70 |
| Videos | **10,584** |
| Bytes | 5,429,976,694 (5.057 GiB) |
| Frames | **247,766** |
| Duration | 2.29 h |

---

# Core-28 Frame Statistics

| Statistic | Frames per video |
|---|---|
| Mean | 19.8 |
| Median | 20 |
| p95 | 29 |
| Min | 9 |
| Max | 70 |

**This is the headline correction of this task.** Earlier planning assumed
roughly 208,912 Core-28 frames. The measured figure is **83,659** — about
2.5x fewer — because KArSL letter clips are very short (median 20 frames
≈ 0.67 s at 30 fps). Every runtime and storage estimate below rests on the
measured number, not the earlier assumption.

Sequence lengths are reported, not modified: no padding, truncation, resampling
or interpolation is performed anywhere in TASK-008B.

---

# Manifest

`datasets/manifests/karsl_core28.csv` — **populated, no longer schema-only.**

| Check | Result |
|---|---|
| Rows | **4,222** |
| Classes | 28 |
| Signers | ['01', '02', '03'] |
| Duplicate sample IDs | **0** |
| Duplicate source paths | **0** |
| Unknown labels | **0** |
| Invalid signers | **0** |
| Invalid partitions | **0** |
| Missing files | **0** |
| Rows without a frame count | **0** |
| Absolute paths committed | **0** — portable **PASS** |

Every row carries `sample_id`, `sign_id`, `label_ar`, `label_index`,
`signer_id`, `official_partition`, `repetition_id`, `source_relative_path`,
`source_sha256`, `source_size_bytes`, `frame_count`, `fps`, `duration_seconds`
and `skeleton_available`. Paths are relative beneath the configurable dataset
root; no `/home/hatim/...` appears in any committed file.

Nothing was fabricated to fill a gap: there were no missing or corrupt files to
substitute for.

---

# LOSO Splits

Regenerated from the populated manifest.

| Fold | Train | Validation | Test | Sum | Classes (tr/va/te) | Held-out leakage |
|---|---|---|---|---|---|---|
| S01 | 2,372 | 448 | 1,402 | 4,222 | 28/28/28 | **0** |
| S02 | 2,373 | 448 | 1,401 | 4,222 | 28/28/28 | **0** |
| S03 | 2,355 | 448 | 1,419 | 4,222 | 28/28/28 | **0** |

Every fold: the held-out signer appears **zero** times in train or validation
and is exactly the test set; all 28 classes are present in all three roles;
**0** unassigned samples and **0** duplicate assignments; the three roles sum to
the full 4,222-row manifest. Remaining signers follow official train → train and
official test → validation, with the pre-existing deterministic class-coverage
repair where a class would otherwise be missing from one role.

---

# Local-Source Workflow

The dataset is already downloaded, so nothing is fetched. Existing options
express this directly and no new CLI surface was invented:

```bash
python scripts/run_task008b_validation.py \
  --dataset-root /home/hatim/datasets/KArSL-502 \
  --work-dir /home/hatim/graduation-project-runs/task008b
```

Two minimal fixes were required in TASK-008A's discovery, both in
`_locate_layout`:

1. **An intervening directory is now tolerated.** The official archive places
   `videos/` between the partition and the class, which the parser could not
   express — it required the three tokens to be adjacent and raised on every
   real path. At most one intervening segment is accepted, so the older flat
   layout still parses and genuine ambiguity still raises.
2. **Non-Core-28 paths are skipped instead of fatal.** The official root also
   holds the number and extended-letter chapters. The function's own docstring
   says it discovers "only valid Core-28 RGB videos", so skipping them is the
   intended behaviour; raising was a bug that only appeared once the root was
   broader than Core-28.

No data is duplicated into the repository, and the authoritative dataset stays
outside Git.

---

# Smoke Manifest

`datasets/manifests/karsl_core28_smoke.csv` — **24 videos**, deterministic.

| Property | Value |
|---|---|
| Signers | ['01', '02', '03'] (all three) |
| Partitions | ['test', 'train'] (both) |
| Distinct classes | 12 |
| Total frames | 592 |
| Frame range | 9 – 70 (median 21) |

**Selection rule** (a pure function of the manifest, no randomness): keep rows
with a known positive frame count; form the six (signer × partition) cells; take
the shortest, median and longest sequence in each; fill the remaining budget
round-robin outward from each cell's median so added rows stay representative
rather than extreme; emit sorted by signer, partition, frame count, sample_id.

Length spread is deliberate — this subset is also the throughput benchmark, and
a uniform-length subset would misestimate the full run.

---

# RTX 3050 Environment

| Property | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| VRAM | 4 GB (4096 MiB) |
| Driver | 610.62 |
| CUDA | 13.0 |
| PyTorch | 2.13.0+cu130 |
| Checkpoint | `wilor_final.ckpt` |
| Precision | **FP32** (`fast_mode=False`) |
| CPU cores | 16 |

No FP16, mixed precision, quantization, alternate checkpoint or different model
was enabled. This is the production configuration.

---

# RTX 3050 Benchmark

The deterministic 24-video subset was run through the complete frozen pipeline
(RGB → WiLoR → TASK-004 → TASK-005 → TASK-006).

| Metric | Value |
|---|---|
| Videos | 24 |
| Frames | 592 |
| Failures | **0** |
| Model load (once) | 47.1 s |
| Worker wall time | 215.2 s |
| Compute time | 168.0 s |
| **Peak VRAM** | **2.62 GiB of 4.00 GiB** |
| Videos/s (excl. load) | 0.143 |

Peak VRAM at 2.62 GiB leaves roughly 1.4 GiB
headroom, consistent with the earlier ~2.8 GB pilot figure. **4 GB is
sufficient**; the card is not memory-constrained for this workload.

---

# WiLoR Throughput

| Rate | FPS |
|---|---|
| WiLoR (POSE) only | **3.712** |
| Full pipeline compute | **3.525** |
| End-to-end excl. model load | 3.522 |
| End-to-end incl. model load | 2.750 |

Per-video POSE throughput varies **1.94 –
6.09 FPS** (median
3.96), so no single clip is used as the basis
for extrapolation. The measured rate is close to, and slightly above, the
earlier ~3.32 FPS pilot figure — but it was re-measured rather than assumed.

---

# Stage Runtime Breakdown

| Stage | Seconds | % of compute | FPS | ms/frame |
|---|---|---|---|---|
| POSE | 159.46 | 94.94% | 3.71 | 269.36 |
| TRACKING | 7.30 | 4.34% | 81.14 | 12.32 |
| KINEMATICS | 0.99 | 0.59% | 600.99 | 1.66 |
| VIRTUAL_GLOVE | 0.22 | 0.13% | 2699.91 | 0.37 |
| serialization / other | 0.13 | 0.08% | — | — |

**WiLoR inference is 94.9% of the compute
cost.** Everything downstream — tracking, kinematics and the virtual glove
combined — is 5.1%. Optimising
anything other than pose inference cannot meaningfully change the total.

---

# Full Core-28 Runtime Estimate

Extrapolated from the measured throughput and the **measured**
83,659 Core-28 frames across 4,222 videos.

Two independent methods:

| Method | Estimate |
|---|---|
| Frames ÷ full-pipeline FPS | 6.59 h |
| Per-video cost model (fit on 24 videos, R²=0.906) | **6.52 h** |

The cost model fits `time = -0.339 s + 0.2977 s/frame`.
The intercept is slightly negative, which is a fitting artefact rather than a
physical quantity — its honest reading is that **per-video fixed overhead is
indistinguishable from zero**, so cost is essentially proportional to frames at
a marginal 3.36 FPS. The two methods agreeing to within 1%
is the useful result.

| | |
|---|---|
| **WiLoR-only estimate** | **6.3 h** |
| **Complete-pipeline best estimate** | **6.5 h** |
| **Conservative range** | **6.5 – 8.2 h** |

The conservative bound adds 25% for sustained thermal throttling on a laptop
GPU, competing desktop load, and process restarts (each costing
47 s of model load). It is a judgement, not a
measurement — the smoke run was under 3 minutes and cannot evidence hour-scale
thermal behaviour.

## COUPLE-OF-HOURS FEASIBLE: **NO**

At a measured 3.52 FPS full-pipeline throughput,
83,659 frames require **6.5 hours**, with
**6.5–8.2 h** the realistic planning range. Reaching
two hours would need roughly 11.6 FPS — about
3.3× the measured rate — which is
not achievable by scheduling changes, since 95%
of the cost is a single FP32 model whose precision this task is not permitted to
change.

The run is nevertheless practical: it is I/O-light, fits in 4 GB VRAM, and
**resume now works correctly** (below), so it can be completed across several
sessions rather than one uninterrupted sitting.

---

# Full Core-28 Storage Estimate

Measured from the smoke run's actual output, scaled by frames.

| Artifact | Bytes/frame | Core-28 estimate |
|---|---|---|
| WiLoR pose | 21,891 | 1.706 GiB |
| TASK-004 tracking | 1,963 | 0.153 GiB |
| TASK-005 kinematics | 602 | 0.047 GiB |
| TASK-006 virtual glove | 1,420 | 0.111 GiB |
| state + provenance | 13 | 0.001 GiB |
| **Total run output** | | **2.02 GiB** |

| | |
|---|---|
| Source videos (already on disk) | 1.67 GiB |
| Estimated run output | 2.02 GiB |
| Source + run | 3.69 GiB |
| **Minimum safe free space** | **5 GiB** |
| **Recommended free space** | **7 GiB** |

The earlier rough 60–100 GB figure is superseded by measurement and was high by
more than an order of magnitude — again because the clips are short. Storage is
not a constraint for Core-28.

---

# Progress UI

The TASK-008A runner already emits a single updating line carrying every
required field. Verified live during the benchmark:

```
TASK-008A shard 00/1: 24/24 (100.0%) current=karsl_core28_s03_sign0038_train_rep001
  stage=VIRTUAL_GLOVE success=24 failed=0 frames=592 rolling_fps=3.52
  elapsed=00:02:48 eta=00:00:00
```

| Requirement | Status |
|---|---|
| completed / total videos | PASS |
| percentage | PASS |
| current sample | PASS |
| current pipeline stage | PASS |
| successful / failed counts | PASS |
| frames processed | PASS |
| rolling WiLoR FPS | PASS |
| elapsed | PASS |
| ETA | PASS |

A unit test asserts each token is present so the display cannot silently
regress.

---

# Resume / Interruption Validation

Resume was exercised against real artifacts, and **two genuine defects were
found and fixed**. Both would have mattered a great deal over a
7-hour run.

## Defect 1 — `--resume` never skipped POSE

The raw pose NPZ stores **one row per detected hand**, so a two-hand video has
twice as many rows as frames (42 rows for 21 frames). Validation compared
`len(frame_index)` against the sidecar's video-frame count, so the check failed
for essentially every completed sample and the most expensive stage —
95% of the cost — was silently recomputed
on every resume. Fixed by comparing the number of **distinct** frame indices.

## Defect 2 — a truncated artifact aborted the sample

An interrupted write leaves a partial `.npz`, and `numpy.load` raises
`zipfile.BadZipFile`, which is not an `OSError` and so escaped the handler. The
sample hard-failed instead of recomputing the damaged stage — exactly the state
an interruption leaves behind. Fixed so any unreadable artifact means "not
usable, recompute" and can never abort a sample.

## Measured behaviour after the fixes

A 6-sample run was completed, one sample's virtual-glove NPZ was truncated to 5
bytes and another's kinematics directory deleted, then the run was resumed over
12 samples:

| Outcome | Count |
|---|---|
| Samples fully skipped (all 4 stages) | **11** |
| Samples recomputing only the damaged stage | **1** |
| Samples fully recomputed | 0 |
| Failures | **0** |

Wall time fell from 78 s to **16.7 s**, most of which is model load.

| Check | Result |
|---|---|
| Interruption/resume: completed valid artifacts skipped | **PASS** |
| Incomplete/corrupt artifact safely recomputed | **PASS** |
| Source and provenance validated (manifest hash, source SHA-256, upstream SHA-256) | **PASS** |
| No duplicate outputs generated | **PASS** |
| `--retry-failed` absent → failed sample left alone (`SKIPPED_FAILED`) | **PASS** |
| `--retry-failed` present → failed sample recovered | **PASS** |

The retry path was exercised by injecting a synthetic `FAILED` state into the
store, so the behaviour was tested rather than assumed.

## Model lifetime

`model_loads = 1` for the whole 24-video shard, at
47.1 s, constructed once before the sample loop and
reused throughout. There is no load/destroy cycle per video.

---

# Tests

```
python -m unittest discover -s tests -p 'test_*.py'
Ran 453 tests
OK

python -m compileall -q evaluation tracking kinematics virtual_glove scripts tests
0 errors
```

**453 passed, 0 failed, 0 errors.** 56 are new TASK-008B tests covering official
workbook parsing, Core-28 verification (including detection of a deliberately
wrong label and a wrong ID range), extended-letter and number-boundary
verification, the SignID 0031 resolution, real filename and layout parsing, local
discovery and determinism, portable manifest paths, deterministic smoke
selection, LOSO generation and leakage rejection, the committed artifacts, Git
exclusion of dataset media, both resume defects, and the progress display.

One pre-existing TASK-008A test asserted the manifest was *schema-only*; since
populating it is this task's purpose, it was updated to assert the new contract
(populated and portable) rather than deleted.

GPU throughput itself is measured in this report, not asserted in a unit test.

---

# Limitations

1. **The benchmark is 24 videos over ~3 minutes.** It cannot evidence sustained
   thermal behaviour over 7 hours; the conservative +25% band is
   a judgement, not a measurement.
2. **The smoke subset averages 24.7 frames/video against Core-28's
   19.8.** Length spread was chosen deliberately, but the mismatch means
   the extrapolation is approximate. The two independent estimation methods
   agreeing to within 1% is reassurance, not proof.
3. **Per-video fixed overhead could not be separated from per-frame cost** at
   this sample size — the fitted intercept is slightly negative. It is reported
   as indistinguishable from zero rather than dressed up as a measurement.
4. **The two upstream anomalies are reported, not repaired.** No source file was
   renamed, moved or deleted. Anyone using the number chapter must handle them.
5. **Repetition identity is the recording timestamp**, and `repetition_id` is a
   deterministic ordinal within each (signer, partition, class) group. It is not
   an official repetition index; no such field exists in the source.
6. **`skeleton_available` is `unknown`** for every row: only the colour modality
   is present locally.
7. **No temporal tensorization was performed** — no padding, truncation,
   resampling or interpolation. Sequence lengths are preserved and reported, and
   TASK-009 will freeze that policy.
8. **The 6.5-hour estimate assumes a single sequential worker.** Whether
   sharding helps on one 4 GB GPU was not tested, and concurrent shards would
   contend for the same VRAM.

---

# Recommendation

**Proceed to the full Core-28 extraction when authorized**, planning for a
**6.5–8.2 hour** run across multiple sessions rather
than one sitting. Resume is now trustworthy and per-stage precise, so
interruption is cheap: restarting costs only the 47 s
model load plus whatever single stage was in flight.

Free at least **5 GiB** (recommended
**7 GiB**). Storage and VRAM are both
comfortable; wall-clock time is the only real cost.

The extraction was **not** started, per instruction.

---

# Verdict

**TASK-008B OFFICIAL KArSL CORE-28 FROZEN — LOCAL EXTRACTION READY**
