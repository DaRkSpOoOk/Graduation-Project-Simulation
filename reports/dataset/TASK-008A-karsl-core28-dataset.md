# Task

TASK-008A — KArSL Core-28 dataset acquisition, Ibex/SLURM extraction
infrastructure, and sensor-dataset pipeline.

# Branch

`luna/task-008a-karsl-core28-dataset`, based on main
`8cafff48c53522bd9e458f56b47507d3a025930d`.

# Scope

This branch prepares the reproducible infrastructure for the first large
Arabic-alphabet virtual-glove dataset:

```text
official KArSL RGB video
  -> WiLoR + MANO
  -> frozen TASK-004 tracking
  -> frozen TASK-005 kinematics
  -> frozen TASK-006 ideal virtual glove
  -> one natural-length virtual-glove sequence per source video
```

It does not run the approximately 4,200-video extraction, train a recognizer,
build an LSTM tensor, create word/continuous-fingerspelling data, or modify
TASK-004, TASK-005 or TASK-006 mathematics. The full Core-28 production run is
therefore intentionally `PENDING`.

# Approach

The implementation separates source acquisition, portable manifests, signer
splits, external stage artifacts and SLURM orchestration. Source RGB and model
outputs remain outside Git. Workers use deterministic manifest-order modulo
shards, persist per-stage provenance and state, and validate an existing stage
before `--resume` skips it.

# Evidence / Sources

The authoritative research sources are the [official KArSL project
page](https://hamzah-luqman.github.io/KArSL/), the [official KArSL
repository](https://github.com/Hamzah-Luqman/KArSL), the [official RGB download
page](https://hamzah-luqman.github.io/KArSL/download_video_502.html), and the
[original KArSL paper](https://doi.org/10.1145/3423420). The official page
describes KArSL-502 as an isolated-sign RGB/depth/skeleton dataset with three
signers and repeated samples, and links the label workbook plus signer/split
RGB resources.

Retrieval date: `2026-09-02` (Asia/Riyadh). At that retrieval the current
official SharePoint label and RGB links returned an administrator-disabled
page. The URLs are retained for provenance; no mirror was substituted and no
bulk KArSL asset was downloaded. The downloader rejects HTML/folder responses
when a direct binary catalog entry is supplied.

The project has permission, as supplied for this task, to obtain data from
[SSHI](https://sshi.sa/), including scraping/data collection as necessary. No
SSHI scraper is built here; that belongs in a later dedicated task.

# Official Labels

The official label target is `KARSL-502_Labels.xlsx`, linked from the official
KArSL page. `evaluation/dataset/core28.py` implements a dependency-free XLSX
reader and validates SignID, Arabic label and English gloss. It uses NFC plus
`strip()` only for equality checks and does not rewrite stored Arabic labels.

The official workbook could not be retrieved during development because its
current link was disabled. Consequently the committed mapping is an
auditable, frozen candidate and not a claim that official source verification
has completed. `scripts/download_karsl_core28.py --labels-only` must succeed
and create `official_labels_verification.json` before production discovery is
allowed.

# Core-28 Mapping

The mapping freezes exactly 28 standard Arabic alphabet entries as SignIDs
`0032`--`0059`, with deterministic label indices `0`--`27`:

| SignID | Arabic | English gloss | index |
|---:|:---:|:---|---:|
| 0032 | ا | alif | 0 |
| 0033 | ب | baa | 1 |
| 0034 | ت | ta | 2 |
| 0035 | ث | tha | 3 |
| 0036 | ج | Jiim | 4 |
| 0037 | ح | Haa | 5 |
| 0038 | خ | kha | 6 |
| 0039 | د | daal | 7 |
| 0040 | ذ | thal | 8 |
| 0041 | ر | raa | 9 |
| 0042 | ز | zay | 10 |
| 0043 | س | siin | 11 |
| 0044 | ش | shiin | 12 |
| 0045 | ص | Saad | 13 |
| 0046 | ض | Daad | 14 |
| 0047 | ط | Taa | 15 |
| 0048 | ظ | Zaa | 16 |
| 0049 | ع | Ayn | 17 |
| 0050 | غ | ghayn | 18 |
| 0051 | ف | faa | 19 |
| 0052 | ق | qaaf | 20 |
| 0053 | ك | kaaf | 21 |
| 0054 | ل | laam | 22 |
| 0055 | م | miim | 23 |
| 0056 | ن | noon | 24 |
| 0057 | ه | haa | 25 |
| 0058 | و | waaw | 26 |
| 0059 | ي | yaa | 27 |

The machine-readable mapping is committed at
`datasets/manifests/karsl_core28_labels.csv`. The candidate membership was
cross-checked against a publicly visible workbook export only as a research
aid; it is not an approved acquisition source and is not included in the
download tooling.

# Extended Letter Candidates

The additional letter-like candidates are retained separately and excluded
from this task: SignIDs `0060`--`0070` (`ة`, `أ`, `ؤ`, `ئ`, `ئـ`, `ء`, `إ`, `آ`,
`ى`, `لا`, `ال`). They are in
`datasets/manifests/karsl_extended_letter_candidates.csv` and in the mapping
document emitted by the code. They are reserved for a possible future
extended/full-vocabulary task and are not sent to the Core-28 extraction.

# Skeleton Modality Investigation

The official project description advertises skeleton data and the dataset is
described as Kinect V2-captured in the research context. The accessible
distribution at retrieval did not provide a usable skeleton archive or a
verified file-format, joint-order, unit or RGB-synchronization specification.
The committed manifest therefore records `skeleton_available=unknown`; no
unverified skeleton file is accepted as production input.

If a verified skeleton distribution becomes available, it may be recorded as
auxiliary metadata for future left/right validation, crop priors or wrist
trajectories. It does not replace WiLoR's detailed per-finger geometry and does
not change TASK-004 or TASK-005.

# Data Storage

The intended Ibex roots are configurable and external to Git:

```text
/ibex/user/$USER/graduation-project-data/karsl/
/ibex/user/$USER/graduation-project-runs/task008a-karsl-core28/
```

The documented subdirectories are `labels`, `archives`, `raw`, `skeleton`,
`depth`, `checksums`, `acquisition`, and run-stage directories for pose,
tracking, kinematics, virtual glove, state, failures, QA and export.

# Git Exclusion

`.gitignore` now protects repository-local raw/external/download/cache,
archive, raw-frame, skeleton/depth, generated artifact, run, log, checkpoint,
model and common video/array/archive extensions. Manifests, splits, reports,
scripts and SLURM files remain trackable. `tests/test_task008a_dataset.py`
executes `git check-ignore` for a dummy raw video and confirms a manifest path
is not ignored.

# Downloader

`scripts/download_karsl_core28.py` supports:

- `--data-root`, `--manifest`, `--labels-file`, `--source-catalog`,
  `--video-root`, and `--split-dir` for portable roots;
- `--labels-only` for official workbook retrieval and exact Core-28
  validation;
- `--source-catalog` for direct binary RGB assets, with `.part` files,
  HTTP Range resume, retries, non-zero checks, progress, expected-size and
  SHA-256 checks;
- `--discover` for deterministic recursive RGB discovery and manifest/split
  construction;
- `--verify` for decoder, size, frame, FPS, duration and declared hash checks;
- `--status` for read-only acquisition status;
- `--dry-run` for source/preflight reporting without downloading or writing;
- `--retry-failed` to continue through catalog failures and record them.

Folder/share links are not treated as video files. The committed
`datasets/manifests/karsl_core28_source_catalog.csv` is intentionally a
schema-only catalog because direct binary URLs were unavailable at retrieval;
it must be populated from the official distribution before a production run.
The preflight reports known expected bytes, existing bytes, remaining bytes,
free space and whether known requirements fit.

# Raw Dataset Integrity

For each discovered video the manifest records a deterministic sample ID,
source dataset/version/modality, SignID/labels/index, signer, official
partition, repetition, relative path/file name, source URL identifier,
SHA-256, size and video metadata when readable. Validation rejects missing,
empty, duplicate, corrupt, non-RGB, non-Core-28, invalid-signer,
invalid-partition, nonportable or checksum-mismatched rows.

Nominal volume is approximately `28 × 3 × 50 ≈ 4,200` sequences, but the
eventual discovered/readable manifest is authoritative. Current development
discovery count: `0`; the committed manifest is a header-only schema because
the official source was inaccessible. Missing, duplicate and corrupt counts
are therefore `PENDING` until official assets are available.

# Manifest

Committed files:

- `datasets/manifests/karsl_core28.csv` — portable schema-only source manifest;
- `datasets/manifests/karsl_core28_labels.csv` — 28-row mapping candidate;
- `datasets/manifests/karsl_extended_letter_candidates.csv` — excluded
  letter-like candidates;
- `datasets/manifests/karsl_core28_source_catalog.csv` — direct-binary catalog
  schema;
- `datasets/manifests/karsl_core28_virtual_glove.csv` — output-index schema.

The manifest has exactly 28 label classes, no duplicate rows or sources in the
template, and uses relative POSIX paths below the configured data root. The
sample count is currently `0` because no source video was downloaded.

# Signer-Independent LOSO Splits

`evaluation/dataset/splits.py` implements three deterministic folds:

```text
Fold S01: S01 test; S02/S03 train+validation
Fold S02: S02 test; S01/S03 train+validation
Fold S03: S03 test; S01/S02 train+validation
```

For the remaining signers, official `train` rows become train and official
`test` rows become validation. If an official partition lacks a class, the
smallest deterministic move from the other remaining role is attempted only
when the donor retains at least one instance. Held-out signers can never be
moved into train or validation. The split validator checks full manifest
coverage, role uniqueness, class coverage, duplicate assignment and leakage.

The three committed split files are schema-only with `0` rows pending source
discovery. Synthetic tests verify deterministic assignment, complete
per-fold assignment and zero held-out-signer leakage.

# Frozen Extraction Pipeline

The orchestrator calls the existing production entry points in this order:

```text
RGB video
  -> pose.wilor.video_processing.process_video_full (full FP32)
  -> tracking.wilor
  -> kinematics
  -> virtual_glove
```

No algorithm is reimplemented here. The raw pose remains in a distinct
`pose/raw` stage; tracking, kinematics and virtual-glove stages are separate
natural-length artifacts. No smoothing, interpolation, forward filling,
cross-hand copying, resampling, padding or truncation is performed.

# Ibex Environment

`slurm/README.md` documents cloning the reviewed branch, checking
`git rev-parse HEAD`, creating an external environment and selecting a
compatible CUDA module. The implementation does not hardcode an account,
QOS, partition, user name or CUDA version. Worker provenance records Python,
NumPy, PyTorch/CUDA, TF32 flags, CPU model/RAM when available, GPU query
output, SLURM IDs, Git commit/branch, WiLoR upstream/checkpoint identity,
full-FP32 mode, TASK-005/TASK-006 contracts, roots and manifest hash.

The validated WiLoR operating contract remains normal FP32 with the existing
model/checkpoints, detector confidence `0.3` and rescale factor `2.0`. No
FP16, mixed precision, quantization or alternate weights are enabled by this
branch. Actual A100 compatibility and throughput remain unmeasured here.

# SLURM Strategy

The scripts are:

- `slurm/task008a_smoke_a100.slurm` — one-worker deterministic smoke path;
- `slurm/task008a_core28_a100.slurm` — default `0-15%4` production array;
- `slurm/task008a_final_qa.slurm` — CPU-only raw/source/run QA and index.

Defaults are one GPU/worker, preferred `a100` constraint, eight CPU cores and
approximately 32 GB RAM for extraction. Partition/account/QOS and log paths
are overrideable with `sbatch`; no institutional account assumption is in
code. The full job is not submitted by this development task.

# Deterministic Sharding

For a fixed manifest file and positive `num_shards`, row position in the
validated manifest is assigned with:

```text
shard_index = manifest_position % num_shards
```

Rows are consumed in CSV order, which is itself deterministic after discovery.
The runner rejects invalid array indices and a sample ID assigned to another
shard. Unit tests verify repeatability, complete coverage and no cross-shard
collision. The default is 16 shards with at most four active workers.

# WiLoR Model Reuse

`run_worker` constructs the full WiLoR pipeline once after preflight and
reuses that object for every assigned sample. It does not load/destroy the
model once per video. Status, dry-run and verify-only modes do not import or
construct WiLoR. Workers preserve the frozen normal FP32 configuration.

# Progress Reporting

Each worker reports shard, completed/total, percentage, current sample, stage,
success/failure count, decoded frames, rolling FPS, elapsed time and ETA once
per sample rather than once per frame. The downloader reports received/total
bytes where available.

# Global Status

```bash
python scripts/run_task008a_karsl_core28.py \
  --manifest datasets/manifests/karsl_core28.csv \
  --run-root "$TASK008A_RUN_ROOT" --status
```

Status reads only state JSON files and the manifest. It reports total,
complete, pending, failed, in-progress, per-stage counts, processed frames and
per-shard updates. It does not load WiLoR or modify artifacts. ETA is printed
by the worker where a completed-sample rate is defensible; global status does
not invent an ETA when it lacks a trustworthy elapsed-time basis.

# Resumability

Each shard owns `state/shard-XX.json`. Each sample has a status of `PENDING`,
the stage-completion values, or `FAILED`, plus attempts, retry count, current
stage, frame count and last error. Stage sidecars bind sample ID, source path,
source SHA-256, manifest SHA-256, stage schema, upstream artifact hash and
frame count. Resume accepts an artifact only after sidecar, schema, full-mode
pose, source and upstream checks pass. A missing/corrupt/mismatched artifact
is not trusted solely because its file exists.

# Failure Recovery

An exception is recorded in both state and shard JSONL failure output with
sample, shard, stage, error type/message, UTC timestamp and retry count. The
worker continues to the next sample. `--resume --retry-failed` retries failed
samples while preserving valid earlier stages. Individual or multiple SLURM
array indices can be resubmitted without changing shard ownership.

# HPC Smoke-Test Plan

Before the full array, use a deterministic 18--24-row Core-28 smoke manifest
covering all signers, multiple letters, relevant official partitions and the
observed one-/two-hand variation. The smoke job runs the same WiLoR →
tracking → kinematics → virtual-glove stages and then CPU QA. Record wall
time, decoded frames, WiLoR FPS, video throughput, peak GPU memory when
available, failures and schema/provenance QA. A systematic environment or
checkpoint error blocks the full submission.

# A100 Benchmark Method

The A100 estimate is intentionally empirical. Measure the smoke run's decoded
frames per elapsed second and video throughput under the complete software
boundary (decode, preprocessing, inference, reconstruction, conversion and
serialization accounting). Extrapolate only after the smoke run, using actual
discovered frame counts and the intended concurrent-worker count. The laptop
RTX 3050 pilot is not used as an A100 performance claim. Runtime is operational
information, not a scientific acceptance threshold.

# Full Extraction Procedure

After official label verification, direct RGB catalog population, manifest
integrity validation and smoke success:

```bash
sbatch slurm/task008a_core28_a100.slurm
squeue -u "$USER"
sacct -j <ARRAY_JOB_ID> --format=JobID,State,Elapsed,AllocTRES,MaxRSS,ExitCode
python scripts/run_task008a_karsl_core28.py --run-root "$TASK008A_RUN_ROOT" --status
```

Use `sbatch --array=7 ...` or `sbatch --array=3,7,11 ...` for recovery, retain
the same shard count, and use `--resume --retry-failed` as documented. The
laptop may disconnect or shut down after submission; artifacts and state live
on Ibex.

# QA Procedure

`slurm/task008a_final_qa.slurm` runs source integrity validation,
`scripts/validate_karsl_core28.py`, and
`scripts/build_karsl_core28_index.py`. QA requires every requested source
sample to have valid pose, tracking, kinematics and virtual-glove artifacts;
checks source hashes, frame/timestamp/state/detection provenance alignment,
normalization, masks, orientation finiteness, per-class/signer success and
natural sequence-length statistics. It reports failures rather than deleting
difficult classes/signers/videos.

# Sensor Dataset Schema

Per source video, the authoritative external output remains natural-length:

```text
virtual_glove/<sample_id>/virtual_glove.npz
virtual_glove/<sample_id>/virtual_glove_meta.json
virtual_glove/<sample_id>/sensor_layout.json
```

The committed output-index schema is
`datasets/manifests/karsl_core28_virtual_glove.csv`. It contains portable
sample/source/label fields, source frame count, stage statuses and bend,
spread and IMU validity fractions. No machine-specific run root is placed in
that manifest.

# ML Feature Contract

TASK-006-ideal-virtual-glove-v1 remains authoritative: per hand there are 15
independent bend channels, 4 adjacent spread channels and one palm IMU package
(19 Hall/magnetic channels and 20 physical sensing packages). The primary
future numerical representation is 15 bend-normalized + 4 spread-normalized +
4 WXYZ quaternion values = 23 values per hand/timestep, or 46 for two hands,
plus masks. Optional ADC and derived body-frame gyro are preserved as
non-authoritative diagnostics; the retired 850--1700 prototype range is not
used. Accelerometer remains deferred. This task does not build a final LSTM
tensor.

# Export Procedure

After real extraction and QA, retain virtual-glove files, manifests, splits,
QA, provenance and compact failure/log summaries in an external export. Raw
RGB is not required for laptop transfer; pose intermediates may remain on
Ibex if provenance permits regeneration. A reproducible archive command is:

```bash
ARCHIVE="${TASK008A_EXPORT_ARCHIVE:-$TASK008A_RUN_ROOT/../task008a-core28-export.tar.zst}"
tar -I 'zstd -T0 -10' \
  -cf "$ARCHIVE" \
  -C "$TASK008A_RUN_ROOT" virtual_glove export qa provenance failures \
  -C "$TASK008A_REPO_ROOT" datasets/manifests datasets/splits reports/dataset
sha256sum "$ARCHIVE"
```

Verify the archive hash after download and record the exact member list.

# Implementation

Primary new code is under `evaluation/dataset/`:

- `core28.py` — source URLs, label parsing/validation and mapping;
- `manifest.py` — portable deterministic video manifest and integrity fields;
- `splits.py` — signer-independent LOSO construction/validation;
- `acquisition.py` — resumable direct-binary download/preflight primitives;
- `orchestrator.py` — stages, sidecars, state, provenance, sharding,
  progress/status and worker lifecycle;
- `qa.py` — extractor-independent external run QA.

CLI scripts and SLURM wrappers are kept separate from frozen model packages.

# Files Changed

`.gitignore`, `datasets/README.md`, the Core-28/candidate/index and split
templates, `research/datasets/karsl_core28.md`, `evaluation/dataset/*`,
`scripts/{download_karsl_core28,run_task008a_karsl_core28,build_karsl_core28_index,validate_karsl_core28}.py`,
`slurm/*`, `tests/test_task008a_dataset.py`, this report, and the compact
infrastructure JSON result.

# How to Run

Development-safe checks:

```bash
python scripts/download_karsl_core28.py --data-root /tmp/karsl --labels-only --dry-run
python scripts/run_task008a_karsl_core28.py \
  --manifest datasets/manifests/karsl_core28.csv \
  --run-root /tmp/task008a-run --status
```

Official-source/HPC workflow and recovery commands are in `slurm/README.md`.
The full extraction command is intentionally documented only; it was not run
in this environment.

# Evaluation

The test suite uses synthetic rows/files and never requires KArSL, WiLoR,
MANO, a checkpoint or a GPU. It exercises exact class mapping, CSV schema,
portable deterministic discovery, duplicate/path/label rejection, LOSO
leakage, modulo shards, Git exclusion, stage validation, state reload, failed
sample retry and read-only status.

# Results

Infrastructure result: READY for manual official-source verification and an
Ibex smoke run. Core-28 source sample count is currently `0` because the
official SharePoint resources were disabled at retrieval. Full production
extraction status is `PENDING`; no sensor dataset is claimed to exist. The
repository suite passed `397` tests and compileall completed with `0` errors.

# Failures / Limitations

The official label workbook and RGB folder links were inaccessible at the
development retrieval date. The mapping therefore remains a verified-looking
candidate that is deliberately gated by a successful official workbook
download/validation marker. Direct binary RGB catalog population and all
video-level metadata/counts are pending. Skeleton availability, format,
joint ordering, units and RGB synchronization were not verifiable and are
recorded as unknown. Actual A100 throughput, VRAM and full-run failures cannot
be reported before the Ibex smoke/full run.

# Performance

No WiLoR or full extraction was run here. Worker-side timing/progress records
decoded frames, elapsed time and rolling FPS; persistent provenance captures
hardware and execution configuration. A100 performance must be measured by
the documented smoke test rather than assumed from the laptop pilot.

# Comparison

This task does not compare pose models. It invokes the frozen WiLoR/TASK-004/
TASK-005/TASK-006 chain and leaves MediaPipe and prior experiments unchanged.
The Core-28 manifest and LOSO split policy are the reproducibility contract
for later TASK-009 recognition work.

# Recommendation

KEEP — retain this infrastructure and proceed to official-source verification
and the manual Ibex smoke test. Do not treat the pending source/download state
as a completed dataset result.

# Reproducibility

The base commit, source URLs, retrieval date, mapping version, exact SignIDs,
Unicode comparison policy, relative path schema, deterministic repetition
ordering, LOSO policy, shard formula, external roots, frozen pipeline
contracts, FP32 WiLoR settings, state/provenance schema and commands are
recorded in this report and `slurm/README.md`. No random selection or
Python-randomized hash is used.

# Next Steps

1. On Ibex, verify the checked-out branch commit and create the external
   environment.
2. Reattempt the official label download and require successful validation.
3. Obtain official direct binary RGB entries, run preflight and build the
   populated manifest.
4. Run and record the A100 smoke test.
5. Submit the 16-shard array, monitor/resume as needed, and run final QA.
6. Freeze the resulting manifest/splits and export only after QA passes.

# Verdict

`TASK-008A IBEX EXTRACTION INFRASTRUCTURE READY — FULL DATASET RUN PENDING`
