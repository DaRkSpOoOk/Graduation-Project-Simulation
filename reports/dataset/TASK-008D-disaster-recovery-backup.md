# TASK-008D — Verified Disaster-Recovery Backup

## Task

Create and independently verify a ZIP backup of the finalized TASK-008 Core-28
virtual-glove production dataset. This task is archival only; it does not change
the dataset or rerun any extraction stage.

## Purpose

Preserve the expensive generated sensor dataset and the compact metadata needed
to interpret it and reconnect it to TASK-009 after a disaster recovery. The
original KArSL RGB source distribution remains at its existing location and is
not copied into this archive.

## Included content

The archive contains:

- the complete production run under `production/`, including `pose/` (and
  `pose/raw/`), `tracking/`, `kinematics/`, `virtual_glove/`, `state/`,
  `provenance/`, and run logs;
- `metadata/manifests/karsl_core28.csv`;
- `metadata/manifests/karsl_core28_labels.csv`;
- `metadata/manifests/karsl_core28_virtual_glove.csv`;
- `metadata/splits/karsl_core28_loso_s01.csv`, `s02.csv`, and `s03.csv`;
- all finalized TASK-008 acquisition, extraction, and QA reports/results;
- `README_BACKUP.md`, `backup_manifest.json`, and `files.sha256`.

Archive hierarchy:

```text
KArSL-Core28-TASK008/
├── README_BACKUP.md
├── backup_manifest.json
├── files.sha256
├── production/
│   ├── pose/
│   ├── tracking/
│   ├── kinematics/
│   ├── virtual_glove/
│   ├── state/
│   ├── provenance/
│   └── logs/
└── metadata/
    ├── manifests/
    ├── splits/
    └── reports/
```

## Excluded content

The archive excludes `.git/`, virtual environments, caches, model/checkpoint
files, generated media, unrelated pilot or resume runs, and the original KArSL
RGB source videos. No excluded item was found in the production run during the
pre-archive scan. The repository already ignores archive extensions (`*.zip`,
`*.tar*`) and generated data paths; no archive file is stored in Git.

## Production source

- Production run: `/home/hatim/graduation-project-runs/task008-core28-full`
- Source dataset identity: KArSL-502
- Source dataset path (not archived): `/home/hatim/datasets/KArSL-502`
- Selected source-video bytes recorded by TASK-008C: `1,794,280,889`
- TASK-008 final verdict: **TASK-008 CORE-28 SENSOR DATASET COMPLETE — READY FOR TASK-009**
- TASK-008C metadata commit: `db36038`
- Production run provenance commit: `5c47de98771154d88ffa1896b96ae8933431ff42`
- Frozen contracts: `TASK-005-final-v2`, `TASK-006-ideal-virtual-glove-v1`

## Production sample count

The finalized QA was independently checked before archiving:

| Check | Result |
|---|---:|
| Manifest samples | 4,222 unique |
| Core-28 labels | 28 |
| Requested | 4,222 |
| Successful/completed | 4,222 |
| Failed | 0 |
| Unaccounted | 0 |
| `pose/raw` sample directories | 4,222 |
| `tracking` sample directories | 4,222 |
| `kinematics` sample directories | 4,222 |
| `virtual_glove` sample directories | 4,222 |
| Source frames | 83,659 |

Every stage directory set matched the 4,222 manifest sample IDs exactly, and
all stage sample directories were non-empty. The three LOSO files each contain
4,222 unique samples with no held-out-signer leakage:

| Fold | Train | Validation | Test |
|---|---:|---:|---:|
| S01 | 2,372 | 448 | 1,402 |
| S02 | 2,373 | 448 | 1,401 |
| S03 | 2,355 | 448 | 1,419 |

## Archive size

- Filename: `KArSL-Core28-TASK008-VirtualGlove-4222-20260904.zip`
- Absolute path: `/home/hatim/graduation-project-backups/KArSL-Core28-TASK008-VirtualGlove-4222-20260904.zip`
- Size: `2,285,452,500` bytes (`2.13 GiB`)
- Archive entries: `50,687`
- Source run apparent size before archiving: `2,255,316,539` bytes (`2.10 GiB`)
- Source run regular-file count: `50,669`
- Source run descendant-directory count: `16,896`

The ZIP uses stored/no-compression entries because the generated NPZ files are
already compressed and ZIP overhead/integrity were preferred over recompression.
ZIP64 support was enabled.

## Archive SHA-256

```text
6d1fa3cdcba11ac72f1416600b08a7f3c84681d7be38666ff939121bfd8be7e6
```

Sidecar:

```text
/home/hatim/graduation-project-backups/KArSL-Core28-TASK008-VirtualGlove-4222-20260904.zip.sha256
```

## Source-file hash strategy

`files.sha256` contains SHA-256 hashes for all `50,686` other regular files in
the archive: the production files, the metadata bundle, `README_BACKUP.md`, and
`backup_manifest.json`. It intentionally does not hash itself to avoid a
recursive hash definition. Paths in the hash list are archive-relative and do
not depend on `/home/hatim`.

## ZIP integrity result

- `unzip -t`: **PASS**; all `50,687` entries reported OK and no errors were
  detected.
- Sidecar verification with `sha256sum -c`: **PASS**.
- Independent stream verification of every `files.sha256` entry against the ZIP
  contents: **PASS**, `50,686 / 50,686`.

## Restore sanity-test result

A full extraction was performed into a temporary directory outside the
production run and was removed after the check. The restored manifest contained
4,222 rows. Deterministic beginning/middle/end samples were checked:

```text
karsl_core28_s01_sign0032_test_rep001
karsl_core28_s02_sign0043_train_rep023
karsl_core28_s03_sign0059_train_rep052
```

For each sample, the restored `pose/raw`, `tracking`, `kinematics`, and
`virtual_glove` files were present; 36 representative NPZ/JSON files were
opened successfully and matched `files.sha256`. Result: **PASS**. The actual
production run was never overwritten or moved.

## Git commit used

The archive metadata was captured from TASK-008C final commit `db36038` on
`opus/task-008c-core28-full-extraction`. This report is being committed on the
dedicated TASK-008D backup branch. The archive and its sidecar remain outside
the repository and are not tracked.

## Limitations

- The archive does not contain the approximately 10,820,626,022-byte original
  KArSL-502 source tree or its RGB videos. Recovery still requires access to
  that source distribution if regeneration is ever needed.
- The archive preserves natural variable-length sequences and existing masks;
  it does not create a TASK-009 tensor, train a model, or add accelerometer
  data.
- Absolute paths in `backup_manifest.json` and this report are provenance
  records only. The archive hierarchy and restore procedure are portable.

## Restore instructions

Keep the ZIP and its `.sha256` sidecar together after uploading them to Google
Drive. On a recovery machine:

```bash
sha256sum -c KArSL-Core28-TASK008-VirtualGlove-4222-20260904.zip.sha256
unzip -t KArSL-Core28-TASK008-VirtualGlove-4222-20260904.zip
unzip KArSL-Core28-TASK008-VirtualGlove-4222-20260904.zip -d <recovery-directory>
```

Then use:

```text
<recovery-directory>/KArSL-Core28-TASK008/production/virtual_glove
<recovery-directory>/KArSL-Core28-TASK008/production/kinematics
<recovery-directory>/KArSL-Core28-TASK008/production/tracking
<recovery-directory>/KArSL-Core28-TASK008/production/pose/raw
<recovery-directory>/KArSL-Core28-TASK008/metadata/manifests
<recovery-directory>/KArSL-Core28-TASK008/metadata/splits
```

with `backup_manifest.json`, `files.sha256`, and the finalized TASK-008 QA
reports. Do not extract over an existing production run.
