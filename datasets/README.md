# datasets/

Contains deterministic manifests and local directory conventions. Raw videos,
model bundles, partial source archives, and generated pose/overlay artifacts are
ignored by Git.

## KArSL milestone 1 pilot

The shared experimental contract is
`manifests/karsl_milestone1_pilot.csv`. It selects signs 0171--0176, the
official `test` split, signer IDs 01--03, and the lexicographically first valid
RGB `.mp4` member for each sign/signer pair (18 videos total).

The official KArSL download page exposes signer/split archives rather than
individual video URLs. The pilot downloader therefore uses HTTP byte ranges to
fetch only the compressed prefix required for those six members from each of
the three `0171-0190.7z` archives, plus the small 7z header at the end. It
extracts only the manifest members and removes the temporary partial archive by
default. It never downloads the full KArSL collection.

Expected local layout after acquisition:

```text
datasets/raw/karsl_milestone1_pilot/<signer>/test/<sign_id>/<source_filename>.mp4
```

Do not commit videos, partial archives, model files, raw pose arrays, or
overlays.

## KArSL Core-28 production preparation

`manifests/karsl_core28_labels.csv` records the deterministic Core-28 mapping
candidate (SignIDs `0032`--`0059`); `manifests/karsl_core28.csv` and the three
`splits/karsl_core28_loso_s*.csv` files are schema-only until the official
KArSL label workbook and RGB assets have been verified.  Use
`scripts/download_karsl_core28.py --labels-only` before discovery.  The
official source URLs are documented in
`research/datasets/karsl_core28.md`; the development retrieval found the
current SharePoint links disabled, so no mirror is substituted.

Production data belongs outside the repository, for example
`/ibex/user/$USER/graduation-project-data/karsl/`.  TASK-008A's extraction
runner and Ibex job-array documentation are in `slurm/` and
`reports/dataset/TASK-008A-karsl-core28-dataset.md`.
