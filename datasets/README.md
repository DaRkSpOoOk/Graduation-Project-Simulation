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
