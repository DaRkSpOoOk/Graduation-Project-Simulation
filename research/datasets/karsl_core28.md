# KArSL Core-28 acquisition note

Status: infrastructure prepared; official binary acquisition and label
verification are still pending.

## Official sources

The source of truth for this task is the [official KArSL project page](https://hamzah-luqman.github.io/KArSL/), its
[official repository](https://github.com/Hamzah-Luqman/KArSL), the
[official RGB download page](https://hamzah-luqman.github.io/KArSL/download_video_502.html),
and the original [KArSL research paper](https://doi.org/10.1145/3423420).
The project page identifies KArSL-502 as a 502-sign isolated-sign dataset
with RGB, depth and skeleton modalities, three signers and repeated samples.

At retrieval on 2026-09-02, the current official label workbook link and the
six signer/partition RGB SharePoint folder links returned an administrator-
disabled page.  The links are retained in `evaluation/dataset/core28.py` and
the downloader refuses HTML/folder pages.  No third-party mirror is used as a
video acquisition source, and no bulk data was downloaded in this task.

## Core-28 mapping candidate

The official KArSL vocabulary is organized with 39 letter-like entries before
the number/word vocabulary.  The committed candidate mapping freezes the first
28 standard Arabic alphabet letters as SignIDs `0032` through `0059` and keeps
SignIDs `0060` through `0070` as extended-letter candidates.  The exact Arabic
cells, English glosses, deterministic zero-based label indices and source-row
convention are in:

- `datasets/manifests/karsl_core28_labels.csv`
- `evaluation/dataset/core28.py`

The source-row value uses the workbook's one-row header convention (`source
row = SignID + 1` for this layout).  `--labels-only` must successfully download
and parse `KARSL-502_Labels.xlsx`, validate every Core-28 row, and write a
hash-bound verification marker before `--discover` may create a production
manifest.  This prevents the candidate mapping from silently becoming a
production authority if the official workbook changes.

The mapping was cross-checked against a publicly visible workbook export only
as a research aid; that export is not an approved source for the production
dataset and is not committed or downloaded by the tooling.

## Skeleton modality

The official project description advertises skeleton data, but the accessible
current distribution did not provide a usable skeleton archive or a verified
joint-order/coordinate specification at retrieval time.  Therefore this task
records skeleton availability as `unknown`, does not claim a file format or
Kinect joint synchronization, and continues with RGB-only acquisition.  If a
verified skeleton distribution becomes available, it may be recorded as
auxiliary metadata for future left/right validation, crop priors or wrist
trajectories.  It will not replace WiLoR's detailed per-finger geometry and it
does not change TASK-004 or TASK-005.

## SSHI note

The project has approval, as supplied for this task, to obtain data from
[SSHI](https://sshi.sa/), including scraping/data collection as necessary.
No SSHI scraper is implemented here.  SSHI source investigation is deferred
to a dedicated later task.
