# TASK-008A Ibex operations

The scripts in this directory prepare and run the Core-28 extraction outside
Git. They do not select an account or QOS: those are institution/allocation
settings and must be supplied by the user when Ibex requires them. Likewise,
the default GPU resource/`a100` constraint is a documented starting point,
not a claim that every Ibex partition has identical syntax.

## Environment and commit pin

Use an external data and run root:

```bash
export KARSL_DATA_ROOT=/ibex/user/$USER/graduation-project-data/karsl
export TASK008A_RUN_ROOT=/ibex/user/$USER/graduation-project-runs/task008a-karsl-core28
export TASK008A_REPO_ROOT=/ibex/user/$USER/Graduation-Project-Simulation
```

Clone/check out the exact reviewed branch and record the commit before doing
anything expensive:

```bash
cd /ibex/user/$USER
git clone git@github.com:DaRkSpOoOk/Graduation-Project-Simulation.git
cd Graduation-Project-Simulation
git fetch origin
git checkout luna/task-008a-karsl-core28-dataset
git rev-parse HEAD
```

Compare that hash with the TASK-008A PR handoff. Do not run from a dirty
checkout. Create a dedicated environment outside the repository using the
project's existing metadata and the WiLoR environment documented by its
reports. Load only a CUDA module compatible with that environment; do not
enable FP16 or mixed precision merely because the node is an A100. The
worker's persisted provenance records Python, NumPy, PyTorch/CUDA, TF32
settings, GPU information, WiLoR mode/checkpoint identity and all SLURM IDs.

If the environment is activated by a shell fragment, set:

```bash
export TASK008A_ENV_ACTIVATE=/ibex/user/$USER/envs/task008a/bin/activate
# Optional, only after choosing a compatible installed module:
export CUDA_MODULE=cuda/12.1
```

The CUDA value above is an example placeholder. Inspect actual WiLoR/PyTorch
compatibility before selecting it.

## Source and labels preflight

The current official SharePoint links were disabled at development retrieval.
Do not substitute a mirror. Run the official label step only when the source
is accessible:

```bash
python scripts/download_karsl_core28.py \
  --data-root "$KARSL_DATA_ROOT" --labels-only --resume
```

It must validate the official `KARSL-502_Labels.xlsx` and create a hash-bound
verification marker. Then populate a catalog of direct binary RGB assets
exported from the official distribution (folder URLs are not accepted as
video files):

```bash
python scripts/download_karsl_core28.py \
  --data-root "$KARSL_DATA_ROOT" \
  --source-catalog "$KARSL_DATA_ROOT/acquisition/source_catalog.csv" \
  --resume --retry-failed
python scripts/download_karsl_core28.py --data-root "$KARSL_DATA_ROOT" --status
python scripts/download_karsl_core28.py --data-root "$KARSL_DATA_ROOT" \
  --discover --verify
```

Discovery writes portable paths to `datasets/manifests/karsl_core28.csv` and
the three LOSO split manifests. It refuses production discovery without
official-label verification. The committed mapping and header templates are
therefore not evidence that the RGB dataset has already been obtained.

Before GPU work, inspect the preflight size/free-space output and require a
non-empty, readable manifest. Raw videos and archives stay under
`KARSL_DATA_ROOT`, never in this checkout.

## Smoke test

Prepare a deterministic 18--24-row manifest covering all signers and the
desired source partitions, then set `TASK008A_SMOKE_MANIFEST` to it. The
smoke job runs the same frozen stages as production with one worker and a
small limit:

```bash
mkdir -p "$TASK008A_RUN_ROOT/smoke/logs"
sbatch \
  --output="$TASK008A_RUN_ROOT/smoke/logs/smoke-%A_%a.out" \
  --error="$TASK008A_RUN_ROOT/smoke/logs/smoke-%A_%a.err" \
  slurm/task008a_smoke_a100.slurm
squeue -u "$USER"
```

Record wall time, decoded frames, effective WiLoR FPS, video throughput, GPU
memory and every failure. Do not submit the full array if the smoke run shows
a systematic import, CUDA, checkpoint, schema or provenance problem.

## Production array

The default plan is 16 deterministic modulo shards, four concurrent workers,
one A100 per worker, eight CPU cores and about 32 GB RAM per worker:

```bash
mkdir -p "$TASK008A_RUN_ROOT/logs"
sbatch \
  --output="$TASK008A_RUN_ROOT/logs/core28-%A_%a.out" \
  --error="$TASK008A_RUN_ROOT/logs/core28-%A_%a.err" \
  slurm/task008a_core28_a100.slurm
```

The array is conceptually `0-15%4`. To use a different partition, account or
QOS, supply the site's required `sbatch --partition=... --account=...` or
`--qos=...` override. To rerun one or several shards:

```bash
sbatch --array=7 slurm/task008a_core28_a100.slurm
sbatch --array=3,7,11 slurm/task008a_core28_a100.slurm
```

Keep `TASK008A_NUM_SHARDS=16` for those reruns. The stable manifest-order
modulo rule ensures that a sample belongs to one shard only. Every worker
loads WiLoR once and reuses it for its entire assigned shard.

## Monitoring and recovery

The runner persists atomic per-shard state, stage sidecars, failure JSONL and
provenance. It validates manifest/source hashes and artifact schemas before
skipping a stage. A corrupt or mismatched output is recomputed; a completed
sample is retained. A failed sample is recorded and does not terminate the
other samples in the shard.

```bash
python scripts/run_task008a_karsl_core28.py \
  --manifest "$TASK008A_MANIFEST" --run-root "$TASK008A_RUN_ROOT" --status
squeue -u "$USER"
sacct -j <ARRAY_JOB_ID> --format=JobID,State,Elapsed,AllocTRES,MaxRSS,ExitCode
tail -f "$TASK008A_RUN_ROOT/logs/core28-<ARRAY_JOB_ID>_<TASK_ID>.out"
```

Resume all assigned work with `--resume`; retry recorded failures with
`--resume --retry-failed`. Status mode is read-only and does not load WiLoR.
Cancel only when intended:

```bash
scancel <ARRAY_JOB_ID>
```

After the array is complete, run CPU-only final QA through an after-ok
dependency:

```bash
mkdir -p "$TASK008A_RUN_ROOT/qa"
QA_JOB=$(sbatch --parsable \
  --dependency=afterok:<ARRAY_JOB_ID> \
  --output="$TASK008A_RUN_ROOT/qa/final-%j.out" \
  --error="$TASK008A_RUN_ROOT/qa/final-%j.err" \
  slurm/task008a_final_qa.slurm)
echo "$QA_JOB"
```

The final QA accounts for every source sample, stage status, source/frame
alignment, validity masks, normalized ranges, sequence lengths, class/signer
coverage and failures. It does not pad, resample, truncate, interpolate or
construct an LSTM tensor.

## Export

After QA, export the compact sensor dataset/index and provenance rather than
raw RGB by default. The authoritative per-video files remain natural-length
`virtual_glove.npz`, metadata and `sensor_layout.json`; pose intermediates can
remain on Ibex when regeneration is possible.

```bash
ARCHIVE="${TASK008A_EXPORT_ARCHIVE:-$TASK008A_RUN_ROOT/../task008a-core28-export.tar.zst}"
tar -I 'zstd -T0 -10' -cf "$ARCHIVE" \
  -C "$TASK008A_RUN_ROOT" virtual_glove export qa provenance failures \
  -C "$TASK008A_REPO_ROOT" datasets/manifests datasets/splits reports/dataset
sha256sum "$ARCHIVE"
```

The exact archive member list should be recorded with the QA result; verify
the SHA-256 after downloading. Do not copy raw KArSL videos to the laptop as
part of the standard ML export.
