#!/usr/bin/env python3
"""Run WiLoR temporal dual-hand tracking over the KArSL pilot.

Reads the validated WiLoR full-mode run by explicit path (never globbing
``runs/wilor*``), derives the LEFT/RIGHT tracked representation for every
manifest sample, and writes the derived stage plus metrics to an ignored
output directory. The raw input is opened read-only and is never modified.

Example:

    python scripts/run_task004a_tracking.py \\
      --manifest datasets/manifests/karsl_milestone1_pilot.csv \\
      --wilor-run /home/hatim/graduation-project-runs/wilor_karsl_pilot_full \\
      --out-dir /home/hatim/graduation-project-runs/wilor_karsl_pilot_tracked
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tracking.wilor import (  # noqa: E402
    TrackerConfig,
    aggregate_metrics,
    compute_metrics,
    load_raw_sequence,
    save_tracked_sequence,
    track_sequence,
)
from tracking.wilor.source import RawInputError  # noqa: E402

EXPECTED_SAMPLES = 18
EXPECTED_FRAMES = 894


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample_ids(manifest: Path) -> list[str]:
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "sample_id" not in rows[0]:
        raise SystemExit(f"Manifest {manifest} has no sample_id column")
    return [row["sample_id"] for row in rows]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "datasets/manifests/karsl_milestone1_pilot.csv")
    parser.add_argument("--wilor-run", type=Path, required=True, help="validated full-mode run directory")
    parser.add_argument("--out-dir", type=Path, required=True, help="ignored output directory for the derived stage")
    parser.add_argument("--config", type=Path, default=None, help="optional tracker config JSON")
    parser.add_argument("--sample-ids", nargs="*", default=None)
    parser.add_argument("--strict-counts", action="store_true", help="require the exact 18-video / 894-frame pilot")
    args = parser.parse_args()

    config = TrackerConfig.from_json(args.config) if args.config else TrackerConfig()
    config.validate()

    manifest = args.manifest.resolve()
    sample_ids = _sample_ids(manifest)
    if args.sample_ids:
        wanted = set(args.sample_ids)
        sample_ids = [sample for sample in sample_ids if sample in wanted]

    run_dir = args.wilor_run.resolve()
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        if summary.get("mode") != "full":
            raise SystemExit(f"Refusing non-full WiLoR run: mode={summary.get('mode')!r}")

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    per_video = []
    records = []
    for sample_id in sample_ids:
        npz_path = run_dir / "raw" / sample_id / "wilor_raw.npz"
        try:
            raw = load_raw_sequence(npz_path, sample_id)
        except RawInputError as error:
            raise SystemExit(f"{sample_id}: {error}") from error

        source = {
            "raw_npz": str(npz_path),
            "raw_npz_sha256": _sha256(npz_path),
            "run_dir": str(run_dir),
            "manifest": str(manifest),
        }
        sequence = track_sequence(raw, config, source=source)
        metrics = compute_metrics(sequence)
        save_tracked_sequence(out_dir, sequence, metrics.to_dict())
        per_video.append(metrics)
        records.append(metrics.to_dict())
        print(
            f"[{sample_id}] frames={metrics.total_frames} "
            f"L={metrics.observed_left_frames} R={metrics.observed_right_frames} "
            f"missL={metrics.missing_left_frames} missR={metrics.missing_right_frames} "
            f"amb={metrics.ambiguous_frames} extra={metrics.extra_detections_total} "
            f"dup={metrics.duplicate_suppressed_detections} reassoc={metrics.reassociation_events}"
        )

    totals = aggregate_metrics(per_video)
    if args.strict_counts:
        if len(per_video) != EXPECTED_SAMPLES or totals.get("total_frames") != EXPECTED_FRAMES:
            raise SystemExit(
                f"Strict pilot check failed: videos={len(per_video)} frames={totals.get('total_frames')}"
            )

    summary_out = {
        "stage": "tracked",
        "schema_version": config.schema_version,
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "wilor_run_dir": str(run_dir),
        "config": config.to_dict(),
        "aggregate": totals,
        "per_video": records,
    }
    (out_dir / "tracking_summary.json").write_text(json.dumps(summary_out, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {out_dir/'tracking_summary.json'}")
    print(
        f"videos={totals['videos']} frames={totals['total_frames']} "
        f"bothTracks={totals['frames_with_both_tracks']} noTrack={totals['frames_with_no_track']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
