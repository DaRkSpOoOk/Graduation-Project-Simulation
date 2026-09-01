#!/usr/bin/env python3
"""Milestone-1 WiLoR pilot benchmark runner.

Reads the shared KArSL pilot manifest (datasets/manifests/karsl_milestone1_pilot.csv,
columns: at minimum ``sample_id`` and ``local_relative_path`` -- see
reports/pose/wilor/TASK-002-wilor-karsl-pilot.md, "Dataset manifest"), runs
WiLoR extraction on each clip, saves immutable raw output (Task 4), computes
extractor-agnostic metrics (evaluation/metrics/hand_pose_metrics.py), and
writes a JSON summary.

Automatically uses the full detector+MANO pipeline if MANO assets are
present (see pose/wilor/model_loader.check_assets); otherwise falls back to
detector-only mode and records why. All generated artifacts go under
``runs/wilor_karsl_pilot/`` (git-ignored).

Usage:
    python evaluation/benchmarks/wilor_karsl_pilot.py \\
        [--manifest datasets/manifests/karsl_milestone1_pilot.csv] \\
        [--out-dir runs/wilor_karsl_pilot] [--limit N]
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

EXTRACTOR_VERSION = "wilor@fcb911312a38fa8badd30d9656a167485d61b8f9"


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sample_id"}
    path_cols = {"local_relative_path", "relative_video_path"}
    if not rows or not required.issubset(rows[0].keys()) or not (path_cols & rows[0].keys()):
        raise ValueError(
            f"Manifest {path} must have a 'sample_id' column and one of {sorted(path_cols)}"
        )
    return rows


def _video_path(row: dict[str, str]) -> Path:
    rel = row.get("local_relative_path") or row.get("relative_video_path")
    return ROOT / rel


def _asdict_jitter(d: dict) -> dict:
    return {k: dataclasses.asdict(v) for k, v in d.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "datasets/manifests/karsl_milestone1_pilot.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs/wilor_karsl_pilot")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from evaluation.metrics.hand_pose_metrics import evaluate_video
    from pose.wilor.model_loader import (
        WilorAssetPaths,
        check_assets,
        load_detector_only,
        load_pipeline,
    )
    from pose.wilor.errors import WilorAssetMissingError
    from pose.wilor.npz_io import save_raw_video_output
    from pose.wilor.video_processing import process_video_detector_only, process_video_full

    assets = WilorAssetPaths.resolve()
    mode = "full"
    blocker: str | None = None
    try:
        check_assets(assets)
        pipeline = load_pipeline(assets)
    except WilorAssetMissingError as exc:
        mode = "detector_only"
        blocker = str(exc)
        pipeline = None

    if mode == "full":
        detector_for_mode = None
    else:
        detector_for_mode = load_detector_only(assets, device=args.device)

    rows = _read_manifest(args.manifest)
    if args.limit:
        rows = rows[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_video: list[dict] = []
    run_start = time.perf_counter()

    for row in rows:
        sample_id = row["sample_id"]
        video_path = _video_path(row)
        if not video_path.exists():
            per_video.append({"sample_id": sample_id, "error": f"video not found: {video_path}"})
            continue

        if mode == "full":
            result = process_video_full(
                video_path,
                sample_id,
                pipeline,
                runtime_confidence=0.3,
                rescale_factor=2.0,
                extractor_version=EXTRACTOR_VERSION,
                checkpoint_id=str(assets.wilor_checkpoint),
            )
        else:
            result = process_video_detector_only(
                video_path,
                sample_id,
                detector_for_mode,
                confidence_threshold=0.3,
                extractor_version=EXTRACTOR_VERSION,
            )

        npz_path = save_raw_video_output(
            args.out_dir / "raw",
            sample_id,
            result.frames,
            result.vertices_by_hand,
            run_metadata={
                "mode": mode,
                "extractor_version": EXTRACTOR_VERSION,
                "source_video": str(video_path.relative_to(ROOT)),
            },
        )

        video_eval = evaluate_video(sample_id, result.frames)
        per_video.append(
            {
                "sample_id": sample_id,
                "mode": mode,
                "npz_path": str(npz_path.relative_to(ROOT)),
                "total_frames_decoded": result.total_frames_decoded,
                "inference_seconds": result.inference_seconds,
                "effective_fps": result.effective_fps,
                "source_fps": result.fps_source,
                "peak_cuda_allocated_bytes": result.peak_cuda_allocated_bytes,
                "peak_cuda_reserved_bytes": result.peak_cuda_reserved_bytes,
                "frame_errors": result.frame_errors,
                "detection": dataclasses.asdict(video_eval.detection),
                "wrist_jitter": _asdict_jitter(video_eval.wrist_jitter),
                "bone_length_variation": _asdict_jitter(video_eval.bone_length_variation),
                "hand_count_changes": [dataclasses.asdict(c) for c in video_eval.hand_count_changes],
                "handedness_swap_candidates": [
                    dataclasses.asdict(c) for c in video_eval.handedness_swap_candidates
                ],
            }
        )
        print(f"[{sample_id}] mode={mode} frames={result.total_frames_decoded} "
              f"fps={result.effective_fps:.2f} hand_frames="
              f"{sum(1 for f in result.frames if f.hand_present)}")

    total_seconds = time.perf_counter() - run_start
    summary = {
        "extractor_version": EXTRACTOR_VERSION,
        "mode": mode,
        "blocker": blocker,
        "manifest": str(args.manifest.relative_to(ROOT)) if args.manifest.is_relative_to(ROOT) else str(args.manifest),
        "n_videos": len(rows),
        "n_processed": sum(1 for v in per_video if "error" not in v),
        "n_failed": sum(1 for v in per_video if "error" in v),
        "total_wall_seconds": total_seconds,
        "per_video": per_video,
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote summary to {summary_path}")
    print(f"mode={mode} processed={summary['n_processed']}/{summary['n_videos']}")
    if blocker:
        print(f"NOTE: running in detector_only mode. Blocker for full mode:\n{blocker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
