"""Matched-scope timing for the two frozen extractors.

The parent process is neutral. Each worker runs in a subprocess whose import
path points at exactly one frozen experimental worktree, preventing package
namespace collisions between the two implementations.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:  # Support the isolated subprocess worker invocation.
    _neutral_root = Path(__file__).resolve().parents[2]
    if str(_neutral_root) not in sys.path:
        sys.path.insert(0, str(_neutral_root))
    from evaluation.comparison.common_contract import EXPECTED_SAMPLE_COUNT, EXPECTED_TOTAL_FRAMES
    from evaluation.comparison.loaders import validate_manifest
else:
    from .common_contract import EXPECTED_SAMPLE_COUNT, EXPECTED_TOTAL_FRAMES
    from .loaders import validate_manifest


RESULT_MARKER = "TASK003A2_RESULT="


def _manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(f"Expected {EXPECTED_SAMPLE_COUNT} manifest rows, got {len(rows)}")
    if len({row["sample_id"] for row in rows}) != EXPECTED_SAMPLE_COUNT:
        raise ValueError("Manifest sample IDs are not unique")
    return rows


def _video_path(row: dict[str, str], video_root: Path) -> Path:
    relative = Path(row["local_relative_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe manifest path: {relative}")
    path = video_root / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _frame_weighted_fps(total_frames: int, total_seconds: float) -> float | None:
    return total_frames / total_seconds if total_seconds > 0 else None


def _run_mediapipe_worker(args: argparse.Namespace) -> dict[str, Any]:
    import cv2
    import mediapipe as mp

    frozen_root = Path(args.frozen_root).resolve()
    sys.path.insert(0, str(frozen_root))
    from pose.mediapipe.extractor import (  # type: ignore[import-not-found]
        MediaPipeConfig,
        _create_landmarker,
        _frame_arrays,
        result_to_common_frames,
    )
    from video_io.reader import iter_video_frames  # type: ignore[import-not-found]

    rows = _manifest_rows(Path(args.manifest).resolve())
    video_root = Path(args.video_root).resolve()
    config = MediaPipeConfig(
        model_path=Path(args.model).resolve(),
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        delegate="CPU",
    )
    per_video: list[dict[str, Any]] = []
    # VIDEO mode retains state, so timestamps must remain monotonic across
    # clips. The offset is only for API sequencing; per-video frame deltas are
    # unchanged and raw experiment timestamps are not rewritten.
    timestamp_offset_ms = 0
    with _create_landmarker(config) as landmarker:
        for row in rows:
            sample_id = row["sample_id"]
            previous_global_timestamp_ms: int | None = None
            frame_count = 0
            converted_hand_count = 0
            video_path = _video_path(row, video_root)
            started = time.perf_counter()
            for frame in iter_video_frames(video_path):
                timestamp_ms = timestamp_offset_ms + frame.timestamp_ms
                if previous_global_timestamp_ms is not None and timestamp_ms <= previous_global_timestamp_ms:
                    timestamp_ms = previous_global_timestamp_ms + 1
                rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                image, world, present, labels, scores, category_indices = _frame_arrays(result, 2)
                converted_hand_count += len(result_to_common_frames(result, frame.frame_index, frame.timestamp_seconds))
                frame_count += 1
                previous_global_timestamp_ms = timestamp_ms
            elapsed = time.perf_counter() - started
            if previous_global_timestamp_ms is not None:
                timestamp_offset_ms = previous_global_timestamp_ms + 1
            per_video.append(
                {
                    "sample_id": sample_id,
                    "total_frames": frame_count,
                    "converted_hand_rows": converted_hand_count,
                    "total_seconds": elapsed,
                    "frame_weighted_fps": _frame_weighted_fps(frame_count, elapsed),
                }
            )
    total_frames = sum(int(row["total_frames"]) for row in per_video)
    total_seconds = sum(float(row["total_seconds"]) for row in per_video)
    if total_frames != EXPECTED_TOTAL_FRAMES:
        raise RuntimeError(f"MediaPipe matched timer decoded {total_frames}, expected {EXPECTED_TOTAL_FRAMES}")
    return {
        "system": "mediapipe",
        "hardware": "CPU delegate",
        "total_frames": total_frames,
        "total_seconds": total_seconds,
        "frame_weighted_fps": _frame_weighted_fps(total_frames, total_seconds),
        "model_loaded_before_timing": True,
        "warmup": "none; first measured inference includes any lazy runtime initialization",
        "per_video": per_video,
    }


def _run_wilor_worker(args: argparse.Namespace) -> dict[str, Any]:
    frozen_root = Path(args.frozen_root).resolve()
    sys.path.insert(0, str(frozen_root))
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Matched WiLoR timing requires the validated CUDA configuration")
    from pose.wilor.config import WilorAssetPaths, WilorRuntimeConfig  # type: ignore[import-not-found]
    from pose.wilor.model_loader import check_assets, load_pipeline  # type: ignore[import-not-found]
    from pose.wilor.video_processing import process_video_full  # type: ignore[import-not-found]

    rows = _manifest_rows(Path(args.manifest).resolve())
    video_root = Path(args.video_root).resolve()
    assets = WilorAssetPaths.resolve()
    check_assets(assets)
    pipeline = load_pipeline(assets, WilorRuntimeConfig(device="cuda", fast_mode=False))
    per_video: list[dict[str, Any]] = []
    for row in rows:
        sample_id = row["sample_id"]
        video_path = _video_path(row, video_root)
        result = process_video_full(
            video_path,
            sample_id,
            pipeline,
            runtime_confidence=0.3,
            rescale_factor=2.0,
            extractor_version="task003a2-neutral-timer",
            checkpoint_id=str(assets.wilor_checkpoint),
        )
        per_video.append(
            {
                "sample_id": sample_id,
                "total_frames": result.total_frames_decoded,
                "converted_hand_rows": len(result.frames),
                "total_seconds": result.inference_seconds,
                "frame_weighted_fps": _frame_weighted_fps(result.total_frames_decoded, result.inference_seconds),
            }
        )
    total_frames = sum(int(row["total_frames"]) for row in per_video)
    total_seconds = sum(float(row["total_seconds"]) for row in per_video)
    if total_frames != EXPECTED_TOTAL_FRAMES:
        raise RuntimeError(f"WiLoR matched timer decoded {total_frames}, expected {EXPECTED_TOTAL_FRAMES}")
    return {
        "system": "wilor",
        "hardware": "CUDA GPU",
        "device": str(pipeline.device),
        "fast_mode": False,
        "detector_confidence": 0.3,
        "rescale_factor": 2.0,
        "total_frames": total_frames,
        "total_seconds": total_seconds,
        "frame_weighted_fps": _frame_weighted_fps(total_frames, total_seconds),
        "model_loaded_before_timing": True,
        "warmup": "none; first measured inference includes any lazy runtime initialization",
        "per_video": per_video,
    }


def _worker_main(args: argparse.Namespace) -> int:
    result = _run_mediapipe_worker(args) if args.worker == "mediapipe" else _run_wilor_worker(args)
    print(RESULT_MARKER + json.dumps(result, sort_keys=True))
    return 0


def _call_worker(system: str, args: argparse.Namespace) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        system,
        "--frozen-root",
        str(args.mediapipe_root if system == "mediapipe" else args.wilor_root),
        "--manifest",
        str(args.manifest),
        "--video-root",
        str(args.video_root),
    ]
    if system == "mediapipe":
        command.extend(["--model", str(args.model)])
    environment = os.environ.copy()
    frozen_root = str(args.mediapipe_root if system == "mediapipe" else args.wilor_root)
    current_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = frozen_root + (os.pathsep + current_pythonpath if current_pythonpath else "")
    if system == "wilor":
        if args.wilor_assets_dir is not None:
            environment["WILOR_ASSETS_DIR"] = str(args.wilor_assets_dir.resolve())
        if args.wilor_source_dir is not None:
            environment["WILOR_SOURCE_DIR"] = str(args.wilor_source_dir.resolve())
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{system} matched worker failed with exit code {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    marker_position = completed.stdout.rfind(RESULT_MARKER)
    if marker_position < 0:
        raise RuntimeError(f"{system} worker did not emit a result marker\n{completed.stdout}\n{completed.stderr}")
    return json.loads(completed.stdout[marker_position + len(RESULT_MARKER) :].strip())


def run_matched_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    validate_manifest(args.manifest, video_root=args.video_root, verify_video_checksums=True)
    mediapipe = _call_worker("mediapipe", args)
    wilor = _call_worker("wilor", args)
    result = {
        "scope": [
            "video decode",
            "required model preprocessing",
            "model inference/reconstruction",
            "conversion into in-memory pose representation",
        ],
        "excluded": [
            "model/checkpoint construction/loading",
            "disk serialization",
            "overlay generation",
            "video encoding",
            "report generation",
        ],
        "interpretation": "practical throughput using each model's validated operating configuration under a matched software timing boundary; not hardware-normalized",
        "mediapipe": mediapipe,
        "wilor": wilor,
    }
    for system in (mediapipe, wilor):
        if system["total_frames"] != EXPECTED_TOTAL_FRAMES:
            raise RuntimeError(f"Matched result for {system['system']} is not frame-complete")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mediapipe-root", type=Path, required=False, default=Path("../Graduation-Project-Simulation-luna"))
    parser.add_argument("--wilor-root", type=Path, required=False, default=Path("../Graduation-Project-Simulation-opus"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=False, default=None)
    parser.add_argument("--wilor-assets-dir", type=Path, required=False, default=None)
    parser.add_argument("--wilor-source-dir", type=Path, required=False, default=None)
    parser.add_argument("--output", type=Path, required=False, default=None)
    parser.add_argument("--worker", choices=("mediapipe", "wilor"), default=None)
    parser.add_argument("--frozen-root", type=Path, default=None)
    args = parser.parse_args()
    if args.worker:
        if args.frozen_root is None:
            parser.error("--frozen-root is required in worker mode")
        if args.worker == "mediapipe" and args.model is None:
            parser.error("--model is required for the MediaPipe worker")
        return _worker_main(args)
    if args.model is None:
        parser.error("--model is required")
    result = run_matched_benchmark(args)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
