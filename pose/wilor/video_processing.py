"""Video-level driver: decode frames (preserving original timestamps), run
per-frame WiLoR extraction, and report timing separate from any
visualization (Task 3: "measure extraction without visualization
separately").

Frame decoding here is intentionally minimal (cv2.VideoCapture) and local to
this extractor -- it is not a shared video_io/ contract, since that area has
no established interface yet at the time of writing (see
reports/pose/wilor/TASK-002-wilor-karsl-pilot.md, Implementation section).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from pose.common.schema import HandPoseFrame

from .frame_extraction import extract_frame_detector_only, extract_frame_full


@dataclass(slots=True)
class VideoProcessingResult:
    sample_id: str
    frames: list[HandPoseFrame]
    vertices_by_hand: dict[tuple[int, int], np.ndarray]
    total_frames_decoded: int
    inference_seconds: float
    fps_source: float
    effective_fps: float
    mode: str
    peak_cuda_allocated_bytes: int | None = None
    peak_cuda_reserved_bytes: int | None = None
    frame_errors: list[dict[str, Any]] = field(default_factory=list)


def _iter_frames(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_index = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            timestamp_seconds = frame_index / fps if fps > 0 else float(frame_index)
            yield frame_index, timestamp_seconds, frame_bgr
            frame_index += 1
    finally:
        cap.release()
    _iter_frames.last_fps = fps  # type: ignore[attr-defined]


def process_video_detector_only(
    video_path: Path,
    sample_id: str,
    detector: Any,
    *,
    confidence_threshold: float,
    extractor_version: str,
    track_cuda_memory: bool = True,
) -> VideoProcessingResult:
    """Detector-only run (no MANO reconstruction). See module docs on why
    this mode exists."""
    import torch  # noqa: PLC0415

    frames: list[HandPoseFrame] = []
    frame_errors: list[dict[str, Any]] = []
    n_decoded = 0

    use_cuda = track_cuda_memory and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    for frame_index, timestamp_seconds, frame_bgr in _iter_frames(video_path):
        n_decoded += 1
        try:
            frames.extend(
                extract_frame_detector_only(
                    detector,
                    frame_bgr,
                    frame_index,
                    timestamp_seconds,
                    confidence_threshold=confidence_threshold,
                    extractor_version=extractor_version,
                )
            )
        except Exception as exc:  # noqa: BLE001 - record explicit failure, do not interpolate
            frame_errors.append({"frame_index": frame_index, "error": repr(exc)})
            frames.append(
                HandPoseFrame(
                    frame_index=frame_index,
                    timestamp_seconds=timestamp_seconds,
                    hand_present=False,
                    extractor_metadata={"extractor": "wilor", "mode": "detector_only"},
                    quality_flags=["extraction_failed", f"error:{type(exc).__name__}"],
                )
            )
    inference_seconds = time.perf_counter() - start
    fps_source = getattr(_iter_frames, "last_fps", 0.0)
    peak_alloc = torch.cuda.max_memory_allocated() if use_cuda else None
    peak_reserved = torch.cuda.max_memory_reserved() if use_cuda else None

    return VideoProcessingResult(
        sample_id=sample_id,
        frames=frames,
        vertices_by_hand={},
        total_frames_decoded=n_decoded,
        inference_seconds=inference_seconds,
        fps_source=fps_source,
        effective_fps=(n_decoded / inference_seconds) if inference_seconds > 0 else 0.0,
        mode="detector_only",
        peak_cuda_allocated_bytes=peak_alloc,
        peak_cuda_reserved_bytes=peak_reserved,
        frame_errors=frame_errors,
    )


def process_video_full(
    video_path: Path,
    sample_id: str,
    pipeline: Any,
    *,
    runtime_confidence: float,
    rescale_factor: float,
    extractor_version: str,
    checkpoint_id: str,
    track_cuda_memory: bool = True,
) -> VideoProcessingResult:
    """Full detector + MANO reconstruction run."""
    import torch  # noqa: PLC0415

    frames: list[HandPoseFrame] = []
    vertices_by_hand: dict[tuple[int, int], np.ndarray] = {}
    frame_errors: list[dict[str, Any]] = []
    n_decoded = 0

    use_cuda = track_cuda_memory and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    for frame_index, timestamp_seconds, frame_bgr in _iter_frames(video_path):
        n_decoded += 1
        try:
            frame_results, frame_vertices = extract_frame_full(
                pipeline,
                frame_bgr,
                frame_index,
                timestamp_seconds,
                runtime_confidence=runtime_confidence,
                rescale_factor=rescale_factor,
                extractor_version=extractor_version,
                checkpoint_id=checkpoint_id,
            )
            frames.extend(frame_results)
            vertices_by_hand.update(frame_vertices)
        except Exception as exc:  # noqa: BLE001 - record explicit failure, do not interpolate
            frame_errors.append({"frame_index": frame_index, "error": repr(exc)})
            frames.append(
                HandPoseFrame(
                    frame_index=frame_index,
                    timestamp_seconds=timestamp_seconds,
                    hand_present=False,
                    extractor_metadata={"extractor": "wilor", "mode": "full"},
                    quality_flags=["extraction_failed", f"error:{type(exc).__name__}"],
                )
            )
    inference_seconds = time.perf_counter() - start
    fps_source = getattr(_iter_frames, "last_fps", 0.0)

    peak_alloc = torch.cuda.max_memory_allocated() if use_cuda else None
    peak_reserved = torch.cuda.max_memory_reserved() if use_cuda else None

    return VideoProcessingResult(
        sample_id=sample_id,
        frames=frames,
        vertices_by_hand=vertices_by_hand,
        total_frames_decoded=n_decoded,
        inference_seconds=inference_seconds,
        fps_source=fps_source,
        effective_fps=(n_decoded / inference_seconds) if inference_seconds > 0 else 0.0,
        mode="full",
        peak_cuda_allocated_bytes=peak_alloc,
        peak_cuda_reserved_bytes=peak_reserved,
        frame_errors=frame_errors,
    )
