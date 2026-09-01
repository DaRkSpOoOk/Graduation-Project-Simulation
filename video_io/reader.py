"""OpenCV video input with container-aware timestamp handling.

The MediaPipe video task requires monotonically increasing millisecond
timestamps.  This module keeps the best source timestamp available for each
decoded frame, while also exposing the integer timestamp actually supplied to
MediaPipe when duplicate millisecond values need a minimal monotonic repair.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterator


def _cv2() -> Any:
    try:
        import cv2
    except ModuleNotFoundError as error:  # pragma: no cover - environment error.
        raise RuntimeError("video_io requires OpenCV; install the project dependencies first") from error
    return cv2


@dataclass(frozen=True, slots=True)
class VideoFrame:
    """One successfully decoded frame and its timestamp provenance."""

    frame_index: int
    timestamp_seconds: float
    timestamp_ms: int
    timestamp_source: str
    timestamp_adjusted_for_monotonicity: bool
    image: Any


@dataclass(frozen=True, slots=True)
class VideoInspection:
    """Metadata and decoder outcome collected before inference."""

    path: str
    decoder_success: bool
    error: str | None
    reported_frame_count: int | None
    decoded_frame_count: int
    fps: float | None
    width: int | None
    height: int | None
    duration_seconds: float | None
    timestamp_source: str
    timestamp_adjustments: int
    frame_count_mismatch: bool
    ffprobe_available: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _ProbeData:
    frame_timestamps: list[float | None]
    duration_seconds: float | None
    fps: float | None
    width: int | None
    height: int | None
    available: bool


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_int(value: Any) -> int | None:
    number = _as_float(value)
    if number is None or number < 0:
        return None
    return int(round(number))


def _parse_rate(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value)
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        try:
            result = float(numerator) / float(denominator)
        except (ValueError, ZeroDivisionError):
            return None
        return result if math.isfinite(result) and result > 0 else None
    result = _as_float(text)
    return result if result and result > 0 else None


def _run_ffprobe(path: Path) -> dict[str, Any] | None:
    if shutil.which("ffprobe") is None:
        return None
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_streams",
        "-show_format",
        "-show_frames",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration:format=duration:frame=best_effort_timestamp_time",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return None
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None


def _probe(path: Path) -> _ProbeData:
    data = _run_ffprobe(path)
    if data is None:
        return _ProbeData([], None, None, None, None, False)
    streams = data.get("streams") or []
    stream = streams[0] if streams else {}
    format_data = data.get("format") or {}
    timestamps: list[float | None] = []
    for frame in data.get("frames") or []:
        timestamps.append(_as_float(frame.get("best_effort_timestamp_time")))
    duration = _as_float(stream.get("duration")) or _as_float(format_data.get("duration"))
    fps = _parse_rate(stream.get("avg_frame_rate")) or _parse_rate(stream.get("r_frame_rate"))
    return _ProbeData(
        frame_timestamps=timestamps,
        duration_seconds=duration,
        fps=fps,
        width=_as_int(stream.get("width")),
        height=_as_int(stream.get("height")),
        available=True,
    )


def _candidate_timestamp(
    frame_index: int,
    cap: Any,
    probe: _ProbeData,
    fps: float | None,
    previous_seconds: float | None,
) -> tuple[float, str]:
    if frame_index < len(probe.frame_timestamps):
        candidate = probe.frame_timestamps[frame_index]
        if candidate is not None:
            return candidate, "ffprobe_best_effort_timestamp"

    position_ms = _as_float(cap.get(_cv2().CAP_PROP_POS_MSEC))
    if position_ms is not None and position_ms >= 0:
        # Some backends return zero for every frame.  Keep the first zero, but
        # use another source after a known positive timestamp has appeared.
        if position_ms > 0 or frame_index == 0:
            return position_ms / 1000.0, "opencv_pos_msec"

    if fps and fps > 0:
        return frame_index / fps, "fps_index_fallback"
    if previous_seconds is not None:
        return previous_seconds + (1.0 / 30.0), "synthetic_30fps_fallback"
    return 0.0, "synthetic_30fps_fallback"


def iter_video_frames(path: str | Path) -> Iterator[VideoFrame]:
    """Yield decoded frames with source and MediaPipe-safe timestamps.

    Source timestamps are never smoothed or interpolated.  When their rounded
    millisecond representation is not strictly increasing, only the separate
    ``timestamp_ms`` value is advanced by one millisecond so the official
    MediaPipe VIDEO API can accept the call.
    """

    cv2 = _cv2()
    video_path = Path(path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Video decoder could not open {video_path}")

    reported_fps = _as_float(cap.get(cv2.CAP_PROP_FPS))
    fps = reported_fps if reported_fps and reported_fps > 0 else None
    probe = _probe(video_path)
    if fps is None:
        fps = probe.fps
    previous_seconds: float | None = None
    previous_ms: int | None = None
    frame_index = 0
    try:
        while True:
            success, image = cap.read()
            if not success:
                break
            source_seconds, source = _candidate_timestamp(frame_index, cap, probe, fps, previous_seconds)
            timestamp_ms = int(round(source_seconds * 1000.0))
            adjusted = previous_ms is not None and timestamp_ms <= previous_ms
            if adjusted:
                timestamp_ms = previous_ms + 1
            previous_seconds = source_seconds
            previous_ms = timestamp_ms
            yield VideoFrame(
                frame_index=frame_index,
                timestamp_seconds=source_seconds,
                timestamp_ms=timestamp_ms,
                timestamp_source=source,
                timestamp_adjusted_for_monotonicity=adjusted,
                image=image,
            )
            frame_index += 1
    finally:
        cap.release()


def inspect_video(path: str | Path) -> VideoInspection:
    """Inspect decoder metadata and count frames before model inference."""

    video_path = Path(path)
    probe = _probe(video_path)
    cv2 = _cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return VideoInspection(
            path=str(video_path),
            decoder_success=False,
            error="decoder_open_failed",
            reported_frame_count=_as_int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            decoded_frame_count=0,
            fps=probe.fps,
            width=probe.width,
            height=probe.height,
            duration_seconds=probe.duration_seconds,
            timestamp_source="unavailable",
            timestamp_adjustments=0,
            frame_count_mismatch=False,
            ffprobe_available=probe.available,
        )
    reported_frames = _as_int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    reported_fps = _as_float(cap.get(cv2.CAP_PROP_FPS))
    reported_width = _as_int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    reported_height = _as_int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    decoded = 0
    timestamp_sources: Counter[str] = Counter()
    adjustments = 0
    last_timestamp: float | None = None
    first_shape: tuple[int, int] | None = None
    try:
        for frame in iter_video_frames(video_path):
            decoded += 1
            timestamp_sources[frame.timestamp_source] += 1
            adjustments += int(frame.timestamp_adjusted_for_monotonicity)
            last_timestamp = frame.timestamp_seconds
            if first_shape is None:
                height, width = frame.image.shape[:2]
                first_shape = (width, height)
    except Exception as error:  # Keep inspection reusable for bad clips.
        return VideoInspection(
            path=str(video_path),
            decoder_success=False,
            error=f"decoder_read_failed: {error}",
            reported_frame_count=reported_frames,
            decoded_frame_count=decoded,
            fps=reported_fps or probe.fps,
            width=reported_width or probe.width,
            height=reported_height or probe.height,
            duration_seconds=probe.duration_seconds,
            timestamp_source=timestamp_sources.most_common(1)[0][0] if timestamp_sources else "unavailable",
            timestamp_adjustments=adjustments,
            frame_count_mismatch=reported_frames is not None and reported_frames != decoded,
            ffprobe_available=probe.available,
        )

    fps = reported_fps or probe.fps
    if fps is None and len(probe.frame_timestamps) > 1:
        deltas = [
            right - left
            for left, right in zip(probe.frame_timestamps, probe.frame_timestamps[1:])
            if left is not None and right is not None and right > left
        ]
        if deltas:
            fps = 1.0 / median(deltas)
    duration = probe.duration_seconds
    if duration is None and last_timestamp is not None:
        duration = last_timestamp + (1.0 / fps if fps and fps > 0 else 0.0)
    width = reported_width or probe.width or (first_shape[0] if first_shape else None)
    height = reported_height or probe.height or (first_shape[1] if first_shape else None)
    return VideoInspection(
        path=str(video_path),
        decoder_success=decoded > 0,
        error=None if decoded > 0 else "decoder_returned_no_frames",
        reported_frame_count=reported_frames,
        decoded_frame_count=decoded,
        fps=fps,
        width=width,
        height=height,
        duration_seconds=duration,
        timestamp_source=timestamp_sources.most_common(1)[0][0] if timestamp_sources else "unavailable",
        timestamp_adjustments=adjustments,
        frame_count_mismatch=reported_frames is not None and reported_frames != decoded,
        ffprobe_available=probe.available,
    )
