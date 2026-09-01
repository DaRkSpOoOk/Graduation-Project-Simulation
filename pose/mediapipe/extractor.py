"""Raw MediaPipe Hand Landmarker extraction for decoded videos.

This module intentionally keeps detector order and detector-provided
handedness in the NPZ output.  Any left/right convenience view is derived at
evaluation time and is never written back over the raw detector output.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pose.common.schema import HandPoseFrame, Landmark2D, Landmark3D
from video_io.reader import VideoInspection, iter_video_frames
from visualization.hand_overlay import draw_hand_overlay


LANDMARK_COUNT = 21
DEFAULT_MAX_HANDS = 2
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)


@dataclass(frozen=True, slots=True)
class MediaPipeConfig:
    """Task options recorded alongside every raw extraction."""

    model_path: Path
    num_hands: int = DEFAULT_MAX_HANDS
    min_hand_detection_confidence: float = 0.5
    min_hand_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    delegate: str = "CPU"

    def __post_init__(self) -> None:
        if self.num_hands < 1:
            raise ValueError("num_hands must be positive")
        for name in (
            "min_hand_detection_confidence",
            "min_hand_presence_confidence",
            "min_tracking_confidence",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
        if self.delegate.upper() not in {"CPU", "GPU"}:
            raise ValueError("delegate must be CPU or GPU")


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    sample_id: str
    status: str
    raw_npz_path: str
    overlay_path: str | None
    frame_count: int
    inference_seconds: float
    runtime_seconds: float
    average_inference_fps: float | None
    effective_processing_fps: float | None
    overlay_error: str | None
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "status": self.status,
            "raw_npz_path": self.raw_npz_path,
            "overlay_path": self.overlay_path,
            "frame_count": self.frame_count,
            "inference_seconds": self.inference_seconds,
            "runtime_seconds": self.runtime_seconds,
            "average_inference_fps": self.average_inference_fps,
            "effective_processing_fps": self.effective_processing_fps,
            "overlay_error": self.overlay_error,
            "error": self.error,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _category_value(category: Any, *names: str) -> Any:
    for name in names:
        value = getattr(category, name, None)
        if value is not None:
            return value
    return None


def _landmark_value(landmark: Any, name: str) -> float:
    value = _category_value(landmark, name)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _result_components(result: Any) -> tuple[list, list, list]:
    image_landmarks = list(getattr(result, "hand_landmarks", []) or [])
    world_landmarks = list(getattr(result, "hand_world_landmarks", []) or [])
    handedness = list(getattr(result, "handedness", []) or [])
    return image_landmarks, world_landmarks, handedness


def result_to_common_frames(result: Any, frame_index: int, timestamp_seconds: float) -> list[HandPoseFrame]:
    """Map one official result into the existing extractor-agnostic schema."""

    image_landmarks, world_landmarks, handedness = _result_components(result)
    common: list[HandPoseFrame] = []
    for detector_index, image_points in enumerate(image_landmarks):
        world_points = world_landmarks[detector_index] if detector_index < len(world_landmarks) else []
        categories = handedness[detector_index] if detector_index < len(handedness) else []
        category = categories[0] if categories else None
        label = _category_value(category, "category_name", "display_name") if category else None
        score_value = _category_value(category, "score") if category else None
        try:
            score = float(score_value) if score_value is not None else None
        except (TypeError, ValueError):
            score = None
        points_2d = [Landmark2D(_landmark_value(point, "x"), _landmark_value(point, "y")) for point in image_points]
        points_3d = [
            Landmark3D(_landmark_value(point, "x"), _landmark_value(point, "y"), _landmark_value(point, "z"))
            for point in world_points
        ]
        flags: list[str] = []
        if len(image_points) != LANDMARK_COUNT:
            flags.append(f"image_landmark_count_{len(image_points)}")
        if len(world_points) != LANDMARK_COUNT:
            flags.append(f"world_landmark_count_{len(world_points)}")
        common.append(
            HandPoseFrame(
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                hand_present=True,
                handedness_label=str(label) if label is not None else None,
                handedness_confidence=score,
                detection_confidence=None,
                landmarks_2d=points_2d,
                landmarks_3d=points_3d,
                wrist_position=points_3d[0] if points_3d else None,
                extractor_metadata={
                    "extractor": "mediapipe_hand_landmarker",
                    "detector_index": detector_index,
                    "screen_landmark_z_preserved_in_npz": True,
                },
                quality_flags=flags,
            )
        )
    return common


def _frame_arrays(result: Any, max_hands: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return exact per-result numeric arrays in detector order."""

    image_landmarks, world_landmarks, handedness = _result_components(result)
    image = np.full((max_hands, LANDMARK_COUNT, 3), np.nan, dtype=np.float32)
    world = np.full((max_hands, LANDMARK_COUNT, 3), np.nan, dtype=np.float32)
    present = np.zeros(max_hands, dtype=np.bool_)
    labels = np.full(max_hands, "", dtype="<U16")
    scores = np.full(max_hands, np.nan, dtype=np.float32)
    category_indices = np.full(max_hands, -1, dtype=np.int32)
    for detector_index, points in enumerate(image_landmarks[:max_hands]):
        present[detector_index] = True
        for landmark_index, point in enumerate(points[:LANDMARK_COUNT]):
            image[detector_index, landmark_index] = [
                _landmark_value(point, "x"),
                _landmark_value(point, "y"),
                _landmark_value(point, "z"),
            ]
        if detector_index < len(world_landmarks):
            for landmark_index, point in enumerate(world_landmarks[detector_index][:LANDMARK_COUNT]):
                world[detector_index, landmark_index] = [
                    _landmark_value(point, "x"),
                    _landmark_value(point, "y"),
                    _landmark_value(point, "z"),
                ]
        categories = handedness[detector_index] if detector_index < len(handedness) else []
        if categories:
            category = categories[0]
            label = _category_value(category, "category_name", "display_name")
            if label is not None:
                labels[detector_index] = str(label)
            score = _category_value(category, "score")
            try:
                scores[detector_index] = float(score)
            except (TypeError, ValueError):
                pass
            category_index = _category_value(category, "index")
            try:
                category_indices[detector_index] = int(category_index)
            except (TypeError, ValueError):
                pass
    return image, world, present, labels, scores, category_indices


def _create_landmarker(config: MediaPipeConfig) -> Any:
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import vision
    except ModuleNotFoundError as error:  # pragma: no cover - environment error.
        raise RuntimeError("MediaPipe is required for extraction; install the project dependencies first") from error

    delegate_name = config.delegate.upper()
    delegate = mp.tasks.BaseOptions.Delegate.GPU if delegate_name == "GPU" else mp.tasks.BaseOptions.Delegate.CPU
    base_options = mp.tasks.BaseOptions(model_asset_path=str(config.model_path), delegate=delegate)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=config.num_hands,
        min_hand_detection_confidence=config.min_hand_detection_confidence,
        min_hand_presence_confidence=config.min_hand_presence_confidence,
        min_tracking_confidence=config.min_tracking_confidence,
    )
    return vision.HandLandmarker.create_from_options(options)


def _empty_payload(max_hands: int) -> dict[str, list]:
    return {
        "frame_indices": [],
        "timestamps_seconds": [],
        "mediapipe_timestamps_ms": [],
        "hand_landmarks_image": [],
        "hand_landmarks_world": [],
        "hand_present": [],
        "handedness_labels": [],
        "handedness_scores": [],
        "handedness_category_indices": [],
        "common_pose_hand_counts": [],
        "max_hands": max_hands,
    }


def _save_payload(path: Path, payload: dict[str, Any], metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "frame_indices": np.asarray(payload["frame_indices"], dtype=np.int64),
        "timestamps_seconds": np.asarray(payload["timestamps_seconds"], dtype=np.float64),
        "mediapipe_timestamps_ms": np.asarray(payload["mediapipe_timestamps_ms"], dtype=np.int64),
        "hand_landmarks_image": np.asarray(payload["hand_landmarks_image"], dtype=np.float32).reshape(
            (-1, payload["max_hands"], LANDMARK_COUNT, 3)
        ),
        "hand_landmarks_world": np.asarray(payload["hand_landmarks_world"], dtype=np.float32).reshape(
            (-1, payload["max_hands"], LANDMARK_COUNT, 3)
        ),
        "hand_present": np.asarray(payload["hand_present"], dtype=np.bool_).reshape((-1, payload["max_hands"])),
        "handedness_labels": np.asarray(payload["handedness_labels"], dtype="<U16").reshape(
            (-1, payload["max_hands"])
        ),
        "handedness_scores": np.asarray(payload["handedness_scores"], dtype=np.float32).reshape(
            (-1, payload["max_hands"])
        ),
        "handedness_category_indices": np.asarray(payload["handedness_category_indices"], dtype=np.int32).reshape(
            (-1, payload["max_hands"])
        ),
        "common_pose_hand_counts": np.asarray(payload["common_pose_hand_counts"], dtype=np.int32),
        "metadata_json": np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
    }
    np.savez_compressed(path, **arrays)


def extract_video(
    video_path: str | Path,
    raw_npz_path: str | Path,
    overlay_path: str | Path | None,
    config: MediaPipeConfig,
    inspection: VideoInspection,
    sample_id: str,
    source_metadata: dict[str, Any] | None = None,
) -> ExtractionResult:
    """Run one VIDEO-mode extraction after validation and save raw NPZ/overlay."""

    if not inspection.decoder_success:
        return ExtractionResult(
            sample_id=sample_id,
            status="skipped_decoder_failure",
            raw_npz_path=str(raw_npz_path),
            overlay_path=str(overlay_path) if overlay_path else None,
            frame_count=inspection.decoded_frame_count,
            inference_seconds=0.0,
            runtime_seconds=0.0,
            average_inference_fps=None,
            effective_processing_fps=None,
            overlay_error=None,
            error=inspection.error,
        )

    started = time.perf_counter()
    payload = _empty_payload(config.num_hands)
    overlay_writer: Any = None
    overlay_error: str | None = None
    inference_seconds = 0.0
    error: str | None = None
    try:
        import cv2

        if overlay_path is not None:
            overlay_file = Path(overlay_path)
            overlay_file.parent.mkdir(parents=True, exist_ok=True)
            width = inspection.width or 0
            height = inspection.height or 0
            fps = inspection.fps or 30.0
            if width > 0 and height > 0:
                overlay_writer = cv2.VideoWriter(
                    str(overlay_file),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
                if not overlay_writer.isOpened():
                    overlay_writer.release()
                    overlay_writer = None
                    overlay_error = "overlay_writer_open_failed"
            else:
                overlay_error = "overlay_dimensions_unavailable"

        with _create_landmarker(config) as landmarker:
            for frame in iter_video_frames(video_path):
                rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
                try:
                    import mediapipe as mp

                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                except Exception as conversion_error:
                    raise RuntimeError(f"MediaPipe image conversion failed: {conversion_error}") from conversion_error
                inference_started = time.perf_counter()
                result = landmarker.detect_for_video(mp_image, frame.timestamp_ms)
                inference_seconds += time.perf_counter() - inference_started
                image, world, present, labels, scores, category_indices = _frame_arrays(result, config.num_hands)
                common_frames = result_to_common_frames(result, frame.frame_index, frame.timestamp_seconds)
                payload["frame_indices"].append(frame.frame_index)
                payload["timestamps_seconds"].append(frame.timestamp_seconds)
                payload["mediapipe_timestamps_ms"].append(frame.timestamp_ms)
                payload["hand_landmarks_image"].append(image)
                payload["hand_landmarks_world"].append(world)
                payload["hand_present"].append(present)
                payload["handedness_labels"].append(labels)
                payload["handedness_scores"].append(scores)
                payload["handedness_category_indices"].append(category_indices)
                payload["common_pose_hand_counts"].append(len(common_frames))
                if overlay_writer is not None:
                    annotated = draw_hand_overlay(
                        frame.image,
                        image,
                        present,
                        labels,
                        scores,
                        frame.frame_index,
                        frame.timestamp_seconds,
                        HAND_CONNECTIONS,
                    )
                    overlay_writer.write(annotated)
    except Exception as extraction_error:  # Preserve already extracted raw frames for diagnosis.
        error = f"{type(extraction_error).__name__}: {extraction_error}"
    finally:
        if overlay_writer is not None:
            overlay_writer.release()

    runtime_seconds = time.perf_counter() - started
    metadata = {
        "schema": "mediapipe_hand_landmarker_raw_v1",
        "stage": "raw_pose",
        "sample_id": sample_id,
        "source_video": str(video_path),
        "source_video_sha256": _sha256(Path(video_path)) if Path(video_path).is_file() else None,
        "mediapipe_version": _mediapipe_version(),
        "running_mode": "VIDEO",
        "num_hands": config.num_hands,
        "min_hand_detection_confidence": config.min_hand_detection_confidence,
        "min_hand_presence_confidence": config.min_hand_presence_confidence,
        "min_tracking_confidence": config.min_tracking_confidence,
        "delegate_requested": config.delegate.upper(),
        "confidence_availability": {
            "handedness_score": "available_per_detected_hand",
            "hand_presence_confidence": "not_exposed_by_python_result_api",
            "hand_detection_confidence": "not_exposed_by_python_result_api",
            "tracking_confidence": "not_exposed_by_python_result_api",
        },
        "timestamp_policy": inspection.timestamp_source,
        "timestamp_adjustments": inspection.timestamp_adjustments,
        "inspection": inspection.to_dict(),
        "runtime_seconds": runtime_seconds,
        "inference_seconds": inference_seconds,
        "source_metadata": source_metadata or {},
        "common_schema": "pose.common.schema.HandPoseFrame used during result mapping; no schema changes",
        "raw_preservation": ["detector_order", "detector_handedness", "no_smoothing", "no_interpolation", "no_identity_correction"],
    }
    _save_payload(Path(raw_npz_path), payload, metadata)
    frame_count = len(payload["frame_indices"])
    status = "success" if error is None and frame_count > 0 else "failed"
    return ExtractionResult(
        sample_id=sample_id,
        status=status,
        raw_npz_path=str(raw_npz_path),
        overlay_path=str(overlay_path) if overlay_path else None,
        frame_count=frame_count,
        inference_seconds=inference_seconds,
        runtime_seconds=runtime_seconds,
        average_inference_fps=frame_count / inference_seconds if inference_seconds > 0 else None,
        effective_processing_fps=frame_count / runtime_seconds if runtime_seconds > 0 else None,
        overlay_error=overlay_error,
        error=error,
    )


def _mediapipe_version() -> str | None:
    try:
        import mediapipe as mp

        return getattr(mp, "__version__", None)
    except ModuleNotFoundError:
        return None
