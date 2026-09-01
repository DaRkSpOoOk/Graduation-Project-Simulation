"""Lightweight, threshold-free baseline metrics for raw hand pose NPZ files."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import numpy as np


LANDMARK_COUNT = 21
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


def _as_json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _finite_float(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _rate(numerator: int, denominator: int) -> float | None:
    return 100.0 * numerator / denominator if denominator else None


def _longest_streak(values: np.ndarray) -> int:
    longest = current = 0
    for value in values.astype(bool):
        current = current + 1 if value else 0
        longest = max(longest, current)
    return int(longest)


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    values_array = np.asarray(list(values), dtype=np.float64)
    if values_array.size == 0:
        return None
    return float(np.percentile(values_array, percentile))


def _summary(values: Iterable[float], threshold: float | None = None) -> dict[str, Any]:
    values_array = np.asarray(list(values), dtype=np.float64)
    values_array = values_array[np.isfinite(values_array)]
    result: dict[str, Any] = {
        "count": int(values_array.size),
        "mean": float(np.mean(values_array)) if values_array.size else None,
        "median": float(np.median(values_array)) if values_array.size else None,
        "p95": float(np.percentile(values_array, 95)) if values_array.size else None,
        "max": float(np.max(values_array)) if values_array.size else None,
    }
    if threshold is not None:
        result["diagnostic_threshold"] = threshold
        result["count_above_diagnostic_threshold"] = int(np.sum(values_array > threshold))
    return result


def _canonical_hands(
    image: np.ndarray,
    world: np.ndarray,
    present: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a derived LEFT/RIGHT view without changing raw detector arrays."""

    frame_count = image.shape[0]
    canonical_image = np.full((frame_count, 2, LANDMARK_COUNT, 3), np.nan, dtype=np.float64)
    canonical_world = np.full((frame_count, 2, LANDMARK_COUNT, 3), np.nan, dtype=np.float64)
    canonical_present = np.zeros((frame_count, 2), dtype=bool)
    canonical_scores = np.full((frame_count, 2), np.nan, dtype=np.float64)
    for frame_index in range(frame_count):
        for detector_index in range(present.shape[1]):
            if not bool(present[frame_index, detector_index]):
                continue
            label = str(labels[frame_index, detector_index]).strip().casefold()
            slot = 0 if label == "left" else 1 if label == "right" else None
            if slot is None:
                continue
            score = float(scores[frame_index, detector_index])
            should_replace = not canonical_present[frame_index, slot]
            if not should_replace and math.isfinite(score):
                previous_score = canonical_scores[frame_index, slot]
                should_replace = not math.isfinite(previous_score) or score > previous_score
            if should_replace:
                canonical_image[frame_index, slot] = image[frame_index, detector_index]
                canonical_world[frame_index, slot] = world[frame_index, detector_index]
                canonical_present[frame_index, slot] = True
                canonical_scores[frame_index, slot] = score
    return canonical_image, canonical_world, canonical_present, canonical_scores


def _wrist_deltas(points: np.ndarray, present: np.ndarray) -> list[float]:
    values: list[float] = []
    for slot in range(points.shape[1]):
        for frame_index in range(1, points.shape[0]):
            if not present[frame_index - 1, slot] or not present[frame_index, slot]:
                continue
            previous = points[frame_index - 1, slot, 0]
            current = points[frame_index, slot, 0]
            if np.all(np.isfinite(previous)) and np.all(np.isfinite(current)):
                values.append(float(np.linalg.norm(current - previous)))
    return values


def _jitter(points: np.ndarray, present: np.ndarray, timestamps: np.ndarray) -> tuple[list[float], list[float]]:
    second_differences: list[float] = []
    accelerations: list[float] = []
    for slot in range(points.shape[1]):
        for frame_index in range(1, points.shape[0] - 1):
            if not np.all(present[frame_index - 1 : frame_index + 2, slot]):
                continue
            p0, p1, p2 = points[frame_index - 1 : frame_index + 2, slot, 0]
            if not all(np.all(np.isfinite(point)) for point in (p0, p1, p2)):
                continue
            dt0 = float(timestamps[frame_index] - timestamps[frame_index - 1])
            dt1 = float(timestamps[frame_index + 1] - timestamps[frame_index])
            if dt0 <= 0 or dt1 <= 0:
                continue
            v0 = (p1 - p0) / dt0
            v1 = (p2 - p1) / dt1
            second_differences.append(float(np.linalg.norm(v1 * dt1 - v0 * dt0)))
            accelerations.append(float(np.linalg.norm(v1 - v0) / ((dt0 + dt1) / 2.0)))
    return second_differences, accelerations


def _bone_variation(world: np.ndarray, present: np.ndarray) -> list[float]:
    coefficients: list[float] = []
    for slot in range(world.shape[1]):
        for start, end in HAND_CONNECTIONS:
            lengths: list[float] = []
            for frame_index in range(world.shape[0]):
                if not present[frame_index, slot]:
                    continue
                first = world[frame_index, slot, start]
                second = world[frame_index, slot, end]
                if np.all(np.isfinite(first)) and np.all(np.isfinite(second)):
                    lengths.append(float(np.linalg.norm(second - first)))
            if len(lengths) >= 2:
                average_length = float(np.mean(lengths))
                if average_length > 0:
                    coefficients.append(float(np.std(lengths) / average_length))
    return coefficients


def evaluate_arrays(
    image_landmarks: np.ndarray,
    world_landmarks: np.ndarray,
    hand_present: np.ndarray,
    handedness_labels: np.ndarray,
    handedness_scores: np.ndarray,
    timestamps_seconds: np.ndarray,
    runtime_seconds: float | None = None,
    inference_seconds: float | None = None,
) -> dict[str, Any]:
    """Calculate baseline statistics from raw detector-order arrays."""

    image = np.asarray(image_landmarks)
    world = np.asarray(world_landmarks)
    present = np.asarray(hand_present, dtype=bool)
    labels = np.asarray(handedness_labels)
    scores = np.asarray(handedness_scores, dtype=np.float64)
    timestamps = np.asarray(timestamps_seconds, dtype=np.float64)
    if image.ndim != 4 or image.shape[2:] != (LANDMARK_COUNT, 3):
        raise ValueError(f"Expected image landmarks [frames,hands,21,3], got {image.shape}")
    if world.shape != image.shape or present.shape != image.shape[:2] or labels.shape != present.shape or scores.shape != present.shape:
        raise ValueError("Raw landmark arrays have incompatible shapes")
    if timestamps.shape != (image.shape[0],):
        raise ValueError("timestamps_seconds must have one value per frame")

    frame_count = int(image.shape[0])
    any_hand = present.any(axis=1)
    canonical_image, canonical_world, canonical_present, canonical_scores = _canonical_hands(
        image, world, present, labels, scores
    )
    both = canonical_present.all(axis=1)
    left = canonical_present[:, 0]
    right = canonical_present[:, 1]

    signatures: list[tuple[str, ...]] = []
    duplicate_label_frames = 0
    for frame_index in range(frame_count):
        frame_labels = [
            str(labels[frame_index, detector_index]).strip().casefold()
            for detector_index in range(present.shape[1])
            if present[frame_index, detector_index] and str(labels[frame_index, detector_index]).strip()
        ]
        if len(frame_labels) != len(set(frame_labels)):
            duplicate_label_frames += 1
        signatures.append(tuple(sorted(frame_labels)))
    label_set_changes = sum(signatures[i] != signatures[i - 1] for i in range(1, frame_count))
    detector_order_label_changes = 0
    order_reversals = 0
    for frame_index in range(1, frame_count):
        previous = [str(value).strip().casefold() for value in labels[frame_index - 1]]
        current = [str(value).strip().casefold() for value in labels[frame_index]]
        for detector_index in range(present.shape[1]):
            if present[frame_index - 1, detector_index] and present[frame_index, detector_index] and previous[detector_index] and current[detector_index]:
                detector_order_label_changes += previous[detector_index] != current[detector_index]
        if present.shape[1] >= 2 and present[frame_index - 1].all() and present[frame_index].all():
            if previous[:2] == ["left", "right"] and current[:2] == ["right", "left"]:
                order_reversals += 1
            elif previous[:2] == ["right", "left"] and current[:2] == ["left", "right"]:
                order_reversals += 1

    crossing_events = 0
    for frame_index in range(1, frame_count):
        if not (both[frame_index - 1] and both[frame_index]):
            continue
        previous_delta = canonical_image[frame_index - 1, 0, 0, 0] - canonical_image[frame_index - 1, 1, 0, 0]
        current_delta = canonical_image[frame_index, 0, 0, 0] - canonical_image[frame_index, 1, 0, 0]
        if np.isfinite(previous_delta) and np.isfinite(current_delta) and previous_delta * current_delta < 0:
            crossing_events += 1

    confidence_values = scores[present & np.isfinite(scores)]
    palm_lengths: list[float] = []
    for slot in range(2):
        for frame_index in range(frame_count):
            if canonical_present[frame_index, slot]:
                wrist = canonical_world[frame_index, slot, 0]
                middle_mcp = canonical_world[frame_index, slot, 9]
                if np.all(np.isfinite(wrist)) and np.all(np.isfinite(middle_mcp)):
                    palm_lengths.append(float(np.linalg.norm(middle_mcp - wrist)))
    second_differences, accelerations = _jitter(canonical_world, canonical_present, timestamps)
    bone_coefficients = _bone_variation(canonical_world, canonical_present)
    world_wrist_jumps = _wrist_deltas(canonical_world, canonical_present)
    image_wrist_jumps = _wrist_deltas(canonical_image, canonical_present)
    detector_order_world_wrist_jumps = _wrist_deltas(world, present)
    detector_order_image_wrist_jumps = _wrist_deltas(image, present)
    observed_world = world[present]
    observed_image = image[present]
    world_finite_rate = float(np.isfinite(observed_world).mean()) if observed_world.size else None
    image_finite_rate = float(np.isfinite(observed_image).mean()) if observed_image.size else None
    thumb_observed = world[present][:, 1:5] if observed_world.size else np.empty((0, 4, 3))
    thumb_missing_rate = float(1.0 - np.isfinite(thumb_observed).all(axis=2).mean()) if thumb_observed.size else None

    result: dict[str, Any] = {
        "total_frames": frame_count,
        "frames_with_no_hands": int((~any_hand).sum()),
        "frames_with_at_least_one_hand": int(any_hand.sum()),
        "frames_with_left_hand": int(left.sum()),
        "frames_with_right_hand": int(right.sum()),
        "frames_with_both_hands": int(both.sum()),
        "missing_frame_percentage": _rate(int((~any_hand).sum()), frame_count),
        "left_hand_detection_rate": _rate(int(left.sum()), frame_count),
        "right_hand_detection_rate": _rate(int(right.sum()), frame_count),
        "both_hand_detection_rate": _rate(int(both.sum()), frame_count),
        "longest_missing_streak_left_frames": _longest_streak(~left),
        "longest_missing_streak_right_frames": _longest_streak(~right),
        "handedness_label_set_changes": int(label_set_changes),
        "detector_order_handedness_changes": int(detector_order_label_changes),
        "duplicate_handedness_frames": int(duplicate_label_frames),
        "potential_identity_instability": {
            "heuristic_event_count": int(duplicate_label_frames + order_reversals + crossing_events),
            "duplicate_label_frames": int(duplicate_label_frames),
            "detector_order_reversals": int(order_reversals),
            "left_right_wrist_x_crossing_events": int(crossing_events),
            "limitation": "heuristic only; no identity ground truth or temporal tracker is present in raw output",
        },
        "handedness_confidence": {
            "count": int(confidence_values.size),
            "mean": _finite_float(np.mean(confidence_values) if confidence_values.size else None),
            "p05": _finite_float(np.percentile(confidence_values, 5) if confidence_values.size else None),
            "p95": _finite_float(np.percentile(confidence_values, 95) if confidence_values.size else None),
        },
        "hand_presence_confidence": None,
        "hand_detection_confidence": None,
        "tracking_confidence": None,
        "confidence_limitation": "The current Python HandLandmarker result object exposes handedness scores but not per-frame presence, detection, or tracking scores.",
        "wrist_coordinate_jumps_world_m": _summary(world_wrist_jumps, threshold=0.10),
        "wrist_coordinate_jumps_image_normalized": _summary(image_wrist_jumps, threshold=0.20),
        "detector_order_wrist_coordinate_jumps_world_m": _summary(detector_order_world_wrist_jumps, threshold=0.10),
        "detector_order_wrist_coordinate_jumps_image_normalized": _summary(detector_order_image_wrist_jumps, threshold=0.20),
        "temporal_jitter_world": {
            "second_difference_m_per_frame": _summary(second_differences),
            "acceleration_m_per_s2": _summary(accelerations),
            "median_palm_length_m": float(np.median(palm_lengths)) if palm_lengths else None,
            "normalized_second_difference": (
                float(np.mean(second_differences) / np.median(palm_lengths)) if second_differences and palm_lengths and np.median(palm_lengths) > 0 else None
            ),
        },
        "bone_length_variation": {
            "observed_bone_coefficients": int(len(bone_coefficients)),
            "mean_coefficient_of_variation": float(np.mean(bone_coefficients)) if bone_coefficients else None,
            "median_coefficient_of_variation": float(np.median(bone_coefficients)) if bone_coefficients else None,
            "max_coefficient_of_variation": float(np.max(bone_coefficients)) if bone_coefficients else None,
        },
        "coordinate_completeness": {
            "image_finite_rate_among_returned_hands": image_finite_rate,
            "world_finite_rate_among_returned_hands": world_finite_rate,
            "thumb_landmark_missing_rate_among_returned_hands": thumb_missing_rate,
        },
        "finger_crossing_behavior": {
            "measured": False,
            "limitation": "No anatomical finger-crossing classifier or ground-truth labels are included in this baseline.",
        },
        "hand_hand_occlusion_behavior": {
            "measured": False,
            "limitation": "The task result does not expose an occlusion cause; missingness and both-hand counts are reported instead.",
        },
        "runtime_seconds": _finite_float(runtime_seconds),
        "inference_seconds": _finite_float(inference_seconds),
        "effective_processing_fps": frame_count / runtime_seconds if runtime_seconds and runtime_seconds > 0 else None,
        "inference_fps": frame_count / inference_seconds if inference_seconds and inference_seconds > 0 else None,
    }
    return {key: _as_json_value(value) for key, value in result.items()}


def evaluate_npz(path: str | Path, runtime_seconds: float | None = None, inference_seconds: float | None = None) -> dict[str, Any]:
    """Load a raw MediaPipe NPZ and calculate metrics."""

    with np.load(path, allow_pickle=False) as data:
        metrics = evaluate_arrays(
            data["hand_landmarks_image"],
            data["hand_landmarks_world"],
            data["hand_present"],
            data["handedness_labels"],
            data["handedness_scores"],
            data["timestamps_seconds"],
            runtime_seconds=runtime_seconds,
            inference_seconds=inference_seconds,
        )
    metrics["raw_npz_path"] = str(path)
    return metrics


def _average_metric(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    return mean(values) if values else None


def aggregate_metrics(records: Iterable[dict[str, Any]], failed_videos: int = 0) -> dict[str, Any]:
    """Aggregate per-video metrics with frame-weighted detection rates."""

    successful = [record for record in records if record.get("total_frames", 0) > 0]
    total_frames = sum(int(record["total_frames"]) for record in successful)
    count_for = lambda key: sum(int(record.get(key, 0)) for record in successful)
    total_runtime = sum(float(record.get("runtime_seconds") or 0.0) for record in successful)
    total_inference = sum(float(record.get("inference_seconds") or 0.0) for record in successful)
    result: dict[str, Any] = {
        "videos_observed": len(successful),
        "videos_failed_or_skipped": int(failed_videos),
        "total_frames": total_frames,
        "frames_with_no_hands": count_for("frames_with_no_hands"),
        "frames_with_at_least_one_hand": count_for("frames_with_at_least_one_hand"),
        "frames_with_left_hand": count_for("frames_with_left_hand"),
        "frames_with_right_hand": count_for("frames_with_right_hand"),
        "frames_with_both_hands": count_for("frames_with_both_hands"),
        "missing_frame_percentage": _rate(count_for("frames_with_no_hands"), total_frames),
        "left_hand_detection_rate": _rate(count_for("frames_with_left_hand"), total_frames),
        "right_hand_detection_rate": _rate(count_for("frames_with_right_hand"), total_frames),
        "both_hand_detection_rate": _rate(count_for("frames_with_both_hands"), total_frames),
        "longest_missing_streak_left_frames": max((int(r.get("longest_missing_streak_left_frames", 0)) for r in successful), default=0),
        "longest_missing_streak_right_frames": max((int(r.get("longest_missing_streak_right_frames", 0)) for r in successful), default=0),
        "handedness_label_set_changes_total": sum(int(r.get("handedness_label_set_changes", 0)) for r in successful),
        "detector_order_handedness_changes_total": sum(int(r.get("detector_order_handedness_changes", 0)) for r in successful),
        "potential_identity_instability_events_total": sum(
            int((r.get("potential_identity_instability") or {}).get("heuristic_event_count", 0)) for r in successful
        ),
        "wrist_coordinate_jump_observations_total": sum(
            int((r.get("wrist_coordinate_jumps_world_m") or {}).get("count", 0)) for r in successful
        ),
        "wrist_coordinate_jumps_above_0_10m_total": sum(
            int((r.get("wrist_coordinate_jumps_world_m") or {}).get("count_above_diagnostic_threshold", 0))
            for r in successful
        ),
        "max_wrist_coordinate_jump_world_m": max(
            (float((r.get("wrist_coordinate_jumps_world_m") or {}).get("max")) for r in successful if (r.get("wrist_coordinate_jumps_world_m") or {}).get("max") is not None),
            default=None,
        ),
        "per_video_mean_wrist_jump_p95_world_m": _average_nested_metric(successful, "wrist_coordinate_jumps_world_m", "p95"),
        "detector_order_wrist_coordinate_jump_observations_total": sum(
            int((r.get("detector_order_wrist_coordinate_jumps_world_m") or {}).get("count", 0)) for r in successful
        ),
        "detector_order_wrist_jumps_above_0_10m_total": sum(
            int((r.get("detector_order_wrist_coordinate_jumps_world_m") or {}).get("count_above_diagnostic_threshold", 0))
            for r in successful
        ),
        "max_detector_order_wrist_jump_world_m": max(
            (
                float((r.get("detector_order_wrist_coordinate_jumps_world_m") or {}).get("max"))
                for r in successful
                if (r.get("detector_order_wrist_coordinate_jumps_world_m") or {}).get("max") is not None
            ),
            default=None,
        ),
        "runtime_seconds_total": total_runtime,
        "inference_seconds_total": total_inference,
        "effective_processing_fps": total_frames / total_runtime if total_runtime > 0 else None,
        "inference_fps": total_frames / total_inference if total_inference > 0 else None,
        "per_video_average_runtime_seconds": _average_metric(successful, "runtime_seconds"),
        "per_video_median_effective_processing_fps": (
            median(float(record["effective_processing_fps"]) for record in successful if record.get("effective_processing_fps") is not None)
            if any(record.get("effective_processing_fps") is not None for record in successful)
            else None
        ),
        "per_video_mean_missing_frame_percentage": _average_metric(successful, "missing_frame_percentage"),
        "per_video_mean_bone_length_cv": _average_nested_metric(successful, "bone_length_variation", "mean_coefficient_of_variation"),
        "per_video_max_bone_length_cv": _average_nested_metric(successful, "bone_length_variation", "max_coefficient_of_variation"),
        "per_video_mean_jitter_m_per_frame": _average_nested_metric(
            successful, "temporal_jitter_world", "second_difference_m_per_frame", "mean"
        ),
        "per_video_mean_jitter_p95_m_per_frame": _average_nested_metric(
            successful, "temporal_jitter_world", "second_difference_m_per_frame", "p95"
        ),
        "per_video_mean_handedness_confidence": _average_nested_metric(successful, "handedness_confidence", "mean"),
    }
    return result


def _average_nested_metric(records: list[dict[str, Any]], outer: str, *inner: str) -> float | None:
    values: list[float] = []
    for record in records:
        value: Any = record.get(outer)
        for key in inner:
            value = value.get(key) if isinstance(value, dict) else None
        if value is not None:
            values.append(float(value))
    return mean(values) if values else None


def load_metadata(path: str | Path) -> dict[str, Any]:
    """Read the JSON metadata string embedded in a raw NPZ."""

    with np.load(path, allow_pickle=False) as data:
        raw = data["metadata_json"].item()
    return json.loads(str(raw))
