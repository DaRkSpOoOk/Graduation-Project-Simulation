"""Neutral, extractor-agnostic metrics for the frozen pilot outputs."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .common_contract import COMMON_HAND_BONES_18, HandRecord, normalize_label, reconstructed_hand


def _finite_confidence(record: HandRecord) -> float:
    value = record.confidence
    return float(value) if value is not None and math.isfinite(float(value)) else float("-inf")


def choose_representatives(records: Iterable[HandRecord]) -> dict[str, HandRecord]:
    """Choose one deterministic valid row per handedness label.

    The highest *native* confidence is preferred within an extractor. The
    scores are never compared between MediaPipe and WiLoR because their
    semantics differ (handedness score versus detector confidence). Ties and
    missing scores fall back to the lowest raw detector/source index.
    """

    selected: dict[str, HandRecord] = {}
    for record in records:
        if not reconstructed_hand(record):
            continue
        label = normalize_label(record.handedness_label)
        if label is None:
            continue
        previous = selected.get(label)
        if previous is None or (_finite_confidence(record), -record.source_index) > (
            _finite_confidence(previous), -previous.source_index
        ):
            selected[label] = record
    return selected


def _group_by_frame(records: Iterable[HandRecord]) -> dict[int, list[HandRecord]]:
    grouped: dict[int, list[HandRecord]] = defaultdict(list)
    for record in records:
        grouped[record.frame_index].append(record)
    return dict(grouped)


def _longest_true_streak(values: Mapping[int, bool], frame_indices: Sequence[int]) -> int:
    longest = 0
    current = 0
    previous_index: int | None = None
    for frame_index in frame_indices:
        if previous_index is None or frame_index != previous_index + 1:
            current = 0
        if values.get(frame_index, False):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
        previous_index = frame_index
    return longest


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _image_position(record: HandRecord) -> np.ndarray | None:
    """Return a normalized image-space hand-centre proxy for both systems."""

    if record.system == "mediapipe" and record.image_landmarks is not None:
        points = np.asarray(record.image_landmarks, dtype=np.float64)
        if points.shape != (21, 3) or not np.isfinite(points[:, :2]).all():
            return None
        xy = points[:, :2]
        return (np.min(xy, axis=0) + np.max(xy, axis=0)) / 2.0
    if record.system == "wilor" and record.mano_references:
        try:
            center = np.asarray(record.mano_references["box_center_xy"], dtype=np.float64)
            image_size = np.asarray(record.mano_references["img_size_wh"], dtype=np.float64)
        except (KeyError, TypeError, ValueError):
            return None
        if center.shape != (2,) or image_size.shape != (2,) or not np.isfinite(center).all() or not np.isfinite(image_size).all():
            return None
        if np.any(image_size <= 0):
            return None
        return center / image_size
    return None


def _common_swap_events(representatives: Mapping[int, Mapping[str, HandRecord]], frame_indices: Sequence[int]) -> int:
    """Count strict lower-cost left/right assignment reversals in 2D.

    Both systems use the same normalized image-space hand-centre proxy. A
    suspected swap is a consecutive-frame pair where assigning current left to
    previous right and current right to previous left has a strictly lower
    total Euclidean displacement than the reported label assignment. This is
    a heuristic, not identity ground truth, and it never changes the selected
    labels or raw arrays.
    """

    count = 0
    previous: tuple[int, dict[str, np.ndarray]] | None = None
    for frame_index in frame_indices:
        current_records = representatives.get(frame_index, {})
        if not {"left", "right"} <= current_records.keys():
            previous = None
            continue
        positions = {
            label: _image_position(current_records[label])
            for label in ("left", "right")
        }
        if any(position is None for position in positions.values()):
            previous = None
            continue
        current = {label: position for label, position in positions.items() if position is not None}
        if previous is not None and frame_index == previous[0] + 1:
            unswapped = float(np.linalg.norm(current["left"] - previous[1]["left"])) + float(
                np.linalg.norm(current["right"] - previous[1]["right"])
            )
            swapped = float(np.linalg.norm(current["left"] - previous[1]["right"])) + float(
                np.linalg.norm(current["right"] - previous[1]["left"])
            )
            if swapped < unswapped:
                count += 1
        previous = (frame_index, current)
    return count


def _normalized_hand_pose(record: HandRecord) -> np.ndarray | None:
    if not reconstructed_hand(record) or record.landmarks_3d is None:
        return None
    points = np.asarray(record.landmarks_3d, dtype=np.float64)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        return None
    scale = float(np.linalg.norm(points[9] - points[0]))
    if not math.isfinite(scale) or scale <= 1e-12:
        return None
    return (points - points[0]) / scale


def _normalized_second_difference(
    representatives: Mapping[int, Mapping[str, HandRecord]], frame_indices: Sequence[int]
) -> dict[str, Any]:
    """Compute the same scale-normalized 21-joint second difference for both."""

    values: list[float] = []
    for label in ("left", "right"):
        poses: dict[int, np.ndarray] = {}
        for frame_index in frame_indices:
            record = representatives.get(frame_index, {}).get(label)
            if record is not None:
                pose = _normalized_hand_pose(record)
                if pose is not None:
                    poses[frame_index] = pose
        for position in range(1, len(frame_indices) - 1):
            previous_index, current_index, next_index = frame_indices[position - 1 : position + 2]
            if next_index != current_index + 1 or current_index != previous_index + 1:
                continue
            if not {previous_index, current_index, next_index} <= poses.keys():
                continue
            second_difference = poses[next_index] - 2.0 * poses[current_index] + poses[previous_index]
            # Mean per-joint L2 makes the summary independent of the number of
            # joints while retaining the exact 21-joint representation.
            values.append(float(np.linalg.norm(second_difference, axis=1).mean()))
    return {
        "operator": "q[t+1] - 2*q[t] + q[t-1]",
        "coordinate_normalization": "root-center at joint 0; divide each frame by norm(joint 9 - joint 0)",
        "units": "dimensionless per frame",
        "distribution": _distribution(values),
    }


def _bone_length_cv(representatives: Mapping[int, Mapping[str, HandRecord]], frame_indices: Sequence[int]) -> dict[str, Any]:
    per_edge: dict[str, list[float]] = {f"{start}-{end}": [] for start, end in COMMON_HAND_BONES_18}
    for frame_index in frame_indices:
        for record in representatives.get(frame_index, {}).values():
            if not reconstructed_hand(record) or record.landmarks_3d is None:
                continue
            points = np.asarray(record.landmarks_3d, dtype=np.float64)
            for start, end in COMMON_HAND_BONES_18:
                length = float(np.linalg.norm(points[end] - points[start]))
                if math.isfinite(length) and length > 0:
                    per_edge[f"{start}-{end}"].append(length)

    per_edge_cv: dict[str, float | None] = {}
    all_cvs: list[float] = []
    for edge, lengths in per_edge.items():
        if len(lengths) < 2:
            per_edge_cv[edge] = None
            continue
        array = np.asarray(lengths, dtype=np.float64)
        average = float(np.mean(array))
        cv = float(np.std(array) / average) if average > 0 else None
        per_edge_cv[edge] = cv
        if cv is not None and math.isfinite(cv):
            all_cvs.append(cv)
    return {
        "edge_set": "COMMON_HAND_BONES_18",
        "edge_count": len(COMMON_HAND_BONES_18),
        "per_edge_cv": per_edge_cv,
        "distribution_across_edges": _distribution(all_cvs),
    }


def evaluate_video(
    sample_id: str,
    records: Sequence[HandRecord],
    total_frames: int,
    *,
    timing: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute harmonized frame/hand metrics for one manifest video."""

    frame_indices = list(range(total_frames))
    grouped = _group_by_frame(records)
    valid_by_frame: dict[int, list[HandRecord]] = {
        frame_index: [record for record in grouped.get(frame_index, []) if reconstructed_hand(record)]
        for frame_index in frame_indices
    }
    representatives = {
        frame_index: choose_representatives(valid_by_frame[frame_index])
        for frame_index in frame_indices
    }

    left_inclusive = {frame_index: any(normalize_label(record.handedness_label) == "left" for record in valid_by_frame[frame_index]) for frame_index in frame_indices}
    right_inclusive = {frame_index: any(normalize_label(record.handedness_label) == "right" for record in valid_by_frame[frame_index]) for frame_index in frame_indices}
    any_hand = {frame_index: bool(valid_by_frame[frame_index]) for frame_index in frame_indices}
    both = {frame_index: left_inclusive[frame_index] and right_inclusive[frame_index] for frame_index in frame_indices}
    left_only = {frame_index: left_inclusive[frame_index] and not right_inclusive[frame_index] for frame_index in frame_indices}
    right_only = {frame_index: right_inclusive[frame_index] and not left_inclusive[frame_index] for frame_index in frame_indices}

    duplicate_left = sum(sum(normalize_label(record.handedness_label) == "left" for record in valid_by_frame[index]) > 1 for index in frame_indices)
    duplicate_right = sum(sum(normalize_label(record.handedness_label) == "right" for record in valid_by_frame[index]) > 1 for index in frame_indices)
    extra_frames = sum(len(valid_by_frame[index]) > 2 for index in frame_indices)

    def count(values: Mapping[int, bool]) -> int:
        return sum(values.values())

    result: dict[str, Any] = {
        "sample_id": sample_id,
        "total_frames": total_frames,
        "frames_left_inclusive": count(left_inclusive),
        "frames_right_inclusive": count(right_inclusive),
        "frames_both": count(both),
        "frames_left_only": count(left_only),
        "frames_right_only": count(right_only),
        "frames_no_hand": total_frames - count(any_hand),
        "frames_at_least_one": count(any_hand),
        "coverage_rates_pct": {
            "left_inclusive": 100.0 * count(left_inclusive) / total_frames if total_frames else None,
            "right_inclusive": 100.0 * count(right_inclusive) / total_frames if total_frames else None,
            "both": 100.0 * count(both) / total_frames if total_frames else None,
            "left_only": 100.0 * count(left_only) / total_frames if total_frames else None,
            "right_only": 100.0 * count(right_only) / total_frames if total_frames else None,
            "no_hand": 100.0 * (total_frames - count(any_hand)) / total_frames if total_frames else None,
            "at_least_one": 100.0 * count(any_hand) / total_frames if total_frames else None,
        },
        "longest_no_hand_streak": _longest_true_streak(
            {index: not any_hand[index] for index in frame_indices}, frame_indices
        ),
        "longest_left_missing_streak": _longest_true_streak(
            {index: not left_inclusive[index] for index in frame_indices}, frame_indices
        ),
        "longest_right_missing_streak": _longest_true_streak(
            {index: not right_inclusive[index] for index in frame_indices}, frame_indices
        ),
        "duplicate_left_events": int(duplicate_left),
        "duplicate_right_events": int(duplicate_right),
        "frames_with_more_than_2_hands": int(extra_frames),
        "extra_hand_events": int(extra_frames),
        "duplicate_policy": {
            "representative": "highest available native confidence; ties/missing confidence use lowest detector/source index",
            "confidence_cross_model_comparison": "never performed; MediaPipe score is handedness confidence and WiLoR score is detector confidence",
        },
        "suspected_swap_events": _common_swap_events(representatives, frame_indices),
        "swap_heuristic": {
            "position": "normalized image-space hand bounding-box centre proxy",
            "event": "strictly lower swapped left/right frame-to-frame displacement than reported assignment",
            "ground_truth": False,
        },
        "bone_length_cv": _bone_length_cv(representatives, frame_indices),
        "scale_normalized_temporal_metric": _normalized_second_difference(representatives, frame_indices),
        "timing": dict(timing) if timing is not None else None,
    }
    return result


def frame_weighted_fps(total_frames: int, total_seconds: float) -> float | None:
    """Standard aggregate FPS: total decoded frames divided by total time."""

    if total_seconds <= 0:
        return None
    return float(total_frames / total_seconds)


def aggregate_metrics(video_metrics: Sequence[Mapping[str, Any]], *, timing: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Frame-weight aggregate of harmonized coverage plus neutral summaries."""

    if not video_metrics:
        raise ValueError("Cannot aggregate an empty metric sequence")
    total_frames = sum(int(metric["total_frames"]) for metric in video_metrics)
    count_keys = (
        "frames_left_inclusive",
        "frames_right_inclusive",
        "frames_both",
        "frames_left_only",
        "frames_right_only",
        "frames_no_hand",
        "frames_at_least_one",
        "duplicate_left_events",
        "duplicate_right_events",
        "frames_with_more_than_2_hands",
        "extra_hand_events",
        "suspected_swap_events",
    )
    counts = {key: sum(int(metric[key]) for metric in video_metrics) for key in count_keys}
    rate_keys = (
        "frames_left_inclusive",
        "frames_right_inclusive",
        "frames_both",
        "frames_left_only",
        "frames_right_only",
        "frames_no_hand",
        "frames_at_least_one",
    )
    rates = {key: 100.0 * counts[key] / total_frames if total_frames else None for key in rate_keys}
    bone_means = [
        metric["bone_length_cv"]["distribution_across_edges"]["mean"]
        for metric in video_metrics
        if metric["bone_length_cv"]["distribution_across_edges"]["mean"] is not None
    ]
    jitter_means = [
        metric["scale_normalized_temporal_metric"]["distribution"]["mean"]
        for metric in video_metrics
        if metric["scale_normalized_temporal_metric"]["distribution"]["mean"] is not None
    ]
    jitter_p95s = [
        metric["scale_normalized_temporal_metric"]["distribution"]["p95"]
        for metric in video_metrics
        if metric["scale_normalized_temporal_metric"]["distribution"]["p95"] is not None
    ]
    aggregate: dict[str, Any] = {
        "videos": len(video_metrics),
        "total_frames": total_frames,
        **counts,
        "coverage_rates_pct": rates,
        "longest_no_hand_streak_max": max(int(metric["longest_no_hand_streak"]) for metric in video_metrics),
        "longest_left_missing_streak_max": max(int(metric["longest_left_missing_streak"]) for metric in video_metrics),
        "longest_right_missing_streak_max": max(int(metric["longest_right_missing_streak"]) for metric in video_metrics),
        "bone_length_cv_mean_of_video_edge_means": mean(bone_means) if bone_means else None,
        "scale_normalized_temporal_metric_mean_of_video_means": mean(jitter_means) if jitter_means else None,
        "scale_normalized_temporal_metric_mean_of_video_p95s": mean(jitter_p95s) if jitter_p95s else None,
        "timing": dict(timing) if timing is not None else None,
    }
    if timing is not None:
        elapsed = float(timing.get("total_seconds", 0.0))
        aggregate["timing"] = {
            **dict(timing),
            "total_frames": total_frames,
            "frame_weighted_fps": frame_weighted_fps(total_frames, elapsed),
        }
    return aggregate
