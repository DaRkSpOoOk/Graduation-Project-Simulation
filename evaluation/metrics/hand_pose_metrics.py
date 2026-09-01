"""Extractor-agnostic metrics over the repository-common ``HandPoseFrame``
sequence (pose/common/schema.py). Works on MediaPipe, WiLoR, or any future
extractor's output as long as it populates the common schema -- no
WiLoR-specific assumptions here (see .github/instructions/evaluation.instructions.md).

Per that guidance, no acceptance thresholds are hard-coded: functions return
descriptive statistics/observations, not pass/fail verdicts.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from pose.common.schema import HandPoseFrame

# Standard OpenPose 21-point hand skeleton connectivity. WiLoR's MANO output
# is remapped to this joint order (see wilor/models/mano_wrapper.py
# mano_to_openpose); MediaPipe Hand Landmarker uses the same 21-point/
# connectivity convention natively.
HAND_BONES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)


def group_by_frame(frames: list[HandPoseFrame]) -> dict[int, list[HandPoseFrame]]:
    grouped: dict[int, list[HandPoseFrame]] = defaultdict(list)
    for f in frames:
        grouped[f.frame_index].append(f)
    return dict(grouped)


@dataclass(slots=True)
class DetectionStats:
    total_frames: int = 0
    frames_no_hand: int = 0
    frames_left_only: int = 0
    frames_right_only: int = 0
    frames_both: int = 0
    frames_unknown_handedness: int = 0
    missing_frame_pct: float = 0.0
    longest_missing_streak: int = 0


def compute_detection_stats(frames: list[HandPoseFrame]) -> DetectionStats:
    grouped = group_by_frame(frames)
    stats = DetectionStats(total_frames=len(grouped))
    longest = current = 0
    for frame_index in sorted(grouped):
        hands = [f for f in grouped[frame_index] if f.hand_present]
        labels = {h.handedness_label for h in hands if h.handedness_label}
        if not hands:
            stats.frames_no_hand += 1
            current += 1
            longest = max(longest, current)
        else:
            current = 0
            if labels == {"left"}:
                stats.frames_left_only += 1
            elif labels == {"right"}:
                stats.frames_right_only += 1
            elif labels == {"left", "right"}:
                stats.frames_both += 1
            else:
                stats.frames_unknown_handedness += 1
    stats.longest_missing_streak = longest
    stats.missing_frame_pct = (
        100.0 * stats.frames_no_hand / stats.total_frames if stats.total_frames else 0.0
    )
    return stats


@dataclass(slots=True)
class JitterStats:
    n_samples: int = 0
    mean: float | None = None
    median: float | None = None
    p95: float | None = None
    max: float | None = None


def _distribution(values: list[float]) -> JitterStats:
    if not values:
        return JitterStats()
    values = sorted(values)
    n = len(values)
    mean = sum(values) / n

    def _pct(p: float) -> float:
        idx = min(n - 1, max(0, int(round(p * (n - 1)))))
        return values[idx]

    return JitterStats(n_samples=n, mean=mean, median=_pct(0.5), p95=_pct(0.95), max=values[-1])


def _euclidean(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def _reference_position(frame: HandPoseFrame) -> tuple[float, float, float] | None:
    """3D wrist joint when available (full/MANO mode); otherwise the 2D
    detector bbox centroid as a position proxy with z=0 (detector_only
    mode). Units/scale therefore differ by mode -- callers should not mix
    jitter values from different extraction modes."""
    if frame.wrist_position is not None:
        return (frame.wrist_position.x, frame.wrist_position.y, frame.wrist_position.z)
    if frame.landmarks_2d:
        cx = sum(p.x for p in frame.landmarks_2d) / len(frame.landmarks_2d)
        cy = sum(p.y for p in frame.landmarks_2d) / len(frame.landmarks_2d)
        return (cx, cy, 0.0)
    return None


def compute_wrist_jitter(frames: list[HandPoseFrame]) -> dict[str, JitterStats]:
    """Frame-to-frame wrist (or, in detector_only mode, bbox-centroid proxy)
    displacement, bucketed by handedness label (no cross-frame identity
    tracking is assumed or performed here -- that belongs in tracking/, see
    AGENTS.md)."""
    by_hand: dict[str, list[tuple[int, tuple[float, float, float]]]] = defaultdict(list)
    for f in frames:
        if not f.hand_present or not f.handedness_label:
            continue
        pos = _reference_position(f)
        if pos is not None:
            by_hand[f.handedness_label].append((f.frame_index, pos))

    result: dict[str, JitterStats] = {}
    for label, points in by_hand.items():
        points.sort(key=lambda p: p[0])
        displacements = [_euclidean(points[i][1], points[i - 1][1]) for i in range(1, len(points))]
        result[label] = _distribution(displacements)
    return result


def compute_bone_lengths(frame: HandPoseFrame) -> list[float]:
    if len(frame.landmarks_3d) < 21:
        return []
    pts = [(lm.x, lm.y, lm.z) for lm in frame.landmarks_3d[:21]]
    return [_euclidean(pts[a], pts[b]) for a, b in HAND_BONES]


def compute_bone_length_variation(frames: list[HandPoseFrame]) -> dict[str, JitterStats]:
    """Per-handedness variation (frame-to-frame absolute change) in each
    bone's length -- a proxy for reconstruction (in)stability, since a
    rigid hand's bone lengths should not change between frames."""
    by_hand: dict[str, list[list[float]]] = defaultdict(list)
    for f in frames:
        if f.hand_present and f.handedness_label:
            lengths = compute_bone_lengths(f)
            if lengths:
                by_hand[f.handedness_label].append(lengths)

    result: dict[str, JitterStats] = {}
    for label, sequences in by_hand.items():
        deltas: list[float] = []
        for i in range(1, len(sequences)):
            deltas.extend(abs(a - b) for a, b in zip(sequences[i], sequences[i - 1]))
        result[label] = _distribution(deltas)
    return result


@dataclass(slots=True)
class HandCountChange:
    frame_index: int
    previous_count: int
    current_count: int


def compute_hand_count_changes(frames: list[HandPoseFrame]) -> list[HandCountChange]:
    grouped = group_by_frame(frames)
    changes: list[HandCountChange] = []
    prev_count: int | None = None
    for frame_index in sorted(grouped):
        count = sum(1 for f in grouped[frame_index] if f.hand_present)
        if prev_count is not None and count != prev_count:
            changes.append(HandCountChange(frame_index, prev_count, count))
        prev_count = count
    return changes


@dataclass(slots=True)
class HandednessSwapCandidate:
    frame_index: int
    reason: str
    swapped_cost: float
    unswapped_cost: float


def _wrist_positions_by_label(hands: list[HandPoseFrame]) -> dict[str, tuple[float, float, float]]:
    out: dict[str, tuple[float, float, float]] = {}
    for h in hands:
        if not h.hand_present or not h.handedness_label:
            continue
        pos = _reference_position(h)
        if pos is not None:
            out[h.handedness_label] = pos
    return out


def compute_potential_handedness_swaps(frames: list[HandPoseFrame]) -> list[HandednessSwapCandidate]:
    """Flags consecutive both-hands frames where swapping the left/right
    labels would have produced a smaller total wrist displacement than the
    reported (unswapped) assignment -- a common WiLoR/MANO-family failure
    mode when the detector's per-crop handedness classification disagrees
    with spatial continuity. This is an observation, not a correction: raw
    output is never modified (Task 4/5)."""
    grouped = group_by_frame(frames)
    frame_indices = sorted(grouped)
    candidates: list[HandednessSwapCandidate] = []

    prev_positions: dict[str, tuple[float, float, float]] | None = None
    for frame_index in frame_indices:
        positions = _wrist_positions_by_label(grouped[frame_index])
        if prev_positions and {"left", "right"} <= positions.keys() and {"left", "right"} <= prev_positions.keys():
            unswapped = _euclidean(positions["left"], prev_positions["left"]) + _euclidean(
                positions["right"], prev_positions["right"]
            )
            swapped = _euclidean(positions["left"], prev_positions["right"]) + _euclidean(
                positions["right"], prev_positions["left"]
            )
            if swapped < unswapped:
                candidates.append(
                    HandednessSwapCandidate(
                        frame_index=frame_index,
                        reason="swapped_assignment_has_lower_total_wrist_displacement",
                        swapped_cost=swapped,
                        unswapped_cost=unswapped,
                    )
                )
        if positions:
            prev_positions = positions
    return candidates


@dataclass(slots=True)
class VideoEvaluation:
    sample_id: str
    detection: DetectionStats
    wrist_jitter: dict[str, JitterStats]
    bone_length_variation: dict[str, JitterStats]
    hand_count_changes: list[HandCountChange]
    handedness_swap_candidates: list[HandednessSwapCandidate]
    frame_error_count: int
    extra: dict[str, Any] = field(default_factory=dict)


def evaluate_video(sample_id: str, frames: list[HandPoseFrame]) -> VideoEvaluation:
    frame_errors = sum(1 for f in frames if "extraction_failed" in f.quality_flags)
    return VideoEvaluation(
        sample_id=sample_id,
        detection=compute_detection_stats(frames),
        wrist_jitter=compute_wrist_jitter(frames),
        bone_length_variation=compute_bone_length_variation(frames),
        hand_count_changes=compute_hand_count_changes(frames),
        handedness_swap_candidates=compute_potential_handedness_swaps(frames),
        frame_error_count=frame_errors,
    )
