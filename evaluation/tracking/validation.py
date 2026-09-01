"""Validate a derived LEFT/RIGHT hand-tracking run against the independent
TASK-004B human benchmark.

Design rules, all deliberate:

* The tracker and the annotations are both treated as frozen. Nothing here
  writes to either, and no threshold defined here is ever fed back into the
  tracker.
* Alignment is exclusively on ``(sample_id, frame_index)``. No timestamp
  matching, no nearest-frame fallback.
* The detector's own handedness label is never used as identity evidence.
  Physical identity is decided only from the human annotation's reference
  points.
* Human ``AMBIGUOUS`` identity frames are excluded from strict identity
  scoring and reported separately; they are never converted into tracker
  failures.
* A hand annotated ``FULLY_OCCLUDED`` or ``OUT_OF_FRAME`` is not required to
  have a pose, so it is never counted as a recall miss.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from evaluation.annotations.task004b import AnnotationRow
from tracking.wilor.npz_io import load_tracked_sequence
from tracking.wilor.schema import CODE_TO_STATE, TRACK_NAMES, TrackState

# --------------------------------------------------------------------------
# Validator constants. These describe how the *evaluation* reads approximate
# human reference points; none of them is a tracker parameter.
# --------------------------------------------------------------------------

#: Tracker states that expose an actual reconstructed pose.
POSED_STATES = frozenset({TrackState.OBSERVED, TrackState.AMBIGUOUS})

#: Annotated states for which a tracker is expected to produce a pose.
EXPECTED_POSE_STATES = frozenset({"VISIBLE", "PARTIALLY_OCCLUDED"})

#: Annotated states for which a tracker must NOT produce a pose.
FORBIDDEN_POSE_STATES = frozenset({"FULLY_OCCLUDED", "OUT_OF_FRAME"})

#: Human confidence levels admitted into strict identity scoring.
STRICT_CONFIDENCE = frozenset({"HIGH", "MEDIUM"})

#: Mean per-hand pixel margin by which the identity assignment must beat the
#: swapped assignment before the coordinate test is treated as decisive. The
#: TASK-004B protocol states its points are approximate wrist/hand-centre
#: references and explicitly "not pixel-perfect keypoint ground truth", so a
#: sub-threshold difference is reported as indeterminate rather than scored.
DECISION_MARGIN_PX = 25.0

#: A coordinate-derived identity flip must persist for at least this many
#: consecutive evaluable frames before it is called a confirmed switch, so a
#: single noisy reference point cannot manufacture one.
SWITCH_PERSISTENCE_FRAMES = 2

DEFAULT_IMAGE_WH = (1920.0, 1080.0)
DEFAULT_FOCAL_LENGTH = 37500.0


class ValidationError(RuntimeError):
    """Raised on any alignment or integrity violation. Never silently fixed."""


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrackedHandView:
    """One canonical track's read-only view of a single tracked frame."""

    track: str
    state: TrackState
    raw_detection_index: int | None
    detector_label: str | None
    box_center_xy: tuple[float, float] | None
    wrist_xy: tuple[float, float] | None

    @property
    def has_pose(self) -> bool:
        return self.state in POSED_STATES and self.box_center_xy is not None


@dataclass(frozen=True, slots=True)
class TrackedFrameView:
    sample_id: str
    frame_index: int
    left: TrackedHandView
    right: TrackedHandView
    number_of_raw_detections: int
    extra_detection_count: int
    tracking_flags: tuple[str, ...]

    def hand(self, track: str) -> TrackedHandView:
        return self.left if track == "left" else self.right


def _project_wrist(
    joints: np.ndarray,
    translation: np.ndarray,
    focal_length: float,
    image_wh: tuple[float, float],
) -> tuple[float, float] | None:
    """Project joint 0 (wrist) to pixels with the frozen pinhole model."""

    if not (np.isfinite(joints).all() and np.isfinite(translation).all()):
        return None
    point = joints[0].astype(np.float64) + translation.astype(np.float64)
    if abs(point[2]) < 1e-9:
        return None
    return (
        focal_length * point[0] / point[2] + image_wh[0] / 2.0,
        focal_length * point[1] / point[2] + image_wh[1] / 2.0,
    )


def load_tracked_frames(
    tracked_run: str | Path,
    sample_id: str,
    *,
    image_wh: tuple[float, float] = DEFAULT_IMAGE_WH,
    focal_length: float = DEFAULT_FOCAL_LENGTH,
) -> dict[int, TrackedFrameView]:
    """Read one tracked sample into frame-indexed read-only views."""

    directory = Path(tracked_run) / sample_id
    if not (directory / "wilor_tracked.npz").is_file():
        raise ValidationError(f"Tracked output missing for {sample_id}: {directory}")
    arrays, metadata = load_tracked_sequence(directory)

    track_order = tuple(metadata.get("track_order", TRACK_NAMES))
    if track_order != TRACK_NAMES:
        raise ValidationError(f"Unexpected track order for {sample_id}: {track_order}")

    frames: dict[int, TrackedFrameView] = {}
    label_lookup = {0: "left", 1: "right"}
    for row in range(len(arrays["frame_index"])):
        frame_index = int(arrays["frame_index"][row])
        if frame_index in frames:
            raise ValidationError(
                f"Duplicated tracker frame {sample_id}:{frame_index}"
            )
        hands: list[TrackedHandView] = []
        for column, track in enumerate(track_order):
            state = CODE_TO_STATE[int(arrays["state_code"][row, column])]
            raw_index = int(arrays["raw_detection_index"][row, column])
            centre = arrays["box_center_xy"][row, column]
            has_centre = bool(np.isfinite(centre).all())
            hands.append(
                TrackedHandView(
                    track=track,
                    state=state,
                    raw_detection_index=None if raw_index < 0 else raw_index,
                    detector_label=label_lookup.get(
                        int(arrays["detector_label_code"][row, column])
                    ),
                    box_center_xy=(float(centre[0]), float(centre[1])) if has_centre else None,
                    wrist_xy=_project_wrist(
                        arrays["landmarks_3d"][row, column],
                        arrays["camera_translation"][row, column],
                        focal_length,
                        image_wh,
                    ),
                )
            )
        import json as _json

        frames[frame_index] = TrackedFrameView(
            sample_id=sample_id,
            frame_index=frame_index,
            left=hands[0],
            right=hands[1],
            number_of_raw_detections=int(arrays["number_of_raw_detections"][row]),
            extra_detection_count=int(arrays["extra_detection_count"][row]),
            tracking_flags=tuple(_json.loads(str(arrays["tracking_flags_json"][row]))),
        )
    return frames


# --------------------------------------------------------------------------
# Alignment
# --------------------------------------------------------------------------


def align(
    annotations: Sequence[AnnotationRow],
    tracked_by_sample: Mapping[str, Mapping[int, TrackedFrameView]],
) -> list[tuple[AnnotationRow, TrackedFrameView]]:
    """Join annotation rows to tracker frames on (sample_id, frame_index).

    Hard-fails on duplicate annotation keys, a missing annotated clip, or a
    tracker frame missing for an annotated source frame.
    """

    seen: set[tuple[str, int]] = set()
    pairs: list[tuple[AnnotationRow, TrackedFrameView]] = []
    for row in annotations:
        key = (row.sample_id, row.frame_index)
        if key in seen:
            raise ValidationError(f"Duplicate annotation key: {key}")
        seen.add(key)
        frames = tracked_by_sample.get(row.sample_id)
        if frames is None:
            raise ValidationError(f"Annotated clip missing from tracker run: {row.sample_id}")
        frame = frames.get(row.frame_index)
        if frame is None:
            raise ValidationError(
                f"Tracker frame missing for annotated frame {row.sample_id}:{row.frame_index}"
            )
        pairs.append((row, frame))
    return pairs


# --------------------------------------------------------------------------
# Metric helpers
# --------------------------------------------------------------------------


def _rate(numerator: int, denominator: int) -> float | None:
    return 100.0 * numerator / denominator if denominator else None


def reference_points(
    row: AnnotationRow, image_wh: tuple[float, float] = DEFAULT_IMAGE_WH
) -> dict[str, tuple[float, float]]:
    """Annotated normalized points converted to pixels.

    TASK-004B defines x = pixel_x / (width - 1), y = pixel_y / (height - 1).
    """

    width, height = image_wh
    points: dict[str, tuple[float, float]] = {}
    if row.left_x is not None and row.left_y is not None:
        points["left"] = (row.left_x * (width - 1.0), row.left_y * (height - 1.0))
    if row.right_x is not None and row.right_y is not None:
        points["right"] = (row.right_x * (width - 1.0), row.right_y * (height - 1.0))
    return points


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass(frozen=True, slots=True)
class IdentityDecision:
    """Outcome of the coordinate-based identity test for a single frame."""

    evaluable: bool
    correct: bool | None
    margin_px: float | None
    identity_distance_px: float | None
    swapped_distance_px: float | None
    reason: str

    @property
    def decisive(self) -> bool:
        """True when the coordinate test separates the two assignments by more
        than the annotation's own approximation error.

        Uses the *absolute* margin: a confidently wrong assignment is just as
        decisive as a confidently right one, and must not be silently dropped
        (that would hide identity switches)."""

        return (
            self.evaluable
            and self.margin_px is not None
            and abs(self.margin_px) >= DECISION_MARGIN_PX
        )


def identity_decision(
    row: AnnotationRow,
    frame: TrackedFrameView,
    *,
    position: str = "box",
    image_wh: tuple[float, float] = DEFAULT_IMAGE_WH,
) -> IdentityDecision:
    """Decide whether tracker LEFT/RIGHT match the annotated physical hands.

    Relative test: the assignment (annotated LEFT -> tracker LEFT, annotated
    RIGHT -> tracker RIGHT) is compared against the swapped assignment. Only
    the *relative* ordering is used, so a systematic offset between an
    approximate human wrist point and the tracker's hand centre cannot by
    itself produce a wrong verdict.
    """

    if row.has_identity_ambiguity or "AMBIGUOUS" in (row.left_visibility, row.right_visibility):
        return IdentityDecision(False, None, None, None, None, "human_identity_ambiguous")
    if row.annotator_confidence not in STRICT_CONFIDENCE:
        return IdentityDecision(False, None, None, None, None, "low_annotator_confidence")

    points = reference_points(row, image_wh)
    if len(points) < 2:
        return IdentityDecision(False, None, None, None, None, "insufficient_reference_points")

    selector = (lambda hand: hand.box_center_xy) if position == "box" else (lambda hand: hand.wrist_xy)
    tracker_points: dict[str, tuple[float, float]] = {}
    for track in TRACK_NAMES:
        hand = frame.hand(track)
        value = selector(hand)
        if hand.state in POSED_STATES and value is not None:
            tracker_points[track] = value
    if len(tracker_points) < 2:
        return IdentityDecision(False, None, None, None, None, "tracker_pose_unavailable")

    identity = _distance(points["left"], tracker_points["left"]) + _distance(
        points["right"], tracker_points["right"]
    )
    swapped = _distance(points["left"], tracker_points["right"]) + _distance(
        points["right"], tracker_points["left"]
    )
    margin = (swapped - identity) / 2.0
    return IdentityDecision(
        evaluable=True,
        correct=identity <= swapped,
        margin_px=margin,
        identity_distance_px=identity / 2.0,
        swapped_distance_px=swapped / 2.0,
        reason="evaluated",
    )


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def visibility_recall(pairs: Iterable[tuple[AnnotationRow, TrackedFrameView]]) -> dict[str, Any]:
    """METRIC 1. Does a pose exist when the human says the hand is there?"""

    counters: dict[str, dict[str, int]] = {
        track: {"expected": 0, "posed": 0, "visible_expected": 0, "visible_posed": 0,
                "partial_expected": 0, "partial_posed": 0}
        for track in TRACK_NAMES
    }
    misses: list[dict[str, Any]] = []
    for row, frame in pairs:
        for track in TRACK_NAMES:
            state = row.left_visibility if track == "left" else row.right_visibility
            if state not in EXPECTED_POSE_STATES:
                continue
            hand = frame.hand(track)
            posed = hand.has_pose
            bucket = counters[track]
            bucket["expected"] += 1
            bucket["posed"] += int(posed)
            key = "visible" if state == "VISIBLE" else "partial"
            bucket[f"{key}_expected"] += 1
            bucket[f"{key}_posed"] += int(posed)
            if not posed:
                misses.append(
                    {
                        "sample_id": row.sample_id,
                        "frame_index": row.frame_index,
                        "track": track,
                        "annotated_state": state,
                        "tracker_state": hand.state.value,
                        "annotator_confidence": row.annotator_confidence,
                        "scene_flags": list(row.scene_flags),
                    }
                )
    overall_expected = sum(c["expected"] for c in counters.values())
    overall_posed = sum(c["posed"] for c in counters.values())
    visible_expected = sum(c["visible_expected"] for c in counters.values())
    visible_posed = sum(c["visible_posed"] for c in counters.values())
    partial_expected = sum(c["partial_expected"] for c in counters.values())
    partial_posed = sum(c["partial_posed"] for c in counters.values())
    return {
        "left": {**counters["left"], "recall_pct": _rate(counters["left"]["posed"], counters["left"]["expected"])},
        "right": {**counters["right"], "recall_pct": _rate(counters["right"]["posed"], counters["right"]["expected"])},
        "overall": {
            "expected": overall_expected,
            "posed": overall_posed,
            "recall_pct": _rate(overall_posed, overall_expected),
        },
        "fully_visible": {
            "expected": visible_expected,
            "posed": visible_posed,
            "recall_pct": _rate(visible_posed, visible_expected),
        },
        "partially_occluded": {
            "expected": partial_expected,
            "posed": partial_posed,
            "recall_pct": _rate(partial_posed, partial_expected),
        },
        "misses": misses,
    }


def false_presence(pairs: Iterable[tuple[AnnotationRow, TrackedFrameView]]) -> dict[str, Any]:
    """METRIC 2. Does the tracker fabricate a pose for a hidden hand?"""

    considered = 0
    fabricated: list[dict[str, Any]] = []
    by_state: dict[str, int] = defaultdict(int)
    for row, frame in pairs:
        for track in TRACK_NAMES:
            state = row.left_visibility if track == "left" else row.right_visibility
            if state not in FORBIDDEN_POSE_STATES:
                continue
            considered += 1
            hand = frame.hand(track)
            by_state[hand.state.value] += 1
            if hand.has_pose:
                fabricated.append(
                    {
                        "sample_id": row.sample_id,
                        "frame_index": row.frame_index,
                        "track": track,
                        "annotated_state": state,
                        "tracker_state": hand.state.value,
                    }
                )
    return {
        "considered_hand_instances": considered,
        "false_presence_count": len(fabricated),
        "false_presence_rate_pct": _rate(len(fabricated), considered),
        "tracker_state_distribution": dict(by_state),
        "events": fabricated,
    }


def identity_accuracy(
    pairs: Sequence[tuple[AnnotationRow, TrackedFrameView]], *, position: str = "box"
) -> dict[str, Any]:
    """METRIC 3. Do the canonical tracks carry the right physical hands?"""

    evaluable = correct = incorrect = 0
    decisive = decisive_correct = 0
    indeterminate = 0
    excluded: dict[str, int] = defaultdict(int)
    incorrect_frames: list[dict[str, Any]] = []
    indeterminate_frames: list[dict[str, Any]] = []

    for row, frame in pairs:
        decision = identity_decision(row, frame, position=position)
        if not decision.evaluable:
            excluded[decision.reason] += 1
            continue
        evaluable += 1
        if decision.correct:
            correct += 1
        else:
            incorrect += 1
            incorrect_frames.append(
                {
                    "sample_id": row.sample_id,
                    "frame_index": row.frame_index,
                    "margin_px": decision.margin_px,
                    "identity_distance_px": decision.identity_distance_px,
                    "swapped_distance_px": decision.swapped_distance_px,
                    "annotator_confidence": row.annotator_confidence,
                    "scene_flags": list(row.scene_flags),
                }
            )
        if decision.decisive:
            decisive += 1
            decisive_correct += int(bool(decision.correct))
        else:
            indeterminate += 1
            indeterminate_frames.append(
                {
                    "sample_id": row.sample_id,
                    "frame_index": row.frame_index,
                    "margin_px": decision.margin_px,
                    "coordinate_test_correct": decision.correct,
                }
            )
    return {
        "position_source": position,
        "decision_margin_px": DECISION_MARGIN_PX,
        "evaluable_frames": evaluable,
        "correct_frames": correct,
        "incorrect_frames": incorrect,
        "accuracy_pct": _rate(correct, evaluable),
        "decisive_frames": decisive,
        "decisive_correct_frames": decisive_correct,
        "decisive_accuracy_pct": _rate(decisive_correct, decisive),
        "indeterminate_frames": indeterminate,
        "excluded_reasons": dict(excluded),
        "incorrect_detail": incorrect_frames,
        "indeterminate_detail": indeterminate_frames[:20],
    }


def identity_switches(
    pairs: Sequence[tuple[AnnotationRow, TrackedFrameView]], *, position: str = "box"
) -> dict[str, Any]:
    """METRIC 4. Confirmed persistent LEFT/RIGHT swaps.

    A switch is confirmed only when the coordinate-derived assignment flips
    and the flipped state persists for at least SWITCH_PERSISTENCE_FRAMES
    consecutive decisive frames, so a single noisy reference point cannot
    manufacture one.
    """

    by_sample: dict[str, list[tuple[AnnotationRow, TrackedFrameView]]] = defaultdict(list)
    for row, frame in pairs:
        by_sample[row.sample_id].append((row, frame))

    confirmed: list[dict[str, Any]] = []
    suspected: list[dict[str, Any]] = []
    for sample_id, entries in by_sample.items():
        entries.sort(key=lambda item: item[0].frame_index)
        decisions = [
            (row, identity_decision(row, frame, position=position)) for row, frame in entries
        ]
        run_state: bool | None = None
        run_length = 0
        run_start: int | None = None
        for row, decision in decisions:
            if not decision.decisive:
                continue
            correct = bool(decision.correct)
            if correct == run_state:
                run_length += 1
            else:
                if run_state is False and run_length >= SWITCH_PERSISTENCE_FRAMES:
                    confirmed.append(
                        {"sample_id": sample_id, "start_frame": run_start, "length": run_length}
                    )
                elif run_state is False:
                    suspected.append(
                        {"sample_id": sample_id, "start_frame": run_start, "length": run_length}
                    )
                run_state, run_length, run_start = correct, 1, row.frame_index
        if run_state is False:
            target = confirmed if run_length >= SWITCH_PERSISTENCE_FRAMES else suspected
            target.append({"sample_id": sample_id, "start_frame": run_start, "length": run_length})
    return {
        "persistence_frames_required": SWITCH_PERSISTENCE_FRAMES,
        "confirmed_switches": len(confirmed),
        "suspected_unresolved_switches": len(suspected),
        "confirmed_detail": confirmed,
        "suspected_detail": suspected,
    }


def reacquisition(
    pairs: Sequence[tuple[AnnotationRow, TrackedFrameView]], *, position: str = "box"
) -> dict[str, Any]:
    """METRIC 5. After a human-annotated disappearance, is the returning
    physical hand put back on the correct canonical track?"""

    by_sample: dict[str, list[tuple[AnnotationRow, TrackedFrameView]]] = defaultdict(list)
    for row, frame in pairs:
        by_sample[row.sample_id].append((row, frame))

    events: list[dict[str, Any]] = []
    for sample_id, entries in by_sample.items():
        entries.sort(key=lambda item: item[0].frame_index)
        for track in TRACK_NAMES:
            absent_run = 0
            for row, frame in entries:
                state = row.left_visibility if track == "left" else row.right_visibility
                if state in FORBIDDEN_POSE_STATES:
                    absent_run += 1
                    continue
                if absent_run and state in EXPECTED_POSE_STATES:
                    hand = frame.hand(track)
                    decision = identity_decision(row, frame, position=position)
                    events.append(
                        {
                            "sample_id": sample_id,
                            "track": track,
                            "return_frame": row.frame_index,
                            "absent_frames": absent_run,
                            "tracker_state": hand.state.value,
                            "pose_restored": hand.has_pose,
                            "identity_evaluable": decision.evaluable,
                            "identity_correct": decision.correct,
                            "margin_px": decision.margin_px,
                            "annotator_confidence": row.annotator_confidence,
                        }
                    )
                absent_run = 0
    scored = [e for e in events if e["pose_restored"] and e["identity_correct"] is not False]
    incorrect = [e for e in events if e["identity_correct"] is False or not e["pose_restored"]]
    return {
        "events": len(events),
        "correct": len(scored),
        "incorrect": len(incorrect),
        "accuracy_pct": _rate(len(scored), len(events)),
        "detail": events,
    }


def occlusion_state_validity(
    pairs: Iterable[tuple[AnnotationRow, TrackedFrameView]]
) -> dict[str, Any]:
    """METRIC 6. What does the tracker's LIKELY_OCCLUDED heuristic coincide
    with in the human annotation?"""

    matrix: dict[str, int] = defaultdict(int)
    total = 0
    for row, frame in pairs:
        for track in TRACK_NAMES:
            hand = frame.hand(track)
            if hand.state is not TrackState.LIKELY_OCCLUDED:
                continue
            total += 1
            state = row.left_visibility if track == "left" else row.right_visibility
            matrix[state] += 1
    return {
        "tracker_likely_occluded_hand_instances": total,
        "human_state_counts": dict(matrix),
        "note": (
            "Coincidence counts only. The tracker flag is a proximity heuristic; "
            "this is not a claim of true occlusion classification."
        ),
    }


def ambiguity_calibration(
    pairs: Sequence[tuple[AnnotationRow, TrackedFrameView]], *, position: str = "box"
) -> dict[str, Any]:
    """METRIC 7. Compare tracker AMBIGUOUS frames with human ambiguity."""

    tracker_ambiguous: list[tuple[str, int]] = []
    human_ambiguous: list[tuple[str, int]] = []
    tracker_ambiguous_on_crossing = 0
    human_ambiguous_tracker_confident: list[dict[str, Any]] = []

    for row, frame in pairs:
        key = (row.sample_id, row.frame_index)
        is_tracker_ambiguous = any(
            frame.hand(track).state is TrackState.AMBIGUOUS for track in TRACK_NAMES
        )
        is_human_ambiguous = row.has_identity_ambiguity or "AMBIGUOUS" in (
            row.left_visibility,
            row.right_visibility,
        )
        if is_tracker_ambiguous:
            tracker_ambiguous.append(key)
            if row.is_crossing:
                tracker_ambiguous_on_crossing += 1
        if is_human_ambiguous:
            human_ambiguous.append(key)
            if not is_tracker_ambiguous:
                decision = identity_decision(row, frame, position=position)
                human_ambiguous_tracker_confident.append(
                    {
                        "sample_id": row.sample_id,
                        "frame_index": row.frame_index,
                        "tracker_states": [frame.hand(t).state.value for t in TRACK_NAMES],
                        "coordinate_test": decision.reason,
                        "coordinate_test_correct": decision.correct,
                    }
                )
    overlap = set(tracker_ambiguous) & set(human_ambiguous)
    crossing_frames = sum(1 for row, _ in pairs if row.is_crossing)
    return {
        "tracker_ambiguous_frames": len(tracker_ambiguous),
        "human_ambiguous_frames": len(human_ambiguous),
        "overlap_frames": len(overlap),
        "crossing_frames": crossing_frames,
        "tracker_ambiguous_on_crossing_frames": tracker_ambiguous_on_crossing,
        "human_ambiguous_but_tracker_confident": len(human_ambiguous_tracker_confident),
        "human_ambiguous_but_tracker_confident_detail": human_ambiguous_tracker_confident,
        "tracker_ambiguous_detail": [
            {"sample_id": s, "frame_index": f} for s, f in tracker_ambiguous
        ],
    }


def stratified_identity(
    pairs: Sequence[tuple[AnnotationRow, TrackedFrameView]], *, position: str = "box"
) -> dict[str, Any]:
    """Identity + recall broken down by difficulty stratum."""

    def stratum_of(row: AnnotationRow, clip_roles: Mapping[str, str]) -> list[str]:
        names: list[str] = []
        if clip_roles.get(row.sample_id) == "control":
            names.append("control")
        if row.is_crossing:
            names.append("crossing")
        if row.is_motion_blurred:
            names.append("motion_blur")
        if "PARTIALLY_OCCLUDED" in (row.left_visibility, row.right_visibility):
            names.append("partially_occluded")
        if "FULLY_OCCLUDED" in (row.left_visibility, row.right_visibility):
            names.append("fully_occluded")
        if row.has_identity_ambiguity or "AMBIGUOUS" in (row.left_visibility, row.right_visibility):
            names.append("human_ambiguous")
        if not names:
            names.append("other")
        return names

    from evaluation.annotations.task004b import CLIP_SPECS

    clip_roles = {spec.sample_id: spec.role for spec in CLIP_SPECS}
    buckets: dict[str, dict[str, int]] = defaultdict(
        lambda: {"frames": 0, "identity_evaluable": 0, "identity_correct": 0,
                 "pose_expected": 0, "pose_present": 0}
    )
    for row, frame in pairs:
        decision = identity_decision(row, frame, position=position)
        for name in stratum_of(row, clip_roles):
            bucket = buckets[name]
            bucket["frames"] += 1
            if decision.evaluable:
                bucket["identity_evaluable"] += 1
                bucket["identity_correct"] += int(bool(decision.correct))
            for track in TRACK_NAMES:
                state = row.left_visibility if track == "left" else row.right_visibility
                if state in EXPECTED_POSE_STATES:
                    bucket["pose_expected"] += 1
                    bucket["pose_present"] += int(frame.hand(track).has_pose)
    return {
        name: {
            **values,
            "identity_accuracy_pct": _rate(values["identity_correct"], values["identity_evaluable"]),
            "recall_pct": _rate(values["pose_present"], values["pose_expected"]),
        }
        for name, values in sorted(buckets.items())
    }


def per_clip_summary(
    pairs: Sequence[tuple[AnnotationRow, TrackedFrameView]], *, position: str = "box"
) -> dict[str, Any]:
    """Per-clip roll-up used by the challenge-clip section of the report."""

    by_sample: dict[str, list[tuple[AnnotationRow, TrackedFrameView]]] = defaultdict(list)
    for row, frame in pairs:
        by_sample[row.sample_id].append((row, frame))

    result: dict[str, Any] = {}
    for sample_id, entries in sorted(by_sample.items()):
        recall = visibility_recall(entries)
        identity = identity_accuracy(entries, position=position)
        switches = identity_switches(entries, position=position)
        reacq = reacquisition(entries, position=position)
        ambiguity = ambiguity_calibration(entries, position=position)
        presence = false_presence(entries)
        result[sample_id] = {
            "frames": len(entries),
            "recall_pct": recall["overall"]["recall_pct"],
            "left_recall_pct": recall["left"]["recall_pct"],
            "right_recall_pct": recall["right"]["recall_pct"],
            "identity_evaluable": identity["evaluable_frames"],
            "identity_correct": identity["correct_frames"],
            "identity_accuracy_pct": identity["accuracy_pct"],
            "confirmed_switches": switches["confirmed_switches"],
            "suspected_switches": switches["suspected_unresolved_switches"],
            "reacquisition_events": reacq["events"],
            "reacquisition_correct": reacq["correct"],
            "tracker_ambiguous_frames": ambiguity["tracker_ambiguous_frames"],
            "human_ambiguous_frames": ambiguity["human_ambiguous_frames"],
            "false_presence_count": presence["false_presence_count"],
            "frames_with_more_than_two_raw_detections": sum(
                1 for _, frame in entries if frame.number_of_raw_detections > 2
            ),
        }
    return result


def extra_detection_case(
    pairs: Sequence[tuple[AnnotationRow, TrackedFrameView]],
    raw_run: str | Path,
    *,
    sample_id: str = "karsl_test_s02_sign0176_repfirst",
    frame_index: int = 39,
    position: str = "box",
) -> dict[str, Any]:
    """METRIC 8. The known third-detection frame, checked independently.

    Verifies that the human reference contains exactly two physical hands,
    that raw WiLoR really produced three reconstructions, that the tracker
    kept exactly two canonical identities, that identity is correct before,
    during and after the event, and that raw provenance is still recorded.
    """

    lookup = {(row.sample_id, row.frame_index): (row, frame) for row, frame in pairs}
    checks: dict[str, Any] = {}
    failures: list[str] = []

    target = lookup.get((sample_id, frame_index))
    if target is None:
        return {"status": "FAIL", "reason": "target frame not present in aligned data"}
    row, frame = target

    # 1. human reference describes exactly two physical hands
    human_states = (row.left_visibility, row.right_visibility)
    two_physical_hands = all(state not in FORBIDDEN_POSE_STATES for state in human_states)
    checks["human_states"] = list(human_states)
    checks["human_describes_two_hands"] = two_physical_hands
    if not two_physical_hands:
        failures.append("human reference does not describe two present hands")

    # 2. raw WiLoR produced three reconstructions
    raw_path = Path(raw_run) / "raw" / sample_id / "wilor_raw.npz"
    raw_count = None
    if raw_path.is_file():
        with np.load(raw_path, allow_pickle=False) as data:
            indices = np.asarray(data["frame_index"]).astype(int)
            present = np.asarray(data["hand_present"], dtype=bool)
            raw_count = int(((indices == frame_index) & present).sum())
    checks["raw_reconstruction_count"] = raw_count
    if raw_count != 3:
        failures.append(f"expected 3 raw reconstructions, found {raw_count}")

    # 3. tracker kept exactly two canonical identities with poses
    posed = [track for track in TRACK_NAMES if frame.hand(track).has_pose]
    checks["tracker_posed_tracks"] = posed
    checks["tracker_reported_raw_detections"] = frame.number_of_raw_detections
    checks["tracker_flags"] = list(frame.tracking_flags)
    if len(posed) != 2:
        failures.append(f"tracker exposed {len(posed)} posed tracks, expected 2")
    if frame.number_of_raw_detections != 3:
        failures.append("tracker did not record that three raw detections existed")

    # 4. provenance for the surviving identities is recorded and distinct
    provenance = {track: frame.hand(track).raw_detection_index for track in TRACK_NAMES}
    checks["raw_detection_provenance"] = provenance
    if provenance["left"] is None or provenance["right"] is None:
        failures.append("provenance missing for a surviving identity")
    elif provenance["left"] == provenance["right"]:
        failures.append("both identities point at the same raw detection")

    # 5. identity correct immediately before, during and after
    window: dict[str, Any] = {}
    for offset in (-1, 0, 1):
        entry = lookup.get((sample_id, frame_index + offset))
        if entry is None:
            window[str(frame_index + offset)] = "unavailable"
            continue
        decision = identity_decision(entry[0], entry[1], position=position)
        window[str(frame_index + offset)] = {
            "evaluable": decision.evaluable,
            "correct": decision.correct,
            "margin_px": decision.margin_px,
        }
        if decision.evaluable and decision.correct is False:
            failures.append(f"identity incorrect at frame {frame_index + offset}")
    checks["identity_window"] = window

    return {"status": "FAIL" if failures else "PASS", "failures": failures, "checks": checks}


def quality_gate_evaluation(
    pairs: Sequence[tuple[AnnotationRow, TrackedFrameView]],
    tracked_run: str | Path,
) -> dict[str, Any]:
    """METRIC 9. Was the quality gate useful, harmful, or inert?

    Counts what the tracker actually rejected across annotated clips, and
    checks whether a pose was accepted on any frame the human marked
    ``FULLY_OCCLUDED``/``OUT_OF_FRAME`` (which would be a clearly bad pose
    the gate should have caught).
    """

    import json as _json

    useful_flags = 0
    quality_rejections = 0
    duplicate_suppressions = 0
    unassigned_extras = 0
    false_rejects: list[dict[str, Any]] = []
    rejection_reasons: dict[str, int] = defaultdict(int)

    annotated_samples = sorted({row.sample_id for row, _ in pairs})
    annotated_frames = {(row.sample_id, row.frame_index) for row, _ in pairs}
    expectation = {
        (row.sample_id, row.frame_index, track): (
            row.left_visibility if track == "left" else row.right_visibility
        )
        for row, _ in pairs
        for track in TRACK_NAMES
    }

    for sample_id in annotated_samples:
        arrays, _ = load_tracked_sequence(Path(tracked_run) / sample_id)
        for row_index in range(len(arrays["frame_index"])):
            frame_index = int(arrays["frame_index"][row_index])
            if (sample_id, frame_index) not in annotated_frames:
                continue
            payload = _json.loads(str(arrays["rejected_detections_json"][row_index]))
            for reason in payload.get("reasons", {}).values():
                rejection_reasons[reason.split(":")[0].split("=")[0]] += 1
                if reason.startswith("quality:"):
                    quality_rejections += 1
                elif reason.startswith("duplicate_same_label"):
                    duplicate_suppressions += 1
                elif reason.startswith("unassigned_extra"):
                    unassigned_extras += 1
            for column, track in enumerate(TRACK_NAMES):
                flags = _json.loads(str(arrays["quality_flags_json"][row_index, column]))
                quality_flags = [f for f in flags if str(f).startswith("LOW_QUALITY")]
                if quality_flags:
                    useful_flags += 1
                    expected = expectation.get((sample_id, frame_index, track))
                    if expected in EXPECTED_POSE_STATES:
                        false_rejects.append(
                            {
                                "sample_id": sample_id,
                                "frame_index": frame_index,
                                "track": track,
                                "flags": quality_flags,
                                "annotated_state": expected,
                            }
                        )

    presence = false_presence(pairs)
    return {
        "annotated_clips": len(annotated_samples),
        "quality_rejected_detections": quality_rejections,
        "duplicate_suppressed_detections": duplicate_suppressions,
        "unassigned_extra_detections": unassigned_extras,
        "low_quality_flags_on_kept_hands": useful_flags,
        "flagged_but_human_says_visible": len(false_rejects),
        "flagged_but_human_says_visible_detail": false_rejects,
        "missed_clearly_bad_pose_events": presence["false_presence_count"],
        "missed_clearly_bad_pose_detail": presence["events"],
        "rejection_reason_counts": dict(rejection_reasons),
    }


def verify_raw_integrity(tracked_run: str | Path) -> dict[str, Any]:
    """Acceptance criterion E: every source raw NPZ is byte-unchanged."""

    import hashlib
    import json as _json

    checked = matched = 0
    mismatched: list[str] = []
    missing: list[str] = []
    for meta_path in sorted(Path(tracked_run).glob("*/wilor_tracked_meta.json")):
        metadata = _json.loads(meta_path.read_text())
        source = metadata.get("source", {})
        raw_path = Path(source.get("raw_npz", ""))
        recorded = source.get("raw_npz_sha256")
        checked += 1
        if not raw_path.is_file():
            missing.append(str(raw_path))
            continue
        digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        if digest == recorded:
            matched += 1
        else:
            mismatched.append(meta_path.parent.name)
    return {
        "tracked_samples_checked": checked,
        "raw_sha256_matched": matched,
        "mismatched": mismatched,
        "missing": missing,
        "all_unchanged": bool(checked and matched == checked and not mismatched and not missing),
    }


def tracker_claimed_reacquisitions(
    pairs: Sequence[tuple[AnnotationRow, TrackedFrameView]], tracked_run: str | Path
) -> dict[str, Any]:
    """Cross-check the tracker's own reacquisition events against the human
    reference.

    METRIC 5's denominator counts *annotated* disappear/reappear events. When
    the benchmark contains none, that metric is vacuous, which would hide a
    tracker that claims a reacquisition the reference does not support. This
    function reports the opposite direction: every reacquisition the tracker
    recorded, and what the human said about that hand on that frame.
    """

    import json as _json

    lookup = {(row.sample_id, row.frame_index): row for row, _ in pairs}
    annotated_samples = sorted({row.sample_id for row, _ in pairs})
    claimed: list[dict[str, Any]] = []
    for sample_id in annotated_samples:
        meta_path = Path(tracked_run) / sample_id / "wilor_tracked_meta.json"
        if not meta_path.is_file():
            continue
        metadata = _json.loads(meta_path.read_text())
        for event in metadata.get("events", []):
            if event.get("event") != "reassociation":
                continue
            row = lookup.get((sample_id, event.get("frame_index")))
            human_state = None
            if row is not None:
                human_state = (
                    row.left_visibility if event.get("track") == "left" else row.right_visibility
                )
            claimed.append(
                {
                    "sample_id": sample_id,
                    "frame_index": event.get("frame_index"),
                    "track": event.get("track"),
                    "frames_absent": event.get("frames_absent"),
                    "human_state_for_that_hand": human_state,
                    "annotator_confidence": row.annotator_confidence if row else None,
                    "supported_by_reference": human_state in EXPECTED_POSE_STATES
                    if human_state is not None
                    else None,
                }
            )
    supported = [event for event in claimed if event["supported_by_reference"] is True]
    contradicted = [event for event in claimed if event["supported_by_reference"] is False]
    return {
        "tracker_claimed_reacquisitions": len(claimed),
        "supported_by_reference": len(supported),
        "contradicted_by_reference": len(contradicted),
        "detail": claimed,
    }
