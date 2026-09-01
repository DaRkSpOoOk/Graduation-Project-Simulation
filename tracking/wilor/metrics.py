"""Neutral tracking metrics for the derived LEFT/RIGHT sequence.

These are descriptive counts over the tracker's own decisions. There is no
identity ground truth in this pilot, so nothing here is an accuracy score:
"suspected identity switch" means the tracker's own evidence looked
inconsistent, not that a switch provably occurred. An independent
annotation/benchmark set is being prepared separately for validation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .schema import TRACK_NAMES, TrackedSequence, TrackState

_OBSERVED_STATES = frozenset({TrackState.OBSERVED, TrackState.AMBIGUOUS})
_MISSING_STATES = frozenset(
    {TrackState.MISSING, TrackState.LIKELY_OCCLUDED, TrackState.REJECTED_QUALITY}
)


@dataclass(slots=True)
class TrackingMetrics:
    sample_id: str
    total_frames: int = 0
    observed_left_frames: int = 0
    observed_right_frames: int = 0
    missing_left_frames: int = 0
    missing_right_frames: int = 0
    likely_occluded_left_frames: int = 0
    likely_occluded_right_frames: int = 0
    ambiguous_frames: int = 0
    quality_rejected_detections: int = 0
    quality_rejected_frames: int = 0
    frames_with_rejected_detections: int = 0
    extra_detection_frames: int = 0
    extra_detections_total: int = 0
    frames_with_more_than_two_raw_detections: int = 0
    duplicate_suppressed_detections: int = 0
    handedness_disagreement_events: int = 0
    both_labels_swapped_frames: int = 0
    reassociation_events: int = 0
    suspected_identity_switch_events: int = 0
    longest_left_missing_run: int = 0
    longest_right_missing_run: int = 0
    frames_with_both_tracks: int = 0
    frames_with_no_track: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _longest_run(flags: list[bool]) -> int:
    longest = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return longest


def compute_metrics(sequence: TrackedSequence) -> TrackingMetrics:
    metrics = TrackingMetrics(sample_id=sequence.sample_id, total_frames=len(sequence.frames))
    missing_runs: dict[str, list[bool]] = {track: [] for track in TRACK_NAMES}

    for frame in sequence.frames:
        observed_count = 0
        for track in TRACK_NAMES:
            hand = frame.hand(track)
            is_observed = hand.state in _OBSERVED_STATES
            missing_runs[track].append(not is_observed)
            if is_observed:
                observed_count += 1
                if track == "left":
                    metrics.observed_left_frames += 1
                else:
                    metrics.observed_right_frames += 1
            else:
                if track == "left":
                    metrics.missing_left_frames += 1
                else:
                    metrics.missing_right_frames += 1
                if hand.state is TrackState.LIKELY_OCCLUDED:
                    if track == "left":
                        metrics.likely_occluded_left_frames += 1
                    else:
                        metrics.likely_occluded_right_frames += 1

        if observed_count == 2:
            metrics.frames_with_both_tracks += 1
        elif observed_count == 0:
            metrics.frames_with_no_track += 1

        if any(frame.hand(track).state is TrackState.AMBIGUOUS for track in TRACK_NAMES):
            metrics.ambiguous_frames += 1

        frame_had_quality_rejection = False
        for reason in frame.rejection_reasons.values():
            if reason.startswith("quality:"):
                metrics.quality_rejected_detections += 1
                frame_had_quality_rejection = True
            elif reason.startswith("duplicate_same_label"):
                metrics.duplicate_suppressed_detections += 1
        if frame_had_quality_rejection:
            metrics.quality_rejected_frames += 1
        if frame.rejected_detection_indices:
            metrics.frames_with_rejected_detections += 1

        if frame.extra_detection_count:
            metrics.extra_detection_frames += 1
            metrics.extra_detections_total += frame.extra_detection_count
        if frame.number_of_raw_detections > 2:
            metrics.frames_with_more_than_two_raw_detections += 1

        if "BOTH_LABELS_SWAPPED" in frame.tracking_flags:
            metrics.both_labels_swapped_frames += 1

    for event in sequence.events:
        kind = event.get("event")
        if kind == "handedness_disagreement":
            metrics.handedness_disagreement_events += 1
        elif kind == "reassociation":
            metrics.reassociation_events += 1

    metrics.suspected_identity_switch_events = _count_suspected_switches(sequence)
    metrics.longest_left_missing_run = _longest_run(missing_runs["left"])
    metrics.longest_right_missing_run = _longest_run(missing_runs["right"])
    return metrics


def _count_suspected_switches(sequence: TrackedSequence) -> int:
    """Heuristic, NOT an accuracy measure.

    A suspected switch is counted when, between two consecutive frames in
    which both tracks are observed, the detector-label agreement of *both*
    tracks flips at once (agree -> disagree or the reverse). That pattern is
    what an identity swap would look like in the absence of ground truth; it
    can also be produced by genuine detector label error.
    """

    suspected = 0
    previous: tuple[bool, bool] | None = None
    for frame in sequence.frames:
        left, right = frame.left, frame.right
        if left.state not in _OBSERVED_STATES or right.state not in _OBSERVED_STATES:
            previous = None
            continue
        if left.detector_label is None or right.detector_label is None:
            previous = None
            continue
        current = (left.detector_label == "left", right.detector_label == "right")
        if previous is not None and current != previous and current[0] == current[1]:
            suspected += 1
        previous = current
    return suspected


def aggregate_metrics(per_video: list[TrackingMetrics]) -> dict[str, Any]:
    """Frame-weighted totals across videos, plus per-video maxima."""

    if not per_video:
        return {}
    summable = [
        "total_frames",
        "observed_left_frames",
        "observed_right_frames",
        "missing_left_frames",
        "missing_right_frames",
        "likely_occluded_left_frames",
        "likely_occluded_right_frames",
        "ambiguous_frames",
        "quality_rejected_detections",
        "quality_rejected_frames",
        "frames_with_rejected_detections",
        "extra_detection_frames",
        "extra_detections_total",
        "frames_with_more_than_two_raw_detections",
        "duplicate_suppressed_detections",
        "handedness_disagreement_events",
        "both_labels_swapped_frames",
        "reassociation_events",
        "suspected_identity_switch_events",
        "frames_with_both_tracks",
        "frames_with_no_track",
    ]
    totals: dict[str, Any] = {"videos": len(per_video)}
    for name in summable:
        totals[name] = sum(getattr(metrics, name) for metrics in per_video)
    totals["longest_left_missing_run_max"] = max(m.longest_left_missing_run for m in per_video)
    totals["longest_right_missing_run_max"] = max(m.longest_right_missing_run for m in per_video)
    frames = totals["total_frames"] or 1
    totals["rates_pct"] = {
        "observed_left": 100.0 * totals["observed_left_frames"] / frames,
        "observed_right": 100.0 * totals["observed_right_frames"] / frames,
        "both_tracks": 100.0 * totals["frames_with_both_tracks"] / frames,
        "no_track": 100.0 * totals["frames_with_no_track"] / frames,
        "ambiguous": 100.0 * totals["ambiguous_frames"] / frames,
    }
    return totals
