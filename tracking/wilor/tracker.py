"""The WiLoR temporal dual-hand tracker state machine.

Per frame the tracker runs a fixed, deterministic pipeline:

1. quality-gate every raw detection (mark, never fabricate);
2. suppress same-label duplicate boxes;
3. solve the exact 2 x N assignment against the two canonical tracks;
4. bind assignments, emit MISSING/LIKELY_OCCLUDED for unbound tracks;
5. record extra/rejected detections, ambiguity, label disagreement and
   reassociation evidence.

The tracker only establishes temporal identity and quality/state metadata.
It computes no joint angles, no sensor values and no LSTM features - those
belong to later tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .association import (
    AssignmentResult,
    TrackPrior,
    solve_assignment,
    suppress_same_label_duplicates,
)
from .config import DEFAULT_CONFIG, TrackerConfig
from .quality import assess_detection
from .schema import (
    TRACK_NAMES,
    RawDetection,
    TrackedFrame,
    TrackedHand,
    TrackedSequence,
    TrackState,
)
from .source import RawSequence

# Frame-level tracking flags.
FLAG_NO_DETECTIONS = "NO_DETECTIONS"
FLAG_EXTRA_DETECTIONS = "EXTRA_DETECTIONS"
FLAG_AMBIGUOUS_ASSIGNMENT = "AMBIGUOUS_ASSIGNMENT"
FLAG_LABEL_DISAGREEMENT = "LABEL_DISAGREEMENT"
FLAG_BOTH_LABELS_SWAPPED = "BOTH_LABELS_SWAPPED"
FLAG_REASSOCIATION = "REASSOCIATION"
FLAG_QUALITY_REJECTION = "QUALITY_REJECTION"
FLAG_SEEDED = "TRACKS_SEEDED"


@dataclass(slots=True)
class _TrackMemory:
    track: str
    last_centre: tuple[float, float] | None = None
    last_normalized_size: float | None = None
    last_orientation: np.ndarray | None = None
    frames_since_observed: int = 0
    observed_once: bool = False
    last_observed_frame: int | None = None
    last_label_agreed: bool | None = None

    def prior(self) -> TrackPrior:
        return TrackPrior(
            track=self.track,
            last_centre=self.last_centre,
            last_normalized_size=self.last_normalized_size,
            frames_since_observed=self.frames_since_observed,
        )

    def observe(self, detection: RawDetection, frame_index: int) -> None:
        self.last_centre = detection.normalized_centre
        self.last_normalized_size = detection.normalized_size
        self.last_orientation = detection.global_orient_rotmat.copy()
        self.frames_since_observed = 0
        self.observed_once = True
        self.last_observed_frame = frame_index

    def miss(self) -> None:
        self.frames_since_observed += 1


def _seed_priors(
    memories: dict[str, _TrackMemory],
    detections: list[RawDetection],
    config: TrackerConfig,
) -> list[dict[str, Any]]:
    """Bootstrap uninitialized tracks from the first usable frame.

    Detector labels are used when they are distinct. When both detections
    carry the same label the image-position convention verified in TASK-003A
    (subject RIGHT hand at the smaller image x for this facing, non-mirrored
    capture, ~95-97% of two-hand frames) breaks the tie. The fallback is
    recorded as evidence, never presented as ground truth.
    """

    if any(memory.observed_once for memory in memories.values()):
        return []
    if not detections:
        return []

    labels = [detection.detector_label for detection in detections]
    events: list[dict[str, Any]] = []
    if len(detections) >= 2 and len({label for label in labels if label}) < 2:
        if not config.seed_right_hand_at_smaller_x:
            return []
        ordered = sorted(detections, key=lambda d: d.normalized_centre[0])
        seeds = {"right": ordered[0], "left": ordered[-1]}
        events.append(
            {
                "event": "seed_by_position",
                "frame_index": detections[0].frame_index,
                "detector_labels": labels,
                "basis": "subject right hand appears at smaller image x (TASK-003A convention)",
            }
        )
        for track, detection in seeds.items():
            memories[track].last_centre = detection.normalized_centre
            memories[track].last_normalized_size = detection.normalized_size
    return events


def track_sequence(
    raw: RawSequence,
    config: TrackerConfig | None = None,
    source: dict[str, Any] | None = None,
) -> TrackedSequence:
    """Derive the LEFT/RIGHT tracked sequence from immutable raw output."""

    config = config or DEFAULT_CONFIG
    config.validate()

    memories = {name: _TrackMemory(track=name) for name in TRACK_NAMES}
    frames: list[TrackedFrame] = []
    events: list[dict[str, Any]] = []

    for frame_index in raw.frame_indices:
        detections = list(raw.detections(frame_index))
        timestamp = raw.timestamps.get(frame_index, float("nan"))
        flags: list[str] = []
        rejection_reasons: dict[int, str] = {}

        # 1. quality gate -------------------------------------------------
        usable: list[RawDetection] = []
        quality_flags_by_raw_index: dict[int, tuple[str, ...]] = {}
        for detection in detections:
            assessment = assess_detection(
                detection,
                config,
                previous_orientation=_orientation_prior(memories, detection),
            )
            quality_flags_by_raw_index[detection.raw_detection_index] = assessment.flags
            if assessment.passed:
                usable.append(detection)
            else:
                rejection_reasons[detection.raw_detection_index] = f"quality:{assessment.reason}"

        if rejection_reasons:
            flags.append(FLAG_QUALITY_REJECTION)

        # 2. same-label duplicate suppression ------------------------------
        kept_positions, suppressed = suppress_same_label_duplicates(usable, config)
        for position, reason in suppressed.items():
            rejection_reasons[usable[position].raw_detection_index] = reason
        candidates = [usable[position] for position in kept_positions]

        # 3. seeding + assignment ------------------------------------------
        seed_events = _seed_priors(memories, candidates, config)
        if seed_events:
            events.extend(seed_events)
            flags.append(FLAG_SEEDED)

        priors = {name: memories[name].prior() for name in TRACK_NAMES}
        assignment = solve_assignment(priors, candidates, config)

        if not detections:
            flags.append(FLAG_NO_DETECTIONS)

        # 4. bind results ---------------------------------------------------
        bound: dict[str, TrackedHand] = {}
        for track in TRACK_NAMES:
            memory = memories[track]
            if track in assignment.mapping:
                detection = candidates[assignment.mapping[track]]
                was_missing = memory.observed_once and memory.frames_since_observed > 0
                state = TrackState.AMBIGUOUS if assignment.ambiguous else TrackState.OBSERVED
                bound[track] = TrackedHand(
                    track=track,
                    state=state,
                    raw_detection_index=detection.raw_detection_index,
                    detector_label=detection.detector_label,
                    detector_confidence=detection.detector_confidence,
                    assignment_cost=assignment.costs.get(track),
                    landmarks_3d=detection.landmarks_3d,
                    hand_pose_rotmat=detection.hand_pose_rotmat,
                    global_orient_rotmat=detection.global_orient_rotmat,
                    betas=detection.betas,
                    camera_translation=detection.camera_translation,
                    box_center_xy=detection.box_center_xy,
                    box_size=detection.box_size,
                    quality_flags=tuple(detection.raw_quality_flags)
                    + quality_flags_by_raw_index.get(detection.raw_detection_index, ()),
                )
                if was_missing:
                    flags.append(FLAG_REASSOCIATION)
                    events.append(
                        {
                            "event": "reassociation",
                            "frame_index": frame_index,
                            "track": track,
                            "frames_absent": memory.frames_since_observed,
                            "assignment_cost": assignment.costs.get(track),
                            "detector_label": detection.detector_label,
                            "detector_label_agreed": detection.detector_label == track,
                            "raw_detection_index": detection.raw_detection_index,
                        }
                    )
                memory.observe(detection, frame_index)
                memory.last_label_agreed = detection.detector_label == track
            else:
                bound[track] = TrackedHand(track=track, state=TrackState.MISSING)
                memory.miss()

        # Refine the state of every unbound track. Order of precedence:
        #   REJECTED_QUALITY  a gated-out detection plausibly belonged here
        #   LIKELY_OCCLUDED   heuristic: the hand was last seen on top of the
        #                     other hand, which is still observed
        #   MISSING           no evidence either way
        image_size = detections[0].image_size_wh if detections else None
        quality_rejected = [
            detection
            for detection in detections
            if rejection_reasons.get(detection.raw_detection_index, "").startswith("quality:")
        ]
        for track in TRACK_NAMES:
            hand = bound[track]
            if hand.state is not TrackState.MISSING:
                continue
            memory = memories[track]

            claimed = _nearest_within_gate(memory, quality_rejected, config)
            if claimed is not None:
                hand.state = TrackState.REJECTED_QUALITY
                hand.raw_detection_index = claimed.raw_detection_index
                hand.detector_label = claimed.detector_label
                hand.detector_confidence = claimed.detector_confidence
                hand.quality_flags = quality_flags_by_raw_index.get(
                    claimed.raw_detection_index, ()
                )
                continue

            other = bound["right" if track == "left" else "left"]
            if (
                other.state in {TrackState.OBSERVED, TrackState.AMBIGUOUS}
                and memory.observed_once
                and memory.last_centre is not None
                and other.box_center_xy is not None
                and image_size is not None
                and memory.frames_since_observed <= config.occlusion_max_age_frames
            ):
                other_centre = (
                    other.box_center_xy[0] / image_size[0],
                    other.box_center_xy[1] / image_size[1],
                )
                distance = float(
                    np.hypot(
                        memory.last_centre[0] - other_centre[0],
                        memory.last_centre[1] - other_centre[1],
                    )
                )
                if distance <= config.occlusion_proximity_radius:
                    hand.state = TrackState.LIKELY_OCCLUDED
                    hand.quality_flags = (f"heuristic_occlusion_distance={distance:.4f}",)

        # 5. frame-level bookkeeping ----------------------------------------
        rejected_indices = tuple(sorted(rejection_reasons))
        extra_count = max(0, len(candidates) - len(assignment.mapping))
        if len(detections) > 2 or extra_count:
            flags.append(FLAG_EXTRA_DETECTIONS)
        for position, detection in enumerate(candidates):
            if position not in assignment.assigned_indices:
                rejection_reasons.setdefault(
                    detection.raw_detection_index, "unassigned_extra_detection"
                )
        rejected_indices = tuple(sorted(rejection_reasons))

        if assignment.ambiguous and assignment.mapping:
            flags.append(FLAG_AMBIGUOUS_ASSIGNMENT)

        disagreements = [track for track in TRACK_NAMES if bound[track].label_disagrees]
        if disagreements:
            flags.append(FLAG_LABEL_DISAGREEMENT)
            events.append(
                {
                    "event": "handedness_disagreement",
                    "frame_index": frame_index,
                    "tracks": disagreements,
                    "detector_labels": {
                        track: bound[track].detector_label for track in disagreements
                    },
                }
            )
        if len(disagreements) == 2:
            flags.append(FLAG_BOTH_LABELS_SWAPPED)

        frames.append(
            TrackedFrame(
                frame_index=frame_index,
                timestamp_seconds=timestamp,
                left=bound["left"],
                right=bound["right"],
                number_of_raw_detections=len(detections),
                extra_detection_count=extra_count,
                rejected_detection_indices=rejected_indices,
                rejection_reasons=dict(rejection_reasons),
                assignment_margin=assignment.margin,
                tracking_flags=tuple(dict.fromkeys(flags)),
            )
        )

    return TrackedSequence(
        sample_id=raw.sample_id,
        frames=frames,
        config=config.to_dict(),
        source=source or {},
        events=events,
    )


def _orientation_prior(
    memories: dict[str, _TrackMemory], detection: RawDetection
) -> np.ndarray | None:
    """Nearest plausible previous orientation for the pose-jump advisory."""

    label = detection.detector_label
    if label in memories and memories[label].last_orientation is not None:
        return memories[label].last_orientation
    return None


def _nearest_within_gate(
    memory: _TrackMemory, detections: list[RawDetection], config: TrackerConfig
) -> RawDetection | None:
    """Closest quality-rejected detection that falls inside a track's gate.

    Used only to explain *why* a track has no pose this frame: it lets the
    state say REJECTED_QUALITY ("a detection was here but was not trusted")
    instead of a bare MISSING. No pose is ever taken from it.
    """

    if not detections:
        return None
    from .association import gate_radius  # local import avoids a cycle

    if memory.last_centre is None:
        # Uninitialized track: any rejected detection explains the absence,
        # chosen deterministically by raw index.
        return min(detections, key=lambda d: d.raw_detection_index)

    radius = gate_radius(memory.prior(), config)
    best: RawDetection | None = None
    best_distance = float("inf")
    for detection in detections:
        centre = detection.normalized_centre
        distance = float(
            np.hypot(centre[0] - memory.last_centre[0], centre[1] - memory.last_centre[1])
        )
        if distance <= radius and (
            distance < best_distance
            or (distance == best_distance and best is not None and detection.raw_detection_index < best.raw_detection_index)
        ):
            best, best_distance = detection, distance
    return best
