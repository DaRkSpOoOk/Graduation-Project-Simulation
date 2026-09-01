"""Derived tracked representation for WiLoR dual-hand output.

This schema is a NEW derived stage. It never replaces raw output: every
observed entry keeps provenance (`raw_detection_index`, the original
detector label/confidence, and the raw quality flags) so the corresponding
row in ``wilor_raw.npz`` can always be recovered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class TrackState(str, Enum):
    """Per-frame state of one canonical track.

    OBSERVED          a raw detection was bound to this track this frame.
    MISSING           no detection was bound; no pose is available.
    LIKELY_OCCLUDED   MISSING, plus a *heuristic* suggestion that the hand was
                      hidden behind/near the other hand. Never ground truth.
    AMBIGUOUS         a detection was bound, but the assignment was close to a
                      competing one; identity is uncertain this frame.
    REJECTED_QUALITY  a detection existed for this track's position but failed
                      the quality gate, so no pose is exposed.
    """

    OBSERVED = "OBSERVED"
    MISSING = "MISSING"
    LIKELY_OCCLUDED = "LIKELY_OCCLUDED"
    AMBIGUOUS = "AMBIGUOUS"
    REJECTED_QUALITY = "REJECTED_QUALITY"


STATE_CODES: dict[TrackState, int] = {
    TrackState.MISSING: 0,
    TrackState.OBSERVED: 1,
    TrackState.AMBIGUOUS: 2,
    TrackState.REJECTED_QUALITY: 3,
    TrackState.LIKELY_OCCLUDED: 4,
}
CODE_TO_STATE: dict[int, TrackState] = {code: state for state, code in STATE_CODES.items()}

# Canonical track order used by every array in the tracked NPZ.
TRACK_NAMES: tuple[str, str] = ("left", "right")
LABEL_CODES: dict[str, int] = {"left": 0, "right": 1}
CODE_TO_LABEL: dict[int, str] = {0: "left", 1: "right"}

POSE_STATES = frozenset({TrackState.OBSERVED, TrackState.AMBIGUOUS})


@dataclass(slots=True)
class RawDetection:
    """One immutable raw WiLoR row, plus derived read-only conveniences."""

    frame_index: int
    raw_detection_index: int
    detector_label: str | None
    detector_confidence: float | None
    landmarks_3d: np.ndarray               # (21, 3) model units, root-relative
    hand_pose_rotmat: np.ndarray           # (15, 3, 3)
    global_orient_rotmat: np.ndarray       # (3, 3)
    betas: np.ndarray                      # (10,)
    camera_translation: np.ndarray         # (3,)
    box_center_xy: tuple[float, float]     # pixels
    box_size: float                        # pixels
    image_size_wh: tuple[float, float]     # pixels
    focal_length: float
    raw_quality_flags: tuple[str, ...] = ()

    @property
    def normalized_centre(self) -> tuple[float, float]:
        width, height = self.image_size_wh
        return (self.box_center_xy[0] / width, self.box_center_xy[1] / height)

    @property
    def normalized_size(self) -> float:
        return self.box_size / self.image_size_wh[0]


@dataclass(slots=True)
class TrackedHand:
    """One canonical track's entry for one frame."""

    track: str                              # "left" | "right"
    state: TrackState
    raw_detection_index: int | None = None
    detector_label: str | None = None
    detector_confidence: float | None = None
    assignment_cost: float | None = None
    landmarks_3d: np.ndarray | None = None
    hand_pose_rotmat: np.ndarray | None = None
    global_orient_rotmat: np.ndarray | None = None
    betas: np.ndarray | None = None
    camera_translation: np.ndarray | None = None
    box_center_xy: tuple[float, float] | None = None
    box_size: float | None = None
    quality_flags: tuple[str, ...] = ()

    @property
    def has_pose(self) -> bool:
        return self.state in POSE_STATES and self.landmarks_3d is not None

    @property
    def label_disagrees(self) -> bool:
        """True when the detector called this hand the opposite identity."""
        return self.detector_label is not None and self.detector_label != self.track


@dataclass(slots=True)
class TrackedFrame:
    frame_index: int
    timestamp_seconds: float
    left: TrackedHand
    right: TrackedHand
    number_of_raw_detections: int = 0
    extra_detection_count: int = 0
    rejected_detection_indices: tuple[int, ...] = ()
    rejection_reasons: dict[int, str] = field(default_factory=dict)
    assignment_margin: float | None = None
    tracking_flags: tuple[str, ...] = ()

    def hand(self, track: str) -> TrackedHand:
        return self.left if track == "left" else self.right


@dataclass(slots=True)
class TrackedSequence:
    sample_id: str
    frames: list[TrackedFrame]
    config: dict[str, Any]
    source: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_frames(self) -> int:
        return len(self.frames)
