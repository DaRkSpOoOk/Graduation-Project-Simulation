"""Conservative pose-quality assessment for tracking.

Design note grounded in measured evidence: because WiLoR's joints come from
the MANO parametric model, bone lengths are almost constant regardless of
whether the crop was a real hand. Across the 1,779 reconstructed rows of the
pilot, palm length spans only 0.0941-0.0970 and the projected joints always
fall inside the detector box. The documented spurious third detection
(frame 39 of ``karsl_test_s02_sign0176_repfirst``) has *perfectly normal*
intrinsic geometry.

So intrinsic geometry checks alone cannot catch a bad WiLoR reconstruction.
They are retained here as a genuine safety net for gross collapse, while the
mechanism that actually removes that spurious hand is same-label duplicate
suppression plus the two-track capacity limit (see ``association.py``).

The gate marks and reports; it never fabricates a replacement pose, and the
raw row always remains recoverable through ``raw_detection_index``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import TrackerConfig
from .schema import RawDetection

# Quality flag vocabulary emitted by this module.
FLAG_LOW_CONFIDENCE = "LOW_QUALITY_CONFIDENCE"
FLAG_NON_FINITE = "LOW_QUALITY_NON_FINITE"
FLAG_GEOMETRY_COLLAPSE = "LOW_QUALITY_GEOMETRY_COLLAPSE"
FLAG_PROJECTION_INCONSISTENT = "LOW_QUALITY_PROJECTION_INCONSISTENT"
FLAG_POSE_JUMP = "LOW_QUALITY_POSE_JUMP"


@dataclass(frozen=True, slots=True)
class QualityAssessment:
    passed: bool
    flags: tuple[str, ...]
    span_palm_ratio: float | None
    projection_centre_offset: float | None

    @property
    def reason(self) -> str:
        return ",".join(self.flags) if self.flags else "ok"


def project_joints(detection: RawDetection) -> np.ndarray | None:
    """Project the reconstructed joints into full-image pixels.

    Uses the same pinhole model as the frozen WiLoR visualization path
    (``pose/wilor/geometry.py: project_points_full_img``).
    """

    if detection.focal_length <= 0:
        return None
    points = detection.landmarks_3d + detection.camera_translation
    depth = points[..., -1:]
    if not np.all(np.isfinite(depth)) or np.any(np.abs(depth) < 1e-9):
        return None
    normalized = points / depth
    width, height = detection.image_size_wh
    intrinsics = np.eye(3)
    intrinsics[0, 0] = intrinsics[1, 1] = detection.focal_length
    intrinsics[0, 2] = width / 2.0
    intrinsics[1, 2] = height / 2.0
    return (intrinsics @ normalized.T).T[..., :2]


def geodesic_degrees(first: np.ndarray, second: np.ndarray) -> float:
    """Rotation distance between two 3x3 rotation matrices, in degrees."""

    relative = np.asarray(first).reshape(3, 3).T @ np.asarray(second).reshape(3, 3)
    cosine = (float(np.trace(relative)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def assess_detection(
    detection: RawDetection,
    config: TrackerConfig,
    previous_orientation: np.ndarray | None = None,
) -> QualityAssessment:
    """Evaluate one raw detection. Failing the gate never deletes raw data."""

    flags: list[str] = []
    passed = True

    arrays = (
        detection.landmarks_3d,
        detection.hand_pose_rotmat,
        detection.global_orient_rotmat,
        detection.betas,
        detection.camera_translation,
    )
    if not all(np.isfinite(np.asarray(array)).all() for array in arrays):
        return QualityAssessment(False, (FLAG_NON_FINITE,), None, None)

    if detection.detector_confidence is not None and (
        detection.detector_confidence < config.min_detection_confidence
    ):
        flags.append(FLAG_LOW_CONFIDENCE)
        passed = False

    joints = detection.landmarks_3d
    palm_length = float(np.linalg.norm(joints[9] - joints[0]))
    span_palm_ratio: float | None = None
    if palm_length <= 0 or not math.isfinite(palm_length):
        flags.append(FLAG_GEOMETRY_COLLAPSE)
        passed = False
    else:
        span = float(np.linalg.norm(joints.max(axis=0) - joints.min(axis=0)))
        span_palm_ratio = span / palm_length
        if not (config.min_span_palm_ratio <= span_palm_ratio <= config.max_span_palm_ratio):
            flags.append(FLAG_GEOMETRY_COLLAPSE)
            passed = False

    projection_offset: float | None = None
    projected = project_joints(detection)
    if projected is not None and detection.box_size > 0:
        centroid = projected.mean(axis=0)
        projection_offset = float(
            math.hypot(
                centroid[0] - detection.box_center_xy[0],
                centroid[1] - detection.box_center_xy[1],
            )
            / detection.box_size
        )
        if projection_offset > config.max_projection_centre_offset:
            flags.append(FLAG_PROJECTION_INCONSISTENT)
            passed = False

    # Advisory only: a large orientation jump is reported, never rejected,
    # because fast genuine rotation is indistinguishable from a bad estimate
    # without ground truth.
    if previous_orientation is not None:
        if geodesic_degrees(previous_orientation, detection.global_orient_rotmat) > (
            config.orientation_jump_warn_degrees
        ):
            flags.append(FLAG_POSE_JUMP)

    return QualityAssessment(passed, tuple(flags), span_palm_ratio, projection_offset)
