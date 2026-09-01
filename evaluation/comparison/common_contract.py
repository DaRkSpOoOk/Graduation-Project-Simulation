"""Shared, extractor-neutral comparison contract.

The raw MediaPipe and WiLoR representations are intentionally not rewritten.
This module only defines the validity predicate and the common 21-point bone
topology used when deriving comparison metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


EXPECTED_MANIFEST_SHA256 = "4a8e0d24f587d9c54ad87ebd896c6df0290b3ebc80885774100d71b29dbc3c0c"
EXPECTED_SAMPLE_COUNT = 18
EXPECTED_TOTAL_FRAMES = 894
EXPECTED_MEDIAPIPE_MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"

# Copied from the frozen implementations for an explicit intersection check.
MEDIAPIPE_HAND_CONNECTIONS_21: tuple[tuple[int, int], ...] = (
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

WILOR_HAND_BONES_20: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)

# The exact set intersection: all phalanx chains plus wrist-to-index and
# wrist-to-pinky. The middle/ring MCP edges are intentionally excluded because
# the frozen implementations use different palm edges there.
COMMON_HAND_BONES_18: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (9, 10),
    (10, 11),
    (11, 12),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)


@dataclass(frozen=True, slots=True)
class HandRecord:
    """One raw output row or one fixed detector slot.

    ``image_landmarks`` uses normalized image coordinates for MediaPipe. WiLoR
    has no serialized full-mode 2D landmarks, so its comparable image-space
    position is derived from ``mano_references['box_center_xy']`` by the
    neutral metric layer. ``landmarks_3d`` is never compared in absolute
    units between extractors.
    """

    system: str
    frame_index: int
    hand_present: bool
    handedness_label: str | None
    confidence: float | None
    detection_confidence: float | None
    image_landmarks: np.ndarray | None
    landmarks_3d: np.ndarray | None
    mano_params: dict[str, Any] | None
    mano_references: dict[str, Any] | None
    mode: str | None
    quality_flags: tuple[str, ...] = ()
    source_index: int = 0


def normalize_label(value: Any) -> str | None:
    """Return only the two labels used by the common left/right contract."""

    if value is None:
        return None
    label = str(value).strip().casefold()
    return label if label in {"left", "right"} else None


def _finite_landmarks(value: np.ndarray | None) -> bool:
    if value is None:
        return False
    array = np.asarray(value)
    return array.shape == (21, 3) and bool(np.isfinite(array).all())


def _nonempty_finite(value: Any) -> bool:
    if value is None:
        return False
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return array.size > 0 and bool(np.isfinite(array).all())


def reconstructed_hand(record: HandRecord) -> bool:
    """Apply the common complete-pose predicate without changing raw output.

    MediaPipe has no MANO requirement: a hand is valid when its 21 image and
    21 world landmarks are present and finite. WiLoR additionally must be in
    exact ``full`` mode and carry the required MANO fields.
    """

    if not record.hand_present:
        return False
    if record.system == "mediapipe":
        return _finite_landmarks(record.image_landmarks) and _finite_landmarks(record.landmarks_3d)
    if record.system == "wilor":
        if record.mode != "full" or not _finite_landmarks(record.landmarks_3d):
            return False
        mano = record.mano_params or {}
        return all(_nonempty_finite(mano.get(field)) for field in (
            "hand_pose_rotmat",
            "global_orient_rotmat",
            "betas",
        ))
    raise ValueError(f"Unknown extractor system: {record.system!r}")


def assert_common_bone_intersection() -> None:
    """Fail loudly if the frozen edge-list transcription drifts."""

    media = set(MEDIAPIPE_HAND_CONNECTIONS_21)
    wilor = set(WILOR_HAND_BONES_20)
    common = set(COMMON_HAND_BONES_18)
    if len(MEDIAPIPE_HAND_CONNECTIONS_21) != 21 or len(WILOR_HAND_BONES_20) != 20:
        raise AssertionError("Frozen source edge-list cardinality changed")
    if common != media & wilor or len(common) != 18:
        raise AssertionError("COMMON_HAND_BONES_18 is not the exact frozen edge intersection")


assert_common_bone_intersection()
