"""Canonical palm frame, and the one place handedness is canonicalized.

THE FRAME
=========

Built from four stable landmarks -- wrist, index MCP, middle MCP, pinky MCP --
which the pilot shows to be near-coplanar (third singular value 0.0056 against
0.077 and 0.050 for the first two).

    f  (distal axis)  = normalize( middle_MCP - wrist )
    u_raw             = index_MCP - pinky_MCP        on the RIGHT hand
                      = pinky_MCP - index_MCP        on the LEFT hand
    n  (palm normal)  = normalize( f x u_raw )
    u  (lateral axis) = n x f
    R                 = [ u | n | f ]   as COLUMNS

``u`` is re-derived as ``n x f`` rather than used raw, so the three axes are
exactly orthonormal even though the four landmarks are only approximately
coplanar. det(R) = +1 always, by construction.

Direction of ``n``, established from an explicit canonical pose (see
``tests/test_kinematics.py::TestPalmFrameConvention``): for a right hand held
with the palm facing the viewer and the fingers pointing up, ``n`` points at
the viewer. So **n is the palmar (volar) normal**; the dorsal normal is -n.

HANDEDNESS
==========

This module contains the ONLY handedness-dependent line in TASK-005A: the
direction of ``u_raw``. Everything else -- flexion, spread -- is built from
dot products of joint differences and is therefore automatically invariant
under reflection, so those channels need no correction at all.

Why the flip is necessary, and why it is not a fudge. Take a right hand and
its exact mirror image, ``M = diag(-1, 1, 1)``, so ``p_left = M p_right``. For
a reflection, ``(M a) x (M b) = -M (a x b)``. Applying the *same* formula to
both hands would therefore give ``n_left = -M n_right``: the normal would point
out of the palm on one hand and into the palm on the other, which is exactly
the sort of silent convention error this task must avoid. Reversing ``u_raw``
on the left hand cancels that sign:

    f_left = M f_right
    u_raw_left = -M u_raw_right          (by the reversed definition)
    n_left = normalize(M f x -M u) = -(-M n_right) = M n_right
    u_left = n_left x f_left = (M n) x (M f) = -M (n x f) = -M u_right

so

    R_left = M @ R_right @ diag(-1, 1, 1)

with det(R_left) = (-1)(+1)(-1) = +1. This identity is asserted directly in
``TestMirroredHands``.

The consequence, stated plainly rather than glossed: the distal axis ``f`` and
the palmar normal ``n`` carry the SAME anatomical meaning on both hands. The
lateral axis ``u`` does not -- it points radially (toward the thumb) on the
right hand and ulnarly (toward the pinky) on the left. That asymmetry is
unavoidable: two mirror-image hands cannot both have three anatomically
identical axes AND both be right-handed coordinate frames. One axis must carry
the handedness, and the lateral axis is the least useful of the three for the
downstream virtual-IMU work, which cares about where the palm faces and where
the fingers point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import (
    MIN_BONE_LENGTH,
    MIN_FRAME_CROSS_NORM,
    MIN_PALM_LANDMARK_SEPARATION_RATIO,
    safe_normalize,
)
from .layout import PALM_INDEX_MCP, PALM_MIDDLE_MCP, PALM_PINKY_MCP, WRIST

FLAG_PALM_NON_FINITE = "PALM_POINTS_NON_FINITE"
FLAG_PALM_ZERO_AXIS = "PALM_AXIS_ZERO_LENGTH"
FLAG_PALM_COLLINEAR = "PALM_POINTS_COLLINEAR"
FLAG_PALM_COINCIDENT = "PALM_LANDMARKS_COINCIDENT"


@dataclass(slots=True)
class PalmFrame:
    """Orthonormal palm frame with det = +1."""

    rotation: np.ndarray          # (3, 3), columns [lateral | palmar normal | distal]
    origin: np.ndarray            # (3,) wrist position, kept for reference only
    lateral: np.ndarray           # (3,)
    normal: np.ndarray            # (3,) palmar
    distal: np.ndarray            # (3,)


def build_palm_frame(joints: np.ndarray, track: str) -> tuple[PalmFrame | None, list[str]]:
    """Construct the canonical palm frame for one hand.

    ``track`` must be "left" or "right"; it selects the lateral-axis direction
    documented in the module docstring and is the only place handedness enters.

    Returns ``(frame, flags)``. ``frame`` is None when the landmarks are
    non-finite, an axis is degenerate, or the defining vectors are collinear.
    """

    flags: list[str] = []
    joints = np.asarray(joints, dtype=np.float64)
    wrist = joints[WRIST]
    index_mcp = joints[PALM_INDEX_MCP]
    middle_mcp = joints[PALM_MIDDLE_MCP]
    pinky_mcp = joints[PALM_PINKY_MCP]

    if not np.isfinite(np.stack([wrist, index_mcp, middle_mcp, pinky_mcp])).all():
        return None, [FLAG_PALM_NON_FINITE]

    palm_length = float(np.linalg.norm(middle_mcp - wrist))
    distal = safe_normalize(middle_mcp - wrist)
    if distal is None:
        return None, [FLAG_PALM_ZERO_AXIS]

    # The four landmarks must be mutually distinct for the frame they define to
    # mean anything. Checked on the points themselves rather than on the axes:
    # a collapsed palm can still yield two non-parallel, non-zero axis vectors
    # and so would otherwise produce a perfectly finite frame for geometry that
    # cannot be a hand.
    landmarks = (
        ("wrist", wrist),
        ("index_MCP", index_mcp),
        ("middle_MCP", middle_mcp),
        ("pinky_MCP", pinky_mcp),
    )
    minimum_separation = MIN_PALM_LANDMARK_SEPARATION_RATIO * palm_length
    for first in range(len(landmarks)):
        for second in range(first + 1, len(landmarks)):
            name_a, point_a = landmarks[first]
            name_b, point_b = landmarks[second]
            separation = float(np.linalg.norm(point_a - point_b))
            if separation < minimum_separation:
                return None, [
                    f"{FLAG_PALM_COINCIDENT}_{name_a}_{name_b}"
                    f"_sep_over_palm={separation / palm_length:.3e}"
                ]

    if track == "left":
        lateral_raw = pinky_mcp - index_mcp
    else:
        lateral_raw = index_mcp - pinky_mcp
    if float(np.linalg.norm(lateral_raw)) < MIN_BONE_LENGTH:
        return None, [FLAG_PALM_ZERO_AXIS]

    normal = safe_normalize(np.cross(distal, lateral_raw), minimum=MIN_FRAME_CROSS_NORM)
    if normal is None:
        # distal and lateral_raw are parallel: the four landmarks lie on a line
        # and no plane is defined.
        return None, [FLAG_PALM_COLLINEAR]

    lateral = safe_normalize(np.cross(normal, distal))
    if lateral is None:
        return None, [FLAG_PALM_COLLINEAR]

    rotation = np.column_stack((lateral, normal, distal))
    if not np.isfinite(rotation).all():
        return None, [FLAG_PALM_COLLINEAR]

    return (
        PalmFrame(
            rotation=rotation,
            origin=wrist.copy(),
            lateral=lateral,
            normal=normal,
            distal=distal,
        ),
        flags,
    )
