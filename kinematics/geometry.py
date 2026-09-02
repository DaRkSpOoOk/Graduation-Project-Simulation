"""Primitive geometry for hand kinematics.

Everything here operates on 3D joint positions. MANO local rotation matrices
are deliberately NOT used to derive any output channel.

Reason: WiLoR reconstructs every hand with the canonical MANO *right*-hand
model and then mirrors the x axis of the joints for a left hand
(``pose/wilor/frame_extraction.py``: ``joints[:, 0] = (2*is_right - 1) *
joints[:, 0]``). The exported 3D joints are therefore true left-hand geometry
in camera space, while ``hand_pose_rotmat`` remains in the un-mirrored
canonical-right convention. Reading anatomical flexion off those rotation
matrices would require inverting an undocumented mirroring, so all kinematics
below are derived from the joints, whose physical meaning is demonstrable.

A useful consequence: every quantity built purely from dot products of joint
differences is invariant under any orthogonal transform, reflections included.
That is what makes a mirrored LEFT/RIGHT pair produce identical flexion and
spread without a single sign flip.
"""

from __future__ import annotations

import numpy as np

# Below this length (in MANO/metre units) a bone is treated as degenerate.
# Real pilot bones are 0.019-0.095 m, so this is two orders of magnitude
# clear of anything genuine.
MIN_BONE_LENGTH = 1e-6

# Conditioning limit for the palm-plane projection used by spread.
#
# The projected norm equals sin(theta), where theta is the angle between the
# finger's proximal phalanx and the palm normal. A finger pointing along the
# normal has NO direction within the palm plane, so its spread is undefined;
# near that pole the projected direction amplifies any angular error in the
# input by 1/sin(theta), and the surviving component is dominated by
# reconstruction noise rather than by anatomy.
#
# The limit is set a priori, by capping that amplification at ~3.9x, rather
# than fitted to the pilot: theta >= 15 degrees. Unlike the TASK-004D
# cross-label rule there is NO natural gap in the observed distribution to
# separate -- it is a smooth continuum (see the TASK-005A report for the
# histogram and a sensitivity table across 5-30 degrees). The threshold is
# therefore a stated conditioning choice, not a discovered boundary.
MIN_SPREAD_PROJECTION_ANGLE_DEG = 15.0
MIN_PROJECTED_NORM = float(np.sin(np.radians(MIN_SPREAD_PROJECTION_ANGLE_DEG)))

# Minimum third-axis magnitude when building the palm frame; below this the
# frame-defining vectors are treated as collinear.
MIN_FRAME_CROSS_NORM = 1e-6

# Minimum separation between any two palm-frame-defining landmarks, expressed
# as a fraction of the palm length ||middle_MCP - wrist||.
#
# Two coincident landmarks cannot define a palm, but the three-vector frame
# construction does not necessarily notice: if the index and middle MCPs are
# the same point, ``middle_MCP - wrist`` and ``index_MCP - pinky_MCP`` are both
# still non-zero and non-parallel, so a fully finite orthonormal frame comes
# out of a palm that has collapsed. The landmarks are therefore checked for
# distinctness directly, rather than inferred from the axes they produce.
#
# The bound is a numerical-degeneracy limit, not an anatomical one. Joints are
# stored as float32, whose relative precision is ~1.2e-7, so a difference of
# two coordinates carries rounding of order 2.4e-7 relative to their
# magnitude. 1e-3 keeps roughly 4200x that noise floor, so a surviving
# separation is real signal rather than storage error. It is scale-relative,
# so it is unaffected by hand size or units.
MIN_PALM_LANDMARK_SEPARATION_RATIO = 1e-3

ORTHONORMAL_TOLERANCE = 1e-5


def safe_normalize(vector: np.ndarray, minimum: float = MIN_BONE_LENGTH) -> np.ndarray | None:
    """Unit vector, or None when the input is non-finite or too short."""

    vector = np.asarray(vector, dtype=np.float64)
    if not np.isfinite(vector).all():
        return None
    norm = float(np.linalg.norm(vector))
    if norm < minimum:
        return None
    return vector / norm


def angle_between_unit(first: np.ndarray, second: np.ndarray) -> float:
    """Angle in degrees between two unit vectors, in [0, 180].

    Uses ``atan2(|a x b|, a . b)`` rather than ``arccos(a . b)``. The arccos
    form is ill-conditioned exactly where hand kinematics spends most of its
    time: its derivative is unbounded as the dot product approaches +-1, so a
    perfectly straight finger reads a spurious ~1e-6 degrees instead of 0. The
    atan2 form is well conditioned across the whole range and returns exact
    zero for exactly-collinear input.
    """

    cross = float(np.linalg.norm(np.cross(first, second)))
    dot = float(np.dot(first, second))
    return float(np.degrees(np.arctan2(cross, dot)))


def turn_angle(previous_point: np.ndarray, joint: np.ndarray, next_point: np.ndarray) -> float | None:
    """Bend angle at ``joint`` of the polyline previous -> joint -> next.

    Defined as the angle between the *incoming bone direction* and the
    *outgoing bone direction*:

        b_in  = joint - previous_point
        b_out = next_point - joint
        angle = arccos( b_in_hat . b_out_hat )

    A straight (collinear, same-direction) chain gives exactly 0 degrees, and
    the value grows monotonically to 180 degrees as the chain folds back on
    itself. It is unsigned by construction, so it satisfies the required
    convention -- straight is 0, more bend is more positive -- on both hands
    with no handedness correction.

    Returns None if either bone is degenerate or non-finite.
    """

    incoming = safe_normalize(np.asarray(joint, dtype=np.float64) - np.asarray(previous_point, dtype=np.float64))
    outgoing = safe_normalize(np.asarray(next_point, dtype=np.float64) - np.asarray(joint, dtype=np.float64))
    if incoming is None or outgoing is None:
        return None
    return angle_between_unit(incoming, outgoing)


def project_onto_plane(vector: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Component of ``vector`` perpendicular to unit ``normal``."""

    vector = np.asarray(vector, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)
    return vector - float(np.dot(vector, normal)) * normal


def is_orthonormal(matrix: np.ndarray, tolerance: float = ORTHONORMAL_TOLERANCE) -> bool:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        return False
    return bool(np.allclose(matrix.T @ matrix, np.eye(3), atol=tolerance))


def orthonormality_error(matrix: np.ndarray) -> float:
    """max |R^T R - I|, a scalar diagnostic."""

    matrix = np.asarray(matrix, dtype=np.float64)
    return float(np.max(np.abs(matrix.T @ matrix - np.eye(3))))


def rotation_matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray | None:
    """Convert a proper rotation matrix to a normalized [w, x, y, z] quaternion.

    Uses Shepperd's method: pick the branch whose denominator is largest so the
    square root is never taken of a near-zero quantity. The sign is then fixed
    so that w >= 0, which makes the output deterministic (q and -q are the same
    rotation, so an unconstrained solver could return either).

    Returns None if the matrix is not a finite proper rotation, or if the
    resulting quaternion cannot be normalized.
    """

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        return None

    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / scale
        x = 0.25 * scale
        y = (matrix[0, 1] + matrix[1, 0]) / scale
        z = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / scale
        x = (matrix[0, 1] + matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / scale
        x = (matrix[0, 2] + matrix[2, 0]) / scale
        y = (matrix[1, 2] + matrix[2, 1]) / scale
        z = 0.25 * scale

    quaternion = np.array([w, x, y, z], dtype=np.float64)
    if not np.isfinite(quaternion).all():
        return None
    norm = float(np.linalg.norm(quaternion))
    if norm < MIN_BONE_LENGTH:
        return None
    quaternion = quaternion / norm
    if quaternion[0] < 0.0:
        quaternion = -quaternion
    return quaternion
