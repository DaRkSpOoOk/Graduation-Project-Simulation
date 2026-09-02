"""Independent rotation and quaternion helpers for TASK-006B.

These helpers are a benchmark oracle, not a production sensor-conversion
implementation.  They intentionally do not import the TASK-005 kinematics
package or any future virtual-glove implementation.
"""

from __future__ import annotations

import math

import numpy as np


ROTATION_ORTHOGONALITY_TOLERANCE = 1e-10
ROTATION_DETERMINANT_TOLERANCE = 1e-10
QUATERNION_NORM_TOLERANCE = 1e-10
MATRIX_QUATERNION_TOLERANCE = 1e-10
ORIENTATION_ANGLE_TOLERANCE_DEG = 1e-10
GYROSCOPE_TOLERANCE_RAD_PER_SECOND = 1e-10


def rotation_matrix_axis(axis: str, degrees: float) -> np.ndarray:
    """Return an exact right-handed principal-axis rotation matrix."""

    radians = math.radians(float(degrees))
    cosine = math.cos(radians)
    sine = math.sin(radians)
    normalized = str(axis).upper()
    if normalized == "X":
        return np.array(
            [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
            dtype=np.float64,
        )
    if normalized == "Y":
        return np.array(
            [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
            dtype=np.float64,
        )
    if normalized == "Z":
        return np.array(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    raise ValueError(f"unknown rotation axis: {axis!r}")


def rotation_matrix_xyz(x_deg: float, y_deg: float, z_deg: float) -> np.ndarray:
    """Compose column-vector rotations as ``Rz @ Ry @ Rx``."""

    return (
        rotation_matrix_axis("Z", z_deg)
        @ rotation_matrix_axis("Y", y_deg)
        @ rotation_matrix_axis("X", x_deg)
    )


def validate_rotation_matrix(
    matrix: object,
    *,
    orthogonality_tolerance: float = ROTATION_ORTHOGONALITY_TOLERANCE,
    determinant_tolerance: float = ROTATION_DETERMINANT_TOLERANCE,
) -> tuple[bool, str]:
    """Validate a finite proper rotation without repairing it."""

    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (3, 3):
        return False, f"rotation_shape={values.shape}"
    if not np.isfinite(values).all():
        return False, "rotation_non_finite"
    orthogonality_error = float(np.max(np.abs(values.T @ values - np.eye(3))))
    if orthogonality_error > orthogonality_tolerance:
        return False, f"rotation_not_orthonormal={orthogonality_error:.3e}"
    determinant_error = abs(float(np.linalg.det(values)) - 1.0)
    if determinant_error > determinant_tolerance:
        return False, f"rotation_determinant={float(np.linalg.det(values)):.12g}"
    return True, ""


def quaternion_wxyz_from_matrix(matrix: object) -> np.ndarray:
    """Convert a proper matrix to a deterministic normalized WXYZ quaternion."""

    values = np.asarray(matrix, dtype=np.float64)
    valid, reason = validate_rotation_matrix(values)
    if not valid:
        raise ValueError(reason)

    trace = float(np.trace(values))
    quaternion = np.empty(4, dtype=np.float64)
    if trace > 0.0:
        root = math.sqrt(trace + 1.0)
        quaternion[0] = 0.5 * root
        scale = 0.5 / root
        quaternion[1] = (values[2, 1] - values[1, 2]) * scale
        quaternion[2] = (values[0, 2] - values[2, 0]) * scale
        quaternion[3] = (values[1, 0] - values[0, 1]) * scale
    else:
        diagonal = np.diag(values)
        index = int(np.argmax(diagonal))
        if index == 0:
            root = math.sqrt(max(0.0, 1.0 + values[0, 0] - values[1, 1] - values[2, 2]))
            quaternion[1] = 0.5 * root
            scale = 0.5 / root if root > 0.0 else 0.0
            quaternion[0] = (values[2, 1] - values[1, 2]) * scale
            quaternion[2] = (values[0, 1] + values[1, 0]) * scale
            quaternion[3] = (values[0, 2] + values[2, 0]) * scale
        elif index == 1:
            root = math.sqrt(max(0.0, 1.0 - values[0, 0] + values[1, 1] - values[2, 2]))
            quaternion[2] = 0.5 * root
            scale = 0.5 / root if root > 0.0 else 0.0
            quaternion[0] = (values[0, 2] - values[2, 0]) * scale
            quaternion[1] = (values[0, 1] + values[1, 0]) * scale
            quaternion[3] = (values[1, 2] + values[2, 1]) * scale
        else:
            root = math.sqrt(max(0.0, 1.0 - values[0, 0] - values[1, 1] + values[2, 2]))
            quaternion[3] = 0.5 * root
            scale = 0.5 / root if root > 0.0 else 0.0
            quaternion[0] = (values[1, 0] - values[0, 1]) * scale
            quaternion[1] = (values[0, 2] + values[2, 0]) * scale
            quaternion[2] = (values[1, 2] + values[2, 1]) * scale

    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("rotation produced a zero or non-finite quaternion")
    quaternion /= norm
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return quaternion


def quaternion_matrix_wxyz(quaternion: object) -> np.ndarray:
    """Convert a nonzero WXYZ quaternion to a rotation matrix."""

    values = np.asarray(quaternion, dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("quaternion must be a finite shape-(4,) vector")
    norm = float(np.linalg.norm(values))
    if norm <= 0.0:
        raise ValueError("quaternion cannot have zero norm")
    w, x, y, z = values / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def validate_quaternion(
    quaternion: object,
    *,
    norm_tolerance: float = QUATERNION_NORM_TOLERANCE,
) -> tuple[bool, str]:
    """Validate finite normalized WXYZ quaternion data without normalization."""

    values = np.asarray(quaternion, dtype=np.float64)
    if values.shape != (4,):
        return False, f"quaternion_shape={values.shape}"
    if not np.isfinite(values).all():
        return False, "quaternion_non_finite"
    norm_error = abs(float(np.linalg.norm(values)) - 1.0)
    if norm_error > norm_tolerance:
        return False, f"quaternion_norm_error={norm_error:.3e}"
    return True, ""


def quaternions_equivalent(first: object, second: object, tolerance: float = 1e-10) -> bool:
    """Return whether two normalized quaternions represent the same rotation."""

    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != (4,) or right.shape != (4,):
        return False
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        return False
    left /= np.linalg.norm(left)
    right /= np.linalg.norm(right)
    return bool(abs(abs(float(np.dot(left, right))) - 1.0) <= tolerance)


def quaternion_angular_distance_deg(first: object, second: object) -> float:
    """Return the shortest angular distance between two WXYZ rotations."""

    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != (4,) or right.shape != (4,):
        raise ValueError("quaternions must have shape (4,)")
    left /= np.linalg.norm(left)
    right /= np.linalg.norm(right)
    dot = min(1.0, max(-1.0, abs(float(np.dot(left, right)))))
    return float(math.degrees(2.0 * math.acos(dot)))


def rotation_angular_distance_deg(first: object, second: object) -> float:
    """Return the shortest angular distance between two proper rotations."""

    relative = np.asarray(first, dtype=np.float64).T @ np.asarray(second, dtype=np.float64)
    cosine = min(1.0, max(-1.0, (float(np.trace(relative)) - 1.0) / 2.0))
    return float(math.degrees(math.acos(cosine)))


def quaternion_multiply_wxyz(first: object, second: object) -> np.ndarray:
    """Hamilton product for WXYZ quaternions."""

    w1, x1, y1, z1 = np.asarray(first, dtype=np.float64)
    w2, x2, y2, z2 = np.asarray(second, dtype=np.float64)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quaternion_conjugate_wxyz(quaternion: object) -> np.ndarray:
    """Return the WXYZ quaternion conjugate."""

    values = np.asarray(quaternion, dtype=np.float64)
    if values.shape != (4,):
        raise ValueError("quaternion must have shape (4,)")
    return np.array([values[0], -values[1], -values[2], -values[3]], dtype=np.float64)


def angular_velocity_from_quaternions(
    first: object,
    second: object,
    delta_seconds: float,
) -> np.ndarray:
    """Compute the legacy WORLD-FRAME rotation-vector velocity in rad/s.

    The relative rotation is ``q_second * conjugate(q_first)``.  The returned
    vector is the axis of ``R_second @ R_first.T`` in world/camera axes.  This
    helper is retained for TASK-006B historical compatibility; the final
    TASK-006 gyro contract uses :func:`angular_velocity_body_frame_from_quaternions`.
    """

    if not math.isfinite(float(delta_seconds)) or float(delta_seconds) <= 0.0:
        raise ValueError("delta_seconds must be finite and positive")
    first_values = np.asarray(first, dtype=np.float64)
    second_values = np.asarray(second, dtype=np.float64)
    valid_first, reason_first = validate_quaternion(first_values)
    valid_second, reason_second = validate_quaternion(second_values)
    if not valid_first:
        raise ValueError(reason_first)
    if not valid_second:
        raise ValueError(reason_second)
    delta = quaternion_multiply_wxyz(second_values, quaternion_conjugate_wxyz(first_values))
    delta /= np.linalg.norm(delta)
    if delta[0] < 0.0:
        delta *= -1.0
    vector = delta[1:]
    vector_norm = float(np.linalg.norm(vector))
    angle = 2.0 * math.atan2(vector_norm, max(-1.0, min(1.0, float(delta[0]))))
    if vector_norm <= 1e-14 or angle <= 1e-14:
        return np.zeros(3, dtype=np.float64)
    return vector / vector_norm * (angle / float(delta_seconds))


def angular_velocity_body_frame_from_quaternions(
    first: object,
    second: object,
    delta_seconds: float,
) -> np.ndarray:
    """Compute BODY-FRAME angular velocity in rad/s.

    For column-vector rotation matrices, ``conjugate(q_first) * q_second``
    represents ``R_first.T @ R_second``.  Its axis is expressed in the earlier
    sample's body axes, matching a palm-mounted gyroscope and the final
    TASK-006 contract.
    """

    if not math.isfinite(float(delta_seconds)) or float(delta_seconds) <= 0.0:
        raise ValueError("delta_seconds must be finite and positive")
    first_values = np.asarray(first, dtype=np.float64)
    second_values = np.asarray(second, dtype=np.float64)
    valid_first, reason_first = validate_quaternion(first_values)
    valid_second, reason_second = validate_quaternion(second_values)
    if not valid_first:
        raise ValueError(reason_first)
    if not valid_second:
        raise ValueError(reason_second)
    delta = quaternion_multiply_wxyz(
        quaternion_conjugate_wxyz(first_values), second_values
    )
    delta /= np.linalg.norm(delta)
    if delta[0] < 0.0:
        delta *= -1.0
    vector = delta[1:]
    vector_norm = float(np.linalg.norm(vector))
    angle = 2.0 * math.atan2(vector_norm, max(-1.0, min(1.0, float(delta[0]))))
    if vector_norm <= 1e-14 or angle <= 1e-14:
        return np.zeros(3, dtype=np.float64)
    return vector / vector_norm * (angle / float(delta_seconds))


__all__ = [
    "GYROSCOPE_TOLERANCE_RAD_PER_SECOND",
    "MATRIX_QUATERNION_TOLERANCE",
    "ORIENTATION_ANGLE_TOLERANCE_DEG",
    "QUATERNION_NORM_TOLERANCE",
    "ROTATION_DETERMINANT_TOLERANCE",
    "ROTATION_ORTHOGONALITY_TOLERANCE",
    "angular_velocity_from_quaternions",
    "angular_velocity_body_frame_from_quaternions",
    "quaternion_angular_distance_deg",
    "quaternion_conjugate_wxyz",
    "quaternion_matrix_wxyz",
    "quaternion_multiply_wxyz",
    "quaternion_wxyz_from_matrix",
    "quaternions_equivalent",
    "rotation_angular_distance_deg",
    "rotation_matrix_axis",
    "rotation_matrix_xyz",
    "validate_quaternion",
    "validate_rotation_matrix",
]
