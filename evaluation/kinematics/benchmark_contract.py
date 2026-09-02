"""Extractor-neutral TASK-005 kinematics output contract.

The validator accepts an adapter's in-memory result and checks shape, finite
values, rotation validity, and quaternion/matrix consistency.  It does not
compute production kinematics from joints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


TRACK_ORDER = ("LEFT", "RIGHT")
FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")
CONTRACT_TOLERANCES = {
    "known_flexion_abs_error_deg": 1.0,
    "known_spread_abs_error_deg": 1.0,
    "rotation_matrix_orthogonality": 1e-5,
    "rotation_matrix_determinant": 1e-5,
    "quaternion_norm": 1e-5,
    "matrix_quaternion_consistency": 1e-5,
    "known_orientation_error_deg": 1.0,
}


class ContractError(ValueError):
    """Raised when a production adapter violates the kinematics contract."""


@dataclass(frozen=True)
class KinematicsResult:
    """The exact result shape TASK-005A must expose to TASK-005D."""

    flexion_deg: np.ndarray
    adjacent_spread_deg: np.ndarray
    palm_rotation_matrix: np.ndarray
    palm_quaternion_wxyz: np.ndarray

    @classmethod
    def from_mapping(cls, result: Mapping[str, object]) -> "KinematicsResult":
        required = (
            "flexion_deg",
            "adjacent_spread_deg",
            "palm_rotation_matrix",
            "palm_quaternion_wxyz",
        )
        missing = [name for name in required if name not in result]
        if missing:
            raise ContractError(f"missing kinematics result fields: {missing}")
        return cls(*(np.asarray(result[name], dtype=np.float64) for name in required))


def coerce_result(result: KinematicsResult | Mapping[str, object]) -> KinematicsResult:
    """Convert the later production adapter's mapping into this contract."""

    if isinstance(result, KinematicsResult):
        return result
    if isinstance(result, Mapping):
        return KinematicsResult.from_mapping(result)
    raise ContractError("result must be KinematicsResult or a mapping with contract fields")


def _require_shape(array: np.ndarray, expected: tuple[int | None, ...], name: str) -> None:
    if array.ndim != len(expected):
        raise ContractError(f"{name} must have rank {len(expected)}, got {array.ndim}")
    for actual, wanted in zip(array.shape, expected):
        if wanted is not None and actual != wanted:
            raise ContractError(f"{name} must have shape {expected}, got {array.shape}")


def quaternion_matrix_wxyz(quaternion: object) -> np.ndarray:
    """Convert a [w,x,y,z] quaternion into a rotation matrix for validation."""

    values = np.asarray(quaternion, dtype=np.float64)
    if values.shape != (4,) or not np.all(np.isfinite(values)):
        raise ContractError("quaternion must be a finite vector with shape (4,)")
    norm = float(np.linalg.norm(values))
    if norm <= 0.0:
        raise ContractError("quaternion norm must be positive")
    w, x, y, z = values / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def validate_result(
    result: KinematicsResult | Mapping[str, object],
    *,
    expected_frames: int | None = None,
) -> KinematicsResult:
    """Hard-validate a production result before benchmark scoring."""

    validated = coerce_result(result)
    flexion = np.asarray(validated.flexion_deg, dtype=np.float64)
    spread = np.asarray(validated.adjacent_spread_deg, dtype=np.float64)
    matrices = np.asarray(validated.palm_rotation_matrix, dtype=np.float64)
    quaternions = np.asarray(validated.palm_quaternion_wxyz, dtype=np.float64)
    frame_count = expected_frames
    if frame_count is None and flexion.ndim >= 1:
        frame_count = int(flexion.shape[0])
    if frame_count is not None and frame_count < 1:
        raise ContractError("kinematics result must contain at least one frame")
    _require_shape(flexion, (frame_count, 2, 5, 3), "flexion_deg")
    _require_shape(spread, (frame_count, 2, 4), "adjacent_spread_deg")
    _require_shape(matrices, (frame_count, 2, 3, 3), "palm_rotation_matrix")
    _require_shape(quaternions, (frame_count, 2, 4), "palm_quaternion_wxyz")
    arrays = (
        (flexion, "flexion_deg"),
        (spread, "adjacent_spread_deg"),
        (matrices, "palm_rotation_matrix"),
        (quaternions, "palm_quaternion_wxyz"),
    )
    for array, name in arrays:
        if not np.all(np.isfinite(array)):
            raise ContractError(f"{name} contains non-finite values")

    matrix_tolerance = CONTRACT_TOLERANCES["rotation_matrix_orthogonality"]
    determinant_tolerance = CONTRACT_TOLERANCES["rotation_matrix_determinant"]
    quaternion_tolerance = CONTRACT_TOLERANCES["quaternion_norm"]
    consistency_tolerance = CONTRACT_TOLERANCES["matrix_quaternion_consistency"]
    identity = np.eye(3)
    for frame in range(int(frame_count)):
        for track in range(2):
            matrix = matrices[frame, track]
            orthogonality_error = float(np.max(np.abs(matrix.T @ matrix - identity)))
            if orthogonality_error > matrix_tolerance:
                raise ContractError(
                    f"palm_rotation_matrix[{frame},{track}] is not orthogonal: {orthogonality_error:g}"
                )
            determinant_error = abs(float(np.linalg.det(matrix)) - 1.0)
            if determinant_error > determinant_tolerance:
                raise ContractError(
                    f"palm_rotation_matrix[{frame},{track}] determinant is invalid: {determinant_error:g}"
                )
            quaternion = quaternions[frame, track]
            norm_error = abs(float(np.linalg.norm(quaternion)) - 1.0)
            if norm_error > quaternion_tolerance:
                raise ContractError(
                    f"palm_quaternion_wxyz[{frame},{track}] norm is invalid: {norm_error:g}"
                )
            converted = quaternion_matrix_wxyz(quaternion)
            consistency_error = float(np.max(np.abs(matrix - converted)))
            if consistency_error > consistency_tolerance:
                raise ContractError(
                    f"matrix/quaternion disagreement at [{frame},{track}]: {consistency_error:g}"
                )
    return KinematicsResult(flexion, spread, matrices, quaternions)
