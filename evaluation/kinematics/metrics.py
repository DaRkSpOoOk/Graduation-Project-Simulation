"""Independent scoring helpers for synthetic TASK-005B cases."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from .benchmark_contract import CONTRACT_TOLERANCES, ContractError, KinematicsResult, coerce_result, validate_result
from .synthetic_hand import SyntheticSequence


def rotation_angle_error_deg(predicted: np.ndarray, expected: np.ndarray) -> float:
    """Return the geodesic angle between two proper rotation matrices."""

    relative = np.asarray(predicted, dtype=np.float64).T @ np.asarray(expected, dtype=np.float64)
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def quaternion_angle_error_deg(predicted: np.ndarray, expected: np.ndarray) -> float:
    """Return sign-invariant angular error for [w,x,y,z] quaternions."""

    first = np.asarray(predicted, dtype=np.float64)
    second = np.asarray(expected, dtype=np.float64)
    first = first / np.linalg.norm(first)
    second = second / np.linalg.norm(second)
    cosine = float(np.clip(abs(np.dot(first, second)), -1.0, 1.0))
    return math.degrees(2.0 * math.acos(cosine))


def evaluate_sequence(
    result: KinematicsResult | Mapping[str, object],
    expected: SyntheticSequence,
) -> dict[str, object]:
    """Validate and compare an adapter result against analytic fixture truth."""

    if not expected.expected_valid:
        raise ContractError(
            f"{expected.case_id} intentionally has invalid geometry: {expected.invalid_reasons}; "
            "a production adapter must reject it before numeric scoring"
        )
    validated = validate_result(result, expected_frames=expected.joints.shape[0])
    flexion_error = np.abs(validated.flexion_deg - expected.flexion_deg)
    spread_error = np.abs(validated.adjacent_spread_deg - expected.adjacent_spread_deg)
    rotation_errors = np.asarray(
        [
            rotation_angle_error_deg(validated.palm_rotation_matrix[frame, track], expected.palm_rotation_matrix[frame, track])
            for frame in range(expected.joints.shape[0])
            for track in range(2)
        ]
    )
    quaternion_errors = np.asarray(
        [
            quaternion_angle_error_deg(validated.palm_quaternion_wxyz[frame, track], expected.palm_quaternion_wxyz[frame, track])
            for frame in range(expected.joints.shape[0])
            for track in range(2)
        ]
    )
    flexion_limit = CONTRACT_TOLERANCES["known_flexion_abs_error_deg"]
    spread_limit = CONTRACT_TOLERANCES["known_spread_abs_error_deg"]
    orientation_limit = CONTRACT_TOLERANCES["known_orientation_error_deg"]
    return {
        "case_id": expected.case_id,
        "frames": int(expected.joints.shape[0]),
        "max_flexion_error_deg": float(np.max(flexion_error)),
        "max_spread_error_deg": float(np.max(spread_error)),
        "max_orientation_error_deg": float(np.max(rotation_errors)),
        "max_quaternion_error_deg": float(np.max(quaternion_errors)),
        "flexion_pass": bool(np.all(flexion_error <= flexion_limit)),
        "spread_pass": bool(np.all(spread_error <= spread_limit)),
        "orientation_pass": bool(np.all(rotation_errors <= orientation_limit)),
        "quaternion_pass": bool(np.all(quaternion_errors <= orientation_limit)),
    }
