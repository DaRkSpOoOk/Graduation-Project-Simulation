"""Independent ideal sensor-model oracle for TASK-006B.

The functions here are deliberately small reference mathematics for tests and
future adapter comparison.  They are not the production conversion stage and
must not be imported by a production virtual-glove pipeline.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .contract import (
    ADC_MAX_COUNT,
    ANGLE_MAX_DEG,
    ANGLE_MIN_DEG,
    HALL_PER_HAND,
    IdealSensorOutput,
    InputContractError,
    KinematicsInput,
    SensorDefinition,
    validate_sensor_catalog,
    build_sensor_catalog,
    validate_kinematics_input,
)
from .orientation import (
    angular_velocity_from_quaternions,
    quaternion_angular_distance_deg,
)


def normalize_angle_deg(angle_deg: float) -> float:
    """Map one in-contract degree value to the authoritative [0, 1] signal.

    Out-of-range and non-finite values are rejected rather than clipped.
    """

    value = float(angle_deg)
    if not math.isfinite(value):
        raise InputContractError("angle_non_finite")
    if value < ANGLE_MIN_DEG or value > ANGLE_MAX_DEG:
        raise InputContractError(f"angle_outside_0_180={value!r}")
    return value / ANGLE_MAX_DEG


def normalized_to_adc_count(normalized: float) -> int:
    """Optionally map a valid normalized value to 12-bit ADC counts.

    Rounding is deterministic round-half-up for non-negative values.  This is
    an optional compatibility representation; normalized floats remain the
    authoritative signal.
    """

    value = float(normalized)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise InputContractError("normalized_outside_0_1")
    return int(math.floor(value * ADC_MAX_COUNT + 0.5))


def hall_adc_counts(
    normalized: object,
    valid: object,
) -> np.ndarray:
    """Return optional ADC counts, using -1 only for invalid channels."""

    values = np.asarray(normalized, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if values.shape != mask.shape:
        raise ValueError(f"normalized and valid shapes differ: {values.shape} vs {mask.shape}")
    counts = np.full(values.shape, -1, dtype=np.int16)
    for index in np.ndindex(values.shape):
        if mask[index]:
            counts[index] = normalized_to_adc_count(float(values[index]))
        elif np.isfinite(values[index]):
            raise InputContractError(f"invalid_channel_has_finite_normalized_value={index}")
    return counts


def reference_ideal_sensor_model(
    data: KinematicsInput,
    catalog: tuple[SensorDefinition, ...] | None = None,
) -> IdealSensorOutput:
    """Convert validated kinematics into the independent ideal sensor oracle.

    Bends and spreads are mapped independently.  A valid palm frame passes its
    matrix and WXYZ quaternion through unchanged to the palm IMU representation;
    no TASK-005 synthetic comparison mapping is applied.
    """

    validate_kinematics_input(data)
    sensor_catalog = build_sensor_catalog() if catalog is None else tuple(catalog)
    catalog_validation = validate_sensor_catalog(sensor_catalog)
    if not catalog_validation["passed"]:
        raise ValueError("invalid sensor catalog: " + ", ".join(catalog_validation["errors"]))

    frames = int(np.asarray(data.frame_index).shape[0])
    hall_normalized = np.full((frames, 2, HALL_PER_HAND), np.nan, dtype=np.float64)
    hall_valid = np.zeros((frames, 2, HALL_PER_HAND), dtype=bool)
    imu_rotation = np.full((frames, 2, 3, 3), np.nan, dtype=np.float64)
    imu_quaternion = np.full((frames, 2, 4), np.nan, dtype=np.float64)
    imu_valid = np.zeros((frames, 2), dtype=bool)

    # Catalog layout is deterministic: 19 Hall definitions then one IMU per
    # hand. The dictionary check above still makes an adapter safe if a caller
    # passes a reordered but complete catalog in future.
    for hand_index, hand in enumerate(("LEFT", "RIGHT")):
        hand_hall = [
            definition
            for definition in sensor_catalog
            if definition.hand == hand and definition.kind == "HALL"
        ]
        if len(hand_hall) != HALL_PER_HAND:
            raise ValueError(f"{hand} must have {HALL_PER_HAND} Hall definitions")
        for hall_index, definition in enumerate(hand_hall):
            if definition.channel == "bend":
                _, finger, joint = definition.source
                finger_index = ("thumb", "index", "middle", "ring", "pinky").index(finger)
                joint_index = ("proximal", "middle", "distal").index(joint)
                source_values = np.asarray(data.flexion_deg, dtype=np.float64)[:, hand_index, finger_index, joint_index]
            elif definition.channel == "spread":
                _, spread = definition.source
                spread_index = (
                    "thumb-index",
                    "index-middle",
                    "middle-ring",
                    "ring-pinky",
                ).index(spread)
                source_values = np.asarray(data.adjacent_spread_deg, dtype=np.float64)[:, hand_index, spread_index]
            else:
                raise ValueError(f"unexpected Hall channel {definition.channel!r}")
            for frame, value in enumerate(source_values):
                if bool(data.hand_present[frame, hand_index]) and np.isfinite(value):
                    hall_normalized[frame, hand_index, hall_index] = normalize_angle_deg(float(value))
                    hall_valid[frame, hand_index, hall_index] = True

        if bool(data.hand_present[:, hand_index].any()):
            palm_valid = np.asarray(data.valid_palm_frame[:, hand_index], dtype=bool)
            present = np.asarray(data.hand_present[:, hand_index], dtype=bool)
            imu_valid[:, hand_index] = present & palm_valid
            imu_rotation[imu_valid[:, hand_index]] = np.asarray(data.palm_rotation_matrix)[
                imu_valid[:, hand_index]
            ]
            imu_quaternion[imu_valid[:, hand_index]] = np.asarray(data.palm_quaternion_wxyz)[
                imu_valid[:, hand_index]
            ]

    return IdealSensorOutput(
        sensor_catalog=sensor_catalog,
        hall_normalized=hall_normalized,
        hall_valid=hall_valid,
        palm_imu_rotation_matrix=imu_rotation,
        palm_imu_quaternion_wxyz=imu_quaternion,
        palm_imu_valid=imu_valid,
    )


def compare_sensor_outputs(
    actual: IdealSensorOutput,
    expected: IdealSensorOutput,
    *,
    angle_tolerance: float = 1e-10,
    orientation_tolerance_deg: float = 1e-10,
) -> dict[str, Any]:
    """Compare a future adapter result with the independent oracle.

    Quaternion comparison uses rotation equivalence, so q and -q are treated
    as the same orientation while the WXYZ shape/order remains required.
    """

    errors: list[str] = []
    if tuple(actual.sensor_catalog) != tuple(expected.sensor_catalog):
        errors.append("sensor_catalog_mismatch")
    if actual.hall_normalized.shape != expected.hall_normalized.shape:
        errors.append("hall_shape_mismatch")
    if actual.hall_valid.shape != expected.hall_valid.shape:
        errors.append("hall_mask_shape_mismatch")
    if actual.palm_imu_rotation_matrix.shape != expected.palm_imu_rotation_matrix.shape:
        errors.append("imu_rotation_shape_mismatch")
    if actual.palm_imu_quaternion_wxyz.shape != expected.palm_imu_quaternion_wxyz.shape:
        errors.append("imu_quaternion_shape_mismatch")

    hall_mask_mismatches = 0
    hall_errors: list[float] = []
    if actual.hall_valid.shape == expected.hall_valid.shape:
        hall_mask_mismatches = int(np.count_nonzero(actual.hall_valid != expected.hall_valid))
        if actual.hall_normalized.shape == expected.hall_normalized.shape:
            mask = expected.hall_valid & actual.hall_valid
            hall_errors = np.abs(actual.hall_normalized[mask] - expected.hall_normalized[mask]).tolist()
            if np.any(~np.isfinite(actual.hall_normalized[mask])):
                errors.append("finite_expected_hall_is_non_finite")
            if np.any(np.isfinite(actual.hall_normalized[~expected.hall_valid])):
                errors.append("invalid_expected_hall_is_finite")
            if np.any(np.asarray(hall_errors) > angle_tolerance):
                errors.append("hall_value_mismatch")
        if hall_mask_mismatches:
            errors.append("hall_validity_mask_mismatch")

    imu_mask_mismatches = 0
    if actual.palm_imu_valid.shape == expected.palm_imu_valid.shape:
        imu_mask_mismatches = int(np.count_nonzero(actual.palm_imu_valid != expected.palm_imu_valid))
    if imu_mask_mismatches:
        errors.append("imu_validity_mask_mismatch")
    matrix_errors: list[float] = []
    quaternion_errors: list[float] = []
    if (
        actual.palm_imu_rotation_matrix.shape == expected.palm_imu_rotation_matrix.shape
        and actual.palm_imu_quaternion_wxyz.shape == expected.palm_imu_quaternion_wxyz.shape
    ):
        for index in np.ndindex(expected.palm_imu_valid.shape):
            if not expected.palm_imu_valid[index]:
                continue
            matrix_errors.append(
                float(
                    np.max(
                        np.abs(
                            actual.palm_imu_rotation_matrix[index]
                            - expected.palm_imu_rotation_matrix[index]
                        )
                    )
                )
            )
            if matrix_errors[-1] > angle_tolerance:
                errors.append("imu_rotation_value_mismatch")
            quaternion_errors.append(
                quaternion_angular_distance_deg(
                    actual.palm_imu_quaternion_wxyz[index],
                    expected.palm_imu_quaternion_wxyz[index],
                )
            )
            if quaternion_errors[-1] > orientation_tolerance_deg:
                errors.append("imu_quaternion_value_mismatch")
        if np.any(np.isfinite(actual.palm_imu_rotation_matrix[~expected.palm_imu_valid])):
            errors.append("invalid_expected_imu_rotation_is_finite")
        if np.any(np.isfinite(actual.palm_imu_quaternion_wxyz[~expected.palm_imu_valid])):
            errors.append("invalid_expected_imu_quaternion_is_finite")

    return {
        "passed": not errors,
        "errors": sorted(set(errors)),
        "hall_mask_mismatches": hall_mask_mismatches,
        "hall_max_abs_error": max(hall_errors, default=0.0),
        "imu_mask_mismatches": imu_mask_mismatches,
        "imu_max_matrix_abs_error": max(matrix_errors, default=0.0),
        "imu_max_quaternion_angular_error_deg": max(quaternion_errors, default=0.0),
    }


def gyro_sequence_from_quaternions(
    quaternions_wxyz: object,
    timestamps_seconds: object,
    valid_mask: object | None = None,
) -> np.ndarray:
    """Return analytic [F,2,3] angular velocity, without bridging gaps."""

    quaternions = np.asarray(quaternions_wxyz, dtype=np.float64)
    timestamps = np.asarray(timestamps_seconds, dtype=np.float64)
    if quaternions.ndim != 3 or quaternions.shape[1:] != (2, 4):
        raise ValueError(f"quaternions must have shape [F,2,4], got {quaternions.shape}")
    if timestamps.shape != (quaternions.shape[0],):
        raise ValueError("timestamp shape does not match quaternion sequence")
    if valid_mask is None:
        valid = np.isfinite(quaternions).all(axis=2)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != quaternions.shape[:2]:
            raise ValueError("gyro valid mask shape does not match quaternion sequence")
    result = np.full((quaternions.shape[0], 2, 3), np.nan, dtype=np.float64)
    for frame in range(1, quaternions.shape[0]):
        delta_seconds = float(timestamps[frame] - timestamps[frame - 1])
        if delta_seconds <= 0.0 or not math.isfinite(delta_seconds):
            raise InputContractError("gyro_timestamps_not_strictly_increasing")
        for hand in range(2):
            if valid[frame, hand] and valid[frame - 1, hand]:
                result[frame, hand] = angular_velocity_from_quaternions(
                    quaternions[frame - 1, hand],
                    quaternions[frame, hand],
                    delta_seconds,
                )
    if result.shape[0]:
        result[0] = np.nan
    return result


__all__ = [
    "compare_sensor_outputs",
    "gyro_sequence_from_quaternions",
    "hall_adc_counts",
    "normalize_angle_deg",
    "normalized_to_adc_count",
    "reference_ideal_sensor_model",
]
