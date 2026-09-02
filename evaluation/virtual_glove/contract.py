"""Independent TASK-006 ideal virtual-glove data contract.

This module describes the logical sensor catalog and validates the frozen
TASK-005-like input shape.  It is deliberately an evaluation contract only;
it does not live in ``virtual_sensors/`` and does not implement production
sensor conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .orientation import (
    MATRIX_QUATERNION_TOLERANCE,
    QUATERNION_NORM_TOLERANCE,
    ROTATION_DETERMINANT_TOLERANCE,
    ROTATION_ORTHOGONALITY_TOLERANCE,
    quaternion_matrix_wxyz,
    validate_quaternion,
    validate_rotation_matrix,
)


SCHEMA_VERSION = "TASK-006-ideal-virtual-glove-v1"
TRACK_ORDER = ("LEFT", "RIGHT")
FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")
CHAIN_ORDER = ("proximal", "middle", "distal")
SPREAD_ORDER = ("thumb-index", "index-middle", "middle-ring", "ring-pinky")
HALL_BEND_COUNT = len(FINGER_ORDER) * len(CHAIN_ORDER)
HALL_SPREAD_COUNT = len(SPREAD_ORDER)
HALL_PER_HAND = HALL_BEND_COUNT + HALL_SPREAD_COUNT
IMU_PER_HAND = 1
SENSORS_PER_HAND = HALL_PER_HAND + IMU_PER_HAND
TOTAL_HALL_COUNT = HALL_PER_HAND * len(TRACK_ORDER)
TOTAL_IMU_COUNT = IMU_PER_HAND * len(TRACK_ORDER)

ANGLE_MIN_DEG = 0.0
ANGLE_MAX_DEG = 180.0
ADC_MAX_COUNT = 4095
INVALID_ADC_COUNT = -1

# These are intentionally stricter than the TASK-005 pilot QA limits.  They
# are for deterministic float64 synthetic algebra and are locked before any
# production TASK-006A output is inspected.
ANGLE_ABSOLUTE_TOLERANCE_DEG = 1e-10
ADC_ABSOLUTE_TOLERANCE_COUNT = 0
MATRIX_ORIENTATION_TOLERANCE = MATRIX_QUATERNION_TOLERANCE
ORIENTATION_ANGULAR_TOLERANCE_DEG = 1e-10


class InputContractError(ValueError):
    """Raised when an input violates the frozen ideal-glove contract."""


class SensorCatalogError(ValueError):
    """Raised when a logical sensor catalog is incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class SensorDefinition:
    """One logical sensor in the ideal glove catalog."""

    sensor_id: str
    hand: str
    kind: str
    channel: str
    source: tuple[str, ...]
    display_marker: str


@dataclass(frozen=True, slots=True)
class KinematicsInput:
    """TASK-005-like arrays consumed by the independent benchmark oracle."""

    flexion_deg: np.ndarray
    adjacent_spread_deg: np.ndarray
    palm_rotation_matrix: np.ndarray
    palm_quaternion_wxyz: np.ndarray
    timestamps_seconds: np.ndarray
    frame_index: np.ndarray
    hand_present: np.ndarray
    valid_palm_frame: np.ndarray
    valid_kinematics: np.ndarray
    source_provenance: tuple[str, ...] | None
    track_order: tuple[str, ...] = TRACK_ORDER
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class IdealSensorOutput:
    """Reference output shape for a future TASK-006 production adapter."""

    sensor_catalog: tuple[SensorDefinition, ...]
    hall_normalized: np.ndarray
    hall_valid: np.ndarray
    palm_imu_rotation_matrix: np.ndarray
    palm_imu_quaternion_wxyz: np.ndarray
    palm_imu_valid: np.ndarray


def _bend_source(finger_index: int, joint_index: int) -> tuple[str, ...]:
    return ("flexion_deg", FINGER_ORDER[finger_index], CHAIN_ORDER[joint_index])


def _spread_source(spread_index: int) -> tuple[str, ...]:
    return ("adjacent_spread_deg", SPREAD_ORDER[spread_index])


def build_sensor_catalog() -> tuple[SensorDefinition, ...]:
    """Build the deterministic 40-sensor logical catalog.

    Per hand the order is 15 bend Hall channels, 4 spread Hall channels, then
    one palm IMU.  IDs are semantic and stable so a later production adapter
    cannot silently swap a joint while keeping the same array shape.
    """

    definitions: list[SensorDefinition] = []
    for hand in TRACK_ORDER:
        for finger_index, finger in enumerate(FINGER_ORDER):
            for joint_index, joint in enumerate(CHAIN_ORDER):
                definitions.append(
                    SensorDefinition(
                        sensor_id=f"{hand}.bend.{finger}.{joint}",
                        hand=hand,
                        kind="HALL",
                        channel="bend",
                        source=_bend_source(finger_index, joint_index),
                        display_marker="H",
                    )
                )
        for spread_index, spread in enumerate(SPREAD_ORDER):
            definitions.append(
                SensorDefinition(
                    sensor_id=f"{hand}.spread.{spread}",
                    hand=hand,
                    kind="HALL",
                    channel="spread",
                    source=_spread_source(spread_index),
                    display_marker="H",
                )
            )
        definitions.append(
            SensorDefinition(
                sensor_id=f"{hand}.palm_imu",
                hand=hand,
                kind="IMU",
                channel="palm_orientation",
                source=("palm_rotation_matrix", "palm_quaternion_wxyz"),
                display_marker="IMU",
            )
        )
    return tuple(definitions)


def validate_sensor_catalog(
    catalog: Iterable[SensorDefinition] | None = None,
) -> dict[str, Any]:
    """Hard-validate sensor IDs, source assignments, counts and markers."""

    values = tuple(build_sensor_catalog() if catalog is None else catalog)
    errors: list[str] = []
    ids = [definition.sensor_id for definition in values]
    if len(ids) != len(set(ids)):
        errors.append("duplicate_sensor_id")
    if len(values) != len(TRACK_ORDER) * SENSORS_PER_HAND:
        errors.append(f"sensor_count={len(values)}")

    expected = {definition.sensor_id: definition for definition in build_sensor_catalog()}
    for definition in values:
        if definition.sensor_id not in expected:
            errors.append(f"unexpected_sensor_id={definition.sensor_id}")
            continue
        reference = expected[definition.sensor_id]
        if definition != reference:
            errors.append(f"definition_mismatch={definition.sensor_id}")

    for hand in TRACK_ORDER:
        hand_values = [definition for definition in values if definition.hand == hand]
        halls = [definition for definition in hand_values if definition.kind == "HALL"]
        imus = [definition for definition in hand_values if definition.kind == "IMU"]
        if len(halls) != HALL_PER_HAND:
            errors.append(f"{hand}_hall_count={len(halls)}")
        if len(imus) != IMU_PER_HAND:
            errors.append(f"{hand}_imu_count={len(imus)}")
        if any(definition.display_marker != "H" for definition in halls):
            errors.append(f"{hand}_hall_marker")
        if any(definition.display_marker != "IMU" for definition in imus):
            errors.append(f"{hand}_imu_marker")

    return {
        "passed": not errors,
        "errors": sorted(set(errors)),
        "sensor_count": len(values),
        "hall_count": sum(definition.kind == "HALL" for definition in values),
        "imu_count": sum(definition.kind == "IMU" for definition in values),
        "sensor_ids": ids,
    }


def _shape_error(name: str, values: np.ndarray, expected: tuple[Any, ...]) -> str | None:
    if values.ndim != len(expected):
        return f"{name}_rank={values.ndim}"
    for actual, wanted in zip(values.shape, expected):
        if wanted != "F" and actual != wanted:
            return f"{name}_shape={values.shape}"
    return None


def _is_all_nan(values: np.ndarray) -> bool:
    return bool(np.isnan(np.asarray(values, dtype=np.float64)).all())


def _finite_angle_error(name: str, values: np.ndarray, errors: list[str]) -> None:
    finite = np.isfinite(values)
    if np.any(values[finite] < ANGLE_MIN_DEG) or np.any(values[finite] > ANGLE_MAX_DEG):
        errors.append(f"{name}_outside_0_180")


def validate_kinematics_input(data: KinematicsInput) -> dict[str, Any]:
    """Validate structure, masks, provenance, angles and orientation.

    This function never clips, normalizes, fills, or repairs values.  It
    returns a compact diagnostic and raises ``InputContractError`` on the
    first complete validation failure so corrupted data cannot silently enter
    the reference sensor model.
    """

    errors: list[str] = []
    arrays = {
        "flexion_deg": np.asarray(data.flexion_deg),
        "adjacent_spread_deg": np.asarray(data.adjacent_spread_deg),
        "palm_rotation_matrix": np.asarray(data.palm_rotation_matrix),
        "palm_quaternion_wxyz": np.asarray(data.palm_quaternion_wxyz),
        "timestamps_seconds": np.asarray(data.timestamps_seconds),
        "frame_index": np.asarray(data.frame_index),
        "hand_present": np.asarray(data.hand_present),
        "valid_palm_frame": np.asarray(data.valid_palm_frame),
        "valid_kinematics": np.asarray(data.valid_kinematics),
    }
    frame_index = arrays["frame_index"]
    frame_count = int(frame_index.shape[0]) if frame_index.ndim >= 1 else 0
    expected_shapes: dict[str, tuple[Any, ...]] = {
        "flexion_deg": ("F", 2, 5, 3),
        "adjacent_spread_deg": ("F", 2, 4),
        "palm_rotation_matrix": ("F", 2, 3, 3),
        "palm_quaternion_wxyz": ("F", 2, 4),
        "timestamps_seconds": ("F",),
        "frame_index": ("F",),
        "hand_present": ("F", 2),
        "valid_palm_frame": ("F", 2),
        "valid_kinematics": ("F", 2),
    }
    for name, expected in expected_shapes.items():
        error = _shape_error(name, arrays[name], expected)
        if error is not None:
            errors.append(error)
        elif "F" in expected and arrays[name].shape[0] != frame_count:
            errors.append(f"{name}_frame_count={arrays[name].shape[0]}")

    if tuple(data.track_order) != TRACK_ORDER:
        errors.append("wrong_track_order")
    if data.schema_version != SCHEMA_VERSION:
        errors.append("wrong_schema_version")
    provenance = data.source_provenance
    if provenance is None or len(provenance) != frame_count or any(
        not isinstance(value, str) or not value.strip() for value in provenance
    ):
        errors.append("missing_or_malformed_provenance")

    for name in ("hand_present", "valid_palm_frame", "valid_kinematics"):
        if arrays[name].shape == expected_shapes[name] or (
            arrays[name].ndim == len(expected_shapes[name])
            and arrays[name].shape[0] == frame_count
        ):
            if arrays[name].dtype != np.bool_:
                errors.append(f"{name}_must_be_bool")

    timestamps = arrays["timestamps_seconds"]
    if timestamps.ndim == 1 and not np.isfinite(timestamps).all():
        errors.append("timestamps_non_finite")
    if timestamps.ndim == 1 and timestamps.size > 1:
        differences = np.diff(timestamps.astype(np.float64, copy=False))
        if np.any(differences == 0.0):
            errors.append("timestamps_duplicate")
        if np.any(differences < 0.0):
            errors.append("timestamps_non_monotonic")

    if frame_index.ndim == 1:
        if not np.issubdtype(frame_index.dtype, np.integer):
            errors.append("frame_index_must_be_integer")
        if frame_index.size and np.any(frame_index < 0):
            errors.append("frame_index_negative")
        if frame_index.size > 1:
            differences = np.diff(frame_index.astype(np.int64, copy=False))
            if np.any(differences == 0):
                errors.append("frame_index_duplicate")
            if np.any(differences < 0):
                errors.append("frame_index_non_monotonic")

    if not errors:
        flexion = arrays["flexion_deg"].astype(np.float64, copy=False)
        spread = arrays["adjacent_spread_deg"].astype(np.float64, copy=False)
        matrices = arrays["palm_rotation_matrix"].astype(np.float64, copy=False)
        quaternions = arrays["palm_quaternion_wxyz"].astype(np.float64, copy=False)
        hand_present = arrays["hand_present"]
        palm_valid = arrays["valid_palm_frame"]
        strict_valid = arrays["valid_kinematics"]
        _finite_angle_error("flexion_deg", flexion, errors)
        _finite_angle_error("adjacent_spread_deg", spread, errors)

        for frame in range(frame_count):
            for hand in range(2):
                present = bool(hand_present[frame, hand])
                palm = bool(palm_valid[frame, hand])
                strict = bool(strict_valid[frame, hand])
                finite_flexion = bool(np.isfinite(flexion[frame, hand]).all())
                finite_spread = bool(np.isfinite(spread[frame, hand]).all())
                finite_orientation = bool(np.isfinite(matrices[frame, hand]).all())
                finite_quaternion = bool(np.isfinite(quaternions[frame, hand]).all())

                if not present:
                    if palm or strict:
                        errors.append(f"missing_pose_valid_flag={frame}:{hand}")
                    if any(
                        np.isfinite(values[frame, hand]).any()
                        for values in (flexion, spread, matrices, quaternions)
                    ):
                        errors.append(f"missing_pose_has_finite_derived={frame}:{hand}")
                    continue

                if palm:
                    valid_matrix, matrix_reason = validate_rotation_matrix(
                        matrices[frame, hand],
                        orthogonality_tolerance=ROTATION_ORTHOGONALITY_TOLERANCE,
                        determinant_tolerance=ROTATION_DETERMINANT_TOLERANCE,
                    )
                    if not valid_matrix:
                        errors.append(f"invalid_palm_rotation={frame}:{hand}:{matrix_reason}")
                    valid_quaternion, quaternion_reason = validate_quaternion(
                        quaternions[frame, hand],
                        norm_tolerance=QUATERNION_NORM_TOLERANCE,
                    )
                    if not valid_quaternion:
                        errors.append(f"invalid_palm_quaternion={frame}:{hand}:{quaternion_reason}")
                    if valid_matrix and valid_quaternion:
                        converted = quaternion_matrix_wxyz(quaternions[frame, hand])
                        disagreement = float(np.max(np.abs(converted - matrices[frame, hand])))
                        if disagreement > MATRIX_QUATERNION_TOLERANCE:
                            errors.append(
                                f"matrix_quaternion_disagreement={frame}:{hand}:{disagreement:.3e}"
                            )
                else:
                    if finite_orientation or finite_quaternion:
                        errors.append(f"invalid_palm_has_orientation={frame}:{hand}")

                expected_strict = bool(palm and finite_flexion and finite_spread)
                if strict != expected_strict:
                    errors.append(f"strict_validity_mismatch={frame}:{hand}")

    result = {
        "passed": not errors,
        "errors": sorted(set(errors)),
        "frame_count": frame_count,
        "track_count": 2 if not errors or arrays["hand_present"].ndim == 2 else None,
        "schema_version": data.schema_version,
    }
    if errors:
        raise InputContractError("; ".join(result["errors"]))
    return result


def expected_hall_sensor_ids(hand: str) -> tuple[str, ...]:
    """Return the 19 stable Hall IDs for one canonical hand name."""

    normalized = str(hand).upper()
    if normalized not in TRACK_ORDER:
        raise SensorCatalogError(f"unknown hand {hand!r}")
    return tuple(
        definition.sensor_id
        for definition in build_sensor_catalog()
        if definition.hand == normalized and definition.kind == "HALL"
    )


def expected_imu_sensor_id(hand: str) -> str:
    """Return the stable palm IMU ID for one canonical hand name."""

    normalized = str(hand).upper()
    if normalized not in TRACK_ORDER:
        raise SensorCatalogError(f"unknown hand {hand!r}")
    return f"{normalized}.palm_imu"


__all__ = [
    "ADC_MAX_COUNT",
    "ANGLE_ABSOLUTE_TOLERANCE_DEG",
    "ANGLE_MAX_DEG",
    "ANGLE_MIN_DEG",
    "CHAIN_ORDER",
    "FINGER_ORDER",
    "HALL_BEND_COUNT",
    "HALL_PER_HAND",
    "HALL_SPREAD_COUNT",
    "IMU_PER_HAND",
    "IdealSensorOutput",
    "InputContractError",
    "KinematicsInput",
    "MATRIX_ORIENTATION_TOLERANCE",
    "ORIENTATION_ANGULAR_TOLERANCE_DEG",
    "SENSORS_PER_HAND",
    "SCHEMA_VERSION",
    "SPREAD_ORDER",
    "SensorCatalogError",
    "SensorDefinition",
    "TOTAL_HALL_COUNT",
    "TOTAL_IMU_COUNT",
    "TRACK_ORDER",
    "build_sensor_catalog",
    "expected_hall_sensor_ids",
    "expected_imu_sensor_id",
    "validate_kinematics_input",
    "validate_sensor_catalog",
]
