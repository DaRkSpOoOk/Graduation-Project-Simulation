"""Self-check runner for the independent TASK-006B benchmark."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .contract import (
    ANGLE_ABSOLUTE_TOLERANCE_DEG,
    HALL_PER_HAND,
    SENSORS_PER_HAND,
    TOTAL_HALL_COUNT,
    TOTAL_IMU_COUNT,
    build_sensor_catalog,
    validate_sensor_catalog,
)
from .orientation import (
    GYROSCOPE_TOLERANCE_RAD_PER_SECOND,
    MATRIX_QUATERNION_TOLERANCE,
    ORIENTATION_ANGLE_TOLERANCE_DEG,
    QUATERNION_NORM_TOLERANCE,
    ROTATION_DETERMINANT_TOLERANCE,
    ROTATION_ORTHOGONALITY_TOLERANCE,
    angular_velocity_body_frame_from_quaternions,
    angular_velocity_from_quaternions,
    quaternion_angular_distance_deg,
    quaternion_matrix_wxyz,
    quaternion_wxyz_from_matrix,
    quaternions_equivalent,
    rotation_matrix_axis,
    rotation_matrix_xyz,
    validate_quaternion,
    validate_rotation_matrix,
)
from .reference import (
    compare_sensor_outputs,
    gyro_sequence_from_quaternions,
    hall_adc_counts,
    normalize_angle_deg,
    normalized_to_adc_count,
    reference_ideal_sensor_model,
)
from .synthetic import (
    KNOWN_ANGLE_VALUES_DEG,
    SyntheticFixture,
    build_fixture_catalog,
)


def _close(first: object, second: object, tolerance: float = ANGLE_ABSOLUTE_TOLERANCE_DEG) -> bool:
    return bool(np.allclose(first, second, atol=tolerance, rtol=0.0, equal_nan=True))


def _validity_checks(fixtures: tuple[SyntheticFixture, ...]) -> dict[str, bool]:
    by_id = {fixture.fixture_id: fixture for fixture in fixtures}
    partial = reference_ideal_sensor_model(by_id["validity_partial_spread"].data)
    palm_invalid = reference_ideal_sensor_model(by_id["validity_whole_palm_invalid"].data)
    missing = reference_ideal_sensor_model(by_id["validity_missing_tracking_pose"].data)

    partial_left_valid = int(partial.hall_valid[0, 0].sum())
    partial_right_valid = int(partial.hall_valid[0, 1].sum())
    return {
        "fully_valid_has_19_finite_hall_channels_per_hand": bool(
            partial_right_valid == HALL_PER_HAND
            and np.isfinite(partial.hall_normalized[0, 1]).all()
        ),
        "partial_spread_preserves_all_bends": bool(
            partial_left_valid == HALL_PER_HAND - 1
            and partial.hall_valid[0, 0, :15].all()
            and not partial.hall_valid[0, 0, 15 + 2]
        ),
        "partial_spread_keeps_imu_valid": bool(partial.palm_imu_valid[0, 0]),
        "whole_palm_invalid_invalidates_only_imu": bool(
            not bool(palm_invalid.palm_imu_valid[0, 0])
            and palm_invalid.hall_valid[0, 0].all()
            and np.isnan(palm_invalid.palm_imu_rotation_matrix[0, 0]).all()
        ),
        "missing_pose_has_no_fabricated_sensor_values": bool(
            not missing.hall_valid[0, 1].any()
            and not missing.palm_imu_valid[0, 1]
            and np.isnan(missing.hall_normalized[0, 1]).all()
            and np.isnan(missing.palm_imu_quaternion_wxyz[0, 1]).all()
        ),
    }


def _orientation_checks() -> dict[str, bool]:
    rotations = (
        np.eye(3, dtype=np.float64),
        rotation_matrix_axis("X", 90.0),
        rotation_matrix_axis("Y", 90.0),
        rotation_matrix_axis("Z", 90.0),
        rotation_matrix_axis("X", 180.0),
        rotation_matrix_axis("Y", 180.0),
        rotation_matrix_axis("Z", 180.0),
        rotation_matrix_xyz(25.0, -40.0, 70.0),
    )
    matrix_checks: list[bool] = []
    quaternion_checks: list[bool] = []
    sign_checks: list[bool] = []
    orientation_errors: list[float] = []
    for matrix in rotations:
        valid_matrix, _ = validate_rotation_matrix(
            matrix,
            orthogonality_tolerance=ROTATION_ORTHOGONALITY_TOLERANCE,
            determinant_tolerance=ROTATION_DETERMINANT_TOLERANCE,
        )
        quaternion = quaternion_wxyz_from_matrix(matrix)
        valid_quaternion, _ = validate_quaternion(
            quaternion,
            norm_tolerance=QUATERNION_NORM_TOLERANCE,
        )
        reconstructed = quaternion_matrix_wxyz(quaternion)
        matrix_checks.append(valid_matrix)
        quaternion_checks.append(
            valid_quaternion
            and float(np.max(np.abs(reconstructed - matrix))) <= MATRIX_QUATERNION_TOLERANCE
        )
        sign_checks.append(quaternions_equivalent(quaternion, -quaternion))
        orientation_errors.append(quaternion_angular_distance_deg(quaternion, quaternion))
    return {
        "known_rotations_are_proper": all(matrix_checks),
        "wxyz_round_trip_within_tolerance": all(quaternion_checks),
        "q_and_negative_q_are_equivalent": all(sign_checks),
        "known_orientation_self_errors_within_tolerance": all(
            error <= ORIENTATION_ANGLE_TOLERANCE_DEG for error in orientation_errors
        ),
    }


def _gyro_checks() -> dict[str, bool]:
    identity = quaternion_wxyz_from_matrix(np.eye(3))
    z90 = quaternion_wxyz_from_matrix(rotation_matrix_axis("Z", 90.0))
    x180 = quaternion_wxyz_from_matrix(rotation_matrix_axis("X", 180.0))
    sequence = np.asarray([[identity, identity], [z90, z90]], dtype=np.float64)
    velocities = gyro_sequence_from_quaternions(sequence, (0.0, 1.0))
    expected_z = np.asarray([0.0, 0.0, np.pi / 2.0])
    direct_x = gyro_sequence_from_quaternions(
        np.asarray([[identity, identity], [x180, x180]], dtype=np.float64),
        (0.0, 2.0),
    )
    expected_x = np.asarray([np.pi / 2.0, 0.0, 0.0])
    initial_matrix = rotation_matrix_xyz(35.0, -20.0, 15.0)
    local_matrix = rotation_matrix_axis("Y", 30.0)
    initial_quaternion = quaternion_wxyz_from_matrix(initial_matrix)
    current_quaternion = quaternion_wxyz_from_matrix(initial_matrix @ local_matrix)
    body_expected = np.asarray([0.0, np.pi / 6.0 / 0.5, 0.0])
    body_direct = angular_velocity_body_frame_from_quaternions(
        initial_quaternion, current_quaternion, 0.5
    )
    legacy_world = angular_velocity_from_quaternions(
        initial_quaternion, current_quaternion, 0.5
    )
    return {
        "zero_to_90_z_in_one_second": bool(
            np.allclose(velocities[1, 0], expected_z, atol=GYROSCOPE_TOLERANCE_RAD_PER_SECOND, rtol=0.0)
        ),
        "zero_to_180_x_in_two_seconds": bool(
            np.allclose(direct_x[1, 0], expected_x, atol=GYROSCOPE_TOLERANCE_RAD_PER_SECOND, rtol=0.0)
        ),
        "initial_velocity_row_is_missing": bool(np.isnan(velocities[0]).all()),
        "pre_rotated_body_frame_matches_local_delta": bool(
            np.allclose(
                body_direct,
                body_expected,
                atol=GYROSCOPE_TOLERANCE_RAD_PER_SECOND,
                rtol=0.0,
            )
        ),
        "legacy_world_helper_is_distinct_after_pre_rotation": bool(
            np.linalg.norm(legacy_world - body_expected) > 1e-8
        ),
    }


def _coverage_checks(fixtures: tuple[SyntheticFixture, ...]) -> dict[str, bool]:
    bend_coverage: dict[tuple[str, ...], set[float]] = {}
    spread_coverage: dict[tuple[str, ...], set[float]] = {}
    for fixture in fixtures:
        if fixture.coverage_kind == "bend" and fixture.coverage_key is not None:
            bend_coverage.setdefault(fixture.coverage_key, set()).add(float(fixture.coverage_angle_deg))
        if fixture.coverage_kind == "spread" and fixture.coverage_key is not None:
            spread_coverage.setdefault(fixture.coverage_key, set()).add(float(fixture.coverage_angle_deg))
    expected_keys = {
        (finger, joint)
        for finger in ("thumb", "index", "middle", "ring", "pinky")
        for joint in ("proximal", "middle", "distal")
    }
    expected_spread = {(name,) for name in ("thumb-index", "index-middle", "middle-ring", "ring-pinky")}
    expected_values = set(float(value) for value in KNOWN_ANGLE_VALUES_DEG)
    return {
        "every_bend_channel_has_all_known_values": set(bend_coverage) == expected_keys
        and all(values == expected_values for values in bend_coverage.values()),
        "every_spread_pair_has_all_known_values": set(spread_coverage) == expected_spread
        and all(values == expected_values for values in spread_coverage.values()),
    }


def _sensor_value_checks(fixtures: tuple[SyntheticFixture, ...]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    values = np.asarray(KNOWN_ANGLE_VALUES_DEG, dtype=np.float64)
    normalized = np.asarray([normalize_angle_deg(value) for value in values])
    checks["known_normalization_values"] = bool(
        np.array_equal(normalized, np.asarray((0.0, 1 / 180, 1 / 6, 0.25, 0.5, 0.75, 179 / 180, 1.0)))
    )
    checks["normalization_is_monotonic"] = bool(np.all(np.diff(normalized) > 0.0))
    checks["normalization_uses_no_dataset_statistics"] = bool(
        normalize_angle_deg(45.0) == 0.25
        and normalize_angle_deg(45.0) == normalize_angle_deg(45.0)
    )
    adc_values = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
    adc = hall_adc_counts(adc_values, np.ones(5, dtype=bool))
    checks["optional_adc_endpoints_and_rounding"] = bool(
        np.array_equal(adc, np.asarray([0, 1024, 2048, 3071, 4095], dtype=np.int16))
    )
    checks["optional_adc_is_monotonic"] = bool(np.all(np.diff(adc) >= 0))
    orientation_fixture = next(fixture for fixture in fixtures if fixture.fixture_id == "orientation_composed")
    orientation_output = reference_ideal_sensor_model(orientation_fixture.data)
    checks["imu_orientation_is_direct_passthrough"] = bool(
        np.array_equal(
            orientation_output.palm_imu_rotation_matrix,
            orientation_fixture.data.palm_rotation_matrix,
        )
        and all(
            quaternions_equivalent(
                orientation_output.palm_imu_quaternion_wxyz[frame, hand],
                orientation_fixture.data.palm_quaternion_wxyz[frame, hand],
            )
            for frame in range(orientation_output.palm_imu_valid.shape[0])
            for hand in range(2)
        )
    )
    checks["all_valid_fixture_outputs_are_deterministic"] = True
    for fixture in fixtures:
        if not fixture.expected_valid:
            continue
        first = reference_ideal_sensor_model(fixture.data)
        second = reference_ideal_sensor_model(fixture.data)
        comparison = compare_sensor_outputs(first, second)
        if not comparison["passed"]:
            checks["all_valid_fixture_outputs_are_deterministic"] = False
            break
    return checks


def _invalid_checks(fixtures: tuple[SyntheticFixture, ...]) -> tuple[dict[str, bool], list[dict[str, Any]]]:
    checks: dict[str, bool] = {}
    results: list[dict[str, Any]] = []
    for fixture in fixtures:
        if fixture.expected_valid:
            continue
        try:
            reference_ideal_sensor_model(fixture.data)
        except Exception as error:  # noqa: BLE001 - benchmark records all hard-fail types
            message = str(error)
            rejected = bool(fixture.expected_error and fixture.expected_error in message)
            results.append(
                {
                    "fixture_id": fixture.fixture_id,
                    "expected_error": fixture.expected_error,
                    "rejected": True,
                    "matching_error": rejected,
                    "error": message,
                }
            )
            checks[fixture.fixture_id] = rejected
        else:
            results.append(
                {
                    "fixture_id": fixture.fixture_id,
                    "expected_error": fixture.expected_error,
                    "rejected": False,
                    "matching_error": False,
                    "error": None,
                }
            )
            checks[fixture.fixture_id] = False
    return checks, results


def run_self_check() -> dict[str, Any]:
    """Run the complete independent benchmark self-check."""

    fixtures = build_fixture_catalog()
    valid = tuple(fixture for fixture in fixtures if fixture.expected_valid)
    invalid = tuple(fixture for fixture in fixtures if not fixture.expected_valid)
    category_counts = Counter(fixture.category for fixture in fixtures)
    catalog_check = validate_sensor_catalog()
    coverage_checks = _coverage_checks(valid)
    orientation_checks = _orientation_checks()
    gyro_checks = _gyro_checks()
    validity_checks = _validity_checks(valid)
    sensor_value_checks = _sensor_value_checks(valid)
    invalid_checks, invalid_results = _invalid_checks(invalid)

    catalog_shape = {
        "sensor_count": len(build_sensor_catalog()),
        "hall_count": TOTAL_HALL_COUNT,
        "imu_count": TOTAL_IMU_COUNT,
        "sensors_per_hand": SENSORS_PER_HAND,
    }
    all_checks: dict[str, bool] = {
        "sensor_catalog_valid": bool(catalog_check["passed"]),
        "catalog_has_19_hall_and_1_imu_per_hand": bool(
            catalog_shape["sensor_count"] == 40
            and catalog_shape["hall_count"] == 38
            and catalog_shape["imu_count"] == 2
        ),
        **coverage_checks,
        **orientation_checks,
        **gyro_checks,
        **validity_checks,
        **sensor_value_checks,
        "all_invalid_fixtures_hard_fail": all(invalid_checks.values()),
    }
    return {
        "schema_version": "TASK-006-ideal-virtual-glove-v1",
        "fixture_count": len(fixtures),
        "valid_fixture_count": len(valid),
        "invalid_fixture_count": len(invalid),
        "category_counts": dict(sorted(category_counts.items())),
        "catalog": catalog_shape | {"validation": catalog_check},
        "known_angle_values_deg": list(KNOWN_ANGLE_VALUES_DEG),
        "coverage": {
            "bend_channels": 15,
            "spread_pairs": 4,
            "values_per_channel": len(KNOWN_ANGLE_VALUES_DEG),
        },
        "tolerances": {
            "angle_absolute_deg": ANGLE_ABSOLUTE_TOLERANCE_DEG,
            "orientation_angular_deg": ORIENTATION_ANGLE_TOLERANCE_DEG,
            "rotation_orthogonality": ROTATION_ORTHOGONALITY_TOLERANCE,
            "rotation_determinant": ROTATION_DETERMINANT_TOLERANCE,
            "quaternion_norm": QUATERNION_NORM_TOLERANCE,
            "matrix_quaternion": MATRIX_QUATERNION_TOLERANCE,
            "gyro_rad_per_second": GYROSCOPE_TOLERANCE_RAD_PER_SECOND,
            "adc_count": 0,
        },
        "checks": all_checks,
        "invalid_fixture_results": invalid_results,
        "passed": all(all_checks.values()),
    }


__all__ = ["run_self_check"]
