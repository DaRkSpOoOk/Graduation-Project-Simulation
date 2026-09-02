"""Tests for the independent TASK-006B ideal virtual-glove benchmark."""

from __future__ import annotations

import dataclasses
import unittest

import numpy as np

from evaluation.virtual_glove import (
    ADC_MAX_COUNT,
    HALL_PER_HAND,
    InputContractError,
    build_fixture_catalog,
    build_sensor_catalog,
    compare_sensor_outputs,
    gyro_sequence_from_quaternions,
    hall_adc_counts,
    normalize_angle_deg,
    quaternion_matrix_wxyz,
    quaternion_wxyz_from_matrix,
    quaternions_equivalent,
    reference_ideal_sensor_model,
    rotation_matrix_axis,
    run_self_check,
    validate_kinematics_input,
    validate_sensor_catalog,
)


class TestTask006BSensorCatalog(unittest.TestCase):
    def test_catalog_has_fixed_counts_and_markers(self) -> None:
        catalog = build_sensor_catalog()
        self.assertEqual(len(catalog), 40)
        for hand in ("LEFT", "RIGHT"):
            hand_values = [definition for definition in catalog if definition.hand == hand]
            self.assertEqual(sum(definition.kind == "HALL" for definition in hand_values), 19)
            self.assertEqual(sum(definition.kind == "IMU" for definition in hand_values), 1)
            self.assertTrue(
                all(definition.display_marker == "H" for definition in hand_values if definition.kind == "HALL")
            )
            self.assertEqual(
                [definition.display_marker for definition in hand_values if definition.kind == "IMU"],
                ["IMU"],
            )
        self.assertTrue(validate_sensor_catalog()["passed"])

    def test_catalog_contains_every_expected_source_assignment(self) -> None:
        catalog = build_sensor_catalog()
        bend_sources = {
            definition.source[1:]
            for definition in catalog
            if definition.kind == "HALL" and definition.channel == "bend"
        }
        spread_sources = {
            definition.source[1:]
            for definition in catalog
            if definition.kind == "HALL" and definition.channel == "spread"
        }
        self.assertEqual(
            bend_sources,
            {
                (finger, joint)
                for finger in ("thumb", "index", "middle", "ring", "pinky")
                for joint in ("proximal", "middle", "distal")
            },
        )
        self.assertEqual(
            spread_sources,
            {(name,) for name in ("thumb-index", "index-middle", "middle-ring", "ring-pinky")},
        )

    def test_catalog_rejects_duplicate_and_wrong_assignment(self) -> None:
        catalog = list(build_sensor_catalog())
        catalog.append(catalog[0])
        self.assertFalse(validate_sensor_catalog(catalog)["passed"])
        catalog = list(build_sensor_catalog())
        catalog[0] = dataclasses.replace(catalog[0], source=("flexion_deg", "index", "distal"))
        result = validate_sensor_catalog(catalog)
        self.assertFalse(result["passed"])
        self.assertIn("definition_mismatch=LEFT.bend.thumb.proximal", result["errors"])


class TestTask006BReferenceModel(unittest.TestCase):
    def test_normalization_known_values_and_no_clipping(self) -> None:
        for degrees, expected in ((0.0, 0.0), (45.0, 0.25), (90.0, 0.5), (135.0, 0.75), (180.0, 1.0)):
            self.assertEqual(normalize_angle_deg(degrees), expected)
        for degrees in (-0.001, 180.001, np.nan, np.inf):
            with self.assertRaises(InputContractError):
                normalize_angle_deg(degrees)

    def test_reference_output_shape_and_comparison_adapter(self) -> None:
        fixture = next(fixture for fixture in build_fixture_catalog() if fixture.fixture_id == "multi_all_channels")
        output = reference_ideal_sensor_model(fixture.data)
        self.assertEqual(output.hall_normalized.shape, (1, 2, HALL_PER_HAND))
        self.assertEqual(output.hall_valid.shape, (1, 2, HALL_PER_HAND))
        self.assertEqual(output.palm_imu_rotation_matrix.shape, (1, 2, 3, 3))
        self.assertTrue(compare_sensor_outputs(output, output)["passed"])

    def test_left_right_local_values_are_equivalent_but_orientation_is_distinct(self) -> None:
        fixture = next(
            fixture
            for fixture in build_fixture_catalog()
            if fixture.fixture_id == "left_right_equivalent_local_values"
        )
        output = reference_ideal_sensor_model(fixture.data)
        self.assertTrue(np.array_equal(output.hall_normalized[0, 0], output.hall_normalized[0, 1]))
        self.assertFalse(
            np.array_equal(
                output.palm_imu_rotation_matrix[0, 0],
                output.palm_imu_rotation_matrix[0, 1],
            )
        )

    def test_partial_spread_preserves_bends_and_imu_mask(self) -> None:
        fixture = next(fixture for fixture in build_fixture_catalog() if fixture.fixture_id == "validity_partial_spread")
        output = reference_ideal_sensor_model(fixture.data)
        self.assertTrue(output.hall_valid[0, 0, :15].all())
        self.assertEqual(int(output.hall_valid[0, 0].sum()), 18)
        self.assertFalse(output.hall_valid[0, 0, 17])
        self.assertTrue(output.palm_imu_valid[0, 0])
        self.assertTrue(output.hall_valid[0, 1].all())

    def test_whole_palm_invalid_preserves_hall_and_invalidates_imu(self) -> None:
        fixture = next(
            fixture
            for fixture in build_fixture_catalog()
            if fixture.fixture_id == "validity_whole_palm_invalid"
        )
        output = reference_ideal_sensor_model(fixture.data)
        self.assertTrue(output.hall_valid[0, 0].all())
        self.assertFalse(output.palm_imu_valid[0, 0])
        self.assertTrue(np.isnan(output.palm_imu_rotation_matrix[0, 0]).all())
        self.assertTrue(output.palm_imu_valid[0, 1])

    def test_missing_tracking_pose_has_no_fabricated_values(self) -> None:
        fixture = next(
            fixture
            for fixture in build_fixture_catalog()
            if fixture.fixture_id == "validity_missing_tracking_pose"
        )
        output = reference_ideal_sensor_model(fixture.data)
        self.assertFalse(output.hall_valid[0, 1].any())
        self.assertFalse(output.palm_imu_valid[0, 1])
        self.assertTrue(np.isnan(output.hall_normalized[0, 1]).all())
        self.assertTrue(np.isnan(output.palm_imu_quaternion_wxyz[0, 1]).all())

    def test_orientation_is_direct_passthrough_and_wxyz_round_trips(self) -> None:
        for degrees, axis in ((90.0, "X"), (90.0, "Y"), (90.0, "Z"), (180.0, "X"), (180.0, "Y"), (180.0, "Z")):
            matrix = rotation_matrix_axis(axis, degrees)
            quaternion = quaternion_wxyz_from_matrix(matrix)
            self.assertTrue(np.allclose(quaternion_matrix_wxyz(quaternion), matrix, atol=1e-10, rtol=0.0))
            self.assertTrue(quaternions_equivalent(quaternion, -quaternion))

    def test_optional_adc_is_12_bit_monotonic_and_deterministic(self) -> None:
        normalized = np.asarray((0.0, 0.25, 0.5, 0.75, 1.0))
        counts = hall_adc_counts(normalized, np.ones(5, dtype=bool))
        self.assertEqual(counts.tolist(), [0, 1024, 2048, 3071, ADC_MAX_COUNT])
        self.assertTrue(np.all(np.diff(counts) >= 0))
        self.assertEqual(hall_adc_counts(np.asarray((np.nan,)), np.asarray((False,))).tolist(), [-1])

    def test_gyro_known_rotations_and_gap_policy(self) -> None:
        identity = quaternion_wxyz_from_matrix(np.eye(3))
        z90 = quaternion_wxyz_from_matrix(rotation_matrix_axis("Z", 90.0))
        quaternions = np.asarray([[identity, identity], [z90, z90]], dtype=np.float64)
        result = gyro_sequence_from_quaternions(quaternions, (0.0, 1.0))
        self.assertTrue(np.allclose(result[1, 0], (0.0, 0.0, np.pi / 2), atol=1e-10, rtol=0.0))
        self.assertTrue(np.isnan(result[0]).all())
        masked = gyro_sequence_from_quaternions(quaternions, (0.0, 1.0), ((True, True), (False, True)))
        self.assertTrue(np.isnan(masked[1, 0]).all())
        self.assertTrue(np.isfinite(masked[1, 1]).all())


class TestTask006BInputValidation(unittest.TestCase):
    def test_all_invalid_fixture_inputs_hard_fail_with_expected_reason(self) -> None:
        invalid = [fixture for fixture in build_fixture_catalog() if not fixture.expected_valid]
        self.assertEqual(len(invalid), 12)
        for fixture in invalid:
            with self.assertRaises(InputContractError) as context:
                validate_kinematics_input(fixture.data)
            self.assertIn(fixture.expected_error, str(context.exception), fixture.fixture_id)

    def test_valid_fixture_inputs_pass(self) -> None:
        valid = [fixture for fixture in build_fixture_catalog() if fixture.expected_valid]
        self.assertEqual(len(valid), 167)
        for fixture in valid:
            self.assertTrue(validate_kinematics_input(fixture.data)["passed"], fixture.fixture_id)

    def test_benchmark_self_check_passes_without_production_imports(self) -> None:
        result = run_self_check()
        self.assertTrue(result["passed"])
        self.assertEqual(result["fixture_count"], 179)
        self.assertEqual(result["valid_fixture_count"], 167)
        self.assertEqual(result["invalid_fixture_count"], 12)
        self.assertFalse(any(name.startswith("virtual_sensors") for name in result["checks"]))


if __name__ == "__main__":
    unittest.main()
