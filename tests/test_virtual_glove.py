"""Synthetic tests for the TASK-006A ideal virtual smart-glove sensor model.

Every fixture is built in memory. Nothing here reads the pilot, the WiLoR
checkpoint, MANO assets, or any generated run directory.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from virtual_glove import (
    ADC_INVALID_SENTINEL,
    ADC_MAX,
    ARRAY_ORDER,
    BEND_SENSOR_IDS,
    CHAIN_ORDER,
    EXPECTED_BEND_SENSORS,
    EXPECTED_HALL_SENSORS,
    EXPECTED_IMU_SENSORS,
    EXPECTED_SENSOR_PACKAGES,
    EXPECTED_SPREAD_SENSORS,
    FINGER_ORDER,
    GloveInputError,
    HALL_SENSOR_IDS,
    IMU_SENSOR_IDS,
    MARKER_HALL,
    MARKER_IMU,
    NORMALIZATION_DIVISOR,
    SENSOR_LAYOUT,
    SPREAD_PAIRS,
    SPREAD_SENSOR_IDS,
    SensorContractViolation,
    accelerometer_feasibility,
    angular_velocity_body_frame,
    build_metadata,
    extract_glove_sequence,
    layout_document,
    load_glove_sequence,
    normalize_angles,
    save_glove_sequence,
    sensor_by_id,
    to_adc_12bit,
)

TRACKS = 2
FINGERS = 5
CHAIN = 3
SPREAD = 4


def rotation_about(axis, degrees: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    theta = np.radians(degrees)
    cross = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + np.sin(theta) * cross + (1.0 - np.cos(theta)) * (cross @ cross)


def quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    trace = float(np.trace(matrix))
    scale = np.sqrt(max(trace + 1.0, 1e-12)) * 2.0
    quaternion = np.array(
        [
            0.25 * scale,
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        ]
    )
    return quaternion / np.linalg.norm(quaternion)


def make_kinematics(frames: int = 6, *, fps: float = 30.0):
    """A minimal synthetic payload in the frozen TASK-005 layout."""

    bend = np.zeros((frames, TRACKS, FINGERS, CHAIN), dtype=np.float32)
    spread = np.zeros((frames, TRACKS, SPREAD), dtype=np.float32)
    rotation = np.zeros((frames, TRACKS, 3, 3), dtype=np.float32)
    quaternion = np.zeros((frames, TRACKS, 4), dtype=np.float32)
    for row in range(frames):
        bend[row] = 30.0
        spread[row] = 20.0
        for track in range(TRACKS):
            matrix = rotation_about((0.0, 0.0, 1.0), 10.0 * row)
            rotation[row, track] = matrix
            quaternion[row, track] = quaternion_wxyz(matrix)
    arrays = {
        "frame_index": np.arange(frames, dtype=np.int32),
        "timestamp_seconds": (np.arange(frames) / fps).astype(np.float64),
        "tracking_state_code": np.ones((frames, TRACKS), dtype=np.int32),
        "source_raw_detection_index": np.tile(
            np.array([[0, 1]], dtype=np.int32), (frames, 1)
        ),
        "valid_kinematics": np.ones((frames, TRACKS), dtype=bool),
        "valid_palm_frame": np.ones((frames, TRACKS), dtype=bool),
        "flexion_deg": bend,
        "adjacent_spread_deg": spread,
        "palm_rotation_matrix": rotation,
        "palm_quaternion_wxyz": quaternion,
    }
    metadata = {
        "track_order": ["left", "right"],
        "finger_order": list(FINGER_ORDER),
        "chain_joint_order": list(CHAIN_ORDER),
        "spread_pairs": [list(p) for p in SPREAD_PAIRS],
        "quaternion_order": "wxyz",
        "schema_version": "hand_kinematics_v1",
        "sample_id": "synthetic",
    }
    return arrays, metadata


class TestBendNormalization(unittest.TestCase):
    """1-3, 5. The normalizer is the 0..180 contract and nothing else."""

    def test_zero_degrees_maps_to_zero(self) -> None:
        value, _ = normalize_angles(np.array([0.0]))
        self.assertEqual(float(value[0]), 0.0)

    def test_ninety_degrees_maps_to_one_half(self) -> None:
        value, _ = normalize_angles(np.array([90.0]))
        self.assertAlmostEqual(float(value[0]), 0.5, places=12)

    def test_one_eighty_degrees_maps_to_one(self) -> None:
        value, _ = normalize_angles(np.array([180.0]))
        self.assertEqual(float(value[0]), 1.0)

    def test_normalization_is_strictly_monotonic(self) -> None:
        angles = np.array([0.0, 1.0, 12.5, 45.0, 90.0, 130.0, 179.9, 180.0])
        values, _ = normalize_angles(angles)
        self.assertTrue(np.all(np.diff(values) > 0.0))

    def test_divisor_is_the_fixed_contract_bound(self) -> None:
        self.assertEqual(NORMALIZATION_DIVISOR, 180.0)


class TestSpreadNormalization(unittest.TestCase):
    """4. Spread uses the identical transfer, not a separate one."""

    def test_known_spread_values(self) -> None:
        values, _ = normalize_angles(np.array([0.0, 90.0, 180.0]), channel="spread")
        np.testing.assert_allclose(values, [0.0, 0.5, 1.0], atol=1e-12)

    def test_spread_and_bend_share_one_transfer(self) -> None:
        angles = np.array([0.0, 33.0, 97.6, 180.0])
        bend, _ = normalize_angles(angles, channel="bend")
        spread, _ = normalize_angles(angles, channel="spread")
        np.testing.assert_array_equal(bend, spread)


class TestNoDatasetFittedNormalization(unittest.TestCase):
    """6. The transfer must not depend on the data it is given."""

    def test_same_angle_maps_identically_regardless_of_batch(self) -> None:
        alone, _ = normalize_angles(np.array([42.0]))
        with_small = normalize_angles(np.array([42.0, 0.5, 1.0]))[0][0]
        with_large = normalize_angles(np.array([42.0, 170.0, 180.0]))[0][0]
        self.assertAlmostEqual(float(alone[0]), float(with_small), places=15)
        self.assertAlmostEqual(float(alone[0]), float(with_large), places=15)

    def test_a_narrow_pilot_like_range_does_not_rescale_to_full_range(self) -> None:
        """Min/max normalization would map the observed max to 1.0. This must not."""

        observed = np.array([0.14, 21.0, 113.65])  # the pilot's bend min/median/max
        values, _ = normalize_angles(observed)
        self.assertLess(float(values.max()), 0.64)
        self.assertGreater(float(values.min()), 0.0)

    def test_no_neutral_offset_is_subtracted(self) -> None:
        """The proximal channel's non-zero neutral is preserved, not removed."""

        arrays, metadata = make_kinematics(frames=2)
        arrays["flexion_deg"][:, :, 1, 0] = 12.9  # index proximal floor
        sequence = extract_glove_sequence(arrays, metadata, "synthetic")
        self.assertAlmostEqual(
            float(sequence.bend_normalized[0, 0, 1, 0]), 12.9 / 180.0, places=6
        )


class TestContractViolations(unittest.TestCase):
    """Out-of-contract values are surfaced, never silently repaired."""

    def test_value_above_contract_raises(self) -> None:
        with self.assertRaises(SensorContractViolation):
            normalize_angles(np.array([180.5]))

    def test_value_below_contract_raises(self) -> None:
        with self.assertRaises(SensorContractViolation):
            normalize_angles(np.array([-0.1]))

    def test_flag_mode_reports_without_clamping(self) -> None:
        values, mask = normalize_angles(np.array([200.0, 90.0]), on_violation="flag")
        self.assertTrue(bool(mask[0]))
        self.assertFalse(bool(mask[1]))
        self.assertGreater(float(values[0]), 1.0)  # not clamped to 1.0

    def test_nan_is_absence_not_violation(self) -> None:
        values, mask = normalize_angles(np.array([np.nan]))
        self.assertTrue(bool(np.isnan(values[0])))
        self.assertFalse(bool(mask[0]))

    def test_extractor_raises_on_out_of_contract_input(self) -> None:
        arrays, metadata = make_kinematics(frames=2)
        arrays["flexion_deg"][0, 0, 0, 0] = 181.0
        with self.assertRaises(SensorContractViolation):
            extract_glove_sequence(arrays, metadata, "synthetic")

    def test_flag_mode_invalidates_rather_than_repairs(self) -> None:
        arrays, metadata = make_kinematics(frames=2)
        arrays["flexion_deg"][0, 0, 0, 0] = 181.0
        sequence = extract_glove_sequence(
            arrays, metadata, "synthetic", on_contract_violation="flag"
        )
        self.assertFalse(bool(sequence.bend_valid[0, 0, 0, 0]))
        self.assertEqual(float(sequence.bend_angle_deg[0, 0, 0, 0]), 181.0)
        self.assertTrue(sequence.contract_violations)


class TestChannelValidity(unittest.TestCase):
    """7-9. Model-B: per-channel masks, never a whole-hand discard."""

    def test_nan_bend_invalidates_only_that_sensor(self) -> None:
        arrays, metadata = make_kinematics()
        arrays["flexion_deg"][1, 0, 2, 1] = np.nan
        sequence = extract_glove_sequence(arrays, metadata, "synthetic")
        self.assertFalse(bool(sequence.bend_valid[1, 0, 2, 1]))
        self.assertEqual(int((~sequence.bend_valid).sum()), 1)
        self.assertTrue(bool(sequence.spread_valid.all()))
        self.assertTrue(bool(sequence.palm_imu_valid.all()))

    def test_nan_spread_affects_only_the_relevant_spread_sensor(self) -> None:
        arrays, metadata = make_kinematics()
        arrays["adjacent_spread_deg"][2, 1, 3] = np.nan
        sequence = extract_glove_sequence(arrays, metadata, "synthetic")
        self.assertFalse(bool(sequence.spread_valid[2, 1, 3]))
        self.assertEqual(int((~sequence.spread_valid).sum()), 1)
        self.assertTrue(bool(sequence.bend_valid.all()))
        self.assertTrue(bool(sequence.palm_imu_valid.all()))

    def test_valid_flexion_survives_strict_valid_kinematics_false(self) -> None:
        """The exact scenario the task calls out: 15 bends + IMU must survive."""

        arrays, metadata = make_kinematics()
        arrays["adjacent_spread_deg"][0, 0, 2] = np.nan
        arrays["valid_kinematics"][0, 0] = False  # strict flag false, palm still fine
        sequence = extract_glove_sequence(arrays, metadata, "synthetic")
        self.assertEqual(int(sequence.bend_valid[0, 0].sum()), EXPECTED_BEND_SENSORS)
        self.assertTrue(bool(sequence.palm_imu_valid[0, 0]))
        self.assertEqual(int(sequence.spread_valid[0, 0].sum()), EXPECTED_SPREAD_SENSORS - 1)
        self.assertFalse(bool(sequence.spread_valid[0, 0, 2]))

    def test_missing_hand_invalidates_every_channel(self) -> None:
        arrays, metadata = make_kinematics()
        arrays["flexion_deg"][3, 1] = np.nan
        arrays["adjacent_spread_deg"][3, 1] = np.nan
        arrays["palm_rotation_matrix"][3, 1] = np.nan
        arrays["palm_quaternion_wxyz"][3, 1] = np.nan
        arrays["valid_kinematics"][3, 1] = False
        arrays["valid_palm_frame"][3, 1] = False
        sequence = extract_glove_sequence(arrays, metadata, "synthetic")
        self.assertFalse(bool(sequence.bend_valid[3, 1].any()))
        self.assertFalse(bool(sequence.spread_valid[3, 1].any()))
        self.assertFalse(bool(sequence.palm_imu_valid[3, 1]))

    def test_masks_come_from_channel_state_not_the_strict_flag(self) -> None:
        arrays, metadata = make_kinematics()
        arrays["valid_kinematics"][:] = False
        sequence = extract_glove_sequence(arrays, metadata, "synthetic")
        self.assertTrue(bool(sequence.bend_valid.all()))
        self.assertTrue(bool(sequence.spread_valid.all()))

    def test_no_values_are_invented_for_invalid_channels(self) -> None:
        arrays, metadata = make_kinematics()
        arrays["flexion_deg"][1, 0, 2, 1] = np.nan
        sequence = extract_glove_sequence(arrays, metadata, "synthetic")
        self.assertTrue(bool(np.isnan(sequence.bend_angle_deg[1, 0, 2, 1])))
        self.assertTrue(bool(np.isnan(sequence.bend_normalized[1, 0, 2, 1])))


class TestTrackOrderAndProvenance(unittest.TestCase):
    """10, 18. LEFT/RIGHT order and source provenance are preserved."""

    def test_track_order_is_left_then_right(self) -> None:
        arrays, metadata = make_kinematics()
        arrays["flexion_deg"][:, 0] = 10.0
        arrays["flexion_deg"][:, 1] = 80.0
        sequence = extract_glove_sequence(arrays, metadata, "synthetic")
        self.assertAlmostEqual(float(sequence.bend_angle_deg[0, 0, 0, 0]), 10.0, places=4)
        self.assertAlmostEqual(float(sequence.bend_angle_deg[0, 1, 0, 0]), 80.0, places=4)

    def test_reordered_tracks_are_refused(self) -> None:
        arrays, metadata = make_kinematics()
        metadata["track_order"] = ["right", "left"]
        with self.assertRaises(GloveInputError):
            extract_glove_sequence(arrays, metadata, "synthetic")

    def test_reordered_fingers_are_refused(self) -> None:
        arrays, metadata = make_kinematics()
        metadata["finger_order"] = ["index", "thumb", "middle", "ring", "pinky"]
        with self.assertRaises(GloveInputError):
            extract_glove_sequence(arrays, metadata, "synthetic")

    def test_non_wxyz_quaternion_order_is_refused(self) -> None:
        arrays, metadata = make_kinematics()
        metadata["quaternion_order"] = "xyzw"
        with self.assertRaises(GloveInputError):
            extract_glove_sequence(arrays, metadata, "synthetic")

    def test_provenance_is_carried_verbatim(self) -> None:
        arrays, metadata = make_kinematics()
        arrays["tracking_state_code"][2, 1] = 4
        arrays["source_raw_detection_index"][3, 0] = 7
        sequence = extract_glove_sequence(arrays, metadata, "synthetic")
        np.testing.assert_array_equal(
            sequence.tracking_state_code, arrays["tracking_state_code"]
        )
        np.testing.assert_array_equal(
            sequence.source_raw_detection_index, arrays["source_raw_detection_index"]
        )
        np.testing.assert_array_equal(sequence.frame_index, arrays["frame_index"])
        np.testing.assert_array_equal(
            sequence.timestamp_seconds, arrays["timestamp_seconds"]
        )

    def test_source_arrays_are_not_mutated(self) -> None:
        arrays, metadata = make_kinematics()
        before = {k: v.copy() for k, v in arrays.items()}
        extract_glove_sequence(arrays, metadata, "synthetic")
        for key, value in before.items():
            np.testing.assert_array_equal(arrays[key], value)

    def test_missing_required_array_is_refused(self) -> None:
        arrays, metadata = make_kinematics()
        del arrays["palm_quaternion_wxyz"]
        with self.assertRaises(GloveInputError):
            extract_glove_sequence(arrays, metadata, "synthetic")


class TestImuOrientationPassthrough(unittest.TestCase):
    """11. Orientation is copied with no convention mutation."""

    def test_matrix_and_quaternion_are_bitwise_copies(self) -> None:
        arrays, metadata = make_kinematics()
        sequence = extract_glove_sequence(arrays, metadata, "synthetic")
        np.testing.assert_array_equal(
            sequence.imu_rotation_matrix, arrays["palm_rotation_matrix"]
        )
        np.testing.assert_array_equal(
            sequence.imu_quaternion_wxyz, arrays["palm_quaternion_wxyz"]
        )

    def test_no_sign_flip_or_renormalization_is_applied(self) -> None:
        arrays, metadata = make_kinematics()
        arrays["palm_quaternion_wxyz"][0, 0] = np.array(
            [0.5, 0.5, 0.5, 0.5], dtype=np.float32
        )
        sequence = extract_glove_sequence(arrays, metadata, "synthetic")
        np.testing.assert_array_equal(
            sequence.imu_quaternion_wxyz[0, 0], np.array([0.5, 0.5, 0.5, 0.5], np.float32)
        )

    def test_no_evaluation_basis_mapping_is_applied_to_either_track(self) -> None:
        """The LEFT/RIGHT comparison bases are evaluation-only."""

        arrays, metadata = make_kinematics()
        arrays["palm_rotation_matrix"][0, 0] = np.eye(3, dtype=np.float32)
        arrays["palm_rotation_matrix"][0, 1] = np.eye(3, dtype=np.float32)
        sequence = extract_glove_sequence(arrays, metadata, "synthetic")
        for track in (0, 1):
            np.testing.assert_array_equal(
                sequence.imu_rotation_matrix[0, track], np.eye(3, dtype=np.float32)
            )

    def test_output_is_a_copy_not_a_view(self) -> None:
        arrays, metadata = make_kinematics()
        sequence = extract_glove_sequence(arrays, metadata, "synthetic")
        sequence.imu_rotation_matrix[0, 0, 0, 0] = 99.0
        self.assertNotEqual(float(arrays["palm_rotation_matrix"][0, 0, 0, 0]), 99.0)


class TestAdcEncoding(unittest.TestCase):
    """12. Optional 12-bit transfer, including both rails."""

    def test_boundaries(self) -> None:
        normalized = np.array([0.0, 0.5, 1.0])
        counts = to_adc_12bit(normalized, np.ones(3, bool))
        np.testing.assert_array_equal(counts, [0, 2048, ADC_MAX])

    def test_full_scale_is_4095_not_the_old_prototype_range(self) -> None:
        self.assertEqual(ADC_MAX, 4095)
        counts = to_adc_12bit(np.array([1.0]), np.ones(1, bool))
        self.assertEqual(int(counts[0]), 4095)
        # the retired prototype's ~850-1700 window must not appear as the scale
        self.assertNotEqual(int(counts[0]), 1700)

    def test_invalid_channels_carry_the_sentinel(self) -> None:
        counts = to_adc_12bit(np.array([0.5, np.nan]), np.array([True, False]))
        self.assertEqual(int(counts[1]), ADC_INVALID_SENTINEL)
        self.assertNotIn(int(counts[1]), range(0, ADC_MAX + 1))

    def test_rounding_is_half_up_and_deterministic(self) -> None:
        normalized = np.array([0.5 / ADC_MAX, 1.5 / ADC_MAX, 2.5 / ADC_MAX])
        counts = to_adc_12bit(normalized, np.ones(3, bool))
        np.testing.assert_array_equal(counts, [1, 2, 3])

    def test_float_mode_keeps_nan_for_invalid(self) -> None:
        values = to_adc_12bit(np.array([1.0, np.nan]), np.array([True, False]), integer=False)
        self.assertAlmostEqual(float(values[0]), float(ADC_MAX), places=3)
        self.assertTrue(bool(np.isnan(values[1])))

    def test_adc_is_monotonic_in_the_angle(self) -> None:
        angles = np.array([0.0, 10.0, 45.0, 90.0, 179.0, 180.0])
        normalized, _ = normalize_angles(angles)
        counts = to_adc_12bit(normalized, np.ones(angles.size, bool))
        self.assertTrue(np.all(np.diff(counts) > 0))


class TestSensorLayout(unittest.TestCase):
    """13-16. The sensor-count contract and display markers."""

    def test_exactly_nineteen_hall_sensor_ids(self) -> None:
        self.assertEqual(len(HALL_SENSOR_IDS), 19)
        self.assertEqual(len(set(HALL_SENSOR_IDS)), 19)
        self.assertEqual(EXPECTED_HALL_SENSORS, 19)

    def test_the_hall_ids_are_exactly_the_specified_set(self) -> None:
        expected = {
            f"H_{finger.upper()}_{chain.upper()}"
            for finger in FINGER_ORDER
            for chain in CHAIN_ORDER
        } | {
            "H_SPREAD_THUMB_INDEX", "H_SPREAD_INDEX_MIDDLE",
            "H_SPREAD_MIDDLE_RING", "H_SPREAD_RING_PINKY",
        }
        self.assertEqual(set(HALL_SENSOR_IDS), expected)

    def test_fifteen_bend_and_four_spread(self) -> None:
        self.assertEqual(len(BEND_SENSOR_IDS), 15)
        self.assertEqual(len(SPREAD_SENSOR_IDS), 4)
        self.assertEqual(EXPECTED_BEND_SENSORS, 15)
        self.assertEqual(EXPECTED_SPREAD_SENSORS, 4)

    def test_exactly_one_palm_imu(self) -> None:
        self.assertEqual(len(IMU_SENSOR_IDS), 1)
        self.assertEqual(IMU_SENSOR_IDS[0], "IMU_PALM")
        self.assertEqual(EXPECTED_IMU_SENSORS, 1)

    def test_twenty_logical_packages(self) -> None:
        self.assertEqual(len(SENSOR_LAYOUT), 20)
        self.assertEqual(EXPECTED_SENSOR_PACKAGES, 20)

    def test_every_hall_entry_uses_marker_H(self) -> None:
        for sensor_id in HALL_SENSOR_IDS:
            self.assertEqual(sensor_by_id(sensor_id).display_marker, MARKER_HALL)
            self.assertEqual(sensor_by_id(sensor_id).display_marker, "H")

    def test_imu_entry_uses_marker_IMU(self) -> None:
        self.assertEqual(sensor_by_id("IMU_PALM").display_marker, MARKER_IMU)
        self.assertEqual(sensor_by_id("IMU_PALM").display_marker, "IMU")

    def test_no_hall_sensor_is_marked_IMU_and_vice_versa(self) -> None:
        markers = {s.sensor_id: s.display_marker for s in SENSOR_LAYOUT}
        self.assertEqual(sum(1 for m in markers.values() if m == "H"), 19)
        self.assertEqual(sum(1 for m in markers.values() if m == "IMU"), 1)

    def test_every_entry_has_the_visualization_fields(self) -> None:
        for sensor in SENSOR_LAYOUT:
            for field in ("sensor_id", "sensor_type", "role", "logical_location",
                          "display_marker", "description"):
                value = getattr(sensor, field)
                self.assertTrue(value, f"{sensor.sensor_id} missing {field}")
            self.assertTrue(sensor.finger is not None or sensor.pair is not None
                            or sensor.sensor_type == "imu_package")

    def test_layout_locations_are_unique(self) -> None:
        locations = [s.logical_location for s in SENSOR_LAYOUT]
        self.assertEqual(len(locations), len(set(locations)))

    def test_bend_sensors_map_to_distinct_array_slots(self) -> None:
        """Three joints of one finger are never aggregated into one value."""

        slots = [sensor_by_id(i).array_index for i in BEND_SENSOR_IDS]
        self.assertEqual(len(slots), 15)
        self.assertEqual(len(set(slots)), 15)

    def test_layout_document_is_json_serializable_and_complete(self) -> None:
        document = layout_document()
        text = json.dumps(document)
        self.assertGreater(len(text), 100)
        self.assertEqual(len(document["sensors"]), 20)
        self.assertEqual(document["per_hand_counts"]["hall_sensors_total"], 19)
        self.assertEqual(document["per_hand_counts"]["logical_sensing_packages"], 20)

    def test_sensor_id_maps_to_the_correct_array_slot(self) -> None:
        arrays, metadata = make_kinematics(frames=2)
        arrays["flexion_deg"][:, :, FINGER_ORDER.index("ring"), CHAIN_ORDER.index("middle")] = 77.0
        sequence = extract_glove_sequence(arrays, metadata, "synthetic")
        spec = sensor_by_id("H_RING_MIDDLE")
        finger_index, chain_index = spec.array_index
        self.assertAlmostEqual(
            float(sequence.bend_angle_deg[0, 0, finger_index, chain_index]), 77.0, places=4
        )


class TestAngularVelocity(unittest.TestCase):
    """19. Derived temporal signal must never bridge a gap."""

    def test_known_constant_rate_is_recovered(self) -> None:
        rotations = np.stack([rotation_about((0, 0, 1), 10.0 * k) for k in range(4)])
        timestamps = np.arange(4) / 30.0
        omega, valid = angular_velocity_body_frame(
            rotations, timestamps, np.ones(4, bool), np.arange(4)
        )
        self.assertFalse(bool(valid[0]))
        self.assertTrue(bool(valid[1:].all()))
        expected = np.radians(10.0) * 30.0
        for k in range(1, 4):
            self.assertAlmostEqual(float(omega[k, 2]), expected, places=9)

    def test_first_frame_is_always_invalid(self) -> None:
        rotations = np.stack([np.eye(3)] * 3)
        omega, valid = angular_velocity_body_frame(
            rotations, np.arange(3) / 30.0, np.ones(3, bool), np.arange(3)
        )
        self.assertFalse(bool(valid[0]))
        self.assertTrue(bool(np.isnan(omega[0]).all()))

    def test_invalid_orientation_is_never_bridged(self) -> None:
        rotations = np.stack([rotation_about((0, 0, 1), 10.0 * k) for k in range(5)])
        valid_in = np.array([True, True, False, True, True])
        omega, valid = angular_velocity_body_frame(
            rotations, np.arange(5) / 30.0, valid_in, np.arange(5)
        )
        self.assertTrue(bool(valid[1]))
        self.assertFalse(bool(valid[2]))
        self.assertFalse(bool(valid[3]), "must not bridge across the invalid frame")
        self.assertTrue(bool(valid[4]))

    def test_non_adjacent_frame_indices_are_never_bridged(self) -> None:
        rotations = np.stack([rotation_about((0, 0, 1), 10.0 * k) for k in range(3)])
        frame_index = np.array([0, 1, 9])  # a gap in the source frames
        omega, valid = angular_velocity_body_frame(
            rotations, np.array([0.0, 1 / 30, 9 / 30]), np.ones(3, bool), frame_index
        )
        self.assertTrue(bool(valid[1]))
        self.assertFalse(bool(valid[2]))

    def test_actual_timestamp_delta_is_used(self) -> None:
        rotations = np.stack([np.eye(3), rotation_about((0, 0, 1), 10.0)])
        fast, _ = angular_velocity_body_frame(
            rotations, np.array([0.0, 1 / 60]), np.ones(2, bool), np.arange(2)
        )
        slow, _ = angular_velocity_body_frame(
            rotations, np.array([0.0, 1 / 30]), np.ones(2, bool), np.arange(2)
        )
        self.assertAlmostEqual(float(fast[1, 2]), 2.0 * float(slow[1, 2]), places=9)

    def test_non_positive_dt_is_rejected(self) -> None:
        rotations = np.stack([np.eye(3), rotation_about((0, 0, 1), 10.0)])
        _, valid = angular_velocity_body_frame(
            rotations, np.array([0.0, 0.0]), np.ones(2, bool), np.arange(2)
        )
        self.assertFalse(bool(valid[1]))

    def test_stationary_hand_reads_zero_rate(self) -> None:
        rotations = np.stack([np.eye(3)] * 3)
        omega, valid = angular_velocity_body_frame(
            rotations, np.arange(3) / 30.0, np.ones(3, bool), np.arange(3)
        )
        self.assertTrue(bool(valid[1:].all()))
        np.testing.assert_allclose(omega[1:], 0.0, atol=1e-12)

    def test_large_jump_is_reported_not_smoothed(self) -> None:
        """TASK-005 has known large orientation jumps; they must stay visible."""

        rotations = np.stack([np.eye(3), rotation_about((0, 0, 1), 102.0)])
        omega, valid = angular_velocity_body_frame(
            rotations, np.array([0.0, 1 / 30]), np.ones(2, bool), np.arange(2)
        )
        self.assertTrue(bool(valid[1]))
        self.assertAlmostEqual(
            float(np.linalg.norm(omega[1])), np.radians(102.0) * 30.0, places=6
        )

    def test_sequence_level_gyro_respects_palm_validity(self) -> None:
        arrays, metadata = make_kinematics(frames=5)
        arrays["valid_palm_frame"][2, 0] = False
        sequence = extract_glove_sequence(arrays, metadata, "synthetic")
        self.assertFalse(bool(sequence.imu_angular_velocity_valid[2, 0]))
        self.assertFalse(bool(sequence.imu_angular_velocity_valid[3, 0]))
        self.assertTrue(bool(sequence.imu_angular_velocity_valid[3, 1]))


class TestAccelerometerDeferred(unittest.TestCase):
    """No accelerometer is fabricated."""

    def test_accelerometer_is_explicitly_deferred(self) -> None:
        report = accelerometer_feasibility()
        self.assertEqual(report["accelerometer"], "DEFER ACCELEROMETER")
        self.assertGreaterEqual(len(report["reasons"]), 3)

    def test_no_accelerometer_array_is_emitted(self) -> None:
        self.assertFalse(any("accel" in name.lower() for name in ARRAY_ORDER))


class TestDeterminismAndRoundTrip(unittest.TestCase):
    """17. Deterministic output and faithful persistence."""

    def test_repeated_extraction_is_identical(self) -> None:
        arrays, metadata = make_kinematics()
        arrays["adjacent_spread_deg"][1, 0, 2] = np.nan
        first = extract_glove_sequence(arrays, metadata, "synthetic")
        second = extract_glove_sequence(arrays, metadata, "synthetic")
        for name in ARRAY_ORDER:
            left, right = getattr(first, name), getattr(second, name)
            if np.asarray(left).dtype.kind == "f":
                np.testing.assert_array_equal(
                    np.nan_to_num(left, nan=-9e9), np.nan_to_num(right, nan=-9e9)
                )
            else:
                np.testing.assert_array_equal(left, right)

    def test_npz_round_trip_preserves_every_array(self) -> None:
        arrays, metadata = make_kinematics()
        arrays["flexion_deg"][0, 0, 0, 0] = np.nan
        sequence = extract_glove_sequence(arrays, metadata, "synthetic")
        meta = build_metadata(
            sequence, kinematics_dir=Path("/nonexistent"), kinematics_sha256="0" * 64,
            kinematics_metadata=metadata, implementation_commit="test",
        )
        with tempfile.TemporaryDirectory() as directory:
            save_glove_sequence(directory, sequence, meta)
            loaded, loaded_meta = load_glove_sequence(directory)
            layout_path = Path(directory) / "sensor_layout.json"
            self.assertTrue(layout_path.is_file())
            layout = json.loads(layout_path.read_text())
        self.assertEqual(set(loaded), set(ARRAY_ORDER))
        for name in ARRAY_ORDER:
            expected = getattr(sequence, name)
            if np.asarray(expected).dtype.kind == "f":
                np.testing.assert_array_equal(
                    np.nan_to_num(loaded[name], nan=-9e9),
                    np.nan_to_num(expected, nan=-9e9),
                )
            else:
                np.testing.assert_array_equal(loaded[name], expected)
        self.assertEqual(loaded_meta["track_order"], ["LEFT", "RIGHT"])
        self.assertEqual(loaded_meta["sensor_counts_per_hand"]["hall_sensors_total"], 19)
        self.assertEqual(len(layout["sensors"]), 20)

    def test_npz_contains_no_pickled_objects(self) -> None:
        arrays, metadata = make_kinematics()
        sequence = extract_glove_sequence(arrays, metadata, "synthetic")
        meta = build_metadata(
            sequence, kinematics_dir=Path("/nonexistent"), kinematics_sha256="0" * 64,
            kinematics_metadata=metadata, implementation_commit="test",
        )
        with tempfile.TemporaryDirectory() as directory:
            save_glove_sequence(directory, sequence, meta)
            with np.load(Path(directory) / "virtual_glove.npz", allow_pickle=False) as data:
                for key in data.files:
                    self.assertNotEqual(data[key].dtype, np.object_)

    def test_output_shapes_match_the_contract(self) -> None:
        arrays, metadata = make_kinematics(frames=7)
        sequence = extract_glove_sequence(arrays, metadata, "synthetic")
        self.assertEqual(sequence.bend_angle_deg.shape, (7, 2, 5, 3))
        self.assertEqual(sequence.bend_normalized.shape, (7, 2, 5, 3))
        self.assertEqual(sequence.bend_valid.shape, (7, 2, 5, 3))
        self.assertEqual(sequence.spread_angle_deg.shape, (7, 2, 4))
        self.assertEqual(sequence.spread_normalized.shape, (7, 2, 4))
        self.assertEqual(sequence.spread_valid.shape, (7, 2, 4))
        self.assertEqual(sequence.imu_rotation_matrix.shape, (7, 2, 3, 3))
        self.assertEqual(sequence.imu_quaternion_wxyz.shape, (7, 2, 4))
        self.assertEqual(sequence.palm_imu_valid.shape, (7, 2))
        self.assertEqual(sequence.imu_angular_velocity_rad_s.shape, (7, 2, 3))


class TestMlContractShape(unittest.TestCase):
    """The 23 primary channels exist, but no tensor is built here."""

    def test_primary_channel_count_is_available_per_hand(self) -> None:
        arrays, metadata = make_kinematics(frames=3)
        sequence = extract_glove_sequence(arrays, metadata, "synthetic")
        per_hand = (
            sequence.bend_normalized[0, 0].size
            + sequence.spread_normalized[0, 0].size
            + sequence.imu_quaternion_wxyz[0, 0].size
        )
        self.assertEqual(per_hand, 23)

    def test_no_stacked_training_tensor_is_produced(self) -> None:
        self.assertFalse(any(
            token in name for name in ARRAY_ORDER
            for token in ("feature", "tensor", "label", "sequence_x", "dataset")
        ))


if __name__ == "__main__":
    unittest.main()
