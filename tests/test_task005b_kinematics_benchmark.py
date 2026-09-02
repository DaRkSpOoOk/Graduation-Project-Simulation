import json
import math
import unittest
from pathlib import Path

import numpy as np

from evaluation.kinematics import (
    CONTRACT_TOLERANCES,
    ContractError,
    FrameParameters,
    GeometryOptions,
    KinematicsResult,
    build_benchmark_catalog,
    coerce_result,
    evaluate_sequence,
    frame_parameters_from_dict,
    frame_parameters_to_dict,
    generate_hand,
    generate_sequence,
    geometry_validity,
    mirror_points,
    rotation_matrix_axis,
    rotation_matrix_xyz,
    validate_result,
)
from evaluation.kinematics.synthetic_hand import FINGER_JOINTS, FINGER_NAMES, quaternion_matrix_wxyz


def _signed_turn_degrees(incoming: np.ndarray, outgoing: np.ndarray, axis: np.ndarray) -> float:
    return math.degrees(math.atan2(float(np.dot(np.cross(incoming, outgoing), axis)), float(np.dot(incoming, outgoing))))


def _segment_directions(hand, finger: int) -> list[np.ndarray]:
    chain = FINGER_JOINTS[finger]
    return [hand.joints[second] - hand.joints[first] for first, second in zip(chain, chain[1:])]


class TestTask005BSyntheticGenerator(unittest.TestCase):
    def test_all_single_bends_have_requested_geometry(self):
        cases = [case for case in build_benchmark_catalog() if case.category == "single_bend"]
        self.assertEqual(len(cases), 45)
        for case in cases:
            sequence = case.generate()
            hand = generate_hand(
                side="RIGHT",
                flexion_deg=case.frames[0].flexion_deg,
                adjacent_spread_deg=case.frames[0].adjacent_spread_deg,
            )
            finger = next(index for index, name in enumerate(FINGER_NAMES) if f"_{name}_" in case.case_id)
            joint = int(case.case_id.split("_joint")[1].split("_")[0])
            requested = float(case.frames[0].flexion_deg[finger][joint])
            straight = np.array(
                [math.sin(math.radians(hand.base_heading_deg[finger])), math.cos(math.radians(hand.base_heading_deg[finger])), 0.0]
            )
            axis = np.cross(straight, np.array([0.0, 0.0, 1.0]))
            directions = _segment_directions(hand, finger)
            incoming = straight if joint == 0 else directions[joint - 1]
            measured = _signed_turn_degrees(incoming, directions[joint], axis)
            self.assertAlmostEqual(measured, requested, places=8, msg=case.case_id)
            self.assertTrue(sequence.expected_valid)

    def test_catalog_contains_required_case_families(self):
        cases = build_benchmark_catalog()
        self.assertEqual(len(cases), 86)
        categories = {case.category for case in cases}
        self.assertTrue({"neutral", "single_bend", "multi_joint_curl", "independent_fingers", "spread", "mirror", "translation", "scale", "quaternion_orientation", "degenerate", "adversarial"} <= categories)
        self.assertEqual(sum(case.category == "degenerate" for case in cases), 5)
        self.assertEqual(sum(case.category == "adversarial" for case in cases), 8)

    def test_spread_geometry_is_the_requested_unsigned_gap(self):
        for case in build_benchmark_catalog():
            if case.category != "spread":
                continue
            hand = generate_hand(adjacent_spread_deg=case.frames[0].adjacent_spread_deg)
            headings = hand.base_heading_deg
            measured = np.abs(np.diff(headings))
            expected = np.asarray(case.frames[0].adjacent_spread_deg, dtype=float)
            np.testing.assert_allclose(measured, expected, atol=1e-10)

    def test_rigid_transform_is_applied_exactly(self):
        flexion = np.asarray([[20.0, 40.0, 60.0], [0, 0, 0], [10, 20, 30], [0, 0, 0], [5, 15, 25]])
        base = generate_hand(side="RIGHT", flexion_deg=flexion)
        rotation = rotation_matrix_xyz(35.0, -20.0, 110.0)
        translation = np.array([12.5, -7.25, 3.0])
        transformed = generate_hand(
            side="RIGHT",
            flexion_deg=flexion,
            global_rotation=rotation,
            translation=translation,
            scale=2.0,
        )
        expected_points = (2.0 * base.joints) @ rotation.T + translation
        np.testing.assert_allclose(transformed.joints, expected_points, atol=1e-12)
        np.testing.assert_allclose(transformed.palm_rotation_matrix, rotation @ base.palm_rotation_matrix, atol=1e-12)

    def test_mirror_is_deterministic_and_local_truth_is_equivalent(self):
        flexion = np.asarray([[60.0, 45.0, 30.0], [30.0, 45.0, 60.0], [0, 0, 0], [20, 10, 5], [0, 0, 0]])
        right = generate_hand(side="RIGHT", flexion_deg=flexion, adjacent_spread_deg=(15, 20, 25, 30))
        left = generate_hand(side="LEFT", flexion_deg=flexion, adjacent_spread_deg=(15, 20, 25, 30))
        np.testing.assert_allclose(left.joints, mirror_points(right.joints), atol=1e-12)
        np.testing.assert_allclose(mirror_points(mirror_points(right.joints)), right.joints, atol=1e-12)
        np.testing.assert_allclose(left.flexion_deg, right.flexion_deg, atol=0.0)
        np.testing.assert_allclose(left.adjacent_spread_deg, right.adjacent_spread_deg, atol=0.0)

    def test_fixture_descriptor_round_trips(self):
        path = Path(__file__).resolve().parent / "fixtures/kinematics/task005b_representative_cases.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["track_order"], ["LEFT", "RIGHT"])
        self.assertEqual(payload["finger_order"], list(FINGER_NAMES))
        for fixture in payload["cases"]:
            parameters = frame_parameters_from_dict(fixture["parameters"])
            encoded = frame_parameters_to_dict(parameters)
            decoded = frame_parameters_from_dict(encoded)
            np.testing.assert_allclose(np.asarray(decoded.flexion_deg), np.asarray(parameters.flexion_deg))
            np.testing.assert_allclose(np.asarray(decoded.global_rotation), np.asarray(parameters.global_rotation), atol=1e-7)
            sequence = generate_sequence(fixture["case_id"], (decoded,))
            self.assertEqual(sequence.expected_valid, fixture["expected_valid"])

    def test_degenerate_cases_are_intentionally_invalid(self):
        cases = [case for case in build_benchmark_catalog() if case.category == "degenerate" or case.case_id == "adversarial_tiny_nonzero_bone"]
        self.assertEqual(len(cases), 6)
        reasons = set()
        for case in cases:
            sequence = case.generate()
            self.assertFalse(sequence.expected_valid, case.case_id)
            reasons.update(sequence.invalid_reasons)
        self.assertTrue(any(reason.endswith("zero_or_tiny_bone") for reason in reasons))
        self.assertTrue(any(reason.endswith("collinear_or_coincident_palm") for reason in reasons))
        self.assertTrue(any(reason.endswith("non_finite_joint") for reason in reasons))

    def test_adversarial_case_parameters_are_present(self):
        cases = {case.case_id: case for case in build_benchmark_catalog()}
        self.assertAlmostEqual(float(cases["adversarial_almost_straight"].frames[0].flexion_deg[1, 0]), 0.1)
        self.assertAlmostEqual(float(cases["adversarial_near_180"].frames[0].flexion_deg[1, 0]), 179.9)
        self.assertEqual(cases["adversarial_finger_crossing"].frames[0].adjacent_spread_deg[1], -140.0)
        self.assertEqual(cases["adversarial_thumb_opposition"].frames[0].flexion_deg[0, 0], 60.0)


class TestTask005BContract(unittest.TestCase):
    def _valid_result(self):
        sequence = generate_sequence("contract", (FrameParameters(), FrameParameters()))
        result = KinematicsResult(
            sequence.flexion_deg,
            sequence.adjacent_spread_deg,
            sequence.palm_rotation_matrix,
            sequence.palm_quaternion_wxyz,
        )
        return sequence, result

    def test_valid_result_has_required_shapes_and_passes(self):
        sequence, result = self._valid_result()
        validated = validate_result(result, expected_frames=2)
        self.assertEqual(validated.flexion_deg.shape, (2, 2, 5, 3))
        score = evaluate_sequence(result, sequence)
        self.assertTrue(score["flexion_pass"])
        self.assertTrue(score["spread_pass"])
        self.assertTrue(score["orientation_pass"])
        self.assertTrue(score["quaternion_pass"])
        self.assertLessEqual(score["max_flexion_error_deg"], CONTRACT_TOLERANCES["known_flexion_abs_error_deg"])

    def test_mapping_is_a_clear_later_adapter_interface(self):
        sequence, result = self._valid_result()
        mapping = {
            "flexion_deg": result.flexion_deg,
            "adjacent_spread_deg": result.adjacent_spread_deg,
            "palm_rotation_matrix": result.palm_rotation_matrix,
            "palm_quaternion_wxyz": result.palm_quaternion_wxyz,
        }
        coerced = coerce_result(mapping)
        np.testing.assert_allclose(coerced.flexion_deg, sequence.flexion_deg)

    def test_bad_shapes_and_nonfinite_values_are_rejected(self):
        sequence, result = self._valid_result()
        with self.assertRaises(ContractError):
            validate_result({"flexion_deg": result.flexion_deg[:, :, :, :2], "adjacent_spread_deg": result.adjacent_spread_deg, "palm_rotation_matrix": result.palm_rotation_matrix, "palm_quaternion_wxyz": result.palm_quaternion_wxyz})
        nonfinite = result.flexion_deg.copy()
        nonfinite[0, 0, 0, 0] = np.nan
        with self.assertRaises(ContractError):
            validate_result(KinematicsResult(nonfinite, result.adjacent_spread_deg, result.palm_rotation_matrix, result.palm_quaternion_wxyz))

    def test_bad_rotation_and_quaternion_are_rejected(self):
        sequence, result = self._valid_result()
        bad_matrix = result.palm_rotation_matrix.copy()
        bad_matrix[0, 0, 0, 0] += 0.1
        with self.assertRaises(ContractError):
            validate_result(KinematicsResult(result.flexion_deg, result.adjacent_spread_deg, bad_matrix, result.palm_quaternion_wxyz))
        bad_quaternion = result.palm_quaternion_wxyz.copy()
        bad_quaternion[0, 0, 0] = 2.0
        with self.assertRaises(ContractError):
            validate_result(KinematicsResult(result.flexion_deg, result.adjacent_spread_deg, result.palm_rotation_matrix, bad_quaternion))

    def test_known_axis_quaternions_have_wxyz_order(self):
        expected = {
            "X": np.array([math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0]),
            "Y": np.array([math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0]),
            "Z": np.array([math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]),
        }
        for axis, quaternion in expected.items():
            rotation = rotation_matrix_axis(axis, 90.0)
            sequence = generate_sequence(
                axis,
                (FrameParameters(global_rotation=rotation),),
            )
            # Track order is LEFT, RIGHT; the RIGHT track has the global
            # rotation directly, while LEFT also includes the side basis.
            np.testing.assert_allclose(sequence.palm_quaternion_wxyz[0, 1], quaternion, atol=1e-8)
            np.testing.assert_allclose(quaternion_matrix_wxyz(quaternion), rotation, atol=1e-8)

    def test_orientation_error_is_rigid_and_scale_independent_in_truth(self):
        base = generate_sequence("base", (FrameParameters(),))
        transformed = generate_sequence(
            "transformed",
            (FrameParameters(global_rotation=rotation_matrix_xyz(20, 30, 40), translation=(100, -50, 12), scale=5.0),),
        )
        np.testing.assert_allclose(transformed.flexion_deg, base.flexion_deg)
        np.testing.assert_allclose(transformed.adjacent_spread_deg, base.adjacent_spread_deg)
        self.assertFalse(np.allclose(transformed.palm_rotation_matrix, base.palm_rotation_matrix))

    def test_invalid_geometry_must_not_be_scored_as_plausible_numeric_truth(self):
        invalid = build_benchmark_catalog()[1]
        # Find a declared invalid case without relying on production formulas.
        invalid = next(case for case in build_benchmark_catalog() if case.case_id == "degenerate_nan_joint")
        sequence = invalid.generate()
        self.assertFalse(sequence.expected_valid)
        valid = geometry_validity(generate_hand())
        self.assertEqual(valid[0], True)
        with self.assertRaises(ContractError):
            evaluate_sequence(
                KinematicsResult(
                    np.zeros((1, 2, 5, 3)),
                    np.zeros((1, 2, 4)),
                    np.tile(np.eye(3), (1, 2, 1, 1)),
                    np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (1, 2, 1)),
                ),
                sequence,
            )


if __name__ == "__main__":
    unittest.main()
