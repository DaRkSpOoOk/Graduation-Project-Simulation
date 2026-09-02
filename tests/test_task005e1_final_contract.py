"""Tests for the versioned TASK-005E1 final contract and B2 truth."""

from __future__ import annotations

import unittest

import numpy as np

from evaluation.kinematics.benchmark_contract import CONTRACT_TOLERANCES
from evaluation.kinematics.final_contract import (
    FINAL_CONTRACT_TOLERANCES,
    FINAL_CONTRACT_VERSION,
    FINAL_SPREAD_MIN_PROJECTED_ANGLE_DEG,
    SIDE_ORIENTATION_MAPPINGS,
    build_final_catalog,
    build_final_sequence,
    map_production_sequence_rotations,
)


def _direct_unsigned_angle(first: np.ndarray, second: np.ndarray) -> float:
    first = first / np.linalg.norm(first)
    second = second / np.linalg.norm(second)
    return float(np.degrees(np.arctan2(np.linalg.norm(np.cross(first, second)), np.dot(first, second))))


class TestTask005E1FinalContract(unittest.TestCase):
    def test_versioned_catalog_preserves_frozen_case_split(self) -> None:
        catalog = build_final_catalog()
        self.assertEqual(len(catalog), 86)
        self.assertEqual(sum(case.expected_valid for case in catalog), 80)
        self.assertEqual(sum(not case.expected_valid for case in catalog), 6)
        self.assertEqual(FINAL_CONTRACT_VERSION, "TASK-005-final-v2")

    def test_proximal_truth_is_direct_output_geometry(self) -> None:
        case = next(case for case in build_final_catalog() if case.case_id == "neutral")
        source = case.generate()
        final = build_final_sequence(case)
        joints = source.joints[0, 0]
        direct = _direct_unsigned_angle(joints[1] - joints[0], joints[2] - joints[1])
        self.assertAlmostEqual(final.flexion_deg[0, 0, 0, 0], direct, places=10)
        # The historical B truth was the requested first bend (zero); the
        # final truth intentionally reflects the observed wrist-to-base turn.
        self.assertAlmostEqual(source.flexion_deg[0, 0, 0, 0], 0.0, places=12)
        self.assertGreater(final.flexion_deg[0, 0, 0, 0], 1.0)

    def test_spread_truth_uses_actual_proximal_phalanx_direction(self) -> None:
        case = next(case for case in build_final_catalog() if case.case_id == "adversarial_near_180")
        source = case.generate()
        final = build_final_sequence(case)
        self.assertTrue(np.allclose(source.adjacent_spread_deg, 10.0))
        self.assertTrue(
            np.allclose(
                final.adjacent_spread_deg,
                np.asarray([[[170.0, 170.0, 10.0, 10.0], [170.0, 170.0, 10.0, 10.0]]]),
                atol=1e-10,
                rtol=0.0,
            )
        )

    def test_conditioning_is_channel_level_and_explicit(self) -> None:
        conditioning = {
            "single_thumb_joint0_90deg",
            "single_index_joint0_90deg",
            "single_middle_joint0_90deg",
            "single_ring_joint0_90deg",
            "single_pinky_joint0_90deg",
            "multi_curl_pinky",
        }
        found: set[str] = set()
        for case in build_final_catalog():
            if not case.expected_valid:
                continue
            final = build_final_sequence(case)
            if np.isnan(final.adjacent_spread_deg).any():
                found.add(case.case_id)
                self.assertTrue(final.valid_palm_frame.all())
                self.assertFalse(final.valid_kinematics.all())
                self.assertTrue(np.isfinite(final.flexion_deg).all())
                self.assertTrue(np.isfinite(final.palm_rotation_matrix).all())
                self.assertTrue(np.isfinite(final.palm_quaternion_wxyz).all())
                self.assertEqual(
                    final.spread_direction_degenerate.shape,
                    (1, 2, 5),
                )
        self.assertEqual(found, conditioning)
        self.assertEqual(FINAL_SPREAD_MIN_PROJECTED_ANGLE_DEG, 15.0)

    def test_side_mappings_are_fixed_proper_and_distinct(self) -> None:
        self.assertNotEqual(
            SIDE_ORIENTATION_MAPPINGS["LEFT"].tolist(),
            SIDE_ORIENTATION_MAPPINGS["RIGHT"].tolist(),
        )
        for matrix in SIDE_ORIENTATION_MAPPINGS.values():
            self.assertTrue(np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-12, rtol=0.0))
            self.assertAlmostEqual(float(np.linalg.det(matrix)), 1.0, places=12)

    def test_side_mapping_reconciles_known_fixture_bases(self) -> None:
        # Use two arbitrary global rotations to prove the mapping is constant,
        # rather than fitted to one orientation case.
        from evaluation.kinematics.synthetic_hand import rotation_matrix_xyz

        for rotation in (
            np.eye(3),
            rotation_matrix_xyz(25.0, -40.0, 70.0),
        ):
            production_basis = np.array(
                [[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
                dtype=np.float64,
            )
            production = np.asarray(
                [rotation @ production_basis, rotation @ production_basis],
                dtype=np.float64,
            )[None, ...]
            mapped = map_production_sequence_rotations(production)
            expected = np.asarray(
                [
                    [rotation @ np.diag((-1.0, 1.0, -1.0)), rotation],
                ],
                dtype=np.float64,
            )
            self.assertTrue(np.allclose(mapped, expected, atol=1e-12, rtol=0.0))

    def test_invalid_coincident_mcp_expectation_is_not_weakened(self) -> None:
        case = next(case for case in build_final_catalog() if case.case_id == "degenerate_coincident_mcp")
        final = build_final_sequence(case)
        self.assertFalse(final.expected_valid)
        self.assertTrue(any("coincident" in reason for reason in final.invalid_reasons))
        self.assertFalse(final.valid_kinematics.any())

    def test_original_task005b_truth_is_not_mutated(self) -> None:
        case = next(case for case in build_final_catalog() if case.case_id == "neutral")
        original = case.generate()
        before_flex = original.flexion_deg.copy()
        before_spread = original.adjacent_spread_deg.copy()
        _ = build_final_sequence(case)
        self.assertTrue(np.array_equal(original.flexion_deg, before_flex))
        self.assertTrue(np.array_equal(original.adjacent_spread_deg, before_spread))

    def test_locked_tolerances_are_unchanged(self) -> None:
        self.assertEqual(FINAL_CONTRACT_TOLERANCES, CONTRACT_TOLERANCES)
        self.assertEqual(FINAL_CONTRACT_TOLERANCES["known_flexion_abs_error_deg"], 1.0)
        self.assertEqual(FINAL_CONTRACT_TOLERANCES["known_spread_abs_error_deg"], 1.0)
        self.assertEqual(FINAL_CONTRACT_TOLERANCES["known_orientation_error_deg"], 1.0)
        self.assertEqual(FINAL_CONTRACT_TOLERANCES["rotation_matrix_orthogonality"], 1e-5)
        self.assertEqual(FINAL_CONTRACT_TOLERANCES["rotation_matrix_determinant"], 1e-5)
        self.assertEqual(FINAL_CONTRACT_TOLERANCES["quaternion_norm"], 1e-5)
        self.assertEqual(FINAL_CONTRACT_TOLERANCES["matrix_quaternion_consistency"], 1e-5)


if __name__ == "__main__":
    unittest.main()
