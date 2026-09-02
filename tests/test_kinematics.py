"""Synthetic tests for TASK-005A hand kinematics.

Every fixture is built in memory from explicit geometry. Nothing here needs the
WiLoR checkpoint, MANO assets, KArSL videos, or any pilot output.

The synthetic hand is a flat canonical RIGHT hand: wrist at the origin, fingers
extending along +y, palm plane in z=0, and the thumb side at -x. Bends are
applied by rotating the remaining chain about an explicit axis, so the expected
angle is known by construction rather than read back from the implementation.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from kinematics import (
    ARRAY_ORDER,
    CHAIN_ORDER,
    FINGER_ORDER,
    SCHEMA_VERSION,
    SPREAD_PAIRS,
    TRACK_ORDER,
    build_metadata,
    build_palm_frame,
    compute_flexion,
    compute_hand_kinematics,
    compute_spread,
    extract_sequence,
    load_kinematics,
    save_kinematics,
)
from kinematics.extractor import POSE_BEARING_STATES
from kinematics.geometry import (
    MIN_PALM_LANDMARK_SEPARATION_RATIO,
    MIN_PROJECTED_NORM,
    MIN_SPREAD_PROJECTION_ANGLE_DEG,
    orthonormality_error,
    rotation_matrix_to_quaternion_wxyz,
)
from kinematics.layout import FINGER_CHAINS, N_JOINTS

MIRROR = np.diag([-1.0, 1.0, 1.0])

# x offset of each finger's MCP, thumb side at -x (canonical right hand).
_MCP_X = {"thumb": -0.045, "index": -0.030, "middle": 0.0, "ring": 0.018, "pinky": 0.038}
_MCP_Y = {"thumb": 0.035, "index": 0.090, "middle": 0.095, "ring": 0.086, "pinky": 0.082}
_SEGMENTS = {
    "thumb": (0.031, 0.027, 0.036),
    "index": (0.033, 0.022, 0.026),
    "middle": (0.032, 0.023, 0.026),
    "ring": (0.029, 0.025, 0.026),
    "pinky": (0.021, 0.019, 0.019),
}


def rotation_about(axis: np.ndarray, degrees: float) -> np.ndarray:
    """Rodrigues rotation matrix."""

    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    theta = np.radians(degrees)
    cross = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + np.sin(theta) * cross + (1.0 - np.cos(theta)) * (cross @ cross)


def straight_hand() -> np.ndarray:
    """Right hand whose every finger chain is *geometrically* straight.

    Each finger's phalanges continue along its own wrist->MCP direction, so all
    four bones of a chain are collinear and every turn angle is exactly zero.
    This is the fixture the "straight chain reads 0" requirement refers to.
    """

    joints = np.zeros((N_JOINTS, 3), dtype=np.float64)
    for finger in FINGER_ORDER:
        chain = FINGER_CHAINS[finger]
        mcp = np.array([_MCP_X[finger], _MCP_Y[finger], 0.0])
        direction = mcp / np.linalg.norm(mcp)
        joints[chain[1]] = mcp
        position = mcp
        for step, length in enumerate(_SEGMENTS[finger]):
            position = position + length * direction
            joints[chain[step + 2]] = position
    return joints


def flat_hand() -> np.ndarray:
    """Anatomically flat right hand: MCPs splayed, fingers parallel to +y.

    This is the realistic posture, and unlike ``straight_hand`` its proximal
    turn angles are NOT zero -- the wrist->MCP vector fans outward from the
    single wrist point and so is not collinear with the proximal phalanx. See
    ``TestProximalReferenceBone``.
    """

    joints = np.zeros((N_JOINTS, 3), dtype=np.float64)
    for finger in FINGER_ORDER:
        chain = FINGER_CHAINS[finger]
        position = np.array([_MCP_X[finger], _MCP_Y[finger], 0.0])
        joints[chain[1]] = position
        for step, length in enumerate(_SEGMENTS[finger]):
            position = position + np.array([0.0, length, 0.0])
            joints[chain[step + 2]] = position
    return joints


def bend_chain(
    joints: np.ndarray, finger: str, chain_index: int, degrees: float, axis=None
) -> np.ndarray:
    """Rotate the chain distal to one joint by ``degrees``.

    ``chain_index`` 0/1/2 selects the proximal/middle/distal joint. Rotating
    everything beyond the pivot rigidly leaves every other joint's turn angle
    untouched, so the bend is exactly isolated.

    The rotation axis must be PERPENDICULAR to the incoming bone for the
    resulting turn angle to equal ``degrees`` -- rotating a vector by theta
    about a non-perpendicular axis turns it by less than theta. By default the
    axis is therefore built as ``incoming_bone x palm_normal``, which flexes
    the finger out of the palm plane. An explicit ``axis`` is used as given
    (callers pass the palm normal itself to abduct within the plane, which is
    perpendicular to any bone lying in that plane).
    """

    joints = joints.copy()
    chain = FINGER_CHAINS[finger]
    pivot_index = chain[chain_index + 1]
    pivot = joints[pivot_index].copy()

    if axis is None:
        incoming = joints[pivot_index] - joints[chain[chain_index]]
        incoming = incoming / np.linalg.norm(incoming)
        axis_vector = np.cross(incoming, np.array([0.0, 0.0, 1.0]))
    else:
        axis_vector = np.asarray(axis, dtype=np.float64)

    rotation = rotation_about(axis_vector, degrees)
    for moved in chain[chain_index + 2:]:
        joints[moved] = pivot + rotation @ (joints[moved] - pivot)
    return joints


def flexion_of(joints: np.ndarray, finger: str, chain_index: int) -> float:
    flexion, _ = compute_flexion(joints)
    return float(flexion[FINGER_ORDER.index(finger), chain_index])


class TestStraightChain(unittest.TestCase):
    """1. A perfectly straight chain must read as zero bend."""

    def test_straight_hand_gives_zero_flexion_everywhere(self) -> None:
        flexion, flags = compute_flexion(straight_hand())
        self.assertEqual(flags, [])
        np.testing.assert_allclose(flexion, np.zeros_like(flexion), atol=1e-9)


class TestKnownAngles(unittest.TestCase):
    """2-4. Known bends must be recovered to within numerical noise."""

    def _check(self, degrees: float) -> None:
        joints = bend_chain(straight_hand(), "index", 1, degrees)
        self.assertAlmostEqual(flexion_of(joints, "index", 1), degrees, places=6)

    def test_30_degree_bend(self) -> None:
        self._check(30.0)

    def test_60_degree_bend(self) -> None:
        self._check(60.0)

    def test_90_degree_bend(self) -> None:
        self._check(90.0)

    def test_monotonic_in_bend_magnitude(self) -> None:
        previous = -1.0
        for degrees in (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 120.0, 150.0):
            value = flexion_of(bend_chain(straight_hand(), "index", 1, degrees), "index", 1)
            self.assertGreater(value, previous)
            previous = value

    def test_bend_is_positive_for_either_rotation_direction(self) -> None:
        """The convention is unsigned: a -70 deg rotation is still +70 of bend."""

        positive = flexion_of(bend_chain(straight_hand(), "index", 1, 70.0), "index", 1)
        negative = flexion_of(bend_chain(straight_hand(), "index", 1, -70.0), "index", 1)
        self.assertAlmostEqual(positive, 70.0, places=6)
        self.assertAlmostEqual(negative, 70.0, places=6)


class TestIsolatedJoints(unittest.TestCase):
    """5-7. A bend at one chain joint must not leak into the others."""

    def _isolated(self, chain_index: int) -> None:
        joints = bend_chain(straight_hand(), "middle", chain_index, 55.0)
        flexion, flags = compute_flexion(joints)
        self.assertEqual(flags, [])
        row = flexion[FINGER_ORDER.index("middle")]
        for index in range(len(CHAIN_ORDER)):
            expected = 55.0 if index == chain_index else 0.0
            self.assertAlmostEqual(row[index], expected, places=6)

    def test_isolated_proximal_bend(self) -> None:
        self._isolated(0)

    def test_isolated_middle_bend(self) -> None:
        self._isolated(1)

    def test_isolated_distal_bend(self) -> None:
        self._isolated(2)

    def test_other_fingers_are_untouched(self) -> None:
        joints = bend_chain(straight_hand(), "middle", 1, 55.0)
        flexion, _ = compute_flexion(joints)
        for finger in FINGER_ORDER:
            if finger == "middle":
                continue
            np.testing.assert_allclose(
                flexion[FINGER_ORDER.index(finger)], np.zeros(3), atol=1e-9
            )


class TestMultipleFingers(unittest.TestCase):
    """8. Independent bends on several fingers stay independent."""

    def test_three_fingers_bent_independently(self) -> None:
        expected = {"index": (0, 25.0), "ring": (1, 65.0), "pinky": (2, 100.0)}
        joints = straight_hand()
        for finger, (chain_index, degrees) in expected.items():
            joints = bend_chain(joints, finger, chain_index, degrees)
        flexion, flags = compute_flexion(joints)
        self.assertEqual(flags, [])
        for finger_index, finger in enumerate(FINGER_ORDER):
            for chain_index in range(len(CHAIN_ORDER)):
                want = 0.0
                if finger in expected and expected[finger][0] == chain_index:
                    want = expected[finger][1]
                self.assertAlmostEqual(flexion[finger_index, chain_index], want, places=6)


def _bent_hand() -> np.ndarray:
    joints = bend_chain(straight_hand(), "index", 1, 70.0)
    joints = bend_chain(joints, "ring", 0, 40.0)
    joints = bend_chain(joints, "thumb", 2, 35.0)
    return joints


class TestProximalReferenceBone(unittest.TestCase):
    """The proximal channel's reference bone, and its non-zero neutral.

    The proximal angle is measured against the wrist->MCP vector. With a
    21-joint skeleton the wrist is a single point, so that vector fans outward
    from the carpus instead of following the metacarpal shaft, and an
    anatomically flat hand therefore reads a finger-specific NON-ZERO proximal
    angle. This is a property of the definition, not a defect, and it is
    asserted here rather than hidden: the pilot shows the same floor
    (index proximal minimum 12.9 deg over 894 frames).

    No per-finger neutral offset is subtracted anywhere. That would be
    calibration rather than geometry, and it is deliberately left to whatever
    consumes these values.
    """

    def test_flat_hand_has_zero_middle_and_distal_flexion(self) -> None:
        flexion, flags = compute_flexion(flat_hand())
        self.assertEqual(flags, [])
        np.testing.assert_allclose(flexion[:, 1], np.zeros(5), atol=1e-9)
        np.testing.assert_allclose(flexion[:, 2], np.zeros(5), atol=1e-9)

    def test_flat_hand_has_a_non_zero_proximal_offset(self) -> None:
        """The offset scales with how far the MCP fans off the palm axis.

        In this fixture the middle MCP sits exactly on the palm's distal axis,
        so its metacarpal is collinear with its finger and its offset really is
        zero. Every other finger fans outward and reads a non-zero neutral.
        """

        flexion, _ = compute_flexion(flat_hand())
        self.assertAlmostEqual(
            float(flexion[FINGER_ORDER.index("middle"), 0]), 0.0, places=9
        )
        for finger in ("thumb", "index", "ring", "pinky"):
            self.assertGreater(
                float(flexion[FINGER_ORDER.index(finger), 0]), 1.0,
                f"{finger} proximal offset should be a documented non-zero",
            )

    def test_the_proximal_channel_is_still_monotone_in_added_bend(self) -> None:
        """What matters downstream: more true bend always reads as more angle."""

        previous = -1.0
        for degrees in (0.0, 10.0, 20.0, 40.0, 60.0, 80.0):
            joints = bend_chain(flat_hand(), "index", 0, degrees)
            value = flexion_of(joints, "index", 0)
            self.assertGreater(value, previous)
            previous = value


class TestRigidInvariance(unittest.TestCase):
    """9-13. Finger-local quantities are rigid-transform invariant."""

    def setUp(self) -> None:
        self.joints = _bent_hand()
        self.kinematics = compute_hand_kinematics(self.joints, "right", "OBSERVED")
        self.assertTrue(self.kinematics.valid)

    def _compare_local(self, transformed: np.ndarray) -> None:
        other = compute_hand_kinematics(transformed, "right", "OBSERVED")
        self.assertTrue(other.valid)
        np.testing.assert_allclose(other.flexion_deg, self.kinematics.flexion_deg, atol=1e-8)
        np.testing.assert_allclose(other.spread_deg, self.kinematics.spread_deg, atol=1e-8)

    def test_translation_invariance(self) -> None:
        self._compare_local(self.joints + np.array([3.7, -12.0, 0.85]))

    def test_uniform_scale_invariance(self) -> None:
        self._compare_local(self.joints * 7.25)

    def test_uniform_scale_invariance_small(self) -> None:
        self._compare_local(self.joints * 0.013)

    def test_global_rotation_invariance_of_flexion_and_spread(self) -> None:
        for axis, degrees in (((0, 0, 1), 37.0), ((1, 1, 0), 115.0), ((0.2, -0.7, 0.4), 263.0)):
            rotation = rotation_about(np.asarray(axis, dtype=np.float64), degrees)
            self._compare_local(self.joints @ rotation.T)

    def test_combined_rigid_transform_and_scale(self) -> None:
        rotation = rotation_about(np.array([0.3, 0.5, -0.8]), 88.0)
        self._compare_local((self.joints @ rotation.T) * 3.5 + np.array([-2.0, 4.0, 9.0]))

    def test_global_rotation_changes_palm_orientation_correctly(self) -> None:
        rotation = rotation_about(np.array([0.2, -0.7, 0.4]), 63.0)
        rotated = compute_hand_kinematics(self.joints @ rotation.T, "right", "OBSERVED")
        self.assertTrue(rotated.valid)
        np.testing.assert_allclose(
            rotated.palm_rotation, rotation @ self.kinematics.palm_rotation, atol=1e-8
        )
        # and it really did change
        self.assertGreater(
            float(np.abs(rotated.palm_rotation - self.kinematics.palm_rotation).max()), 0.1
        )

    def test_translation_and_scale_do_not_change_palm_orientation(self) -> None:
        moved = compute_hand_kinematics(
            self.joints * 4.0 + np.array([1.0, 2.0, 3.0]), "right", "OBSERVED"
        )
        np.testing.assert_allclose(
            moved.palm_rotation, self.kinematics.palm_rotation, atol=1e-8
        )


class TestPalmFrameConvention(unittest.TestCase):
    """The palm normal's anatomical direction, pinned by an explicit pose."""

    def test_normal_points_out_of_the_palm_on_a_right_hand(self) -> None:
        # Right hand, palm facing +z (the viewer), fingers up (+y), thumb at -x.
        joints = straight_hand()
        frame, flags = build_palm_frame(joints, "right")
        self.assertEqual(flags, [])
        np.testing.assert_allclose(frame.distal, [0.0, 1.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(frame.normal, [0.0, 0.0, 1.0], atol=1e-6)
        # lateral points to the thumb side (radial) on a right hand
        self.assertLess(frame.lateral[0], -0.99)

    def test_matrix_columns_are_the_documented_axes(self) -> None:
        frame, _ = build_palm_frame(_bent_hand(), "right")
        np.testing.assert_allclose(frame.rotation[:, 0], frame.lateral, atol=1e-12)
        np.testing.assert_allclose(frame.rotation[:, 1], frame.normal, atol=1e-12)
        np.testing.assert_allclose(frame.rotation[:, 2], frame.distal, atol=1e-12)


class TestMirroredHands(unittest.TestCase):
    """14. A mirrored LEFT hand must produce equivalent local kinematics."""

    def setUp(self) -> None:
        self.right_joints = _bent_hand()
        self.left_joints = self.right_joints @ MIRROR.T
        self.right = compute_hand_kinematics(self.right_joints, "right", "OBSERVED")
        self.left = compute_hand_kinematics(self.left_joints, "left", "OBSERVED")

    def test_both_hands_are_valid(self) -> None:
        self.assertTrue(self.right.valid)
        self.assertTrue(self.left.valid)

    def test_flexion_is_identical_not_negated(self) -> None:
        np.testing.assert_allclose(self.left.flexion_deg, self.right.flexion_deg, atol=1e-8)
        # the point of the convention: a +70 bend is +70 on both hands
        index = FINGER_ORDER.index("index")
        self.assertAlmostEqual(float(self.right.flexion_deg[index, 1]), 70.0, places=6)
        self.assertAlmostEqual(float(self.left.flexion_deg[index, 1]), 70.0, places=6)

    def test_spread_is_identical(self) -> None:
        np.testing.assert_allclose(self.left.spread_deg, self.right.spread_deg, atol=1e-8)

    def test_palm_frames_satisfy_the_documented_mirror_identity(self) -> None:
        np.testing.assert_allclose(
            self.left.palm_rotation, MIRROR @ self.right.palm_rotation @ MIRROR, atol=1e-8
        )

    def test_both_palm_frames_are_right_handed(self) -> None:
        for kinematics in (self.left, self.right):
            self.assertAlmostEqual(float(np.linalg.det(kinematics.palm_rotation)), 1.0, places=9)

    def test_palm_normal_is_anatomically_consistent_across_hands(self) -> None:
        """The normal mirrors with the hand, rather than flipping into it."""

        np.testing.assert_allclose(
            self.left.palm_rotation[:, 1], MIRROR @ self.right.palm_rotation[:, 1], atol=1e-8
        )
        np.testing.assert_allclose(
            self.left.palm_rotation[:, 2], MIRROR @ self.right.palm_rotation[:, 2], atol=1e-8
        )

    def test_applying_the_right_rule_to_a_left_hand_would_flip_the_normal(self) -> None:
        """Guards the canonicalization: without it the normal is wrong."""

        wrong, _ = build_palm_frame(self.left_joints, "right")
        correct, _ = build_palm_frame(self.left_joints, "left")
        np.testing.assert_allclose(wrong.normal, -correct.normal, atol=1e-9)


class TestSpread(unittest.TestCase):
    """12 (spread half) plus the palm-relative property."""

    def test_spread_is_positive_for_a_splayed_hand(self) -> None:
        spread, flags = compute_spread(flat_hand(), np.array([0.0, 0.0, 1.0]))
        self.assertEqual(flags, [])
        self.assertEqual(spread.shape, (len(SPREAD_PAIRS),))
        self.assertTrue(np.isfinite(spread).all())
        self.assertTrue((spread >= 0.0).all())

    def test_spread_responds_to_abduction(self) -> None:
        joints = flat_hand()
        base, _ = compute_spread(joints, np.array([0.0, 0.0, 1.0]))
        # abduct the index finger within the palm plane
        spread_joints = bend_chain(joints, "index", 0, 20.0, axis=(0.0, 0.0, 1.0))
        frame, _ = build_palm_frame(spread_joints, "right")
        moved, _ = compute_spread(spread_joints, frame.normal)
        pair = [i for i, p in enumerate(SPREAD_PAIRS) if p == ("index", "middle")][0]
        self.assertGreater(abs(float(moved[pair]) - float(base[pair])), 15.0)

    def test_spread_is_unchanged_by_global_rotation(self) -> None:
        joints = _bent_hand()
        first = compute_hand_kinematics(joints, "right", "OBSERVED")
        rotation = rotation_about(np.array([0.5, 0.1, -0.9]), 141.0)
        second = compute_hand_kinematics(joints @ rotation.T, "right", "OBSERVED")
        np.testing.assert_allclose(second.spread_deg, first.spread_deg, atol=1e-8)


class TestQuaternionAndMatrix(unittest.TestCase):
    """15-17. Orientation output must be a clean unit rotation."""

    def setUp(self) -> None:
        self.kinematics = compute_hand_kinematics(_bent_hand(), "right", "OBSERVED")

    def test_quaternion_is_normalized(self) -> None:
        norm = float(np.linalg.norm(self.kinematics.palm_quaternion))
        self.assertAlmostEqual(norm, 1.0, places=12)

    def test_quaternion_order_is_wxyz_and_w_non_negative(self) -> None:
        self.assertEqual(self.kinematics.palm_quaternion.shape, (4,))
        self.assertGreaterEqual(float(self.kinematics.palm_quaternion[0]), 0.0)

    def test_quaternion_round_trips_to_the_rotation_matrix(self) -> None:
        w, x, y, z = self.kinematics.palm_quaternion
        rebuilt = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        np.testing.assert_allclose(rebuilt, self.kinematics.palm_rotation, atol=1e-9)

    def test_rotation_matrix_is_orthonormal(self) -> None:
        self.assertLess(orthonormality_error(self.kinematics.palm_rotation), 1e-12)

    def test_rotation_determinant_is_plus_one(self) -> None:
        self.assertAlmostEqual(float(np.linalg.det(self.kinematics.palm_rotation)), 1.0, places=12)

    def test_quaternion_sign_is_deterministic_for_a_negative_trace_rotation(self) -> None:
        rotation = rotation_about(np.array([0.0, 0.0, 1.0]), 179.0)
        quaternion = rotation_matrix_to_quaternion_wxyz(rotation)
        self.assertIsNotNone(quaternion)
        self.assertGreaterEqual(float(quaternion[0]), 0.0)
        self.assertAlmostEqual(float(np.linalg.norm(quaternion)), 1.0, places=12)


class TestTrackingStatePolicy(unittest.TestCase):
    """18-19. States without a usable pose must produce NaN, never a guess."""

    def _assert_all_nan(self, kinematics) -> None:
        self.assertFalse(kinematics.valid)
        self.assertTrue(np.isnan(kinematics.flexion_deg).all())
        self.assertTrue(np.isnan(kinematics.spread_deg).all())
        self.assertTrue(np.isnan(kinematics.palm_rotation).all())
        self.assertTrue(np.isnan(kinematics.palm_quaternion).all())

    def test_missing_hand_is_all_nan(self) -> None:
        kinematics = compute_hand_kinematics(_bent_hand(), "right", "MISSING")
        self._assert_all_nan(kinematics)
        self.assertIn("NO_POSE_STATE_MISSING", kinematics.flags)

    def test_likely_occluded_hand_is_all_nan(self) -> None:
        kinematics = compute_hand_kinematics(_bent_hand(), "right", "LIKELY_OCCLUDED")
        self._assert_all_nan(kinematics)
        self.assertIn("NO_POSE_STATE_LIKELY_OCCLUDED", kinematics.flags)

    def test_rejected_quality_hand_is_all_nan(self) -> None:
        kinematics = compute_hand_kinematics(_bent_hand(), "right", "REJECTED_QUALITY")
        self._assert_all_nan(kinematics)
        self.assertIn("NO_POSE_STATE_REJECTED_QUALITY", kinematics.flags)

    def test_a_present_pose_is_ignored_when_the_state_says_no_pose(self) -> None:
        """Geometry is available but the tracker says it is not usable."""

        self._assert_all_nan(compute_hand_kinematics(straight_hand(), "right", "MISSING"))

    def test_ambiguous_state_keeps_its_pose_and_is_flagged(self) -> None:
        kinematics = compute_hand_kinematics(_bent_hand(), "right", "AMBIGUOUS")
        self.assertTrue(kinematics.valid)
        self.assertIn("TRACK_STATE_AMBIGUOUS", kinematics.flags)
        self.assertTrue(np.isfinite(kinematics.flexion_deg).all())

    def test_pose_bearing_states_match_the_tracker_definition(self) -> None:
        """kinematics restates POSE_STATES; the two must not diverge."""

        from tracking.wilor.schema import POSE_STATES

        self.assertEqual({state.value for state in POSE_STATES}, set(POSE_BEARING_STATES))


class TestDegenerateGeometry(unittest.TestCase):
    """20-21. Degenerate input is flagged, never crashed on."""

    def test_zero_length_bone_is_rejected(self) -> None:
        joints = straight_hand()
        chain = FINGER_CHAINS["index"]
        joints[chain[2]] = joints[chain[1]]  # collapse MCP->PIP to zero length
        kinematics = compute_hand_kinematics(joints, "right", "OBSERVED")
        self.assertFalse(kinematics.valid)
        self.assertTrue(any(flag.startswith("ZERO_LENGTH_BONE_index") for flag in kinematics.flags))
        index = FINGER_ORDER.index("index")
        self.assertTrue(np.isnan(kinematics.flexion_deg[index, 0]))
        self.assertTrue(np.isnan(kinematics.flexion_deg[index, 1]))
        # unaffected fingers keep their finite values
        self.assertTrue(np.isfinite(kinematics.flexion_deg[FINGER_ORDER.index("ring")]).all())

    def test_collinear_palm_points_are_rejected(self) -> None:
        joints = straight_hand()
        # put wrist, index MCP, middle MCP and pinky MCP on one line
        for index, offset in ((5, 0.02), (9, 0.04), (17, 0.06)):
            joints[index] = np.array([0.0, offset, 0.0])
        frame, flags = build_palm_frame(joints, "right")
        self.assertIsNone(frame)
        self.assertIn("PALM_POINTS_COLLINEAR", flags)
        kinematics = compute_hand_kinematics(joints, "right", "OBSERVED")
        self.assertFalse(kinematics.valid)
        self.assertTrue(np.isnan(kinematics.palm_rotation).all())
        self.assertTrue(np.isnan(kinematics.palm_quaternion).all())

    def test_zero_length_palm_axis_is_rejected(self) -> None:
        joints = straight_hand()
        joints[9] = joints[0]  # middle MCP coincides with the wrist
        frame, flags = build_palm_frame(joints, "right")
        self.assertIsNone(frame)
        self.assertIn("PALM_AXIS_ZERO_LENGTH", flags)

    def test_non_finite_joints_are_rejected(self) -> None:
        joints = straight_hand()
        joints[7, 1] = np.nan
        kinematics = compute_hand_kinematics(joints, "right", "OBSERVED")
        self.assertFalse(kinematics.valid)
        self.assertIn("JOINTS_NON_FINITE", kinematics.flags)
        self.assertTrue(np.isnan(kinematics.flexion_deg).all())

    def test_infinite_joints_are_rejected(self) -> None:
        joints = straight_hand()
        joints[3, 0] = np.inf
        kinematics = compute_hand_kinematics(joints, "right", "OBSERVED")
        self.assertFalse(kinematics.valid)
        self.assertIn("JOINTS_NON_FINITE", kinematics.flags)

    def test_wrong_joint_shape_is_rejected(self) -> None:
        kinematics = compute_hand_kinematics(np.zeros((20, 3)), "right", "OBSERVED")
        self.assertFalse(kinematics.valid)
        self.assertIn("JOINTS_WRONG_SHAPE", kinematics.flags)

    def test_finger_parallel_to_the_palm_normal_flags_spread(self) -> None:
        joints = flat_hand()
        # point the index proximal phalanx straight out of the palm plane, so
        # it has no direction at all once projected into that plane
        chain = FINGER_CHAINS["index"]
        joints[chain[2]] = joints[chain[1]] + np.array([0.0, 0.0, 0.033])
        joints[chain[3]] = joints[chain[2]] + np.array([0.0, 0.0, 0.022])
        joints[chain[4]] = joints[chain[3]] + np.array([0.0, 0.0, 0.026])
        frame, _ = build_palm_frame(joints, "right")
        spread, flags = compute_spread(joints, frame.normal)
        self.assertTrue(any(flag.startswith("SPREAD_DIRECTION_DEGENERATE") for flag in flags))
        self.assertTrue(np.isnan(spread[SPREAD_PAIRS.index(("thumb", "index"))]))
        self.assertTrue(np.isnan(spread[SPREAD_PAIRS.index(("index", "middle"))]))

    def test_validity_means_every_channel_is_finite(self) -> None:
        """The invariant the pilot diagnostics rely on."""

        for joints, state in (
            (straight_hand(), "OBSERVED"),
            (_bent_hand(), "OBSERVED"),
            (_bent_hand(), "AMBIGUOUS"),
            (_bent_hand(), "MISSING"),
            (np.full((N_JOINTS, 3), np.nan), "OBSERVED"),
        ):
            kinematics = compute_hand_kinematics(joints, "right", state)
            all_finite = (
                np.isfinite(kinematics.flexion_deg).all()
                and np.isfinite(kinematics.spread_deg).all()
                and np.isfinite(kinematics.palm_rotation).all()
                and np.isfinite(kinematics.palm_quaternion).all()
            )
            self.assertEqual(kinematics.valid, bool(all_finite))


def _tracked_arrays(frames: int = 6) -> tuple[dict, dict]:
    """A minimal synthetic tracked NPZ payload, in the TASK-004 layout."""

    right = _bent_hand()
    left = right @ MIRROR.T
    landmarks = np.zeros((frames, 2, N_JOINTS, 3), dtype=np.float32)
    state = np.ones((frames, 2), dtype=np.int32)  # OBSERVED
    for row in range(frames):
        landmarks[row, 0] = left
        landmarks[row, 1] = right
    state[2, 1] = 0  # MISSING
    state[3, 1] = 4  # LIKELY_OCCLUDED
    state[4, 0] = 3  # REJECTED_QUALITY
    state[5, 0] = 2  # AMBIGUOUS
    arrays = {
        "frame_index": np.arange(frames, dtype=np.int32),
        "timestamp_seconds": (np.arange(frames) / 30.0).astype(np.float64),
        "state_code": state,
        "raw_detection_index": np.tile(np.array([[0, 1]], dtype=np.int32), (frames, 1)),
        "landmarks_3d": landmarks,
    }
    metadata = {
        "track_order": list(TRACK_ORDER),
        "state_codes": {
            "MISSING": 0, "OBSERVED": 1, "AMBIGUOUS": 2,
            "REJECTED_QUALITY": 3, "LIKELY_OCCLUDED": 4,
        },
        "schema_version": "wilor_tracked_v1",
        "sample_id": "synthetic",
    }
    return arrays, metadata


class TestSequenceExtraction(unittest.TestCase):
    """18-19 at sequence level, plus 25 (schema/order preservation)."""

    def setUp(self) -> None:
        self.arrays, self.metadata = _tracked_arrays()
        self.sequence = extract_sequence(self.arrays, self.metadata, "synthetic")

    def test_output_shapes_match_the_contract(self) -> None:
        frames = 6
        self.assertEqual(self.sequence.frame_index.shape, (frames,))
        self.assertEqual(self.sequence.timestamp_seconds.shape, (frames,))
        self.assertEqual(self.sequence.tracking_state_code.shape, (frames, 2))
        self.assertEqual(self.sequence.source_raw_detection_index.shape, (frames, 2))
        self.assertEqual(self.sequence.valid_kinematics.shape, (frames, 2))
        self.assertEqual(self.sequence.flexion_deg.shape, (frames, 2, 5, 3))
        self.assertEqual(self.sequence.adjacent_spread_deg.shape, (frames, 2, 4))
        self.assertEqual(self.sequence.palm_rotation_matrix.shape, (frames, 2, 3, 3))
        self.assertEqual(self.sequence.palm_quaternion_wxyz.shape, (frames, 2, 4))
        self.assertEqual(self.sequence.kinematic_flags_json.shape, (frames, 2))

    def test_dtypes_are_plain_numeric(self) -> None:
        self.assertEqual(self.sequence.flexion_deg.dtype, np.float32)
        self.assertEqual(self.sequence.adjacent_spread_deg.dtype, np.float32)
        self.assertEqual(self.sequence.palm_rotation_matrix.dtype, np.float32)
        self.assertEqual(self.sequence.palm_quaternion_wxyz.dtype, np.float32)
        self.assertEqual(self.sequence.valid_kinematics.dtype, np.bool_)

    def test_non_pose_states_are_masked_to_nan(self) -> None:
        for row, column in ((2, 1), (3, 1), (4, 0)):
            self.assertFalse(self.sequence.valid_kinematics[row, column])
            self.assertTrue(np.isnan(self.sequence.flexion_deg[row, column]).all())
            self.assertTrue(np.isnan(self.sequence.adjacent_spread_deg[row, column]).all())
            self.assertTrue(np.isnan(self.sequence.palm_rotation_matrix[row, column]).all())
            self.assertTrue(np.isnan(self.sequence.palm_quaternion_wxyz[row, column]).all())

    def test_ambiguous_state_stays_valid_and_flagged(self) -> None:
        self.assertTrue(self.sequence.valid_kinematics[5, 0])
        flags = json.loads(str(self.sequence.kinematic_flags_json[5, 0]))
        self.assertIn("TRACK_STATE_AMBIGUOUS", flags)

    def test_tracking_state_and_provenance_are_preserved_verbatim(self) -> None:
        np.testing.assert_array_equal(
            self.sequence.tracking_state_code, self.arrays["state_code"]
        )
        np.testing.assert_array_equal(
            self.sequence.source_raw_detection_index, self.arrays["raw_detection_index"]
        )
        np.testing.assert_array_equal(self.sequence.frame_index, self.arrays["frame_index"])

    def test_left_and_right_columns_keep_the_task004_order(self) -> None:
        """Column 0 is LEFT, column 1 is RIGHT, and the mirror pair agrees."""

        self.assertEqual(TRACK_ORDER, ("left", "right"))
        np.testing.assert_allclose(
            self.sequence.flexion_deg[0, 0], self.sequence.flexion_deg[0, 1], atol=1e-5
        )

    def test_unexpected_track_order_is_refused(self) -> None:
        metadata = dict(self.metadata)
        metadata["track_order"] = ["right", "left"]
        with self.assertRaises(ValueError):
            extract_sequence(self.arrays, metadata, "synthetic")


class TestDeterminismAndRoundTrip(unittest.TestCase):
    """22-24. Repeatability, input immutability, and NPZ persistence."""

    def test_repeated_execution_is_bit_identical(self) -> None:
        arrays, metadata = _tracked_arrays()
        first = extract_sequence(arrays, metadata, "synthetic")
        second = extract_sequence(arrays, metadata, "synthetic")
        for name in ("flexion_deg", "adjacent_spread_deg", "palm_rotation_matrix", "palm_quaternion_wxyz"):
            np.testing.assert_array_equal(
                np.nan_to_num(getattr(first, name), nan=-999.0),
                np.nan_to_num(getattr(second, name), nan=-999.0),
            )
        np.testing.assert_array_equal(first.valid_kinematics, second.valid_kinematics)
        np.testing.assert_array_equal(first.kinematic_flags_json, second.kinematic_flags_json)

    def test_source_tracked_arrays_are_not_modified(self) -> None:
        arrays, metadata = _tracked_arrays()
        before = {key: value.copy() for key, value in arrays.items()}
        extract_sequence(arrays, metadata, "synthetic")
        for key, value in before.items():
            np.testing.assert_array_equal(arrays[key], value)

    def test_npz_round_trip_preserves_every_array(self) -> None:
        arrays, metadata = _tracked_arrays()
        sequence = extract_sequence(arrays, metadata, "synthetic")
        meta = build_metadata(
            sequence,
            tracked_dir=Path("/nonexistent"),
            tracked_sha256="0" * 64,
            tracked_metadata=metadata,
            implementation_commit="test",
        )
        with tempfile.TemporaryDirectory() as directory:
            save_kinematics(directory, sequence, meta)
            loaded, loaded_meta = load_kinematics(directory)

        np.testing.assert_array_equal(loaded["frame_index"], sequence.frame_index)
        np.testing.assert_array_equal(loaded["valid_kinematics"], sequence.valid_kinematics)
        np.testing.assert_array_equal(loaded["tracking_state_code"], sequence.tracking_state_code)
        np.testing.assert_array_equal(
            loaded["source_raw_detection_index"], sequence.source_raw_detection_index
        )
        np.testing.assert_array_equal(loaded["kinematic_flags_json"], sequence.kinematic_flags_json)
        for name in ("flexion_deg", "adjacent_spread_deg", "palm_rotation_matrix", "palm_quaternion_wxyz"):
            np.testing.assert_array_equal(
                np.nan_to_num(loaded[name], nan=-999.0),
                np.nan_to_num(getattr(sequence, name), nan=-999.0),
            )
            self.assertEqual(loaded[name].dtype, np.float32)
        self.assertEqual(loaded_meta["schema_version"], SCHEMA_VERSION)
        self.assertEqual(loaded_meta["track_order"], list(TRACK_ORDER))
        self.assertEqual(loaded_meta["finger_order"], list(FINGER_ORDER))
        self.assertEqual(loaded_meta["chain_joint_order"], list(CHAIN_ORDER))
        self.assertEqual(loaded_meta["quaternion_order"], "wxyz")

    def test_npz_contains_no_pickled_objects(self) -> None:
        arrays, metadata = _tracked_arrays()
        sequence = extract_sequence(arrays, metadata, "synthetic")
        meta = build_metadata(
            sequence, tracked_dir=Path("/nonexistent"), tracked_sha256="0" * 64,
            tracked_metadata=metadata, implementation_commit="test",
        )
        with tempfile.TemporaryDirectory() as directory:
            save_kinematics(directory, sequence, meta)
            # allow_pickle=False is the actual guarantee; this must not raise
            with np.load(Path(directory) / "hand_kinematics.npz", allow_pickle=False) as data:
                self.assertEqual(set(data.files), set(ARRAY_ORDER))
                for key in data.files:
                    self.assertNotEqual(data[key].dtype, np.object_)


class TestSpreadConditioning(unittest.TestCase):
    """The palm-plane projection has a stated conditioning limit."""

    def test_threshold_matches_the_documented_angle(self) -> None:
        self.assertAlmostEqual(MIN_SPREAD_PROJECTION_ANGLE_DEG, 15.0, places=9)
        self.assertAlmostEqual(
            MIN_PROJECTED_NORM, float(np.sin(np.radians(15.0))), places=12
        )

    def test_finger_just_inside_the_limit_still_yields_spread(self) -> None:
        joints = self._index_tilted_from_normal(20.0)
        frame, _ = build_palm_frame(joints, "right")
        spread, flags = compute_spread(joints, frame.normal)
        self.assertEqual(flags, [])
        self.assertTrue(np.isfinite(spread).all())

    def test_finger_just_outside_the_limit_is_rejected(self) -> None:
        joints = self._index_tilted_from_normal(10.0)
        frame, _ = build_palm_frame(joints, "right")
        spread, flags = compute_spread(joints, frame.normal)
        self.assertIn("SPREAD_DIRECTION_DEGENERATE_index", flags)
        self.assertTrue(np.isnan(spread[SPREAD_PAIRS.index(("index", "middle"))]))

    @staticmethod
    def _index_tilted_from_normal(degrees_from_normal: float) -> np.ndarray:
        """Index proximal phalanx at a chosen angle away from the palm normal."""

        joints = flat_hand()
        chain = FINGER_CHAINS["index"]
        theta = np.radians(degrees_from_normal)
        direction = np.array([0.0, np.sin(theta), np.cos(theta)])
        position = joints[chain[1]]
        for step, length in enumerate((0.033, 0.022, 0.026)):
            position = position + length * direction
            joints[chain[step + 2]] = position
        return joints


class TestCoincidentPalmLandmarks(unittest.TestCase):
    """TASK-005E2. Two palm landmarks at one point cannot define a palm.

    The three-vector frame construction does not catch this on its own: with a
    coincident index and middle MCP, ``middle_MCP - wrist`` and
    ``index_MCP - pinky_MCP`` are both still non-zero and non-parallel, so a
    fully finite orthonormal frame comes out of a collapsed palm. The
    landmarks are therefore tested for distinctness directly.

    The rule is a numerical-degeneracy bound (1e-3 of palm length, ~4200x the
    float32 difference-rounding floor), not an anatomical one, and no fixture
    identifier or coordinate appears in it.
    """

    @staticmethod
    def _with_mcp_separation(ratio: float) -> np.ndarray:
        """Flat hand whose index MCP sits ``ratio`` * palm length from middle."""

        joints = flat_hand()
        palm_length = float(np.linalg.norm(joints[9] - joints[0]))
        joints[5] = joints[9] + np.array([-ratio * palm_length, 0.0, 0.0])
        return joints

    def test_exactly_coincident_mcps_are_rejected(self) -> None:
        joints = flat_hand()
        joints[5] = joints[9].copy()
        frame, flags = build_palm_frame(joints, "right")
        self.assertIsNone(frame)
        self.assertTrue(any(f.startswith("PALM_LANDMARKS_COINCIDENT") for f in flags))
        self.assertIn("index_MCP_middle_MCP", flags[0])

    def test_the_frame_would_otherwise_have_looked_valid(self) -> None:
        """Guards the reason this check is needed at all.

        With coincident MCPs the two axis-defining vectors stay non-zero and
        non-parallel, so without the landmark test a finite frame is produced.
        """

        joints = flat_hand()
        joints[5] = joints[9].copy()
        distal = joints[9] - joints[0]
        lateral_raw = joints[5] - joints[17]
        self.assertGreater(float(np.linalg.norm(distal)), 1e-6)
        self.assertGreater(float(np.linalg.norm(lateral_raw)), 1e-6)
        self.assertGreater(float(np.linalg.norm(np.cross(distal, lateral_raw))), 1e-6)

    def test_every_landmark_pair_is_covered(self) -> None:
        for coincide_with, moved in ((0, 5), (0, 9), (0, 17), (5, 9), (5, 17), (9, 17)):
            joints = flat_hand()
            joints[moved] = joints[coincide_with].copy()
            frame, flags = build_palm_frame(joints, "right")
            self.assertIsNone(frame, f"joint {moved} == joint {coincide_with} was accepted")
            self.assertTrue(
                any(
                    f.startswith("PALM_LANDMARKS_COINCIDENT")
                    or f in ("PALM_AXIS_ZERO_LENGTH", "PALM_POINTS_COLLINEAR")
                    for f in flags
                ),
                flags,
            )

    def test_near_but_distinct_landmarks_are_accepted(self) -> None:
        """A valid palm is not rejected merely for having close MCPs."""

        joints = self._with_mcp_separation(0.01)  # 10x the threshold
        frame, flags = build_palm_frame(joints, "right")
        self.assertIsNotNone(frame)
        self.assertEqual(flags, [])
        kinematics = compute_hand_kinematics(joints, "right", "OBSERVED")
        self.assertTrue(kinematics.palm_frame_valid)

    def test_threshold_boundary_behaves_monotonically(self) -> None:
        for ratio in (0.5, 0.1, 0.01, 0.002):
            self.assertIsNotNone(
                build_palm_frame(self._with_mcp_separation(ratio), "right")[0],
                f"ratio {ratio} above the bound should be accepted",
            )
        for ratio in (0.0, 1e-5, 1e-4):
            self.assertIsNone(
                build_palm_frame(self._with_mcp_separation(ratio), "right")[0],
                f"ratio {ratio} below the bound should be rejected",
            )

    def test_threshold_is_the_documented_numerical_bound(self) -> None:
        self.assertAlmostEqual(MIN_PALM_LANDMARK_SEPARATION_RATIO, 1e-3, places=12)
        # comfortably above the float32 difference-rounding floor
        self.assertGreater(
            MIN_PALM_LANDMARK_SEPARATION_RATIO, 1000.0 * float(np.finfo(np.float32).eps)
        )

    def test_rejection_is_translation_invariant(self) -> None:
        joints = flat_hand()
        joints[5] = joints[9].copy()
        for offset in ([3.7, -12.0, 0.85], [0.0, 0.0, 0.0], [-100.0, 55.0, 2.0]):
            frame, flags = build_palm_frame(joints + np.array(offset), "right")
            self.assertIsNone(frame)
            self.assertTrue(flags[0].startswith("PALM_LANDMARKS_COINCIDENT"))

    def test_rejection_is_scale_invariant(self) -> None:
        joints = flat_hand()
        joints[5] = joints[9].copy()
        for scale in (0.001, 0.5, 1.0, 7.25, 1000.0):
            frame, _ = build_palm_frame(joints * scale, "right")
            self.assertIsNone(frame)
        # and acceptance is scale invariant too
        valid = self._with_mcp_separation(0.01)
        for scale in (0.001, 1.0, 1000.0):
            self.assertIsNotNone(build_palm_frame(valid * scale, "right")[0])

    def test_mirrored_left_hand_gives_the_same_verdict(self) -> None:
        joints = flat_hand()
        joints[5] = joints[9].copy()
        right_frame, right_flags = build_palm_frame(joints, "right")
        left_frame, left_flags = build_palm_frame(joints @ MIRROR.T, "left")
        self.assertIsNone(right_frame)
        self.assertIsNone(left_frame)
        self.assertEqual(right_flags, left_flags)

    def test_mirrored_left_hand_accepts_what_right_accepts(self) -> None:
        joints = self._with_mcp_separation(0.01)
        self.assertIsNotNone(build_palm_frame(joints, "right")[0])
        self.assertIsNotNone(build_palm_frame(joints @ MIRROR.T, "left")[0])

    def test_normal_synthetic_hands_remain_valid(self) -> None:
        for joints in (straight_hand(), flat_hand(), _bent_hand()):
            for track, points in (("right", joints), ("left", joints @ MIRROR.T)):
                frame, flags = build_palm_frame(points, track)
                self.assertIsNotNone(frame)
                self.assertEqual(flags, [])

    def test_rejection_propagates_to_both_validity_flags_and_nan(self) -> None:
        joints = flat_hand()
        joints[5] = joints[9].copy()
        kinematics = compute_hand_kinematics(joints, "right", "OBSERVED")
        self.assertFalse(kinematics.valid)
        self.assertFalse(kinematics.palm_frame_valid)
        self.assertTrue(np.isnan(kinematics.flexion_deg).all())
        self.assertTrue(np.isnan(kinematics.spread_deg).all())
        self.assertTrue(np.isnan(kinematics.palm_rotation).all())
        self.assertTrue(np.isnan(kinematics.palm_quaternion).all())
        self.assertTrue(any(f.startswith("PALM_LANDMARKS_COINCIDENT") for f in kinematics.flags))

    def test_real_pilot_geometry_has_two_orders_of_magnitude_clearance(self) -> None:
        """The tightest real pair measured over the pilot is 0.2648."""

        self.assertLess(MIN_PALM_LANDMARK_SEPARATION_RATIO, 0.2648 / 100.0)


class TestValidityFlags(unittest.TestCase):
    """valid_kinematics is strict; valid_palm_frame tracks orientation only."""

    def test_a_clean_hand_sets_both_flags(self) -> None:
        kinematics = compute_hand_kinematics(_bent_hand(), "right", "OBSERVED")
        self.assertTrue(kinematics.valid)
        self.assertTrue(kinematics.palm_frame_valid)

    def test_an_undefined_spread_clears_only_the_strict_flag(self) -> None:
        joints = TestSpreadConditioning._index_tilted_from_normal(5.0)
        kinematics = compute_hand_kinematics(joints, "right", "OBSERVED")
        self.assertFalse(kinematics.valid)
        self.assertTrue(kinematics.palm_frame_valid)
        # the sound channels survive rather than being discarded
        self.assertTrue(np.isfinite(kinematics.flexion_deg).all())
        self.assertTrue(np.isfinite(kinematics.palm_rotation).all())
        self.assertTrue(np.isfinite(kinematics.palm_quaternion).all())

    def test_no_pose_clears_both_flags(self) -> None:
        kinematics = compute_hand_kinematics(_bent_hand(), "right", "MISSING")
        self.assertFalse(kinematics.valid)
        self.assertFalse(kinematics.palm_frame_valid)

    def test_bad_palm_frame_clears_both_flags(self) -> None:
        joints = flat_hand()
        joints[9] = joints[0]
        kinematics = compute_hand_kinematics(joints, "right", "OBSERVED")
        self.assertFalse(kinematics.valid)
        self.assertFalse(kinematics.palm_frame_valid)

    def test_sequence_exposes_both_arrays(self) -> None:
        arrays, metadata = _tracked_arrays()
        sequence = extract_sequence(arrays, metadata, "synthetic")
        self.assertEqual(sequence.valid_palm_frame.shape, sequence.valid_kinematics.shape)
        self.assertEqual(sequence.valid_palm_frame.dtype, np.bool_)
        # a strictly-valid hand is always palm-frame valid
        self.assertTrue(
            bool(np.all(~sequence.valid_kinematics | sequence.valid_palm_frame))
        )


class TestNoTemporalFiltering(unittest.TestCase):
    """TASK-005A is strictly per-frame."""

    def test_each_frame_depends_only_on_its_own_joints(self) -> None:
        arrays, metadata = _tracked_arrays(frames=6)
        full = extract_sequence(arrays, metadata, "synthetic")

        # Perturb one frame only; no other frame's output may move.
        perturbed = {key: value.copy() for key, value in arrays.items()}
        perturbed["landmarks_3d"][1, 1] = bend_chain(_bent_hand(), "pinky", 0, 44.0)
        other = extract_sequence(perturbed, metadata, "synthetic")

        self.assertFalse(
            np.allclose(other.flexion_deg[1, 1], full.flexion_deg[1, 1], atol=1e-6)
        )
        for row in (0, 2, 3, 4, 5):
            np.testing.assert_array_equal(
                np.nan_to_num(other.flexion_deg[row], nan=-999.0),
                np.nan_to_num(full.flexion_deg[row], nan=-999.0),
            )

    def test_a_gap_is_not_filled_from_neighbours(self) -> None:
        arrays, metadata = _tracked_arrays()
        sequence = extract_sequence(arrays, metadata, "synthetic")
        # frame 2 RIGHT is MISSING while frames 1 and 3 are usable
        self.assertTrue(sequence.valid_kinematics[1, 1])
        self.assertFalse(sequence.valid_kinematics[2, 1])
        self.assertTrue(np.isnan(sequence.flexion_deg[2, 1]).all())


if __name__ == "__main__":
    unittest.main()
