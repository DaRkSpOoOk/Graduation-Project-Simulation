"""TASK-007G visual-acceptance regression tests.

These cover the failures the TASK-007G visual pass was opened for:

* the presentation rig profile is internally consistent;
* recorded wrist motion is bounded and cannot unframe the composition;
* per-sequence state is reset, so pressing a second letter does not inherit
  the previous letter's orientation reference (the reported "orbit" bug);
* the exported left hand is an exact mirror of the right one; and
* on real TASK-008 sequences the two hands stay put, stay separated and never
  overlap.

The tests that need the local GLB or the external TASK-008 run root skip
cleanly when those are not present, so CI without local assets still runs.
"""

from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np

from smart_glove_app.app.hand_pose_solver import (
    CONTRACT_DEGREE_SCALE,
    HandPoseSolver,
    axis_angle_quaternion,
    clamp_quaternion_angle,
    identity_quaternion,
    quaternion_angle_deg,
    quaternion_multiply,
)
from smart_glove_app.rendering.presentation_rig import (
    FINGERS,
    SIDES,
    SPREAD_PAIRS,
    PresentationRigError,
    load_presentation_rig,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GLB_PATH = PROJECT_ROOT / "assets-local" / "blendswap_hands_v1" / "task007g_presentation_hands.glb"
RUN_ROOT = PROJECT_ROOT.parent / "graduation-project-runs" / "task008-core28-full"
MANIFEST = PROJECT_ROOT / "datasets" / "manifests" / "karsl_core28.csv"
LABELS = PROJECT_ROOT / "datasets" / "manifests" / "karsl_core28_labels.csv"
CATALOG = PROJECT_ROOT / "visualizer" / "catalog" / "core28_exemplars.json"

ACCEPTANCE_CHARACTERS = "ابحمدي"


class _Frame:
    """Minimal stand-in for one frozen TASK-008 playback frame."""

    def __init__(
        self,
        *,
        bend: np.ndarray | None = None,
        spread: np.ndarray | None = None,
        palm: np.ndarray | None = None,
        palm_valid: bool = True,
        state: str = "OBSERVED",
    ) -> None:
        self.bend_normalized = np.zeros((2, 5, 3)) if bend is None else bend
        self.bend_valid = np.ones((2, 5, 3), dtype=bool)
        self.spread_normalized = np.zeros((2, 4)) if spread is None else spread
        self.spread_valid = np.ones((2, 4), dtype=bool)
        self.palm_quaternion_wxyz = (
            np.tile(np.asarray([1.0, 0.0, 0.0, 0.0]), (2, 1)) if palm is None else palm
        )
        self.palm_imu_valid = np.asarray([palm_valid, palm_valid])
        self._state = state

    def hand(self, side: str):  # noqa: D401 - mirrors the real contract surface
        return type("Hand", (), {"state": self._state, "present": True})()


class PresentationRigProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rig = load_presentation_rig()

    def test_profile_declares_both_hands_and_every_finger(self) -> None:
        self.assertEqual(set(self.rig.roots), set(SIDES))
        self.assertEqual(set(self.rig.armatures), set(SIDES))
        self.assertEqual(set(self.rig.chains), set(FINGERS))

    def test_every_articulated_bone_is_required(self) -> None:
        required = set(self.rig.required_bones)
        for finger, chain in self.rig.chains.items():
            self.assertIn(chain.metacarpal, required, finger)
            for joint in chain.joints:
                self.assertIn(joint, required, joint)
        self.assertIn(self.rig.wrist_bone, required)

    def test_no_bone_receives_both_a_bend_and_a_spread_channel(self) -> None:
        """Single-purpose bones are what keep the two channel families from
        silently overwriting one another, as they could in TASK-007F."""

        bend_bones = {joint for chain in self.rig.chains.values() for joint in chain.joints}
        spread_bones = {target.bone for target in self.rig.spread_targets}
        self.assertEqual(bend_bones & spread_bones, set())
        self.assertNotIn(self.rig.wrist_bone, bend_bones | spread_bones)

    def test_spread_neutrals_cover_the_frozen_task006_pairs(self) -> None:
        self.assertEqual(set(self.rig.spread_neutral_deg), set(SPREAD_PAIRS))

    def test_palm_and_back_views_are_declared_and_distinct(self) -> None:
        palm = self.rig.view_euler_deg("PALM")
        back = self.rig.view_euler_deg("BACK")
        self.assertNotEqual(palm, back)
        self.assertAlmostEqual(abs(back[1] - palm[1]) % 360.0, 180.0, places=6)

    def test_left_and_right_are_placed_symmetrically_and_apart(self) -> None:
        left = self.rig.root_position("LEFT")
        right = self.rig.root_position("RIGHT")
        self.assertLess(left[0], 0.0, "LEFT must stay on the left")
        self.assertGreater(right[0], 0.0, "RIGHT must stay on the right")
        self.assertAlmostEqual(left[0], -right[0], places=6)

    def test_unknown_view_mode_is_rejected(self) -> None:
        with self.assertRaises(PresentationRigError):
            self.rig.view_euler_deg("ORBIT")


class HandPoseSolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rig = load_presentation_rig()
        self.solver = HandPoseSolver(self.rig)

    def test_neutral_pose_is_identity_on_every_required_bone(self) -> None:
        for side in SIDES:
            pose = self.solver.neutral_pose(side)
            self.assertEqual(set(pose.bones_wxyz), set(self.rig.required_bones))
            for bone, value in pose.bones_wxyz.items():
                self.assertAlmostEqual(quaternion_angle_deg(value), 0.0, places=6, msg=bone)

    def test_bend_flexes_toward_the_palm_with_one_shared_sign(self) -> None:
        bend = np.zeros((2, 5, 3))
        bend[:, :, :] = 60.0 / CONTRACT_DEGREE_SCALE
        poses = self.solver.frame_pose(_Frame(bend=bend))
        for side in SIDES:
            for chain in self.rig.chains.values():
                value = poses[side].bones_wxyz[chain.joints[0]]
                # negative rotation about the shared local flexion axis
                self.assertLess(value[1], 0.0, f"{side}/{chain.joints[0]}")
        self.assertEqual(
            poses["LEFT"].bones_wxyz["index_1"], poses["RIGHT"].bones_wxyz["index_1"],
            "the mirrored rig must use identical signs on both hands",
        )

    def test_bend_is_clamped_to_the_anatomical_joint_limit(self) -> None:
        bend = np.zeros((2, 5, 3))
        bend[:, :, :] = 1.0  # 180 degrees before clamping
        poses = self.solver.frame_pose(_Frame(bend=bend))
        for chain in self.rig.chains.values():
            for joint, limit in zip(chain.joints, chain.joint_limits_deg):
                angle = quaternion_angle_deg(poses["RIGHT"].bones_wxyz[joint])
                self.assertLessEqual(angle, limit + 1e-6, joint)

    def test_spread_is_clamped_and_measured_from_the_middle_finger(self) -> None:
        spread = np.zeros((2, 4))
        spread[:, :] = 1.0  # 180 degrees of raw spread on every pair
        poses = self.solver.frame_pose(_Frame(spread=spread))
        low, high = self.rig.spread_clamp_deg
        bound = max(abs(low), abs(high))
        for target in self.rig.spread_targets:
            angle = quaternion_angle_deg(poses["LEFT"].bones_wxyz[target.bone])
            self.assertLessEqual(angle, len(target.sum_of) * bound + 1e-6, target.bone)
        middle_meta = self.rig.chains["middle"].metacarpal
        self.assertAlmostEqual(
            quaternion_angle_deg(poses["LEFT"].bones_wxyz[middle_meta]), 0.0, places=6,
            msg="the middle finger is the spread reference and must not abduct",
        )

    def test_recorded_wrist_rotation_is_hard_clamped(self) -> None:
        """A large recorded palm rotation must never reach the scene unbounded."""

        limit = self.rig.wrist_max_angle_deg
        first = np.tile(np.asarray([1.0, 0.0, 0.0, 0.0]), (2, 1))
        self.solver.frame_pose(_Frame(palm=first))
        big = axis_angle_quaternion("Y", 150.0)
        poses = self.solver.frame_pose(_Frame(palm=np.tile(big, (2, 1))))
        for side in SIDES:
            angle = quaternion_angle_deg(poses[side].bones_wxyz[self.rig.wrist_bone])
            self.assertLessEqual(angle, limit + 1e-6, side)
            self.assertGreater(angle, 0.0, side)

    def test_clamp_preserves_small_rotations_exactly(self) -> None:
        small = axis_angle_quaternion("X", 4.0)
        self.assertAlmostEqual(quaternion_angle_deg(clamp_quaternion_angle(small, 25.0)), 4.0, places=6)

    def test_first_valid_frame_of_a_sequence_has_zero_wrist_delta(self) -> None:
        odd = axis_angle_quaternion("Z", 88.0)
        poses = self.solver.frame_pose(_Frame(palm=np.tile(odd, (2, 1))))
        for side in SIDES:
            self.assertAlmostEqual(
                quaternion_angle_deg(poses[side].bones_wxyz[self.rig.wrist_bone]), 0.0, places=6,
                msg="a sign must start from its own first frame, not from a global reference",
            )

    def test_reset_clears_the_wrist_reference_between_signs(self) -> None:
        """Regression for the reported orbit-after-pressing-a-letter behaviour.

        Without reset the second sign is measured against the first sign's
        reference frame, so it opens at an inherited orientation.
        """

        first_sign = np.tile(axis_angle_quaternion("Y", 30.0), (2, 1))
        self.solver.frame_pose(_Frame(palm=first_sign))

        second_sign = np.tile(axis_angle_quaternion("Y", 95.0), (2, 1))
        without_reset = self.solver.frame_pose(_Frame(palm=second_sign))
        self.assertGreater(
            quaternion_angle_deg(without_reset["LEFT"].bones_wxyz[self.rig.wrist_bone]), 1.0
        )

        self.solver.reset()
        with_reset = self.solver.frame_pose(_Frame(palm=second_sign))
        self.assertAlmostEqual(
            quaternion_angle_deg(with_reset["LEFT"].bones_wxyz[self.rig.wrist_bone]), 0.0, places=6
        )

    def test_invalid_channels_hold_the_last_valid_presentation_value(self) -> None:
        bend = np.zeros((2, 5, 3))
        bend[:, :, :] = 45.0 / CONTRACT_DEGREE_SCALE
        good = _Frame(bend=bend)
        held = self.solver.frame_pose(good)["RIGHT"].bones_wxyz["index_1"]

        dropped = _Frame(bend=bend)
        dropped.bend_valid = np.zeros((2, 5, 3), dtype=bool)
        after = self.solver.frame_pose(dropped)["RIGHT"]
        self.assertEqual(after.bones_wxyz["index_1"], held)
        self.assertEqual(after.bend_valid_count, 0)

    def test_missing_hand_state_is_reported_as_dimmed(self) -> None:
        poses = self.solver.frame_pose(_Frame(state="MISSING"))
        self.assertTrue(poses["LEFT"].dimmed)
        self.assertFalse(self.solver.frame_pose(_Frame(state="OBSERVED"))["LEFT"].dimmed)

    def test_non_finite_palm_input_does_not_produce_a_non_finite_pose(self) -> None:
        broken = np.full((2, 4), np.nan)
        poses = self.solver.frame_pose(_Frame(palm=broken))
        for side in SIDES:
            for value in poses[side].bones_wxyz.values():
                self.assertTrue(all(math.isfinite(v) for v in value))


@unittest.skipUnless(GLB_PATH.is_file(), f"presentation GLB not available at {GLB_PATH}")
class ExportedAssetTests(unittest.TestCase):
    """Structural checks on the exported presentation asset itself."""

    @classmethod
    def setUpClass(cls) -> None:
        from tools.visual_acceptance.rig_probe import GlbHierarchy

        cls.hierarchy = GlbHierarchy(GLB_PATH)
        cls.rig = load_presentation_rig()
        cls.index = {}
        for root in cls.hierarchy.roots:
            side = "LEFT" if "LEFT" in cls.hierarchy.nodes[root].name else "RIGHT"
            cls.index[side] = {node.name: node.index for _, node in cls.hierarchy.walk(root)}

    def test_both_presentation_roots_exist_at_the_origin(self) -> None:
        names = {self.hierarchy.nodes[i].name for i in self.hierarchy.roots}
        self.assertEqual(names, set(self.rig.roots.values()))
        for root in self.hierarchy.roots:
            self.assertEqual(tuple(self.hierarchy.nodes[root].translation), (0.0, 0.0, 0.0))

    def test_every_required_bone_is_present_on_both_hands(self) -> None:
        for side in SIDES:
            missing = [b for b in self.rig.required_bones if b not in self.index[side]]
            self.assertEqual(missing, [], f"{side} is missing bones")

    def test_the_two_hands_are_the_same_persons_hands(self) -> None:
        from tools.visual_acceptance.rig_probe import skinned_vertices

        world = self.hierarchy.world_transforms()
        sizes = {}
        for side in SIDES:
            node = next(
                self.hierarchy.nodes[i]
                for i in self.index[side].values()
                if self.hierarchy.nodes[i].mesh is not None
            )
            vertices = skinned_vertices(self.hierarchy, node, world)
            sizes[side] = vertices.max(0) - vertices.min(0)
        np.testing.assert_allclose(sizes["LEFT"], sizes["RIGHT"], atol=1e-4)

    def test_the_default_pose_faces_the_camera_rather_than_edge_on(self) -> None:
        """TASK-007F exported hands whose depth exceeded their on-screen size,
        which is why they read as a small blob instead of two palms."""

        from tools.visual_acceptance.rig_probe import skinned_vertices

        world = self.hierarchy.world_transforms()
        for side in SIDES:
            node = next(
                self.hierarchy.nodes[i]
                for i in self.index[side].values()
                if self.hierarchy.nodes[i].mesh is not None
            )
            width, height, depth = np.ptp(skinned_vertices(self.hierarchy, node, world), axis=0)
            self.assertGreater(height, 2.0 * depth, f"{side} is not facing the camera")
            self.assertGreater(width, depth, f"{side} is not facing the camera")

    def test_the_asset_carries_real_sculpted_geometry(self) -> None:
        for mesh in self.hierarchy.json["meshes"]:
            for primitive in mesh["primitives"]:
                count = self.hierarchy.json["accessors"][primitive["attributes"]["POSITION"]]["count"]
                self.assertGreater(count, 20000, f"{mesh['name']} looks like an un-subdivided cage")

    def test_the_default_material_is_a_light_skin_tone_not_graphite(self) -> None:
        for material in self.hierarchy.json["materials"]:
            red, green, blue, _ = material["pbrMetallicRoughness"]["baseColorFactor"]
            self.assertGreater(red, 0.35, "base colour is too dark to read as skin")
            self.assertGreater(red, green)
            self.assertGreater(green, blue)
            self.assertLess(material["pbrMetallicRoughness"]["metallicFactor"], 0.05)


@unittest.skipUnless(
    GLB_PATH.is_file() and RUN_ROOT.is_dir(),
    "presentation GLB and external TASK-008 run root are both required",
)
class RealSequenceFramingTests(unittest.TestCase):
    """The acceptance property: playing a sign must not move the composition."""

    @classmethod
    def setUpClass(cls) -> None:
        from tools.visual_acceptance.rig_probe import GlbHierarchy

        cls.hierarchy = GlbHierarchy(GLB_PATH)
        cls.rig = load_presentation_rig()
        cls.index, cls.mesh_node = {}, {}
        for root in cls.hierarchy.roots:
            side = "LEFT" if "LEFT" in cls.hierarchy.nodes[root].name else "RIGHT"
            cls.index[side] = {n.name: n.index for _, n in cls.hierarchy.walk(root)}
            cls.mesh_node[side] = next(
                cls.hierarchy.nodes[i]
                for i in cls.index[side].values()
                if cls.hierarchy.nodes[i].mesh is not None
            )
        cls.base = {
            side: {b: cls.hierarchy.nodes[cls.index[side][b]].rotation_wxyz for b in cls.rig.required_bones}
            for side in SIDES
        }

    def _sequences(self):
        from visualizer.app.integration import load_sequence_for_item
        from visualizer.mapping import Core28Resolver
        from visualizer.queue import PlaybackQueue

        resolver = Core28Resolver(labels_path=LABELS, catalog_path=CATALOG)
        for character in ACCEPTANCE_CHARACTERS:
            queue = PlaybackQueue(resolver)
            item = next(i for i in queue.enqueue_text(character, mode="canonical") if i.item_type == "sign")
            yield character, load_sequence_for_item(item, run_root=RUN_ROOT, manifest_path=MANIFEST)

    def test_hands_stay_framed_separated_and_on_their_own_side(self) -> None:
        from tools.visual_acceptance.rig_probe import skinned_vertices

        solver = HandPoseSolver(self.rig)
        offsets = {side: self.rig.root_position(side)[0] for side in SIDES}
        for character, sequence in self._sequences():
            solver.reset()
            for index, frame in enumerate(sequence.frames):
                poses = solver.frame_pose(frame)
                overrides = {
                    self.index[side][bone]: quaternion_multiply(self.base[side][bone], value)
                    for side in SIDES
                    for bone, value in poses[side].bones_wxyz.items()
                }
                world = self.hierarchy.world_transforms(overrides)
                spans = {}
                for side in SIDES:
                    vertices = skinned_vertices(self.hierarchy, self.mesh_node[side], world)
                    spans[side] = (
                        float(vertices[:, 0].min()) + offsets[side],
                        float(vertices[:, 0].max()) + offsets[side],
                    )
                    self.assertLessEqual(
                        quaternion_angle_deg(poses[side].bones_wxyz[self.rig.wrist_bone]),
                        self.rig.wrist_max_angle_deg + 1e-6,
                        f"{character} {side} frame {index}",
                    )
                self.assertLess(spans["LEFT"][1], 0.0, f"{character} frame {index}: LEFT crossed centre")
                self.assertGreater(spans["RIGHT"][0], 0.0, f"{character} frame {index}: RIGHT crossed centre")
                self.assertLess(
                    spans["LEFT"][1], spans["RIGHT"][0],
                    f"{character} frame {index}: the hands overlap",
                )


if __name__ == "__main__":
    unittest.main()
