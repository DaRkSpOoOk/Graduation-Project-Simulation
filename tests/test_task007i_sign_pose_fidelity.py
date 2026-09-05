"""TASK-007I source-grounded presentation retargeting tests.

These tests exercise the presentation-only landmark guidance added after the
TASK-007H visual audit.  The external TASK-008 run and the local, ignored GLB
assets are optional, so the structural and mathematical checks still run in a
clean checkout while the source-grounded checks skip without those artifacts.
"""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from smart_glove_app.app.hand_pose_solver import (
    HandPoseSolver,
    quaternion_to_matrix_wxyz,
)
from smart_glove_app.rendering.presentation_rig import FINGERS, SIDES, load_presentation_rig
from smart_glove_app.rendering.rig_pose_calibration import GlbPoseCalibration


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT.parent / "graduation-project-runs" / "task008-core28-full"
MANIFEST = PROJECT_ROOT / "datasets" / "manifests" / "karsl_core28.csv"
ASSET_DIR = PROJECT_ROOT / "assets-local" / "blendswap_hands_v1"
ASSET_PATHS = {
    "LEFT": ASSET_DIR / "task007g_hand_left.glb",
    "RIGHT": ASSET_DIR / "task007g_hand_right.glb",
}


def _world_rotations(solver: HandPoseSolver, side: str, bones: dict[str, tuple[float, ...]]):
    calibration = solver._landmark_calibration[side]
    assert calibration is not None
    wrist = solver.rig.wrist_bone
    result = {
        wrist: calibration.world_rotations[wrist]
        @ quaternion_to_matrix_wxyz(bones[wrist])
    }
    for finger in FINGERS:
        parent = wrist
        chain = solver.rig.chains[finger]
        for bone in (chain.metacarpal, *chain.joints):
            result[bone] = (
                result[parent]
                @ calibration.local_rotations[bone]
                @ quaternion_to_matrix_wxyz(bones[bone])
            )
            parent = bone
    return result


class RigCalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rig = load_presentation_rig()

    def test_landmark_mapping_is_a_proper_rotation(self) -> None:
        mapping = np.asarray(
            self.rig.raw["landmark_retargeting"]["source_to_presentation_matrix"],
            dtype=float,
        )
        np.testing.assert_allclose(mapping.T @ mapping, np.eye(3), atol=1e-7)
        self.assertAlmostEqual(float(np.linalg.det(mapping)), 1.0, places=7)

    def test_shortest_swing_reproduces_a_target_direction(self) -> None:
        source = np.asarray((0.0, 1.0, 0.0))
        target = np.asarray((-1.0, 1.0, 0.0))
        target /= np.linalg.norm(target)
        swing = HandPoseSolver._swing_to_direction(
            source,
            target,
            np.asarray((1.0, 0.0, 0.0)),
        )
        np.testing.assert_allclose(swing @ source, target, atol=1e-7)
        np.testing.assert_allclose(swing.T @ swing, np.eye(3), atol=1e-7)
        self.assertAlmostEqual(float(np.linalg.det(swing)), 1.0, places=7)


@unittest.skipUnless(
    RUN_ROOT.is_dir() and all(path.is_file() for path in ASSET_PATHS.values()),
    "external TASK-008 run root and both presentation GLBs are required",
)
class SourceGroundedRetargetingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from visualizer.app.integration import load_sequence_for_item
        from visualizer.mapping import Core28Resolver
        from visualizer.queue import PlaybackQueue

        cls.rig = load_presentation_rig()
        cls.resolver = Core28Resolver()
        queue = PlaybackQueue(cls.resolver)
        item = queue.enqueue_character("م", mode="canonical")
        cls.sequence = load_sequence_for_item(
            item,
            run_root=RUN_ROOT,
            manifest_path=MANIFEST,
        )
        assert cls.sequence is not None

    def test_both_glbs_have_an_immutable_rest_calibration(self) -> None:
        for side, path in ASSET_PATHS.items():
            calibration = GlbPoseCalibration.from_glb(path, self.rig.required_bones)
            self.assertEqual(set(calibration.local_rotations), set(self.rig.required_bones))
            self.assertEqual(set(calibration.world_rotations), set(self.rig.required_bones))
            self.assertEqual(calibration.parent_names[self.rig.wrist_bone], "wrist_root")

    def test_guided_pose_hits_stored_segment_directions(self) -> None:
        solver = HandPoseSolver(self.rig, rig_asset_paths=ASSET_PATHS)
        for frame in self.sequence.frames:
            poses = solver.frame_pose(frame)
            for side in SIDES:
                calibration = solver._landmark_calibration[side]
                assert calibration is not None
                targets = solver._landmark_targets(
                    frame.hand(side).landmarks_3d,
                    side,
                    calibration,
                )
                self.assertIsNotNone(targets)
                assert targets is not None
                actual = _world_rotations(solver, side, dict(poses[side].bones_wxyz))
                wrist = self.rig.wrist_bone
                shape_world = (
                    calibration.world_rotations[wrist]
                    @ quaternion_to_matrix_wxyz(poses[side].bones_wxyz[wrist])
                    @ calibration.world_rotations[wrist].T
                )
                for finger in FINGERS:
                    chain = self.rig.chains[finger]
                    for bone in (chain.metacarpal, *chain.joints):
                        actual_direction = actual[bone][:, 1]
                        target_direction = shape_world @ targets[bone][:, 1]
                        self.assertAlmostEqual(
                            float(np.degrees(np.arccos(np.clip(np.dot(actual_direction, target_direction), -1.0, 1.0)))),
                            0.0,
                            places=4,
                            msg=f"{side}/{bone}",
                        )

    def test_guidance_does_not_mutate_source_arrays_on_partial_masks(self) -> None:
        from visualizer.app.integration import load_sequence_for_item
        from visualizer.mapping import Core28Resolver
        from visualizer.queue import PlaybackQueue

        resolver = Core28Resolver()
        queue = PlaybackQueue(resolver)
        item = queue.enqueue_character("د", mode="canonical")
        sequence = load_sequence_for_item(item, run_root=RUN_ROOT, manifest_path=MANIFEST)
        assert sequence is not None
        before = [
            (
                np.array(frame.bend_normalized, copy=True),
                np.array(frame.spread_normalized, copy=True),
                np.array(frame.bend_valid, copy=True),
                np.array(frame.spread_valid, copy=True),
            )
            for frame in sequence.frames
        ]
        solver = HandPoseSolver(self.rig, rig_asset_paths=ASSET_PATHS)
        for frame in sequence.frames:
            solver.frame_pose(frame)
        for frame, snapshot in zip(sequence.frames, before):
            for current, original in zip(
                (
                    frame.bend_normalized,
                    frame.spread_normalized,
                    frame.bend_valid,
                    frame.spread_valid,
                ),
                snapshot,
            ):
                np.testing.assert_array_equal(
                    current,
                    original,
                    err_msg="presentation retargeting changed a scientific source array",
                )

        track = 0
        self.assertFalse(solver._spread_valid_for_base(sequence.frames[0], track, "ring"))
        self.assertFalse(solver._spread_valid_for_base(sequence.frames[0], track, "pinky"))


if __name__ == "__main__":
    unittest.main()
