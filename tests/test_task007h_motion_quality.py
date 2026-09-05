"""TASK-007H motion-boundary and presentation-transition tests."""

from __future__ import annotations

import inspect
from pathlib import Path
import unittest

import numpy as np

from smart_glove_app.app.hand_pose_solver import (
    HandPose,
    HandPoseSolver,
    axis_angle_quaternion,
    quaternion_angle_deg,
)
from smart_glove_app.app.motion_quality import (
    PresentationTransition,
    TransitionConfig,
    interpolate_pose_maps,
    pose_distance_degrees,
)
from smart_glove_app.app.playback_controller import (
    PersistentPlaybackController,
    PlaybackBoundaryTrace,
)
from smart_glove_app.rendering.presentation_rig import (
    FINGERS,
    SIDES,
    SPREAD_PAIRS,
    load_presentation_rig,
)


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT.parent / "graduation-project-runs" / "task008-core28-full"
MANIFEST = ROOT / "datasets" / "manifests" / "karsl_core28.csv"
LABELS = ROOT / "datasets" / "manifests" / "karsl_core28_labels.csv"
CATALOG = ROOT / "visualizer" / "catalog" / "core28_exemplars.json"


def _pose_map(angle_deg: float) -> dict[str, HandPose]:
    rig = load_presentation_rig()
    solver = HandPoseSolver(rig)
    result: dict[str, HandPose] = {}
    for side in SIDES:
        pose = solver.neutral_pose(side)
        bones = dict(pose.bones_wxyz)
        bones["index_1"] = tuple(
            float(value) for value in axis_angle_quaternion("X", angle_deg)
        )
        result[side] = HandPose(
            side=side,
            bones_wxyz=bones,
            state="OBSERVED",
            dimmed=False,
            bend_valid_count=15,
            spread_valid_count=4,
            wrist_valid=True,
            wrist_angle_deg=0.0,
        )
    return result


class _Frame:
    """Small frozen-frame stand-in for channel-isolation tests."""

    def __init__(self) -> None:
        self.bend_normalized = np.zeros((2, 5, 3), dtype=float)
        self.bend_valid = np.ones((2, 5, 3), dtype=bool)
        self.spread_normalized = np.zeros((2, 4), dtype=float)
        self.spread_valid = np.ones((2, 4), dtype=bool)
        self.palm_quaternion_wxyz = np.tile(
            np.asarray([1.0, 0.0, 0.0, 0.0]), (2, 1)
        )
        self.palm_imu_valid = np.ones(2, dtype=bool)

    def hand(self, side: str):  # noqa: D401 - mirrors the real frame contract
        return type("Hand", (), {"state": "OBSERVED", "present": True})()


class TransitionControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rig = load_presentation_rig()
        self.config = TransitionConfig()

    def test_transition_is_direct_and_uses_slerp_with_eased_parameter(self) -> None:
        first = _pose_map(70.0)
        second = _pose_map(10.0)
        transition = PresentationTransition(
            first,
            second,
            started_at=10.0,
            rig=self.rig,
            config=self.config,
        )

        held = transition.sample(10.04)
        self.assertEqual(held.phase, "BOUNDARY_HOLD")
        self.assertAlmostEqual(
            quaternion_angle_deg(held.poses["LEFT"].bones_wxyz["index_1"]),
            70.0,
            places=5,
        )

        midpoint = transition.sample(10.08 + transition.plan.duration_ms / 1000.0 / 2.0)
        self.assertEqual(midpoint.phase, "BLENDING")
        midpoint_angle = quaternion_angle_deg(
            midpoint.poses["LEFT"].bones_wxyz["index_1"]
        )
        self.assertGreater(midpoint_angle, 10.0)
        self.assertLess(midpoint_angle, 70.0)
        self.assertNotAlmostEqual(midpoint_angle, 0.0, places=5)

        complete = transition.sample(
            10.08 + transition.plan.duration_ms / 1000.0 + 0.001
        )
        self.assertTrue(complete.done)
        self.assertEqual(complete.phase, "COMPLETE")
        self.assertAlmostEqual(
            quaternion_angle_deg(complete.poses["LEFT"].bones_wxyz["index_1"]),
            10.0,
            places=5,
        )

    def test_duration_adapts_to_pose_distance_inside_requested_range(self) -> None:
        small = PresentationTransition(
            _pose_map(20.0),
            _pose_map(10.0),
            started_at=0.0,
            rig=self.rig,
            config=self.config,
        )
        large = PresentationTransition(
            _pose_map(170.0),
            _pose_map(0.0),
            started_at=0.0,
            rig=self.rig,
            config=self.config,
        )
        self.assertGreater(large.plan.distance_degrees, small.plan.distance_degrees)
        self.assertGreater(large.plan.duration_ms, small.plan.duration_ms)
        self.assertGreaterEqual(small.plan.duration_ms, 150.0)
        self.assertLessEqual(large.plan.duration_ms, 350.0)

    def test_interpolation_does_not_mutate_endpoint_pose_maps(self) -> None:
        first = _pose_map(40.0)
        second = _pose_map(90.0)
        before_first = first["LEFT"].bones_wxyz["index_1"]
        before_second = second["LEFT"].bones_wxyz["index_1"]
        result = interpolate_pose_maps(first, second, 0.5)
        self.assertEqual(first["LEFT"].bones_wxyz["index_1"], before_first)
        self.assertEqual(second["LEFT"].bones_wxyz["index_1"], before_second)
        self.assertEqual(result["LEFT"].state, "TRANSITION")


class IndependentRigChannelTests(unittest.TestCase):
    """Verify that each frozen TASK-005 channel reaches its own rig joint."""

    def setUp(self) -> None:
        self.rig = load_presentation_rig()
        self.solver = HandPoseSolver(self.rig)

    def test_all_fifteen_bend_channels_are_independent(self) -> None:
        for finger_index, finger in enumerate(FINGERS):
            chain = self.rig.chains[finger]
            for joint_index, target_bone in enumerate(chain.joints):
                self.solver.reset()
                frame = _Frame()
                frame.bend_normalized[:, finger_index, joint_index] = 45.0 / 180.0
                for spread_index, pair in enumerate(SPREAD_PAIRS):
                    frame.spread_normalized[:, spread_index] = (
                        self.rig.spread_neutral_deg[pair] / 180.0
                    )
                poses = self.solver.frame_pose(frame)
                for side in SIDES:
                    for bone, quaternion in poses[side].bones_wxyz.items():
                        angle = quaternion_angle_deg(quaternion)
                        if bone == target_bone:
                            self.assertGreater(
                                angle,
                                1.0,
                                f"{side}/{finger}[{joint_index}] did not move",
                            )
                        else:
                            self.assertAlmostEqual(
                                angle,
                                0.0,
                                places=6,
                                msg=f"{side}/{finger}[{joint_index}] also moved {bone}",
                            )

    def test_each_spread_channel_reaches_only_its_declared_base_targets(self) -> None:
        expected_targets = {
            "thumb-index": {"thumb_meta"},
            "index-middle": {"thumb_meta", "index_meta"},
            "middle-ring": {"ring_meta", "pinky_meta"},
            "ring-pinky": {"pinky_meta"},
        }
        for spread_index, pair in enumerate(SPREAD_PAIRS):
            self.solver.reset()
            frame = _Frame()
            for index, name in enumerate(SPREAD_PAIRS):
                frame.spread_normalized[:, index] = (
                    self.rig.spread_neutral_deg[name] / 180.0
                )
            frame.spread_normalized[:, spread_index] += 20.0 / 180.0
            poses = self.solver.frame_pose(frame)
            spread_bones = {target.bone for target in self.rig.spread_targets}
            for side in SIDES:
                for bone in spread_bones:
                    angle = quaternion_angle_deg(poses[side].bones_wxyz[bone])
                    if bone in expected_targets[pair]:
                        self.assertGreater(angle, 1.0, f"{side}/{pair} did not move")
                    else:
                        self.assertAlmostEqual(
                            angle,
                            0.0,
                            places=6,
                            msg=f"{side}/{pair} also moved {bone}",
                        )


class PlaybackBoundaryTraceTests(unittest.TestCase):
    def test_nominal_60hz_playback_visits_every_source_anchor_and_terminal_frame(
        self,
    ) -> None:
        timestamps = tuple(index / 30.0 for index in range(20))
        frame_indices = tuple(range(100, 120))
        playback = PersistentPlaybackController(timestamps, frame_indices)
        trace = PlaybackBoundaryTrace.for_sequence("sample", "م", frame_indices)

        state = playback.play(0.0)
        trace.record(state.position)
        now = 0.0
        while playback.playing:
            now += 1.0 / 60.0
            state = playback.tick(now)
            trace.record(state.position)
        trace.mark_queue_advance()

        self.assertEqual(trace.displayed_positions, list(range(20)))
        self.assertEqual(trace.first_source_frame, 100)
        self.assertEqual(trace.final_source_frame, 119)
        self.assertTrue(trace.all_source_positions_presented)
        self.assertTrue(trace.last_frame_presented)
        self.assertTrue(trace.queue_advanced_after_final)
        self.assertFalse(trace.early_queue_advance)
        self.assertFalse(playback.playing)
        self.assertTrue(playback.at_end)

    def test_queue_advance_before_terminal_anchor_is_explicitly_detected(self) -> None:
        trace = PlaybackBoundaryTrace.for_sequence("sample", "ا", (0, 1, 2))
        trace.record(0)
        trace.mark_queue_advance()
        self.assertFalse(trace.last_frame_presented)
        self.assertFalse(trace.queue_advanced_after_final)
        self.assertTrue(trace.early_queue_advance)


@unittest.skipUnless(RUN_ROOT.is_dir(), "external TASK-008 run root is not present")
class Core28BoundaryAuditTests(unittest.TestCase):
    def test_all_canonical_core28_sequences_are_complete_and_contiguous(self) -> None:
        from visualizer.app.integration import load_sequence_for_item
        from visualizer.mapping import Core28Resolver
        from visualizer.queue import PlaybackQueue

        resolver = Core28Resolver(labels_path=LABELS, catalog_path=CATALOG)
        self.assertEqual(len(resolver.catalog.entries), 28)
        for character in resolver.supported_characters():
            queue = PlaybackQueue(resolver)
            item = queue.enqueue_character(character, mode="canonical")
            sequence = load_sequence_for_item(
                item, run_root=RUN_ROOT, manifest_path=MANIFEST
            )
            self.assertIsNotNone(sequence, character)
            assert sequence is not None
            self.assertEqual(sequence.frame_indices, tuple(range(len(sequence))))
            self.assertTrue(
                all(hand.present for frame in sequence.frames for hand in frame.hands),
                character,
            )
            self.assertTrue(
                np.isfinite(np.asarray(sequence.timestamps)).all(), character
            )


class RecognitionIsolationTests(unittest.TestCase):
    def test_transition_api_cannot_be_passed_to_the_recognition_bridge(self) -> None:
        from smart_glove_app.app.recognition_bridge import RecognitionBridge

        bridge = RecognitionBridge()
        self.assertNotIn(
            "PresentationTransition", inspect.signature(bridge.predict).parameters
        )
        transition = PresentationTransition(
            _pose_map(0.0),
            _pose_map(30.0),
            started_at=0.0,
            rig=load_presentation_rig(),
            config=TransitionConfig(),
        )
        self.assertEqual(transition.sample(0.01).phase, "BOUNDARY_HOLD")
        self.assertIsNone(bridge.predict(object()))

    def test_repeated_core28_characters_remain_two_queue_events(self) -> None:
        from visualizer.mapping import Core28Resolver
        from visualizer.queue import PlaybackQueue

        resolver = Core28Resolver(labels_path=LABELS, catalog_path=CATALOG)
        queue = PlaybackQueue(resolver)
        items = queue.enqueue_text("مم", mode="canonical")
        self.assertEqual(len(items), 2)
        self.assertEqual([item.character for item in items], ["م", "م"])
        self.assertEqual([item.sample_id for item in items], [items[0].sample_id] * 2)


if __name__ == "__main__":
    unittest.main()
