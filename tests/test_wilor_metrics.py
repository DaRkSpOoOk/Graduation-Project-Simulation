import unittest

from evaluation.metrics.hand_pose_metrics import (
    compute_bone_length_variation,
    compute_detection_stats,
    compute_hand_count_changes,
    compute_potential_handedness_swaps,
    compute_wrist_jitter,
)
from pose.common.schema import HandPoseFrame, Landmark3D


def _frame(frame_index, hand_present, label=None, wrist=None):
    return HandPoseFrame(
        frame_index=frame_index,
        timestamp_seconds=frame_index / 30.0,
        hand_present=hand_present,
        handedness_label=label,
        wrist_position=wrist,
    )


class TestDetectionStats(unittest.TestCase):
    def test_counts_and_missing_streak(self) -> None:
        frames = [
            _frame(0, True, "left"),
            _frame(0, True, "right"),
            _frame(1, False),
            _frame(2, False),
            _frame(3, True, "right"),
            _frame(4, True, "left"),
        ]
        stats = compute_detection_stats(frames)
        self.assertEqual(stats.total_frames, 5)
        self.assertEqual(stats.frames_both, 1)
        self.assertEqual(stats.frames_no_hand, 2)
        self.assertEqual(stats.frames_right_only, 1)
        self.assertEqual(stats.frames_left_only, 1)
        self.assertEqual(stats.longest_missing_streak, 2)
        self.assertAlmostEqual(stats.missing_frame_pct, 40.0)

    def test_empty_input(self) -> None:
        stats = compute_detection_stats([])
        self.assertEqual(stats.total_frames, 0)
        self.assertEqual(stats.missing_frame_pct, 0.0)


class TestWristJitter(unittest.TestCase):
    def test_displacement_between_consecutive_same_hand_frames(self) -> None:
        frames = [
            _frame(0, True, "right", Landmark3D(0.0, 0.0, 0.0)),
            _frame(1, True, "right", Landmark3D(3.0, 4.0, 0.0)),  # distance 5
            _frame(2, True, "right", Landmark3D(3.0, 4.0, 0.0)),  # distance 0
        ]
        jitter = compute_wrist_jitter(frames)
        self.assertIn("right", jitter)
        self.assertEqual(jitter["right"].n_samples, 2)
        self.assertAlmostEqual(jitter["right"].max, 5.0)
        self.assertAlmostEqual(jitter["right"].mean, 2.5)

    def test_no_wrist_position_yields_empty(self) -> None:
        frames = [_frame(0, True, "right", None)]
        jitter = compute_wrist_jitter(frames)
        self.assertEqual(jitter, {})


class TestHandCountChanges(unittest.TestCase):
    def test_detects_sudden_count_change(self) -> None:
        frames = [
            _frame(0, True, "left"),
            _frame(0, True, "right"),
            _frame(1, True, "left"),
            _frame(2, False),
        ]
        changes = compute_hand_count_changes(frames)
        self.assertEqual(len(changes), 2)
        self.assertEqual((changes[0].previous_count, changes[0].current_count), (2, 1))
        self.assertEqual((changes[1].previous_count, changes[1].current_count), (1, 0))


class TestBoneLengthVariation(unittest.TestCase):
    def test_zero_variation_for_identical_rigid_frames(self) -> None:
        joints = [Landmark3D(x=float(i), y=0.0, z=0.0) for i in range(21)]
        frames = [
            HandPoseFrame(
                frame_index=i,
                timestamp_seconds=i / 30.0,
                hand_present=True,
                handedness_label="right",
                landmarks_3d=joints,
            )
            for i in range(3)
        ]
        variation = compute_bone_length_variation(frames)
        self.assertIn("right", variation)
        self.assertAlmostEqual(variation["right"].max, 0.0)

    def test_missing_joints_are_skipped_not_fabricated(self) -> None:
        frames = [_frame(0, True, "right")]  # no landmarks_3d
        variation = compute_bone_length_variation(frames)
        self.assertEqual(variation, {})


class TestPotentialHandednessSwaps(unittest.TestCase):
    def test_flags_spatially_inconsistent_relabeling(self) -> None:
        frame0 = [
            _frame(0, True, "left", Landmark3D(0.0, 0.0, 0.0)),
            _frame(0, True, "right", Landmark3D(10.0, 0.0, 0.0)),
        ]
        # Labels look swapped: new "left" is near old "right" and vice versa.
        frame1 = [
            _frame(1, True, "left", Landmark3D(10.0, 0.0, 1.0)),
            _frame(1, True, "right", Landmark3D(0.0, 0.0, 1.0)),
        ]
        candidates = compute_potential_handedness_swaps(frame0 + frame1)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].frame_index, 1)
        self.assertLess(candidates[0].swapped_cost, candidates[0].unswapped_cost)

    def test_consistent_motion_is_not_flagged(self) -> None:
        frame0 = [
            _frame(0, True, "left", Landmark3D(0.0, 0.0, 0.0)),
            _frame(0, True, "right", Landmark3D(10.0, 0.0, 0.0)),
        ]
        frame1 = [
            _frame(1, True, "left", Landmark3D(0.5, 0.0, 0.0)),
            _frame(1, True, "right", Landmark3D(10.5, 0.0, 0.0)),
        ]
        candidates = compute_potential_handedness_swaps(frame0 + frame1)
        self.assertEqual(candidates, [])


if __name__ == "__main__":
    unittest.main()
