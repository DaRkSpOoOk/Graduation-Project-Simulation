import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np  # noqa: F401

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from pose.common.schema import HandPoseFrame, Landmark3D

if HAS_NUMPY:
    from pose.wilor.npz_io import hand_pose_frames_from_npz, save_raw_video_output


@unittest.skipUnless(HAS_NUMPY, "numpy not installed (see pose/wilor/requirements.txt)")
class TestWilorNpzIo(unittest.TestCase):
    def test_round_trip_preserves_presence_and_failure_frames(self) -> None:
        frames = [
            HandPoseFrame(
                frame_index=0,
                timestamp_seconds=0.0,
                hand_present=True,
                handedness_label="right",
                detection_confidence=0.91,
                landmarks_3d=[Landmark3D(x=float(i), y=float(i) * 2, z=float(i) * 3) for i in range(21)],
                wrist_position=Landmark3D(x=0.0, y=0.0, z=0.0),
                mano_params={"betas": [0.1] * 10},
                mano_references={"camera_translation_xyz": [0.1, 0.2, 0.3]},
                extractor_metadata={"extractor": "wilor", "mode": "full"},
            ),
            HandPoseFrame(
                frame_index=1,
                timestamp_seconds=0.033,
                hand_present=False,
                extractor_metadata={"extractor": "wilor", "mode": "full"},
                quality_flags=["extraction_failed", "error:RuntimeError"],
            ),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            npz_path = save_raw_video_output(Path(tmp), "sample_1", frames)
            self.assertTrue(npz_path.exists())

            restored = hand_pose_frames_from_npz(npz_path)

        self.assertEqual(len(restored), 2)
        self.assertTrue(restored[0].hand_present)
        self.assertEqual(restored[0].handedness_label, "right")
        self.assertAlmostEqual(restored[0].detection_confidence, 0.91, places=5)
        self.assertEqual(len(restored[0].landmarks_3d), 21)
        self.assertEqual(restored[0].mano_params["betas"], [0.1] * 10)

        self.assertFalse(restored[1].hand_present)
        self.assertIn("extraction_failed", restored[1].quality_flags)
        self.assertEqual(restored[1].landmarks_3d, [])

    def test_no_interpolation_of_missing_frames(self) -> None:
        """A gap frame must round-trip as hand_present=False, never as an
        interpolated/guessed position (Task 4: raw output immutability)."""
        frames = [
            HandPoseFrame(frame_index=0, timestamp_seconds=0.0, hand_present=True, handedness_label="left"),
            HandPoseFrame(frame_index=1, timestamp_seconds=0.033, hand_present=False),
            HandPoseFrame(frame_index=2, timestamp_seconds=0.066, hand_present=True, handedness_label="left"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            npz_path = save_raw_video_output(Path(tmp), "sample_2", frames)
            restored = hand_pose_frames_from_npz(npz_path)

        self.assertEqual([f.hand_present for f in restored], [True, False, True])


if __name__ == "__main__":
    unittest.main()
