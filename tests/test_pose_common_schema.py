import unittest

from pose.common.schema import HandPoseFrame, Landmark2D, Landmark3D


class TestPoseCommonSchema(unittest.TestCase):
    def test_optional_fields_are_supported(self) -> None:
        frame = HandPoseFrame(
            frame_index=12,
            timestamp_seconds=0.4,
            hand_present=False,
        )

        self.assertFalse(frame.hand_present)
        self.assertEqual(frame.landmarks_2d, [])
        self.assertEqual(frame.landmarks_3d, [])
        self.assertIsNone(frame.handedness_label)

    def test_serialization_contains_required_keys(self) -> None:
        frame = HandPoseFrame(
            frame_index=2,
            timestamp_seconds=0.066,
            hand_present=True,
            handedness_label="left",
            handedness_confidence=0.95,
            detection_confidence=0.91,
            landmarks_2d=[Landmark2D(0.1, 0.2)],
            landmarks_3d=[Landmark3D(0.1, 0.2, 0.3)],
            wrist_position=Landmark3D(0.1, 0.2, 0.3),
            extractor_metadata={"source": "placeholder"},
            quality_flags=["ok"],
        )

        as_dict = frame.to_dict()
        self.assertEqual(as_dict["frame_index"], 2)
        self.assertEqual(as_dict["handedness_label"], "left")
        self.assertEqual(len(as_dict["landmarks_2d"]), 1)
        self.assertEqual(len(as_dict["landmarks_3d"]), 1)


if __name__ == "__main__":
    unittest.main()
