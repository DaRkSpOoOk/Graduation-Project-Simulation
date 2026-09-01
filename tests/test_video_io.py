import tempfile
import unittest
from pathlib import Path

import numpy as np

from video_io.reader import inspect_video, iter_video_frames


class TestVideoIO(unittest.TestCase):
    def test_inspection_counts_decoded_frames_and_uses_timestamps(self) -> None:
        import cv2

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.avi"
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                5.0,
                (64, 48),
            )
            if not writer.isOpened():
                self.skipTest("OpenCV MJPG writer is unavailable in this environment")
            for index in range(6):
                writer.write(np.full((48, 64, 3), index * 20, dtype=np.uint8))
            writer.release()

            inspection = inspect_video(path)
            self.assertTrue(inspection.decoder_success)
            self.assertEqual(inspection.decoded_frame_count, 6)
            self.assertEqual((inspection.width, inspection.height), (64, 48))
            self.assertIsNotNone(inspection.fps)
            self.assertGreater(inspection.duration_seconds or 0.0, 0.0)

            frames = list(iter_video_frames(path))
            self.assertEqual([frame.frame_index for frame in frames], list(range(6)))
            self.assertEqual(len({frame.timestamp_ms for frame in frames}), 6)
            self.assertEqual(frames[0].timestamp_seconds, 0.0)
            self.assertTrue(all(right.timestamp_seconds >= left.timestamp_seconds for left, right in zip(frames, frames[1:])))


if __name__ == "__main__":
    unittest.main()
