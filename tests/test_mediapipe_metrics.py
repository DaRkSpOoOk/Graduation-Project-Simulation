import unittest

import numpy as np

from evaluation.metrics.mediapipe_baseline import evaluate_arrays


class TestMediaPipeMetrics(unittest.TestCase):
    def test_detection_missingness_and_runtime_metrics(self) -> None:
        frame_count = 5
        image = np.full((frame_count, 2, 21, 3), np.nan, dtype=np.float32)
        world = np.full_like(image, np.nan)
        present = np.array(
            [
                [True, True],
                [True, False],
                [False, False],
                [False, True],
                [True, True],
            ],
            dtype=bool,
        )
        labels = np.array(
            [
                ["Left", "Right"],
                ["Left", ""],
                ["", ""],
                ["", "Right"],
                ["Left", "Right"],
            ],
            dtype="<U8",
        )
        scores = np.full((frame_count, 2), 0.9, dtype=np.float32)
        for frame_index in range(frame_count):
            for hand_index in range(2):
                if present[frame_index, hand_index]:
                    points = np.arange(21, dtype=np.float32)[:, None] * np.array([[0.01, 0.0, 0.0]], dtype=np.float32)
                    image[frame_index, hand_index] = points
                    world[frame_index, hand_index] = points

        metrics = evaluate_arrays(
            image,
            world,
            present,
            labels,
            scores,
            np.arange(frame_count, dtype=np.float64) / 5.0,
            runtime_seconds=2.0,
            inference_seconds=1.0,
        )
        self.assertEqual(metrics["total_frames"], 5)
        self.assertEqual(metrics["frames_with_no_hands"], 1)
        self.assertEqual(metrics["frames_with_at_least_one_hand"], 4)
        self.assertEqual(metrics["frames_with_both_hands"], 2)
        self.assertEqual(metrics["longest_missing_streak_left_frames"], 2)
        self.assertEqual(metrics["longest_missing_streak_right_frames"], 2)
        self.assertAlmostEqual(metrics["both_hand_detection_rate"], 40.0)
        self.assertAlmostEqual(metrics["effective_processing_fps"], 2.5)
        self.assertEqual(metrics["hand_presence_confidence"], None)
        self.assertEqual(metrics["hand_detection_confidence"], None)
        self.assertEqual(metrics["tracking_confidence"], None)


if __name__ == "__main__":
    unittest.main()
