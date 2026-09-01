import unittest

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

if HAS_NUMPY:
    from pose.wilor.frame_extraction import extract_frame_detector_only


class _FakeTensor:
    """Minimal numpy-backed stand-in for the .detach().cpu().numpy()/.item()
    duck-typed interface our adapter uses on ultralytics' torch tensors, so
    this test does not need torch installed (see pose/wilor/requirements.txt
    -- torch is a heavy optional dependency, this is a lightweight test)."""

    def __init__(self, array):
        self._array = np.asarray(array)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._array

    def squeeze(self):
        return _FakeTensor(self._array.squeeze())

    def item(self):
        return self._array.item()

    def __len__(self):
        return len(self._array)

    def __getitem__(self, i):
        return _FakeTensor(self._array[i])


class _FakeBoxes:
    def __init__(self, data_rows, classes):
        self.data = _FakeTensor(np.array(data_rows, dtype=float))
        self.cls = _FakeTensor(np.array(classes, dtype=float))

    def __len__(self):
        return len(self.data._array)


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeDetector:
    def __init__(self, result):
        self._result = result

    def __call__(self, frame_bgr, conf, verbose=False):
        return [self._result]


@unittest.skipUnless(HAS_NUMPY, "numpy not installed (see pose/wilor/requirements.txt)")
class TestExtractFrameDetectorOnly(unittest.TestCase):
    def test_no_detection_marks_hand_absent(self) -> None:
        detector = _FakeDetector(_FakeResult(_FakeBoxes([], [])))
        frame = np.zeros((10, 10, 3), dtype=np.uint8)

        frames = extract_frame_detector_only(
            detector, frame, frame_index=5, timestamp_seconds=0.166,
            confidence_threshold=0.3, extractor_version="test@0",
        )

        self.assertEqual(len(frames), 1)
        self.assertFalse(frames[0].hand_present)
        self.assertIn("no_hand_detected", frames[0].quality_flags)
        self.assertIn("detector_only_no_mano", frames[0].quality_flags)

    def test_two_hands_produce_left_and_right_records(self) -> None:
        # cls: 0.0 -> left, 1.0 -> right (WiLoR detector convention)
        boxes = _FakeBoxes(
            data_rows=[[10, 20, 30, 40, 0.95], [50, 60, 70, 80, 0.88]],
            classes=[0.0, 1.0],
        )
        detector = _FakeDetector(_FakeResult(boxes))
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        frames = extract_frame_detector_only(
            detector, frame, frame_index=1, timestamp_seconds=0.033,
            confidence_threshold=0.3, extractor_version="test@0",
        )

        self.assertEqual(len(frames), 2)
        labels = {f.handedness_label for f in frames}
        self.assertEqual(labels, {"left", "right"})
        for f in frames:
            self.assertTrue(f.hand_present)
            self.assertEqual(len(f.landmarks_2d), 4)
            self.assertIsNotNone(f.detection_confidence)
            self.assertEqual(f.extractor_metadata["mode"], "detector_only")


if __name__ == "__main__":
    unittest.main()
