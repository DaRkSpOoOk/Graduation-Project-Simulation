"""Guards on the tracking-side raw loader.

``tracking.wilor.source.is_complete_wilor_pose`` deliberately restates the
WiLoR half of ``evaluation.comparison.common_contract.reconstructed_hand``
so that ``tracking`` does not import ``evaluation`` at runtime (the pipeline
order is extraction -> tracking -> evaluation). These tests assert the two
predicates cannot silently diverge.
"""

from __future__ import annotations

import unittest

import numpy as np

from evaluation.comparison.common_contract import HandRecord, reconstructed_hand
from tracking.wilor.source import is_complete_wilor_pose, normalize_label


def _mano(valid: bool = True) -> dict:
    if not valid:
        return {"hand_pose_rotmat": [], "global_orient_rotmat": [], "betas": []}
    return {
        "hand_pose_rotmat": np.tile(np.eye(3), (15, 1, 1)).tolist(),
        "global_orient_rotmat": np.eye(3).tolist(),
        "betas": [0.0] * 10,
    }


def _record(
    *,
    mode: str = "full",
    landmarks: np.ndarray | None = None,
    mano: dict | None = None,
    flags: tuple[str, ...] = (),
    present: bool = True,
) -> HandRecord:
    return HandRecord(
        system="wilor",
        frame_index=0,
        hand_present=present,
        handedness_label="left",
        confidence=0.9,
        detection_confidence=0.9,
        image_landmarks=None,
        landmarks_3d=np.zeros((21, 3)) if landmarks is None else landmarks,
        mano_params=_mano() if mano is None else mano,
        mano_references={},
        mode=mode,
        quality_flags=flags,
    )


class TestPredicateParity(unittest.TestCase):
    def _both(self, record: HandRecord) -> tuple[bool, bool]:
        tracking_view = is_complete_wilor_pose(
            record.mode, record.landmarks_3d, record.mano_params, record.quality_flags
        )
        comparison_view = reconstructed_hand(record)
        return tracking_view, comparison_view

    def test_complete_full_mode_pose_accepted_by_both(self) -> None:
        tracking_view, comparison_view = self._both(_record())
        self.assertTrue(tracking_view)
        self.assertEqual(tracking_view, comparison_view)

    def test_detector_only_mode_rejected_by_both(self) -> None:
        tracking_view, comparison_view = self._both(
            _record(mode="detector_only", flags=("detector_only_no_mano",))
        )
        self.assertFalse(tracking_view)
        self.assertEqual(tracking_view, comparison_view)

    def test_missing_mano_rejected_by_both(self) -> None:
        tracking_view, comparison_view = self._both(_record(mano=_mano(valid=False)))
        self.assertFalse(tracking_view)
        self.assertEqual(tracking_view, comparison_view)

    def test_non_finite_joints_rejected_by_both(self) -> None:
        landmarks = np.zeros((21, 3))
        landmarks[3, 1] = np.nan
        tracking_view, comparison_view = self._both(_record(landmarks=landmarks))
        self.assertFalse(tracking_view)
        self.assertEqual(tracking_view, comparison_view)

    def test_wrong_joint_shape_rejected_by_both(self) -> None:
        tracking_view, comparison_view = self._both(_record(landmarks=np.zeros((20, 3))))
        self.assertFalse(tracking_view)
        self.assertEqual(tracking_view, comparison_view)


class TestLabelNormalization(unittest.TestCase):
    def test_recognized_labels(self) -> None:
        self.assertEqual(normalize_label("Left"), "left")
        self.assertEqual(normalize_label(" RIGHT "), "right")

    def test_unrecognized_labels_become_none(self) -> None:
        for value in ("", "both", None, "unknown"):
            self.assertIsNone(normalize_label(value))


if __name__ == "__main__":
    unittest.main()
