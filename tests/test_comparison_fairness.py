from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from evaluation.comparison.common_contract import (
    COMMON_HAND_BONES_18,
    MEDIAPIPE_HAND_CONNECTIONS_21,
    WILOR_HAND_BONES_20,
    HandRecord,
    reconstructed_hand,
)
from evaluation.comparison.harmonized_metrics import (
    choose_representatives,
    evaluate_video,
    frame_weighted_fps,
)
from evaluation.comparison.loaders import InputValidationError, ManifestContract, validate_manifest, validate_wilor_run


def _points(offset: float = 0.0) -> np.ndarray:
    values = np.zeros((21, 3), dtype=np.float64)
    values[:, 0] = np.linspace(0.0 + offset, 0.2 + offset, 21)
    values[:, 1] = np.linspace(0.0, 0.2, 21)
    values[:, 2] = np.linspace(0.0, 0.1, 21)
    return values


def _mediapipe(frame_index: int, label: str, *, offset: float = 0.0, confidence: float = 0.8) -> HandRecord:
    points = _points(offset)
    return HandRecord(
        system="mediapipe",
        frame_index=frame_index,
        hand_present=True,
        handedness_label=label,
        confidence=confidence,
        detection_confidence=None,
        image_landmarks=points.copy(),
        landmarks_3d=points.copy(),
        mano_params=None,
        mano_references=None,
        mode="VIDEO",
        source_index=0,
    )


def _wilor(frame_index: int, label: str = "left") -> HandRecord:
    return HandRecord(
        system="wilor",
        frame_index=frame_index,
        hand_present=True,
        handedness_label=label,
        confidence=0.3,
        detection_confidence=0.3,
        image_landmarks=None,
        landmarks_3d=_points(),
        mano_params={
            "hand_pose_rotmat": [[[1.0, 0.0, 0.0]]],
            "global_orient_rotmat": [[[1.0, 0.0, 0.0]]],
            "betas": [0.0] * 10,
        },
        mano_references={"box_center_xy": [100.0, 100.0], "img_size_wh": [200.0, 200.0]},
        mode="full",
        source_index=0,
    )


class ComparisonFairnessTests(unittest.TestCase):
    def test_detector_only_wilor_run_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "summary.json").write_text(json.dumps({"mode": "detector_only"}), encoding="utf-8")
            contract = ManifestContract(Path("manifest.csv"), tuple(), {}, "manifest-sha")
            with self.assertRaises(InputValidationError):
                validate_wilor_run(run, contract)

    def test_full_wilor_row_passes_reconstructed_predicate(self) -> None:
        self.assertTrue(reconstructed_hand(_wilor(0)))

    def test_nonfinite_wilor_joints_fail_reconstructed_predicate(self) -> None:
        record = _wilor(0)
        record = replace(record, landmarks_3d=record.landmarks_3d.copy())
        record.landmarks_3d[3, 1] = np.nan
        self.assertFalse(reconstructed_hand(record))

    def test_wilor_full_fast_mode_is_not_accepted_as_full(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "summary.json").write_text(
                json.dumps({"mode": "full_fast"}), encoding="utf-8"
            )
            contract = ManifestContract(Path("manifest.csv"), tuple(), {}, "manifest-sha")
            with self.assertRaises(InputValidationError):
                validate_wilor_run(run, contract)

    def test_valid_mediapipe_21_plus_21_passes(self) -> None:
        self.assertTrue(reconstructed_hand(_mediapipe(0, "left")))

    def test_coverage_reports_inclusive_and_exclusive_counts(self) -> None:
        records = [
            _mediapipe(0, "left"),
            _mediapipe(1, "left"),
            _mediapipe(1, "right", offset=0.5),
            _mediapipe(2, "right", offset=0.5),
        ]
        result = evaluate_video("synthetic", records, 4)
        self.assertEqual(result["frames_left_inclusive"], 2)
        self.assertEqual(result["frames_right_inclusive"], 2)
        self.assertEqual(result["frames_both"], 1)
        self.assertEqual(result["frames_left_only"], 1)
        self.assertEqual(result["frames_right_only"], 1)
        self.assertEqual(result["frames_no_hand"], 1)
        self.assertEqual(result["frames_at_least_one"], 3)

    def test_missing_streaks_distinguish_no_hand_and_channels(self) -> None:
        records = [_mediapipe(0, "left"), _mediapipe(1, "right", offset=0.5)]
        result = evaluate_video("synthetic", records, 4)
        self.assertEqual(result["longest_no_hand_streak"], 2)
        self.assertEqual(result["longest_left_missing_streak"], 3)
        self.assertEqual(result["longest_right_missing_streak"], 2)

    def test_common_bones_are_exact_frozen_intersection(self) -> None:
        self.assertEqual(len(COMMON_HAND_BONES_18), 18)
        self.assertEqual(set(COMMON_HAND_BONES_18), set(MEDIAPIPE_HAND_CONNECTIONS_21) & set(WILOR_HAND_BONES_20))

    def test_duplicate_normalization_keeps_highest_native_confidence(self) -> None:
        low = _mediapipe(0, "left", confidence=0.2)
        high = _mediapipe(0, "left", confidence=0.9)
        high = HandRecord(
            system=high.system,
            frame_index=high.frame_index,
            hand_present=high.hand_present,
            handedness_label=high.handedness_label,
            confidence=high.confidence,
            detection_confidence=high.detection_confidence,
            image_landmarks=high.image_landmarks,
            landmarks_3d=high.landmarks_3d,
            mano_params=high.mano_params,
            mano_references=high.mano_references,
            mode=high.mode,
            quality_flags=high.quality_flags,
            source_index=1,
        )
        chosen = choose_representatives([low, high])
        self.assertIs(chosen["left"], high)
        result = evaluate_video("synthetic", [low, high, _mediapipe(0, "right", offset=0.5)], 1)
        self.assertEqual(result["duplicate_left_events"], 1)
        self.assertEqual(result["frames_with_more_than_2_hands"], 1)

    def test_frame_weighted_fps(self) -> None:
        self.assertAlmostEqual(frame_weighted_fps(894, 40.0), 22.35)

    def test_common_swap_heuristic_uses_same_normalized_2d_rule(self) -> None:
        records = [
            _mediapipe(0, "left", offset=-0.3),
            _mediapipe(0, "right", offset=0.3),
            _mediapipe(1, "left", offset=0.3),
            _mediapipe(1, "right", offset=-0.3),
        ]
        result = evaluate_video("synthetic", records, 2)
        self.assertEqual(result["suspected_swap_events"], 1)

    def test_scale_normalized_temporal_metric_is_common(self) -> None:
        records = [_mediapipe(0, "left"), _mediapipe(1, "left", offset=0.01), _mediapipe(2, "left", offset=0.03)]
        result = evaluate_video("synthetic", records, 3)
        metric = result["scale_normalized_temporal_metric"]
        self.assertEqual(metric["operator"], "q[t+1] - 2*q[t] + q[t-1]")
        self.assertEqual(metric["distribution"]["count"], 1)

    def test_manifest_path_validation_rejects_parent_traversal_without_glob(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            path.write_text(
                "sample_id,local_relative_path,checksum_sha256,sign_id,signer_id,split,repetition_id\n"
                "bad,../runs/wilor_karsl_pilot/a.npz,,0171,01,test,lexicographically_first_valid_mp4\n",
                encoding="utf-8",
            )
            with self.assertRaises(InputValidationError):
                validate_manifest(path, expected_sha256=None, verify_video_checksums=False)


if __name__ == "__main__":
    unittest.main()
