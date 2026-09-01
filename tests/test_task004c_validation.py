"""Tests for the TASK-004C validation layer.

All fixtures are synthetic. Nothing here needs the WiLoR checkpoint, MANO
assets, KArSL videos, or a generated run directory.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.annotations.task004b import AnnotationRow
from evaluation.tracking.validation import (
    DECISION_MARGIN_PX,
    ValidationError,
    align,
    ambiguity_calibration,
    false_presence,
    identity_accuracy,
    identity_decision,
    identity_switches,
    load_tracked_frames,
    occlusion_state_validity,
    reacquisition,
    reference_points,
    verify_raw_integrity,
    visibility_recall,
)
from tracking.wilor.schema import TrackState
from evaluation.tracking.validation import TrackedFrameView, TrackedHandView

W, H = 1920.0, 1080.0


def annotation(
    frame_index: int,
    *,
    sample_id: str = "clip",
    left: str = "VISIBLE",
    right: str = "VISIBLE",
    left_xy: tuple[float, float] | None = (0.7, 0.5),
    right_xy: tuple[float, float] | None = (0.3, 0.5),
    flags: tuple[str, ...] = (),
    confidence: str = "HIGH",
) -> AnnotationRow:
    return AnnotationRow(
        sample_id=sample_id,
        frame_index=frame_index,
        left_visibility=left,
        left_x=None if left_xy is None else left_xy[0],
        left_y=None if left_xy is None else left_xy[1],
        right_visibility=right,
        right_x=None if right_xy is None else right_xy[0],
        right_y=None if right_xy is None else right_xy[1],
        scene_flags=flags,
        annotator_confidence=confidence,
        notes="",
    )


def hand(
    track: str,
    state: TrackState,
    centre: tuple[float, float] | None,
    *,
    raw_index: int | None = 0,
    detector_label: str | None = None,
) -> TrackedHandView:
    return TrackedHandView(
        track=track,
        state=state,
        raw_detection_index=raw_index,
        detector_label=detector_label or track,
        box_center_xy=centre,
        wrist_xy=centre,
    )


def frame(
    frame_index: int,
    left: TrackedHandView,
    right: TrackedHandView,
    *,
    sample_id: str = "clip",
    raw_detections: int = 2,
    flags: tuple[str, ...] = (),
) -> TrackedFrameView:
    return TrackedFrameView(
        sample_id=sample_id,
        frame_index=frame_index,
        left=left,
        right=right,
        number_of_raw_detections=raw_detections,
        extra_detection_count=0,
        tracking_flags=flags,
    )


#: A frame whose tracker positions match the annotation exactly.
def aligned_frame(index: int, *, swapped: bool = False, **kwargs) -> TrackedFrameView:
    left_px = (0.7 * (W - 1), 0.5 * (H - 1))
    right_px = (0.3 * (W - 1), 0.5 * (H - 1))
    if swapped:
        left_px, right_px = right_px, left_px
    return frame(
        index,
        hand("left", TrackState.OBSERVED, left_px),
        hand("right", TrackState.OBSERVED, right_px),
        **kwargs,
    )


class TestAlignment(unittest.TestCase):
    def test_aligns_on_sample_and_frame_index(self) -> None:
        rows = [annotation(0), annotation(1)]
        tracked = {"clip": {0: aligned_frame(0), 1: aligned_frame(1)}}
        pairs = align(rows, tracked)
        self.assertEqual([p[0].frame_index for p in pairs], [0, 1])

    def test_duplicate_annotation_key_hard_fails(self) -> None:
        rows = [annotation(0), annotation(0)]
        tracked = {"clip": {0: aligned_frame(0)}}
        with self.assertRaises(ValidationError):
            align(rows, tracked)

    def test_missing_annotated_clip_hard_fails(self) -> None:
        with self.assertRaises(ValidationError):
            align([annotation(0)], {})

    def test_missing_tracker_frame_hard_fails(self) -> None:
        with self.assertRaises(ValidationError):
            align([annotation(5)], {"clip": {0: aligned_frame(0)}})


class TestReferencePoints(unittest.TestCase):
    def test_normalized_points_use_width_minus_one(self) -> None:
        points = reference_points(annotation(0, left_xy=(1.0, 1.0), right_xy=(0.0, 0.0)))
        self.assertAlmostEqual(points["left"][0], W - 1.0)
        self.assertAlmostEqual(points["left"][1], H - 1.0)
        self.assertAlmostEqual(points["right"][0], 0.0)

    def test_missing_coordinates_are_omitted(self) -> None:
        points = reference_points(annotation(0, left_xy=None))
        self.assertNotIn("left", points)
        self.assertIn("right", points)


class TestVisibilityScoring(unittest.TestCase):
    def test_pose_present_for_visible_hand_counts_as_recall(self) -> None:
        result = visibility_recall([(annotation(0), aligned_frame(0))])
        self.assertEqual(result["overall"]["expected"], 2)
        self.assertEqual(result["overall"]["recall_pct"], 100.0)

    def test_missing_pose_for_visible_hand_is_a_miss(self) -> None:
        broken = frame(
            0,
            hand("left", TrackState.MISSING, None, raw_index=None),
            hand("right", TrackState.OBSERVED, (0.3 * (W - 1), 0.5 * (H - 1))),
        )
        result = visibility_recall([(annotation(0), broken)])
        self.assertEqual(result["left"]["recall_pct"], 0.0)
        self.assertEqual(result["right"]["recall_pct"], 100.0)
        self.assertEqual(len(result["misses"]), 1)

    def test_fully_occluded_hand_is_never_a_recall_miss(self) -> None:
        row = annotation(0, right="FULLY_OCCLUDED", right_xy=None)
        view = frame(
            0,
            hand("left", TrackState.OBSERVED, (0.7 * (W - 1), 0.5 * (H - 1))),
            hand("right", TrackState.LIKELY_OCCLUDED, None, raw_index=None),
        )
        result = visibility_recall([(row, view)])
        self.assertEqual(result["overall"]["expected"], 1)
        self.assertEqual(result["overall"]["recall_pct"], 100.0)

    def test_partial_and_visible_are_reported_separately(self) -> None:
        rows = [annotation(0), annotation(1, left="PARTIALLY_OCCLUDED")]
        views = [aligned_frame(0), aligned_frame(1)]
        result = visibility_recall(list(zip(rows, views)))
        self.assertEqual(result["fully_visible"]["expected"], 3)
        self.assertEqual(result["partially_occluded"]["expected"], 1)


class TestFalsePresence(unittest.TestCase):
    def test_pose_during_full_occlusion_is_false_presence(self) -> None:
        row = annotation(0, right="FULLY_OCCLUDED", right_xy=None)
        result = false_presence([(row, aligned_frame(0))])
        self.assertEqual(result["considered_hand_instances"], 1)
        self.assertEqual(result["false_presence_count"], 1)
        self.assertEqual(result["false_presence_rate_pct"], 100.0)

    def test_no_pose_during_full_occlusion_is_clean(self) -> None:
        row = annotation(0, right="FULLY_OCCLUDED", right_xy=None)
        view = frame(
            0,
            hand("left", TrackState.OBSERVED, (0.7 * (W - 1), 0.5 * (H - 1))),
            hand("right", TrackState.LIKELY_OCCLUDED, None, raw_index=None),
        )
        result = false_presence([(row, view)])
        self.assertEqual(result["false_presence_count"], 0)


class TestIdentityMatching(unittest.TestCase):
    def test_matching_positions_are_correct_identity(self) -> None:
        decision = identity_decision(annotation(0), aligned_frame(0))
        self.assertTrue(decision.evaluable)
        self.assertTrue(decision.correct)
        self.assertGreater(decision.margin_px, DECISION_MARGIN_PX)

    def test_swapped_positions_are_incorrect_identity(self) -> None:
        decision = identity_decision(annotation(0), aligned_frame(0, swapped=True))
        self.assertTrue(decision.evaluable)
        self.assertFalse(decision.correct)

    def test_human_ambiguous_frames_are_excluded_not_failed(self) -> None:
        row = annotation(0, right="AMBIGUOUS", flags=("IDENTITY_AMBIGUOUS",))
        decision = identity_decision(row, aligned_frame(0, swapped=True))
        self.assertFalse(decision.evaluable)
        self.assertIsNone(decision.correct)
        self.assertEqual(decision.reason, "human_identity_ambiguous")
        summary = identity_accuracy([(row, aligned_frame(0, swapped=True))])
        self.assertEqual(summary["evaluable_frames"], 0)
        self.assertEqual(summary["incorrect_frames"], 0)
        self.assertEqual(summary["excluded_reasons"]["human_identity_ambiguous"], 1)

    def test_low_confidence_frames_are_excluded(self) -> None:
        decision = identity_decision(annotation(0, confidence="LOW"), aligned_frame(0))
        self.assertFalse(decision.evaluable)
        self.assertEqual(decision.reason, "low_annotator_confidence")

    def test_missing_reference_points_are_excluded(self) -> None:
        decision = identity_decision(annotation(0, left_xy=None), aligned_frame(0))
        self.assertEqual(decision.reason, "insufficient_reference_points")

    def test_absent_tracker_pose_is_excluded(self) -> None:
        view = frame(
            0,
            hand("left", TrackState.MISSING, None, raw_index=None),
            hand("right", TrackState.OBSERVED, (0.3 * (W - 1), 0.5 * (H - 1))),
        )
        self.assertEqual(identity_decision(annotation(0), view).reason, "tracker_pose_unavailable")

    def test_near_tie_is_reported_indeterminate(self) -> None:
        centre = (0.5 * (W - 1), 0.5 * (H - 1))
        view = frame(
            0,
            hand("left", TrackState.OBSERVED, centre),
            hand("right", TrackState.OBSERVED, centre),
        )
        decision = identity_decision(annotation(0), view)
        self.assertTrue(decision.evaluable)
        self.assertFalse(decision.decisive)
        summary = identity_accuracy([(annotation(0), view)])
        self.assertEqual(summary["indeterminate_frames"], 1)
        self.assertEqual(summary["decisive_frames"], 0)


class TestIdentitySwitches(unittest.TestCase):
    def test_persistent_swap_is_a_confirmed_switch(self) -> None:
        pairs = [(annotation(i), aligned_frame(i)) for i in range(3)]
        pairs += [(annotation(i), aligned_frame(i, swapped=True)) for i in range(3, 7)]
        result = identity_switches(pairs)
        self.assertEqual(result["confirmed_switches"], 1)
        self.assertEqual(result["confirmed_detail"][0]["start_frame"], 3)

    def test_single_frame_wobble_is_not_confirmed(self) -> None:
        pairs = [(annotation(i), aligned_frame(i)) for i in range(3)]
        pairs.append((annotation(3), aligned_frame(3, swapped=True)))
        pairs += [(annotation(i), aligned_frame(i)) for i in range(4, 7)]
        result = identity_switches(pairs)
        self.assertEqual(result["confirmed_switches"], 0)
        self.assertEqual(result["suspected_unresolved_switches"], 1)

    def test_clean_sequence_has_no_switches(self) -> None:
        pairs = [(annotation(i), aligned_frame(i)) for i in range(6)]
        result = identity_switches(pairs)
        self.assertEqual(result["confirmed_switches"], 0)
        self.assertEqual(result["suspected_unresolved_switches"], 0)


class TestReacquisition(unittest.TestCase):
    def test_correct_return_after_annotated_absence(self) -> None:
        pairs = []
        for i in range(2):
            pairs.append((annotation(i), aligned_frame(i)))
        for i in range(2, 4):
            row = annotation(i, right="FULLY_OCCLUDED", right_xy=None)
            view = frame(
                i,
                hand("left", TrackState.OBSERVED, (0.7 * (W - 1), 0.5 * (H - 1))),
                hand("right", TrackState.LIKELY_OCCLUDED, None, raw_index=None),
            )
            pairs.append((row, view))
        pairs.append((annotation(4), aligned_frame(4)))
        result = reacquisition(pairs)
        self.assertEqual(result["events"], 1)
        self.assertEqual(result["correct"], 1)
        self.assertEqual(result["accuracy_pct"], 100.0)
        self.assertEqual(result["detail"][0]["absent_frames"], 2)

    def test_return_to_the_wrong_track_is_incorrect(self) -> None:
        pairs = [(annotation(0), aligned_frame(0))]
        row = annotation(1, right="FULLY_OCCLUDED", right_xy=None)
        pairs.append(
            (
                row,
                frame(
                    1,
                    hand("left", TrackState.OBSERVED, (0.7 * (W - 1), 0.5 * (H - 1))),
                    hand("right", TrackState.MISSING, None, raw_index=None),
                ),
            )
        )
        pairs.append((annotation(2), aligned_frame(2, swapped=True)))
        result = reacquisition(pairs)
        self.assertEqual(result["events"], 1)
        self.assertEqual(result["incorrect"], 1)

    def test_no_annotated_absence_yields_no_events(self) -> None:
        pairs = [(annotation(i), aligned_frame(i)) for i in range(4)]
        self.assertEqual(reacquisition(pairs)["events"], 0)


class TestOcclusionAndAmbiguity(unittest.TestCase):
    def test_likely_occluded_is_cross_tabulated_not_scored(self) -> None:
        row = annotation(0, right="FULLY_OCCLUDED", right_xy=None)
        view = frame(
            0,
            hand("left", TrackState.OBSERVED, (0.7 * (W - 1), 0.5 * (H - 1))),
            hand("right", TrackState.LIKELY_OCCLUDED, None, raw_index=None),
        )
        result = occlusion_state_validity([(row, view)])
        self.assertEqual(result["tracker_likely_occluded_hand_instances"], 1)
        self.assertEqual(result["human_state_counts"]["FULLY_OCCLUDED"], 1)

    def test_ambiguity_overlap_is_counted(self) -> None:
        row = annotation(0, right="AMBIGUOUS", flags=("IDENTITY_AMBIGUOUS", "HAND_CROSSING"))
        view = frame(
            0,
            hand("left", TrackState.AMBIGUOUS, (0.7 * (W - 1), 0.5 * (H - 1))),
            hand("right", TrackState.AMBIGUOUS, (0.3 * (W - 1), 0.5 * (H - 1))),
        )
        result = ambiguity_calibration([(row, view)])
        self.assertEqual(result["tracker_ambiguous_frames"], 1)
        self.assertEqual(result["human_ambiguous_frames"], 1)
        self.assertEqual(result["overlap_frames"], 1)
        self.assertEqual(result["tracker_ambiguous_on_crossing_frames"], 1)

    def test_human_ambiguous_but_tracker_confident_is_recorded(self) -> None:
        row = annotation(0, right="AMBIGUOUS", flags=("IDENTITY_AMBIGUOUS",))
        result = ambiguity_calibration([(row, aligned_frame(0))])
        self.assertEqual(result["human_ambiguous_but_tracker_confident"], 1)
        self.assertEqual(result["overlap_frames"], 0)


class TestRawIntegrityAndSchema(unittest.TestCase):
    def test_raw_integrity_detects_a_changed_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "clip"
            sample.mkdir()
            raw = root / "wilor_raw.npz"
            raw.write_bytes(b"original")
            import hashlib

            digest = hashlib.sha256(b"original").hexdigest()
            (sample / "wilor_tracked_meta.json").write_text(
                json.dumps({"source": {"raw_npz": str(raw), "raw_npz_sha256": digest}})
            )
            self.assertTrue(verify_raw_integrity(root)["all_unchanged"])
            raw.write_bytes(b"tampered")
            result = verify_raw_integrity(root)
            self.assertFalse(result["all_unchanged"])
            self.assertEqual(result["mismatched"], ["clip"])

    def test_results_schema_contains_required_top_level_keys(self) -> None:
        path = Path(__file__).resolve().parents[1] / "reports/tracking/TASK-004C-validation-results.json"
        if not path.is_file():
            self.skipTest("validation results not generated in this environment")
        document = json.loads(path.read_text())
        for key in (
            "frozen_inputs",
            "annotation_integrity",
            "visibility_recall",
            "false_presence",
            "identity_accuracy",
            "identity_switches",
            "reacquisition",
            "tracker_claimed_reacquisitions",
            "occlusion_state_validity",
            "ambiguity_calibration",
            "extra_detection",
            "quality_gate",
            "raw_integrity",
            "acceptance",
            "verdict",
        ):
            self.assertIn(key, document)
        self.assertEqual(
            document["frozen_inputs"]["tracker_commit"],
            "00ec1d7de21837012fa3eb8faecbf635ac2503d6",
        )
        self.assertEqual(
            document["frozen_inputs"]["annotation_commit"],
            "012d58a989a079dbeca6e5cb49b26c384dd80c21",
        )


class TestFrozenConfigProtection(unittest.TestCase):
    """The validator must not have altered the frozen tracker configuration."""

    def test_tracker_config_matches_the_frozen_values(self) -> None:
        path = Path(__file__).resolve().parents[1] / "configs/tracking/wilor_tracker.json"
        config = json.loads(path.read_text())
        self.assertEqual(config["base_gate_radius"], 0.08)
        self.assertEqual(config["proximity_ambiguity_radius"], 0.02)
        self.assertEqual(config["duplicate_iou_threshold"], 0.5)
        self.assertEqual(config["ambiguity_margin"], 0.10)
        self.assertEqual(config["min_detection_confidence"], 0.30)
        self.assertEqual(config["schema_version"], "wilor_tracked_v1")


if __name__ == "__main__":
    unittest.main()
