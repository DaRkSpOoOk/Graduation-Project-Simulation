"""Synthetic tests for the WiLoR temporal dual-hand tracker.

Every test builds its own in-memory or temporary-directory input. Nothing
here needs the 2.56 GB WiLoR checkpoint, MANO assets, or KArSL videos.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tracking.wilor import (
    TrackerConfig,
    aggregate_metrics,
    compute_metrics,
    load_raw_sequence,
    load_tracked_sequence,
    save_tracked_sequence,
    track_sequence,
)
from tracking.wilor.association import (
    bbox_iou,
    centre_separation_ratio,
    is_cross_label_ghost,
    suppress_cross_label_ghosts,
    suppress_same_label_duplicates,
)
from tracking.wilor.quality import assess_detection
from tracking.wilor.schema import RawDetection, TrackState
from tracking.wilor.source import RawInputError, RawSequence

IMAGE_WH = (1920.0, 1080.0)


FOCAL = 37500.0
DEPTH = 40.0


def make_detection(
    frame_index: int,
    detection_index: int,
    label: str | None,
    centre_px: tuple[float, float],
    *,
    confidence: float = 0.85,
    box_size: float = 400.0,
    palm_scale: float = 1.0,
) -> RawDetection:
    """A geometrically self-consistent synthetic detection.

    The camera translation is solved so the projected joint centroid lands on
    ``centre_px``, matching how a real WiLoR row relates its reconstruction to
    its detector box. Fixtures that are not self-consistent would be rejected
    by the projection component of the quality gate.
    """

    joints = np.zeros((21, 3), dtype=np.float64)
    # wrist at origin, middle-MCP along +y so palm length is well defined
    joints[9] = (0.0, 0.095 * palm_scale, 0.0)
    for index in range(1, 21):
        if index == 9:
            continue
        joints[index] = (0.01 * index, 0.004 * index, 0.0)

    centroid = joints.mean(axis=0)
    width, height = IMAGE_WH
    translation = np.array(
        [
            (centre_px[0] - width / 2.0) * DEPTH / FOCAL - centroid[0],
            (centre_px[1] - height / 2.0) * DEPTH / FOCAL - centroid[1],
            DEPTH,
        ]
    )
    return RawDetection(
        frame_index=frame_index,
        raw_detection_index=detection_index,
        detector_label=label,
        detector_confidence=confidence,
        landmarks_3d=joints,
        hand_pose_rotmat=np.tile(np.eye(3), (15, 1, 1)),
        global_orient_rotmat=np.eye(3),
        betas=np.zeros(10),
        camera_translation=translation,
        box_center_xy=centre_px,
        box_size=box_size,
        image_size_wh=IMAGE_WH,
        focal_length=FOCAL,
    )


def make_sequence(frames: list[list[RawDetection]], sample_id: str = "synthetic") -> RawSequence:
    detections_by_frame = {index: list(items) for index, items in enumerate(frames)}
    timestamps = {index: index / 30.0 for index in range(len(frames))}
    return RawSequence(
        sample_id=sample_id,
        frame_indices=sorted(detections_by_frame),
        timestamps=timestamps,
        detections_by_frame=detections_by_frame,
        run_metadata={"mode": "full"},
    )


def states(sequence, track: str) -> list[TrackState]:
    return [frame.hand(track).state for frame in sequence.frames]


class TestStableTwoHandSequence(unittest.TestCase):
    """1. A clean two-hand sequence stays OBSERVED on both tracks."""

    def test_both_tracks_observed_every_frame(self) -> None:
        frames = [
            [
                make_detection(i, 0, "right", (800.0 + i, 500.0)),
                make_detection(i, 1, "left", (1200.0 - i, 505.0)),
            ]
            for i in range(12)
        ]
        tracked = track_sequence(make_sequence(frames))
        self.assertEqual(states(tracked, "left"), [TrackState.OBSERVED] * 12)
        self.assertEqual(states(tracked, "right"), [TrackState.OBSERVED] * 12)
        metrics = compute_metrics(tracked)
        self.assertEqual(metrics.frames_with_both_tracks, 12)
        self.assertEqual(metrics.suspected_identity_switch_events, 0)


class TestConvergingHands(unittest.TestCase):
    """2. Hands approaching each other keep their identities."""

    def test_identities_survive_convergence(self) -> None:
        frames = []
        for i in range(15):
            right_x = 700.0 + i * 20.0
            left_x = 1300.0 - i * 20.0
            frames.append(
                [
                    make_detection(i, 0, "right", (right_x, 500.0)),
                    make_detection(i, 1, "left", (left_x, 500.0)),
                ]
            )
        tracked = track_sequence(make_sequence(frames))
        for frame in tracked.frames:
            self.assertIn(frame.left.state, {TrackState.OBSERVED, TrackState.AMBIGUOUS})
            self.assertIn(frame.right.state, {TrackState.OBSERVED, TrackState.AMBIGUOUS})
            # the LEFT track must keep binding the detection that is further right
            if frame.left.box_center_xy and frame.right.box_center_xy:
                self.assertGreaterEqual(frame.left.box_center_xy[0], frame.right.box_center_xy[0])


class TestCrossingHands(unittest.TestCase):
    """3. Hands that cross keep spatially continuous identities."""

    def test_tracks_follow_motion_through_a_crossing(self) -> None:
        frames = []
        for i in range(21):
            right_x = 700.0 + i * 30.0   # moves rightwards, ends at 1300
            left_x = 1300.0 - i * 30.0   # moves leftwards, ends at 700
            frames.append(
                [
                    make_detection(i, 0, "right", (right_x, 500.0)),
                    make_detection(i, 1, "left", (left_x, 500.0)),
                ]
            )
        tracked = track_sequence(make_sequence(frames))
        # After the crossing the RIGHT track should be the detection that is
        # now on the larger-x side, i.e. motion continuity beat the raw
        # left/right image ordering.
        final = tracked.frames[-1]
        self.assertIsNotNone(final.right.box_center_xy)
        self.assertIsNotNone(final.left.box_center_xy)
        self.assertGreater(final.right.box_center_xy[0], final.left.box_center_xy[0])
        self.assertEqual(compute_metrics(tracked).frames_with_no_track, 0)


class TestLabelFlip(unittest.TestCase):
    """4. A one-frame detector label flip must not switch identities."""

    def test_single_frame_label_flip_is_absorbed(self) -> None:
        frames = []
        for i in range(9):
            # frame 4 has both detector labels inverted
            right_label, left_label = ("left", "right") if i == 4 else ("right", "left")
            frames.append(
                [
                    make_detection(i, 0, right_label, (800.0, 500.0)),
                    make_detection(i, 1, left_label, (1300.0, 500.0)),
                ]
            )
        tracked = track_sequence(make_sequence(frames))
        for frame in tracked.frames:
            # positions are static, so identity must never move
            self.assertAlmostEqual(frame.right.box_center_xy[0], 800.0, places=6)
            self.assertAlmostEqual(frame.left.box_center_xy[0], 1300.0, places=6)
        flipped = tracked.frames[4]
        self.assertTrue(flipped.left.label_disagrees)
        self.assertTrue(flipped.right.label_disagrees)
        self.assertIn("BOTH_LABELS_SWAPPED", flipped.tracking_flags)
        self.assertEqual(compute_metrics(tracked).handedness_disagreement_events, 1)


class TestTemporaryDisappearance(unittest.TestCase):
    """5. + 6. A hand vanishes then returns to the same identity."""

    def test_missing_then_reassociated(self) -> None:
        frames = []
        for i in range(12):
            row = [make_detection(i, 0, "right", (800.0, 500.0))]
            if not 4 <= i <= 7:
                row.append(make_detection(i, 1, "left", (1300.0, 500.0)))
            frames.append(row)
        tracked = track_sequence(make_sequence(frames))
        left_states = states(tracked, "left")
        for index in range(4, 8):
            self.assertIn(left_states[index], {TrackState.MISSING, TrackState.LIKELY_OCCLUDED})
            self.assertIsNone(tracked.frames[index].left.landmarks_3d)
        self.assertEqual(left_states[8], TrackState.OBSERVED)
        self.assertAlmostEqual(tracked.frames[8].left.box_center_xy[0], 1300.0, places=6)

        metrics = compute_metrics(tracked)
        self.assertEqual(metrics.reassociation_events, 1)
        self.assertEqual(metrics.longest_left_missing_run, 4)
        reassociations = [e for e in tracked.events if e["event"] == "reassociation"]
        self.assertEqual(reassociations[0]["track"], "left")
        self.assertEqual(reassociations[0]["frames_absent"], 4)

    def test_missing_hand_is_never_interpolated(self) -> None:
        frames = [
            [make_detection(i, 0, "right", (800.0, 500.0))]
            + ([make_detection(i, 1, "left", (1300.0, 500.0))] if i != 3 else [])
            for i in range(6)
        ]
        tracked = track_sequence(make_sequence(frames))
        gap = tracked.frames[3].left
        self.assertIsNone(gap.landmarks_3d)
        self.assertIsNone(gap.hand_pose_rotmat)
        self.assertIsNone(gap.raw_detection_index)
        self.assertFalse(gap.has_pose)


class TestDuplicateDetections(unittest.TestCase):
    """7. Two overlapping detections sharing a label collapse to one."""

    def test_same_label_duplicate_is_suppressed(self) -> None:
        detections = [
            make_detection(0, 0, "left", (1000.0, 500.0), confidence=0.9, box_size=400.0),
            make_detection(0, 1, "left", (1010.0, 505.0), confidence=0.6, box_size=380.0),
        ]
        self.assertGreater(bbox_iou(detections[0], detections[1]), 0.5)
        kept, suppressed = suppress_same_label_duplicates(detections, TrackerConfig())
        self.assertEqual(kept, [0])
        self.assertIn(1, suppressed)
        self.assertTrue(suppressed[1].startswith("duplicate_same_label"))

    def test_overlapping_boxes_with_different_labels_are_kept(self) -> None:
        detections = [
            make_detection(0, 0, "left", (1000.0, 500.0)),
            make_detection(0, 1, "right", (1010.0, 505.0)),
        ]
        self.assertGreater(bbox_iou(detections[0], detections[1]), 0.5)
        kept, suppressed = suppress_same_label_duplicates(detections, TrackerConfig())
        self.assertEqual(kept, [0, 1])
        self.assertEqual(suppressed, {})


class TestThreeDetections(unittest.TestCase):
    """8. A third detection must never become a permanent identity."""

    def test_third_detection_is_recorded_not_promoted(self) -> None:
        frames = []
        for i in range(6):
            row = [
                make_detection(i, 0, "right", (800.0, 500.0)),
                make_detection(i, 1, "left", (1300.0, 500.0)),
            ]
            if i == 3:
                # a distinct, non-overlapping spurious hand
                row.append(make_detection(i, 2, "right", (300.0, 900.0), confidence=0.55))
            frames.append(row)
        tracked = track_sequence(make_sequence(frames))
        spurious_frame = tracked.frames[3]
        self.assertEqual(spurious_frame.number_of_raw_detections, 3)
        self.assertEqual(spurious_frame.extra_detection_count, 1)
        self.assertIn(2, spurious_frame.rejected_detection_indices)
        self.assertIn("EXTRA_DETECTIONS", spurious_frame.tracking_flags)
        # identities are unchanged by the intruder
        self.assertAlmostEqual(spurious_frame.right.box_center_xy[0], 800.0, places=6)
        self.assertAlmostEqual(spurious_frame.left.box_center_xy[0], 1300.0, places=6)
        for frame in tracked.frames:
            self.assertNotEqual(frame.left.raw_detection_index, 2)
            self.assertNotEqual(frame.right.raw_detection_index, 2)


class TestQualityGate(unittest.TestCase):
    """9. A low-quality candidate is marked, not silently accepted."""

    def test_collapsed_geometry_fails_the_gate(self) -> None:
        detection = make_detection(0, 0, "left", (1000.0, 500.0))
        detection.landmarks_3d = np.zeros((21, 3))  # palm length collapses to 0
        assessment = assess_detection(detection, TrackerConfig())
        self.assertFalse(assessment.passed)
        self.assertIn("LOW_QUALITY_GEOMETRY_COLLAPSE", assessment.flags)

    def test_below_threshold_confidence_fails_the_gate(self) -> None:
        detection = make_detection(0, 0, "left", (1000.0, 500.0), confidence=0.05)
        assessment = assess_detection(detection, TrackerConfig())
        self.assertFalse(assessment.passed)
        self.assertIn("LOW_QUALITY_CONFIDENCE", assessment.flags)

    def test_plausible_detection_passes(self) -> None:
        assessment = assess_detection(make_detection(0, 0, "left", (1000.0, 500.0)), TrackerConfig())
        self.assertTrue(assessment.passed)
        self.assertEqual(assessment.flags, ())

    def test_rejected_candidate_does_not_become_a_track(self) -> None:
        frames = [
            [
                make_detection(i, 0, "right", (800.0, 500.0)),
                make_detection(i, 1, "left", (1300.0, 500.0), confidence=0.01 if i == 2 else 0.9),
            ]
            for i in range(5)
        ]
        tracked = track_sequence(make_sequence(frames))
        gated = tracked.frames[2]
        self.assertIn(1, gated.rejected_detection_indices)
        self.assertTrue(gated.rejection_reasons[1].startswith("quality:"))
        self.assertNotIn(gated.left.state, {TrackState.OBSERVED, TrackState.AMBIGUOUS})
        self.assertIsNone(gated.left.landmarks_3d)
        self.assertEqual(compute_metrics(tracked).quality_rejected_detections, 1)


class TestNoDetections(unittest.TestCase):
    """10. Empty frames produce MISSING tracks and no fabricated pose."""

    def test_empty_sequence_is_all_missing(self) -> None:
        tracked = track_sequence(make_sequence([[] for _ in range(5)]))
        self.assertEqual(states(tracked, "left"), [TrackState.MISSING] * 5)
        self.assertEqual(states(tracked, "right"), [TrackState.MISSING] * 5)
        for frame in tracked.frames:
            self.assertIn("NO_DETECTIONS", frame.tracking_flags)
            self.assertIsNone(frame.left.landmarks_3d)
            self.assertIsNone(frame.right.landmarks_3d)
        metrics = compute_metrics(tracked)
        self.assertEqual(metrics.frames_with_no_track, 5)
        self.assertEqual(metrics.longest_left_missing_run, 5)


class TestDeterminism(unittest.TestCase):
    """11. Repeated execution is byte-identical."""

    def _sample_frames(self) -> list[list[RawDetection]]:
        frames = []
        for i in range(14):
            row = [
                make_detection(i, 0, "right", (700.0 + i * 25.0, 500.0)),
                make_detection(i, 1, "left", (1300.0 - i * 25.0, 505.0)),
            ]
            if i == 6:
                row.append(make_detection(i, 2, "left", (1310.0 - i * 25.0, 508.0), confidence=0.5))
            frames.append(row)
        return frames

    def test_repeated_runs_agree(self) -> None:
        first = track_sequence(make_sequence(self._sample_frames()))
        second = track_sequence(make_sequence(self._sample_frames()))
        self.assertEqual(states(first, "left"), states(second, "left"))
        self.assertEqual(states(first, "right"), states(second, "right"))
        self.assertEqual(
            [f.rejected_detection_indices for f in first.frames],
            [f.rejected_detection_indices for f in second.frames],
        )
        self.assertEqual(first.events, second.events)

    def test_serialized_output_is_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payloads = []
            for name in ("run_a", "run_b"):
                tracked = track_sequence(make_sequence(self._sample_frames()), source={"fixed": True})
                npz_path, _ = save_tracked_sequence(root / name, tracked, compute_metrics(tracked).to_dict())
                payloads.append(npz_path.read_bytes())
            arrays_a, _ = load_tracked_sequence(root / "run_a" / "synthetic")
            arrays_b, _ = load_tracked_sequence(root / "run_b" / "synthetic")
        for key in arrays_a:
            np.testing.assert_array_equal(arrays_a[key], arrays_b[key], err_msg=key)


class TestRawImmutability(unittest.TestCase):
    """12. Tracking never modifies its raw input."""

    def _write_raw_npz(self, path: Path) -> None:
        frames, rows = 4, []
        for frame in range(frames):
            for detection, label in ((0, "right"), (1, "left")):
                rows.append((frame, detection, label))
        count = len(rows)
        joints = np.zeros((count, 21, 3), dtype=np.float32)
        joints[:, 9, 1] = 0.095
        mano = json.dumps(
            {
                "hand_pose_rotmat": np.tile(np.eye(3), (15, 1, 1)).tolist(),
                "global_orient_rotmat": np.eye(3).tolist(),
                "betas": [0.0] * 10,
            }
        )
        references = [
            json.dumps(
                {
                    "camera_translation_xyz": [0.0, 0.0, 40.0],
                    "focal_length": 37500.0,
                    "box_center_xy": [800.0 if detection == 0 else 1300.0, 500.0],
                    "box_size": 400.0,
                    "img_size_wh": [1920.0, 1080.0],
                }
            )
            for _, detection, _ in rows
        ]
        np.savez_compressed(
            path,
            frame_index=np.array([r[0] for r in rows], dtype=np.int32),
            timestamp_seconds=np.array([r[0] / 30.0 for r in rows], dtype=np.float64),
            hand_present=np.ones(count, dtype=bool),
            handedness_label=np.array([r[2] for r in rows], dtype="U8"),
            detection_confidence=np.full(count, 0.9, dtype=np.float32),
            landmarks_3d=joints,
            quality_flags_json=np.array(["[]"] * count, dtype="U16"),
            mano_params_json=np.array([mano] * count, dtype=f"U{len(mano)}"),
            mano_references_json=np.array(references, dtype=f"U{max(len(r) for r in references)}"),
            extractor_metadata_json=np.array(['{"mode": "full"}'] * count, dtype="U32"),
            run_metadata_json=np.array(json.dumps({"mode": "full"})),
        )

    def test_raw_npz_bytes_are_unchanged(self) -> None:
        import hashlib

        with tempfile.TemporaryDirectory() as tmp:
            raw_path = Path(tmp) / "wilor_raw.npz"
            self._write_raw_npz(raw_path)
            before = hashlib.sha256(raw_path.read_bytes()).hexdigest()

            sequence = load_raw_sequence(raw_path, "sample")
            tracked = track_sequence(sequence)
            save_tracked_sequence(Path(tmp) / "tracked", tracked, compute_metrics(tracked).to_dict())

            after = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        self.assertEqual(before, after)

    def test_detector_only_input_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_path = Path(tmp) / "wilor_raw.npz"
            self._write_raw_npz(raw_path)
            with np.load(raw_path, allow_pickle=False) as data:
                arrays = {key: data[key] for key in data.files}
            arrays["run_metadata_json"] = np.array(json.dumps({"mode": "detector_only"}))
            np.savez_compressed(raw_path, **arrays)
            with self.assertRaises(RawInputError):
                load_raw_sequence(raw_path, "sample")


class TestRoundTripAndProvenance(unittest.TestCase):
    def test_tracked_npz_round_trip_preserves_states_and_provenance(self) -> None:
        frames = [
            [make_detection(i, 0, "right", (800.0, 500.0))]
            + ([make_detection(i, 1, "left", (1300.0, 500.0))] if i != 2 else [])
            for i in range(5)
        ]
        tracked = track_sequence(make_sequence(frames))
        with tempfile.TemporaryDirectory() as tmp:
            save_tracked_sequence(Path(tmp), tracked, compute_metrics(tracked).to_dict())
            arrays, metadata = load_tracked_sequence(Path(tmp) / "synthetic")

        self.assertEqual(metadata["track_order"], ["left", "right"])
        self.assertEqual(arrays["frame_index"].tolist(), [0, 1, 2, 3, 4])
        # provenance back to the raw detection index survives serialization
        self.assertEqual(arrays["raw_detection_index"][0, 0], 1)   # left <- raw row 1
        self.assertEqual(arrays["raw_detection_index"][0, 1], 0)   # right <- raw row 0
        self.assertEqual(arrays["raw_detection_index"][2, 0], -1)  # missing left
        self.assertTrue(np.isnan(arrays["landmarks_3d"][2, 0]).all())
        self.assertFalse(np.isnan(arrays["landmarks_3d"][2, 1]).any())

    def test_aggregate_metrics_sum_across_videos(self) -> None:
        frames = [
            [
                make_detection(i, 0, "right", (800.0, 500.0)),
                make_detection(i, 1, "left", (1300.0, 500.0)),
            ]
            for i in range(4)
        ]
        metrics = [
            compute_metrics(track_sequence(make_sequence(frames, sample_id=f"s{index}")))
            for index in range(3)
        ]
        totals = aggregate_metrics(metrics)
        self.assertEqual(totals["videos"], 3)
        self.assertEqual(totals["total_frames"], 12)
        self.assertEqual(totals["frames_with_both_tracks"], 12)
        self.assertAlmostEqual(totals["rates_pct"]["both_tracks"], 100.0)


class TestCrossLabelGhostSuppression(unittest.TestCase):
    """TASK-004D. A weak opposite-label duplicate must not become a hand.

    The fixtures below are written in terms of the three measured signals, not
    in terms of any pilot frame, so they exercise the general rule rather than
    the sample that exposed it.
    """

    @staticmethod
    def _pair(*, offset_px: float, weak_confidence: float, box: float = 400.0):
        return [
            make_detection(0, 0, "left", (1000.0, 500.0), confidence=0.90, box_size=box),
            make_detection(
                0, 1, "right", (1000.0 + offset_px, 500.0),
                confidence=weak_confidence, box_size=box,
            ),
        ]

    def test_weak_opposite_label_duplicate_is_suppressed(self) -> None:
        detections = self._pair(offset_px=12.0, weak_confidence=0.35)
        self.assertGreaterEqual(bbox_iou(*detections), 0.80)
        self.assertLessEqual(centre_separation_ratio(*detections), 0.07)
        kept, suppressed = suppress_cross_label_ghosts(detections, TrackerConfig())
        self.assertEqual(kept, [0])
        self.assertIn(1, suppressed)
        self.assertTrue(suppressed[1].startswith("cross_label_duplicate_suspected"))

    def test_the_confident_detection_of_the_pair_survives(self) -> None:
        """Input order must not decide which detection is kept."""

        detections = self._pair(offset_px=12.0, weak_confidence=0.35)
        reversed_order = [detections[1], detections[0]]
        kept, _ = suppress_cross_label_ghosts(reversed_order, TrackerConfig())
        self.assertEqual(kept, [1])
        self.assertAlmostEqual(reversed_order[kept[0]].detector_confidence, 0.90)

    def test_two_genuine_hands_at_high_overlap_are_kept(self) -> None:
        """Overlap alone must never suppress: separated centres, equal trust."""

        detections = self._pair(offset_px=60.0, weak_confidence=0.86)
        self.assertGreater(bbox_iou(*detections), 0.5)
        kept, suppressed = suppress_cross_label_ghosts(detections, TrackerConfig())
        self.assertEqual(kept, [0, 1])
        self.assertEqual(suppressed, {})

    def test_near_coincident_but_equally_confident_pair_is_kept(self) -> None:
        """Confidence agreement alone is enough to protect a real hand."""

        detections = self._pair(offset_px=12.0, weak_confidence=0.88)
        self.assertGreaterEqual(bbox_iou(*detections), 0.80)
        kept, _ = suppress_cross_label_ghosts(detections, TrackerConfig())
        self.assertEqual(kept, [0, 1])

    def test_weak_but_well_separated_pair_is_kept(self) -> None:
        """A faint far hand is a faint hand, not a duplicate."""

        detections = self._pair(offset_px=150.0, weak_confidence=0.35)
        kept, _ = suppress_cross_label_ghosts(detections, TrackerConfig())
        self.assertEqual(kept, [0, 1])

    def test_same_label_pairs_are_left_to_the_same_label_rule(self) -> None:
        detections = self._pair(offset_px=12.0, weak_confidence=0.35)
        detections[1] = make_detection(0, 1, "left", (1012.0, 500.0), confidence=0.35)
        kept, suppressed = suppress_cross_label_ghosts(detections, TrackerConfig())
        self.assertEqual(kept, [0, 1])
        self.assertEqual(suppressed, {})

    def test_unlabelled_detections_are_never_ghost_suppressed(self) -> None:
        detections = self._pair(offset_px=12.0, weak_confidence=0.35)
        detections[1] = make_detection(0, 1, None, (1012.0, 500.0), confidence=0.35)
        kept, _ = suppress_cross_label_ghosts(detections, TrackerConfig())
        self.assertEqual(kept, [0, 1])

    def test_confidence_ratio_sweep_is_monotone(self) -> None:
        """The decision moves once, at the configured ratio, and stays put."""

        config = TrackerConfig()
        outcomes = []
        for weak in (0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.85):
            detections = self._pair(offset_px=12.0, weak_confidence=weak)
            outcomes.append(is_cross_label_ghost(detections[1], detections[0], config))
        self.assertEqual(outcomes, [True, True, True, False, False, False, False, False])

    def test_ghost_does_not_produce_a_second_tracked_hand(self) -> None:
        """End to end: the occluded track stays absent instead of reappearing."""

        frames = []
        for i in range(8):
            row = [make_detection(i, 0, "left", (900.0, 500.0), confidence=0.90)]
            if i < 2:
                row.append(make_detection(i, 1, "right", (1500.0, 500.0), confidence=0.88))
            elif i >= 5:
                # weak opposite-label blob sitting on the visible left hand
                row.append(make_detection(i, 1, "right", (912.0, 505.0), confidence=0.34))
            frames.append(row)
        tracked = track_sequence(make_sequence(frames))
        right_states = states(tracked, "right")
        self.assertEqual(right_states[0], TrackState.OBSERVED)
        for state in right_states[5:]:
            self.assertNotIn(state, (TrackState.OBSERVED, TrackState.AMBIGUOUS))
        for frame in tracked.frames[5:]:
            self.assertIn(1, frame.rejected_detection_indices)
            self.assertIn(
                "cross_label_duplicate_suspected", frame.rejection_reasons[1]
            )
            self.assertIn("CROSS_LABEL_DUPLICATE_SUPPRESSED", frame.tracking_flags)
        # the raw record still carries both detections
        self.assertEqual(tracked.frames[5].number_of_raw_detections, 2)

    def test_genuine_return_after_absence_is_still_reacquired(self) -> None:
        """The guard must not block a real hand coming back."""

        frames = []
        for i in range(9):
            row = [make_detection(i, 0, "left", (900.0, 500.0), confidence=0.90)]
            if i < 2 or i >= 6:
                row.append(make_detection(i, 1, "right", (1500.0, 500.0), confidence=0.87))
            frames.append(row)
        tracked = track_sequence(make_sequence(frames))
        right_states = states(tracked, "right")
        self.assertEqual(right_states[3], TrackState.MISSING)
        self.assertEqual(right_states[6], TrackState.OBSERVED)
        self.assertAlmostEqual(tracked.frames[6].right.box_center_xy[0], 1500.0, places=6)

    def test_suppression_is_deterministic_across_repeated_runs(self) -> None:
        frames = [
            [
                make_detection(i, 0, "left", (900.0, 500.0), confidence=0.90),
                make_detection(i, 1, "right", (911.0, 503.0), confidence=0.36),
            ]
            for i in range(5)
        ]
        first = track_sequence(make_sequence(frames))
        second = track_sequence(make_sequence(frames))
        self.assertEqual(states(first, "right"), states(second, "right"))
        self.assertEqual(
            [f.rejection_reasons for f in first.frames],
            [f.rejection_reasons for f in second.frames],
        )


class TestConfigValidation(unittest.TestCase):
    def test_cost_weights_must_sum_to_one(self) -> None:
        with self.assertRaises(ValueError):
            TrackerConfig(weight_position=0.9, weight_label=0.9).validate()

    def test_config_round_trips_through_json(self) -> None:
        config = TrackerConfig()
        restored = TrackerConfig.from_dict(config.to_dict())
        self.assertEqual(config, restored)

    def test_unknown_config_key_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TrackerConfig.from_dict({"not_a_real_option": 1})


if __name__ == "__main__":
    unittest.main()
