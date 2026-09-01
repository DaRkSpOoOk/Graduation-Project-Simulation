import unittest
from pathlib import Path

from evaluation.annotations.task004b import (
    AnnotationError,
    ClipSpec,
    annotation_statistics,
    read_annotations,
    validate_against_manifest,
    validate_rows,
)


def _row(sample_id="sample", frame_index="0", **overrides):
    row = {
        "sample_id": sample_id,
        "frame_index": frame_index,
        "left_visibility": "VISIBLE",
        "left_x": "0.25",
        "left_y": "0.50",
        "right_visibility": "VISIBLE",
        "right_x": "0.75",
        "right_y": "0.50",
        "scene_flags": "",
        "annotator_confidence": "HIGH",
        "notes": "synthetic source-frame judgment",
    }
    row.update(overrides)
    return row


class TestTask004BAnnotations(unittest.TestCase):
    def test_valid_visibility_enums_and_coordinates(self):
        spec = (ClipSpec("sample", 1, "control", "test"),)
        rows = validate_rows([_row()], clip_specs=spec)
        self.assertEqual(rows[0].left_visibility, "VISIBLE")
        self.assertEqual(rows[0].right_x, 0.75)

    def test_invalid_visibility_is_rejected(self):
        with self.assertRaises(AnnotationError):
            validate_rows([_row(left_visibility="MAYBE")], clip_specs=(ClipSpec("sample", 1, "control", "test"),))

    def test_duplicate_sample_frame_is_rejected(self):
        spec = (ClipSpec("sample", 1, "control", "test"),)
        with self.assertRaises(AnnotationError):
            validate_rows([_row(), _row()], clip_specs=spec)

    def test_frame_range_and_complete_coverage_are_rejected(self):
        spec = (ClipSpec("sample", 2, "control", "test"),)
        with self.assertRaises(AnnotationError):
            validate_rows([_row(frame_index="2")], clip_specs=spec)
        with self.assertRaises(AnnotationError):
            validate_rows([_row()], clip_specs=spec)

    def test_coordinate_range_and_pairing_are_rejected(self):
        spec = (ClipSpec("sample", 1, "control", "test"),)
        with self.assertRaises(AnnotationError):
            validate_rows([_row(left_x="1.1")], clip_specs=spec)
        with self.assertRaises(AnnotationError):
            validate_rows([_row(right_x="")], clip_specs=spec)

    def test_all_selected_videos_must_be_represented(self):
        specs = (
            ClipSpec("one", 1, "challenge", "test"),
            ClipSpec("two", 1, "control", "test"),
        )
        with self.assertRaises(AnnotationError):
            validate_rows([_row(sample_id="one")], clip_specs=specs)

    def test_statistics_count_frames_not_duplicate_rows(self):
        spec = (ClipSpec("sample", 2, "challenge", "test"),)
        rows = validate_rows(
            [
                _row(frame_index="0", right_visibility="AMBIGUOUS", scene_flags="HAND_CROSSING;IDENTITY_AMBIGUOUS"),
                _row(frame_index="1", left_visibility="PARTIALLY_OCCLUDED", right_visibility="FULLY_OCCLUDED", scene_flags="MOTION_BLUR"),
            ],
            clip_specs=spec,
        )
        stats = annotation_statistics(rows)
        self.assertEqual(stats["frames"], 2)
        self.assertEqual(stats["ambiguous_identity_frames"], 1)
        self.assertEqual(stats["flags_frame_counts"]["HAND_CROSSING"], 1)
        self.assertEqual(stats["partially_occluded_hand_labels"], 1)
        self.assertEqual(stats["fully_occluded_hand_labels"], 1)

    def test_committed_annotation_file_parses(self):
        path = Path(__file__).resolve().parents[1] / "evaluation/annotations/task004_hand_identity_visibility.csv"
        rows = read_annotations(path)
        self.assertEqual(len(rows), 399)
        self.assertEqual(len({row.sample_id for row in rows}), 8)

    def test_committed_annotations_reference_shared_manifest(self):
        root = Path(__file__).resolve().parents[1]
        rows = validate_against_manifest(
            root / "evaluation/annotations/task004_hand_identity_visibility.csv",
            root / "datasets/manifests/karsl_milestone1_pilot.csv",
        )
        self.assertEqual(len(rows), 399)


if __name__ == "__main__":
    unittest.main()
