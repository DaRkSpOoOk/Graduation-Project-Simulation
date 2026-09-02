"""Tests for TASK-008C dataset-level QA and the pose-alignment QA fix.

All fixtures are synthetic and built in temporary directories. Nothing here
reads the production run, the dataset, or requires a GPU.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.dataset.final_qa import (
    POSE_BEARING_CODES,
    TRACKING_STATE_NAMES,
    aggregate_dataset,
    coverage_accounting,
)
from evaluation.dataset.orchestrator import RunPaths

FINGERS, CHAIN, SPREAD, TRACKS = 5, 3, 4, 2


def _write_glove(root: Path, sample_id: str, frames: int, *,
                 state_code: np.ndarray | None = None,
                 bend_invalid: int = 0, spread_invalid: int = 0,
                 imu_invalid: int = 0) -> None:
    paths = RunPaths(root)
    glove_dir = paths.virtual_glove / sample_id
    glove_dir.mkdir(parents=True, exist_ok=True)
    bend_valid = np.ones((frames, TRACKS, FINGERS, CHAIN), dtype=bool)
    spread_valid = np.ones((frames, TRACKS, SPREAD), dtype=bool)
    imu_valid = np.ones((frames, TRACKS), dtype=bool)
    if bend_invalid:
        bend_valid.reshape(-1)[:bend_invalid] = False
    if spread_invalid:
        spread_valid.reshape(-1)[:spread_invalid] = False
    if imu_invalid:
        imu_valid.reshape(-1)[:imu_invalid] = False
    if state_code is None:
        state_code = np.ones((frames, TRACKS), dtype=np.int32)
    np.savez_compressed(
        glove_dir / "virtual_glove.npz",
        frame_index=np.arange(frames, dtype=np.int32),
        bend_valid=bend_valid, spread_valid=spread_valid,
        palm_imu_valid=imu_valid, tracking_state_code=state_code.astype(np.int32),
    )
    tracked_dir = paths.tracking / sample_id
    tracked_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(tracked_dir / "wilor_tracked.npz",
                        frame_index=np.arange(frames, dtype=np.int32))


def _row(sample_id: str, frames: int, *, signer="01", sign_id="0032",
         partition="train", label="ا") -> dict[str, str]:
    return {"sample_id": sample_id, "sign_id": sign_id, "label_ar": label,
            "signer_id": signer, "official_partition": partition,
            "frame_count": str(frames), "source_relative_path": f"{signer}/x/{sample_id}.mp4"}


class TestPoseAlignmentQaFix(unittest.TestCase):
    """The raw pose NPZ has one row per detected hand, not per frame.

    Comparing its row vector directly against the per-frame stages rejected
    every two-hand sample. This is the same defect class TASK-008B fixed in the
    resume validator, and it would have failed the entire 4,222-sample dataset.
    """

    def test_distinct_pose_frames_match_per_frame_stages(self) -> None:
        frames = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
        times = np.array([0.0, 0.0, 0.1, 0.1, 0.2, 0.2])
        distinct = np.unique(frames)
        np.testing.assert_array_equal(distinct, np.array([0, 1, 2]))
        per_frame = np.array([times[frames == v][0] for v in distinct])
        np.testing.assert_allclose(per_frame, [0.0, 0.1, 0.2])

    def test_row_count_comparison_would_have_failed(self) -> None:
        """Documents why the direct comparison is wrong."""

        pose_rows = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
        kinematics_frames = np.array([0, 1, 2], dtype=np.int32)
        self.assertNotEqual(pose_rows.shape, kinematics_frames.shape)
        self.assertEqual(np.unique(pose_rows).shape, kinematics_frames.shape)

    def test_single_hand_video_is_unaffected(self) -> None:
        pose_rows = np.array([0, 1, 2], dtype=np.int32)
        np.testing.assert_array_equal(np.unique(pose_rows), pose_rows)

    def test_inconsistent_timestamps_within_a_frame_are_detectable(self) -> None:
        frames = np.array([0, 0], dtype=np.int32)
        times = np.array([0.0, 0.5])
        rows = times[frames == 0]
        self.assertFalse(bool(np.all(rows == rows[0])))

    def test_qa_module_uses_distinct_pose_frames(self) -> None:
        source = Path(__file__).resolve().parents[1] / "evaluation/dataset/qa.py"
        text = source.read_text()
        self.assertIn("np.unique(pose_frames)", text)
        self.assertNotIn(
            '_assert_equal("frame_index", np.asarray(pose["frame_index"]), frame_index',
            text,
            "the direct pose row comparison must not come back",
        )


class TestDatasetAggregation(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.rows = [
            _row("a", 10, signer="01", sign_id="0032", partition="train"),
            _row("b", 20, signer="02", sign_id="0033", partition="train"),
            _row("c", 30, signer="03", sign_id="0032", partition="test"),
        ]
        for row in self.rows:
            _write_glove(self.root, row["sample_id"], int(row["frame_count"]))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_totals_and_fractions(self) -> None:
        result = aggregate_dataset(self.rows, self.root)
        totals = result["sensor_validity_totals"]
        self.assertEqual(result["samples_aggregated"], 3)
        self.assertEqual(totals["bend_total"], 60 * TRACKS * FINGERS * CHAIN)
        self.assertEqual(totals["bend_valid_fraction"], 1.0)
        self.assertEqual(totals["spread_valid_fraction"], 1.0)
        self.assertEqual(totals["imu_valid_fraction"], 1.0)

    def test_invalid_channels_lower_only_their_own_fraction(self) -> None:
        _write_glove(self.root, "a", 10, spread_invalid=5)
        result = aggregate_dataset(self.rows, self.root)
        totals = result["sensor_validity_totals"]
        self.assertEqual(totals["bend_valid_fraction"], 1.0)
        self.assertLess(totals["spread_valid_fraction"], 1.0)
        self.assertEqual(totals["imu_valid_fraction"], 1.0)

    def test_breakdowns_cover_signer_class_and_partition(self) -> None:
        result = aggregate_dataset(self.rows, self.root)
        self.assertEqual(sorted(result["by_signer"]), ["01", "02", "03"])
        self.assertEqual(sorted(result["by_class"]), ["0032", "0033"])
        self.assertEqual(sorted(result["by_partition"]), ["test", "train"])
        self.assertEqual(result["by_class"]["0032"]["videos"], 2)
        self.assertEqual(result["by_class"]["0032"]["frames"], 40)

    def test_sequence_lengths_are_reported_not_altered(self) -> None:
        result = aggregate_dataset(self.rows, self.root)
        lengths = result["sequence_lengths"]
        self.assertEqual(lengths["min"], 10)
        self.assertEqual(lengths["max"], 30)
        self.assertEqual(lengths["sum"], 60)
        integrity = result["temporal_integrity"]
        self.assertTrue(integrity["output_equals_manifest_frames"])
        self.assertFalse(integrity["padding_performed"])
        self.assertFalse(integrity["resampling_performed"])
        self.assertFalse(integrity["interpolation_performed"])

    def test_a_length_change_is_detected(self) -> None:
        _write_glove(self.root, "a", 11)  # manifest says 10
        result = aggregate_dataset(self.rows, self.root)
        integrity = result["temporal_integrity"]
        self.assertFalse(integrity["output_equals_manifest_frames"])
        self.assertEqual(integrity["length_mismatch_count"], 1)
        self.assertEqual(integrity["length_mismatches"][0]["sample_id"], "a")

    def test_missing_output_is_reported_not_skipped_silently(self) -> None:
        rows = self.rows + [_row("d", 12)]
        result = aggregate_dataset(rows, self.root)
        self.assertEqual(result["samples_missing_output"], ["d"])
        self.assertEqual(result["samples_aggregated"], 3)

    def test_tracking_states_are_named_and_counted(self) -> None:
        state = np.full((10, TRACKS), 1, dtype=np.int32)
        state[0, 1] = 0   # MISSING
        state[1, 1] = 4   # LIKELY_OCCLUDED
        state[2, 0] = 2   # AMBIGUOUS
        _write_glove(self.root, "a", 10, state_code=state)
        result = aggregate_dataset(self.rows, self.root)
        combined = result["tracking_states"]["combined"]
        self.assertEqual(combined["MISSING"], 1)
        self.assertEqual(combined["LIKELY_OCCLUDED"], 1)
        self.assertEqual(combined["AMBIGUOUS"], 1)
        self.assertIn("OBSERVED", combined)

    def test_hand_availability_counts_only_pose_bearing_states(self) -> None:
        state = np.full((10, TRACKS), 1, dtype=np.int32)
        state[:4, 1] = 0     # RIGHT missing on 4 frames
        state[:2, 0] = 4     # LEFT occluded on 2 frames
        _write_glove(self.root, "a", 10, state_code=state)
        result = aggregate_dataset(self.rows, self.root)
        avail = result["hand_availability"]
        # sample a contributes 8 LEFT, 6 RIGHT, 6 both; b and c contribute 20 each
        self.assertEqual(avail["left_available"], 8 + 20 + 30)
        self.assertEqual(avail["right_available"], 6 + 20 + 30)
        self.assertEqual(avail["both_available"], 6 + 20 + 30)

    def test_occluded_hand_is_never_counted_as_available(self) -> None:
        state = np.full((5, TRACKS), 4, dtype=np.int32)  # all LIKELY_OCCLUDED
        _write_glove(self.root, "a", 5, state_code=state)
        rows = [_row("a", 5)]
        result = aggregate_dataset(rows, self.root)
        self.assertEqual(result["hand_availability"]["left_available"], 0)
        self.assertEqual(result["hand_availability"]["right_available"], 0)
        self.assertEqual(result["hand_availability"]["neither_available"], 5)

    def test_state_names_match_the_frozen_task004_contract(self) -> None:
        from tracking.wilor.schema import STATE_CODES

        for state, code in STATE_CODES.items():
            self.assertEqual(TRACKING_STATE_NAMES[code], state.value)

    def test_pose_bearing_codes_match_the_frozen_contract(self) -> None:
        from tracking.wilor.schema import POSE_STATES, STATE_CODES

        self.assertEqual({STATE_CODES[s] for s in POSE_STATES}, set(POSE_BEARING_CODES))

    def test_worst_class_lists_are_ordered_ascending(self) -> None:
        _write_glove(self.root, "b", 20, spread_invalid=100)
        result = aggregate_dataset(self.rows, self.root)
        worst = result["worst_classes_by_spread_validity"]
        self.assertEqual(worst[0]["sign_id"], "0033")
        fractions = [w["spread_valid_fraction"] for w in worst]
        self.assertEqual(fractions, sorted(fractions))


class TestCoverageAccounting(unittest.TestCase):
    def _state(self, mapping: dict[str, str]) -> dict[str, object]:
        return {"samples": {k: {"status": v} for k, v in mapping.items()}}

    def test_all_successful(self) -> None:
        rows = [_row("a", 5), _row("b", 5)]
        result = coverage_accounting(rows, self._state(
            {"a": "VIRTUAL_GLOVE_DONE", "b": "VIRTUAL_GLOVE_DONE"}))
        self.assertEqual(result["requested"], 2)
        self.assertEqual(result["successful"], 2)
        self.assertEqual(result["failed"], 0)
        self.assertEqual(result["unaccounted_count"], 0)
        self.assertTrue(result["all_accounted"])

    def test_failures_are_counted_not_hidden(self) -> None:
        rows = [_row("a", 5), _row("b", 5)]
        result = coverage_accounting(rows, self._state(
            {"a": "VIRTUAL_GLOVE_DONE", "b": "FAILED"}))
        self.assertEqual(result["successful"], 1)
        self.assertEqual(result["failed"], 1)
        self.assertTrue(result["all_accounted"])

    def test_a_sample_absent_from_state_is_unaccounted(self) -> None:
        rows = [_row("a", 5), _row("b", 5)]
        result = coverage_accounting(rows, self._state({"a": "VIRTUAL_GLOVE_DONE"}))
        self.assertEqual(result["unaccounted_count"], 1)
        self.assertEqual(result["unaccounted"], ["b"])
        self.assertFalse(result["all_accounted"])

    def test_a_half_finished_sample_is_not_counted_as_success(self) -> None:
        rows = [_row("a", 5)]
        result = coverage_accounting(rows, self._state({"a": "TRACKING_DONE"}))
        self.assertEqual(result["successful"], 0)
        self.assertEqual(result["incomplete_count"], 1)
        self.assertFalse(result["all_accounted"])


if __name__ == "__main__":
    unittest.main()
