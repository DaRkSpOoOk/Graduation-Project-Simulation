"""Synthetic tests for TASK-005C kinematics QA tooling."""

from __future__ import annotations

import csv
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.kinematics_qa.validator import validate_runs

ROOT = Path(__file__).resolve().parents[1]


def _rotation_z(degrees: float) -> np.ndarray:
    radians = np.deg2rad(degrees)
    c = float(np.cos(radians))
    s = float(np.sin(radians))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _build_base_arrays(frame_count: int = 4) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]]:
    frame_index = np.arange(frame_count, dtype=np.int32)
    timestamps = frame_index.astype(np.float64) / 30.0

    tracked_state = np.array([[1, 1], [1, 0], [2, 1], [1, 1]], dtype=np.int8)
    tracked_source = np.array([[0, 1], [0, -1], [1, 0], [2, 1]], dtype=np.int16)

    valid = np.array([[True, True], [True, False], [True, True], [True, True]], dtype=bool)
    flexion = np.full((frame_count, 2, 5, 3), np.nan, dtype=np.float64)
    spread = np.full((frame_count, 2, 4), np.nan, dtype=np.float64)
    rotation = np.full((frame_count, 2, 3, 3), np.nan, dtype=np.float64)
    quaternion = np.full((frame_count, 2, 4), np.nan, dtype=np.float64)

    for row in range(frame_count):
        for hand in range(2):
            if not valid[row, hand]:
                continue
            rotation[row, hand] = _rotation_z(5.0 * row + hand)
            angle = np.deg2rad((5.0 * row + hand) / 2.0)
            quaternion[row, hand] = np.array([np.cos(angle), 0.0, 0.0, np.sin(angle)], dtype=np.float64)
            for finger in range(5):
                for joint in range(3):
                    flexion[row, hand, finger, joint] = 10.0 + row + hand + finger * 0.2 + joint * 0.05
            for spread_idx in range(4):
                spread[row, hand, spread_idx] = 2.0 + row + hand + spread_idx * 0.3

    kinematics_arrays = {
        "frame_index": frame_index.copy(),
        "timestamp_seconds": timestamps.copy(),
        "tracking_state_code": tracked_state.copy(),
        "source_raw_detection_index": tracked_source.copy(),
        "valid_kinematics": valid.copy(),
        "flexion_deg": flexion.copy(),
        "adjacent_spread_deg": spread.copy(),
        "palm_rotation_matrix": rotation.copy(),
        "palm_quaternion_wxyz": quaternion.copy(),
        "kinematic_flags_json": np.full((frame_count, 2), "[]", dtype="U16"),
    }

    tracked_arrays = {
        "frame_index": frame_index.copy(),
        "timestamp_seconds": timestamps.copy(),
        "state_code": tracked_state.copy(),
        "raw_detection_index": tracked_source.copy(),
    }

    meta = {
        "sample_id": "sample_001",
        "track_order": ["LEFT", "RIGHT"],
        "finger_order": ["thumb", "index", "middle", "ring", "pinky"],
        "quaternion_convention": ["w", "x", "y", "z"],
    }
    return kinematics_arrays, tracked_arrays, meta


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _write_sample(
    root: Path,
    sample_id: str,
    kinematics_arrays: dict[str, np.ndarray],
    tracked_arrays: dict[str, np.ndarray],
    meta: dict[str, object],
) -> tuple[Path, Path]:
    tracked_dir = root / "tracked" / sample_id
    kine_dir = root / "kinematics" / sample_id
    tracked_dir.mkdir(parents=True, exist_ok=True)
    kine_dir.mkdir(parents=True, exist_ok=True)
    _write_npz(tracked_dir / "wilor_tracked.npz", tracked_arrays)
    _write_npz(kine_dir / "hand_kinematics.npz", kinematics_arrays)
    (kine_dir / "hand_kinematics_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return root / "tracked", root / "kinematics"


def _run_validator(
    mutate: callable | None = None,
    *,
    add_extra_kinematics_sample: bool = False,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        kine, tracked, meta = _build_base_arrays()
        if mutate is not None:
            mutate(kine, tracked, meta)
        tracked_run, kinematics_run = _write_sample(root, "sample_001", kine, tracked, meta)

        if add_extra_kinematics_sample:
            extra_kine, _, extra_meta = _build_base_arrays()
            extra_meta["sample_id"] = "sample_extra"
            _write_npz(kinematics_run / "sample_extra" / "hand_kinematics.npz", extra_kine)
            (kinematics_run / "sample_extra" / "hand_kinematics_meta.json").write_text(
                json.dumps(extra_meta, sort_keys=True), encoding="utf-8"
            )

        summary, csv_rows = validate_runs(tracked_run, kinematics_run)
    return summary, csv_rows


class TestTask005CKinematicsQa(unittest.TestCase):
    def test_01_completely_valid_fixture(self) -> None:
        summary, rows = _run_validator()
        self.assertTrue(summary["passed"])
        self.assertTrue(summary["contract_validation"]["passed"])
        self.assertTrue(summary["tracking_alignment"]["passed"])
        self.assertGreater(len(rows), 0)

    def test_02_required_npz_field_missing(self) -> None:
        def mutate(kine, *_):
            del kine["flexion_deg"]

        summary, _ = _run_validator(mutate)
        self.assertFalse(summary["contract_validation"]["passed"])
        self.assertIn("sample_001", summary["contract_validation"]["sample_failures"])

    def test_03_wrong_flexion_shape(self) -> None:
        def mutate(kine, *_):
            kine["flexion_deg"] = np.full((4, 2, 5, 2), np.nan)

        summary, _ = _run_validator(mutate)
        self.assertFalse(summary["contract_validation"]["passed"])

    def test_04_wrong_spread_shape(self) -> None:
        def mutate(kine, *_):
            kine["adjacent_spread_deg"] = np.full((4, 2, 5), np.nan)

        summary, _ = _run_validator(mutate)
        self.assertFalse(summary["contract_validation"]["passed"])

    def test_05_wrong_track_dimension(self) -> None:
        def mutate(kine, *_):
            kine["tracking_state_code"] = np.zeros((4, 3), dtype=np.int8)

        summary, _ = _run_validator(mutate)
        self.assertFalse(summary["contract_validation"]["passed"])

    def test_06_mismatching_frame_count(self) -> None:
        def mutate(kine, *_):
            for key, value in list(kine.items()):
                kine[key] = value[:3]

        summary, _ = _run_validator(mutate)
        self.assertFalse(summary["tracking_alignment"]["passed"])
        self.assertEqual(summary["tracking_alignment"]["mismatches"][0]["field"], "frame_count")

    def test_07_duplicate_frame_index(self) -> None:
        def mutate(kine, *_):
            kine["frame_index"] = np.array([0, 1, 1, 3], dtype=np.int32)

        summary, _ = _run_validator(mutate)
        joined = "\n".join(summary["contract_validation"]["sample_failures"]["sample_001"])
        self.assertIn("duplicate", joined)

    def test_08_non_monotonic_frame_indices(self) -> None:
        def mutate(kine, *_):
            kine["frame_index"] = np.array([0, 2, 1, 3], dtype=np.int32)

        summary, _ = _run_validator(mutate)
        joined = "\n".join(summary["contract_validation"]["sample_failures"]["sample_001"])
        self.assertIn("monotonically increasing", joined)

    def test_09_non_monotonic_timestamps(self) -> None:
        def mutate(kine, *_):
            kine["timestamp_seconds"] = np.array([0.0, 0.1, 0.05, 0.2], dtype=np.float64)

        summary, _ = _run_validator(mutate)
        joined = "\n".join(summary["contract_validation"]["sample_failures"]["sample_001"])
        self.assertIn("timestamp_seconds is not monotonically increasing", joined)

    def test_10_invalid_hand_with_finite_values(self) -> None:
        def mutate(kine, *_):
            kine["valid_kinematics"][1, 1] = False
            kine["flexion_deg"][1, 1, 0, 0] = 12.0

        summary, _ = _run_validator(mutate)
        self.assertGreater(summary["invalid_mask_violations"]["count"], 0)

    def test_11_valid_hand_with_nan_output(self) -> None:
        def mutate(kine, *_):
            kine["valid_kinematics"][0, 0] = True
            kine["adjacent_spread_deg"][0, 0, 0] = np.nan

        summary, _ = _run_validator(mutate)
        self.assertGreater(summary["non_finite_violations"]["count"], 0)

    def test_12_non_orthogonal_rotation_matrix(self) -> None:
        def mutate(kine, *_):
            kine["palm_rotation_matrix"][0, 0] = np.array(
                [[1.0, 0.2, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
            )

        summary, _ = _run_validator(mutate)
        self.assertGreater(summary["rotation_errors"]["orthogonality"]["max"], 0.0)

    def test_13_rotation_determinant_minus_one(self) -> None:
        def mutate(kine, *_):
            kine["palm_rotation_matrix"][0, 0] = np.diag([1.0, 1.0, -1.0])

        summary, _ = _run_validator(mutate)
        self.assertEqual(summary["rotation_errors"]["determinant_non_positive"]["count"], 1)

    def test_14_non_unit_quaternion(self) -> None:
        def mutate(kine, *_):
            kine["palm_quaternion_wxyz"][0, 0] = np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float64)

        summary, _ = _run_validator(mutate)
        self.assertGreater(summary["quaternion_errors"]["norm_abs_error"]["max"], 0.9)

    def test_15_matrix_quaternion_disagreement(self) -> None:
        def mutate(kine, *_):
            kine["palm_quaternion_wxyz"][0, 0] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            kine["palm_rotation_matrix"][0, 0] = _rotation_z(90.0)

        summary, _ = _run_validator(mutate)
        self.assertGreater(
            summary["quaternion_errors"]["matrix_quaternion_angular_disagreement_deg"]["max"],
            80.0,
        )

    def test_16_task004_task005_frame_mismatch(self) -> None:
        def mutate(kine, *_):
            kine["frame_index"] = np.array([0, 2, 3, 4], dtype=np.int32)

        summary, _ = _run_validator(mutate)
        mismatch_fields = [item["field"] for item in summary["tracking_alignment"]["mismatches"]]
        self.assertIn("frame_index", mismatch_fields)

    def test_17_source_detection_provenance_mismatch(self) -> None:
        def mutate(kine, *_):
            kine["source_raw_detection_index"][0, 1] = 99

        summary, _ = _run_validator(mutate)
        mismatch_fields = [item["field"] for item in summary["tracking_alignment"]["mismatches"]]
        self.assertIn("source_raw_detection_index", mismatch_fields)

    def test_18_deterministic_json_results(self) -> None:
        first, _ = _run_validator(add_extra_kinematics_sample=True)
        second, _ = _run_validator(add_extra_kinematics_sample=True)
        self.assertEqual(
            json.dumps(first, sort_keys=True, separators=(",", ":")),
            json.dumps(second, sort_keys=True, separators=(",", ":")),
        )

    def test_cli_writes_json_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kine, tracked, meta = _build_base_arrays()
            tracked_run, kinematics_run = _write_sample(root, "sample_001", kine, tracked, meta)
            json_out = root / "out" / "summary.json"
            csv_out = root / "out" / "summary.csv"
            result = subprocess.run(
                [
                    "python",
                    str(ROOT / "scripts/validate_task005_kinematics.py"),
                    "--tracked-run",
                    str(tracked_run),
                    "--kinematics-run",
                    str(kinematics_run),
                    "--output-json",
                    str(json_out),
                    "--output-csv",
                    str(csv_out),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(json_out.is_file())
            self.assertTrue(csv_out.is_file())
            with csv_out.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertGreater(len(rows), 0)


if __name__ == "__main__":
    unittest.main()
