"""Synthetic tests for TASK-006C virtual-glove QA tooling."""

from __future__ import annotations

import csv
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.virtual_glove_qa.contract import (
    FINGER_NAMES,
    JOINT_NAMES,
    SPREAD_NAMES,
    VIRTUAL_GLOVE_META_NAME,
    VIRTUAL_GLOVE_NPZ_NAME,
    sha256_file,
)
from evaluation.virtual_glove_qa.validator import validate_runs, write_csv, write_json

ROOT = Path(__file__).resolve().parents[1]


def _rotation_z(degrees: float) -> np.ndarray:
    radians = np.deg2rad(degrees)
    cosine = float(np.cos(radians))
    sine = float(np.sin(radians))
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _quaternion_z(degrees: float) -> np.ndarray:
    radians = np.deg2rad(degrees) / 2.0
    return np.array([np.cos(radians), 0.0, 0.0, np.sin(radians)], dtype=np.float64)


def _sensor_layout() -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for hand in ("LEFT", "RIGHT"):
        for finger in FINGER_NAMES:
            for joint in JOINT_NAMES:
                entries.append(
                    {
                        "sensor_id": f"{hand}_BEND_{finger}_{joint}",
                        "sensor_type": "hall",
                        "logical_location": {
                            "family": "bend",
                            "hand": hand,
                            "finger": finger,
                            "joint": joint,
                        },
                        "display_marker": "H",
                        "description": f"{hand} {finger} {joint} bend Hall sensor",
                    }
                )
        for pair in SPREAD_NAMES:
            entries.append(
                {
                    "sensor_id": f"{hand}_SPREAD_{pair}",
                    "sensor_type": "magnetic",
                    "logical_location": {
                        "family": "spread",
                        "hand": hand,
                        "spread_pair": pair,
                    },
                    "display_marker": "H",
                    "description": f"{hand} {pair} spread Hall sensor",
                }
            )
        entries.append(
            {
                "sensor_id": f"{hand}_PALM_IMU",
                "sensor_type": "imu",
                "logical_location": {"family": "imu", "hand": hand, "location": "palm"},
                "display_marker": "IMU",
                "description": f"{hand} palm orientation IMU",
            }
        )
    return entries


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_fixture(root: Path, *, partial_spread: bool = False) -> tuple[Path, Path, str]:
    kinematics_run = root / "task005_run"
    virtual_run = root / "task006_run"
    sample_id = "sample_a"
    source_dir = kinematics_run / sample_id
    glove_dir = virtual_run / sample_id
    source_dir.mkdir(parents=True)
    glove_dir.mkdir(parents=True)

    frame_count = 4
    frame_index = np.arange(100, 100 + frame_count, dtype=np.int32)
    timestamps = np.arange(frame_count, dtype=np.float64) / 30.0
    state = np.ones((frame_count, 2), dtype=np.int8)
    raw_index = np.tile(np.array([[2, 3]], dtype=np.int16), (frame_count, 1))
    flexion = np.empty((frame_count, 2, 5, 3), dtype=np.float64)
    spread = np.empty((frame_count, 2, 4), dtype=np.float64)
    rotations = np.empty((frame_count, 2, 3, 3), dtype=np.float64)
    quaternions = np.empty((frame_count, 2, 4), dtype=np.float64)
    for row in range(frame_count):
        for hand in range(2):
            flexion[row, hand] = 12.0 + row + hand + np.arange(15, dtype=np.float64).reshape(5, 3)
            spread[row, hand] = 8.0 + row + hand + np.arange(4, dtype=np.float64)
            if partial_spread and row == 1 and hand == 0:
                spread[row, hand, 2] = np.nan
            rotations[row, hand] = _rotation_z(4.0 * row + 2.0 * hand)
            quaternions[row, hand] = _quaternion_z(4.0 * row + 2.0 * hand)
    palm_valid = np.ones((frame_count, 2), dtype=bool)

    source_arrays = {
        "frame_index": frame_index,
        "timestamp_seconds": timestamps,
        "tracking_state_code": state,
        "source_raw_detection_index": raw_index,
        "valid_palm_frame": palm_valid,
        "flexion_deg": flexion,
        "adjacent_spread_deg": spread,
        "palm_rotation_matrix": rotations,
        "palm_quaternion_wxyz": quaternions,
    }
    np.savez_compressed(source_dir / "hand_kinematics.npz", **source_arrays)
    source_meta = {
        "schema_version": "TASK-005-final-v2",
        "stage": "kinematics",
        "task": "TASK-005F",
        "sample_id": sample_id,
        "total_frames": frame_count,
        "track_order": ["left", "right"],
    }
    _write_json(source_dir / "hand_kinematics_meta.json", source_meta)

    bend_valid = np.isfinite(flexion)
    spread_valid = np.isfinite(spread)
    bend_normalized = np.divide(
        flexion,
        180.0,
        out=np.full_like(flexion, np.nan),
        where=bend_valid,
    )
    spread_normalized = np.divide(
        spread,
        180.0,
        out=np.full_like(spread, np.nan),
        where=spread_valid,
    )
    gyro = np.full((frame_count, 2, 3), np.nan, dtype=np.float64)
    gyro_valid = np.ones((frame_count, 2), dtype=bool)
    for row in range(frame_count):
        for hand in range(2):
            gyro[row, hand] = [row + hand, 2.0 * row, -float(hand)]
    adc_bend = np.where(bend_valid, np.rint(bend_normalized * 4095.0), np.nan)
    adc_spread = np.where(spread_valid, np.rint(spread_normalized * 4095.0), np.nan)
    glove_arrays = {
        "frame_index": frame_index.copy(),
        "timestamp_seconds": timestamps.copy(),
        "tracking_state_code": state.copy(),
        "source_raw_detection_index": raw_index.copy(),
        "bend_angle_deg": flexion.copy(),
        "bend_normalized": bend_normalized,
        "bend_valid": bend_valid,
        "spread_angle_deg": spread.copy(),
        "spread_normalized": spread_normalized,
        "spread_valid": spread_valid,
        "imu_rotation_matrix": rotations.copy(),
        "imu_quaternion_wxyz": quaternions.copy(),
        "palm_imu_valid": palm_valid.copy(),
        "bend_adc_12bit": adc_bend,
        "spread_adc_12bit": adc_spread,
        "imu_angular_velocity_rad_s": gyro,
        "imu_angular_velocity_valid": gyro_valid,
    }
    np.savez_compressed(glove_dir / VIRTUAL_GLOVE_NPZ_NAME, **glove_arrays)
    glove_meta = {
        "schema_version": "task006_virtual_glove_v1",
        "stage": "virtual_glove",
        "task": "TASK-006A",
        "sample_id": sample_id,
        "total_frames": frame_count,
        "track_order": ["LEFT", "RIGHT"],
        "sensor_counts": {"bend_hall": 15, "spread_hall": 4, "hall_total": 19, "palm_imu": 1},
        "sensor_layout": _sensor_layout(),
        "normalization": {
            "method": "fixed_angle_divisor",
            "formula": "angle_deg / 180",
            "angle_divisor_deg": 180,
            "range": [0, 1],
            "fit_scope": "global",
        },
        "adc_transfer": {
            "bits": 12,
            "min_code": 0,
            "max_code": 4095,
            "mapping": "round(normalized * 4095)",
            "tolerance_codes": 1,
        },
        "source": {
            "task": "TASK-005F",
            "stage": "kinematics",
            "sample_id": sample_id,
            "kinematics_run": str(kinematics_run),
            "kinematics_npz_sha256": sha256_file(source_dir / "hand_kinematics.npz"),
            "kinematics_schema_version": source_meta["schema_version"],
        },
    }
    _write_json(glove_dir / VIRTUAL_GLOVE_META_NAME, glove_meta)
    return kinematics_run, virtual_run, sample_id


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **arrays)


def _load_meta(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


class TestTask006CVirtualGloveQA(unittest.TestCase):
    def _validated_fixture(self, *, partial_spread: bool = False):
        temporary = tempfile.TemporaryDirectory()
        paths = _build_fixture(Path(temporary.name), partial_spread=partial_spread)
        self.addCleanup(temporary.cleanup)
        return paths

    def test_valid_complete_run(self) -> None:
        kinematics_run, virtual_run, _ = self._validated_fixture()
        summary, rows = validate_runs(kinematics_run, virtual_run)
        self.assertTrue(summary["passed"], summary)
        self.assertEqual(summary["verdict"], "VIRTUAL-GLOVE QA TOOLING READY")
        self.assertEqual(summary["sensor_layout"]["expected"]["run_total"]["all_sensors"], 40)
        self.assertEqual(len(rows), 38 * 2 + 6)

    def test_malformed_shape_fails_schema(self) -> None:
        kinematics_run, virtual_run, sample_id = self._validated_fixture()
        path = virtual_run / sample_id / VIRTUAL_GLOVE_NPZ_NAME
        arrays = _load_npz(path)
        arrays["bend_angle_deg"] = arrays["bend_angle_deg"][:, :, :, :2]
        _save_npz(path, arrays)
        summary, _ = validate_runs(kinematics_run, virtual_run)
        self.assertFalse(summary["passed"])
        self.assertIn("bend_angle_deg shape mismatch", " ".join(summary["schema_validation"]["sample_failures"][sample_id]))

    def test_missing_sensor_fails_layout(self) -> None:
        kinematics_run, virtual_run, sample_id = self._validated_fixture()
        path = virtual_run / sample_id / VIRTUAL_GLOVE_META_NAME
        metadata = _load_meta(path)
        metadata["sensor_layout"] = metadata["sensor_layout"][:-1]
        _write_json(path, metadata)
        summary, _ = validate_runs(kinematics_run, virtual_run)
        self.assertFalse(summary["sensor_layout"]["passed"])
        self.assertTrue(any("exactly 40" in failure for failure in summary["sensor_layout"]["failures"]))

    def test_duplicate_sensor_id_fails_layout(self) -> None:
        kinematics_run, virtual_run, sample_id = self._validated_fixture()
        path = virtual_run / sample_id / VIRTUAL_GLOVE_META_NAME
        metadata = _load_meta(path)
        layout = metadata["sensor_layout"]
        layout[1]["sensor_id"] = layout[0]["sensor_id"]
        _write_json(path, metadata)
        summary, _ = validate_runs(kinematics_run, virtual_run)
        self.assertFalse(summary["sensor_layout"]["passed"])
        self.assertTrue(any("duplicate sensor ID" in failure for failure in summary["sensor_layout"]["failures"]))

    def test_missing_h_marker_fails_layout(self) -> None:
        kinematics_run, virtual_run, sample_id = self._validated_fixture()
        path = virtual_run / sample_id / VIRTUAL_GLOVE_META_NAME
        metadata = _load_meta(path)
        metadata["sensor_layout"][0]["display_marker"] = ""
        _write_json(path, metadata)
        summary, _ = validate_runs(kinematics_run, virtual_run)
        self.assertFalse(summary["sensor_layout"]["passed"])
        self.assertTrue(any("display_marker must be 'H'" in failure for failure in summary["sensor_layout"]["failures"]))

    def test_missing_imu_marker_fails_layout(self) -> None:
        kinematics_run, virtual_run, sample_id = self._validated_fixture()
        path = virtual_run / sample_id / VIRTUAL_GLOVE_META_NAME
        metadata = _load_meta(path)
        imu = next(entry for entry in metadata["sensor_layout"] if entry["sensor_type"] == "imu")
        imu["display_marker"] = "H"
        _write_json(path, metadata)
        summary, _ = validate_runs(kinematics_run, virtual_run)
        self.assertFalse(summary["sensor_layout"]["passed"])
        self.assertTrue(any("display_marker must be 'IMU'" in failure for failure in summary["sensor_layout"]["failures"]))

    def test_wrong_normalization_fails_with_reference(self) -> None:
        kinematics_run, virtual_run, sample_id = self._validated_fixture()
        path = virtual_run / sample_id / VIRTUAL_GLOVE_NPZ_NAME
        arrays = _load_npz(path)
        arrays["bend_normalized"][0, 0, 0, 0] = 0.9
        _save_npz(path, arrays)
        summary, _ = validate_runs(kinematics_run, virtual_run)
        self.assertFalse(summary["normalization"]["passed"])
        violation = next(item for item in summary["normalization"]["violations"] if item["channel"] == "bend.thumb.MCP")
        self.assertEqual(violation["sample_id"], sample_id)
        self.assertEqual(violation["frame_index"], 100)
        self.assertEqual(violation["track"], "LEFT")

    def test_out_of_range_normalized_fails(self) -> None:
        kinematics_run, virtual_run, sample_id = self._validated_fixture()
        path = virtual_run / sample_id / VIRTUAL_GLOVE_NPZ_NAME
        arrays = _load_npz(path)
        arrays["bend_normalized"][0, 0, 0, 0] = 1.1
        _save_npz(path, arrays)
        summary, _ = validate_runs(kinematics_run, virtual_run)
        self.assertFalse(summary["normalization"]["passed"])
        self.assertTrue(any(item["reason"] == "normalized_out_of_range" for item in summary["normalization"]["violations"]))

    def test_run_specific_min_max_metadata_fails(self) -> None:
        kinematics_run, virtual_run, sample_id = self._validated_fixture()
        path = virtual_run / sample_id / VIRTUAL_GLOVE_META_NAME
        metadata = _load_meta(path)
        metadata["normalization"] = {
            "method": "run_min_max",
            "fit_scope": "run",
            "min": 0.0,
            "max": 90.0,
        }
        _write_json(path, metadata)
        summary, _ = validate_runs(kinematics_run, virtual_run)
        self.assertFalse(summary["normalization"]["passed"])
        self.assertIn(sample_id, summary["normalization"]["metadata_failures"])
        self.assertTrue(summary["normalization"]["metadata"][sample_id]["run_specific_evidence"])

    def test_valid_partial_spread_nan_is_channel_level_legal(self) -> None:
        kinematics_run, virtual_run, _ = self._validated_fixture(partial_spread=True)
        summary, _ = validate_runs(kinematics_run, virtual_run)
        self.assertTrue(summary["passed"], summary)
        self.assertGreater(summary["validity_masks"]["partial_channel_examples"].__len__(), 0)
        self.assertEqual(summary["nan_propagation"]["count"], 0)

    def test_illegal_nan_in_valid_channel_fails(self) -> None:
        kinematics_run, virtual_run, sample_id = self._validated_fixture()
        path = virtual_run / sample_id / VIRTUAL_GLOVE_NPZ_NAME
        arrays = _load_npz(path)
        arrays["bend_angle_deg"][0, 0, 0, 0] = np.nan
        _save_npz(path, arrays)
        summary, _ = validate_runs(kinematics_run, virtual_run)
        self.assertFalse(summary["validity_masks"]["passed"])
        self.assertTrue(any(item["reason"] == "valid_channel_has_non_finite_value" for item in summary["validity_masks"]["violations"]))

    def test_provenance_mismatch_fails(self) -> None:
        kinematics_run, virtual_run, sample_id = self._validated_fixture()
        path = virtual_run / sample_id / VIRTUAL_GLOVE_META_NAME
        metadata = _load_meta(path)
        metadata["source"]["sample_id"] = "other_sample"
        _write_json(path, metadata)
        summary, _ = validate_runs(kinematics_run, virtual_run)
        self.assertFalse(summary["provenance"]["passed"])
        self.assertTrue(any("sample_id" in failure for failure in summary["provenance"]["failures"][sample_id]))

    def test_frame_mismatch_is_explicit(self) -> None:
        kinematics_run, virtual_run, sample_id = self._validated_fixture()
        path = virtual_run / sample_id / VIRTUAL_GLOVE_NPZ_NAME
        arrays = _load_npz(path)
        arrays["frame_index"][2] = 999
        _save_npz(path, arrays)
        summary, _ = validate_runs(kinematics_run, virtual_run)
        self.assertFalse(summary["alignment"]["passed"])
        mismatch = next(item for item in summary["alignment"]["mismatches"] if item["field"] == "frame_index")
        self.assertEqual(mismatch["first_mismatch_position"], [2])
        self.assertEqual(mismatch["source"], 102)
        self.assertEqual(mismatch["virtual_glove"], 999)

    def test_rotation_fault_fails(self) -> None:
        kinematics_run, virtual_run, sample_id = self._validated_fixture()
        path = virtual_run / sample_id / VIRTUAL_GLOVE_NPZ_NAME
        arrays = _load_npz(path)
        arrays["imu_rotation_matrix"][0, 0, 0, 0] = 2.0
        _save_npz(path, arrays)
        summary, _ = validate_runs(kinematics_run, virtual_run)
        self.assertFalse(summary["rotation_quality"]["passed"])
        self.assertTrue(any(item["reason"] == "orthogonality_tolerance" for item in summary["rotation_quality"]["violations"]))

    def test_quaternion_fault_fails(self) -> None:
        kinematics_run, virtual_run, sample_id = self._validated_fixture()
        path = virtual_run / sample_id / VIRTUAL_GLOVE_NPZ_NAME
        arrays = _load_npz(path)
        arrays["imu_quaternion_wxyz"][0, 0] = [2.0, 0.0, 0.0, 0.0]
        _save_npz(path, arrays)
        summary, _ = validate_runs(kinematics_run, virtual_run)
        self.assertFalse(summary["rotation_quality"]["passed"])
        self.assertTrue(any(item["reason"] == "quaternion_norm_tolerance" for item in summary["rotation_quality"]["violations"]))

    def test_optional_adc_mismatch_fails(self) -> None:
        kinematics_run, virtual_run, sample_id = self._validated_fixture()
        path = virtual_run / sample_id / VIRTUAL_GLOVE_NPZ_NAME
        arrays = _load_npz(path)
        arrays["spread_adc_12bit"][0, 0, 0] += 100.0
        _save_npz(path, arrays)
        summary, _ = validate_runs(kinematics_run, virtual_run)
        self.assertFalse(summary["adc"]["passed"])
        self.assertTrue(any(item["reason"] == "adc_normalized_disagreement" for item in summary["adc"]["violations"]))

    def test_optional_adc_out_of_range_fails(self) -> None:
        kinematics_run, virtual_run, sample_id = self._validated_fixture()
        path = virtual_run / sample_id / VIRTUAL_GLOVE_NPZ_NAME
        arrays = _load_npz(path)
        arrays["bend_adc_12bit"][0, 0, 0, 0] = 4096.0
        _save_npz(path, arrays)
        summary, _ = validate_runs(kinematics_run, virtual_run)
        self.assertFalse(summary["adc"]["passed"])
        self.assertTrue(any(item["reason"] == "adc_out_of_range" for item in summary["adc"]["violations"]))

    def test_deterministic_json(self) -> None:
        kinematics_run, virtual_run, _ = self._validated_fixture()
        summary_a, rows_a = validate_runs(kinematics_run, virtual_run)
        summary_b, rows_b = validate_runs(kinematics_run, virtual_run)
        self.assertEqual(summary_a, summary_b)
        with tempfile.TemporaryDirectory() as output_dir:
            first = Path(output_dir) / "first.json"
            second = Path(output_dir) / "second.json"
            write_json(first, summary_a)
            write_json(second, summary_b)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(rows_a, rows_b)

    def test_deterministic_csv(self) -> None:
        kinematics_run, virtual_run, _ = self._validated_fixture()
        _, rows = validate_runs(kinematics_run, virtual_run)
        with tempfile.TemporaryDirectory() as output_dir:
            first = Path(output_dir) / "first.csv"
            second = Path(output_dir) / "second.csv"
            write_csv(first, rows)
            write_csv(second, rows)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with first.open(newline="", encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), len(rows))

    def test_cli_execution(self) -> None:
        kinematics_run, virtual_run, _ = self._validated_fixture()
        with tempfile.TemporaryDirectory() as output_dir:
            output = Path(output_dir)
            command = [
                "python",
                str(ROOT / "scripts/validate_task006_virtual_glove.py"),
                "--kinematics-run",
                str(kinematics_run),
                "--virtual-glove-run",
                str(virtual_run),
                "--output-json",
                str(output / "summary.json"),
                "--output-csv",
                str(output / "summary.csv"),
            ]
            completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            self.assertTrue((output / "summary.json").is_file())
            self.assertTrue((output / "summary.csv").is_file())


if __name__ == "__main__":
    unittest.main()
