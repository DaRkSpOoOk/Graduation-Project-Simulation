"""Tests for the TASK-008C finalization checks.

The production run is 4,222 external sequences; none of it is read here. Every
fixture is a synthetic run root built in a temporary directory, so each check is
exercised on both a conforming dataset and a deliberately corrupted one.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.dataset.final_qa import (
    ANGLE_DEGREE_CEILING,
    EXPECTED_CHAIN_ORDER,
    EXPECTED_FINGER_ORDER,
    EXPECTED_SPREAD_PAIRS,
    sequence_length_statistics,
    verify_dataset_contract,
    verify_label_integrity,
    verify_loso_folds,
)
from evaluation.dataset.orchestrator import RunPaths

FINGERS, CHAIN, SPREAD, TRACKS = 5, 3, 4, 2


def _layout() -> dict:
    sensors = []
    for finger_index, finger in enumerate(EXPECTED_FINGER_ORDER):
        for chain_index, joint in enumerate(EXPECTED_CHAIN_ORDER):
            sensors.append({
                "sensor_id": f"H_{finger.upper()}_{joint.upper()}", "array": "bend_angle_deg",
                "array_index": [finger_index, chain_index], "role": "bend", "finger": finger,
                "joint": joint, "pair": None, "display_marker": "H",
                "sensor_type": "hall_bend_angular",
            })
    for pair_index, pair in enumerate(EXPECTED_SPREAD_PAIRS):
        sensors.append({
            "sensor_id": f"H_SPREAD_{pair[0].upper()}_{pair[1].upper()}",
            "array": "spread_angle_deg", "array_index": [pair_index], "role": "spread",
            "finger": None, "joint": None, "pair": list(pair), "display_marker": "H",
            "sensor_type": "hall_spread_angular",
        })
    sensors.append({
        "sensor_id": "IMU_PALM", "array": "imu_quaternion_wxyz", "array_index": [],
        "role": "orientation", "finger": None, "joint": None, "pair": None,
        "display_marker": "IMU", "sensor_type": "imu_orientation",
    })
    return {
        "layout_version": "ideal_virtual_glove_v1",
        "finger_order": list(EXPECTED_FINGER_ORDER),
        "chain_joint_order": list(EXPECTED_CHAIN_ORDER),
        "per_hand_counts": {"bend_hall_sensors": 15, "spread_hall_sensors": 4,
                            "hall_sensors_total": 19, "imu_packages": 1,
                            "logical_sensing_packages": 20},
        "sensors": sensors,
    }


def _write_sample(root: Path, sample_id: str, frames: int, **overrides) -> None:
    paths = RunPaths(root)
    glove_dir = paths.virtual_glove / sample_id
    glove_dir.mkdir(parents=True, exist_ok=True)

    bend_deg = np.full((frames, TRACKS, FINGERS, CHAIN), 30.0)
    spread_deg = np.full((frames, TRACKS, SPREAD), 20.0)
    bend_valid = np.ones_like(bend_deg, dtype=bool)
    spread_valid = np.ones_like(spread_deg, dtype=bool)
    imu_valid = np.ones((frames, TRACKS), dtype=bool)
    quaternion = np.zeros((frames, TRACKS, 4))
    quaternion[..., 0] = 1.0
    frame_index = np.arange(frames, dtype=np.int32)
    timestamps = frame_index / 30.0

    for key, value in overrides.items():
        locals_map = {
            "bend_deg": bend_deg, "spread_deg": spread_deg, "bend_valid": bend_valid,
            "spread_valid": spread_valid, "imu_valid": imu_valid, "quaternion": quaternion,
        }
        if key in locals_map:
            locals_map[key][...] = value
    frame_index = overrides.get("frame_index", frame_index)
    timestamps = overrides.get("timestamps", timestamps)
    bend_deg = overrides.get("bend_deg_array", bend_deg)
    spread_deg = overrides.get("spread_deg_array", spread_deg)
    bend_valid = overrides.get("bend_valid_array", bend_valid)
    quaternion = overrides.get("quaternion_array", quaternion)

    bend_deg = np.where(bend_valid, bend_deg, np.nan)
    spread_deg = np.where(spread_valid, spread_deg, np.nan)
    np.savez_compressed(
        glove_dir / "virtual_glove.npz",
        frame_index=np.asarray(frame_index, dtype=np.int32),
        timestamp_seconds=np.asarray(timestamps, dtype=np.float64),
        bend_angle_deg=bend_deg.astype(np.float32),
        bend_normalized=(bend_deg / ANGLE_DEGREE_CEILING).astype(np.float32),
        bend_valid=bend_valid,
        spread_angle_deg=spread_deg.astype(np.float32),
        spread_normalized=(spread_deg / ANGLE_DEGREE_CEILING).astype(np.float32),
        spread_valid=spread_valid,
        palm_imu_valid=imu_valid,
        imu_quaternion_wxyz=quaternion.astype(np.float32),
        tracking_state_code=np.ones((frames, TRACKS), dtype=np.int32),
    )
    layout = overrides.get("layout", _layout())
    (glove_dir / "sensor_layout.json").write_text(json.dumps(layout), encoding="utf-8")


def _row(sample_id: str, frames: int, *, signer="01", sign_id="0032", partition="train",
         label="ا", label_index="0", repetition="rep001") -> dict[str, str]:
    """A full-schema manifest row, so the frozen validators accept the fixture."""

    return {
        "sample_id": sample_id, "source_dataset": "KArSL", "dataset_version": "KArSL-502",
        "modality": "RGB", "sign_id": sign_id, "label_ar": label,
        "label_en_if_available": "x", "label_index": label_index, "signer_id": signer,
        "official_partition": partition, "repetition_id": repetition,
        "source_relative_path": f"{signer}/{partition}/videos/{sign_id}/{sample_id}.mp4",
        "source_file_name": f"{sample_id}.mp4", "source_url": "",
        "source_sha256": "a" * 64, "source_size_bytes": "1024", "container": "mp4",
        "width": "1920", "height": "1080", "fps": "30.000000", "frame_count": str(frames),
        "duration_seconds": f"{frames / 30.0:.6f}", "skeleton_available": "unknown",
    }


class TestSequenceLengthStatistics(unittest.TestCase):
    def test_reports_every_percentile_the_report_cites(self) -> None:
        block = sequence_length_statistics(list(range(1, 101)))
        for key in ("count", "min", "max", "mean", "p5", "p25", "median", "p75", "p95"):
            self.assertIn(key, block)
        self.assertEqual(block["min"], 1)
        self.assertEqual(block["max"], 100)
        self.assertAlmostEqual(block["median"], 50.5)

    def test_empty_input_is_not_an_error(self) -> None:
        self.assertEqual(sequence_length_statistics([]), {"count": 0})


class TestDatasetContract(unittest.TestCase):
    def test_conforming_dataset_passes_every_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [_row("a", 5), _row("b", 7)]
            for row in rows:
                _write_sample(root, row["sample_id"], int(row["frame_count"]))
            result = verify_dataset_contract(rows, root)
        self.assertTrue(result["contract_intact"])
        self.assertEqual(result["samples_checked"], 2)
        self.assertEqual(result["violation_count"], 0)
        self.assertTrue(result["sensor_layout"]["channel_order_identical_across_samples"])
        self.assertEqual(result["sensor_layout"]["two_hand_hall_channels"], 38)
        self.assertEqual(result["sensor_layout"]["two_hand_imu_packages"], 2)

    def test_non_monotonic_frame_index_is_a_violation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [_row("a", 4)]
            _write_sample(root, "a", 4, frame_index=np.array([0, 2, 1, 3], dtype=np.int32))
            result = verify_dataset_contract(rows, root)
        self.assertFalse(result["contract_intact"])
        self.assertFalse(result["temporal"]["frame_index_strictly_increasing"])

    def test_non_monotonic_timestamp_is_a_violation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_sample(root, "a", 4, timestamps=np.array([0.0, 0.2, 0.1, 0.3]))
            result = verify_dataset_contract([_row("a", 4)], root)
        self.assertFalse(result["temporal"]["timestamp_strictly_increasing"])

    def test_finite_value_on_an_invalid_channel_is_flagged_as_imputation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = RunPaths(root)
            _write_sample(root, "a", 3)
            path = paths.virtual_glove / "a" / "virtual_glove.npz"
            with np.load(path, allow_pickle=False) as data:
                arrays = {key: data[key] for key in data.files}
            arrays["bend_valid"][0, 0, 0, 0] = False  # value stays 30.0, not NaN
            np.savez_compressed(path, **arrays)
            result = verify_dataset_contract([_row("a", 3)], root)
        self.assertEqual(result["imputed_invalid_channels"], 1)
        self.assertFalse(result["contract_intact"])

    def test_out_of_range_angle_is_a_violation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_sample(root, "a", 3, bend_deg=190.0)
            result = verify_dataset_contract([_row("a", 3)], root)
        self.assertFalse(result["contract_intact"])
        self.assertTrue(any("bend angle outside" in v["problem"] for v in result["violations"]))

    def test_negative_w_quaternion_violates_the_sign_convention(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quaternion = np.zeros((3, TRACKS, 4))
            quaternion[..., 0] = -1.0
            _write_sample(root, "a", 3, quaternion_array=quaternion)
            result = verify_dataset_contract([_row("a", 3)], root)
        self.assertEqual(result["quaternion"]["negative_w_count"], 6)
        self.assertFalse(result["contract_intact"])

    def test_non_unit_quaternion_is_a_violation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quaternion = np.zeros((3, TRACKS, 4))
            quaternion[..., 0] = 0.5
            _write_sample(root, "a", 3, quaternion_array=quaternion)
            result = verify_dataset_contract([_row("a", 3)], root)
        self.assertGreater(result["quaternion"]["max_unit_norm_error"], 1e-4)
        self.assertFalse(result["contract_intact"])

    def test_reordered_bend_channels_are_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            layout = _layout()
            layout["sensors"][0], layout["sensors"][1] = layout["sensors"][1], layout["sensors"][0]
            _write_sample(root, "a", 3, layout=layout)
            result = verify_dataset_contract([_row("a", 3)], root)
        self.assertFalse(result["contract_intact"])
        self.assertTrue(result["sensor_layout"]["contract_problems"])

    def test_layout_differing_between_samples_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_sample(root, "a", 3)
            other = _layout()
            other["sensors"][0]["sensor_id"] = "H_RENAMED"
            _write_sample(root, "b", 3, layout=other)
            result = verify_dataset_contract([_row("a", 3), _row("b", 3)], root)
        self.assertEqual(result["sensor_layout"]["distinct_layouts"], 2)
        self.assertFalse(result["sensor_layout"]["channel_order_identical_across_samples"])
        self.assertFalse(result["contract_intact"])

    def test_wrong_per_hand_counts_are_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            layout = _layout()
            layout["per_hand_counts"]["hall_sensors_total"] = 5
            _write_sample(root, "a", 3, layout=layout)
            result = verify_dataset_contract([_row("a", 3)], root)
        self.assertTrue(any("hall_sensors_total" in p["problem"]
                            for p in result["sensor_layout"]["contract_problems"]))

    def test_missing_output_is_reported_not_skipped_silently(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = verify_dataset_contract([_row("ghost", 3)], Path(tmp))
        self.assertEqual(result["samples_checked"], 0)
        self.assertEqual(result["violation_count"], 1)


class TestLabelIntegrity(unittest.TestCase):
    def _labels(self) -> list[dict[str, str]]:
        return [{"sign_id": f"{32 + i:04d}", "label_ar": chr(0x0627 + i),
                 "label_index": str(i)} for i in range(28)]

    def _rows(self) -> list[dict[str, str]]:
        return [_row(f"s{i}", 5, sign_id=f"{32 + i:04d}", label=chr(0x0627 + i),
                     label_index=str(i)) for i in range(28)]

    def test_intact_labels_pass(self) -> None:
        result = verify_label_integrity(self._rows(), self._labels())
        self.assertTrue(result["labels_intact"])
        self.assertTrue(result["class_count_correct"])
        self.assertEqual(result["distinct_classes"], 28)
        self.assertTrue(result["label_index_contiguous"])

    def test_label_text_drift_is_detected(self) -> None:
        rows = self._rows()
        rows[3]["label_ar"] = "؟"
        result = verify_label_integrity(rows, self._labels())
        self.assertFalse(result["labels_intact"])

    def test_label_index_drift_is_detected(self) -> None:
        rows = self._rows()
        rows[3]["label_index"] = "99"
        result = verify_label_integrity(rows, self._labels())
        self.assertFalse(result["labels_intact"])

    def test_non_core28_sign_is_rejected(self) -> None:
        rows = self._rows()
        rows[0]["sign_id"] = "0060"
        result = verify_label_integrity(rows, self._labels())
        self.assertFalse(result["labels_intact"])

    def test_missing_source_provenance_is_reported(self) -> None:
        rows = self._rows()
        rows[0]["source_sha256"] = ""
        result = verify_label_integrity(rows, self._labels())
        self.assertFalse(result["labels_intact"])


class TestLosoFolds(unittest.TestCase):
    def _manifest(self) -> list[dict[str, str]]:
        # The frozen manifest validator cross-checks label_ar/label_en against
        # the real Core-28 table, so the fixture uses the real records.
        from evaluation.dataset.core28 import core28_records

        rows = []
        for signer in ("01", "02", "03"):
            for label_index, record in enumerate(core28_records()):
                for repetition in range(3):
                    sign_id = f"{int(record.sign_id):04d}"
                    rows.append(_row(
                        f"karsl_core28_s{signer}_sign{sign_id}_train_rep{repetition:03d}",
                        5, signer=signer, sign_id=sign_id, label=record.label_ar,
                        label_index=str(label_index), repetition=f"rep{repetition:03d}",
                    ))
                    rows[-1]["label_en_if_available"] = record.label_en
        return rows

    def _folds(self, manifest: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
        from evaluation.dataset.splits import build_loso_splits
        return {signer: build_loso_splits(manifest, signer) for signer in ("01", "02", "03")}

    def test_generated_folds_are_intact(self) -> None:
        manifest = self._manifest()
        result = verify_loso_folds(self._folds(manifest), manifest)
        self.assertTrue(result["loso_intact"])
        self.assertEqual(result["fold_count"], 3)
        for signer, entry in result["folds"].items():
            self.assertEqual(entry["test_signers"], [signer])
            self.assertFalse(entry["held_out_signer_leakage"])
            self.assertTrue(entry["covers_manifest_exactly"])

    def test_held_out_signer_leaking_into_train_is_detected(self) -> None:
        manifest = self._manifest()
        folds = self._folds(manifest)
        for row in folds["01"]:
            if row["role"] == "test":
                row["role"] = "train"
                break
        result = verify_loso_folds(folds, manifest)
        self.assertFalse(result["loso_intact"])
        self.assertTrue(result["folds"]["01"]["held_out_signer_leakage"])

    def test_fold_missing_a_sample_is_detected(self) -> None:
        manifest = self._manifest()
        folds = self._folds(manifest)
        folds["02"] = folds["02"][:-1]
        result = verify_loso_folds(folds, manifest)
        self.assertFalse(result["loso_intact"])
        self.assertFalse(result["folds"]["02"]["covers_manifest_exactly"])


if __name__ == "__main__":
    unittest.main()
