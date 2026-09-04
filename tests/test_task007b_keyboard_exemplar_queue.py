"""Synthetic and production-contract tests for TASK-007B."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from visualizer.catalog import write_catalog_csv
from visualizer.catalog.builder import CatalogBuildError, build_catalog
from visualizer.keyboard import Core28Keyboard
from visualizer.mapping import Core28Mapping, Core28Resolver, UnsupportedCharacterError
from visualizer.queue import PlaybackQueue, QueueState, UnsupportedTextError

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "datasets/manifests/karsl_core28_labels.csv"
CATALOG = ROOT / "visualizer/catalog/core28_exemplars.json"
EXPECTED = (
    ("0032", "ا"), ("0033", "ب"), ("0034", "ت"), ("0035", "ث"),
    ("0036", "ج"), ("0037", "ح"), ("0038", "خ"), ("0039", "د"),
    ("0040", "ذ"), ("0041", "ر"), ("0042", "ز"), ("0043", "س"),
    ("0044", "ش"), ("0045", "ص"), ("0046", "ض"), ("0047", "ط"),
    ("0048", "ظ"), ("0049", "ع"), ("0050", "غ"), ("0051", "ف"),
    ("0052", "ق"), ("0053", "ك"), ("0054", "ل"), ("0055", "م"),
    ("0056", "ن"), ("0057", "ه"), ("0058", "و"), ("0059", "ي"),
)
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
JOINTS = ("proximal", "middle", "distal")
PAIRS = (("thumb", "index"), ("index", "middle"), ("middle", "ring"), ("ring", "pinky"))


def _layout() -> dict:
    sensors = []
    for finger_index, finger in enumerate(FINGERS):
        for joint_index, joint in enumerate(JOINTS):
            sensors.append({
                "sensor_id": f"H_{finger}_{joint}", "array": "bend_angle_deg",
                "array_index": [finger_index, joint_index], "role": "bend", "finger": finger,
                "joint": joint, "pair": None, "display_marker": "H",
                "sensor_type": "hall_bend_angular", "logical_location": f"dorsal_{finger}_{joint}",
                "description": f"Hall bend sensor for {finger} {joint}.",
            })
    for index, pair in enumerate(PAIRS):
        sensors.append({
            "sensor_id": f"H_SPREAD_{index}", "array": "spread_angle_deg",
            "array_index": [index], "role": "spread", "finger": None, "joint": None,
            "pair": list(pair), "display_marker": "H", "sensor_type": "hall_spread_angular",
            "logical_location": f"interdigital_web_{pair[0]}_{pair[1]}",
            "description": f"Hall spread sensor for {pair[0]}-{pair[1]}.",
        })
    sensors.append({
        "sensor_id": "IMU_PALM", "array": "imu_quaternion_wxyz", "array_index": [],
        "role": "palm_orientation", "finger": None, "joint": None, "pair": None,
        "display_marker": "IMU", "sensor_type": "imu_orientation",
        "logical_location": "dorsal_palm_centre", "description": "Palm IMU.",
    })
    return {
        "layout_version": "ideal_virtual_glove_v1",
        "finger_order": list(FINGERS), "chain_joint_order": list(JOINTS),
        "per_hand_counts": {"bend_hall_sensors": 15, "spread_hall_sensors": 4,
                            "hall_sensors_total": 19, "imu_packages": 1,
                            "logical_sensing_packages": 20},
        "sensors": sensors,
    }


def _arrays(frames: int = 4) -> dict[str, np.ndarray]:
    bend_valid = np.ones((frames, 2, 5, 3), dtype=bool)
    spread_valid = np.ones((frames, 2, 4), dtype=bool)
    imu_valid = np.ones((frames, 2), dtype=bool)
    bend_deg = np.full((frames, 2, 5, 3), 36.0, dtype=np.float32)
    spread_deg = np.full((frames, 2, 4), 18.0, dtype=np.float32)
    rotation = np.broadcast_to(np.eye(3, dtype=np.float32), (frames, 2, 3, 3)).copy()
    quaternion = np.zeros((frames, 2, 4), dtype=np.float32)
    quaternion[..., 0] = 1.0
    return {
        "frame_index": np.arange(frames, dtype=np.int32),
        "timestamp_seconds": np.arange(frames, dtype=np.float64) / 30.0,
        "bend_angle_deg": bend_deg,
        "bend_normalized": (bend_deg / 180.0).astype(np.float32),
        "bend_valid": bend_valid,
        "spread_angle_deg": spread_deg,
        "spread_normalized": (spread_deg / 180.0).astype(np.float32),
        "spread_valid": spread_valid,
        "imu_rotation_matrix": rotation,
        "imu_quaternion_wxyz": quaternion,
        "palm_imu_valid": imu_valid,
        "tracking_state_code": np.ones((frames, 2), dtype=np.int32),
        "source_raw_detection_index": np.zeros((frames, 2), dtype=np.int32),
        "bend_adc_12bit": np.full((frames, 2, 5, 3), 819, dtype=np.int32),
        "spread_adc_12bit": np.full((frames, 2, 4), 410, dtype=np.int32),
    }


def _write_fixture(
    root: Path,
    *,
    bad: str | None = None,
    bad_class: str = "0032",
    frames: int = 4,
) -> tuple[Path, dict[str, str]]:
    with LABELS.open(newline="", encoding="utf-8") as handle:
        labels = list(csv.DictReader(handle))
    fields = [
        "sample_id", "sign_id", "label_ar", "label_index", "signer_id", "official_partition",
        "repetition_id", "source_relative_path", "source_sha256", "source_frame_count",
        "sequence_length", "virtual_glove_relative_path", "pose_status", "tracking_status",
        "kinematics_status", "virtual_glove_status", "bend_valid_fraction", "spread_valid_fraction",
        "imu_valid_fraction", "pose_bearing_hand_fraction", "manifest_sha256", "contract_version",
    ]
    rows: list[dict[str, str]] = []
    chosen: dict[str, str] = {}
    for label in labels:
        sign_id = label["sign_id"]
        sample_id = f"synthetic_{sign_id}"
        chosen[sign_id] = sample_id
        directory = root / "virtual_glove" / sample_id
        directory.mkdir(parents=True, exist_ok=True)
        arrays = _arrays(frames)
        layout = _layout()
        if sign_id == bad_class:
            if bad == "shape":
                arrays["bend_angle_deg"] = arrays["bend_angle_deg"][:, :, :, :2]
            elif bad == "wrong_normalization":
                arrays["bend_normalized"][0, 0, 0, 0] = 0.3
            elif bad == "out_of_range_normalized":
                arrays["bend_normalized"][0, 0, 0, 0] = 1.2
            elif bad == "illegal_nan":
                arrays["bend_angle_deg"][0, 0, 0, 0] = np.nan
            elif bad == "partial_spread_nan":
                arrays["spread_valid"][0, 0, 0] = False
                arrays["spread_angle_deg"][0, 0, 0] = np.nan
                arrays["spread_normalized"][0, 0, 0] = np.nan
                arrays["spread_adc_12bit"][0, 0, 0] = -1
            elif bad == "rotation_fault":
                arrays["imu_rotation_matrix"][0, 0, 0, 0] = 2.0
            elif bad == "quaternion_fault":
                arrays["imu_quaternion_wxyz"][0, 0, 0] = 2.0
            elif bad == "adc_mismatch":
                arrays["bend_adc_12bit"][0, 0, 0, 0] = 0
            elif bad == "missing_sensor":
                layout["sensors"].pop()
            elif bad == "duplicate_sensor":
                layout["sensors"][1]["sensor_id"] = layout["sensors"][0]["sensor_id"]
            elif bad == "missing_hall_marker":
                layout["sensors"][0]["display_marker"] = ""
            elif bad == "missing_imu_marker":
                layout["sensors"][-1]["display_marker"] = ""
        np.savez_compressed(directory / "virtual_glove.npz", **arrays)
        (directory / "sensor_layout.json").write_text(json.dumps(layout), encoding="utf-8")
        if bad == "provenance_mismatch" and sign_id == bad_class:
            (directory / "virtual_glove_meta.json").write_text(
                json.dumps({"sample_id": "not-the-sample"}), encoding="utf-8"
            )
        row = {
            "sample_id": sample_id, "sign_id": sign_id, "label_ar": label["label_ar"],
            "label_index": label["label_index"], "signer_id": "01", "official_partition": "train",
            "repetition_id": "rep001", "source_relative_path": f"01/train/{sign_id}.mp4",
            "source_sha256": "a" * 64, "source_frame_count": str(frames), "sequence_length": str(frames),
            "virtual_glove_relative_path": f"virtual_glove/{sample_id}/virtual_glove.npz",
            "pose_status": "POSE_DONE", "tracking_status": "TRACKING_DONE",
            "kinematics_status": "KINEMATICS_DONE", "virtual_glove_status": "VIRTUAL_GLOVE_DONE",
            "bend_valid_fraction": "1.0", "spread_valid_fraction": "1.0", "imu_valid_fraction": "1.0",
            "pose_bearing_hand_fraction": "1.0", "manifest_sha256": "b" * 64,
            "contract_version": "TASK-005-final-v2;TASK-006-ideal-virtual-glove-v1",
        }
        if bad == "frame_mismatch" and sign_id == bad_class:
            row["source_frame_count"] = str(frames + 1)
        rows.append(row)
    manifest = root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return manifest, chosen


class TestCore28MappingAndQueue(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.resolver = Core28Resolver(labels_path=LABELS, catalog_path=CATALOG)

    def test_all_28_authoritative_mappings_and_indices(self) -> None:
        self.assertEqual(tuple((label.sign_id, label.character) for label in self.resolver.mapping.labels), EXPECTED)
        self.assertEqual(len({label.character for label in self.resolver.mapping.labels}), 28)
        for index, (sign_id, character) in enumerate(EXPECTED):
            result = self.resolver.resolve_character(character)
            self.assertEqual((result.sign_id, result.label_index), (sign_id, index))

    def test_unsupported_character_is_not_mapped(self) -> None:
        with self.assertRaises(UnsupportedCharacterError) as raised:
            self.resolver.resolve_character("أ")
        self.assertIn("Unsupported Core-28 character: أ", str(raised.exception))
        with self.assertRaises(UnsupportedCharacterError):
            self.resolver.resolve_character("لا")

    def test_arabic_unicode_and_repeated_muhammad(self) -> None:
        queue = PlaybackQueue(self.resolver)
        queue.enqueue_text("محمد")
        self.assertEqual([item.character for item in queue.items], list("محمد"))
        self.assertEqual([item.sign_id for item in queue.items], ["0055", "0037", "0055", "0039"])
        self.assertEqual(queue.items[0].sample_id, queue.items[2].sample_id)
        self.assertEqual(self.resolver.resolve_character("\u0645").character, "م")

    def test_queue_order_advance_pop_clear_and_reset(self) -> None:
        queue = PlaybackQueue(self.resolver)
        queue.enqueue_text("محمد")
        self.assertEqual(queue.remaining, 4)
        self.assertEqual(queue.completed, 0)
        self.assertEqual(queue.peek().character, "م")
        self.assertEqual(queue.start().state, QueueState.PLAYING)
        self.assertEqual(queue.advance().character, "ح")
        self.assertEqual(queue.completed, 1)
        popped = queue.pop()
        self.assertEqual((popped.character, popped.state), ("ح", QueueState.COMPLETED))
        queue.clear()
        self.assertEqual((queue.remaining, queue.completed, queue.current), (0, 0, None))
        queue.enqueue_text("مد")
        queue.start()
        queue.advance()
        queue.reset()
        self.assertEqual(queue.remaining, 2)
        self.assertTrue(all(item.state == QueueState.PENDING for item in queue.items))

    def test_spaces_are_explicit_gaps_and_unsupported_text_is_atomic(self) -> None:
        queue = PlaybackQueue(self.resolver)
        queue.enqueue_text("م ح\nد")
        self.assertEqual([item.item_type for item in queue.items], ["sign", "gap", "sign", "gap", "sign"])
        self.assertEqual(queue.items[1].transition_policy, "neutral_gap")
        before = queue.items
        with self.assertRaises(UnsupportedTextError) as raised:
            queue.enqueue_text("مأد")
        self.assertEqual(queue.items, before)
        self.assertEqual(raised.exception.issues[0].position, 1)
        reported = queue.enqueue_text("مأد", unsupported_policy="report")
        self.assertEqual([item.character for item in reported], ["م", "د"])
        self.assertEqual(queue.last_unsupported[0].character, "أ")
        with self.assertRaises(UnsupportedTextError) as compound:
            queue.enqueue_text("لا")
        self.assertEqual(compound.exception.issues[0].character, "لا")

    def test_signer_and_seeded_random_modes_are_deterministic(self) -> None:
        first = self.resolver.resolve_character("م", mode="signer02")
        self.assertEqual(first.signer_id, "02")
        random_a = self.resolver.resolve_character("م", mode="random", rng_seed=17)
        random_b = self.resolver.resolve_character("م", mode="random", rng_seed=17)
        self.assertEqual(random_a.sample_id, random_b.sample_id)
        with self.assertRaises(ValueError):
            self.resolver.resolve_character("م", mode="random")

    def test_renderer_neutral_descriptor_and_production_samples(self) -> None:
        result = self.resolver.resolve_character("م")
        descriptor = result.sequence_descriptor
        self.assertEqual(descriptor.sample_id, result.sample_id)
        self.assertTrue(descriptor.virtual_glove_relative_path.startswith("virtual_glove/"))
        self.assertTrue(descriptor.absolute_path("virtual_glove").is_file())
        self.assertTrue(descriptor.absolute_path("kinematics").is_file())
        with (ROOT / "datasets/manifests/karsl_core28_virtual_glove.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            manifest_ids = {row["sample_id"] for row in csv.DictReader(handle)}
        self.assertTrue(all(entry.sample_id in manifest_ids for entry in self.resolver.catalog.entries))

    def test_cli_execution(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts/demo_task007b_queue.py"), "--text", "محمد", "--no-contract-demo"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("1. م -> 0055 ->", completed.stdout)
        self.assertIn("3. م -> 0055 ->", completed.stdout)
        self.assertIn("4. د -> 0039 ->", completed.stdout)


class TestKeyboardLayout(unittest.TestCase):
    def test_core28_keys_and_rtl_rows_preserve_mapping(self) -> None:
        keyboard = Core28Keyboard(Core28Mapping(LABELS))
        self.assertEqual(len(keyboard.keys), 28)
        self.assertEqual(sum(len(row) for row in keyboard.rtl_rows), 28)
        self.assertEqual({key.character for row in keyboard.rtl_rows for key in row}, set(label.character for label in keyboard.mapping.labels))
        self.assertTrue(keyboard.validate_text("محمد").is_valid)
        self.assertFalse(keyboard.validate_text("أ").is_valid)


class TestSyntheticCatalogBuilder(unittest.TestCase):
    def _build(self, bad: str | None = None) -> tuple[dict, Path]:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _ = _write_fixture(root, bad=bad)
            output = root / "catalog.json"
            payload = build_catalog(manifest_path=manifest, run_root=root, labels_path=LABELS, output_path=output)
            return payload, output

    def test_complete_catalog_and_deterministic_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _ = _write_fixture(root)
            first = root / "first.json"
            second = root / "second.json"
            build_catalog(manifest_path=manifest, run_root=root, labels_path=LABELS, output_path=first)
            build_catalog(manifest_path=manifest, run_root=root, labels_path=LABELS, output_path=second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            payload = json.loads(first.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["entries"]), 28)
            self.assertEqual({entry["sample_id"] for entry in payload["entries"]}, {f"synthetic_{sid}" for sid, _ in EXPECTED})

    def test_deterministic_compact_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _ = _write_fixture(root)
            payload = build_catalog(manifest_path=manifest, run_root=root, labels_path=LABELS)
            first, second = root / "first.csv", root / "second.csv"
            write_catalog_csv(first, payload)
            write_catalog_csv(second, payload)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with first.open(newline="", encoding="utf-8") as handle:
                self.assertEqual(sum(1 for _ in csv.DictReader(handle)), 28)

    def test_malformed_shapes_and_contract_values_are_rejected(self) -> None:
        for bad in (
            "shape", "wrong_normalization", "out_of_range_normalized", "illegal_nan",
            "rotation_fault", "quaternion_fault", "adc_mismatch", "provenance_mismatch", "frame_mismatch",
            "missing_sensor", "duplicate_sensor", "missing_hall_marker", "missing_imu_marker",
        ):
            with self.subTest(bad=bad), self.assertRaises(CatalogBuildError):
                self._build(bad)

    def test_partial_spread_nan_is_legal_and_not_all_or_nothing(self) -> None:
        payload, _ = self._build("partial_spread_nan")
        selected = next(entry for entry in payload["entries"] if entry["sign_id"] == "0032")
        self.assertLess(selected["metrics"]["spread_valid_fraction"], 1.0)
        self.assertEqual(selected["metrics"]["bend_valid_fraction"], 1.0)
        self.assertEqual(selected["metrics"]["imu_valid_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
