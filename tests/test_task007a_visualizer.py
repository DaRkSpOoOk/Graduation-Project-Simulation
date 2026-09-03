"""Unit tests for the TASK-007A renderer-facing playback foundation."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from visualizer import ArtifactValidationError, PlaybackController, load_sequence
from visualizer.contract import (
    CHAIN_ORDER,
    FINGER_ORDER,
    SPREAD_PAIRS,
    TRACK_ORDER,
    validate_sensor_layout,
)
from visualizer.geometry import FINGER_CHAINS, sensor_marker_positions, sequence_bounds


PRODUCTION_RUN = Path("/home/hatim/graduation-project-runs/task008-core28-full")
PRODUCTION_MANIFEST = Path("datasets/manifests/karsl_core28.csv")
PRODUCTION_SAMPLE = "karsl_core28_s02_sign0043_train_rep023"


def _layout_payload() -> dict:
    sensors = []
    for finger_index, finger in enumerate(FINGER_ORDER):
        for chain_index, chain in enumerate(CHAIN_ORDER):
            sensors.append(
                {
                    "sensor_id": f"H_{finger.upper()}_{chain.upper()}",
                    "sensor_type": "hall_bend_angular",
                    "finger": finger,
                    "pair": None,
                    "joint": chain,
                    "role": "bend",
                    "logical_location": f"dorsal_{finger}_{chain}",
                    "display_marker": "H",
                    "description": "synthetic bend sensor",
                    "array": "bend_angle_deg",
                    "array_index": [finger_index, chain_index],
                }
            )
    for pair_index, pair in enumerate(SPREAD_PAIRS):
        sensors.append(
            {
                "sensor_id": f"H_SPREAD_{pair[0].upper()}_{pair[1].upper()}",
                "sensor_type": "hall_spread_angular",
                "finger": None,
                "pair": list(pair),
                "joint": None,
                "role": "spread",
                "logical_location": f"web_{pair[0]}_{pair[1]}",
                "display_marker": "H",
                "description": "synthetic spread sensor",
                "array": "spread_angle_deg",
                "array_index": [pair_index],
            }
        )
    sensors.append(
        {
            "sensor_id": "IMU_PALM",
            "sensor_type": "imu_package",
            "finger": None,
            "pair": None,
            "joint": None,
            "role": "palm_orientation",
            "logical_location": "dorsal_palm_centre",
            "display_marker": "IMU",
            "description": "synthetic palm IMU",
            "array": "imu_quaternion_wxyz",
            "array_index": [],
        }
    )
    return {
        "layout_version": "ideal_virtual_glove_v1",
        "track_order": ["LEFT", "RIGHT"],
        "finger_order": list(FINGER_ORDER),
        "chain_joint_order": list(CHAIN_ORDER),
        "spread_pairs": [list(pair) for pair in SPREAD_PAIRS],
        "display_markers": {"hall": "H", "imu": "IMU"},
        "sensors": sensors,
    }


def _write_synthetic_sequence(root: Path) -> tuple[Path, str]:
    sample_id = "synthetic_task007a"
    n_frames = 2
    frame_index = np.array([0, 1], dtype=np.int32)
    timestamps = np.array([0.0, 0.1], dtype=np.float64)
    landmarks = np.full((n_frames, 2, 21, 3), np.nan, dtype=np.float32)
    # Keep left/right visibly separated so screen position cannot be mistaken
    # for the identity source of the loader.
    landmarks[0, 0] = np.arange(63, dtype=np.float32).reshape(21, 3) - 30.0
    landmarks[1, 0] = landmarks[0, 0] + 0.1
    landmarks[1, 1] = landmarks[0, 0] + 10.0
    raw_landmarks = landmarks[0, 0].copy()

    raw_dir = root / "pose" / "raw" / sample_id
    tracking_dir = root / "tracking" / sample_id
    kinematics_dir = root / "kinematics" / sample_id
    glove_dir = root / "virtual_glove" / sample_id
    for directory in (raw_dir, tracking_dir, kinematics_dir, glove_dir):
        directory.mkdir(parents=True)

    np.savez_compressed(
        raw_dir / "wilor_raw.npz",
        frame_index=frame_index,
        timestamp_seconds=timestamps,
        hand_present=np.array([True, True]),
        handedness_label=np.array(["LEFT", "LEFT"]),
        handedness_confidence=np.ones(2, dtype=np.float32),
        detection_confidence=np.ones(2, dtype=np.float32),
        landmarks_3d=np.stack([raw_landmarks, raw_landmarks + 0.1]),
        vertices_keys=np.array(["0:0", "1:0"]),
        vertices=np.stack(
            [
                np.full((4, 3), 10.0, dtype=np.float32),
                np.full((4, 3), 20.0, dtype=np.float32),
            ]
        ),
    )
    state_codes = {"MISSING": 0, "OBSERVED": 1, "AMBIGUOUS": 2, "REJECTED_QUALITY": 3, "LIKELY_OCCLUDED": 4}
    tracking_state = np.array([[1, 0], [1, 1]], dtype=np.int8)
    raw_detection_index = np.array([[0, -1], [0, 0]], dtype=np.int16)
    np.savez_compressed(
        tracking_dir / "wilor_tracked.npz",
        frame_index=frame_index,
        timestamp_seconds=timestamps,
        state_code=tracking_state,
        raw_detection_index=raw_detection_index,
        landmarks_3d=landmarks,
    )
    (tracking_dir / "wilor_tracked_meta.json").write_text(
        json.dumps({"track_order": ["left", "right"], "state_codes": state_codes}), encoding="utf-8"
    )
    np.savez_compressed(
        kinematics_dir / "hand_kinematics.npz",
        frame_index=frame_index,
        timestamp_seconds=timestamps,
        tracking_state_code=tracking_state,
        source_raw_detection_index=raw_detection_index,
    )

    bend = np.zeros((n_frames, 2, 5, 3), dtype=np.float32)
    bend[0, 0, 0, 0] = 0.25
    bend_valid = np.ones_like(bend, dtype=bool)
    bend_valid[0, 1] = False
    spread = np.zeros((n_frames, 2, 4), dtype=np.float32)
    spread_valid = np.ones_like(spread, dtype=bool)
    spread_valid[0, 1] = False
    quaternion = np.zeros((n_frames, 2, 4), dtype=np.float32)
    quaternion[..., 0] = 1.0
    imu_valid = np.array([[True, False], [True, True]])
    np.savez_compressed(
        glove_dir / "virtual_glove.npz",
        frame_index=frame_index,
        timestamp_seconds=timestamps,
        bend_normalized=bend,
        spread_normalized=spread,
        bend_valid=bend_valid,
        spread_valid=spread_valid,
        imu_quaternion_wxyz=quaternion,
        palm_imu_valid=imu_valid,
        tracking_state_code=tracking_state.astype(np.int32),
        source_raw_detection_index=raw_detection_index.astype(np.int32),
    )
    (glove_dir / "virtual_glove_meta.json").write_text(
        json.dumps({"track_order": ["LEFT", "RIGHT"]}), encoding="utf-8"
    )
    (glove_dir / "sensor_layout.json").write_text(json.dumps(_layout_payload()), encoding="utf-8")

    manifest = root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "label_ar", "label_index", "signer_id"])
        writer.writeheader()
        writer.writerow({"sample_id": sample_id, "label_ar": "ا", "label_index": "0", "signer_id": "01"})
    return manifest, sample_id


class Task007AVisualizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.run_root = Path(self.temp.name) / "run"
        self.manifest, self.sample_id = _write_synthetic_sequence(self.run_root)
        self.sequence = load_sequence(self.run_root, self.sample_id, manifest_path=self.manifest)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_sensor_layout_has_exact_marker_counts(self) -> None:
        self.assertEqual(len(self.sequence.sensor_layout), 20)
        self.assertEqual(sum(sensor.role == "bend" for sensor in self.sequence.sensor_layout), 15)
        self.assertEqual(sum(sensor.role == "spread" for sensor in self.sequence.sensor_layout), 4)
        self.assertEqual(sum(sensor.display_marker == "IMU" for sensor in self.sequence.sensor_layout), 1)
        self.assertEqual(sum(sensor.display_marker == "H" for sensor in self.sequence.sensor_layout), 19)

    def test_mesh_key_association_uses_frame_and_raw_detection(self) -> None:
        self.assertTrue(np.all(self.sequence.frame_at(0).hand("LEFT").mesh_vertices == 10.0))
        self.assertIsNone(self.sequence.frame_at(0).hand("RIGHT").mesh_vertices)
        self.assertTrue(np.all(self.sequence.frame_at(1).hand("RIGHT").mesh_vertices == 20.0))

    def test_left_right_identity_is_preserved(self) -> None:
        self.assertEqual(self.sequence.frame_at(1).hand("LEFT").raw_detection_index, 0)
        self.assertEqual(self.sequence.frame_at(1).hand("RIGHT").raw_detection_index, 0)
        self.assertLess(self.sequence.frame_at(1).hand("LEFT").landmarks_3d[0, 0], self.sequence.frame_at(1).hand("RIGHT").landmarks_3d[0, 0])

    def test_sensor_readings_distinguish_valid_zero_from_invalid(self) -> None:
        left = {item.sensor.sensor_id: item for item in self.sequence.sensor_readings(0, "LEFT")}
        right = {item.sensor.sensor_id: item for item in self.sequence.sensor_readings(0, "RIGHT")}
        self.assertTrue(left["H_THUMB_PROXIMAL"].valid)
        self.assertEqual(left["H_THUMB_PROXIMAL"].value, 0.25)
        self.assertFalse(right["H_THUMB_PROXIMAL"].valid)
        self.assertIsNone(right["H_THUMB_PROXIMAL"].value)
        self.assertFalse(right["IMU_PALM"].valid)

    def test_sensor_marker_positions_cover_all_layout_entries(self) -> None:
        positions = sensor_marker_positions(self.sequence.frame_at(1).hand("LEFT").landmarks_3d, self.sequence.sensor_layout)
        self.assertEqual(set(positions), {sensor.sensor_id for sensor in self.sequence.sensor_layout})
        self.assertTrue(all(value is not None for value in positions.values()))
        self.assertTrue(np.allclose(positions["H_INDEX_PROXIMAL"], self.sequence.frame_at(1).hand("LEFT").landmarks_3d[5]))

    def test_missing_hand_has_no_invented_geometry(self) -> None:
        hand = self.sequence.frame_at(0).hand("RIGHT")
        self.assertFalse(hand.present)
        self.assertEqual(hand.state, "MISSING")
        readings = self.sequence.sensor_readings(0, "RIGHT")
        self.assertTrue(all(not reading.valid for reading in readings))

    def test_frame_and_timestamp_synchronization(self) -> None:
        self.assertEqual(self.sequence.frame_indices, (0, 1))
        self.assertEqual(self.sequence.timestamps, (0.0, 0.1))
        self.assertEqual(self.sequence.position_for_frame(1), 1)
        with self.assertRaises(KeyError):
            self.sequence.position_for_frame(99)

    def test_sequence_bounds_are_finite(self) -> None:
        lower, upper = sequence_bounds(self.sequence)
        self.assertTrue(np.isfinite(lower).all())
        self.assertTrue(np.isfinite(upper).all())
        self.assertTrue(np.all(upper > lower))

    def test_playback_scrub_and_timestamp_progress(self) -> None:
        controller = PlaybackController((0.0, 0.1, 0.5), (10, 11, 12))
        controller.seek_frame(11)
        self.assertEqual(controller.position, 1)
        controller.play(now=0.0)
        self.assertEqual(controller.tick(now=0.09), 1)
        self.assertEqual(controller.tick(now=0.11), 1)
        self.assertEqual(controller.tick(now=0.40), 2)
        self.assertFalse(controller.playing)

    def test_playback_restart_and_speed(self) -> None:
        controller = PlaybackController((0.0, 1.0), (0, 1), speed=2.0)
        controller.play(now=10.0)
        self.assertEqual(controller.tick(now=10.25), 0)
        controller.restart()
        self.assertEqual(controller.position, 0)
        self.assertFalse(controller.playing)
        with self.assertRaises(ValueError):
            controller.set_speed(0)

    def test_malformed_layout_is_rejected(self) -> None:
        payload = _layout_payload()
        payload["sensors"] = payload["sensors"][:-1]
        with self.assertRaises(ValueError):
            validate_sensor_layout(payload)

    def test_missing_required_artifact_is_rejected(self) -> None:
        (self.run_root / "virtual_glove" / self.sample_id / "virtual_glove.npz").unlink()
        with self.assertRaises(ArtifactValidationError):
            load_sequence(self.run_root, self.sample_id, manifest_path=self.manifest)

    def test_no_mesh_falls_back_to_tracked_landmarks(self) -> None:
        raw_path = self.run_root / "pose" / "raw" / self.sample_id / "wilor_raw.npz"
        with np.load(raw_path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files if key not in {"vertices", "vertices_keys"}}
        np.savez_compressed(raw_path, **arrays)
        sequence = load_sequence(self.run_root, self.sample_id, manifest_path=self.manifest)
        self.assertEqual(sequence.geometry_source, "tracked_landmarks_3d")
        self.assertIsNone(sequence.frame_at(0).hand("LEFT").mesh_vertices)
        self.assertTrue(sequence.frame_at(0).hand("LEFT").present)

    @unittest.skipUnless(PRODUCTION_RUN.is_dir(), "TASK-008 production run is not available")
    def test_real_production_artifact_loading(self) -> None:
        sequence = load_sequence(PRODUCTION_RUN, PRODUCTION_SAMPLE, manifest_path=PRODUCTION_MANIFEST)
        self.assertGreater(len(sequence), 0)
        self.assertEqual(sequence.geometry_source, "stored_mano_vertices+tracked_landmarks_3d")
        self.assertEqual(sequence.frame_at(0).hand("LEFT").mesh_vertices.shape[1:], (3,))
        self.assertEqual(sequence.frame_at(0).hand("LEFT").mesh_vertices.shape[0], 778)
        self.assertEqual(len(sequence.sensor_layout), 20)


if __name__ == "__main__":
    unittest.main()
