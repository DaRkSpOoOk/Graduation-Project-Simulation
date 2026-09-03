"""Adversarial tests for the TASK-009A frozen sequence-input contract.

Every fixture is synthetic and built in a temporary directory. Nothing here
reads the 4,222-sequence production run, and no model is constructed.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from recognition.data import (
    CONTRACT_VERSION,
    FEATURE_SETS,
    HAND_ORDER,
    NUM_CLASSES,
    BatchError,
    SequenceContractError,
    SequenceInputConfig,
    VirtualGloveSequenceDataset,
    build_feature_tensor,
    channel_index,
    channel_names,
    collate_sequences,
    concat_features_and_masks,
    contract_document,
    family_slice,
    feature_dimension,
    hand_slice,
    key_padding_mask,
    load_fold,
    load_index,
    load_sequence_arrays,
    relative_to_first_valid,
    verify_sensor_layout,
)
from recognition.data.contract import (
    CHAIN_ORDER,
    FINGER_ORDER,
    SPREAD_PAIRS,
)

ROOT = Path(__file__).resolve().parents[1]
FINGERS, CHAIN, SPREAD, HANDS = 5, 3, 4, 2
INDEX_FIELDS = (
    "sample_id,sign_id,label_ar,label_index,signer_id,official_partition,repetition_id,"
    "source_relative_path,source_sha256,source_frame_count,sequence_length,"
    "virtual_glove_relative_path,pose_status,tracking_status,kinematics_status,"
    "virtual_glove_status,bend_valid_fraction,spread_valid_fraction,imu_valid_fraction,"
    "pose_bearing_hand_fraction,manifest_sha256,contract_version"
)


def _layout() -> dict:
    sensors = []
    for f, finger in enumerate(FINGER_ORDER):
        for c, joint in enumerate(CHAIN_ORDER):
            sensors.append({"sensor_id": f"H_{finger}_{joint}", "array": "bend_angle_deg",
                            "array_index": [f, c], "role": "bend", "finger": finger,
                            "joint": joint, "pair": None, "display_marker": "H"})
    for i, pair in enumerate(SPREAD_PAIRS):
        sensors.append({"sensor_id": f"H_SPREAD_{i}", "array": "spread_angle_deg",
                        "array_index": [i], "role": "spread", "finger": None, "joint": None,
                        "pair": list(pair), "display_marker": "H"})
    sensors.append({"sensor_id": "IMU_PALM", "array": "imu_quaternion_wxyz", "array_index": [],
                    "role": "orientation", "finger": None, "joint": None, "pair": None,
                    "display_marker": "IMU"})
    return {"layout_version": "ideal_virtual_glove_v1", "finger_order": list(FINGER_ORDER),
            "chain_joint_order": list(CHAIN_ORDER), "sensors": sensors}


def _arrays(frames: int, *, hand_present: np.ndarray | None = None,
            spread_valid: np.ndarray | None = None,
            bend_value: float = 0.25, quaternion: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """A synthetic sequence obeying the frozen TASK-006 schema."""

    if hand_present is None:
        hand_present = np.ones((frames, HANDS), dtype=bool)
    bend_valid = np.repeat(hand_present[:, :, None], FINGERS * CHAIN, axis=2).reshape(
        frames, HANDS, FINGERS, CHAIN)
    if spread_valid is None:
        spread_valid = np.repeat(hand_present[:, :, None], SPREAD, axis=2)
    spread_valid = spread_valid & hand_present[:, :, None]
    if quaternion is None:
        quaternion = np.zeros((frames, HANDS, 4), dtype=np.float32)
        quaternion[..., 0] = 1.0
    bend = np.where(bend_valid, bend_value, np.nan).astype(np.float32)
    spread = np.where(spread_valid, 0.5, np.nan).astype(np.float32)
    return {
        "frame_index": np.arange(frames, dtype=np.int32),
        "timestamp_seconds": (np.arange(frames) / 30.0).astype(np.float64),
        "bend_normalized": bend, "bend_valid": bend_valid,
        "spread_normalized": spread, "spread_valid": spread_valid,
        "imu_quaternion_wxyz": np.asarray(quaternion, dtype=np.float32),
        "palm_imu_valid": hand_present.copy(),
        "tracking_state_code": np.where(hand_present, 1, 0).astype(np.int32),
    }


def _write_sample(root: Path, sample_id: str, arrays: dict[str, np.ndarray],
                  layout: dict | None = None) -> None:
    directory = root / "virtual_glove" / sample_id
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / "virtual_glove.npz", **arrays)
    (directory / "sensor_layout.json").write_text(json.dumps(layout or _layout()), encoding="utf-8")


def _index_row(sample_id: str, frames: int, *, signer="01", sign_id="0032",
               label_index=0, partition="train") -> str:
    path = f"virtual_glove/{sample_id}/virtual_glove.npz"
    return (f"{sample_id},{sign_id},x,{label_index},{signer},{partition},rep001,"
            f"{signer}/v.mp4,{'a' * 64},{frames},{frames},{path},POSE_DONE,TRACKING_DONE,"
            f"KINEMATICS_DONE,VIRTUAL_GLOVE_DONE,1.0,1.0,1.0,1.0,{'b' * 64},v1")


def _write_index(root: Path, rows: list[str]) -> Path:
    path = root / "index.csv"
    path.write_text(INDEX_FIELDS + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


class TestFeatureOrder(unittest.TestCase):
    def test_exact_feature_order_and_dimensions(self) -> None:
        self.assertEqual(feature_dimension("bend_only"), 30)
        self.assertEqual(feature_dimension("bend_spread"), 38)
        self.assertEqual(feature_dimension("full"), 46)
        names = channel_names("full")
        self.assertEqual(len(names), 46)
        self.assertEqual(names[0], "LEFT/bend/thumb/proximal")
        self.assertEqual(names[14], "LEFT/bend/pinky/distal")
        self.assertEqual(names[15], "LEFT/spread/thumb-index")
        self.assertEqual(names[18], "LEFT/spread/ring-pinky")
        self.assertEqual(names[19], "LEFT/palm_quaternion/w")
        self.assertEqual(names[22], "LEFT/palm_quaternion/z")
        self.assertEqual(names[23], "RIGHT/bend/thumb/proximal")
        self.assertEqual(names[45], "RIGHT/palm_quaternion/z")

    def test_every_channel_name_is_unique(self) -> None:
        for feature_set in FEATURE_SETS:
            names = channel_names(feature_set)
            self.assertEqual(len(names), len(set(names)), feature_set)

    def test_hand_and_family_slices_agree_with_names(self) -> None:
        for feature_set in FEATURE_SETS:
            names = channel_names(feature_set)
            for hand in HAND_ORDER:
                for name in names[hand_slice(feature_set, hand)]:
                    self.assertTrue(name.startswith(f"{hand}/"))
            for name in names[family_slice(feature_set, "RIGHT", "bend")]:
                self.assertIn("/bend/", name)

    def test_ablation_dimensions_are_subsets_in_the_same_order(self) -> None:
        full = channel_names("full")
        for feature_set in ("bend_only", "bend_spread"):
            for hand in HAND_ORDER:
                subset = channel_names(feature_set)[hand_slice(feature_set, hand)]
                reference = full[hand_slice("full", hand)][: len(subset)]
                self.assertEqual(subset, reference)

    def test_channel_index_rejects_a_channel_the_set_lacks(self) -> None:
        self.assertEqual(channel_index("full", "RIGHT/palm_quaternion/w"), 42)
        with self.assertRaises(KeyError):
            channel_index("bend_only", "LEFT/spread/thumb-index")


class TestLeftRightOrdering(unittest.TestCase):
    def test_left_right_never_swaps_when_only_right_is_present(self) -> None:
        frames = 4
        present = np.zeros((frames, HANDS), dtype=bool)
        present[:, 1] = True  # RIGHT only
        item = build_feature_tensor(_arrays(frames, hand_present=present), SequenceInputConfig())
        left = item["feature_valid"][:, hand_slice("full", "LEFT")]
        right = item["feature_valid"][:, hand_slice("full", "RIGHT")]
        self.assertFalse(left.any())
        self.assertTrue(right.all())
        self.assertFalse(item["hand_present"][:, 0].any())
        self.assertTrue(item["hand_present"][:, 1].all())

    def test_left_right_never_swaps_when_only_left_is_present(self) -> None:
        frames = 4
        present = np.zeros((frames, HANDS), dtype=bool)
        present[:, 0] = True
        item = build_feature_tensor(_arrays(frames, hand_present=present), SequenceInputConfig())
        self.assertTrue(item["feature_valid"][:, hand_slice("full", "LEFT")].all())
        self.assertFalse(item["feature_valid"][:, hand_slice("full", "RIGHT")].any())

    def test_hand_blocks_are_not_reordered_by_channel_count(self) -> None:
        # RIGHT carries more valid channels than LEFT; ordering must not react.
        frames = 3
        present = np.ones((frames, HANDS), dtype=bool)
        spread_valid = np.ones((frames, HANDS, SPREAD), dtype=bool)
        spread_valid[:, 0, :] = False  # LEFT loses all spread
        item = build_feature_tensor(
            _arrays(frames, hand_present=present, spread_valid=spread_valid), SequenceInputConfig())
        self.assertFalse(item["feature_valid"][:, family_slice("full", "LEFT", "spread")].any())
        self.assertTrue(item["feature_valid"][:, family_slice("full", "RIGHT", "spread")].all())


class TestMissingHands(unittest.TestCase):
    def test_a_single_missing_hand_does_not_drop_the_timestep(self) -> None:
        frames = 5
        present = np.ones((frames, HANDS), dtype=bool)
        present[2, 0] = False  # LEFT missing at t=2 only
        item = build_feature_tensor(_arrays(frames, hand_present=present), SequenceInputConfig())
        self.assertEqual(item["length"], frames)
        self.assertFalse(item["feature_valid"][2, hand_slice("full", "LEFT")].any())
        self.assertTrue(item["feature_valid"][2, hand_slice("full", "RIGHT")].all())

    def test_neither_hand_frame_is_handled_not_rejected(self) -> None:
        frames = 3
        present = np.ones((frames, HANDS), dtype=bool)
        present[1, :] = False
        item = build_feature_tensor(_arrays(frames, hand_present=present), SequenceInputConfig())
        self.assertEqual(item["length"], frames)
        self.assertFalse(item["feature_valid"][1].any())
        self.assertFalse(item["hand_present"][1].any())
        self.assertTrue(np.isfinite(item["values"]).all())

    def test_spread_invalid_while_bend_stays_valid(self) -> None:
        frames = 4
        spread_valid = np.ones((frames, HANDS, SPREAD), dtype=bool)
        spread_valid[:, :, 0] = False  # thumb-index ill-conditioned
        item = build_feature_tensor(_arrays(frames, spread_valid=spread_valid), SequenceInputConfig())
        for hand in HAND_ORDER:
            self.assertTrue(item["feature_valid"][:, family_slice("full", hand, "bend")].all())
            spread_block = item["feature_valid"][:, family_slice("full", hand, "spread")]
            self.assertFalse(spread_block[:, 0].any())
            self.assertTrue(spread_block[:, 1:].all())


class TestValidZeroVersusInvalidFill(unittest.TestCase):
    def test_real_zero_measurement_is_distinguishable_from_invalid_fill(self) -> None:
        frames = 3
        present = np.ones((frames, HANDS), dtype=bool)
        present[1, 0] = False
        arrays = _arrays(frames, hand_present=present, bend_value=0.0)
        item = build_feature_tensor(arrays, SequenceInputConfig())
        values, valid = item["values"], item["feature_valid"]
        bend_left = family_slice("full", "LEFT", "bend")
        # t=0: a genuine 0.0 reading, still marked valid.
        self.assertTrue(np.all(values[0, bend_left] == 0.0))
        self.assertTrue(np.all(valid[0, bend_left]))
        # t=1: no measurement at all -- same stored value, mask says otherwise.
        self.assertTrue(np.all(values[1, bend_left] == 0.0))
        self.assertFalse(np.any(valid[1, bend_left]))

    def test_nan_is_never_silently_converted_without_a_mask(self) -> None:
        frames = 3
        arrays = _arrays(frames)
        arrays["bend_normalized"][1, 0, 0, 0] = np.nan  # NaN while mask says valid
        with self.assertRaises(SequenceContractError):
            build_feature_tensor(arrays, SequenceInputConfig())

    def test_invalid_channels_never_leave_a_nan_in_the_tensor(self) -> None:
        frames = 4
        present = np.ones((frames, HANDS), dtype=bool)
        present[0, 1] = False
        item = build_feature_tensor(_arrays(frames, hand_present=present), SequenceInputConfig())
        self.assertTrue(np.isfinite(item["values"]).all())


class TestQuaternionPolicy(unittest.TestCase):
    def _rotation_sequence(self, frames: int) -> np.ndarray:
        angles = np.linspace(0.0, 1.2, frames)
        quaternion = np.zeros((frames, HANDS, 4), dtype=np.float32)
        for hand in range(HANDS):
            offset = 0.3 * hand
            quaternion[:, hand, 0] = np.cos((angles + offset) / 2.0)
            quaternion[:, hand, 3] = np.sin((angles + offset) / 2.0)
        return quaternion

    def test_absolute_policy_copies_the_stored_quaternion_verbatim(self) -> None:
        frames = 5
        quaternion = self._rotation_sequence(frames)
        item = build_feature_tensor(_arrays(frames, quaternion=quaternion), SequenceInputConfig())
        for hand_position, hand in enumerate(HAND_ORDER):
            block = item["values"][:, family_slice("full", hand, "quaternion")]
            np.testing.assert_allclose(block, quaternion[:, hand_position], atol=1e-6)

    def test_wxyz_ordering_is_preserved(self) -> None:
        frames = 2
        quaternion = np.zeros((frames, HANDS, 4), dtype=np.float32)
        quaternion[:, :, 0], quaternion[:, :, 1] = 0.6, 0.8  # w, x
        item = build_feature_tensor(_arrays(frames, quaternion=quaternion), SequenceInputConfig())
        block = family_slice("full", "LEFT", "quaternion")
        self.assertAlmostEqual(float(item["values"][0, block.start + 0]), 0.6, places=6)
        self.assertAlmostEqual(float(item["values"][0, block.start + 1]), 0.8, places=6)
        self.assertAlmostEqual(float(item["values"][0, block.start + 2]), 0.0, places=6)

    def test_relative_policy_makes_the_reference_frame_the_identity(self) -> None:
        frames = 6
        quaternion = self._rotation_sequence(frames)
        config = SequenceInputConfig(quaternion_policy="relative_first_valid")
        item = build_feature_tensor(_arrays(frames, quaternion=quaternion), config)
        for hand in HAND_ORDER:
            first = item["values"][0, family_slice("full", hand, "quaternion")]
            np.testing.assert_allclose(first, [1.0, 0.0, 0.0, 0.0], atol=1e-6)

    def test_relative_policy_preserves_the_rotation_between_frames(self) -> None:
        frames = 6
        quaternion = self._rotation_sequence(frames).astype(np.float64)
        relative, references = relative_to_first_valid(quaternion, np.ones((frames, HANDS), dtype=bool))
        self.assertEqual(references, [0, 0])
        for hand in range(HANDS):
            for t in range(frames):
                # angle(q_ref, q_t) must equal angle(identity, q_rel_t)
                dot_absolute = abs(float(np.dot(quaternion[0, hand], quaternion[t, hand])))
                dot_relative = abs(float(relative[t, hand, 0]))
                self.assertAlmostEqual(dot_absolute, dot_relative, places=6)

    def test_relative_reference_is_the_first_valid_frame_of_that_hand(self) -> None:
        frames = 6
        present = np.ones((frames, HANDS), dtype=bool)
        present[:2, 0] = False  # LEFT only becomes valid at t=2
        quaternion = self._rotation_sequence(frames).astype(np.float64)
        relative, references = relative_to_first_valid(quaternion, present)
        self.assertEqual(references, [2, 0])
        np.testing.assert_allclose(relative[2, 0], [1.0, 0.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(relative[0, 1], [1.0, 0.0, 0.0, 0.0], atol=1e-6)
        # Frames before the reference are untouched and remain masked out.
        np.testing.assert_allclose(relative[0, 0], quaternion[0, 0], atol=1e-12)

    def test_relative_policy_on_a_never_valid_hand_has_no_reference(self) -> None:
        frames = 4
        present = np.ones((frames, HANDS), dtype=bool)
        present[:, 0] = False
        config = SequenceInputConfig(quaternion_policy="relative_first_valid")
        item = build_feature_tensor(_arrays(frames, hand_present=present, quaternion=None), config)
        self.assertEqual(item["quaternion_reference_frame"][0], None)
        self.assertFalse(item["feature_valid"][:, family_slice("full", "LEFT", "quaternion")].any())

    def test_relative_output_keeps_the_w_non_negative_convention(self) -> None:
        frames = 8
        angles = np.linspace(0.0, 3.0, frames)
        quaternion = np.zeros((frames, HANDS, 4))
        quaternion[:, :, 0] = np.cos(angles[:, None] / 2.0)
        quaternion[:, :, 3] = np.sin(angles[:, None] / 2.0)
        relative, _ = relative_to_first_valid(quaternion, np.ones((frames, HANDS), dtype=bool))
        self.assertTrue((relative[..., 0] >= 0.0).all())

    def test_quaternion_policy_is_rejected_for_a_set_without_quaternions(self) -> None:
        with self.assertRaises(ValueError):
            SequenceInputConfig(feature_set="bend_only", quaternion_policy="relative_first_valid")


class TestSchemaRejection(unittest.TestCase):
    def test_malformed_npz_is_rejected_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.npz"
            path.write_bytes(b"PK\x03\x04 this is not a valid npz")
            with self.assertRaises(SequenceContractError) as caught:
                load_sequence_arrays(path)
            self.assertIn("unreadable NPZ", str(caught.exception))

    def test_missing_array_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            arrays = _arrays(3)
            arrays.pop("spread_valid")
            path = Path(tmp) / "partial.npz"
            np.savez_compressed(path, **arrays)
            with self.assertRaises(SequenceContractError) as caught:
                load_sequence_arrays(path)
            self.assertIn("missing arrays", str(caught.exception))

    def test_wrong_shape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            arrays = _arrays(3)
            arrays["spread_normalized"] = arrays["spread_normalized"][:, :, :3]
            arrays["spread_valid"] = arrays["spread_valid"][:, :, :3]
            path = Path(tmp) / "wrong.npz"
            np.savez_compressed(path, **arrays)
            with self.assertRaises(SequenceContractError):
                load_sequence_arrays(path)

    def test_non_boolean_mask_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            arrays = _arrays(3)
            arrays["bend_valid"] = arrays["bend_valid"].astype(np.int8)
            path = Path(tmp) / "mask.npz"
            np.savez_compressed(path, **arrays)
            with self.assertRaises(SequenceContractError):
                load_sequence_arrays(path)

    def test_sensor_layout_reorder_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp) / "good.json"
            good.write_text(json.dumps(_layout()), encoding="utf-8")
            verify_sensor_layout(good)  # does not raise

            reordered = _layout()
            reordered["sensors"][0], reordered["sensors"][1] = (
                reordered["sensors"][1], reordered["sensors"][0])
            bad = Path(tmp) / "bad.json"
            bad.write_text(json.dumps(reordered), encoding="utf-8")
            with self.assertRaises(SequenceContractError):
                verify_sensor_layout(bad)

    def test_spread_pair_reorder_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            layout = _layout()
            spread = [s for s in layout["sensors"] if s["role"] == "spread"]
            spread[0]["pair"], spread[1]["pair"] = spread[1]["pair"], spread[0]["pair"]
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps(layout), encoding="utf-8")
            with self.assertRaises(SequenceContractError):
                verify_sensor_layout(path)

    def test_duplicate_sample_id_in_index_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_index(Path(tmp), [_index_row("a", 5), _index_row("a", 5)])
            with self.assertRaises(SequenceContractError):
                load_index(path)

    def test_label_index_outside_the_frozen_range_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_index(Path(tmp), [_index_row("a", 5, label_index=NUM_CLASSES)])
            with self.assertRaises(SequenceContractError):
                load_index(path)

    def test_index_length_disagreement_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_sample(root, "a", _arrays(5))
            path = _write_index(root, [_index_row("a", 9)])  # index claims 9, file holds 5
            dataset = VirtualGloveSequenceDataset(load_index(path), root, SequenceInputConfig())
            with self.assertRaises(SequenceContractError):
                dataset[0]


class TestVariableLengthBatching(unittest.TestCase):
    def _items(self, lengths: list[int], config: SequenceInputConfig | None = None):
        config = config or SequenceInputConfig()
        items = []
        for position, length in enumerate(lengths):
            item = build_feature_tensor(_arrays(length), config)
            item.update(sample_id=f"s{position}", sign_id="0032", label_ar="x",
                        label_index=position % NUM_CLASSES, signer_id="01",
                        official_partition="train")
            items.append(item)
        return items, config

    def test_extreme_lengths_nine_and_seventy(self) -> None:
        items, config = self._items([9, 70])
        batch = collate_sequences(items, config)
        self.assertEqual(tuple(batch["values"].shape), (2, 70, 46))
        self.assertEqual(batch["lengths"].tolist(), [9, 70])
        self.assertEqual(int(batch["frame_valid"][0].sum()), 9)
        self.assertEqual(int(batch["frame_valid"][1].sum()), 70)

    def test_padding_mask_matches_lengths_exactly(self) -> None:
        items, config = self._items([12, 5, 30, 17])
        batch = collate_sequences(items, config)
        self.assertTrue(torch.equal(batch["frame_valid"].sum(dim=1), batch["lengths"]))
        self.assertTrue(torch.equal(key_padding_mask(batch), ~batch["frame_valid"]))

    def test_padding_never_alters_real_frames(self) -> None:
        items, config = self._items([6, 20])
        alone = collate_sequences([items[0]], config)
        together = collate_sequences(items, config)
        self.assertTrue(torch.equal(alone["values"][0, :6], together["values"][0, :6]))
        self.assertTrue(torch.equal(alone["feature_valid"][0, :6], together["feature_valid"][0, :6]))

    def test_padding_is_masked_in_every_mask_simultaneously(self) -> None:
        items, config = self._items([4, 11])
        batch = collate_sequences(items, config)
        padded = ~batch["frame_valid"]
        self.assertFalse(batch["feature_valid"][padded].any())
        self.assertFalse(batch["hand_present"][padded].any())
        self.assertTrue(torch.all(batch["values"][padded] == 0.0))

    def test_three_states_stay_distinct_in_a_batch(self) -> None:
        frames = 4
        present = np.ones((frames, HANDS), dtype=bool)
        present[2, 0] = False
        config = SequenceInputConfig()
        item = build_feature_tensor(_arrays(frames, hand_present=present, bend_value=0.0), config)
        item.update(sample_id="a", sign_id="0032", label_ar="x", label_index=0,
                    signer_id="01", official_partition="train")
        short = build_feature_tensor(_arrays(2, bend_value=0.0), config)
        short.update(sample_id="b", sign_id="0032", label_ar="x", label_index=0,
                     signer_id="01", official_partition="train")
        batch = collate_sequences([item, short], config)
        values, valid, frame_valid = batch["values"], batch["feature_valid"], batch["frame_valid"]
        left = hand_slice("full", "LEFT")
        self.assertTrue(bool(valid[0, 0, left].all()) and bool(frame_valid[0, 0]))       # real 0.0
        self.assertFalse(bool(valid[0, 2, left].any())); self.assertTrue(bool(frame_valid[0, 2]))  # invalid
        self.assertFalse(bool(frame_valid[1, 3]))                                          # padding
        self.assertEqual(float(values[0, 0, left.start]), 0.0)
        self.assertEqual(float(values[0, 2, left.start]), 0.0)
        self.assertEqual(float(values[1, 3, left.start]), 0.0)

    def test_pack_padded_sequence_accepts_the_batch(self) -> None:
        items, config = self._items([9, 3, 15])
        batch = collate_sequences(items, config)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            batch["values"], batch["lengths"], batch_first=True, enforce_sorted=False)
        restored, restored_lengths = torch.nn.utils.rnn.pad_packed_sequence(packed, batch_first=True)
        self.assertTrue(torch.equal(restored_lengths, batch["lengths"]))
        self.assertTrue(torch.equal(restored, batch["values"]))

    def test_metadata_stays_aligned_with_tensor_rows(self) -> None:
        items, config = self._items([4, 8, 6])
        items[1]["sample_id"] = "middle"
        items[1]["label_index"] = 17
        items[1]["signer_id"] = "03"
        batch = collate_sequences(items, config)
        self.assertEqual(batch["sample_ids"][1], "middle")
        self.assertEqual(int(batch["labels"][1]), 17)
        self.assertEqual(batch["signer_ids"][1], "03")
        self.assertEqual(int(batch["lengths"][1]), 8)

    def test_dimension_mismatch_between_items_is_rejected(self) -> None:
        full_items, full_config = self._items([5])
        bend_config = SequenceInputConfig(feature_set="bend_only")
        bend_item = build_feature_tensor(_arrays(5), bend_config)
        bend_item.update(sample_id="b", sign_id="0032", label_ar="x", label_index=0,
                         signer_id="01", official_partition="train")
        with self.assertRaises(BatchError):
            collate_sequences(full_items + [bend_item], full_config)

    def test_ablation_batches_have_the_declared_dimension(self) -> None:
        for feature_set, expected in (("bend_only", 30), ("bend_spread", 38), ("full", 46)):
            config = SequenceInputConfig(feature_set=feature_set)
            item = build_feature_tensor(_arrays(7), config)
            item.update(sample_id="a", sign_id="0032", label_ar="x", label_index=0,
                        signer_id="01", official_partition="train")
            batch = collate_sequences([item], config)
            self.assertEqual(tuple(batch["values"].shape), (1, 7, expected), feature_set)
            self.assertEqual(tuple(batch["feature_valid"].shape), (1, 7, expected), feature_set)

    def test_concat_helper_doubles_the_channel_axis(self) -> None:
        items, config = self._items([5, 9])
        batch = collate_sequences(items, config)
        joined = concat_features_and_masks(batch)
        self.assertEqual(tuple(joined.shape), (2, 9, 92))
        self.assertTrue(torch.equal(joined[..., :46], batch["values"]))

    def test_empty_batch_is_rejected(self) -> None:
        with self.assertRaises(BatchError):
            collate_sequences([], SequenceInputConfig())


class TestLosoContract(unittest.TestCase):
    def _corpus(self, tmp: Path):
        rows, split_rows = [], []
        for signer in ("01", "02", "03"):
            for label_index in range(2):
                sample_id = f"s{signer}_c{label_index}"
                rows.append(_index_row(sample_id, 5, signer=signer,
                                       sign_id=f"{32 + label_index:04d}", label_index=label_index))
                split_rows.append((sample_id, signer))
        index_path = _write_index(tmp, rows)
        return load_index(index_path), split_rows

    def _write_split(self, tmp: Path, held_out: str, split_rows, override=None) -> None:
        lines = ["sample_id,signer_id,fold,role"]
        for sample_id, signer in split_rows:
            role = "test" if signer == held_out else "train"
            if override and sample_id in override:
                role = override[sample_id]
            lines.append(f"{sample_id},{signer},S{held_out},{role}")
        (tmp / f"karsl_core28_loso_s{held_out}.csv").write_text("\n".join(lines) + "\n",
                                                                encoding="utf-8")

    def test_fold_loads_and_isolates_the_held_out_signer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records, split_rows = self._corpus(root)
            self._write_split(root, "02", split_rows)
            fold = load_fold(root, "02", records)
            self.assertEqual(fold.signers("test"), {"02"})
            self.assertNotIn("02", fold.signers("train"))

    def test_held_out_signer_leaking_into_train_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records, split_rows = self._corpus(root)
            self._write_split(root, "02", split_rows, override={"s02_c0": "train"})
            with self.assertRaises(SequenceContractError) as caught:
                load_fold(root, "02", records)
            self.assertIn("leaks into train", str(caught.exception))

    def test_a_sample_in_two_roles_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records, split_rows = self._corpus(root)
            self._write_split(root, "01", split_rows)
            path = root / "karsl_core28_loso_s01.csv"
            path.write_text(path.read_text(encoding="utf-8") + "s02_c0,02,S01,validation\n",
                            encoding="utf-8")
            with self.assertRaises(SequenceContractError):
                load_fold(root, "01", records)

    def test_incomplete_coverage_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records, split_rows = self._corpus(root)
            self._write_split(root, "01", split_rows[:-1])
            with self.assertRaises(SequenceContractError):
                load_fold(root, "01", records)

    def test_fold_contents_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records, split_rows = self._corpus(root)
            self._write_split(root, "03", split_rows)
            first = load_fold(root, "03", records)
            second = load_fold(root, "03", list(reversed(records)))
            self.assertEqual([r.sample_id for r in first.roles["train"]],
                             [r.sample_id for r in second.roles["train"]])


class TestDatasetEndToEnd(unittest.TestCase):
    def test_dataset_round_trip_on_a_synthetic_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lengths = {"a": 9, "b": 70, "c": 23}
            for position, (sample_id, length) in enumerate(lengths.items()):
                _write_sample(root, sample_id, _arrays(length))
            index = _write_index(root, [_index_row(sid, n, label_index=i)
                                        for i, (sid, n) in enumerate(lengths.items())])
            records = load_index(index)
            config = SequenceInputConfig(verify_layout="all")
            dataset = VirtualGloveSequenceDataset(records, root, config)
            self.assertEqual(len(dataset), 3)
            batch = collate_sequences([dataset[i] for i in range(3)], config)
            self.assertEqual(tuple(batch["values"].shape), (3, 70, 46))
            self.assertEqual(sorted(batch["lengths"].tolist()), [9, 23, 70])
            self.assertEqual(batch["labels"].tolist(), [0, 1, 2])

    def test_preload_and_lazy_paths_agree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_sample(root, "a", _arrays(11))
            index = _write_index(root, [_index_row("a", 11)])
            records = load_index(index)
            lazy = VirtualGloveSequenceDataset(records, root, SequenceInputConfig())
            eager = VirtualGloveSequenceDataset(records, root, SequenceInputConfig(preload=True))
            np.testing.assert_array_equal(lazy[0]["values"], eager[0]["values"])
            np.testing.assert_array_equal(lazy[0]["feature_valid"], eager[0]["feature_valid"])

    def test_layout_verification_rejects_a_reordered_production_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            layout = _layout()
            layout["sensors"][3], layout["sensors"][4] = layout["sensors"][4], layout["sensors"][3]
            _write_sample(root, "a", _arrays(5), layout=layout)
            index = _write_index(root, [_index_row("a", 5)])
            with self.assertRaises(SequenceContractError):
                VirtualGloveSequenceDataset(load_index(index), root, SequenceInputConfig())


class TestFrozenConfiguration(unittest.TestCase):
    def test_committed_config_matches_the_code(self) -> None:
        path = ROOT / "configs/recognition/task009a_sequence_input_v1.json"
        committed = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(committed, contract_document())

    def test_contract_version_is_recorded_everywhere(self) -> None:
        document = contract_document()
        self.assertEqual(document["contract_version"], CONTRACT_VERSION)
        self.assertEqual(document["config_defaults"]["contract_version"], CONTRACT_VERSION)

    def test_config_rejects_unknown_options(self) -> None:
        with self.assertRaises(ValueError):
            SequenceInputConfig(feature_set="everything")
        with self.assertRaises(ValueError):
            SequenceInputConfig(quaternion_policy="euler")
        with self.assertRaises(ValueError):
            SequenceInputConfig(verify_layout="sometimes")


if __name__ == "__main__":
    unittest.main()
