"""TASK-009A: load one finalized virtual-glove sequence as masked ML tensors.

The dataset is read-only with respect to the frozen TASK-008 production tree.
It performs *lightweight* runtime contract validation -- array presence, shapes,
value ranges -- and never re-hashes source videos: TASK-008C already established
integrity over all 4,222 sequences with a full SHA-256 pass.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .contract import (
    BEND_CHANNELS_PER_HAND,
    CHAIN_ORDER,
    CONTRACT_VERSION,
    FINGER_ORDER,
    HAND_ORDER,
    LABEL_INDEX_RANGE,
    POSE_BEARING_CODES,
    QUATERNION_CHANNELS_PER_HAND,
    SOURCE_LAYOUT_VERSION,
    SPREAD_CHANNELS_PER_HAND,
    SPREAD_PAIRS,
    SequenceInputConfig,
    families_for,
)

REQUIRED_ARRAYS: tuple[str, ...] = (
    "frame_index",
    "timestamp_seconds",
    "bend_normalized",
    "bend_valid",
    "spread_normalized",
    "spread_valid",
    "imu_quaternion_wxyz",
    "palm_imu_valid",
    "tracking_state_code",
)
NUM_HANDS = len(HAND_ORDER)


class SequenceContractError(ValueError):
    """A stored sequence does not satisfy the frozen TASK-006/008 schema."""


@dataclass(frozen=True)
class SequenceRecord:
    """One row of the finalized index, with nothing inferred from the path."""

    sample_id: str
    sign_id: str
    label_ar: str
    label_index: int
    signer_id: str
    official_partition: str
    repetition_id: str
    source_frame_count: int
    sequence_length: int
    virtual_glove_relative_path: str


def load_index(path: str | Path) -> list[SequenceRecord]:
    """Read the finalized TASK-008C index.

    Labels, signer and class all come from authoritative manifest columns.
    Nothing is parsed out of a directory name or inferred from file ordering.
    """

    rows: list[SequenceRecord] = []
    seen: set[str] = set()
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sample_id = row["sample_id"]
            if sample_id in seen:
                raise SequenceContractError(f"duplicate sample_id in index: {sample_id}")
            seen.add(sample_id)
            label_index = int(row["label_index"])
            low, high = LABEL_INDEX_RANGE
            if not low <= label_index <= high:
                raise SequenceContractError(
                    f"{sample_id}: label_index {label_index} outside [{low}, {high}]"
                )
            rows.append(
                SequenceRecord(
                    sample_id=sample_id,
                    sign_id=row["sign_id"],
                    label_ar=row["label_ar"],
                    label_index=label_index,
                    signer_id=row["signer_id"],
                    official_partition=row["official_partition"],
                    repetition_id=row.get("repetition_id", ""),
                    source_frame_count=int(row["source_frame_count"]) if row.get("source_frame_count") else 0,
                    sequence_length=int(row["sequence_length"]) if row.get("sequence_length") else 0,
                    virtual_glove_relative_path=row["virtual_glove_relative_path"],
                )
            )
    if not rows:
        raise SequenceContractError(f"index {path} contains no rows")
    return rows


def verify_sensor_layout(path: str | Path) -> None:
    """Assert a stored ``sensor_layout.json`` still matches the frozen order.

    A silent channel reordering upstream would relabel every ML feature without
    changing a single shape, so this is checked against the layout file rather
    than trusted.
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("layout_version") != SOURCE_LAYOUT_VERSION:
        raise SequenceContractError(
            f"sensor layout version {payload.get('layout_version')!r} != {SOURCE_LAYOUT_VERSION!r}"
        )
    if tuple(payload.get("finger_order") or ()) != FINGER_ORDER:
        raise SequenceContractError("sensor layout finger_order differs from the frozen order")
    if tuple(payload.get("chain_joint_order") or ()) != CHAIN_ORDER:
        raise SequenceContractError("sensor layout chain_joint_order differs from the frozen order")

    sensors = list(payload.get("sensors") or ())
    bend = [s for s in sensors if s.get("role") == "bend"]
    spread = [s for s in sensors if s.get("role") == "spread"]
    expected_bend = [
        (finger, joint, [f, c])
        for f, finger in enumerate(FINGER_ORDER)
        for c, joint in enumerate(CHAIN_ORDER)
    ]
    actual_bend = [(s.get("finger"), s.get("joint"), list(s.get("array_index") or ())) for s in bend]
    if actual_bend != expected_bend:
        raise SequenceContractError("bend channel order or array_index differs from the frozen layout")
    expected_spread = [(list(pair), [i]) for i, pair in enumerate(SPREAD_PAIRS)]
    actual_spread = [(list(s.get("pair") or ()), list(s.get("array_index") or ())) for s in spread]
    if actual_spread != expected_spread:
        raise SequenceContractError("spread channel order or array_index differs from the frozen layout")


def load_sequence_arrays(path: str | Path) -> dict[str, np.ndarray]:
    """Read one ``virtual_glove.npz`` and validate the frozen schema.

    ``allow_pickle=False`` throughout: these files are data, never code.
    """

    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as data:
            missing = [name for name in REQUIRED_ARRAYS if name not in data.files]
            if missing:
                raise SequenceContractError(f"{path.name}: missing arrays {missing}")
            arrays = {name: np.asarray(data[name]) for name in REQUIRED_ARRAYS}
    except SequenceContractError:
        raise
    except Exception as error:  # noqa: BLE001 - a corrupt NPZ raises many types
        raise SequenceContractError(f"{path}: unreadable NPZ ({type(error).__name__}: {error})") from error

    frames = int(arrays["frame_index"].shape[0])
    if frames <= 0:
        raise SequenceContractError(f"{path.name}: zero-length sequence")
    expected = {
        "frame_index": (frames,),
        "timestamp_seconds": (frames,),
        "bend_normalized": (frames, NUM_HANDS, len(FINGER_ORDER), len(CHAIN_ORDER)),
        "bend_valid": (frames, NUM_HANDS, len(FINGER_ORDER), len(CHAIN_ORDER)),
        "spread_normalized": (frames, NUM_HANDS, SPREAD_CHANNELS_PER_HAND),
        "spread_valid": (frames, NUM_HANDS, SPREAD_CHANNELS_PER_HAND),
        "imu_quaternion_wxyz": (frames, NUM_HANDS, QUATERNION_CHANNELS_PER_HAND),
        "palm_imu_valid": (frames, NUM_HANDS),
        "tracking_state_code": (frames, NUM_HANDS),
    }
    for name, shape in expected.items():
        if tuple(arrays[name].shape) != shape:
            raise SequenceContractError(
                f"{path.name}: {name} shape {tuple(arrays[name].shape)} != {shape}"
            )
    for name in ("bend_valid", "spread_valid", "palm_imu_valid"):
        if arrays[name].dtype != np.bool_:
            raise SequenceContractError(f"{path.name}: {name} must be boolean, got {arrays[name].dtype}")
    return arrays


def _quaternion_conjugate(quaternion: np.ndarray) -> np.ndarray:
    out = np.array(quaternion, dtype=np.float64, copy=True)
    out[..., 1:] *= -1.0
    return out


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hamilton product of WXYZ quaternions, broadcasting over leading axes."""

    lw, lx, ly, lz = (left[..., i] for i in range(4))
    rw, rx, ry, rz = (right[..., i] for i in range(4))
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def _canonicalize_sign(quaternion: np.ndarray) -> np.ndarray:
    """Enforce w >= 0, preserving the rotation (q and -q are the same rotation)."""

    out = np.array(quaternion, dtype=np.float64, copy=True)
    flip = out[..., 0] < 0.0
    out[flip] *= -1.0
    return out


def relative_to_first_valid(
    quaternion: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, list[int | None]]:
    """Re-express each hand's palm orientation relative to its own first valid frame.

    ``q_rel(t) = conjugate(q_ref) * q(t)``, with ``q_ref`` the first valid palm
    quaternion of that physical hand in that sequence. This is strictly causal:
    a frame earlier than ``t_ref`` has no valid orientation by construction, so
    no earlier timestep is ever defined using later information. Hands are
    treated independently, so a missing LEFT does not shift RIGHT's reference.

    Returns the transformed quaternions (invalid entries left untouched) and the
    reference frame index per hand, ``None`` when a hand is never valid.
    """

    values = np.array(quaternion, dtype=np.float64, copy=True)
    references: list[int | None] = []
    for hand in range(values.shape[1]):
        valid_frames = np.flatnonzero(valid[:, hand])
        if valid_frames.size == 0:
            references.append(None)
            continue
        reference_frame = int(valid_frames[0])
        references.append(reference_frame)
        conjugate = _quaternion_conjugate(values[reference_frame, hand])
        selected = values[valid_frames, hand, :]
        values[valid_frames, hand, :] = _canonicalize_sign(
            _quaternion_multiply(np.broadcast_to(conjugate, selected.shape), selected)
        )
    return values, references


def build_feature_tensor(
    arrays: Mapping[str, np.ndarray], config: SequenceInputConfig
) -> dict[str, Any]:
    """Assemble one sequence into ``values`` plus its masks.

    Invalid channels are written as ``config.invalid_fill_value`` *and* marked
    False in ``feature_valid``. The fill is a mechanical tensor placeholder, not
    a measurement: a real 0.0 reading keeps ``feature_valid=True`` and stays
    distinguishable from it.
    """

    frames = int(arrays["frame_index"].shape[0])
    families = families_for(config.feature_set)

    bend = np.asarray(arrays["bend_normalized"], dtype=np.float64).reshape(
        frames, NUM_HANDS, BEND_CHANNELS_PER_HAND
    )
    bend_valid = np.asarray(arrays["bend_valid"], dtype=bool).reshape(
        frames, NUM_HANDS, BEND_CHANNELS_PER_HAND
    )
    spread = np.asarray(arrays["spread_normalized"], dtype=np.float64)
    spread_valid = np.asarray(arrays["spread_valid"], dtype=bool)
    palm_valid = np.asarray(arrays["palm_imu_valid"], dtype=bool)
    quaternion = np.asarray(arrays["imu_quaternion_wxyz"], dtype=np.float64)

    quaternion_reference: list[int | None] = [None] * NUM_HANDS
    if "quaternion" in families and config.quaternion_policy == "relative_first_valid":
        quaternion, quaternion_reference = relative_to_first_valid(quaternion, palm_valid)

    blocks: list[np.ndarray] = []
    valid_blocks: list[np.ndarray] = []
    for name in families:
        if name == "bend":
            blocks.append(bend)
            valid_blocks.append(bend_valid)
        elif name == "spread":
            blocks.append(spread)
            valid_blocks.append(spread_valid)
        else:
            blocks.append(quaternion)
            # One IMU package per hand: its validity applies to all four
            # components together, so it is broadcast rather than invented.
            valid_blocks.append(np.repeat(palm_valid[:, :, None], QUATERNION_CHANNELS_PER_HAND, axis=2))

    # [T, hand, channel] -> [T, hand * channel], hand-major.
    values = np.concatenate(blocks, axis=2).reshape(frames, -1)
    feature_valid = np.concatenate(valid_blocks, axis=2).reshape(frames, -1)

    # NaN never reaches a tensor: every non-finite slot must already be masked.
    non_finite_but_valid = int((~np.isfinite(values)) [feature_valid].sum())
    if non_finite_but_valid:
        raise SequenceContractError(
            f"{non_finite_but_valid} channels are marked valid but hold a non-finite value"
        )
    values = np.where(feature_valid, values, config.invalid_fill_value)

    state_code = np.asarray(arrays["tracking_state_code"], dtype=np.int64)
    hand_present = np.isin(state_code, np.asarray(POSE_BEARING_CODES, dtype=np.int64))

    return {
        "values": values.astype(np.float32),
        "feature_valid": feature_valid,
        "hand_present": hand_present,
        "tracking_state_code": state_code.astype(np.int16),
        "frame_index": np.asarray(arrays["frame_index"], dtype=np.int64),
        "timestamp_seconds": np.asarray(arrays["timestamp_seconds"], dtype=np.float64),
        "length": frames,
        "quaternion_reference_frame": quaternion_reference,
    }


class VirtualGloveSequenceDataset:
    """A ``torch.utils.data.Dataset``-compatible view of the frozen corpus.

    Deliberately not a subclass of ``torch.utils.data.Dataset``: the mapping
    protocol (``__len__``/``__getitem__``) is all a DataLoader needs, and this
    keeps the module importable and testable without torch.
    """

    def __init__(
        self,
        records: Sequence[SequenceRecord],
        run_root: str | Path,
        config: SequenceInputConfig | None = None,
    ) -> None:
        self.records = list(records)
        self.run_root = Path(run_root)
        self.config = config or SequenceInputConfig()
        self._cache: dict[int, dict[str, Any]] = {}
        if not self.records:
            raise SequenceContractError("dataset constructed with no records")

        if self.config.verify_layout != "none":
            targets = self.records if self.config.verify_layout == "all" else self.records[:1]
            for record in targets:
                verify_sensor_layout(self._path(record).parent / "sensor_layout.json")
        if self.config.preload:
            for position in range(len(self.records)):
                self._cache[position] = self._build(position)

    def _path(self, record: SequenceRecord) -> Path:
        return self.run_root / record.virtual_glove_relative_path

    def __len__(self) -> int:
        return len(self.records)

    def _build(self, position: int) -> dict[str, Any]:
        record = self.records[position]
        arrays = load_sequence_arrays(self._path(record))
        item = build_feature_tensor(arrays, self.config)
        if record.sequence_length and item["length"] != record.sequence_length:
            raise SequenceContractError(
                f"{record.sample_id}: stored length {item['length']} != index "
                f"{record.sequence_length}"
            )
        item.update(
            sample_id=record.sample_id,
            sign_id=record.sign_id,
            label_ar=record.label_ar,
            label_index=record.label_index,
            signer_id=record.signer_id,
            official_partition=record.official_partition,
        )
        return item

    def __getitem__(self, position: int) -> dict[str, Any]:
        if position in self._cache:
            return self._cache[position]
        return self._build(position)

    @property
    def feature_dim(self) -> int:
        return self.config.feature_dim

    def describe(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "sequences": len(self.records),
            "run_root": str(self.run_root),
            **self.config.to_dict(),
        }


__all__ = [
    "REQUIRED_ARRAYS", "SequenceContractError", "SequenceRecord", "load_index",
    "verify_sensor_layout", "load_sequence_arrays", "relative_to_first_valid",
    "build_feature_tensor", "VirtualGloveSequenceDataset",
]
