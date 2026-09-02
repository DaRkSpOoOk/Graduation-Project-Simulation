"""Fixed TASK-005 contract definitions and structural checks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

KINEMATICS_NPZ_NAME = "hand_kinematics.npz"
KINEMATICS_META_NAME = "hand_kinematics_meta.json"
TRACKED_NPZ_NAME = "wilor_tracked.npz"
TRACKED_META_NAME = "wilor_tracked_meta.json"

TRACK_NAMES = ("LEFT", "RIGHT")
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")

# Superseding TASK-005E1 contract.  ``valid_kinematics`` is a strict summary
# flag; it does not erase finite channels when a valid palm frame has only a
# geometrically undefined spread channel.  The validator enforces this
# channel-level (Model-B) rule in addition to the structural fields below.
VALIDITY_CONTRACT_VERSION = "TASK-005-final-v2-model-B"
VALIDITY_CONTRACT_NAME = "channel_level_validity"

REQUIRED_ARRAYS: dict[str, tuple[int, ... | str]] = {
    "frame_index": ("F",),
    "timestamp_seconds": ("F",),
    "tracking_state_code": ("F", 2),
    "source_raw_detection_index": ("F", 2),
    "valid_kinematics": ("F", 2),
    "valid_palm_frame": ("F", 2),
    "flexion_deg": ("F", 2, 5, 3),
    "adjacent_spread_deg": ("F", 2, 4),
    "palm_rotation_matrix": ("F", 2, 3, 3),
    "palm_quaternion_wxyz": ("F", 2, 4),
    "kinematic_flags_json": ("F", 2),
}


@dataclass(frozen=True, slots=True)
class SampleKinematics:
    sample_id: str
    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]
    path: Path


class ContractError(RuntimeError):
    """Raised for hard contract failures (missing files, unreadable arrays)."""


def list_sample_ids(run_dir: Path, npz_name: str) -> list[str]:
    sample_ids: list[str] = []
    for entry in sorted(run_dir.iterdir(), key=lambda p: p.name):
        if entry.is_dir() and (entry / npz_name).is_file():
            sample_ids.append(entry.name)
    return sample_ids


def load_kinematics_sample(run_dir: Path, sample_id: str) -> SampleKinematics:
    sample_dir = run_dir / sample_id
    npz_path = sample_dir / KINEMATICS_NPZ_NAME
    meta_path = sample_dir / KINEMATICS_META_NAME
    if not npz_path.is_file():
        raise ContractError(f"{sample_id}: missing {KINEMATICS_NPZ_NAME}")
    if not meta_path.is_file():
        raise ContractError(f"{sample_id}: missing {KINEMATICS_META_NAME}")

    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return SampleKinematics(sample_id=sample_id, arrays=arrays, metadata=metadata, path=sample_dir)


def _shape_matches(shape: tuple[int, ...], spec: tuple[int, ... | str], frame_count: int) -> bool:
    if len(shape) != len(spec):
        return False
    for got, expected in zip(shape, spec):
        if expected == "F":
            if got != frame_count:
                return False
        elif got != expected:
            return False
    return True


def _meta_track_order_ok(value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        # TASK-004/TASK-005A store the same fixed order in lowercase, while
        # the human-facing QA contract names the tracks in uppercase.
        return tuple(str(v).upper() for v in value) == TRACK_NAMES
    return False


def _meta_finger_order_ok(value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value) == FINGER_NAMES
    return False


def _meta_quaternion_ok(value: Any, order: Any = None) -> bool:
    if isinstance(value, (list, tuple)):
        return tuple(str(v).lower() for v in value) == ("w", "x", "y", "z")
    if isinstance(value, str):
        normalized = value.replace(" ", "").lower()
        if normalized in {"[w,x,y,z]", "wxyz", "(w,x,y,z)"}:
            return True
    # TASK-005A carries the order in a separate explicit field and uses the
    # convention field for normalization/sign policy prose.
    if isinstance(order, str):
        return order.replace(" ", "").lower() in {"wxyz", "[w,x,y,z]", "(w,x,y,z)"}
    return False


def validate_sample_contract(sample: SampleKinematics) -> dict[str, Any]:
    arrays = sample.arrays
    failures: list[str] = []

    missing = sorted(set(REQUIRED_ARRAYS) - set(arrays))
    if missing:
        failures.append(f"missing arrays: {missing}")
        return {"passed": False, "failures": failures, "frame_count": 0}

    frame_index = np.asarray(arrays["frame_index"])
    if frame_index.ndim != 1:
        failures.append(f"frame_index must be rank-1, got shape {frame_index.shape}")
        return {"passed": False, "failures": failures, "frame_count": 0}

    frame_count = int(frame_index.shape[0])
    for name, spec in REQUIRED_ARRAYS.items():
        shape = tuple(np.asarray(arrays[name]).shape)
        if not _shape_matches(shape, spec, frame_count):
            failures.append(
                f"{name} shape mismatch: expected {spec} with F={frame_count}, got {shape}"
            )

    for validity_name in ("valid_kinematics", "valid_palm_frame"):
        validity_dtype = np.asarray(arrays[validity_name]).dtype
        if validity_dtype != np.bool_:
            failures.append(f"{validity_name} must be bool, got {validity_dtype}")

    if frame_count > 1:
        frame_diff = np.diff(frame_index.astype(np.int64, copy=False))
        if np.any(frame_diff <= 0):
            if np.any(frame_diff == 0):
                failures.append("frame_index contains duplicate entries")
            if np.any(frame_diff < 0):
                failures.append("frame_index is not monotonically increasing")

    timestamps = np.asarray(arrays["timestamp_seconds"], dtype=np.float64)
    if not np.isfinite(timestamps).all():
        failures.append("timestamp_seconds contains non-finite values")
    if timestamps.size > 1 and np.any(np.diff(timestamps) < 0):
        failures.append("timestamp_seconds is not monotonically increasing")

    if not _meta_track_order_ok(sample.metadata.get("track_order")):
        failures.append(
            f"metadata track_order must be {list(TRACK_NAMES)}, got {sample.metadata.get('track_order')!r}"
        )
    if not _meta_finger_order_ok(sample.metadata.get("finger_order")):
        failures.append(
            f"metadata finger_order must be {list(FINGER_NAMES)}, got {sample.metadata.get('finger_order')!r}"
        )
    if not _meta_quaternion_ok(
        sample.metadata.get("quaternion_convention"),
        sample.metadata.get("quaternion_order"),
    ):
        failures.append(
            "metadata quaternion_convention must encode [w, x, y, z], "
            f"got convention={sample.metadata.get('quaternion_convention')!r}, "
            f"order={sample.metadata.get('quaternion_order')!r}"
        )

    return {
        "passed": not failures,
        "failures": failures,
        "frame_count": frame_count,
        "track_count": int(np.asarray(arrays["tracking_state_code"]).shape[1]),
    }
