"""Serialization of the derived tracked representation.

Output layout per video (a NEW stage; raw input is never touched):

    <out_dir>/<sample_id>/wilor_tracked.npz
    <out_dir>/<sample_id>/wilor_tracked_meta.json

The NPZ holds plain numeric arrays indexed by ``[frame, track]`` with track
order ``("left", "right")``; no pickled Python objects are stored. Only the
inherently variable-length fields (flag lists, rejection reasons) are
serialized as JSON strings, matching the convention already used by
``pose/wilor/npz_io.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .schema import (
    CODE_TO_LABEL,
    CODE_TO_STATE,
    LABEL_CODES,
    STATE_CODES,
    TRACK_NAMES,
    TrackedFrame,
    TrackedHand,
    TrackedSequence,
    TrackState,
)

TRACKED_NPZ_NAME = "wilor_tracked.npz"
TRACKED_META_NAME = "wilor_tracked_meta.json"

_NAN3 = np.full(3, np.nan, dtype=np.float32)


def _hand_arrays(hand: TrackedHand) -> dict[str, np.ndarray]:
    def or_nan(value: Any, shape: tuple[int, ...]) -> np.ndarray:
        if value is None:
            return np.full(shape, np.nan, dtype=np.float32)
        return np.asarray(value, dtype=np.float32).reshape(shape)

    return {
        "landmarks_3d": or_nan(hand.landmarks_3d, (21, 3)),
        "hand_pose_rotmat": or_nan(hand.hand_pose_rotmat, (15, 3, 3)),
        "global_orient_rotmat": or_nan(hand.global_orient_rotmat, (3, 3)),
        "betas": or_nan(hand.betas, (10,)),
        "camera_translation": or_nan(hand.camera_translation, (3,)),
        "box_center_xy": or_nan(hand.box_center_xy, (2,)),
    }


def save_tracked_sequence(
    output_dir: str | Path, sequence: TrackedSequence, metrics: dict[str, Any] | None = None
) -> tuple[Path, Path]:
    """Write the tracked NPZ plus its JSON sidecar. Returns both paths."""

    directory = Path(output_dir) / sequence.sample_id
    directory.mkdir(parents=True, exist_ok=True)
    frames = sequence.frames
    n = len(frames)
    t = len(TRACK_NAMES)

    arrays: dict[str, np.ndarray] = {
        "frame_index": np.array([f.frame_index for f in frames], dtype=np.int32),
        "timestamp_seconds": np.array([f.timestamp_seconds for f in frames], dtype=np.float64),
        "state_code": np.zeros((n, t), dtype=np.int8),
        "raw_detection_index": np.full((n, t), -1, dtype=np.int16),
        "detector_label_code": np.full((n, t), -1, dtype=np.int8),
        "detector_confidence": np.full((n, t), np.nan, dtype=np.float32),
        "assignment_cost": np.full((n, t), np.nan, dtype=np.float32),
        "landmarks_3d": np.full((n, t, 21, 3), np.nan, dtype=np.float32),
        "hand_pose_rotmat": np.full((n, t, 15, 3, 3), np.nan, dtype=np.float32),
        "global_orient_rotmat": np.full((n, t, 3, 3), np.nan, dtype=np.float32),
        "betas": np.full((n, t, 10), np.nan, dtype=np.float32),
        "camera_translation": np.full((n, t, 3), np.nan, dtype=np.float32),
        "box_center_xy": np.full((n, t, 2), np.nan, dtype=np.float32),
        "box_size": np.full((n, t), np.nan, dtype=np.float32),
        "assignment_margin": np.full(n, np.nan, dtype=np.float32),
        "number_of_raw_detections": np.zeros(n, dtype=np.int16),
        "extra_detection_count": np.zeros(n, dtype=np.int16),
    }
    quality_flags_json = np.empty((n, t), dtype="U256")
    rejected_json = np.empty(n, dtype="U512")
    tracking_flags_json = np.empty(n, dtype="U256")

    for row, frame in enumerate(frames):
        arrays["assignment_margin"][row] = (
            np.nan if frame.assignment_margin is None else frame.assignment_margin
        )
        arrays["number_of_raw_detections"][row] = frame.number_of_raw_detections
        arrays["extra_detection_count"][row] = frame.extra_detection_count
        rejected_json[row] = json.dumps(
            {
                "indices": list(frame.rejected_detection_indices),
                "reasons": {str(k): v for k, v in frame.rejection_reasons.items()},
            }
        )
        tracking_flags_json[row] = json.dumps(list(frame.tracking_flags))
        for column, track in enumerate(TRACK_NAMES):
            hand = frame.hand(track)
            arrays["state_code"][row, column] = STATE_CODES[hand.state]
            if hand.raw_detection_index is not None:
                arrays["raw_detection_index"][row, column] = hand.raw_detection_index
            if hand.detector_label in LABEL_CODES:
                arrays["detector_label_code"][row, column] = LABEL_CODES[hand.detector_label]
            if hand.detector_confidence is not None:
                arrays["detector_confidence"][row, column] = hand.detector_confidence
            if hand.assignment_cost is not None:
                arrays["assignment_cost"][row, column] = hand.assignment_cost
            if hand.box_size is not None:
                arrays["box_size"][row, column] = hand.box_size
            for name, value in _hand_arrays(hand).items():
                arrays[name][row, column] = value
            quality_flags_json[row, column] = json.dumps(list(hand.quality_flags))

    arrays["quality_flags_json"] = quality_flags_json
    arrays["rejected_detections_json"] = rejected_json
    arrays["tracking_flags_json"] = tracking_flags_json

    npz_path = directory / TRACKED_NPZ_NAME
    np.savez_compressed(npz_path, **arrays)

    meta = {
        "schema_version": sequence.config.get("schema_version", "wilor_tracked_v1"),
        "stage": "tracked",
        "sample_id": sequence.sample_id,
        "track_order": list(TRACK_NAMES),
        "state_codes": {state.value: code for state, code in STATE_CODES.items()},
        "label_codes": LABEL_CODES,
        "total_frames": len(frames),
        "config": sequence.config,
        "source": sequence.source,
        "events": sequence.events,
        "metrics": metrics or {},
        "raw_immutability": "derived stage only; raw NPZ is opened read-only and never modified",
    }
    meta_path = directory / TRACKED_META_NAME
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return npz_path, meta_path


def load_tracked_sequence(directory: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a tracked NPZ into plain dictionaries (arrays + metadata)."""

    path = Path(directory)
    npz_path = path / TRACKED_NPZ_NAME if path.is_dir() else path
    meta_path = npz_path.parent / TRACKED_META_NAME
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    metadata = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    return arrays, metadata


def states_from_arrays(arrays: dict[str, Any]) -> list[dict[str, TrackState]]:
    """Convenience: decode per-frame track states from a loaded NPZ."""

    codes = np.asarray(arrays["state_code"])
    return [
        {track: CODE_TO_STATE[int(codes[row, column])] for column, track in enumerate(TRACK_NAMES)}
        for row in range(codes.shape[0])
    ]


def decode_label(code: int) -> str | None:
    return CODE_TO_LABEL.get(int(code))
