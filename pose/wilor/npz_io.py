"""Immutable per-video raw-output serialization (Task 4: raw output
immutability).

Layout: one NPZ per video at
``<output_dir>/<sample_id>/wilor_raw.npz`` containing a long-format table
(one row per detected hand per frame; frames with zero hands still get one
row with ``hand_present=False``) plus a ragged store of mesh vertices when
the full pipeline (with MANO) was used.

This module never interpolates, smooths, or "fixes" failed frames -- a
failed/undetected frame is stored explicitly (``hand_present=False``,
``quality_flags`` populated) rather than dropped or imputed.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np

from pose.common.schema import HandPoseFrame

_LANDMARK3D_ROWS = 21  # MANO->OpenPose joint count (see mano_wrapper.py mano_to_openpose)
_SCALAR_FIELDS = (
    "frame_index",
    "timestamp_seconds",
    "hand_present",
    "handedness_label",
    "handedness_confidence",
    "detection_confidence",
)


def _pack_landmarks_3d(frame: HandPoseFrame) -> np.ndarray:
    arr = np.full((_LANDMARK3D_ROWS, 3), np.nan, dtype=np.float32)
    for i, lm in enumerate(frame.landmarks_3d[:_LANDMARK3D_ROWS]):
        arr[i] = (lm.x, lm.y, lm.z)
    return arr


def save_raw_video_output(
    output_dir: Path,
    sample_id: str,
    frames: list[HandPoseFrame],
    vertices_by_hand: dict[tuple[int, int], np.ndarray] | None = None,
    run_metadata: dict[str, Any] | None = None,
) -> Path:
    video_dir = output_dir / sample_id
    video_dir.mkdir(parents=True, exist_ok=True)
    npz_path = video_dir / "wilor_raw.npz"

    n = len(frames)
    frame_index = np.array([f.frame_index for f in frames], dtype=np.int32)
    timestamp_seconds = np.array([f.timestamp_seconds for f in frames], dtype=np.float64)
    hand_present = np.array([f.hand_present for f in frames], dtype=bool)
    handedness = np.array([f.handedness_label or "" for f in frames], dtype="U8")
    handedness_confidence = np.array(
        [f.handedness_confidence if f.handedness_confidence is not None else np.nan for f in frames],
        dtype=np.float32,
    )
    detection_confidence = np.array(
        [f.detection_confidence if f.detection_confidence is not None else np.nan for f in frames],
        dtype=np.float32,
    )
    landmarks_3d = np.stack([_pack_landmarks_3d(f) for f in frames]) if n else np.zeros((0, _LANDMARK3D_ROWS, 3))
    quality_flags = np.array([json.dumps(f.quality_flags) for f in frames], dtype="U256")
    mano_params_json = np.array([json.dumps(f.mano_params) if f.mano_params else "" for f in frames], dtype="U8192")
    mano_references_json = np.array(
        [json.dumps(f.mano_references) if f.mano_references else "" for f in frames], dtype="U2048"
    )
    extractor_metadata_json = np.array([json.dumps(f.extractor_metadata) for f in frames], dtype="U512")

    arrays: dict[str, Any] = dict(
        frame_index=frame_index,
        timestamp_seconds=timestamp_seconds,
        hand_present=hand_present,
        handedness_label=handedness,
        handedness_confidence=handedness_confidence,
        detection_confidence=detection_confidence,
        landmarks_3d=landmarks_3d,
        quality_flags_json=quality_flags,
        mano_params_json=mano_params_json,
        mano_references_json=mano_references_json,
        extractor_metadata_json=extractor_metadata_json,
        run_metadata_json=np.array(json.dumps(run_metadata or {})),
    )

    vertices_by_hand = vertices_by_hand or {}
    if vertices_by_hand:
        keys = sorted(vertices_by_hand.keys())
        arrays["vertices_keys"] = np.array([f"{fi}:{hi}" for fi, hi in keys], dtype="U32")
        arrays["vertices"] = np.stack([vertices_by_hand[k] for k in keys]).astype(np.float32)

    np.savez_compressed(npz_path, **arrays)
    return npz_path


def load_raw_video_output(npz_path: Path) -> dict[str, Any]:
    with np.load(npz_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def hand_pose_frames_from_npz(npz_path: Path) -> list[HandPoseFrame]:
    """Reconstruct HandPoseFrame objects (without embedded mesh vertices) for
    evaluation code that operates on the common schema."""
    from pose.common.schema import Landmark3D  # noqa: PLC0415

    data = load_raw_video_output(npz_path)
    n = len(data["frame_index"])
    result: list[HandPoseFrame] = []
    for i in range(n):
        lm3d = data["landmarks_3d"][i]
        landmarks_3d = [
            Landmark3D(x=float(row[0]), y=float(row[1]), z=float(row[2]))
            for row in lm3d
            if not np.isnan(row).any()
        ]
        mano_params_raw = str(data["mano_params_json"][i])
        mano_refs_raw = str(data["mano_references_json"][i])
        result.append(
            HandPoseFrame(
                frame_index=int(data["frame_index"][i]),
                timestamp_seconds=float(data["timestamp_seconds"][i]),
                hand_present=bool(data["hand_present"][i]),
                handedness_label=str(data["handedness_label"][i]) or None,
                handedness_confidence=(
                    None if np.isnan(data["handedness_confidence"][i]) else float(data["handedness_confidence"][i])
                ),
                detection_confidence=(
                    None if np.isnan(data["detection_confidence"][i]) else float(data["detection_confidence"][i])
                ),
                landmarks_3d=landmarks_3d,
                wrist_position=landmarks_3d[0] if landmarks_3d else None,
                mano_params=json.loads(mano_params_raw) if mano_params_raw else None,
                mano_references=json.loads(mano_refs_raw) if mano_refs_raw else None,
                extractor_metadata=json.loads(str(data["extractor_metadata_json"][i])),
                quality_flags=json.loads(str(data["quality_flags_json"][i])),
            )
        )
    return result


assert {f.name for f in fields(HandPoseFrame)} >= set(_SCALAR_FIELDS), "schema drift"
